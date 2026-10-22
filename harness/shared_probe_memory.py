"""Setting-owned immutable memory, shared by two independent Probe runs.

The owner lives in one worker process. Consumer adapters and sessions are private;
only prepared checkpoint memory and its read-only query transports are retained.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from harness.checkpoint import read_branch_namespace
from harness.probe_repeat_source import REFERENCE_CONDITIONS


class SharedMemoryError(RuntimeError):
    pass


@dataclass
class SharedQueryResource:
    client: Any
    new_adapter: Callable[[], Any]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _remove_shared_file_view(directory: Path) -> None:
    """Only the drained owner may make its accepted view writable to remove it."""
    for root, _, _ in os.walk(directory, followlinks=False):
        Path(root).chmod(0o700)
    shutil.rmtree(directory)


class SharedProbeMemoryOwner:
    def __init__(self, root: Path, *, restore_slots_root: Path | None = None,
                 restore_concurrency: int = 1, pressure_file: Path | None = None) -> None:
        if restore_concurrency < 1:
            raise ValueError("restore_concurrency must be positive")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.restore_slots_root = Path(restore_slots_root or self.root / "restore-slots")
        self.restore_concurrency = restore_concurrency
        self.pressure_file = Path(pressure_file) if pressure_file is not None else None
        self._lock = (self.root / "owner.lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._lock.close()
            raise
        self._entries: dict[str, dict[str, Any]] = {}
        self._preparations: dict[str, asyncio.Task] = {}
        self._tasks: set[asyncio.Task] = set()
        self._source_identity: tuple[str, str, str | None] | None = None
        self._stopping = False
        self._closed = False
        self._close_task: asyncio.Task | None = None

    def request_stop(self) -> None:
        self._stopping = True

    def is_stop_requested(self) -> bool:
        return self._stopping

    async def _thread(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        # shield keeps the actual worker observable when the awaiting Probe is cancelled.
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return await self._wait_actual(task)

    @staticmethod
    async def _wait_actual(task: asyncio.Task) -> Any:
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            # The worker has ended before its caller can release a restore lock.
            task.exception() if not task.cancelled() else None
            raise asyncio.CancelledError
        return task.result()

    @asynccontextmanager
    async def _restore_slot(self):
        self.restore_slots_root.mkdir(parents=True, exist_ok=True)
        stream = None
        try:
            while stream is None:
                if self._stopping:
                    raise SharedMemoryError("setting owner is stopping")
                if self.pressure_file is not None and self.pressure_file.exists():
                    pressure = json.loads(self.pressure_file.read_text())
                    if pressure.get("paused"):
                        await asyncio.sleep(0.25)
                        continue
                for index in range(self.restore_concurrency):
                    candidate = (self.restore_slots_root / f"restore-{index}.lock").open("a+")
                    try:
                        fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close()
                        continue
                    stream = candidate
                    break
                if stream is None:
                    await asyncio.sleep(0.1)
            yield
        finally:
            if stream is not None:
                stream.close()  # stable lock inode; never unlink

    def _save(self, entry: dict[str, Any]) -> None:
        _atomic(entry["manifest_path"], entry["record"])

    async def _prepare(self, manager: Any, checkpoint_id: str, key: str) -> dict[str, Any]:
        directory = self.root / key
        manifest_path = self.root / f"{key}.json"
        record = {"schema_version": 1, "source_run_dir": str(manager.run_dir.resolve()),
                  "checkpoint_id": checkpoint_id,
                  "query_identity": manager.shared_query_identity,
                  "memory_condition": manager.memory_condition, "state": "preparing", "consumers": {}}
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        entry = {"directory": directory, "manager": manager, "manifest_path": manifest_path,
                 "record": record, "query": None}
        self._entries[key] = entry
        if previous is not None:
            record.update(previous)
            if previous.get("state") != "ready" or any(
                value != "finished" for value in previous.get("consumers", {}).values()
            ):
                entry["retain_only"] = True
                raise SharedMemoryError(f"Unresolved shared memory retained for reconciliation: {manifest_path}")
        elif directory.exists():
            raise SharedMemoryError(f"Unrecorded shared directory must be reconciled: {directory}")
        else:
            self._save(entry)  # intent before any remote operation
        try:
            async with self._restore_slot():
                started = time.monotonic()
                if previous is None:
                    await self._thread(manager.prepare_shared_memory, checkpoint_id, directory)
                namespace = (read_branch_namespace(directory) or {}).get("namespace")
                if manager.memory_condition != "nanobot_file_memory":
                    if manager.shared_query_factory is None:
                        raise SharedMemoryError("Backend has no shared read-only query factory")
                    query = manager.shared_query_factory(namespace, directory, manager.run_dir.name)
                    if inspect.isawaitable(query):
                        query = await query
                    entry["query"] = query
                record.update(state="ready", namespace=namespace,
                              prepare_seconds=time.monotonic() - started,
                              restore_count=1, directory=str(directory))
                self._save(entry)
                return entry
        except BaseException as exc:
            record.update(state="unresolved", error=f"{type(exc).__name__}: {exc}")
            self._save(entry)
            raise

    async def branch(self, manager: Any, checkpoint_id: str, branch_dir: Path) -> Any:
        if self._closed or self._stopping:
            raise SharedMemoryError("Shared owner is closed or stopping")
        if manager.memory_condition in REFERENCE_CONDITIONS:
            # A reference arm reads none of the source's memory: the
            # orchestrator empties the branch's and binds its own adapter. So
            # there is no shared entry, only the branch's own copy of the pin.
            await self._thread(manager.branch_from, checkpoint_id, branch_dir)
            return None
        if manager.memory_condition not in {"simple_rag", "mem0", "memos", "nanobot_file_memory"}:
            raise SharedMemoryError(f"Shared read-only backend has not been validated: {manager.memory_condition}")
        identity = (str(manager.run_dir.resolve()), manager.memory_condition, manager.shared_query_identity)
        if self._source_identity is not None and self._source_identity != identity:
            raise SharedMemoryError("A setting owner cannot mix sources or retrieval configurations")
        self._source_identity = identity
        key = _digest([identity, checkpoint_id])
        if key not in self._preparations:
            self._preparations[key] = asyncio.create_task(self._prepare(manager, checkpoint_id, key))
        entry = await asyncio.shield(self._preparations[key])
        if entry["record"]["state"] != "ready":
            raise SharedMemoryError("Shared memory is no longer ready")
        query = entry["query"]
        if query is not None:
            check_ready = getattr(query.client, "ensure_query_ready", None)
            if check_ready is not None:
                try:
                    check_ready()
                except Exception as exc:
                    reason = f"Shared query client is not ready: {type(exc).__name__}: {exc}"
                    entry["record"].update(state="cleanup_pending", error=reason)
                    self._save(entry)
                    raise SharedMemoryError(reason) from exc
        consumer = str(Path(branch_dir).resolve())
        entry["record"]["consumers"][consumer] = "active"
        self._save(entry)
        await self._thread(manager.branch_from, checkpoint_id, branch_dir,
                           shared_memory_dir=entry["directory"])
        if query is not None:
            return query.new_adapter()
        if manager.memory_condition == "nanobot_file_memory":
            from memory.conditions.nanobot_file_memory import NanobotFileMemoryAdapter
            memories = list((entry["directory"] / "pa_workspace").glob("*/memory"))
            if len(memories) != 1:
                raise SharedMemoryError("Expected one persona memory directory")
            return NanobotFileMemoryAdapter(memories[0])
        return None

    def release_consumer(self, branch_dir: Path, *, failed: bool = False) -> None:
        consumer = str(Path(branch_dir).resolve())
        for entry in self._entries.values():
            if consumer in entry["record"]["consumers"]:
                entry["record"]["consumers"][consumer] = "failed" if failed else "finished"
                self._save(entry)
                return

    async def close(self) -> bool:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        return await self._wait_actual(self._close_task)

    async def _close(self) -> bool:
        if self._closed:
            return all(entry["record"].get("state") == "released" for entry in self._entries.values())
        self._stopping = True
        self._closed = True
        ok = True
        try:
            await asyncio.gather(*self._preparations.values(), return_exceptions=True)
            if self._tasks:
                await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
            for entry in self._entries.values():
                record = entry["record"]
                if entry.get("retain_only"):
                    ok = False
                    continue
                try:
                    query = entry["query"]
                    if query is not None:
                        closed = await self._thread(query.client.close, 60.0)
                        if closed is False:
                            raise SharedMemoryError("Actual backend requests have not drained")
                    if record["state"] != "ready" or any(v == "active" for v in record["consumers"].values()):
                        raise SharedMemoryError("Unresolved consumers or preparation; memory retained")
                    record["state"] = "releasing"
                    self._save(entry)
                    await self._thread(entry["manager"].release_branch_namespace, entry["directory"])
                    if entry["manager"].memory_condition == "nanobot_file_memory":
                        await self._thread(_remove_shared_file_view, entry["directory"])
                    record["state"] = "released"
                    self._save(entry)
                except Exception as exc:
                    ok = False
                    record.update(state="cleanup_pending", error=f"{type(exc).__name__}: {exc}")
                    self._save(entry)
        finally:
            self._lock.close()
        return ok
