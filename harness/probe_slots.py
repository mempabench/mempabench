"""Workspace-wide Probe batch slots, defaulting to two active rounds
(`probe_batch_slot`)."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Callable

BATCH_SLOTS = 2
MAX_BATCH_WORKERS = 4
CAPACITY_FILE = "capacity.json"


@contextmanager
def _capacity_lock(root: Path, *, exclusive: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "capacity.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def batch_capacity(root: Path) -> int:
    """Read the shared capacity; unchanged workspaces retain their two slots."""
    path = Path(root) / CAPACITY_FILE
    if not path.exists():
        return BATCH_SLOTS
    value = json.loads(path.read_text())["batch_slots"]
    if type(value) is not int or value < BATCH_SLOTS:
        raise ValueError("Probe batch capacity must be at least 2 for existing producers")
    return value


def configure_batch_slots(root: Path, capacity: int) -> None:
    """Configure the existing slot pool without unlinking guards or busy slots."""
    if type(capacity) is not int or capacity < BATCH_SLOTS:
        raise ValueError("Probe batch capacity must be at least 2 for existing producers")
    root = Path(root).resolve()
    with _capacity_lock(root, exclusive=True):
        # Existing producers use the same slot-N.lock files, including older
        # checkouts that only know about slots 0 and 1. Never bypass their locks.
        for path in root.glob("slot-*.lock"):
            index = int(path.stem.removeprefix("slot-"))
            if index < capacity:
                continue
            with path.open("r+") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("Cannot remove an occupied Probe batch slot") from exc
        temporary = root / (CAPACITY_FILE + ".tmp")
        temporary.write_text(json.dumps({"batch_slots": capacity}) + "\n")
        temporary.replace(root / CAPACITY_FILE)


def active_batch_slots(root: Path) -> list[dict]:
    """Return only kernel-held slots; stale PID metadata is never activity."""
    active = []
    for path in sorted(Path(root).glob("slot-*.lock")):
        with path.open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                try:
                    row = json.load(stream)
                except (ValueError, OSError):
                    row = {"metadata_pending": True}
                active.append({**row, "path": str(path)})
    return active


def workspace_root(project_root: Path) -> Path:
    value = os.environ.get("MEMPA_WORKSPACE_ROOT")
    root = Path(value) if value else project_root / "workspace"
    if not root.is_absolute():
        raise ValueError("MEMPA_WORKSPACE_ROOT must be absolute")
    return root.resolve()


@asynccontextmanager
async def probe_batch_slot(
    root: Path | None,
    *,
    stop_requested: Callable[[], bool],
    workers: int,
    progress: Callable[[str], None] = lambda _: None,
    poll_seconds: float = 0.25,
):
    """Wait without dispatching; keep stable lock inodes and release on every exit.

    The caller retains its run producer lock while waiting. Each acquired slot
    covers the entire batch (at most four workers), including its entry audit.
    None permits isolated in-process test executors to use their existing path.
    """
    if root is None:
        yield not stop_requested()
        return
    if not 1 <= workers <= MAX_BATCH_WORKERS:
        raise ValueError("global Probe slots require 1..4 workers per batch")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    held = None
    announced = False
    try:
        while not stop_requested():
            with _capacity_lock(root):
                capacity = batch_capacity(root)
                for index in range(capacity):
                    fd = os.open(root / f"slot-{index}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                    stream = os.fdopen(fd, "r+")
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        stream.close()
                        continue
                    except BaseException:
                        stream.close()
                        raise
                    held = stream
                    held.seek(0)
                    held.truncate()
                    json.dump({"pid": os.getpid(), "slot": index, "workers": workers,
                               "acquired_at": datetime.now(timezone.utc).isoformat()}, held)
                    held.flush()
                    progress(f"Probe batch slot acquired: {index + 1}/{capacity}")
                    break
            if held is not None:
                break
            if not announced:
                progress("waiting for a global Probe batch slot; all pending Probes remain pending")
                announced = True
            await asyncio.sleep(poll_seconds)
        yield held is not None and not stop_requested()
    finally:
        if held is not None:
            # Never unlink: another process may already have opened this inode.
            held.close()
