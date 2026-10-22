"""Real checkpoint preparation with offline MemOS administration and transport."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import threading

import httpx
import pytest

from harness.checkpoint import read_branch_namespace
from harness.run_ledger import RunLedger
from harness.shared_probe_memory import (
    SharedMemoryError,
    SharedProbeMemoryOwner,
    SharedQueryResource,
)
from memory.backends.memos_adapter import MemOSAdapter
from memory.backends.memos_client import MemOSHttpClient
from memory.scope import MemoryRetrievalBudget, MemoryScope, RetrieveRequest
from test_harness.test_checkpoint import (
    FakeMemosClient,
    PERSONA,
    RUN_ID,
    _branch_session,
    _build_run_dir,
    _external_manager,
    _manager,
    _write_session,
)


class StatefulMemos(FakeMemosClient):
    """Use the existing checkpoint validator; retain the actual imported IDs."""

    def __init__(self):
        super().__init__()
        self.cubes = {}
        self.query_clients = []
        self.query_handler = None
        self.import_entered = None
        self.import_finish = None

    def export_cube(self, cube_id):
        if cube_id in self.cubes:
            self.calls.append(("export", cube_id))
            return deepcopy(self.cubes[cube_id])
        return super().export_cube(cube_id)

    def import_cube_as(self, payload, target_cube_id):
        if self.import_entered is not None:
            self.import_entered.set()
            assert self.import_finish.wait(timeout=5), "test must release import"
        receipt = super().import_cube_as(payload, target_cube_id)
        self.cubes[target_cube_id] = deepcopy(payload)
        return receipt

    def delete_cube(self, cube_id):
        self.cubes.pop(cube_id, None)
        return super().delete_cube(cube_id)

    def queries(self, namespace, directory, source_run_id):
        def handle(request):
            if self.query_handler is not None:
                self.query_handler(request)
            body = json.loads(request.content)
            assert body["readable_cube_ids"] == [namespace]
            node = self.cubes[namespace]["nodes"][0]["properties"]
            return httpx.Response(200, json={
                "code": 200, "message": "ok", "data": {
                    "text_mem": [{"cube_id": namespace, "memories": [{
                        "id": node["id"], "memory": node["memory"],
                        "metadata": {"relativity": 0.9},
                    }]}],
                },
            })

        client = MemOSHttpClient(
            api_base="http://memos.test", transport=httpx.MockTransport(handle)
        ).read_only_for_cube(namespace)
        self.query_clients.append(client)
        return SharedQueryResource(
            client=client,
            new_adapter=lambda: MemOSAdapter(
                client=client, cube_id=namespace, memory_session_id=f"{source_run_id}_main"
            ),
        )


def _setting(tmp_path):
    run = _build_run_dir(tmp_path)
    _write_session(run, cursors=(3, 3, 3))
    ledger = RunLedger(run, RUN_ID)
    client = StatefulMemos()
    manager = _external_manager(
        run, ledger, condition="memos", namespace="source-cube", client=client
    )
    manager.shared_query_factory = client.queries
    manager.shared_query_identity = "original-session-and-frozen-retrieval-config"
    ledger.mark_running("acc_108")
    manager.save("acc_108", pinned=True)
    client.calls.clear()
    return manager, client


def _records(owner):
    return [json.loads(path.read_text()) for path in owner.root.glob("*.json")]


def _calls(client, operation):
    return [call for call in client.calls if call[0] == operation]


def _request():
    return RetrieveRequest(
        query="preferences", scope=MemoryScope(run_id="repeat02", persona_id=PERSONA),
        session_id="repeat02-private-session", budget=MemoryRetrievalBudget(max_items=2),
    )


def _locked(path):
    with Path(path).open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


@pytest.mark.asyncio
async def test_restore_once_across_rounds_and_keep_private_execution_state(tmp_path):
    manager, admin = _setting(tmp_path)
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    first = tmp_path / "repeat02" / "final01"
    second = tmp_path / "repeat03" / "final01"
    try:
        first_adapter = await owner.branch(manager, "acc_108", first)
        namespace = read_branch_namespace(first)["namespace"]
        owner.release_consumer(first)
        assert manager.release_branch_namespace(first) is None
        assert not _calls(admin, "delete")
        assert namespace in admin.cubes
        assert not admin.query_clients[0]._http.is_closed

        second_adapter = await owner.branch(manager, "acc_108", second)
        assert read_branch_namespace(second)["namespace"] == namespace
        assert first_adapter is not second_adapter
        assert first_adapter._client is second_adapter._client
        assert len(_calls(admin, "import")) == 1
        assert len(admin.query_clients) == 1
        _branch_session(first).write_text("private first-round conversation")
        (first / "canonical_stage/email/inbox.json").write_text("private scenario")
        assert "private" not in _branch_session(second).read_text()
        assert (second / "canonical_stage/email/inbox.json").read_text() == "[]"
        owner.release_consumer(second)
        assert await owner.close()
        assert _calls(admin, "delete") == [("delete", namespace)]
        assert _records(owner)[0]["state"] == "released"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_concurrent_questions_prepare_a_checkpoint_once(tmp_path):
    manager, admin = _setting(tmp_path)
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    branches = [tmp_path / f"repeat{repeat}" / f"final{question}" for repeat in (2, 3) for question in range(3)]
    try:
        adapters = await asyncio.gather(*(owner.branch(manager, "acc_108", path) for path in branches))
        assert len({id(adapter) for adapter in adapters}) == len(branches)
        assert len({id(adapter._client) for adapter in adapters}) == 1
        assert len(_calls(admin, "import")) == 1
        for branch in branches:
            owner.release_consumer(branch)
        assert await owner.close()
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_different_pins_and_setting_owners_have_distinct_namespaces(tmp_path):
    manager, admin = _setting(tmp_path)
    manager.ledger.mark_running("acc_060")
    manager.save("acc_060", pinned=True)
    admin.calls.clear()
    first = SharedProbeMemoryOwner(tmp_path / "setting-one")
    second = SharedProbeMemoryOwner(tmp_path / "setting-two")
    try:
        paths = [tmp_path / f"branch-{index}" for index in range(3)]
        await first.branch(manager, "acc_060", paths[0])
        await first.branch(manager, "acc_108", paths[1])
        await second.branch(manager, "acc_108", paths[2])
        namespaces = [read_branch_namespace(path)["namespace"] for path in paths]
        assert len(set(namespaces)) == 3
        native_ids = [admin.cubes[namespace]["nodes"][0]["properties"]["id"] for namespace in namespaces]
        assert len(set(native_ids)) == 3
        assert len(_calls(admin, "import")) == 3
        first.release_consumer(paths[0])
        first.release_consumer(paths[1])
        assert await first.close()
        assert namespaces[2] in admin.cubes
        second.release_consumer(paths[2])
        assert await second.close()
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_one_owner_refuses_a_different_source_or_query_contract(tmp_path):
    manager, admin = _setting(tmp_path / "source-one")
    other, _ = _setting(tmp_path / "source-two")
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    branch = tmp_path / "repeat02"
    try:
        await owner.branch(manager, "acc_108", branch)
        with pytest.raises(SharedMemoryError, match="mix sources"):
            await owner.branch(other, "acc_108", tmp_path / "other-source")
        manager.shared_query_identity = "changed-retrieval-settings"
        with pytest.raises(SharedMemoryError, match="mix sources"):
            await owner.branch(manager, "acc_108", tmp_path / "other-query")
        assert len(_calls(admin, "import")) == 1
        owner.release_consumer(branch)
        assert await owner.close()
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_active_consumer_prevents_cleanup(tmp_path):
    manager, admin = _setting(tmp_path)
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    await owner.branch(manager, "acc_108", tmp_path / "repeat02")
    assert not await owner.close()
    assert not _calls(admin, "delete")
    assert "active" in _records(owner)[0]["consumers"].values()


@pytest.mark.asyncio
async def test_unknown_remote_request_retains_cube_even_when_retry_succeeds(tmp_path, monkeypatch):
    from memory.backends import memos_client

    manager, admin = _setting(tmp_path)
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    branch = tmp_path / "repeat02"
    adapter = await owner.branch(manager, "acc_108", branch)
    calls = 0

    def timeout_first(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("unknown server completion", request=request)

    monkeypatch.setattr(memos_client, "_RETRY_PAUSE_SECONDS", 0)
    admin.query_handler = timeout_first
    assert (await adapter.retrieve(_request())).raw
    assert calls == 2
    owner.release_consumer(branch)
    assert not await owner.close()
    assert not _calls(admin, "delete")
    record = _records(owner)[0]
    assert record["state"] == "cleanup_pending"
    assert "may still be running" in record["error"]


@pytest.mark.asyncio
async def test_cancelled_probe_query_is_drained_before_namespace_deletion(tmp_path):
    manager, admin = _setting(tmp_path)
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    branch = tmp_path / "repeat02"
    adapter = await owner.branch(manager, "acc_108", branch)
    entered, finish = threading.Event(), threading.Event()

    def wait_query(request):
        entered.set()
        assert finish.wait(timeout=5)

    admin.query_handler = wait_query
    task = asyncio.create_task(adapter.retrieve(_request()))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        owner.release_consumer(branch, failed=True)
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0.02)
        assert not closing.done()
        assert not _calls(admin, "delete")
        assert _locked(owner.root / "owner.lock")
    finally:
        finish.set()
        if closing is not None:
            assert await closing
        else:
            await owner.close()
    assert len(_calls(admin, "delete")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer_state", ["active", "failed"])
async def test_pending_prior_owner_is_not_rebuilt_or_forgotten(tmp_path, consumer_state):
    manager, admin = _setting(tmp_path)
    root = tmp_path / "shared"
    first = SharedProbeMemoryOwner(root)
    branch = tmp_path / "repeat02"
    await first.branch(manager, "acc_108", branch)
    if consumer_state == "failed":
        first.release_consumer(branch, failed=True)
    before = _records(first)[0]
    # Simulate losing only the process and its connections, leaving its intent.
    first._lock.close()
    admin.query_clients[0].close()
    second = SharedProbeMemoryOwner(root)
    try:
        with pytest.raises(SharedMemoryError, match="reconciliation"):
            await second.branch(manager, "acc_108", tmp_path / "repeat03")
        assert len(_calls(admin, "import")) == 1
        assert not await second.close()
        after = _records(second)[0]
        for key in ("namespace", "consumers", "directory"):
            assert after[key] == before[key]
        assert not _calls(admin, "delete")
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_cancelled_close_holds_owner_lock_until_workers_finish(tmp_path):
    manager, admin = _setting(tmp_path)
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    branch = tmp_path / "repeat02"
    adapter = await owner.branch(manager, "acc_108", branch)
    entered, finish = threading.Event(), threading.Event()

    def wait_query(request):
        entered.set()
        assert finish.wait(timeout=5)

    admin.query_handler = wait_query
    query = asyncio.create_task(adapter.retrieve(_request()))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        query.cancel()
        with pytest.raises(asyncio.CancelledError):
            await query
        owner.release_consumer(branch, failed=True)
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0.02)
        closing.cancel()
        await asyncio.sleep(0.02)
        assert _locked(owner.root / "owner.lock")
        assert not _calls(admin, "delete")
    finally:
        finish.set()
        if closing is not None:
            with pytest.raises(asyncio.CancelledError):
                await closing
        else:
            await owner.close()
    assert not _locked(owner.root / "owner.lock")


@pytest.mark.asyncio
async def test_cancelled_preparation_keeps_restore_lock_until_import_finishes(tmp_path):
    manager, admin = _setting(tmp_path)
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    admin.import_entered, admin.import_finish = threading.Event(), threading.Event()
    branch = asyncio.create_task(owner.branch(manager, "acc_108", tmp_path / "repeat02"))
    try:
        assert await asyncio.to_thread(admin.import_entered.wait, 2)
        preparation = next(iter(owner._preparations.values()))
        preparation.cancel()
        await asyncio.sleep(0.02)
        assert _locked(owner.restore_slots_root / "restore-0.lock")
    finally:
        admin.import_finish.set()
        with pytest.raises(asyncio.CancelledError):
            await branch
        assert not await owner.close()
    assert not _calls(admin, "delete")
    assert len(_calls(admin, "import")) == 1


@pytest.mark.asyncio
async def test_file_memory_is_shared_but_sessions_and_write_boundaries_are_private(tmp_path):
    from pa.agent.memory import MarkdownMemoryStore

    run = _build_run_dir(tmp_path)
    _write_session(run, cursors=(3, 3, 3))
    ledger = RunLedger(run, RUN_ID)
    ledger.mark_running("acc_108")
    manager = _manager(run, ledger)
    manager.save("acc_108", pinned=True)
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    branches = [tmp_path / repeat for repeat in ("repeat02", "repeat03")]
    try:
        adapters = [await owner.branch(manager, "acc_108", path) for path in branches]
        memory_dirs = [path / "pa_workspace" / PERSONA / "memory" for path in branches]
        assert all(path.is_symlink() for path in memory_dirs)
        assert memory_dirs[0].resolve() == memory_dirs[1].resolve()
        assert adapters[0] is not adapters[1]
        assert (await adapters[0].retrieve(_request())).raw == (await adapters[1].retrieve(_request())).raw
        assert (memory_dirs[0] / "MEMORY.md").read_text() == "pa memory\n"
        assert not (run / "pa_workspace" / PERSONA / "memory/.mempa-read-only").exists()
        for adapter in adapters:
            with pytest.raises(PermissionError):
                await adapter.reset_scope(_request().scope)
        store = MarkdownMemoryStore(branches[0] / "pa_workspace" / PERSONA, log_scope="test")
        for write in (store.write_long_term, store.append_history):
            with pytest.raises(PermissionError):
                write("must not reach shared memory")
        _branch_session(branches[0]).write_text("private repeat02")
        assert "private repeat02" not in _branch_session(branches[1]).read_text()
        for path in branches:
            owner.release_consumer(path)
        assert await owner.close()
    finally:
        await owner.close()
