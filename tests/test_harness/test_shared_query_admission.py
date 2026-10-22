"""Unknown remote completion blocks admission before any new private branch."""

import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from harness.shared_probe_memory import SharedMemoryError, SharedProbeMemoryOwner, SharedQueryResource
from memory.backends import memos_client
from memory.backends.mem0_adapter import Mem0Adapter
from memory.backends.memos_client import MemOSHttpClient, MemOSTransportError
from test_harness.test_checkpoint import RUN_ID, _external_manager, _mem0_run
from test_harness.test_shared_probe_memory import _setting
from test_memory.test_mem0_remote_drain import SnapshotAdmin, _client


@pytest.mark.parametrize("backend", ["memos", "mem0"])
@pytest.mark.parametrize("state", ["draining", "closed"])
def test_readiness_is_local_and_refuses_finished_lifecycle(monkeypatch, backend, state):
    if backend == "memos":
        network = Mock()
        client = MemOSHttpClient(
            api_base="http://memos.invalid", transport=httpx.MockTransport(network),
        ).read_only_for_cube("shared")
    else:
        client, network, _ = _client(monkeypatch, httpx.ReadTimeout)
    try:
        assert client.ensure_query_ready() is None
        assert client.ensure_query_ready() is None
        if state == "draining":
            client.drain(0)
        else:
            client.close(0)
        with pytest.raises(RuntimeError, match="draining|closed"):
            client.ensure_query_ready()
        network.assert_not_called()
    finally:
        client.close(0)


@pytest.mark.parametrize("backend", ["memos", "mem0", "simple_rag"])
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.RemoteProtocolError])
@pytest.mark.asyncio
async def test_unknown_remote_blocks_new_consumer_but_other_pin_continues(
    tmp_path, monkeypatch, backend, error,
):
    clients = []
    if backend == "memos":
        manager, admin = _setting(tmp_path)
        clients = admin.query_clients
        calls = 0

        def fail_first(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise error("remote completion unknown", request=request)

        admin.query_handler = fail_first
        monkeypatch.setattr(memos_client, "_RETRY_PAUSE_SECONDS", 0)
    else:
        run, ledger, admin_client = _mem0_run(tmp_path)
        admin = SnapshotAdmin()
        admin_client.qdrant_admin = admin
        manager = _external_manager(
            run, ledger, condition=backend, namespace="source", client=admin_client,
        )
        manager.save("acc_108", pinned=True)

        def factory(namespace, directory, source_run_id):
            client, network, _ = _client(monkeypatch, error, namespace=namespace, run_id=RUN_ID)
            if clients:
                network.side_effect = lambda request: httpx.Response(200, json={
                    "result": {"points": []}, "status": "ok", "time": 0.0,
                })
            clients.append(client)
            return SharedQueryResource(client=client, new_adapter=lambda: Mem0Adapter(client=client))

        manager.shared_query_factory = factory
    manager.ledger.mark_running("acc_060")
    manager.save("acc_060", pinned=True)
    manager.branch_from = Mock(wraps=manager.branch_from)
    owner = SharedProbeMemoryOwner(tmp_path / "shared")

    async def query(client):
        if backend == "memos":
            return await asyncio.to_thread(client.search_memories, {
                "query": "fixed query", "user_id": "user_a", "session_id": "original_main",
                "readable_cube_ids": [client.cube_id], "filter": {},
            })
        return await asyncio.to_thread(
            client.search, "fixed query", filters={"user_id": "user_a", "run_id": RUN_ID},
        )

    first = tmp_path / "repeat02" / "first"
    rejected = tmp_path / "repeat03" / "must-not-exist"
    unrelated = tmp_path / "repeat03" / "other-pin"
    try:
        await owner.branch(manager, "acc_108", first)
        if backend == "memos" and error is httpx.ReadTimeout:
            # MemOS's original retry may succeed, but cannot resolve the first request.
            assert await query(clients[0])
            assert calls == 2
        elif backend == "memos":
            with pytest.raises(MemOSTransportError):
                await query(clients[0])
            assert calls == 1
        else:
            with pytest.raises(ResponseHandlingException):
                await query(clients[0])
        owner.release_consumer(first, failed=True)
        with pytest.raises(RuntimeError, match="unresolved|may still be running"):
            clients[0].ensure_query_ready()
        with pytest.raises(SharedMemoryError, match="query client is not ready"):
            await owner.branch(manager, "acc_108", rejected)
        assert not rejected.exists()
        assert manager.branch_from.call_count == 1
        bad = next(iter(owner._entries.values()))
        record = json.loads(bad["manifest_path"].read_text())
        assert record["state"] == "cleanup_pending"
        assert "may still be running" in record["error"]
        assert record["consumers"] == {str(first.resolve()): "failed"}
        assert not owner.is_stop_requested()

        await owner.branch(manager, "acc_060", unrelated)
        assert unrelated.is_dir()
        assert len(clients) == 2
        assert await query(clients[1]) is not None
        owner.release_consumer(unrelated)
        assert not await owner.close()
        records = {entry["record"]["checkpoint_id"]: entry["record"] for entry in owner._entries.values()}
        assert records["acc_108"]["state"] == "cleanup_pending"
        assert records["acc_060"]["state"] == "released"
        remaining = admin.cubes if backend == "memos" else admin.names
        assert records["acc_108"]["namespace"] in remaining
        assert records["acc_060"]["namespace"] not in remaining
    finally:
        await owner.close()
