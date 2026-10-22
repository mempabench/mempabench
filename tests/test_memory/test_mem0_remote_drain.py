"""Uncertain remote Qdrant completion must outlive finished local threads."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from harness.shared_probe_memory import SharedProbeMemoryOwner, SharedQueryResource
from memory.backends import mem0_read_only as ro
from memory.backends.mem0_adapter import Mem0Adapter
from test_harness.test_checkpoint import FakeQdrant, RUN_ID, _external_manager, _mem0_run


class FailingBody(httpx.SyncByteStream):
    def __init__(self, error):
        self.error = error
        self.closed = False

    def __iter__(self):
        yield b'{"result":'
        raise self.error("server may still be running")

    def close(self):
        self.closed = True


def _client(monkeypatch, error, *, body=False, namespace="shared", run_id="source_run"):
    stream = FailingBody(error)

    def handler(request):
        if body:
            return httpx.Response(200, stream=stream)
        raise error("server completion unknown", request=request)

    network = Mock(side_effect=handler)
    monkeypatch.setattr(ro.httpx, "HTTPTransport", lambda: httpx.MockTransport(network))
    qdrant = ro._query_qdrant(host="qdrant.invalid", port=6333, namespaces={namespace})
    qdrant.close = Mock(wraps=qdrant.close)

    def search(**kwargs):
        qdrant.query_points(collection_name=namespace, query=[1.0, 0.0])
        return {"results": []}

    memory = SimpleNamespace(search=search, embedding_model=SimpleNamespace(client=Mock()))
    client = ro.ReadOnlyMem0Client(memory, qdrant=qdrant, history_path=Path("/unused/history.sqlite3"),
        collection_name=namespace, qdrant_host="qdrant.invalid", qdrant_port=6333,
        expected_run_id=run_id, embedding_dimensions=2, qdrant_distance="Cosine",
        qdrant_on_disk=True, qdrant_bm25_enabled=False, enable_entity_memory=False)
    return client, network, stream


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.RemoteProtocolError])
@pytest.mark.parametrize("body", [False, True], ids=["headers", "body"])
def test_remote_unknown_is_recorded_before_qdrant_wraps_it_and_survives_close(monkeypatch, error, body):
    client, network, stream = _client(monkeypatch, error, body=body)
    with pytest.raises(ResponseHandlingException):
        client.search("fixed query", filters={"user_id": "user_a", "run_id": "source_run"})
    assert client._active == 0
    assert client._qdrant._probe_read_only_transport.unresolved_remote_requests == 1
    if body:
        assert stream.closed
    with pytest.raises(ro.Mem0UnresolvedRequestError, match="retain the shared namespace"):
        client.search("another query", filters={"user_id": "user_a", "run_id": "source_run"})
    assert network.call_count == 1
    with pytest.raises(ro.Mem0UnresolvedRequestError):
        client.drain(0)
    for _ in range(2):
        with pytest.raises(ro.Mem0UnresolvedRequestError):
            client.close(0)
    client._qdrant.close.assert_called_once()
    client._memory.embedding_model.client.close.assert_called_once()


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout])
def test_pre_delivery_connection_failures_do_not_claim_unknown_remote_work(monkeypatch, error):
    client, network, _ = _client(monkeypatch, error)
    with pytest.raises(ResponseHandlingException):
        client.search("query", filters={"user_id": "user_a", "run_id": "source_run"})
    assert network.call_count == 1
    assert client._qdrant._probe_read_only_transport.unresolved_remote_requests == 0
    assert client.close(0) is True


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_real_vendor_keyword_catch_cannot_turn_unknown_into_success(monkeypatch, error):
    from mem0.vector_stores.qdrant import Qdrant
    from qdrant_client.models import SparseVector

    client, network, _ = _client(monkeypatch, error)
    # Use the actual catch-all keyword path without allocating storage or an encoder.
    store = object.__new__(Qdrant)
    store.client = client._qdrant
    store.collection_name = "shared"
    store._has_bm25_slot = True
    store._encode_bm25 = lambda query: SparseVector(indices=[1], values=[1.0])

    def swallowed(**kwargs):
        assert store.keyword_search("fixed query") is None
        return {"results": []}

    client._memory.search = swallowed
    with pytest.raises(ro.Mem0UnresolvedRequestError, match="may still be running"):
        client.search("query", filters={"user_id": "user_a", "run_id": "source_run"})
    assert network.call_count == 1
    with pytest.raises(ro.Mem0UnresolvedRequestError):
        client.close(0)


class SnapshotAdmin(FakeQdrant):
    def __init__(self):
        super().__init__()
        self.names = {"source"}

    def collection_exists(self, collection_name):
        return collection_name in self.names

    def recover_snapshot(self, *, collection_name, location):
        super().recover_snapshot(collection_name=collection_name, location=location)
        self.names.add(collection_name)

    def scroll(self, **kwargs):
        return [], None

    def delete_collection(self, *, collection_name):
        super().delete_collection(collection_name=collection_name)
        self.names.remove(collection_name)


@pytest.mark.parametrize("condition", ["mem0", "simple_rag"])
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.RemoteProtocolError])
@pytest.mark.asyncio
async def test_owner_retains_namespace_after_local_thread_ended(tmp_path, monkeypatch, condition, error):
    run, ledger, admin_client = _mem0_run(tmp_path)
    admin = SnapshotAdmin()
    admin_client.qdrant_admin = admin
    manager = _external_manager(run, ledger, condition=condition, namespace="source", client=admin_client)
    manager.save("acc_108", pinned=True)
    clients = []

    def factory(namespace, directory, source_run_id):
        client, _, _ = _client(monkeypatch, error, body=True, namespace=namespace, run_id=RUN_ID)
        clients.append(client)
        return SharedQueryResource(client=client, new_adapter=lambda: Mem0Adapter(client=client))

    manager.shared_query_factory = factory
    owner = SharedProbeMemoryOwner(tmp_path / "shared")
    branch = tmp_path / "repeat02"
    await owner.branch(manager, "acc_108", branch)
    client = clients[0]
    with pytest.raises(ResponseHandlingException):
        await asyncio.to_thread(client.search, "query", filters={"user_id": "user_a", "run_id": RUN_ID})
    assert client._active == 0
    owner.release_consumer(branch, failed=True)
    assert not await owner.close()
    entry = next(iter(owner._entries.values()))
    assert entry["record"]["state"] == "cleanup_pending"
    assert "may still be running" in entry["record"]["error"]
    assert entry["record"]["namespace"] in admin.names
    assert not any(call[0] == "delete_collection" for call in admin.calls)
    assert client._closed
