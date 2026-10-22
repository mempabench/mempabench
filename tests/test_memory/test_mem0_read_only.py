"""No-network checks of shared Probe reads and their lifecycle boundary."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from memory.backends import mem0_read_only as ro
from memory.backends.mem0_adapter import Mem0Adapter
from memory.backends.mem0_client import create_read_only_mem0_client, _ValidatedMem0Client
from memory.scope import AdapterMode, MemoryScope, RetrieveRequest, MemoryRetrievalBudget


_SCOPE = {"user_id": "user_a", "run_id": "source_run"}


@pytest.mark.parametrize("method,suffix", [
    ("PUT", ""), ("DELETE", ""), ("PATCH", ""),
    ("PUT", "/points"), ("POST", "/points/delete"),
    ("POST", "/points/payload"), ("PUT", "/index"),
    ("POST", "/snapshots"), ("PUT", "/snapshots/recover"),
])
def test_transport_rejects_writes_before_any_network(method, suffix):
    network = Mock(return_value=httpx.Response(200))
    transport = ro.ReadOnlyQdrantTransport({"shared"}, httpx.MockTransport(network))
    with pytest.raises(PermissionError, match="request refused"):
        transport.handle_request(httpx.Request(method, f"http://qdrant/collections/shared{suffix}"))
    network.assert_not_called()


@pytest.mark.parametrize("path", [
    "/collections/other/points/query", "/collections/shared_entities/points/query",
    "/collections/shared%2Fother/points/query", "/collections",
    "/collections/shared/points/query/../delete",
])
def test_transport_rejects_other_namespaces_and_unknown_routes(path):
    network = Mock(return_value=httpx.Response(200))
    transport = ro.ReadOnlyQdrantTransport({"shared"}, httpx.MockTransport(network))
    with pytest.raises(PermissionError):
        transport.handle_request(httpx.Request("POST", "http://qdrant" + path))
    network.assert_not_called()


def test_real_qdrant_sdk_cannot_bypass_transport_write_guard(monkeypatch):
    from qdrant_client.http.exceptions import ResponseHandlingException

    network = Mock(return_value=httpx.Response(200, json={"result": {"exists": True}, "status": "ok", "time": 0.0}))
    monkeypatch.setattr(ro.httpx, "HTTPTransport", lambda: httpx.MockTransport(network))
    client = ro._query_qdrant(host="qdrant.invalid", port=6333, namespaces={"shared"})
    assert client.collection_exists("shared") is True
    assert network.call_count == 1
    for mutate in (
        lambda: client.delete_collection("shared"),
        lambda: client.create_payload_index("shared", "run_id", "keyword"),
        lambda: client.recover_snapshot("shared", "file:///snapshot"),
        lambda: client.collection_exists("other"),
    ):
        with pytest.raises((PermissionError, ResponseHandlingException)):
            mutate()
    assert network.call_count == 1
    client.close()


def test_query_body_cannot_lookup_another_collection():
    network = Mock(return_value=httpx.Response(200))
    transport = ro.ReadOnlyQdrantTransport({"shared"}, httpx.MockTransport(network))
    request = httpx.Request("POST", "http://qdrant/collections/shared/points/query", json={
        "prefetch": [{"lookup_from": {"collection": "other"}}],
    })
    with pytest.raises(PermissionError):
        transport.handle_request(request)
    assert transport.denied_requests == 1
    network.assert_not_called()


class FakeQdrant:
    def __init__(self, *, bm25, entity_exists):
        self.bm25 = bm25
        self.names = {"shared"} | ({"shared_entities"} if entity_exists else set())
        self.reads = []
        self.writes = []
        self.closed = 0
        self.points = [
            SimpleNamespace(id="first", score=0.8, payload={
                **_SCOPE, "data": "Use concise answers", "role": "user",
                "session_id": "acc_001", "turn_id": "turn_1",
            }),
            SimpleNamespace(id="second", score=0.7, payload={
                **_SCOPE, "data": "Explain critical assumptions", "role": "assistant",
                "session_id": "acc_002", "turn_id": "turn_2",
            }),
        ]
        self.entities = ([SimpleNamespace(id="entity", score=0.95, payload={
            **_SCOPE, "data": "Alice", "linked_memory_ids": ["second"],
        })] if entity_exists else [])

    def get_collection(self, collection_name):
        assert collection_name in self.names
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
            vectors=SimpleNamespace(size=2, distance=SimpleNamespace(value="Cosine"), on_disk=True),
            sparse_vectors={"bm25": {}} if self.bm25 else None,
        )))

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.names])

    def collection_exists(self, collection_name):
        return collection_name in self.names

    def scroll(self, collection_name, **kwargs):
        return (self.entities if collection_name.endswith("_entities") else self.points), None

    def create_collection(self, collection_name, **kwargs):
        self.writes.append(("create_collection", collection_name))
        self.names.add(collection_name)

    def create_payload_index(self, **kwargs):
        self.writes.append(("index", kwargs))

    def query_points(self, collection_name, **kwargs):
        self.reads.append((collection_name, kwargs))
        if collection_name.endswith("_entities"):
            return SimpleNamespace(points=self.entities)
        if kwargs.get("using") == "bm25":
            return SimpleNamespace(points=[SimpleNamespace(id="second", score=3.0)])
        return SimpleNamespace(points=self.points)

    def close(self):
        self.closed += 1


def _storage_kwargs(tmp_path, *, bm25=False, entity=False):
    return dict(
        collection_name="shared", qdrant_host="qdrant.invalid", qdrant_port=6333,
        expected_run_id="source_run", allow_existing_collection=True,
        embedding_api_base="https://embedding.invalid/v1", embedding_provider="openai",
        embedding_model="text-embedding-3-small", embedding_dimensions=2,
        qdrant_distance="Cosine", qdrant_on_disk=True, qdrant_bm25_enabled=bm25,
        enable_entity_memory=entity, history_db_path=tmp_path / "history.sqlite3",
        allow_existing_history_db=True, require_existing_history_db=False,
    )


@pytest.mark.parametrize("bm25,entity,entity_exists", [
    (False, False, False), (True, False, False),
    (True, True, True), (True, True, False),
])
@pytest.mark.asyncio
async def test_factory_reuses_resources_and_matches_vendor_search(
    monkeypatch, tmp_path, bm25, entity, entity_exists,
):
    from mem0 import Memory
    from mem0.memory import main as vendor
    from mem0.utils.factory import EmbedderFactory, VectorStoreFactory
    from mem0.vector_stores.qdrant import Qdrant

    monkeypatch.setenv("OPENAI_EMB_API_KEY", "test-key")
    monkeypatch.setattr(vendor, "MEM0_TELEMETRY", False)
    monkeypatch.setattr("mem0.memory.telemetry.MEM0_TELEMETRY", False)
    monkeypatch.setattr(vendor, "lemmatize_for_bm25", lambda query: query)
    monkeypatch.setattr(vendor, "extract_entities", lambda query: [("PERSON", "Alice")])
    monkeypatch.setattr(ro, "_require_entity_nlp", lambda: None)
    import numpy as np
    encoder = SimpleNamespace(embed=lambda texts: [SimpleNamespace(
        indices=np.array([1]), values=np.array([1.0]),
    )])
    monkeypatch.setattr(Qdrant, "_get_bm25_encoder", lambda self: encoder)
    qdrant = FakeQdrant(bm25=bm25, entity_exists=entity_exists)
    opens = Mock(return_value=qdrant)
    monkeypatch.setattr(ro, "_query_qdrant", opens)
    embedded = []

    def embed(query, memory_action):
        embedded.append(query)
        return [1.0, 0.0]

    embedder = SimpleNamespace(embed=embed, client=Mock())
    embeddings = Mock(return_value=embedder)
    monkeypatch.setattr(EmbedderFactory, "create", embeddings)
    # The new constructor must not allocate a vendor Memory/SQLite instance.
    original_init = Memory.__init__
    init = Mock(side_effect=AssertionError("query factory used full Memory init"))
    monkeypatch.setattr(Memory, "__init__", init)
    client = create_read_only_mem0_client(
        infer=entity, **_storage_kwargs(tmp_path, bm25=bm25, entity=entity),
        allow_missing_entity_collection=entity and not entity_exists,
    )
    assert opens.call_count == embeddings.call_count == 1
    init.assert_not_called()
    assert client._memory.db is None and client._memory.llm is None
    assert qdrant.writes == []

    shared_adapters = [Mem0Adapter(client=client, infer=entity) for _ in range(2)]
    scope = MemoryScope(run_id="source_run", persona_id="user_a")
    request = RetrieveRequest(
        query="Alice response style", scope=scope, session_id="probe",
        budget=MemoryRetrievalBudget(max_items=2, max_tokens=4000),
    )
    await shared_adapters[0].setup_scope(scope)
    await shared_adapters[0].set_mode(AdapterMode(read_only=True, reason="repeat02"))
    assert shared_adapters[1]._scope is None
    assert shared_adapters[1]._mode.read_only is False
    first, second = await asyncio.gather(*(adapter.retrieve(request) for adapter in shared_adapters))
    assert first.raw == second.raw and first.raw is not second.raw
    assert first.trace is not second.trace
    assert embedded.count(request.query) == 2
    assert opens.call_count == embeddings.call_count == 1
    assert qdrant.writes == []

    # Independently build the original vendor Memory through its normal
    # constructor, using the same deterministic transport and embedding data.
    monkeypatch.setattr(Memory, "__init__", original_init)
    monkeypatch.setattr(VectorStoreFactory, "create", lambda provider, cfg: Qdrant(
        collection_name=cfg.collection_name, embedding_model_dims=cfg.embedding_model_dims,
        client=qdrant, on_disk=cfg.on_disk, enable_bm25=cfg.enable_bm25,
    ))
    baseline = Memory(config=client._memory.config)
    baseline_client = _ValidatedMem0Client(
        baseline, collection_name="shared", qdrant_host="qdrant.invalid", qdrant_port=6333,
        expected_run_id="source_run", embedding_dimensions=2, qdrant_distance="Cosine",
        qdrant_on_disk=True, qdrant_bm25_enabled=bm25, enable_entity_memory=entity,
    )
    original_adapter = Mem0Adapter(client=baseline_client, infer=entity)
    original = await original_adapter.retrieve(request)
    assert first.raw == original.raw
    assert shared_adapters[0].format_context(first) == original_adapter.format_context(original)
    assert first.trace.native_request == original.trace.native_request
    assert first.trace.native_response == original.trace.native_response
    baseline.close()
    assert client.close(1) is True


@pytest.mark.parametrize("failure", ["lost_entity", "wrong_run"])
def test_factory_fails_closed_before_embedding_setup(monkeypatch, tmp_path, failure):
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "test-key")
    monkeypatch.setattr(ro, "_require_entity_nlp", lambda: None)
    qdrant = FakeQdrant(bm25=False, entity_exists=False)
    if failure == "wrong_run":
        qdrant.points[0].payload["run_id"] = "different_source"
    monkeypatch.setattr(ro, "_query_qdrant", lambda **kwargs: qdrant)
    from mem0.utils.factory import EmbedderFactory
    embeddings = Mock(side_effect=AssertionError("opened embedding before failed source check"))
    monkeypatch.setattr(EmbedderFactory, "create", embeddings)
    with pytest.raises((ValueError, FileNotFoundError)):
        create_read_only_mem0_client(infer=True, **_storage_kwargs(tmp_path, entity=True))
    embeddings.assert_not_called()
    assert qdrant.closed == 1 and qdrant.writes == []


def _blocking_client(entered, release):
    def search(**kwargs):
        entered.set()
        assert release.wait(5)
        return {"results": []}

    qdrant = Mock()
    qdrant._probe_read_only_transport = None
    return ro.ReadOnlyMem0Client(
        SimpleNamespace(search=search, embedding_model=SimpleNamespace(client=Mock())),
        qdrant=qdrant, history_path=Path("/unused/history.sqlite3"),
        collection_name="shared", qdrant_host="unused", qdrant_port=6333,
        expected_run_id="source_run", embedding_dimensions=2, qdrant_distance="Cosine",
        qdrant_on_disk=True, qdrant_bm25_enabled=False, enable_entity_memory=False,
    )


@pytest.mark.asyncio
async def test_cancelled_awaiter_does_not_allow_closing_active_search():
    entered, release = threading.Event(), threading.Event()
    client = _blocking_client(entered, release)
    task = asyncio.create_task(asyncio.to_thread(client.search, "query", filters=_SCOPE))
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.close(0) is False
    client._qdrant.close.assert_not_called()
    with pytest.raises(RuntimeError, match="draining"):
        client.search("next", filters=_SCOPE)
    release.set()
    assert await asyncio.to_thread(client.close, 2) is True
    assert client.close(0) is True
    client._qdrant.close.assert_called_once()
    client._memory.embedding_model.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_reset_is_rejected_even_if_adapter_read_only_mode_is_lost():
    client = _blocking_client(threading.Event(), threading.Event())
    adapter = Mem0Adapter(client=client)
    scope = MemoryScope(run_id="source_run", persona_id="user_a")
    with pytest.raises(PermissionError):
        await adapter.reset_scope(scope)
    with pytest.raises(PermissionError):
        client.add([{"role": "user", "content": "write"}], **_SCOPE)
    await adapter.set_mode(AdapterMode(read_only=True))
    with pytest.raises(PermissionError, match="read-only"):
        await adapter.reset_scope(scope)
    client.close()


def test_vendor_cannot_swallow_a_rejected_write_and_return_partial_results():
    client = _blocking_client(threading.Event(), threading.Event())
    guard = ro.ReadOnlyQdrantTransport({"shared"}, httpx.MockTransport(Mock()))
    client._qdrant._probe_read_only_transport = guard

    def search(**kwargs):
        try:
            guard.handle_request(httpx.Request("DELETE", "http://qdrant/collections/shared"))
        except PermissionError:
            pass
        return {"results": []}

    client._memory.search = search
    with pytest.raises(PermissionError, match="forbidden"):
        client.search("query", filters=_SCOPE)
    client.close()


@pytest.mark.parametrize("infer", [False, True])
@pytest.mark.asyncio
async def test_two_repeat_identities_read_original_source_scope(infer):
    client = _blocking_client(threading.Event(), threading.Event())
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return {"results": [{
            "id": "original_memory", "memory": "Use concise answers", "score": 0.9,
            **_SCOPE, "role": "user", "metadata": {"session_id": "acc_001", "turn_id": "turn_1"},
        }]}

    client._memory.search = search
    scopes = [MemoryScope(run_id=repeat, persona_id="user_a") for repeat in ("repeat02", "repeat03")]
    adapters = [Mem0Adapter(client=client, infer=infer, retrieval_run_id="source_run") for _ in scopes]
    requests = [RetrieveRequest(
        query="response style", scope=scope, session_id="final_probe_01",
        budget=MemoryRetrievalBudget(max_items=2),
    ) for scope in scopes]
    for adapter, scope in zip(adapters, scopes):
        await adapter.setup_scope(scope)
        await adapter.set_mode(AdapterMode(read_only=True))
    results = await asyncio.gather(*(adapter.retrieve(request) for adapter, request in zip(adapters, requests)))
    assert len(calls) == 2
    assert all(call["filters"] == _SCOPE for call in calls)
    assert [result.trace.scope.run_id for result in results] == ["repeat02", "repeat03"]
    assert [adapter._scope.run_id for adapter in adapters] == ["repeat02", "repeat03"]
    assert all(result.raw[0]["run_id"] == "source_run" for result in results)
    assert all(result.trace.native_request["filters"]["run_id"] == "source_run" for result in results)
    assert [request.scope.run_id for request in requests] == ["repeat02", "repeat03"]
    client.close()


@pytest.mark.asyncio
async def test_source_scope_override_does_not_change_read_write_retrieval():
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return {"results": []}

    adapter = Mem0Adapter(client=SimpleNamespace(search=search), retrieval_run_id="source_run")
    scope = MemoryScope(run_id="ordinary_run", persona_id="user_a")
    result = await adapter.retrieve(RetrieveRequest(
        query="response style", scope=scope, session_id="acc_001",
        budget=MemoryRetrievalBudget(max_items=2),
    ))
    assert calls[0]["filters"]["run_id"] == result.trace.scope.run_id == "ordinary_run"
