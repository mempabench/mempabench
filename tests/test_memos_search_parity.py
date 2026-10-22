"""Exercise the pinned product/search algorithm without a service or provider.

This covers the deployed fast/text/mmr configuration with an empty plugin hook
registry. It does not certify fine/mixture search, arbitrary runtime plugins, or
live database drivers. Only storage reads and query embeddings are substitutes;
handler, cube view, tree search, recall, cosine reranking and MMR remain real.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import httpx
import pytest
import yaml

import memos.api.handlers  # noqa: F401 - resolve upstream handler import cycle
from memos.api.config import APIConfig
from memos.api.handlers.base_handler import HandlerDependencies
from memos.api.handlers.search_handler import SearchHandler
from memos.api.product_models import APISearchRequest
from memos.graph_dbs.neo4j_community import Neo4jCommunityGraphDB
from memos.memories.textual.tree import TreeTextMemory
from memos.memories.textual.tree_text_memory.retrieve.searcher import Searcher
from memos.plugins import hooks
from memos.reranker.cosine_local import CosineLocalReranker

from memory.backends.memos_adapter import MemOSAdapter
from memory.backends.memos_client import MemOSHttpClient
from memory.scope import MemoryRetrievalBudget, MemoryScope, RetrieveRequest


ROOT = Path(__file__).resolve().parents[1]
CUBE = "offline_frozen_cube"
SOURCE_SESSION = "original_main"
QUERY = "What are the user's project and travel preferences?"


@pytest.fixture
def deployed_config(monkeypatch):
    """Evaluate actual API config from committed compose constants, offline."""
    env = yaml.safe_load((ROOT / "deploy/memos/compose.yaml").read_text())[
        "services"
    ]["memos"]["environment"]
    for key, value in env.items():
        if "${" not in str(value):
            monkeypatch.setenv(key, str(value))
    for key in ("FAST_GRAPH", "BM25_CALL", "VEC_COT_CALL", "FULLTEXT_CALL", "INCLUDE_EMBEDDING"):
        monkeypatch.delenv(key, raising=False)
    for key in ("OPENAI_API_KEY", "MOS_EMBEDDER_API_KEY"):
        monkeypatch.setenv(key, "offline-test-placeholder")
    monkeypatch.setenv("OPENAI_API_BASE", "http://provider.invalid/v1")
    monkeypatch.setenv("MOS_CHAT_MODEL", "offline-model")
    monkeypatch.setenv("NEO4J_PASSWORD", "offline-password")
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden")))
    config = APIConfig.get_default_cube_config()
    assert config.text_mem.backend == "tree_text"
    assert config.text_mem.config.graph_db.backend == "neo4j-community"
    assert config.text_mem.config.reranker.backend == "cosine_local"
    assert config.text_mem.config.search_strategy == {
        "fast_graph": False, "bm25": False, "cot": False, "fulltext": False,
    }
    assert not config.text_mem.config.reorganize
    assert config.text_mem.config.internet_retriever is None
    assert not APIConfig.is_scheduler_enabled()
    return config


def _node(number, text, vector, kind):
    return {
        "id": str(UUID(int=number)), "memory": text,
        "metadata": {
            "memory_type": kind, "embedding": vector, "source": "conversation",
            "user_id": "user_a", "session_id": SOURCE_SESSION,
            "created_at": "2026-09-20T00:00:00", "updated_at": "2026-09-20T00:00:00",
            "usage": ["original-consumption-evidence"],
        },
    }


@pytest.fixture
def product_search(monkeypatch, deployed_config):
    # Scrambled recall order, a repeated text with another ID, one low score,
    # three memory classes, and non-collinear vectors exercise actual ranking.
    nodes = [
        _node(3, "Travel plans allow flexible dates.", [0.6, 0.8, 0.0], "LongTermMemory"),
        _node(1, "The project deadline is Friday.", [1.0, 0.0, 0.0], "WorkingMemory"),
        _node(5, "An irrelevant old note.", [0.1, 0.9, 0.0], "UserMemory"),
        _node(2, "The user prefers quiet accommodation.", [0.8, 0.0, 0.6], "UserMemory"),
        _node(4, "The project deadline is Friday.", [0.7, 0.7, 0.0], "LongTermMemory"),
    ]
    unchanged = deepcopy(nodes)
    by_id = {node["id"]: node for node in nodes}
    vector = Mock(name="vector_store")

    def vector_search(*, query_vector, top_k, filter):
        assert query_vector == [1.0, 0.0, 0.0]
        assert filter["user_name"] == CUBE
        assert filter["status"] == "activated"
        assert filter["vector_sync"] == "success"
        # Scores intentionally differ from cosine reranking scores.
        return [SimpleNamespace(id=node["id"], score=0.95 - index * 0.02)
                for index, node in enumerate(nodes)
                if node["metadata"]["memory_type"] == filter["memory_type"]][:top_k]

    vector.search.side_effect = vector_search
    graph = Mock(spec=Neo4jCommunityGraphDB, name="graph_store")
    graph.vec_db = vector
    graph.config = SimpleNamespace(user_name=CUBE)
    graph.search_by_embedding.side_effect = lambda **kw: Neo4jCommunityGraphDB.search_by_embedding(graph, **kw)

    def get_nodes(ids, **kwargs):
        assert kwargs["user_name"] == CUBE
        assert kwargs["include_embedding"] is True
        return [deepcopy(by_id[node_id]) for node_id in reversed(ids)]

    def working(**kwargs):
        assert kwargs["user_name"] == CUBE
        return [deepcopy(node) for node in nodes if node["metadata"]["memory_type"] == kwargs["scope"]]

    graph.get_nodes.side_effect = get_nodes
    graph.get_all_memory_items.side_effect = working
    embedder = Mock(name="query_embedder")
    embedder.embed.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    llm = Mock(name="unused_dispatcher_llm")
    scheduler = Mock(name="disabled_scheduler")
    agent = Mock(name="unused_deepsearch_agent")
    reranker = CosineLocalReranker()
    tree = object.__new__(TreeTextMemory)
    tree.dispatcher_llm, tree.graph_store, tree.embedder = llm, graph, embedder
    tree.reranker, tree.bm25_retriever, tree.internet_retriever = reranker, None, None
    tree.search_strategy = deployed_config.text_mem.config.search_strategy
    tree.include_embedding, tree.tokenizer = False, None
    handler = SearchHandler(HandlerDependencies(
        naive_mem_cube=SimpleNamespace(text_mem=tree), mem_scheduler=scheduler,
        searcher=SimpleNamespace(graph_store=graph), deepsearch_agent=agent,
        reranker=reranker, graph_db=graph, vector_db=vector,
    ))
    searchers = []
    original_init = Searcher.__init__

    def observe_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._usage_executor.submit = Mock(wraps=self._usage_executor.submit)
        searchers.append(self)

    monkeypatch.setattr(Searcher, "__init__", observe_init)
    # Real hook decorator is exercised; production plugin discovery is outside
    # this offline test's scope and must not run during test import.
    monkeypatch.setattr(hooks, "_hooks", defaultdict(list))
    requests = []

    def transport(request):
        assert request.method == "POST" and request.url.path == "/product/search"
        body = json.loads(request.content)
        requests.append(deepcopy(body))
        parsed = APISearchRequest.model_validate(body)
        assert parsed.mode == "fast" and parsed.dedup == "mmr"
        assert parsed.relativity == 0.45
        assert not parsed.internet_search
        assert not parsed.include_preference and not parsed.search_tool_memory and not parsed.include_skill_memory
        response = handler.handle_search_memories(parsed)
        return httpx.Response(200, json=response.model_dump(mode="json"))

    admin = MemOSHttpClient(api_base="http://memos.invalid", transport=httpx.MockTransport(transport))
    shared = admin.read_only_for_cube(CUBE)
    state = SimpleNamespace(
        admin=admin, shared=shared, requests=requests, nodes=nodes, unchanged=unchanged,
        graph=graph, vector=vector, scheduler=scheduler, agent=agent, llm=llm,
        embedder=embedder, searchers=searchers,
    )
    try:
        yield state
    finally:
        shared.close()
        for searcher in searchers:
            searcher._usage_executor.shutdown(wait=True)


def _request(repeat, budget):
    return RetrieveRequest(
        query=QUERY, scope=MemoryScope(run_id=repeat, persona_id="user_a"),
        session_id=f"private-{repeat}", budget=MemoryRetrievalBudget(max_items=budget),
    )


def _adapter(client):
    return MemOSAdapter(client=client, cube_id=CUBE, memory_session_id=SOURCE_SESSION)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [2, 10])
async def test_real_fast_search_old_shared_adapter_multihit_parity_and_no_writes(product_search, budget):
    state = product_search
    old = _adapter(state.admin)
    expected = await old.retrieve(_request("original", budget))
    expected_context = old.format_context(expected)
    assert expected.trace.retrieved_count == min(budget, 3)
    assert expected.trace.top_score == pytest.approx(1.0)
    assert expected.trace.native_ids == [str(UUID(int=n)) for n in (1, 2, 3)][:budget]
    memories = expected.raw[0]["text_mem"][0]["memories"]
    assert [memory["metadata"]["relativity"] for memory in memories] == pytest.approx([1.0, 0.8, 0.6][:budget])
    assert all(memory["metadata"]["embedding"] == [] for memory in memories)
    assert str(UUID(int=4)) not in expected.trace.native_ids  # repeated text
    assert str(UUID(int=5)) not in expected.trace.native_ids  # relativity threshold
    assert "The project deadline is Friday." in expected_context
    assert "An irrelevant old note." not in expected_context

    results = []
    for repeat in ("repeat02", "repeat03"):
        adapter = _adapter(state.shared)
        request = _request(repeat, budget)
        await adapter.setup_scope(request.scope)
        actual = await adapter.retrieve(request)
        results.append(actual)
        assert actual.raw == expected.raw
        assert actual.trace.native_response == expected.trace.native_response
        assert actual.trace.native_request == expected.trace.native_request
        assert actual.trace.native_ids == expected.trace.native_ids
        assert actual.trace.top_score == expected.trace.top_score
        assert actual.trace.retrieved_count == expected.trace.retrieved_count
        assert actual.trace.warnings == expected.trace.warnings
        assert adapter.format_context(actual) == expected_context
        assert actual.trace.scope.run_id == repeat

    assert state.requests == [state.requests[0]] * 3
    assert len(state.searchers) == 3 and state.embedder.embed.call_count == 3
    assert state.requests[0]["session_id"] == SOURCE_SESSION
    assert state.requests[0]["chat_history"] == []
    assert state.nodes == state.unchanged
    assert state.vector.search.call_count >= 3
    assert {call[0] for call in state.vector.mock_calls} == {"search"}
    assert {call[0] for call in state.graph.mock_calls} == {
        "get_all_memory_items", "search_by_embedding", "get_nodes",
    }
    for component in (state.scheduler, state.agent, state.llm):
        assert component.mock_calls == []
    for searcher in state.searchers:
        searcher._usage_executor.submit.assert_not_called()
    results[0].raw[0]["text_mem"][0]["memories"][0]["memory"] = "private mutation"
    assert results[1].raw == expected.raw


@pytest.mark.asyncio
async def test_search_hooks_are_a_server_side_write_boundary_not_an_http_guarantee(product_search):
    """Fault injection records why arbitrary enabled plugins need separate review."""
    state = product_search

    def unexpected_write(*, request):
        state.graph.update_node(str(UUID(int=1)), {"usage": ["plugin-write"]}, user_name=CUBE)

    hooks.register_hook("search.before", unexpected_write)
    result = await _adapter(state.shared).retrieve(_request("repeat02", 10))
    assert result.trace.retrieved_count == 3
    state.graph.update_node.assert_called_once()
    assert len(state.requests) == 1
