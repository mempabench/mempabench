"""Shared MemOS transport lifecycle and request-boundary tests; no live service."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import threading
from unittest.mock import Mock

import httpx
import pytest

from memory.backends import memos_client, memos_read_only
from memory.backends.memos_adapter import MemOSAdapter
from memory.backends.memos_client import MemOSClientError, MemOSHttpClient
from memory.backends.memos_read_only import MemOSUnresolvedRequestError
from memory.events import MemoryEvent
from memory.results import WriteStatus
from memory.scope import AdapterMode, MemoryRetrievalBudget, MemoryScope, RetrieveRequest


def _response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    cube = (body.get("readable_cube_ids") or [body.get("mem_cube_id")])[0]
    return httpx.Response(200, json={
        "code": 200,
        "message": "ok",
        "data": {
            "text_mem": [{
                "cube_id": cube,
                "memories": [{
                    "id": "memory-1", "memory": body.get("query", "memory"),
                    "metadata": {"relativity": 0.8},
                }],
            }],
        },
    })


def _admin(handler=_response) -> MemOSHttpClient:
    return MemOSHttpClient(
        api_base="http://memos.test:8000", transport=httpx.MockTransport(handler)
    )


def _search(cube="cube-1", query="query"):
    return {
        "query": query, "user_id": "user_a", "session_id": "original_main",
        "readable_cube_ids": [cube], "filter": {"and": [{"k": "v"}]},
    }


def _adapter(client):
    return MemOSAdapter(client=client, cube_id="cube-1", memory_session_id="original_main")


def _request(repeat):
    return RetrieveRequest(
        query=f"query-{repeat}",
        scope=MemoryScope(run_id=repeat, persona_id="user_a"),
        session_id=f"execution-{repeat}",
        budget=MemoryRetrievalBudget(max_items=2),
    )


def test_one_pool_for_both_repeats_and_no_query_result_cache(monkeypatch):
    constructors = []
    requests = []
    original = httpx.Client

    def construct(**kwargs):
        client = original(**kwargs)
        constructors.append(client)
        return client

    def handler(request):
        requests.append(request)
        return _response(request)

    monkeypatch.setattr(memos_read_only.httpx, "Client", construct)
    client = _admin(handler).read_only_for_cube("cube-1")
    assert not requests  # Opening query resources does not initialize a cube.
    try:
        for _repeat in ("repeat02", "repeat03"):
            client.search_memories(_search())
        assert len(constructors) == 1
        assert len(requests) == 2
        assert not constructors[0].is_closed
    finally:
        client.close()
    assert constructors[0].is_closed
    client.close()


@pytest.mark.parametrize("operation", [
    "add", "delete_memory", "delete_cube", "import_cube", "export_cube",
    "reset", "update", "init", "search/../add", "search?operation=add",
])
def test_write_management_and_nonexact_routes_rejected_before_transport(operation):
    handler = Mock(side_effect=_response)
    client = _admin(handler).read_only_for_cube("cube-1")
    try:
        with pytest.raises(PermissionError):
            client._request_json(
                "POST", f"/product/{operation}", operation=operation,
                json_body={**_search(), "writable_cube_ids": ["cube-1"]},
            )
        handler.assert_not_called()
    finally:
        client.close()


def test_public_write_methods_and_cube_escalation_rejected():
    handler = Mock(side_effect=_response)
    client = _admin(handler).read_only_for_cube("cube-1")
    try:
        operations = [
            lambda: client.add_memories({"writable_cube_ids": ["cube-1"]}),
            lambda: client.delete_memories({"writable_cube_ids": ["cube-1"], "memory_ids": ["m"]}),
            lambda: client.import_cube_as({}, "cube-1"),
            lambda: client.delete_cube("cube-1"),
            lambda: client.export_cube("cube-1"),
            lambda: client.search_memories(_search("other-cube")),
            lambda: client.get_memories({"mem_cube_id": "other-cube"}),
            lambda: client.read_only_for_cube("other-cube"),
            lambda: client._request_json("POST", "/product/search", operation="add", json_body=_search()),
        ]
        for operation in operations:
            with pytest.raises(PermissionError):
                operation()
        handler.assert_not_called()
    finally:
        client.close()


def test_shared_and_original_search_preserve_payload_result_and_filters():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _response(request)

    admin = _admin(handler)
    shared = admin.read_only_for_cube("cube-1")
    try:
        assert admin.search_memories(_search()) == shared.search_memories(_search())
        assert seen == [_search(), _search()]
        assert shared.get_memories({"mem_cube_id": "cube-1"})["text_mem"][0]["cube_id"] == "cube-1"
    finally:
        shared.close()


def test_eight_settings_have_independent_cube_bindings_and_pool_lifetimes():
    barrier = threading.Barrier(8)

    def handler(request):
        barrier.wait(timeout=3)
        return _response(request)

    clients = [_admin(handler).read_only_for_cube(f"cube-{index}") for index in range(8)]
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = [
                pool.submit(client.search_memories, _search(f"cube-{index}"))
                for index, client in enumerate(clients)
            ]
            results = [future.result(timeout=4) for future in pending]
        assert [result["text_mem"][0]["cube_id"] for result in results] == [f"cube-{index}" for index in range(8)]
        clients[0].close()
        assert all(not client._http.is_closed for client in clients[1:])
    finally:
        for client in clients:
            client.close()


def test_pinned_memos_query_usage_hook_does_not_write_or_schedule_work():
    from memos.memories.textual.tree_text_memory.retrieve.searcher import Searcher

    searcher = object.__new__(Searcher)
    searcher.graph_store = Mock()
    searcher._usage_executor = Mock()
    item = Mock()
    searcher._update_usage_history([item], {"session_id": "original_main"}, "cube-1")
    searcher.graph_store.assert_not_called()
    assert not searcher.graph_store.mock_calls
    assert not searcher._usage_executor.mock_calls
    assert not item.mock_calls


@pytest.mark.asyncio
async def test_two_adapters_keep_original_retrieval_session_and_independent_traces():
    barrier = threading.Barrier(2)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        barrier.wait(timeout=3)
        return _response(request)

    client = _admin(handler).read_only_for_cube("cube-1")
    first, second = _adapter(client), _adapter(client)
    await first.setup_scope(_request("repeat02").scope)
    await second.setup_scope(_request("repeat03").scope)
    await first.set_mode(AdapterMode(read_only=True, reason="repeat02"))
    try:
        results = await asyncio.gather(
            first.retrieve(_request("repeat02")), second.retrieve(_request("repeat03"))
        )
        assert [result.trace.scope.run_id for result in results] == ["repeat02", "repeat03"]
        assert first._scope != second._scope
        assert first._adapter_mode != second._adapter_mode
        assert {body["session_id"] for body in seen} == {"original_main"}
        assert all(body["chat_history"] == [] for body in seen)
        assert all(body["readable_cube_ids"] == ["cube-1"] for body in seen)
        assert [result.trace.native_request["query"] for result in results] == ["query-repeat02", "query-repeat03"]
        results[0].trace.native_response["text_mem"][0]["memories"][0]["memory"] = "changed"
        assert "changed" not in str(results[1].trace.native_response)
    finally:
        client.close()


@pytest.mark.asyncio
async def test_shared_adapter_writes_skip_and_reset_rejects_even_without_mode_flag():
    handler = Mock(side_effect=_response)
    client = _admin(handler).read_only_for_cube("cube-1")
    adapter = _adapter(client)
    event = MemoryEvent(
        event_id="e", run_id="r", persona_id="user_a", agent_id="pa",
        session_id="s", turn_id="t", timestamp=datetime.now(timezone.utc),
        messages=[{"role": "user", "content": "hello"}],
    )
    try:
        assert (await adapter.record_event(event)).status == WriteStatus.skipped_read_only
        with pytest.raises(PermissionError, match="read-only"):
            await adapter.reset_scope(_request("repeat02").scope)
        handler.assert_not_called()
    finally:
        client.close()


@pytest.mark.asyncio
async def test_cancelled_await_does_not_drain_running_http_worker():
    entered, finish = threading.Event(), threading.Event()

    def handler(request):
        entered.set()
        if not finish.wait(timeout=5):
            raise AssertionError("test did not release HTTP worker")
        return _response(request)

    client = _admin(handler).read_only_for_cube("cube-1")
    task = asyncio.create_task(_adapter(client).retrieve(_request("repeat02")))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.active_requests == 1
        with pytest.raises(TimeoutError, match="retain the shared cube"):
            client.close(timeout_seconds=0)
        assert not client._http.is_closed
        with pytest.raises(MemOSClientError, match="draining"):
            client.search_memories(_search())
    finally:
        finish.set()
        await asyncio.to_thread(client.close, 2)
    assert client.active_requests == 0
    assert client._http.is_closed


def test_close_waits_until_the_actual_worker_finishes():
    entered, finish, closing = threading.Event(), threading.Event(), threading.Event()

    def handler(request):
        entered.set()
        assert finish.wait(timeout=3)
        return _response(request)

    client = _admin(handler).read_only_for_cube("cube-1")
    with ThreadPoolExecutor(max_workers=2) as pool:
        query = pool.submit(client.search_memories, _search())
        try:
            assert entered.wait(timeout=2)

            def close():
                closing.set()
                client.close(timeout_seconds=2)

            closed = pool.submit(close)
            assert closing.wait(timeout=2)
            assert not closed.done()
            assert not client._http.is_closed
        finally:
            finish.set()
        query.result(timeout=2)
        closed.result(timeout=2)
    assert client._http.is_closed


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectTimeout])
def test_retry_success_does_not_hide_potentially_running_server_request(monkeypatch, failure):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure("test timeout", request=request)
        return _response(request)

    monkeypatch.setattr(memos_client, "_RETRY_PAUSE_SECONDS", 0)
    client = _admin(handler).read_only_for_cube("cube-1")
    assert client.search_memories(_search())["text_mem"]
    assert calls == 2
    if failure is httpx.ReadTimeout:
        assert client.unresolved_remote_requests == 1
        with pytest.raises(MemOSUnresolvedRequestError, match="retain the shared cube"):
            client.drain()
        with pytest.raises(MemOSUnresolvedRequestError):
            client.close()
    else:
        assert client.unresolved_remote_requests == 0
        client.close()
    assert client._http.is_closed
