import asyncio
import json
import os

import pytest

from harness.runtime_activity import Activity, observe_model_calls


def test_concurrent_waits_and_exception_leave_accurate_state_without_payload(tmp_path):
    activity = Activity(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def provider(prompt, api_key):
        started.set()
        await release.wait()
        raise ValueError("provider failed")

    async def exercise():
        task = asyncio.create_task(activity.wrap(provider, "simulator")("PRIVATE PROMPT", "PRIVATE KEY"))
        await started.wait()
        with activity.request("runtime_judge"):
            text = activity.path.read_text()
            row = json.loads(text)
            assert len(row["requests"]) == 2 and row["pid"] == os.getpid()
            assert "PRIVATE" not in text
        release.set()
        with pytest.raises(ValueError, match="provider failed"):
            await task
        assert json.loads(activity.path.read_text())["requests"] == []

    asyncio.run(exercise())


def test_stream_close_clears_wait_and_closes_original_stream(tmp_path):
    activity = Activity(tmp_path)
    closed = []

    async def provider():
        try:
            yield "one"
            yield "two"
        finally:
            closed.append(True)

    async def exercise():
        stream = activity.wrap(provider, "pa_or_memory_lm")()
        assert await anext(stream) == "one"
        assert len(json.loads(activity.path.read_text())["requests"]) == 1
        await stream.aclose()
        assert closed == [True]
        assert json.loads(activity.path.read_text())["requests"] == []

    asyncio.run(exercise())


def test_scoped_hooks_restore_provider_methods_even_after_failure(tmp_path):
    from pa.providers.openai_compat_provider import OpenAICompatProvider
    original = OpenAICompatProvider.chat
    with pytest.raises(RuntimeError):
        with observe_model_calls(tmp_path):
            assert OpenAICompatProvider.chat is not original
            raise RuntimeError("stop")
    assert OpenAICompatProvider.chat is original
    assert json.loads((tmp_path / "runtime_activity.json").read_text())["state"] == "stopped"


@pytest.mark.asyncio
async def test_parallel_run_contexts_attribute_requests_without_reinstalling_hooks(tmp_path, monkeypatch):
    from harness.runtime_activity import observe_run
    from pa.providers.openai_compat_provider import OpenAICompatProvider

    both, release = asyncio.Event(), asyncio.Event()
    entered = []

    async def provider(self, payload):
        entered.append(payload)
        if len(entered) == 2:
            both.set()
        await release.wait()
        return payload

    monkeypatch.setattr(OpenAICompatProvider, "chat", provider)
    runs = [tmp_path / name for name in ("repeat02", "repeat03")]
    for run in runs:
        run.mkdir()

    async def call(run):
        with observe_run(run):
            return await OpenAICompatProvider.chat(None, "PRIVATE " + run.name)

    with observe_model_calls(tmp_path):
        installed = OpenAICompatProvider.chat
        tasks = [asyncio.create_task(call(run)) for run in runs]
        await asyncio.wait_for(both.wait(), 2)
        aggregate = json.loads((tmp_path / "runtime_activity.json").read_text())
        assert {row["run_dir"] for row in aggregate["requests"]} == {str(run) for run in runs}
        for run in runs:
            assert len(json.loads((run / "runtime_activity.json").read_text())["requests"]) == 1
        assert OpenAICompatProvider.chat is installed
        assert "PRIVATE" not in (tmp_path / "runtime_activity.json").read_text()
        release.set()
        assert await asyncio.gather(*tasks) == ["PRIVATE repeat02", "PRIVATE repeat03"]
    assert OpenAICompatProvider.chat is provider
    for run in runs:
        row = json.loads((run / "runtime_activity.json").read_text())
        assert row["state"] == "stopped" and row["requests"] == []


@pytest.mark.asyncio
async def test_readonly_mem0_search_has_one_record_and_thread_context_per_call(tmp_path):
    import threading
    from harness.runtime_activity import observe_run
    from memory.backends.mem0_client import Mem0ResponseError
    from test_memory.test_mem0_read_only import _blocking_client, _SCOPE

    entered, release = threading.Event(), threading.Event()
    client = _blocking_client(entered, release)
    run = tmp_path / "repeat02"
    run.mkdir()
    try:
        with observe_model_calls(tmp_path), observe_run(run):
            task = asyncio.create_task(asyncio.to_thread(client.search, "PRIVATE QUERY", filters=_SCOPE))
            assert await asyncio.to_thread(entered.wait, 2)
            aggregate = json.loads((tmp_path / "retrieval_activity.json").read_text())
            assert len(aggregate["requests"]) == 1 and aggregate["requests"][0]["run_dir"] == str(run)
            assert len(json.loads((run / "retrieval_activity.json").read_text())["requests"]) == 1
            release.set()
            assert await task == {"results": []}
            with pytest.raises(Mem0ResponseError):
                await asyncio.to_thread(client.search, "", filters=_SCOPE)
    finally:
        release.set()
        client.close()
    for directory in (tmp_path, run):
        text = (directory / "retrieval_activity.jsonl").read_text()
        records = [json.loads(line) for line in text.splitlines()]
        assert len(records) == 2  # The subclass and its super call are not counted twice.
        assert [row["success"] for row in records] == [True, False]
        assert records[1]["error_type"] == "Mem0ResponseError"
        assert all(row["elapsed_ms"] >= 0 and row["run_dir"] == str(run) for row in records)
        assert "PRIVATE" not in text and "query" not in records[0]
        final = json.loads((directory / "retrieval_activity.json").read_text())
        assert final["state"] == "stopped" and final["requests"] == []


def test_readonly_memos_inherited_search_records_once_including_internal_retry(tmp_path, monkeypatch):
    import httpx
    from harness.runtime_activity import observe_run
    from memory.backends import memos_client
    from memory.backends.memos_client import MemOSHttpClient

    calls = []
    monkeypatch.setattr(memos_client, "_RETRY_PAUSE_SECONDS", 0)

    def search(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("local fake timeout", request=request)
        return httpx.Response(200, json={"code": 200, "message": "ok", "data": {
            "text_mem": [{"cube_id": "shared", "memories": []}]}})

    run = tmp_path / "repeat03"
    run.mkdir()
    client = MemOSHttpClient(api_base="http://unused.test", transport=httpx.MockTransport(search)).read_only_for_cube("shared")
    try:
        with observe_model_calls(tmp_path), observe_run(run):
            result = client.search_memories({"query": "PRIVATE QUERY", "readable_cube_ids": ["shared"]})
        assert result == {"text_mem": [{"cube_id": "shared", "memories": []}]} and len(calls) == 2
        for directory in (tmp_path, run):
            text = (directory / "retrieval_activity.jsonl").read_text()
            records = [json.loads(line) for line in text.splitlines()]
            assert len(records) == 1 and records[0]["backend"] == "memos"
            assert records[0]["boundary"] == "native_logical_search" and records[0]["success"]
            assert "PRIVATE" not in text
    finally:
        client.close()


@pytest.mark.asyncio
async def test_scoped_thread_is_marked_draining_until_it_actually_exits(tmp_path):
    import threading
    from harness.runtime_activity import observe_run

    activity = Activity(tmp_path)
    entered, release = threading.Event(), threading.Event()
    run = tmp_path / "repeat02"
    run.mkdir()

    def call():
        entered.set()
        assert release.wait(3)

    with observe_run(run):
        task = asyncio.create_task(asyncio.to_thread(activity.wrap(call, "simulator")))
        assert await asyncio.to_thread(entered.wait, 2)
    assert json.loads((run / "runtime_activity.json").read_text())["state"] == "draining"
    release.set()
    await task
    final = json.loads((run / "runtime_activity.json").read_text())
    assert final["state"] == "stopped" and final["requests"] == []


def test_readonly_mem0_post_search_refusal_is_one_failed_query(tmp_path):
    import threading
    from types import SimpleNamespace
    from test_memory.test_mem0_read_only import _blocking_client, _SCOPE

    release = threading.Event()
    release.set()
    client = _blocking_client(threading.Event(), release)
    guard = SimpleNamespace(unresolved_remote_requests=0, denied_requests=0)
    client._qdrant._probe_read_only_transport = guard

    def native_search(**kwargs):
        guard.denied_requests += 1
        return {"results": []}

    client._memory.search = native_search
    try:
        with observe_model_calls(tmp_path):
            with pytest.raises(PermissionError, match="forbidden Qdrant request"):
                client.search("PRIVATE QUERY", filters=_SCOPE)
        records = [json.loads(line) for line in (tmp_path / "retrieval_activity.jsonl").read_text().splitlines()]
        assert len(records) == 1 and not records[0]["success"]
        assert records[0]["error_type"] == "PermissionError"
    finally:
        client.close()
