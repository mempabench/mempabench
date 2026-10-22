"""Provider avoidance at the request: what counts against an upstream, and the
`provider.ignore` every later request for that model carries.

The state file itself is covered in `tests/test_harness/test_provider_avoidance.py`.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from harness.provider_avoidance import STATE_FILENAME, ProviderAvoidance
from pa.agent.ix import TurnToolRegistry
from pa.agent.memory import MarkdownMemoryStore, MemoryConsolidationContractError
from pa.agent.runner import AgentRunner, AgentRunSpec
from pa.agent.tools.registry import ToolRegistry
from pa.providers.openai_compat_provider import OpenAICompatProvider
from pa.providers.registry import find_by_name

MODEL = "deepseek/deepseek-v4-flash"
OTHER_MODEL = "z-ai/glm-5.3-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"
#: What OpenRouter answered on 2026-10-19 to a `deepseek/deepseek-v4-flash`
#: request whose `provider.ignore` named all 16 of the model's upstreams.
ALL_IGNORED_BODY = {
    "message": "All providers have been ignored. To change your default ignored "
               "providers, visit: https://openrouter.ai/settings/privacy",
    "code": 404,
    "metadata": {
        "routing_funnel": [{"step": "Initial Endpoints", "endpoint_count": 16}],
        "failed_routing_step": "Filter by Ignored Providers",
    },
}


class _Completions:
    """Stands in for `AsyncOpenAI().chat.completions`: scripted outcomes, recorded requests."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.requests.append(request)
        assert self.outcomes, "unexpected extra request"
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _provider(monkeypatch, tmp_path, outcomes, *, threshold: int = 3,
              extra_body: dict | None = None) -> tuple[OpenAICompatProvider, _Completions]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = OpenAICompatProvider(
        api_key="test-key",
        spec=find_by_name("openrouter"),
        default_model=MODEL,
        extra_body=extra_body,
        provider_avoidance=ProviderAvoidance(tmp_path, threshold=threshold),
    )
    completions = _Completions(outcomes)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider._CHAT_RETRY_DELAYS = (0, 0, 0)
    return provider, completions


def _reply(upstream: str | None, *, content: str | None = "ok", finish_reason: str = "stop",
           tool_calls: list[tuple[str, dict]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{index}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
            for index, (name, arguments) in enumerate(tool_calls)
        ]
    reply: dict[str, Any] = {
        "id": "gen-test", "model": MODEL,
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    if upstream is not None:
        reply["provider"] = upstream
    return reply


def _http_error(status: int, body: dict[str, Any]) -> openai.APIStatusError:
    request = httpx.Request("POST", URL)
    response = httpx.Response(status, request=request, json={"error": body})
    error_type = {401: openai.AuthenticationError, 404: openai.NotFoundError,
                  429: openai.RateLimitError, 502: openai.InternalServerError}[status]
    return error_type(f"Error code: {status}", response=response, body=body)


def _provider_error(upstream: str, status: int = 502) -> openai.APIStatusError:
    return _http_error(status, {"message": "Provider returned error", "code": status,
                                "metadata": {"raw": "upstream failed", "provider_name": upstream}})


def _failures(tmp_path) -> dict[str, int]:
    path = tmp_path / STATE_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())["models"].get(MODEL, {}).get("failures", {})


def _ignore(request: dict[str, Any]) -> list[str] | None:
    return ((request.get("extra_body") or {}).get("provider") or {}).get("ignore")


async def _chat(provider: OpenAICompatProvider):
    return await provider.chat(messages=[{"role": "user", "content": "hi"}], model=MODEL)


# ── what counts, against whom ─────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    _provider_error("DigitalOcean"),
    _provider_error("DigitalOcean", status=429),
    _reply("DigitalOcean", content=None, finish_reason="error"),
    _reply("DigitalOcean", content=None, finish_reason="length"),
    _reply("DigitalOcean", content="  ", finish_reason="length",
           tool_calls=[("save_memory", {})]),
], ids=["http_502", "http_429", "error_response", "empty_length", "length_empty_tool_args"])
async def test_a_provider_failure_counts_against_the_upstream_it_names(monkeypatch, tmp_path, outcome):
    provider, _ = _provider(monkeypatch, tmp_path, [outcome])
    response = await _chat(provider)
    assert response.served_by is not None and response.served_by.provider == "DigitalOcean"
    assert _failures(tmp_path) == {"DigitalOcean": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    openai.APITimeoutError(request=httpx.Request("POST", URL)),
    openai.APIConnectionError(request=httpx.Request("POST", URL)),
    _http_error(401, {"message": "No auth credentials found", "code": 401}),
    _reply(None, content=None, finish_reason="error"),
    _reply("DigitalOcean", content="cut short but said something", finish_reason="length"),
    _reply("DigitalOcean", content=None, finish_reason="length",
           tool_calls=[("save_memory", {"history_entry": "x"})]),
    _reply("DigitalOcean"),
], ids=["timeout", "connection", "gateway_401", "error_without_provider",
        "length_with_text", "length_with_tool_args", "success"])
async def test_nothing_counts_without_a_named_upstream_or_a_failure(monkeypatch, tmp_path, outcome):
    provider, _ = _provider(monkeypatch, tmp_path, [outcome])
    await _chat(provider)
    assert not (tmp_path / STATE_FILENAME).exists()


class _Stream:
    """An SSE stream as the SDK hands it over: chunks to iterate asynchronously."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat", "chat_stream"])
async def test_a_state_file_that_cannot_be_updated_stops_the_call(monkeypatch, tmp_path, method):
    """Never returned as one more provider error: the run would go on uncounted."""
    error_reply = _reply("DigitalOcean", content=None, finish_reason="error")
    if method == "chat_stream":
        # A mid-stream error: HTTP 200, then a chunk naming the upstream.
        error_reply = _Stream([{"provider": "DigitalOcean",
                                "choices": [{"delta": {}, "finish_reason": "error"}]}])

    def broken_mid_request():
        (tmp_path / STATE_FILENAME).write_text("{")
        return error_reply

    provider, _ = _provider(monkeypatch, tmp_path, [broken_mid_request])
    with pytest.raises(ValueError, match="is invalid"):
        await getattr(provider, method)(messages=[{"role": "user", "content": "hi"}], model=MODEL)


@pytest.mark.asyncio
async def test_successes_in_between_do_not_reset_the_count(monkeypatch, tmp_path):
    outcomes = [_provider_error("DigitalOcean"), _reply("DigitalOcean")] * 2 + [_provider_error("DigitalOcean")]
    provider, completions = _provider(monkeypatch, tmp_path, [*outcomes, _reply("Baidu")])
    for _ in range(6):
        await _chat(provider)
    assert _failures(tmp_path) == {"DigitalOcean": 3}
    assert _ignore(completions.requests[-1]) == ["DigitalOcean"]


# ── the ignore list in the request ────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_retry_right_after_the_third_failure_skips_that_upstream(monkeypatch, tmp_path, capsys):
    provider, completions = _provider(
        monkeypatch, tmp_path,
        [_provider_error("DigitalOcean")] * 3 + [_reply("Baidu", content="answer")],
    )

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}], model=MODEL)

    assert response.content == "answer"
    assert [_ignore(request) for request in completions.requests] == [None, None, None, ["DigitalOcean"]]
    stderr = capsys.readouterr().err
    assert f"provider_avoided model={MODEL} provider=DigitalOcean failures=3" in stderr
    assert "upstream_provider=Baidu provider_ignore=DigitalOcean" in stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_body, expected", [
    (None, {"provider": {"ignore": ["DigitalOcean"]}}),
    # `providers.probe_phase` routing that still allows fallbacks.
    ({"provider": {"order": ["Baidu", "Parasail"], "ignore": ["Azure"]}},
     {"provider": {"order": ["Baidu", "Parasail"], "ignore": ["Azure", "DigitalOcean"]}}),
    ({"provider": {"order": ["Baidu"], "allow_fallbacks": True}, "transforms": []},
     {"provider": {"order": ["Baidu"], "allow_fallbacks": True, "ignore": ["DigitalOcean"]},
      "transforms": []}),
], ids=["no_routing", "probe_phase_routing", "routing_with_fallbacks"])
async def test_the_ignore_list_is_merged_into_the_existing_routing(monkeypatch, tmp_path, extra_body, expected):
    ProviderAvoidance(tmp_path, threshold=1).record_failure(
        MODEL, "DigitalOcean", kind="provider_error", failure="x")
    before = json.loads(json.dumps(extra_body))
    provider, completions = _provider(monkeypatch, tmp_path, [_reply("Baidu"), _reply("Baidu")],
                                      threshold=1, extra_body=extra_body)

    await _chat(provider)
    await provider.chat(messages=[{"role": "user", "content": "hi"}], model="z-ai/glm-5.3-flash")

    assert completions.requests[0]["extra_body"] == expected
    # Another model keeps its own routing, and the provider's own routing is untouched.
    assert completions.requests[1].get("extra_body") == extra_body
    assert provider.extra_body == before


@pytest.mark.asyncio
async def test_a_pinned_model_is_never_counted_and_never_given_an_ignore_list(monkeypatch, tmp_path):
    pinned = {"provider": {"order": ["SiliconFlow"], "allow_fallbacks": False}}
    # The same model, unpinned elsewhere in the run, has had SiliconFlow ignored.
    unpinned = ProviderAvoidance(tmp_path, threshold=3)
    for _ in range(3):
        unpinned.record_failure(MODEL, "SiliconFlow", kind="provider_error", failure="an unpinned request")
    provider, completions = _provider(monkeypatch, tmp_path, [
        _provider_error("SiliconFlow"),
        _reply("SiliconFlow", content=None, finish_reason="length"),
        _reply("SiliconFlow", content=None, finish_reason="tool_calls", tool_calls=[("save_memory", {})]),
    ], extra_body=pinned)

    await _chat(provider)
    await _chat(provider)
    rejected = await _chat(provider)
    provider.record_contract_failure(rejected, "pa_ix_render: required_render_carrier")

    assert all(request["extra_body"] == pinned for request in completions.requests)
    assert _failures(tmp_path) == {"SiliconFlow": 3}


@pytest.mark.asyncio
async def test_threshold_zero_sends_no_ignore_list_and_counts_nothing(monkeypatch, tmp_path):
    ProviderAvoidance(tmp_path, threshold=1).record_failure(
        MODEL, "DigitalOcean", kind="provider_error", failure="x")
    state = (tmp_path / STATE_FILENAME).read_text()
    provider, completions = _provider(
        monkeypatch, tmp_path, [_provider_error("Baidu")] * 4, threshold=0)

    for _ in range(4):
        await _chat(provider)

    assert all("extra_body" not in request for request in completions.requests)
    assert (tmp_path / STATE_FILENAME).read_text() == state


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat_with_retry", "chat_stream_with_retry"])
async def test_no_upstream_left_clears_the_models_avoidance_and_resends_the_request(
        monkeypatch, tmp_path, capsys, method):
    """The ignore list is kept across resumes, so without the reset every later
    request for the model, and every resume, would be refused the same way."""
    avoidance = ProviderAvoidance(tmp_path, threshold=1)
    for upstream in ("DigitalOcean", "Baidu"):
        avoidance.record_failure(MODEL, upstream, kind="provider_error", failure="x")
    avoidance.record_failure(OTHER_MODEL, "DigitalOcean", kind="provider_error", failure="x")
    other_model = json.loads((tmp_path / STATE_FILENAME).read_text())["models"][OTHER_MODEL]
    answer = _reply("Parasail", content="answer")
    if method == "chat_stream_with_retry":
        answer = _Stream([{"provider": "Parasail",
                           "choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}])
    provider, completions = _provider(
        monkeypatch, tmp_path, [_http_error(404, ALL_IGNORED_BODY), answer], threshold=1)

    response = await getattr(provider, method)(messages=[{"role": "user", "content": "hi"}], model=MODEL)

    assert response.content == "answer"
    refused, resent = completions.requests
    assert _ignore(refused) == ["Baidu", "DigitalOcean"]
    # The same request, without the ignore list.
    assert "extra_body" not in resent
    assert resent == {key: value for key, value in refused.items() if key != "extra_body"}
    assert _failures(tmp_path) == {}
    assert avoidance.ignored_providers(MODEL) == []
    # Another model's counts and ignored upstreams are not touched.
    assert json.loads((tmp_path / STATE_FILENAME).read_text())["models"][OTHER_MODEL] == other_model
    resets = [line for line in capsys.readouterr().err.splitlines() if "provider_avoidance_reset" in line]
    assert resets == [f"provider_avoidance_reset model={MODEL} cleared=Baidu,DigitalOcean"]


@pytest.mark.asyncio
@pytest.mark.parametrize("resend_outcome, failures", [
    (_provider_error("Parasail"), {"Parasail": 1}),
    # Only the first refusal resets: the resend is sent once.
    (_http_error(404, ALL_IGNORED_BODY), {}),
], ids=["provider_error", "refused_again"])
async def test_a_resend_that_fails_is_an_ordinary_failure_counted_from_zero(
        monkeypatch, tmp_path, capsys, resend_outcome, failures):
    avoidance = ProviderAvoidance(tmp_path, threshold=3)
    for _ in range(3):
        avoidance.record_failure(MODEL, "DigitalOcean", kind="provider_error", failure="x")
    provider, completions = _provider(
        monkeypatch, tmp_path, [_http_error(404, ALL_IGNORED_BODY), resend_outcome])

    response = await _chat(provider)

    assert response.finish_reason == "error"
    assert len(completions.requests) == 2
    assert _failures(tmp_path) == failures
    assert capsys.readouterr().err.count("provider_avoidance_reset") == 1


# ── a caller's contract rejection ─────────────────────────────────────────


def _save_memory(upstream: str, arguments: dict | None, *, finish_reason: str = "tool_calls",
                 content: str | None = None) -> dict[str, Any]:
    calls = None if arguments is None else [("save_memory", arguments)]
    return _reply(upstream, content=content, finish_reason=finish_reason, tool_calls=calls)


@pytest.mark.asyncio
async def test_only_an_empty_consolidation_reply_counts_against_the_upstream_that_served_it(monkeypatch, tmp_path):
    """`save_memory` with `{}`, as DigitalOcean answered over and over on
    2026-10-19, is an empty reply: the upstream's failure. Prose instead of the
    call, or the call with only some of its fields, is the model's own failure,
    the same on every upstream, so it is not counted."""
    valid = {"history_entry": "[2026-10-19 09:00] noted", "memory_update": "# Memory\n- noted"}
    provider, completions = _provider(monkeypatch, tmp_path, [
        _save_memory("DigitalOcean", {}),
        _save_memory("DigitalOcean", None, finish_reason="stop", content="I could not do that."),
        _save_memory("DigitalOcean", {"history_entry": "only half"}),
        _save_memory("DigitalOcean", {}),
        _save_memory("Baidu", valid),
        _save_memory("DigitalOcean", {}),
        _save_memory("Parasail", valid),
    ])
    store = MarkdownMemoryStore(tmp_path / "workspace", log_scope="pa_file_memory")
    messages = [{"role": "user", "content": "remember this", "timestamp": "2026-10-19T09:00"}]

    outcomes = []
    for _ in range(7):
        try:
            outcomes.append(await store.consolidate(messages, provider, MODEL))
        except MemoryConsolidationContractError as exc:
            outcomes.append(str(exc).split(":")[0])

    assert outcomes == [
        "Missing save_memory fields", "No save_memory call", "Missing save_memory fields",
        "Missing save_memory fields", True, "Missing save_memory fields", True,
    ]
    assert _failures(tmp_path) == {"DigitalOcean": 3}
    assert [_ignore(request) for request in completions.requests] == [None] * 6 + [["DigitalOcean"]]
    ignored = json.loads((tmp_path / STATE_FILENAME).read_text())["models"][MODEL]["ignored"]
    assert ignored["DigitalOcean"]["failure_kind"] == "contract_rejected"
    assert ignored["DigitalOcean"]["failure"].startswith("file_consolidation: Missing save_memory fields")


@pytest.mark.asyncio
async def test_a_response_the_provider_and_the_caller_both_reject_counts_once(monkeypatch, tmp_path):
    provider, _ = _provider(monkeypatch, tmp_path, [_save_memory("DigitalOcean", {}, finish_reason="length")])
    store = MarkdownMemoryStore(tmp_path / "workspace", log_scope="pa_file_memory")
    with pytest.raises(MemoryConsolidationContractError):
        await store.consolidate([{"role": "user", "content": "x", "timestamp": "2026-10-19T09:00"}],
                                provider, MODEL)
    assert _failures(tmp_path) == {"DigitalOcean": 1}
    entry = json.loads((tmp_path / STATE_FILENAME).read_text())
    assert entry["models"][MODEL]["ignored"] == {}


def _ix(name: str, **arguments: Any) -> tuple[str, dict]:
    return name, arguments


def test_pa_replies_that_say_something_without_the_render_payload_are_not_counted(monkeypatch, tmp_path):
    """Two replies from DigitalOcean with prose but without the IX render
    payload, with the threshold at 1: the model left the payload out, as it did
    on four upstreams in thirty minutes on 2026-10-20, so no upstream is
    charged and the pinned third repair is sent with no ignore list."""
    provider, completions = _provider(monkeypatch, tmp_path, [
        _reply("DigitalOcean", content=None, finish_reason="tool_calls", tool_calls=[_ix(
            "IX_reasoning_visibility", setting="Show", evidence="Synthetic request.",
            application="Give the reason.")]),
        _reply("DigitalOcean", content="First final response without carrier."),
        _reply("DigitalOcean", content="Second final response without carrier."),
        _reply("Baidu", content=None, finish_reason="tool_calls", tool_calls=[_ix(
            "IX_render_reasoning_visibility", user_facing_rationale="A synthetic reason.")]),
        _reply("Baidu", content="Completed."),
    ], threshold=1)
    context = {"session_active_ix": ["reasoning_visibility"], "tool_action_types": {}}
    tools = TurnToolRegistry(ToolRegistry(), ["reasoning_visibility"], ix_context=context)

    result = asyncio.run(AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Synthetic turn."}],
        tools=tools, model=MODEL, max_iterations=16,
        required_tool_names=tools.required_ix_tool_names, ix_context=context, turn_index=3)))

    assert result.stop_reason == "completed"
    assert result.final_content == "Completed."
    assert completions.requests[3]["tool_choice"]["function"]["name"] == "IX_render_reasoning_visibility"
    assert [_ignore(request) for request in completions.requests] == [None] * 5
    assert not (tmp_path / STATE_FILENAME).exists()
