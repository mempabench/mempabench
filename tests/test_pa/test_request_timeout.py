"""The per-request wall-clock deadline on the PA's provider calls.

acc_039 hung nine minutes on one OpenRouter call with nothing on
stderr. These pin the deadline that ends that: a stalled request must come back
as a transient error while the run is still young enough to retry it. The
deadline must also leave a call room to return the plan's whole output cap.
"""

import asyncio
import time

import httpx
import pytest

from pa.providers import openai_compat_provider
from pa.providers.base import LLMProvider
from pa.providers.openai_compat_provider import (
    OpenAICompatProvider,
    _request_timeout,
)


def test_the_default_deadline_is_600s_with_a_20s_connect_leg(monkeypatch) -> None:
    monkeypatch.delenv("MEMPA_LLM_REQUEST_TIMEOUT_S", raising=False)

    timeout = _request_timeout()

    assert timeout.read == 600.0
    assert timeout.connect == 20.0


@pytest.mark.parametrize(
    ("env_value", "deadline"), [(None, 600.0), ("45", 45.0)], ids=["default", "env"]
)
def test_a_pa_call_goes_out_with_the_600s_deadline_or_the_env_override(
    monkeypatch, env_value, deadline
) -> None:
    """A call may return the plan's whole output cap (`pa_max_tokens`); a
    deadline shorter than that takes would turn every long answer into a
    timeout. Read off the request the provider actually sends."""
    if env_value is None:
        monkeypatch.delenv("MEMPA_LLM_REQUEST_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("MEMPA_LLM_REQUEST_TIMEOUT_S", env_value)
    sent: list[dict] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request.extensions["timeout"])
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0,
            "model": "test/model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    real_client = openai_compat_provider.AsyncOpenAI
    monkeypatch.setattr(
        openai_compat_provider,
        "AsyncOpenAI",
        lambda **kwargs: real_client(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
        ),
    )
    provider = OpenAICompatProvider(
        api_key="test-key", api_base="http://pa.invalid/v1", default_model="test/model"
    )

    response = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

    assert response.finish_reason == "stop"
    assert sent == [
        {"connect": 20.0, "read": deadline, "write": deadline, "pool": deadline}
    ]


def test_the_env_var_overrides_the_total(monkeypatch) -> None:
    monkeypatch.setenv("MEMPA_LLM_REQUEST_TIMEOUT_S", "12.5")

    timeout = _request_timeout()

    assert timeout.read == 12.5
    assert timeout.connect == 20.0


def test_a_stalled_request_becomes_a_transient_error_within_the_deadline(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("MEMPA_LLM_REQUEST_TIMEOUT_S", "1")

    async def run() -> tuple[object, float]:
        released = asyncio.Event()

        async def never_respond(reader, writer) -> None:
            # Accept the connection, then answer nothing at all — exactly the
            # shape of the acc_039 hang. Released only once the call has given
            # up, so the server can shut down.
            await released.wait()
            writer.close()

        server = await asyncio.start_server(never_respond, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            provider = OpenAICompatProvider(
                api_key="test-key",
                api_base=f"http://127.0.0.1:{port}/v1",
                default_model="test/model",
            )
            started = time.monotonic()
            response = await provider.chat([{"role": "user", "content": "hi"}])
            return response, time.monotonic() - started
        finally:
            released.set()
            server.close()

    response, elapsed = asyncio.run(run())

    assert elapsed < 30, "the deadline did not bound the call"
    assert response.finish_reason == "error"
    assert LLMProvider._is_transient_error(response.error_text), response.error_text
    assert " finish_reason=timeout" in capsys.readouterr().err
