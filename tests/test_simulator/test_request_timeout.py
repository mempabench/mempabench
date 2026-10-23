"""The per-request wall-clock deadline on the simulator's provider calls.

Live, acc_039 hung nine minutes on one OpenRouter call with nothing on
stderr. These pin the deadline that ends that: a stalled request must come back
as a transient error while the run is still young enough to retry it.
"""

import asyncio
import time

from simulator.providers.base import LLMProvider
from simulator.providers.openai_compat_provider import (
    OpenAICompatProvider,
    _request_timeout,
)


def test_the_default_deadline_is_180s_with_a_20s_connect_leg(monkeypatch) -> None:
    monkeypatch.delenv("MEMPA_LLM_REQUEST_TIMEOUT_S", raising=False)

    timeout = _request_timeout()

    assert timeout.read == 180.0
    assert timeout.connect == 20.0


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
