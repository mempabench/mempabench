"""The per-call stderr line the simulator prints for live run monitoring."""

from simulator.providers.base import LLMResponse
from simulator.providers.openai_compat_provider import _log_openrouter_call


def test_the_call_line_carries_the_usual_fields(capsys) -> None:
    _log_openrouter_call(
        "google/gemini-3.8-flash", 0.0,
        LLMResponse(content="ok", usage={"prompt_tokens": 1, "completion_tokens": 2}),
        cost=0.0004,
    )

    line = capsys.readouterr().err.strip()
    assert line.startswith("openrouter_call model=google/gemini-3.8-flash")
    assert "prompt_tokens=1" in line
    assert "completion_tokens=2" in line
    assert "cost_usd=0.0004" in line
    assert line.endswith(" finish_reason=stop")


def test_the_call_line_says_why_the_model_stopped(capsys) -> None:
    _log_openrouter_call(
        "google/gemini-3.8-flash", 0.0,
        LLMResponse(content=None, usage={"prompt_tokens": 1, "completion_tokens": 4092},
                    finish_reason="length"),
        cost=None,
    )

    assert capsys.readouterr().err.strip().endswith(" finish_reason=length")
