import pytest

from pa.providers.openai_compat_provider import (
    OpenAICompatProvider as PAOpenAICompatProvider,
)
from simulator.providers.openai_compat_provider import (
    OpenAICompatProvider as SimulatorOpenAICompatProvider,
)


@pytest.mark.parametrize(
    "provider_type",
    [PAOpenAICompatProvider, SimulatorOpenAICompatProvider],
)
def test_empty_error_choice_preserves_provider_diagnostics(provider_type) -> None:
    provider = object.__new__(provider_type)
    response = provider._parse({
        "id": "request_123",
        "model": "upstream-model",
        "error": {"code": 504, "message": "upstream timeout"},
        "choices": [{
            "finish_reason": "error",
            "message": {"role": "assistant", "content": None},
            "native_finish_reason": "provider_error",
        }],
    })

    assert response.finish_reason == "error"
    assert response.content is None
    assert response.error_detail == {
        "response_id": "request_123",
        "model": "upstream-model",
        "response_error": {"code": 504, "message": "upstream timeout"},
        "raw_choice": {
            "finish_reason": "error",
            "message": {"role": "assistant", "content": None},
            "native_finish_reason": "provider_error",
        },
    }
    assert "upstream timeout" in response.error_text
    assert provider._is_transient_error(response.error_text)
