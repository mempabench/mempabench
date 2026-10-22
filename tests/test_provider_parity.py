"""Parity tests for the PA/Simulator provider output contract.

These tests intentionally compare only fields consumed by the benchmark. Provider
request policy, headers, retry behavior, and private implementation details remain
independent.
"""

from types import SimpleNamespace

import pytest

from pa.providers.anthropic_provider import AnthropicProvider as PAAnthropicProvider
from pa.providers.openai_compat_provider import (
    OpenAICompatProvider as PAOpenAICompatProvider,
)
from simulator.providers.anthropic_provider import (
    AnthropicProvider as SimulatorAnthropicProvider,
)
from simulator.providers.openai_compat_provider import (
    OpenAICompatProvider as SimulatorOpenAICompatProvider,
)


OPENAI_PROVIDERS = [PAOpenAICompatProvider, SimulatorOpenAICompatProvider]
ANTHROPIC_PROVIDERS = [PAAnthropicProvider, SimulatorAnthropicProvider]


@pytest.mark.parametrize("provider_type", OPENAI_PROVIDERS)
def test_openai_compatible_tool_call_usage_and_finish_reason_contract(
    provider_type,
) -> None:
    provider = object.__new__(provider_type)

    provider_response = provider._parse({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "upstream-id",
                    "type": "function",
                    "provider_marker": "tool-call-metadata",
                    "extra_content": {"thought_signature": "signed"},
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Shanghai"}',
                        "provider_marker": "function-metadata",
                    },
                }],
            },
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
        },
    })

    assert provider_response.content is None
    assert provider_response.finish_reason == "tool_calls"
    assert provider_response.usage == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
    }
    assert len(provider_response.tool_calls) == 1
    tool_call = provider_response.tool_calls[0]
    assert len(tool_call.id) == 9 and tool_call.id.isalnum()
    assert tool_call.name == "lookup_weather"
    assert tool_call.arguments == {"city": "Shanghai"}
    assert tool_call.extra_content == {"thought_signature": "signed"}
    assert tool_call.provider_specific_fields == {
        "provider_marker": "tool-call-metadata"
    }
    assert tool_call.function_provider_specific_fields == {
        "provider_marker": "function-metadata"
    }


@pytest.mark.parametrize("provider_type", OPENAI_PROVIDERS)
def test_openai_compatible_error_detail_contract(provider_type) -> None:
    provider = object.__new__(provider_type)

    provider_response = provider._parse({
        "id": "request_123",
        "model": "upstream-model",
        "error": {"code": 504, "message": "upstream timeout"},
        "choices": [{
            "finish_reason": "error",
            "message": {"role": "assistant", "content": None},
            "native_finish_reason": "provider_error",
        }],
    })

    assert provider_response.finish_reason == "error"
    assert provider_response.error_detail == {
        "response_id": "request_123",
        "model": "upstream-model",
        "response_error": {"code": 504, "message": "upstream timeout"},
        "raw_choice": {
            "finish_reason": "error",
            "message": {"role": "assistant", "content": None},
            "native_finish_reason": "provider_error",
        },
    }


@pytest.mark.parametrize("provider_type", ANTHROPIC_PROVIDERS)
def test_anthropic_tool_call_usage_and_finish_reason_contract(provider_type) -> None:
    provider_response = provider_type._parse_response(SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Checking."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_123",
                name="lookup_weather",
                input={"city": "Shanghai"},
            ),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=2,
        ),
    ))

    assert provider_response.content == "Checking."
    assert provider_response.finish_reason == "tool_calls"
    assert provider_response.usage == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 2,
    }
    assert len(provider_response.tool_calls) == 1
    tool_call = provider_response.tool_calls[0]
    assert tool_call.id == "toolu_123"
    assert tool_call.name == "lookup_weather"
    assert tool_call.arguments == {"city": "Shanghai"}
