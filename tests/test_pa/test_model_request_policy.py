from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from pa.providers.base import LLMResponse, LLMTaskIdentity
from pa.providers.openai_compat_provider import OpenAICompatProvider
from pa.providers.registry import find_by_name
from pa.providers.request_policy import (
    ModelRequestPolicyError,
    load_model_policy_catalog,
    resolve_model_request_policy,
    validate_policy_snapshot,
)


class _RecordingCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _success_response():
    return {
        "choices": [
            {
                "message": {"content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }


def _openrouter_provider(model: str) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="test-key",
        spec=find_by_name("openrouter"),
        default_model=model,
    )


@pytest.mark.asyncio
async def test_openrouter_does_not_inject_explicit_cache_control():
    provider = _openrouter_provider("openai/gpt-5.5")
    completions = _RecordingCompletions([_success_response()])
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    response = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.7,
        task_identity=LLMTaskIdentity(logical_task_key="pa-turn-1"),
    )

    assert response.content == "ok"
    request = completions.requests[0]
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert "extra_headers" not in request


@pytest.mark.asyncio
async def test_openrouter_retries_do_not_add_task_affinity_headers(
    monkeypatch,
):
    provider = _openrouter_provider("openai/gpt-5.5")
    provider._CHAT_RETRY_DELAYS = (0,)
    completions = _RecordingCompletions(
        [
            RuntimeError("connection timeout"),
            _success_response(),
            _success_response(),
        ]
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    first_identity = LLMTaskIdentity(logical_task_key="pa-turn-1")
    first_response = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hello"}],
        task_identity=first_identity,
    )
    second_response = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "next"}],
        task_identity=LLMTaskIdentity(logical_task_key="pa-turn-2"),
    )

    assert first_response.content == "ok"
    assert second_response.content == "ok"
    assert all("extra_headers" not in request for request in completions.requests)


def test_removed_pingu_provider_fails_closed():
    with pytest.raises(
        ModelRequestPolicyError,
        match="provider_name='pingu'.*model='unregistered/model'",
    ):
        resolve_model_request_policy("pingu", "unregistered/model")


def test_openrouter_provider_default_cannot_run_benchmark_role():
    policy = resolve_model_request_policy("openrouter", "unregistered/model")

    assert policy.policy_source == "provider_default"
    with pytest.raises(
        ModelRequestPolicyError,
        match="not validated for role",
    ):
        policy.require_exact_role("pa")


def test_all_pa_backbones_have_exact_openrouter_policies():
    project_root = Path(__file__).resolve().parents[2]
    pa_models = yaml.safe_load(
        (project_root / "configs/pa_models.yaml").read_text(encoding="utf-8")
    )["models"]
    catalog = load_model_policy_catalog()

    exact_openrouter_models = {
        policy.model
        for policy in catalog.models
        if policy.provider_name == "openrouter" and "pa" in policy.validated_roles
    }

    assert exact_openrouter_models == set(pa_models)
    for model in pa_models:
        resolve_model_request_policy("openrouter", model).require_exact_role("pa")


def test_gpt55_has_exact_memory_side_roles():
    policy = resolve_model_request_policy("openrouter", "openai/gpt-5.5")

    policy.require_exact_role("memory_extraction")
    policy.require_exact_role("consolidation")


def test_provider_uses_explicit_request_policy_catalog(tmp_path):
    catalog = tmp_path / "model_request_policies.yaml"
    catalog.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "provider_defaults: []",
                "models:",
                "  - provider_name: openrouter",
                "    model: openai/gpt-5.5",
                "    temperature: omit",
                "    completion_tokens_parameter: max_completion_tokens",
                "    reasoning_effort: omit",
                "    supports_tools: true",
                "    supports_forced_tool_choice: true",
                "    validated_roles: [pa]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    provider = OpenAICompatProvider(
        api_key="test-key",
        spec=find_by_name("openrouter"),
        default_model="openai/gpt-5.5",
        request_policy_catalog_path=catalog,
    )

    policy = provider._request_policy("openai/gpt-5.5")

    assert policy.policy_source == "model"
    assert policy.temperature == "omit"
    assert policy.completion_tokens_parameter == "max_completion_tokens"


def test_a_frozen_policy_snapshot_loads_as_frozen_and_rejects_bad_values():
    policy = resolve_model_request_policy("openrouter", "openai/gpt-5.5")
    snapshot = policy.snapshot()
    snapshot["temperature"] = "omit"

    loaded = validate_policy_snapshot(snapshot, field_name="pa_request_policy")

    assert loaded.temperature == "omit"
    with pytest.raises(ModelRequestPolicyError, match="Invalid frozen policy snapshot"):
        validate_policy_snapshot(
            {**snapshot, "temperature": "sometimes"},
            field_name="pa_request_policy",
        )


class _IdentityRecordingProvider(OpenAICompatProvider):
    async def chat(self, **kwargs):
        self.recorded_identities.append(self.current_task_identity)
        if len(self.recorded_identities) == 1:
            return LLMResponse(
                content="Error calling LLM: connection timeout",
                finish_reason="error",
            )
        return LLMResponse(content="ok")


@pytest.mark.asyncio
async def test_base_retry_scope_preserves_task_identity():
    provider = _IdentityRecordingProvider(
        api_key="test",
        default_model="test-model",
        spec=find_by_name("custom"),
    )
    provider.recorded_identities = []
    provider._CHAT_RETRY_DELAYS = (0,)
    identity = LLMTaskIdentity(logical_task_key="stable-task")

    await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hello"}],
        task_identity=identity,
    )

    assert provider.recorded_identities == [identity, identity]


def test_removed_pingu_provider_is_not_registered():
    assert find_by_name("pingu") is None
