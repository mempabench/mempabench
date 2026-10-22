"""PA upstream routing comes from `provider_routing` in the policy catalog."""

from __future__ import annotations

from pathlib import Path

from harness.pa_factory import AssistantLLMProviderProfile, build_pa_from_provider_profile


def _pa_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace/runs/m/c/user_a/run1/pa_workspace/user_a"
    workspace.mkdir(parents=True)
    return workspace


def _build(tmp_path: Path, model: str):
    profile = AssistantLLMProviderProfile(provider_name="openrouter", api_key="test-key")
    loop = build_pa_from_provider_profile(
        provider_profile=profile,
        workspace=_pa_workspace(tmp_path),
        pa_model=model,
        pa_max_tokens=1024,
    )
    return loop.pa_llm_provider


def test_hy4_preview_goes_only_to_siliconflow_and_logs_raw_responses(tmp_path: Path):
    provider = _build(tmp_path, "tencent/hy4-preview")
    assert provider.extra_body == {
        "provider": {"order": ["SiliconFlow"], "allow_fallbacks": False}
    }
    assert provider.raw_response_log == (
        tmp_path / "workspace/logs/raw_pa/run1/raw_pa_responses.jsonl"
    ).resolve()


def test_a_model_without_routing_keeps_the_default(tmp_path: Path):
    provider = _build(tmp_path, "openai/gpt-5.5")
    assert provider.extra_body is None
    assert provider.raw_response_log is None
