"""Contract tests for MemPA's patched MemOS LLM request path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memos.configs.llm import OpenAILLMConfig
from memos.context.context import ContextThreadPoolExecutor
from memos.llms.openai import OpenAILLM
from memos.mempa_runtime import memory_extraction_task, runtime_contract


def _write_policy_catalog(
    path: Path,
    *,
    temperature: str = "send",
    completion_tokens_parameter: str = "max_completion_tokens",
) -> None:
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "provider_defaults: []",
                "models:",
                "  - provider_name: openrouter",
                "    model: openai/gpt-5.5",
                f"    temperature: {temperature}",
                "    completion_tokens_parameter: "
                f"{completion_tokens_parameter}",
                "    reasoning_effort: send",
                "    supports_tools: true",
                "    supports_forced_tool_choice: true",
                "    validated_roles: [memory_extraction]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _configure_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    temperature: str = "send",
    completion_tokens_parameter: str = "max_completion_tokens",
) -> None:
    catalog_path = tmp_path / "model_request_policies.yaml"
    _write_policy_catalog(
        catalog_path,
        temperature=temperature,
        completion_tokens_parameter=completion_tokens_parameter,
    )
    values = {
        "MEMPA_MODEL_REQUEST_POLICY_PATH": str(catalog_path),
        "MEMPA_MEMOS_SOURCE_COMMIT": "test-source-commit",
        "MEMPA_MEMOS_SOURCE_SHA256": "a" * 64,
        "MEMPA_MEMORY_LM_PROVIDER_NAME": "openrouter",
        "MEMPA_MEMORY_LM_TASK_AFFINITY_REQUIRED": "false",
        "MEMPA_MEMORY_LM_TASK_AFFINITY_NAMESPACE": "",
        "MOS_CHAT_MODEL_PROVIDER": "openai",
        "MOS_CHAT_MODEL": "openai/gpt-5.5",
        "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
        "OPENAI_API_KEY": "api-key",
        "MEMRADER_MODEL": "openai/gpt-5.5",
        "MEMRADER_API_BASE": "https://openrouter.ai/api/v1",
        "MEMRADER_API_KEY": "api-key",
        "MEMREADER_GENERAL_MODEL": "openai/gpt-5.5",
        "MEMREADER_GENERAL_API_BASE": "https://openrouter.ai/api/v1",
        "MEMREADER_GENERAL_API_KEY": "api-key",
        "MEMREADER_ENABLE_BACKUP": "false",
        "MOS_EMBEDDER_PROVIDER": "openai",
        "MOS_EMBEDDER_MODEL": "text-embedding-3-small",
        "MOS_EMBEDDER_API_BASE": "https://api.openai.com/v1",
        "EMBEDDING_DIMENSION": "1536",
        "MOS_RERANKER_BACKEND": "cosine_local",
        "GRAPH_DB_BACKEND": "neo4j-community",
        "NEO4J_URI": "bolt://neo4j:7687",
        "NEO4J_DB_NAME": "neo4j",
        "QDRANT_HOST": "qdrant",
        "QDRANT_PORT": "6333",
        "ENABLE_CHAT_API": "false",
        "ENABLE_ACTIVATION_MEMORY": "false",
        "ENABLE_PREFERENCE_MEMORY": "false",
        "MEMPA_QDRANT_IMAGE": "qdrant/qdrant:v1.15.3",
        "MEMPA_NEO4J_IMAGE": "neo4j:5.26.6",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _llm(monkeypatch: pytest.MonkeyPatch) -> tuple[OpenAILLM, Mock]:
    create = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            model_dump_json=lambda: json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("memos.llms.openai.openai.Client", Mock(return_value=client))
    llm = OpenAILLM(
        OpenAILLMConfig(
            model_name_or_path="openai/gpt-5.5",
            api_key="api-key",
            api_base="https://openrouter.ai/api/v1",
            temperature=0.2,
            max_tokens=321,
        )
    )
    return llm, create


def test_gpt_request_uses_frozen_policy_without_affinity_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    llm, create = _llm(monkeypatch)
    messages = [{"role": "user", "content": "extract preference"}]

    with memory_extraction_task("run_001:session_001:turn_001"):
        llm.generate(messages)
        llm.generate(messages)

    first = create.call_args_list[0].kwargs
    second = create.call_args_list[1].kwargs
    assert first["temperature"] == 0.2
    assert first["max_completion_tokens"] == 321
    assert "max_tokens" not in first
    assert "top_p" not in first
    assert "extra_headers" not in first
    assert "extra_headers" not in second


def test_policy_can_omit_temperature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(
        monkeypatch,
        tmp_path,
        temperature="omit",
        completion_tokens_parameter="max_tokens",
    )
    llm, create = _llm(monkeypatch)

    with memory_extraction_task("run_001:session_001:turn_001"):
        llm.generate([{"role": "user", "content": "extract"}])

    request = create.call_args.kwargs
    assert "temperature" not in request
    assert request["max_tokens"] == 321
    assert "max_completion_tokens" not in request


def test_openrouter_requests_do_not_depend_on_parent_task_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    llm, create = _llm(monkeypatch)

    with memory_extraction_task("run_001:session_001:turn_001"):
        llm.generate([{"role": "user", "content": "first"}])
        llm.generate([{"role": "user", "content": "second"}])
    with memory_extraction_task("run_001:session_001:turn_002"):
        llm.generate([{"role": "user", "content": "first"}])

    assert len(create.call_args_list) == 3
    assert all(
        "extra_headers" not in call.kwargs for call in create.call_args_list
    )


def test_openrouter_request_does_not_require_parent_task_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    llm, create = _llm(monkeypatch)

    assert llm.generate([{"role": "user", "content": "extract"}]) == "ok"
    assert "extra_headers" not in create.call_args.kwargs


def test_context_thread_pool_preserves_memory_extraction_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    llm, create = _llm(monkeypatch)

    with memory_extraction_task("run_001:session_001:turn_001"):
        with ContextThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                llm.generate,
                [{"role": "user", "content": "extract"}],
            ).result()

    assert result == "ok"
    assert "extra_headers" not in create.call_args.kwargs


def test_runtime_contract_reports_the_memory_profile_and_excludes_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)

    contract = runtime_contract()

    assert contract["memory_profile"] == "text_memory"
    assert contract["schema_version"] == 2
    assert contract["memos_source_sha256"] == "a" * 64
    assert contract["activation_memory_enabled"] is False
    assert contract["preference_memory_enabled"] is False
    assert set(contract["memory_side_model_roles"]) == {
        "chat",
        "memory_reader",
        "general_reader",
    }
    assert "api-key" not in json.dumps(contract)


def test_runtime_contract_rejects_role_model_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMREADER_GENERAL_MODEL", "unexpected-model")

    with pytest.raises(ValueError, match="do not use one fixed model"):
        runtime_contract()


def test_memory_side_lm_rejects_backup_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr("memos.llms.openai.openai.Client", Mock())

    with pytest.raises(ValueError, match="does not allow a backup"):
        OpenAILLM(
            OpenAILLMConfig(
                model_name_or_path="openai/gpt-5.5",
                api_key="api-key",
                api_base="https://openrouter.ai/api/v1",
                backup_client=True,
                backup_api_key="backup-key",
                backup_api_base="https://backup.example/v1",
                backup_model_name_or_path="backup-model",
            )
        )
