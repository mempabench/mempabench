"""Tests for Mem0 client construction and extraction behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import openai
import pytest

from harness.provider_avoidance import STATE_FILENAME, ProviderAvoidance
from memory.backends.mem0_client import (
    MEM0_EXTRACTION_MAX_OUTPUT_TOKENS,
    Mem0PersistenceError,
    Mem0ResponseError,
    _MemoryExtractionLLM,
    _ScopedMem0Client,
    _ValidatedMem0Client,
    _assert_qdrant_collection_ready,
    _drop_qdrant_collection_for_run,
    _inspect_qdrant_collection,
    _prepare_history_db_path,
    _probe_qdrant_collection,
    _qdrant_admin,
    _storage_config,
    _validate_add_response,
    _validate_search_response,
    create_mem0_client,
    create_read_only_mem0_client,
    create_simple_rag_mem0_client,
)
from memory.backends.qdrant_runtime import QDRANT_OPERATION_TIMEOUT_SECONDS


_CLIENT_STORAGE_KWARGS = {
    "embedding_api_base": "https://api.openai.com/v1",
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "embedding_dimensions": 1536,
    "qdrant_distance": "Cosine",
    "qdrant_on_disk": True,
    "qdrant_bm25_enabled": False,
}
_CLIENT_VALIDATION_KWARGS = {
    "embedding_dimensions": 1536,
    "qdrant_distance": "Cosine",
    "qdrant_on_disk": True,
    "qdrant_bm25_enabled": False,
    "enable_entity_memory": False,
}
_ADD_SCOPE = {
    "user_id": "user_a",
    "run_id": "run_001",
    "metadata": {
        "session_id": "acc_003",
        "turn_id": "beat_02",
        "timestamp": "2026-08-03T10:00:00+08:00",
    },
}


def _factory_storage_kwargs(
    tmp_path: Path,
    *,
    enable_entity_memory: bool,
) -> dict[str, Any]:
    return {
        **_CLIENT_STORAGE_KWARGS,
        "enable_entity_memory": enable_entity_memory,
        "history_db_path": tmp_path / "backend_state" / "mem0_history.sqlite3",
        "allow_existing_history_db": False,
        "require_existing_history_db": False,
    }


_QDRANT_SCHEMA_KWARGS = {
    "expected_embedding_dimensions": 1536,
    "expected_distance": "Cosine",
    "expected_on_disk": True,
    "expected_bm25_enabled": False,
    "expected_entity_memory_enabled": False,
}


def test_simple_rag_client_uses_dedicated_embedding_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_qdrant(monkeypatch)
    captured: dict[str, Any] = {}

    class FakeMemory:
        @classmethod
        def from_config(cls, config: dict[str, Any]) -> object:
            captured.update(config)
            return object()

    fake_mem0 = ModuleType("mem0")
    fake_mem0.Memory = FakeMemory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong_general_key")
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "embedding_key")

    monkeypatch.setattr(
        "memory.backends.mem0_client._assert_qdrant_collection_ready",
        lambda **_: None,
    )
    client = create_simple_rag_mem0_client(
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        allow_existing_collection=False,
        **_factory_storage_kwargs(tmp_path, enable_entity_memory=False),
    )

    assert captured["embedder"] == {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "embedding_dims": 1536,
            "api_key": "embedding_key",
            "openai_base_url": "https://api.openai.com/v1",
        },
    }
    assert captured["vector_store"]["config"]["collection_name"] == (
        "mempa_run_001_mem0"
    )
    assert captured["vector_store"]["config"]["client"].timeout == (
        QDRANT_OPERATION_TIMEOUT_SECONDS
    )
    assert captured["initialize_llm"] is False
    assert captured["enable_entity_memory"] is False
    assert captured["history_db_path"] == str(
        tmp_path / "backend_state" / "mem0_history.sqlite3"
    )
    assert isinstance(client, _ValidatedMem0Client)


def test_simple_rag_client_rejects_missing_dedicated_embedding_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "wrong_general_key")
    monkeypatch.delenv("OPENAI_EMB_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_EMB_API_KEY is unset"):
        create_simple_rag_mem0_client(
            collection_name="mempa_run_001_mem0",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            expected_run_id="run_001",
            allow_existing_collection=False,
            **_factory_storage_kwargs(tmp_path, enable_entity_memory=False),
        )


@pytest.mark.parametrize("standard_history", [True, False])
@pytest.mark.parametrize('extraction_budget', [None, 32000])
def test_openrouter_client_factory_builds_scoped_client_with_explicit_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    standard_history: bool,
    extraction_budget: int | None,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr("memory.backends.mem0_client._require_entity_nlp", lambda: None)
    captured: dict[str, Any] = {}
    memory = SimpleNamespace(llm=None)

    class FakeMemory:
        @classmethod
        def from_config(cls, config: dict[str, Any]) -> object:
            captured["config"] = config
            return memory

    class FakeExtractionLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured["llm_kwargs"] = kwargs

    fake_mem0 = ModuleType("mem0")
    fake_mem0.Memory = FakeMemory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0)
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "embedding-key")
    monkeypatch.setattr(
        "memory.backends.mem0_client._MemoryExtractionLLM",
        FakeExtractionLLM,
    )
    monkeypatch.setattr(
        "memory.backends.mem0_client._assert_qdrant_collection_ready",
        lambda **kwargs: captured.update(readiness=kwargs),
    )

    storage = _factory_storage_kwargs(tmp_path, enable_entity_memory=True)
    if not standard_history:
        storage["history_db_path"] = tmp_path / "custom_history.sqlite3"
    avoidance = ProviderAvoidance(tmp_path / "run", threshold=3)
    client = create_mem0_client(
        provider_name="openrouter",
        api_key="api-key",
        api_base="https://openrouter.ai/api/v1",
        memory_side_model="openai/gpt-5.5",
        provider_avoidance=avoidance,
        collection_name="mempa_run_001_mem0",
        qdrant_host="qdrant.internal",
        qdrant_port=7444,
        expected_run_id="run_001",
        allow_existing_collection=True,
        **storage,
        **({'extraction_max_output_tokens': extraction_budget}
           if extraction_budget is not None else {}),
    )

    assert isinstance(client, _ScopedMem0Client)
    assert client._memory is memory
    assert memory.llm is client._llm
    assert captured["llm_kwargs"] == {
        "provider_name": "openrouter",
        "api_key": "api-key",
        "api_base": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-5.5",
        "extraction_max_output_tokens": extraction_budget or MEM0_EXTRACTION_MAX_OUTPUT_TOKENS,
        "failure_evidence_dir": (tmp_path / "private_memory_diagnostics") if standard_history else None,
        "provider_avoidance": avoidance,
    }
    assert captured["readiness"] == {
        "collection_name": "mempa_run_001_mem0",
        "qdrant_host": "qdrant.internal",
        "qdrant_port": 7444,
        "expected_run_id": "run_001",
        "allow_existing_collection": True,
        "drop_existing_collection": False,
        **{
            **_QDRANT_SCHEMA_KWARGS,
            "expected_entity_memory_enabled": True,
        },
    }
    assert captured["config"]["initialize_llm"] is False
    assert captured["config"]["enable_entity_memory"] is True
    assert captured["config"]["history_db_path"] == str(storage["history_db_path"])
    assert captured["config"]["llm"]["config"] == {
        "model": "openai/gpt-5.5",
        "api_key": "api-key",
        "openai_base_url": "https://openrouter.ai/api/v1",
    }
    vector_config = captured["config"]["vector_store"]["config"]
    qdrant = vector_config.pop("client")
    assert qdrant.timeout == QDRANT_OPERATION_TIMEOUT_SECONDS
    assert vector_config == {
        "collection_name": "mempa_run_001_mem0",
        "embedding_model_dims": 1536,
        "host": "qdrant.internal",
        "port": 7444,
        "path": None,
        "on_disk": True,
        "enable_bm25": False,
    }
    assert client._collection_name == "mempa_run_001_mem0"
    assert client._qdrant_host == "qdrant.internal"
    assert client._qdrant_port == 7444
    assert client._expected_run_id == "run_001"


def test_scoped_extraction_client_sets_logical_task_from_event_identity() -> None:
    llm = SimpleNamespace(
        logical_task_keys=[],
        request_succeeded=True,
    )
    llm.set_logical_task_key = llm.logical_task_keys.append
    memory = SimpleNamespace(
        add=lambda **kwargs: {
            "results": [
                {"id": "memory-1", "memory": "hello", "event": "ADD"}
            ]
        },
        get=lambda memory_id: {
            "id": memory_id,
            "memory": "hello",
            "user_id": "user_a",
            "run_id": "run_001",
            "metadata": _ADD_SCOPE["metadata"],
        },
    )
    client = _ScopedMem0Client(
        memory,
        llm,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    result = client.add(
        messages=[{"role": "user", "content": "hello"}],
        **_ADD_SCOPE,
    )

    assert result == {
        "results": [
            {"id": "memory-1", "memory": "hello", "event": "ADD"}
        ]
    }
    assert llm.logical_task_keys == ["mem0_add:run_001:acc_003:beat_02"]


def test_scoped_extraction_client_rejects_add_without_successful_extraction() -> None:
    llm = SimpleNamespace(request_succeeded=False)
    llm.set_logical_task_key = lambda _: None
    memory = SimpleNamespace(add=lambda **kwargs: {"results": []})
    client = _ScopedMem0Client(
        memory,
        llm,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    with pytest.raises(
        RuntimeError,
        match="run_id='run_001'.*session_id='acc_003'.*turn_id='beat_02'",
    ):
        client.add(
            messages=[{"role": "user", "content": "hello"}],
            **_ADD_SCOPE,
        )


_EXTRACTION_MODEL = "deepseek/deepseek-v4-flash"


def _reply(content: str, **usage: Any) -> dict[str, Any]:
    return {
        "choices": [
            {"message": {"content": content}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            **usage,
        },
    }


def _metered_extraction_llm(
    monkeypatch: pytest.MonkeyPatch,
    replies: list[Any],
    *, extraction_max_output_tokens: int = MEM0_EXTRACTION_MAX_OUTPUT_TOKENS,
    provider_avoidance: ProviderAvoidance | None = None,
) -> tuple[_MemoryExtractionLLM, list[dict[str, Any]]]:
    """A real extraction LLM whose only fake is the HTTP transport.

    Everything above the wire — the provider, its request policy, its
    `openrouter_call` line — is the production object.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = _MemoryExtractionLLM(
        provider_name="openrouter",
        api_key="test-key",
        api_base="https://openrouter.ai/api/v1",
        model=_EXTRACTION_MODEL,
        extraction_max_output_tokens=extraction_max_output_tokens,
        **({"provider_avoidance": provider_avoidance} if provider_avoidance is not None else {}),
    )
    requests: list[dict[str, Any]] = []
    remaining = iter(replies)

    async def create(**kwargs: Any) -> dict[str, Any]:
        requests.append(kwargs)
        reply = next(remaining)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    llm._provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return llm, requests


def _extraction_client(llm: _MemoryExtractionLLM) -> _ScopedMem0Client:
    """A scoped client whose mem0 stand-in extracts the way mem0 does."""

    def add(**kwargs: Any) -> dict[str, Any]:
        llm.generate_response(
            messages=[
                {"role": "system", "content": "extract"},
                {"role": "user", "content": "pool"},
            ],
            response_format={"type": "json_object"},
        )
        return {"results": [{"id": "memory-1", "memory": "hello", "event": "ADD"}]}

    memory = SimpleNamespace(
        add=add,
        get=lambda memory_id: {
            "id": memory_id,
            "memory": "hello",
            "user_id": "user_a",
            "run_id": "run_001",
            "metadata": _ADD_SCOPE["metadata"],
        },
    )
    return _ScopedMem0Client(
        memory,
        llm,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )


def test_mem0_add_prints_one_metered_openrouter_call_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm, _ = _metered_extraction_llm(
        monkeypatch,
        [
            _reply(
                '{"memory": [{"text": "hello"}]}',
                prompt_tokens=1234,
                completion_tokens=56,
                total_tokens=1290,
                cost=0.00042,
            )
        ],
    )
    client = _extraction_client(llm)
    capsys.readouterr()

    client.add(messages=[{"role": "user", "content": "hello"}], **_ADD_SCOPE)

    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("openrouter_call ")
    ]
    assert len(lines) == 1
    fields = dict(pair.split("=", 1) for pair in lines[0].split()[1:])
    assert fields["model"] == _EXTRACTION_MODEL
    assert fields["prompt_tokens"] == "1234"
    assert fields["completion_tokens"] == "56"
    assert fields["cost_usd"] == "0.00042"
    assert fields["task"] == "mem0_add:run_001:acc_003:beat_02"
    assert float(fields["seconds"]) >= 0


@pytest.mark.parametrize('budget', [MEM0_EXTRACTION_MAX_OUTPUT_TOKENS, 32000])
def test_extraction_request_carries_mem0_json_mode_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
) -> None:
    llm, requests = _metered_extraction_llm(
        monkeypatch, [_reply('{"memory": []}')], extraction_max_output_tokens=budget,
    )
    llm.set_logical_task_key("mem0_add:run_001:acc_003:beat_02")

    result = llm.generate_response(
        messages=[{"role": "user", "content": "pool"}],
        response_format={"type": "json_object"},
    )

    assert result == '{"memory": []}'
    assert len(requests) == 1
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[0]["model"] == _EXTRACTION_MODEL
    assert requests[0]["messages"] == [{"role": "user", "content": "pool"}]
    assert requests[0]["max_tokens"] == budget
    assert requests[0]["temperature"] == 0.1
    assert "tools" not in requests[0]
    assert "tool_choice" not in requests[0]


def test_extraction_request_uses_the_explicit_frozen_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm, requests = _metered_extraction_llm(
        monkeypatch,
        [_reply('{"memory": []}')],
        extraction_max_output_tokens=32_000,
    )
    llm.set_logical_task_key("mem0_add:run_001:acc_003:beat_02")

    assert llm.generate_response(
        messages=[{"role": "user", "content": "pool"}],
        response_format={"type": "json_object"},
    ) == '{"memory": []}'
    assert requests[0]["max_tokens"] == 32_000


def test_extraction_request_carries_tools_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "add_memory", "parameters": {"type": "object"}},
        }
    ]
    llm, requests = _metered_extraction_llm(
        monkeypatch,
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "add_memory",
                                        "arguments": '{"text": "hello"}',
                                    }
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        ],
    )
    llm.set_logical_task_key("mem0_add:run_001:acc_003:beat_02")

    result = llm.generate_response(
        messages=[{"role": "user", "content": "pool"}],
        tools=tools,
        tool_choice="auto",
    )

    assert requests[0]["tools"] == tools
    assert requests[0]["tool_choice"] == "auto"
    assert result == {
        "content": None,
        "tool_calls": [
            {"name": "add_memory", "arguments": '{"text": "hello"}'}
        ],
    }


def test_extraction_retries_invalid_json_without_affinity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm, requests = _metered_extraction_llm(
        monkeypatch, [_reply(""), _reply('{"memory": []}')]
    )
    llm.set_logical_task_key("mem0_add:run_001:acc_003:beat_02")
    monkeypatch.setattr("memory.backends.mem0_client.time.sleep", lambda _: None)

    result = llm.generate_response(
        messages=[{"role": "user", "content": "test"}],
        response_format={"type": "json_object"},
    )

    assert result == '{"memory": []}'
    assert len(requests) == 2
    assert all("extra_headers" not in request for request in requests)
    assert llm.request_succeeded is True


def test_extraction_length_empty_content_passes_to_native_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty length content is passed to native Mem0 without a second call."""
    llm, requests = _metered_extraction_llm(
        monkeypatch,
        [
            {
                "choices": [
                    {"message": {"content": ""}, "finish_reason": "length"}
                ],
                "usage": {
                    "prompt_tokens": 11185,
                    "completion_tokens": 16000,
                    "total_tokens": 27185,
                    "completion_tokens_details": {"reasoning_tokens": 16412},
                },
            },
            _reply('{"memory": []}'),
        ],
    )
    llm.set_logical_task_key("mem0_add:run_001:acc_003:beat_02")

    assert llm.generate_response(
        messages=[{"role": "user", "content": "pool"}],
        response_format={"type": "json_object"},
    ) == ""
    assert len(requests) == 1
    assert llm.request_succeeded is True


def _upstream_error(upstream: str) -> openai.InternalServerError:
    """A 502 OpenRouter relays from `upstream`, as `error.metadata.provider_name` names it."""
    body = {"message": "Provider returned error", "code": 502,
            "metadata": {"raw": "upstream failed", "provider_name": upstream}}
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.InternalServerError(
        "Error code: 502", response=httpx.Response(502, request=request, json={"error": body}), body=body)


def _ignore_list(request: dict[str, Any]) -> list[str] | None:
    return ((request.get("extra_body") or {}).get("provider") or {}).get("ignore")


def test_extraction_requests_skip_an_upstream_the_run_avoids_and_count_against_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Mem0 extraction counts into the run's state and the retry after the third failure skips it."""
    avoidance = ProviderAvoidance(tmp_path, threshold=3)
    llm, requests = _metered_extraction_llm(
        monkeypatch,
        [_upstream_error("DigitalOcean")] * 3 + [{**_reply('{"memory": []}'), "provider": "Baidu"}],
        provider_avoidance=avoidance,
    )
    llm._provider._CHAT_RETRY_DELAYS = (0, 0, 0)
    llm.set_logical_task_key("mem0_add:run_001:acc_003:beat_02")

    result = llm.generate_response(
        messages=[{"role": "user", "content": "pool"}],
        response_format={"type": "json_object"},
    )

    assert result == '{"memory": []}'
    assert [_ignore_list(request) for request in requests] == [None, None, None, ["DigitalOcean"]]
    entry = json.loads((tmp_path / STATE_FILENAME).read_text())["models"][_EXTRACTION_MODEL]
    assert entry["failures"] == {"DigitalOcean": 3}
    assert list(entry["ignored"]) == ["DigitalOcean"]


def test_pinned_extraction_routing_is_never_counted_or_given_an_ignore_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    pinned = {"provider": {"order": ["SiliconFlow"], "allow_fallbacks": False}}
    avoidance = ProviderAvoidance(tmp_path, threshold=1)
    avoidance.record_failure(_EXTRACTION_MODEL, "SiliconFlow", kind="provider_error", failure="unpinned")
    state = (tmp_path / STATE_FILENAME).read_text()
    llm, requests = _metered_extraction_llm(
        monkeypatch,
        [_upstream_error("SiliconFlow"), {**_reply('{"memory": []}'), "provider": "SiliconFlow"}],
        provider_avoidance=avoidance,
    )
    llm._provider.extra_body = pinned
    llm._provider._CHAT_RETRY_DELAYS = (0, 0, 0)
    llm.set_logical_task_key("mem0_add:run_001:acc_003:beat_02")

    llm.generate_response(
        messages=[{"role": "user", "content": "pool"}],
        response_format={"type": "json_object"},
    )

    assert [request["extra_body"] for request in requests] == [pinned, pinned]
    assert (tmp_path / STATE_FILENAME).read_text() == state


def test_the_read_only_probe_client_takes_the_mainline_llm_arguments_with_the_run_avoidance(
    tmp_path: Path,
) -> None:
    """The shared probe query factory hands it the mainline client's LLM arguments,
    the run's avoidance among them; it opens no provider, so it only accepts them."""
    with pytest.raises(ValueError, match="already restored collection"):
        create_read_only_mem0_client(
            infer=True,
            provider_name="openrouter",
            api_key="api-key",
            api_base="https://openrouter.ai/api/v1",
            memory_side_model=_EXTRACTION_MODEL,
            extraction_max_output_tokens=MEM0_EXTRACTION_MAX_OUTPUT_TOKENS,
            provider_avoidance=ProviderAvoidance(tmp_path / "run", threshold=3),
            collection_name="mempa_run_001_mem0",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            expected_run_id="run_001",
            allow_existing_collection=False,
            **_factory_storage_kwargs(tmp_path, enable_entity_memory=False),
        )


@pytest.mark.parametrize(
    "response, expected_error",
    [
        (None, "must be a dictionary"),
        ({}, "requires a results list"),
        ({"results": ["bad"]}, "must be a dictionary"),
        (
            {"results": [{"id": "", "memory": "text", "event": "ADD"}]},
            "non-empty string id",
        ),
        (
            {"results": [{"id": "m1", "memory": "text", "event": "UPSERT"}]},
            "unsupported event",
        ),
        (
            {"results": [{"id": "m1", "memory": "", "event": "ADD"}]},
            "non-empty memory text",
        ),
    ],
)
def test_add_response_validation_rejects_malformed_values(
    response: Any,
    expected_error: str,
) -> None:
    with pytest.raises(Mem0ResponseError, match=expected_error):
        _validate_add_response(response)


@pytest.mark.parametrize(
    "response, expected_error",
    [
        (None, "must be a dictionary"),
        ({}, "requires a results list"),
        ({"results": [{"id": "m1", "memory": "", "score": 0.9}]}, "memory text"),
        ({"results": [{"id": "m1", "memory": "text", "score": "high"}]}, "finite number"),
    ],
)
def test_search_response_validation_rejects_malformed_values(
    response: Any,
    expected_error: str,
) -> None:
    with pytest.raises(Mem0ResponseError, match=expected_error):
        _validate_search_response(response)


def test_validated_client_rejects_unreadable_reported_write() -> None:
    memory = SimpleNamespace(
        add=lambda **_: {
            "results": [
                {"id": "memory-1", "memory": "hello", "event": "ADD"}
            ]
        },
        get=lambda _: None,
    )
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    with pytest.raises(
        Mem0PersistenceError,
        match="run_id='run_001'.*id='memory-1'.*read_value=None",
    ):
        client.add(
            messages=[{"role": "user", "content": "hello"}],
            **_ADD_SCOPE,
        )


def test_simple_rag_rejects_partial_raw_dialogue_persistence() -> None:
    memory = SimpleNamespace(
        add=lambda **_: {
            "results": [
                {
                    "id": "memory-1",
                    "memory": "first",
                    "event": "ADD",
                    "role": "user",
                }
            ]
        },
        get=lambda _: None,
    )
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    with pytest.raises(
        Mem0PersistenceError,
        match="exactly one chunk.*expected_count=2 actual_count=1",
    ):
        client.add(
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
            user_id="user_a",
            run_id="run_001",
            metadata=_ADD_SCOPE["metadata"],
            infer=False,
        )


def test_simple_rag_rejects_read_back_from_wrong_run() -> None:
    memory = SimpleNamespace(
        add=lambda **_: {
            "results": [
                {
                    "id": "memory-1",
                    "memory": "hello",
                    "event": "ADD",
                    "role": "user",
                }
            ]
        },
        get=lambda _: {
            "id": "memory-1",
            "memory": "hello",
            "role": "user",
            "user_id": "user_a",
            "run_id": "run_other",
            "metadata": {
                "session_id": "acc_001",
                "turn_id": "open",
                "timestamp": "2026-08-03T10:00:00+08:00",
            },
        },
    )
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    with pytest.raises(
        Mem0PersistenceError,
        match="read-back does not match.*run_other",
    ):
        client.add(
            messages=[{"role": "user", "content": "hello"}],
            user_id="user_a",
            run_id="run_001",
            metadata={
                "session_id": "acc_001",
                "turn_id": "open",
                "timestamp": "2026-08-03T10:00:00+08:00",
            },
            infer=False,
        )


def test_client_rejects_add_for_different_run_before_backend_call() -> None:
    memory = SimpleNamespace(add=Mock())
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    with pytest.raises(
        Mem0ResponseError,
        match="does not match.*expected_run_id='run_001'.*run_other",
    ):
        client.add(
            messages=[{"role": "user", "content": "hello"}],
            user_id="user_a",
            run_id="run_other",
            metadata=_ADD_SCOPE["metadata"],
            infer=False,
        )

    memory.add.assert_not_called()


def test_validated_client_rejects_search_with_invalid_score() -> None:
    memory = SimpleNamespace(
        search=lambda **_: {
            "results": [
                {"id": "memory-1", "memory": "hello", "score": "high"}
            ]
        }
    )
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    with pytest.raises(Mem0ResponseError, match="value='high'"):
        client.search(
            query="hello",
            filters={"user_id": "user_a", "run_id": "run_001"},
        )


def test_validated_client_accepts_search_results_in_requested_scope() -> None:
    result = {
        "results": [
            {
                "id": "memory-1",
                "memory": "hello",
                "score": 0.8,
                "user_id": "user_a",
                "run_id": "run_001",
            }
        ]
    }
    memory = SimpleNamespace(search=Mock(return_value=result))
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    assert client.search(
        query="hello",
        filters={"user_id": "user_a", "run_id": "run_001"},
        top_k=10,
        threshold=0.1,
        rerank=False,
    ) == result


def test_validated_client_rejects_search_result_outside_scope() -> None:
    memory = SimpleNamespace(
        search=Mock(
            return_value={
                "results": [
                    {
                        "id": "memory-1",
                        "memory": "wrong run",
                        "score": 0.8,
                        "user_id": "user_a",
                        "run_id": "run_other",
                    }
                ]
            }
        )
    )
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    with pytest.raises(
        Mem0PersistenceError,
        match="outside the requested scope.*run_other",
    ):
        client.search(
            query="hello",
            filters={"user_id": "user_a", "run_id": "run_001"},
        )


class _FakeQdrantClient:
    collection_exists_value = False
    entity_collection_exists_value = False
    existing_collections: frozenset[str] = frozenset()
    deleted_collections: list[str] = []
    scroll_pages: list[list[Any]] = []
    connection_error: Exception | None = None
    vector_size = 1536
    vector_distance = "Cosine"
    vectors_on_disk = True
    sparse_vectors: dict[str, Any] | None = None
    init_kwargs: list[dict[str, Any]] = []

    def __init__(self, *, host: str, port: int, timeout: int | None = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        type(self).init_kwargs.append({"host": host, "port": port, "timeout": timeout})

    def get_collections(self) -> object:
        if self.connection_error is not None:
            raise self.connection_error
        return object()

    def collection_exists(self, collection_name: str) -> bool:
        if collection_name in self.existing_collections:
            return True
        if collection_name.endswith("_entities"):
            return self.entity_collection_exists_value
        return self.collection_exists_value

    def delete_collection(self, collection_name: str) -> None:
        cls = type(self)
        cls.deleted_collections.append(collection_name)
        cls.existing_collections = cls.existing_collections - {collection_name}
        if collection_name.endswith("_entities"):
            cls.entity_collection_exists_value = False
        else:
            cls.collection_exists_value = False

    def get_collection(self, collection_name: str) -> Any:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(
                        size=self.vector_size,
                        distance=SimpleNamespace(value=self.vector_distance),
                        on_disk=self.vectors_on_disk,
                    ),
                    sparse_vectors=self.sparse_vectors,
                )
            )
        )

    def scroll(self, **kwargs: Any) -> tuple[list[Any], int | None]:
        page_index = kwargs.get("offset") or 0
        if page_index >= len(self.scroll_pages):
            return [], None
        next_page_offset = (
            page_index + 1
            if page_index + 1 < len(self.scroll_pages)
            else None
        )
        return self.scroll_pages[page_index], next_page_offset


def _install_fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_qdrant = ModuleType("qdrant_client")
    fake_qdrant.QdrantClient = _FakeQdrantClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant)


def test_qdrant_rejects_existing_collection_for_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", True)

    with pytest.raises(FileExistsError, match="mempa_run_001_mem0"):
        _assert_qdrant_collection_ready(
            collection_name="mempa_run_001_mem0",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            expected_run_id="run_001",
            allow_existing_collection=False,
            **_QDRANT_SCHEMA_KWARGS,
        )


def _arm_overwrite_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing: set[str],
) -> list[str]:
    """A fake Qdrant holding exactly `existing`, recording every drop."""
    _install_fake_qdrant(monkeypatch)
    deleted: list[str] = []
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", False)
    monkeypatch.setattr(
        _FakeQdrantClient,
        "entity_collection_exists_value",
        False,
    )
    monkeypatch.setattr(
        _FakeQdrantClient,
        "existing_collections",
        frozenset(existing),
    )
    monkeypatch.setattr(_FakeQdrantClient, "deleted_collections", deleted)
    return deleted


def test_qdrant_overwrite_drops_this_runs_collection_before_creating_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted = _arm_overwrite_fake(
        monkeypatch,
        existing={"mempa_run_001_mem0"},
    )

    _assert_qdrant_collection_ready(
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        allow_existing_collection=False,
        drop_existing_collection=True,
        **_QDRANT_SCHEMA_KWARGS,
    )

    assert deleted == ["mempa_run_001_mem0"]


def test_qdrant_overwrite_drops_the_entity_sibling_of_this_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted = _arm_overwrite_fake(
        monkeypatch,
        existing={"mempa_run_001_mem0", "mempa_run_001_mem0_entities"},
    )

    _assert_qdrant_collection_ready(
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        allow_existing_collection=False,
        drop_existing_collection=True,
        **_QDRANT_SCHEMA_KWARGS,
    )

    assert deleted == [
        "mempa_run_001_mem0",
        "mempa_run_001_mem0_entities",
    ]


def test_qdrant_overwrite_never_touches_another_runs_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted = _arm_overwrite_fake(
        monkeypatch,
        existing={"mempa_run_001_mem0", "mempa_run_999_mem0"},
    )

    _assert_qdrant_collection_ready(
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        allow_existing_collection=False,
        drop_existing_collection=True,
        **_QDRANT_SCHEMA_KWARGS,
    )

    assert deleted == ["mempa_run_001_mem0"]
    assert "mempa_run_999_mem0" not in deleted


def test_qdrant_without_overwrite_refuses_instead_of_dropping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted = _arm_overwrite_fake(
        monkeypatch,
        existing={"mempa_run_001_mem0"},
    )

    with pytest.raises(FileExistsError, match="mempa_run_001_mem0"):
        _assert_qdrant_collection_ready(
            collection_name="mempa_run_001_mem0",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            expected_run_id="run_001",
            allow_existing_collection=False,
            **_QDRANT_SCHEMA_KWARGS,
        )

    assert deleted == []


def test_qdrant_rejects_entity_collection_when_entity_memory_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", True)
    monkeypatch.setattr(
        _FakeQdrantClient,
        "entity_collection_exists_value",
        True,
    )

    with pytest.raises(
        ValueError,
        match="Entity memory is disabled.*mempa_run_001_mem0_entities",
    ):
        _assert_qdrant_collection_ready(
            collection_name="mempa_run_001_mem0",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            expected_run_id="run_001",
            allow_existing_collection=True,
            **_QDRANT_SCHEMA_KWARGS,
        )


def test_qdrant_rejects_collection_owned_by_different_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", True)
    monkeypatch.setattr(
        _FakeQdrantClient,
        "scroll_pages",
        [[SimpleNamespace(id="point-1", payload={"run_id": "run_other"})]],
    )

    with pytest.raises(ValueError, match="stored_run_id='run_other'"):
        _assert_qdrant_collection_ready(
            collection_name="mempa_run_001_mem0",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            expected_run_id="run_001",
            allow_existing_collection=True,
            **_QDRANT_SCHEMA_KWARGS,
        )


def test_qdrant_rejects_wrong_owner_after_valid_first_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", True)
    monkeypatch.setattr(
        _FakeQdrantClient,
        "scroll_pages",
        [
            [SimpleNamespace(id="point-1", payload={"run_id": "run_001"})],
            [SimpleNamespace(id="point-2", payload={"run_id": "run_other"})],
        ],
    )

    with pytest.raises(
        ValueError,
        match="point_id='point-2'.*stored_run_id='run_other'",
    ):
        _probe_qdrant_collection(
            collection_name="mempa_run_001_mem0",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            expected_run_id="run_001",
            require_collection=True,
            allow_existing_collection=True,
            **_QDRANT_SCHEMA_KWARGS,
        )


def test_qdrant_runtime_health_requires_expected_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", False)

    with pytest.raises(
        FileNotFoundError,
        match="collection_name='mempa_run_001_mem0'.*expected_run_id='run_001'",
    ):
        _probe_qdrant_collection(
            collection_name="mempa_run_001_mem0",
            qdrant_host="qdrant.internal",
            qdrant_port=7444,
            expected_run_id="run_001",
            require_collection=True,
            allow_existing_collection=True,
            **_QDRANT_SCHEMA_KWARGS,
        )


def test_qdrant_runtime_health_reports_failed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(
        _FakeQdrantClient,
        "connection_error",
        ConnectionRefusedError("refused"),
    )

    with pytest.raises(
        ConnectionError,
        match="host='qdrant.internal'.*port=7444.*ConnectionRefusedError: refused",
    ):
        _probe_qdrant_collection(
            collection_name="mempa_run_001_mem0",
            qdrant_host="qdrant.internal",
            qdrant_port=7444,
            expected_run_id="run_001",
            require_collection=True,
            allow_existing_collection=True,
            **_QDRANT_SCHEMA_KWARGS,
        )


def test_qdrant_rejects_collection_with_wrong_vector_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", True)
    monkeypatch.setattr(_FakeQdrantClient, "vector_size", 3072)

    with pytest.raises(
        ValueError,
        match="schema does not match.*embedding_dimensions.*3072",
    ):
        _probe_qdrant_collection(
            collection_name="mempa_run_001_mem0",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            expected_run_id="run_001",
            require_collection=True,
            allow_existing_collection=True,
            **_QDRANT_SCHEMA_KWARGS,
        )


def test_client_health_probes_embedding_and_validates_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", True)
    embed_calls: list[tuple[str, str]] = []

    def embed(text: str, memory_action: str) -> list[float]:
        embed_calls.append((text, memory_action))
        return [0.1] * 1536

    memory = SimpleNamespace(
        embedding_model=SimpleNamespace(embed=embed),
    )
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    assert client.health() >= 0
    assert embed_calls == [("MemPA embedding readiness probe", "search")]


def test_native_entity_ids_inspects_the_owned_entity_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memory.backends.mem0_client._inspect_qdrant_collection",
        lambda **kwargs: (calls.append(kwargs) or ({}, {"entity-1", "entity-2"})),
    )
    client = _ValidatedMem0Client(
        SimpleNamespace(),
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **{**_CLIENT_VALIDATION_KWARGS, "enable_entity_memory": True},
    )

    assert client.native_entity_ids() == {"entity-1", "entity-2"}
    assert calls == [{
        "collection_name": "mempa_run_001_mem0_entities",
        "qdrant_host": "127.0.0.1",
        "qdrant_port": 6333,
        "expected_run_id": "run_001",
        "require_collection": True,
        "expected_embedding_dimensions": 1536,
        "expected_distance": "Cosine",
        "expected_on_disk": True,
        "expected_bm25_enabled": False,
        "expected_entity_memory_enabled": False,
    }]


def test_client_health_rejects_wrong_embedding_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", True)
    memory = SimpleNamespace(
        embedding_model=SimpleNamespace(
            embed=lambda *_args, **_kwargs: [0.1] * 3
        ),
    )
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    with pytest.raises(
        Mem0ResponseError,
        match="wrong dimensions.*expected=1536 actual=3",
    ):
        client.health()


def test_history_db_path_requires_explicit_run_ownership(tmp_path: Path) -> None:
    history_db_path = tmp_path / "backend_state" / "mem0_history.sqlite3"
    history_db_path.parent.mkdir(parents=True)
    history_db_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        _prepare_history_db_path(
            history_db_path,
            allow_existing_history_db=False,
            require_existing_history_db=False,
        )

    missing_path = tmp_path / "other" / "mem0_history.sqlite3"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _prepare_history_db_path(
            missing_path,
            allow_existing_history_db=True,
            require_existing_history_db=True,
        )


def test_entity_memory_rejects_missing_nlp_before_backend_initialization(monkeypatch, tmp_path):
    from memory.backends import mem0_client
    monkeypatch.setattr(mem0_client, "version", lambda name: (_ for _ in ()).throw(mem0_client.PackageNotFoundError(name)))
    monkeypatch.setattr(mem0_client, "_assert_qdrant_collection_ready", lambda **kwargs: pytest.fail("Backend must not be touched"))
    with pytest.raises(RuntimeError, match="locked spaCy"):
        create_mem0_client(
            provider_name="openrouter", api_key="test", api_base="https://openrouter.ai/api/v1",
            memory_side_model="deepseek/deepseek-v4-flash", collection_name="untouched",
            qdrant_host="127.0.0.1", qdrant_port=6333, expected_run_id="run",
            allow_existing_collection=False, **_factory_storage_kwargs(tmp_path, enable_entity_memory=True),
        )


# ── The pool passes the raw-dialogue role check ─────────

from memory.backends.mem0_client import (  # noqa: E402
    _RAW_DIALOGUE_ROLES,
    _validated_raw_dialogue_messages,
)

from test_memory.pool_fixtures import full_experience_pool  # noqa: E402


class _EchoMemory:
    """Persists one raw chunk per visible message, byte for byte."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        results = []
        for index, message in enumerate(messages):
            memory_id = f"memory-{index}"
            record = {
                "id": memory_id,
                "memory": message["content"],
                "event": "ADD",
                "role": message["role"],
            }
            self.store[memory_id] = {
                **record,
                "user_id": kwargs.get("user_id"),
                "agent_id": kwargs.get("agent_id"),
                "run_id": kwargs.get("run_id"),
                "metadata": kwargs.get("metadata"),
            }
            results.append(record)
        return {"results": results}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        return self.store.get(memory_id)


def test_pool_with_tagged_assistant_blocks_passes_the_role_check() -> None:
    pool = full_experience_pool()

    validated = _validated_raw_dialogue_messages(pool.messages)

    assert validated == [
        {"role": m["role"], "content": m["content"]} for m in pool.messages
    ]
    assert {m["role"] for m in validated} <= _RAW_DIALOGUE_ROLES
    tagged = [m for m in validated if m["content"].startswith("[")]
    assert tagged, "the pool must carry at least one tagged block"
    assert all(m["role"] == "assistant" for m in tagged)


def test_simple_rag_persists_every_pool_message_byte_equal() -> None:
    pool = full_experience_pool()
    memory = _EchoMemory()
    client = _ValidatedMem0Client(
        memory,
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        **_CLIENT_VALIDATION_KWARGS,
    )

    result = client.add(messages=pool.messages, infer=False, **_ADD_SCOPE)

    assert [item["memory"] for item in result["results"]] == [
        m["content"] for m in pool.messages
    ]
    assert [item["role"] for item in result["results"]] == [
        m["role"] for m in pool.messages
    ]


def test_qdrant_admin_clients_carry_a_120s_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every Qdrant handle we build must outlast a slow snapshot request.

    The library default is 5 s; the live user_b mem0 run died on a snapshot
    POST that took 6.50 s.
    """
    _install_fake_qdrant(monkeypatch)
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(_FakeQdrantClient, "init_kwargs", captured)
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", False)
    monkeypatch.setattr(_FakeQdrantClient, "connection_error", RuntimeError("stop"))

    _drop_qdrant_collection_for_run(
        collection_name="mempa_run_001_mem0",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
    )
    with pytest.raises(ConnectionError):
        _inspect_qdrant_collection(
            collection_name="mempa_run_001_mem0",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            expected_run_id="run_001",
            require_collection=True,
            expected_embedding_dimensions=1536,
            expected_distance="Cosine",
            expected_on_disk=True,
            expected_bm25_enabled=False,
            expected_entity_memory_enabled=False,
        )
    _qdrant_admin(SimpleNamespace(_qdrant_host="127.0.0.1", _qdrant_port=6333))

    assert [kwargs["timeout"] for kwargs in captured] == [120, 120, 120]


def test_authorized_execution_budget_reaches_real_mem0_request_without_changing_defaults(monkeypatch):
    llm, requests = _metered_extraction_llm(monkeypatch, [_reply('{"memory": []}')], extraction_max_output_tokens=32000)
    llm.set_logical_task_key("mem0_add:run_001:acc_003:beat_02")
    assert llm.generate_response(messages=[{"role":"user","content":"pool"}],response_format={"type":"json_object"}) == '{"memory": []}'
    assert requests[0]["max_tokens"] == 32000
    assert MEM0_EXTRACTION_MAX_OUTPUT_TOKENS == 16000
    assert requests[0]["temperature"] == 0.1


@pytest.mark.parametrize("budget",[0,-1,True,"32000"])
def test_invalid_mem0_budget_rejected_before_provider_construction(budget):
    with pytest.raises(ValueError,match="positive integer"):
        _MemoryExtractionLLM(provider_name="openrouter",api_key="offline",api_base="https://invalid.test",model=_EXTRACTION_MODEL,extraction_max_output_tokens=budget)
