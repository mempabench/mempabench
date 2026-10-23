"""Contract tests for the self-hosted MemOS HTTP client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from memory.backends import memos_client
from memory.backends.memos_client import (
    MEMOS_MEMPA_PATCH_VERSION,
    MEMOS_SOURCE_SHA256,
    MemOSHttpClient,
    MemOSResponseError,
    MemOSTransportError,
)


def _grouped_data() -> dict[str, Any]:
    return {
        "text_mem": [
            {
                "cube_id": "cube_001",
                "memories": [
                    {
                        "id": "text_1",
                        "memory": "User prefers concise updates.",
                        "metadata": {"relativity": 0.91},
                    }
                ],
                "total_nodes": 1,
            }
        ],
        "act_mem": [],
        "para_mem": [],
        "pref_mem": [
            {
                "cube_id": "cube_001",
                "memories": [
                    {
                        "id": "pref_1",
                        "memory": "Use concise status updates.",
                        "score": 0.87,
                    }
                ],
                "total_nodes": 1,
            }
        ],
        "tool_mem": [],
        "skill_mem": [],
        "pref_note": "The user values brevity.",
    }


def _runtime_contract() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "memos_version": "2.0.15",
        "memos_submodule_commit": "1cf8a78fda66ebb529f2b599a66741fbf5c9a220",
        "memos_source_sha256": MEMOS_SOURCE_SHA256,
        "mempa_patch_version": MEMOS_MEMPA_PATCH_VERSION,
        "memory_side_provider_name": "openrouter",
        "memory_side_model": "openai/gpt-5.5",
        "memory_side_api_base": "https://openrouter.ai/api/v1",
        "memory_side_model_roles": {
            "chat": "openai/gpt-5.5",
            "general_reader": "openai/gpt-5.5",
            "memory_reader": "openai/gpt-5.5",
        },
        "memory_side_request_policy_sha256": "a" * 64,
        "memory_side_temperature_policy": "send",
        "memory_side_completion_tokens_parameter": "max_completion_tokens",
        "memory_side_reasoning_effort_policy": "send",
        "memory_side_task_affinity_required": False,
        "memory_side_task_affinity_namespace": "",
        "memory_side_backup_enabled": False,
        "memory_profile": "text_memory",
        "activation_memory_enabled": False,
        "preference_memory_enabled": False,
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_api_base": "https://api.openai.com/v1",
        "embedding_dimensions": 1536,
        "reranker_backend": "cosine_local",
        "qdrant_host": "qdrant",
        "qdrant_port": 6333,
        "qdrant_image": "qdrant/qdrant:v1.15.3",
        "neo4j_backend": "neo4j-community",
        "neo4j_uri": "bolt://neo4j:7687",
        "neo4j_database": "neo4j",
        "neo4j_image": "neo4j:5.26.6",
        "chat_api_enabled": False,
    }


def _health_payload(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "memos",
        "version": "1.0.1",
        "mempa_patch_version": MEMOS_MEMPA_PATCH_VERSION,
        "runtime_contract": contract or _runtime_contract(),
        "dependencies": {"qdrant": "healthy", "neo4j": "healthy"},
    }


def _client(handler: Any) -> MemOSHttpClient:
    return MemOSHttpClient(
        api_base="http://memos.test:8000",
        expected_runtime_contract=_runtime_contract(),
        transport=httpx.MockTransport(handler),
    )


def _envelope(data: Any, *, code: int = 200, message: str = "ok") -> dict[str, Any]:
    return {"code": code, "message": message, "data": data}


def _search_request() -> dict[str, Any]:
    return {
        "query": "updates",
        "user_id": "user_a",
        "readable_cube_ids": ["cube_001"],
    }


def test_requires_absolute_api_base() -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        MemOSHttpClient(api_base="localhost:8000")


def test_rejects_credentials_in_api_base() -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        MemOSHttpClient(api_base="http://user:secret@memos.test:8000")


@pytest.mark.parametrize("timeout", [0.0, float("nan"), float("inf"), -float("inf")])
def test_rejects_non_finite_or_non_positive_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MemOSHttpClient(
            api_base="http://memos.test:8000",
            timeout_seconds=timeout,
        )


def test_health_validates_service_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            200,
            json=_health_payload(),
        )

    assert _client(handler).health()["status"] == "healthy"


def test_health_rejects_process_only_false_success() -> None:
    client = _client(
        lambda _: httpx.Response(
            200,
            json={**_health_payload(), "status": "starting"},
        )
    )
    with pytest.raises(MemOSResponseError, match="invalid health response"):
        client.health()


def test_health_rejects_service_without_required_mempa_patch() -> None:
    client = _client(
        lambda _: httpx.Response(
            200,
            json={"status": "healthy", "service": "memos", "version": "1.0.1"},
        )
    )

    with pytest.raises(MemOSResponseError, match="mempa_patch_version=None"):
        client.health()


def test_health_returns_a_differing_runtime_contract_for_the_caller_to_record() -> None:
    actual = _runtime_contract()
    actual["memory_side_model"] = "different-model"
    client = _client(
        lambda _: httpx.Response(200, json=_health_payload(actual))
    )

    assert client.health()["runtime_contract"]["memory_side_model"] == "different-model"


def test_health_rejects_unavailable_dependency() -> None:
    payload = _health_payload()
    payload["dependencies"]["neo4j"] = "unavailable"
    client = _client(lambda _: httpx.Response(200, json=payload))

    with pytest.raises(MemOSResponseError, match="dependency health is invalid"):
        client.health()


def test_add_unwraps_and_validates_native_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/product/add"
        return httpx.Response(
            200,
            json=_envelope(
                [
                    {
                        "memory_id": "mem_1",
                        "memory": "User prefers concise updates.",
                        "memory_type": "PreferenceMemory",
                        "cube_id": "cube_001",
                    }
                ]
            ),
        )

    result = _client(handler).add_memories(
        {"messages": [], "writable_cube_ids": ["cube_001"]}
    )
    assert result[0]["memory_id"] == "mem_1"


def test_add_allows_valid_empty_extraction() -> None:
    client = _client(lambda _: httpx.Response(200, json=_envelope([])))
    assert client.add_memories(
        {"messages": [], "writable_cube_ids": ["cube_001"]}
    ) == []


def test_add_rejects_item_without_native_id() -> None:
    client = _client(
        lambda _: httpx.Response(
            200,
            json=_envelope([{"memory": "missing id"}]),
        )
    )
    with pytest.raises(MemOSResponseError, match="non-empty memory_id/id"):
        client.add_memories(
            {"messages": [], "writable_cube_ids": ["cube_001"]}
        )


@pytest.mark.parametrize(
    ("response_item", "error_pattern"),
    [
        (
            {
                "memory_id": "mem_1",
                "memory": "User prefers concise updates.",
                "cube_id": "wrong_cube",
            },
            "cube_id.*wrong_cube.*expected.*cube_001",
        ),
        (
            {"memory_id": "mem_1", "memory": "  ", "cube_id": "cube_001"},
            "non-empty memory text",
        ),
    ],
)
def test_add_rejects_uncommitted_response_evidence(
    response_item: dict[str, Any],
    error_pattern: str,
) -> None:
    client = _client(
        lambda _: httpx.Response(200, json=_envelope([response_item]))
    )

    with pytest.raises(MemOSResponseError, match=error_pattern):
        client.add_memories(
            {"messages": [], "writable_cube_ids": ["cube_001"]}
        )


def test_add_rejects_duplicate_native_ids() -> None:
    item = {
        "memory_id": "mem_1",
        "memory": "User prefers concise updates.",
        "cube_id": "cube_001",
    }
    client = _client(
        lambda _: httpx.Response(200, json=_envelope([item, dict(item)]))
    )

    with pytest.raises(MemOSResponseError, match="duplicate.*mem_1"):
        client.add_memories(
            {"messages": [], "writable_cube_ids": ["cube_001"]}
        )


def test_add_rejects_conflicting_id_fields() -> None:
    client = _client(
        lambda _: httpx.Response(
            200,
            json=_envelope(
                [
                    {
                        "id": "mem_1",
                        "memory_id": "mem_2",
                        "memory": "User prefers concise updates.",
                        "cube_id": "cube_001",
                    }
                ]
            ),
        )
    )

    with pytest.raises(MemOSResponseError, match="conflicting id fields"):
        client.add_memories(
            {"messages": [], "writable_cube_ids": ["cube_001"]}
        )


@pytest.mark.parametrize("operation", ["get_memory", "search"])
def test_grouped_memory_response_preserves_cube_buckets(operation: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/product/{operation}"
        return httpx.Response(200, json=_envelope(_grouped_data()))

    client = _client(handler)
    if operation == "get_memory":
        result = client.get_memories({"mem_cube_id": "cube_001"})
    else:
        result = client.search_memories(_search_request())
    assert result["text_mem"][0]["cube_id"] == "cube_001"
    assert result["text_mem"][0]["memories"][0]["id"] == "text_1"


def test_search_rejects_old_flat_fake_shape() -> None:
    invalid = {
        "text_mem": [{"id": "text_1", "memory": "flat"}],
        "pref_mem": [],
        "tool_mem": [],
        "skill_mem": [],
    }
    client = _client(lambda _: httpx.Response(200, json=_envelope(invalid)))
    with pytest.raises(MemOSResponseError, match="requires cube_id"):
        client.search_memories(_search_request())


@pytest.mark.parametrize(
    "score",
    [None, "bad", True, -0.1, 1.1],
)
def test_search_requires_valid_text_memory_relativity(score: Any) -> None:
    invalid = _grouped_data()
    memory = invalid["text_mem"][0]["memories"][0]
    if score is None:
        memory.pop("metadata", None)
    else:
        memory["metadata"] = {"relativity": score}
    client = _client(lambda _: httpx.Response(200, json=_envelope(invalid)))

    with pytest.raises(
        MemOSResponseError,
        match="metadata.relativity must be a finite number between 0 and 1",
    ):
        client.search_memories(_search_request())


@pytest.mark.parametrize("group_name", ["act_mem", "para_mem"])
def test_search_validates_activity_and_paragraph_groups(
    group_name: str,
) -> None:
    invalid = _grouped_data()
    invalid[group_name] = [{"id": "memory_1", "memory": "flat"}]
    client = _client(lambda _: httpx.Response(200, json=_envelope(invalid)))

    with pytest.raises(
        MemOSResponseError,
        match=rf"{group_name}\[0\] requires cube_id",
    ):
        client.search_memories(_search_request())


@pytest.mark.parametrize("operation", ["get_memory", "search"])
def test_grouped_memory_response_rejects_unrequested_cube(operation: str) -> None:
    invalid = _grouped_data()
    invalid["text_mem"][0]["cube_id"] = "cube_other_run"
    client = _client(lambda _: httpx.Response(200, json=_envelope(invalid)))

    with pytest.raises(
        MemOSResponseError,
        match="cube_other_run.*does not match requested cube ids.*cube_001",
    ):
        if operation == "get_memory":
            client.get_memories({"mem_cube_id": "cube_001"})
        else:
            client.search_memories(_search_request())


def test_search_requires_explicit_readable_cube_ids() -> None:
    client = _client(lambda _: pytest.fail("request must not be sent"))

    with pytest.raises(ValueError, match="readable_cube_ids"):
        client.search_memories({"query": "updates", "user_id": "user_a"})


def test_search_rejects_api_failure_code() -> None:
    client = _client(
        lambda _: httpx.Response(
            200,
            json=_envelope(None, code=500, message="search backend failed"),
        )
    )
    with pytest.raises(MemOSResponseError, match="code is not 200"):
        client.search_memories(_search_request())


def test_response_requires_data_field() -> None:
    client = _client(
        lambda _: httpx.Response(200, json={"code": 200, "message": "ok"})
    )
    with pytest.raises(MemOSResponseError, match="missing data"):
        client.add_memories(
            {"messages": [], "writable_cube_ids": ["cube_001"]}
        )


def test_delete_requires_native_success_status() -> None:
    client = _client(
        lambda _: httpx.Response(
            200,
            json=_envelope({"status": "failure"}, message="delete failed"),
        )
    )
    with pytest.raises(MemOSResponseError, match="delete did not succeed"):
        client.delete_memories({"memory_ids": ["mem_1"]})


def test_delete_requires_deleted_count_to_match_explicit_ids() -> None:
    client = _client(
        lambda _: httpx.Response(
            200,
            json=_envelope(
                {"status": "success", "deleted_count": 1},
                message="delete succeeded",
            ),
        )
    )

    with pytest.raises(MemOSResponseError, match="expected_count=2 deleted_count=1"):
        client.delete_memories({"memory_ids": ["mem_1", "mem_2"]})


def test_delete_requires_integer_deleted_count() -> None:
    client = _client(
        lambda _: httpx.Response(
            200,
            json=_envelope({"status": "success"}, message="delete succeeded"),
        )
    )

    with pytest.raises(MemOSResponseError, match="deleted_count must be an integer"):
        client.delete_memories({"memory_ids": ["mem_1"]})


def test_http_error_contains_operation_and_status() -> None:
    client = _client(lambda _: httpx.Response(503, text="service unavailable"))
    with pytest.raises(
        MemOSResponseError,
        match="operation='search'.*status_code=503",
    ):
        client.search_memories(_search_request())


def test_transport_error_preserves_cause() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    with pytest.raises(MemOSTransportError, match="operation='health'") as error:
        client.health()
    assert isinstance(error.value.__cause__, httpx.ConnectError)


def _add_request() -> dict[str, Any]:
    return {"messages": [], "writable_cube_ids": ["cube_001"]}


def test_add_replays_once_after_a_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(memos_client, "_RETRY_PAUSE_SECONDS", 0.0)
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, json=_envelope([]))

    assert _client(handler).add_memories(_add_request()) == []
    assert attempts == ["/product/add", "/product/add"]
    assert (
        "memos_transport_retry op=add attempt=2 reason=ConnectTimeout: timed out"
        in capsys.readouterr().out
    )


def test_add_raises_after_a_second_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memos_client, "_RETRY_PAUSE_SECONDS", 0.0)
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        raise httpx.ConnectTimeout("timed out", request=request)

    with pytest.raises(MemOSTransportError, match="operation='add'") as error:
        _client(handler).add_memories(_add_request())
    assert len(attempts) == 2
    assert isinstance(error.value.__cause__, httpx.ConnectTimeout)


def test_add_is_not_replayed_after_a_read_timeout() -> None:
    """A read timeout means MemOS got the add and may still be running it.

    `/product/add` mints a fresh UUID per extracted memory and carries no
    dedupe key, so replaying it would write the same memories twice.
    """
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(MemOSTransportError, match="operation='add'"):
        _client(handler).add_memories(_add_request())
    assert attempts == ["/product/add"]


def test_search_replays_once_after_a_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(memos_client, "_RETRY_PAUSE_SECONDS", 0.0)
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=_envelope(_grouped_data()))

    result = _client(handler).search_memories(_search_request())
    assert result["pref_note"] == "The user values brevity."
    assert attempts == ["/product/search", "/product/search"]
    assert (
        "memos_transport_retry op=search attempt=2 reason=ReadTimeout: timed out"
        in capsys.readouterr().out
    )


def test_add_waits_ten_minutes_while_other_operations_wait_three() -> None:
    """The add carries its own long timeout; every other operation keeps 180 s."""
    read_timeouts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        read_timeouts.append(request.extensions["timeout"]["read"])
        data = [] if request.url.path == "/product/add" else _grouped_data()
        return httpx.Response(200, json=_envelope(data))

    client = _client(handler)
    client.add_memories(_add_request())
    client.search_memories(_search_request())

    assert read_timeouts == [600.0, 180.0]


# ── checkpoint ────────────────────────────────────────────────────


def test_export_cube_uses_frozen_cube_id() -> None:
    """The export names the cube it was asked for, verbatim."""
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "code": 200,
                "message": "ok",
                "data": {"cube_id": "cube_frozen", "memories": []},
            },
        )

    client = MemOSHttpClient(
        api_base="http://memos.local",
        transport=httpx.MockTransport(handler),
    )

    payload = client.export_cube("cube_frozen")

    assert payload == {"cube_id": "cube_frozen", "memories": []}
    assert seen == [("/product/export_cube", {"mem_cube_id": "cube_frozen"})]


def _import_snapshot() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "storage": "neo4j-community-qdrant",
        "cube_id": "cube_001",
        "nodes": [{"labels": ["Memory"], "properties": {"id": "n1"}}],
        "edges": [],
        "points": [{"id": "n1", "vector": [0.1], "payload": {}}],
    }


def test_import_cube_returns_the_service_counts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(
                {"status": "success", "cube_id": "cube_001", "node_count": 1, "point_count": 1}
            ),
        )

    data = _client(handler).import_cube_as(_import_snapshot(), "cube_001")

    assert data["point_count"] == 1


def test_import_cube_rejects_a_count_that_misses_the_snapshot() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(
                {"status": "success", "cube_id": "cube_001", "node_count": 1, "point_count": 0}
            ),
        )

    with pytest.raises(MemOSResponseError, match="restored point_count does not match"):
        _client(handler).import_cube_as(_import_snapshot(), "cube_001")
