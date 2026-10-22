from __future__ import annotations

import json

import pytest

from harness.memory_io import MemoryIOAuditError, MemoryIORecorder


def _operation(run_id: str, *, status: str = "succeeded") -> dict:
    return {
        "run_id": run_id,
        "step_id": "acc_001",
        "session_id": f"{run_id}_main",
        "turn_id": "acc_001:beat_01",
        "backend": "mem0",
        "operation": "search",
        "adapter_request": {"query": "hello"},
        "backend_request": {
            "query": "hello",
            "filters": {"run_id": run_id, "user_id": "user_a"},
        },
        "backend_response": {"results": []},
        "status": status,
        "native_ids": [],
        "native_events": [],
        "result_count": 0,
        "top_score": None,
        "latency_ms": 1.0,
        "audit_only_labels": {},
        "error": None,
    }


def test_recorder_resumes_with_next_sequence(tmp_path) -> None:
    path = tmp_path / "memory_io.jsonl"
    MemoryIORecorder(path, run_id="run_001").record(_operation("run_001"))
    MemoryIORecorder(path, run_id="run_001").record(_operation("run_001"))

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["sequence"] for record in records] == [1, 2]


def test_recorder_reads_validated_existing_records(tmp_path) -> None:
    path = tmp_path / "memory_io.jsonl"
    recorder = MemoryIORecorder(path, run_id="run_001")
    recorder.record(_operation("run_001"))

    records = recorder.read_records()

    assert len(records) == 1
    assert records[0]["run_id"] == "run_001"
    assert records[0]["sequence"] == 1


@pytest.mark.parametrize(
    "sensitive_field",
    [
        "api_key",
        "Authorization",
        "access_token",
        "client_secret",
        "neo4j_auth",
        "storage_access_key",
        "database_password",
        "service_credentials",
    ],
)
def test_recorder_rejects_sensitive_fields(
    tmp_path,
    sensitive_field: str,
) -> None:
    operation = _operation("run_001")
    secret_value = "must-not-appear-in-error"
    operation["backend_request"][sensitive_field] = secret_value

    with pytest.raises(
        MemoryIOAuditError,
        match=f"backend_request.{sensitive_field}",
    ) as exc_info:
        MemoryIORecorder(
            tmp_path / "memory_io.jsonl",
            run_id="run_001",
        ).record(operation)
    assert secret_value not in str(exc_info.value)


@pytest.mark.parametrize("field", ["sequence", "timestamp"])
def test_recorder_rejects_caller_owned_sequence_or_timestamp(
    tmp_path,
    field: str,
) -> None:
    operation = _operation("run_001")
    operation[field] = "caller-value"

    with pytest.raises(
        MemoryIOAuditError,
        match=f"recorder-owned field: field='{field}' value='caller-value'",
    ):
        MemoryIORecorder(
            tmp_path / "memory_io.jsonl",
            run_id="run_001",
        ).record(operation)


def test_recorder_strips_nested_vector_payloads_without_mutating_input(
    tmp_path,
) -> None:
    path = tmp_path / "memory_io.jsonl"
    operation = _operation("run_001")
    operation["backend_response"] = {
        "results": [
            {
                "id": "memory_001",
                "embedding": [0.1, 0.2],
                "details": {
                    "dense_vector": [0.3, 0.4],
                    "embedding_model": "text-embedding-3-small",
                },
            }
        ],
        "vectors": [[0.5, 0.6]],
    }

    MemoryIORecorder(path, run_id="run_001").record(operation)

    record = json.loads(path.read_text(encoding="utf-8"))
    result = record["backend_response"]["results"][0]
    assert "embedding" not in result
    assert "dense_vector" not in result["details"]
    assert result["details"]["embedding_model"] == "text-embedding-3-small"
    assert "vectors" not in record["backend_response"]
    assert operation["backend_response"]["results"][0]["embedding"] == [
        0.1,
        0.2,
    ]


def test_recorder_rejects_existing_records_from_another_run(tmp_path) -> None:
    path = tmp_path / "memory_io.jsonl"
    MemoryIORecorder(path, run_id="run_001").record(_operation("run_001"))

    with pytest.raises(
        MemoryIOAuditError,
        match="different run.*expected_run_id='run_002'.*value='run_001'",
    ):
        MemoryIORecorder(path, run_id="run_002")


def test_recorder_does_not_echo_invalid_json_content(tmp_path) -> None:
    path = tmp_path / "memory_io.jsonl"
    secret_value = "must-not-appear-in-json-error"
    path.write_text(f'{{"run_id":"run_001","secret":"{secret_value}"', encoding="utf-8")

    with pytest.raises(MemoryIOAuditError, match="invalid JSON") as exc_info:
        MemoryIORecorder(path, run_id="run_001")
    assert secret_value not in str(exc_info.value)


def test_recorder_rejects_secret_echoed_inside_error_text(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "real-secret-value")
    operation = _operation("run_001")
    operation["error"] = {
        "type": "RemoteError",
        "message": "request rejected for real-secret-value",
    }

    with pytest.raises(
        MemoryIOAuditError,
        match="configured secret.*error.message",
    ):
        MemoryIORecorder(
            tmp_path / "memory_io.jsonl",
            run_id="run_001",
        ).record(operation)


@pytest.mark.parametrize(
    "environment_name",
    [
        "MEMOS_NEO4J_PASSWORD",
        "MEMOS_CLIENT_SECRET",
        "MEMOS_SERVICE_CREDENTIALS",
        "NEO4J_AUTH",
        "AWS_SECRET_ACCESS_KEY",
        "SERVICE_CONNECTION_STRING",
    ],
)
def test_recorder_rejects_values_from_recognized_secret_environment_names(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
) -> None:
    monkeypatch.setenv(environment_name, "configured-secret-value")
    operation = _operation("run_001")
    operation["error"] = {
        "type": "RemoteError",
        "message": "request rejected for configured-secret-value",
    }

    with pytest.raises(MemoryIOAuditError, match="configured secret"):
        MemoryIORecorder(
            tmp_path / "memory_io.jsonl",
            run_id="run_001",
        ).record(operation)


@pytest.mark.parametrize(
    ("environment_name", "environment_value", "leaked_fragment"),
    [
        ("NEO4J_AUTH", "neo4j/review-secret-value", "review-secret-value"),
        ("AUTHORIZATION", "Bearer bearer-secret-value", "bearer-secret-value"),
        (
            "SERVICE_DATABASE_URL",
            "postgresql://user:url-secret-value@db.example/app",
            "url-secret-value",
        ),
        (
            "SERVICE_CONNECTION_STRING",
            "Server=db.example;User=user;Password=field-secret-value;",
            "field-secret-value",
        ),
    ],
)
def test_recorder_rejects_secret_fragment_from_composite_environment_value(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    environment_value: str,
    leaked_fragment: str,
) -> None:
    monkeypatch.setenv(environment_name, environment_value)
    operation = _operation("run_001")
    operation["error"] = {
        "type": "RemoteError",
        "message": f"request rejected for {leaked_fragment}",
    }

    with pytest.raises(MemoryIOAuditError, match="configured secret"):
        MemoryIORecorder(
            tmp_path / "memory_io.jsonl",
            run_id="run_001",
        ).record(operation)
