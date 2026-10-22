"""Append-only, secret-free audit stream for memory backend operations."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


_SENSITIVE_KEYS = {
    "access_key",
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "authorization_header",
    "client_secret",
    "connection_string",
    "credential",
    "credentials",
    "dsn",
    "password",
    "private_key",
    "refresh_token",
    "secret",
}

_SECRET_ENV_NAME_PARTS = frozenset(
    {
        "AUTH",
        "AUTHORIZATION",
        "CREDENTIAL",
        "CREDENTIALS",
        "PASSWORD",
        "SECRET",
        "TOKEN",
    }
)

_SECRET_ENV_NAME_SUFFIXES = (
    "_ACCESS_KEY",
    "_API_KEY",
    "_APIKEY",
    "_CONNECTION_STRING",
    "_DATABASE_URL",
    "_DB_URL",
    "_DSN",
    "_PRIVATE_KEY",
)

_VECTOR_PAYLOAD_KEYS = {
    "embedding",
    "embeddings",
    "vector",
    "vectors",
}

_RECORDER_OWNED_FIELDS = {"sequence", "timestamp"}


class MemoryIOAuditError(RuntimeError):
    """The memory audit stream is invalid or unsafe to write."""


def _is_secret_environment_name(name: str) -> bool:
    normalized = name.upper().replace("-", "_")
    parts = frozenset(part for part in normalized.split("_") if part)
    return bool(parts.intersection(_SECRET_ENV_NAME_PARTS)) or normalized.endswith(
        _SECRET_ENV_NAME_SUFFIXES
    )


def _secret_environment_values(name: str, value: str) -> set[str]:
    """Return whole and embedded credential values for one sensitive variable."""
    if not value or not _is_secret_environment_name(name):
        return set()
    secrets = {value} if len(value) >= 8 else set()
    normalized = name.upper().replace("-", "_")
    if normalized.endswith("_AUTH") or normalized == "AUTHORIZATION":
        for separator in ("/", ":", " "):
            if separator in value:
                candidate = value.rsplit(separator, 1)[1].strip()
                if len(candidate) >= 4:
                    secrets.add(candidate)
    if normalized.endswith(
        ("_CONNECTION_STRING", "_DATABASE_URL", "_DB_URL", "_DSN")
    ):
        password = urlsplit(value).password
        if password:
            candidate = unquote(password)
            if len(candidate) >= 4:
                secrets.add(candidate)
        for match in re.finditer(
            r"(?:^|[;?&])\s*(?:password|pwd|token|secret|accountkey|"
            r"accesskey|apikey)\s*=\s*([^;?&]+)",
            value,
            flags=re.IGNORECASE,
        ):
            candidate = unquote(match.group(1)).strip().strip("'\"")
            if len(candidate) >= 4:
                secrets.add(candidate)
    return secrets


class MemoryIORecorder:
    def __init__(self, path: Path | str, *, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError(f"Memory audit run_id must not be empty: value={run_id!r}")
        self.path = Path(path)
        self.run_id = run_id
        self._secret_values = tuple(
            sorted(
                {
                    secret
                    for name, value in os.environ.items()
                    for secret in _secret_environment_values(name, value)
                },
                key=len,
                reverse=True,
            )
        )
        self._sequence = self._read_last_sequence()

    def read_records(self) -> list[dict[str, Any]]:
        """Read and validate all existing records for this run."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        last_sequence = 0
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MemoryIOAuditError(
                        "Memory audit contains invalid JSON: "
                        f"path={str(self.path)!r} line={line_number} "
                        f"column={exc.colno!r}"
                    ) from exc
                if not isinstance(record, dict):
                    raise MemoryIOAuditError(
                        "Memory audit record must be an object: "
                        f"path={str(self.path)!r} line={line_number} "
                        f"value_type={type(record).__name__}"
                    )
                if record.get("run_id") != self.run_id:
                    raise MemoryIOAuditError(
                        "Memory audit contains a record for a different run: "
                        f"path={str(self.path)!r} line={line_number} "
                        f"expected_run_id={self.run_id!r} "
                        f"value={record.get('run_id')!r}"
                    )
                sequence = record.get("sequence")
                if not isinstance(sequence, int) or sequence <= last_sequence:
                    raise MemoryIOAuditError(
                        "Memory audit sequence must increase strictly: "
                        f"path={str(self.path)!r} line={line_number} "
                        f"previous={last_sequence!r} value={sequence!r}"
                    )
                records.append(record)
                last_sequence = sequence
        return records

    def record(self, operation: dict[str, Any]) -> None:
        for field in sorted(_RECORDER_OWNED_FIELDS.intersection(operation)):
            raise MemoryIOAuditError(
                "Memory audit operation must not define a recorder-owned field: "
                f"field={field!r} value={operation[field]!r}"
            )
        operation_run_id = operation.get("run_id")
        if operation_run_id != self.run_id:
            raise MemoryIOAuditError(
                "Memory audit operation belongs to a different run: "
                f"expected_run_id={self.run_id!r} value={operation_run_id!r}"
            )
        normalized = self._strip_vector_payloads(self._normalize(operation))
        self._reject_sensitive_fields(normalized)
        self._reject_secret_values(normalized)
        next_sequence = self._sequence + 1
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": next_sequence,
            **normalized,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._sequence = next_sequence

    def _read_last_sequence(self) -> int:
        records = self.read_records()
        return records[-1]["sequence"] if records else 0

    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return cls._normalize(value.value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise MemoryIOAuditError(
                        "Memory audit dictionary keys must be strings: "
                        f"value={key!r}"
                    )
                normalized[key] = cls._normalize(item)
            return normalized
        if isinstance(value, (list, tuple)):
            return [cls._normalize(item) for item in value]
        return str(value)

    @classmethod
    def _strip_vector_payloads(cls, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                normalized_key = key.casefold().replace("-", "_")
                if (
                    normalized_key in _VECTOR_PAYLOAD_KEYS
                    or normalized_key.endswith("_embedding")
                    or normalized_key.endswith("_embeddings")
                    or normalized_key.endswith("_vector")
                    or normalized_key.endswith("_vectors")
                ):
                    continue
                sanitized[key] = cls._strip_vector_payloads(item)
            return sanitized
        if isinstance(value, list):
            return [cls._strip_vector_payloads(item) for item in value]
        return value

    @classmethod
    def _reject_sensitive_fields(cls, value: Any, *, path: str = "$") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = key.casefold().replace("-", "_")
                if (
                    normalized_key in _SENSITIVE_KEYS
                    or normalized_key.endswith("_api_key")
                    or normalized_key.endswith("_access_key")
                    or normalized_key.endswith("_auth")
                    or normalized_key.endswith("_authorization")
                    or normalized_key.endswith("_connection_string")
                    or normalized_key.endswith("_dsn")
                    or normalized_key.endswith("_password")
                    or normalized_key.endswith("_private_key")
                    or normalized_key.endswith("_secret")
                    or normalized_key.endswith("_credential")
                    or normalized_key.endswith("_credentials")
                ):
                    raise MemoryIOAuditError(
                        "Refusing to write a sensitive memory audit field: "
                        f"path={path}.{key}"
                    )
                cls._reject_sensitive_fields(item, path=f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                cls._reject_sensitive_fields(item, path=f"{path}[{index}]")

    def _reject_secret_values(self, value: Any, *, path: str = "$") -> None:
        if isinstance(value, str):
            if any(secret in value for secret in self._secret_values):
                raise MemoryIOAuditError(
                    "Refusing to write a memory audit value containing a "
                    f"configured secret: path={path}"
                )
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self._reject_secret_values(item, path=f"{path}.{key}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                self._reject_secret_values(item, path=f"{path}[{index}]")
