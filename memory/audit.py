"""Boundary for append-only memory operation audit records."""

from __future__ import annotations

from typing import Any, Protocol


class MemoryIOAuditSink(Protocol):
    def record(self, operation: dict[str, Any]) -> None: ...
