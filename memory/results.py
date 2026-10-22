"""Output types: enums, traces, receipts, retrieve results, health."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from memory.scope import AdapterMode, MemoryScope


# ── Enums ──────────────────────────────────────────────────────────

class HealthStatus(str, Enum):
    ok = "ok"
    degraded = "degraded"
    unavailable = "unavailable"


class ConsistencyModel(str, Enum):
    immediate = "immediate"
    eventual = "eventual"
    async_derived = "async_derived"


class WriteStatus(str, Enum):
    committed = "committed"
    accepted_no_change = "accepted_no_change"
    queued = "queued"
    deferred_to_file_consolidation = "deferred_to_file_consolidation"
    skipped_read_only = "skipped_read_only"
    skipped_no_memory = "skipped_no_memory"


# ── Trace ──────────────────────────────────────────────────────────

@dataclass
class Trace:
    """Structured observability record for a single adapter operation.

    Populated by the adapter after each record_event / retrieve call.
    Consumed by the benchmark harness for PET / IQI analysis — never
    injected into the Nanobot prompt unless debug mode is on.
    """

    backend_name: str
    scope: MemoryScope
    native_operation: str
    native_ids: list[str]
    latency_ms: float
    consistency_model: ConsistencyModel
    native_events: list[str] = field(default_factory=list)
    token_count: int | None = None
    retrieved_count: int | None = None
    top_score: float | None = None
    oldest_retrieved_at: datetime | None = None
    newest_retrieved_at: datetime | None = None
    representation_version: str | None = None
    last_derived_at: datetime | None = None
    native_request: dict[str, Any] | None = None
    native_response: Any | None = None
    warnings: list[str] = field(default_factory=list)


# ── Receipts ───────────────────────────────────────────────────────

@dataclass
class WriteReceipt:
    status: WriteStatus
    native_ids: list[str] = field(default_factory=list)
    native_events: list[str] = field(default_factory=list)
    trace: Trace | None = None
    #: Native write calls this one turn's pool was split into. Every backend
    #: writes a turn in one call except mem0's `infer=True` path, which must
    #: split a pool that exceeds the embedding input cap; `native_ids` then
    #: aggregates every chunk's ids, in chunk order.
    chunk_count: int = 1


@dataclass
class ModeReceipt:
    mode: AdapterMode
    previous_mode: AdapterMode | None = None


@dataclass
class ResetReceipt:
    scope: MemoryScope
    cleared: bool = True
    warnings: list[str] = field(default_factory=list)


# ── Retrieve result ────────────────────────────────────────────────

@dataclass
class RetrieveResult:
    """Raw backend results plus trace.

    ``raw`` keeps native provenance (ids, scores, timestamps) for
    PET/IQI analysis.  ``format_context`` on the adapter turns this
    into a prompt-ready string.
    """

    raw: list[dict[str, Any]]
    trace: Trace | None = None


# ── Health / readiness ─────────────────────────────────────────────

@dataclass
class BackendHealth:
    status: HealthStatus
    backend_name: str
    latency_ms: float
    consistency_model: ConsistencyModel
    native_memory_types: list[str] | None = None
    native_ingest_modes: list[str] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class BackendReadiness:
    ready: bool
    pending_count: int = 0
    warnings: list[str] = field(default_factory=list)
