"""Async-first adapter contract and capability protocols."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from memory.events import MemoryEvent
from memory.scope import AdapterMode, MemoryRetrievalBudget, MemoryScope, RetrieveRequest
from memory.results import (
    BackendHealth,
    BackendReadiness,
    ModeReceipt,
    ResetReceipt,
    RetrieveResult,
    Trace,
    WriteReceipt,
)


# ── Core contract ──────────────────────────────────────────────────

class PromptHistoryEvictionPolicy(str, Enum):
    after_file_consolidation = "after_file_consolidation"
    after_adapter_commit = "after_adapter_commit"
    without_persistence = "without_persistence"


@runtime_checkable
class MemoryAdapter(Protocol):
    """Minimum contract every memory backend must satisfy.

    Nanobot core depends only on this protocol — never on backend SDK
    objects.  Adapters map IDs, build requests, format context, and
    report traces; they do not rewrite backend extraction/retrieval.
    """

    backend_name: str
    prompt_history_eviction_policy: PromptHistoryEvictionPolicy

    async def setup_scope(self, scope: MemoryScope) -> None: ...
    async def set_mode(self, mode: AdapterMode) -> ModeReceipt: ...
    async def record_event(self, event: MemoryEvent) -> WriteReceipt: ...
    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult: ...
    def format_context(self, result: RetrieveResult) -> str: ...
    async def reset_scope(self, scope: MemoryScope) -> ResetReceipt: ...
    async def health(self) -> BackendHealth: ...


# ── Optional capability protocols ─────────────────────────────────
# These serve the benchmark harness and backend-native ablation.
# They are not required by the Nanobot main loop.

@runtime_checkable
class SupportsFeedback(Protocol):
    async def apply_feedback(self, feedback: Any) -> Any: ...


@runtime_checkable
class SupportsBulkIngest(Protocol):
    async def record_events(self, events: list[MemoryEvent]) -> list[WriteReceipt]: ...


@runtime_checkable
class SupportsReadiness(Protocol):
    async def readiness(self, scope: MemoryScope) -> BackendReadiness: ...
    async def wait_until_ready(self, scope: MemoryScope, timeout_s: float) -> BackendReadiness: ...


@runtime_checkable
class SupportsNativeMutation(Protocol):
    async def native_mutation(self, request: Any) -> Any: ...


@runtime_checkable
class SupportsProvenance(Protocol):
    async def provenance(self, request: Any) -> Any: ...


@runtime_checkable
class SupportsNativeQuery(Protocol):
    async def native_query(self, request: Any) -> Any: ...
