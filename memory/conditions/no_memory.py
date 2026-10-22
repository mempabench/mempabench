"""No-memory controlled condition — always returns empty context."""

from __future__ import annotations

from memory.adapter import PromptHistoryEvictionPolicy
from memory.events import MemoryEvent
from memory.results import (
    BackendHealth,
    ConsistencyModel,
    HealthStatus,
    ModeReceipt,
    ResetReceipt,
    RetrieveResult,
    Trace,
    WriteReceipt,
    WriteStatus,
)
from memory.scope import AdapterMode, MemoryScope, RetrieveRequest


class NoMemoryAdapter:
    """Controlled condition: no memory storage or retrieval.

    Satisfies MemoryAdapter so the harness can select it by config
    without special-casing.  All writes are no-ops and all retrieves
    return empty context.
    """

    backend_name: str = "no_memory"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.without_persistence

    def __init__(self) -> None:
        self._mode = AdapterMode()

    async def setup_scope(self, scope: MemoryScope) -> None:
        pass

    async def set_mode(self, mode: AdapterMode) -> ModeReceipt:
        prev = self._mode
        self._mode = mode
        return ModeReceipt(mode=mode, previous_mode=prev)

    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        return WriteReceipt(
            status=WriteStatus.skipped_no_memory,
            trace=self._trace(event, "record_event"),
        )

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        return RetrieveResult(
            raw=[],
            trace=Trace(
                backend_name=self.backend_name,
                scope=request.scope,
                native_operation="retrieve",
                native_ids=[],
                latency_ms=0.0,
                consistency_model=ConsistencyModel.immediate,
                retrieved_count=0,
            ),
        )

    def format_context(self, result: RetrieveResult) -> str:
        return ""

    async def reset_scope(self, scope: MemoryScope) -> ResetReceipt:
        return ResetReceipt(scope=scope)

    async def health(self) -> BackendHealth:
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=0.0,
            consistency_model=ConsistencyModel.immediate,
        )

    def _trace(self, event: MemoryEvent, operation: str) -> Trace:
        return Trace(
            backend_name=self.backend_name,
            scope=MemoryScope(run_id=event.run_id, persona_id=event.persona_id),
            native_operation=operation,
            native_ids=[],
            latency_ms=0.0,
            consistency_model=ConsistencyModel.immediate,
        )
