"""NanoBot file-memory condition exposed through the MemoryAdapter contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memory.adapter import PromptHistoryEvictionPolicy
from memory.events import MemoryEvent
from memory.file.backend import FileMemoryBackend
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


class NanobotFileMemoryAdapter:
    """Baseline adapter exposing NanoBot Markdown memory.

    Wraps ``FileMemoryBackend`` without changing its consolidation
    behaviour.  ``record_event`` delegates to the backend's ``add``
    (which is currently a no-op — consolidation lives in the PA loop).
    """

    backend_name: str = "nanobot_file_memory"
    prompt_history_eviction_policy = (
        PromptHistoryEvictionPolicy.after_file_consolidation
    )

    def __init__(self, memory_dir: Path | str) -> None:
        self._file_state = FileMemoryBackend(Path(memory_dir))
        self._mode = AdapterMode()

    async def setup_scope(self, scope: MemoryScope) -> None:
        pass

    async def set_mode(self, mode: AdapterMode) -> ModeReceipt:
        prev = self._mode
        self._mode = mode
        return ModeReceipt(mode=mode, previous_mode=prev)

    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        scope = MemoryScope(run_id=event.run_id, persona_id=event.persona_id)
        if self._mode.read_only:
            return WriteReceipt(
                status=WriteStatus.skipped_read_only,
                trace=self._trace(scope, "record_event"),
            )
        await self._file_state.add(event.messages)
        return WriteReceipt(
            status=WriteStatus.deferred_to_file_consolidation,
            trace=self._trace(scope, "record_event"),
        )

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        results = await self._file_state.search(
            query=request.query,
            user_id=request.scope.persona_id,
            limit=request.budget.max_items or 10,
        )
        return RetrieveResult(
            raw=results,
            trace=self._trace(
                request.scope,
                "retrieve",
                retrieved_count=len(results),
            ),
        )

    def format_context(self, result: RetrieveResult) -> str:
        # MEMORY.md already reaches the prompt whole under "# Memory"
        # (`ContextBuilder.build_system_prompt`, pa/agent/context.py). Formatting
        # it here as well put it in a second time, under "# Retrieved Memory"
        # and cut to the retrieval token budget. `retrieve` still runs, so the
        # memory_io audit keeps what MEMORY.md held on each turn.
        return ""

    async def reset_scope(self, scope: MemoryScope) -> ResetReceipt:
        if self._mode.read_only:
            raise PermissionError("Read-only file memory cannot be reset")
        self._file_state.reset()
        return ResetReceipt(scope=scope)

    async def health(self) -> BackendHealth:
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=0.0,
            consistency_model=ConsistencyModel.immediate,
        )

    def _trace(
        self,
        scope: MemoryScope,
        operation: str,
        *,
        retrieved_count: int | None = None,
    ) -> Trace:
        return Trace(
            backend_name=self.backend_name,
            scope=scope,
            native_operation=operation,
            native_ids=[],
            latency_ms=0.0,
            consistency_model=ConsistencyModel.immediate,
            retrieved_count=retrieved_count,
        )
