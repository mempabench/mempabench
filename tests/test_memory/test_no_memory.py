"""Tests for the no_memory controlled condition provider."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory.adapter import MemoryAdapter
from memory.conditions.no_memory import NoMemoryAdapter
from memory.events import MemoryEvent
from memory.results import (
    HealthStatus,
    ConsistencyModel,
    WriteStatus,
)
from memory.scope import (
    AdapterMode,
    MemoryRetrievalBudget,
    MemoryScope,
    RetrieveRequest,
)


def _scope() -> MemoryScope:
    return MemoryScope(run_id="run_001", persona_id="user_a")


def _event() -> MemoryEvent:
    return MemoryEvent(
        event_id="evt_1",
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
        session_id="s01",
        turn_id="t01",
        timestamp=datetime.now(timezone.utc),
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    )


def _request() -> RetrieveRequest:
    return RetrieveRequest(
        query="preferences",
        scope=_scope(),
        session_id="s01",
        budget=MemoryRetrievalBudget(max_tokens=4000),
    )


class TestNoMemoryAdapter:
    @pytest.fixture()
    def adapter(self) -> NoMemoryAdapter:
        return NoMemoryAdapter()

    def test_is_memory_adapter(self, adapter: NoMemoryAdapter) -> None:
        assert isinstance(adapter, MemoryAdapter)

    def test_backend_name(self, adapter: NoMemoryAdapter) -> None:
        assert adapter.backend_name == "no_memory"

    @pytest.mark.asyncio
    async def test_record_event_returns_skipped(self, adapter: NoMemoryAdapter) -> None:
        receipt = await adapter.record_event(_event())
        assert receipt.status == WriteStatus.skipped_no_memory
        assert receipt.trace is not None
        assert receipt.trace.backend_name == "no_memory"

    @pytest.mark.asyncio
    async def test_record_does_not_affect_retrieve(self, adapter: NoMemoryAdapter) -> None:
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        result = await adapter.retrieve(_request())
        assert result.raw == []

    @pytest.mark.asyncio
    async def test_retrieve_always_empty(self, adapter: NoMemoryAdapter) -> None:
        result = await adapter.retrieve(_request())
        assert result.raw == []
        assert result.trace is not None
        assert result.trace.retrieved_count == 0

    @pytest.mark.asyncio
    async def test_format_context_empty(self, adapter: NoMemoryAdapter) -> None:
        result = await adapter.retrieve(_request())
        assert adapter.format_context(result) == ""

    @pytest.mark.asyncio
    async def test_set_mode_read_only(self, adapter: NoMemoryAdapter) -> None:
        receipt = await adapter.set_mode(AdapterMode(read_only=True, reason="test"))
        assert receipt.mode.read_only is True
        assert receipt.previous_mode is not None
        assert receipt.previous_mode.read_only is False

    @pytest.mark.asyncio
    async def test_set_mode_back_to_readwrite(self, adapter: NoMemoryAdapter) -> None:
        await adapter.set_mode(AdapterMode(read_only=True, reason="test"))
        receipt = await adapter.set_mode(AdapterMode(read_only=False, reason="accumulation"))
        assert receipt.mode.read_only is False
        assert receipt.previous_mode is not None
        assert receipt.previous_mode.read_only is True

    @pytest.mark.asyncio
    async def test_health(self, adapter: NoMemoryAdapter) -> None:
        h = await adapter.health()
        assert h.status == HealthStatus.ok
        assert h.backend_name == "no_memory"
        assert h.consistency_model == ConsistencyModel.immediate

    @pytest.mark.asyncio
    async def test_reset_scope(self, adapter: NoMemoryAdapter) -> None:
        result = await adapter.reset_scope(_scope())
        assert result.cleared is True
        assert result.scope == _scope()

    @pytest.mark.asyncio
    async def test_setup_scope_succeeds(self, adapter: NoMemoryAdapter) -> None:
        await adapter.setup_scope(_scope())  # should not raise
