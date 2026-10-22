"""Tests for the nanobot_file_memory baseline adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from memory.adapter import MemoryAdapter
from memory.conditions.nanobot_file_memory import NanobotFileMemoryAdapter
from memory.events import MemoryEvent
from memory.file.backend import FileMemoryBackend
from memory.registry import default_registry
from memory.results import HealthStatus, ConsistencyModel, WriteStatus
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


def _request(scope: MemoryScope | None = None) -> RetrieveRequest:
    return RetrieveRequest(
        query="preferences",
        scope=scope or _scope(),
        session_id="s01",
        budget=MemoryRetrievalBudget(max_tokens=4000),
    )


class TestRegistryIntegration:
    def test_default_registry_includes_file_memory(self) -> None:
        reg = default_registry()
        assert "nanobot_file_memory" in reg

    def test_adapter_kind_is_condition(self) -> None:
        reg = default_registry()
        assert reg.adapter_kind("nanobot_file_memory") == "condition"

    def test_create_requires_memory_dir(self, tmp_path: Path) -> None:
        reg = default_registry()
        adapter = reg.create("nanobot_file_memory", memory_dir=tmp_path / "mem")
        assert isinstance(adapter, MemoryAdapter)
        assert adapter.backend_name == "nanobot_file_memory"

    def test_create_without_memory_dir_raises(self) -> None:
        reg = default_registry()
        with pytest.raises(TypeError):
            reg.create("nanobot_file_memory")


class TestNanobotFileMemoryAdapter:
    @pytest.fixture()
    def adapter(self, tmp_path: Path) -> NanobotFileMemoryAdapter:
        return NanobotFileMemoryAdapter(memory_dir=tmp_path / "mem")

    @pytest.mark.asyncio
    async def test_is_memory_adapter(
        self,
        adapter: NanobotFileMemoryAdapter,
    ) -> None:
        assert isinstance(adapter, MemoryAdapter)

    @pytest.mark.asyncio
    async def test_preexisting_memory_is_retrieved_but_not_formatted(
        self, tmp_path: Path
    ) -> None:
        """The audit still sees MEMORY.md; the prompt gets it from `# Memory` only."""
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("User prefers formal tone.", encoding="utf-8")

        adapter = NanobotFileMemoryAdapter(memory_dir=mem_dir)
        result = await adapter.retrieve(_request())
        assert result.raw == [
            {"content": "## Long-term Memory\nUser prefers formal tone."}
        ]
        assert adapter.format_context(result) == ""

    @pytest.mark.asyncio
    async def test_empty_memory_returns_empty_context(
        self,
        adapter: NanobotFileMemoryAdapter,
    ) -> None:
        result = await adapter.retrieve(_request())
        assert adapter.format_context(result) == ""

    @pytest.mark.asyncio
    async def test_record_event_is_deferred_to_file_consolidation(
        self,
        adapter: NanobotFileMemoryAdapter,
    ) -> None:
        await adapter.setup_scope(_scope())
        receipt = await adapter.record_event(_event())
        assert receipt.status == WriteStatus.deferred_to_file_consolidation
        # FileMemoryBackend.add is a no-op — no content created
        result = await adapter.retrieve(_request())
        assert adapter.format_context(result) == ""

    @pytest.mark.asyncio
    async def test_read_only_skips_write(
        self,
        adapter: NanobotFileMemoryAdapter,
    ) -> None:
        await adapter.set_mode(AdapterMode(read_only=True, reason="test_session"))
        receipt = await adapter.record_event(_event())
        assert receipt.status == WriteStatus.skipped_read_only
        assert receipt.trace is not None

    @pytest.mark.asyncio
    async def test_set_mode_tracks_state(
        self,
        adapter: NanobotFileMemoryAdapter,
    ) -> None:
        r1 = await adapter.set_mode(AdapterMode(read_only=True, reason="test"))
        assert r1.mode.read_only is True
        assert r1.previous_mode is not None
        assert r1.previous_mode.read_only is False

        r2 = await adapter.set_mode(AdapterMode(read_only=False, reason="accum"))
        assert r2.previous_mode is not None
        assert r2.previous_mode.read_only is True

    @pytest.mark.asyncio
    async def test_reset_scope_clears_files(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("Some memory", encoding="utf-8")
        (mem_dir / "HISTORY.md").write_text("Some history", encoding="utf-8")

        adapter = NanobotFileMemoryAdapter(memory_dir=mem_dir)
        reset = await adapter.reset_scope(_scope())
        assert reset.cleared is True

        assert (mem_dir / "MEMORY.md").read_text(encoding="utf-8") == ""
        assert (mem_dir / "HISTORY.md").read_text(encoding="utf-8") == ""

    @pytest.mark.asyncio
    async def test_health(self, adapter: NanobotFileMemoryAdapter) -> None:
        h = await adapter.health()
        assert h.status == HealthStatus.ok
        assert h.backend_name == "nanobot_file_memory"
        assert h.consistency_model == ConsistencyModel.immediate

    @pytest.mark.asyncio
    async def test_retrieve_trace(
        self,
        adapter: NanobotFileMemoryAdapter,
    ) -> None:
        result = await adapter.retrieve(_request())
        assert result.trace is not None
        assert result.trace.backend_name == "nanobot_file_memory"
        assert result.trace.native_operation == "retrieve"


class TestFileMemoryBackendStillWorks:
    """Ensure the original FileMemoryBackend is untouched."""

    def test_direct_usage(self, tmp_path: Path) -> None:
        backend = FileMemoryBackend(tmp_path / "direct")
        (tmp_path / "direct").mkdir(parents=True, exist_ok=True)
        mem_file = tmp_path / "direct" / "MEMORY.md"
        mem_file.write_text("Direct test", encoding="utf-8")
        ctx = backend.get_context_string("user_a")
        assert "Direct test" in ctx
