"""Contract-level tests for the M1 adapter scaffold.

Tests cover:
- All public types can be constructed with minimal valid values.
- A tiny fake adapter satisfies the MemoryAdapter protocol.
- Read-only mode returns skipped_read_only.
- Empty retrieval produces empty format_context.
- Trace fields are recorded without inventing scores/timestamps.
- Capability protocols can be checked structurally.
- Every real adapter presents every pool item it receives.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from memory.adapter import (
    MemoryAdapter,
    PromptHistoryEvictionPolicy,
    SupportsBulkIngest,
    SupportsFeedback,
    SupportsNativeMutation,
    SupportsNativeQuery,
    SupportsProvenance,
    SupportsReadiness,
)
from memory.events import MemoryEvent
from memory.results import (
    BackendHealth,
    BackendReadiness,
    ConsistencyModel,
    HealthStatus,
    ModeReceipt,
    ResetReceipt,
    RetrieveResult,
    Trace,
    WriteReceipt,
    WriteStatus,
)
from memory.scope import (
    AdapterMode,
    MemoryRetrievalBudget,
    MemoryScope,
    RetrieveRequest,
)


# ── Helpers ────────────────────────────────────────────────────────

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
            {"role": "assistant", "content": "Hi there"},
        ],
    )


def _budget() -> MemoryRetrievalBudget:
    return MemoryRetrievalBudget(max_tokens=4000, max_items=10)


# ── Fake adapter ───────────────────────────────────────────────────

class FakeAdapter:
    """Minimal adapter that satisfies the MemoryAdapter protocol."""

    backend_name: str = "fake"
    prompt_history_eviction_policy = (
        PromptHistoryEvictionPolicy.after_adapter_commit
    )

    def __init__(self) -> None:
        self._mode = AdapterMode()
        self._store: list[MemoryEvent] = []

    async def setup_scope(self, scope: MemoryScope) -> None:
        pass

    async def set_mode(self, mode: AdapterMode) -> ModeReceipt:
        prev = self._mode
        self._mode = mode
        return ModeReceipt(mode=mode, previous_mode=prev)

    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        if self._mode.read_only:
            return WriteReceipt(
                status=WriteStatus.skipped_read_only,
                trace=Trace(
                    backend_name=self.backend_name,
                    scope=MemoryScope(
                        run_id=event.run_id,
                        persona_id=event.persona_id,
                    ),
                    native_operation="record_event",
                    native_ids=[],
                    latency_ms=0.0,
                    consistency_model=ConsistencyModel.immediate,
                ),
            )
        self._store.append(event)
        return WriteReceipt(
            status=WriteStatus.committed,
            native_ids=[event.event_id],
            trace=Trace(
                backend_name=self.backend_name,
                scope=MemoryScope(
                    run_id=event.run_id,
                    persona_id=event.persona_id,
                ),
                native_operation="record_event",
                native_ids=[event.event_id],
                latency_ms=0.1,
                consistency_model=ConsistencyModel.immediate,
            ),
        )

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        raw = [
            {"content": m["content"]}
            for ev in self._store
            if ev.run_id == request.scope.run_id
            for m in ev.messages
        ]
        return RetrieveResult(
            raw=raw,
            trace=Trace(
                backend_name=self.backend_name,
                scope=request.scope,
                native_operation="retrieve",
                native_ids=[],
                latency_ms=0.1,
                consistency_model=ConsistencyModel.immediate,
                retrieved_count=len(raw),
                # no scores/timestamps — don't invent them
            ),
        )

    def format_context(self, result: RetrieveResult) -> str:
        if not result.raw:
            return ""
        lines = [item.get("content", "") for item in result.raw]
        return f'<memory-context backend="fake">\n' + "\n".join(lines) + "\n</memory-context>"

    async def reset_scope(self, scope: MemoryScope) -> ResetReceipt:
        self._store = [e for e in self._store if e.run_id != scope.run_id]
        return ResetReceipt(scope=scope)

    async def health(self) -> BackendHealth:
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=0.0,
            consistency_model=ConsistencyModel.immediate,
        )


# ── Type construction tests ────────────────────────────────────────

class TestTypeConstruction:
    def test_memory_scope(self) -> None:
        s = MemoryScope(run_id="r", persona_id="p")
        assert s.run_id == "r"
        assert s.backend_namespace is None
        assert s.agent_id is None

    def test_adapter_mode_defaults(self) -> None:
        m = AdapterMode()
        assert m.read_only is False
        assert m.reason == ""

    def test_memory_retrieval_budget(self) -> None:
        b = MemoryRetrievalBudget(max_tokens=8000)
        assert b.max_tokens == 8000
        assert b.max_items is None

    def test_memory_event_two_messages(self) -> None:
        ev = _event()
        assert len(ev.messages) == 2
        assert not hasattr(ev, "scenario_description")
        assert not hasattr(ev, "interaction_context")
        assert not hasattr(ev, "active_preference_attributes")
        assert not hasattr(ev, "metadata")

    def test_retrieve_request_carries_budget(self) -> None:
        rr = RetrieveRequest(
            query="preference",
            scope=_scope(),
            session_id="s01",
            budget=_budget(),
        )
        assert rr.budget.max_tokens == 4000
        assert rr.budget.max_items == 10

    def test_trace_optional_fields_default_none(self) -> None:
        t = Trace(
            backend_name="x",
            scope=_scope(),
            native_operation="op",
            native_ids=[],
            latency_ms=1.0,
            consistency_model=ConsistencyModel.immediate,
        )
        assert t.token_count is None
        assert t.top_score is None
        assert t.oldest_retrieved_at is None
        assert t.newest_retrieved_at is None
        assert t.representation_version is None
        assert t.last_derived_at is None
        assert t.warnings == []

    def test_write_receipt(self) -> None:
        wr = WriteReceipt(status=WriteStatus.committed, native_ids=["id1"])
        assert wr.status == WriteStatus.committed

    def test_mode_receipt(self) -> None:
        mr = ModeReceipt(mode=AdapterMode(read_only=True, reason="test"))
        assert mr.mode.read_only is True
        assert mr.previous_mode is None

    def test_reset_receipt(self) -> None:
        rr = ResetReceipt(scope=_scope())
        assert rr.cleared is True

    def test_retrieve_result(self) -> None:
        rr = RetrieveResult(raw=[{"content": "x"}])
        assert len(rr.raw) == 1
        assert rr.trace is None

    def test_backend_health(self) -> None:
        bh = BackendHealth(
            status=HealthStatus.ok,
            backend_name="test",
            latency_ms=5.0,
            consistency_model=ConsistencyModel.eventual,
        )
        assert bh.native_memory_types is None
        assert bh.native_ingest_modes is None

    def test_backend_readiness(self) -> None:
        br = BackendReadiness(ready=True)
        assert br.pending_count == 0

    def test_enums(self) -> None:
        assert HealthStatus.ok.value == "ok"
        assert ConsistencyModel.async_derived.value == "async_derived"
        assert WriteStatus.skipped_read_only.value == "skipped_read_only"


# ── Protocol structural checks ─────────────────────────────────────

class TestProtocolConformance:
    def test_fake_adapter_is_memory_adapter(self) -> None:
        assert isinstance(FakeAdapter(), MemoryAdapter)

    def test_non_adapter_fails_check(self) -> None:
        assert not isinstance(object(), MemoryAdapter)


# ── Fake adapter behaviour tests ───────────────────────────────────

class TestFakeAdapterBehaviour:
    @pytest.fixture()
    def adapter(self) -> FakeAdapter:
        return FakeAdapter()

    @pytest.mark.asyncio
    async def test_record_and_retrieve(self, adapter: FakeAdapter) -> None:
        await adapter.setup_scope(_scope())
        ev = _event()
        receipt = await adapter.record_event(ev)
        assert receipt.status == WriteStatus.committed
        assert ev.event_id in receipt.native_ids

        result = await adapter.retrieve(
            RetrieveRequest(query="hello", scope=_scope(), session_id="s01", budget=_budget())
        )
        assert result.trace is not None
        assert result.trace.retrieved_count == 2  # two messages
        ctx = adapter.format_context(result)
        assert "Hello" in ctx
        assert 'backend="fake"' in ctx

    @pytest.mark.asyncio
    async def test_read_only_skips_write(self, adapter: FakeAdapter) -> None:
        await adapter.setup_scope(_scope())
        mode_receipt = await adapter.set_mode(AdapterMode(read_only=True, reason="test_session"))
        assert mode_receipt.mode.read_only is True
        assert mode_receipt.previous_mode is not None
        assert mode_receipt.previous_mode.read_only is False

        receipt = await adapter.record_event(_event())
        assert receipt.status == WriteStatus.skipped_read_only

        # nothing stored — retrieve returns empty
        result = await adapter.retrieve(
            RetrieveRequest(query="x", scope=_scope(), session_id="s01", budget=_budget())
        )
        assert result.raw == []

    @pytest.mark.asyncio
    async def test_empty_format_context(self, adapter: FakeAdapter) -> None:
        result = RetrieveResult(raw=[])
        assert adapter.format_context(result) == ""

    @pytest.mark.asyncio
    async def test_trace_no_invented_scores(self, adapter: FakeAdapter) -> None:
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        result = await adapter.retrieve(
            RetrieveRequest(query="x", scope=_scope(), session_id="s01", budget=_budget())
        )
        assert result.trace is not None
        assert result.trace.top_score is None
        assert result.trace.oldest_retrieved_at is None
        assert result.trace.newest_retrieved_at is None

    @pytest.mark.asyncio
    async def test_reset_scope(self, adapter: FakeAdapter) -> None:
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        reset = await adapter.reset_scope(_scope())
        assert reset.cleared is True

        result = await adapter.retrieve(
            RetrieveRequest(query="x", scope=_scope(), session_id="s01", budget=_budget())
        )
        assert result.raw == []

    @pytest.mark.asyncio
    async def test_health(self, adapter: FakeAdapter) -> None:
        h = await adapter.health()
        assert h.status == HealthStatus.ok
        assert h.backend_name == "fake"
        assert h.consistency_model == ConsistencyModel.immediate


# ── Capability protocol structural checks ──────────────────────────

class _FakeWithCapabilities:
    """Fake that also satisfies optional capability protocols."""

    async def apply_feedback(self, feedback: Any) -> Any:
        return None

    async def record_events(self, events: list[MemoryEvent]) -> list[WriteReceipt]:
        return []

    async def readiness(self, scope: MemoryScope) -> BackendReadiness:
        return BackendReadiness(ready=True)

    async def wait_until_ready(self, scope: MemoryScope, timeout_s: float) -> BackendReadiness:
        return BackendReadiness(ready=True)

    async def native_mutation(self, request: Any) -> Any:
        return None

    async def provenance(self, request: Any) -> Any:
        return None

    async def native_query(self, request: Any) -> Any:
        return None


class TestCapabilityProtocols:
    def test_supports_feedback(self) -> None:
        assert isinstance(_FakeWithCapabilities(), SupportsFeedback)

    def test_supports_bulk_ingest(self) -> None:
        assert isinstance(_FakeWithCapabilities(), SupportsBulkIngest)

    def test_supports_readiness(self) -> None:
        assert isinstance(_FakeWithCapabilities(), SupportsReadiness)

    def test_supports_native_mutation(self) -> None:
        assert isinstance(_FakeWithCapabilities(), SupportsNativeMutation)

    def test_supports_provenance(self) -> None:
        assert isinstance(_FakeWithCapabilities(), SupportsProvenance)

    def test_supports_native_query(self) -> None:
        assert isinstance(_FakeWithCapabilities(), SupportsNativeQuery)

    def test_plain_object_does_not_match(self) -> None:
        o = object()
        assert not isinstance(o, SupportsFeedback)
        assert not isinstance(o, SupportsBulkIngest)
        assert not isinstance(o, SupportsReadiness)
        assert not isinstance(o, SupportsNativeMutation)
        assert not isinstance(o, SupportsProvenance)
        assert not isinstance(o, SupportsNativeQuery)


# ── Every adapter presents every item it receives ────────

from pathlib import Path  # noqa: E402

from memory.backends.mem0_adapter import Mem0Adapter  # noqa: E402
from memory.backends.memos_adapter import MemOSAdapter  # noqa: E402
from memory.conditions.nanobot_file_memory import NanobotFileMemoryAdapter  # noqa: E402
from memory.conditions.no_memory import NoMemoryAdapter  # noqa: E402
from memory.file.backend import FileMemoryBackend  # noqa: E402

from test_memory.pool_fixtures import (  # noqa: E402
    expected_contents,
    full_experience_pool,
    pool_event,
)
from test_memory.test_mem0_adapter import FakeMem0Client  # noqa: E402
from test_memory.test_memos_adapter import FakeMemOSClient  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend",
    ["mem0", "memos", "nanobot_file_memory", "no_memory"],
)
async def test_adapter_presents_every_item_it_receives(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No adapter drops, filters or reorders a pool message."""
    pool = full_experience_pool()
    event = pool_event(pool)
    before = [dict(message) for message in event.messages]
    expected = expected_contents(pool)

    if backend == "mem0":
        mem0_client = FakeMem0Client()
        adapter: Any = Mem0Adapter(client=mem0_client)

        def presented() -> list[str]:
            return [m["content"] for m in mem0_client.add_calls[-1]["messages"]]

    elif backend == "memos":
        memos_client = FakeMemOSClient()
        adapter = MemOSAdapter(
            client=memos_client, cube_id="cube_001", memory_session_id="run_001_main"
        )

        def presented() -> list[str]:
            return [m["content"] for m in memos_client.add_calls[-1]["messages"]]

    elif backend == "nanobot_file_memory":
        # Wrap `FileMemoryBackend.add` rather than replace it, so the real
        # documented no-op (memory/file/backend.py:29-35) still runs and the
        # "writes no MEMORY.md" post-condition below can actually fail.
        seen: list[list[dict[str, Any]]] = []
        real_add = FileMemoryBackend.add

        async def _recording_add(
            self: FileMemoryBackend,
            messages: list[dict[str, Any]],
            metadata: Any = None,
        ) -> None:
            seen.append(messages)
            return await real_add(self, messages, metadata)

        monkeypatch.setattr(FileMemoryBackend, "add", _recording_add)
        adapter = NanobotFileMemoryAdapter(tmp_path)

        def presented() -> list[str]:
            return [m["content"] for m in seen[-1]]

    else:
        adapter = NoMemoryAdapter()
        expected = []

        def presented() -> list[str]:
            return []

    await adapter.setup_scope(_scope())
    receipt = await adapter.record_event(event)

    assert presented() == expected
    assert event.messages == before
    assert event.items == pool.items

    # Per-backend post-conditions.
    if backend == "nanobot_file_memory":
        # Step 7 wires the journal; today `record_event` still defers.
        assert receipt.status == WriteStatus.deferred_to_file_consolidation
        assert (tmp_path / "MEMORY.md").exists() is False
    elif backend == "no_memory":
        assert receipt.status == WriteStatus.skipped_no_memory
