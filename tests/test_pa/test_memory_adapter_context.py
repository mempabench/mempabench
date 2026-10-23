"""Tests for PA prompt-context memory adapter integration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from memory.adapter import PromptHistoryEvictionPolicy
from memory.event_writer import build_turn_pool
from pa.agent.context import ContextBuilder
from pa.providers.base import LLMProvider, LLMResponse
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


# ── Fake adapter ───────────────────────────────────────────────────

class FakeAdapterForPA:
    """Fake MemoryAdapter that returns predictable context."""

    backend_name: str = "fake_pa"
    prompt_history_eviction_policy = (
        PromptHistoryEvictionPolicy.without_persistence
    )

    def __init__(self, context_text: str = "User prefers formal tone.") -> None:
        self._context_text = context_text
        self._mode = AdapterMode()
        self.setup_scope_calls: list[MemoryScope] = []
        self.retrieve_calls: list[RetrieveRequest] = []
        self.record_events: list[MemoryEvent] = []
        self.last_retrieve_result: RetrieveResult | None = None

    async def setup_scope(self, scope: MemoryScope) -> None:
        self.setup_scope_calls.append(scope)

    async def set_mode(self, mode: AdapterMode) -> ModeReceipt:
        prev = self._mode
        self._mode = mode
        return ModeReceipt(mode=mode, previous_mode=prev)

    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        self.record_events.append(event)
        return WriteReceipt(status=WriteStatus.skipped_no_memory)

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        self.retrieve_calls.append(request)
        result = RetrieveResult(
            raw=[{"memory": self._context_text}],
            trace=Trace(
                backend_name=self.backend_name,
                scope=request.scope,
                native_operation="retrieve",
                native_ids=[],
                latency_ms=0.0,
                consistency_model=ConsistencyModel.immediate,
                retrieved_count=1,
                native_request={
                    "query": request.query,
                    "filters": {
                        "run_id": request.scope.run_id,
                        "user_id": request.scope.persona_id,
                    },
                },
                native_response={
                    "results": [{"memory": self._context_text}]
                },
            ),
        )
        self.last_retrieve_result = result
        return result

    def format_context(self, result: RetrieveResult) -> str:
        if not result.raw:
            return ""
        text = result.raw[0].get("memory", "")
        if not text:
            return ""
        return f'<memory-context backend="{self.backend_name}">\n{text}\n</memory-context>'

    async def reset_scope(self, scope: MemoryScope) -> ResetReceipt:
        return ResetReceipt(scope=scope)

    async def health(self) -> BackendHealth:
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=0.0,
            consistency_model=ConsistencyModel.immediate,
        )


class FailingAdapter(FakeAdapterForPA):
    """Adapter that raises on retrieve."""

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        raise RuntimeError("Simulated retrieval failure")


class CommittedAdapter(FakeAdapterForPA):
    prompt_history_eviction_policy = (
        PromptHistoryEvictionPolicy.after_adapter_commit
    )

    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        self.record_events.append(event)
        return WriteReceipt(
            status=WriteStatus.committed,
            native_ids=["memory-1"],
            trace=Trace(
                backend_name=self.backend_name,
                scope=MemoryScope(
                    run_id=event.run_id,
                    persona_id=event.persona_id,
                ),
                native_operation="add",
                native_ids=["memory-1"],
                latency_ms=1.0,
                consistency_model=ConsistencyModel.immediate,
                native_request={
                    "messages": event.messages,
                    "user_id": event.persona_id,
                    "run_id": event.run_id,
                },
                native_response={
                    "results": [
                        {
                            "id": "memory-1",
                            "memory": "stored",
                            "event": "ADD",
                        }
                    ]
                },
            ),
        )


class QueuedAdapter(CommittedAdapter):
    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        self.record_events.append(event)
        return WriteReceipt(status=WriteStatus.queued)


class NoNewMemoryAdapter(CommittedAdapter):
    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        self.record_events.append(event)
        return WriteReceipt(status=WriteStatus.accepted_no_change)


class RecordingProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[dict[str, Any]]] = []

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(content="done")


class ErrorProvider(RecordingProvider):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(
            content="Error: provider throughput limit",
            finish_reason="error",
        )


class BudgetCountingProvider(RecordingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.estimation_models: list[str | None] = []

    def estimate_prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
    ) -> tuple[int, str]:
        self.estimation_models.append(model)
        return len(messages[0]["content"]), "test_pa_counter"


# ── ContextBuilder tests ───────────────────────────────────────────

class TestContextBuilderAdapterMemory:
    def test_adapter_memory_appears_in_system_prompt(self, tmp_path: Path) -> None:
        cb = ContextBuilder(tmp_path)
        prompt = cb.build_system_prompt(
            adapter_memory_context='<memory-context backend="fake">\nformal tone\n</memory-context>'
        )
        assert "# Retrieved Memory" in prompt
        assert "formal tone" in prompt
        assert 'backend="fake"' in prompt

    def test_no_adapter_memory_no_section(self, tmp_path: Path) -> None:
        cb = ContextBuilder(tmp_path)
        prompt = cb.build_system_prompt()
        assert "# Retrieved Memory" not in prompt

    def test_file_memory_and_adapter_memory_coexist(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("Existing file memory content", encoding="utf-8")

        cb = ContextBuilder(tmp_path)
        prompt = cb.build_system_prompt(
            adapter_memory_context='<memory-context backend="fake">\nadapter content\n</memory-context>'
        )
        assert "# Memory" in prompt
        assert "Existing file memory content" in prompt
        assert "# Retrieved Memory" in prompt
        assert "adapter content" in prompt

    def test_build_messages_passes_adapter_context(self, tmp_path: Path) -> None:
        cb = ContextBuilder(tmp_path)
        msgs = cb.build_messages(
            history=[],
            current_message="Hello",
            adapter_memory_context="adapter context here",
        )
        system_content = msgs[0]["content"]
        assert "# Retrieved Memory" in system_content
        assert "adapter context here" in system_content

    def test_build_messages_without_adapter_context(self, tmp_path: Path) -> None:
        cb = ContextBuilder(tmp_path)
        msgs = cb.build_messages(history=[], current_message="Hello")
        system_content = msgs[0]["content"]
        assert "# Retrieved Memory" not in system_content


# ── AgentLoop adapter retrieval tests ──────────────────────────────

class TestAgentLoopAdapterRetrieval:
    @pytest.mark.asyncio
    async def test_separate_acc_loops_continue_one_persisted_session(
        self,
        tmp_path: Path,
    ) -> None:
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus
        from pa.session.manager import SessionManager

        session_key = "run_001_main"
        SessionManager(tmp_path).create(session_key)

        first_provider = RecordingProvider()
        first_loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=first_provider,
            workspace=tmp_path,
            pa_model="test-model",
        )
        await first_loop.process_direct_message(
            "acc_001 preference evidence",
            session_key=session_key,
        )

        second_provider = RecordingProvider()
        second_loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=second_provider,
            workspace=tmp_path,
            pa_model="test-model",
        )
        await second_loop.process_direct_message(
            "acc_002 follow-up",
            session_key=session_key,
        )

        second_request = second_provider.calls[0]
        assert any(
            message.get("role") == "user"
            and message.get("content") == "acc_001 preference evidence"
            for message in second_request
        )
        persisted = SessionManager(tmp_path).get(session_key)
        assert [message["content"] for message in persisted.messages] == [
            "acc_001 preference evidence",
            "done",
            "acc_002 follow-up",
            "done",
        ]

    @pytest.mark.asyncio
    async def test_retrieve_adapter_memory_with_metadata(self, tmp_path: Path) -> None:
        from pa.agent.loop import AgentLoop

        adapter = FakeAdapterForPA()
        loop = AgentLoop.__new__(AgentLoop)
        loop.memory_adapter = adapter
        loop.memory_io_recorder = None

        result = await loop._retrieve_adapter_memory(
            {
                "run_id": "run_001",
                "persona_id": "user_a",
                "scenario_description": "greeting",
                "interaction_context": "work",
                "active_preference_attributes": ["tone_formality"],
                "memory_retrieval_budget_items": 5,
            },
            session_key="stable:pa-session",
        )

        assert result is not None
        assert "formal tone" in result
        assert len(adapter.retrieve_calls) == 1
        req = adapter.retrieve_calls[0]
        assert req.scope.run_id == "run_001"
        assert req.scope.persona_id == "user_a"
        assert req.session_id == "stable:pa-session"
        assert not hasattr(req, "scenario_description")
        assert not hasattr(req, "interaction_context")
        assert not hasattr(req, "active_preference_attributes")
        assert req.budget.max_items == 5

    @pytest.mark.asyncio
    async def test_retrieve_uses_current_message_when_query_missing(self, tmp_path: Path) -> None:
        from pa.agent.loop import AgentLoop

        adapter = FakeAdapterForPA()
        loop = AgentLoop.__new__(AgentLoop)
        loop.memory_adapter = adapter
        loop.memory_io_recorder = None

        await loop._retrieve_adapter_memory(
            {"run_id": "run_001", "persona_id": "user_a"},
            current_message="How should I phrase this?",
            session_key="stable:pa-session",
        )

        assert adapter.retrieve_calls[0].query == "How should I phrase this?"

    @pytest.mark.asyncio
    async def test_retrieve_applies_shared_pa_model_token_budget(self) -> None:
        from pa.agent.loop import AgentLoop

        provider = BudgetCountingProvider()
        adapter = FakeAdapterForPA("preference detail " * 30)
        loop = AgentLoop.__new__(AgentLoop)
        loop.memory_adapter = adapter
        loop.memory_io_recorder = None
        loop.pa_llm_provider = provider
        loop.pa_model = "pa-backbone-model"

        result = await loop._retrieve_adapter_memory(
            {
                "run_id": "run_001",
                "persona_id": "user_a",
                "memory_retrieval_budget_tokens": 120,
            },
            session_key="stable:pa-session",
        )

        assert result is not None
        assert len(result) <= 120
        assert result.startswith('<memory-context backend="fake_pa">')
        assert result.endswith("</memory-context>")
        assert provider.estimation_models
        assert set(provider.estimation_models) == {"pa-backbone-model"}
        assert adapter.retrieve_calls[0].budget.max_tokens == 120
        assert adapter.last_retrieve_result is not None
        trace = adapter.last_retrieve_result.trace
        assert trace is not None
        assert trace.token_count == len(result)
        assert any("shared retrieval token budget" in warning for warning in trace.warnings)

    @pytest.mark.asyncio
    async def test_skip_when_no_adapter(self, tmp_path: Path) -> None:
        from pa.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.memory_adapter = None
        loop.memory_io_recorder = None

        result = await loop._retrieve_adapter_memory(
            {"run_id": "r", "persona_id": "p"},
            session_key="stable:pa-session",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_run_id_fails(self, tmp_path: Path) -> None:
        from pa.agent.loop import AdapterRetrievalError, AgentLoop

        adapter = FakeAdapterForPA()
        loop = AgentLoop.__new__(AgentLoop)
        loop.memory_adapter = adapter
        loop.memory_io_recorder = None

        with pytest.raises(AdapterRetrievalError, match="run_id"):
            await loop._retrieve_adapter_memory(
                {"persona_id": "user_a"},
                session_key="stable:pa-session",
            )
        assert len(adapter.retrieve_calls) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "value", "error_pattern"),
        [
            ("memory_retrieval_budget_tokens", 0, "token budget"),
            ("memory_retrieval_budget_tokens", True, "token budget"),
            ("memory_retrieval_budget_items", -1, "item budget"),
            ("memory_retrieval_budget_items", 1.5, "item budget"),
        ],
    )
    async def test_invalid_retrieval_budget_fails_before_adapter_call(
        self,
        field: str,
        value: object,
        error_pattern: str,
    ) -> None:
        from pa.agent.loop import AdapterRetrievalError, AgentLoop

        adapter = FakeAdapterForPA()
        loop = AgentLoop.__new__(AgentLoop)
        loop.memory_adapter = adapter
        loop.memory_io_recorder = None

        with pytest.raises(AdapterRetrievalError, match=error_pattern):
            await loop._retrieve_adapter_memory(
                {
                    "run_id": "run_001",
                    "persona_id": "user_a",
                    field: value,
                },
                session_key="stable:pa-session",
            )
        assert adapter.retrieve_calls == []

    @pytest.mark.asyncio
    async def test_missing_persona_id_fails(self, tmp_path: Path) -> None:
        from pa.agent.loop import AdapterRetrievalError, AgentLoop

        adapter = FakeAdapterForPA()
        loop = AgentLoop.__new__(AgentLoop)
        loop.memory_adapter = adapter
        loop.memory_io_recorder = None

        with pytest.raises(AdapterRetrievalError, match="persona_id"):
            await loop._retrieve_adapter_memory(
                {"run_id": "run_001"},
                session_key="stable:pa-session",
            )
        assert len(adapter.retrieve_calls) == 0

    @pytest.mark.asyncio
    async def test_retrieval_failure_stops_with_scope_values(self, tmp_path: Path) -> None:
        from pa.agent.loop import AdapterRetrievalError, AgentLoop

        adapter = FailingAdapter()
        loop = AgentLoop.__new__(AgentLoop)
        loop.memory_adapter = adapter
        loop.memory_io_recorder = None

        with pytest.raises(
            AdapterRetrievalError,
            match="run_id='run_001'.*persona_id='user_a'",
        ):
            await loop._retrieve_adapter_memory({
                "run_id": "run_001",
                "persona_id": "user_a",
            }, session_key="stable:pa-session")

    @pytest.mark.asyncio
    async def test_user_id_is_not_accepted_as_persona_id(self, tmp_path: Path) -> None:
        from pa.agent.loop import AdapterRetrievalError, AgentLoop

        adapter = FakeAdapterForPA()
        loop = AgentLoop.__new__(AgentLoop)
        loop.memory_adapter = adapter
        loop.memory_io_recorder = None

        with pytest.raises(AdapterRetrievalError, match="persona_id"):
            await loop._retrieve_adapter_memory({
                "run_id": "run_001",
                "user_id": "user_b",
            }, session_key="stable:pa-session")
        assert adapter.retrieve_calls == []

    @pytest.mark.asyncio
    async def test_process_direct_injects_adapter_context_into_system_prompt(
        self, tmp_path: Path
    ) -> None:
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus

        adapter = FakeAdapterForPA()
        provider = RecordingProvider()
        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=provider,
            workspace=tmp_path,
            pa_model="test-model",
            memory_adapter=adapter,
        )
        loop.sessions.create("test:memory")

        await loop.process_direct_message(
            "How should I answer this?",
            session_key="test:memory",
            metadata={"run_id": "run_001", "persona_id": "user_a"},
        )

        assert provider.calls
        system_prompt = provider.calls[0][0]["content"]
        assert "# Retrieved Memory" in system_prompt
        assert "User prefers formal tone." in system_prompt
        assert adapter.retrieve_calls[0].query == "How should I answer this?"

    @pytest.mark.asyncio
    async def test_process_direct_stops_before_persistence_on_provider_error(
        self, tmp_path: Path
    ) -> None:
        from pa.agent.loop import AgentLoop, AgentProviderError
        from pa.bus.queue import MessageBus

        adapter = FakeAdapterForPA()
        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=ErrorProvider(),
            workspace=tmp_path,
            pa_model="failing-model",
            memory_adapter=adapter,
        )
        loop.sessions.create("test:memory")

        with pytest.raises(
            AgentProviderError,
            match="model='failing-model'.*provider throughput limit",
        ):
            await loop.process_direct_message(
                "This turn must not be persisted.",
                session_key="test:memory",
                metadata={"run_id": "run_001", "persona_id": "user_a"},
            )

        assert loop.sessions.get("test:memory").messages == []
        assert adapter.record_events == []

    def test_adapter_selection_controls_native_file_memory(self, tmp_path: Path) -> None:
        from memory.conditions.nanobot_file_memory import (
            NanobotFileMemoryAdapter,
        )
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus

        external_loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=RecordingProvider(),
            workspace=tmp_path / "external",
            pa_model="test-model",
            memory_adapter=FakeAdapterForPA(),
        )
        file_loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=RecordingProvider(),
            workspace=tmp_path / "file",
            pa_model="test-model",
            memory_adapter=NanobotFileMemoryAdapter(
                tmp_path / "file" / "memory"
            ),
        )

        assert external_loop.native_file_memory_enabled is False
        assert file_loop.native_file_memory_enabled is True
        assert "MEMORY.md" not in external_loop.context.build_system_prompt()
        assert "HISTORY.md" not in external_loop.context.build_system_prompt()
        assert "Long-term memory" in file_loop.context.build_system_prompt()

    @pytest.mark.asyncio
    async def test_file_memory_prompt_carries_memory_md_once(
        self, tmp_path: Path
    ) -> None:
        """MEMORY.md reaches the File memory PA once, whole, under `# Memory`.

        It used to arrive twice: in full under `# Memory` and again through
        the adapter under `# Retrieved Memory`, cut to the retrieval budget.
        """
        from memory.conditions.nanobot_file_memory import (
            NanobotFileMemoryAdapter,
        )
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus

        marker = "User wants bullet-point answers at work."
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "MEMORY.md").write_text(
            f"# Preferences\n- {marker}\n", encoding="utf-8"
        )
        provider = RecordingProvider()
        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=provider,
            workspace=tmp_path,
            pa_model="test-model",
            memory_adapter=NanobotFileMemoryAdapter(tmp_path / "memory"),
        )
        loop.sessions.create("test:memory")

        await loop.process_direct_message(
            "Summarise the thread.",
            session_key="test:memory",
            metadata={
                "run_id": "run_001",
                "persona_id": "user_a",
                "memory_retrieval_budget_tokens": 4000,
                "memory_retrieval_budget_items": 10,
            },
        )

        system_prompt = provider.calls[0][0]["content"]
        assert system_prompt.count(marker) == 1
        assert f"# Memory\n\n## Long-term Memory\n# Preferences\n- {marker}" in (
            system_prompt
        )
        assert "# Retrieved Memory" not in system_prompt

    @pytest.mark.asyncio
    async def test_external_condition_prompt_keeps_retrieved_memory(
        self, tmp_path: Path
    ) -> None:
        """The File memory fix leaves an external backend's prompt as it was."""
        from memory.backends.mem0_adapter import Mem0Adapter
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus

        marker = "User wants bullet-point answers at work."

        class SearchOnlyMem0Client:
            def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
                return {"results": [{"id": "m1", "memory": marker, "score": 0.9}]}

        adapter = Mem0Adapter(client=SearchOnlyMem0Client())
        await adapter.set_mode(AdapterMode(read_only=True, reason="test"))
        provider = RecordingProvider()
        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=provider,
            workspace=tmp_path,
            pa_model="test-model",
            memory_adapter=adapter,
        )
        loop.sessions.create("test:memory")

        await loop.process_direct_message(
            "Summarise the thread.",
            session_key="test:memory",
            metadata={"run_id": "run_001", "persona_id": "user_a"},
        )

        system_prompt = provider.calls[0][0]["content"]
        assert system_prompt.count(marker) == 1
        assert (
            "# Retrieved Memory\n\n"
            f'<memory-context backend="mem0">\n{marker}\n</memory-context>'
        ) in system_prompt
        assert "# Memory\n" not in system_prompt

    def test_consolidation_lm_can_be_configured_independently(self, tmp_path: Path) -> None:
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus

        pa_provider = RecordingProvider()
        memory_provider = RecordingProvider()
        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=pa_provider,
            workspace=tmp_path,
            pa_model="pa-model",
            consolidation_llm_provider=memory_provider,
            consolidation_model="fixed-memory-model",
        )

        assert loop.memory_consolidator.pa_llm_provider is pa_provider
        assert loop.memory_consolidator.pa_model == "pa-model"
        assert (
            loop.memory_consolidator.consolidation_llm_provider
            is memory_provider
        )
        assert loop.memory_consolidator.consolidation_model == "fixed-memory-model"

    @pytest.mark.asyncio
    async def test_strict_consolidation_failure_is_not_backgrounded(
        self,
        tmp_path: Path,
    ) -> None:
        from pa.agent.loop import AgentLoop
        from pa.agent.memory import MemoryConsolidationError
        from pa.bus.queue import MessageBus
        from pa.session.manager import Session

        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=RecordingProvider(),
            workspace=tmp_path,
        )

        async def fail_consolidation(
            _session: Session, *, stage: str = "legacy",
        ) -> None:
            raise MemoryConsolidationError("INVALID_CONSOLIDATION")

        loop._compact_prompt_history_if_needed = fail_consolidation

        with pytest.raises(MemoryConsolidationError, match="INVALID_CONSOLIDATION"):
            await loop._compact_prompt_history_after_turn(
                Session(key="test:strict")
            )

        assert loop._background_tasks == []

    @pytest.mark.asyncio
    async def test_process_direct_records_visible_turn_to_adapter(
        self, tmp_path: Path
    ) -> None:
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus

        adapter = FakeAdapterForPA()
        provider = RecordingProvider()
        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=provider,
            workspace=tmp_path,
            pa_model="test-model",
            memory_adapter=adapter,
        )
        loop.sessions.create("test:memory")

        benchmark_turn_timestamp = datetime(
            2026, 10, 1, 9, 0, tzinfo=timezone.utc
        )
        await loop.process_direct_message(
            "Remember my preference.",
            session_key="test:memory",
            timestamp=benchmark_turn_timestamp,
            metadata={
                "run_id": "run_001",
                "persona_id": "user_a",
                "agent_id": "nanobot",
                "session_id": "s01",
                "scenario_description": "preference",
                "interaction_context": "work",
                "active_preference_attributes": ["tone_formality"],
            },
        )

        assert len(adapter.record_events) == 1
        event = adapter.record_events[0]
        assert event.run_id == "run_001"
        assert event.persona_id == "user_a"
        assert event.agent_id == "nanobot"
        assert event.session_id == "test:memory"
        assert event.timestamp == benchmark_turn_timestamp
        assert not hasattr(event, "scenario_description")
        assert not hasattr(event, "interaction_context")
        assert not hasattr(event, "active_preference_attributes")
        assert event.messages == [
            {"role": "user", "content": "Remember my preference."},
            {"role": "assistant", "content": "done"},
        ]

    @pytest.mark.asyncio
    async def test_process_direct_rejects_naive_explicit_timestamp(self) -> None:
        from pa.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        with pytest.raises(ValueError, match="timestamp must include a timezone"):
            await loop.process_direct_message(
                "Message",
                timestamp=datetime(2026, 10, 1, 9, 0),
            )

    @pytest.mark.asyncio
    async def test_memory_io_separates_audit_labels_from_backend_requests(
        self,
        tmp_path: Path,
    ) -> None:
        from harness.memory_io import MemoryIORecorder
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus

        audit_path = tmp_path / "memory_io.jsonl"
        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=RecordingProvider(),
            workspace=tmp_path,
            memory_adapter=CommittedAdapter(),
            require_committed_adapter_writes=True,
            memory_io_recorder=MemoryIORecorder(
                audit_path,
                run_id="run_001",
            ),
        )
        loop.sessions.create("run_001_main")

        await loop.process_direct_message(
            "Please draft this.",
            session_key="run_001_main",
            metadata={
                "run_id": "run_001",
                "persona_id": "user_a",
                "script_session_id": "acc_001",
                "turn_id": "acc_001:beat_01",
                "scenario_description": "hidden scenario",
                "interaction_context": "work",
                "active_preference_attributes": ["tone_formality"],
                "memory_retrieval_budget_tokens": 4000,
            },
        )

        records = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [record["operation"] for record in records] == [
            "retrieve",
            "add",
        ]
        assert [record["sequence"] for record in records] == [1, 2]
        assert isinstance(records[0]["token_count"], int)
        assert records[0]["token_count"] <= 4000
        assert records[0]["warnings"] == []
        for record in records:
            assert record["step_id"] == "acc_001"
            assert record["session_id"] == "run_001_main"
            assert record["audit_only_labels"] == {
                "active_preference_attributes": ["tone_formality"],
                "interaction_context": "work",
                "scenario_description": "hidden scenario",
                "script_session_id": "acc_001",
            }
            backend_request = json.dumps(
                record["backend_request"],
                ensure_ascii=False,
            )
            assert "hidden scenario" not in backend_request
            assert "tone_formality" not in backend_request

    @pytest.mark.asyncio
    async def test_internal_ix_reminder_is_not_persisted_or_written_to_memory(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime

        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus
        from pa.session.manager import Session

        adapter = FakeAdapterForPA()
        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=RecordingProvider(),
            workspace=tmp_path,
            memory_adapter=adapter,
        )
        reminder = {
            "role": "user",
            "content": "Internal IX control instruction.",
            "_mempa_internal": True,
        }
        turn = [
            {"role": "user", "content": "Summarize the paper."},
            reminder,
            {"role": "assistant", "content": "Summary."},
        ]
        session = Session(key="test:internal-reminder")

        loop._save_turn(session, turn, 0)
        await loop._write_turn_to_memory_adapter(
            {"run_id": "run_001", "persona_id": "user_a"},
            build_turn_pool(turn),
            timestamp=datetime.now(),
            session_key=session.key,
            turn_id="turn_1",
        )

        expected = [
            {"role": "user", "content": "Summarize the paper."},
            {"role": "assistant", "content": "Summary."},
        ]
        assert [
            {"role": item["role"], "content": item["content"]}
            for item in session.messages
        ] == expected
        assert adapter.record_events[0].messages == expected

    @pytest.mark.asyncio
    async def test_committed_adapter_write_advances_only_adapter_index(
        self,
        tmp_path: Path,
    ) -> None:
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus

        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=RecordingProvider(),
            workspace=tmp_path,
            memory_adapter=CommittedAdapter(),
            require_committed_adapter_writes=True,
        )
        loop.sessions.create("test:committed")

        await loop.process_direct_message(
            "Remember this.",
            session_key="test:committed",
            metadata={"run_id": "run_001", "persona_id": "user_a"},
        )

        session = loop.sessions.get("test:committed")
        assert session.adapter_committed_end_index == len(session.messages)
        assert session.file_consolidated_end_index == 0
        assert session.prompt_history_start_index == 0

    @pytest.mark.asyncio
    async def test_queued_adapter_write_fails_formal_run(
        self,
        tmp_path: Path,
    ) -> None:
        from pa.agent.loop import AdapterWriteCommitError, AgentLoop
        from pa.bus.queue import MessageBus

        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=RecordingProvider(),
            workspace=tmp_path,
            memory_adapter=QueuedAdapter(),
            require_committed_adapter_writes=True,
        )
        loop.sessions.create("test:queued")

        with pytest.raises(AdapterWriteCommitError, match="status=queued"):
            await loop.process_direct_message(
                "Remember this.",
                session_key="test:queued",
                metadata={"run_id": "run_001", "persona_id": "user_a"},
            )

        session = loop.sessions.get("test:queued")
        assert session.adapter_committed_end_index == 0
        loop.sessions.invalidate("test:queued")
        persisted = loop.sessions.get("test:queued")
        assert persisted.messages
        assert persisted.adapter_committed_end_index == 0

    @pytest.mark.asyncio
    async def test_successful_no_new_memory_advances_adapter_index(
        self,
        tmp_path: Path,
    ) -> None:
        from pa.agent.loop import AgentLoop
        from pa.bus.queue import MessageBus

        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=RecordingProvider(),
            workspace=tmp_path,
            memory_adapter=NoNewMemoryAdapter(),
            require_committed_adapter_writes=True,
        )
        loop.sessions.create("test:no-new-memory")

        await loop.process_direct_message(
            "No durable fact in this closing acknowledgment.",
            session_key="test:no-new-memory",
            metadata={"run_id": "run_001", "persona_id": "user_a"},
        )

        session = loop.sessions.get("test:no-new-memory")
        assert session.adapter_committed_end_index == len(session.messages)

    @pytest.mark.asyncio
    async def test_process_direct_fails_adapter_retrieval_without_scope(
        self, tmp_path: Path
    ) -> None:
        from pa.agent.loop import AdapterRetrievalError, AgentLoop
        from pa.bus.queue import MessageBus

        adapter = FakeAdapterForPA()
        provider = RecordingProvider()
        loop = AgentLoop(
            bus=MessageBus(),
            pa_llm_provider=provider,
            workspace=tmp_path,
            pa_model="test-model",
            memory_adapter=adapter,
        )
        loop.sessions.create("test:memory")

        with pytest.raises(AdapterRetrievalError, match="missing_fields"):
            await loop.process_direct_message("Hello", session_key="test:memory")

        assert adapter.record_events == []
