from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.memory_runtime import MemoryRuntime, capabilities_for
from harness.run_plan import RunStep
from memory.adapter import PromptHistoryEvictionPolicy
from memory.conditions.nanobot_file_memory import NanobotFileMemoryAdapter
from memory.event_writer import record_adapter_write_commit, write_visible_memory_event
from memory.results import (
    BackendHealth,
    BackendReadiness,
    ConsistencyModel,
    HealthStatus,
    WriteReceipt,
    WriteStatus,
)


class FakeAdapter:
    backend_name = "fake"

    def __init__(self):
        self.modes = []
        self.scopes = []

    async def setup_scope(self, scope):
        self.scopes.append(scope)

    async def set_mode(self, mode):
        self.modes.append(mode)
        return None

    async def health(self):
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=1.0,
            consistency_model=ConsistencyModel.immediate,
        )


@pytest.mark.asyncio
async def test_memory_runtime_applies_read_write():
    adapter = FakeAdapter()
    runtime = MemoryRuntime(adapter, run_id="run_001", persona_id="user_a")

    await runtime.apply_mode("read_write", reason="acc_001")

    assert adapter.modes[-1].read_only is False
    assert adapter.modes[-1].reason == "acc_001"


@pytest.mark.asyncio
async def test_memory_runtime_applies_read_only():
    adapter = FakeAdapter()
    runtime = MemoryRuntime(adapter, run_id="run_001", persona_id="user_a")

    await runtime.apply_mode("read_only", reason="pretest")

    assert adapter.modes[-1].read_only is True
    assert adapter.modes[-1].reason == "pretest"


class _AsyncDerivedAdapter(FakeAdapter):
    backend_name = "async_derived_fake"

    async def health(self):
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=1.0,
            consistency_model=ConsistencyModel.async_derived,
        )


class _ReadyAdapter(_AsyncDerivedAdapter):
    backend_name = "ready_fake"

    def __init__(self, *, ready: bool = True):
        super().__init__()
        self._ready = ready
        self.waited = []

    async def readiness(self, scope):
        return BackendReadiness(ready=self._ready)

    async def wait_until_ready(self, scope, timeout_s):
        self.waited.append(timeout_s)
        return BackendReadiness(ready=self._ready, pending_count=0 if self._ready else 2)


class _Session:
    def __init__(self, committed: int, count: int) -> None:
        self.adapter_committed_end_index = committed
        self.file_consolidated_end_index = 0
        self.messages = [{"role": "user"}] * count


@pytest.mark.asyncio
async def test_await_checkpoint_ready_requires_readiness_for_async_backends():
    # async_derived without SupportsReadiness: a checkpoint cannot prove
    # anything locally, so it must refuse.
    runtime = MemoryRuntime(
        _AsyncDerivedAdapter(), run_id="run_001", persona_id="user_a"
    )
    with pytest.raises(RuntimeError):
        await runtime.await_checkpoint_ready(5.0)

    # async_derived that implements SupportsReadiness: the backend is asked.
    adapter = _ReadyAdapter()
    runtime = MemoryRuntime(adapter, run_id="run_001", persona_id="user_a")
    await runtime.await_checkpoint_ready(7.0)
    assert adapter.waited == [7.0]

    not_ready = _ReadyAdapter(ready=False)
    runtime = MemoryRuntime(not_ready, run_id="run_001", persona_id="user_a")
    with pytest.raises(RuntimeError):
        await runtime.await_checkpoint_ready(7.0)

    # immediate backend: passes on the local assertion.
    immediate = FakeAdapter()
    immediate.prompt_history_eviction_policy = (
        PromptHistoryEvictionPolicy.after_adapter_commit
    )
    runtime = MemoryRuntime(immediate, run_id="run_001", persona_id="user_a")
    await runtime.await_checkpoint_ready(5.0, session=_Session(3, 3))
    with pytest.raises(RuntimeError):
        await runtime.await_checkpoint_ready(5.0, session=_Session(1, 3))


def test_capabilities_follow_backend():
    """Only backends with a real export/restore path may declare support."""
    supported = {
        "nanobot_file_memory",
        "no_memory",
        "mem0",
        "simple_rag",
        "memos",
    }
    for backend in supported:
        capabilities = capabilities_for(backend)
        assert capabilities.supports_checkpoint is True, backend
        assert capabilities.supports_restore is True, backend

    for backend in ("", None, "not_a_backend"):
        capabilities = capabilities_for(backend)
        assert capabilities.supports_checkpoint is False, backend
        assert capabilities.supports_restore is False, backend
        assert capabilities.strategy == "none", backend

    assert capabilities_for("no_memory").strategy == "none"
    assert capabilities_for("nanobot_file_memory").strategy == "scope_clone"

    adapter = FakeAdapter()
    adapter.backend_name = "mem0"
    runtime = MemoryRuntime(adapter, run_id="run_001", persona_id="user_a")
    assert runtime.capabilities == capabilities_for("mem0")


# ---------------------------------------------------------------------------
# the three consolidation entry points read the turn journal
# ---------------------------------------------------------------------------
#
# The file-memory condition used to consolidate
# `session.messages[file_consolidated_end_index:]` through
# the two-message projection the pool replaces. The pool is
# projection 3, built at live time from the raw provider slice, and the
# persisted slice cannot re-derive it. So the journal, not the
# message list, is what mid-session compaction and the pinned flush consume.

import asyncio
import json
from typing import Any

from pa.agent.loop import AgentLoop
from pa.agent.tools.base import Tool
from pa.bus.queue import MessageBus
from pa.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from pa.session.manager import SessionManager

JOURNAL_SESSION_KEY = "benchmark:main"


class _Echo(Tool):
    @property
    def name(self) -> str:
        return "task_echo"

    @property
    def description(self) -> str:
        return "Echo a value."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    async def execute(self, value: str = "") -> dict[str, str]:
        return {"value": value}


class _ScriptProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = list(responses)

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)


class _CapturingMemoryProvider(LLMProvider):
    """The consolidation LM: records the prompt it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.generation = GenerationSettings(max_tokens=256)
        self.prompts: list[str] = []

    def get_default_model(self) -> str:
        return "fake-memory-model"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self.prompts.append(
            "\n".join(str(message.get("content") or "") for message in messages)
        )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="save_1", name="save_memory",
                arguments={"history_entry": "H", "memory_update": "M"},
            )],
            finish_reason="tool_calls",
        )


def _journal_loop(
    workspace: Path, provider: LLMProvider, memory_provider: LLMProvider,
) -> AgentLoop:
    """A file-memory PA: no adapter, so the eviction policy is the file one."""
    loop = AgentLoop(
        bus=MessageBus(), pa_llm_provider=provider, workspace=workspace,
        pa_model="test-model",
        consolidation_llm_provider=memory_provider,
        consolidation_model="fake-memory-model",
    )
    loop.tools.register(_Echo())
    loop.sessions.create(JOURNAL_SESSION_KEY)
    return loop


async def _journal_turn(loop: AgentLoop, text: str):
    return await loop.process_direct_message(
        text,
        session_key=JOURNAL_SESSION_KEY,
        metadata={"run_id": "run_001", "persona_id": "user_a",
                  "script_session_id": "acc_001"},
    )


def _order_turn(value: str, reply: str) -> list[LLMResponse]:
    return [
        LLMResponse(
            content=f"Let me look up {value}.",
            reasoning_content=f"The order id is {value}.",
            tool_calls=[ToolCallRequest(id=f"echo_{value}", name="task_echo",
                                        arguments={"value": value})],
        ),
        LLMResponse(content=reply),
    ]


def _force_compaction(loop: AgentLoop, session) -> None:
    loop.memory_consolidator.estimate_session_prompt_tokens = (
        lambda _session: (25_000, "test")
    )


def test_consolidation_receives_the_full_rendering(tmp_path: Path) -> None:
    """Mid-session compaction hands the LM the pool, not the old projection.

    The old projection kept the `user`/`assistant` text and dropped
    everything else, so MEMORY.md never learned that the PA reasoned about the
    order id or looked it up. The journal record carries the whole turn.
    """
    async def run() -> None:
        memory_provider = _CapturingMemoryProvider()
        provider = _ScriptProvider(
            _order_turn("MB-55021", "It is out for delivery.")
            + _order_turn("MB-55022", "That one shipped yesterday.")
        )
        loop = _journal_loop(tmp_path, provider, memory_provider)

        await _journal_turn(loop, "Any news on my order?")
        await _journal_turn(loop, "And the second one?")

        session = loop.sessions.get(JOURNAL_SESSION_KEY)
        _force_compaction(loop, session)
        await loop.memory_consolidator.maybe_compact_prompt_history(session)

        prompt = "\n".join(memory_provider.prompts)
        # The person's text and the delivered reply, as before.
        assert "Any news on my order?" in prompt
        assert "It is out for delivery." in prompt
        # And the three things projection 2 threw away.
        assert "[reasoning source=reasoning_content]" in prompt
        assert "The order id is MB-55021." in prompt
        assert "[tool_call name=task_echo" in prompt
        assert "[prose]" in prompt
        assert "Let me look up MB-55021." in prompt
        # The journal cursor moved with the message cursor.
        assert session.file_consolidated_end_index > 0

    asyncio.run(run())


def test_suppressed_outreach_turn_reaches_file_consolidation(
    tmp_path: Path,
) -> None:
    """A `[n, n)` turn has no message to consolidate and must reach the files.

    `proactive_outreach` is scored on the decision to stay silent, so the one
    turn whose whole content is that decision cannot be the one the file
    condition drops. A message-index rule cannot carry it; the journal record
    can.
    """
    async def run() -> None:
        memory_provider = _CapturingMemoryProvider()
        provider = _ScriptProvider([
            LLMResponse(
                content=None,
                reasoning_content="The user hates nudges.",
                tool_calls=[ToolCallRequest(
                    id="ix_proactive", name="IX_proactive_outreach",
                    arguments={
                        "setting": "Low",
                        "evidence": "Retrieved preference evidence.",
                        "application": "suppress this contact.",
                        "outreach_action": "suppress",
                        "outreach_message": "The store closes at 6 PM.",
                    },
                )],
            ),
            LLMResponse(content="Internal completion text."),
        ] + _order_turn("MB-55021", "It is out for delivery.")
          + _order_turn("MB-55022", "That one shipped yesterday."))
        loop = _journal_loop(tmp_path, provider, memory_provider)

        await loop.process_environment_event(
            trigger_id="pickup_deadline", source_type="deadline",
            prior_thread="The comic order is ready for pickup.",
            detail="The store closes in two hours.",
            user_impact="User A may want time to collect the order today.",
            session_key=JOURNAL_SESSION_KEY,
            metadata={"run_id": "run_001", "persona_id": "user_a"},
            timestamp=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
        )
        await _journal_turn(loop, "Any news on my order?")
        await _journal_turn(loop, "And the second one?")

        session = loop.sessions.get(JOURNAL_SESSION_KEY)
        _force_compaction(loop, session)
        await loop.memory_consolidator.maybe_compact_prompt_history(session)

        prompt = "\n".join(memory_provider.prompts)
        assert "[environment_event" in prompt
        assert "The store closes in two hours." in prompt
        assert "[no_reply decision=suppress]" in prompt
        # The fixed instruction suffix is contract enforcement, not memory.
        assert "Use IX_proactive_outreach" not in prompt

    asyncio.run(run())


class _FlushHandle:
    """Only the pieces `_flush_uncompacted_turns` reaches on a real PA."""

    def __init__(self, workspace: Path, session_key: str) -> None:
        self.prompt_history_eviction_policy = (
            PromptHistoryEvictionPolicy.after_file_consolidation
        )
        self.sessions = SessionManager(workspace)
        self.session_key = session_key
        self.archived: list[list[dict[str, Any]]] = []

    def checkpoint_session(self):
        self.sessions.invalidate(self.session_key)
        return self.sessions.get(self.session_key)

    def turn_journal(self):
        from memory.event_writer import turn_journal_for

        return turn_journal_for(self.sessions, self.session_key)


    async def archive_messages_to_files(self, messages) -> bool:
        self.archived.append([dict(message) for message in messages])
        return True

    def save_session(self, session) -> None:
        self.sessions.save(session)

    async def drain_background_tasks(self) -> None:
        return None


def test_pinned_flush_consumes_journal_when_message_tail_is_empty(
    tmp_path: Path,
) -> None:
    """The flush must run with nothing left in `session.messages`.

    A pinned checkpoint is the only copy a probe branches from. A suppressed
    outreach as the last turn of a step leaves the message tail empty, and the
    old flush archived `session.messages[file_consolidated_end_index:]`, which
    is `[]` -- so the turn never reached the memory files and every probe
    branched from a memory that had never heard of it.
    """
    async def run() -> None:
        from harness.orchestrator import Orchestrator

        provider = _ScriptProvider([
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(
                    id="ix_proactive", name="IX_proactive_outreach",
                    arguments={
                        "setting": "Low",
                        "evidence": "Retrieved preference evidence.",
                        "application": "suppress this contact.",
                        "outreach_action": "suppress",
                        "outreach_message": "The store closes at 6 PM.",
                    },
                )],
            ),
            LLMResponse(content="Internal completion text."),
        ])
        loop = _journal_loop(tmp_path, provider, _CapturingMemoryProvider())
        await loop.process_environment_event(
            trigger_id="pickup_deadline", source_type="deadline",
            prior_thread="The comic order is ready for pickup.",
            detail="The store closes in two hours.",
            user_impact="User A may want time to collect the order today.",
            session_key=JOURNAL_SESSION_KEY,
            metadata={"run_id": "run_001", "persona_id": "user_a"},
            timestamp=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
        )

        session = loop.sessions.get(JOURNAL_SESSION_KEY)
        assert session.messages == []
        assert session.file_consolidated_end_index == len(session.messages)

        handle = _FlushHandle(tmp_path, JOURNAL_SESSION_KEY)
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.pa_checkpoint = handle
        await orchestrator._flush_uncompacted_turns()

        assert len(handle.archived) == 1
        blob = json.dumps(handle.archived[0], default=str)
        assert "[environment_event" in blob
        assert "[no_reply decision=suppress]" in blob

        # A second flush finds the record already consolidated and re-archives
        # nothing: the journal's own mark, not the message cursor, says so.
        await orchestrator._flush_uncompacted_turns()
        assert len(handle.archived) == 1

    asyncio.run(run())
class _ReadOnlyGateAdapter(FakeAdapter):
    """The read-only gate every adapter implements as `record_event`'s first
    statement (`mem0_adapter.py:71-73`, `memos_adapter.py:100-102`,
    `nanobot_file_memory.py:52-56`), with the native call recorded so a probe
    can be shown not to have reached it.
    """

    backend_name = "read_only_gate_fake"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.after_adapter_commit

    def __init__(self):
        super().__init__()
        self.events = []
        self.native_writes = []
        self._read_only = False

    async def set_mode(self, mode):
        self._read_only = mode.read_only
        return await super().set_mode(mode)

    async def record_event(self, event):
        self.events.append(event)
        if self._read_only:
            return WriteReceipt(status=WriteStatus.skipped_read_only)
        self.native_writes.append(event)
        return WriteReceipt(status=WriteStatus.committed, native_ids=[event.event_id])


class _RecordingFileAdapter(NanobotFileMemoryAdapter):
    """The real file condition, with `record_event` and the native
    `FileMemoryBackend.add` behind it both recorded.
    """

    backend_name = "recording_nanobot_file_memory"

    def __init__(self, memory_dir):
        super().__init__(memory_dir)
        self.modes = []
        self.events = []
        self.native_writes = []
        original_add = self._file_state.add

        async def _recording_add(messages, metadata=None):
            self.native_writes.append(messages)
            return await original_add(messages, metadata)

        self._file_state.add = _recording_add

    async def set_mode(self, mode):
        self.modes.append(mode)
        return await super().set_mode(mode)

    async def record_event(self, event):
        self.events.append(event)
        return await super().record_event(event)


def _probe_step(kind: str) -> RunStep:
    """A probe step as the plan validator fixes it (`harness/run_plan.py:787-807`)."""
    return RunStep(
        step_id=f"{kind}_001",
        kind=kind,
        persona_id="user_a",
        script_path=Path("Authoring/sessions/probe.yaml"),
        benchmark_session_timestamp=datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc),
        memory_mode="read_only",
        stage_policy="discard",
        skip_consolidation=True,
        before_acc_num=7 if kind == "pre_event_probe" else None,
    )


_PROBE_TURN = [
    {"role": "user", "content": "Which of the two would you pick for me?"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "She has asked for a recommendation, not a menu.",
        "tool_calls": [
            {
                "id": "call_probe_01",
                "type": "function",
                "function": {
                    "name": "mcp_state_calendar_read",
                    "arguments": '{"date": "2026-09-13"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": '{"events": []}',
        "tool_call_id": "call_probe_01",
        "name": "mcp_state_calendar_read",
    },
    {"role": "assistant", "content": "The second one. Your afternoon is clear."},
]


@pytest.mark.parametrize("kind", ["pre_event_probe", "final_probe"])
@pytest.mark.asyncio
async def test_probe_step_writes_nothing_and_moves_no_cursor(
    kind: str, tmp_path: Path
) -> None:
    """The read-only gate is what makes a test probe retrieve-only.

    The pool change rewrites every write path, so this pins the three layers
    -- the step's fixed `memory_mode`, the mode applied once per
    step, and the per-adapter refusal -- against both eviction policies: the
    cursor-carrying one (`after_adapter_commit`) and the file condition, whose
    real state is MEMORY.md / HISTORY.md.
    """
    step = _probe_step(kind)
    assert step.memory_mode == "read_only"
    assert step.stage_policy == "discard"

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("# Long-term\n- She walks Glenmont.\n", encoding="utf-8")
    (memory_dir / "HISTORY.md").write_text("## 2026-09-12\nA settled afternoon.\n", encoding="utf-8")
    before = {
        name: (memory_dir / name).read_bytes() for name in ("MEMORY.md", "HISTORY.md")
    }

    adapters = [_ReadOnlyGateAdapter(), _RecordingFileAdapter(memory_dir)]
    for adapter in adapters:
        runtime = MemoryRuntime(adapter, run_id="run_001", persona_id="user_a")
        await runtime.apply_mode(step.memory_mode, reason=step.step_id)
        assert adapter.modes[-1].read_only is True

        # `committed` starts BEHIND the message count on purpose. With
        # `committed == count` both branches of `record_adapter_write_commit`
        # return the same number (`max(committed_index, current_index)`,
        # `memory/event_writer.py:347-353`), so assertion (ii) below could not
        # tell a committed write from a skipped one. At 0 it can: a
        # `WriteStatus.committed` receipt would advance it to 4.
        session = _Session(committed=0, count=4)
        session.file_consolidated_end_index = 2

        receipt = await write_visible_memory_event(
            adapter=adapter,
            messages=_PROBE_TURN,
            run_id="run_001",
            persona_id="user_a",
            agent_id="nanobot",
            session_id="run_001_probe_branch",
            turn_id=f"{step.step_id}:turn_01",
            event_id=f"run_001:{step.step_id}:turn_01",
            timestamp=step.benchmark_session_timestamp,
            step_id=step.step_id,
        )

        # (i) `record_event` IS entered on a probe turn -- the gate is its first
        # statement -- and every receipt is `skipped_read_only` with no native
        # write behind it.
        assert adapter.events, adapter.backend_name
        assert receipt is not None
        assert receipt.status is WriteStatus.skipped_read_only, adapter.backend_name
        assert adapter.native_writes == [], adapter.backend_name

        # (ii) neither cursor moves.
        session.adapter_committed_end_index = record_adapter_write_commit(
            receipt=receipt,
            session_id="run_001_probe_branch",
            current_index=len(session.messages),
            committed_index=session.adapter_committed_end_index,
            eviction_policy=adapter.prompt_history_eviction_policy,
            require_committed=False,
        )
        assert session.adapter_committed_end_index == 0, adapter.backend_name
        assert session.file_consolidated_end_index == 2, adapter.backend_name

    # (iii) the file condition's own state is byte-identical.
    assert {
        name: (memory_dir / name).read_bytes() for name in ("MEMORY.md", "HISTORY.md")
    } == before


# ---------------------------------------------------------------------------
# The forced-event line's own journal record.
# ---------------------------------------------------------------------------

FE_SESSION_KEY = "benchmark:fe_main"


async def _drive_forced_exchange(workspace: Path, adapter, session_key: str):
    """One forced exchange through the executor's own two commit methods.

    `_prepare_exchange` writes the pending journal record and
    `_commit_session_and_memory` extends the session, writes the pool to the
    adapter and marks the record written. This exchange is pure dialogue, so
    no tool is reached through the registry and the executor needs nothing of
    the stage but its `run_id`.
    """
    from types import SimpleNamespace

    from harness.forced_event_executor import ForcedEventExecutor
    from harness.forced_event_replay import GatedReplay
    from harness.forced_event_schema import MessageItem
    from pa.agent.tools.registry import ToolRegistry
    from pa.session.manager import SessionManager

    sessions = SessionManager(workspace)
    transcript = SimpleNamespace(
        event=SimpleNamespace(event_id="evolv_05", persona_id="user_a")
    )
    executor = ForcedEventExecutor(
        transcript_path=workspace / "evolv_05.yaml",
        project_root=workspace,
        benchmark_session_timestamp=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
        stage_runtime=SimpleNamespace(run_id="run_001", state_root=workspace),
        session_manager=sessions,
        pa_session_key=session_key,
        memory_adapter=adapter,
        memory_io_recorder=None,
        writer=None,
        require_committed_adapter_writes=False,
    )
    executor._replay = GatedReplay(gate=None)
    executor._open_journal()
    prepared = await executor._prepare_exchange(
        transcript,
        exchange_index=1,
        items=[
            MessageItem(
                type="message", offset_seconds=0, source="user",
                visibility="both",
                content="The Glenmont office wrote about the trail entrance.",
            ),
            MessageItem(
                type="message", offset_seconds=20, source="pa",
                visibility="both",
                content="I left the reply drafted and unsent until you read it.",
            ),
        ],
        registry=ToolRegistry(),
        actual_call_ids=[],
        actual_action_ids=[],
    )
    await executor._commit_session_and_memory(transcript, [prepared])
    return sessions, prepared


class _PromptEchoMemoryProvider(LLMProvider):
    """The consolidation LM, writing its own prompt back out as MEMORY.md.

    MEMORY.md then holds exactly the text the LM read, so the assertion can
    compare the flushed file against the rendered pool byte for byte instead
    of against whatever a fake chose to summarise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.generation = GenerationSettings(max_tokens=256)

    def get_default_model(self) -> str:
        return "fake-memory-model"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="save_1", name="save_memory",
                arguments={
                    "history_entry": "[2026-10-01 09:00] forced event",
                    "memory_update": messages[-1]["content"],
                },
            )],
            finish_reason="tool_calls",
        )


class _MarkdownFlushHandle:
    """A flush handle whose archive really writes MEMORY.md and HISTORY.md."""

    def __init__(self, workspace: Path, session_key: str, provider) -> None:
        from pa.agent.memory import MarkdownMemoryStore
        from pa.session.manager import SessionManager

        self.prompt_history_eviction_policy = (
            PromptHistoryEvictionPolicy.after_file_consolidation
        )
        self.sessions = SessionManager(workspace)
        self.session_key = session_key
        self.store = MarkdownMemoryStore(workspace, log_scope="fe_file_memory")
        self.provider = provider

    def checkpoint_session(self):
        self.sessions.invalidate(self.session_key)
        return self.sessions.get(self.session_key)

    def turn_journal(self):
        from memory.event_writer import turn_journal_for

        return turn_journal_for(self.sessions, self.session_key)

    async def archive_messages_to_files(self, messages) -> bool:
        return await self.store.consolidate(
            [dict(message) for message in messages],
            self.provider,
            "fake-memory-model",
        )

    def save_session(self, session) -> None:
        self.sessions.save(session)

    async def drain_background_tasks(self) -> None:
        return None


@pytest.mark.asyncio
async def test_forced_exchange_reaches_the_memory_files_under_the_file_condition(
    tmp_path: Path,
) -> None:
    """A forced event must reach MEMORY.md/HISTORY.md.

    `NanobotFileMemoryAdapter.record_event` is a documented no-op that returns
    `deferred_to_file_consolidation` (`memory/conditions/nanobot_file_memory.py:50-61`),
    and the pinned flush reads the journal rather than
    `session.messages` — so an executor that writes no record writes nothing
    to the files at all, whatever the session holds. With the record in place
    the flush consumes it and the rendered pool is what the consolidation LM
    reads.
    """
    from types import SimpleNamespace

    from harness.orchestrator import Orchestrator
    from pa.session.manager import SessionManager

    SessionManager(tmp_path).create(FE_SESSION_KEY)
    adapter = NanobotFileMemoryAdapter(tmp_path / "memory")
    sessions, prepared = await _drive_forced_exchange(
        tmp_path, adapter, FE_SESSION_KEY
    )

    handle = _MarkdownFlushHandle(
        tmp_path, FE_SESSION_KEY, _PromptEchoMemoryProvider()
    )
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.pa_checkpoint = handle
    await orchestrator._flush_uncompacted_turns()

    memory = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    history = (tmp_path / "memory" / "HISTORY.md").read_text(encoding="utf-8")
    assert history.strip()
    # The flushed text is the rendered pool, message for message -- not the
    # session's own messages, and not a two-line projection of them.
    assert prepared.pool.messages
    for message in prepared.pool.messages:
        assert message["content"] in memory
    # And it carries the turn's own clock, which is what the record holds.
    assert "2026-10-01T09:00" in memory

    # The journal's second completion track is marked, so a second flush
    # re-archives nothing.
    record = handle.turn_journal().record_for("evolv_05:exchange_01")
    assert record.file_consolidated is True
    # A pure-dialogue exchange realized nothing, and says so rather than
    # leaving `realized` absent -- which is treated as an error.
    assert record.realized == {"results": {}, "action_ids": []}
