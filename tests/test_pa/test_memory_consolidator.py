"""Prompt-history compaction for the nanobot_file_memory condition.

These tests pin the write side of the file-memory condition: consolidation
only fires once the running prompt crosses the token budget, and when it
fires it must (a) overwrite MEMORY.md with the long-term update, (b) append
the summary to HISTORY.md, and (c) advance the session's consolidation
index so the same messages are not re-archived.
"""

import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from loguru import logger

from memory.adapter import PromptHistoryEvictionPolicy
from memory.event_writer import (
    build_turn_pool,
    consolidation_records,
    turn_journal_for,
)
from memory.turn_journal import JournalRecord
from pa.agent.memory import (
    MarkdownMemoryStore,
    MemoryConsolidationError,
    MemoryConsolidator,
    PromptHistoryEvictionBlockedError,
)
from pa.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from pa.session.manager import Session, SessionManager


class FakeConsolidationProvider(LLMProvider):
    """Returns a fixed save_memory tool call without hitting a real LLM."""

    def __init__(self, *, memory_update: str, history_entry: str, max_tokens: int = 256):
        super().__init__()
        self.generation = GenerationSettings(max_tokens=max_tokens)
        self._memory_update = memory_update
        self._history_entry = history_entry
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self.calls.append({
            "tool_choice": tool_choice,
            "model": model,
            "max_tokens": max_tokens,
            "task_identity": self.current_task_identity,
            "messages": messages,
        })
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments={
                        "history_entry": self._history_entry,
                        "memory_update": self._memory_update,
                    },
                )
            ],
            finish_reason="tool_calls",
        )

    def get_default_model(self) -> str:
        return "fake-model"


@pytest.mark.asyncio
async def test_markdown_consolidation_log_identifies_memory_scope(tmp_path):
    provider = FakeConsolidationProvider(
        memory_update="memory",
        history_entry="[2026-07-28 12:00] history",
    )
    store = MarkdownMemoryStore(
        tmp_path,
        log_scope="pa_file_memory",
    )
    log_messages: list[str] = []
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        await store.consolidate(
            [{"role": "user", "content": "remember this"}],
            provider,
            "fake-memory-model",
        )
    finally:
        logger.remove(sink_id)

    assert any(
        "Markdown memory consolidation committed: "
        "scope=pa_file_memory" in message
        for message in log_messages
    )
    assert provider.calls[0]["max_tokens"] == 256


class MissingToolCallProvider(FakeConsolidationProvider):
    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self.calls.append({
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "task_identity": self.current_task_identity,
        })
        return LLMResponse(content="Could not consolidate.", finish_reason="stop")


class DetailedErrorProvider(FakeConsolidationProvider):
    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        return LLMResponse(
            content=None,
            finish_reason="error",
            error_detail={
                "response_id": "request_123",
                "response_error": {
                    "code": "invalid_request",
                    "message": "provider rejected request",
                },
            },
        )


class UnsupportedForcedToolProvider(FakeConsolidationProvider):
    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self.calls.append({"tool_choice": tool_choice, "max_tokens": max_tokens})
        if len(self.calls) == 1:
            return LLMResponse(
                content="tool_choice does not support forced functions",
                finish_reason="error",
            )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="call_1",
                name="save_memory",
                arguments={
                    "history_entry": self._history_entry,
                    "memory_update": self._memory_update,
                },
            )],
            finish_reason="tool_calls",
        )


@pytest.mark.asyncio
async def test_file_consolidation_auto_fallback_keeps_explicit_budget(tmp_path):
    provider = UnsupportedForcedToolProvider(
        memory_update="memory",
        history_entry="[2026-09-20 14:00] history",
    )
    store = MarkdownMemoryStore(tmp_path, log_scope="pa_file_memory")

    await store.consolidate(
        [{"role": "user", "content": "remember this"}],
        provider,
        "fake-memory-model",
        max_tokens=8192,
    )

    assert [call["max_tokens"] for call in provider.calls] == [8192, 8192]
    assert provider.calls[0]["tool_choice"]["function"]["name"] == "save_memory"
    assert provider.calls[1]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_consolidation_contract_error_preserves_provider_detail(tmp_path):
    provider = DetailedErrorProvider(memory_update="", history_entry="")
    store = MarkdownMemoryStore(tmp_path, log_scope="provider_error_test")

    with pytest.raises(
        MemoryConsolidationError,
        match="provider rejected request",
    ):
        await store.consolidate(
            [{"role": "user", "content": "remember this"}],
            provider,
            "fake-model",
        )


class FakeEstimationProvider(LLMProvider):
    def __init__(self, estimates: list[int]):
        super().__init__()
        self._estimates = iter(estimates)
        self.calls: list[dict] = []

    def estimate_prompt_tokens(self, messages, tools, model):
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        return next(self._estimates), "fake-pa-counter"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        raise AssertionError("PA estimation provider must not generate consolidation")

    def get_default_model(self) -> str:
        return "fake-pa-model"


def _build_messages(history, current_message, channel=None, chat_id=None):
    """Minimal stand-in for AgentLoop.context.build_messages (token-probe view)."""
    return list(history) + [{"role": "user", "content": current_message}]


def _make_consolidator(
    workspace: Path,
    provider: LLMProvider,
    context_window_tokens: int,
    *,
    prompt_history_eviction_policy: PromptHistoryEvictionPolicy = (
        PromptHistoryEvictionPolicy.after_file_consolidation
    ),
    require_committed_adapter_writes: bool = False,
    pa_max_completion_tokens: int | None = None,
):
    return MemoryConsolidator(
        workspace=workspace,
        pa_llm_provider=provider,
        pa_model="fake-pa-model",
        consolidation_llm_provider=provider,
        consolidation_model="fake-memory-model",
        session_manager=SessionManager(workspace),
        pa_context_window_tokens=context_window_tokens,
        build_pa_prompt_messages=_build_messages,
        get_pa_tool_definitions=lambda: [],
        prompt_history_eviction_policy=prompt_history_eviction_policy,
        pa_max_completion_tokens=(
            provider.generation.max_tokens
            if pa_max_completion_tokens is None
            else pa_max_completion_tokens
        ),
        require_committed_adapter_writes=require_committed_adapter_writes,
    )


def _make_strict_consolidator(
    workspace: Path,
    provider: LLMProvider,
    context_window_tokens: int,
):
    return _make_consolidator(
        workspace,
        provider,
        context_window_tokens,
    )


@pytest.mark.asyncio
async def test_file_consolidation_uses_8192_without_changing_provider_default(tmp_path):
    provider = FakeConsolidationProvider(
        memory_update="memory",
        history_entry="[2026-09-20 14:00] history",
        max_tokens=4096,
    )
    consolidator = _make_consolidator(
        tmp_path,
        provider,
        context_window_tokens=65_536,
        pa_max_completion_tokens=8192,
    )

    await consolidator.consolidate_messages_to_files(
        [{"role": "user", "content": "remember this"}],
        task_identity=None,
    )

    assert provider.generation.max_tokens == 4096
    assert provider.calls[0]["max_tokens"] == 8192


def _chatty_session(key: str, turns: int, chars: int) -> Session:
    """Alternating user/assistant turns, each large enough to move the needle."""
    session = Session(key=key)
    for i in range(turns):
        role = "user" if i % 2 == 0 else "assistant"
        session.messages.append(
            {"role": role, "content": f"turn {i}: " + ("x " * chars)}
        )
    return session


def _seed_journal(workspace: Path, session: Session, *, stride: int = 2) -> None:
    """One journal record per turn, as the loop would have written them.

    After §9 step 7 the file condition consolidates journal records, not
    `session.messages` -- the persisted slice cannot re-derive the pool
    (§6.1 (b)) -- so a session built by hand has to carry its own journal or
    there is nothing for compaction to hand the LM.
    """
    journal = turn_journal_for(SessionManager(workspace), session.key)
    for index in range(0, len(session.messages), stride):
        chunk = session.messages[index:index + stride]
        journal.append_pending(JournalRecord.for_pool(
            turn_id=f"turn_{index}",
            span=(index, index + len(chunk)),
            timestamp=str(chunk[0].get("timestamp") or ""),
            pool=build_turn_pool([dict(message) for message in chunk]),
        ))


def test_snap_pulls_a_cut_inside_a_record_back_to_the_last_whole_one():
    """The consolidated prefix is a union of whole journal records (§6.1 (e)).

    `select_compaction_end_index` cuts at any `role: user` message, and a record
    can own more than one, so the cut can land inside one record's span. The
    snap moves it back to the end of the last record that is wholly inside the
    prefix, leaving the straddling record's head in the prompt.
    """
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        session = _chatty_session("cli:snap", turns=6, chars=10)
        _seed_journal(ws, session, stride=2)  # records [0,2) [2,4) [4,6)
        journal = turn_journal_for(SessionManager(ws), session.key)
        snap = MemoryConsolidator.snap_end_index_to_journal_records

        assert snap(journal, 3) == 2  # interior to [2,4): back to the [2,4) head
        assert snap(journal, 4) == 4  # already whole: unchanged
        assert snap(journal, snap(journal, 3)) == 2  # and idempotent
        assert snap(journal, 6) == 6  # the whole journal
        assert snap(journal, 1) == 0  # no whole record inside the prefix


@pytest.mark.asyncio
async def test_below_budget_does_not_consolidate():
    """A short session stays in context with no files or index movement."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = FakeConsolidationProvider(memory_update="M", history_entry="H")
        consolidator = _make_consolidator(ws, provider, context_window_tokens=65_536)
        session = _chatty_session("cli:local", turns=2, chars=10)

        await consolidator.maybe_compact_prompt_history(session)

        assert session.file_consolidated_end_index == 0
        assert session.prompt_history_start_index == 0
        assert provider.calls == []  # consolidation LLM never invoked


@pytest.mark.asyncio
async def test_context_only_compaction_does_not_write_file_memory():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = FakeConsolidationProvider(
            memory_update="unused",
            history_entry="unused",
        )
        consolidator = _make_consolidator(
            ws,
            provider,
            context_window_tokens=2_000,
            prompt_history_eviction_policy=(
                PromptHistoryEvictionPolicy.after_adapter_commit
            ),
        )
        session = Session(key="test:external-memory")
        for index in range(12):
            session.add_message("user", f"request {index} " + ("x " * 120))
            session.add_message("assistant", f"response {index} " + ("y " * 120))

        session.adapter_committed_end_index = len(session.messages)
        await consolidator.maybe_compact_prompt_history(session)

        assert session.file_consolidated_end_index == 0
        assert session.prompt_history_start_index > 0
        assert provider.calls == []
        assert not (ws / "memory" / "MEMORY.md").exists()
        assert not (ws / "memory" / "HISTORY.md").exists()


@pytest.mark.asyncio
async def test_prompt_compaction_cannot_cross_adapter_commit_index(monkeypatch):
    """Token pressure must fail rather than hide an uncommitted adapter turn."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = FakeConsolidationProvider(memory_update="unused", history_entry="unused")
        consolidator = _make_consolidator(
            ws,
            provider,
            context_window_tokens=2_000,
            prompt_history_eviction_policy=(
                PromptHistoryEvictionPolicy.after_adapter_commit
            ),
            require_committed_adapter_writes=True,
        )
        session = _chatty_session("test:blocked", turns=8, chars=120)
        monkeypatch.setattr(
            consolidator,
            "estimate_session_prompt_tokens",
            lambda _session: (1_000, "test"),
        )

        with pytest.raises(
            PromptHistoryEvictionBlockedError,
            match="INVALID_PROMPT_HISTORY_EVICTION_UNCOMMITTED",
        ):
            await consolidator.maybe_compact_prompt_history(session)

        assert session.prompt_history_start_index == 0


def test_experimental_budget_caps_native_budget_at_20000():
    """The experiment lowers the normal budget without exceeding smaller safe windows."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = FakeConsolidationProvider(memory_update="M", history_entry="H")

        normal = _make_consolidator(ws, provider, context_window_tokens=65_536)
        assert normal.safe_pa_prompt_limit_tokens() == 64_256
        assert normal.prompt_compaction_trigger_tokens() == 20_000

        small = _make_consolidator(ws, provider, context_window_tokens=2_000)
        assert small.safe_pa_prompt_limit_tokens() == 720
        assert small.prompt_compaction_trigger_tokens() == 720


@pytest.mark.asyncio
async def test_experimental_budget_triggers_at_20000(monkeypatch):
    """The normal-window experiment fires at 20k and targets the 10k half-budget."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = FakeConsolidationProvider(memory_update="M", history_entry="H")
        consolidator = _make_consolidator(ws, provider, context_window_tokens=65_536)
        session = _chatty_session("cli:threshold", turns=4, chars=10)
        _seed_journal(ws, session)
        estimates = iter([(20_000, "test"), (9_999, "test")])
        monkeypatch.setattr(
            consolidator,
            "estimate_session_prompt_tokens",
            lambda _session: next(estimates),
        )

        await consolidator.maybe_compact_prompt_history(session)

        assert provider.calls
        assert session.file_consolidated_end_index > 0
        assert (
            session.prompt_history_start_index
            == session.file_consolidated_end_index
        )


@pytest.mark.asyncio
async def test_pa_provider_estimates_and_memory_provider_consolidates():
    """Trigger accounting belongs to the PA; summary generation belongs to memory LM."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        pa_llm_provider = FakeEstimationProvider([1_000, 300])
        consolidation_llm_provider = FakeConsolidationProvider(
            memory_update="M",
            history_entry="H",
        )
        consolidator = MemoryConsolidator(
            workspace=ws,
            pa_llm_provider=pa_llm_provider,
            pa_model="pa-backbone",
            consolidation_llm_provider=consolidation_llm_provider,
            consolidation_model="fixed-memory-lm",
            session_manager=SessionManager(ws),
            pa_context_window_tokens=2_000,
            build_pa_prompt_messages=_build_messages,
            get_pa_tool_definitions=lambda: [
                {"type": "function", "function": {"name": "tool"}}
            ],
            prompt_history_eviction_policy=(
                PromptHistoryEvictionPolicy.after_file_consolidation
            ),
            pa_max_completion_tokens=256,
        )
        session = _chatty_session("cli:separate-models", turns=8, chars=120)
        _seed_journal(ws, session)

        await consolidator.maybe_compact_prompt_history(session)

        assert [call["model"] for call in pa_llm_provider.calls] == [
            "pa-backbone",
            "pa-backbone",
        ]
        assert [call["model"] for call in consolidation_llm_provider.calls] == [
            "fixed-memory-lm"
        ]
        assert session.file_consolidated_end_index > 0


@pytest.mark.asyncio
async def test_over_budget_writes_both_files_and_advances_indices():
    """Crossing the budget archives old turns into MEMORY.md + HISTORY.md."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = FakeConsolidationProvider(
            memory_update="User prefers concise answers.",
            history_entry="[2026-06-09] Discussed answer length.",
        )
        # Tiny window so a handful of turns crosses budget:
        # budget = 2000 - 256 - 1024 = 720 tokens, target = 360.
        consolidator = _make_consolidator(ws, provider, context_window_tokens=2_000)
        session = _chatty_session("cli:local", turns=8, chars=120)
        _seed_journal(ws, session)

        await consolidator.maybe_compact_prompt_history(session)

        assert 0 < session.file_consolidated_end_index < len(session.messages)
        assert (
            session.prompt_history_start_index
            == session.file_consolidated_end_index
        )
        assert provider.calls, "consolidation LLM should have been invoked"

        memory_md = (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8")
        history_md = (ws / "memory" / "HISTORY.md").read_text(encoding="utf-8")
        assert memory_md == "User prefers concise answers."  # overwritten
        assert "Discussed answer length." in history_md  # appended

        # Indices persist so a fresh load will not re-archive the same chunk.
        reloaded = SessionManager(ws).get("cli:local")
        assert (
            reloaded.file_consolidated_end_index
            == session.file_consolidated_end_index
        )
        assert (
            reloaded.prompt_history_start_index
            == session.prompt_history_start_index
        )


@pytest.mark.asyncio
async def test_auto_compaction_consolidates_the_pool_and_not_held_prose():
    """R2, restated for step 7: both file entry points read the same journal.

    The shape is no longer the two-message projection. §7 rules the PA's own
    IX call and the world traffic IN -- they are the PA's experience, and a
    wrong guess is remembered as a wrong guess -- while a held question is
    still OUT, because it reached nobody.
    """
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = FakeConsolidationProvider(memory_update="M", history_entry="H")
        consolidator = _make_consolidator(ws, provider, context_window_tokens=2_000)
        session = Session(key="cli:ix")
        for index in range(8):
            session.add_message("user", f"ask {index}: " + ("x " * 120))
            session.add_message(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": "IX_autonomy_level",
                            "arguments": '{"setting": "Suggest"}',
                        },
                    }
                ],
            )
            session.add_message(
                "tool",
                f"IX_autonomy_level accepted: Suggest ({index})",
                tool_call_id=f"call_{index}",
            )
            session.add_message(
                "assistant",
                f"HELD_SECRET_Q2 ({index})",
                _mempa_internal=True,
            )
            session.add_message("assistant", f"reply {index}: " + ("y " * 120))
        _seed_journal(ws, session, stride=5)
        owed = turn_journal_for(SessionManager(ws), session.key).records()

        consolidated_chunks: list[list[dict]] = []
        unprojected = consolidator.consolidate_messages_to_files

        async def _capture(messages, task_identity=None):
            consolidated_chunks.append(list(messages))
            return await unprojected(messages, task_identity=task_identity)

        consolidator.consolidate_messages_to_files = _capture

        await consolidator.maybe_compact_prompt_history(session)

        end = session.file_consolidated_end_index
        assert 0 < end <= len(session.messages)
        consolidated = [record for chunk in consolidated_chunks for record in chunk]
        # Byte-identical to what the pinned flush would send for the same
        # records: one source, one renderer.
        assert consolidated == consolidation_records(
            [record for record in owed if record.span[1] <= end]
        )
        assert {record["role"] for record in consolidated} == {"user", "assistant"}

        prompts = "\n".join(
            message["content"]
            for call in provider.calls
            for message in call["messages"]
        )
        # The PA's own selection and the world traffic are experience (§7).
        assert "IX_autonomy_level" in prompts
        assert "[ix_selection" in prompts or "[tool_call" in prompts
        # A held question reached nobody, so it is still not a memory.
        assert "HELD_SECRET_Q2" not in prompts
        assert "ask 0:" in prompts and "reply 0:" in prompts


@pytest.mark.asyncio
async def test_strict_failure_stops_without_raw_archive_or_index_advance(monkeypatch):
    """Formal runs must fail instead of silently accepting degraded memory."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = MissingToolCallProvider(memory_update="", history_entry="")
        consolidator = _make_consolidator(
            ws,
            provider,
            context_window_tokens=2_000,
            pa_max_completion_tokens=8192,
        )
        (ws / "memory" / "MEMORY.md").write_text("existing memory", encoding="utf-8")
        (ws / "memory" / "HISTORY.md").write_text("existing history\n", encoding="utf-8")
        session = _chatty_session("cli:strict", turns=8, chars=120)
        _seed_journal(ws, session)
        monkeypatch.setattr(
            consolidator,
            "estimate_session_prompt_tokens",
            lambda _session: (1_000, "test"),
        )

        with pytest.raises(MemoryConsolidationError, match="INVALID_CONSOLIDATION"):
            await consolidator.maybe_compact_prompt_history(session)

        assert len(provider.calls) == 3
        assert {call["max_tokens"] for call in provider.calls} == {8192}
        assert len({
            call["task_identity"].logical_task_key
            for call in provider.calls
        }) == 1
        assert session.file_consolidated_end_index == 0
        assert session.prompt_history_start_index == 0
        assert (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8") == "existing memory"
        assert (ws / "memory" / "HISTORY.md").read_text(encoding="utf-8") == "existing history\n"


@pytest.mark.asyncio
async def test_explicit_archive_fails_without_raw_history_fallback():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = MissingToolCallProvider(memory_update="", history_entry="")
        consolidator = _make_consolidator(ws, provider, context_window_tokens=2_000)
        messages = _chatty_session("cli:interactive", turns=4, chars=20).messages

        with pytest.raises(MemoryConsolidationError, match="explicit_archive"):
            await consolidator.archive_messages_to_files(messages)

        assert len(provider.calls) == 3
        assert not (ws / "memory" / "HISTORY.md").exists()


@pytest.mark.asyncio
async def test_distinct_consolidation_chunks_use_distinct_task_identities():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        provider = FakeConsolidationProvider(
            memory_update="memory",
            history_entry="history",
        )
        consolidator = _make_consolidator(
            ws,
            provider,
            context_window_tokens=2_000,
        )
        messages = [{"role": "user", "content": "remember"}]

        await consolidator.require_file_consolidation(
            messages,
            session_key="run_001_main",
            chunk_start_index=0,
            chunk_end_index=2,
        )
        await consolidator.require_file_consolidation(
            messages,
            session_key="run_001_main",
            chunk_start_index=2,
            chunk_end_index=4,
        )

        task_keys = [
            call["task_identity"].logical_task_key
            for call in provider.calls
        ]
        assert task_keys[0] != task_keys[1]


@pytest.mark.asyncio
async def test_probe_branch_skips_compaction_at_every_entry_point():
    """B16: a probe skips compaction outright, not just the LM write in it."""
    import inspect

    from pa.agent.loop import AgentLoop
    from pa.bus.queue import MessageBus

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        def _loop(*, skip: bool) -> AgentLoop:
            provider = FakeConsolidationProvider(
                memory_update="m", history_entry="h"
            )
            loop = AgentLoop(
                bus=MessageBus(),
                pa_llm_provider=provider,
                workspace=workspace,
                pa_model="fake-pa-model",
                consolidation_llm_provider=provider,
                consolidation_model="fake-memory-model",
                skip_consolidation=skip,
            )
            return loop

        session = _chatty_session("probe_session", turns=20, chars=4000)

        probe = _loop(skip=True)
        seen: list[Session] = []

        async def spy(compacted: Session, **kwargs) -> None:
            seen.append(compacted)

        probe.memory_consolidator.maybe_compact_prompt_history = spy
        await probe._compact_prompt_history_if_needed(session)
        await probe._compact_prompt_history_after_turn(session)
        assert seen == []
        # No consolidation LLM call was made either.
        assert probe.pa_llm_provider.calls == []

        # The default is unchanged: a mainline step still compacts.
        mainline = _loop(skip=False)
        mainline.memory_consolidator.maybe_compact_prompt_history = spy
        await mainline._compact_prompt_history_if_needed(session)
        await mainline._compact_prompt_history_after_turn(session)
        assert len(seen) == 2

    # All three compaction entry points in the loop funnel through the one
    # gated helper, so gating it gates all of them.
    source = inspect.getsource(AgentLoop)
    assert source.count("maybe_compact_prompt_history") == 1
    assert source.count("_compact_prompt_history_if_needed(session)") == 1
    assert source.count("_compact_prompt_history_after_turn(session)") == 2


class _StubAdapter:
    """Minimal adapter: only the pieces a cut-point wait actually touches."""

    def __init__(self, policy: PromptHistoryEvictionPolicy) -> None:
        self.backend_name = f"stub_{policy.value}"
        self.prompt_history_eviction_policy = policy

    async def setup_scope(self, scope):
        return None

    async def set_mode(self, mode):
        return None

    async def health(self):
        from memory.results import BackendHealth, ConsistencyModel, HealthStatus

        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=1.0,
            consistency_model=ConsistencyModel.immediate,
        )


@pytest.mark.asyncio
async def test_checkpoint_wait_drains_without_writing():
    """Step 14: a cut point only waits. It never writes and never advances.

    The two convergence actions are fixed and ordered — drain the background
    archives, then let the adapter settle — and neither of them may trigger a
    consolidation, move a cursor, or touch the memory files. Turns that were
    never compacted stay in the session and reach the snapshot as they are;
    putting them into the memory files is step 16a's job, and only on pinned
    steps. All three backends take this same path, with no branching.
    """
    import inspect

    from harness.memory_runtime import MemoryRuntime
    from harness.orchestrator import Orchestrator
    from pa.agent.loop import AgentLoop
    from pa.bus.queue import MessageBus

    policies = [
        PromptHistoryEvictionPolicy.after_file_consolidation,
        PromptHistoryEvictionPolicy.after_adapter_commit,
        PromptHistoryEvictionPolicy.without_persistence,
    ]
    observed: list[tuple[str, list[str], tuple[int, int, int], int]] = []

    for policy in policies:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            provider = FakeConsolidationProvider(
                memory_update="must not be written",
                history_entry="must not be appended",
            )
            adapter = _StubAdapter(policy)
            loop = AgentLoop(
                bus=MessageBus(),
                pa_llm_provider=provider,
                workspace=workspace,
                pa_model="fake-pa-model",
                consolidation_llm_provider=provider,
                consolidation_model="fake-memory-model",
                memory_adapter=adapter,
            )

            # A session well past the compaction threshold that was never
            # compacted: exactly the state a cut point must leave alone.
            session = _chatty_session("run_001_main", turns=20, chars=4000)
            session.adapter_committed_end_index = len(session.messages)
            cursors_before = (
                session.file_consolidated_end_index,
                session.prompt_history_start_index,
                session.adapter_committed_end_index,
            )
            messages_before = [dict(m) for m in session.messages]

            calls: list[str] = []
            archived: list[list] = []

            async def _archive(messages, _archived=archived):
                _archived.append(list(messages))
                return True

            loop.memory_consolidator.archive_messages_to_files = _archive

            background_ran: list[str] = []

            async def _pending_archive(_ran=background_ran):
                _ran.append("done")

            loop._schedule_background(_pending_archive())
            assert loop._background_tasks

            runtime = MemoryRuntime(
                adapter, run_id="run_001", persona_id="user_a"
            )
            inner_ready = runtime.await_checkpoint_ready

            async def _ready(timeout_s, *, session=None, _inner=inner_ready):
                calls.append("await_checkpoint_ready")
                return await _inner(timeout_s, session=session)

            runtime.await_checkpoint_ready = _ready

            original_drain = loop.drain_background_tasks

            async def _drain(_inner=original_drain):
                calls.append("drain_background_tasks")
                await _inner()

            loop.drain_background_tasks = _drain

            # The cut-point wait, exactly as the orchestrator composes it.
            await loop.drain_background_tasks()
            await runtime.await_checkpoint_ready(300.0, session=session)

            history = workspace / "memory" / "HISTORY.md"
            memory_file = workspace / "memory" / "MEMORY.md"

            assert background_ran == ["done"], policy
            assert loop._background_tasks == [], policy
            # Only the two convergence actions happened, in that order.
            assert calls == [
                "drain_background_tasks",
                "await_checkpoint_ready",
            ], policy
            # No consolidation was triggered, by any route.
            assert archived == [], policy
            assert provider.calls == [], policy
            # No cursor moved.
            assert (
                session.file_consolidated_end_index,
                session.prompt_history_start_index,
                session.adapter_committed_end_index,
            ) == cursors_before, policy
            # The never-compacted turns are still in the session verbatim.
            assert session.messages == messages_before, policy
            # The memory files were not written or appended to.
            assert not history.exists(), policy
            assert not memory_file.exists(), policy

            observed.append(
                (
                    policy.value,
                    calls,
                    cursors_before,
                    len(session.messages),
                )
            )

    # Same path for all three backends: identical call sequence, identical
    # cursor and message outcome. Only the backend label differs.
    assert len({(c[1][0], c[1][1], c[2], c[3]) for c in observed}) == 1
    assert [entry[0] for entry in observed] == [p.value for p in policies]

    # The orchestrator's cut-point wait is those two calls and nothing else:
    # no consolidation, no cursor assignment, no eviction-policy branching.
    wait_source = inspect.getsource(Orchestrator._await_checkpoint_ready)
    assert "drain_background_tasks()" in wait_source
    assert "await_checkpoint_ready(" in wait_source
    assert "archive_messages_to_files" not in wait_source
    assert "prompt_history_eviction_policy" not in wait_source
    assert "_end_index =" not in wait_source
    assert "_start_index =" not in wait_source


class FakeTurnProvider(LLMProvider):
    """One PA reply, no tools — enough to make the loop persist a turn."""

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        return LLMResponse(content="Checked the citation notes.")

    def get_default_model(self) -> str:
        return "fake-pa-model"


class PromptEchoProvider(FakeConsolidationProvider):
    """Writes the consolidation prompt back out as the memory update.

    MEMORY.md then holds exactly the text the consolidation LM read, so every
    timestamp that reached the prompt is one the assertion can see.
    """

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self._memory_update = messages[-1]["content"]
        return await super().chat(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
        )


@pytest.mark.asyncio
async def test_benchmark_clock_is_the_only_date_in_the_memory_files(tmp_path):
    """The turn's own clock, not the machine's, reaches MEMORY.md.

    A benchmark session runs at its scripted timestamp, but the turn used to be
    persisted with `datetime.now()`, and `_format_messages` puts that stamp in
    front of every line the consolidation LM reads. In regress13 the LM copied
    it: October sessions whose MEMORY.md said 2026-09-07, the day the batch
    happened to run. Driving one turn at a fixed benchmark timestamp and
    echoing the prompt into MEMORY.md, the only date in the file must be the
    benchmark one.
    """
    from pa.agent.loop import AgentLoop
    from pa.bus.queue import MessageBus

    benchmark_timestamp = datetime(2026, 10, 1, 9, 5, tzinfo=timezone.utc)
    session_key = "regress__acc_001_main"
    SessionManager(tmp_path).create(session_key)

    loop = AgentLoop(
        bus=MessageBus(),
        pa_llm_provider=FakeTurnProvider(),
        workspace=tmp_path,
        pa_model="fake-pa-model",
    )
    await loop.process_direct_message(
        "Check the Braithwaite entry in the citation notes.",
        session_key=session_key,
        timestamp=benchmark_timestamp,
    )

    session = SessionManager(tmp_path).get(session_key)
    assert session.messages
    assert [message["timestamp"] for message in session.messages] == [
        benchmark_timestamp.isoformat()
    ] * len(session.messages)

    store = MarkdownMemoryStore(tmp_path, log_scope="pa_file_memory")
    journal = turn_journal_for(SessionManager(tmp_path), session_key)
    await store.consolidate(
        consolidation_records(journal.records()),
        PromptEchoProvider(memory_update="", history_entry="[2026-10-01 09:05] x"),
        "fake-memory-model",
    )

    memory = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "2026-10-01T09:05" in memory
    # Not "today's date is absent": *no* date but the benchmark one is there,
    # which stays true whatever day the suite runs on.
    assert set(re.findall(r"\d{4}-\d{2}-\d{2}", memory)) == {"2026-10-01"}


@pytest.mark.asyncio
@pytest.mark.parametrize("archived_turns", [0, 2])
async def test_fully_committed_current_turn_waits_for_next_boundary(tmp_path, monkeypatch, archived_turns):
    provider = FakeConsolidationProvider(memory_update="unused", history_entry="unused")
    consolidator = _make_consolidator(tmp_path, provider, 65_536,
        prompt_history_eviction_policy=PromptHistoryEvictionPolicy.after_adapter_commit,
        require_committed_adapter_writes=True)
    session = Session(key="test:committed-current")
    for i in range(archived_turns + 1):
        session.add_message("user", f"request {i}")
        session.add_message("assistant", f"response {i}")
    session.prompt_history_start_index = archived_turns * 2
    session.adapter_committed_end_index = len(session.messages)
    monkeypatch.setattr(consolidator, "estimate_session_prompt_tokens", lambda s: (25_000, "test"))
    await consolidator.maybe_compact_prompt_history(session)
    assert session.prompt_history_start_index == archived_turns * 2
    assert session.adapter_committed_end_index == len(session.messages)
    assert provider.calls == []
    # A subsequent committed turn creates an eligible boundary again.
    session.add_message("user", "next request")
    session.add_message("assistant", "next answer")
    session.adapter_committed_end_index = len(session.messages)
    monkeypatch.setattr(consolidator, "estimate_session_prompt_tokens",
        lambda s: (25_000 if s.prompt_history_start_index == archived_turns * 2 else 5_000, "test"))
    await consolidator.maybe_compact_prompt_history(session)
    assert session.prompt_history_start_index == (archived_turns + 1) * 2
    assert provider.calls == []


@pytest.mark.asyncio
async def test_committed_current_turn_still_respects_model_capacity(tmp_path, monkeypatch):
    provider = FakeConsolidationProvider(memory_update="unused", history_entry="unused")
    consolidator = _make_consolidator(tmp_path, provider, 65_536,
        prompt_history_eviction_policy=PromptHistoryEvictionPolicy.after_adapter_commit,
        require_committed_adapter_writes=True)
    session = _chatty_session("test:oversized-current", turns=1, chars=100)
    session.adapter_committed_end_index = len(session.messages)
    monkeypatch.setattr(consolidator, "estimate_session_prompt_tokens",
        lambda s: (consolidator.safe_pa_prompt_limit_tokens(), "test"))
    with pytest.raises(RuntimeError, match="PROMPT_HISTORY_CAPACITY"):
        await consolidator.maybe_compact_prompt_history(session)
    assert session.prompt_history_start_index == 0


# ---------------------------------------------------------------------------
# §9 step 5 · `[no_reply]` is the one item whose only content is delimiters
# ---------------------------------------------------------------------------

def test_no_reply_block_survives_format_messages() -> None:
    """A suppressed outreach must reach the consolidation LM, not vanish.

    `_format_messages` skips any message whose content is falsy
    (`pa/agent/memory.py:123-124`), and `[no_reply]`'s body is empty by design
    (§3.3). The always-emit-delimiters rule is what saves it: the content is
    the delimiter pair, never `""`. Without that rule the file condition would
    drop a scored interaction choice — `proactive_outreach` is scored on
    exactly the decision to stay silent — silently.
    """
    pool = build_turn_pool(
        [{"role": "assistant", "content": "Internal completion text."}],
        environment_event={
            "source_id": "deadline",
            "action_id": "pickup_deadline",
            "body": "[Environment event; no user request is currently active]",
        },
        delivered="",
    )
    records = [
        {**message, "timestamp": "2026-10-01T09:00:00+00:00"}
        for message in pool.messages
    ]

    rendered = MarkdownMemoryStore._format_messages(records)

    assert "[no_reply decision=suppress]" in rendered
    assert rendered.count("ASSISTANT") == len(pool.messages)


def test_no_reply_content_is_never_the_empty_string_a_backend_refuses() -> None:
    """The same rule, against the other reader that drops empty content.

    `_validated_raw_dialogue_messages` raises `Mem0ResponseError` on an
    empty-content message (`memory/backends/mem0_client.py:153-157`), so a
    renderer that suppressed an empty block would not merely lose the item —
    it would fail the whole `simple_rag` add for the turn.
    """
    pool = build_turn_pool([], delivered="")

    assert [message["content"] for message in pool.messages] == [
        "[no_reply decision=suppress]\n[/no_reply]"
    ]
    assert all(
        isinstance(message["content"], str) and message["content"].strip()
        for message in pool.messages
    )
