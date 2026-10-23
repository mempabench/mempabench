"""Resume of the turn pool across a crash.

These are acceptance tests written before the code they describe. That
code has two parts: `memory/turn_journal.py` -- the
"append-only turn journal, one record per pool write" -- and the resume entry
point that reads it and re-writes the pending pools. The **module has landed**
and these tests are written to its API; the **entry point has not**, and the
module says so itself ("Wiring this module is NOT done here"), so the file is
still skipped by name at import. Once the entry point lands the tests run for
real and their own assertions, never an import, decide.

**Two things a reader should check before trusting that sentence**, because a
skipped file asserts nothing:

- the journal records these tests read back are written by the **production**
  commit window, not by the test body. Every turn is driven through
  `AgentLoop.process_environment_event`, whose branch calls
  `event_writer.record_turn_journal` twice (`pa/agent/loop.py:1564`, `:1584`).
  The tests bind that seam to the journal
  module and nothing else: production decides the `turn_id`, the span, the pool
  and the ordering against `sessions.save`, which is what makes the rule
  ("the record is durable BEFORE the turn's first `sessions.save`") assertable
  here at all. The crashes are likewise production-side -- a monkeypatched
  `AgentLoop` method raises -- so the state a resume finds on disk is the state
  the real code left;
- that the assertions below are satisfiable, and non-vacuous, is checked by
  `test_resume_pool_stub_proof.py`, which runs this whole suite twice: green
  against the **real landed journal** plus a throwaway resume function, red on
  its own assertions against a journal that keeps a span and a hash. That proof
  runs today, while this file is skipped.

The journal calls below are the landed `memory.turn_journal` API --
`JournalRecord.for_pool(...)` into `append_pending(record)`,
`mark_written(turn_id, receipt)`, `pending_records()`, `record.pool`. The one
name still unfixed is the resume entry point, `replay_pending_turns`, which is
this suite's proposal for a function that does not exist yet; whichever commit
lands it either takes that name or updates the two call sites here. What is
*not* negotiable is the behaviour each test asserts, which each docstring
states. The import is at module scope on purpose: a resume entry point landed
under a different name fails loudly here instead of turning these tests into
silent xfails.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

import memory.event_writer as event_writer
from memory.adapter import PromptHistoryEvictionPolicy
from memory.conditions.nanobot_file_memory import NanobotFileMemoryAdapter
from memory.results import RetrieveResult, WriteReceipt, WriteStatus
from memory.scope import MemoryScope
from pa.agent.loop import AgentLoop
from pa.bus.queue import MessageBus
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from pa.session.manager import SessionManager

_journal = pytest.importorskip(
    "memory.turn_journal",
    reason=(
        "the step that adds `memory/turn_journal.py` has not landed: the "
        "append-only turn journal that persists the rendered pool per turn_id, "
        "and the resume entry point that replays its pending records. These "
        "tests are that step's acceptance criteria and cannot run before it. "
        "While it is absent, `test_resume_pool_stub_proof.py` is what keeps "
        "them honest -- it runs this suite against two throwaway journals and "
        "shows it passing against one and failing against the other."
    ),
)

if not hasattr(_journal, "replay_pending_turns"):
    # The journal MODULE has landed; the resume ENTRY POINT has not. The module
    # says so itself -- "Wiring this module is NOT done here" -- and leaves step
    # 9 the job of reading `resume_pending()` and re-writing those pools:
    # `TurnJournal` exposes `resume_pending() -> ResumePlan` and nothing that
    # calls an adapter or logs a rewritten turn id.
    #
    # These tests deliberately do NOT hand-roll that loop in the test body. An
    # earlier round of this file did, and was refuted for it: a suite that
    # writes the replay itself goes green on a journal module alone, with zero
    # resume logic in production, and satisfies the log check
    # ("the resume log lists rewritten turn ids") with its own logging. So the
    # suite waits for the production entry point, and
    # `test_resume_pool_stub_proof.py` keeps it honest meanwhile -- it now runs
    # this suite against the REAL landed journal plus a throwaway resume
    # function, and against an inert one that loses the pool.
    pytest.skip(
        "`memory.turn_journal` has landed but the resume entry point has not: "
        "the module exposes `TurnJournal.resume_pending()` and no function that "
        "re-writes those pools to the adapter and logs the turn ids it rewrote "
        "(at-least-once). These tests drive that entry point rather than "
        "hand-rolling the replay, so they run when it lands.",
        allow_module_level=True,
    )

from memory.turn_journal import (  # noqa: E402
    JournalRecord,
    TurnJournal,
    replay_pending_turns,
)


SESSION_KEY = "benchmark:main"
TIMESTAMP = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)
SCOPE = MemoryScope(run_id="run_001", persona_id="user_a", agent_id="nanobot")

#: The three things `session.messages` cannot give a resume back:
#: intermediate assistant prose (blanked at `pa/agent/ix/collector.py:1403`),
#: the final iteration's reasoning (dropped at `:1433`), and the PA's own IX
#: call, which lives only in the raw provider slice. A journal that persisted a
#: span, a hash, or the lossy slice carries none of them.
INTERMEDIATE_PROSE = "The store closes at six, so this is worth one line."
FINAL_REASONING = "One line is enough; she has said she hates being nudged twice."


class RecordingAdapter:
    """Records every `record_event` it receives, so a replay is countable."""

    backend_name = "resume_recording_fake"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.after_adapter_commit

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def setup_scope(self, scope) -> None:
        return None

    async def set_mode(self, mode):
        return None

    async def retrieve(self, request):
        return RetrieveResult(raw=[])

    def format_context(self, result) -> str:
        return ""

    async def record_event(self, event) -> WriteReceipt:
        self.events.append(event)
        return WriteReceipt(status=WriteStatus.committed, native_ids=[event.event_id])


class _ScriptProvider(LLMProvider):
    """The same two responses for every turn, so two runs render alike."""

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        if any(message.get("role") == "tool" for message in messages):
            return LLMResponse(
                content="Internal completion text.",
                reasoning_content=FINAL_REASONING,
            )
        return LLMResponse(
            content=INTERMEDIATE_PROSE,
            reasoning_content="She asked to be told about pickups, once.",
            tool_calls=[ToolCallRequest(
                id="ix_proactive",
                name="IX_proactive_outreach",
                arguments={
                    "setting": "High",
                    "evidence": "Retrieved preference evidence.",
                    "application": "send this contact.",
                    "outreach_action": "send",
                    "outreach_message": "The comic store closes at 6 PM today.",
                },
            )],
        )


def _loop(workspace: Path, adapter: RecordingAdapter) -> AgentLoop:
    loop = AgentLoop(
        bus=MessageBus(),
        pa_llm_provider=_ScriptProvider(),
        workspace=workspace,
        pa_model="test-model",
        memory_adapter=adapter,
        require_committed_adapter_writes=False,
    )
    loop.sessions.create(SESSION_KEY)
    return loop


def _bind_journal(
    monkeypatch: pytest.MonkeyPatch,
    journal_path: Path,
    trace: list[tuple[str, ...]],
) -> None:
    """Bind the production journal seam to the journal module, and trace it.

    `event_writer.record_turn_journal` is the one named place a pool write
    announces its record (`memory/event_writer.py:235`);
    the ACC line left it a documented no-op so it would not guess this module's
    API. Binding it here is the whole of the test's involvement: every argument
    -- `turn_id`, `span`, `timestamp`, `pool`, `state` -- and the moment of the
    call are production's. `SessionManager.save` is traced beside it so the
    required ordering can be asserted rather than assumed.
    """
    def _record(
        *, turn_id, session_id, span, timestamp, pool, state,
        journal=None, receipt=None,
    ):
        # `journal` is production's own, resolved beside the session file; the
        # bridge substitutes this test's path so the trace and the file are in
        # one place. Everything else is production's, `receipt` included.
        bound = TurnJournal(journal_path)
        if state == "pending":
            bound.append_pending(JournalRecord.for_pool(
                turn_id=turn_id,
                span=span,
                pool=pool,
                timestamp=timestamp.isoformat(),
            ))
        else:
            bound.mark_written(
                turn_id,
                receipt or WriteReceipt(status=WriteStatus.committed),
            )
        trace.append(("journal", turn_id, state))

    original_save = SessionManager.save

    def _save(self, session):
        trace.append(("save", session.key))
        return original_save(self, session)

    monkeypatch.setattr(event_writer, "record_turn_journal", _record)
    monkeypatch.setattr(SessionManager, "save", _save)


async def _environment_turn(loop: AgentLoop, turn_id: str, *, index: int):
    """One environment turn, driven through the production entry point."""
    return await loop.process_environment_event(
        trigger_id=f"pickup_deadline_{index:02d}",
        source_type="deadline",
        prior_thread="The comic order is ready for pickup.",
        detail=f"The store closes in {index + 1} hours.",
        user_impact="User A may want time to collect the order today.",
        session_key=SESSION_KEY,
        metadata={"run_id": "run_001", "persona_id": "user_a", "turn_id": turn_id},
        timestamp=TIMESTAMP + timedelta(minutes=index),
    )


def _turn_slice(trace: list[tuple[str, ...]], marker: str) -> list[tuple[str, ...]]:
    """The trace entries a single turn produced, by its own marker."""
    start = trace.index(("turn", marker))
    end = next(
        (i for i in range(start + 1, len(trace)) if trace[i][0] == "turn"),
        len(trace),
    )
    return trace[start + 1:end]


def _assert_durable_before_first_save(
    trace: list[tuple[str, ...]], turn_id: str
) -> None:
    """The record is durable BEFORE the turn's first `sessions.save`."""
    entries = _turn_slice(trace, turn_id)
    pending = entries.index(("journal", turn_id, "pending"))
    saves = [i for i, entry in enumerate(entries) if entry[0] == "save"]
    assert saves, f"turn {turn_id} saved the session at least once"
    assert pending < saves[0], (
        f"turn {turn_id}: the journal record must be durable before the turn's "
        f"first sessions.save; trace={entries}"
    )


@pytest.mark.asyncio
async def test_rederived_pool_equals_the_uninterrupted_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance check: "the re-derived pool for turn N+1 is byte-equal to
    the pool the uninterrupted run produced for it -- which is the assertion
    that fails today, on the losses in (b)."

    Two runs, two provenances. The uninterrupted run is a complete turn, and
    the pool it produced is the one its adapter received. The interrupted run
    is a second, independently driven turn in its own workspace that dies
    before the adapter call; all it leaves is the journal record, and the
    resume re-reads that. Neither value is derived from the other, which is
    what makes the equality mean anything.

    The artefact the resume re-reads is the journal record, not
    `session.messages`: "persist the pool itself. Write the rendered pool
    messages (or the `items` list) beside the turn, keyed by `turn_id`,
    so a resume re-reads a rendered artefact instead of re-deriving one."
    """
    # Run 1, uninterrupted: the pool its adapter received is the reference.
    uninterrupted_adapter = RecordingAdapter()
    uninterrupted = _loop(tmp_path / "uninterrupted", uninterrupted_adapter)
    await _environment_turn(uninterrupted, "turn_01", index=0)
    assert [event.turn_id for event in uninterrupted_adapter.events] == ["turn_01"]
    reference = uninterrupted_adapter.events[0]

    # Run 2, interrupted after the journal record and before the adapter call.
    # Nothing of run 1 is in scope: its own workspace, its own loop, its own
    # adapter, and the only thing that survives the crash is the journal file.
    trace: list[tuple[str, ...]] = []
    workspace = tmp_path / "interrupted"
    journal_path = workspace / "turn_journal.jsonl"
    _bind_journal(monkeypatch, journal_path, trace)
    interrupted_adapter = RecordingAdapter()
    interrupted = _loop(workspace, interrupted_adapter)

    async def _die(self, metadata, pool, *, timestamp, session_key, turn_id):
        raise RuntimeError("crash after the journal record, before the adapter call")

    monkeypatch.setattr(AgentLoop, "_write_turn_to_memory_adapter", _die)
    trace.append(("turn", "turn_01"))
    with pytest.raises(RuntimeError):
        await _environment_turn(interrupted, "turn_01", index=0)
    assert interrupted_adapter.events == []
    _assert_durable_before_first_save(trace, "turn_01")

    # The resume opens the journal again -- a second `TurnJournal` over the
    # same path, nothing carried in memory -- and reads the rendered artefact
    # back.
    pending = TurnJournal(journal_path).pending_records()
    assert [record.turn_id for record in pending] == ["turn_01"]
    rederived = pending[0].pool

    assert rederived.messages == reference.messages
    assert rederived.items == list(reference.items)

    # The three things `session.messages` loses, named one by
    # one, so a journal that persisted only a span, a hash or the lossy slice
    # cannot pass this.
    text = "\n".join(str(message["content"]) for message in rederived.messages)
    assert INTERMEDIATE_PROSE in text  # intermediate prose
    assert "ix_selection" in text  # the PA's own IX choice
    assert FINAL_REASONING in text  # final-iteration reasoning


@pytest.mark.asyncio
async def test_resume_writes_pools_after_the_committed_turn_and_none_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance checks: "assert (i) a pool receipt exists for every
    turn N+1 and after, (ii) no second receipt exists for any turn <= N".

    The setup, with N = 1: three turns driven through the production
    commit window against a fake adapter that records every `record_event`;
    the adapter stops being reachable after turn N, so turns N+1 and N+2 leave
    durable `pending` records and no receipt; the process state is dropped
    (a fresh `SessionManager` re-reads the run directory) and the resume runs.

    This is the first crash point -- "after the journal record,
    before the adapter call". The rule: "the authority is the journal
    record's state, not a message index".
    """
    trace: list[tuple[str, ...]] = []
    workspace = tmp_path / "pa_workspace"
    journal_path = workspace / "turn_journal.jsonl"
    _bind_journal(monkeypatch, journal_path, trace)
    adapter = RecordingAdapter()
    loop = _loop(workspace, adapter)

    # Turn N -- written before the crash: receipt, journal state and cursor.
    trace.append(("turn", "turn_01"))
    await _environment_turn(loop, "turn_01", index=0)
    assert [event.turn_id for event in adapter.events] == ["turn_01"]
    _assert_durable_before_first_save(trace, "turn_01")

    # Turns N+1 and N+2 -- the journal record is durable and the session is
    # saved, then the process dies before the adapter call.
    original_write = AgentLoop._write_turn_to_memory_adapter

    async def _die(self, metadata, pool, *, timestamp, session_key, turn_id):
        raise RuntimeError("crash after the journal record, before the adapter call")

    monkeypatch.setattr(AgentLoop, "_write_turn_to_memory_adapter", _die)
    for index, turn_id in ((1, "turn_02"), (2, "turn_03")):
        trace.append(("turn", turn_id))
        with pytest.raises(RuntimeError):
            await _environment_turn(loop, turn_id, index=index)
        _assert_durable_before_first_save(trace, turn_id)
    assert [event.turn_id for event in adapter.events] == ["turn_01"]
    monkeypatch.setattr(AgentLoop, "_write_turn_to_memory_adapter", original_write)

    # The process state is dropped; the run directory is what is left.
    resumed_session = SessionManager(workspace).get(SESSION_KEY)
    rewritten = await replay_pending_turns(
        journal=TurnJournal(journal_path),
        adapter=adapter,
        scope=SCOPE,
        session=resumed_session,
        require_committed=True,
    )

    # (i) a pool receipt exists for every turn after N; (ii) turn N, whose
    # record is `written`, is not written a second time.
    assert rewritten == ["turn_02", "turn_03"]
    assert [event.turn_id for event in adapter.events] == [
        "turn_01", "turn_02", "turn_03",
    ]
    assert [event.turn_id for event in adapter.events].count("turn_01") == 1
    assert TurnJournal(journal_path).pending_records() == []

    # The rewritten pools are the ones the interrupted turns rendered, not a
    # re-derivation from the lossy slice.
    for event in adapter.events[1:]:
        text = "\n".join(str(message["content"]) for message in event.messages)
        assert INTERMEDIATE_PROSE in text
        assert FINAL_REASONING in text

    # The cursor catches up to the turn the resume wrote, and neither file
    # cursor moved on the way.
    assert resumed_session.adapter_committed_end_index == len(
        resumed_session.messages
    )
    assert resumed_session.file_consolidated_end_index == 0
    assert resumed_session.prompt_history_start_index == 0


@pytest.mark.asyncio
async def test_crash_between_adapter_ack_and_cursor_save_replays_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At-least-once, declared.

    "a crash after the adapter returns and before the cursor is saved leaves a
    written pool with a stale cursor, and the resume writes it again ... every
    resume logs the turn ids it rewrote". That is the second
    crash point, and it is production that crashes there: the turn's
    `_record_adapter_write_commit` raises, which is exactly between the
    adapter's receipt (`pa/agent/loop.py:1575`) and the cursor and journal
    state that follow it (`:1580-1592`).

    The replay is driven through the production entry point, so the log line
    the row asks for ("the resume log lists rewritten turn ids") is the resume
    path's own, not the test's. The authority for the skip is the
    journal record's state, never a message index, and neither
    `file_consolidated_end_index` nor `prompt_history_start_index` may move on
    the failed turn.
    """
    trace: list[tuple[str, ...]] = []
    workspace = tmp_path / "pa_workspace"
    journal_path = workspace / "turn_journal.jsonl"
    _bind_journal(monkeypatch, journal_path, trace)
    adapter = RecordingAdapter()
    loop = _loop(workspace, adapter)

    def _die(self, session, receipt, *, adapter_committed_end_index):
        raise RuntimeError("crash after the adapter ack, before the cursor save")

    monkeypatch.setattr(AgentLoop, "_record_adapter_write_commit", _die)
    trace.append(("turn", "turn_01"))
    with pytest.raises(RuntimeError):
        await _environment_turn(loop, "turn_01", index=0)

    # The adapter acknowledged, and neither the cursor nor the journal state
    # reached disk behind it.
    assert [event.turn_id for event in adapter.events] == ["turn_01"]
    assert ("journal", "turn_01", "written") not in trace
    _assert_durable_before_first_save(trace, "turn_01")

    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO")
    try:
        resumed_session = SessionManager(workspace).get(SESSION_KEY)
        rewritten = await replay_pending_turns(
            journal=TurnJournal(journal_path),
            adapter=adapter,
            scope=SCOPE,
            session=resumed_session,
            require_committed=True,
        )
    finally:
        logger.remove(sink)

    # The turn was written twice -- at-least-once, declared; no dedupe and no
    # exactly-once claim is made anywhere here -- and the resume path logged
    # the id it rewrote.
    assert rewritten == ["turn_01"]
    assert [event.turn_id for event in adapter.events] == ["turn_01", "turn_01"]
    assert any("turn_01" in line for line in lines)

    # Nothing else moved: the pool bytes are the same on both writes, the
    # journal holds no pending record, and neither file cursor advanced on the
    # failed turn.
    assert adapter.events[0].messages == adapter.events[1].messages
    assert TurnJournal(journal_path).pending_records() == []
    assert resumed_session.file_consolidated_end_index == 0
    assert resumed_session.prompt_history_start_index == 0


@pytest.mark.asyncio
async def test_file_condition_resume_replays_nothing_and_mutates_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file condition's resume is a no-op: nothing replayed, nothing marked.

    `nanobot_file_memory` is the one condition whose adapter write is not the
    write: `record_event` returns `deferred_to_file_consolidation` and the
    backend's `add` is a documented no-op (`memory/file/backend.py:29-35`) --
    the real write happens later, at consolidation. So a `pending` record on
    this condition is not a lost write, and replaying it would put nothing in
    the memory files while still stamping the record `written` with a receipt
    for a write that never happened. Its completion track is the journal's
    second one, `file_consolidated`, which the consolidation entry points own.

    The discriminator is the adapter's declared
    `prompt_history_eviction_policy`, asserted below so a change to it breaks
    here rather than silently turning the replay back on.
    """
    trace: list[tuple[str, ...]] = []
    workspace = tmp_path / "pa_workspace"
    journal_path = workspace / "turn_journal.jsonl"
    _bind_journal(monkeypatch, journal_path, trace)

    # The adapter a file-condition run actually holds: `scripts/run_orchestrator
    # .py:1785-1788` builds it over `<workspace>/memory`, so `memory_runtime
    # .adapter` is this object and never `None`.
    memory_dir = workspace / "memory"
    adapter = NanobotFileMemoryAdapter(memory_dir)
    assert (
        adapter.prompt_history_eviction_policy
        is PromptHistoryEvictionPolicy.after_file_consolidation
    )
    memory_file = memory_dir / "MEMORY.md"
    history_file = memory_dir / "HISTORY.md"
    memory_file.write_text("# Memory\n- She hates being nudged twice.\n", encoding="utf-8")
    history_file.write_text("[2026-09-13] USER: Anything on the comic order?\n", encoding="utf-8")

    loop = _loop(workspace, adapter)

    async def _die(self, metadata, pool, *, timestamp, session_key, turn_id):
        raise RuntimeError("crash after the journal record, before the adapter call")

    monkeypatch.setattr(AgentLoop, "_write_turn_to_memory_adapter", _die)
    for index, turn_id in ((0, "turn_01"), (1, "turn_02")):
        trace.append(("turn", turn_id))
        with pytest.raises(RuntimeError):
            await _environment_turn(loop, turn_id, index=index)
        _assert_durable_before_first_save(trace, turn_id)

    journal_bytes = journal_path.read_bytes()
    memory_bytes = memory_file.read_bytes()
    history_bytes = history_file.read_bytes()
    assert [
        record.turn_id for record in TurnJournal(journal_path).pending_records()
    ] == ["turn_01", "turn_02"]

    resumed_session = SessionManager(workspace).get(SESSION_KEY)
    rewritten = await replay_pending_turns(
        journal=TurnJournal(journal_path),
        adapter=adapter,
        scope=SCOPE,
        session=resumed_session,
        require_committed=True,
    )

    # Replays nothing, marks nothing, mutates no file -- the journal included,
    # whose `written` mark would otherwise be appended to it.
    assert rewritten == []
    assert [
        record.turn_id for record in TurnJournal(journal_path).pending_records()
    ] == ["turn_01", "turn_02"]
    assert journal_path.read_bytes() == journal_bytes
    assert memory_file.read_bytes() == memory_bytes
    assert history_file.read_bytes() == history_bytes
    assert resumed_session.file_consolidated_end_index == 0
    assert resumed_session.prompt_history_start_index == 0


@pytest.mark.asyncio
async def test_a_fresh_run_has_no_journal_and_replays_without_reading_the_session(
    tmp_path: Path,
) -> None:
    """A run with no session file yet must start, not die in the resume.

    A live run never reached its first API call: `Orchestrator.run`
    calls `_replay_pending_turns` at start-up unconditionally, and that read
    `handle.checkpoint_session()` unguarded. On a fresh run there is no session
    file, and `SessionManager.get` raises `SessionNotFoundError` rather than
    inventing one (`pa/session/manager.py:197-205`), so every fresh run aborted
    with

        SessionNotFoundError: Session not found: key='live3_pool_acc_001_main'

    No session means no turn, so there is nothing to replay: the resume returns
    instead of dying. `checkpoint_session` here raises the way a fresh run's
    does, and the handle's other methods fail the test if the resume goes on to
    replay or save anything.
    """
    from pa.session.manager import SessionNotFoundError

    from harness.orchestrator import Orchestrator

    class _FreshRunHandle:
        def turn_journal(self) -> TurnJournal:
            raise AssertionError("a fresh run has no journal to read")

        def checkpoint_session(self) -> Any:
            raise SessionNotFoundError("no session file on a fresh run")

        def save_session(self, session: Any) -> None:
            raise AssertionError("a fresh run saves no session in the resume")

    class _Runtime:
        adapter = RecordingAdapter()
        scope = SCOPE

    class _Stub:
        pa_checkpoint = _FreshRunHandle()
        memory_runtime = _Runtime()

        def progress(self, message: str) -> None:
            raise AssertionError(f"a fresh run rewrites no pool: {message!r}")

    await Orchestrator._replay_pending_turns(_Stub())

    assert _Runtime.adapter.events == []
