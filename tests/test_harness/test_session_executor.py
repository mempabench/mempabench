import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from harness.run_plan import RunStep
from harness.session_executor import SessionExecutor, failed_step_artifacts
from harness.state_runtime import StageStepRuntime
from harness.transcript import (
    READ_CLASS_TOOL_SUFFIXES,
    STATE_TOOL_NAME_PREFIX,
    declared_plan_arguments,
    declared_plan_tools,
    tool_outside_plan_events,
)
from pa.agent.ix.render_types import (
    IX_CONTRACT_VERSION,
    IX_RENDER_SCHEMA_VERSION,
)
from simulator.output import SessionInvalidError
from state_server.store import SESSION_CLOSING_MARKER


class FakeLoop:
    def __init__(self, writer):
        self.writer = writer

    async def run_session(self):
        events = await self.run_session_events()
        for event in events:
            self.writer.log_event(event)
        return events

    async def run_session_events(self):
        return [
            {
                "t": 0,
                "kind": "session_start",
                "persona": "user_a",
                "session_id": "acc_001",
                "scenario": "smoke",
                "context": "work",
                "beat_names": ["b1"],
                "model": "pa-model",
                "pa_model": "pa-model",
                "simulator_model": "simulator-model",
            },
            {"t": 1, "kind": "beat_enter", "beat": "b1", "active_skills": []},
            {
                "t": 2,
                "kind": "simulator_turn",
                "beat": "b1",
                "message": "hi",
                "factual_check": [],
                "turn_assessment": {"next_beat": "end"},
                "parse_warnings": [],
            },
            {
                "t": 3,
                "kind": "pa_turn",
                "beat": "b1",
                "outbound": "hello",
                "tool_events": [{"name": "document_read", "status": "ok"}],
                "ix_required": [],
                "ix_called": [],
                "ix_missing": [],
            },
            {"t": 4, "kind": "session_end", "simulator_turns": 1, "beat_count": 1},
        ]


class FailingLoop:
    async def run_session(self):
        raise RuntimeError("boom")


class CompressedLoop(FakeLoop):
    """A compressed session: the probe beat, then the reaction, then nothing.

    Events go out one at a time, so the test can ask what the state root looked
    like at the moment each PA turn was about to run -- which is the whole
    question the session-closing marker answers.
    """

    def __init__(self, writer, state_root):
        super().__init__(writer)
        self.state_root = Path(state_root)
        self.closed_at = {}

    async def run_session(self):
        events = await self.run_session_events()
        events = [
            *events[:1],
            {"t": 1, "kind": "beat_enter", "beat": "b1_probe", "active_skills": ["x"]},
            {**events[2], "t": 2, "beat": "b1_probe"},
            {**events[3], "t": 3, "beat": "b1_probe"},
            {"t": 4, "kind": "beat_enter", "beat": "b2_reaction", "active_skills": []},
            {**events[2], "t": 5, "beat": "b2_reaction"},
            {**events[3], "t": 6, "beat": "b2_reaction"},
            {"t": 7, "kind": "session_end", "simulator_turns": 2, "beat_count": 2},
        ]
        for event in events:
            if event["kind"] == "pa_turn":
                self.closed_at[event["beat"]] = (
                    self.state_root / SESSION_CLOSING_MARKER
                ).exists()
            self.writer.log_event(event)
        return events


def _compressed_stage(tmp_path: Path, reaction_index: int | None) -> StageStepRuntime:
    state_root = tmp_path / "stage_work" / "acc_035"
    state_root.mkdir(parents=True)
    return StageStepRuntime(
        run_id="run_001",
        user_id="user_a",
        session_id="acc_035",
        step_id="acc_035",
        state_root=state_root,
        project_root=tmp_path,
        enabled_tools=["document_read"],
        compressed_reaction_index=reaction_index,
    )


@pytest.mark.asyncio
async def test_the_pas_final_turn_of_a_compressed_session_runs_closed(tmp_path: Path):
    """The wiring, end to end: the executor installs the closing hook.

    The probe beat's PA turn is open -- the session is still going -- and the
    reaction beat's is not, which is the turn `acc_035` t8 spent
    writing a roster change nobody had asked for.
    """
    stage = _compressed_stage(tmp_path, 2)
    loop = CompressedLoop(None, stage.state_root)

    def factory(_step, writer, _stage):
        loop.writer = writer
        return loop

    await SessionExecutor(factory).execute(
        _step(), steps_dir=tmp_path / "steps", stage_runtime=stage
    )

    assert loop.closed_at == {"b1_probe": False, "b2_reaction": True}


@pytest.mark.asyncio
async def test_an_uncompressed_session_is_never_closed(tmp_path: Path):
    """No reaction index, no hook, no marker -- on any turn."""
    stage = _compressed_stage(tmp_path, None)
    loop = CompressedLoop(None, stage.state_root)

    def factory(_step, writer, _stage):
        loop.writer = writer
        return loop

    await SessionExecutor(factory).execute(
        _step(), steps_dir=tmp_path / "steps", stage_runtime=stage
    )

    assert loop.closed_at == {"b1_probe": False, "b2_reaction": False}
    assert not (stage.state_root / SESSION_CLOSING_MARKER).exists()


class InvalidSessionLoop(FakeLoop):
    """A session the simulator invalidated: `session_invalid` and a raise.

    Same shape `SimulatorLoop` produces — the end event is written before the
    error leaves the loop — so the executor reads the rule off the transcript
    it already has rather than off the exception.
    """

    async def run_session(self):
        for event in await self.run_session_events():
            if event["kind"] == "session_end":
                continue
            self.writer.log_event(event)
        self.writer.log_event(
            {
                "t": 4,
                "kind": "session_invalid",
                "rule": "beat_ask_unfulfilled",
                "beat": "b1",
                "attempts": 3,
                "simulator_turns": 1,
                "beat_count": 1,
            }
        )
        raise SessionInvalidError("beat_ask_unfulfilled", beat="b1", attempts=3)


class ForcedOutreachLoop(FakeLoop):
    """A trigger turn the wrapper steered under `MEMPA_VERIFY_OUTREACH`.

    Shaped the way `pa/agent/loop.py` leaves it: the model's own selection is
    still in `tool_events`, and the override sits beside it as one more
    `ix_event`. Nothing in the simulator wrote either.
    """

    async def run_session_events(self):
        events = await super().run_session_events()
        for event in events:
            if event["kind"] != "pa_turn":
                continue
            event["tool_events"] = [
                *event["tool_events"],
                {
                    "ix": True,
                    "name": "IX_proactive_outreach",
                    "attribute": "proactive_outreach",
                    "setting": "High",
                    "outreach_action": "send",
                },
                {
                    "ix_event": True,
                    "name": "verification_override",
                    "attribute": "proactive_outreach",
                    "model_outreach_action": "send",
                    "model_setting": "High",
                    "outreach_action": "suppress",
                    "setting": "Low",
                    "source": "MEMPA_VERIFY_OUTREACH",
                },
            ]
        return events


class RetryingLoop(FakeLoop):
    async def run_session(self):
        self.writer.log_validation_attempt(
            {
                "phase": "simulator_turn",
                "attempt": 1,
                "error_codes": ["message_block_missing"],
                "raw_response": "invalid raw output",
            }
        )
        return await super().run_session()


def _step() -> RunStep:
    return RunStep(
        step_id="acc_001",
        kind="accumulation",
        persona_id="user_a",
        script_path="missing.yaml",
        context="work",
        target_cell="cell_a",
        acc_num=1,
        benchmark_session_timestamp=datetime(
            2026, 10, 1, 9, 0, tzinfo=timezone.utc
        ),
        memory_mode="read_write",
        stage_policy="commit",
        placeholder=True,
    )


@pytest.mark.asyncio
async def test_session_executor_writes_artifacts(tmp_path: Path):
    executor = SessionExecutor(lambda _step, writer, _stage: FakeLoop(writer))

    result = await executor.execute(_step(), steps_dir=tmp_path / "steps")

    assert result.beats == 1
    assert result.tool_calls == 1
    for path in result.artifacts.values():
        assert Path(path).exists()
    assert "memory_mode: read_write" in Path(result.artifacts["meta"]).read_text(encoding="utf-8")
    toolcalls = json.loads(Path(result.artifacts["pa_toolcalls"]).read_text(encoding="utf-8"))
    assert toolcalls["meta"]["step_id"] == "acc_001"
    assert toolcalls["meta"]["target_cell"] == "cell_a"
    assert toolcalls["meta"]["memory_mode"] == "read_write"
    assert toolcalls["meta"]["pa_model"] == "pa-model"
    assert toolcalls["meta"]["simulator_model"] == "simulator-model"
    meta = Path(result.artifacts["meta"]).read_text(encoding="utf-8")
    assert "pa_model: pa-model" in meta
    assert "simulator_model: simulator-model" in meta


@pytest.mark.asyncio
async def test_session_executor_propagates_loop_error(tmp_path: Path):
    executor = SessionExecutor(lambda _step, _writer, _stage: FailingLoop())

    with pytest.raises(RuntimeError, match="boom"):
        await executor.execute(_step(), steps_dir=tmp_path / "steps")


@pytest.mark.asyncio
async def test_an_invalid_session_names_its_rule_in_meta(tmp_path: Path):
    """A step whose simulator failed as an instrument
    says so in its own directory, not only in the ledger, so a read can count
    the rule without reopening the transcript."""
    executor = SessionExecutor(
        lambda _step, writer, _stage: InvalidSessionLoop(writer)
    )

    with pytest.raises(SessionInvalidError) as exc_info:
        await executor.execute(_step(), steps_dir=tmp_path / "steps")

    artifacts = failed_step_artifacts(exc_info.value)
    meta = yaml.safe_load(Path(artifacts["meta"]).read_text(encoding="utf-8"))
    assert meta["session_invalid"] == {
        "rule": "beat_ask_unfulfilled",
        "beat": "b1",
        "attempts": 3,
    }
    assert meta["failure_reason"].startswith("SessionInvalidError: SESSION_INVALID")
    # The audit trail is still written: a discarded session is the one a reader
    # most needs to be able to open.
    assert Path(artifacts["transcript_jsonl"]).exists()


@pytest.mark.asyncio
async def test_a_finished_session_records_no_invalid_rule(tmp_path: Path):
    executor = SessionExecutor(lambda _step, writer, _stage: FakeLoop(writer))

    result = await executor.execute(_step(), steps_dir=tmp_path / "steps")

    meta = yaml.safe_load(Path(result.artifacts["meta"]).read_text(encoding="utf-8"))
    assert "session_invalid" not in meta


@pytest.mark.asyncio
async def test_a_forced_outreach_branch_is_stamped_on_transcript_and_meta(
    tmp_path: Path,
):
    """The verification aid.

    The wrapper's stamp rides inside `pa_turn.tool_events`, where only a
    reader that opens tool events would find it. The executor lifts it to a
    top-level transcript event and to `meta.yaml`, so a step directory picked
    up later cannot be mistaken for a scored session.
    """
    executor = SessionExecutor(
        lambda _step, writer, _stage: ForcedOutreachLoop(writer)
    )

    result = await executor.execute(_step(), steps_dir=tmp_path / "steps")

    meta = yaml.safe_load(Path(result.artifacts["meta"]).read_text(encoding="utf-8"))
    assert meta["verification_override"] == {
        "outreach_action": "suppress",
        "source": "MEMPA_VERIFY_OUTREACH",
    }
    events = [
        json.loads(line)
        for line in Path(result.artifacts["transcript_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    stamps = [
        event for event in events if event["kind"] == "verification_override"
    ]
    assert len(stamps) == 1
    assert stamps[0]["turn"] == 3
    assert stamps[0]["beat"] == "b1"
    assert stamps[0]["outreach_action"] == "suppress"
    # What the PA would have done unforced survives in the same record.
    assert stamps[0]["model_outreach_action"] == "send"
    assert (
        "VERIFICATION OVERRIDE — NOT A SCORED RUN"
        in Path(result.artifacts["transcript_md"]).read_text(encoding="utf-8")
    )


@pytest.mark.asyncio
async def test_an_ordinary_session_carries_no_verification_stamp(tmp_path: Path):
    executor = SessionExecutor(lambda _step, writer, _stage: FakeLoop(writer))

    result = await executor.execute(_step(), steps_dir=tmp_path / "steps")

    meta = yaml.safe_load(Path(result.artifacts["meta"]).read_text(encoding="utf-8"))
    assert "verification_override" not in meta
    assert "verification_override" not in (
        Path(result.artifacts["transcript_jsonl"]).read_text(encoding="utf-8")
    )


def test_session_executor_registers_validation_attempt_artifact(
    tmp_path: Path,
):
    async def run():
        executor = SessionExecutor(
            lambda _step, writer, _stage: RetryingLoop(writer)
        )

        result = await executor.execute(_step(), steps_dir=tmp_path / "steps")

        attempts = Path(result.artifacts["simulator_validation_attempts"])
        assert attempts.name == "simulator_validation_attempts.jsonl"
        assert "invalid raw output" in attempts.read_text(encoding="utf-8")
        assert "simulator_emotion_events" not in result.artifacts
        assert "simulator_emotion_state" not in result.artifacts
        meta = Path(result.artifacts["meta"]).read_text(encoding="utf-8")
        assert "simulator_emotion" not in meta

    asyncio.run(run())


@pytest.mark.asyncio
async def test_session_executor_preserves_step_state_audit(tmp_path: Path):
    state_root = tmp_path / "stage_work" / "acc_001"
    state_root.mkdir(parents=True)
    (state_root / "tool_log.jsonl").write_text(
        '{"tool":"weather_forecast_read"}\n', encoding="utf-8"
    )
    stage = StageStepRuntime(
        run_id="run_001",
        user_id="user_a",
        session_id="acc_001",
        step_id="acc_001",
        state_root=state_root,
        project_root=tmp_path,
        enabled_tools=["weather_forecast_read"],
    )
    executor = SessionExecutor(lambda _step, writer, _stage: FakeLoop(writer))

    result = await executor.execute(
        _step(),
        steps_dir=tmp_path / "steps",
        stage_runtime=stage,
    )

    assert Path(result.artifacts["tool_log"]).is_file()
    assert Path(result.artifacts["state_diff"]).read_text(encoding="utf-8") == ""


class IXWrapperLoop(FakeLoop):
    """A PA turn that used the wrapper: IX tools, a notice, one real call."""

    async def run_session(self):
        events = await super().run_session_events()
        events[3]["tool_events"] = [
            {"name": "IX_autonomy_level", "status": "ok", "ix": True,
             "attribute": "autonomy_level", "setting": "Reactive", "args": {}},
            {"name": "IX_render_autonomy_level", "status": "ok", "args": {"plan_id": "p1"}},
            {"name": "IX_continue_autonomy_level", "status": "ok", "args": {}},
            {"name": "IX_capability_resolve", "status": "ok", "args": {}},
            {"name": "document_read", "status": "ok",
             "args": {"path": "a.md", "_ix_notice": "Checking.", "interaction_item_id": "i1"}},
            {"process": True, "stage": "start", "message": "legacy control event"},
        ]
        events[3]["ix_render"] = {
            "schema_version": IX_RENDER_SCHEMA_VERSION,
            "ix_contract_version": IX_CONTRACT_VERSION,
            "episodes": {},
            "released": {"batches": [], "questions": [], "plan_executed": []},
            "held": {"batches": [], "questions": [], "plan_remaining": [],
                     "calls": [], "cancelled": [], "outreach_message": ""},
            "gate": None,
            "consent": [],
            "notices": [],
            "completion": {"released": True, "text": "Done."},
            "reasoning": {"shown": False, "source": "user_facing_rationale",
                          "provider_reasoning_present": False},
            "certainties": [],
            "proactive_decision": None,
            "trace": [],
        }
        for event in events:
            self.writer.log_event(event)
        return events


@pytest.mark.asyncio
async def test_tool_call_count_excludes_ix_and_internal_tools(tmp_path: Path):
    executor = SessionExecutor(lambda _step, writer, _stage: IXWrapperLoop(writer))

    result = await executor.execute(_step(), steps_dir=tmp_path / "steps")

    # Four IX tools and one legacy control event on that turn; one real call.
    assert result.tool_calls == 1
    meta = Path(result.artifacts["meta"]).read_text(encoding="utf-8")
    assert f"ix_contract_version: {IX_CONTRACT_VERSION}" in meta
    assert f"ix_render_schema_version: {IX_RENDER_SCHEMA_VERSION}" in meta


def _blob(**overrides) -> dict:
    blob = {
        "schema_version": IX_RENDER_SCHEMA_VERSION,
        "ix_contract_version": IX_CONTRACT_VERSION,
        "episodes": {},
        "released": {"batches": [], "questions": [], "plan_executed": []},
        "held": {"batches": [], "questions": [], "plan_remaining": [],
                 "calls": [], "cancelled": [], "outreach_message": ""},
        "gate": None,
        "consent": [],
        "notices": [],
        "completion": {"released": False, "text": ""},
        "reasoning": {"shown": False, "source": "user_facing_rationale",
                      "provider_reasoning_present": False},
        "certainties": [],
        "proactive_decision": None,
        "trace": [],
    }
    blob.update(overrides)
    return blob


class LoopThatFailsAfterTwoTurns:
    """Two beats render fine, then the contract check kills the step."""

    def __init__(self, writer):
        self.writer = writer
        self.ix_render_records = [
            {"kind": "direct", "ix_render": _blob(trace=["turn 1"])},
            {"kind": "direct", "ix_render": _blob(trace=["turn 2"])},
        ]

    async def run_session(self):
        self.writer.log_event({"t": 0, "kind": "session_start", "persona": "user_a",
                               "session_id": "acc_001", "pa_model": "pa-model"})
        for index in (1, 2):
            self.writer.log_event(
                {"t": index * 2 - 1, "kind": "beat_enter", "beat": f"b{index}",
                 "active_skills": []}
            )
            self.writer.log_event(
                {"t": index * 2, "kind": "pa_turn", "beat": f"b{index}",
                 "outbound": "ok", "tool_events": [], "ix_required": [],
                 "ix_called": [], "ix_missing": []}
            )
        raise RuntimeError(
            "PA turn failed the IX render contract: a required "
            "IX_render_<attribute> payload was missing twice"
        )


@pytest.mark.asyncio
async def test_a_failed_step_still_writes_its_audit_artifacts(tmp_path: Path):
    """The run that most needs an audit trail was the one left without one.

    A step that dies on the render contract used to leave only
    `transcript.jsonl` and `state/`: no merge, so every `pa_turn.ix_render` was
    `null` — including the turns that rendered fine long before the failure.
    """
    executor = SessionExecutor(
        lambda _step, writer, _stage: LoopThatFailsAfterTwoTurns(writer)
    )

    with pytest.raises(RuntimeError, match="IX render contract"):
        await executor.execute(_step(), steps_dir=tmp_path / "steps")

    step_dir = tmp_path / "steps" / "acc_001"
    toolcalls = json.loads((step_dir / "pa_toolcalls.json").read_text(encoding="utf-8"))
    assert [beat["beat"] for beat in toolcalls["beats"]] == ["b1", "b2"]
    assert [beat["ix_render"][0]["trace"] for beat in toolcalls["beats"]] == [
        ["turn 1"], ["turn 2"]
    ]
    meta = (step_dir / "meta.yaml").read_text(encoding="utf-8")
    assert "failure_reason: 'RuntimeError: PA turn failed the IX render contract" in meta
    assert (step_dir / "transcript.md").is_file()


@pytest.mark.asyncio
async def test_a_failed_step_hands_its_artifact_paths_to_the_caller(tmp_path: Path):
    """Writing the artifacts is half the job — the ledger has to learn where.

    The result object carrying the paths dies with the raised exception, so
    the paths ride on the exception instead and `mark_failed` records them.
    """
    executor = SessionExecutor(
        lambda _step, writer, _stage: LoopThatFailsAfterTwoTurns(writer)
    )

    with pytest.raises(RuntimeError, match="IX render contract") as caught:
        await executor.execute(_step(), steps_dir=tmp_path / "steps")

    step_dir = tmp_path / "steps" / "acc_001"
    artifacts = failed_step_artifacts(caught.value)
    assert artifacts["transcript_jsonl"] == str(step_dir / "transcript.jsonl")
    assert artifacts["pa_toolcalls"] == str(step_dir / "pa_toolcalls.json")
    assert artifacts["meta"] == str(step_dir / "meta.yaml")


def test_failed_step_artifacts_of_an_unrelated_error_is_empty() -> None:
    assert failed_step_artifacts(RuntimeError("nothing was written")) == {}


@pytest.mark.asyncio
async def test_a_failing_artifact_write_does_not_mask_the_step_error(tmp_path: Path):
    """Best effort means best effort: the step's own error is what propagates."""

    class _Loop(LoopThatFailsAfterTwoTurns):
        def __init__(self, writer):
            super().__init__(writer)
            # A malformed blob makes the version read raise during the write.
            self.ix_render_records = [{"ix_render": {"schema_version": "bogus"}}]

    executor = SessionExecutor(lambda _step, writer, _stage: _Loop(writer))

    with pytest.raises(RuntimeError, match="IX render contract"):
        await executor.execute(_step(), steps_dir=tmp_path / "steps")

    meta = (tmp_path / "steps" / "acc_001" / "meta.yaml").read_text(encoding="utf-8")
    assert "ix_render_version_error" in meta


@pytest.mark.asyncio
async def test_a_step_without_a_rendered_turn_records_no_version(tmp_path: Path):
    executor = SessionExecutor(lambda _step, writer, _stage: FakeLoop(writer))

    result = await executor.execute(_step(), steps_dir=tmp_path / "steps")

    meta = Path(result.artifacts["meta"]).read_text(encoding="utf-8")
    assert "ix_contract_version" not in meta


# ── tool_outside_plan ──────────────────────────────────────────────


def _draft_declaring(tmp_path: Path, tools: list[str]) -> Path:
    """A session draft whose atomic steps name exactly `tools`."""
    path = tmp_path / "draft.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "task_execution": {
                    "atomic_steps": [
                        {"step_id": f"tool_step_{index}", "actor": "tool", "tool": tool}
                        for index, tool in enumerate(tools)
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return path


class _CallingLoop:
    """One beat and one PA turn per entry in `turns`, calling the tools it names."""

    turns: list[list[str]] = []

    def __init__(self, writer):
        self.writer = writer

    async def run_session(self):
        events: list[dict] = [
            {
                "t": 0,
                "kind": "session_start",
                "persona": "user_a",
                "session_id": "acc_001",
                "pa_model": "pa-model",
                "simulator_model": "simulator-model",
            }
        ]
        for index, tools in enumerate(self.turns):
            beat = f"b{index + 1}"
            events.append(
                {"t": index * 2 + 1, "kind": "beat_enter", "beat": beat,
                 "active_skills": []}
            )
            events.append(
                {
                    "t": index * 2 + 2,
                    "kind": "pa_turn",
                    "beat": beat,
                    "outbound": "ok",
                    "tool_events": [
                        {
                            "name": f"{STATE_TOOL_NAME_PREFIX}{tool}",
                            "status": "ok",
                            "args": {},
                        }
                        for tool in tools
                    ],
                    "ix_required": [],
                    "ix_called": [],
                    "ix_missing": [],
                }
            )
        events.append(
            {"t": 99, "kind": "session_end", "simulator_turns": len(self.turns),
             "beat_count": len(self.turns)}
        )
        for event in events:
            self.writer.log_event(event)
        return events


def _outside_plan_records(result) -> list[dict]:
    return [
        event
        for event in (
            json.loads(line)
            for line in Path(result.artifacts["transcript_jsonl"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        if event.get("kind") == "tool_outside_plan"
    ]


@pytest.mark.asyncio
async def test_every_undeclared_write_call_is_recorded(tmp_path: Path):
    """`acc_023`: the PA sent a real email the draft never planned.

    Sent twice, across two beats, and both calls are recorded. The old rule
    kept only the first per tool name, which is why `acc_039` showed
    2 records for 4 refused email writes and `acc_011` 1 for 6 refused
    `document_update`s — the count read as "tried once" either way. The
    declared tools around it still record nothing.
    """
    step = _step().model_copy(
        update={
            "script_path": _draft_declaring(
                tmp_path, ["calendar_event_create", "email_draft_create"]
            )
        }
    )

    class _Loop(_CallingLoop):
        turns = [
            ["calendar_event_create", "email_message_send"],
            ["email_draft_create", "email_message_send"],
        ]

    executor = SessionExecutor(lambda _step, writer, _stage: _Loop(writer))
    result = await executor.execute(step, steps_dir=tmp_path / "steps")

    assert _outside_plan_records(result) == [
        {
            "kind": "tool_outside_plan",
            "tool": "email_message_send",
            "tool_class": "write",
            "turn": 2,
            "beat": "b1",
            "call_index": 1,
        },
        {
            "kind": "tool_outside_plan",
            "tool": "email_message_send",
            "tool_class": "write",
            "turn": 4,
            "beat": "b2",
            "call_index": 1,
        },
    ]
    # Diagnostic only: all four calls ran and still count as tool calls.
    assert result.tool_calls == 4


@pytest.mark.asyncio
async def test_repeated_undeclared_writes_inside_one_turn_are_told_apart(
    tmp_path: Path,
):
    """`acc_011`: six refused `document_update`s inside three turns.

    Same tool, same turn, same beat — without `call_index` the records would be
    byte-identical and a reader could not say which call of the turn each one
    is.
    """
    step = _step().model_copy(
        update={"script_path": _draft_declaring(tmp_path, ["document_read"])}
    )

    class _Loop(_CallingLoop):
        turns = [["document_read", "document_update", "document_update"]]

    executor = SessionExecutor(lambda _step, writer, _stage: _Loop(writer))
    result = await executor.execute(step, steps_dir=tmp_path / "steps")

    assert _outside_plan_records(result) == [
        {
            "kind": "tool_outside_plan",
            "tool": "document_update",
            "tool_class": "write",
            "turn": 2,
            "beat": "b1",
            "call_index": 1,
        },
        {
            "kind": "tool_outside_plan",
            "tool": "document_update",
            "tool_class": "write",
            "turn": 2,
            "beat": "b1",
            "call_index": 2,
        },
    ]


@pytest.mark.asyncio
async def test_an_undeclared_read_is_recorded_as_a_read(tmp_path: Path):
    """The read calls are recorded too.

    `acc_023` made roughly ninety out-of-plan reads and the records
    said none, because the read class was dropped before anything was written.
    The per-model read behaviour is analysed later, so every call is kept —
    `tool_class` is what tells the two apart afterwards.
    """
    step = _step().model_copy(
        update={"script_path": _draft_declaring(tmp_path, ["calendar_event_create"])}
    )

    class _Loop(_CallingLoop):
        turns = [["document_read", "document_read", "place_search"]]

    executor = SessionExecutor(lambda _step, writer, _stage: _Loop(writer))
    result = await executor.execute(step, steps_dir=tmp_path / "steps")

    assert _outside_plan_records(result) == [
        {
            "kind": "tool_outside_plan",
            "tool": "document_read",
            "tool_class": "read",
            "turn": 2,
            "beat": "b1",
            "call_index": 0,
        },
        {
            "kind": "tool_outside_plan",
            "tool": "document_read",
            "tool_class": "read",
            "turn": 2,
            "beat": "b1",
            "call_index": 1,
        },
        {
            "kind": "tool_outside_plan",
            "tool": "place_search",
            "tool_class": "read",
            "turn": 2,
            "beat": "b1",
            "call_index": 2,
        },
    ]


@pytest.mark.asyncio
async def test_read_and_write_in_one_turn_are_recorded_under_their_own_class(
    tmp_path: Path,
):
    """One turn, one record each, told apart by `tool_class` and not by name.

    A batch review counts writes and reads separately — the reads are the
    per-model behaviour question, the writes are the ones that changed a world
    the draft never planned to change.
    """
    step = _step().model_copy(
        update={"script_path": _draft_declaring(tmp_path, ["calendar_event_create"])}
    )

    class _Loop(_CallingLoop):
        turns = [["place_search", "email_message_send"]]

    executor = SessionExecutor(lambda _step, writer, _stage: _Loop(writer))
    result = await executor.execute(step, steps_dir=tmp_path / "steps")

    assert [
        (record["tool"], record["tool_class"])
        for record in _outside_plan_records(result)
    ] == [("place_search", "read"), ("email_message_send", "write")]


def test_the_read_class_suffixes_agree_with_the_tool_catalog() -> None:
    """The suffix rule stands in for `allowed_effects`; the catalog must agree.

    `tool_outside_plan_events` sorts read from write by tool name so it needs
    no catalog at run time. That only holds while the catalog keeps the two
    apart by suffix — a new read-only `*_list`, or a `*_read` that writes,
    would make the diagnostic miss a write or flag a probe, silently.
    """
    catalog_path = (
        Path(__file__).resolve().parents[2] / "data" / "state_tools" / "catalog.yaml"
    )
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    read_only_effects = {"read", "verify"}

    missorted = []
    for name, spec in (catalog.get("tools") or {}).items():
        effects = set(spec.get("allowed_effects") or [])
        assert effects, f"{name} declares no allowed_effects"
        reads_only = effects <= read_only_effects
        if name.endswith(READ_CLASS_TOOL_SUFFIXES) != reads_only:
            missorted.append(f"{name} allowed_effects={sorted(effects)}")
    assert not missorted, "suffix rule disagrees with the catalog: " + "; ".join(
        missorted
    )


def _pa_turn_with(tool: str, args: dict) -> dict:
    return {
        "t": 3,
        "kind": "pa_turn",
        "beat": "b1",
        "tool_events": [{"name": f"mcp_state_{tool}", "args": args, "status": "ok"}],
    }


def test_a_declared_tool_pointed_off_plan_is_its_own_finding(tmp_path: Path) -> None:
    """`acc_019` wrote to a file no atomic step names through a
    perfectly declared `document_update`, and the name-only rule read it as
    clean. The argument comparison catches it, under its own record kind so the
    name-only counts stay comparable."""
    draft = tmp_path / "draft.yaml"
    draft.write_text(
        yaml.safe_dump(
            {
                "task_execution": {
                    "atomic_steps": [
                        {
                            "step_id": "s1",
                            "tool": "document_update",
                            "tool_arguments": {"path": "draft_intro.md"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    events = [_pa_turn_with("document_update", {"path": "draft_section3_w_notes.md"})]

    records = tool_outside_plan_events(
        events,
        declared_plan_tools(draft),
        declared_plan_arguments(draft),
    )

    assert [r["kind"] for r in records] == ["tool_arguments_off_plan"]
    assert records[0]["tool"] == "document_update"
    assert records[0]["tool_class"] == "write"
    assert records[0]["args"] == {"path": "draft_section3_w_notes.md"}


def test_runtime_added_arguments_do_not_make_a_declared_call_off_plan(
    tmp_path: Path,
) -> None:
    """A draft names the arguments that carry the story; the runtime fills in
    ids and defaults the author never wrote. Only the declared keys are
    compared, so those additions are not a finding."""
    draft = tmp_path / "draft.yaml"
    draft.write_text(
        yaml.safe_dump(
            {
                "task_execution": {
                    "atomic_steps": [
                        {
                            "step_id": "s1",
                            "tool": "document_update",
                            "tool_arguments": {"path": "draft_intro.md"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    events = [
        _pa_turn_with(
            "document_update", {"path": "draft_intro.md", "user_id": "user_a"}
        )
    ]

    assert (
        tool_outside_plan_events(
            events, declared_plan_tools(draft), declared_plan_arguments(draft)
        )
        == []
    )


def test_a_gate_payload_param_is_compared_exactly(tmp_path: Path) -> None:
    """A gate binds its payload fields, so a body that is only nearly the one
    the plan wrote is exactly the case worth seeing."""
    draft = tmp_path / "draft.yaml"
    draft.write_text(
        yaml.safe_dump(
            {
                "task_execution": {
                    "atomic_steps": [
                        {
                            "step_id": "s1",
                            "tool": "email_message_send",
                            "tool_arguments": {"body": "the agreed line"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    events = [_pa_turn_with("email_message_send", {"body": "the agreed line, edited"})]

    records = tool_outside_plan_events(
        events, declared_plan_tools(draft), declared_plan_arguments(draft)
    )

    assert [r["kind"] for r in records] == ["tool_arguments_off_plan"]


def test_omitting_the_declared_arguments_keeps_the_name_only_behaviour(
    tmp_path: Path,
) -> None:
    """Every caller that has not been taught the arguments yet must keep seeing
    exactly what it saw before."""
    draft = tmp_path / "draft.yaml"
    draft.write_text(
        yaml.safe_dump(
            {
                "task_execution": {
                    "atomic_steps": [
                        {
                            "step_id": "s1",
                            "tool": "document_update",
                            "tool_arguments": {"path": "draft_intro.md"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    events = [_pa_turn_with("document_update", {"path": "somewhere_else.md"})]

    assert tool_outside_plan_events(events, declared_plan_tools(draft)) == []


@pytest.mark.asyncio
async def test_the_text_judge_pack_is_written_beside_the_transcript(
    tmp_path: Path,
) -> None:
    """`render_session_html` was the only thing that ever built the pack, and
    it built it in-process, so nothing on disk said what the text-face judge
    would actually have been handed for a finished session."""
    step = _step()
    executor = SessionExecutor(lambda _step, writer, _stage: FakeLoop(writer))

    result = await executor.execute(step, steps_dir=tmp_path / "steps")

    pack_path = tmp_path / "steps" / step.step_id / "text_judge_pack.json"
    assert result.artifacts["text_judge_pack"] == str(pack_path)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    assert "turns" in pack
