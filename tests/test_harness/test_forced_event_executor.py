from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from harness.forced_event_executor import (
    ForcedEventExecutionError,
    ForcedEventExecutor,
)
from harness.memory_io import MemoryIORecorder
from harness.state_runtime import StageRuntime, preserve_step_state_artifacts
from harness.transcript import TranscriptWriter, load_events, write_markdown
from memory.adapter import PromptHistoryEvictionPolicy
from memory.results import WriteReceipt, WriteStatus
from pa.agent.loop import (
    ENVIRONMENT_EVENT_HEADER,
    ENVIRONMENT_EVENT_ROLE,
    render_environment_event,
)
from pa.agent.memory import MemoryConsolidator
from pa.agent.tools.registry import ToolRegistry
from pa.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from pa.session.manager import SessionManager
from pa.utils.helpers import estimate_message_tokens


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT_PATH = (
    PROJECT_ROOT
    / "tests/fixtures/authoring/forced_event_transcript_evolv_05.yaml"
)
BENCHMARK_TIMESTAMP = datetime(
    2026, 12, 8, 18, 30, tzinfo=timezone.utc
)


class RecordingCommittedAdapter:
    backend_name = "gate_c_recording"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.after_adapter_commit

    def __init__(self) -> None:
        self.scopes = []
        self.events = []

    async def setup_scope(self, scope) -> None:
        self.scopes.append(scope)

    async def record_event(self, event) -> WriteReceipt:
        self.events.append(event)
        return WriteReceipt(
            status=WriteStatus.committed,
            native_ids=[event.event_id],
        )


def _gate_c_runtime(tmp_path: Path):
    run_id = "gate_c_evolv_05"
    stage = StageRuntime(
        run_id=run_id,
        user_id="user_a",
        fixture_root=PROJECT_ROOT / "data/state_fixtures",
        runs_root=tmp_path / "runs",
        project_root=PROJECT_ROOT,
    )
    step = stage.begin_step(
        step_id="evolv_05",
        session_id="evolv_05",
        runtime_artifact_path=TRANSCRIPT_PATH,
    )
    sessions = SessionManager(tmp_path / "pa_workspace")
    session_key = f"{run_id}_main"
    sessions.create(session_key)
    adapter = RecordingCommittedAdapter()
    recorder = MemoryIORecorder(
        tmp_path / "memory_io.jsonl",
        run_id=run_id,
    )
    step_dir = tmp_path / "steps" / "evolv_05"
    return stage, step, sessions, session_key, adapter, recorder, step_dir


def _executor(
    *,
    transcript_path: Path,
    step,
    sessions,
    session_key: str,
    adapter,
    recorder,
    writer,
) -> ForcedEventExecutor:
    return ForcedEventExecutor(
        transcript_path=transcript_path,
        project_root=PROJECT_ROOT,
        benchmark_session_timestamp=BENCHMARK_TIMESTAMP,
        stage_runtime=step,
        session_manager=sessions,
        pa_session_key=session_key,
        memory_adapter=adapter,
        memory_io_recorder=recorder,
        writer=writer,
        require_committed_adapter_writes=True,
    )


def _third_party_message_transcript(tmp_path: Path) -> Path:
    """The fixture with its delivery rewritten as a direct third-party message.

    An `environment_action` and a `MessageItem` carrying a `SourceRef` source
    are the two ways a third party speaks in a transcript, and neither is
    the user or the PA, so neither reaches memory. No
    landed transcript and no fixture authors the second shape, so the tests
    build it: the delivery becomes a third-party `message`, the PA answers it
    inside the same exchange, and the user's reply moves to the next exchange.
    That leaves an exchange whose only user-or-PA message is the PA's, which is
    what makes the projection check exact.
    """
    data = copy.deepcopy(yaml.safe_load(TRANSCRIPT_PATH.read_text(encoding="utf-8")))
    action = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "environment_action"
    )
    index, exchange = next(
        (index, exchange)
        for index, exchange in enumerate(data["exchanges"])
        if action in exchange["items"]
    )
    exchange["items"][exchange["items"].index(action)] = {
        "type": "message",
        "offset_seconds": action["offset_seconds"],
        "source": {"kind": "third_party", "id": "Glenmont Trail Office"},
        "visibility": "both",
        "content": action["content"],
    }
    exchange["items"].insert(
        1,
        {
            "type": "message",
            "offset_seconds": action["offset_seconds"] + 5,
            "source": "pa",
            "visibility": "both",
            "content": "That closure lands on the day you pinned.",
        },
    )
    data["exchanges"][index : index + 1] = [
        {"items": exchange["items"][:2]},
        {"items": exchange["items"][2:]},
    ]
    # A direct message is conversation only: it lands in no inbox, so there is
    # no environment action left to assert and no delivered path in the diff.
    data["assertions"]["expected_environment_actions"] = []
    path = tmp_path / "third_party_message.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _memory_visible_exchange_count(data: dict) -> int:
    """Exchanges that produce a non-empty pool (and hence one memory event).

    Every exchange carries a PA-visible spoken line from the user or the PA --
    the editorial rule of `Exchange._require_pa_visible_text` -- so every
    exchange renders at least one pool message and commits. Counting the spoken
    line rather than `len(exchanges)` keeps the check on what the pool is built
    from.
    """
    count = 0
    for exchange in data["exchanges"]:
        for item in exchange["items"]:
            if (
                item["type"] == "message"
                and isinstance(item["source"], str)
                and item.get("visibility", "both") in {"pa", "both"}
            ):
                count += 1
                break
    return count


def _expected_state_diff_count(data: dict) -> int:
    expected = data["assertions"]["expected_persistent_state_diff"]
    own = 0 if expected == "none" else expected["state_diff_count"]
    # third-party deliveries land in the world too (inbox / SMS thread)
    delivered = sum(
        1
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "environment_action"
        and item["source"]["kind"] == "third_party"
    )
    return own + delivered

@pytest.mark.asyncio
async def test_executor_runs_frozen_event_without_live_models(tmp_path: Path) -> None:
    transcript_data = yaml.safe_load(TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    transcript_items = [
        item
        for exchange in transcript_data["exchanges"]
        for item in exchange["items"]
    ]
    expected_call_ids = transcript_data["assertions"]["required_tool_calls"]
    expected_tool_names = [
        item["name"].removeprefix("mcp_state_")
        for item in transcript_items
        if item["type"] == "pa_tool_call"
    ]
    stage, step, sessions, session_key, adapter, recorder, step_dir = (
        _gate_c_runtime(tmp_path)
    )
    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=TRANSCRIPT_PATH,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        events = await executor.run_session()
    state_artifacts = preserve_step_state_artifacts(step, step_dir)
    markdown_path = write_markdown(step_dir / "transcript.jsonl")
    stage.finish_step(step, "commit")

    assert events[-1] == {
        "t": max(item["offset_seconds"] for item in transcript_items) + 1,
        "kind": "session_end",
        "exchanges": len(transcript_data["exchanges"]),
        "tool_calls": len(expected_call_ids),
        "environment_actions": len(
            transcript_data["assertions"]["expected_environment_actions"]
        ),
    }
    persisted = sessions.get(session_key)
    tool_call_ids = [
        call["id"]
        for message in persisted.messages
        for call in message.get("tool_calls", [])
    ]
    assert tool_call_ids == expected_call_ids
    assert [
        message["tool_call_id"]
        for message in persisted.messages
        if message["role"] == "tool"
    ] == tool_call_ids
    assert persisted.adapter_committed_end_index == len(persisted.messages)
    assert len(adapter.events) == _memory_visible_exchange_count(transcript_data)
    assert all(
        message["role"] in {"user", "assistant"}
        for event in adapter.events
        for message in event.messages
    )
    assert all(
        "target_attribute" not in message["content"]
        for event in adapter.events
        for message in event.messages
    )
    # The delivered third-party message is in the PA's
    # own session -- and so in the prompt history of every later turn -- and in
    # the pool the adapter is handed, as a tagged block on an `assistant`
    # message. It is never the person's own speech.
    environment_contents = [
        item["content"]
        for item in transcript_items
        if item["type"] == "environment_action"
    ]
    assert environment_contents
    assert all(
        content in (message.get("content") for message in persisted.messages)
        for content in environment_contents
    )
    for content in environment_contents:
        carrying = [
            message
            for event in adapter.events
            for message in event.messages
            if content in message["content"]
        ]
        assert len(carrying) == 1
        assert carrying[0]["role"] == "assistant"
        assert carrying[0]["content"].startswith("[third_party ")

    tool_records = [
        json.loads(line)
        for line in Path(state_artifacts["tool_log"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["tool"] for record in tool_records] == expected_tool_names
    state_diff_records = [
        json.loads(line)
        for line in Path(state_artifacts["state_diff"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(state_diff_records) == _expected_state_diff_count(transcript_data)
    assert not (stage.canonical_stage / "tool_log.jsonl").exists()
    assert not (stage.canonical_stage / "state_diff.jsonl").exists()

    assert not (step_dir / "environment_events.jsonl").exists()
    assert len(recorder.read_records()) == _memory_visible_exchange_count(transcript_data)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Environment Audit" not in markdown
    environment_sources = [
        item["source"]["id"]
        for exchange in transcript_data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "environment_action"
    ]
    for source_id in environment_sources:
        assert f"deliver_message` from {source_id}" in markdown
    first_call = next(
        item
        for exchange in transcript_data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "pa_tool_call"
    )
    # The live world answers the call at replay time: the recorded detail is
    # the real handler's result for the transcript's arguments.
    assert any(
        event["kind"] == "pa_turn"
        and event["tool_events"]
        and event["tool_events"][0]["name"] == first_call["name"]
        and event["tool_events"][0]["detail"]["date"] == first_call["arguments"]["date"]
        and event["tool_events"][0]["detail"]["forecast"]
        for event in load_events(step_dir / "transcript.jsonl")
    )


@pytest.mark.asyncio
async def test_executor_does_not_read_source_event_draft(tmp_path: Path) -> None:
    data = yaml.safe_load(TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    data["event"]["source_event_path"] = (
        "Authoring/sessions/forced/drafts/user_a/does_not_exist.yaml"
    )
    runtime_transcript = tmp_path / "runtime_transcript.yaml"
    runtime_transcript.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    stage, step, sessions, session_key, adapter, recorder, step_dir = (
        _gate_c_runtime(tmp_path)
    )

    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=runtime_transcript,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        events = await executor.run_session()

    assert events[-1]["kind"] == "session_end"
    assert len(adapter.events) == _memory_visible_exchange_count(data)


@pytest.mark.asyncio
async def test_executor_fails_before_dialogue_when_pinned_content_missing(
    tmp_path: Path,
) -> None:
    data = yaml.safe_load(TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    weather_call = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item.get("name") == "mcp_state_weather_forecast_read"
    )
    weather_call["expected_contains"] = {
        "forecast": {"summary": "Clear skies all night"}
    }
    bad_transcript = tmp_path / "bad_expected_result.yaml"
    bad_transcript.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    stage, step, sessions, session_key, adapter, recorder, step_dir = (
        _gate_c_runtime(tmp_path)
    )

    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=bad_transcript,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        with pytest.raises(ForcedEventExecutionError, match="pinned content"):
            await executor.run_session()
    artifacts = preserve_step_state_artifacts(step, step_dir)
    stage.finish_step(step, "discard")

    assert sessions.get(session_key).messages == []
    assert adapter.events == []
    assert Path(artifacts["tool_log"]).read_text(encoding="utf-8").strip()
    assert not (step_dir / "environment_events.jsonl").exists()


@pytest.mark.asyncio
async def test_executor_fails_before_dialogue_when_environment_action_is_missing(
    tmp_path: Path,
) -> None:
    stage, step, sessions, session_key, adapter, recorder, step_dir = (
        _gate_c_runtime(tmp_path)
    )
    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=TRANSCRIPT_PATH,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        executor._prepare_environment_action = lambda *args, **kwargs: None
        with pytest.raises(
            ForcedEventExecutionError,
            match="Expected environment actions",
        ):
            await executor.run_session()
    artifacts = preserve_step_state_artifacts(step, step_dir)
    stage.finish_step(step, "discard")

    assert sessions.get(session_key).messages == []
    assert adapter.events == []
    assert Path(artifacts["tool_log"]).read_text(encoding="utf-8").strip()
    assert not (step_dir / "environment_events.jsonl").exists()


GATED_TRANSCRIPT_PATH = (
    PROJECT_ROOT / "tests/fixtures/authoring/forced_event_transcript_gated.yaml"
)

#: What the resolved gate for the gated fixture looks like, as
#: `harness/gate.py:resolve_session_gate` would hand it to the state server.
#: The test passes it to `begin_step` directly so the replay is exercised
#: without a persona outline behind it.
FIXTURE_GATE = {
    "tool": "email_draft_create",
    "param": "body",
    "task_fact": "entrance_question",
    "dictated_text": (
        "Ask which entrance stays open and say I am walking the lower loop "
        "that afternoon."
    ),
    "key_terms": ["entrance stays open", "lower loop"],
    "gate_cue": "what the note should say",
    "reveal_beat": "b3_probe_dictate_note",
    "pre_reveal_write_tools": [],
}


def _gated_runtime(tmp_path: Path):
    run_id = "gate_c_evolv_05_gated"
    stage = StageRuntime(
        run_id=run_id,
        user_id="user_a",
        fixture_root=PROJECT_ROOT / "data/state_fixtures",
        runs_root=tmp_path / "runs",
        project_root=PROJECT_ROOT,
    )
    step = stage.begin_step(
        step_id="evolv_05",
        session_id="evolv_05",
        runtime_artifact_path=GATED_TRANSCRIPT_PATH,
        gate=dict(FIXTURE_GATE),
    )
    sessions = SessionManager(tmp_path / "pa_workspace")
    session_key = f"{run_id}_main"
    sessions.create(session_key)
    return stage, step, sessions, session_key, tmp_path / "steps" / "evolv_05_gated"


@pytest.mark.asyncio
async def test_a_gated_call_is_held_before_the_reveal_and_runs_after(
    tmp_path: Path,
) -> None:
    """The forced line records a held call the way the accumulation line does.

    The first `email_draft_create` is composed before the user has dictated the
    body, so it never reaches the world: it is recorded with `status: held` and
    the wrapper's own held result. The second carries the user's information
    and executes.
    """
    stage, step, sessions, session_key, step_dir = _gated_runtime(tmp_path)
    adapter = RecordingCommittedAdapter()
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id="gate_c_evolv_05_gated")
    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=GATED_TRANSCRIPT_PATH,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        events = await executor.run_session()
    state_artifacts = preserve_step_state_artifacts(step, step_dir)
    stage.finish_step(step, "commit")

    tool_events = [
        event["tool_events"][0] for event in events if event["kind"] == "pa_turn"
    ]
    assert [event["status"] for event in tool_events] == ["held", "ok"]
    held, executed = tool_events
    assert held["call_id"] == "call_gated_early"
    assert held["detail"] == {
        "status": "held",
        "detail": (
            "not executed this turn — waiting for the user; "
            "do not describe it as done"
        ),
    }
    assert executed["call_id"] == "call_gated_release"
    # The held call never reached the world, so only one draft was written.
    assert events[-1]["tool_calls"] == 1
    tool_records = [
        json.loads(line)
        for line in Path(state_artifacts["tool_log"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    # The held call never reached the store, so it leaves no row: the gate
    # arrived, the release carried the user's information, and that one write
    # ran -- the same three rows a gated accumulation session leaves.
    assert [record["tool"] for record in tool_records] == [
        "gate_loaded",
        "gate_satisfied",
        "email_draft_create",
    ]
    # The PA's own session still sees a well-formed call/result pair for both.
    persisted = sessions.get(session_key)
    assert [
        message["tool_call_id"]
        for message in persisted.messages
        if message["role"] == "tool"
    ] == ["call_gated_early", "call_gated_release"]


@pytest.mark.asyncio
async def test_a_gated_call_the_writer_did_not_hold_is_refused(
    tmp_path: Path,
) -> None:
    """Nothing holds itself: an unmarked pre-reveal gate call is an error.

    Left to run it would reach the store and be refused there, which reads as a
    broken event rather than as the hold the session exists to show.
    """
    data = yaml.safe_load(GATED_TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    for exchange in data["exchanges"]:
        for item in exchange["items"]:
            if item.get("call_id") == "call_gated_early":
                item["held"] = False
    data["assertions"]["held_tool_calls"] = []
    data["assertions"]["required_tool_calls"] = [
        "call_gated_early",
        "call_gated_release",
    ]
    unheld = tmp_path / "unheld.yaml"
    unheld.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    stage, step, sessions, session_key, step_dir = _gated_runtime(tmp_path)

    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=unheld,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=RecordingCommittedAdapter(),
            recorder=MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id="gate_c_evolv_05_gated"),
            writer=writer,
        )
        with pytest.raises(
            ForcedEventExecutionError, match="before the user has dictated"
        ):
            await executor.run_session()
    stage.finish_step(step, "discard")


@pytest.mark.asyncio
async def test_a_hold_with_no_gate_behind_it_is_refused(tmp_path: Path) -> None:
    """A `held` claim is checked against the live gate, never taken."""
    from dataclasses import replace

    stage, step, sessions, session_key, step_dir = _gated_runtime(tmp_path)
    step = replace(step, gate=None)

    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=GATED_TRANSCRIPT_PATH,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=RecordingCommittedAdapter(),
            recorder=MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id="gate_c_evolv_05_gated"),
            writer=writer,
        )
        with pytest.raises(
            ForcedEventExecutionError, match="nothing holds it"
        ):
            await executor.run_session()
    stage.finish_step(step, "discard")


ALL_TRANSCRIPTS = sorted(
    (PROJECT_ROOT / "Authoring/sessions/forced/transcripts/user_a").glob("*/transcript.yaml")
)


@pytest.mark.parametrize(
    "transcript_path", ALL_TRANSCRIPTS, ids=[p.parent.name for p in ALL_TRANSCRIPTS]
)
@pytest.mark.asyncio
async def test_every_forced_transcript_replays_against_the_world(
    tmp_path: Path, transcript_path: Path
) -> None:
    """Each committed transcript must run end to end on a fresh L0 world:
    ``$ref`` arguments resolve, delivered third-party messages are readable
    through the PA's own tools, and pinned read content matches."""
    from harness.benchmark_clock import benchmark_session_timestamps
    from scripts.outline_schema import CompactOutline

    data = yaml.safe_load(transcript_path.read_text(encoding="utf-8"))
    event_id = data["event"]["event_id"]
    acc_num = data["event"]["source_acc_num"]
    outline = CompactOutline.model_validate(
        yaml.safe_load(
            (PROJECT_ROOT / "Authoring/outline/drafts/user_a/outline.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    session_timestamp = benchmark_session_timestamps(
        {
            item.acc_num: {
                "time_position": item.story.time_position,
                "date": item.story.date,
                "hour": item.story.hour,
            }
            for item in outline.sessions
        }
    )[acc_num]
    run_id = f"replay_{event_id}"
    stage = StageRuntime(
        run_id=run_id,
        user_id="user_a",
        fixture_root=PROJECT_ROOT / "data/state_fixtures",
        runs_root=tmp_path / "runs",
        project_root=PROJECT_ROOT,
    )
    step = stage.begin_step(
        step_id=event_id,
        session_id=event_id,
        runtime_artifact_path=transcript_path,
        session_date=session_timestamp.date().isoformat(),
        session_timestamp=session_timestamp.isoformat(),
    )
    sessions = SessionManager(tmp_path / "pa_workspace")
    session_key = f"{run_id}_main"
    sessions.create(session_key)
    adapter = RecordingCommittedAdapter()
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=run_id)
    step_dir = tmp_path / "steps" / event_id
    with TranscriptWriter(step_dir) as writer:
        executor = ForcedEventExecutor(
            transcript_path=transcript_path,
            project_root=PROJECT_ROOT,
            benchmark_session_timestamp=session_timestamp,
            stage_runtime=step,
            session_manager=sessions,
            pa_session_key=session_key,
            memory_adapter=adapter,
            memory_io_recorder=recorder,
            writer=writer,
            require_committed_adapter_writes=True,
        )
        events = await executor.run_session()
    stage.finish_step(step, "commit")

    assert events[-1]["kind"] == "session_end"
    assert events[-1]["tool_calls"] == len(data["assertions"]["required_tool_calls"])
    delivered = [
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "environment_action" and item["source"]["kind"] == "third_party"
    ]
    for item in delivered:
        store_file = "email/inbox.jsonl" if item["channel"] == "email" else "message/threads.jsonl"
        assert item["content"] in (stage.canonical_stage / store_file).read_text(encoding="utf-8")


RESOLVED_GATED_TRANSCRIPT_PATH = (
    PROJECT_ROOT
    / "tests/fixtures/authoring/forced_event_transcript_evolv_01_gated.yaml"
)


def _fixture_project_root(tmp_path: Path) -> Path:
    """A project root whose user_a outline is the fixture the FE draft names.

    `resolve_session_gate` reads the persona's outline off the project root, and
    the worked FE fixtures are authored against `outline_acc_015.yaml`, not
    against the live user_a outline (whose acc_015 now declares a different
    gate). Same construction as
    `test_gate.py:test_resolve_gates_the_schema_written_forced_draft_fixture`.
    """
    root = tmp_path / "fixture_root"
    outline_dir = root / "Authoring/outline/drafts/user_a"
    outline_dir.mkdir(parents=True)
    (outline_dir / "outline.yaml").write_text(
        (PROJECT_ROOT / "tests/fixtures/authoring/outline_acc_015.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    for name in ("data", "tests"):
        (root / name).symlink_to(PROJECT_ROOT / name)
    # The transcript names its draft at the canonical Authoring path, as every
    # frozen transcript does; the worked draft fixture stands in for it here.
    draft_dir = root / "Authoring/sessions/forced/drafts/user_a"
    draft_dir.mkdir(parents=True)
    (draft_dir / "evolv_01.yaml").write_text(
        (PROJECT_ROOT / "tests/fixtures/authoring/forced_event_evolv_01.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    return root


@pytest.mark.asyncio
async def test_a_resolved_gate_holds_then_releases_the_authored_call(
    tmp_path: Path,
) -> None:
    """Draft -> transcript -> `harness/gate.py` resolver -> executor replay.

    The gate is not injected here: it is resolved from the transcript itself,
    which is the artifact the runtime freezes, through the outline entry of the
    acc the event sits on. That is the whole chain the transcript layer was
    missing -- before `task_facts` / `reveal_at` reached the transcript schema,
    this resolution raised on the real gated artifact.

    The PA composes the review draft's body before User A has dictated his
    caveat, so that call is held; the user's own wording opens the gate at the
    message bound to the fact, and the retry writes the world.
    """
    from harness.gate import resolve_session_gate

    fixture_root = _fixture_project_root(tmp_path)
    gate = resolve_session_gate(RESOLVED_GATED_TRANSCRIPT_PATH, fixture_root)
    assert gate is not None
    assert gate["tool"] == "email_draft_create"
    assert gate["task_fact"] == "reyes_handoff_caveat_sentence"

    run_id = "resolved_gate_evolv_01"
    stage = StageRuntime(
        run_id=run_id,
        user_id="user_a",
        fixture_root=PROJECT_ROOT / "data/state_fixtures",
        runs_root=tmp_path / "runs",
        project_root=PROJECT_ROOT,
    )
    step = stage.begin_step(
        step_id="evolv_01",
        session_id="evolv_01",
        runtime_artifact_path=RESOLVED_GATED_TRANSCRIPT_PATH,
        gate=gate,
    )
    sessions = SessionManager(tmp_path / "pa_workspace")
    session_key = f"{run_id}_main"
    sessions.create(session_key)
    adapter = RecordingCommittedAdapter()
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=run_id)
    step_dir = tmp_path / "steps" / "evolv_01"
    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=RESOLVED_GATED_TRANSCRIPT_PATH,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        events = await executor.run_session()
    state_artifacts = preserve_step_state_artifacts(step, step_dir)
    drafts = (step.state_root / "email/drafts.jsonl").read_text(encoding="utf-8")
    stage.finish_step(step, "commit")

    tool_events = [
        event["tool_events"][0] for event in events if event["kind"] == "pa_turn"
    ]
    assert [event["status"] for event in tool_events] == ["ok", "held", "ok", "ok"]
    assert tool_events[1]["call_id"] == "call_evolv_01_02_held"
    # Only the released write reached the world: the stored draft holds the
    # user's own wording, never the body the PA composed before he spoke.
    assert "the 3.3 notation is not settled" in drafts
    assert "It is in good shape." not in drafts
    tool_records = [
        json.loads(line)
        for line in Path(state_artifacts["tool_log"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [record["tool"] for record in tool_records] == [
        "gate_loaded",
        "document_read",
        "gate_satisfied",
        "email_draft_create",
        "email_draft_read",
    ]


def test_the_gated_transcript_binds_to_its_draft() -> None:
    """The freeze check: the transcript's gated values are the draft's own."""
    from harness.forced_event_schema import (
        ForcedEventTranscriptLoadError,
        load_forced_event_transcript,
        validate_gated_values_match_draft,
    )
    from scripts.session_authoring_schema import ForcedEventDraft

    transcript = load_forced_event_transcript(RESOLVED_GATED_TRANSCRIPT_PATH)
    draft = ForcedEventDraft.model_validate(
        yaml.safe_load(
            (
                PROJECT_ROOT / "tests/fixtures/authoring/forced_event_evolv_01.yaml"
            ).read_text(encoding="utf-8")
        )
    )
    validate_gated_values_match_draft(transcript, draft)
    assert transcript.reveal_at == draft.simulation.reveal_at

    drifted = transcript.model_copy(
        update={
            "task_facts": {
                key: f"{value} Rephrased."
                for key, value in transcript.task_facts.items()
            }
        }
    )
    with pytest.raises(ForcedEventTranscriptLoadError, match="task_facts"):
        validate_gated_values_match_draft(drifted, draft)


async def _replay_gated_transcript(
    tmp_path: Path,
    transcript_path: Path,
    *,
    run_id: str,
) -> tuple[list[dict], object, ForcedEventExecutor]:
    """Run one gated transcript through the whole chain: resolver -> executor.

    Same construction as
    `test_a_resolved_gate_holds_then_releases_the_authored_call`, returned
    rather than asserted so a test can look at the world, the events, or the
    reference results the replay collected.
    """
    from harness.gate import resolve_session_gate

    fixture_root = _fixture_project_root(tmp_path)
    gate = resolve_session_gate(transcript_path, fixture_root)
    stage = StageRuntime(
        run_id=run_id,
        user_id="user_a",
        fixture_root=PROJECT_ROOT / "data/state_fixtures",
        runs_root=tmp_path / "runs",
        project_root=PROJECT_ROOT,
    )
    step = stage.begin_step(
        step_id="evolv_01",
        session_id="evolv_01",
        runtime_artifact_path=transcript_path,
        gate=gate,
    )
    sessions = SessionManager(tmp_path / "pa_workspace")
    session_key = f"{run_id}_main"
    sessions.create(session_key)
    with TranscriptWriter(tmp_path / "steps" / "evolv_01") as writer:
        executor = _executor(
            transcript_path=transcript_path,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=RecordingCommittedAdapter(),
            recorder=MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=run_id),
            writer=writer,
        )
        events = await executor.run_session()
    return events, step, executor


def _with_task_fact_reference(tmp_path: Path) -> Path:
    """The gated fixture, with the released write's gated argument written as
    `$task_facts.<key>` instead of repeating the dictated value as a literal.

    This is the shape a real gated FE transcript arrives in: the generator
    copies every non-authored argument from the draft step verbatim, and a
    draft binds its gate as `gate_param: $task_facts.<task_fact>`, so the
    reference reaches the executor unrewritten.
    """
    data = yaml.safe_load(RESOLVED_GATED_TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    replaced = 0
    for exchange in data["exchanges"]:
        for item in exchange["items"]:
            if item.get("type") != "pa_tool_call" or item.get("held"):
                continue
            if "body" in item.get("arguments", {}):
                item["arguments"]["body"] = (
                    "$task_facts.reyes_handoff_caveat_sentence"
                )
                replaced += 1
    assert replaced == 1
    path = tmp_path / "forced_event_transcript_evolv_01_ref.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_a_task_facts_reference_resolves_after_the_reveal(
    tmp_path: Path,
) -> None:
    """`$task_facts.<key>` is a producer the reference table cannot hold.

    Before this, the executor resolved arguments against tool results alone, so
    a gated call written the way the draft writes it -- `body:
    $task_facts.reyes_handoff_caveat_sentence` -- died as a reference to a
    producer that had not run. It is not a producer: it is the value the user
    holds, and it resolves out of the transcript's own `task_facts`, through
    the same function the accumulation validator uses.
    """
    transcript_path = _with_task_fact_reference(tmp_path)
    events, step, _ = await _replay_gated_transcript(
        tmp_path, transcript_path, run_id="task_fact_ref_evolv_01"
    )
    tool_events = [
        event["tool_events"][0] for event in events if event["kind"] == "pa_turn"
    ]
    # The attempt before the reveal is still held; the retry after it carries
    # the user's dictated wording, resolved from the reference.
    assert [event["status"] for event in tool_events] == ["ok", "held", "ok", "ok"]
    released = [
        event
        for event in tool_events
        if event["name"] == "mcp_state_email_draft_create"
        and event["status"] == "ok"
    ]
    assert released[0]["args"]["body"].startswith("Section 3.2 is preliminary")
    assert "$task_facts" not in released[0]["args"]["body"]
    drafts = (step.state_root / "email/drafts.jsonl").read_text(encoding="utf-8")
    assert "the 3.3 notation is not settled" in drafts


def test_a_user_only_reveal_message_is_refused_by_the_validator(
    tmp_path: Path,
) -> None:
    """A `visibility: user` reveal would open the gate on words the PA never
    read, so it never loads."""
    from harness.forced_event_schema import (
        ForcedEventTranscriptLoadError,
        load_forced_event_transcript,
    )

    data = yaml.safe_load(RESOLVED_GATED_TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    hidden = 0
    for exchange in data["exchanges"]:
        for item in exchange["items"]:
            if item.get("type") == "message" and item.get("reveals"):
                item["visibility"] = "user"
                hidden += 1
    assert hidden == 1
    path = tmp_path / "user_only_reveal.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    with pytest.raises(ForcedEventTranscriptLoadError, match="PA-visible"):
        load_forced_event_transcript(path)


def test_a_user_only_reveal_message_is_refused_at_replay() -> None:
    """The executor's half of the same rule, on the code it actually runs.

    The schema refuses such a transcript before the executor sees it, so this
    exercises the replay routine directly -- the one the executor calls for
    every message, and the only thing between a gate and a reveal the PA was
    never shown if a transcript ever reaches it another way.
    """
    from harness.forced_event_replay import ForcedEventReplayError, GatedReplay

    replay = GatedReplay(
        gate={
            "tool": "document_update",
            "param": "replacement",
            "task_fact": "fact",
            "key_terms": ["crossover"],
        },
        task_facts={"fact": "the crossover depth is bounded"},
    )
    with pytest.raises(ForcedEventReplayError, match="PA-visible"):
        replay.observe_message(
            source="user",
            visibility="user",
            reveals=["fact"],
            content="the crossover depth is bounded",
        )
    assert not replay.gate_is_open()
    assert replay.observe_message(
        source="user",
        visibility="both",
        reveals=["fact"],
        content="the crossover depth is bounded",
    )
    assert replay.gate_is_open()


def test_a_forced_event_must_end_on_a_pa_visible_user_message(
    tmp_path: Path,
) -> None:
    """The user's stated rule is the event. Said out of the PA's hearing, it is
    not in the session every later probe measures."""
    from harness.forced_event_schema import (
        ForcedEventTranscriptLoadError,
        load_forced_event_transcript,
    )

    data = yaml.safe_load(RESOLVED_GATED_TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    last_message = [
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item.get("type") == "message"
    ][-1]
    last_message["visibility"] = "user"
    # The hidden rule moves in with the exchange before it: alone it would be an
    # exchange with no PA-visible text at all, which the shape rule catches
    # first, and this test is about the rule the PA never hears.
    hidden = data["exchanges"].pop()["items"]
    data["exchanges"][-1]["items"].extend(hidden)
    path = tmp_path / "hidden_rule.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    with pytest.raises(ForcedEventTranscriptLoadError, match="last message"):
        load_forced_event_transcript(path)


def test_environment_action_content_is_in_the_preview_as_a_tagged_block() -> None:
    """The pool is the PA's whole experience.

    A delivered third-party line is part of that experience -- the PA perceived
    it, and the person's next line answers it -- so it is in the preview. What
    it is never is the person's own speech, so it sits in a tagged block on an
    `assistant` record, not as `role: "user"` text.
    """
    from harness.forced_event_executor import projected_memory_records
    from harness.forced_event_schema import (
        EnvironmentActionItem,
        load_forced_event_transcript,
    )

    transcript = load_forced_event_transcript(TRANSCRIPT_PATH)
    items = [item for exchange in transcript.exchanges for item in exchange.items]
    actions = [item for item in items if isinstance(item, EnvironmentActionItem)]
    assert actions

    records = projected_memory_records(transcript)
    for action in actions:
        assert action.visibility in {"pa", "both"}
        carrying = [
            record for record in records if action.content in record["content"]
        ]
        assert len(carrying) == 1
        assert carrying[0]["role"] == "assistant"
        assert carrying[0]["content"].startswith("[third_party ")
        assert f"action_id={action.action_id}" in carrying[0]["content"]
    assert "Then write back and ask them which entrance stays open." in [
        record["content"] for record in records
    ]


def test_the_preview_carries_the_tool_calls_the_pool_will_carry() -> None:
    """What the review skill reads of a forced transcript, without a run.

    Under the memory pool the reviewer no longer reads a dialogue-only
    projection: the call the PA made is in it, as a `[tool_call]` block, and so
    is a placeholder for the result the world has not produced yet. The
    reviewer's remaining question is the spoken one -- whether the person says
    the boundary -- not whether tool text survives, because it now does.
    """
    from harness.forced_event_executor import projected_memory_records
    from harness.forced_event_schema import (
        PAToolCallItem,
        load_forced_event_transcript,
    )

    transcript = load_forced_event_transcript(RESOLVED_GATED_TRANSCRIPT_PATH)
    records = projected_memory_records(transcript)
    text = "\n".join(record["content"] for record in records)
    items = [item for exchange in transcript.exchanges for item in exchange.items]
    calls = [item for item in items if isinstance(item, PAToolCallItem)]

    assert calls
    assert {record["role"] for record in records} == {"user", "assistant"}
    # One record per PA-visible spoken item, and two per call: the declaration
    # and the not-executed placeholder.
    assert len(records) == 2 * len(calls) + len(
        [
            item
            for item in items
            if not isinstance(item, PAToolCallItem)
            and getattr(item, "visibility", None) in {"pa", "both"}
        ]
    )
    for call in calls:
        assert f"name={call.name}" in text
        assert f"id={call.call_id}" in text
    assert "\nnot executed\n[/tool_result]" in text

    # The reveal, and the rule the user states after it.
    assert "I only found that because I asked what was stored" in text
    assert "stop and ask me, then wait for my wording" in text


def test_a_third_party_message_is_in_the_preview_as_a_third_party_block(
    tmp_path: Path,
) -> None:
    """The other shape a third party speaks in reaches the preview too.

    A `MessageItem` whose `source` is a `SourceRef` is routed by that source's
    own `kind`, exactly as a delivered environment action is, so both shapes
    render as the same block and neither borrows the person's role.
    """
    from harness.forced_event_executor import projected_memory_records
    from harness.forced_event_schema import MessageItem, load_forced_event_transcript

    transcript = load_forced_event_transcript(
        _third_party_message_transcript(tmp_path)
    )
    items = [item for exchange in transcript.exchanges for item in exchange.items]
    third_party = [
        item
        for item in items
        if isinstance(item, MessageItem) and not isinstance(item.source, str)
    ]
    assert len(third_party) == 1
    assert third_party[0].visibility == "both"

    records = projected_memory_records(transcript)
    carrying = [
        record
        for record in records
        if third_party[0].content in record["content"]
    ]
    assert len(carrying) == 1
    assert carrying[0]["role"] == "assistant"
    assert carrying[0]["content"].startswith("[third_party ")
    assert "action_id" not in carrying[0]["content"].split("\n")[0]
    assert "That closure lands on the day you pinned." in [
        record["content"] for record in records
    ]


@pytest.mark.asyncio
async def test_a_third_party_message_reaches_the_adapter_as_a_tagged_block(
    tmp_path: Path,
) -> None:
    """The live half of the same rule, with the commit index kept whole.

    The third party's line is in `session.messages` for the PA's prompt and in
    the pool for the adapter, and in the pool it is a `[third_party]` block on
    an `assistant` message. Every exchange still commits:
    `adapter_committed_end_index` matches `len(session.messages)` and the run's
    checkpoint accepts the step.
    """
    transcript_path = _third_party_message_transcript(tmp_path)
    data = yaml.safe_load(transcript_path.read_text(encoding="utf-8"))
    third_party_content = next(
        item["content"]
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "message" and not isinstance(item["source"], str)
    )
    stage, step, sessions, session_key, adapter, recorder, step_dir = (
        _gate_c_runtime(tmp_path)
    )
    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=transcript_path,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        await executor.run_session()
    stage.finish_step(step, "commit")

    persisted = sessions.get(session_key)
    assert third_party_content in [
        message.get("content") for message in persisted.messages
    ]
    carrying = [
        message
        for event in adapter.events
        for message in event.messages
        if third_party_content in message["content"]
    ]
    assert len(carrying) == 1
    assert carrying[0]["role"] == "assistant"
    assert carrying[0]["content"].startswith("[third_party ")
    assert persisted.adapter_committed_end_index == len(persisted.messages)
    assert len(adapter.events) == _memory_visible_exchange_count(data)


def _with_realized_call_reference(tmp_path: Path) -> Path:
    """The fixture plus a call whose argument names an earlier realized id.

    `call_fixture_02` mints a `draft_id` the transcript cannot know
    (`state_server/store.py:1801`, `_mint_id`), and the added `call_fixture_03`
    reads it back through `$call_fixture_02.draft_id` (`harness/tool_refs.py`).
    That pairing is what makes a re-execution visible: a second run that
    executes `call_fixture_02` again mints a second draft, so the reference in
    the next exchange resolves to an id the first run never produced.
    """
    data = copy.deepcopy(yaml.safe_load(TRANSCRIPT_PATH.read_text(encoding="utf-8")))
    data["exchanges"].insert(
        len(data["exchanges"]) - 1,
        {
            "items": [
                {
                    "type": "pa_tool_call",
                    "offset_seconds": 170,
                    "call_id": "call_fixture_03",
                    "name": "mcp_state_email_draft_read",
                    "arguments": {"draft_id": "$call_fixture_02.draft_id"},
                },
                {
                    "type": "message",
                    "offset_seconds": 175,
                    "source": "pa",
                    "visibility": "both",
                    "content": "I read it back once so you know what is sitting there.",
                },
            ]
        },
    )
    data["assertions"]["required_tool_calls"].append("call_fixture_03")
    path = tmp_path / "realized_call_reference.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# The memory pool: the PA's whole first-person experience of an exchange.
# ---------------------------------------------------------------------------


def _third_party_and_environment_transcript(tmp_path: Path) -> Path:
    """The fixture with both shapes a non-user voice takes, in one exchange.

    An `environment_action` (a delivery the world made) and a `MessageItem`
    carrying a `SourceRef` source (a third party speaking straight into the
    conversation) are the two, and they are routed to different tags by their
    own `source.kind`. The fixture authors the first; the second is added here
    beside it so one exchange carries both, with distinct offsets.
    """
    data = copy.deepcopy(yaml.safe_load(TRANSCRIPT_PATH.read_text(encoding="utf-8")))
    action = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "environment_action"
    )
    exchange = next(
        exchange
        for exchange in data["exchanges"]
        if action in exchange["items"]
    )
    exchange["items"].insert(
        exchange["items"].index(action) + 1,
        {
            "type": "message",
            "offset_seconds": action["offset_seconds"] + 10,
            "source": {"kind": "environment", "id": "calendar_sync"},
            "visibility": "both",
            "content": "Calendar sync moved the Glenmont walk to the afternoon slot.",
        },
    )
    path = tmp_path / "third_party_and_environment.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _draft_records(state_root: Path) -> list[dict]:
    path = state_root / "email" / "drafts.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(
    "crash_point",
    ["after_preparation", "after_world_call_before_memory_ack"],
)
@pytest.mark.asyncio
async def test_resume_does_not_re_execute_a_realized_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_point: str
) -> None:
    """A resume may replay a memory write, never a world action.

    There are two interruption points, and both are here.

    `after_preparation` interrupts once every exchange is prepared and before
    any commit -- the point that matters, because "preparation of every
    exchange completes before any commit" -- so every world call of the run is
    realized and no memory write is.

    `after_world_call_before_memory_ack` interrupts inside `_prepare_exchange`
    after `call_fixture_02`'s realization is recorded. The interruption point is
    "after the world call but before the memory ack", and the realization is
    recorded on the line after the call --
    `self._replay.results[item.call_id] = actual_result`
    (`harness/forced_event_executor.py:347`, straight after
    `await registry.execute(...)` at `:345`). So the crash is tripped at the
    top of the NEXT registry call, `call_fixture_03`'s
    `mcp_state_email_draft_read`: at that instant the minted `draft_id` is in
    `self._replay.results` where a turn journal can persist it, the draft
    exists in the world, and `_commit_session_and_memory` (`:568`) has not run,
    so nothing has reached the adapter. Raising inside `registry.execute`
    itself would be a crash *before* `:347`, leaving nothing journaled and
    making the resume's "re-execute nothing" unsatisfiable.

    The two points differ in what the resume may legitimately do. After
    `after_preparation` every world call of the run is realized, so the resume
    calls the registry for nothing. After
    `after_world_call_before_memory_ack` the calls that never ran -- here
    `call_fixture_03` -- run for the first time on resume;
    what may never happen is a second execution of a call
    that was already realized.
    """
    transcript_path = _with_realized_call_reference(tmp_path)
    (
        stage,
        step,
        sessions,
        session_key,
        adapter,
        recorder,
        step_dir,
    ) = _gate_c_runtime(tmp_path)

    executed: list[str] = []
    original_execute = ToolRegistry.execute
    original_commit = ForcedEventExecutor._commit_session_and_memory
    armed = {"crash": True}

    async def _counting_execute(self, name, params):
        if (
            crash_point == "after_world_call_before_memory_ack"
            and armed["crash"]
            and name == "mcp_state_email_draft_read"
        ):
            # `call_fixture_02` is realized and recorded
            # (`harness/forced_event_executor.py:347`); this call has not
            # reached the world, so it is not counted as executed either.
            armed["crash"] = False
            raise RuntimeError("crash after the world call, before the memory ack")
        executed.append(name)
        return await original_execute(self, name, params)

    async def _crash_before_commit(self, transcript, prepared):
        if crash_point == "after_preparation" and armed["crash"]:
            armed["crash"] = False
            raise RuntimeError("crash after preparation, before any commit")
        return await original_commit(self, transcript, prepared)

    monkeypatch.setattr(ToolRegistry, "execute", _counting_execute)
    monkeypatch.setattr(
        ForcedEventExecutor,
        "_commit_session_and_memory",
        _crash_before_commit,
        raising=True,
    )
    first = _executor(
        transcript_path=transcript_path,
        step=step,
        sessions=sessions,
        session_key=session_key,
        adapter=adapter,
        recorder=recorder,
        writer=TranscriptWriter(step_dir / "first"),
    )
    with pytest.raises(RuntimeError):
        await first.run_session()

    realized = _draft_records(step.state_root)
    assert len(realized) == 1, "the interrupted run realized exactly one draft"
    calls_before_resume = list(executed)
    assert calls_before_resume, "the interrupted run reached the world"
    assert "mcp_state_email_draft_create" in calls_before_resume

    # The crash is spent -- `armed` is False -- so the resume runs the real
    # commit path with the same counting registry still installed.
    assert armed["crash"] is False
    resumed_adapter = RecordingCommittedAdapter()
    resumed = _executor(
        transcript_path=transcript_path,
        step=step,
        sessions=sessions,
        session_key=session_key,
        adapter=resumed_adapter,
        recorder=recorder,
        writer=TranscriptWriter(step_dir / "resumed"),
    )
    await resumed.run_session()

    # No realized call is executed a second time: the prefix is untouched, and
    # the only calls the resume makes are the ones the crash never reached.
    # `after_preparation` reached them all, so it makes none.
    assert executed[:len(calls_before_resume)] == calls_before_resume
    expected_tail = (
        [] if crash_point == "after_preparation" else ["mcp_state_email_draft_read"]
    )
    assert executed[len(calls_before_resume):] == expected_tail
    assert executed.count("mcp_state_email_draft_create") == 1

    # The world holds one draft, and the pool the resume commits carries that
    # same id -- both from the call that produced it and from the reference the
    # next exchange resolved against it.
    after = _draft_records(step.state_root)
    assert [record["draft_id"] for record in after] == [
        record["draft_id"] for record in realized
    ]
    draft_id = realized[0]["draft_id"]
    pool_text = "\n".join(
        message["content"]
        for event in resumed_adapter.events
        for message in event.messages
    )
    assert draft_id in pool_text


def _blocks(content: str) -> list[str]:
    """The opening delimiter of every block in one pool message."""
    return [
        line
        for line in content.split("\n")
        if line.startswith("[") and not line.startswith("[/") and line.endswith("]")
    ]


def test_third_party_environment_action_renders_as_third_party_block(
    tmp_path: Path,
) -> None:
    """The preview routes each non-user voice by its own `source.kind`.

    The fixture's delivery is a `third_party` environment action, so it renders
    as `[third_party]` and not as `[environment_event]` -- the tag follows the
    item's declared kind, never the call site that produced it.
    """
    from harness.forced_event_executor import projected_memory_records
    from harness.forced_event_schema import load_forced_event_transcript

    transcript = load_forced_event_transcript(
        _third_party_and_environment_transcript(tmp_path)
    )
    records = projected_memory_records(transcript)
    contents = [record["content"] for record in records]

    third_party = [
        content for content in contents if content.startswith("[third_party ")
    ]
    assert len(third_party) == 1
    assert "source_kind=third_party" in third_party[0]
    assert 'source_id="Glenmont Trail Office"' in third_party[0]
    assert "action_id=" in third_party[0]
    assert "closing the lower loop" in third_party[0]
    assert third_party[0].endswith("[/third_party]")
    # Never the person's own voice.
    assert all(
        record["role"] == "assistant"
        for record in records
        if record["content"].startswith("[third_party ")
    )


def test_environment_message_without_action_id_omits_the_attribute(
    tmp_path: Path,
) -> None:
    """`action_id=` is emitted only when the item has one.

    A `MessageItem` with a `SourceRef` source carries no `action_id`
    (`harness/forced_event_schema.py` `MessageItem`), so the attribute is
    absent -- it is not emitted empty.
    """
    from harness.forced_event_executor import projected_memory_records
    from harness.forced_event_schema import load_forced_event_transcript

    transcript = load_forced_event_transcript(
        _third_party_and_environment_transcript(tmp_path)
    )
    contents = [record["content"] for record in projected_memory_records(transcript)]
    environment = [
        content for content in contents if content.startswith("[environment_event ")
    ]
    assert len(environment) == 1
    assert "source_kind=environment" in environment[0]
    assert "source_id=calendar_sync" in environment[0]
    assert "action_id" not in environment[0]
    assert "Calendar sync moved" in environment[0]


def _environment_delivery_transcript(tmp_path: Path) -> Path:
    """The fixture's delivery, re-sourced as the world rather than a person.

    The fixture authors its `environment_action` as `third_party`. The same
    item with `source.kind: environment` is the shape ACC's own environment
    event takes, and is what this module's ACC-parity test needs.
    """
    data = copy.deepcopy(yaml.safe_load(TRANSCRIPT_PATH.read_text(encoding="utf-8")))
    action = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "environment_action"
    )
    action["source"] = {"kind": "environment", "id": "calendar_sync"}
    path = tmp_path / "environment_delivery.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_environment_action_reaches_the_pa_in_accs_role_and_wrapper(
    tmp_path: Path,
) -> None:
    """One environment event, one shape.

    ACC delivers an environment event to the PA as `role: "system"` carrying a
    body that opens with `[Environment event; no user request is currently
    active]` (`pa/agent/loop.py`, `process_environment_event` and the
    `sender_id == "environment"` arm of `_process_message`). The forced-event
    executor used to append its own environment delivery as `role: "user"` with
    the bare text, which tells the PA the *person* said it -- the same
    misattribution the memory side already fixed by routing on `source.kind`.

    The two expectations below are written as ACC's contract, not as a
    restatement of the helper, so this fails on the old executor rather than
    only on a missing import.
    """
    transcript_path = _environment_delivery_transcript(tmp_path)
    data = yaml.safe_load(transcript_path.read_text(encoding="utf-8"))
    action = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "environment_action"
    )
    stage, step, sessions, session_key, adapter, recorder, step_dir = (
        _gate_c_runtime(tmp_path)
    )
    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=transcript_path,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        await executor.run_session()
    stage.finish_step(step, "commit")

    persisted = sessions.get(session_key)
    carrying = [
        message
        for message in persisted.messages
        if action["content"] in (message.get("content") or "")
    ]
    assert len(carrying) == 1
    delivered = carrying[0]

    # ACC's contract, spelled out.
    assert delivered["role"] == "system"
    assert delivered["content"] == (
        "[Environment event; no user request is currently active]\n"
        f"{action['content']}"
    )

    # ...and the executor gets there by importing ACC's own definitions rather
    # than carrying a second copy of them, so the two cannot drift apart.
    assert delivered["role"] == ENVIRONMENT_EVENT_ROLE
    assert delivered["content"] == render_environment_event(action["content"])
    assert delivered["content"].startswith(ENVIRONMENT_EVENT_HEADER)


@pytest.mark.asyncio
async def test_environment_action_enters_memory_as_tagged_block_not_user_role(
    tmp_path: Path,
) -> None:
    """The live pool carries the delivered world line, never as the person.

    The pool is the PA's complete first-person
    experience, so what a third party said is in it -- but a third party's
    words are never attributed to the person, so the block sits on an
    `assistant` message.
    """
    transcript_path = _third_party_and_environment_transcript(tmp_path)
    data = yaml.safe_load(transcript_path.read_text(encoding="utf-8"))
    action = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "environment_action"
    )
    stage, step, sessions, session_key, adapter, recorder, step_dir = (
        _gate_c_runtime(tmp_path)
    )
    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=transcript_path,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        await executor.run_session()
    stage.finish_step(step, "commit")

    pool = [
        message for event in adapter.events for message in event.messages
    ]
    carrying = [
        message for message in pool if action["content"] in message["content"]
    ]
    assert len(carrying) == 1
    assert carrying[0]["role"] == "assistant"
    assert carrying[0]["content"].startswith("[third_party ")
    # The user's own line is the only `role: "user"` message in the pool, and
    # it keeps its exact bytes -- `simple_rag` compares them.
    user_lines = {
        item["content"]
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "message" and item["source"] == "user"
    }
    assert {
        message["content"] for message in pool if message["role"] == "user"
    } == user_lines
    persisted = sessions.get(session_key)
    assert persisted.adapter_committed_end_index == len(persisted.messages)


@pytest.mark.asyncio
async def test_the_live_pool_carries_tool_calls_results_and_both_source_tags(
    tmp_path: Path,
) -> None:
    """Everything the PA did and perceived in the exchange reaches the adapter.

    The dialogue stays byte-equal and untagged; the tool call, its result, the
    delivered third-party line and the environment line are each their own
    tagged block on an `assistant` message.
    """
    transcript_path = _third_party_and_environment_transcript(tmp_path)
    data = yaml.safe_load(transcript_path.read_text(encoding="utf-8"))
    calls = [
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "pa_tool_call"
    ]
    stage, step, sessions, session_key, adapter, recorder, step_dir = (
        _gate_c_runtime(tmp_path)
    )
    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=transcript_path,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        await executor.run_session()
    stage.finish_step(step, "commit")

    pool = [message for event in adapter.events for message in event.messages]
    tags = [tag for message in pool for tag in _blocks(message["content"])]
    assert sum(tag.startswith("[tool_call ") for tag in tags) == len(calls)
    assert sum(tag.startswith("[tool_result ") for tag in tags) == len(calls)
    assert sum(tag.startswith("[third_party ") for tag in tags) == 1
    assert sum(tag.startswith("[environment_event ") for tag in tags) == 1
    # Dialogue is bare: the person's text and the PA's replies keep their
    # bytes, so nothing untagged gained a character.
    spoken = {
        item["content"]
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "message" and isinstance(item["source"], str)
    }
    bare = {
        message["content"]
        for message in pool
        if not message["content"].startswith("[")
    }
    assert bare == spoken
    assert all(
        message["role"] in {"user", "assistant"} for message in pool
    )
    persisted = sessions.get(session_key)
    assert persisted.adapter_committed_end_index == len(persisted.messages)


@pytest.mark.asyncio
async def test_two_items_in_one_exchange_carry_distinct_offsets_on_the_wire(
    tmp_path: Path,
) -> None:
    """`offset=` is the per-item time, in the content every backend ingests.

    Offsets are relative to `benchmark_session_timestamp`, never to the
    exchange stamp, and a `[tool_call]` / `[tool_result]` inherits the offset
    of the item that produced it. Bare dialogue carries no tag and so no
    offset.
    """
    transcript_path = _third_party_and_environment_transcript(tmp_path)
    data = yaml.safe_load(transcript_path.read_text(encoding="utf-8"))
    action = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "environment_action"
    )
    environment_message = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "message" and not isinstance(item["source"], str)
    )
    call = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == "pa_tool_call"
    )
    assert action["offset_seconds"] != environment_message["offset_seconds"]
    stage, step, sessions, session_key, adapter, recorder, step_dir = (
        _gate_c_runtime(tmp_path)
    )
    with TranscriptWriter(step_dir) as writer:
        executor = _executor(
            transcript_path=transcript_path,
            step=step,
            sessions=sessions,
            session_key=session_key,
            adapter=adapter,
            recorder=recorder,
            writer=writer,
        )
        await executor.run_session()
    stage.finish_step(step, "commit")

    pool = [message for event in adapter.events for message in event.messages]
    tags = [tag for message in pool for tag in _blocks(message["content"])]
    third_party = next(tag for tag in tags if tag.startswith("[third_party "))
    environment = next(tag for tag in tags if tag.startswith("[environment_event "))
    assert third_party.endswith(f"offset={action['offset_seconds']}]")
    assert environment.endswith(
        f"offset={environment_message['offset_seconds']}]"
    )
    call_tags = [
        tag
        for tag in tags
        if tag.startswith(("[tool_call ", "[tool_result "))
        and f"id={call['call_id']}" in tag
    ]
    assert len(call_tags) == 2
    assert all(
        tag.endswith(f"offset={call['offset_seconds']}]") for tag in call_tags
    )
    assert not any(
        "offset=" in message["content"]
        for message in pool
        if not message["content"].startswith("[")
    )


def test_the_preview_of_a_landed_transcript_is_its_dialogue_plus_tool_blocks(
) -> None:
    """The five landed transcripts carry only dialogue and tool calls.

    So their preview is exactly the dialogue-only pool with a `[tool_call]` and
    a `[tool_result]` whose body is `not executed` spliced in at each call:
    nothing else in them has a tag.
    """
    from harness.forced_event_executor import projected_memory_records
    from harness.forced_event_schema import (
        EnvironmentActionItem,
        MessageItem,
        PAToolCallItem,
        load_forced_event_transcript,
    )
    from memory.event_writer import build_turn_pool

    landed = sorted(
        (PROJECT_ROOT / "Authoring/sessions/forced/transcripts/user_a").glob(
            "evolv_0*/transcript.yaml"
        )
    )
    assert len(landed) == 5
    for path in landed:
        transcript = load_forced_event_transcript(path)
        items = [item for exchange in transcript.exchanges for item in exchange.items]
        assert not [item for item in items if isinstance(item, EnvironmentActionItem)]
        calls = [item for item in items if isinstance(item, PAToolCallItem)]
        assert calls

        records = projected_memory_records(transcript)
        tagged = [
            record for record in records if record["content"].startswith("[")
        ]
        assert len(tagged) == 2 * len(calls)
        assert [
            record["content"].split("\n")[0].split(" ")[0].lstrip("[")
            for record in tagged
        ] == ["tool_call", "tool_result"] * len(calls)
        assert all(
            record["content"].endswith("\nnot executed\n[/tool_result]")
            for record in tagged
            if record["content"].startswith("[tool_result ")
        )

        dialogue = [
            {
                "role": "assistant" if item.source == "pa" else "user",
                "content": item.content,
            }
            for item in items
            if isinstance(item, MessageItem)
            and isinstance(item.source, str)
            and item.visibility in {"pa", "both"}
        ]
        assert [
            record for record in records if not record["content"].startswith("[")
        ] == build_turn_pool(dialogue).messages


@pytest.mark.asyncio
async def test_crash_between_pending_and_written_resumes_through_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memory write is replayed, the world action is not.

    The window this pins is the one the journal exists for: the pending record
    is durable, the adapter has acknowledged, and the process dies before
    `mark_written`. The record is then the only thing that knows the pool --
    `session.messages` cannot re-derive it -- so the resume reads it back and
    writes it again under the *same* turn id. What must not happen a second
    time is the world call behind it: the draft `call_fixture_02` minted stays
    one draft.
    """
    from types import SimpleNamespace

    from memory.event_writer import turn_journal_for
    from memory.turn_journal import PENDING, replay_pending_turns
    import harness.forced_event_executor as executor_module

    (
        stage,
        step,
        sessions,
        session_key,
        adapter,
        recorder,
        step_dir,
    ) = _gate_c_runtime(tmp_path)

    original_record = executor_module.record_turn_journal

    def _crash_before_written(**kwargs):
        if kwargs.get("state") == "written":
            raise RuntimeError("crash after the adapter ack, before mark_written")
        return original_record(**kwargs)

    monkeypatch.setattr(
        executor_module, "record_turn_journal", _crash_before_written
    )
    crashed = _executor(
        transcript_path=TRANSCRIPT_PATH,
        step=step,
        sessions=sessions,
        session_key=session_key,
        adapter=adapter,
        recorder=recorder,
        writer=TranscriptWriter(step_dir / "crashed"),
    )
    with pytest.raises(RuntimeError):
        await crashed.run_session()

    drafts_after_crash = _draft_records(step.state_root)
    assert len(drafts_after_crash) == 1, "the crashed run minted exactly one draft"

    journal = turn_journal_for(sessions, session_key)
    pending = journal.resume_pending()
    assert pending.turn_ids, "the pending record survived the crash"
    assert pending.turn_ids[0].startswith("evolv_05:exchange_")
    assert all(record.state == PENDING for record in pending.records)
    # The realized artefacts rode with it, so a resumed *executor* would skip
    # the world call -- and so the pool is replayable without one.
    assert any(record.realized["results"] for record in pending.records)

    resumed_adapter = RecordingCommittedAdapter()
    session = sessions.get(session_key)
    rewritten = await replay_pending_turns(
        journal=journal,
        adapter=resumed_adapter,
        scope=SimpleNamespace(
            run_id=step.run_id, persona_id="user_a", agent_id="nanobot"
        ),
        session=session,
    )

    # Same turn ids, same pools, and the records are marked written this time.
    assert rewritten == pending.turn_ids
    assert [event.turn_id for event in resumed_adapter.events] == pending.turn_ids
    assert [event.messages for event in resumed_adapter.events] == [
        record.pool.messages for record in pending.records
    ]
    assert not journal.pending_records()

    # The replay is a memory write and nothing else: the world still holds one
    # draft, and the mutation count did not move.
    assert _draft_records(step.state_root) == drafts_after_crash


# ---------------------------------------------------------------------------
# a mid-event compaction sweep against the turn journal
# ---------------------------------------------------------------------------


class _SaveMemoryProvider(LLMProvider):
    """Answers every consolidation call with one valid `save_memory` call."""

    def __init__(self, *, max_tokens: int = 256):
        super().__init__()
        self.generation = GenerationSettings(max_tokens=max_tokens)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self.calls.append(messages)
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments={
                        "history_entry": "[2026-12-08 18:30] forced event",
                        "memory_update": "long term",
                    },
                )
            ],
            finish_reason="tool_calls",
        )

    def get_default_model(self) -> str:
        return "fake-model"


@pytest.mark.asyncio
async def test_a_mid_event_sweep_consolidates_whole_journal_records_only(
    tmp_path: Path,
) -> None:
    """A sweep that stops mid-event must cut on a journal-record boundary.

    The consumption rule is per TURN, not per event: "automatic
    compaction consumes exactly the journal prefix whose turns end at or before
    the chosen `compaction_end_index`". So an event's exchanges being split
    across two sweeps is by design -- an FE exchange *is* a turn. What is not by
    design is a cut that lands inside one exchange, which is what this pins:

    A. every consolidated record's full span lies inside the consolidated prefix;
    B. every message in the consolidated prefix belongs to a consolidated record.

    The prompt estimate is stubbed rather than shrunk-window-driven because a
    real shrunk window sweeps straight to the last user turn and never exercises
    an interior boundary; the stub is the same device
    `tests/test_pa/test_memory_consolidator.py` uses, aimed at one round.
    """
    from memory.event_writer import turn_journal_for

    (
        stage,
        step,
        sessions,
        session_key,
        adapter,
        recorder,
        step_dir,
    ) = _gate_c_runtime(tmp_path)
    executor = _executor(
        transcript_path=TRANSCRIPT_PATH,
        step=step,
        sessions=sessions,
        session_key=session_key,
        adapter=adapter,
        recorder=recorder,
        writer=TranscriptWriter(step_dir),
    )
    await executor.run_session()

    session = sessions.get(session_key)
    journal = turn_journal_for(sessions, session_key)
    assert len(journal.records()) >= 3, "the fixture event is a multi-exchange one"

    provider = _SaveMemoryProvider()
    consolidator = MemoryConsolidator(
        workspace=tmp_path / "pa_workspace",
        pa_llm_provider=provider,
        pa_model="fake-pa-model",
        consolidation_llm_provider=provider,
        consolidation_model="fake-memory-model",
        session_manager=sessions,
        pa_context_window_tokens=1600,
        build_pa_prompt_messages=lambda history, current_message, **_: list(history),
        get_pa_tool_definitions=lambda: [],
        prompt_history_eviction_policy=(
            PromptHistoryEvictionPolicy.after_file_consolidation
        ),
        pa_max_completion_tokens=256,
    )

    # The boundary `select_compaction_end_index` would pick if the sweep had to
    # remove everything up to it: a `role: user` message strictly inside one
    # record's span. Derived from the transcript rather than hard-coded, so a
    # fixture edit moves it instead of silently disarming the test.
    interior_boundary = next(
        index
        for record in journal.records()
        for index in range(record.span[0] + 1, record.span[1])
        if session.messages[index].get("role") == "user"
    )
    target_tokens = consolidator.prompt_compaction_trigger_tokens() // 2
    over_budget = target_tokens + sum(
        estimate_message_tokens(message)
        for message in session.messages[:interior_boundary]
    )
    estimates = [over_budget]
    consolidator.estimate_session_prompt_tokens = lambda _session: (
        (estimates.pop(0), "stub") if estimates else (target_tokens, "stub")
    )

    # What the token arithmetic alone picks, before the record-boundary snap.
    # Recorded rather than asserted afterwards, because the snap is what the
    # cursors keep -- without this the test could pass on a sweep that never
    # reached the interior boundary at all.
    selected_boundaries: list[int] = []
    select_end_index = consolidator.select_compaction_end_index

    def _recording_select(*args, **kwargs):
        boundary = select_end_index(*args, **kwargs)
        if boundary is not None:
            selected_boundaries.append(boundary[0])
        return boundary

    consolidator.select_compaction_end_index = _recording_select

    await consolidator.maybe_compact_prompt_history(session)
    # Compaction marks its own handle on the same path; this is the documented
    # way the other one catches up (`memory/turn_journal.py:TurnJournal.reload`).
    journal.reload()

    # The sweep really did fire, really did stop inside the event, and the cut
    # it kept is not the raw one the token arithmetic asked for.
    assert provider.calls, "one consolidation call ran"
    assert selected_boundaries[:1] == [interior_boundary]
    consolidated_end_index = session.file_consolidated_end_index
    assert consolidated_end_index < interior_boundary
    # The head of the straddling record stays in the PA's prompt until its own
    # record is consolidated: never out of the prompt and not yet in the files.
    assert session.prompt_history_start_index == consolidated_end_index
    consolidated = [
        record for record in journal.records() if record.file_consolidated
    ]
    assert consolidated, "the sweep consolidated part of the event"
    assert len(consolidated) < len(journal.records()), "and stopped mid-event"

    # A. no journal record is half in the prefix and half out of it.
    straddling = [
        record.turn_id
        for record in consolidated
        if record.span[1] > consolidated_end_index
    ]
    assert straddling == [], (
        "a consolidated record reaches past the consolidated prefix: "
        f"{straddling}"
    )

    # B. no message is inside the prefix without a consolidated record of its own.
    covered = {
        index for record in consolidated for index in range(*record.span)
    }
    orphaned = sorted(set(range(consolidated_end_index)) - covered)
    assert orphaned == [], (
        "messages sit inside the consolidated prefix with no consolidated "
        f"record: {orphaned}"
    )


# ---------------------------------------------------------------------------
# A transcript's `task_facts` are text by contract -- the generator writes them
# as `str(value)` and the schema compares the same stringification, because the
# gate matches key terms against that text. A tool parameter has a declared JSON
# type. `GatedReplay.resolve_arguments` is where the two meet, for the offline
# pin and the live run alike, so the catalog's type is applied there.
# ---------------------------------------------------------------------------


_FAKE_PARAMETER_TYPES = {
    "reimbursement_submission_create": {
        "purpose": "string",
        "amount": "number",
        "copies": "integer",
        "urgent": "boolean",
        "attachments": "array",
    }
}


def test_a_task_fact_is_coerced_to_the_catalogs_declared_type(monkeypatch) -> None:
    from harness import forced_event_replay
    from harness.forced_event_replay import GatedReplay

    monkeypatch.setattr(
        forced_event_replay,
        "_catalog_parameter_types",
        lambda: _FAKE_PARAMETER_TYPES,
    )
    replay = GatedReplay(
        gate=None,
        task_facts={
            "purpose": "Desk lamp",
            "amount": "38.5",
            "whole": "3",
            "copies": "2",
            "urgent": "true",
            "calm": "false",
        },
    )
    resolved = replay.resolve_arguments(
        call_id="call_01",
        arguments={
            "purpose": "$task_facts.purpose",
            "amount": "$task_facts.amount",
            "copies": "$task_facts.copies",
            "urgent": "$task_facts.urgent",
        },
        tool="mcp_state_reimbursement_submission_create",
    )
    assert resolved["amount"] == 38.5 and isinstance(resolved["amount"], float)
    assert resolved["copies"] == 2 and isinstance(resolved["copies"], int)
    assert resolved["urgent"] is True
    # A string parameter is handed over exactly as the user said it.
    assert resolved["purpose"] == "Desk lamp"

    # `number` round-trips the draft's own type: '3' was an int.
    whole = replay.resolve_arguments(
        call_id="call_02",
        arguments={"amount": "$task_facts.whole"},
        tool="mcp_state_reimbursement_submission_create",
    )
    assert whole["amount"] == 3 and isinstance(whole["amount"], int)

    false = replay.resolve_arguments(
        call_id="call_03",
        arguments={"urgent": "$task_facts.calm"},
        tool="mcp_state_reimbursement_submission_create",
    )
    assert false["urgent"] is False


def test_without_a_tool_name_nothing_is_coerced(monkeypatch) -> None:
    from harness import forced_event_replay
    from harness.forced_event_replay import GatedReplay

    monkeypatch.setattr(
        forced_event_replay,
        "_catalog_parameter_types",
        lambda: _FAKE_PARAMETER_TYPES,
    )
    replay = GatedReplay(gate=None, task_facts={"amount": "38.5"})
    assert replay.resolve_arguments(
        call_id="call_01", arguments={"amount": "$task_facts.amount"}
    ) == {"amount": "38.5"}


def test_text_that_will_not_convert_is_left_for_the_store_to_refuse() -> None:
    from harness.forced_event_replay import _coerced

    # Mangling it would hide the defect; the store refuses it by name instead.
    assert _coerced("thirty-eight fifty", "number") == "thirty-eight fifty"
    assert _coerced("maybe", "boolean") == "maybe"
    assert _coerced("1.5", "integer") == "1.5"
    # Non-strings and the container types are never touched.
    assert _coerced(38.5, "number") == 38.5
    assert _coerced(["a.pdf"], "array") == ["a.pdf"]
    assert _coerced({"k": "v"}, "object") == {"k": "v"}
    assert _coerced("Desk lamp", "string") == "Desk lamp"
