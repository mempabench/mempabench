"""The wrapper's render block has to survive the trip to the artifacts.

The simulator emits `pa_turn` **without** the PA's `OutboundMessage.metadata`
— by design, it must never read `ix_render` — so without the harness merge the
block exists only inside the PA and reaches no artifact at all. That is exactly
what live run acc003_simgemini_005 showed: 8/8 turns rendered, every transcript
`pa_turn` missing `metadata`, `pa_toolcalls.json` `beats[*].ix_render == []`.

    PYTHONPATH=. python -m pytest tests/test_harness/test_ix_render_transcript_merge.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from harness.judge_pack import build_text_judge_pack, scan_banned_keys
from harness.pa_factory import RecordingPA, ix_render_records_from_loop
from harness.transcript import (
    TranscriptWriter,
    extract_pa_toolcalls,
    load_events,
    merge_ix_render_into_transcript,
    write_pa_toolcalls,
)
from pa.agent.ix.render_types import IX_CONTRACT_VERSION, IX_RENDER_SCHEMA_VERSION
from pa.bus.events import OutboundMessage
from simulator.loop import SimulatorLoop
from simulator.output import SimulatorOutput, TurnAssessment
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

HELD_BODY = "HELD-BODY-MARKER the deposit is not paid yet."


def _ix_render(turn: int) -> dict:
    return {
        "schema_version": IX_RENDER_SCHEMA_VERSION,
        "ix_contract_version": IX_CONTRACT_VERSION,
        # Closed: this fixture is about the merge and the text face, and it is
        # the same PA turn on every beat. An OPEN episode is the harness's
        # `has_pending_interaction`, which correctly keeps the beat open for the
        # user's answer — pacing this fixture is not what these tests assert.
        # `held` still carries the body, so the text face still has to strip it.
        "episodes": {
            "topic_management": {
                "episode_id": f"bs{turn}",
                "rounds_open": 1,
                "closed": True,
            }
        },
        "released": {"batches": [f"b{turn}"], "questions": [], "plan_executed": []},
        "held": {
            "batches": [{"batch_id": "b9", "label": "hotel", "body": HELD_BODY}],
            "questions": [],
            "plan_remaining": [],
            "calls": [],
            "cancelled": [],
            "outreach_message": "",
        },
        "gate": None,
        "consent": [],
        # The judge doc's M count reads these two off the block.
        "notices": [
            {"order": 0, "tool": "document_read", "text": f"Checking note {turn}."}
        ],
        "completion": {"released": True, "text": f"Done with {turn}."},
        "reasoning": {
            "shown": False,
            "source": "user_facing_rationale",
            "provider_reasoning_present": False,
        },
        "certainties": [],
        "proactive_decision": None,
        "trace": [f"topic_management -> release {turn}"],
    }


class _ScriptedProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "scripted-model"

    async def chat(self, *args, **kwargs) -> LLMResponse:
        raise AssertionError("the scripted backend never calls the provider")


class _ScriptedBackend:
    name = "scripted"

    def __init__(self, messages: list[str]) -> None:
        self.messages = list(messages)

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        return SimulatorOutput(
            message=self.messages.pop(0),
            turn_assessment=TurnAssessment(next_beat="advance"),
            raw_response="scripted",
        )

    async def drain_memory_tasks(self) -> None:
        return None


class _RenderingPA:
    """A PA whose outbound carries `ix_render`, like the real one does."""

    def __init__(self) -> None:
        self.turn = 0

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
        # `RecordingPA` pins the model-visible chat identity, so the real PA's
        # two routing kwargs always arrive.
        channel: str = "benchmark",
        chat_id: str = "main",
    ) -> OutboundMessage:
        self.turn += 1
        return OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=f"PA reply {self.turn}",
            metadata={
                "tool_events": [
                    {"name": "IX_render_process_visibility", "status": "ok", "args": {}},
                    {"name": "mcp_state_document_read", "status": "ok", "args": {}},
                ],
                "ix_render": _ix_render(self.turn),
            },
        )


def _script() -> SessionScript:
    return SessionScript(
        session_id="merge_check",
        scenario="merge check",
        context="personal",
        task="Get the thing resolved.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(beat="open", goal="Start it.", active_skills=[]),
            Beat(beat="wrap", goal="Finish it.", active_skills=[]),
        ],
    )


def _run_session(tmp_path: Path, *, merge: bool = True) -> Path:
    """Drive a real SimulatorLoop through a RecordingPA, then merge."""
    run_dir = tmp_path / "step"

    async def scenario() -> None:
        pa = RecordingPA(_RenderingPA())
        with TranscriptWriter(run_dir) as writer:
            loop = SimulatorLoop(
                _ScriptedProvider(),
                PersonaIdentity(name="User A", speech_style={}),
                PreferenceMatrix.from_yaml(
                    "data/personas/user_a_preferences.yaml"
                ),
                _script(),
                pa,
                backend=_ScriptedBackend(["open turn", "wrap turn"]),
                transcript_writer=writer,
            )
            await loop.run_session()

        if not merge:
            return
        # The simulator wrote nothing about ix_render; the harness puts it back.
        report = merge_ix_render_into_transcript(
            run_dir / "transcript.jsonl",
            ix_render_records_from_loop(loop),
        )
        assert report["pa_turns"] == report["records"] == report["merged"] == 2

    asyncio.run(scenario())
    return run_dir


def test_the_simulator_alone_records_no_ix_render(tmp_path: Path) -> None:
    """The bug this merge exists for, reproduced: skip the merge and it is gone.

    This is the live-run symptom exactly — `pa_turn` events with no `metadata`
    at all — and it is also the reason the merge cannot move into the
    simulator: the simulator stays blind to `ix_render` on purpose.
    """
    run_dir = _run_session(tmp_path, merge=False)

    events = load_events(run_dir / "transcript.jsonl")
    pa_turns = [event for event in events if event["kind"] == "pa_turn"]

    assert len(pa_turns) == 2
    assert all("metadata" not in event for event in pa_turns)
    assert extract_pa_toolcalls(run_dir / "transcript.jsonl")["beats"][0][
        "ix_render"
    ] == []


def test_transcript_pa_turn_carries_the_merged_ix_render(tmp_path: Path) -> None:
    run_dir = _run_session(tmp_path)

    events = load_events(run_dir / "transcript.jsonl")
    pa_turns = [event for event in events if event["kind"] == "pa_turn"]

    assert len(pa_turns) == 2
    for index, event in enumerate(pa_turns, start=1):
        blob = event["metadata"]["ix_render"]
        assert blob["schema_version"] == IX_RENDER_SCHEMA_VERSION
        assert blob["ix_contract_version"] == IX_CONTRACT_VERSION
        # Turn order, not content matching: the n-th record is the n-th turn.
        assert blob["released"]["batches"] == [f"b{index}"]


def test_pa_toolcalls_beats_carry_the_ix_render_records(tmp_path: Path) -> None:
    run_dir = _run_session(tmp_path)

    records = extract_pa_toolcalls(run_dir / "transcript.jsonl")
    rendered = [
        blob for beat in records["beats"] for blob in beat["ix_render"]
    ]

    assert len(rendered) == 2
    # The judge doc's M count for process_visibility reads notices/completion
    # off this block; both survive the trip.
    assert [notice["text"] for blob in rendered for notice in blob["notices"]] == [
        "Checking note 1.",
        "Checking note 2.",
    ]
    assert [blob["completion"]["text"] for blob in rendered] == [
        "Done with 1.",
        "Done with 2.",
    ]
    written = write_pa_toolcalls(run_dir / "transcript.jsonl")
    assert json.loads(written.read_text(encoding="utf-8")) == records


def test_the_text_face_still_strips_the_merged_block(tmp_path: Path) -> None:
    run_dir = _run_session(tmp_path)

    pack = build_text_judge_pack(load_events(run_dir / "transcript.jsonl"))

    assert scan_banned_keys(pack) == []
    text = json.dumps(pack, ensure_ascii=False)
    assert "ix_render" not in text
    assert HELD_BODY not in text
    assert "IX_render_process_visibility" not in text
    # What the text face keeps: the reply and the real tool call.
    pa_turns = [turn for turn in pack["turns"] if turn["kind"] == "pa_turn"]
    assert [turn["public_turn"] for turn in pa_turns] == ["PA reply 1", "PA reply 2"]
    assert [
        event["name"] for turn in pa_turns for event in turn["task_tool_events"]
    ] == ["mcp_state_document_read", "mcp_state_document_read"]


# --- forced-event steps ------------------------------------------------------


def _forced_step(tmp_path: Path) -> Path:
    """A forced step's transcript: authored `pa_turn`s, no wrapper anywhere."""
    run_dir = tmp_path / "forced"
    with TranscriptWriter(run_dir) as writer:
        writer.log_event({"t": 0, "kind": "session_start", "session_id": "fe1"})
        for index in (1, 2):
            writer.log_event(
                {
                    "t": index * 10,
                    "kind": "pa_turn",
                    "forced_event": True,
                    "exchange": 0,
                    "outbound": "",
                    "tool_events": [
                        {
                            "call_id": f"c{index}",
                            "name": "mcp_state_document_read",
                            "status": "ok",
                            "args": {"path": "trip.md"},
                            "detail": "trip contents",
                        }
                    ],
                }
            )
        writer.log_event({"t": 99, "kind": "session_end"})
    return run_dir


def test_a_forced_step_is_not_a_merge_mismatch(tmp_path: Path) -> None:
    """Forced events replay authored text and never call the PA LLM, so their
    `pa_turn`s carry no record and must not be counted as wrapper turns."""
    run_dir = _forced_step(tmp_path)

    report = merge_ix_render_into_transcript(run_dir / "transcript.jsonl", [])

    assert report == {"pa_turns": 0, "records": 0, "merged": 0}
    # Which is what `harness/session_executor.py` reads to decide whether to
    # record `ix_render_merge` in the step's meta.
    assert report["records"] == report["pa_turns"]


def test_wrapper_turns_with_no_records_are_reported_as_a_mismatch(
    tmp_path: Path,
) -> None:
    """The bug: the empty-records early return claimed `pa_turns: 0` whatever
    the transcript held, so a session whose records went missing compared
    `0 != 0` and looked like a PA that simply never rendered."""
    run_dir = _run_session(tmp_path, merge=False)

    report = merge_ix_render_into_transcript(run_dir / "transcript.jsonl", [])

    assert report == {"pa_turns": 2, "records": 0, "merged": 0}
    assert report["records"] != report["pa_turns"]


def test_forced_turns_do_not_shift_the_records_of_a_mixed_transcript(
    tmp_path: Path,
) -> None:
    """Alignment is positional, so a forced turn in the middle would consume a
    record that belongs to the wrapper turn after it."""
    run_dir = _run_session(tmp_path, merge=False)
    path = run_dir / "transcript.jsonl"
    events = load_events(path)
    index = next(i for i, event in enumerate(events) if event["kind"] == "pa_turn")
    events.insert(
        index,
        {"t": 0, "kind": "pa_turn", "forced_event": True, "outbound": "", "tool_events": []},
    )
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    report = merge_ix_render_into_transcript(
        path, [{"ix_render": _ix_render(1)}, {"ix_render": _ix_render(2)}]
    )

    assert report == {"pa_turns": 2, "records": 2, "merged": 2}
    wrapper_turns = [
        event
        for event in load_events(path)
        if event["kind"] == "pa_turn" and not event.get("forced_event")
    ]
    assert [
        event["metadata"]["ix_render"]["released"]["batches"] for event in wrapper_turns
    ] == [["b1"], ["b2"]]
    forced = next(
        event
        for event in load_events(path)
        if event["kind"] == "pa_turn" and event.get("forced_event")
    )
    assert "metadata" not in forced


# --- the rewrite is atomic ---------------------------------------------------


def test_a_crash_mid_merge_leaves_the_transcript_intact(
    tmp_path: Path, monkeypatch
) -> None:
    """The merge runs at every step boundary, before the checkpoint is
    published, and the transcript is that step's only audit copy. A rewrite in
    place would leave it half-written; the swap either happens or does not."""
    run_dir = _run_session(tmp_path, merge=False)
    path = run_dir / "transcript.jsonl"
    original = path.read_bytes()

    def _boom(_src, _dst):
        raise OSError("disk went away")

    monkeypatch.setattr("harness.transcript.os.replace", _boom)
    with pytest.raises(OSError, match="disk went away"):
        merge_ix_render_into_transcript(
            path, [{"ix_render": _ix_render(1)}, {"ix_render": _ix_render(2)}]
        )

    assert path.read_bytes() == original
    assert load_events(path)  # still parseable, not a half line


def test_the_merge_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    run_dir = _run_session(tmp_path)

    assert [p.name for p in run_dir.glob("transcript.jsonl*")] == [
        "transcript.jsonl"
    ]
