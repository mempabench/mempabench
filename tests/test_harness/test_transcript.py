"""Unit tests for harness/transcript.py — JSONL writer + Markdown renderer.

Pure file I/O, no LLM. Run:
    PYTHONPATH=. python -m pytest tests/test_harness/test_transcript.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.session_executor import _count_tool_calls
from harness.transcript import (
    TranscriptWriter,
    extract_pa_toolcalls,
    load_events,
    render_markdown,
    write_pa_toolcalls,
    write_markdown,
    write_meta,
)
from pa.agent.ix.events import IX_SELECTION_PAYLOAD_TYPE
from pa.agent.ix.render_types import (
    IX_CONTRACT_VERSION,
    IX_RENDER_SCHEMA_VERSION,
    IX_RENDER_OPTIONAL_KEYS,
    IX_RENDER_TOP_LEVEL_KEYS,
    IXContractVersionError,
    from_ix_render_metadata,
)


def _synthetic_session_events() -> list[dict]:
    """A 1-beat session with 1 simulator+PA pair, plus session bookends."""
    return [
        {
            "t": 0, "kind": "session_start",
            "persona": "User A", "session_id": "session_01",
            "context": "personal", "scenario": "email_drafting",
            "model": "test-model", "beat_names": ["open"], "session_key": "sim:test",
        },
        {
            "t": 1,
            "kind": "beat_enter",
            "beat": "open",
            "active_skills": ["topic_management"],
            "active_ix_tools": ["IX_topic_management"],
        },
        {
            "t": 2, "kind": "simulator_turn", "beat": "open",
            "message": "Draft an email about the elevator only.",
            "factual_check": [],
            "turn_assessment": {"next_beat": "stay"},
            "parse_warnings": [],
        },
        {
            "t": 3, "kind": "pa_turn", "beat": "open",
            "outbound": "Sure, I'll draft that email now.",
            "tool_events": [],
            "ix_required": ["IX_topic_management"],
            "ix_called": [],
            "ix_missing": ["IX_topic_management"],
            "chat_id": "direct", "empty": False,
        },
        {"t": 4, "kind": "session_end", "simulator_turns": 1, "beat_count": 1},
    ]


def test_writer_appends_jsonl_and_creates_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "test-run" / "user_a" / "session_01"
    with TranscriptWriter(run_dir) as tw:
        for evt in _synthetic_session_events():
            tw.log_event(evt)

    assert run_dir.exists()
    jsonl = run_dir / "transcript.jsonl"
    assert jsonl.exists()

    lines = [ln for ln in jsonl.read_text().splitlines() if ln.strip()]
    assert len(lines) == 5
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["kind"] == "session_start"
    assert parsed[2]["turn_assessment"]["next_beat"] == "stay"
    assert "emotion_event" not in parsed[2]


def test_writer_flushes_each_event(tmp_path: Path) -> None:
    """Even without close(), the JSONL file should have all events visible
    (each log_event flushes — survives crash)."""
    run_dir = tmp_path / "session"
    tw = TranscriptWriter(run_dir)
    tw.log_event({"t": 0, "kind": "session_start", "persona": "X"})
    tw.log_event({"t": 1, "kind": "session_end", "simulator_turns": 0, "beat_count": 0})
    # Read BEFORE close to verify flushing
    events = load_events(tw.jsonl_path)
    assert len(events) == 2
    tw.close()


def test_a_session_with_no_retries_still_writes_an_empty_attempts_file(
    tmp_path: Path,
) -> None:
    """`acc_004`, `acc_007` and `acc_019` had no
    `simulator_validation_attempts.jsonl` at all, so a read could not tell an
    unretried session from an unrecorded one."""
    run_dir = tmp_path / "session"
    with TranscriptWriter(run_dir) as tw:
        tw.log_event({"t": 0, "kind": "session_start"})

    assert tw.validation_attempts_path.is_file()
    assert tw.validation_attempts_path.read_text(encoding="utf-8") == ""


def test_reopening_a_writer_does_not_truncate_recorded_attempts(
    tmp_path: Path,
) -> None:
    """The executor opens a second writer on the same directory to append its
    own records, and the file it finds there is not blanked."""
    run_dir = tmp_path / "session"
    with TranscriptWriter(run_dir) as tw:
        tw.log_validation_attempt({"attempt": 1, "error_codes": ["style"]})

    with TranscriptWriter(run_dir):
        pass

    assert load_events(tw.validation_attempts_path) == [
        {"attempt": 1, "error_codes": ["style"]}
    ]


def test_load_events_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "transcript.jsonl"
    p.write_text('{"t":0,"kind":"a"}\n\n{"t":1,"kind":"b"}\n\n')
    events = load_events(p)
    assert [e["kind"] for e in events] == ["a", "b"]


def test_render_markdown_contains_expected_sections(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    with TranscriptWriter(run_dir) as tw:
        for evt in _synthetic_session_events():
            tw.log_event(evt)

    md = render_markdown(run_dir / "transcript.jsonl")

    # Header
    assert "# Session: session_01" in md
    assert "**User A**" in md
    assert "`personal`" in md
    # Beat heading
    assert "## Beat: `open`" in md
    assert "topic_management" in md
    assert "**[Active IX tools]** `IX_topic_management`" in md
    # Visible turn pair
    assert "[Simulator → PA]" in md
    assert "[PA → Simulator]" in md
    assert "Draft an email about the elevator only." in md
    assert "Sure, I'll draft that email now." in md
    # Eval-only collapsible
    assert "<details><summary>eval-only (not visible to PA)</summary>" in md
    assert "next_beat" in md
    # Session end footer
    assert "**Session end**" in md
    assert "simulator turns: 1" in md


def test_render_markdown_includes_tool_and_state_audit(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    with TranscriptWriter(run_dir) as tw:
        for evt in _synthetic_session_events():
            tw.log_event(evt)
    state_dir = run_dir / "state"
    state_dir.mkdir()
    (state_dir / "tool_log.jsonl").write_text(
        json.dumps({
            "t": 1,
            "tool": "email_draft_create",
            "args": {"to": "management@example.test", "subject": "Elevator"},
            "result_summary": {"draft_id": "draft_0001", "status": "saved"},
            "status": "ok",
        }) + "\n",
        encoding="utf-8",
    )
    (state_dir / "state_diff.jsonl").write_text(
        json.dumps({
            "t": 1,
            "namespace": "email.drafts",
            "op": "append",
            "id": "draft_0001",
            "summary": "Saved draft: Elevator",
        }) + "\n",
        encoding="utf-8",
    )

    md = render_markdown(run_dir / "transcript.jsonl")
    assert "## Tool And State Audit" in md
    assert "`email_draft_create`" in md
    assert "`email.drafts` append `draft_0001`" in md


def test_render_markdown_includes_validation_and_session_end_reflection(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    events = [
        {
            "t": 0,
            "kind": "simulator_validation_failure",
            "phase": "simulator_turn",
            "attempt": 1,
            "error_codes": ["message_block_missing"],
        },
        {
            "t": 1,
            "kind": "session_end_reflection",
            "reflection": "That went better than I expected.",
            "final_emotion": {
                "trigger": "PA delivered the final reply",
                "reaction": "relieved",
            },
            "self_report": {
                "felt_understood_in_this_context": 4,
                "shift_perception": "no_shift_here",
                "felt_effort": 2,
                "noticed_contradictions": "None noticed.",
                "overall_experience": 4,
            },
        },
    ]
    transcript.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    md = render_markdown(transcript)

    assert "Simulator output validation failed" in md
    assert "message_block_missing" in md
    assert "## Session-end self-report" in md
    assert "That went better than I expected." in md
    assert "- trigger: PA delivered the final reply" in md
    assert "- reaction: relieved" in md
    assert "- felt_understood_in_this_context: 4" in md
    assert "- shift_perception: no_shift_here" in md
    assert "- overall_experience: 4" in md


def test_render_markdown_aligns_tool_events_with_pa_turn(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["tool_events"] = [
        {
            "name": "mcp_state_email_draft_create",
            "status": "error",
            "args": {"subject": "Elevator"},
            "detail": "{'draft_id': 'draft_0001'}",
            "error": {"code": "invalid_type", "message": "bad subject"},
        }
    ]
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    md = render_markdown(run_dir / "transcript.jsonl")
    assert "**[PA task/tool calls]**" in md
    assert "`mcp_state_email_draft_create`" in md
    assert "subject" in md
    assert md.index("**[PA task/tool calls]**") < md.index("**[PA → Simulator]**")

    audit = extract_pa_toolcalls(run_dir / "transcript.jsonl")
    call = audit["beats"][0]["calls"][0]
    assert call["status"] == "error"
    assert call["error"] == {"code": "invalid_type", "message": "bad subject"}


def test_render_markdown_separates_ix_audit_from_task_tool_calls(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["tool_events"] = [
        {
            "name": "IX_topic_management",
            "status": "ok",
            "args": {"setting": "One-at-a-time"},
            "detail": '{"type":"ix_preference_selection"}',
            "ix": True,
            "attribute": "topic_management",
            "setting": "One-at-a-time",
            "evidence": "User asked to keep the email about one issue only.",
            "application": "Keep the draft scoped to elevator repair.",
        },
        {
            "name": "mcp_state_email_draft_create",
            "status": "ok",
            "args": {"subject": "Elevator"},
            "detail": "{'draft_id': 'draft_0001'}",
        },
    ]
    events[3]["ix_called"] = ["IX_topic_management"]
    events[3]["ix_missing"] = []
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    md = render_markdown(run_dir / "transcript.jsonl")
    assert "**[IX audit]**" in md
    assert "- required: `IX_topic_management`" in md
    assert "- called: `IX_topic_management`" in md
    assert "- missing: (none)" in md
    assert "**[IX selections]**" in md
    assert "`topic_management`: `One-at-a-time`" in md
    assert "Keep the draft scoped to elevator repair." in md
    assert "**[PA task/tool calls]**" in md
    assert "`mcp_state_email_draft_create`" in md
    assert md.index("**[IX selections]**") < md.index("**[PA task/tool calls]**")


def test_a_render_call_is_not_listed_as_an_empty_selection(tmp_path: Path) -> None:
    """acc_011 printed `reasoning_visibility: ``` on every render turn.

    `IX_render_*` carries a payload, not a setting, so it has no selection row.
    """
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["tool_events"] = [
        {
            "name": "IX_render_topic_management",
            "status": "ok",
            "args": {"batches": []},
            "detail": '{"type":"ix_render_payload"}',
            "ix": True,
            "attribute": "topic_management",
            "setting": "",
            "evidence": "",
            "application": "",
        },
    ]
    events[3]["ix_called"] = ["IX_render_topic_management"]
    events[3]["ix_missing"] = []
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    md = render_markdown(run_dir / "transcript.jsonl")
    assert "**[IX audit]**" in md
    assert "`topic_management`: ``" not in md
    assert "**[IX selections]**" not in md


def test_write_pa_toolcalls_extracts_grouped_judge_ready_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["tool_events"] = [
        {
            "name": "IX_topic_management",
            "status": "ok",
            "args": {"setting": "One-at-a-time"},
            "detail": '{"type":"ix_preference_selection"}',
            "ix": True,
            "attribute": "topic_management",
            "setting": "One-at-a-time",
            "evidence": "User asked to keep the email about one issue only.",
            "application": "Keep the draft scoped to elevator repair.",
        },
        {
            "name": "mcp_state_email_draft_create",
            "status": "ok",
            "args": {"subject": "Elevator"},
            "detail": "{'draft_id': 'draft_0001'}",
        },
    ]
    events[3]["ix_called"] = ["IX_topic_management"]
    events[3]["ix_missing"] = []
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    records = extract_pa_toolcalls(run_dir / "transcript.jsonl")
    assert records["meta"]["session_id"] == "session_01"
    assert len(records["beats"]) == 1
    beat = records["beats"][0]
    assert beat["beat"] == "open"
    assert beat["active_ix_tools"] == ["IX_topic_management"]
    assert beat["ix_required"] == ["IX_topic_management"]
    assert beat["ix_called"] == ["IX_topic_management"]
    assert beat["ix_missing"] == []
    assert beat["ix_render"] == []
    assert beat["autonomy_events"] == []
    assert [call["name"] for call in beat["calls"]] == [
        "IX_topic_management",
        "mcp_state_email_draft_create",
    ]
    assert beat["calls"][0]["call_index"] == 0
    assert beat["calls"][0]["type"] == "ix"
    assert beat["calls"][0]["attribute"] == "topic_management"
    assert beat["calls"][0]["setting"] == "One-at-a-time"
    assert beat["calls"][1]["call_index"] == 1
    assert beat["calls"][1]["type"] == "task"

    out_path = write_pa_toolcalls(run_dir / "transcript.jsonl")
    assert out_path == run_dir / "pa_toolcalls.json"
    assert json.loads(out_path.read_text(encoding="utf-8")) == records


def _ix_render_blob(**overrides) -> dict:
    """An `ix_render` blob: one released batch, one held one, a gate."""
    blob = {
        "schema_version": IX_RENDER_SCHEMA_VERSION,
        "ix_contract_version": IX_CONTRACT_VERSION,
        "episodes": {
            "topic_management": {
                "episode_id": "bs1",
                "rounds_open": 1,
                "closed": False,
            },
        },
        "released": {"batches": ["b1"], "questions": [], "plan_executed": ["a1"]},
        "held": {
            "batches": [
                {
                    "batch_id": "b2",
                    "label": "hotel",
                    "body": "HELD-BODY-MARKER the hotel needs a deposit by Friday.",
                }
            ],
            "questions": [],
            "plan_remaining": [{"id": "a3", "intent": "send the email"}],
            "calls": [
                {
                    "interaction_item_id": "i3",
                    "plan_item_id": "a3",
                    "tool": "mcp_state_email_message_send",
                    "reason": "external_action_floor",
                    "status": "held_for_confirmation",
                }
            ],
            "cancelled": [],
            "outreach_message": "",
        },
        "gate": {
            "attribute": "autonomy_level",
            "item_id": "a3",
            "reason": "external_action_floor",
            "action_type": "external_action",
            "message_source": "_ix_confirmation_message",
        },
        "consent": [
            {
                "attribute": "autonomy_level",
                "item_id": "a3",
                "pa_decision": "release",
                "sideband_decision": "allow",
                "released": True,
                "user_evidence": "go ahead",
                "evidence_verified": True,
            }
        ],
        "notices": [
            {"order": 0, "tool": "mcp_state_document_read", "text": "Checking your list."}
        ],
        "completion": {"released": False, "text": "COMPLETION-MARKER all set."},
        "reasoning": {
            "shown": True,
            "source": "user_facing_rationale",
            "provider_reasoning_present": False,
        },
        "certainties": [{"ref": "c1", "level": "low", "rendered": "(confidence: low)"}],
        "proactive_decision": None,
        "trace": ["TRACE-MARKER topic_management -> release 1/2"],
    }
    blob.update(overrides)
    return blob


def test_ix_render_block_round_trips_through_from_ix_render_metadata(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["ix_render"] = _ix_render_blob()
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    records = extract_pa_toolcalls(run_dir / "transcript.jsonl")
    beat = records["beats"][0]
    assert len(beat["ix_render"]) == 1
    record = beat["ix_render"][0]
    assert record["t"] == events[3]["t"]
    assert record["schema_version"] == IX_RENDER_SCHEMA_VERSION
    assert record["ix_contract_version"] == IX_CONTRACT_VERSION
    # The held body survives verbatim: the IX face has to tell "never written"
    # from "written and held".
    assert record["held"]["batches"][0]["body"].startswith("HELD-BODY-MARKER")
    # `autonomy_events` now records the gate, not the old interception.
    assert beat["autonomy_events"] == [
        {"t": events[3]["t"], **_ix_render_blob()["gate"]}
    ]

    read_back = from_ix_render_metadata(events[3]["ix_render"])
    assert set(read_back) == set(IX_RENDER_TOP_LEVEL_KEYS) - IX_RENDER_OPTIONAL_KEYS


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", "ix_render/0.9.0"),
        ("ix_contract_version", "ix-wrapper/0.9.0"),
    ],
)
def test_ix_render_block_with_a_stale_version_fails_fast(
    tmp_path: Path, field: str, value: str
) -> None:
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["ix_render"] = _ix_render_blob(**{field: value})
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    with pytest.raises(IXContractVersionError):
        extract_pa_toolcalls(run_dir / "transcript.jsonl")


def test_render_markdown_shows_held_content_gate_and_consent(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["ix_render"] = _ix_render_blob()
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    md = render_markdown(run_dir / "transcript.jsonl")
    assert "**[IX render]**" in md
    assert "HELD-BODY-MARKER" in md
    assert "held call `mcp_state_email_message_send`" in md
    assert "gate: `autonomy_level`" in md
    assert "| `release` | `allow` | True |" in md
    assert "Checking your list." in md
    assert "COMPLETION-MARKER" in md
    assert "TRACE-MARKER" in md


def test_identical_held_call_rows_collapse_to_one_line(tmp_path: Path) -> None:
    """acc_039 t19 printed 25 identical `held call` rows for one act.

    Rendering only: the ledger keeps every row, so the audit plane still counts
    the refusals -- the page shows the run of them once, with how many.
    """
    run_dir = tmp_path / "session"
    spent = {
        "interaction_item_id": "apt_book_1",
        "plan_item_id": "step1",
        "tool": "mcp_state_appointment_create",
        "reason": "release_spent",
        "status": "held_for_binding",
    }
    blob = _ix_render_blob()
    blob["held"]["calls"] = [spent, dict(spent), dict(spent), blob["held"]["calls"][0]]
    events = _synthetic_session_events()
    events[3]["ix_render"] = blob
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    md = render_markdown(run_dir / "transcript.jsonl")
    rows = [line for line in md.split("\n") if line.startswith("- held call")]
    assert len(rows) == 2
    assert rows[0].endswith("repeat=3")
    assert "repeat" not in rows[1]
    logged = json.loads(
        (run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()[3]
    )
    assert len(logged["ix_render"]["held"]["calls"]) == 4


def test_eval_block_shows_the_consent_sideband(tmp_path: Path) -> None:
    """The gate's ground truth belongs on the IX face's main read."""
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[2]["consent_decisions"] = [
        {"interaction_item_id": "item_email_1", "decision": "deny"}
    ]
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    md = render_markdown(run_dir / "transcript.jsonl")
    assert "**consent_decisions**:" in md
    assert "- `item_email_1`: `deny`" in md

    # A turn with no sideband says so, the way factual_check already does.
    bare_dir = tmp_path / "bare"
    with TranscriptWriter(bare_dir) as tw:
        for evt in _synthetic_session_events():
            tw.log_event(evt)
    assert "**consent_decisions**: (none)" in render_markdown(
        bare_dir / "transcript.jsonl"
    )


def test_ix_tools_are_never_counted_as_task_tool_calls(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["tool_events"] = [
        {"name": "IX_render_topic_management", "status": "ok", "args": {"batch_set_id": "bs1"}},
        {"name": "IX_continue_autonomy_level", "status": "ok", "args": {}},
        {"name": "IX_capability_resolve", "status": "ok", "args": {}},
        {"name": "mcp_state_document_read", "status": "ok", "args": {"path": "a.md"}},
        {"process": True, "stage": "start", "message": "legacy control event"},
        {"ix_event": True, "name": "ix_consent_denied", "item_id": "i3"},
    ]
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    records = extract_pa_toolcalls(run_dir / "transcript.jsonl")
    calls = records["beats"][0]["calls"]
    # The wrapper's control records stay on the IX face — dropping them here
    # would take the gate out of the audit trail, not just out of the pack.
    assert [call["name"] for call in calls] == [
        "IX_render_topic_management",
        "IX_continue_autonomy_level",
        "IX_capability_resolve",
        "mcp_state_document_read",
        "ix_consent_denied",
    ]
    assert [call["type"] for call in calls] == ["ix", "ix", "ix", "task", "ix"]
    assert _count_tool_calls(load_events(run_dir / "transcript.jsonl")) == 1


def test_ix_calls_keep_the_payload_type_tj_selects_b2_by(tmp_path: Path) -> None:
    """`selections[]` takes only `payload_type == "ix_preference_selection"`.

    The per-turn payload tools are `type == "ix"` too and their `setting` is the
    empty string, so with `payload_type` dropped the documented selector had
    nothing to filter on and `selected_setting` came back empty.
    """
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["tool_events"] = [
        {
            "name": "IX_topic_management",
            "status": "ok",
            "args": {"setting": "One-at-a-time"},
            "ix": True,
            "payload_type": IX_SELECTION_PAYLOAD_TYPE,
            "attribute": "topic_management",
            "setting": "One-at-a-time",
            "evidence": "User asked to keep the email about one issue only.",
            "application": "Keep the draft scoped to elevator repair.",
        },
        {
            "name": "IX_render_topic_management",
            "status": "ok",
            "args": {"batch_set_id": "bs1"},
            "ix": True,
            "payload_type": "ix_render_payload",
            "attribute": "topic_management",
            "setting": "",
            "evidence": "",
            "application": "",
        },
    ]
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    calls = extract_pa_toolcalls(run_dir / "transcript.jsonl")["beats"][0]["calls"]
    selections = [
        call for call in calls
        if call.get("payload_type") == IX_SELECTION_PAYLOAD_TYPE
    ]

    assert [call["name"] for call in selections] == ["IX_topic_management"]
    assert selections[0]["setting"] == "One-at-a-time"
    # The render payload stays on the IX face — it is just not a selection.
    assert [call["payload_type"] for call in calls] == [
        IX_SELECTION_PAYLOAD_TYPE,
        "ix_render_payload",
    ]


def test_ix_control_records_keep_the_ids_tj_correlates_them_by(
    tmp_path: Path,
) -> None:
    """Cancellations and false claims are correlated through
    each control record's own `item_id` / `item_ids`.

    The derived view rebuilt a fixed field set, so an `ix_consent_denied` row
    arrived with its name and nothing else — not joinable to the call it
    refused.
    """
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["tool_events"] = [
        {
            "name": "mcp_state_email_message_send",
            "status": "ok",
            "args": {"to": "team@example.test", "interaction_item_id": "i3"},
        },
        {
            "ix_event": True,
            "name": "ix_consent_denied",
            "item_id": "i3",
            "sideband_decision": "deny",
        },
        {
            "ix_event": True,
            "name": "ix_claims_unexecuted_action",
            "tools": ["mcp_state_email_message_send"],
            "item_ids": ["i3"],
        },
        {
            "ix_event": True,
            "name": "ix_hold_forced_close",
            "attribute": "topic_management",
            "episode_id": "bs1",
            "cancelled": 2,
        },
    ]
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    calls = extract_pa_toolcalls(run_dir / "transcript.jsonl")["beats"][0]["calls"]
    control = {call["name"]: call for call in calls if call.get("ix_event")}

    assert set(control) == {
        "ix_consent_denied", "ix_claims_unexecuted_action", "ix_hold_forced_close",
    }
    denied = control["ix_consent_denied"]
    assert denied["type"] == "ix"
    assert denied["sideband_decision"] == "deny"
    # The join: the refused call is findable by that same id.
    assert [
        call["name"] for call in calls
        if call["args"].get("interaction_item_id") == denied["item_id"]
    ] == ["mcp_state_email_message_send"]
    assert control["ix_claims_unexecuted_action"]["item_ids"] == ["i3"]
    assert control["ix_hold_forced_close"]["cancelled"] == 2
    # A control record is not a selection: the four IX payload fields belong to
    # `tool_event["ix"]` entries only, and `payload_type` filters it out of the
    # selections.
    assert "setting" not in denied
    assert denied.get("payload_type") is None


def test_render_markdown_includes_interaction_tool_message(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    events = _synthetic_session_events()
    events[3]["tool_events"] = [
        {
            "name": "mcp_state_Message_task_scope",
            "status": "ok",
            "args": {"included_scope": "elevator repair"},
            "detail": '{"message": "I will keep this scoped to the elevator."}',
            "interaction": True,
            "message": "I will keep this scoped to the elevator.",
        }
    ]
    with TranscriptWriter(run_dir) as tw:
        for evt in events:
            tw.log_event(evt)

    md = render_markdown(run_dir / "transcript.jsonl")
    assert "`mcp_state_Message_task_scope`" in md
    assert "interaction_message: I will keep this scoped to the elevator." in md


def test_write_markdown_writes_sibling_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    with TranscriptWriter(run_dir) as tw:
        for evt in _synthetic_session_events():
            tw.log_event(evt)

    md_path = write_markdown(run_dir / "transcript.jsonl")
    assert md_path == run_dir / "transcript.md"
    assert md_path.exists()
    assert "# Session: session_01" in md_path.read_text()


def test_write_meta_writes_yaml(tmp_path: Path) -> None:
    run_dir = tmp_path / "session"
    path = write_meta(run_dir, {"persona": "User A", "model": "test-model", "context": "work"})
    assert path == run_dir / "meta.yaml"
    text = path.read_text()
    assert "persona: User A" in text
    assert "model: test-model" in text
    assert "created_at:" in text


def test_render_handles_pa_turn_without_pair(tmp_path: Path) -> None:
    """If a pa_turn arrives with no prior simulator_turn, renderer doesn't crash
    and labels it as unpaired."""
    run_dir = tmp_path / "session"
    with TranscriptWriter(run_dir) as tw:
        tw.log_event({"t": 0, "kind": "session_start", "session_id": "s", "persona": "P"})
        tw.log_event({"t": 1, "kind": "pa_turn", "outbound": "orphan reply"})
        tw.log_event({"t": 2, "kind": "session_end", "simulator_turns": 0, "beat_count": 0})

    md = render_markdown(run_dir / "transcript.jsonl")
    assert "no paired simulator_turn" in md
    assert "orphan reply" in md


def test_render_markdown_shows_the_outreach_path(tmp_path: Path) -> None:
    events = [
        {"t": 0, "kind": "session_start", "persona": "User A", "session_id": "acc_007",
         "context": "personal", "scenario": "pickup", "model": "m",
         "beat_names": ["opening_after_send", "opening_after_suppress"], "session_key": "k"},
        {"t": 1, "kind": "environment_trigger", "trigger_id": "pickup_deadline",
         "source_type": "deadline", "detail": "Closes at 6.", "prior_thread": "Order ready.",
         "user_impact": "Pickup today."},
        {"t": 2, "kind": "pa_turn", "beat": "proactive_trigger", "proactive_event": True,
         "outbound": "", "outreach_decision": "suppress", "outreach_delivered": False,
         "tool_events": [], "ix_required": ["IX_proactive_outreach"], "ix_called": [],
         "ix_missing": [], "empty": True},
        {"t": 3, "kind": "outreach_path", "path": "suppress",
         "opening_beat": "opening_after_suppress", "skipped_beat": "opening_after_send",
         "resume_at": "2026-10-04T16:20:00+00:00"},
        {"t": 4, "kind": "session_end", "simulator_turns": 0, "beat_count": 0},
    ]
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    text = write_markdown(transcript).read_text(encoding="utf-8")
    assert "**[PA outreach decision: suppress]**" in text
    assert "contact suppressed → `opening_after_suppress`" in text
    assert "comes back on their own at 2026-10-04T16:20:00+00:00" in text
