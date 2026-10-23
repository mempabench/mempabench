"""Text-face judge packing — T11 (banned-field / banned-string).

Pure data, no LLM. Run:
    PYTHONPATH=. python -m pytest tests/test_harness/test_judge_pack.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.judge_pack import (
    IX_CONTROL_VOCABULARY,
    IX_SETTING_LABELS,
    TEXT_JUDGE_PA_DELIVERED_VISIBILITY,
    TEXT_JUDGE_VISIBILITY,
    build_text_judge_pack,
    scan_banned_keys,
    scan_banned_strings,
    text_judge_projection,
)
from harness.transcript import TranscriptWriter, load_events
from pa.agent.ix.render_types import (
    IX_CONTRACT_VERSION,
    IX_EVENT_NAMES,
    IX_RENDER_SCHEMA_VERSION,
    Batch,
    GateRecord,
    IXRenderInput,
    RenderedTurn,
    ToolCallRecord,
)

HELD_BODY = "The hotel wants a deposit by Friday; I have not booked it."
GATE_REASON = "external_action_floor"
TRACE_LINE = "topic_management=One-at-a-time -> release 1/2"
PROVIDER_REASONING = "The user seems annoyed, so I will keep this short."

#: The strings that may never appear anywhere in a text-face pack, free text
#: included: held bodies, the gate reason, the trace and provider reasoning.
LEAK_NEEDLES = (HELD_BODY, GATE_REASON, TRACE_LINE, PROVIDER_REASONING)


def _ix_render_blob() -> dict:
    return {
        "schema_version": IX_RENDER_SCHEMA_VERSION,
        "ix_contract_version": IX_CONTRACT_VERSION,
        "episodes": {
            "topic_management": {"episode_id": "bs1", "rounds_open": 1, "closed": False}
        },
        "released": {"batches": ["b1"], "questions": [], "plan_executed": ["a1"]},
        "held": {
            "batches": [{"batch_id": "b2", "label": "hotel", "body": HELD_BODY}],
            "questions": [],
            "plan_remaining": [{"id": "a3", "intent": "send the email"}],
            "calls": [
                {
                    "interaction_item_id": "i3",
                    "plan_item_id": "a3",
                    "tool": "mcp_state_email_message_send",
                    "reason": GATE_REASON,
                    "status": "held_for_confirmation",
                }
            ],
            "cancelled": [],
            "outreach_message": "",
        },
        "gate": {
            "attribute": "autonomy_level",
            "item_id": "a3",
            "reason": GATE_REASON,
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
        "completion": {"released": False, "text": "All set."},
        "reasoning": {
            "shown": True,
            "source": "user_facing_rationale",
            "provider_reasoning_present": True,
        },
        "certainties": [{"ref": "c1", "level": "low", "rendered": "(confidence: low)"}],
        "proactive_decision": None,
        "trace": [TRACE_LINE],
    }


def _session_events() -> list[dict]:
    """One held / gated / consented exchange plus the surrounding bookends."""
    return [
        {"t": 0, "kind": "session_start", "session_id": "s1", "persona": "User A"},
        {"t": 1, "kind": "beat_enter", "beat": "open", "active_skills": ["autonomy_level"]},
        {
            "t": 2,
            "kind": "simulator_turn",
            "beat": "open",
            "message": "Book the hotel and mail the team.",
            "turn_assessment": {"next_beat": "stay"},
        },
        {
            "t": 3,
            "kind": "pa_turn",
            "beat": "open",
            "outbound": "I read your trip note. Shall I send the email to the team?",
            "tool_events": [
                {
                    "name": "IX_autonomy_level",
                    "status": "ok",
                    "ix": True,
                    "attribute": "autonomy_level",
                    "setting": "Reactive",
                    "evidence": "The user asked to be consulted before sending.",
                    "application": "Ask before the send.",
                    "args": {"setting": "Reactive"},
                },
                {"name": "IX_render_autonomy_level", "status": "ok", "args": {"plan_id": "p1"}},
                {
                    "name": "mcp_state_document_read",
                    "status": "ok",
                    "args": {
                        "path": "trip.md",
                        "_ix_notice": "I am checking your trip note.",
                        "interaction_item_id": "i1",
                    },
                    "detail": "trip contents",
                },
                {"process": True, "stage": "start", "message": "legacy control event"},
                # The wrapper's own control record. Its name carries no
                # attribute or setting string, so only the predicate can keep
                # "a gate fired and consent was refused" off the text face.
                {"ix_event": True, "name": "ix_consent_denied", "item_id": "i3"},
                {"ix_event": True, "name": "beat_gate_turn"},
            ],
            "ix_render": _ix_render_blob(),
        },
        {"t": 4, "kind": "session_end", "simulator_turns": 1, "beat_count": 1},
    ]


def test_pack_keeps_the_public_turn_and_executed_task_events(tmp_path: Path) -> None:
    pack = build_text_judge_pack(_session_events())

    kinds = [turn["kind"] for turn in pack["turns"]]
    assert kinds == ["user_message", "pa_turn"]
    pa_turn = pack["turns"][1]
    # The gate's ask reached the user, so it stays — it is part of public_turn.
    assert "Shall I send the email to the team?" in pa_turn["public_turn"]
    assert [event["name"] for event in pa_turn["task_tool_events"]] == [
        "mcp_state_document_read"
    ]
    # Neither the IX tools nor the wrapper's control records are task calls.
    assert "ix_consent_denied" not in json.dumps(pack, ensure_ascii=False)
    assert "beat_gate_turn" not in json.dumps(pack, ensure_ascii=False)
    # The injected per-call fields are stripped out of the args that remain.
    assert pa_turn["task_tool_events"][0]["args"] == {"path": "trip.md"}


def test_pack_has_no_banned_key_and_no_ix_tool_event() -> None:
    pack = build_text_judge_pack(_session_events())

    assert scan_banned_keys(pack) == []
    text = json.dumps(pack, ensure_ascii=False)
    assert "ix_render" not in text
    assert "IX_" not in text


def test_pack_has_no_attribute_setting_or_held_string() -> None:
    pack = build_text_judge_pack(_session_events())

    assert scan_banned_strings(pack, needles=LEAK_NEEDLES) == []


def test_pack_never_carries_the_consent_sideband() -> None:
    """Assert, don't filter.

    `consent_decisions` is the evaluation side's ground truth for the gate,
    PA-blind like `factual_check` and `turn_assessment`. The whitelist is what
    keeps it out — a `simulator_turn` contributes its `t` / `beat` / `text` and
    nothing else — and this test is what keeps that true when the whitelist is
    next edited. The key is in `TEXT_JUDGE_BANNED_KEYS` as well, so the
    assembly scan now catches it on any path the whitelist does not cover.
    """
    events = _session_events()
    events[2]["consent_decisions"] = [
        {"interaction_item_id": "consent_item_9", "decision": "deny"}
    ]

    pack = build_text_judge_pack(events)

    user_message = pack["turns"][0]
    assert user_message["kind"] == "user_message"
    assert sorted(user_message) == ["beat", "kind", "t", "text"]
    assert all(
        "consent_decisions" not in turn
        for turn in pack["turns"]
        if turn["kind"] == "user_message"
    )
    text = json.dumps(pack, ensure_ascii=False)
    assert "consent_decisions" not in text
    assert "consent_item_9" not in text


def test_meta_is_reduced_to_the_text_face_allow_list() -> None:
    """`meta` was copied verbatim, so the caller's whole metadata dict rode in.

    `thinking_blocks` is the fourth provider-reasoning path, `consent_decisions`
    is the gate's PA-blind ground truth (and `id` is an alias
    `consent_decisions_from_metadata` accepts), and `target_setting` is the
    ground truth the text face is supposed to be blind to. None of the three
    was a banned key on this path, so both scans returned empty.
    """
    pack = build_text_judge_pack(
        _session_events(),
        meta={
            "run_id": "acc001_simgemini_005",
            "persona_id": "user_a",
            "context": "work",
            "pa_model": "deepseek/deepseek-v4-flash",
            "thinking_blocks": [{"type": "thinking", "thinking": PROVIDER_REASONING}],
            "reasoning_content": PROVIDER_REASONING,
            "consent_decisions": [{"id": "i3", "decision": "allow"}],
            "target_setting": "Reactive",
        },
    )

    # The legitimate provenance fields survive.
    assert pack["meta"] == {
        "run_id": "acc001_simgemini_005",
        "persona_id": "user_a",
        "context": "work",
        "pa_model": "deepseek/deepseek-v4-flash",
    }
    blob = json.dumps(pack, ensure_ascii=False)
    for dropped in (
        "thinking_blocks", "reasoning_content", "consent_decisions",
        PROVIDER_REASONING, "target_setting", "Reactive",
    ):
        assert dropped not in blob


def test_the_assembly_scan_refuses_a_leak_the_whitelists_missed() -> None:
    """The two scans now run on the finished object, every time.

    A banned key nested inside a tool argument survives `_strip_injected_fields`
    (which only looks at the top level of `args`), so it is exactly the case the
    assembly check exists for — before, `turns` was scanned only by whichever
    test remembered to call a scan, and `meta` was not scanned at all.
    """
    events = _session_events()
    events[3]["tool_events"] = [
        {
            "name": "mcp_state_document_create",
            "status": "ok",
            "args": {
                "path": "notes.md",
                "provenance": {"reasoning_content": PROVIDER_REASONING},
            },
        }
    ]

    with pytest.raises(ValueError, match="reasoning_content"):
        build_text_judge_pack(events)


def test_the_same_session_keeps_all_of_it_on_the_ix_face(tmp_path: Path) -> None:
    """The removals are a packing rule, not a loss: the IX face still has it."""
    run_dir = tmp_path / "session"
    with TranscriptWriter(run_dir) as writer:
        for event in _session_events():
            writer.log_event(event)

    ix_face = json.dumps(load_events(run_dir / "transcript.jsonl"), ensure_ascii=False)
    # Provider reasoning is not in this list: the blob records only whether it was
    # present (`reasoning.provider_reasoning_present`), never its text.
    for needle in (
        HELD_BODY, GATE_REASON, TRACE_LINE, "autonomy_level", "Reactive",
        "ix_render", "ix_consent_denied", "beat_gate_turn",
    ):
        assert needle in ix_face


def test_scan_banned_strings_reports_a_leak_it_is_given() -> None:
    leaked = {
        "turns": [
            {
                "kind": "pa_turn",
                "public_turn": "Done.",
                "task_tool_events": [{"name": "log", "args": {"note": TRACE_LINE}}],
            }
        ]
    }

    hits = scan_banned_strings(leaked, needles=LEAK_NEEDLES)

    assert any(TRACE_LINE in hit for hit in hits)
    assert any("topic_management" in hit for hit in hits)


def test_free_text_is_only_scanned_for_the_needles_it_is_given() -> None:
    """An LLM writing "Low" to the user is prose, not a setting leak.

    Attribute and setting names are checked on the IX-shaped structure of the
    pack; scanning the user-visible prose for them would report noise.
    """
    assert "Low" in IX_SETTING_LABELS
    pack = {"turns": [{"kind": "pa_turn", "public_turn": "Low tide is at six."}]}

    assert scan_banned_strings(pack, needles=LEAK_NEEDLES) == []
    assert scan_banned_strings({"ix_render": {"setting": "Low"}}) == [
        "ix_render.setting: Low"
    ]


def test_a_document_the_pa_wrote_is_prose_not_a_setting_leak() -> None:
    """Both acc_011 runs reported one structural hit on a `document_create` whose
    body said "Low priority". The argument is PA prose in exactly the sense
    `public_turn` is, so the vocabulary scan has no business there.
    """
    pack = {
        "turns": [
            {
                "kind": "pa_turn",
                "public_turn": "Saved to your desktop.",
                "task_tool_events": [
                    {
                        "name": "document_create",
                        "args": {
                            "path": "my_desktop/prep.md",
                            "content": "- Low priority — mention the wrist ache second.",
                        },
                    }
                ],
            }
        ]
    }

    assert scan_banned_strings(pack, needles=LEAK_NEEDLES) == []


def test_a_held_body_inside_a_document_is_still_a_leak() -> None:
    """Free text is not unscanned text: the needles still apply to `content`."""
    pack = {
        "turns": [
            {
                "kind": "pa_turn",
                "public_turn": "Saved to your desktop.",
                "task_tool_events": [
                    {
                        "name": "document_create",
                        "args": {"path": "my_desktop/prep.md", "content": HELD_BODY},
                    }
                ],
            }
        ]
    }

    hits = scan_banned_strings(pack, needles=LEAK_NEEDLES)

    assert hits == [f"turns[0].task_tool_events[0].args.content: {HELD_BODY}"]


# --- setting labels vs control vocabulary -------------------------------------
#
# The structural half used to scan the whole `IX_VOCABULARY` as
# substrings over every leaf outside `FREE_TEXT_KEYS`, so `priority: "High"`,
# a reply that said "keep it casual" and the word "Highlight" all failed the
# pack. Labels are now whole tokens on IX-shaped leaves only; the wrapper's
# control surface stays banned everywhere.


def test_a_task_argument_named_priority_high_is_not_a_setting_leak() -> None:
    """(a) `High` is a task priority far more often than it is a setting."""
    events = _session_events()
    events[3]["tool_events"] = [
        {
            "name": "mcp_state_task_create",
            "status": "ok",
            "args": {"title": "Confirm the hotel", "priority": "High"},
        }
    ]

    pack = build_text_judge_pack(events)

    assert pack["turns"][1]["task_tool_events"][0]["args"] == {
        "title": "Confirm the hotel",
        "priority": "High",
    }
    assert scan_banned_strings(pack, needles=LEAK_NEEDLES) == []


def test_outbound_prose_may_use_the_words_the_settings_are_named_after() -> None:
    """(b) The judge rows quote outbound prose; prose owns these words."""
    events = _session_events()
    events[3]["outbound"] = (
        "keep it casual, show me the moderate option. Casual is fine either "
        "way — Show me the Detailed version and I will keep the Terse one for "
        "later. High priority, Low effort."
    )

    pack = build_text_judge_pack(events)

    assert scan_banned_strings(pack, needles=LEAK_NEEDLES) == []


def test_a_setting_on_an_ix_shaped_structural_leaf_is_still_a_leak() -> None:
    """(c) The leak the scan exists for: the selection reaching the text face
    through structure rather than through what the PA said."""
    leaked = {
        "ix_render": {
            "selections": [{"attribute": "tone_formality", "setting": "Formal"}]
        }
    }

    hits = scan_banned_strings(leaked)

    assert "ix_render.selections[0].setting: Formal" in hits
    assert "ix_render.selections[0].attribute: tone_formality" in hits

    # And the assembly refuses it: a nested blob survives `_strip_injected_fields`,
    # which only looks at the top level of `args`.
    events = _session_events()
    events[3]["tool_events"] = [
        {
            "name": "mcp_state_document_create",
            "status": "ok",
            "args": {
                "path": "notes.md",
                "provenance": {"ix_render": {"selections": [{"setting": "Formal"}]}},
            },
        }
    ]
    with pytest.raises(ValueError, match="Formal"):
        build_text_judge_pack(events)


def test_highlight_under_an_ix_key_does_not_match_the_setting_high() -> None:
    """(d) Whole tokens, not substrings: `Highlight` / `Showcase` / `Lowell`."""
    assert {"High", "Show", "Low"} <= IX_SETTING_LABELS

    assert scan_banned_strings({"ix_render": {"label": "Highlight a Showcase for Lowell"}}) == []
    assert scan_banned_strings({"ix_render": {"label": "High"}}) == [
        "ix_render.label: High"
    ]


def test_a_control_marker_in_outbound_prose_is_still_a_leak() -> None:
    """(e) `IX_CONTROL_VOCABULARY` is banned in free text too — a `[[…]]`
    position marker never reaches the user by accident."""
    assert "[[question:" in IX_CONTROL_VOCABULARY

    events = _session_events()
    events[3]["outbound"] = "Which hotel do you want? [[question:q1]]"

    with pytest.raises(ValueError, match=r"\[\[question:"):
        build_text_judge_pack(events)


def test_an_ix_tool_name_in_a_text_face_tool_event_is_still_a_leak() -> None:
    """(f) Tool names are checked as before: by `scan_banned_keys` on the `name`
    key, and now by the control-vocabulary scan wherever else the name lands."""
    pack = {
        "turns": [
            {
                "kind": "pa_turn",
                "public_turn": "Done.",
                "task_tool_events": [{"name": "IX_verbosity", "args": {}}],
            }
        ]
    }

    assert scan_banned_keys(pack) == ["turns[0].task_tool_events[0].name"]
    assert scan_banned_strings(pack) == [
        "turns[0].task_tool_events[0].name: IX_verbosity"
    ]

    events = _session_events()
    events[3]["tool_events"] = [
        {
            "name": "mcp_state_document_create",
            "status": "ok",
            "args": {"path": "notes.md", "source_tool": "IX_verbosity"},
        }
    ]
    with pytest.raises(ValueError, match="IX_verbosity"):
        build_text_judge_pack(events)


def test_audit_event_names_are_control_vocabulary_but_ix_is_not_a_prefix() -> None:
    """(g) The lowercase audit names are control vocabulary too.

    They are taken from `IX_EVENT_NAMES` at import time and banned as **exact
    names**, not as a bare `ix_` prefix: this half of the scan is a substring
    match, so `ix_` alone would fail an ordinary `prefix_length` argument.
    """
    assert set(IX_EVENT_NAMES) <= IX_CONTROL_VOCABULARY
    assert "ix_" not in IX_CONTROL_VOCABULARY

    denied = _session_events()
    denied[3]["outbound"] = "I stopped there (ix_consent_denied) — shall I go on?"
    with pytest.raises(ValueError, match="ix_consent_denied"):
        build_text_judge_pack(denied)

    gated = _session_events()
    gated[3]["outbound"] = "That was a beat_gate_turn, so I paused."
    with pytest.raises(ValueError, match="beat_gate_turn"):
        build_text_judge_pack(gated)

    innocent = _session_events()
    innocent[3]["outbound"] = "The prefix_length and matrix_size fields look right."
    pack = build_text_judge_pack(innocent)
    assert [
        turn["public_turn"] for turn in pack["turns"] if turn["kind"] == "pa_turn"
    ] == ["The prefix_length and matrix_size fields look right."]


def test_text_judge_projection_keeps_executed_calls_only() -> None:
    """The contract of the `render_types` stub, as the packer implements it."""
    rendered = RenderedTurn(
        message="I read your note. Shall I send it?",
        executed_call_ids=["call_read"],
        held_call_ids=["call_send"],
        trace=[TRACE_LINE],
    )
    render_input = IXRenderInput(
        segments=["...", HELD_BODY],
        provider_reasoning=[PROVIDER_REASONING],
        calls=[
            ToolCallRecord(
                order=0,
                tool_name="mcp_state_document_read",
                args={"path": "trip.md", "interaction_item_id": "i1"},
                tool_call_id="call_read",
                executed=True,
            ),
            ToolCallRecord(
                order=1,
                tool_name="mcp_state_email_message_send",
                args={"to": "team@example.test"},
                tool_call_id="call_send",
            ),
            ToolCallRecord(
                order=2,
                tool_name="IX_render_topic_management",
                args={"batch_set_id": "bs1"},
                tool_call_id="call_ix",
            ),
        ],
        gate=GateRecord(attribute="autonomy_level", item_id="a3", reason=GATE_REASON),
    )

    projection = text_judge_projection(rendered, render_input)

    assert projection.public_turn == rendered.message
    assert [event["name"] for event in projection.task_tool_events] == [
        "mcp_state_document_read"
    ]
    assert projection.task_tool_events[0]["args"] == {"path": "trip.md"}
    packed = {
        "public_turn": projection.public_turn,
        "task_tool_events": projection.task_tool_events,
    }
    assert scan_banned_keys(packed) == []
    assert scan_banned_strings(packed, needles=LEAK_NEEDLES) == []


def test_held_batch_body_never_reaches_the_pack() -> None:
    """A batch the wrapper held is invisible to the text face by construction."""
    events = _session_events()
    pack = build_text_judge_pack(events)

    assert Batch(batch_id="b2", label="hotel", body=HELD_BODY).body == HELD_BODY
    assert HELD_BODY not in json.dumps(pack, ensure_ascii=False)


# --- forced events -----------------------------------------------------------
#
# A forced step never calls the PA LLM: `harness/forced_event_executor.py`
# replays authored speech for both sides through the one `message` kind and
# emits a separate `pa_turn` per authored tool call. So the text face has to
# read `source` to know whose turn a `message` is, and has to drop the one
# visibility neither party ever saw.

GOD_ONLY_NOTE = "Setup: the deposit deadline is what this step is testing."


def _forced_event_events() -> list[dict]:
    """One forced step's transcript, in the executor's own event shape."""
    return [
        {"t": 0, "kind": "session_start", "session_id": "fe1", "persona": "User A"},
        {
            "t": 10,
            "kind": "message",
            "exchange": 0,
            "source": "user",
            "visibility": "both",
            "content": "Did the hotel come back to you?",
        },
        {
            "t": 20,
            "kind": "message",
            "exchange": 0,
            "source": "pa",
            "visibility": "both",
            "content": "They did. Shall I pay the deposit now?",
        },
        {
            "t": 30,
            "kind": "message",
            "exchange": 0,
            "source": {"kind": "third_party", "id": "hotel_lugano"},
            "visibility": "pa",
            "content": "Hotel Lugano: the deposit is due Friday.",
        },
        {
            "t": 40,
            "kind": "message",
            "exchange": 0,
            "source": {"kind": "environment", "id": "authoring_note"},
            "visibility": "god_only",
            "content": GOD_ONLY_NOTE,
        },
        {
            "t": 50,
            "kind": "pa_turn",
            "forced_event": True,
            "exchange": 0,
            "outbound": "",
            "tool_events": [
                {
                    "call_id": "c1",
                    "name": "mcp_state_document_read",
                    "status": "ok",
                    "args": {"path": "trip.md"},
                    "detail": "trip contents",
                }
            ],
        },
        {"t": 60, "kind": "session_end", "simulator_turns": 0, "beat_count": 0},
    ]


def test_forced_pa_speech_is_packed_as_a_pa_turn_not_a_user_message() -> None:
    """The bug: `source` was never read, so the PA's authored line arrived at
    the judge as something the user had said."""
    pack = build_text_judge_pack(_forced_event_events())

    kinds = [turn["kind"] for turn in pack["turns"]]
    assert kinds == ["user_message", "pa_turn", "user_message", "pa_turn"]

    pa_speech = pack["turns"][1]
    assert pa_speech["public_turn"] == "They did. Shall I pay the deposit now?"
    # Authored speech went through no wrapper, so it owns no tool events; the
    # step's real calls are the `pa_turn` event that follows.
    assert pa_speech["task_tool_events"] == []
    assert "text" not in pa_speech

    # A third-party message is something the PA received, not something it
    # said — the executor's own role mapping, and now the pack's.
    assert pack["turns"][2]["text"] == "Hotel Lugano: the deposit is due Friday."
    assert [
        event["name"] for event in pack["turns"][3]["task_tool_events"]
    ] == ["mcp_state_document_read"]


def test_god_only_setup_never_reaches_the_text_face() -> None:
    """The old filter tested for `eval_only`, a value the vocabulary never had,
    so every `god_only` item passed straight through."""
    pack = build_text_judge_pack(_forced_event_events())

    assert GOD_ONLY_NOTE not in json.dumps(pack, ensure_ascii=False)
    assert "god_only" not in json.dumps(pack, ensure_ascii=False)


def test_the_text_face_filter_uses_the_real_visibility_vocabulary() -> None:
    """`TEXT_JUDGE_VISIBILITY` is a copy; this is what keeps it a true one."""
    from typing import get_args

    from harness.forced_event_schema import Visibility

    vocabulary = set(get_args(Visibility))
    assert TEXT_JUDGE_VISIBILITY <= vocabulary
    assert vocabulary - TEXT_JUDGE_VISIBILITY == {"god_only"}


# --- forced environment actions ----------------------------------------------
#
# A forced third-party message is delivered as an `EnvironmentActionItem`, not
# as a `message`: the executor records it as `kind="environment_action"` and,
# when it executed and the PA could see it, writes its content into the PA's
# history and memory as a **user** message. The text pack handled neither.

CANCELLATION_NOTICE = "Hotel Lugano: your booking for the 14th has been cancelled."
NEVER_DELIVERED_NOTICE = "Hotel Lugano: your late checkout is confirmed."


def _environment_action_events() -> list[dict]:
    """The executor's own `environment_action` shape: one notice the PA was
    handed, one neither party saw, one that never ran."""
    return [
        {"t": 0, "kind": "session_start", "session_id": "fe2", "persona": "User A"},
        {
            "t": 10,
            "kind": "environment_action",
            "exchange": 0,
            "action_id": "act_1",
            "action": "deliver_message",
            "source": {"kind": "third_party", "id": "hotel_lugano"},
            "visibility": "both",
            "content": CANCELLATION_NOTICE,
            "status": "executed",
        },
        {
            "t": 20,
            "kind": "environment_action",
            "exchange": 0,
            "action_id": "act_2",
            "action": "deliver_message",
            "source": {"kind": "environment", "id": "authoring_note"},
            "visibility": "god_only",
            "content": GOD_ONLY_NOTE,
            "status": "executed",
        },
        {
            "t": 30,
            "kind": "environment_action",
            "exchange": 0,
            "action_id": "act_3",
            "action": "deliver_message",
            "source": {"kind": "third_party", "id": "hotel_lugano"},
            "visibility": "both",
            "content": NEVER_DELIVERED_NOTICE,
            "status": "missing",
        },
        {
            "t": 40,
            "kind": "pa_turn",
            "forced_event": True,
            "exchange": 0,
            "outbound": "I saw the cancellation. Shall I look for another room?",
            "tool_events": [],
        },
        {"t": 50, "kind": "session_end", "simulator_turns": 0, "beat_count": 0},
    ]


def test_an_executed_pa_visible_environment_action_becomes_a_user_message() -> None:
    """The bug: the PA and its memory were handed the cancellation notice, but
    the text pack skipped the whole kind, so the judge read the PA's reply with
    the input that caused it missing."""
    pack = build_text_judge_pack(_environment_action_events())

    assert [turn["kind"] for turn in pack["turns"]] == ["user_message", "pa_turn"]
    notice = pack["turns"][0]
    assert notice["text"] == CANCELLATION_NOTICE
    # Only the visible content crosses: no action id, no source, no status, and
    # none of the delivery's world-state payload.
    assert sorted(notice) == ["beat", "kind", "t", "text"]
    assert pack["turns"][1]["public_turn"].startswith("I saw the cancellation")


def test_a_god_only_or_unexecuted_environment_action_stays_off_the_text_face() -> None:
    pack = build_text_judge_pack(_environment_action_events())
    blob = json.dumps(pack, ensure_ascii=False)

    assert GOD_ONLY_NOTE not in blob
    assert NEVER_DELIVERED_NOTICE not in blob
    for plumbing in ("act_1", "hotel_lugano", "deliver_message", "executed"):
        assert plumbing not in blob


def test_the_environment_action_filter_matches_what_the_executor_delivers() -> None:
    """`TEXT_JUDGE_PA_DELIVERED_VISIBILITY` mirrors the executor's own
    `item.visibility in {"pa", "both"}` check; this keeps the copy a true one.

    It is narrower than `TEXT_JUDGE_VISIBILITY`: a `user`-only action never
    entered the PA's history, so unlike a `user`-visible `message` it is not
    part of the exchange the text judge scores.
    """
    from typing import get_args

    from harness.forced_event_schema import Visibility

    assert TEXT_JUDGE_PA_DELIVERED_VISIBILITY < TEXT_JUDGE_VISIBILITY
    assert set(get_args(Visibility)) - TEXT_JUDGE_PA_DELIVERED_VISIBILITY == {
        "user", "god_only",
    }


def test_a_simulator_turn_contributes_only_t_beat_and_text() -> None:
    """The `simulator_turn` branch is a whitelist of three
    keys. `consent_decisions` (the evaluation-side truth the PA is blind to),
    `factual_check`, `turn_assessment` and `parse_warnings` never cross, and
    this pin is what keeps them out — not an entry in `TEXT_JUDGE_BANNED_KEYS`.
    """
    consent = [{"attribute": "autonomy_level", "item_id": "a3", "decision": "deny"}]
    events = [
        {
            "t": 1,
            "kind": "simulator_turn",
            "beat": "beat1",
            "message": "Can you book the hotel?",
            "consent_decisions": consent,
            "factual_check": [{"fact": "deposit", "expected": "Friday", "pa_said": ""}],
            "turn_assessment": {"next_beat": "beat2"},
            "parse_warnings": ["trailing whitespace"],
        },
        {"t": 2, "kind": "pa_turn", "beat": "beat1", "outbound": "Sure.", "tool_events": []},
    ]
    pack = build_text_judge_pack(events)

    user_message = pack["turns"][0]
    assert user_message == {
        "t": 1, "kind": "user_message", "beat": "beat1", "text": "Can you book the hotel?",
    }
    blob = json.dumps(pack, ensure_ascii=False)
    for withheld in ("consent_decisions", "deny", "factual_check", "turn_assessment", "parse_warnings"):
        assert withheld not in blob


# ------------------------------------------------------------------------------
# Input boundary — tool_log allow-list, proactive exception
# ------------------------------------------------------------------------------

from harness.judge_pack import (  # noqa: E402
    TEXT_JUDGE_OUTREACH_KEYS,
    TEXT_JUDGE_TOOL_LOG_CONTROL_ARG_KEYS,
    TEXT_JUDGE_TOOL_LOG_RESULT_KEYS,
    TEXT_JUDGE_TRIGGER_KEYS,
    text_judge_tool_log,
)
from pa.agent.ix.render_types import TEXT_JUDGE_BANNED_KEYS  # noqa: E402


def _store_row(t: int, tool: str, args: dict, result: dict, status: str = "ok") -> dict:
    """The exact shape `state_server/store.py` `log_tool` writes."""
    return {"t": t, "timestamp": "2026-10-04T09:00:00+00:00", "run_id": "run_x",
            "user_id": "user_a", "session_id": "acc_001", "tool": tool,
            "args": args, "result_summary": result, "status": status}


def test_tool_log_projection_is_an_allow_list_over_row_args_and_result() -> None:
    rows = text_judge_tool_log([
        _store_row(2, "appointment_create",
                   {"provider_id": "pv_0003", "reason": "knee follow-up",
                    "gate": {"param": "reason", "matched": True, "satisfied": True, "open": True},
                    "_ix_notice": "Booking it now.", "interaction_item_id": "i1"},
                   {"appointment_id": "ap_0009", "status": "booked"}),
        _store_row(1, "contact_search", {"query": "Sarah"},
                   {"matches": 1, "ids": ["ct_0012"], "truncated": False}),
    ])
    assert [row["t"] for row in rows] == [1, 2]
    assert set(rows[1]) == {"t", "tool", "status", "args", "result_summary"}
    assert rows[1]["args"] == {"provider_id": "pv_0003", "reason": "knee follow-up"}
    assert rows[1]["result_summary"] == {"appointment_id": "ap_0009", "status": "booked"}
    assert rows[0]["result_summary"] == {"matches": 1, "ids": ["ct_0012"], "truncated": False}


def test_tool_log_projection_drops_gate_audit_rows_ix_rows_and_refusal_vocabulary() -> None:
    rows = text_judge_tool_log([
        _store_row(0, "gate_loaded",
                   {"gate_tool": "appointment_create", "gate_param": "reason",
                    "task_fact": "appointment_reason", "reveal_beat": "probe",
                    "pre_reveal_write_tools": []},
                   {"loaded": True, "open": False}),
        _store_row(1, "appointment_create", {"gate_tool": "appointment_create", "gate_param": "reason"},
                   {"error": "gate_refusal", "gate_refusal": True, "closed": True}, status="error"),
        _store_row(2, "gate_satisfied",
                   {"gate_tool": "appointment_create", "gate_param": "reason", "task_fact": "x"},
                   {"satisfied": True}),
        _store_row(3, "email_draft_read", {"draft_id": "dr_0001"}, {"error": "not_found"}, status="error"),
        _store_row(4, "scholarly_paper_read", {"paper_id": "pp_0007"},
                   {"error_code": "capability_limit", "boundary_kind": "missing_data"},
                   status="capability_limit"),
        {"t": 5, "tool": "IX_render_autonomy_level", "args": {"plan_id": "p1"},
         "result_summary": {"released": ["a1"]}, "status": "ok"},
    ])
    assert [(row["t"], row["tool"], row["status"]) for row in rows] == [
        (1, "appointment_create", "error"),
        (3, "email_draft_read", "error"),
        (4, "scholarly_paper_read", "capability_limit"),
    ]
    # the refused call stays as an errored call; the store's own vocabulary goes
    assert rows[0]["args"] == {} and rows[0]["result_summary"] == {}
    # a world outcome and the capability boundary are business evidence
    assert rows[1]["result_summary"] == {"error": "not_found"}
    assert rows[2]["result_summary"] == {"error_code": "capability_limit", "boundary_kind": "missing_data"}
    assert "gate" in TEXT_JUDGE_TOOL_LOG_CONTROL_ARG_KEYS
    assert not {"gate_refusal", "closed", "session_closing", "loaded", "open", "satisfied"} & TEXT_JUDGE_TOOL_LOG_RESULT_KEYS


def _proactive_events(decision: str) -> list[dict]:
    delivered = decision == "send"
    blob = _ix_render_blob()
    blob["proactive_decision"] = {"action": decision, "setting": "Low" if decision == "suppress" else "High"}
    blob["held"]["outreach_message"] = "" if delivered else HELD_BODY
    return [
        {"t": 1, "kind": "environment_trigger", "trigger_id": "pickup_deadline",
         "source_type": "deadline", "prior_thread": "Order ready.", "detail": "Closes at 6.",
         "user_impact": "Pickup today."},
        {"t": 2, "kind": "pa_turn", "beat": "proactive_trigger", "proactive_event": True,
         "outbound": "Your order closes at 6 today." if delivered else "",
         "outreach_decision": decision, "outreach_delivered": delivered,
         "tool_events": [
             {"name": "IX_proactive_outreach", "status": "ok", "ix": True,
              "attribute": "proactive_outreach", "setting": "High" if delivered else "Low",
              "outreach_action": decision, "outreach_message": "Your order closes at 6 today.",
              "args": {"setting": "High" if delivered else "Low"}},
         ],
         "ix_render": blob},
    ]


@pytest.mark.parametrize("decision", ["send", "suppress"])
def test_a_proactive_turn_carries_exactly_the_outreach_exception(decision: str) -> None:
    pack = build_text_judge_pack(_proactive_events(decision))

    trigger, turn = pack["turns"]
    assert trigger["kind"] == "environment_trigger"
    assert set(trigger) == {"t", "kind", *TEXT_JUDGE_TRIGGER_KEYS}
    assert trigger["detail"] == "Closes at 6." and trigger["trigger_id"] == "pickup_deadline"
    assert set(turn) == {"t", "kind", "beat", "public_turn", "task_tool_events", *TEXT_JUDGE_OUTREACH_KEYS}
    assert turn["proactive_event"] is True
    assert turn["outreach_decision"] == decision
    assert turn["outreach_delivered"] is (decision == "send")
    assert turn["public_turn"] == ("Your order closes at 6 today." if decision == "send" else "")
    assert turn["task_tool_events"] == []
    # the exception opens nothing else: the setting, the ix_render block and the
    # archived message stay on the IX face, and both scans still hold
    text = json.dumps(pack, ensure_ascii=False)
    assert "proactive_decision" not in text and "outreach_message" not in text
    assert "outreach_action" not in text and HELD_BODY not in text
    assert scan_banned_keys(pack) == [] and scan_banned_strings(pack, needles=LEAK_NEEDLES) == []


def test_the_outreach_exception_stays_disjoint_from_the_banned_keys() -> None:
    assert not set(TEXT_JUDGE_OUTREACH_KEYS) & TEXT_JUDGE_BANNED_KEYS
    assert not set(TEXT_JUDGE_TRIGGER_KEYS) & TEXT_JUDGE_BANNED_KEYS
    assert "proactive_decision" in TEXT_JUDGE_BANNED_KEYS


def test_a_plain_turn_carries_no_outreach_key() -> None:
    pack = build_text_judge_pack(_session_events())
    assert all(key not in pack["turns"][1] for key in TEXT_JUDGE_OUTREACH_KEYS)
