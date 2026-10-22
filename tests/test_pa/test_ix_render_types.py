"""P0 contract tests for `pa.agent.ix.render_types`.

Every construction below is a worked example of a turn budget, a batch, or a
serialised JSON shape.
"""

from __future__ import annotations

import dataclasses
import json
import re
import typing
from pathlib import Path

import pytest

from pa.agent.ix import render_types
from pa.agent.ix.render_types import (
    HELD_STATUS_BY_ATTRIBUTE,
    NO_ASK_HOLD_REASONS,
    held_status,
    BASE_TURNS,
    BATCH_MARKER_RE,
    BOUNDARY_FAILURE_REASONS,
    CAPABILITY_LIMIT_ERROR_CODE,
    CAPABILITY_LIMIT_RESULT_KEYS,
    CAPABILITY_LIMIT_TOOL_STATUS,
    CERTAINTY_MARKER_RE,
    CLOSE_ROUNDS,
    CONSENT_DECISIONS_BLOCK,
    EPISODE_ID_FIELDS,
    GLOBAL_TURN_CAP,
    HELD_STATUS_BY_ATTRIBUTE,
    HOLD_BEARING_ATTRIBUTES,
    HOLD_ROUNDS_HARD_CAP,
    IX_CAPABILITY_RESOLVE_TOOL,
    IX_CONFIRMATION_MESSAGE_FIELD,
    IX_CONTINUE_TOOL_NAMES,
    IX_CONTRACT_VERSION,
    IX_EVENT_NAMES,
    IX_NOTICE_FIELD,
    IX_RENDER_OPTIONAL_KEYS,
    IX_RENDER_SCHEMA_VERSION,
    IX_RENDER_TOOL_NAMES,
    IX_RENDER_TOP_LEVEL_KEYS,
    IX_SELECTION_TOOL_NAMES,
    IX_TOOL_NAME_PREFIX,
    INTERACTION_ITEM_ID_FIELD,
    NON_HOLD_RENDERED_ATTRIBUTES,
    PASS_THROUGH_ATTRIBUTES,
    PER_CALL_INJECTED_FIELDS,
    PROJECTION_NAMES,
    QUESTION_MARKER_RE,
    RENDER_PAYLOAD_FIELDS,
    RENDERED_ATTRIBUTES,
    SIDEBAND_CONSENT_VALUES,
    STOP_REASON_IX_CONTRACT_FAILURE,
    STOP_REASON_IX_GATE,
    STOP_REASON_IX_HOLD,
    TEXT_JUDGE_BANNED_KEYS,
    TURN_BUDGET_PARAMS,
    UNIT_CAP,
    UNKNOWN_TOOL_ACTION_TYPE,
    AuditTrace,
    Batch,
    BoundaryKind,
    BoundaryState,
    CapabilityLimitResult,
    CapabilityStage,
    Certainty,
    ConsentDecision,
    ConsentDecisionValue,
    ConsentRecord,
    GateRecord,
    HeldItem,
    HoldEpisode,
    IXContractVersionError,
    IXHold,
    IXRenderInput,
    MemoryProjection,
    PlanItem,
    ProactiveDecision,
    PublicTurn,
    Question,
    ReleaseDecision,
    RenderedTurn,
    RequiredContext,
    RuntimeHistory,
    StopReason,
    TextJudgeProjection,
    ToolCallRecord,
    ToolResultStatus,
    TurnPayloads,
    UNRECOGNISED_MARKER_RE,
    from_ix_render_metadata,
    text_judge_projection,
    to_ix_render_metadata,
    turn_budget,
)
from pa.agent.ix.specs import IX_PASSIVE_SPECS

# --- worked examples ---------------------------------------------------------

# IX_render_autonomy_level(plan_id="p1", plan=[a1, a2, a3], done=False)
PLAN_A1_A3 = [
    PlanItem(
        id="a1",
        intent="check the stock and hold rules for this issue at Dale's shop",
        is_key=False,
        bound_item_ids=["i1"],
    ),
    PlanItem(
        id="a2",
        intent="draft the extension request to Dale",
        is_key=False,
        bound_item_ids=["i2"],
    ),
    PlanItem(
        id="a3",
        intent="send the email",
        is_key=True,
        bound_item_ids=["i3"],
    ),
]

# One batch each for the calendar, hotel and invoice topics in a
# single turn, each bound to its own interaction_item_id
BATCHES_B1_B3 = [
    Batch(
        batch_id="b1", label="calendar",
        body="I have scheduled Monday morning.", call_binding=["i1"],
    ),
    Batch(
        batch_id="b2", label="hotel",
        body="The hotel still has two room types available.", call_binding=["i2"],
    ),
    Batch(
        batch_id="b3", label="invoice",
        body="I have found the invoice.", call_binding=["i3"],
    ),
]

# OutboundMessage.metadata["ix_render"]
IX_RENDER_EXAMPLE = {
    "schema_version": "ix_render/1.1.0",
    "ix_contract_version": "ix-wrapper/2.1.0",
    "episodes": {
        "autonomy_level": {"episode_id": "p1", "rounds_open": 1, "closed": False},
        "topic_management": {"episode_id": "bs1", "rounds_open": 1, "closed": False},
    },
    "released": {"batches": ["b1"], "questions": ["q1", "q2"], "plan_executed": ["a1"]},
    "held": {
        "batches": [{"batch_id": "b2", "label": "hotel", "body": "…verbatim…"}],
        "questions": [],
        "plan_remaining": [{"id": "a3", "intent": "send the email"}],
        "calls": [
            {
                "interaction_item_id": "i3",
                "plan_item_id": "a3",
                "tool": "email_message_send",
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
            "user_evidence": "go ahead and send it",
            "evidence_verified": True,
            "tool_changed": False,
        }
    ],
    "notices": [
        {"order": 0, "tool": "document_read", "text": "Let me look at your pull list first"}
    ],
    "completion": {"released": True, "text": "Monday morning is now scheduled."},
    "reasoning": {
        "shown": True,
        "source": "user_facing_rationale",
        "provider_reasoning_present": False,
    },
    "certainties": [{"ref": "c1", "level": "low", "rendered": "(confidence: low)"}],
    "proactive_decision": None,
    "trace": ["topic_management=One-at-a-time -> release 1/3", "..."],
}


# --- constants ---------------------------------------------------------------


def test_contract_and_schema_versions_are_present() -> None:
    assert IX_CONTRACT_VERSION == "ix-wrapper/2.1.0"
    assert IX_RENDER_SCHEMA_VERSION == "ix_render/1.1.0"


def test_beat_budget_constants_match_the_plan() -> None:
    assert (BASE_TURNS, CLOSE_ROUNDS, GLOBAL_TURN_CAP) == (3, 1, 8)
    assert UNIT_CAP == {
        "autonomy_level": 3,
        "topic_management": 3,
        "information_elicitation": 3,
        "capability_boundary": 1,
    }
    assert HOLD_BEARING_ATTRIBUTES == frozenset(UNIT_CAP)


def test_per_turn_tool_names_and_injected_field_names() -> None:
    assert IX_RENDER_TOOL_NAMES == {
        attr: f"IX_render_{attr}"
        for attr in (
            "reasoning_visibility",
            "uncertainty_expression",
            "process_visibility",
            "autonomy_level",
            "topic_management",
            "information_elicitation",
        )
    }
    assert IX_CONTINUE_TOOL_NAMES == {
        attr: f"IX_continue_{attr}" for attr in HOLD_BEARING_ATTRIBUTES
    }
    assert IX_CAPABILITY_RESOLVE_TOOL == "IX_capability_resolve"
    assert PER_CALL_INJECTED_FIELDS == (
        "_ix_notice",
        "interaction_item_id",
        "_ix_confirmation_message",
    )
    assert (IX_NOTICE_FIELD, INTERACTION_ITEM_ID_FIELD, IX_CONFIRMATION_MESSAGE_FIELD) == (
        PER_CALL_INJECTED_FIELDS
    )
    assert CONSENT_DECISIONS_BLOCK == "consent_decisions"
    assert CAPABILITY_LIMIT_ERROR_CODE == "capability_limit"
    assert UNKNOWN_TOOL_ACTION_TYPE == "external_action"


def test_five_projection_names() -> None:
    assert PROJECTION_NAMES == (
        "public_turn",
        "runtime_history",
        "memory_projection",
        "text_judge_projection",
        "audit_trace",
    )


# --- turn_budget ---------------------------------------------------


def test_turn_budget_worked_cases() -> None:
    # no hold-bearing attribute this beat -> keep the existing MAX_TURNS_PER_BEAT
    assert turn_budget([]) == 3
    assert turn_budget(["tone_formality", "verbosity", "solution_breadth"]) == 3
    # autonomy alone: min(8, 3 + 3 + 1)
    assert turn_budget(["autonomy_level"]) == 7
    # capability alone: min(8, 3 + 1 + 1)
    assert turn_budget(["capability_boundary"]) == 5
    # autonomy + topic: 3 + 3 + 3 + 1 = 10 -> hits the global hard cap
    assert turn_budget(["autonomy_level", "topic_management"]) == GLOBAL_TURN_CAP
    # all four hold kinds active: 3 + (3+3+3+1) + 1 = 14 -> hits the global hard cap
    assert turn_budget(sorted(HOLD_BEARING_ATTRIBUTES)) == GLOBAL_TURN_CAP


def test_turn_budget_does_not_leak_the_setting() -> None:
    """`UNIT_CAP` is per attribute: `Organize` and `One-at-a-time` get exactly the same budget."""
    assert turn_budget(["topic_management"]) == turn_budget(["topic_management"])
    # a repeated skill name is not billed twice
    assert turn_budget(["autonomy_level", "autonomy_level"]) == turn_budget(["autonomy_level"])


# --- attribute grouping ------------------------------------------------------


def test_every_rendered_attribute_is_hold_bearing_or_listed_as_non_hold() -> None:
    assert RENDERED_ATTRIBUTES <= set(IX_PASSIVE_SPECS)
    for attribute in RENDERED_ATTRIBUTES:
        assert (attribute in UNIT_CAP) ^ (attribute in NON_HOLD_RENDERED_ATTRIBUTES), attribute
    assert NON_HOLD_RENDERED_ATTRIBUTES == {
        "reasoning_visibility",
        "uncertainty_expression",
        "process_visibility",
        "proactive_outreach",
    }


def test_rendered_and_pass_through_partition_the_taxonomy() -> None:
    assert RENDERED_ATTRIBUTES | PASS_THROUGH_ATTRIBUTES == set(IX_PASSIVE_SPECS)
    assert not (RENDERED_ATTRIBUTES & PASS_THROUGH_ATTRIBUTES)


def test_hold_bearing_attributes_have_episode_id_field_and_held_status() -> None:
    assert set(EPISODE_ID_FIELDS) == HOLD_BEARING_ATTRIBUTES
    assert set(HELD_STATUS_BY_ATTRIBUTE) == HOLD_BEARING_ATTRIBUTES
    assert set(HELD_STATUS_BY_ATTRIBUTE.values()) == {
        "held_for_confirmation",
        "held_for_handoff",
        "held_for_batch",
        "held_for_questions",
    }


# --- construction from the plan's examples -----------------------------------


def test_plan_items_a1_a3_from_section_3_3() -> None:
    assert [item.id for item in PLAN_A1_A3] == ["a1", "a2", "a3"]
    assert [item.is_key for item in PLAN_A1_A3] == [False, False, True]
    assert PLAN_A1_A3[2].bound_item_ids == ["i3"]


def test_the_real_call_carries_binding_id_and_confirmation_message() -> None:
    """email_message_send(..., interaction_item_id="i3", _ix_confirmation_message=...)"""
    call = ToolCallRecord(
        order=2,
        tool_name="email_message_send",
        args={
            "to": "dale@…", "subject": "…",
            "body": "<the body the draft tool just returned>",
        },
        tool_call_id="call_3",
        interaction_item_id="i3",
        ask_text=(
            "I have written the extension request from the draft above, "
            "addressed to dale@… — shall I send it now?"
        ),
        action_type="external_action",
        executed=False,
    )
    assert call.interaction_item_id == PLAN_A1_A3[2].bound_item_ids[0]
    assert call.action_type == "external_action"
    assert call.executed is False


def test_batches_bind_only_interaction_item_ids() -> None:
    assert [batch.batch_id for batch in BATCHES_B1_B3] == ["b1", "b2", "b3"]
    bindings = [binding for batch in BATCHES_B1_B3 for binding in batch.call_binding]
    assert bindings == ["i1", "i2", "i3"]


def test_hold_episode_and_ix_hold_are_episode_scoped() -> None:
    episode = HoldEpisode(
        episode_id="p1",
        attribute="autonomy_level",
        opened_at_turn=1,
        rounds_open=1,
        plan_remaining=[PLAN_A1_A3[2]],
        last_gate=GateRecord(
            attribute="autonomy_level",
            item_id="a3",
            reason="external_action_floor",
            asked_at_turn=1,
            action_type="external_action",
        ),
        calls_held=[
            {
                "interaction_item_id": "i3",
                "plan_item_id": "a3",
                "tool": "email_message_send",
                "reason": "external_action_floor",
                "status": "held_for_confirmation",
            }
        ],
    )
    hold = IXHold(open_episodes={"autonomy_level": episode})
    assert hold.open_episodes["autonomy_level"].closed is False
    assert hold.open_episodes["autonomy_level"].rounds_open == 1
    assert hold.finished_episodes == []
    # closed / rounds_open / last_gate / cancelled all belong to the episode, not to IXHold
    assert not hasattr(hold, "rounds_open")
    assert not hasattr(hold, "closed")


def test_frozen_dataclasses_reject_rebinding() -> None:
    item = PLAN_A1_A3[0]
    with pytest.raises(Exception):
        item.id = "a9"  # type: ignore[misc]


def test_boundary_state_question_and_consent_shapes() -> None:
    boundary = BoundaryState(
        blocked_goal="get the full text of the non-Markovian boundary dephasing paper",
        boundary_kind="missing_data",
        recovery_remaining=[
            PlanItem(id="r1", intent="find the collaborator's email address"),
            PlanItem(id="r2", intent="switch to an alternative paper on the same topic"),
        ],
        handoff_message=(
            "This one is not in the searchable catalogue. Shall I find Imran's "
            "email address and write to ask him for the manuscript?"
        ),
    )
    assert boundary.boundary_kind == "missing_data"
    assert [item.id for item in boundary.recovery_remaining] == ["r1", "r2"]

    question = Question(id="q1", text="What is the budget ceiling?", about="budget")
    assert question.id == "q1"

    # The sideband carries only {interaction_item_id, decision}
    assert ConsentDecision(interaction_item_id="i3", decision="allow").decision == "allow"
    # Carrier D: the PA's verdict + user_evidence for the audit
    assert ReleaseDecision(
        id="a3", decision="release", user_evidence="go ahead and send it"
    ).decision == "release"


def test_render_input_and_rendered_turn_round_trip_the_gate_round() -> None:
    payloads = TurnPayloads(
        user_facing_rationale="Only em_0100 of the two mentions 4B.",
        certainties=[Certainty(ref="c1", level="low")],
        completion_message="Monday morning is now scheduled.",
        plan_id="p1",
        plan=PLAN_A1_A3,
        batch_set_id="bs1",
        item_count=3,
        batches=BATCHES_B1_B3,
    )
    render_input = IXRenderInput(
        turn_index=1,
        segments=["Check the stock first, then draft."],
        provider_reasoning=["<think>...</think>"],
        calls=[
            ToolCallRecord(
                order=0,
                tool_name="document_read",
                tool_call_id="call_1",
                notice="Let me look at your pull list first",
                interaction_item_id="i1",
                action_type="read",
                executed=True,
            ),
            ToolCallRecord(
                order=1,
                tool_name="email_message_send",
                tool_call_id="call_2",
                interaction_item_id="i3",
                ask_text="Shall I send it now?",
                action_type="external_action",
                executed=False,
            ),
        ],
        gate=GateRecord(
            attribute="autonomy_level",
            item_id="a3",
            reason="external_action_floor",
            action_type="external_action",
        ),
        gate_remaining=[PLAN_A1_A3[2]],
        payloads=payloads,
        consent_decisions=[ConsentDecision(interaction_item_id="i3", decision="deny")],
    )
    assert render_input.payloads.plan_id == "p1"
    assert render_input.gate is not None and render_input.gate.item_id == "a3"

    rendered = RenderedTurn(
        message="Shall I send it now?",
        held=[
            HeldItem(
                status="held_for_confirmation",
                interaction_item_id="i3",
                attribute="autonomy_level",
                tool="email_message_send",
                plan_item_id="a3",
                episode_id="p1",
                reason="external_action_floor",
            )
        ],
        executed_call_ids=["call_1"],
        held_call_ids=["call_2"],
        trace=["autonomy_level=Self-directed -> gate a3 (external_action_floor)"],
    )
    # Mixed tool calls: the two lists are disjoint, and together they are
    # every call this turn declared
    assert set(rendered.executed_call_ids) & set(rendered.held_call_ids) == set()
    assert set(rendered.executed_call_ids) | set(rendered.held_call_ids) == {
        call.tool_call_id for call in render_input.calls
    }


def test_required_context_slot_shape() -> None:
    ctx = RequiredContext(
        session_active_ix=("autonomy_level",),
        hold=IXHold(
            open_episodes={
                "autonomy_level": HoldEpisode(
                    episode_id="p1",
                    attribute="autonomy_level",
                    plan_remaining=[PLAN_A1_A3[2]],
                )
            }
        ),
        response_interaction_item_ids=("i3",),
        produces_user_visible_message=True,
        capability_limit_seen=False,
    )
    assert ctx.hold.open_episodes["autonomy_level"].plan_remaining[0].id == "a3"


# --- projections ------------------------------------------------------


def test_projection_containers_hold_the_turn_envelope() -> None:
    public = PublicTurn(message="Shall I send it now?")
    user_message = {"role": "user", "content": "Please handle the extension for me."}
    assistant_message = {"role": "assistant", "content": public.message}

    history = RuntimeHistory(
        user_message=user_message,
        tool_protocol=[
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ],
        assistant_message=assistant_message,
    )
    # hard constraint 1: the current user message exactly once; the last
    # assistant message equals public_turn verbatim
    assert [m for m in history.messages if m.get("role") == "user"] == [user_message]
    assert history.messages[-1]["content"] == public.message
    declared = [
        call["id"]
        for m in history.tool_protocol
        for call in m.get("tool_calls", [])
    ]
    results = [m["tool_call_id"] for m in history.tool_protocol if m.get("role") == "tool"]
    assert declared == results

    # hard constraint 2: memory_projection is exactly two messages, and the
    # assistant one equals public_turn verbatim
    memory = MemoryProjection(user_message=user_message, assistant_message=assistant_message)
    assert len(memory.messages) == 2
    assert memory.messages[1]["content"] == public.message

    text_face = TextJudgeProjection(
        public_turn=public.message,
        task_tool_events=[{"name": "document_read", "args": {}, "status": "ok", "t": 0}],
    )
    assert text_face.public_turn == public.message
    assert all(
        key not in event for event in text_face.task_tool_events for key in TEXT_JUDGE_BANNED_KEYS
    )

    audit = AuditTrace(
        segments=["Check the stock first, then draft."],
        provider_reasoning=["<think>...</think>"],
        held=[
            HeldItem(
                status="held_for_batch",
                interaction_item_id="i2",
                attribute="topic_management",
                tool="hotel_search",
                batch_id="b2",
            )
        ],
        consent=[
            ConsentRecord(
                attribute="autonomy_level",
                item_id="a3",
                pa_decision="release",
                sideband_decision="deny",
                released=False,
            )
        ],
        trace=["topic_management=One-at-a-time -> release 1/3"],
        ix_render=IX_RENDER_EXAMPLE,
    )
    assert audit.consent[0].released is False
    assert audit.ix_render["schema_version"] == IX_RENDER_SCHEMA_VERSION


def test_text_judge_projection_builder_is_deferred_to_p4() -> None:
    with pytest.raises(NotImplementedError):
        text_judge_projection(RenderedTurn(message="x"), IXRenderInput())


# --- `ix_render` JSON --------------------------------------------------------------


def test_text_judge_banned_keys_cover_every_section_4_4_key() -> None:
    # The optional keys are the ones a turn may simply not have: no carrier
    # block, no hold exit, no closing round. The 1.0.0 records that predate
    # them never wrote any of the three.
    assert set(IX_RENDER_TOP_LEVEL_KEYS) - IX_RENDER_OPTIONAL_KEYS == set(
        IX_RENDER_EXAMPLE
    )
    assert IX_RENDER_OPTIONAL_KEYS <= set(IX_RENDER_TOP_LEVEL_KEYS)
    assert set(IX_RENDER_TOP_LEVEL_KEYS) <= TEXT_JUDGE_BANNED_KEYS
    assert "ix_render" in TEXT_JUDGE_BANNED_KEYS
    assert "ix_directives" in TEXT_JUDGE_BANNED_KEYS
    for key in ("setting", "evidence", "application", *PER_CALL_INJECTED_FIELDS):
        assert key in TEXT_JUDGE_BANNED_KEYS


def test_from_ix_render_metadata_accepts_the_section_4_4_example() -> None:
    assert from_ix_render_metadata(IX_RENDER_EXAMPLE) == IX_RENDER_EXAMPLE


def test_ix_render_metadata_json_round_trip() -> None:
    blob = to_ix_render_metadata(
        RenderedTurn(
            message="Shall I send it now?",
            proactive=None,
            trace=list(IX_RENDER_EXAMPLE["trace"]),
        ),
        episodes=IX_RENDER_EXAMPLE["episodes"],
        released=IX_RENDER_EXAMPLE["released"],
        held=IX_RENDER_EXAMPLE["held"],
        gate=GateRecord(
            attribute="autonomy_level",
            item_id="a3",
            reason="external_action_floor",
            asked_at_turn=1,
            action_type="external_action",
        ),
        consent=[
            ConsentRecord(
                attribute="autonomy_level",
                item_id="a3",
                pa_decision="release",
                sideband_decision="allow",
                released=True,
                user_evidence="go ahead and send it",
                evidence_verified=True,
            )
        ],
        notices=IX_RENDER_EXAMPLE["notices"],
        completion=IX_RENDER_EXAMPLE["completion"],
        reasoning=IX_RENDER_EXAMPLE["reasoning"],
        certainties=IX_RENDER_EXAMPLE["certainties"],
    )
    assert blob == IX_RENDER_EXAMPLE
    # A turn with no carrier block writes no `raw_blocks`, and the reader takes
    # it back exactly as it was written.
    assert list(blob) == [
        key for key in IX_RENDER_TOP_LEVEL_KEYS if key not in IX_RENDER_OPTIONAL_KEYS
    ]
    assert from_ix_render_metadata(json.loads(json.dumps(blob))) == IX_RENDER_EXAMPLE

    # …and a turn that had one carries it verbatim, on the audit plane only.
    with_block = to_ix_render_metadata(
        RenderedTurn(message="x"),
        raw_blocks=[{
            "kind": "certainties",
            "text": (
                "<certainties>\n"
                "- ref: c1 | level: low | claim: Monday morning is free\n"
                "</certainties>"
            ),
        }],
    )
    assert with_block["raw_blocks"][0]["kind"] == "certainties"
    assert "raw_blocks" in TEXT_JUDGE_BANNED_KEYS
    assert from_ix_render_metadata(json.loads(json.dumps(with_block)))["raw_blocks"] == (
        with_block["raw_blocks"]
    )


def test_the_stop_reason_and_the_closing_round_are_optional_audit_keys() -> None:
    """`ix_hold` had no record in any artefact: `pa_turn` carries no
    `stop_reason`, and the closing request wrote no field, no event and
    no trace line — it was visible only as an extra provider call."""
    blob = to_ix_render_metadata(
        RenderedTurn(message="x", trace=["topic_management episode bs1 -> held, round 1"]),
        stop_reason="ix_hold",
        closing_iteration=True,
    )
    assert blob["stop_reason"] == "ix_hold"
    assert blob["closing_iteration"] is True
    assert blob["trace"][-1] == "closing iteration ran"
    assert from_ix_render_metadata(json.loads(json.dumps(blob)))["stop_reason"] == "ix_hold"

    empty = to_ix_render_metadata(RenderedTurn(message="x"), closing_iteration=False)
    assert empty["closing_iteration"] is False
    assert empty["trace"] == ["closing iteration empty"]

    # Neither key is written by a turn that had neither, and both are banned
    # from the text face like every other `ix_render` key.
    plain = to_ix_render_metadata(RenderedTurn(message="x"))
    assert "stop_reason" not in plain and "closing_iteration" not in plain
    assert {"stop_reason", "closing_iteration"} <= TEXT_JUDGE_BANNED_KEYS


def test_a_1_0_0_record_still_reads_under_the_1_1_0_reader() -> None:
    """Contract 12 · inside one major only optional fields are added, so an old record needs no legacy reader."""
    old = {**IX_RENDER_EXAMPLE, "schema_version": "ix_render/1.0.0"}
    assert from_ix_render_metadata(old)["schema_version"] == "ix_render/1.0.0"
    with pytest.raises(IXContractVersionError):
        from_ix_render_metadata({**IX_RENDER_EXAMPLE, "schema_version": "ix_render/2.0.0"})


def test_proactive_decision_reaches_the_blob() -> None:
    """Contract 11: under `suppress` the candidate message goes only to `held.outreach_message`, and `public_turn` is empty."""
    decision = ProactiveDecision(
        action="suppress",
        setting="Low",
        message="The flight you booked has been rescheduled.",
    )
    blob = to_ix_render_metadata(
        RenderedTurn(message="", proactive=decision),
        held={
            "batches": [],
            "questions": [],
            "plan_remaining": [],
            "calls": [],
            "cancelled": [],
            "outreach_message": decision.message,
        },
    )
    assert blob["proactive_decision"] == {
        "action": "suppress",
        "setting": "Low",
        "message": "The flight you booked has been rescheduled.",
    }
    assert blob["held"]["outreach_message"] == decision.message
    assert from_ix_render_metadata(blob)["proactive_decision"] is not None


def test_version_boundary_fails_fast() -> None:
    """Contract 12: a version mismatch fails fast, and is never silently reinterpreted under the new semantics."""
    legacy = dict(IX_RENDER_EXAMPLE)
    legacy["schema_version"] = "ix_render/0.9.0"
    with pytest.raises(IXContractVersionError):
        from_ix_render_metadata(legacy)

    legacy = dict(IX_RENDER_EXAMPLE)
    legacy["ix_contract_version"] = "ix-wrapper/0.9.0"
    with pytest.raises(IXContractVersionError):
        from_ix_render_metadata(legacy)

    missing = {k: v for k, v in IX_RENDER_EXAMPLE.items() if k != "schema_version"}
    with pytest.raises(ValueError):
        from_ix_render_metadata(missing)

    extra = {**IX_RENDER_EXAMPLE, "ix_directives": []}
    with pytest.raises(ValueError):
        from_ix_render_metadata(extra)


# --- the rendered attributes' setting domains ----------------------------------


def test_every_setting_literal_is_exactly_its_spec_settings() -> None:
    """A `*Setting` Literal freezes the taxonomy domain and nothing outside it.

    There is no option beyond the attribute's own settings: the PA always names
    one of them, so the frozen type domain and `IX_PASSIVE_SPECS` have to agree
    member for member and in order.
    """
    setting_literals = {
        name: alias
        for name, alias in vars(render_types).items()
        if name.endswith("Setting") and typing.get_args(alias)
    }
    # one value domain per rendered attribute, eight in all
    assert len(setting_literals) == len(RENDERED_ATTRIBUTES)
    for name, alias in setting_literals.items():
        stem = name.removesuffix("Setting")
        attribute = re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()
        assert attribute in RENDERED_ATTRIBUTES, name
        assert typing.get_args(alias) == IX_PASSIVE_SPECS[attribute].settings, name


# --- the enums' value domains --------------------------------------------------


def test_enumerations_freeze_their_value_sets() -> None:
    assert set(typing.get_args(CapabilityStage)) == {
        "select_setting",
        "declare_boundary",
        "apply_setting",
    }
    assert set(typing.get_args(BoundaryKind)) == {
        "missing_data",
        "missing_capability",
        "action_not_permitted",
        "external_dependency",
    }
    assert set(typing.get_args(StopReason)) == {
        "ix_gate",
        "ix_hold",
        "ix_contract_failure",
    }
    assert (
        STOP_REASON_IX_GATE, STOP_REASON_IX_HOLD, STOP_REASON_IX_CONTRACT_FAILURE
    ) == ("ix_gate", "ix_hold", "ix_contract_failure")
    assert set(typing.get_args(ToolResultStatus)) == {"ok", "error", "capability_limit"}
    assert CAPABILITY_LIMIT_TOOL_STATUS in typing.get_args(ToolResultStatus)
    assert IX_EVENT_NAMES == (
        "ix_binding_invalid",
        "ix_gate_no_message",
        "ix_render_payload_missing",
        "ix_render_payload_forced",
        "ix_release_missing",
        "ix_release_folded",
        "ix_release_spent",
        "ix_release_inherited",
        "ix_release_preauthorised",
        "ix_preauthorisation_expired",
        "ix_consent_denied",
        "ix_consent_unresolved",
        "ix_consent_unasked",
        "ix_consent_tool_changed",
        "ix_selection_invalid",
        "ix_selection_duplicate",
        "ix_batch_undivided",
        "ix_batch_rebound",
        "ix_questions_undivided",
        "ix_questions_unmarked",
        "ix_hold_forced_close",
        "ix_hold_closing_tool_call_ignored",
        "ix_close_no_message",
        "ix_completion_without_content",
        "ix_markup_stripped",
        "ix_dsml_calls_recovered",
        "ix_certainty_claim_not_found",
        "ix_certainties_all_dropped",
        "ix_leadin_dropped",
        "ix_rationale_work_log",
        "ix_claims_unexecuted_action",
        "ix_boundary_unresolved",
        "ix_prose_ask_unbound",
        "ix_empty_reply",
        "beat_gate_turn",
        "provider_truncated",
        "turn_truncated",
        "provider_error_retried",
        "tool_failure_strike",
        "tool_retry_cap_reached",
        "ix_reselection_ignored",
    )
    assert len(set(IX_EVENT_NAMES)) == len(IX_EVENT_NAMES)
    # `ix_boundary_unresolved` is a failure diagnostic,
    # and its reason has only two values
    assert BOUNDARY_FAILURE_REASONS == ("missing", "invalid_payload")
    assert "ix_boundary_unresolved" in IX_EVENT_NAMES


# --- how the tool names are derived -------------------------------------


def test_render_tools_cover_six_of_the_eight_rendered_attributes() -> None:
    assert len(RENDERED_ATTRIBUTES) == 8
    assert len(IX_RENDER_TOOL_NAMES) == 6
    assert set(IX_RENDER_TOOL_NAMES) < RENDERED_ATTRIBUTES
    # the other two rendered attributes each have their own carrier and do not
    # go through IX_render_*
    assert RENDERED_ATTRIBUTES - set(IX_RENDER_TOOL_NAMES) == {
        # carrier E: the forced IX_capability_resolve
        "capability_boundary",
        # stays on IX_proactive_outreach, re-offered on every event turn
        "proactive_outreach",
    }
    for names in (
        IX_RENDER_TOOL_NAMES.values(),
        IX_CONTINUE_TOOL_NAMES.values(),
        IX_SELECTION_TOOL_NAMES,
        (IX_CAPABILITY_RESOLVE_TOOL,),
    ):
        assert all(name.startswith(IX_TOOL_NAME_PREFIX) for name in names)


def test_selection_tool_names_match_the_fourteen_passive_specs() -> None:
    """The literal constants must agree with `specs.py`, the source of truth (the module deliberately does not import specs; see its comment)."""
    assert len(IX_SELECTION_TOOL_NAMES) == 14
    assert IX_SELECTION_TOOL_NAMES == tuple(spec.name for spec in IX_PASSIVE_SPECS.values())
    assert set(IX_SELECTION_TOOL_NAMES) == {
        f"{IX_TOOL_NAME_PREFIX}{attribute}" for attribute in IX_PASSIVE_SPECS
    }


def test_render_payload_fields_map_every_tool_to_real_turn_payload_fields() -> None:
    assert set(RENDER_PAYLOAD_FIELDS) == set(IX_RENDER_TOOL_NAMES.values())
    payload_fields = {f.name for f in dataclasses.fields(TurnPayloads)}
    listed = [name for names in RENDER_PAYLOAD_FIELDS.values() for name in names]
    assert set(listed) <= payload_fields
    # one payload field hangs off exactly one carrier tool
    assert len(listed) == len(set(listed))
    # every field no IX_render_* claims belongs to carrier D / E
    assert payload_fields - set(listed) == {
        "boundary_id",  # carrier E · IX_capability_resolve
        "boundary",  # carrier E · IX_capability_resolve
        "release_decisions",  # carrier D · IX_continue_<attribute>
        "closing_message",  # carrier D · IX_continue_<attribute>
    }


# --- contract 5 · the capability_limit result shape ------------


def test_capability_limit_result_shape() -> None:
    assert CAPABILITY_LIMIT_RESULT_KEYS == ("ok", "error_code", "boundary_kind")
    # The module enables `from __future__ import annotations`, so the TypedDict
    # annotations are strings at runtime and `__required_keys__` cannot see
    # `NotRequired` (PEP 563) -- static checkers still read it correctly.
    # So the check here is against the declaration itself: the three required
    # keys + the one optional key `detail`.
    hints = typing.get_type_hints(CapabilityLimitResult, include_extras=True)
    assert tuple(hints)[:3] == CAPABILITY_LIMIT_RESULT_KEYS
    assert set(hints) == {*CAPABILITY_LIMIT_RESULT_KEYS, "detail"}
    assert typing.get_origin(hints["detail"]) is typing.NotRequired
    assert typing.get_args(hints["ok"]) == (False,)
    assert typing.get_args(hints["error_code"]) == (CAPABILITY_LIMIT_ERROR_CODE,)
    assert hints["boundary_kind"] == BoundaryKind
    # none of the three required keys is NotRequired
    assert all(
        typing.get_origin(hints[key]) is not typing.NotRequired
        for key in CAPABILITY_LIMIT_RESULT_KEYS
    )

    # acc_026: the entry is searchable but the full text is not obtainable (the
    # fixture explicitly marks it unobtainable)
    result: CapabilityLimitResult = {
        "ok": False,
        "error_code": CAPABILITY_LIMIT_ERROR_CODE,
        "boundary_kind": "missing_data",
        "detail": "fulltext_available: false",
    }
    assert all(key in result for key in CAPABILITY_LIMIT_RESULT_KEYS)
    assert result["ok"] is False
    assert result["boundary_kind"] in typing.get_args(BoundaryKind)

    # detail may be omitted
    minimal: CapabilityLimitResult = {
        "ok": False,
        "error_code": "capability_limit",
        "boundary_kind": "missing_capability",
    }
    assert set(minimal) == set(CAPABILITY_LIMIT_RESULT_KEYS)


# --- contract 1 · RequiredContext's marker input ------------------------


def test_required_context_carries_response_text_and_markers() -> None:
    text = (
        "[[batch:b1]] I have scheduled Monday morning."
        "[[batch:b2]] The hotel still has two room types available."
        "[[q:q1]] What is the budget ceiling?"
        "[[c:c1]] I am not too sure about this one."
    )
    assert BATCH_MARKER_RE.findall(text) == ["b1", "b2"]
    assert QUESTION_MARKER_RE.findall(text) == ["q1"]
    assert CERTAINTY_MARKER_RE.findall(text) == ["c1"]

    ctx = RequiredContext(
        session_active_ix=("topic_management",),
        response_text=text,
        response_markers=(
            *(f"batch:{ref}" for ref in BATCH_MARKER_RE.findall(text)),
            *(f"q:{ref}" for ref in QUESTION_MARKER_RE.findall(text)),
            *(f"c:{ref}" for ref in CERTAINTY_MARKER_RE.findall(text)),
        ),
    )
    assert ctx.response_markers == ("batch:b1", "batch:b2", "q:q1", "c:c1")
    # A [[batch:…]] in the prose makes IX_render_topic_management required,
    # even on a turn with no task call
    assert ctx.response_tool_names == ()
    assert any(marker.startswith("batch:") for marker in ctx.response_markers)

    # the defaults for a pure-conversation turn: no marker and no task call ->
    # neither offered nor reminded about
    blank = RequiredContext()
    assert (blank.response_text, blank.response_markers) == ("", ())


@pytest.mark.parametrize(
    "pattern,shapes",
    [
        (QUESTION_MARKER_RE, ("[[q:q1]]", "[[q1]]", "[[question:q1]]")),
        (BATCH_MARKER_RE, ("[[batch:b1]]", "[[b1]]", "[[b:b1]]")),
        (CERTAINTY_MARKER_RE, ("[[c:c1]]", "[[c1]]", "[[certainty:c1]]")),
    ],
)
def test_markers_are_read_in_every_shape_the_model_writes(pattern, shapes) -> None:
    """The canonical form is `[[q:q1]]`; the lenient ones cost nothing to read.

    Live, both acc_009 runs wrote `[[q1]]` on every question. The strict pattern
    matched none of them, so `released.questions` was `[]` on all ten
    elicitation turns and the raw bracket was delivered to the user verbatim.
    """
    for shape in shapes:
        assert pattern.findall(f"prose {shape} more prose") == [shape[2:-2].split(":")[-1]]
    # `[[q:1]]` — the prefix is the kind, the capture is the id, whatever it looks like
    kind = shapes[0][2]
    assert pattern.findall(f"[[{kind}:1]]") == ["1"]


def test_a_bare_marker_of_one_kind_is_never_read_as_another() -> None:
    """Leniency stops at the kind letter: `[[b1]]` is a batch and only a batch."""
    text = "[[b1]] batch prose [[q1]] a question [[c1]] a claim"
    assert BATCH_MARKER_RE.findall(text) == ["b1"]
    assert QUESTION_MARKER_RE.findall(text) == ["q1"]
    assert CERTAINTY_MARKER_RE.findall(text) == ["c1"]


def test_a_marker_shape_nobody_recognises_is_still_a_marker() -> None:
    """It carries no usable id, and it must not reach the user either."""
    text = "before [[topic:1]] after [[q:q1]] fine"
    for pattern in (BATCH_MARKER_RE, QUESTION_MARKER_RE, CERTAINTY_MARKER_RE):
        assert "topic" not in "".join(pattern.findall(text))
    # The catch-all sees every bracket pair; the renderer runs it last, on what
    # the three known kinds left behind.
    assert [match.group(0) for match in UNRECOGNISED_MARKER_RE.finditer(text)] == [
        "[[topic:1]]", "[[q:q1]]",
    ]


# --- contract 10 · the fail-closed action type default ------------------


def test_action_type_defaults_are_fail_closed() -> None:
    gate = GateRecord(attribute="autonomy_level", item_id="a3", reason="external_action_floor")
    call = ToolCallRecord(order=0, tool_name="unknown_tool")
    assert gate.action_type == UNKNOWN_TOOL_ACTION_TYPE == "external_action"
    assert call.action_type == UNKNOWN_TOOL_ACTION_TYPE
    # status is narrowed to three stable codes, no longer an open string
    assert call.status == "ok"
    assert set(typing.get_args(ToolResultStatus)) == {"ok", "error", "capability_limit"}


# --- contract 9 · TURN_BUDGET_PARAMS ---------------------------


def test_turn_budget_params_reproduce_turn_budget() -> None:
    assert TURN_BUDGET_PARAMS == {
        "base": BASE_TURNS,
        "close_rounds": CLOSE_ROUNDS,
        "global_cap": GLOBAL_TURN_CAP,
        "unit_cap": dict(UNIT_CAP),
    }
    # A pure-data copy: the harness can inject or serialise it without reaching
    # this module's definition.
    assert TURN_BUDGET_PARAMS["unit_cap"] is not UNIT_CAP

    def from_params(active_skills: list[str]) -> int:
        unit_cap = TURN_BUDGET_PARAMS["unit_cap"]
        units = sum(
            unit_cap[attr] for attr in dict.fromkeys(active_skills) if attr in unit_cap
        )
        if units == 0:
            return TURN_BUDGET_PARAMS["base"]
        return min(
            TURN_BUDGET_PARAMS["global_cap"],
            TURN_BUDGET_PARAMS["base"] + units + TURN_BUDGET_PARAMS["close_rounds"],
        )

    for active in (
        [],
        ["tone_formality", "verbosity"],
        ["autonomy_level"],
        ["capability_boundary"],
        ["autonomy_level", "topic_management"],
        sorted(HOLD_BEARING_ATTRIBUTES),
    ):
        assert from_params(active) == turn_budget(active), active


#: Every `reason` `pa/agent/runner.py` can put on a held row, so the invariant
#: below is checked against the real set and a new one has to be classified.
ALL_HOLD_REASONS = (
    "after_open_gate", "questions_gate", "release_spent", "batch_not_released",
    "release_missing", "consent_unasked", "consent_deny", "consent_unclear",
    "consent_missing", "external_action_floor", "draft_gate",
    "internal_write_gate", "binding_invalid", "batch_undivided", "handoff",
    "ix_selection_invalid", "capability_handoff",
)


def test_every_hold_reason_the_runner_emits_is_known_here() -> None:
    """The invariant below is only worth as much as this list's completeness.

    A new hold reason has to be classified as asking or not asking, so it is
    read off the runner rather than trusted to stay in step by hand.
    """
    source = (Path(__file__).resolve().parents[2] / "pa/agent/runner.py").read_text()
    emitted: set[str] = set()
    for line in source.splitlines():
        # `attribute, reason = <attribute>, "<reason>"` — the reason is second.
        pair = re.search(r'\battribute, reason = .+?, "([a-z_]+)"', line)
        if pair:
            emitted.add(pair.group(1))
            continue
        # `reason = "<reason>"` / `reason="<reason>"`, but never `stop_reason`.
        single = re.search(r'(?<![_\w])reason\s*=\s*"([a-z_]+)"', line)
        if single:
            emitted.add(single.group(1))
    # `ix_boundary_unresolved` reuses the keyword for an event field, not a hold.
    unknown = emitted - set(ALL_HOLD_REASONS) - {"missing", "invalid_payload"}
    assert not unknown, f"new hold reason(s) to classify: {sorted(unknown)}"
    # And the classification really covers the gates as well as the refusals.
    assert NO_ASK_HOLD_REASONS <= set(ALL_HOLD_REASONS)


@pytest.mark.parametrize("reason", sorted(NO_ASK_HOLD_REASONS))
@pytest.mark.parametrize("attribute", sorted(HELD_STATUS_BY_ATTRIBUTE))
def test_a_hold_that_asks_nothing_is_never_held_for_confirmation(
    attribute: str, reason: str
) -> None:
    """Round 7 finding: `held_for_confirmation` + `gate: null` + `asked: false`.

    In live run `v7r06` t18 all six `binding_invalid` rows were written as
    `held_for_confirmation` -- that status means "the user has been asked and we
    are waiting for an answer", while that beat opened no gate and asked
    nothing, so three reads simply vanished. The invariant is checked over the
    full `reason × attribute` cross product: as long as the reason itself asks
    nobody, no attribute the call site passes can yield
    `held_for_confirmation`.
    """
    assert held_status(attribute, reason) != "held_for_confirmation"


def test_a_gate_reason_still_gets_its_gates_status() -> None:
    """The reason map must not swallow the four real gates."""
    assert held_status("autonomy_level", "external_action_floor") == "held_for_confirmation"
    assert held_status("capability_boundary", "handoff") == "held_for_handoff"
    assert held_status("autonomy_level", "consent_deny") == "held_for_confirmation"
    # An attribute nobody recognises falls to the contract status, never to the
    # one that claims the user was asked.
    assert held_status("", "") == "held_for_binding"


def test_hold_rounds_hard_cap_is_reachable_under_the_beat_budget() -> None:
    """The round-4 forced close has to fall inside the beat budget."""
    assert HOLD_ROUNDS_HARD_CAP == 4
    for attribute in sorted(HOLD_BEARING_ATTRIBUTES):
        assert turn_budget([attribute]) >= HOLD_ROUNDS_HARD_CAP, attribute


# --- contract 8 · the sideband's input value domain ------------


def test_sideband_input_excludes_the_audit_only_missing_value() -> None:
    assert SIDEBAND_CONSENT_VALUES == ("allow", "deny", "unclear")
    assert set(typing.get_args(ConsentDecisionValue)) - set(SIDEBAND_CONSENT_VALUES) == {
        "missing"
    }
    # `missing` is an audit state the wrapper infers, not a verdict the
    # simulator may write
    assert ConsentRecord(
        attribute="autonomy_level",
        item_id="a3",
        pa_decision="release",
        sideband_decision="missing",
        released=False,
    ).released is False
    assert CONSENT_DECISIONS_BLOCK == "consent_decisions"
