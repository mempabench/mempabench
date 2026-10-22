"""P1 · the pure renderer: one case per rendered attribute per setting.

Also covers the unset fallback every rendered attribute must have. The worked
examples are the three-batch travel turn, the Dale plan with the email gate,
and the process notice / completion pair.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pa.agent.ix.collector import hold_prompt, rebound_batch_ids
from pa.agent.ix.render import render
from pa.agent.ix.render_types import (
    HOLD_ROUNDS_HARD_CAP,
    PASS_THROUGH_ATTRIBUTES,
    RENDERED_ATTRIBUTES,
    Assumption,
    Batch,
    BoundaryState,
    Certainty,
    GateRecord,
    HoldEpisode,
    IXHold,
    IXRenderInput,
    PlanItem,
    ProactiveDecision,
    Question,
    ReleaseDecision,
    ToolCallRecord,
    TurnPayloads,
)

#: The unset fallback: the attribute has no selection this session. Only
#: reachable for an attribute the session never exposed — an exposed one's
#: `IX_<attribute>` tool is `required == "always"`.
UNSET = ""

# --- the plan's worked examples ---------------------------------------------

THREE_BATCHES = (
    "Here is where the three of them stand.\n\n"
    "[[batch:b1]] Flights: the 09:40 is still open.\n\n"
    "[[batch:b2]] Hotel: two rooms left at the Kimpton.\n\n"
    "[[batch:b3]] Invoice: finance has not signed it off."
)

BATCH_PAYLOADS = TurnPayloads(
    batch_set_id="bs1",
    item_count=3,
    batches=[
        Batch("b1", "Flights", "", ["i1"]),
        Batch("b2", "Hotel", "", ["i2"]),
        Batch("b3", "Invoice", "", ["i3"]),
    ],
)

THREE_QUESTIONS = (
    "I can book it once I know three things.\n\n"
    "[[q:q1]] Which airport do you want to fly out of?\n\n"
    "[[q:q2]] Aisle or window?\n\n"
    "[[q:q3]] Should I put it on the team card?"
)

QUESTION_PAYLOADS = TurnPayloads(
    elicitation_id="el1",
    questions=[
        Question("q1", "Which airport?", "departure"),
        Question("q2", "Aisle or window?", "seat"),
        Question("q3", "Which card?", "payment"),
    ],
    assumptions=[Assumption("departure", "SFO, the one you always use")],
)

DALE_PLAN = [
    PlanItem("a1", "Check the stock at Dale's shop", False, ["i1"]),
    PlanItem("a2", "Draft the extension request", False, ["i2"]),
    PlanItem("a3", "Send the email", True, ["i3"]),
]

ASK = "The extension request is written and addressed to dale@…, shall I send it?"


def _input(**kwargs) -> IXRenderInput:
    kwargs.setdefault("segments", ["Done."])
    return IXRenderInput(**kwargs)


#: One executed task call: the closing line is only released over one (owner
#: decision 2026-09-07), so the completion tests below give their turn some work.
_TASK_CALL = ToolCallRecord(
    order=0, tool_name="document_update", tool_call_id="c1",
    action_type="internal_write", executed=True,
)


def _gate_turn(setting_attribute: str = "autonomy_level") -> IXRenderInput:
    """One read executed, the external action stopped before it ran (§3.3 (part 2)·D)."""
    return _input(
        segments=["Checked the shop, the pull list is up to date."],
        calls=[
            ToolCallRecord(
                order=0,
                tool_name="document_read",
                tool_call_id="tc1",
                notice="Let me look at your pull list first.",
                interaction_item_id="i1",
                action_type="read",
                executed=True,
            ),
            ToolCallRecord(
                order=1,
                tool_name="email_message_send",
                tool_call_id="tc2",
                notice="Now I will send the request.",
                interaction_item_id="i3",
                ask_text=ASK,
                action_type="external_action",
                executed=False,
            ),
        ],
        gate=GateRecord(
            attribute=setting_attribute,
            item_id="a3",
            reason="external_action_floor",
            action_type="external_action",
        ),
        gate_remaining=[DALE_PLAN[2]],
        payloads=TurnPayloads(plan_id="p1", plan=DALE_PLAN),
    )


def _open_episode(attribute: str, rounds_open: int, **held) -> IXHold:
    return IXHold(
        open_episodes={
            attribute: HoldEpisode(
                episode_id="e1", attribute=attribute, rounds_open=rounds_open, **held
            )
        }
    )


# --- reasoning_visibility (§3.2) --------------------------------------------


@pytest.mark.parametrize(
    "setting,shown",
    [("Show", True), ("Hide", False), (UNSET, False)],
)
def test_reasoning_visibility_releases_only_under_show(setting: str, shown: bool) -> None:
    rationale = "Only em_0100 mentions 4B: its stack id matches the batch you gave me."
    rendered = render(
        _input(
            segments=["It is the first of the two emails."],
            payloads=TurnPayloads(user_facing_rationale=rationale),
        ),
        {"reasoning_visibility": setting},
    )
    assert (rationale in rendered.message) is shown
    assert "It is the first of the two emails." in rendered.message
    if shown:
        # §3.2: the rationale is rendered as a paragraph *before* the answer.
        assert rendered.message.index(rationale) < rendered.message.index("It is the first")


def test_provider_reasoning_is_never_rendered() -> None:
    """§3.2 — provider reasoning is audit-only, whatever the setting."""
    rendered = render(
        _input(
            segments=["The 09:40 works."],
            provider_reasoning=["<think>maybe the 07:15 instead</think>"],
            payloads=TurnPayloads(user_facing_rationale="The 09:40 has the shortest layover."),
        ),
        {"reasoning_visibility": "Show"},
    )
    assert "07:15" not in rendered.message
    assert "think" not in rendered.message


# --- uncertainty_expression (§3.1 uncertainty row) --------------------------


@pytest.mark.parametrize(
    "setting,expected",
    [
        ("Express", ["(confidence: low)", "(confidence: high)"]),
        ("Hide", []),
        (UNSET, []),
    ],
)
def test_uncertainty_expression_parentheses_per_setting(
    setting: str, expected: list[str]
) -> None:
    rendered = render(
        _input(
            segments=["The invoice is unpaid[[c:c1]] and the flight is confirmed[[c:c2]]."],
            payloads=TurnPayloads(
                certainties=[Certainty("c1", "low"), Certainty("c2", "high")]
            ),
        ),
        {"uncertainty_expression": setting},
    )
    for text in expected:
        assert text in rendered.message
    for text in ("(confidence: low)", "(confidence: high)", "(confidence: medium)"):
        if text not in expected:
            assert text not in rendered.message
    assert "[[c:" not in rendered.message


# --- process_visibility (§3.1 process row, T14) -----------------------------


@pytest.mark.parametrize(
    "setting,notices,completion",
    [
        ("Silent", 0, False),
        ("Bookend", 1, True),
        ("Full narration", 2, True),
        (UNSET, 0, False),
    ],
)
def test_process_visibility_notices_and_completion(
    setting: str, notices: int, completion: bool
) -> None:
    render_input = _input(
        segments=["Monday 10:00 is booked."],
        calls=[
            ToolCallRecord(
                order=0,
                tool_name="calendar_search",
                tool_call_id="tc1",
                notice="Let me look for a free slot on Monday morning.",
                executed=True,
            ),
            ToolCallRecord(
                order=1,
                tool_name="calendar_event_create",
                tool_call_id="tc2",
                notice="Putting the meeting in now.",
                executed=True,
            ),
        ],
        payloads=TurnPayloads(completion_message="Monday morning is in the calendar."),
    )
    rendered = render(render_input, {"process_visibility": setting})

    assert rendered.message.count("Let me look for a free slot") == (1 if notices >= 1 else 0)
    assert rendered.message.count("Putting the meeting in now") == (1 if notices >= 2 else 0)
    assert ("Monday morning is in the calendar." in rendered.message) is completion
    assert "Monday 10:00 is booked." in rendered.message
    if setting == "Bookend":
        # first notice + result + completion, in that order
        assert rendered.message.index("Let me look") < rendered.message.index("Monday 10:00")
        assert rendered.message.index("Monday 10:00") < rendered.message.index(
            "Monday morning is in the calendar."
        )


def test_completion_message_is_never_taken_from_the_prose() -> None:
    """§3.1 process row — an ordinary last paragraph is not a completion line."""
    rendered = render(
        _input(segments=["Monday 10:00 is booked. All done."], payloads=TurnPayloads()),
        {"process_visibility": "Bookend"},
    )
    assert rendered.message == "Monday 10:00 is booked. All done."


@pytest.mark.parametrize("setting", ["Bookend", "Full narration"])
def test_a_completion_the_prose_already_ends_with_is_not_repeated(setting: str) -> None:
    """Round-7 finding: the completion sentence shipped twice.

    Live `v7r01` t13 and t22 under `Full narration` — the model wrote the
    sentence as its own closing prose *and* put the same sentence in
    `completion_message`, and the renderer appended the field without looking at
    what the prose already ended with. Three of the five turns were clean, so it
    is the model's habit, not the setting's.
    """
    line = "All four citation-notes actions are complete."
    rendered = render(
        _input(
            segments=[f"Removing the duplicate entry.\n\n{line}"],
            payloads=TurnPayloads(completion_message=line),
            calls=[_TASK_CALL],
        ),
        {"process_visibility": setting},
    )

    assert rendered.message.count(line) == 1
    assert rendered.message.endswith(line)
    assert any("completion already in prose" in step for step in rendered.trace)
    # Not appended, but the user did read it: the audit plane says released.
    assert rendered.completion_delivered is True
    assert rendered.completion_skipped == ""


@pytest.mark.parametrize("setting", ["Bookend", "Full narration"])
def test_a_completion_the_prose_repeats_in_a_code_span_is_not_repeated(
    setting: str,
) -> None:
    """acc_003 t=13: the same sentence twice, differing only by backticks.

    The prose named the citation key bare and `completion_message` put it in a
    code span, so the exact-string tail check missed it and the renderer
    appended the sentence a second time — `bookend.md` reception item 5 in its
    purest form.
    """
    rendered = render(
        _input(
            segments=[
                "All Lindqvist citations standardized to lindqvist2022trace."
            ],
            payloads=TurnPayloads(
                completion_message=(
                    "All Lindqvist citations standardized to `lindqvist2022trace`."
                ),
            ),
            calls=[_TASK_CALL],
        ),
        {"process_visibility": setting},
    )

    assert rendered.message.count("Lindqvist citations standardized") == 1
    assert any("completion already in prose" in step for step in rendered.trace)
    assert rendered.completion_delivered is True
    assert rendered.completion_skipped == ""


@pytest.mark.parametrize("setting", ["Bookend", "Full narration"])
def test_a_completion_that_is_not_the_prose_tail_is_still_appended(setting: str) -> None:
    """The dedupe is a tail check, not a licence to drop the field."""
    rendered = render(
        _input(
            segments=["Removing the duplicate entry."],
            payloads=TurnPayloads(completion_message="The bibliography is clean."),
            calls=[_TASK_CALL],
        ),
        {"process_visibility": setting},
    )

    assert rendered.message.endswith("The bibliography is clean.")
    assert "Removing the duplicate entry." in rendered.message


@pytest.mark.parametrize("setting", ["Bookend", "Full narration"])
def test_a_gate_turn_skips_the_completion_line_and_records_why(setting: str) -> None:
    """§4.4 — the gate's ask ends the turn, so no closing line is delivered.

    `loop.py` used to write `completion.released` from the setting plus a
    non-empty `completion_message`, independent of which branch the renderer
    took. On a gate turn the audit blob then claimed a closing sentence the
    reply did not carry, which is exactly the reading the blob exists to give.
    """
    line = "Everything on the pull list is handled."
    render_input = _gate_turn()
    rendered = render(
        replace(
            render_input,
            payloads=replace(render_input.payloads, completion_message=line),
        ),
        {"autonomy_level": "Reactive", "process_visibility": setting},
    )

    assert rendered.message.rstrip().endswith(ASK)
    assert line not in rendered.message
    assert rendered.completion_delivered is False
    assert rendered.completion_skipped == "gate"


@pytest.mark.parametrize("in_prose", [False, True])
def test_a_completion_the_wrapper_filter_strips_is_not_delivered(in_prose: bool) -> None:
    """The residual after 8edf352 — the flag was set before the filter ran.

    `completion_delivered` is decided while the parts are being assembled, and
    `strip_wrapper_vocabulary` runs afterwards on the joined message. A closing
    line that named the wrapper was therefore cut from `message` while the audit
    plane went on reporting `completion.released: true` — the one reading the
    field exists to give. Both delivery routes are checked: the appended field
    and the same sentence already standing as the prose tail.
    """
    line = "Both files are saved; the tool definition rejected the first attempt."
    rendered = render(
        _input(
            segments=[f"Saved both files.\n\n{line}"] if in_prose else ["Saved both files."],
            payloads=TurnPayloads(completion_message=line),
            calls=[_TASK_CALL],
        ),
        {"process_visibility": "Bookend"},
    )

    assert "tool definition" not in rendered.message
    assert rendered.message.strip() == "Saved both files."
    assert rendered.completion_delivered is False
    assert rendered.completion_skipped == "stripped"
    # Withheld, not lost: the sentence the PA wrote still reaches the audit plane.
    assert [row.text for row in rendered.withheld_narration] == [line]
    assert any("stripped by the wrapper filter" in step for step in rendered.trace)


def test_a_clean_completion_is_untouched_by_the_post_strip_check() -> None:
    """The counterpart: an ordinary closing line still reports delivered."""
    line = "The bibliography is clean."
    rendered = render(
        _input(
            segments=["Removing the duplicate entry."],
            payloads=TurnPayloads(completion_message=line),
            calls=[_TASK_CALL],
        ),
        {"process_visibility": "Bookend"},
    )

    assert rendered.message.endswith(line)
    assert rendered.completion_delivered is True
    assert rendered.completion_skipped == ""
    assert rendered.withheld_narration == []


# --- autonomy_level (§3.3) --------------------------------------------------


@pytest.mark.parametrize("setting", ["Reactive", "Self-directed"])
def test_autonomy_gate_delivers_the_per_call_ask_verbatim(setting: str) -> None:
    rendered = render(_gate_turn(), {"autonomy_level": setting})

    assert ASK in rendered.message
    assert rendered.message.rstrip().endswith(ASK)
    assert rendered.executed_call_ids == ["tc1"]
    assert rendered.held_call_ids == ["tc2"]
    held = rendered.held
    assert [(item.status, item.tool, item.plan_item_id) for item in held] == [
        ("held_for_confirmation", "email_message_send", "a3")
    ]
    episode = rendered.hold_next.open_episodes["autonomy_level"]
    assert [item.id for item in episode.plan_remaining] == ["a3"]


@pytest.mark.parametrize("setting", ["Autonomous", UNSET])
def test_autonomy_gate_off_settings_render_the_same_shape(setting: str) -> None:
    """§3.3 (part 2)·B — these two open no gate, so the renderer has nothing to hold."""
    render_input = _input(
        segments=["Sent the extension request to Dale."],
        calls=[
            ToolCallRecord(
                order=0,
                tool_name="email_message_send",
                tool_call_id="tc1",
                interaction_item_id="i3",
                ask_text=ASK,
                action_type="external_action",
                executed=True,
            )
        ],
        payloads=TurnPayloads(plan_id="p1", plan=DALE_PLAN),
    )
    rendered = render(render_input, {"autonomy_level": setting})

    assert rendered.message == "Sent the extension request to Dale."
    assert ASK not in rendered.message
    assert rendered.held == []
    assert rendered.hold_next.open_episodes == {}
    assert rendered.held_call_ids == []


def test_gate_without_a_written_ask_is_recorded_not_invented() -> None:
    """§3.3 (part 2)·C — the renderer never writes the ask for the PA."""
    render_input = _gate_turn()
    silent_call = render_input.calls[1]
    render_input = IXRenderInput(
        segments=render_input.segments,
        calls=[render_input.calls[0], ToolCallRecord(
            order=silent_call.order,
            tool_name=silent_call.tool_name,
            tool_call_id=silent_call.tool_call_id,
            interaction_item_id=silent_call.interaction_item_id,
            action_type=silent_call.action_type,
            executed=False,
        )],
        gate=render_input.gate,
        gate_remaining=render_input.gate_remaining,
        payloads=render_input.payloads,
    )
    rendered = render(render_input, {"autonomy_level": "Reactive"})

    assert ASK not in rendered.message
    assert any("ix_gate_no_message" in step for step in rendered.trace)


# --- capability_boundary (§3.5) ---------------------------------------------


def test_find_and_hand_off_ends_the_turn_with_the_handoff_message() -> None:
    handoff = "That paper is not in the retrievable catalogue. Shall I find Imran's address?"
    render_input = _input(
        segments=["I could not pull the full text."],
        calls=[
            ToolCallRecord(
                order=0,
                tool_name="scholarly_paper_read",
                tool_call_id="tc1",
                interaction_item_id="i1",
                status="capability_limit",
                executed=False,
            )
        ],
        gate=GateRecord(attribute="capability_boundary", item_id="r1", reason="handoff"),
        payloads=TurnPayloads(
            boundary_id="bd1",
            boundary=BoundaryState(
                blocked_goal="get the full text",
                boundary_kind="missing_data",
                handoff_message=handoff,
            ),
        ),
    )
    rendered = render(render_input, {"capability_boundary": "Find and hand off"})

    assert rendered.message.rstrip().endswith(handoff)
    assert [item.status for item in rendered.held] == ["held_for_handoff"]
    assert rendered.hold_next.open_episodes["capability_boundary"].boundary is not None


@pytest.mark.parametrize("setting", ["Suggest alternatives", UNSET])
def test_capability_boundary_without_a_gate_holds_nothing(setting: str) -> None:
    """§3.5 (part 3) — no gate means the PA keeps going; nothing is held."""
    rendered = render(
        _input(
            segments=["The full text is not available; I will try the preprint instead."],
            payloads=TurnPayloads(
                boundary_id="bd1",
                boundary=BoundaryState(
                    blocked_goal="get the full text",
                    boundary_kind="missing_data",
                    handoff_message="Shall I ask Imran?",
                ),
            ),
        ),
        {"capability_boundary": setting},
    )
    assert "Shall I ask Imran?" not in rendered.message
    assert rendered.hold_next.open_episodes == {}


# --- topic_management (§3.4) ------------------------------------------------


@pytest.mark.parametrize(
    "setting,released,held",
    [
        ("Organize", ["Flights", "Hotel", "Invoice"], []),
        ("One-at-a-time", ["Flights"], ["b2", "b3"]),
        (UNSET, ["Flights", "Hotel", "Invoice"], []),
    ],
)
def test_topic_management_batch_release(
    setting: str, released: list[str], held: list[str]
) -> None:
    rendered = render(
        _input(segments=[THREE_BATCHES], payloads=BATCH_PAYLOADS),
        {"topic_management": setting},
    )
    for label in ("Flights", "Hotel", "Invoice"):
        assert (label in rendered.message) is (label in released)
    assert "[[batch:" not in rendered.message
    episode = rendered.hold_next.open_episodes.get("topic_management")
    assert [batch.batch_id for batch in (episode.batches_held if episode else [])] == held
    if held:
        # the held bodies are kept verbatim for the IX face, never delivered
        assert "Kimpton" in episode.batches_held[0].body
        assert "Kimpton" not in rendered.message


def test_unmarked_prose_ships_on_a_turn_with_held_batches() -> None:
    """§8 item 2 option A — the framing line goes out, the held bodies do not."""
    rendered = render(
        _input(segments=[THREE_BATCHES], payloads=BATCH_PAYLOADS),
        {"topic_management": "One-at-a-time"},
    )
    assert "Here is where the three of them stand." in rendered.message
    assert "finance has not signed it off" not in rendered.message


# --- information_elicitation (§3.4 (part 2)) --------------------------------------


@pytest.mark.parametrize(
    "setting,released,held",
    [
        ("Infer", [], ["q1", "q2", "q3"]),
        ("Structured", ["airport", "Aisle", "team card"], []),
        ("Iterative", ["airport"], ["q2", "q3"]),
        (UNSET, ["airport", "Aisle", "team card"], []),
    ],
)
def test_information_elicitation_question_release(
    setting: str, released: list[str], held: list[str]
) -> None:
    rendered = render(
        _input(segments=[THREE_QUESTIONS], payloads=QUESTION_PAYLOADS),
        {"information_elicitation": setting},
    )
    for needle in ("airport", "Aisle", "team card"):
        assert (needle in rendered.message) is (needle in released)
    assert "[[q:" not in rendered.message
    assert "I can book it once I know three things." in rendered.message
    episode = rendered.hold_next.open_episodes.get("information_elicitation")
    assert [question.id for question in (episode.questions_held if episode else [])] == held


#: regress13 acc_009 t=10 verbatim: a sentence that stands on its own, then the
#: sentence whose object is the block `Infer` is about to hold.
LEADIN_QUESTIONS = (
    "Happy to hear the draft structure is acceptable! "
    "To finalize it, I need two details from you:\n\n"
    "[[q:q1]] How long should the review window be?\n\n"
    "[[q:q2]] How would you like the comments workflow to work?"
)


def test_infer_drops_the_lead_in_the_held_questions_leave_dangling() -> None:
    """regress13 acc_009 — three turns reached the user as a colon with nothing after it."""
    rendered = render(
        _input(segments=[LEADIN_QUESTIONS], payloads=QUESTION_PAYLOADS),
        {"information_elicitation": "Infer"},
    )
    assert rendered.message == "Happy to hear the draft structure is acceptable!"
    assert rendered.leadin_dropped == "To finalize it, I need two details from you:"
    assert rendered.prose_held is True
    assert "held block -> lead-in sentence dropped" in rendered.trace


def test_a_released_question_keeps_its_lead_in() -> None:
    """`Iterative` releases the first block, so the lead-in still introduces something."""
    rendered = render(
        _input(segments=[LEADIN_QUESTIONS], payloads=QUESTION_PAYLOADS),
        {"information_elicitation": "Iterative"},
    )
    assert "I need two details from you:" in rendered.message
    assert "How long should the review window be?" in rendered.message
    assert rendered.leadin_dropped == ""


def test_framing_that_is_not_a_lead_in_survives_a_full_hold() -> None:
    """The narrowness of the rule, stated as a test.

    `I can book it once I know three things.` introduces the same three held
    questions and ends on a full stop, and it says something the user can act
    on. Only a colon left pointing at nothing, or the object-naming form, goes.
    """
    rendered = render(
        _input(segments=[THREE_QUESTIONS], payloads=QUESTION_PAYLOADS),
        {"information_elicitation": "Infer"},
    )
    assert rendered.message == "I can book it once I know three things."
    assert rendered.leadin_dropped == ""


def test_a_lead_in_without_a_colon_goes_on_its_object() -> None:
    """The second shape: acc_009's own wording, punctuated as a sentence."""
    rendered = render(
        _input(
            segments=[
                "I have the draft. I need two details from you.\n\n"
                "[[q:q1]] How long should the review window be?"
            ],
            payloads=QUESTION_PAYLOADS,
        ),
        {"information_elicitation": "Infer"},
    )
    assert rendered.message == "I have the draft."
    assert rendered.leadin_dropped == "I need two details from you."


#: regress15 acc_009 t7 verbatim shape: the model wrote its questions as a
#: markdown list and put each `[[q:…]]` marker *inside* the list item, so every
#: section carries the next item's `- ` on its tail.
BULLETED_QUESTIONS = (
    "So Monday and Wednesday both look free. "
    "Before I write the draft, I just need a few clarifications from you:\n\n"
    "- [[q:q1]] How long should the review session be?\n\n"
    "- [[q:q2]] Would you like written comments before the call?\n\n"
    "- [[q:q3]] Is there a particular focus you want raised?"
)


def test_a_bulleted_question_list_leaves_no_orphan_bullet_behind() -> None:
    """regress15 acc_009 t7 — the turn shipped a lead-in and a lone `-`.

    The excised item's own bullet stayed on the framing, so the lead-in rule
    read the last "sentence" as `-`, matched neither of its two shapes, and left
    the colon standing. What reached the user was
    *"…I just need a few clarifications from you:"* followed by nothing.
    """
    rendered = render(
        _input(segments=[BULLETED_QUESTIONS], payloads=QUESTION_PAYLOADS),
        {"information_elicitation": "Infer"},
    )
    assert rendered.message == "So Monday and Wednesday both look free."
    assert rendered.leadin_dropped == (
        "Before I write the draft, I just need a few clarifications from you:"
    )
    assert "held block -> orphan list marker dropped" in rendered.trace


def test_a_released_question_does_not_ship_the_held_ones_bullet() -> None:
    """`Iterative` keeps the lead-in, and the bullet of the held pair still goes.

    The last *delivered* section is the one carrying the orphan here, not the
    framing, so the lead-in rule never looks at it.
    """
    rendered = render(
        _input(segments=[BULLETED_QUESTIONS], payloads=QUESTION_PAYLOADS),
        {"information_elicitation": "Iterative"},
    )
    assert "clarifications from you:" in rendered.message
    assert rendered.message.endswith("How long should the review session be?")


def test_structured_closes_the_episode_when_everything_is_released() -> None:
    """§4.1 — the episode closes, so a later task can open a new elicitation_id."""
    # The carried question is one this turn releases: "everything" has to mean
    # the episode's own pending set too, or the episode is not empty and does
    # not close.
    hold = _open_episode(
        "information_elicitation", 1, questions_held=[Question("q1", "old", "departure")]
    )
    rendered = render(
        _input(segments=[THREE_QUESTIONS], payloads=QUESTION_PAYLOADS, hold=hold),
        {"information_elicitation": "Structured"},
    )
    assert "information_elicitation" not in rendered.hold_next.open_episodes
    assert rendered.hold_next.finished_episodes[-1].closed is True


# --- proactive_outreach (§3.1 proactive row, contract 11) -------------------


@pytest.mark.parametrize(
    "setting,action,delivered",
    [
        ("High", "send", True),
        ("Low", "suppress", False),
        (UNSET, "send", True),
        (UNSET, "suppress", False),
    ],
)
def test_proactive_decision_drives_delivery(
    setting: str, action: str, delivered: bool
) -> None:
    message = "The comic shop closes at 6 tonight, and your pull list is still waiting."
    rendered = render(
        _input(
            segments=[""],
            proactive_decision=ProactiveDecision(action=action, setting=setting, message=message),
        ),
        {"proactive_outreach": setting},
    )
    assert rendered.message == (message if delivered else "")
    assert rendered.proactive is not None
    assert rendered.proactive.action == action
    # The candidate is written either way; suppression keeps it for the audit side.
    assert rendered.proactive.message == message


def test_proactive_setting_overrides_a_contradicting_action() -> None:
    rendered = render(
        _input(
            segments=[""],
            proactive_decision=ProactiveDecision(
                action="suppress", setting="High", message="Your order is ready."
            ),
        ),
        {"proactive_outreach": "High"},
    )
    assert rendered.proactive is not None
    assert rendered.proactive.action == "send"
    assert rendered.message == "Your order is ready."


# --- pass-through attributes (§3.1, §3.6) -----------------------------------


@pytest.mark.parametrize("attribute", sorted(PASS_THROUGH_ATTRIBUTES))
def test_pass_through_attributes_are_not_transformed(attribute: str) -> None:
    text = "Two options fit: the 09:40 or the 13:15. I would take the 09:40."
    for setting in ("Low", "High", "Terse", "Formal", UNSET):
        rendered = render(_input(segments=[text]), {attribute: setting})
        assert rendered.message == text
        assert rendered.held == []
        assert rendered.hold_next.open_episodes == {}


# --- the unset fallback, one case per rendered attribute --------------------


def _all_three_marker_kinds() -> IXRenderInput:
    """One turn carrying a batch, a question and a certainty marker at once."""
    return _input(
        segments=[
            "Morning.\n\n[[batch:b1]] Flights are open[[c:c1]].\n\n"
            "[[q:q1]] Which airport?\n\n[[batch:b2]] The hotel has two rooms."
        ],
        calls=[
            ToolCallRecord(
                order=0,
                tool_name="calendar_search",
                tool_call_id="tc1",
                notice="Checking the calendar.",
                interaction_item_id="i1",
                ask_text=ASK,
                executed=True,
            )
        ],
        payloads=TurnPayloads(
            user_facing_rationale="Because the stack ids line up.",
            certainties=[Certainty("c1", "low")],
            completion_message="All three are handled.",
            batch_set_id="bs1",
            batches=[Batch("b1", "Flights", "", []), Batch("b2", "Hotel", "", [])],
            elicitation_id="el1",
            questions=[Question("q1", "Which airport?", "departure")],
        ),
    )


def _assert_unset_fallback(rendered) -> None:
    # every batch and question body goes out, nothing is withheld
    for needle in ("Flights are open", "Which airport?", "two rooms"):
        assert needle in rendered.message
    # markers are still removed deterministically, all three kinds
    for marker in ("[[batch:", "[[q:", "[[c:"):
        assert marker not in rendered.message
    # no internal payload field is released
    for internal in ("Because the stack ids", "All three are handled.", "(confidence: ", ASK):
        assert internal not in rendered.message
    # no gate, so no hold and no held call
    assert rendered.held == []
    assert rendered.hold_next.open_episodes == {}
    assert rendered.held_call_ids == []


@pytest.mark.parametrize("attribute", sorted(RENDERED_ATTRIBUTES))
def test_the_unset_fallback_strips_markers_holds_nothing_releases_nothing(
    attribute: str,
) -> None:
    _assert_unset_fallback(render(_all_three_marker_kinds(), {attribute: UNSET}))


def test_a_turn_with_no_selection_at_all_still_strips_all_three_marker_kinds() -> None:
    """The renderer's floor: with an empty `settings` map every marker still comes
    out and no internal payload field is released.

    This is the state of an attribute the session never exposed. An attribute it
    *did* expose cannot reach it — `IX_<attribute>` is `required == "always"`, so
    a turn that never calls it fails the render contract instead of rendering
    (`tests/test_ix_wrapper/test_render_paths.py`
    `test_an_exposed_attribute_that_is_never_selected_never_renders`).
    """
    _assert_unset_fallback(render(_all_three_marker_kinds(), {}))
    # …and with no `settings` argument supplied at all.
    _assert_unset_fallback(render(_all_three_marker_kinds()))


# --- hold bookkeeping (§4.1, §3.7 (part 4)) ---------------------------------------


def test_an_empty_batch_render_does_not_rename_the_open_episode() -> None:
    """acc_023 b_review_proposals declared `bs_002` with `item_count: 0`.

    A payload naming no batch is a no-op declaration, and it used to be enough
    to rename the open episode to a batch set that holds nothing.
    """
    hold = _open_episode(
        "topic_management", 1, batches_held=[Batch("b2", "Hotel", "old", [])],
    )
    rendered = render(
        _input(
            segments=["Nothing new to divide."],
            payloads=TurnPayloads(batch_set_id="bs2", item_count=0, batches=[]),
            hold=hold,
        ),
        {"topic_management": "One-at-a-time"},
    )

    assert rendered.hold_next.open_episodes["topic_management"].episode_id == "e1"
    assert rendered.hold_next.batch_bindings == {}


def test_a_batch_id_keeps_the_binding_it_was_first_declared_with() -> None:
    """acc_023 t18 / t20 — the session's only memory of what `b1` meant.

    A fully released batch set closes no episode, so `open_episodes` and
    `finished_episodes` between them remember nothing about it. The record is
    what a redeclaration is checked against, so a rebinding must not move it.
    """
    first = render(
        _input(
            segments=["[[batch:b1]] Deadline logged."],
            payloads=TurnPayloads(
                batch_set_id="bs_001",
                item_count=1,
                batches=[Batch("b1", "Calendar deadline", "", ["cal_1"])],
            ),
        ),
        {"topic_management": "Organize"},
    )
    assert first.hold_next.batch_bindings == {("bs_001", "b1"): ("cal_1",)}
    assert first.hold_next.open_episodes == {}

    second = render(
        _input(
            segments=["[[batch:b1]] Sending the confirmation."],
            payloads=TurnPayloads(
                batch_set_id="bs_001",
                item_count=1,
                batches=[Batch("b1", "Send confirmation email", "", ["snd_1"])],
            ),
            hold=first.hold_next,
        ),
        {"topic_management": "Organize"},
    )

    assert second.hold_next.batch_bindings == {("bs_001", "b1"): ("cal_1",)}
    assert rebound_batch_ids(
        TurnPayloads(
            batch_set_id="bs_001",
            item_count=1,
            batches=[Batch("b1", "Send confirmation email", "", ["snd_1"])],
        ),
        first.hold_next,
    ) == ["b1"]


def test_a_new_batch_set_may_re_use_the_item_ids_of_a_closed_one() -> None:
    """acc_023 t18 — `report_2026_02` opened with `b1`, and round 4 refused it.

    A batch id is a handle inside its set, and the `batch_set_id` description
    tells the PA to open a new id once the set has closed. Keying the binding by
    the id alone made that a rebinding: `ix_batch_rebound` twice, two turns with
    `email_draft_create` held and nothing else in them, and the wrapper's own
    stall text read out to the user.
    """
    first = render(
        _input(
            segments=["[[batch:b1]] Deadline logged."],
            payloads=TurnPayloads(
                batch_set_id="report_2026_01",
                item_count=1,
                batches=[Batch("b1", "Calendar reminder", "", ["cal_1"])],
            ),
        ),
        {"topic_management": "Organize"},
    )

    reopened = TurnPayloads(
        batch_set_id="report_2026_02",
        item_count=1,
        batches=[Batch("b1", "Draft reply to Dr. Okafor", "", ["draft_1"])],
    )
    assert rebound_batch_ids(reopened, first.hold_next) == []

    second = render(
        _input(
            segments=["[[batch:b1]] Drafting the reply."],
            payloads=reopened,
            hold=first.hold_next,
        ),
        {"topic_management": "Organize"},
    )

    assert second.hold_next.batch_bindings == {
        ("report_2026_01", "b1"): ("cal_1",),
        ("report_2026_02", "b1"): ("draft_1",),
    }


def test_hold_round_four_cancels_and_delivers_the_closing_line() -> None:
    hold = _open_episode(
        "topic_management", HOLD_ROUNDS_HARD_CAP - 1, batches_held=[Batch("b2", "Hotel", "old", [])]
    )
    payloads = TurnPayloads(
        batch_set_id="bs1",
        item_count=3,
        batches=BATCH_PAYLOADS.batches,
        closing_message="I am dropping the hotel and the invoice for now.",
    )
    rendered = render(
        _input(segments=[THREE_BATCHES], payloads=payloads, hold=hold),
        {"topic_management": "One-at-a-time"},
    )

    assert "I am dropping the hotel and the invoice for now." in rendered.message
    assert "topic_management" not in rendered.hold_next.open_episodes
    cancelled = rendered.hold_next.finished_episodes[-1].cancelled
    assert [entry["id"] for entry in cancelled] == ["b2", "b3"]
    assert "Kimpton" not in rendered.message


def test_hold_round_four_without_a_closing_line_is_recorded() -> None:
    hold = _open_episode("topic_management", HOLD_ROUNDS_HARD_CAP - 1)
    rendered = render(
        _input(segments=[THREE_BATCHES], payloads=BATCH_PAYLOADS, hold=hold),
        {"topic_management": "One-at-a-time"},
    )
    assert any("ix_close_no_message" in step for step in rendered.trace)


def test_hold_rounds_accumulate_across_turns() -> None:
    hold = _open_episode("topic_management", 1, batches_held=[Batch("b2", "Hotel", "old", [])])
    rendered = render(
        _input(segments=[THREE_BATCHES], payloads=BATCH_PAYLOADS, hold=hold),
        {"topic_management": "One-at-a-time"},
    )
    assert rendered.hold_next.open_episodes["topic_management"].rounds_open == 2
    assert rendered.hold_next.open_episodes["topic_management"].episode_id == "bs1"


def _held_call(item_id: str, tool: str = "email_message_send") -> dict:
    """One `calls_held` row of an open episode, in the shape the renderer writes."""
    return {
        "interaction_item_id": item_id,
        "tool": tool,
        "plan_item_id": "",
        "batch_id": "",
        "reason": "external_action_floor",
        "asked": True,
    }


def _carried(rendered) -> list[str]:
    episode = rendered.hold_next.open_episodes.get("autonomy_level")
    return [] if episode is None else [
        call["interaction_item_id"] for call in episode.calls_held
    ]


def test_a_keep_decision_leaves_the_item_held() -> None:
    """§3.7 (part 3) — `keep` is a normal answer, not the end of the episode."""
    hold = _open_episode("autonomy_level", 1, calls_held=[_held_call("i1")])
    rendered = render(
        _input(
            segments=["I will wait for your word on the email."],
            payloads=TurnPayloads(
                release_decisions=[ReleaseDecision(id="i1", decision="keep")]
            ),
            hold=hold,
        ),
        {"autonomy_level": "Reactive"},
    )
    assert _carried(rendered) == ["i1"]
    assert rendered.hold_next.finished_episodes == []


def test_a_turn_that_calls_no_continuation_keeps_the_episode_open() -> None:
    """No `IX_continue_<attribute>` at all: nothing was decided, nothing closes."""
    hold = _open_episode("autonomy_level", 1, calls_held=[_held_call("i1")])
    rendered = render(
        _input(segments=["Still on my list."], hold=hold),
        {"autonomy_level": "Reactive"},
    )
    assert _carried(rendered) == ["i1"]
    assert rendered.hold_next.finished_episodes == []


def test_handling_one_held_item_does_not_drop_the_others() -> None:
    """i1 ran, i2 was skipped, i3 was never mentioned — only i3 stays pending."""
    hold = _open_episode(
        "autonomy_level",
        1,
        calls_held=[_held_call("i1"), _held_call("i2"), _held_call("i3")],
    )
    rendered = render(
        _input(
            segments=["The email is out."],
            calls=[
                ToolCallRecord(
                    order=0,
                    tool_name="email_message_send",
                    tool_call_id="tc1",
                    interaction_item_id="i1",
                    action_type="external_action",
                    executed=True,
                )
            ],
            payloads=TurnPayloads(
                release_decisions=[
                    ReleaseDecision(id="i1", decision="release"),
                    ReleaseDecision(id="i2", decision="skip"),
                ]
            ),
            hold=hold,
        ),
        {"autonomy_level": "Reactive"},
    )
    assert _carried(rendered) == ["i3"]


def test_a_carried_item_reaches_the_hard_cap_and_is_cancelled() -> None:
    """§3.7 (part 4) — the rounds keep counting while the item is still waiting."""
    hold = _open_episode(
        "autonomy_level", HOLD_ROUNDS_HARD_CAP - 1, calls_held=[_held_call("i1")]
    )
    rendered = render(
        _input(segments=["Still waiting on you."], hold=hold),
        {"autonomy_level": "Reactive"},
    )
    assert "autonomy_level" not in rendered.hold_next.open_episodes
    assert rendered.hold_next.finished_episodes[-1].closed is True
    assert [episode.attribute for episode in rendered.cancelled_now] == ["autonomy_level"]


def test_a_cancelled_call_hold_says_which_call_it_dropped() -> None:
    """regress15 acc_039 t27 — `ix_hold_forced_close` reported `cancelled: 0`.

    The episode was holding one thing, the held call `apt_book_001`, and
    `_cancelled_entries` only ever looked at batches, questions and plan items.
    So the cap closed the episode, wiped `calls_held`, and the booking the user
    was still waiting on left no record at all: never executed, never cancelled,
    never repaired.
    """
    hold = _open_episode(
        "autonomy_level",
        HOLD_ROUNDS_HARD_CAP - 1,
        calls_held=[_held_call("apt_book_001", tool="appointment_create")],
    )
    rendered = render(
        _input(segments=["Still holding the booking."], hold=hold),
        {"autonomy_level": "Reactive"},
    )
    cancelled = rendered.cancelled_now[0].cancelled
    assert [(row["id"], row["kind"]) for row in cancelled] == [
        ("apt_book_001", "call")
    ]


def _carried_questions(rendered) -> list[str]:
    episode = rendered.hold_next.open_episodes.get("information_elicitation")
    return [] if episode is None else [question.id for question in episode.questions_held]


def _carried_batches(rendered) -> list[str]:
    episode = rendered.hold_next.open_episodes.get("topic_management")
    return [] if episode is None else [batch.batch_id for batch in episode.batches_held]


_HELD_QUESTIONS = [
    Question("q2", "Aisle or window?", "seat"),
    Question("q3", "Which card should I put it on?", "payment"),
]


def test_carried_questions_survive_a_turn_with_no_continuation() -> None:
    """Live `v11s02` acc_009: `q2` / `q3` held on t3 were simply gone on t8.

    The episode was closed with `cancelled: null` and no close line, because
    the questions were read off this turn's holds alone.
    """
    hold = _open_episode("information_elicitation", 1, questions_held=_HELD_QUESTIONS)
    rendered = render(
        _input(segments=["Here is what I found so far."], hold=hold),
        {"information_elicitation": "Iterative"},
    )
    assert _carried_questions(rendered) == ["q2", "q3"]
    assert rendered.hold_next.open_episodes["information_elicitation"].rounds_open == 2
    assert rendered.hold_next.finished_episodes == []
    reminder = hold_prompt(rendered.hold_next)
    assert "q2 (seat)" in reminder and "q3 (payment)" in reminder


def test_carried_batches_survive_a_turn_with_no_continuation() -> None:
    """The same rule on the `topic_management` side of the episode."""
    hold = _open_episode(
        "topic_management",
        1,
        batches_held=[Batch("b2", "Hotel", "old", ["i2"]), Batch("b3", "Invoice", "old", ["i3"])],
    )
    rendered = render(
        _input(segments=["Still working through those two."], hold=hold),
        {"topic_management": "One-at-a-time"},
    )
    assert _carried_batches(rendered) == ["b2", "b3"]
    assert rendered.hold_next.open_episodes["topic_management"].rounds_open == 2
    reminder = hold_prompt(rendered.hold_next)
    assert "b2 (Hotel)" in reminder and "b3 (Invoice)" in reminder


def test_a_continuation_that_releases_one_question_leaves_the_other_held() -> None:
    """`q2` went out under its own marker; `keep` settles nothing."""
    hold = _open_episode("information_elicitation", 1, questions_held=_HELD_QUESTIONS)
    rendered = render(
        _input(
            segments=["One more thing.\n\n[[q:q2]] Aisle or window?"],
            payloads=TurnPayloads(
                elicitation_id="el1",
                questions=[Question("q2", "Aisle or window?", "seat")],
                release_decisions=[
                    ReleaseDecision(id="q2", decision="release"),
                    ReleaseDecision(id="q3", decision="keep"),
                ],
            ),
            hold=hold,
        ),
        {"information_elicitation": "Iterative"},
    )
    assert _carried_questions(rendered) == ["q3"]
    assert "Aisle or window?" in rendered.message


def test_carried_questions_reach_the_cap_and_are_cancelled() -> None:
    """§3.7 (part 4), owner decision 2026-09-06 — the cap writes no close line.

    The reply is the PA's prose and nothing else. The cancellation is a fact of
    the audit plane: the ids are on the finished episode, `close_line_source` is
    `none`, and the trace still records the PA's missing `closing_message`.
    """
    hold = _open_episode(
        "information_elicitation", HOLD_ROUNDS_HARD_CAP - 1, questions_held=_HELD_QUESTIONS
    )
    rendered = render(
        _input(segments=["Nothing new from me."], hold=hold),
        {"information_elicitation": "Iterative"},
    )
    assert "information_elicitation" not in rendered.hold_next.open_episodes
    cancelled = rendered.hold_next.finished_episodes[-1].cancelled
    assert [entry["id"] for entry in cancelled] == ["q2", "q3"]
    assert rendered.message == "Nothing new from me."
    assert rendered.close_line_source == "none"
    assert any("ix_close_no_message" in step for step in rendered.trace)


def test_a_cap_closed_turn_delivers_the_pa_prose_and_nothing_else() -> None:
    """Owner decision 2026-09-06 (§10 item d): no template text on the force-close.

    Live `v11s02` t11 answered the held question in prose and the renderer then
    appended "I have not done meeting_duration; it is on hold." underneath it:
    the reply contradicted itself, and a snake_case intent id reached the user.
    Neither the label, the id, nor the held wording may appear now.
    """
    hold = _open_episode(
        "information_elicitation", HOLD_ROUNDS_HARD_CAP - 1, questions_held=_HELD_QUESTIONS
    )
    rendered = render(
        _input(segments=["[[q:q1]] Which airport?\n\nThat is all for now."], hold=hold),
        {"information_elicitation": "Iterative"},
    )

    assert rendered.message == "Which airport?\n\nThat is all for now."
    for leaked in (
        "on hold", "I have not done", "did not get to",
        "seat", "payment", "q2", "q3",
        "Aisle or window?", "Which card should I put it on?",
    ):
        assert leaked not in rendered.message, leaked
    # The audit plane keeps every cancelled id.
    assert [
        entry["id"] for entry in rendered.hold_next.finished_episodes[-1].cancelled
    ] == ["q2", "q3"]
    assert [episode.attribute for episode in rendered.cancelled_now] == [
        "information_elicitation"
    ]
    assert rendered.close_line_source == "none"


def test_a_cap_closed_turn_still_delivers_the_pa_s_own_closing_message() -> None:
    """The PA-authored line is PA text, so the decision does not touch it."""
    hold = _open_episode(
        "information_elicitation", HOLD_ROUNDS_HARD_CAP - 1, questions_held=_HELD_QUESTIONS
    )
    rendered = render(
        _input(
            segments=["Nothing new from me."],
            payloads=TurnPayloads(
                closing_message="I am letting the seat and the card go for now.",
            ),
            hold=hold,
        ),
        {"information_elicitation": "Iterative"},
    )

    assert "I am letting the seat and the card go for now." in rendered.message
    assert rendered.close_line_source == "payload"


def test_a_fresh_hold_joins_the_carried_ones_without_duplicating_them() -> None:
    """`q2` is held again on top of being carried; `q3` is only carried.

    Sorted, so a `q2` counted twice fails the same assertion as a `q3` dropped.
    """
    hold = _open_episode("information_elicitation", 1, questions_held=_HELD_QUESTIONS)
    rendered = render(
        _input(
            segments=[
                "[[q:q1]] Which airport?\n\n"
                "[[q:q2]] Aisle or window?\n\n"
                "[[q:q4]] Do you want the lounge pass?"
            ],
            payloads=TurnPayloads(
                elicitation_id="el1",
                questions=[
                    Question("q1", "Which airport?", "departure"),
                    Question("q2", "Aisle or window?", "seat"),
                    Question("q4", "Lounge pass?", "lounge"),
                ],
            ),
            hold=hold,
        ),
        {"information_elicitation": "Iterative"},
    )
    assert sorted(_carried_questions(rendered)) == ["q2", "q3", "q4"]


def test_a_bare_release_folds_the_carried_questions_into_the_reply() -> None:
    """§4.3, owner decision 2026-09-06 — `release` alone settles a question.

    The hold reminder tells the PA that "a reply carrying none of them is
    released whole", so a released question with no `[[q:…]]` marker was asked
    in the reply's own prose. It settles, nothing is carried, and with the
    episode empty it closes. `ix_release_folded` is the audit row that lets the
    batch review find every beat this happened on.
    """
    hold = _open_episode("information_elicitation", 1, questions_held=_HELD_QUESTIONS)
    rendered = render(
        _input(
            segments=["Aisle or window, and which card should I put it on?"],
            payloads=TurnPayloads(
                elicitation_id="el1",
                release_decisions=[
                    ReleaseDecision(id="q2", decision="release"),
                    ReleaseDecision(id="q3", decision="release"),
                ],
            ),
            hold=hold,
        ),
        {"information_elicitation": "Iterative"},
    )
    assert _carried_questions(rendered) == []
    assert "information_elicitation" not in rendered.hold_next.open_episodes
    assert rendered.hold_next.finished_episodes[-1].closed is True
    assert rendered.release_folded == [
        {
            "attribute": "information_elicitation",
            "episode_id": "e1",
            "question_ids": ["q2", "q3"],
        }
    ]


def test_a_bare_release_on_one_question_leaves_the_kept_one_held() -> None:
    """`q2` folded into the prose, `q3` kept: only `q3` survives the turn."""
    hold = _open_episode("information_elicitation", 1, questions_held=_HELD_QUESTIONS)
    rendered = render(
        _input(
            segments=["Do you want the aisle or the window?"],
            payloads=TurnPayloads(
                elicitation_id="el1",
                release_decisions=[
                    ReleaseDecision(id="q2", decision="release"),
                    ReleaseDecision(id="q3", decision="keep"),
                ],
            ),
            hold=hold,
        ),
        {"information_elicitation": "Iterative"},
    )
    assert _carried_questions(rendered) == ["q3"]
    assert [row["question_ids"] for row in rendered.release_folded] == [["q2"]]


def test_a_released_question_that_carries_its_marker_is_not_folded() -> None:
    """The marker went out, so nothing was folded and no row is written."""
    hold = _open_episode("information_elicitation", 1, questions_held=_HELD_QUESTIONS)
    rendered = render(
        _input(
            segments=["One more thing.\n\n[[q:q2]] Aisle or window?"],
            payloads=TurnPayloads(
                elicitation_id="el1",
                questions=[Question("q2", "Aisle or window?", "seat")],
                release_decisions=[
                    ReleaseDecision(id="q2", decision="release"),
                    ReleaseDecision(id="q3", decision="keep"),
                ],
            ),
            hold=hold,
        ),
        {"information_elicitation": "Iterative"},
    )
    assert _carried_questions(rendered) == ["q3"]
    assert rendered.release_folded == []


def test_a_batch_released_without_its_marker_stays_pending() -> None:
    """Only questions fold. A batch settles on delivery, or on a bound call.

    `release` is the PA's intent (§3.7 (part 3)); a batch that was neither
    re-marked nor executed is still the user's to receive.
    """
    hold = _open_episode(
        "topic_management", 1, batches_held=[Batch("b2", "Hotel", "old", ["i2"])]
    )
    rendered = render(
        _input(
            segments=["Still working through the hotel."],
            payloads=TurnPayloads(
                batch_set_id="bs1",
                release_decisions=[ReleaseDecision(id="b2", decision="release")],
            ),
            hold=hold,
        ),
        {"topic_management": "One-at-a-time"},
    )
    assert _carried_batches(rendered) == ["b2"]
    assert rendered.release_folded == []


def test_a_bare_release_at_the_round_cap_folds_instead_of_cancelling() -> None:
    """The fold is applied before the cap: round 4 must not cancel a settled id.

    `rounds_open == 3` means this turn is the one that reaches
    `HOLD_ROUNDS_HARD_CAP`. The PA answers both carried questions in its own
    prose and reports `release` on each, so by §4.3 they are settled — there is
    nothing left for the cap to cancel, and the user must not be told "it is on
    hold" about a question they were just asked.
    """
    hold = _open_episode(
        "information_elicitation", HOLD_ROUNDS_HARD_CAP - 1, questions_held=_HELD_QUESTIONS
    )
    rendered = render(
        _input(
            segments=["Aisle or window, and which card should I put it on?"],
            payloads=TurnPayloads(
                elicitation_id="el1",
                release_decisions=[
                    ReleaseDecision(id="q2", decision="release"),
                    ReleaseDecision(id="q3", decision="release"),
                ],
            ),
            hold=hold,
        ),
        {"information_elicitation": "Iterative"},
    )
    assert [row["question_ids"] for row in rendered.release_folded] == [["q2", "q3"]]
    assert rendered.cancelled_now == []
    assert rendered.hold_next.finished_episodes[-1].cancelled == []
    assert "on hold" not in rendered.message
    assert rendered.message == "Aisle or window, and which card should I put it on?"


# --- the §4.3 hold reminder: one sentence per marker kind --------------------


#: The two rules the reminder states, each in its own sentence (§4.3). A carried
#: question with a bare `release` folds into the reply; a carried batch has no
#: such fold and is still owed when the episode closes.
_RELEASED_WHOLE = "a reply carrying none of them is released whole"
_BATCH_STILL_HELD = "one you leave unmarked is still held when the episode closes"


def test_a_batch_only_reminder_does_not_promise_the_question_fold() -> None:
    """`released whole` is the question rule; saying it over a batch is a lie."""
    hold = _open_episode(
        "topic_management",
        1,
        batches_held=[Batch("b2", "Hotel", "old", ["i2"]), Batch("b3", "Invoice", "old", ["i3"])],
    )
    reminder = hold_prompt(hold)

    assert _RELEASED_WHOLE not in reminder
    assert _BATCH_STILL_HELD in reminder
    assert "[[batch:id]]" in reminder and "[[q:id]]" not in reminder
    # The sentence carries its own ids, so the rule and its subjects stay together.
    assert f"{_BATCH_STILL_HELD}: b2, b3." in reminder


def test_a_question_only_reminder_keeps_the_released_whole_sentence() -> None:
    """The wording the fold rule depends on is unchanged for questions."""
    hold = _open_episode("information_elicitation", 1, questions_held=_HELD_QUESTIONS)
    reminder = hold_prompt(hold)

    assert _BATCH_STILL_HELD not in reminder
    assert "[[batch:id]]" not in reminder and "[[q:id]]" in reminder
    assert f"{_RELEASED_WHOLE}: q2, q3." in reminder


def test_a_turn_holding_both_kinds_gets_both_sentences_each_with_its_own_ids() -> None:
    """Two open episodes, one per marker kind — neither rule may swallow the other."""
    hold = IXHold(open_episodes={
        "topic_management": HoldEpisode(
            episode_id="bs1",
            attribute="topic_management",
            rounds_open=1,
            batches_held=[Batch("b2", "Hotel", "old", ["i2"])],
        ),
        "information_elicitation": HoldEpisode(
            episode_id="el1",
            attribute="information_elicitation",
            rounds_open=1,
            questions_held=_HELD_QUESTIONS,
        ),
    })
    reminder = hold_prompt(hold)

    assert f"{_BATCH_STILL_HELD}: b2." in reminder
    assert f"{_RELEASED_WHOLE}: q2, q3." in reminder
    batch_line, question_line = reminder.splitlines()
    assert _RELEASED_WHOLE not in batch_line
    assert _BATCH_STILL_HELD not in question_line


def test_trivial_turn_passes_the_final_segment_through() -> None:
    """§2.3 — no IX selection at all is the plain pass-through, not a bypass."""
    hold = IXHold()
    rendered = render(
        IXRenderInput(segments=["thinking out loud", "The 09:40 works."], hold=hold), {}
    )
    assert rendered.message == "The 09:40 works."
    assert rendered.hold_next == hold
    assert rendered.trace  # the audit trail exists even on a trivial turn


# --- prose hygiene on a turn that stopped early (round-2 findings 2/4/5) -----


def test_an_unrecognised_marker_never_reaches_the_user() -> None:
    """§3.1.1 — a bracket shape the wrapper cannot read is still internal."""
    rendered = render(
        _input(segments=["The 09:40 works. [[topic:1]] I will hold the rest."]), {},
    )

    assert "[[" not in rendered.message
    assert "topic" not in rendered.message
    assert any(
        step.startswith("unrecognised marker stripped") for step in rendered.trace
    )


def test_a_stripped_marker_leaves_no_orphaned_emphasis() -> None:
    """Round-3 finding 3: the delivered ask was three stray `**` lines.

    `v3g05` t3. The model wrote `**[[q1]]** What is …`; the section split takes
    the marker *and* the prose after it, so the opening `**` stayed on the
    framing part and the closing `**` opened the question's. Joined back
    together they became emphasis around nothing.
    """
    rendered = render(
        _input(
            segments=[
                "To get started, I need a few details:\n\n"
                "**[[q1]]** What is Prof. Reyes's email address?\n\n"
                "**[[q2]]** Which days that week are free?"
            ],
            payloads=QUESTION_PAYLOADS,
        ),
        {"information_elicitation": "Iterative"},
    )

    assert rendered.message == (
        "To get started, I need a few details:\n\n"
        "What is Prof. Reyes's email address?"
    )
    # The held question left no delimiter of its own behind either.
    assert "**" not in rendered.message


def test_a_marker_inside_the_bold_leaves_no_half_delimiter() -> None:
    """Round-7 finding: round 3's repair does not cover this shape.

    Round 3's case was `**[[q1]]** text` — bold *around the marker*, so both
    halves ended up adjacent and the empty-pair pass closed them. `v7r07` t3
    wrote `**[[q:q1]] Who is Prof. Reyes?**` — bold *spanning the question* —
    so the opening `**` stays on the framing part (removed as an emphasis-only
    line) and the closing `**` is stranded mid-line, where neither of the two
    passes can see it. Four lines shipped that way. Parity is per line: the turn
    held four `**` in total, an even number, so a whole-text check finds nothing.
    """
    rendered = render(
        _input(
            segments=[
                "I need a few details first.\n\n"
                "**[[q:q1]] Who is Prof. Reyes?** Could you share his email?\n\n"
                "**[[q:q2]] What is your name?** I need it to look up your calendar."
            ],
            payloads=QUESTION_PAYLOADS,
        ),
        {"information_elicitation": "Structured"},
    )

    assert "**" not in rendered.message
    assert "Who is Prof. Reyes? Could you share his email?" in rendered.message
    assert "What is your name? I need it to look up your calendar." in rendered.message


def test_real_bold_survives_the_orphan_cleanup() -> None:
    """The cleanup only closes pairs the marker emptied (`v3g06` t3's shape)."""
    rendered = render(
        _input(
            segments=[
                "[[q:q1]] **Prof. Reyes' message** — do you have his email?\n\n"
                "**Your availability** — **any day** that week?"
            ],
            payloads=QUESTION_PAYLOADS,
        ),
        {"information_elicitation": "Structured"},
    )

    assert "**Prof. Reyes' message**" in rendered.message
    assert "**Your availability** — **any day** that week?" in rendered.message


def test_a_gate_turn_drops_the_half_sentence_it_stopped_on() -> None:
    """Round-2 finding 4: `Appointment confirmed! Now let me add it to your`.

    The completed sentence in front of it is true and is kept; the ask the gate
    delivers is the message, and it is untouched.
    """
    render_input = replace(
        _gate_turn(),
        segments=["Appointment confirmed! Now let me add it to your"],
    )
    rendered = render(render_input, {"autonomy_level": "Reactive"})

    assert "Appointment confirmed!" in rendered.message
    assert "Now let me add it" not in rendered.message
    assert rendered.message.rstrip().endswith(ASK)
    assert "trailing fragment dropped" in rendered.trace


def test_a_finished_sentence_without_a_full_stop_is_left_alone() -> None:
    """Only a truncation is cut back, never the model's own last sentence."""
    render_input = replace(
        _gate_turn(), segments=["Let me gather the material from your research folder"],
    )
    rendered = render(render_input, {"autonomy_level": "Reactive"})

    assert "Let me gather the material from your research folder" in rendered.message
    assert "trailing fragment dropped" not in rendered.trace


def test_an_answer_to_a_repair_message_is_not_delivered() -> None:
    """Round-2 finding 5: the repair works, but the PA answered it out loud."""
    render_input = replace(
        _gate_turn(),
        segments=[
            "You're right — since this is an external booking action, I need to "
            "confirm with you first. Let me re-send that.\n\n"
            "I have the 10:00 slot confirmed available."
        ],
    )
    rendered = render(render_input, {"autonomy_level": "Reactive"})

    assert "You're right" not in rendered.message
    assert "I have the 10:00 slot confirmed available." in rendered.message
    assert "meta acknowledgement dropped" in rendered.trace


def test_the_acknowledgement_cut_never_lands_inside_a_file_name() -> None:
    """acc_011 b3: the turn reached the user opening `md`. Its contents are…`.

    The dot the scan stopped at was the one in `clinic_visit_notes.md`, inside a
    code span. The cut belongs on the sentence end after it.
    """
    from pa.agent.ix.render import strip_meta_acknowledgement

    text, dropped = strip_meta_acknowledgement(
        "You're right — I located the file at `my_desktop/clinic_visit_notes.md`. "
        "Its contents are as follows:\n\n```\n# clinic visit — notes to self\n```"
    )

    assert dropped is True
    assert text.startswith("Its contents are as follows:")
    assert "md`." not in text


def test_an_acknowledgement_with_no_sentence_end_drops_the_whole_line() -> None:
    """No clean sentence end on the first line, so the cut falls on the line end."""
    from pa.agent.ix.render import strip_meta_acknowledgement

    text, dropped = strip_meta_acknowledgement(
        "You're right, re-sending `notes.md` now\nThe 10:00 slot is free."
    )

    assert dropped is True
    assert text == "The 10:00 slot is free."


def test_a_hard_wrapped_acknowledgement_leaves_no_orphan_fragment() -> None:
    """acc_003 t=22: the beat's summary paragraph read `years as shown above.`

    The acknowledgement was hard-wrapped, so the line-end cut took its first
    line and shipped the continuation as a paragraph of its own — a lowercase
    clause with its subject removed. The cut belongs on the paragraph boundary.
    """
    from pa.agent.ix.render import strip_meta_acknowledgement

    text, dropped = strip_meta_acknowledgement(
        "You're right — I've updated both entries to show the corrected\n"
        "years as shown above.\n\n"
        "The two corrected records are confirmed in the file."
    )

    assert dropped is True
    assert text == "The two corrected records are confirmed in the file."


def test_a_whole_sentence_after_the_wrapped_line_is_kept() -> None:
    """Only a fragment goes with the cut; real prose in that paragraph stays."""
    from pa.agent.ix.render import strip_meta_acknowledgement

    text, dropped = strip_meta_acknowledgement(
        "You're right — I've updated both entries to show the corrected\n"
        "years as shown above. Both records now carry the 2021 date."
    )

    assert dropped is True
    assert text.endswith("Both records now carry the 2021 date.")


# --- §3.7 (part 3) · consent is scoped to what the ask named -----------------------


def test_only_the_ids_the_ask_named_are_marked_asked() -> None:
    """The gate's own item, plus any other held id the ask names verbatim."""
    render_input = replace(
        _gate_turn(),
        calls=[
            replace(_gate_turn().calls[1], interaction_item_id="i3", ask_text=ASK),
            ToolCallRecord(
                order=2,
                tool_name="calendar_event_search",
                tool_call_id="tc3",
                interaction_item_id="i9",
                action_type="read",
                executed=False,
            ),
        ],
        gate=GateRecord(
            attribute="autonomy_level",
            item_id="i3",
            reason="external_action_floor",
            action_type="external_action",
        ),
    )
    rendered = render(render_input, {"autonomy_level": "Reactive"})

    asked = {item.interaction_item_id: item.asked for item in rendered.held}
    assert asked == {"i3": True, "i9": False}
    episode = rendered.hold_next.open_episodes["autonomy_level"]
    assert {
        call["interaction_item_id"]: call["asked"] for call in episode.calls_held
    } == {"i3": True, "i9": False}
