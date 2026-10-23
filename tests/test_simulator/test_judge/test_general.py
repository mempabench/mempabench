"""The general form: clause split, what it never asks, cue absorption, chunks."""

from __future__ import annotations

from simulator.judge import catalogue as cat
from simulator.judge.general import (
    GENERAL_AVOID_CHUNK,
    build_general_groups,
    split_clauses,
)

AVOID = (
    "Do not grade the assistant with evaluation terminology; do not assign a "
    "numerical score. Do not tell it what to do next."
)


def _ids(ctx: cat.TurnContext) -> list[list[str]]:
    """The question ids of each call, in the order they go out."""
    return [group.question_ids for group in build_general_groups(ctx)]


def _flat(ctx: cat.TurnContext) -> list[str]:
    return [qid for group in _ids(ctx) for qid in group]


def _avoid(n: int) -> str:
    return " ".join(f"Do not do thing {index}." for index in range(1, n + 1))


def test_avoid_splits_on_sentence_terminators_keeping_each_clause_verbatim():
    clauses = split_clauses(AVOID)
    assert clauses == [
        "Do not grade the assistant with evaluation terminology; do not assign "
        "a numerical score.",
        "Do not tell it what to do next.",
    ]
    for clause in clauses:
        assert clause in AVOID


def test_each_avoid_clause_is_its_own_traceable_question():
    ctx = cat.TurnContext(candidate_turn="That is fine.", beat_avoid=AVOID)
    groups = build_general_groups(ctx)
    assert [group.question_ids for group in groups] == [["avoid_1", "avoid_2"]]
    assert groups[0].num_predict_kind == "general"


def test_the_instruction_is_asked_only_on_the_turns_it_binds():
    ctx = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_user_instruction="Ask the assistant to read the file.",
        instruction_binds_this_turn=True,
    )
    assert _ids(ctx) == [["user_instruction"]]

    ctx.instruction_binds_this_turn = False
    assert build_general_groups(ctx) == []


def test_must_ask_and_beat_position_are_inputs_never_questions():
    ctx = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_avoid="Do not grade the assistant.",
        must_ask=True,
        beat_position="beat_03",
    )
    (group,) = build_general_groups(ctx)
    assert group.question_ids == ["avoid_1"]
    assert group.blocks["turn_owes_an_ask"] == "yes"
    assert group.blocks["beat_position"] == "beat_03"
    _, prompt = cat.render(group)
    assert "<turn_owes_an_ask>\n[1] yes" in prompt


def test_the_cue_is_absorbed_on_the_cue_beats_opening_turn():
    ctx = cat.TurnContext(
        candidate_turn="I need this settled before Monday.",
        beat_avoid="Do not grade the assistant.",
        cue="the printed copy has to be settled by Monday",
        cue_owed=True,
    )
    assert _ids(ctx) == [["gate_cue_missing"], ["avoid_1"]]

    ctx.cue_owed = False
    assert _ids(ctx) == [["avoid_1"]]


def test_every_general_question_points_at_the_candidate_turn():
    ctx = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_avoid=AVOID,
        beat_user_instruction="Ask it to read the file.",
        cue="a cue",
        cue_owed=True,
    )
    for group in build_general_groups(ctx):
        assert set(group.question_blocks().values()) == {("candidate_turn",)}


# ------------------------------------------------------------------- chunking


def test_a_turn_with_nothing_of_its_own_to_judge_makes_no_call():
    assert build_general_groups(cat.TurnContext(candidate_turn="That is fine.")) == []


def test_the_head_call_carries_the_instruction_and_the_cue_and_nothing_else():
    ctx = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_avoid=_avoid(5),
        beat_user_instruction="Ask it to read the file.",
        instruction_binds_this_turn=True,
        cue="a cue",
        cue_owed=True,
    )
    assert _ids(ctx) == [
        ["user_instruction", "gate_cue_missing"],
        ["avoid_1", "avoid_2", "avoid_3", "avoid_4"],
        ["avoid_5"],
    ]


def test_avoid_is_chunked_and_the_clause_index_stays_global():
    ctx = cat.TurnContext(candidate_turn="That is fine.", beat_avoid=_avoid(9))
    groups = build_general_groups(ctx)
    assert [group.question_ids for group in groups] == [
        ["avoid_1", "avoid_2", "avoid_3", "avoid_4"],
        ["avoid_5", "avoid_6", "avoid_7", "avoid_8"],
        ["avoid_9"],
    ]
    assert all(len(group.questions) <= GENERAL_AVOID_CHUNK for group in groups)
    # Each chunk carries only the clauses it asks about, under its global index.
    assert sorted(key for key in groups[1].blocks if key.startswith("avoid::")) == [
        "avoid::5",
        "avoid::6",
        "avoid::7",
        "avoid::8",
    ]
    assert groups[2].blocks["avoid::9"] == "Do not do thing 9."


def test_four_avoid_clauses_are_still_one_call():
    ctx = cat.TurnContext(candidate_turn="That is fine.", beat_avoid=_avoid(4))
    assert _ids(ctx) == [["avoid_1", "avoid_2", "avoid_3", "avoid_4"]]


def test_every_question_is_asked_exactly_once_across_the_chunks():
    ctx = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_avoid=_avoid(9),
        beat_user_instruction="Ask it to read the file.",
        instruction_binds_this_turn=True,
    )
    flat = _flat(ctx)
    assert flat == ["user_instruction"] + [f"avoid_{i}" for i in range(1, 10)]
    assert len(set(flat)) == len(flat)


# ------------------------------------------------------- scope clauses


SCOPE_AVOID = (
    "This turn judges only whether claims carry bracketed ratings. Do not "
    "comment on the length of the response."
)

_SCOPE_SENTENCE = (
    "Some clauses only say what this turn is about, judges, or evaluates: a "
    "clause like that forbids nothing, so answer obeyed with empty evidence."
)


def test_the_avoid_lead_no_longer_carries_the_scope_clause_rule():
    ctx = cat.TurnContext(candidate_turn="That is fine.", beat_avoid=SCOPE_AVOID)
    (group,) = build_general_groups(ctx)
    # The scope sentence is authoring metadata, so it is never put to the
    # judge; the clause after it keeps its global index.
    assert group.question_ids == ["avoid_2"]
    for question in group.questions:
        assert _SCOPE_SENTENCE not in question.text


def test_the_scope_sentence_is_not_put_to_the_judge():
    ctx = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_avoid=(
            "This turn judges only whether claims carry bracketed ratings. "
            "Do not comment on the length of the response. "
            "Do not tell it what to do next."
        ),
    )
    (group,) = build_general_groups(ctx)
    assert group.question_ids == ["avoid_2", "avoid_3"]
    assert sorted(key for key in group.blocks if key.startswith("avoid::")) == [
        "avoid::2",
        "avoid::3",
    ]
    for question in group.questions:
        assert "This turn judges only" not in question.text
    _, prompt = cat.render(group)
    assert "This turn judges only" not in prompt

    # An `avoid` without a scope sentence is untouched.
    plain = cat.TurnContext(candidate_turn="That is fine.", beat_avoid=_avoid(3))
    (group,) = build_general_groups(plain)
    assert group.question_ids == ["avoid_1", "avoid_2", "avoid_3"]

    # A beat whose `avoid` is only the scope sentence asks nothing.
    only = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_avoid="This turn judges only whether claims carry bracketed ratings.",
    )
    assert build_general_groups(only) == []


def test_the_scope_instruction_is_out_of_the_rendered_prompt():
    ctx = cat.TurnContext(candidate_turn="That is fine.", beat_avoid=SCOPE_AVOID)
    (group,) = build_general_groups(ctx)
    _, prompt = cat.render(group)
    assert "a clause like that forbids nothing" not in prompt


# ----------------------------------------------------- the beat's goal


_GOAL_SENTENCE = (
    "<beat_goal> names the thing the line is there to evaluate, and saying "
    "how that thing landed — its reasons and its wording included — is "
    "obeyed. The clause:"
)


def test_every_avoid_chunk_carries_the_beats_goal():
    ctx = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_avoid=_avoid(5),
        beat_intent="React to the assistant tagging each claim with a confidence level.",
    )
    groups = build_general_groups(ctx)
    assert len(groups) == 2
    for group in groups:
        assert group.blocks["beat_goal"] == (
            "React to the assistant tagging each claim with a confidence level."
        )
        _, prompt = cat.render(group)
        assert "<beat_goal>" in prompt
        assert "tagging each claim with a confidence level" in prompt


def test_the_goal_falls_back_to_the_instruction_and_is_omitted_when_neither():
    ctx = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_avoid=SCOPE_AVOID,
        beat_user_instruction="Say how the unprompted message landed.",
        instruction_binds_this_turn=False,
    )
    (group,) = build_general_groups(ctx)
    assert group.blocks["beat_goal"] == "Say how the unprompted message landed."

    bare = cat.TurnContext(candidate_turn="That is fine.", beat_avoid=SCOPE_AVOID)
    (group,) = build_general_groups(bare)
    assert "beat_goal" not in group.blocks


def test_the_avoid_lead_keeps_the_goal_exemption_without_the_ban_list():
    ctx = cat.TurnContext(
        candidate_turn="That is fine.",
        beat_avoid=SCOPE_AVOID,
        beat_intent="React to the confidence tagging.",
    )
    (group,) = build_general_groups(ctx)
    for question in group.questions:
        assert _GOAL_SENTENCE in question.text
    _, prompt = cat.render(group)
    # The list of acts is commented out: the judge
    # read it as a standing ban and stopped reading the clause, which is 32 of
    # the 33 `avoid` findings of a live user_b run.
    assert "Every other act this clause bans is still violated" not in prompt
