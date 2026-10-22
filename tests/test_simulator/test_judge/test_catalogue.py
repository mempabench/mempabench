"""The rule catalogue: shape, question ids, and the composes."""

from __future__ import annotations

import json

import pytest

from simulator.judge import catalogue as cat
from simulator.judge.guards import (
    ASK_GATE_VALUES,
    VERDICT_VALUES,
    YES_NO_NA_VALUES,
    YES_NO_VALUES,
    Verdict,
    check_evidence,
)

#: Every rule handed to the judge.
EXPECTED_RULES = {
    "carries_an_ask",
    "reaction_asserts_unexecuted_pa_action",
    "reaction_misstates_contractions",
    "probe_fact_missing",
    "reaction_authorises_pa_action",
    # Commented out, owner ruling 2026-09-13 ("先comment"): the rule is not
    # asked this round; gold, prompt and builder stay. Eight rules, not nine.
    # "reaction_states_future_policy",
    "withheld_fact_before_ask",
    "reaction_beat_off_attribute",
    "reaction_names_setting",
}


def yes(question_id: str, evidence: str = "four whole words here") -> Verdict:
    return Verdict(question_id, "yes", evidence, "")


def no(question_id: str, evidence: str = "four whole words here") -> Verdict:
    return Verdict(question_id, "no", evidence, "")


def verdicts(*items) -> dict[str, Verdict]:
    return {item.question_id: item for item in items}


# ------------------------------------------------------------------- shape


def test_every_section_nine_rule_is_in_the_catalogue():
    assert set(cat.BY_RULE) == EXPECTED_RULES


def test_no_rule_the_contract_leaves_in_code_is_in_the_catalogue():
    """These stay mechanical or style checks in code, or were deleted."""
    for rule in (
        "grades_pa_before_acting",
        "reviewer_voice_in_message",
        "quoted_span_not_in_pa_turn",
        "dictation_missing_key_terms",
        "offstage_work_reported",
        "revealed_gated_fact_early",
        "key_term_removal_agreed",
    ):
        assert rule not in cat.BY_RULE


#: The rules that return a verdict of their own; everything else is built out
#: of sub-questions a compose function reads.
SINGLE_VERDICT_RULES = {
    "reaction_states_future_policy",
    "reaction_beat_off_attribute",
    "reaction_names_setting",
}


def test_sub_questions_answer_yes_no_and_nothing_else():
    """The vocabulary split of the 2026-09-09 amendment: only a whole-rule
    verdict gets the connotative words."""
    ctx = cat.TurnContext(
        candidate_turn="You may now proceed to the next entry.",
        beat_avoid="Do not tell it what to do next beyond the year field.",
        fact_values={"a_fact": "he works from home on Monday"},
        cue="a cue",
    )
    for rule, spec in cat.BY_RULE.items():
        allowed = set(spec.build(ctx).allowed().values())
        if rule == "carries_an_ask":
            assert allowed == {ASK_GATE_VALUES}, rule
        elif rule in SINGLE_VERDICT_RULES:
            assert allowed == {VERDICT_VALUES}, rule
        elif rule == "reaction_misstates_contractions":
            # q2 carries the extra `not_applicable` for "q1 found no report"
            # (audit finding I, 2026-09-12).
            assert allowed == {YES_NO_VALUES, YES_NO_NA_VALUES}, rule
        else:
            assert allowed == {YES_NO_VALUES}, rule


#: The owner's ruling of 2026-09-09 ("没违反不引用"), one row per question: which
#: answers point at words that are IN the evidence block, and so must be backed
#: by a verbatim quote. Everything not listed asserts an ABSENCE and quotes
#: nothing. Read off what each question asserts, not off its vocabulary — three
#: of them assert presence on `obeyed`.
QUOTES_ON_TABLE = {
    "carries_an_ask": ("ask",),
    "reaction_asserts_unexecuted_pa_action__q1": ("yes",),
    "reaction_misstates_contractions__q1": ("yes",),
    "reaction_misstates_contractions__q2": ("yes",),
    "probe_fact_missing:a_fact": ("yes",),
    "reaction_authorises_pa_action__q1": ("yes",),
    "reaction_authorises_pa_action__qc1": ("yes",),
    "reaction_authorises_pa_action__q2": ("yes",),
    "reaction_authorises_pa_action__q2_first_person": ("yes",),
    "reaction_authorises_pa_action__q2_conditional": ("yes",),
    # Commented out, owner ruling 2026-09-13 ("先comment"): the rule is not
    # asked this round; gold, prompt and builder stay.
    # "reaction_states_future_policy": ("violated",),
    "withheld_fact_before_ask__q_handed": ("yes",),
    "withheld_fact_before_ask__q_wording": ("yes",),
    "withheld_fact_before_ask__q_names": ("yes",),
    "reaction_beat_off_attribute": ("violated",),
    "reaction_names_setting": ("violated",),
    "user_instruction": ("obeyed",),
    "gate_cue_missing": ("obeyed",),
    "avoid_1": ("violated",),
}


def _every_question(ctx):
    from simulator.judge.general import build_general_groups

    groups = [spec.build(ctx) for spec in cat.CATALOGUE]
    groups.extend(build_general_groups(ctx))
    return [question for group in groups for question in group.questions]


def test_every_question_declares_which_answers_it_quotes_on():
    ctx = cat.TurnContext(
        candidate_turn="You may now proceed to the next entry.",
        beat_avoid="Do not tell it what to do next beyond the year field.",
        beat_user_instruction="Ask about the year field.",
        instruction_binds_this_turn=True,
        fact_values={"a_fact": "he works from home on Monday"},
        cue="a cue",
        cue_owed=True,
    )
    got = {q.question_id: tuple(q.quotes_on) for q in _every_question(ctx)}
    assert got == QUOTES_ON_TABLE


def test_a_quoting_answer_is_always_one_the_question_allows():
    ctx = cat.TurnContext(
        candidate_turn="Book it.",
        beat_avoid="Do not act except on the year field.",
        beat_user_instruction="Ask about the year field.",
        instruction_binds_this_turn=True,
        fact_values={"a_fact": "one"},
        cue="a cue",
        cue_owed=True,
    )
    for question in _every_question(ctx):
        assert question.quotes_on, question.question_id
        assert set(question.quotes_on) <= set(question.answers), question.question_id
        assert "not_applicable" not in question.quotes_on, question.question_id


def test_a_quotes_on_answer_the_question_does_not_allow_is_refused():
    with pytest.raises(ValueError):
        cat.Question("q", "text", "candidate_turn", YES_NO_VALUES, ("violated",))


def test_every_group_is_uniform_so_the_format_union_is_exact():
    """`client.verdict_schema` unions the pairs; that is only sound while no
    group mixes vocabularies. A sub-question adding `not_applicable` is the one
    allowed widening (audit finding I, 2026-09-12) — it stays inside the yes/no
    vocabulary, and `parse_verdicts` still holds each question to its own set."""
    ctx = cat.TurnContext(
        candidate_turn="Book it.",
        beat_avoid="Do not act except on the year field.",
        fact_values={"a": "one", "b": "two"},
    )
    for rule, spec in cat.BY_RULE.items():
        sets = set(spec.build(ctx).allowed().values())
        assert sets in ({s} for s in sets) or sets <= {
            YES_NO_VALUES,
            YES_NO_NA_VALUES,
        }, rule
        assert len(sets) == 1 or sets == {YES_NO_VALUES, YES_NO_NA_VALUES}, rule


def test_two_step_rules_return_both_steps_in_one_call():
    ctx = cat.TurnContext(
        candidate_turn="You never asked me how long the review should be.",
        pa_turns=("How long should the review session be?",),
        previous_user_turns=("Draft the note.",),
    )
    for rule in ("reaction_misstates_contractions",):
        group = cat.BY_RULE[rule].build(ctx)
        assert group.question_ids == [f"{rule}__q1", f"{rule}__q2"], rule
        assert group.num_predict_kind == "wide"


def test_the_question_text_comes_from_the_rule_prompt_file():
    """The prompt files are the rule; nothing here restates their text."""
    for rule, spec in cat.BY_RULE.items():
        prompt = cat.RULE_PROMPTS[rule]
        group = spec.build(
            cat.TurnContext(
                candidate_turn="x", beat_avoid="Do not act.", fact_values={"k": "v"}
            )
        )
        assert group.preamble == prompt.preamble, rule
        assert group.questions, rule


def test_every_rule_declares_where_its_prompt_came_from():
    for rule in cat.BY_RULE:
        prompt = cat.RULE_PROMPTS[rule]
        assert prompt.status in ("measured", "adapted", "unmeasured"), rule
        assert prompt.source.strip(), rule


def test_the_measured_prompts_carry_the_conditions_the_one_liners_dropped():
    """The English summary is a gloss; these are the sentences that earned the
    numbers and that the phase-1 one-liners lost."""
    ask = cat.RULE_PROMPTS["carries_an_ask"]
    assert "reacts to what the assistant did" in ask.preamble
    # The two owner patches were removed on the ruling of 2026-09-09: the
    # measured text stands alone, and nothing about not-acting or dictated
    # content is added to it.
    assert "suppresses that sentence only" not in ask.preamble
    assert "reads out because the assistant asked" not in ask.preamble

    withheld = cat.RULE_PROMPTS["withheld_fact_before_ask"]
    assert "An OFFER is not this question." in withheld.text(
        "withheld_fact_before_ask__q_wording"
    )
    handed = withheld.text("withheld_fact_before_ask__q_handed")
    # the narrowing the step-0 replay of the deleted rule's own text forced:
    # the slot noun is worked out first, and a yes is a value, not a mention
    assert "That noun names a SLOT." in handed
    assert "the thing: written into the note" in handed
    assert "never fills the slot is a no." in handed

    policy = cat.RULE_PROMPTS["reaction_states_future_policy"]
    assert "coordinated clause" in policy.text("reaction_states_future_policy")


def test_asserts_asks_one_question_and_is_shown_no_tool_log():
    """Owner ruling B, 2026-09-10: q2 is code, so the group asks only what the
    turn claims and the log it no longer reads is not printed to it."""
    rule = "reaction_asserts_unexecuted_pa_action"
    ctx = cat.TurnContext(
        candidate_turn="You have already booked the table.",
        pa_turns=("Want me to book it?",),
        tool_log=("calendar_read(2026-09-09)",),
    )
    group = cat.BY_RULE[rule].build(ctx)
    assert group.question_ids == [f"{rule}__q1"]
    assert group.question_blocks()[f"{rule}__q1"] == ("candidate_turn",)
    assert "tool_log" not in group.blocks
    assert "Want me to book it?" in group.blocks["assistant_turns_so_far"]


def test_a_tool_log_line_carries_the_arguments_and_the_status():
    """A name on its own does not say WHICH record a write touched, and live
    regress22 `acc_006` was refused four times over seven `document_update`
    calls that had all come back ok. IX calls, the wrapper's own control
    records and errored calls are not lines here; a long value is clipped."""
    lines = cat.tool_log_lines(
        [
            {"name": "IX_task_expansion", "ix": True, "status": "ok",
             "args": {"setting": "Low"}},
            {"name": "ix_render_payload_forced", "ix_event": True},
            {"name": "mcp_state_email_message_send", "status": "error",
             "error": "gate refused", "args": {"to": "marcus@example.com"}},
            {"name": "mcp_state_document_update", "status": "ok",
             "args": {"path": "documents/research/citation_notes.md",
                      "anchor": "b" * 200, "replacement": "", "dry_run": False}},
        ]
    )

    assert lines == (
        'mcp_state_document_update {"path": "documents/research/citation_notes.md", '
        '"anchor": "' + "b" * cat.TOOL_LOG_ARG_CHARS + '…", '
        '"replacement": "", "dry_run": "false"} status=ok',
    )


def test_a_tool_log_line_with_no_arguments_is_the_name_and_the_status():
    assert cat.format_tool_log_line("mcp_state_document_read", {}, "ok") == (
        "mcp_state_document_read status=ok"
    )


def test_withheld_asks_the_handing_over_question_and_both_exemptions():
    """One question for the handing over and one for each exemption (owner
    ruling A, 2026-09-09).

    They read three different blocks. `q_handed` quotes the wording the turn
    handed over, so its evidence is the candidate turn. `q_wording` reads
    `pa_previous_turn`, the only turn that can have asked for the wording of
    THIS sentence. "Did the assistant ever ask what the thing is" is a question
    about the whole session, so `q_names` reads that block.
    """
    ctx = cat.TurnContext(
        candidate_turn="The focus is the hard questions.",
        pa_previous_turn="How long should the session run?",
        pa_turns=(
            "What should the review focus on?",
            "How long should the session run?",
        ),
        gate_key_terms=("hard", "focus"),
        gate_reveal_beat="beat_2",
        beat_id="beat_1",
    )
    group = cat.BY_RULE["withheld_fact_before_ask"].build(ctx)
    assert group.question_ids == [
        "withheld_fact_before_ask__q_handed",
        "withheld_fact_before_ask__q_wording",
        "withheld_fact_before_ask__q_names",
    ]
    points_at = group.question_blocks()
    assert points_at["withheld_fact_before_ask__q_handed"] == ("candidate_turn",)
    assert points_at["withheld_fact_before_ask__q_wording"] == ("pa_previous_turn",)
    assert points_at["withheld_fact_before_ask__q_names"] == (
        "assistant_turns_so_far",
    )
    assert group.blocks["assistant_turns_so_far"] == cat._joined(ctx.pa_turns)
    # the pre-reveal window the merged question took over comes in as context
    assert group.blocks["gate_reveal_beat"] == "beat_2"
    assert group.blocks["beat_id"] == "beat_1"


def test_withheld_pre_reveal_arm_asks_q_handed_alone():
    """Ruling A leaves both exemptions unread before the reveal beat, so the
    call does not ask them and does not print the two blocks only they read
    (audit finding H, restored by the owner 2026-09-12)."""
    ctx = cat.TurnContext(
        candidate_turn="The focus is the hard questions.",
        pa_previous_turn="How long should the session run?",
        pa_turns=(
            "What should the review focus on?",
            "How long should the session run?",
        ),
        gate_dictated_text="The focus is whether the numbers hold up.",
        gate_key_terms=("hard", "focus"),
        gate_reveal_beat="beat_2",
        beat_id="beat_1",
        pre_reveal=True,
    )
    group = cat.BY_RULE["withheld_fact_before_ask"].build(ctx)
    assert group.question_ids == ["withheld_fact_before_ask__q_handed"]
    assert group.question_blocks() == {
        "withheld_fact_before_ask__q_handed": ("candidate_turn",)
    }
    assert "pa_previous_turn" not in group.blocks
    assert "assistant_turns_so_far" not in group.blocks
    assert set(group.blocks) == {
        "candidate_turn",
        "gate_dictated_text",
        "gate_key_terms",
        "gate_reveal_beat",
        "beat_id",
        "key_terms_delivered",
    }
    # the question's own text is the same on both arms; only the frame changed
    prompt = cat.RULE_PROMPTS["withheld_fact_before_ask"]
    assert group.questions[0].text == prompt.text(
        "withheld_fact_before_ask__q_handed"
    )
    assert group.preamble == prompt.text("preamble_pre_reveal")
    assert group.preamble != prompt.preamble


def test_withheld_takes_a_question_from_an_earlier_assistant_turn():
    """An assistant that asked what the thing is two turns ago has asked, and
    the evidence for that is a verbatim span of the turn it appears in."""
    ctx = cat.TurnContext(
        candidate_turn="The focus is the hard questions.",
        pa_previous_turn="How long should the session run?",
        pa_turns=(
            "What should the review focus on?",
            "How long should the session run?",
        ),
    )
    group = cat.BY_RULE["withheld_fact_before_ask"].build(ctx)
    verdict = Verdict(
        "withheld_fact_before_ask__q_names",
        "yes",
        "assistant_turns_so_far#1",
        "focus — it asked two turns ago",
    )
    (resolved,) = check_evidence(
        [verdict], group.blocks, group.quotes_on(), group.question_blocks()
    )
    assert resolved.evidence == "What should the review focus on?"
    assert cat.compose_withheld_fact_before_ask(
        verdicts(
            Verdict("withheld_fact_before_ask__q_handed", "no", "", ""),
            Verdict("withheld_fact_before_ask__q_wording", "no", "", ""),
            resolved,
        ),
        ctx,
    ) is False


def test_withheld_is_told_the_key_term_count_rather_than_asked_for_it():
    delivered = cat.BY_RULE["withheld_fact_before_ask"].build(
        cat.TurnContext(missing_key_terms=())
    )
    assert delivered.blocks["key_terms_delivered"] == "yes"
    short = cat.BY_RULE["withheld_fact_before_ask"].build(
        cat.TurnContext(missing_key_terms=("line-by-line",))
    )
    assert short.blocks["key_terms_delivered"] == "no, missing: line-by-line"


def test_no_persona_or_attribute_name_reaches_the_prompt():
    """Two verbatim blocks plus the catalogue's own context, nothing else."""
    ctx = cat.TurnContext(
        candidate_turn="Book it.",
        pa_previous_turn="Shall I book it?",
        contractions=("I'll",),
    )
    _, prompt = cat.render(cat.BY_RULE["reaction_misstates_contractions"].build(ctx))
    assert "tone_formality" not in prompt
    assert "reception" not in prompt


# --------------------------------------------------------------- probe facts


def test_probe_fact_missing_asks_once_per_fact_in_one_call():
    ctx = cat.TurnContext(
        candidate_turn="I am working from home tomorrow morning.",
        fact_values={"monday_morning_at_home": "he is working from home on Monday morning",
                     "student_name": "Farid Rahimi"},
    )
    group = cat.BY_RULE["probe_fact_missing"].build(ctx)
    assert group.question_ids == [
        "probe_fact_missing:monday_morning_at_home",
        "probe_fact_missing:student_name",
    ]


def test_probe_fact_compose_uses_the_harness_on_request_flag():
    ctx = cat.TurnContext(
        fact_values={"said": "a", "unsaid": "b", "held": "c"},
        on_request_fact_keys=frozenset({"held"}),
        pa_asked=False,
    )
    answers = verdicts(
        yes("probe_fact_missing:said"),
        Verdict("probe_fact_missing:unsaid", "no", "", ""),
        Verdict("probe_fact_missing:held", "no", "", ""),
    )
    # `held` is on-request and the assistant has not asked, so it is not owed
    assert cat.compose_probe_fact_missing(answers, ctx) == ["unsaid"]

    ctx.pa_asked = True
    assert cat.compose_probe_fact_missing(answers, ctx) == ["unsaid", "held"]


# ---------------------------------------------------------------- exemptions


def test_exception_clauses_are_cut_out_of_avoid():
    avoid = (
        "Do not tell it what to do next beyond the year field. Do not grade the "
        "assistant."
    )
    assert cat.exception_clauses(avoid) == ["beyond the year field"]


def test_five_exception_clauses_is_an_authoring_error():
    avoid = " ".join(f"do not act except case {i};" for i in range(5))
    with pytest.raises(ValueError, match="draft-validator"):
        cat.exception_clauses(avoid)


def test_four_exception_clauses_are_allowed():
    avoid = " ".join(f"do not act except case {i};" for i in range(4))
    assert len(cat.exception_clauses(avoid)) == 4


def test_authorises_keeps_the_measured_five_questions_plus_one_per_carveout():
    ctx = cat.TurnContext(
        candidate_turn="You may now proceed to the next entry.",
        beat_avoid="Do not tell it what to do next beyond the year field.",
    )
    group = cat.BY_RULE["reaction_authorises_pa_action"].build(ctx)
    assert group.question_ids == [
        "reaction_authorises_pa_action__q1",
        "reaction_authorises_pa_action__qc1",
        "reaction_authorises_pa_action__q2",
        "reaction_authorises_pa_action__q2_first_person",
        "reaction_authorises_pa_action__q2_conditional",
    ]


def test_authorises_compose_ands_every_exemption_in_code():
    """The whole point of the 0-miss / 0-false-alarm form: the model answers
    and the code does every AND and NOT."""
    ctx = cat.TurnContext()
    rule = "reaction_authorises_pa_action"
    q1, qc1, qc2 = f"{rule}__q1", f"{rule}__qc1", f"{rule}__qc2"
    q2, first, cond = f"{rule}__q2", f"{rule}__q2_first_person", f"{rule}__q2_conditional"

    def compose(**answers):
        return cat.compose_authorises_pa_action(
            verdicts(*(yes(k) if v else no(k) for k, v in answers.items())), ctx
        )

    fires = {q1: True, qc1: False, q2: True, first: False, cond: False}
    assert compose(**fires) is True
    assert compose(**{**fires, q1: False}) is False       # avoid never banned it
    assert compose(**{**fires, qc1: True}) is False       # the ban is carved out
    assert compose(**{**fires, q2: False}) is False       # nothing was released
    assert compose(**{**fires, first: True}) is False     # the speaker's own step
    assert compose(**{**fires, cond: True}) is False      # held or conditional
    assert compose(**{**fires, qc1: False, qc2: True}) is False


# ------------------------------------------------------------ other composes


def test_asserts_is_q1_alone():
    """No second half any more (owner ruling 2026-09-12). The session boolean
    cleared every claim on a turn as soon as anything at all had run, so the
    turn that credits a send after an order went through was never asked
    about; the acts are handed over as a list and the question is asked
    against it."""
    a1 = "reaction_asserts_unexecuted_pa_action__q1"
    ordered = cat.TurnContext(executed_tools=("mcp_state_shopping_order_create",))

    assert cat.compose_asserts_unexecuted(verdicts(yes(a1)), ordered) is True
    assert cat.compose_asserts_unexecuted(verdicts(no(a1)), ordered) is False


def test_executed_acts_are_the_calls_that_changed_state():
    """The catalog's own classification (`allowed_effects`), off the prefixed
    name the PA calls the tool by. A read is not an act and an IX selection is
    not an act; every write is one, whether or not it left the sandbox, and
    they keep call order."""
    acts = cat.TurnContext(
        executed_tools=(
            "mcp_state_calendar_event_search",
            "IX_task_expansion",
            "mcp_state_document_update",
            "mcp_state_email_message_send",
        )
    ).executed_acts

    assert acts == (
        "document_update — edit an existing document in the current MemPABench run state",
        "email_message_send — send an email from the current MemPABench run state",
    )
    assert cat.TurnContext().executed_acts == ()


def test_an_in_sandbox_write_is_an_act_and_the_read_before_it_is_not():
    """Live user_b `acc_001` on the mem0 line (2026-09-13): the PA read the
    bistro staff notes and ran `document_update` on them, the list came out
    "none yet" because `internal_write` was off the effect set, and four
    rewrites of the user thanking it for that edit were all judged fictions
    (`INVALID_SIMULATOR_OUTPUT`). The question is about any act, not only the
    acts that left the sandbox."""
    acts = cat.TurnContext(
        executed_tools=("mcp_state_document_read", "mcp_state_document_update")
    ).executed_acts

    assert acts == (
        "document_update — edit an existing document in the current MemPABench run state",
    )
    assert cat.TurnContext(executed_tools=("mcp_state_document_read",)).executed_acts == ()


def test_one_act_however_many_calls_it_took():
    """The list says what was done, not how often: two sends are one act, and
    a question asked against a list with the same line twice invites the judge
    to count rather than to read."""
    assert (
        cat.TurnContext(
            executed_tools=("mcp_state_message_send", "mcp_state_message_send")
        ).executed_acts
        == ("message_send — send an informal text message to a personal contact",)
    )


def test_a_delivered_outreach_is_one_of_the_acts():
    """Live regress24m `acc_029`: the PA decided `send`, the harness delivered
    the message, and no tool name carries it because the outreach never enters
    the tool record. The actor's true "having it sent was unwelcome" then read
    as a claimed act with none executed, and the session died. It is a line of
    the list like any other, so the judge can match a claim to it."""
    sent = cat.TurnContext(executed_tools=(), outreach_sent=True)

    assert sent.executed_acts == (cat.OUTREACH_ACT,)
    assert cat.TurnContext(executed_tools=()).executed_acts == ()


def test_the_acts_block_is_printed_even_when_there_are_none():
    """`render_blocks` drops a block with no text, and the question names this
    one: with nothing executed the judge would be asked about a block that is
    not in its prompt."""
    group = cat.build_asserts_unexecuted(cat.TurnContext(candidate_turn="You sent it."))

    assert group.blocks["executed_acts"] == cat.NO_EXECUTED_ACTS
    assert "<executed_acts>" in cat.render(group)[1]


def test_named_noun_reads_the_noun_off_every_shape_the_prompt_can_produce():
    """Regress23 finding 1: the prompt now asks for `noun: one sentence`,
    because live the judge answered with the bare word — or nothing. The
    separator is not worth a schema field, so the reader takes the first
    word run and the prompt carries the shape."""
    for reason in (
        "focus: it asked what the review should focus on",
        "focus \u2014 it asked what the review should focus on",
        "focus. It asked what the review should focus on",
        "  focus: leading space",
        "Focus: capitalised",
        "focus",
    ):
        assert cat.named_noun(reason) == "focus", reason
    # nothing to read is not a noun, and the compose below must not exempt on it
    assert cat.named_noun("") == ""


def test_a_blank_reason_cannot_exempt_the_withheld_rule():
    """Fail-closed on finding 1's degenerate answers: `named_noun` returns "",
    `asked_by_name` is false, and the rule fires rather than being excused by
    a word that names no slot."""
    ctx = cat.TurnContext(missing_key_terms=())
    blank = verdicts(
        Verdict("withheld_fact_before_ask__q_handed", "no", "", "n/a"),
        Verdict("withheld_fact_before_ask__q_wording", "no", "", "n/a"),
        Verdict(
            "withheld_fact_before_ask__q_names",
            "yes",
            "What should the review focus on?",
            "",
        ),
    )
    assert cat.compose_withheld_fact_before_ask(blank, ctx) is True


def test_withheld_needs_the_named_noun_inside_the_quoted_question():
    ctx = cat.TurnContext(missing_key_terms=())
    handed_no = Verdict("withheld_fact_before_ask__q_handed", "no", "", "")
    wording_no = Verdict("withheld_fact_before_ask__q_wording", "no", "", "")

    asked_properly = verdicts(
        handed_no,
        wording_no,
        Verdict(
            "withheld_fact_before_ask__q_names",
            "yes",
            "What should the review focus on?",
            "focus — it asked for the focus of the review",
        ),
    )
    assert cat.compose_withheld_fact_before_ask(asked_properly, ctx) is False

    # the same answer, but the question it quoted never names the thing
    asked_about_something_else = verdicts(
        handed_no,
        wording_no,
        Verdict(
            "withheld_fact_before_ask__q_names",
            "yes",
            "How long should the review session be?",
            "focus — it asked about the review",
        ),
    )
    assert cat.compose_withheld_fact_before_ask(asked_about_something_else, ctx) is True


def test_withheld_fires_on_either_half_of_the_handing_over():
    """(delivered OR q_handed) AND NOT q_wording AND NOT asked_by_name.

    The two halves of the handing over are the two arming windows the merge
    joined: past the reveal beat the code counts the gate terms, and before it
    the judge is the only one who can say a turn made a stand-in up.
    """
    handed_no = Verdict("withheld_fact_before_ask__q_handed", "no", "", "")
    wording_no = Verdict("withheld_fact_before_ask__q_wording", "no", "", "")
    names_no = Verdict("withheld_fact_before_ask__q_names", "no", "", "")

    # delivered by code, and the judge saw nothing handed over of its own
    delivered = cat.TurnContext(missing_key_terms=())
    assert cat.compose_withheld_fact_before_ask(
        verdicts(handed_no, wording_no, names_no), delivered
    ) is True

    # nothing delivered, but the judge says the turn handed wording over anyway
    invented = cat.TurnContext(missing_key_terms=("chest tightening",))
    handed_yes = Verdict(
        "withheld_fact_before_ask__q_handed",
        "yes",
        "The reason for the visit is a routine preventive checkup",
        "",
    )
    assert cat.compose_withheld_fact_before_ask(
        verdicts(handed_yes, wording_no, names_no), invented
    ) is True

    # the same handing over, but the assistant had asked for the wording
    assert cat.compose_withheld_fact_before_ask(
        verdicts(
            handed_yes,
            Verdict(
                "withheld_fact_before_ask__q_wording",
                "yes",
                "What would you like it to say?",
                "",
            ),
            names_no,
        ),
        invented,
    ) is False

    # the same handing over, but the assistant had asked what the thing is
    assert cat.compose_withheld_fact_before_ask(
        verdicts(
            handed_yes,
            wording_no,
            Verdict(
                "withheld_fact_before_ask__q_names",
                "yes",
                "What is the reason for the visit?",
                "reason — it asked what the visit is for",
            ),
        ),
        invented,
    ) is False

    # neither half: no terms delivered and nothing handed over
    assert cat.compose_withheld_fact_before_ask(
        verdicts(handed_no, wording_no, names_no), invented
    ) is False


def test_withheld_is_exempted_by_a_request_for_the_wording():
    """Restored with `q_wording` itself (owner ruling A, 2026-09-09): the code's
    own key-term count fires the rule, and this is the answer that excuses it."""
    ctx = cat.TurnContext(missing_key_terms=())
    answers = verdicts(
        Verdict("withheld_fact_before_ask__q_handed", "yes", "the focus", ""),
        Verdict(
            "withheld_fact_before_ask__q_wording",
            "yes",
            "What would you like it to say?",
            "",
        ),
        Verdict("withheld_fact_before_ask__q_names", "no", "", ""),
    )
    assert cat.compose_withheld_fact_before_ask(answers, ctx) is False


def test_withheld_pre_reveal_takes_q_handed_alone():
    """Before the reveal beat neither exemption excuses the handing over (owner
    ruling A, 2026-09-09): the sentence is not the actor's to give yet, however
    early the assistant asked for it."""
    handed_yes = Verdict(
        "withheld_fact_before_ask__q_handed",
        "yes",
        "The reason for the visit is a routine preventive checkup",
        "",
    )
    wording_yes = Verdict(
        "withheld_fact_before_ask__q_wording",
        "yes",
        "What would you like it to say?",
        "",
    )
    names_yes = Verdict(
        "withheld_fact_before_ask__q_names",
        "yes",
        "What is the reason for the visit?",
        "reason — it asked what the visit is for",
    )
    answers = verdicts(handed_yes, wording_yes, names_yes)

    pre_reveal = cat.TurnContext(
        missing_key_terms=("chest tightening",), pre_reveal=True
    )
    assert cat.compose_withheld_fact_before_ask(answers, pre_reveal) is True

    # the same answers on the on-request arm, where both exemptions do apply
    on_request = cat.TurnContext(missing_key_terms=("chest tightening",))
    assert cat.compose_withheld_fact_before_ask(answers, on_request) is False


def test_withheld_pre_reveal_ignores_code_delivery():
    """Code's key-term count is the on-request arm's half of the handing over.
    Before the reveal there is no real sentence to count against, so only the
    judge's answer decides."""
    delivered = cat.TurnContext(missing_key_terms=(), pre_reveal=True)
    wording_no = Verdict("withheld_fact_before_ask__q_wording", "no", "", "")
    names_no = Verdict("withheld_fact_before_ask__q_names", "no", "", "")

    handed_no = Verdict("withheld_fact_before_ask__q_handed", "no", "", "")
    assert cat.compose_withheld_fact_before_ask(
        verdicts(handed_no, wording_no, names_no), delivered
    ) is False

    handed_yes = Verdict(
        "withheld_fact_before_ask__q_handed", "yes", "the focus is the hard questions", ""
    )
    assert cat.compose_withheld_fact_before_ask(
        verdicts(handed_yes, wording_no, names_no), delivered
    ) is True


def test_authorises_pa_action_does_not_print_the_assistants_own_turns():
    """Audit finding G (2026-09-12): no sub-question reads that block, and the
    assistant's own go-aheads inside it pull q2 onto the wrong speaker."""
    group = cat.build_authorises_pa_action(
        cat.TurnContext(
            candidate_turn="Go ahead and book it.",
            beat_avoid="Do not tell it what to do next.",
            pa_turns=("Shall I go ahead and book it?",),
        )
    )
    assert "assistant_turns_so_far" not in group.blocks
    assert "assistant_turns_so_far" not in group.preamble


def test_misstates_contractions_q2_is_not_applicable_when_q1_found_no_report():
    """Audit finding I (2026-09-12): q1 = no leaves q2 nothing to judge, and
    compose never reads it on that branch."""
    q1, q2 = "reaction_misstates_contractions__q1", "reaction_misstates_contractions__q2"
    group = cat.build_misstates_contractions(cat.TurnContext(candidate_turn="Fine."))
    assert group.allowed()[q2] == YES_NO_NA_VALUES
    no_report = verdicts(Verdict(q1, "no", "", ""), Verdict(q2, "not_applicable", "", ""))
    assert cat.compose_misstates_contractions(no_report, cat.TurnContext()) is False


# ----------------------------------------------------------------- rendering


def test_every_shape_has_a_prompt_file_with_a_system_block():
    for shape in ("per_rule", "two_step", "general"):
        system, body = cat.load_prompt(shape)
        assert "Do not write a code fence" in system
        assert "{{context}}" in body and "{{questions}}" in body


def test_render_fills_both_placeholders():
    ctx = cat.TurnContext(candidate_turn="Book it now please.", pa_previous_turn="Shall I?")
    _, prompt = cat.render(cat.BY_RULE["carries_an_ask"].build(ctx))
    assert "{{" not in prompt
    assert "Book it now please." in prompt


# ------------------------------------------- evidence spans, and empty groups


def test_a_probe_fact_may_quote_an_earlier_turn_as_well_as_this_one():
    """The preamble says in as many words that a fact delivered on an earlier
    turn counts. Checked against the candidate turn alone, every correct `yes`
    of that kind fails the guard — and under phase 2 a failed guard voids the
    session."""
    ctx = cat.TurnContext(
        candidate_turn="That is all for the appointment itself.",
        previous_user_turns=("I am working from home tomorrow morning.",),
        fact_values={"monday_morning_at_home": "he is working from home on Monday morning"},
    )
    group = cat.BY_RULE["probe_fact_missing"].build(ctx)
    question = group.questions[0]
    assert question.evidence_block_names == ("candidate_turn", "earlier_user_turns")

    assert group.blocks["earlier_user_turns"] == (
        "I am working from home tomorrow morning."
    )
    assert group.blocks["candidate_turn"] == "That is all for the appointment itself."

    _, prompt = cat.render(group)
    assert "<candidate_turn> or <earlier_user_turns>" in prompt


def test_a_probe_yes_quoting_an_earlier_turn_passes_the_guard():
    from simulator.judge.guards import check_evidence, parse_verdicts

    ctx = cat.TurnContext(
        candidate_turn="That is all for the appointment itself.",
        previous_user_turns=("I am working from home tomorrow morning.",),
        fact_values={"monday_morning_at_home": "he is working from home on Monday morning"},
    )
    group = cat.BY_RULE["probe_fact_missing"].build(ctx)
    raw = json.dumps(
        {
            "verdicts": [
                {
                    "question_id": "probe_fact_missing:monday_morning_at_home",
                    "verdict": "yes",
                    "evidence": "earlier_user_turns#1",
                    "reason": "delivered on the earlier turn",
                }
            ]
        }
    )
    answers = parse_verdicts(raw, allowed=group.allowed())
    (resolved,) = check_evidence(  # would raise before this fix
        answers, group.blocks, group.quotes_on(), group.question_blocks()
    )
    assert resolved.evidence == "I am working from home tomorrow morning."


def test_a_beat_with_no_required_facts_builds_no_questions():
    """413 of the 546 corpus records owe no facts at all."""
    group = cat.BY_RULE["probe_fact_missing"].build(cat.TurnContext(fact_values={}))
    assert group.is_empty
    assert group.allowed() == {}


def test_a_group_with_questions_is_not_empty():
    group = cat.BY_RULE["probe_fact_missing"].build(
        cat.TurnContext(fact_values={"k": "v"})
    )
    assert not group.is_empty


# --------------------------------------------- the two draft-stage rules


def test_the_off_attribute_rule_is_told_which_attribute_is_the_target():
    """The judge gets no attribute name; this one question cannot be put
    without it, so the exception is a block and is written down as one."""
    group = cat.build_reaction_beat_off_attribute(
        cat.TurnContext(
            candidate_turn="Say something about how long that was.",
            target_attribute="guidance_level",
        )
    )
    assert group.blocks["target_attribute"] == "guidance_level"
    assert group.question_ids == ["reaction_beat_off_attribute"]
    assert cat.compose_reaction_beat_off_attribute(
        verdicts(Verdict("reaction_beat_off_attribute", "violated", "a b c d", "")),
        cat.TurnContext(),
    )


def test_the_setting_name_rule_reads_the_candidate_turn_alone():
    group = cat.build_reaction_names_setting(
        cat.TurnContext(candidate_turn="Be more Proactive from now on.")
    )
    assert group.question_ids == ["reaction_names_setting"]
    assert set(group.blocks) == {"pa_previous_turn", "candidate_turn"}
    assert not cat.compose_reaction_names_setting(
        verdicts(Verdict("reaction_names_setting", "obeyed", "", "")), cat.TurnContext()
    )


def test_both_new_rules_are_unmeasured_and_render():
    for rule in ("reaction_beat_off_attribute", "reaction_names_setting"):
        assert cat.RULE_PROMPTS[rule].status == "unmeasured"
        ctx = cat.TurnContext(candidate_turn="Keep it short.", target_attribute="verbosity")
        system, prompt = cat.render(cat.BY_RULE[rule].build(ctx))
        assert system.strip() and "{{" not in prompt


def test_the_evidence_tail_is_a_short_field_line():
    """The grammar of a reference lives in the system prompt, once; the
    per-question tail carries only what varies (framework audit 2026-09-12)."""
    group = cat.QuestionGroup(
        rule="r",
        shape="general",
        questions=(
            cat.Question("q1", "Body.", "candidate_turn", cat.VERDICT_VALUES),
            cat.Question("q2", "Body.", "candidate_turn", cat.VERDICT_VALUES, ()),
        ),
        blocks={"candidate_turn": "One."},
    )
    rendered = cat.render_questions(group)
    assert 'evidence: on "violated" → <candidate_turn>#N; otherwise empty' in rendered
    assert "evidence: always empty" in rendered
    assert "point your evidence at a numbered sentence" not in rendered


def test_carries_an_ask_speaks_the_schemas_values_and_asks_no_negative_quote():
    prompt = cat.RULE_PROMPTS["carries_an_ask"]
    body = prompt.preamble + prompt.text("carries_an_ask")
    assert "YES" not in body and "Answer NO" not in body
    assert '"ask"' in body and '"not_applicable"' in body
    # `check_evidence` rejects evidence on `not_applicable`, so the question
    # must not ask for it (framework audit 2026-09-12).
    assert "otherwise the sentence that best shows" not in body


def test_probe_fact_missing_asks_about_both_blocks():
    body = cat.RULE_PROMPTS["probe_fact_missing"].text(
        "probe_fact_missing:<fact_key>", fact="x"
    )
    assert "<candidate_turn> or in <earlier_user_turns>" in body
    assert "stated in this turn" not in body


def test_two_step_body_references_a_sentence_rather_than_quoting():
    body = (cat.PROMPTS_DIR / "two_step.md").read_text(encoding="utf-8")
    assert "quote the words that make it so" not in body
    assert "reference the numbered sentence" in body
