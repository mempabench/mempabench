"""P3 · producing and rendering `user_facing_rationale` + appending the `certainties` parenthesis.

IX wrapper plan §3.2 / §3.1 uncertainty row / §4.4
(Obsidian `Plans/IX_Interaction_Tool_Development_Log.md`),
plus the unset fallback both entry points must have.
"""

from __future__ import annotations

import pytest

from pa.agent.ix.rationale import (
    CERTAINTY_BLOCK_INSTRUCTION,
    CERTAINTY_LABELS,
    RATIONALE_SOURCE,
    certainty_annotation,
    claim_end,
    has_certainty_block,
    normalize_rationale,
    parse_certainty_block,
    render_certainties,
    render_rationale,
    strip_certainty_block,
)
from pa.agent.ix.render_types import (
    CERTAINTY_MARKER_RE,
    Certainty,
)
from pa.providers.openai_compat_provider import OpenAICompatProvider

RATIONALE = (
    "Only em_0100 of the two mentions 4B: its stack id matches the batch "
    "number you gave me, and the other one is dated outside the window"
)

#: The unset fallback: the attribute has no selection this session.
UNSET = ""

REASONING_SETTINGS = ("Show", "Hide", UNSET)
UNCERTAINTY_SETTINGS = ("Express", "Hide", UNSET)


# ---------------------------------------------------------------------------
# reasoning_visibility (§3.2)
# ---------------------------------------------------------------------------


def test_rationale_is_validated_identically_in_every_setting() -> None:
    """Whether it is required must not leak the setting: all three validate identically (§3.2)."""
    assert normalize_rationale(RATIONALE) == RATIONALE
    assert normalize_rationale("  " + RATIONALE + "\n") == RATIONALE
    for empty in ("", "   ", None, 42, ["a"], {"text": "a"}):
        assert normalize_rationale(empty) == ""

    for setting in REASONING_SETTINGS:
        missing = render_rationale(setting, None)
        assert missing.prefix == ""
        assert missing.shown is False
        assert missing.reasoning["source"] == RATIONALE_SOURCE


def test_show_releases_the_pa_written_rationale() -> None:
    result = render_rationale("Show", RATIONALE)
    assert result.prefix == RATIONALE
    assert result.shown is True
    assert result.reasoning == {
        "shown": True,
        "source": RATIONALE_SOURCE,
        "provider_reasoning_present": False,
    }
    assert result.trace == ["reasoning_visibility=Show -> rationale released (133 chars)"]


@pytest.mark.parametrize("setting", ["Hide", UNSET])
def test_hide_and_unset_release_nothing(setting: str) -> None:
    """`Hide` discards and an unselected attribute applies no preference -- both branches visibly release nothing."""
    result = render_rationale(setting, RATIONALE)
    assert result.prefix == ""
    assert result.shown is False
    assert result.reasoning["shown"] is False
    assert result.reasoning["source"] == RATIONALE_SOURCE
    assert RATIONALE not in "".join(result.trace)


def test_show_with_an_empty_field_does_not_synthesise_a_sentence() -> None:
    result = render_rationale("Show", "   ")
    assert result.prefix == ""
    assert result.shown is False
    assert "empty" in result.trace[0]


def test_unknown_setting_falls_back_to_the_unset_branch_and_says_so() -> None:
    result = render_rationale("Summarize", RATIONALE)
    assert result.prefix == ""
    assert result.shown is False
    assert result.trace == [
        "reasoning_visibility=Summarize -> unset fallback, nothing released"
    ]


def test_provider_reasoning_is_only_a_boolean_never_a_render_source() -> None:
    """§3.2: provider reasoning goes to audit_trace only, and `source` is always `user_facing_rationale`."""
    result = render_rationale("Show", RATIONALE, provider_reasoning_present=True)
    assert result.reasoning["provider_reasoning_present"] is True
    assert result.reasoning["source"] == RATIONALE_SOURCE
    assert result.prefix == RATIONALE

    hidden = render_rationale("Hide", RATIONALE, provider_reasoning_present=True)
    assert hidden.prefix == ""
    assert hidden.reasoning["provider_reasoning_present"] is True


# ---------------------------------------------------------------------------
# uncertainty_expression (§3.1 that row, §4.4)
# ---------------------------------------------------------------------------

TEXT = "Monday morning is free[[c:c1]], and room 302 is still open[[c:c2]]."
CERTAINTIES = [Certainty(ref="c1", level="low"), Certainty(ref="c2", level="high")]


def test_certainty_annotation_matches_the_format_frozen_in_the_plan() -> None:
    """§3.1's uncertainty row hard-codes `(confidence: low)`; §4.4's `rendered` is identical, verbatim."""
    assert certainty_annotation("low") == "(confidence: low)"
    assert certainty_annotation("medium") == "(confidence: medium)"
    assert certainty_annotation("high") == "(confidence: high)"
    assert certainty_annotation("unknown") == ""
    assert set(CERTAINTY_LABELS) == {"high", "medium", "low"}


def test_express_annotates_every_marker() -> None:
    result = render_certainties("Express", TEXT, CERTAINTIES)
    assert result.text == (
        "Monday morning is free(confidence: low), "
        "and room 302 is still open(confidence: high)."
    )
    assert result.rows == [
        {"ref": "c1", "level": "low", "rendered": "(confidence: low)"},
        {"ref": "c2", "level": "high", "rendered": "(confidence: high)"},
    ]
    assert result.trace[0] == "uncertainty_expression=Express -> annotated 2/2 markers"


@pytest.mark.parametrize("setting", ["Hide", UNSET])
def test_hide_and_unset_annotate_nothing(setting: str) -> None:
    result = render_certainties(setting, TEXT, CERTAINTIES)
    assert result.text == "Monday morning is free, and room 302 is still open."
    assert [row["rendered"] for row in result.rows] == ["", ""]
    assert result.rows == [
        {"ref": "c1", "level": "low", "rendered": ""},
        {"ref": "c2", "level": "high", "rendered": ""},
    ]


@pytest.mark.parametrize("setting", [*UNCERTAINTY_SETTINGS, "Summarize"])
def test_markers_are_stripped_in_every_branch(setting: str) -> None:
    """Contract 6: not one `[[c:*]]` may reach the user verbatim, under the unset fallback and an unknown setting too."""
    result = render_certainties(setting, TEXT, CERTAINTIES)
    assert "[[c:" not in result.text
    assert CERTAINTY_MARKER_RE.search(result.text) is None


def test_unknown_marker_ref_is_stripped_without_an_annotation() -> None:
    result = render_certainties(
        "Express",
        "The ticket can still be rebooked[[c:cX]].",
        [Certainty(ref="c1", level="low")],
    )
    assert result.text == "The ticket can still be rebooked."
    assert result.rows == [{"ref": "c1", "level": "low", "rendered": ""}]
    assert "unknown marker ref cX stripped" in " ".join(result.trace)
    assert "ref c1 has no [[c:…]] marker in the text" in " ".join(result.trace)


def test_duplicate_ref_keeps_the_first_level_for_both_rows() -> None:
    result = render_certainties(
        "Express",
        "Just this one spot[[c:c1]].",
        [Certainty(ref="c1", level="low"), Certainty(ref="c1", level="high")],
    )
    assert result.text == "Just this one spot(confidence: low)."
    assert [row["rendered"] for row in result.rows] == ["(confidence: low)", "(confidence: low)"]
    assert "duplicate ref c1 ignored" in " ".join(result.trace)


def test_empty_certainties_on_an_unmarked_answer_is_a_no_op() -> None:
    """The PA gives an empty list when the prose has no marker -- the render must pass it straight through (§3.1 that row's required rule)."""
    for setting in UNCERTAINTY_SETTINGS:
        result = render_certainties(setting, "Monday morning is already scheduled.", [])
        assert result.text == "Monday morning is already scheduled."
        assert result.rows == []


def test_the_same_ref_twice_in_the_text_is_annotated_twice() -> None:
    result = render_certainties(
        "Express",
        "Said once[[c:c1]], and said again[[c:c1]].",
        [Certainty(ref="c1", level="low")],
    )
    assert result.text == "Said once(confidence: low), and said again(confidence: low)."
    assert result.trace[0] == "uncertainty_expression=Express -> annotated 2/2 markers"


# ---------------------------------------------------------------------------
# the claim channel: tool calls are produced before the prose, so the position
# comes from locating the `claim`, not from a marker
# ---------------------------------------------------------------------------

CLAIM_TEXT = (
    "Monday morning is free, so I can put the review there.\n\n"
    "Room 302 is still open, so I am holding it."
)
CLAIM_CERTAINTIES = [
    Certainty(ref="c1", level="low", claim="Monday morning is free"),
    Certainty(ref="c2", level="high", claim="Room 302 is still open"),
]


def test_express_annotates_after_each_claim_it_finds() -> None:
    """The parenthesis lands after the claim's passage, with no marker anywhere in the prose."""
    result = render_certainties("Express", CLAIM_TEXT, CLAIM_CERTAINTIES)
    assert result.text == (
        "Monday morning is free(confidence: low), so I can put the review there.\n\n"
        "Room 302 is still open(confidence: high), so I am holding it."
    )
    # `claim` / `strategy` ride along for the audit plane (§4.4 certainties[]).
    assert result.rows == [
        {
            "ref": "c1", "level": "low", "rendered": "(confidence: low)",
            "claim": "Monday morning is free", "strategy": "exact",
        },
        {
            "ref": "c2", "level": "high", "rendered": "(confidence: high)",
            "claim": "Room 302 is still open", "strategy": "exact",
        },
    ]
    assert "uncertainty_expression=Express -> annotated 2/2 claims" in result.trace
    assert "uncertainty_expression=Express -> annotated 0/0 markers" in result.trace


@pytest.mark.parametrize("setting", ["Hide", UNSET])
def test_hide_and_unset_insert_nothing_for_a_found_claim(setting: str) -> None:
    """A found claim is still not annotated -- the setting decides whether to insert, the lookup only where (§3.1 that row)."""
    result = render_certainties(setting, CLAIM_TEXT, CLAIM_CERTAINTIES)
    assert result.text == CLAIM_TEXT
    assert [row["rendered"] for row in result.rows] == ["", ""]
    assert "(confidence: " not in result.text


def test_a_claim_that_is_not_in_the_reply_inserts_nothing_and_is_traced() -> None:
    """The payload was given, so this is not a payload miss -- it only writes one trace line (round 2 finding 1)."""
    result = render_certainties(
        "Express",
        "I will not touch the roster until you have checked the paper quizzes.",
        [Certainty(
            ref="farid_quiz", level="high", claim="Farid's answers are worth full marks"
        )],
    )
    assert result.text == (
        "I will not touch the roster until you have checked the paper quizzes."
    )
    # `strategy: not_found` is the difference this row exists to record.
    assert result.rows == [{
        "ref": "farid_quiz", "level": "high", "rendered": "",
        "claim": "Farid's answers are worth full marks", "strategy": "not_found",
    }]
    assert "certainty ref farid_quiz: claim not found" in result.trace
    assert "uncertainty_expression=Express -> annotated 0/1 claims" in result.trace


def test_a_claim_is_found_across_a_line_break_and_case_difference() -> None:
    """Compare after whitespace collapsing; when the case-sensitive pass misses, go case-insensitive."""
    result = render_certainties(
        "Express",
        "The clinic slot on\n  Tuesday  is open, so I booked it.",
        [Certainty(ref="c1", level="low", claim="the clinic slot on Tuesday is open")],
    )
    assert result.text == (
        "The clinic slot on\n  Tuesday  is open(confidence: low), so I booked it."
    )


def test_the_marker_wins_when_a_ref_has_both() -> None:
    """When the model happens to write a marker it is placed there, and the claim does not insert a second one."""
    result = render_certainties(
        "Express",
        "Monday morning is free[[c:c1]], so I can put the review there.",
        [Certainty(ref="c1", level="low", claim="Monday morning is free")],
    )
    assert result.text == (
        "Monday morning is free(confidence: low), so I can put the review there."
    )
    assert result.text.count("(confidence: low)") == 1
    assert "uncertainty_expression=Express -> annotated 1/1 markers" in result.trace
    assert not [line for line in result.trace if "claims" in line]


def test_a_ref_with_a_claim_does_not_report_a_missing_marker() -> None:
    """On the claim channel "no marker" is not a finding, so stop reporting it (or the trace is all noise)."""
    result = render_certainties("Express", CLAIM_TEXT, CLAIM_CERTAINTIES)
    assert not [line for line in result.trace if "has no [[c:…]] marker" in line]


# ---------------------------------------------------------------------------
# lookup normalisation: round 3 finding 1 (one full stop made two verbatim claims miss)
# ---------------------------------------------------------------------------

PERIOD_TEXT = (
    "The email has been sent to Farid Rahimi successfully (sent ID: **sent_0001**)."
)
PERIOD_CLAIM = "The email has been sent to Farid Rahimi successfully."


def test_a_trailing_full_stop_no_longer_hides_a_verbatim_claim() -> None:
    """Round 3 finding 1: in the prose it is a clause, and in the block the model wrote it as a sentence."""
    match = claim_end(PERIOD_TEXT, PERIOD_CLAIM)
    assert match is not None
    assert match.strategy == "normalized"
    assert PERIOD_TEXT[: match.end].endswith("successfully")

    result = render_certainties(
        "Express", PERIOD_TEXT, [Certainty(ref="c1", level="high", claim=PERIOD_CLAIM)]
    )
    assert result.text == (
        "The email has been sent to Farid Rahimi successfully(confidence: high)"
        " (sent ID: **sent_0001**)."
    )
    assert result.rows == [{
        "ref": "c1", "level": "high", "rendered": "(confidence: high)",
        "claim": "The email has been sent to Farid Rahimi successfully.",
        "strategy": "normalized",
    }]


def test_markdown_emphasis_inside_the_span_is_normalised_away() -> None:
    """The model bolded half a sentence in the prose, and copied the asterisk-free version into the block."""
    text = "Farid Rahimi's **Q1 score is now 9/10**, and the roster note is in."
    result = render_certainties(
        "Express",
        text,
        [Certainty(ref="c1", level="medium", claim="Farid Rahimi's Q1 score is now 9/10")],
    )
    # The parenthesis lands outside the bold -- it may not go inside `**…**`.
    assert result.text == (
        "Farid Rahimi's **Q1 score is now 9/10**(confidence: medium), and the roster note is in."
    )
    assert "certainty ref c1: located (normalized), annotated" in result.trace


def test_backticks_and_case_do_not_break_an_otherwise_verbatim_claim() -> None:
    result = render_certainties(
        "Express",
        "The invoice `INV-4021` is still unpaid, so I held the transfer.",
        [Certainty(ref="c1", level="low", claim="the invoice INV-4021 is still unpaid")],
    )
    assert "still unpaid(confidence: low)" in result.text


def test_a_reworded_claim_is_found_by_token_overlap() -> None:
    """80% of the tokens matching in order counts as the same passage -- the last pass, taken only when the first two miss."""
    text = "Yes, Farid's submission qualifies on the exchange implication."
    match = claim_end(
        text, "Farid's submission qualifies for the exchange implication."
    )
    assert match is not None
    assert match.strategy == "token_overlap"
    # Past the full stop, not before it: an overlap end is snapped to the end of
    # the clause it landed in (`_clause_boundary`), and here that is the end of
    # the reply.
    assert text[: match.end].endswith("implication.")


#: acc_035 t=18, verbatim. The prose sentence and the claim the payload declared
#: for it: the claim inserts "on quiz 1 question 2" before the closing phrase.
ACC_035_PROSE = (
    "Farid Rahimi is the only student I can positively identify as affected by "
    "the quiz 1, question 2 ambiguity. I cannot determine the exact identities "
    "or total count of other students who lost the same point from the "
    "available documents. You may want to check the physical graded quiz "
    'sheets to see who else answered "two".'
)
ACC_035_CLAIM = (
    "I cannot determine the exact identities or total count of other students "
    "who lost the same point on quiz 1 question 2 from the available documents."
)


def test_a_claim_with_a_long_insertion_is_still_found() -> None:
    """acc_035 t=18 — 21 of 26 tokens present, 81%, and the pass returned None.

    The in-order walk could skip at most two needle tokens, and the insertion is
    five. The first `low` claim in four rounds went unbracketed, on the one cell
    that exists to produce that bracket. The skip window is now the complement
    of the threshold: 20% of the needle is the most that may go missing in
    total, so nothing wider than that could reach the threshold anyway.
    """
    match = claim_end(ACC_035_PROSE, ACC_035_CLAIM)

    assert match is not None
    assert match.strategy == "token_overlap"
    # The sentence's own full stop, snapped to by `_clause_boundary`.
    assert ACC_035_PROSE[: match.end].endswith("from the available documents.")


#: regress18 acc_035, PA turn t=3, verbatim from the run's own transcript. The
#: prose sentence carries one extra word ("therefore") against the claim the
#: `<certainties>` block declared for it, so only `token_overlap` can find it —
#: and the preamble matters: the pass's window is bounded at `2 * len(needle) + 4`
#: haystack tokens, so a window that starts in the note above collects enough of
#: the claim to pass the threshold while its tail ("than as a phase") falls off
#: the end. The last matched token is then "rather", mid-clause.
ACC_035_T3_PROSE = (
    'out the exchange got the implication right and the count "wrong" by the '
    "key. wording is the problem, not the physics.*\n"
    "\n"
    "Farid's answer was exactly that pattern: σ × σ = 1 + ψ has two fusion "
    "outcomes, and exchange therefore acts on a degenerate space rather than "
    "as a phase."
)
ACC_035_T3_CLAIM = (
    "σ × σ = 1 + ψ has two fusion outcomes, and exchange acts on a degenerate "
    "space rather than as a phase."
)


def test_a_token_overlap_marker_lands_on_the_clause_boundary() -> None:
    """regress18 acc_035 t=3: the marker went in as `rather(confidence: high) than`.

    The overlap pass says *which passage* the claim is, not where it ends, so
    its index is only a starting point — the marker is pushed to the end of the
    clause it landed in.
    """
    match = claim_end(ACC_035_T3_PROSE, ACC_035_T3_CLAIM)

    assert match is not None
    assert match.strategy == "token_overlap"
    assert ACC_035_T3_PROSE[: match.end].endswith("as a phase.")

    result = render_certainties(
        "Express",
        ACC_035_T3_PROSE,
        [Certainty(ref="c1", level="high", claim=ACC_035_T3_CLAIM)],
    )
    assert "rather(confidence: high) than" not in result.text
    assert "as a phase.(confidence: high)" in result.text


def test_a_token_overlap_match_with_no_boundary_ahead_is_not_found() -> None:
    """No clause end to push the marker to → the not-found path, not a splice.

    `ix_certainty_claim_not_found` is what the read tables already count; a
    marker inside a word pair is not an acceptable second-best.
    """
    text = ACC_035_T3_PROSE.replace(
        "rather than as a phase.", "rather than as a phase and both halves hold"
    )

    assert claim_end(text, ACC_035_T3_CLAIM) is None


def test_a_genuinely_different_sentence_is_still_not_found() -> None:
    """The fallback is not "anything matches": too little overlap is still not found."""
    assert claim_end(
        "I have not touched the roster yet.",
        "The email has been sent to Farid Rahimi successfully.",
    ) is None


def test_the_strategy_that_matched_is_recorded_in_the_trace() -> None:
    result = render_certainties(
        "Express", CLAIM_TEXT, [CLAIM_CERTAINTIES[0]]
    )
    assert "certainty ref c1: located (exact), annotated" in result.trace


# ---------------------------------------------------------------------------
# round 3 finding 4: `Hide`'s "no parenthesis" has to be falsifiable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setting", ["Hide", UNSET])
def test_a_located_but_withheld_claim_leaves_a_trace_line(setting: str) -> None:
    """Found but withheld under this setting, and never found at all, must be separable from `ix_render`."""
    result = render_certainties(setting, CLAIM_TEXT, CLAIM_CERTAINTIES)
    assert result.text == CLAIM_TEXT
    assert [row["rendered"] for row in result.rows] == ["", ""]
    label = setting or "unset"
    for ref in ("c1", "c2"):
        assert f"certainty ref {ref}: located (exact), withheld ({label})" in result.trace
    assert "claim not found" not in " ".join(result.trace)


# ---------------------------------------------------------------------------
# §5.3 carrier F · the `<certainties>` block at the end of the reply
# ---------------------------------------------------------------------------

BLOCK_REPLY = (
    "Monday morning is free, so I put the review there.\n"
    "Room 302 is still open as well.\n"
    "<certainties>\n"
    "- ref: c1 | level: low | claim: Monday morning is free\n"
    "- ref: c2 | level: high | claim: Room 302 is still open as well\n"
    "</certainties>"
)


def test_the_block_is_parsed_into_certainty_rows() -> None:
    block = parse_certainty_block(BLOCK_REPLY)
    assert block.found is True
    assert block.rows == [
        Certainty(ref="c1", level="low", claim="Monday morning is free"),
        Certainty(ref="c2", level="high", claim="Room 302 is still open as well"),
    ]
    assert "<certainties>" not in block.text
    assert block.text.endswith("Room 302 is still open as well.")
    assert "certainty block: 2 row(s) read from the reply" in block.trace


def test_an_empty_block_is_a_valid_payload() -> None:
    """With no claim whose confidence matters this turn, an empty block is the answer -- not a missing field."""
    block = parse_certainty_block("All booked.\n<certainties></certainties>")
    assert block.found is True
    assert block.rows == []
    assert block.text == "All booked."


def test_a_reply_without_a_block_is_left_alone() -> None:
    block = parse_certainty_block("All booked.")
    assert block.found is False
    assert block.rows == []
    assert block.text == "All booked."
    assert has_certainty_block("All booked.") is False
    assert strip_certainty_block("All booked.") == "All booked."


def test_a_malformed_row_is_traced_and_skipped_not_fatal() -> None:
    block = parse_certainty_block(
        "Done.\n<certainties>\n- level: high | claim: Done\n"
        "- ref: c2 | level: high | claim: Done\n</certainties>"
    )
    assert [row.ref for row in block.rows] == ["c2"]
    assert any("line not parsed" in line for line in block.trace)


def test_a_duplicate_ref_in_the_block_is_dropped() -> None:
    block = parse_certainty_block(
        "Done.\n<certainties>\n- ref: c1 | level: high | claim: Done\n"
        "- ref: c1 | level: low | claim: Done\n</certainties>"
    )
    assert [row.level for row in block.rows] == ["high"]
    assert any("duplicate or empty ref" in line for line in block.trace)


@pytest.mark.parametrize("setting", [*UNCERTAINTY_SETTINGS, "Summarize"])
def test_the_block_never_survives_into_the_rendered_text(setting: str) -> None:
    """The contract: the block is a carrier, not prose -- under all four settings plus an unknown one, not one character may leak."""
    block = parse_certainty_block(BLOCK_REPLY)
    result = render_certainties(setting, BLOCK_REPLY, block.rows)
    assert "<certainties>" not in result.text
    assert "</certainties>" not in result.text
    assert "ref: c1" not in result.text
    assert result.text.startswith("Monday morning is free")
    assert "trailing block stripped from the reply" in " ".join(result.trace)


def test_express_annotates_the_claims_the_block_quoted() -> None:
    block = parse_certainty_block(BLOCK_REPLY)
    result = render_certainties("Express", BLOCK_REPLY, block.rows)
    assert result.text == (
        "Monday morning is free(confidence: low), so I put the review there.\n"
        "Room 302 is still open as well(confidence: high)."
    )
    assert [row["rendered"] for row in result.rows] == ["(confidence: low)", "(confidence: high)"]


def test_the_block_instruction_is_setting_blind() -> None:
    """Both settings read the same passage, and writing the block or not leaks no setting (same required rule as §3.2)."""
    for setting in ("Express", "Hide"):
        assert setting not in CERTAINTY_BLOCK_INSTRUCTION
    assert "<certainties></certainties>" in CERTAINTY_BLOCK_INSTRUCTION
    assert "word for word" in CERTAINTY_BLOCK_INSTRUCTION


# ---------------------------------------------------------------------------
# optional audit capture: the three provider reasoning keys (§3.2 appendix, §7.2 P3 optional item)
# ---------------------------------------------------------------------------


def _parse_message(message: dict) -> object:
    provider = object.__new__(OpenAICompatProvider)
    return provider._parse({
        "choices": [{"finish_reason": "stop", "message": message}],
    })


def test_provider_still_reads_reasoning_content() -> None:
    response = _parse_message(
        {"role": "assistant", "content": "ok", "reasoning_content": "kimi thoughts"}
    )
    assert response.reasoning_content == "kimi thoughts"
    assert response.content == "ok"


def test_provider_reads_the_openrouter_reasoning_key() -> None:
    """The field name OpenRouter normalises to is `reasoning` (§3.2 appendix, item 1)."""
    response = _parse_message(
        {"role": "assistant", "content": "ok", "reasoning": "openrouter thoughts"}
    )
    assert response.reasoning_content == "openrouter thoughts"


def test_provider_reads_structured_reasoning_details() -> None:
    response = _parse_message({
        "role": "assistant",
        "content": "ok",
        "reasoning_details": [{"type": "reasoning.text", "text": "structured thinking"}],
    })
    assert response.reasoning_content is not None
    assert "structured thinking" in response.reasoning_content


def test_reasoning_content_wins_over_the_other_two_keys() -> None:
    response = _parse_message({
        "role": "assistant",
        "content": "ok",
        "reasoning_content": "primary",
        "reasoning": "secondary",
        "reasoning_details": [{"text": "tertiary"}],
    })
    assert response.reasoning_content == "primary"


def test_no_reasoning_anywhere_stays_none() -> None:
    response = _parse_message({"role": "assistant", "content": "ok"})
    assert response.reasoning_content is None


def test_captured_provider_reasoning_never_reaches_the_rendered_text() -> None:
    """§3.2: captured provider reasoning goes to audit_trace only, and the render source is still the PA's own field."""
    response = _parse_message(
        {"role": "assistant", "content": "ok", "reasoning": "private reasoning"}
    )
    rendered = render_rationale(
        "Show", RATIONALE, provider_reasoning_present=bool(response.reasoning_content)
    )
    assert "private reasoning" not in rendered.prefix
    assert rendered.prefix == RATIONALE
    assert rendered.reasoning["provider_reasoning_present"] is True
