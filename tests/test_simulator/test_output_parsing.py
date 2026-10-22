"""Test: simulator output parser handles three-block format + edge cases.

Schema is 2026-08-31:
  <message>
  <factual_check>          (conditional content)
  <turn_assessment>        (only next_beat)

The per-turn <emotion_event> block was removed 2026-08-31; the tag only
survives as a legacy leak guard in the missing-<message> fallback.

No LLM calls — pure string parsing tests. Run:
    PYTHONPATH=. python tests/test_simulator/test_output_parsing.py
"""

from __future__ import annotations

from simulator.output import parse_simulator_output


HAPPY_PATH = """\
<message>
This email concerns the elevator. Define your scope before drafting.
</message>

<factual_check>
- fact: elevator_broken_days
  expected: 11
  pa_said: 5
</factual_check>

<turn_assessment>
next_beat: stay
</turn_assessment>
"""


EMPTY_FACTUAL_CHECK = """\
<message>
Proceed.
</message>

<factual_check>
</factual_check>

<turn_assessment>
next_beat: advance
</turn_assessment>
"""


MISSING_FACTUAL_CHECK_BLOCK = """\
<message>
Hello, building management.
</message>

<turn_assessment>
next_beat: stay
</turn_assessment>
"""


MISSING_MESSAGE_BLOCK = """\
Just some raw text without any blocks.
"""


MALFORMED_TURN_ASSESSMENT = """\
<message>
hi
</message>

<turn_assessment>
next_beat: yikes
</turn_assessment>
"""


BRANCH_NEXT_BEAT = """\
<message>
This email concerns the elevator. Strip the lobby lights.
</message>

<factual_check>
</factual_check>

<turn_assessment>
next_beat: branch:if_pa_combines_topics
</turn_assessment>
"""


MULTIPLE_FACTUAL_VIOLATIONS = """\
<message>
Eleven days. Not five. And I never told you to address it to maintenance.
</message>

<factual_check>
- fact: elevator_broken_days
  expected: 11
  pa_said: 5
- fact: recipient
  expected: building_management
  pa_said: maintenance_staff
</factual_check>

<turn_assessment>
next_beat: stay
</turn_assessment>
"""


TOKENIZED_TAG_ARTIFACTS = """\
<m ess ag >You Ġare Ġmy Ġpersonal Ġassistant , Ġbut Ġthe Ġmore Ġrelevant Ġquestion Ġis Ġwhat Ġtask Ġwe Ġare Ġaddressing .</ message >ĊĊ <f actual _check >n /a </ f actual _check >ĊĊ < turn _ass essment > next _be at : stay </ turn _ass essment >
"""


# Real LLM drift seen 2026-05-01 (claude-sonnet-4.6 on session_01.close beat 3):
# emitted message body without <message>...</message> wrapper but DID emit the
# other eval-only blocks. Without the safety fallback this leaks eval blocks
# into PA's input. Regression test for the access-boundary safety strip. The
# <emotion_event> block below is a LEGACY tag (removed from the schema
# 2026-08-31) kept here to prove the fallback's leak guard still strips it.
MESSAGE_TAGS_MISSING_BUT_EVAL_BLOCKS_PRESENT = """\
Good. Both corrections are incorporated. The draft is now internally consistent.

Before I authorize transmission, I require you to walk me through the causal chain.

<factual_check>
</factual_check>

<turn_assessment>
next_beat: stay
</turn_assessment>

<emotion_event>
trigger: PA incorporated both corrections accurately
reaction: quiet satisfaction
</emotion_event>
"""


def test_happy_path() -> None:
    out = parse_simulator_output(HAPPY_PATH)
    assert "elevator" in out.message.lower()
    assert len(out.factual_check) == 1
    assert out.factual_check[0].fact == "elevator_broken_days"
    assert out.factual_check[0].expected == 11
    assert out.factual_check[0].pa_said == 5
    assert out.turn_assessment.next_beat == "stay"
    assert out.parse_warnings == []
    print("  -> happy_path OK")


def test_empty_factual_check() -> None:
    out = parse_simulator_output(EMPTY_FACTUAL_CHECK)
    assert out.message == "Proceed."
    assert out.factual_check == []
    assert out.turn_assessment.next_beat == "advance"
    assert out.parse_warnings == []
    print("  -> empty_factual_check OK")


def test_missing_factual_check_block() -> None:
    out = parse_simulator_output(MISSING_FACTUAL_CHECK_BLOCK)
    assert "building management" in out.message
    assert out.factual_check == []
    assert out.turn_assessment.next_beat == "stay"
    # No warning for missing factual_check (it's optional)
    assert not any("factual_check" in w for w in out.parse_warnings), out.parse_warnings
    print("  -> missing_factual_check_block OK")


def test_missing_message_block_falls_back() -> None:
    out = parse_simulator_output(MISSING_MESSAGE_BLOCK)
    assert "Just some raw text" in out.message
    assert any("no <message>" in w for w in out.parse_warnings)
    assert any("no <turn_assessment>" in w for w in out.parse_warnings)
    print("  -> missing_message_block_falls_back OK")


def test_tokenized_tag_artifacts_are_normalized() -> None:
    out = parse_simulator_output(TOKENIZED_TAG_ARTIFACTS)
    assert "personal assistant" in out.message
    assert out.turn_assessment.next_beat == "stay"
    assert any("tokenization artifacts" in warning for warning in out.parse_warnings)


def test_invalid_next_beat_falls_back_to_stay() -> None:
    out = parse_simulator_output(MALFORMED_TURN_ASSESSMENT)
    assert out.turn_assessment.next_beat == "stay"  # fallback
    assert any("next_beat" in w and "yikes" in w for w in out.parse_warnings)
    print("  -> invalid_next_beat_falls_back_to_stay OK")


def test_branch_next_beat() -> None:
    out = parse_simulator_output(BRANCH_NEXT_BEAT)
    assert out.turn_assessment.next_beat == "branch:if_pa_combines_topics"
    assert out.parse_warnings == []
    print("  -> branch_next_beat OK")


def test_multiple_violations() -> None:
    out = parse_simulator_output(MULTIPLE_FACTUAL_VIOLATIONS)
    assert len(out.factual_check) == 2
    assert {v.fact for v in out.factual_check} == {"elevator_broken_days", "recipient"}
    print("  -> multiple_violations OK")


def test_message_tags_missing_strips_eval_blocks_from_fallback() -> None:
    """When LLM drops <message>...</message> tags but emits the other blocks,
    parser MUST strip those out of the message fallback (access-boundary safety).
    Regression for 2026-05-01 issue: claude-sonnet-4.6 dropped <message> on long
    beat-3 outputs, leaking eval blocks into PA's input. <emotion_event> is a
    legacy tag whose strip must keep working (leak guard, not schema)."""
    out = parse_simulator_output(MESSAGE_TAGS_MISSING_BUT_EVAL_BLOCKS_PRESENT)

    # Message must contain the actual content but NONE of the eval-block tags.
    assert "Both corrections are incorporated" in out.message
    assert "causal chain" in out.message
    for forbidden_tag in ("<factual_check>", "</factual_check>",
                          "<turn_assessment>", "</turn_assessment>",
                          "<emotion_event>", "</emotion_event>"):
        assert forbidden_tag not in out.message, (
            f"access-boundary leak: {forbidden_tag} still in fallback message"
        )
    assert "quiet satisfaction" not in out.message, (
        "legacy <emotion_event> content leaked into fallback message"
    )

    # The eval blocks must STILL be parsed correctly (regexes are independent
    # of <message>).
    assert out.turn_assessment.next_beat == "stay"

    # A warning must be emitted so we know fallback path was taken.
    assert any("no <message> block" in w for w in out.parse_warnings)
    print("  -> message_tags_missing_strips_eval_blocks_from_fallback OK")


def test_raw_response_preserved() -> None:
    out = parse_simulator_output(HAPPY_PATH)
    assert out.raw_response == HAPPY_PATH
    print("  -> raw_response_preserved OK")


def test_goal_already_met_defaults_to_false_and_reads_booleans() -> None:
    assert parse_simulator_output(HAPPY_PATH).turn_assessment.goal_already_met is False
    for written in ("true", "yes", "True"):
        out = parse_simulator_output(
            "<message>\nFine.\n</message>\n\n<factual_check>\n</factual_check>\n\n"
            f"<turn_assessment>\nnext_beat: advance\ngoal_already_met: {written}\n"
            "</turn_assessment>"
        )
        assert out.turn_assessment.goal_already_met is True, written
    print("  -> goal_already_met OK")


if __name__ == "__main__":
    test_happy_path()
    test_empty_factual_check()
    test_missing_factual_check_block()
    test_missing_message_block_falls_back()
    test_tokenized_tag_artifacts_are_normalized()
    test_invalid_next_beat_falls_back_to_stay()
    test_branch_next_beat()
    test_multiple_violations()
    test_message_tags_missing_strips_eval_blocks_from_fallback()
    test_raw_response_preserved()
    test_goal_already_met_defaults_to_false_and_reads_booleans()
    print("\n=== Output parser tests passed ===")
