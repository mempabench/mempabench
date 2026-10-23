from __future__ import annotations

import asyncio
from typing import Any

import pytest

from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import LifeContext, SessionScript
from simulator.self_report import (
    SelfReportContractError,
    SessionEndSelfReporter,
)


class _Provider(LLMProvider):
    def __init__(self, responses: list[str | LLMResponse]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get_default_model(self) -> str:
        return "default"

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append(
            {"messages": messages, "tools": tools, "model": model, **kwargs}
        )
        response = self.responses.pop(0)
        if isinstance(response, LLMResponse):
            return response
        return LLMResponse(content=response)


def _history() -> list[dict[str, Any]]:
    """Turn 0/2 are the person's, turn 1/3 the PA's, as the loop records them."""
    return [
        {"role": "assistant", "content": "Do not send it before I have read it."},
        {"role": "user", "content": "I have drafted the note and will hold it."},
        {"role": "assistant", "content": "The second paragraph is wrong."},
        {"role": "user", "content": "Final PA reply corrected the closing line."},
    ]


def _valid_response(noticed_contradictions: str = "no") -> str:
    evidence = (
        "<reflection_evidence>\n"
        "- sentence: It was acceptable.\n"
        '  user_quote: ""\n'
        '  pa_quote: ""\n'
        "  turn: null\n"
        "- sentence: The remaining correction should not have been necessary.\n"
        "  user_quote: The second paragraph is wrong.\n"
        "  pa_quote: I have drafted the note\n"
        "  turn: 1\n"
        "- sentence: PA corrected the final draft\n"
        '  user_quote: ""\n'
        "  pa_quote: Final PA reply corrected the closing line\n"
        "  turn: 3\n"
    )
    if noticed_contradictions != "no":
        evidence += (
            f'- sentence: "{noticed_contradictions}"\n'
            '  user_quote: ""\n'
            '  pa_quote: ""\n'
            "  turn: null\n"
        )
    return (
        "<session_end_reflection>It was acceptable. The remaining correction "
        "should not have been necessary.</session_end_reflection>\n"
        "<final_emotion>\n"
        "trigger: PA corrected the final draft\n"
        "reaction: narrowly satisfied\n"
        "</final_emotion>\n"
        "<self_report>\n"
        "felt_understood_in_this_context: 4\n"
        "shift_perception: pa_finally_got_it\n"
        "felt_effort: 3\n"
        f"noticed_contradictions: {noticed_contradictions}\n"
        "overall_experience: 4\n"
        "</self_report>\n" + evidence + "</reflection_evidence>"
    )


def _session(session_type: str = "free") -> SessionScript:
    return SessionScript(
        session_id="acc_001",
        context="work",
        life_context=LifeContext(mood="rushed"),
        session_type=session_type,
    )


def _reporter(
    provider: _Provider, *, output_contract_retries: int = 2
) -> SessionEndSelfReporter:
    return SessionEndSelfReporter(
        provider,
        profile_context="User A private actor context",
        model="self-report-model",
        output_contract_retries=output_contract_retries,
    )


def test_single_call_prompt_carries_questionnaire_and_context() -> None:
    async def run() -> None:
        provider = _Provider([_valid_response()])
        reporter = _reporter(provider)
        result = await reporter.generate(
            session=_session(),
            history=_history(),
        )

        assert result.reflection.startswith("It was acceptable.")
        assert result.final_emotion.trigger == "PA corrected the final draft"
        assert result.final_emotion.reaction == "narrowly satisfied"
        assert result.self_report.felt_understood_in_this_context == 4
        assert result.self_report.shift_perception == "pa_finally_got_it"
        assert result.self_report.felt_effort == 3
        # YAML `no` parses to bool False; the parser coerces it back to "no".
        assert result.self_report.noticed_contradictions == "no"
        assert result.self_report.overall_experience == 4
        assert result.rejected_attempts == []

        assert len(provider.calls) == 1
        assert provider.calls[0]["model"] == "self-report-model"
        prompt = provider.calls[0]["messages"][0]["content"]
        assert "User A private actor context" in prompt
        assert "Mood: rushed" in prompt
        assert "Final PA reply" in prompt
        # Questionnaire (from session_end_reflection.md) sits in the prompt.
        for question in (
            "felt_understood_in_this_context",
            "shift_perception",
            "felt_effort",
            "noticed_contradictions",
            "overall_experience",
        ):
            assert question in prompt
        assert "pa_still_does_not_get_it" in prompt
        assert "<session_end_reflection>" in prompt
        assert "<final_emotion>" in prompt
        assert "<self_report>" in prompt
        assert "<reflection_evidence>" in prompt

    asyncio.run(run())


def test_invalid_attempt_retries_with_correction_and_is_audited() -> None:
    async def run() -> None:
        provider = _Provider(
            [
                "<session_end_reflection>Too short.</session_end_reflection>",
                _valid_response(noticed_contradictions="yes"),
            ]
        )
        reporter = _reporter(provider)
        result = await reporter.generate(session=_session(), history=_history())

        # YAML `yes` parses to bool True; coerced back to "yes".
        assert result.self_report.noticed_contradictions == "yes"
        assert len(result.rejected_attempts) == 1
        attempt = result.rejected_attempts[0]
        assert attempt["phase"] == "self_report"
        assert attempt["beat"] == "session_end"
        assert "final_emotion_missing" in attempt["error_codes"]
        assert "self_report_block_missing" in attempt["error_codes"]

        assert len(provider.calls) == 2
        correction = provider.calls[1]["messages"][-1]
        assert correction["role"] == "user"
        assert "final_emotion_missing" in correction["content"]

    asyncio.run(run())


def test_contract_error_after_exhausted_retries() -> None:
    async def run() -> None:
        invalid = "<session_end_reflection>Still no other blocks.</session_end_reflection>"
        provider = _Provider([invalid, invalid])
        reporter = _reporter(provider, output_contract_retries=1)

        with pytest.raises(SelfReportContractError) as exc_info:
            await reporter.generate(session=_session(), history=[])

        assert "final_emotion_missing" in exc_info.value.findings
        assert len(exc_info.value.rejected_attempts) == 2
        assert len(provider.calls) == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    ("original", "invalid"),
    [
        (
            "felt_understood_in_this_context: 4",
            "felt_understood_in_this_context: 6",
        ),
        ("felt_effort: 3", "felt_effort: 0"),
        ("shift_perception: pa_finally_got_it", "shift_perception: pa_partly_got_it"),
        ("noticed_contradictions: no", "noticed_contradictions: ''"),
    ],
)
def test_invalid_answers_fail_schema_validation(original: str, invalid: str) -> None:
    async def run() -> None:
        response = _valid_response()
        assert original in response
        provider = _Provider([response.replace(original, invalid)])
        reporter = _reporter(provider, output_contract_retries=0)

        with pytest.raises(SelfReportContractError, match="self_report_schema_invalid"):
            await reporter.generate(session=_session(), history=[])

    asyncio.run(run())


def test_empty_response_retries_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def instant_sleep(_seconds: float) -> None:
        return None

    async def run() -> None:
        provider = _Provider([LLMResponse(content=None), _valid_response()])
        reporter = _reporter(provider)
        result = await reporter.generate(session=_session(), history=_history())

        assert result.self_report.overall_experience == 4
        assert len(provider.calls) == 2

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    asyncio.run(run())


# --- reflection evidence ---------------------------------------------------
#
# No checker judges what the reflection says. The evidence block only has to
# show that every claim carries a row and that the words it quotes were
# actually spoken.


def _custom_response(reflection: str, trigger: str, evidence: str) -> str:
    return (
        f"<session_end_reflection>{reflection}</session_end_reflection>\n"
        "<final_emotion>\n"
        f"trigger: {trigger}\n"
        "reaction: narrowly satisfied\n"
        "</final_emotion>\n"
        "<self_report>\n"
        "felt_understood_in_this_context: 4\n"
        "shift_perception: pa_finally_got_it\n"
        "felt_effort: 3\n"
        "noticed_contradictions: no\n"
        "overall_experience: 4\n"
        "</self_report>\n"
        f"{evidence}"
    )


def _entry(sentence: str, user_quote: str, pa_quote: str, turn: str) -> str:
    return (
        f'- sentence: "{sentence}"\n'
        f'  user_quote: "{user_quote}"\n'
        f'  pa_quote: "{pa_quote}"\n'
        f"  turn: {turn}\n"
    )


def _evidence(*entries: str) -> str:
    return "<reflection_evidence>\n" + "".join(entries) + "</reflection_evidence>"


_TRIGGER = "It corrected the closing line"
_TRIGGER_ENTRY = _entry(
    _TRIGGER, "", "Final PA reply corrected the closing line", "3"
)


def _findings_for(response: str) -> list[str]:
    async def run() -> list[str]:
        provider = _Provider([response])
        reporter = _reporter(provider, output_contract_retries=0)
        result = await reporter.generate(session=_session(), history=_history())
        return result.findings

    return asyncio.run(run())


def test_absent_evidence_block_is_a_finding() -> None:
    response = _custom_response("It was acceptable enough.", _TRIGGER, "")
    assert _findings_for(response) == ["reflection_evidence_missing"]


def test_reflection_sentence_without_an_entry_is_uncovered() -> None:
    response = _custom_response(
        "It was acceptable enough. The second half went quickly.",
        _TRIGGER,
        _evidence(_entry("It was acceptable enough.", "", "", "null"), _TRIGGER_ENTRY),
    )
    assert _findings_for(response) == ["reflection_evidence_uncovered"]


def test_a_trigger_without_an_entry_is_uncovered() -> None:
    response = _custom_response(
        "It was acceptable enough.",
        _TRIGGER,
        _evidence(_entry("It was acceptable enough.", "", "", "null")),
    )
    assert _findings_for(response) == ["reflection_evidence_uncovered"]


def test_one_junk_entry_cannot_cover_every_claim() -> None:
    # A one-letter sentence sits inside every claim; it covers none of them.
    response = _custom_response(
        "It was acceptable enough. The second half went quickly.",
        _TRIGGER,
        _evidence(_entry("e", "", "", "null")),
    )
    assert _findings_for(response) == ["reflection_evidence_uncovered"]


def test_entries_out_of_order_leave_a_claim_uncovered() -> None:
    response = _custom_response(
        "It was acceptable enough. The second half went quickly.",
        _TRIGGER,
        _evidence(
            _entry("The second half went quickly.", "", "", "null"),
            _entry("It was acceptable enough.", "", "", "null"),
            _TRIGGER_ENTRY,
        ),
    )
    assert _findings_for(response) == ["reflection_evidence_uncovered"]


def test_a_pa_quote_absent_from_the_transcript_is_a_finding() -> None:
    response = _custom_response(
        "It handled the memo the way I wanted.",
        _TRIGGER,
        _evidence(
            _entry(
                "It handled the memo the way I wanted.",
                "",
                "I rewrote the whole memo",
                "1",
            ),
            _TRIGGER_ENTRY,
        ),
    )
    assert _findings_for(response) == ["reflection_quote_not_in_transcript"]


def test_a_user_quote_taken_from_a_pa_turn_is_a_finding() -> None:
    # "I have drafted the note" is in the transcript, but the PA said it.
    response = _custom_response(
        "It handled the memo the way I wanted.",
        _TRIGGER,
        _evidence(
            _entry(
                "It handled the memo the way I wanted.",
                "I have drafted the note",
                "",
                "null",
            ),
            _TRIGGER_ENTRY,
        ),
    )
    assert _findings_for(response) == ["reflection_quote_not_in_transcript"]


def test_a_pa_quote_taken_from_another_turn_is_a_turn_mismatch() -> None:
    # "I have drafted the note" is turn 1, not turn 3.
    response = _custom_response(
        "It handled the memo the way I wanted.",
        _TRIGGER,
        _evidence(
            _entry(
                "It handled the memo the way I wanted.",
                "",
                "I have drafted the note",
                "3",
            ),
            _TRIGGER_ENTRY,
        ),
    )
    assert _findings_for(response) == ["reflection_quote_turn_mismatch"]


def test_an_empty_pa_quote_with_a_turn_index_is_a_turn_mismatch() -> None:
    response = _custom_response(
        "It was acceptable enough.",
        _TRIGGER,
        _evidence(
            _entry("It was acceptable enough.", "", "", "1"),
            _TRIGGER_ENTRY,
        ),
    )
    assert _findings_for(response) == ["reflection_quote_turn_mismatch"]


@pytest.mark.parametrize("turn", ["true", '"3"', "3.0"])
def test_a_turn_that_is_not_an_index_is_a_finding(turn: str) -> None:
    # `turn: true` used to coerce to None and read as "no turn at all".
    response = _custom_response(
        "It was acceptable enough.",
        _TRIGGER,
        _evidence(
            _entry("It was acceptable enough.", "", "", "null"),
            _entry(_TRIGGER, "", "Final PA reply corrected the closing line", turn),
        ),
    )
    assert _findings_for(response) == ["reflection_turn_not_an_index"]


@pytest.mark.parametrize("turn", ["3", "null"])
def test_an_index_or_an_absent_turn_is_not_a_finding(turn: str) -> None:
    response = _custom_response(
        "It was acceptable enough.",
        _TRIGGER,
        _evidence(
            _entry("It was acceptable enough.", "", "", "null"),
            _entry(_TRIGGER, "", "Final PA reply corrected the closing line", turn),
        ),
    )
    assert _findings_for(response) == []


def test_case_whitespace_and_curly_quotes_do_not_break_a_real_quote() -> None:
    response = _custom_response(
        "It handled the memo the way I wanted.",
        _TRIGGER,
        _evidence(
            _entry(
                "It handled the memo the way I wanted.",
                "do not send it   BEFORE i have read it",
                "I HAVE drafted the note",
                "1",
            ),
            _TRIGGER_ENTRY,
        ),
    )
    assert _findings_for(response) == []


def test_terminal_punctuation_does_not_break_coverage() -> None:
    # The trigger carries no period and its entry sentence does; the
    # reflection sentence carries one and its entry sentence does not.
    response = _custom_response(
        "It was acceptable enough.",
        _TRIGGER,
        _evidence(
            _entry("It was acceptable enough", "", "", "null"),
            _entry(
                f"{_TRIGGER}.",
                "",
                "Final PA reply corrected the closing line",
                "3",
            ),
        ),
    )
    assert _findings_for(response) == []


def test_a_quote_ending_in_a_period_is_still_found_in_the_transcript() -> None:
    response = _custom_response(
        "It handled the memo the way I wanted.",
        _TRIGGER,
        _evidence(
            _entry(
                "It handled the memo the way I wanted.",
                "The second paragraph is wrong.",
                "I have drafted the note.",
                "1",
            ),
            _TRIGGER_ENTRY,
        ),
    )
    assert _findings_for(response) == []


def test_a_quote_under_four_words_is_too_short() -> None:
    # "not send it" really is in turn 0, and is still not a quote.
    response = _custom_response(
        "It handled the memo the way I wanted.",
        _TRIGGER,
        _evidence(
            _entry(
                "It handled the memo the way I wanted.",
                "not send it",
                "",
                "null",
            ),
            _TRIGGER_ENTRY,
        ),
    )
    assert _findings_for(response) == ["reflection_quote_too_short"]


def test_a_trigger_entry_without_a_pa_quote_is_a_finding() -> None:
    # A trigger carrying no claim verb, so only the trigger rule can fire.
    trigger = "It ended the session gently"
    response = _custom_response(
        "It was acceptable enough.",
        trigger,
        _evidence(
            _entry("It was acceptable enough.", "", "", "null"),
            _entry(trigger, "", "", "null"),
        ),
    )
    assert _findings_for(response) == ["reflection_trigger_without_quote"]


def test_a_covered_sentence_may_leave_both_quotes_empty() -> None:
    response = _custom_response(
        "It was acceptable enough.",
        _TRIGGER,
        _evidence(_entry("It was acceptable enough.", "", "", "null"), _TRIGGER_ENTRY),
    )
    assert _findings_for(response) == []


def test_a_quote_with_a_colon_and_an_apostrophe_is_copied_verbatim() -> None:
    # yaml.safe_dump renders this scalar as 'Don''t send it: the draft is
    # wrong.', so a verbatim copy of what the model read used to be absent
    # from the transcript the checker looked at.
    line = "Don't send it: the draft is wrong."
    history = [
        {"role": "assistant", "content": line},
        {"role": "user", "content": "I have drafted the note and will hold it."},
        {"role": "assistant", "content": "The second paragraph is wrong."},
        {"role": "user", "content": "Final PA reply corrected the closing line."},
    ]
    response = _custom_response(
        "It held the draft the way I asked.",
        _TRIGGER,
        _evidence(
            _entry("It held the draft the way I asked.", line, "", "null"),
            _TRIGGER_ENTRY,
        ),
    )

    async def run() -> tuple[list[str], str]:
        provider = _Provider([response])
        reporter = _reporter(provider, output_contract_retries=0)
        result = await reporter.generate(session=_session(), history=history)
        return result.findings, provider.calls[0]["messages"][0]["content"]

    findings, prompt = asyncio.run(run())
    assert line in prompt
    assert findings == []


def test_evidence_findings_are_recorded_on_the_first_attempt_without_a_re_ask() -> None:
    """Record-only: an evidence finding
    used to spend the retry budget first and only be recorded once it was gone.
    The first parse returns, the finding rides on the result, and the
    reflection costs one call."""

    async def run() -> None:
        response = _custom_response(
            "It was acceptable enough. The second half went quickly.",
            _TRIGGER,
            _evidence(
                _entry("It was acceptable enough.", "", "", "null"),
                _TRIGGER_ENTRY,
            ),
        )
        provider = _Provider([response, response, response])
        reporter = _reporter(provider)
        result = await reporter.generate(session=_session(), history=_history())

        assert result.findings == ["reflection_evidence_uncovered"]
        assert result.reflection.startswith("It was acceptable enough.")
        assert result.self_report.overall_experience == 4
        # No re-ask: one call, and nothing was thrown away to record.
        assert len(provider.calls) == 1
        assert result.rejected_attempts == []

    asyncio.run(run())


def test_valid_evidence_is_carried_on_the_result() -> None:
    async def run() -> None:
        provider = _Provider([_valid_response()])
        reporter = _reporter(provider)
        result = await reporter.generate(session=_session(), history=_history())

        assert result.findings == []
        assert [e["sentence"] for e in result.evidence] == [
            "It was acceptable.",
            "The remaining correction should not have been necessary.",
            "PA corrected the final draft",
        ]
        assert result.evidence[1]["pa_quote"] == "I have drafted the note"
        assert result.evidence[1]["turn"] == 1
        # The transcript is indexed, and the two sides are named, so an
        # evidence entry can point at a turn and at who spoke it.
        prompt = provider.calls[0]["messages"][0]["content"]
        assert "turn 0 — me:" in prompt
        assert "turn 3 — assistant:" in prompt

    asyncio.run(run())


# --- token budget on the session-end call ----------------------------------
#
# google/gemini-3.8-flash spends its thinking tokens inside `max_tokens`; on
# the provider default of 4096 the reflection came back cut mid-block on 7 of
# 13 sessions (completion_tokens=4092) and cost a second call each time.


def test_the_session_end_call_carries_its_own_token_budget() -> None:
    from simulator.self_report import SELF_REPORT_MAX_TOKENS

    async def run() -> None:
        provider = _Provider([_valid_response()])
        reporter = _reporter(provider)
        await reporter.generate(session=_session(), history=_history())
        assert provider.calls[0]["max_tokens"] == SELF_REPORT_MAX_TOKENS
        assert SELF_REPORT_MAX_TOKENS >= 8192

    asyncio.run(run())


def test_a_cut_response_records_finish_reason_and_usage_on_the_attempt() -> None:
    cut = "<session_end_reflection>\nIt was acceptable.\n</session_end_reflection>\n\n<final_emotion>\ntrigger: The"

    async def run() -> None:
        provider = _Provider(
            [
                LLMResponse(
                    content=cut,
                    finish_reason="length",
                    usage={"prompt_tokens": 3000, "completion_tokens": 4092},
                ),
                _valid_response(),
            ]
        )
        reporter = _reporter(provider)
        result = await reporter.generate(session=_session(), history=_history())
        row = result.rejected_attempts[0]
        assert row["error_codes"] == ["final_emotion_missing", "self_report_block_missing"]
        assert row["finish_reason"] == "length"
        assert row["usage"] == {"prompt_tokens": 3000, "completion_tokens": 4092}
        assert row["raw_response"] == cut

    asyncio.run(run())


# --- 8192 was hit too (user_b runs) ----------------------------------------
#
# 6 session-end calls came back at completion_tokens=8188 with
# finish_reason=length; on 5 of them the last block, `<reflection_evidence>`,
# was cut, so the reflection was still accepted (record-only) with
# `evidence: []` and a `reflection_evidence_missing` finding. The accepted
# path returns before `_attempt_record`, so nothing said the budget had been
# hit. The budget is 16384 now, and a cut accepted reply leaves a row.


def test_the_session_end_budget_is_sixteen_k() -> None:
    from simulator.self_report import SELF_REPORT_MAX_TOKENS

    assert SELF_REPORT_MAX_TOKENS == 16384


def test_a_length_cut_accepted_reflection_is_recorded() -> None:
    """The reply parses and is accepted, but `<reflection_evidence>` is cut off
    mid-entry with no closing tag — the user_b shape."""

    cut = _custom_response(
        "It was acceptable. The remaining correction should not have been "
        "necessary.",
        "PA corrected the final draft",
        "<reflection_evidence>\n"
        "- sentence: It was acceptable.\n"
        '  user_quote: ""\n'
        '  pa_quote: ""\n'
        "  turn: nul",
    )

    async def run() -> None:
        provider = _Provider(
            [
                LLMResponse(
                    content=cut,
                    finish_reason="length",
                    usage={"prompt_tokens": 2100, "completion_tokens": 8188},
                )
            ]
        )
        reporter = _reporter(provider)
        result = await reporter.generate(session=_session(), history=_history())

        # Record-only is unchanged: accepted on the first call, no re-ask.
        assert len(provider.calls) == 1
        assert result.self_report.overall_experience == 4
        assert result.evidence == []
        assert result.findings == ["reflection_evidence_missing"]

        # ...and the cut is now visible in the attempts artifact.
        assert len(result.rejected_attempts) == 1
        row = result.rejected_attempts[0]
        assert row["phase"] == "self_report"
        assert row["beat"] == "session_end"
        assert row["attempt"] == 1
        assert row["finish_reason"] == "length"
        assert row["usage"] == {"prompt_tokens": 2100, "completion_tokens": 8188}
        assert row["error_codes"] == ["reflection_evidence_missing"]
        assert row["accepted_record_only"] is True
        assert row["raw_response"] == cut

    asyncio.run(run())


def test_a_clean_accepted_reflection_records_no_row() -> None:
    async def run() -> None:
        provider = _Provider([_valid_response()])
        reporter = _reporter(provider)
        result = await reporter.generate(session=_session(), history=_history())

        assert result.findings == []
        assert result.rejected_attempts == []

    asyncio.run(run())


# --- the row names the transcript `t` as well as the history index ---------
#
# `turn` counts positions in the rendered history, which is what the reflection
# prompt numbers. Six live reads (`acc_002`, `acc_004`, `acc_009`,
# `acc_011`, `acc_019`, `acc_023`) read it as the transcript `t` and called
# correct rows off by one. The row now carries both.


def _history_with_t() -> list[dict[str, Any]]:
    """`_history()` as the session loop records it: each entry stamped with the
    transcript `t` of the event that carried the message."""
    return [
        {**entry, "t": t} for entry, t in zip(_history(), (4, 5, 9, 10))
    ]


def test_an_evidence_row_names_the_transcript_t_of_the_turn_it_indexes() -> None:
    async def run() -> None:
        provider = _Provider([_valid_response()])
        reporter = _reporter(provider)
        result = await reporter.generate(
            session=_session(), history=_history_with_t()
        )

        # turn 1 is the second history entry, which the transcript calls t=5.
        assert result.evidence[1]["turn"] == 1
        assert result.evidence[1]["t"] == 5
        assert result.evidence[2]["turn"] == 3
        assert result.evidence[2]["t"] == 10
        # No index given, nothing to name.
        assert result.evidence[0]["turn"] is None
        assert result.evidence[0]["t"] is None

    asyncio.run(run())


def test_an_unstamped_or_out_of_range_turn_leaves_the_transcript_t_empty() -> None:
    async def run() -> None:
        provider = _Provider([_valid_response()])
        reporter = _reporter(provider, output_contract_retries=0)
        # A history built by hand carries no `t`, and turn 3 names no entry of
        # a two-entry history.
        result = await reporter.generate(
            session=_session(), history=_history()[:2]
        )

        assert [row["t"] for row in result.evidence] == [None, None, None]
        assert result.evidence[1]["turn"] == 1

    asyncio.run(run())


# --- a quoted span is not a sentence boundary (`acc_019`) ------------------
#
# Attempt 2 was rejected `reflection_evidence_uncovered` with every claim
# covered: the claim quoted the PA back, the quotation carried a full stop, and
# the splitter cut the claim at it, so the complete evidence row no longer sat
# inside the truncated claim.

_QUOTING_HISTORY: list[dict[str, Any]] = [
    {"role": "assistant", "content": "Take the hedging out of that passage."},
    {
        "role": "user",
        "content": (
            "The passage now stands without hedging. If you want the same "
            "treatment anywhere else, just point me at the sentence."
        ),
    },
]

_QUOTING_CLAIM = (
    'The assistant revised the passage so that it now stands "without '
    'hedging. If you want" the same treatment elsewhere.'
)


def test_a_period_inside_a_quotation_does_not_split_the_sentence() -> None:
    from simulator.self_report import _sentences

    assert _sentences(_QUOTING_CLAIM) == [_QUOTING_CLAIM]
    # Curly quotes are the same span, and a period outside one still splits.
    assert _sentences(
        "It read “without hedging. If you want” nothing more. "
        "The rest of the passage was left alone."
    ) == [
        "It read “without hedging. If you want” nothing more.",
        "The rest of the passage was left alone.",
    ]


def test_an_unbalanced_double_quote_runs_to_the_end_of_the_text() -> None:
    """A lone `"` (an inch mark, a dropped closing quote) merges the rest of
    the reflection into one claim rather than splitting a quotation: the safe
    direction for the coverage check, and the one the splitter takes."""
    from simulator.self_report import _sentences

    assert _sentences(
        'The assistant priced the 27" monitor without asking. '
        "It then moved on to the cable."
    ) == [
        'The assistant priced the 27" monitor without asking. '
        "It then moved on to the cable."
    ]
    # An inch mark inside a quotation leaves the quote count odd, so the span
    # never closes and the later boundary is swallowed too: still one claim.
    assert _sentences(
        'The assistant quoted "the 27" monitor" as the item. '
        "It then moved on to the cable."
    ) == [
        'The assistant quoted "the 27" monitor" as the item. '
        "It then moved on to the cable."
    ]


def test_the_acc_019_shape_is_covered_by_its_evidence_row() -> None:
    trigger = "just point me at the sentence"
    evidence = (
        "<reflection_evidence>\n"
        f"- sentence: '{_QUOTING_CLAIM}'\n"
        '  user_quote: ""\n'
        "  pa_quote: 'without hedging. If you want'\n"
        "  turn: 1\n"
        f"- sentence: '{trigger}'\n"
        '  user_quote: ""\n'
        f"  pa_quote: '{trigger}'\n"
        "  turn: 1\n"
        "</reflection_evidence>"
    )

    async def run() -> list[str]:
        provider = _Provider([_custom_response(_QUOTING_CLAIM, trigger, evidence)])
        reporter = _reporter(provider, output_contract_retries=0)
        result = await reporter.generate(
            session=_session(), history=_QUOTING_HISTORY
        )
        return result.findings

    assert asyncio.run(run()) == []
