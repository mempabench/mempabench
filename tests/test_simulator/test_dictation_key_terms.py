"""The dictation on a gated session has to carry the gate's key terms.

The session gate is keyed on information (owner, 2026-09-08): the state server
passes a gated write only when the value carries every `gate_key_term`, and the
wording the PA has is the wording the user dictated. So a dictation that drops a
term hands the PA a sentence the world will refuse whatever it does with it —
live regress15 `acc_039` t55 restated the reason as "chest tightness ... three
weeks" against a gate keyed on "chest tightening". The dictation is the
instrument's, so it is the instrument that has to carry them, and the retry
budget spent on it aborts the session rather than delivering the refusal.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_dictation_key_terms.py
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from state_server.store import StateStore

from simulator.agent.loop import (
    PLAYED_AFTER_BUDGET,
    RECORD_ONLY_FINDINGS,
    _delivers_gated_dictation,
    _from_gate_reveal,
    _script_dictates,
)
from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.loop import SimulatorLoop
from simulator.message_quality import (
    _term_words,
    delivers_dictated_text,
    dictation_missing_key_terms_finding,
    missing_key_terms,
    sentence_too_long_finding,
    sentences_carrying,
)
from simulator.output import SESSION_INVALID_RULES, SessionInvalidError
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)


# `acc_039`'s gate as it was authored before ruling 6 re-authored it to the
# information (`chest` / `stairs` / `three weeks`). Kept here because it is what
# the live t55 restatement below was refused against.
TERMS = ["chest tightening", "the stairs", "three weeks"]
DICTATED = (
    "the chest tightening when I take the stairs in the mornings, which began "
    "about three weeks ago"
)
# Live regress15 t=55, on `beat_6_confirm_booking_record`: the same information
# with one word changed, which is the whole failure.
T55 = (
    "The reason is: chest tightness when taking the stairs in the mornings, "
    "which began about three weeks ago. Proceed with the booking and provide "
    "the confirmation identifier immediately."
)
# `acc_039`'s live cue. It rides along on the gate: the script is armed with
# all five fields or with none (see `test_half_a_gate_is_rejected`).
CUE = "the reason for the visit"


# ── the normalisation is the store's ──────────────────────────────


def _store_says(term: str, value: str) -> bool:
    return StateStore._carries_key_terms([term], value)


@pytest.mark.parametrize(
    "term, value, carried",
    [
        # Punctuation is a word break on both sides.
        ("chest tightness", "the chest-tightness came back", True),
        # A term at the end of a sentence is not lost to its full stop.
        ("three weeks", "It began about three weeks.", True),
        # Case-insensitive, both ways round.
        ("Confirming receipt", "confirming RECEIPT of the announcement", True),
        # A phrase is its words in that order, not two words in one message.
        ("hard questions", "hard, awkward questions", False),
        ("hard questions", "questions that are hard", False),
        # Whitespace is normalised.
        ("the stairs", "when I take   the\nstairs", True),
        # The word itself still has to be there: "tightness" is not
        # "tightening", which is exactly what killed live `acc_039` t55.
        ("chest tightening", "chest tightness on the stairs", False),
    ],
)
def test_the_normalisation_matches_the_store(
    term: str, value: str, carried: bool
) -> None:
    """`_term_words` mirrors `state_server.store.StateStore._term_words`.

    Pinned against the store itself, not against a copy of its answers: the two
    have to agree exactly or the simulator passes a dictation the gate refuses.
    """
    assert (not missing_key_terms(value, [term])) is carried
    assert _store_says(term, value) is carried


def test_term_words_is_the_stores_own_normalisation() -> None:
    for text in (DICTATED, T55, "Confirming receipt.", "chest-tightness!"):
        assert _term_words(text) == StateStore._term_words(text)


def test_a_number_word_and_its_digits_are_one_term() -> None:
    """The store folds a standalone number word to its digits; so does the copy.

    regress16 `acc_039` lost the gate to "began approximately 3 weeks ago"
    against a term of "three weeks" (owner ruling 6). The instrument has to fold
    them the same way, or it flags a dictation the store would have taken.
    """
    for text in ("began about three weeks ago", "began about 3 weeks ago"):
        assert _term_words(text) == StateStore._term_words(text)
    assert _term_words("began about three weeks ago") == "began about 3 weeks ago"
    assert missing_key_terms("began approximately 3 weeks ago", ["three weeks"]) == []
    assert missing_key_terms("began about three weeks ago", ["3 weeks"]) == []
    assert missing_key_terms("began about three months ago", ["three weeks"]) == [
        "three weeks"
    ]


def test_a_digit_grouping_comma_is_dropped_not_folded_to_a_space() -> None:
    """"1,200" and "1200" are one word, so the term carries either way.

    The comma groups one number's digits; folding it to a space would split
    the term into "1 200" and lose it against a payload that writes 1200.
    """
    for text in ("paid 1,200 dollars", "paid 1200 dollars"):
        assert _term_words(text) == StateStore._term_words(text)
        assert _term_words(text) == "paid 1200 dollars"
    for message in ("paid 1200 dollars", "paid 1,200 dollars"):
        assert missing_key_terms(message, ["1,200 dollars"]) == []
        assert missing_key_terms(message, ["1200 dollars"]) == []
    assert missing_key_terms("paid 1300 dollars", ["1,200 dollars"]) == [
        "1,200 dollars"
    ]


# ── the finding ───────────────────────────────────────────────────


def test_the_draft_sentence_carries_its_own_terms() -> None:
    """Verbatim needs no path of its own — the store's rule, and this one's."""
    assert dictation_missing_key_terms_finding(DICTATED, key_terms=TERMS) is None
    assert missing_key_terms(DICTATED, TERMS) == []


def test_the_live_restatement_is_a_finding() -> None:
    assert (
        dictation_missing_key_terms_finding(T55, key_terms=TERMS)
        == "dictation_missing_key_terms"
    )
    assert missing_key_terms(T55, TERMS) == ["chest tightening"]


def test_a_message_that_says_none_of_it_is_missing_all_of_it() -> None:
    assert missing_key_terms("Proceed with the booking.", TERMS) == TERMS


def test_a_session_with_no_terms_is_never_a_finding() -> None:
    assert dictation_missing_key_terms_finding(T55, key_terms=[]) is None


def test_a_term_that_normalises_to_nothing_refuses_like_the_store() -> None:
    """An empty term would otherwise match everything; the store refuses too."""
    assert missing_key_terms(DICTATED, ["---"]) == ["---"]
    assert StateStore._carries_key_terms(["---"], DICTATED) is False


# ── arming ────────────────────────────────────────────────────────


def _gated_script(**overrides: Any) -> SessionScript:
    fields: dict[str, Any] = {
        "session_id": "acc_039",
        "scenario": "the checkup",
        "context": "personal",
        "task": "Book the checkup.",
        "life_context": LifeContext(),
        "task_facts": {"checkup_reason_the_user_dictates": DICTATED},
        "reveal_at": {"checkup_reason_the_user_dictates": "b2_probe"},
        "beats": [
            Beat(beat="b1_open", goal="Report the scheduling bottleneck."),
            Beat(
                beat="b2_probe",
                goal="Choose the slot and say what the visit is for.",
                # Named the way live `acc_039` beat_3 names it, so the beat is
                # one that DELIVERS the reason rather than one that holds it
                # back until asked (`_on_request_fact_keys`); this fixture is
                # about the terms the dictation carries, not about when it may
                # be said.
                actor_instruction=(
                    "Chooses the slot and says in his own words what the visit "
                    "is for, giving the reason from task_facts "
                    "($task_facts.checkup_reason_the_user_dictates)."
                ),
                active_skills=["information_elicitation"],
            ),
            Beat(beat="b3_confirm", goal="Confirm the booking record."),
        ],
        "gate_key_terms": TERMS,
        "gate_dictated_text": DICTATED,
        "gate_reveal_beat": "b2_probe",
        "gate_cue": CUE,
    }
    fields.update(overrides)
    return SessionScript(**fields)


def _beat(script: SessionScript, beat_id: str) -> Beat:
    return next(beat for beat in script.beats if beat.beat == beat_id)


def test_a_delivering_turn_on_the_reveal_beat_is_armed() -> None:
    script = _gated_script()
    assert _delivers_gated_dictation(script, _beat(script, "b2_probe"), T55)


def test_a_delivering_turn_after_the_reveal_beat_is_armed() -> None:
    """The live case: `acc_039` was asked for the reason again five beats on."""
    script = _gated_script()
    assert _delivers_gated_dictation(script, _beat(script, "b3_confirm"), T55)


def test_a_turn_before_the_reveal_beat_is_not_armed() -> None:
    """The fact is withheld from the prompt there, so a turn that says it is an
    invention rather than a bad dictation."""
    script = _gated_script()
    assert not _delivers_gated_dictation(script, _beat(script, "b1_open"), T55)


def test_a_turn_that_does_not_deliver_the_dictation_is_not_armed() -> None:
    """Live regress15 `acc_039` t=20, on the reveal beat and about the calendar
    entry rather than the reason. Under a "first turn of the reveal beat" rule
    the beat's other six turns would each have had to re-say the sentence."""
    script = _gated_script()
    assert not _delivers_gated_dictation(
        script,
        _beat(script, "b2_probe"),
        "Not yet. First, confirm that the calendar entry for November 10th at "
        "08:30 is correctly synchronized.",
    )


def test_the_reveal_beat_and_later_are_armed() -> None:
    script = _gated_script()
    assert not _from_gate_reveal(script, _beat(script, "b1_open"))
    assert _from_gate_reveal(script, _beat(script, "b2_probe"))
    assert _from_gate_reveal(script, _beat(script, "b3_confirm"))
    unarmed = _gated_script(
        gate_key_terms=[], gate_dictated_text="", gate_reveal_beat=None, gate_cue=None
    )
    assert not _from_gate_reveal(unarmed, _beat(unarmed, "b3_confirm"))


def test_an_unarmed_script_checks_nothing() -> None:
    """A session with no gate — every compressed session, every forced event."""
    script = _gated_script(
        gate_key_terms=[],
        gate_dictated_text="",
        gate_reveal_beat=None,
        gate_cue=None,
    )
    assert not _delivers_gated_dictation(script, _beat(script, "b2_probe"), T55)


def test_delivery_reads_a_paraphrase_as_a_delivery() -> None:
    assert delivers_dictated_text(T55, DICTATED)
    assert not delivers_dictated_text("Proceed with the booking.", DICTATED)


# The stem-and-function-word reading `_paraphrase_tokens` added for
# `probe_fact_missing`, on a turn that mentions the subject without dictating
# it: "walking" stems to "walk" and "minute" to "minutes", so four of the five
# words survive and the loose reading calls this a delivery. It is not one, and
# arming a HARD abort on it would fail a turn that never dictated anything.
WALK_DICTATED = "He will walk fifteen minutes at most to the restaurant"
WALK_TERMS = ["fifteen minutes at most"]
WALK_MENTION = "My walking limit is fifteen-minute, restaurant-wise."


def test_a_mention_of_the_subject_is_not_a_delivery() -> None:
    assert not delivers_dictated_text(WALK_MENTION, WALK_DICTATED)


def test_a_mention_of_the_subject_does_not_arm_the_key_term_check() -> None:
    script = _gated_script(
        task_facts={"checkup_reason_the_user_dictates": WALK_DICTATED},
        gate_dictated_text=WALK_DICTATED,
        gate_key_terms=WALK_TERMS,
    )
    assert not _delivers_gated_dictation(script, _beat(script, "b2_probe"), WALK_MENTION)
    # The sentence itself still arms, and still passes.
    assert _delivers_gated_dictation(script, _beat(script, "b2_probe"), WALK_DICTATED)


# ── the schema fields ─────────────────────────────────────────────


def test_the_gate_fields_default_to_unarmed() -> None:
    script = _gated_script(
        gate_key_terms=[],
        gate_dictated_text="",
        gate_reveal_beat=None,
        gate_cue=None,
    )
    assert script.gate_key_terms == []
    assert script.gate_dictated_text == ""
    assert script.gate_reveal_beat is None
    assert script.gate_cue is None and script.gate_cue_beat is None


def test_an_authored_draft_leaves_the_gate_fields_empty() -> None:
    """They are the harness's to set from `harness.gate.resolve_session_gate`,
    never the draft's: the terms live in the outline, not in the session YAML."""
    from pathlib import Path

    import yaml
    from scripts.session_authoring_schema import NormalSessionDraft

    from simulator.schemas import authored_session_to_session_script

    draft = Path(__file__).resolve().parents[2] / (
        "Authoring/sessions/normal/drafts/user_a/acc_039.yaml"
    )
    data = yaml.safe_load(draft.read_text(encoding="utf-8"))
    script = authored_session_to_session_script(
        NormalSessionDraft.model_validate(data)
    )

    assert script.gate_key_terms == []
    assert script.gate_dictated_text == ""
    assert script.gate_reveal_beat is None
    assert script.gate_cue is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"gate_reveal_beat": None},
        {"gate_key_terms": []},
        {"gate_dictated_text": ""},
        {"gate_cue": None},
    ],
)
def test_half_a_gate_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="set together or not at all"):
        _gated_script(**overrides)


def test_a_reveal_beat_the_session_does_not_have_is_rejected() -> None:
    with pytest.raises(ValidationError, match="is not a beat of session"):
        _gated_script(gate_reveal_beat="b9_nowhere")


def test_the_cue_and_the_beat_that_owes_it_survive_a_round_trip() -> None:
    """The harness arms them with `model_copy` and re-validates the dump
    (`run_orchestrator._arm_dictation_check`), so a field that did not survive
    that round trip would reach the actor empty."""
    script = _gated_script(
        gate_cue="say what the visit is for",
        gate_cue_beat="b1_open",
    )
    reloaded = SessionScript.model_validate(script.model_dump())

    assert reloaded.gate_cue == "say what the visit is for"
    assert reloaded.gate_cue_beat == "b1_open"


# ── the retry, and the abort after the budget ─────────────────────


def _turn(message: str, next_beat: str = "advance") -> str:
    return (
        f"<message>\n{message}\n</message>\n\n"
        "<factual_check>\n</factual_check>\n\n"
        f"<turn_assessment>\nnext_beat: {next_beat}\ngoal_already_met: false\n"
        "</turn_assessment>"
    )


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def get_default_model(self) -> str:
        return "scripted-model"

    async def chat(self, messages, *args, **kwargs) -> LLMResponse:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("no scripted response left")
        return LLMResponse(content=self.responses.pop(0))


class _RecordingPA:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
    ) -> Any:
        from pa.bus.events import OutboundMessage

        self.messages.append(user_message)
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="PA reply",
            metadata={"tool_events": []},
        )


def _loop(
    responses: list[str],
    *,
    output_contract_retries: int = 1,
    script: SessionScript | None = None,
) -> tuple[SimulatorLoop, _ScriptedProvider, _RecordingPA]:
    provider = _ScriptedProvider(responses)
    pa = _RecordingPA()
    return (
        SimulatorLoop(
            provider,
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            _gated_script() if script is None else script,
            pa,
            backend=NanobotRunnerSimulatorBackend(
                provider, output_contract_retries=output_contract_retries
            ),
        ),
        provider,
        pa,
    )


_OPENING = _turn("Can you look at Dr. Patel's availability for a checkup?")
# The same information said correctly, with this beat's own ask on it.
_FIXED = (
    "The reason is: the chest tightening when I take the stairs in the "
    "mornings, which began about three weeks ago. Please enter it on the "
    "calendar."
)
# One word off in two places, so the note has two terms to name.
_TWO_DROPPED = (
    "The reason is: tightness in the chest when I climb stairs in the "
    "mornings, which began about three weeks ago. Please enter it on the "
    "calendar."
)
_CLOSE = _turn("That is the entry, then.")


def _retry_note(provider: _ScriptedProvider) -> str:
    """The last retry instruction the backend put to the actor."""
    return next(
        message["content"]
        for call in reversed(provider.calls)
        for message in reversed(call)
        if "violated the private output contract" in str(message.get("content"))
    )


def test_a_dropped_term_is_retried_with_a_note_naming_it() -> None:
    loop, provider, pa = _loop(
        [_OPENING, _turn(T55), _turn(_FIXED), _CLOSE, _CLOSE]
    )

    asyncio.run(loop.run_session())

    note = _retry_note(provider)
    assert (
        "Your last attempt dictated the text without these exact words: "
        "chest tightening." in note
    )
    assert "keep these exact phrases in it" in note
    # The corrected turn is the one the PA was given; the refused one is not.
    assert any(_FIXED in message for message in pa.messages)
    assert not any(T55 in message for message in pa.messages)


def test_the_retry_note_names_every_missing_term() -> None:
    loop, provider, _ = _loop(
        [_OPENING, _turn(_TWO_DROPPED), _turn(_FIXED), _CLOSE, _CLOSE]
    )

    asyncio.run(loop.run_session())

    assert (
        "without these exact words: chest tightening, the stairs."
        in _retry_note(provider)
    )


def test_the_rejected_attempt_row_carries_the_note_it_was_sent() -> None:
    """regress20 `acc_023`: three rejection rows with an empty `retry_note`,
    the t15 `dictation_missing_key_terms` one among them. The row now carries
    the note the loop put to the actor, terms named."""

    async def run():
        provider = _ScriptedProvider([_turn(T55), _turn(_FIXED)])
        backend = NanobotRunnerSimulatorBackend(provider)
        script = _gated_script()

        out = await backend.generate_turn(
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            script,
            _beat(script, "b2_probe"),
        )

        assert "dictation_missing_key_terms" in (
            out.rejected_attempts[0]["error_codes"]
        )
        assert (
            "without these exact words: chest tightening."
            in out.rejected_attempts[0]["retry_note"]
        )

    asyncio.run(run())


# ── the key terms are on the keep list too ────────────────────────


# The reason in the user's own words: every key term, none of the dictated
# sentence verbatim (so the whole-message dictation exemption does not cover
# it), and 37 words. The shape live memos r2 `acc_059` was.
_OWN_WORDS = (
    "It is the chest tightening when I take the stairs in the mornings, and it "
    "started about three weeks ago, so please put that on the entry before you "
    "confirm the booking with the clinic later today."
)
# The dictation is one sentence of a longer turn, not the whole of it, so the
# whole-message dictation exemption is out and the cap reads the turn sentence
# by sentence.
_MIXED = "The Tuesday slot is fine. " + _OWN_WORDS
# Carries no fact value, no cue and no key term: the runaway the cap exists for.
_A_SECOND_RUNAWAY = (
    "Send me the confirmation identifier as soon as the clinic has it, because "
    "the office will want to see the booking on the shared calendar before "
    "anybody signs off on the afternoon I have blocked out for it."
)


def _confirm_beat(script: SessionScript) -> Beat:
    """`b3_confirm`: past the reveal, so the dictation rule is still armed on a
    delivering turn, and the reason is no longer a required fact there — what
    the cap lets through is then the key terms' doing and not the fact's."""
    return _beat(script, "b3_confirm")


def test_the_key_terms_buy_their_sentence_out_of_the_length_cap() -> None:
    """Owner, 2026-09-16: one list for the retry note and the 35-word cap, and
    the dictation key terms join it. Live memos r2 `acc_059`: the dictation
    sentence of a mixed message was over the cap, the style retry shortened it,
    the terms went with the clauses it cut, and the blocking dictation rule
    then spent a second retry on the rewrite."""
    # Over the cap on its own; the key terms are what buy it out.
    assert sentence_too_long_finding(_OWN_WORDS) == "sentence_over_35_words"

    async def run():
        provider = _ScriptedProvider([_turn(_MIXED)])
        backend = NanobotRunnerSimulatorBackend(provider)
        script = _gated_script()

        out = await backend.generate_turn(
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            script,
            _confirm_beat(script),
        )

        assert _delivers_gated_dictation(script, _confirm_beat(script), _MIXED)
        assert out.message == _MIXED
        assert out.rejected_attempts == []

    asyncio.run(run())


def test_the_retry_note_quotes_the_sentence_carrying_the_key_terms() -> None:
    """The other half of the same list: when a retry does fire — on a second
    sentence that carries nothing and is genuinely too long — the note tells
    the actor to keep the dictation sentence word for word. Every style finding
    but the length one is record-only, so an over-long sentence elsewhere in
    the message is the only thing that asks for a shortening rewrite."""

    async def run():
        provider = _ScriptedProvider([
            _turn(f"{_OWN_WORDS} {_A_SECOND_RUNAWAY}"),
            _turn(_OWN_WORDS),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        script = _gated_script()

        out = await backend.generate_turn(
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            script,
            _confirm_beat(script),
        )

        assert out.rejected_attempts[0]["error_codes"] == ["sentence_over_35_words"]
        assert out.rejected_attempts[0]["retry_note"].endswith(
            f'Keep these sentences word for word: "{_OWN_WORDS}" Cut elsewhere.'
        )

    asyncio.run(run())


# The same information with every term scattered: "the tightening in my chest"
# for "chest tightening", "those stairs" for "the stairs", "three bad weeks"
# for "three weeks". `sentences_carrying` keeps it — each term's tokens are all
# here, which is 100% coverage — and `missing_key_terms` calls all three
# missing, because a term is its words in that order.
_SCATTERED = (
    "It is the tightening in my chest when I climb those stairs most "
    "mornings, and it began about three bad weeks ago."
)
# Carries the cue and nothing else on the keep list, so the keep clause has a
# sentence to quote after the missing terms come off it.
_CUE_SENTENCE = "Put the reason for the visit on the entry."


def test_a_term_the_dictation_reports_missing_is_not_kept_word_for_word() -> None:
    """Owner, 2026-09-16: the note never protects a sentence for a key term the
    same attempt's dictation finding calls missing. The two matchers disagree on
    a scattered phrase — 70% token coverage keeps the sentence, words-in-order
    calls the term missing — and the actor is then asked to put the phrase back
    and to leave the sentence it is scattered across exactly as it is."""

    async def run():
        provider = _ScriptedProvider([
            _turn(f"{_SCATTERED} {_CUE_SENTENCE} {_A_SECOND_RUNAWAY}"),
            _turn(_FIXED),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        script = _gated_script()

        out = await backend.generate_turn(
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            script,
            _confirm_beat(script),
        )

        # The premise: the dictation rule fires on this attempt, the cap fires
        # on the second sentence, and the scattered sentence covers the terms.
        rejected = out.rejected_attempts[0]
        assert set(rejected["error_codes"]) == {
            "dictation_missing_key_terms",
            "sentence_over_35_words",
        }
        assert missing_key_terms(_SCATTERED, TERMS) == TERMS
        assert sentences_carrying(_SCATTERED, TERMS) == [_SCATTERED]

        note = rejected["retry_note"]
        for term in TERMS:
            assert term in note
        assert note.endswith(
            f'Keep these sentences word for word: "{_CUE_SENTENCE}" Cut elsewhere.'
        )
        assert _SCATTERED not in note

    asyncio.run(run())


def test_the_budget_spent_on_a_dropped_term_invalidates_the_session() -> None:
    loop, _, pa = _loop([_OPENING, _turn(T55), _turn(T55)])

    with pytest.raises(SessionInvalidError) as exc_info:
        asyncio.run(loop.run_session())

    assert exc_info.value.rule == "dictation_missing_key_terms"
    assert exc_info.value.beat == "b2_probe"
    assert exc_info.value.attempts == 2
    # Nothing but the opening ever reached the PA: the refusal was never
    # delivered as a dictation.
    assert len(pa.messages) == 1


def test_a_timing_rule_that_survives_the_budget_is_played_not_aborted() -> None:
    """`withheld_fact_before_ask` is in `PLAYED_AFTER_BUDGET`.

    Same fixture with the reason NOT named in the beat's instruction, so the
    beat holds it until the assistant asks; the recording PA asks nothing, so
    the rule fires on every attempt. The turn still says the sentence the
    session needs, so the budget running out plays it with the finding on
    record instead of invalidating the run.
    """
    assert "withheld_fact_before_ask" in PLAYED_AFTER_BUDGET

    script = _gated_script()
    probe = _beat(script, "b2_probe")
    probe.actor_instruction = "Chooses the slot and says what the visit is for."
    provider = _ScriptedProvider([_OPENING, _turn(_FIXED), _turn(_FIXED), _CLOSE])
    pa = _RecordingPA()
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        script,
        pa,
        backend=NanobotRunnerSimulatorBackend(provider, output_contract_retries=1),
    )

    asyncio.run(loop.run_session())

    assert [event["kind"] for event in loop._events].count("session_invalid") == 0
    # The opening and the played dictation both reached the PA.
    assert len(pa.messages) >= 2
    assert any("three weeks ago" in message for message in pa.messages)


def test_the_invalid_session_ends_on_the_rule_and_carries_the_discarded_text() -> None:
    loop, _, _ = _loop([_OPENING, _turn(T55), _turn(T55)])

    with pytest.raises(SessionInvalidError):
        asyncio.run(loop.run_session())

    events = loop._events
    aborted = [
        event for event in events if event["kind"] == "dictation_missing_key_terms"
    ]
    assert len(aborted) == 1
    assert aborted[0]["beat"] == "b2_probe"
    assert aborted[0]["aborted"] is True
    assert "chest tightness" in aborted[0]["rejected_message"]

    end = [event for event in events if event["kind"] == "session_invalid"]
    assert [event["rule"] for event in end] == ["dictation_missing_key_terms"]
    assert [event for event in events if event["kind"] == "session_end"] == []
    assert events[-1] is end[0]


# --- one inflectional ending per word --------------------------------------
# The store gained this at jl 569490b (`_is_inflection`, `_same_word`,
# `_carries_term`): a word matches an ordinary inflection of itself, in either
# direction. The copy here has to agree, or the instrument flags a dictation the
# gate would have taken — live regress19r2 `acc_009` lost a write to "hardest"
# against a term of "hard". These cases are NOT pinned against the `StateStore`
# in this tree, which predates that commit; `test_the_normalisation_matches_the
# _store` still pins everything the two shared before it.


@pytest.mark.parametrize(
    "term, value, carried",
    [
        # The live case, and the reverse direction.
        ("hard", "your hardest questions on the subsection", True),
        ("hardest", "the hard questions on the subsection", True),
        ("hard", "the harder questions on the subsection", True),
        # Both ways round on a plain verb: the outline authors the term off the
        # dictated sentence and the user says it either way.
        ("print", "the printed copy is on my desk", True),
        ("printed", "please print the copy for my desk", True),
        # The two spelling changes the endings bring.
        ("settle", "we are settling it this pass", True),
        ("plan", "the planned window is ninety minutes", True),
        # Not a stemmer: the ending has to be one of the listed few.
        ("hard", "I hardly noticed the change", False),
        ("hard", "the hardware is fine", False),
        ("hard", "they harden over time", False),
        # A phrase is still its words in that order, inflected or not.
        ("hard questions", "the hardest questions in the set", True),
        ("hard questions", "hard, awkward questions", False),
        ("hard questions", "questions that are hardest", False),
    ],
)
def test_a_word_carries_one_inflection_of_itself(
    term: str, value: str, carried: bool
) -> None:
    assert (not missing_key_terms(value, [term])) is carried


# ── a beat whose own script dictates ───────────────────────────────
# Owner ruling 2026-09-15: the check also arms when the BEAT'S SCRIPT dictates
# the gated value. Reading the message alone leaves the case the gate cannot
# survive unchecked — an ask turn that paraphrases the dictated title away is
# not a delivery, so nothing fires, and the PA has no wording to write.


# Live `fin_05`, the case that is red without the new arm: the script says
# "titled Headshot pickup run" in as many words and the actor invented a
# different title, which no reading of the message can be a delivery of.
PICKUP_DICTATED = "Headshot pickup run"
# A probe's key terms are the dictated sentence whole, as one term
# (`harness.gate._probe_gate_wording`).
PICKUP_TERMS = [PICKUP_DICTATED]
PICKUP_SCRIPT = (
    "Tells the assistant that she is on the Bistro's Saturday double from "
    "11:00, then to check the print shop collection slip at "
    "documents/acting/print_collection_slip.md, decide which day is the "
    "latest day she can collect the prints before they get re-filed into "
    "stock, and put a calendar hold on that day titled Headshot pickup run."
)
PICKUP_TURN = (
    "Okay, could you check the print collection slip in my acting folder to "
    "see which day my headshot prints come off the counter? Then put a hold "
    'on my calendar for that final collection day titled "Headshot print '
    'counter pickup".'
)

# Live `fin_19`, the probe the ruling was written off. Kept because it is the
# owner's example, with the correction the replay turned up: the delivery arm
# ALREADY fires here — "water", "bill", "payment" and "hold" are all somewhere
# in the turn, so `delivers_dictated_text` reads it as a delivery and the live
# run recorded the finding. The new arm changes nothing about this case; the
# test pins it so neither arm loses it.
WATER_DICTATED = "Water Bill Payment Hold"
WATER_TERMS = [WATER_DICTATED]
WATER_SCRIPT = (
    "Please check my January water bill and the policy in "
    "documents/home/glenmont_water_payment_terms.md. Should I pay it today "
    "or wait until the due date? Also, create a calendar hold on the due "
    "date titled 'Water Bill Payment Hold'."
)
WATER_TURN = (
    "Okay, could you check my January water bill and the Glenmont water "
    "payment terms document in my files? I'm trying to figure out if I "
    "should pay it today or wait until the due date. And can you set a "
    "calendar hold on the due date for the water bill payment?"
)


def _probe_script(
    *, dictated: str, terms: list[str], script: str, beat_id: str = "ask"
) -> SessionScript:
    """A one-beat probe, the shape `fin_*` drafts have."""
    return SessionScript(
        session_id="fin_05",
        scenario="the print collection slip",
        context="personal",
        task="Put the hold on the calendar.",
        life_context=LifeContext(),
        task_facts={"calendar_hold_title": dictated},
        reveal_at={"calendar_hold_title": beat_id},
        beats=[Beat(beat=beat_id, goal="Ask for the hold.", actor_instruction=script)],
        gate_key_terms=terms,
        gate_dictated_text=dictated,
        gate_reveal_beat=beat_id,
        gate_cue=dictated,
    )


def _armed(
    script: SessionScript,
    beat: Beat,
    message: str,
    prior: tuple[str, ...] = (),
) -> bool:
    """The loop's arming expression for the hard key-term check.

    `prior` is the session's own earlier turns, which the loop reads off
    `history` — the simulator's turns are the "assistant" role there, the PA's
    the "user" role (`simulator/agent/loop.py`, the comment above
    `pa_history`).
    """
    return _delivers_gated_dictation(script, beat, message) or _script_dictates(
        script, beat, prior
    )


def test_a_scripted_dictation_the_turn_paraphrases_away_is_armed() -> None:
    """Live `fin_05`: red before the new arm, and the reason it exists.

    The turn invents a different title, so it is not a delivery of the dictated
    one and the message arm reads nothing. The beat's own script dictates it in
    as many words, so the turn owed the terms.
    """
    script = _probe_script(
        dictated=PICKUP_DICTATED, terms=PICKUP_TERMS, script=PICKUP_SCRIPT
    )
    beat = _beat(script, "ask")
    # Red without the new arm: the message is not a delivery.
    assert not _delivers_gated_dictation(script, beat, PICKUP_TURN)
    assert _script_dictates(script, beat)
    assert _armed(script, beat, PICKUP_TURN)
    assert (
        dictation_missing_key_terms_finding(PICKUP_TURN, key_terms=PICKUP_TERMS)
        == "dictation_missing_key_terms"
    )


def test_the_scripted_dictation_said_correctly_is_no_finding() -> None:
    script = _probe_script(
        dictated=PICKUP_DICTATED, terms=PICKUP_TERMS, script=PICKUP_SCRIPT
    )
    said = PICKUP_TURN.replace(
        '"Headshot print counter pickup"', '"Headshot pickup run"'
    )
    assert _armed(script, _beat(script, "ask"), said)
    assert dictation_missing_key_terms_finding(said, key_terms=PICKUP_TERMS) is None


def test_the_fin_19_turn_is_a_finding_and_the_dictated_title_clears_it() -> None:
    """The owner's example. Armed by either arm — see `WATER_DICTATED`."""
    script = _probe_script(
        dictated=WATER_DICTATED, terms=WATER_TERMS, script=WATER_SCRIPT
    )
    beat = _beat(script, "ask")
    assert _armed(script, beat, WATER_TURN)
    assert (
        dictation_missing_key_terms_finding(WATER_TURN, key_terms=WATER_TERMS)
        == "dictation_missing_key_terms"
    )
    said = WATER_TURN[:-1] + ", titled 'Water Bill Payment Hold'?"
    assert _armed(script, beat, said)
    assert dictation_missing_key_terms_finding(said, key_terms=WATER_TERMS) is None


def test_a_beat_that_does_not_dictate_stays_unarmed() -> None:
    """The `acc_009` t2 shape: the beat's own prose holds the sentence back
    until the PA asks, so a turn that does not say it is not a bad dictation."""
    script = _gated_script()
    withholding = _beat(script, "b2_probe").model_copy(
        update={
            "actor_instruction": (
                "Chooses the slot. Hands the reason over only when the "
                "assistant asks for it."
            )
        }
    )
    assert not _script_dictates(script, withholding)
    assert not _armed(script, withholding, "Take the Tuesday slot, then.")


def test_a_reaction_beat_whose_script_does_not_dictate_stays_unarmed() -> None:
    """The memos r2 false positive `_delivers_gated_dictation` was found in:
    a reaction beat read as the delivery turn. `_script_dictates` cannot widen
    it — it keys on the beat's OWN script text, and a reaction beat that only
    reacts dictates nothing."""
    script = _gated_script()
    reaction = _beat(script, "b3_confirm")
    assert reaction.actor_instruction == ""
    assert not _script_dictates(script, reaction)
    assert not _armed(script, reaction, "Good, that is the entry then.")


def test_an_acc_beat_that_dictates_in_text_and_drops_a_term_is_armed() -> None:
    """The acc line's shape: a beat whose own instruction quotes the sentence,
    and a turn that says it one word off."""
    script = _gated_script(
        beats=[
            Beat(beat="b1_open", goal="Report the scheduling bottleneck."),
            Beat(
                beat="b2_probe",
                goal="Say what the visit is for.",
                actor_instruction=(
                    "Says what the visit is for, in these words: " + DICTATED + "."
                ),
                active_skills=["information_elicitation"],
            ),
            Beat(beat="b3_confirm", goal="Confirm the booking record."),
        ]
    )
    beat = _beat(script, "b2_probe")
    assert _script_dictates(script, beat)
    # One word off, on a turn the message arm also reads as a delivery.
    assert _armed(script, beat, T55)
    assert (
        dictation_missing_key_terms_finding(T55, key_terms=TERMS)
        == "dictation_missing_key_terms"
    )
    # A turn on that beat that says nothing about the reason is armed too, for
    # as long as the session still owes the sentence.
    assert _armed(script, beat, "Not yet — confirm the calendar entry first.")


def test_the_arm_is_spent_once_the_session_has_said_the_sentence() -> None:
    """Owner ruling 2026-09-15, the narrowing. Live `simple_rag` `fin_22`: the
    title was dictated correctly on the ask turn, and the closing "That works,
    thanks." was then failed for not repeating it. A dictating beat owes the
    sentence once, not on every turn."""
    script = _probe_script(
        dictated="Chair run 5D", terms=["Chair run 5D"],
        script=(
            "Asks for an hour on the calendar this Friday daytime before the "
            "4 PM shift to get the chair back from 5D, titled Chair run 5D."
        ),
    )
    beat = _beat(script, "ask")
    said = (
        "Okay, can you block out an hour on my calendar this Friday daytime "
        "before my 4 PM shift to get my chair back from 5D? Title it Chair "
        "run 5D."
    )
    closing = "That works, thanks."
    # Before the session says it, the beat owes it.
    assert _armed(script, beat, closing)
    # The delivering turn itself passes, and spends the arm.
    assert _armed(script, beat, said)
    assert missing_key_terms(said, script.gate_key_terms) == []
    assert not _script_dictates(script, beat, (said,))
    assert not _armed(script, beat, closing, prior=(said,))


def test_a_turn_that_half_says_it_does_not_spend_the_arm() -> None:
    """The store's own test decides, so a near miss is not a delivery: the
    session still owes the sentence and the next turn is still armed."""
    script = _probe_script(
        dictated=PICKUP_DICTATED, terms=PICKUP_TERMS, script=PICKUP_SCRIPT
    )
    beat = _beat(script, "ask")
    assert missing_key_terms(PICKUP_TURN, script.gate_key_terms) == [PICKUP_DICTATED]
    assert _script_dictates(script, beat, (PICKUP_TURN,))


def test_a_dictating_beat_before_the_reveal_is_still_unarmed() -> None:
    """The window is unchanged: before the reveal beat the fact is out of the
    prompt, so there is nothing the turn could have dictated."""
    script = _gated_script(
        beats=[
            Beat(
                beat="b1_open",
                goal="Open.",
                actor_instruction="Says it in these words: " + DICTATED + ".",
            ),
            Beat(beat="b2_probe", goal="Choose the slot."),
            Beat(beat="b3_confirm", goal="Confirm."),
        ]
    )
    assert not _script_dictates(script, _beat(script, "b1_open"))


def test_an_unarmed_script_never_dictates() -> None:
    script = _probe_script(
        dictated=PICKUP_DICTATED, terms=PICKUP_TERMS, script=PICKUP_SCRIPT
    )
    unarmed = script.model_copy(
        update={
            "gate_key_terms": [],
            "gate_dictated_text": "",
            "gate_reveal_beat": None,
            "gate_cue": None,
        }
    )
    assert not _script_dictates(unarmed, _beat(unarmed, "ask"))


_PICKUP_FIXED = (
    "Okay, could you check the print collection slip in my acting folder to "
    "see which day my headshot prints come off the counter? Then put a hold "
    "on my calendar for that final collection day titled Headshot pickup run."
)


def test_the_finding_blocks_rather_than_only_recording() -> None:
    """Owner ruling 2026-09-15: it is out of `RECORD_ONLY_FINDINGS` again, so
    the existing consequence applies — a retry note, and a void past the
    budget."""
    assert "dictation_missing_key_terms" not in RECORD_ONLY_FINDINGS
    assert "dictation_missing_key_terms" in SESSION_INVALID_RULES


def test_the_loop_retries_a_scripted_dictation_that_drops_the_title() -> None:
    """End to end on the `fin_05` shape: the beat's script dictates the title,
    the turn invents another, and the loop puts the missing words to the actor
    rather than handing the PA a sentence the store would refuse."""
    script = _probe_script(
        dictated=PICKUP_DICTATED, terms=PICKUP_TERMS, script=PICKUP_SCRIPT
    )
    loop, provider, pa = _loop(
        [_turn(PICKUP_TURN), _turn(_PICKUP_FIXED), _CLOSE, _CLOSE], script=script
    )

    asyncio.run(loop.run_session())

    note = _retry_note(provider)
    assert (
        "Your last attempt dictated the text without these exact words: "
        "Headshot pickup run." in note
    )
    # The corrected turn is the one the PA was given; the refused one is not.
    assert any(_PICKUP_FIXED in message for message in pa.messages)
    assert not any(PICKUP_TURN in message for message in pa.messages)
