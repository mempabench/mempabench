"""The beat that carries the gate's cue has to actually say it.

The cue is the phrase a gated session's own prose gives the user so that a
question about the withheld fact is a natural thing for the PA to ask (owner
ruling 2026-09-08: "a gate nothing in the task motivates is not a gate"). Live
regress18 lost it on three of the five gated sessions — `acc_009` t2, `acc_039`
t3 and `acc_011` t7 — every one of them to a style
retry, and in all three the PA then wrote instead of asking, so the gate was
measured on refusals rather than on the design. Nothing saw it go: the cue is
prose in `user_instruction`, not a `task_facts` key.

The rule was retried and then played (owner, 2026-09-08); since the owner's
ruling of 2026-09-14 ("只记") it is recorded and played: a missing cue makes it
likely the PA never asks, but unlike a dropped key term it hands the PA nothing
the world will refuse, so the session still measures something.

Phase 3 step 5 (2026-09-09): the cue is a semantic question, so the local judge
answers it — inside the `general` group, on the cue beat's opening turn only —
and the content-word regex that used to answer it is gone. There is no
fallback: with `MEMPA_JUDGE=off` the rule is not judged at all,
so every test below that is about the finding runs with a live judge. What the
harness still owns is the arming (which turn is asked) and the consequence
(record, and play).

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_gate_cue.py
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.judge.session import JudgeSession
from simulator.loop import SimulatorLoop
from simulator.message_quality import sentence_too_long_finding
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

# The judge double the rest of the ladder is tested with. `tests/` is on
# `sys.path` under pytest (`pythonpath = ["."]`, and the rootdir insertion),
# so this is the same class `test_wiring.py` scripts, not a second copy.
from test_simulator.judge_fake import ScriptedJudgeClient


# `acc_039`'s live gate and its live cue: the opening is all scheduling, so the
# reason for the visit only becomes askable once the user says the clinic wants
# it. The cue is what the style retry deleted at t3 and inverted at t6.
TERMS = ["chest tightening", "the stairs", "three weeks"]
DICTATED = (
    "the chest tightening when I take the stairs in the mornings, which began "
    "about three weeks ago"
)
CUE = "the reason for the visit"


# ── the schema fields ─────────────────────────────────────────────


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
                active_skills=["information_elicitation"],
            ),
            Beat(beat="b3_confirm", goal="Confirm the booking record."),
        ],
        "gate_key_terms": TERMS,
        "gate_dictated_text": DICTATED,
        "gate_reveal_beat": "b2_probe",
        "gate_cue": CUE,
        "gate_cue_beat": "b1_open",
    }
    fields.update(overrides)
    return SessionScript(**fields)


def test_the_cue_fields_default_to_unset() -> None:
    script = SessionScript(
        session_id="acc_001",
        scenario="no gate",
        context="personal",
        task="Anything.",
        life_context=LifeContext(),
        beats=[Beat(beat="b1_open")],
    )
    assert script.gate_cue is None and script.gate_cue_beat is None


def test_a_cue_beat_with_no_cue_is_rejected() -> None:
    with pytest.raises(ValidationError, match="gate_cue_beat requires gate_cue"):
        SessionScript(
            session_id="acc_001",
            scenario="no gate",
            context="personal",
            task="Anything.",
            life_context=LifeContext(),
            beats=[Beat(beat="b1_open")],
            gate_cue_beat="b1_open",
        )


def test_a_cue_beat_the_session_does_not_have_is_rejected() -> None:
    with pytest.raises(ValidationError, match="gate_cue_beat 'b9_nowhere'"):
        _gated_script(gate_cue_beat="b9_nowhere")


def test_a_cue_with_no_beat_is_allowed() -> None:
    """Live `acc_023`: the cue is in the scenario, on no beat of its own. The
    gate is armed; the per-beat check is not."""
    script = _gated_script(gate_cue_beat=None)
    assert script.gate_cue == CUE and script.gate_cue_beat is None


# ── arming, the retry, and the play after the budget ──────────────


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
    judge: Any | None = None,
    **script_overrides: Any,
) -> tuple[SimulatorLoop, _ScriptedProvider, _RecordingPA]:
    provider = _ScriptedProvider(responses)
    pa = _RecordingPA()
    return (
        SimulatorLoop(
            provider,
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            _gated_script(**script_overrides),
            pa,
            backend=NanobotRunnerSimulatorBackend(
                provider, output_contract_retries=1
            ),
            judge=judge,
        ),
        provider,
        pa,
    )


def _cue_judge(
    *verdicts: tuple[str, str], ask: tuple[str, str] | None = None
) -> tuple[JudgeSession, ScriptedJudgeClient]:
    """A live judge whose `gate_cue_missing` answers are scripted in order.

    Every other question of the `general` group falls through to the client's
    default, `not_applicable`, which the guards let answer with no evidence.
    `ask` answers the one ask-gate call these scripts make: `b2_probe` is their
    only declaring beat, so only its opening turn owes an ask.
    """
    answers: dict[str, Any] = {
        "gate_cue_missing": list(verdicts),
        # The probe turns below stand in for the beat that dictates the reason,
        # and `probe_fact_missing` has been hard since the owner's ruling of
        # 2026-09-13, so without this every one of these scripts would void on
        # a rule none of them is about. The judge reads the fact in the turn.
        "probe_fact_missing:checkup_reason_the_user_dictates": (
            "yes",
            "candidate_turn#1",
        ),
    }
    if ask is not None:
        answers["carries_an_ask"] = [ask]
    client = ScriptedJudgeClient(answers)
    return JudgeSession(mode="live", client=client), client


def _asked(client: ScriptedJudgeClient, question_id: str) -> int:
    """How many calls carried that question. The arming, read off the wire."""
    return sum(1 for questions in client.calls if question_id in questions)


# Verbatim spans of the candidates the ask gate sees, for the guards.
_ASK_CONFIRM = ("ask", "Can you confirm the entry")
_ASK_NO_CUE = ("ask", "Can you find me a checkup slot")
_ASK_LOOK_FOR = ("ask", "Can you look for a checkup slot")
_VIOLATED_NO_CUE = ("violated", "Can you find me a checkup slot")
_OBEYED_WITH_CUE = ("obeyed", "The clinic wants the reason for the visit")


# The live t3 the retry accepted, and the same turn with the cue kept.
_NO_CUE = "Can you find me a checkup slot with Dr. Patel in the next fortnight?"
_WITH_CUE = (
    "Can you look for a checkup slot with Dr. Patel in the next fortnight? The "
    "clinic wants the reason for the visit before it will hold one."
)
# Every beat after the first is a probe beat, so its turn has to carry an ask,
# and each one has to differ from the last (the judge's repeat check, removed 2026-09-12).
_LATER = [
    _turn("Can you confirm the entry now?"),
    _turn("Is the hold showing on the calendar?"),
    _turn("Anything else you need from me?"),
]


def _retry_note(provider: _ScriptedProvider) -> str:
    return next(
        message["content"]
        for call in reversed(provider.calls)
        for message in reversed(call)
        if "violated the private output contract" in str(message.get("content"))
    )


def test_a_cue_beat_turn_without_the_cue_is_recorded_and_played() -> None:
    """Owner ruling 2026-09-14 ("只记"): the cue check still runs on the cue
    beat's opening turn and still writes its judge row, but a missing cue buys
    no rewrite — the turn the judge refused is the one the PA is given."""

    judge, _ = _cue_judge(_VIOLATED_NO_CUE, ask=_ASK_LOOK_FOR)
    loop, provider, pa = _loop(
        [_turn(_NO_CUE), _turn(_WITH_CUE), *_LATER], judge=judge
    )

    asyncio.run(loop.run_session())

    with pytest.raises(StopIteration):
        _retry_note(provider)
    assert any(_NO_CUE in message for message in pa.messages)
    assert [
        event["beat"]
        for event in loop._events
        if event["kind"] == "simulator_turn_diagnostics"
        and "gate_cue_missing" in event["diagnostics"]
    ] == ["b1_open"]


def test_only_the_first_turn_of_the_cue_beat_is_checked() -> None:
    """The drafts ask for the cue in its own sentence in the beat's opening.
    Demanding it again on the beat's later turns would only have the actor
    repeat the same phrase until the beat ends."""
    judge, client = _cue_judge(_OBEYED_WITH_CUE, ask=_ASK_CONFIRM)
    loop, provider, pa = _loop(
        [
            _turn(_WITH_CUE, next_beat="stay"),
            _turn(_NO_CUE),
            *_LATER[:2],
        ],
        judge=judge,
    )

    asyncio.run(loop.run_session())

    # Asked once, on the opening. The second turn of the same beat is never
    # put to the judge, so no answer of its could have thrown it away.
    assert _asked(client, "gate_cue_missing") == 1
    assert not any(
        "violated the private output contract" in str(message.get("content"))
        for call in provider.calls
        for message in call
    )
    assert any(_NO_CUE in message for message in pa.messages)


def test_a_turn_on_another_beat_is_not_checked_for_the_cue() -> None:
    """Only the beat the harness named carries the cue; the beats after it are
    about the dictation."""
    judge, client = _cue_judge(_OBEYED_WITH_CUE, ask=_ASK_NO_CUE)
    loop, provider, pa = _loop(
        [_turn(_WITH_CUE), _turn(_NO_CUE), *_LATER[:2]], judge=judge
    )

    asyncio.run(loop.run_session())

    assert _asked(client, "gate_cue_missing") == 1
    assert not any(
        "violated the private output contract" in str(message.get("content"))
        for call in provider.calls
        for message in call
    )
    assert any(_NO_CUE in message for message in pa.messages)


def test_an_unset_cue_checks_nothing() -> None:
    """Every compressed session and every forced event, and `acc_023`, whose
    cue sits in the scenario with no beat of its own."""
    judge, client = _cue_judge(ask=_ASK_CONFIRM)
    loop, provider, pa = _loop(
        [_turn(_NO_CUE), *_LATER], judge=judge, gate_cue_beat=None
    )

    asyncio.run(loop.run_session())

    assert _asked(client, "gate_cue_missing") == 0
    assert not any(
        "violated the private output contract" in str(message.get("content"))
        for call in provider.calls
        for message in call
    )
    assert any(_NO_CUE in message for message in pa.messages)


def test_a_dropped_cue_never_costs_the_session() -> None:
    """It was retry-then-play (owner, 2026-09-08) and is play-and-record now. A
    missing cue makes it likely the PA never asks, but it hands the PA nothing
    the world will refuse, so the session still measures something and runs on."""
    judge, _ = _cue_judge(_VIOLATED_NO_CUE, ask=_ASK_NO_CUE)
    loop, _, pa = _loop([_turn(_NO_CUE), _turn(_NO_CUE), *_LATER], judge=judge)

    events = asyncio.run(loop.run_session())

    assert any(_NO_CUE in message for message in pa.messages)
    assert [event for event in events if event["kind"] == "session_invalid"] == []
    assert len([event for event in events if event["kind"] == "session_end"]) == 1


def test_the_played_turn_carries_the_rule_as_a_diagnostic() -> None:
    judge, _ = _cue_judge(_VIOLATED_NO_CUE, ask=_ASK_NO_CUE)
    loop, _, _ = _loop([_turn(_NO_CUE), _turn(_NO_CUE), *_LATER], judge=judge)

    events = asyncio.run(loop.run_session())

    recorded = [
        event
        for event in events
        if event["kind"] == "simulator_turn_diagnostics"
        and "gate_cue_missing" in event["diagnostics"]
    ]
    assert [event["beat"] for event in recorded] == ["b1_open"]


# ── the cue is owed against the style retry too ───────────────────


_LONG_TURN_WITH_THE_CUE = (
    "Dr. Patel's clinic wants the reason for the visit before it will hold a "
    "slot. Can you find me a checkup in the next fortnight, ideally on a "
    "morning when the traffic across town is still light and the waiting room "
    "has not filled up with the school run crowd that always seems to arrive "
    "right before I do?"
)


def test_the_style_retry_note_quotes_the_sentence_carrying_the_cue() -> None:
    """The regress18 defect itself: a style check fired on the cue
    beat's first attempt and the shortened rewrite paid for the cut with the
    cue (`acc_009` t2, `acc_039` t3, `acc_011` t7)."""
    loop, provider, _ = _loop(
        [
            _turn(_LONG_TURN_WITH_THE_CUE),
            _turn(_WITH_CUE),
            *_LATER,
        ]
    )

    asyncio.run(loop.run_session())

    note = _retry_note(provider)
    assert (
        'Keep these sentences word for word: "Dr. Patel\'s clinic wants the '
        'reason for the visit before it will hold a slot." Cut elsewhere.'
        in note
    )


_LONG_TURN_RESTATING_THE_CUE = (
    "The volume number confirms the date. I will give the clinic the reason "
    "for the visit once we are ready to book. Shall we look at the rest of "
    "the calendar now, since the whole of next week is still open apart from "
    "the Thursday seminar and I would rather settle the rest of it in one go "
    "than come back to it twice?"
)


def test_the_style_retry_note_quotes_a_cue_restated_on_a_later_beat() -> None:
    """Live regress20 `acc_003` t10: a b2 shortening retry with an empty note,
    and the rewrite dropped b2's restatement of the cue while keeping a
    question that widened the task. The cue is owed by its own beat, but a
    later beat that says it again is saying the sentence the gate depends on.
    """
    loop, provider, _ = _loop(
        [
            _turn(_WITH_CUE),
            _turn(_LONG_TURN_RESTATING_THE_CUE),
            _turn("Please take the earliest slot you found."),
            *_LATER,
        ]
    )

    asyncio.run(loop.run_session())

    notes = [
        message["content"]
        for call in provider.calls
        for message in call
        if "violated the private output contract" in str(message.get("content"))
    ]
    assert any(
        'Keep these sentences word for word: "I will give the clinic the '
        'reason for the visit once we are ready to book." Cut elsewhere.'
        in note
        for note in notes
    )


_ONE_LONG_SENTENCE_RESTATING_THE_CUE = (
    "I will give the clinic the reason for the visit as soon as the booking "
    "is held, since the whole of next week is open apart from the Thursday "
    "seminar and I would rather settle the rest of the calendar in one go "
    "than come back to it twice."
)


def test_a_later_beats_cue_restatement_is_not_counted_by_the_length_cap() -> None:
    """Owner, 2026-09-16: the sentences the note protects and the sentences the
    35-word cap does not count are one list. They were two, and live memos r2
    `acc_108` had the restated cue in only one of them: a retry note telling
    the actor to keep the 37-word sentence that was the whole message, and a
    cap still counting it, so every attempt was asked for a shorter turn and
    for that sentence whole."""
    # The sentence is over the cap on its own; the restated cue is what buys it
    # out, and before the two lists were joined nothing did.
    assert (
        sentence_too_long_finding(_ONE_LONG_SENTENCE_RESTATING_THE_CUE)
        == "sentence_over_35_words"
    )

    loop, provider, _ = _loop(
        [
            _turn(_WITH_CUE),
            _turn(_ONE_LONG_SENTENCE_RESTATING_THE_CUE),
            *_LATER,
        ]
    )

    asyncio.run(loop.run_session())

    with pytest.raises(StopIteration):
        _retry_note(provider)


_ONE_LONG_SENTENCE_WITHOUT_THE_CUE = (
    "I will confirm the booking as soon as the hold shows, since the whole of "
    "next week is open apart from the Thursday seminar and I would rather "
    "settle the rest of the calendar in one go than come back to it twice."
)


def test_a_long_sentence_carrying_nothing_kept_is_still_counted() -> None:
    """The control: joining the two lists widens the cap's exemption to the
    cue and the gate's key terms, and to nothing else. A sentence that carries
    none of them is the runaway the cap exists for."""
    loop, provider, _ = _loop(
        [
            _turn(_WITH_CUE),
            _turn(_ONE_LONG_SENTENCE_WITHOUT_THE_CUE),
            _turn("Please take the earliest slot you found."),
            *_LATER,
        ]
    )

    asyncio.run(loop.run_session())

    note = _retry_note(provider)
    assert "over 35 words" in note
    assert "Keep these sentences word for word" not in note
