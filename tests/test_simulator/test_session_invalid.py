"""The simulator is the instrument: its own failures invalidate the session.

A still-malformed
three-block output used to be played on its parse fallback, routing the beat on
a `next_beat` the actor never wrote. It ends the session as invalid instead:
nothing reaches the PA, the transcript ends on `session_invalid` naming the
rule, and the raise is what the harness turns into the step's status and a
non-zero rc.

`beat_ask_unfulfilled` — the beat's scripted ask never got produced — came
through the same door until the owner's ruling of 2026-09-14 switched the
forced-ask check off altogether: the turn is played like any other, the judge
is never asked whether it carries an ask, and nothing is recorded. The rule
keeps its place in `SESSION_INVALID_RULES` because the commented-out check in
`simulator/loop.py` still names it. Section (a) is what that looks like.

The PA's own contract failures are the other way round — the PA is the subject,
so they stay continue-and-record (ruling 4). The last two tests hold that line.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_session_invalid.py
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pa.bus.events import OutboundMessage

from test_simulator.judge_fake import ScriptedJudgeClient

from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.judge.session import JudgeSession
from simulator.loop import SimulatorLoop
from simulator.output import (
    SESSION_INVALID_RULES,
    SessionInvalidError,
    SimulatorOutputContractError,
)
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)


def _turn(message: str, next_beat: str = "advance") -> str:
    return (
        f"<message>\n{message}\n</message>\n\n"
        "<factual_check>\n</factual_check>\n\n"
        f"<turn_assessment>\nnext_beat: {next_beat}\ngoal_already_met: true\n"
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


class _CountingPA:
    """Records every message delivered to the PA, and what it replied with."""

    def __init__(self, tool_events: list[dict] | None = None) -> None:
        self.messages: list[str] = []
        self.tool_events = list(tool_events or [])

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
    ) -> OutboundMessage:
        self.messages.append(user_message)
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="PA reply",
            metadata={"tool_events": self.tool_events},
        )


def _script() -> SessionScript:
    return SessionScript(
        session_id="session_invalid",
        scenario="the assistant runs ahead",
        context="personal",
        task="Get tonight's dinner sorted.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(beat="b1_open", goal="Have the assistant read Marcus's message."),
            Beat(
                beat="b2_probe_replacement",
                goal="Ask for a replacement venue.",
                active_skills=["solution_breadth"],
            ),
            Beat(beat="b3_react", goal="Take the recommended venue."),
        ],
    )


#: The ask gate is the judge's alone (contract §9), so every session here runs
#: one. These are the candidate turns that put nothing to the assistant; every
#: other turn reads as an ask, quoted out of the turn itself.
_NO_ASK = (
    "Right, you have already settled it.",
    "Fine, that is settled then.",
    "Nothing further from me.",
    "That one will do.",
)


def _judge() -> JudgeSession:
    return JudgeSession(mode="live", client=ScriptedJudgeClient(no_ask=_NO_ASK))


def _run(
    responses: list[str],
    *,
    pa: _CountingPA | None = None,
    output_contract_retries: int = 3,
) -> tuple[list[dict], _ScriptedProvider, _CountingPA]:
    """Run one session; return (events, provider, pa) or raise the loop's error.

    The events are read off the loop rather than the return value, because an
    invalid session raises instead of returning them.
    """
    provider = _ScriptedProvider(responses)
    recording_pa = pa or _CountingPA()
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script(),
        recording_pa,
        backend=NanobotRunnerSimulatorBackend(
            provider, output_contract_retries=output_contract_retries
        ),
        judge=_judge(),
    )
    events = asyncio.run(loop.run_session())
    return events, provider, recording_pa


def _run_expecting_invalid(
    responses: list[str], *, output_contract_retries: int = 3
) -> tuple[SessionInvalidError, list[dict], _ScriptedProvider, _CountingPA]:
    provider = _ScriptedProvider(responses)
    pa = _CountingPA()
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script(),
        pa,
        backend=NanobotRunnerSimulatorBackend(
            provider, output_contract_retries=output_contract_retries
        ),
        judge=_judge(),
    )
    with pytest.raises(SessionInvalidError) as exc_info:
        asyncio.run(loop.run_session())
    return exc_info.value, loop._events, provider, pa


def _of_kind(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["kind"] == kind]


def _gate_calls(events: list[dict]) -> list[dict]:
    """Every `carries_an_ask` the session put to the judge. The loop writes one
    `judge_call` per answered question group, so an empty list is the proof
    that the commented-out check costs nothing."""
    return [e for e in _of_kind(events, "judge_call") if e["rule"] == "carries_an_ask"]


# ── (a) beat_ask_unfulfilled ──────────────────────────────────────


_ACKNOWLEDGEMENTS = [
    _turn("Anything from Marcus?"),
    _turn("Right, you have already settled it."),
    _turn("Fine, that is settled then."),
    _turn("Nothing further from me."),
]


def test_a_beat_ask_that_never_comes_is_played_and_nothing_is_recorded() -> None:
    """The forced-ask check is off (owner ruling 2026-09-14): the probe beat's
    first turn asks nothing and is played like any other turn. The judge is
    never asked about it, and no event, diagnostic or rejected row is left."""
    events, provider, pa = _run(_ACKNOWLEDGEMENTS + [_turn("That one will do.")])

    assert _of_kind(events, "session_invalid") == []
    assert len(_of_kind(events, "session_end")) == 1
    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []
    assert _of_kind(events, "simulator_turn_diagnostics") == []
    # The probe beat's acknowledgement went to the PA like any turn.
    assert pa.messages[:2] == [
        "Anything from Marcus?",
        "Right, you have already settled it.",
    ]
    # One generation on the beat: nothing is re-asked.
    assert len(provider.calls) == 3


@pytest.mark.skip(reason="forced-ask check off, owner ruling 2026-09-14")
def test_the_unfulfilled_event_carries_the_played_text_without_an_abort() -> None:
    """Same shape `beat_ask_retry` uses for what it threw away: the event is a
    record of the candidate, truncated, not the artifact. `aborted` is now
    false — nothing is thrown away, the candidate is played."""
    events, _, _ = _run(_ACKNOWLEDGEMENTS + [_turn("That one will do.")])
    unfulfilled = _of_kind(events, "beat_ask_unfulfilled")

    assert len(unfulfilled) == 1
    assert unfulfilled[0]["beat"] == "b2_probe_replacement"
    assert unfulfilled[0]["attempts"] == 1
    assert unfulfilled[0]["aborted"] is False
    assert unfulfilled[0]["rejected_message"] == "Right, you have already settled it."


def test_the_candidate_the_gate_read_becomes_the_simulator_turn() -> None:
    events, _, _ = _run(_ACKNOWLEDGEMENTS + [_turn("That one will do.")])
    turns = _of_kind(events, "simulator_turn")

    assert [e["beat"] for e in turns][:2] == ["b1_open", "b2_probe_replacement"]
    assert "Right, you have already settled it." in [e["message"] for e in turns]
    # And the beat that ran out of budget got its PA turn like any other.
    assert [e["beat"] for e in _of_kind(events, "pa_turn")][:2] == [
        "b1_open",
        "b2_probe_replacement",
    ]


def test_an_invalid_session_ends_on_session_invalid_not_session_end() -> None:
    """Scripted on `simulator_output_format_invalid`, which still voids: the
    point here is the generic abort path once turns have already been played,
    and the ask rule that used to carry it is record-only since 2026-09-14."""
    _, events, _, _ = _run_expecting_invalid(
        [_turn("Anything from Marcus?"), _MALFORMED], output_contract_retries=0
    )
    end = _of_kind(events, "session_invalid")

    assert _of_kind(events, "session_end") == []
    assert len(end) == 1
    assert end[0]["rule"] == "simulator_output_format_invalid"
    assert end[0]["beat"] == "b2_probe_replacement"
    assert end[0]["attempts"] == 1
    assert end[0]["simulator_turns"] == 1
    # It is the last thing on the record: nothing runs after the abort.
    assert events[-1] is end[0]


# ── (b) simulator_output_format_invalid ───────────────────────────


_MALFORMED = "<message>So where do we eat instead?</message>"


def test_a_permanently_malformed_output_ends_the_session_as_invalid() -> None:
    error, events, provider, pa = _run_expecting_invalid(
        [_MALFORMED], output_contract_retries=0
    )

    assert error.rule == "simulator_output_format_invalid"
    assert error.beat == "b1_open"
    assert error.attempts == 1
    # It died on the opening beat, so the PA was never called at all.
    assert pa.messages == []
    assert len(provider.calls) == 1


def test_the_format_invalid_event_carries_the_abort_and_the_raw_head() -> None:
    _, events, _, _ = _run_expecting_invalid(
        [_MALFORMED], output_contract_retries=0
    )
    invalid = _of_kind(events, "simulator_output_format_invalid")

    assert len(invalid) == 1
    assert invalid[0]["beat"] == "b1_open"
    assert invalid[0]["attempts"] == 1
    assert invalid[0]["aborted"] is True
    assert invalid[0]["rejected_message"] == _MALFORMED


def test_the_format_invalid_abort_ends_on_session_invalid() -> None:
    _, events, _, _ = _run_expecting_invalid(
        [_MALFORMED], output_contract_retries=0
    )
    end = _of_kind(events, "session_invalid")

    assert _of_kind(events, "session_end") == []
    assert _of_kind(events, "simulator_turn") == []
    assert [e["rule"] for e in end] == ["simulator_output_format_invalid"]
    assert end[0]["beat"] == "b1_open"
    assert events[-1] is end[0]


def test_the_retry_budget_is_spent_before_the_session_is_invalidated() -> None:
    """Three retries by default, so four attempts before the abort."""
    error, _, provider, _ = _run_expecting_invalid([_MALFORMED] * 4)

    assert len(provider.calls) == 4
    assert error.attempts == 4


def test_a_malformed_first_attempt_that_is_corrected_plays_normally() -> None:
    events, _, pa = _run(
        [
            _MALFORMED,
            _turn("Anything from Marcus?"),
            _turn("So where do we eat instead?"),
            _turn("That one will do."),
        ]
    )

    assert _of_kind(events, "session_invalid") == []
    assert len(_of_kind(events, "session_end")) == 1
    assert pa.messages[0] == "Anything from Marcus?"


# ── the two rules are the only ones that abort ────────────────────


def test_only_the_three_simulator_rules_invalidate_a_session() -> None:
    """Three rules of the simulator's own, then the judge's five.

    The judge's are a different kind of instrument failure — nothing about the
    turn is wrong, the reader of the turn did not answer — but they share this
    door exactly (local judge contract §4, §13 i): the session is void, there
    is no fallback and no diagnostic-and-play path. `judge_unavailable` shares
    the door and then goes further: the loop re-raises the underlying
    `JudgeUnavailable` so the round stops, because a judge that is gone is gone
    for every session after this one too.
    """
    assert SESSION_INVALID_RULES == (
        "beat_ask_unfulfilled",
        "simulator_output_format_invalid",
        "dictation_missing_key_terms",
        "judge_unparseable",
        "judge_evidence_invalid",
        "judge_timeout",
        "judge_budget_exhausted",
        "judge_unavailable",
    )


def test_another_output_contract_failure_still_raises_its_own_error() -> None:
    """A hard finding that is not the format class keeps the path it had: the
    contract error propagates, and no `session_invalid` is written.

    An authoring leak, not the invented path this used to be scripted with:
    every judgement about the turn is record-only since the owner's ruling of
    2026-09-14, so the findings left blocking are the ones whose message cannot
    be played as it stands.
    """
    leaks_authoring = _turn("As the simulator I need the venue list today.")
    provider = _ScriptedProvider([leaks_authoring] * 4)
    pa = _CountingPA()
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script(),
        pa,
        backend=NanobotRunnerSimulatorBackend(provider),
    )

    with pytest.raises(SimulatorOutputContractError) as exc_info:
        asyncio.run(loop.run_session())

    assert exc_info.value.findings == ["authoring_metadata_in_message"]
    assert not isinstance(exc_info.value, SessionInvalidError)
    assert _of_kind(loop._events, "session_invalid") == []


def test_a_pa_side_contract_failure_does_not_invalidate_the_session() -> None:
    """Ruling 4: the PA is the subject, so its own contract failures are
    recorded and the session runs on. `ix_contract_failure` and the certainty
    events ride the PA's `tool_events` and never reach the abort path."""
    pa = _CountingPA(
        tool_events=[
            {"name": "ix_certainty_claim_not_found", "ix_event": True},
            {"name": "ix_contract_failure", "ix_event": True},
        ]
    )
    events, _, _ = _run(
        [
            _turn("Anything from Marcus?"),
            _turn("So where do we eat instead?"),
            _turn("That one will do."),
        ],
        pa=pa,
    )

    assert _of_kind(events, "session_invalid") == []
    assert len(_of_kind(events, "session_end")) == 1
    assert len(pa.messages) == 3
