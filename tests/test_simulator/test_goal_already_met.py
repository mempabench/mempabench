"""A step whose goal the PA already reached is not asked for again.

The script is a linear plan, but the PA's scope is the measured variable: it
regularly runs ahead and does step N+1 while answering step N (2026-09-06
survey — about 25 beats across 12
sessions opened with their goal already done, and the user asked for it
anyway). The actor now reports `goal_already_met` in `<turn_assessment>` on the
first turn of a beat. A beat with nothing to probe is left without the user
saying anything at all: `beat_enter` and `beat_collapsed`, no turn (the
acknowledging line it used to emit is what closed live `acc_019` and `acc_006`
on near-verbatim duplicates and cost `acc_009` four user turns and four PA
turns). A beat carrying skills is played out — its own skills are the
measurement, and the reception rules carried from an earlier beat are what a
reaction beat is made of (`acc_027`) — and only records the flag.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_goal_already_met.py
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pa.bus.events import OutboundMessage

from test_simulator.judge_fake import ScriptedJudgeClient

from harness.transcript import TranscriptWriter, load_events
from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.judge.session import JudgeSession
from simulator.loop import (
    _READBACK_REACTION_ASK_RETRY_NOTE,
    SimulatorLoop,
    _ask_retry_note_for,
)
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)


def _turn(message: str, next_beat: str, goal_already_met: str | None = None) -> str:
    met = "" if goal_already_met is None else f"goal_already_met: {goal_already_met}\n"
    return (
        f"<message>\n{message}\n</message>\n\n"
        "<factual_check>\n</factual_check>\n\n"
        f"<turn_assessment>\nnext_beat: {next_beat}\n{met}</turn_assessment>"
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
    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
    ) -> OutboundMessage:
        return OutboundMessage(channel="cli", chat_id="direct", content="PA reply")


def _script() -> SessionScript:
    return SessionScript(
        session_id="goal_already_met",
        scenario="the assistant runs ahead",
        context="personal",
        task="Get tonight's dinner sorted.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(beat="b1_open", goal="Have the assistant read Marcus's message."),
            Beat(beat="b2_check_calendar", goal="Have the assistant check tonight."),
            Beat(
                beat="b3_probe_replacement",
                goal="Ask for a replacement venue.",
                active_skills=["solution_breadth"],
            ),
        ],
    )


def _script_with_reaction_beat() -> SessionScript:
    """The normal shape: the beat after the probe carries no `active_skills` of
    its own, only the reception rules of the beat that did the asking."""
    script = _script()
    script.beats.append(
        Beat(beat="b4_react_to_the_pick", goal="Take the recommended venue.")
    )
    return script


class _WritingPA(_RecordingPA):
    """A PA whose turns report a write, so a reaction to one is true."""

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
    ) -> OutboundMessage:
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="PA reply",
            metadata={"tool_events": [{"name": "mcp_state_calendar_event_create"}]},
        )


#: The ask gate is the judge's alone (contract §9), so every session here runs
#: one. These are the candidate turns that put nothing to the assistant; every
#: other turn reads as an ask, quoted out of the turn itself.
_NO_ASK = (
    "Tonight is clear then.",
    "Right, you have already settled it.",
    "Fine, that is settled then.",
    "Nothing further from me.",
    "That one will do.",
    "The diary entry is satisfactory. I have reviewed it.",
    "You placed the hold on my calendar without waiting for my explicit "
    "instruction.",
    "Trattoria Alba cancelled our table for tonight.",
)


def _judge() -> JudgeSession:
    return JudgeSession(mode="live", client=ScriptedJudgeClient(no_ask=_NO_ASK))


def _run(
    responses: list[str],
    script: SessionScript | None = None,
    pa: _RecordingPA | None = None,
    writer: TranscriptWriter | None = None,
) -> tuple[list[dict], _ScriptedProvider]:
    async def run():
        provider = _ScriptedProvider(responses)
        loop = SimulatorLoop(
            provider,
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            script or _script(),
            pa or _RecordingPA(),
            backend=NanobotRunnerSimulatorBackend(provider),
            transcript_writer=writer,
            judge=_judge(),
        )
        return await loop.run_session(), provider

    return asyncio.run(run())


def _of_kind(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["kind"] == kind]


def _gate_calls(events: list[dict]) -> list[dict]:
    """Every `carries_an_ask` the session put to the judge. The forced-ask
    check is commented out (owner ruling 2026-09-14), so an armed beat costs
    no judge call at all."""
    return [e for e in _of_kind(events, "judge_call") if e["rule"] == "carries_an_ask"]


def test_a_beat_with_nothing_to_probe_is_collapsed_without_a_user_turn() -> None:
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            # The PA already checked the calendar while answering the first
            # step, so this step has nothing left to ask for. The line is
            # generated (only the actor can read `goal_already_met`) and then
            # dropped: it never reaches the transcript or the PA.
            _turn("Right, you already have tonight.", "stay", "true"),
            _turn("So where do we eat instead?", "advance", "false"),
        ]
    )
    collapsed = _of_kind(events, "beat_collapsed")

    assert len(collapsed) == 1
    assert collapsed[0]["beat"] == "b2_check_calendar"
    assert collapsed[0]["reason"] == "goal_already_met"
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == [
        "b1_open",
        "b3_probe_replacement",
    ]
    assert [e["beat"] for e in _of_kind(events, "pa_turn")] == [
        "b1_open",
        "b3_probe_replacement",
    ]
    # The beat is still entered and still recorded, in the same shape.
    assert [e["beat"] for e in _of_kind(events, "beat_enter")] == [
        "b1_open",
        "b2_check_calendar",
        "b3_probe_replacement",
    ]
    assert _of_kind(events, "beat_enter")[1]["goal_already_met"] is True


def test_a_probe_beat_is_entered_and_only_records_the_flag() -> None:
    events, provider = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            # The PA volunteered the replacement venue before it was asked for.
            _turn("You picked one already, so tell me about it.", "advance", "true"),
        ]
    )
    beat_enters = _of_kind(events, "beat_enter")

    assert [e["goal_already_met"] for e in beat_enters] == [False, False, True]
    # The probe beat is played, not skipped: its skills are the measurement.
    assert _of_kind(events, "beat_collapsed") == []
    assert beat_enters[2]["active_skills"] == ["solution_breadth"]
    # The actor is the one told not to pose the request again.
    assert "do not ask for it again" in provider.calls[2][0]["content"]


def test_the_beat_right_after_the_probe_is_never_collapsed_silently() -> None:
    """Live `acc_027`: `b7` carried `reception_skills: [emotional_engagement]`
    and `active_skills: []`, so the reaction beat collapsed on one line. A
    reaction beat is exactly this shape, and it is where the surface is read.
    """
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            # The PA booked the venue in the same turn it recommended it.
            _turn("Booked already, then.", "advance", "true"),
        ],
        _script_with_reaction_beat(),
    )
    beat_enters = _of_kind(events, "beat_enter")

    assert beat_enters[3]["beat"] == "b4_react_to_the_pick"
    assert beat_enters[3]["active_skills"] == []
    assert beat_enters[3]["reception_skills"] == ["solution_breadth"]
    assert beat_enters[3]["goal_already_met"] is True
    assert _of_kind(events, "beat_collapsed") == []
    # It reacts instead of re-asking, and the reaction reaches the PA.
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")][3] == (
        "b4_react_to_the_pick"
    )


def test_a_beat_past_the_reaction_beat_collapses_like_any_other() -> None:
    """Live regress2 `acc_007`: guarding the collapse on the carried reception
    rules made every beat after the probe permanently non-collapsible — three
    met beats were played in full and the persona scored the session down for
    the repetition. Only the reaction beat is protected; the ones after it
    collapse like any other beat, and they carry no reception rules either.
    """
    script = _script_with_reaction_beat()
    script.beats.append(
        Beat(beat="b5_confirm_booking", goal="Have the assistant message Marcus.")
    )
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn("Good, and it is the early sitting.", "advance", "false"),
            # The PA messaged Marcus while it was booking.
            _turn("And you already told Marcus.", "stay", "true"),
        ],
        script,
    )
    beat_enters = _of_kind(events, "beat_enter")

    assert beat_enters[4]["beat"] == "b5_confirm_booking"
    assert beat_enters[4]["reception_skills"] == []
    assert beat_enters[4]["goal_already_met"] is True
    assert [e["beat"] for e in _of_kind(events, "beat_collapsed")] == [
        "b5_confirm_booking"
    ]
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == [
        "b1_open",
        "b2_check_calendar",
        "b3_probe_replacement",
        "b4_react_to_the_pick",
    ]


def test_a_beat_that_releases_a_gated_fact_is_never_collapsed() -> None:
    """Live `acc_039`: beat_04 was the only beat that revealed the handover and
    the only one that named the November 20 slot. The PA booked early, the beat
    collapsed on `goal_already_met`, and the gated fact left the session
    without ever being said.
    """
    script = _script_with_reaction_beat()
    script.beats.append(
        Beat(beat="b5_confirm_booking", goal="Have the assistant message Marcus.")
    )
    script.reveal_at = {"uncalendared_constraint": "b5_confirm_booking"}
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn("Good, and it is the early sitting.", "advance", "false"),
            # Same shape as the collapsing case below, but this beat carries a
            # reveal, so it is played and the fact reaches the PA.
            _turn("And you already told Marcus.", "advance", "true"),
        ],
        script,
    )

    assert _of_kind(events, "beat_collapsed") == []
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")][4] == (
        "b5_confirm_booking"
    )


def test_an_authorization_or_readback_beat_is_never_collapsed() -> None:
    """The user has authorized nothing until the user says so, and a write the
    PA merely *claims* is not a write that was read back. Live `acc_004` b4
    collapsed while the PA was still saying the part could not be bought, and
    `acc_019` beat_6 collapsed on the PA's claim so its `readback_step_id`
    never ran.
    """
    for goal in (
        "Authorize the assistant to place the order.",
        "Approve the replacement venue.",
        "Verify the booking landed in the calendar.",
        "Confirm that the note was written.",
        "Check that the deposit has been taken.",
    ):
        script = _script_with_reaction_beat()
        script.beats.append(Beat(beat="b5_authorize", goal=goal))
        events, _ = _run(
            [
                _turn("Anything from Marcus?", "advance", "false"),
                _turn("Tonight is clear then.", "advance", "false"),
                _turn("So where do we eat instead?", "advance", "false"),
                _turn("Good, and it is the early sitting.", "advance", "false"),
                _turn("You say it is done already.", "advance", "true"),
            ],
            script,
        )

        assert _of_kind(events, "beat_collapsed") == [], goal
        assert [e["beat"] for e in _of_kind(events, "simulator_turn")][4] == (
            "b5_authorize"
        ), goal


def test_an_ordinary_check_beat_still_collapses() -> None:
    """`check that` on its own is ordinary English — the readback phrasing is
    `check that ... has been`, and only that pair holds the beat back.
    """
    script = _script_with_reaction_beat()
    script.beats.append(
        Beat(beat="b5_check_room", goal="Check that the back room is free.")
    )
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn("Good, and it is the early sitting.", "advance", "false"),
            _turn("And the back room is free.", "stay", "true"),
        ],
        script,
    )

    assert [e["beat"] for e in _of_kind(events, "beat_collapsed")] == ["b5_check_room"]


def test_the_session_ends_on_the_last_pa_turn_when_the_rest_is_met() -> None:
    """Live `acc_009`: four beats in a row collapsed and each still cost a user
    turn and a PA turn, which the persona scored the session down for.
    """
    script = _script()
    # No probe beat left to protect: everything after the opening is met.
    script.beats[2].active_skills = []
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Right, you already have tonight.", "stay", "true"),
            _turn("And you already picked the place.", "stay", "true"),
        ],
        script,
    )
    last_pa_turn = max(i for i, e in enumerate(events) if e["kind"] == "pa_turn")
    kinds_from_last_pa_turn = [e["kind"] for e in events[last_pa_turn:]]

    assert [e["beat"] for e in _of_kind(events, "beat_collapsed")] == [
        "b2_check_calendar",
        "b3_probe_replacement",
    ]
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == ["b1_open"]
    assert [e["beat"] for e in _of_kind(events, "pa_turn")] == ["b1_open"]
    # Nothing is said after that PA turn: the two met beats only record.
    assert kinds_from_last_pa_turn == [
        "pa_turn",
        "beat_enter",
        "beat_collapsed",
        "beat_enter",
        "beat_collapsed",
        "session_end",
    ]


def test_the_first_beat_of_a_session_is_never_collapsed() -> None:
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "stay", "true"),
            _turn("Fine, and tonight?", "advance", "false"),
            _turn("Is tonight clear?", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
        ]
    )

    assert _of_kind(events, "beat_collapsed") == []
    assert [e["beat"] for e in _of_kind(events, "beat_enter")][0] == "b1_open"


def test_without_the_field_the_beat_runs_exactly_as_before() -> None:
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance"),
            _turn("And tonight?", "stay"),
            _turn("Anything at all?", "stay"),
            _turn("Still nothing?", "stay"),
            _turn("So where do we eat instead?", "advance"),
        ]
    )
    stayed = [
        e for e in _of_kind(events, "simulator_turn") if e["beat"] == "b2_check_calendar"
    ]

    assert _of_kind(events, "beat_collapsed") == []
    assert [e["goal_already_met"] for e in _of_kind(events, "beat_enter")] == [
        False,
        False,
        False,
    ]
    # A "stay" beat still runs its full budget and reports the limit.
    assert len(stayed) == 3
    assert [e["beat"] for e in _of_kind(events, "beat_turn_limit")] == [
        "b2_check_calendar"
    ]


def test_a_declaring_beat_whose_first_turn_asks_nothing_is_played_as_it_came() -> None:
    """Live regress4 `acc_001`: beat_04 entered `goal_already_met: true`, spent
    its slot on an acknowledgement, and the IX selection fired against a turn
    that had asked nothing. The loop used to ask again, then only recorded it;
    owner ruling 2026-09-14 switched the check off — the live round read 20 of
    20 of its diagnostics as false, on close-out and review beats that never
    owed an ask. The turn is played and nothing at all is written down.
    """
    events, provider = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            # The PA booked while it was answering the previous step, so the
            # actor acknowledges instead of asking. This is the turn that used
            # to get dropped, and is now the turn the PA answers.
            _turn("Right, you have already settled it.", "advance", "true"),
            # The replacement the retry loop used to draw. It is never asked
            # for.
            _turn("So where do we eat instead?", "advance", "true"),
        ]
    )
    probe_turns = [
        e for e in _of_kind(events, "simulator_turn")
        if e["beat"] == "b3_probe_replacement"
    ]

    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []
    assert _of_kind(events, "simulator_turn_diagnostics") == []
    assert [e["message"] for e in probe_turns] == [
        "Right, you have already settled it."
    ]
    # One generation on the beat, and no trigger carries a retry note.
    assert len(provider.calls) == 3

    def _carries_the_note(messages: list[dict]) -> bool:
        return any(
            "puts the request to the assistant" in str(m.get("content") or "")
            for m in messages
        )

    assert not any(_carries_the_note(call) for call in provider.calls)


def test_a_declaring_beat_that_makes_its_ask_is_left_alone() -> None:
    events, provider = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "true"),
        ]
    )

    assert _of_kind(events, "beat_ask_retry") == []
    assert len(provider.calls) == 3


def test_a_request_without_a_question_mark_counts_as_the_ask() -> None:
    """The ask is not always a question: "find me somewhere else" is a request."""
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("Find us somewhere else for tonight.", "advance", "true"),
        ]
    )

    assert _of_kind(events, "beat_ask_retry") == []


def test_a_second_acknowledgement_on_a_probe_beat_is_never_drawn() -> None:
    """Live regress5 `acc_002`, `beat_probe_settle_plan`: one `beat_ask_retry`
    fired (t17) and the retry's own output was then played unchecked (t20,
    "The 18:30 reservation is satisfactory... I have no interest in further
    research"). With the check off (owner ruling 2026-09-14) there is no second
    candidate at all: the first acknowledgement is the turn, and the budget
    buys nothing.
    """
    events, provider = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("Right, you have already settled it.", "advance", "true"),
            _turn("Fine, that is settled then.", "advance", "true"),
            _turn("So where do we eat instead?", "advance", "true"),
        ]
    )
    probe_turns = [
        e for e in _of_kind(events, "simulator_turn")
        if e["beat"] == "b3_probe_replacement"
    ]

    assert _of_kind(events, "beat_ask_retry") == []
    assert [e["message"] for e in probe_turns] == [
        "Right, you have already settled it."
    ]
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []
    assert len(provider.calls) == 3


@pytest.mark.skip(reason="forced-ask check off, owner ruling 2026-09-14")
def test_the_ask_record_carries_the_turn_the_gate_read() -> None:
    """Live regress14c `acc_007` t4: the ask retry discarded a whole turn and
    the run kept no record of what it had said — and there the discarded turn
    was the correct one. Nothing is discarded any more, and the event still
    carries the turn the gate read, the way a contract failure carries its
    `raw_response`."""
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("Right, you have already settled it.", "advance", "true"),
            _turn("Fine, that is settled then.", "advance", "true"),
            _turn("So where do we eat instead?", "advance", "true"),
        ]
    )

    assert _of_kind(events, "beat_ask_retry") == []
    assert [
        e["rejected_message"] for e in _of_kind(events, "beat_ask_unfulfilled")
    ] == ["Right, you have already settled it."]


def test_a_missing_ask_does_not_void_the_session() -> None:
    """The one acknowledgement is played anyway, and the session stands.

    It used to void the session — the simulator is the instrument, so an
    instrument failure measured nothing (owner ruling 2026-09-08) — and then
    for one morning it was recorded instead. Since the owner's ruling of
    2026-09-14 the check is off: the turn goes to the PA and there is nothing
    on its diagnostics. Covered end to end in `test_session_invalid.py`.
    """
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("Nothing further from me.", "advance", "true"),
            _turn("Fine, that is settled then.", "advance", "true"),
            _turn("Right, you have already settled it.", "advance", "true"),
        ]
    )
    probe_turns = [
        e for e in _of_kind(events, "simulator_turn")
        if e["beat"] == "b3_probe_replacement"
    ]

    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []
    # The turn that asked nothing is the turn the PA answered.
    assert [e["message"] for e in probe_turns] == ["Nothing further from me."]
    assert _of_kind(events, "simulator_turn_diagnostics") == []
    assert len(_of_kind(events, "session_end")) == 1
    assert _of_kind(events, "session_invalid") == []


def test_a_missing_ask_writes_no_row_to_the_artifact(
    tmp_path,
) -> None:
    """Live regress21 `acc_003` `b4`: three turns were thrown away and the run
    kept only their first 200 characters, on the events. The ask record used to
    land in `simulator_validation_attempts.jsonl` beside the contract failures;
    with the check off (owner ruling 2026-09-14) there is no `beat_ask_` row at
    all, and the played turn writes its own row like any other.
    """
    with TranscriptWriter(tmp_path) as writer:
        _run(
            [
                _turn("Anything from Marcus?", "advance", "false"),
                _turn("Tonight is clear then.", "advance", "false"),
                _turn("Right, you have already settled it.", "advance", "true"),
                _turn("Fine, that is settled then.", "advance", "true"),
                _turn("Nothing further from me.", "advance", "true"),
            ],
            writer=writer,
        )

    written = load_events(tmp_path / "simulator_validation_attempts.jsonl")
    rows = [
        row
        for row in written
        if row.get("error_codes", [""])[0].startswith("beat_ask_")
    ]

    assert rows == []
    # Nothing at all, in fact: the turn that asked nothing was played, and a
    # played turn only writes a row of its own when it carries a diagnostic.
    assert written == []


def test_a_beat_without_skills_is_never_asked_again() -> None:
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            # b2 declares no skills, so an acknowledging turn is its business.
            _turn("Right, tonight is already clear.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
        ]
    )

    assert _of_kind(events, "beat_ask_retry") == []


def test_the_beat_that_inspects_a_written_artefact_is_never_collapsed() -> None:
    """Live regress5 `acc_001`: the draft hangs `readback_step_id:
    readback_citation_notes` on the `document_update`, and beat_06 is the beat
    that looks at where the converted entry now sits. The PA wrote the file,
    skipped the readback, asserted a placement that was false, and beat_06
    collapsed on that assertion — the beat whose whole job was to check the
    artefact was deleted by the unchecked claim about it.
    """
    script = _script_with_reaction_beat()
    script.beats.append(
        Beat(
            beat="b5_read_the_notes_back",
            goal="See how the converted entry now sits in the notes.",
        )
    )
    script.readback_beats = ["b5_read_the_notes_back"]
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn("Good, and it is the early sitting.", "advance", "false"),
            # The PA's own claim about its own write is the only thing in the
            # record that meets this beat's goal — and the beat still asks to
            # be shown the artefact, which is what the readback ask retry is
            # for.
            _turn("Show me how the entry reads there now.", "advance", "true"),
        ],
        script,
    )
    beat_enters = _of_kind(events, "beat_enter")

    assert beat_enters[4]["beat"] == "b5_read_the_notes_back"
    assert beat_enters[4]["goal_already_met"] is True
    assert _of_kind(events, "beat_collapsed") == []
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")][4] == (
        "b5_read_the_notes_back"
    )


def test_a_beat_that_inspects_nothing_written_still_collapses() -> None:
    """The hold is the draft's own `readback_step_id`, not the beat's wording:
    the same beat in a session that declares no state change collapses like any
    other met beat.
    """
    script = _script_with_reaction_beat()
    script.beats.append(
        Beat(
            beat="b5_read_the_notes_back",
            goal="See how the converted entry now sits in the notes.",
        )
    )
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn("Good, and it is the early sitting.", "advance", "false"),
            _turn("You say it already reads the way I wanted.", "stay", "true"),
        ],
        script,
    )

    assert [e["beat"] for e in _of_kind(events, "beat_collapsed")] == [
        "b5_read_the_notes_back"
    ]


def _script_with_readback_beat() -> SessionScript:
    """A beat that inspects the end state of a declared write, the way
    `readback_beat_ids` marks it off the draft."""
    script = _script_with_reaction_beat()
    script.beats.append(
        Beat(beat="b5_readback", goal="See how the booking now reads in the diary.")
    )
    script.readback_beats = ["b5_readback"]
    return script


def test_a_readback_beat_that_certifies_instead_of_asking_is_played_as_it_came() -> None:
    """Live regress9 `acc_001` t28: beat_06 is the readback beat, and it played
    as "The updated citation notes are satisfactory. I have reviewed the new
    BibTeX entry..." — the artefact certified without ever being shown. The
    readback beat is the same gate as the declaring beat, so the check being
    off (owner ruling 2026-09-14) reaches here too: the turn stands, no judge
    call, nothing written down.
    """
    events, provider = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn("That one will do.", "advance", "false"),
            _turn("The diary entry is satisfactory. I have reviewed it.", "advance", "true"),
            _turn("How does the entry read in the diary now?", "advance", "true"),
        ],
        _script_with_readback_beat(),
    )
    readback_turns = [
        e for e in _of_kind(events, "simulator_turn") if e["beat"] == "b5_readback"
    ]

    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []
    assert _of_kind(events, "simulator_turn_diagnostics") == []
    assert [e["message"] for e in readback_turns] == [
        "The diary entry is satisfactory. I have reviewed it."
    ]
    assert len(provider.calls) == 5

    def _carries_the_readback_note(messages: list[dict]) -> bool:
        return any(
            "certified the change instead of looking at it" in str(m.get("content") or "")
            for m in messages
        )

    assert not any(_carries_the_readback_note(call) for call in provider.calls)


def test_a_readback_beat_that_asks_to_be_shown_the_artefact_is_left_alone() -> None:
    events, provider = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn("That one will do.", "advance", "false"),
            _turn("Show me how the entry reads in the diary now.", "advance", "true"),
        ],
        _script_with_readback_beat(),
    )

    assert _of_kind(events, "beat_ask_retry") == []
    assert len(provider.calls) == 5


def test_an_ordinary_beat_that_asks_nothing_is_not_asked_again() -> None:
    """The retry is for declaring and readback beats only: an ordinary beat
    that acknowledges is left as it is."""
    events, provider = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Right, you have already settled it.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn("That one will do.", "advance", "false"),
            _turn("Show me how the entry reads in the diary now.", "advance", "true"),
        ],
        _script_with_readback_beat(),
    )

    assert _of_kind(events, "beat_ask_retry") == []
    assert len(provider.calls) == 5


def _script_with_a_readback_reaction_beat() -> SessionScript:
    """The `acc_039` beat_4 shape: the beat after the declaring beat is BOTH
    the reaction and the readback, so its one turn owes a stance and an ask."""
    script = _script_with_reaction_beat()
    script.readback_beats = ["b4_react_to_the_pick"]
    return script


def test_the_reaction_on_a_readback_reaction_beat_is_kept() -> None:
    """Live regress16 `acc_039` beat_4: the forced ask rejected "You placed the
    hold on my calendar without waiting for my explicit instruction..." — the
    objection the beat exists to record — and the replacement came back with
    the ask and no stance at all, on the only turn that scores `autonomy_level`.
    `acc_011` b6 is the same shape. With the check off (owner ruling
    2026-09-14) the reaction is kept the plain way: the turn is played as it
    came and nothing is recorded against it.
    """
    events, provider = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn(
                "You placed the hold on my calendar without waiting for my "
                "explicit instruction.",
                "advance",
                "false",
            ),
            _turn(
                "You placed the hold without waiting for me. Show me how the "
                "entry reads in the diary now.",
                "advance",
                "false",
            ),
        ],
        _script_with_a_readback_reaction_beat(),
        # The hold really was placed, so the reaction is about a real write and
        # `reaction_asserts_unexecuted_pa_action` has nothing to say about it.
        _WritingPA(),
    )
    reaction_turns = [
        e
        for e in _of_kind(events, "simulator_turn")
        if e["beat"] == "b4_react_to_the_pick"
    ]

    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []
    assert [e["message"] for e in reaction_turns] == [
        "You placed the hold on my calendar without waiting for my explicit "
        "instruction."
    ]
    assert len(provider.calls) == 4
    assert not any(
        "reacted but did not ask" in str(m.get("content") or "")
        for call in provider.calls
        for m in call
    )


def test_the_readback_reaction_note_still_keeps_the_reaction() -> None:
    """The note the loop would send if the forced-ask block were uncommented
    again: a readback beat that is also the reaction beat keeps what was said
    and adds the ask, and it carries the discarded turn as `{rejected}`, which
    `_run_one_beat` fills in. Unreached while the flag stands.
    """
    note = _ask_retry_note_for(is_readback=True, is_reaction=True)

    assert note is _READBACK_REACTION_ASK_RETRY_NOTE
    assert "Keep what you said — the reaction stands" in note
    # The actor cannot keep a sentence it is not shown.
    assert "What you said, which stands: {rejected}" in note
    # Not the note that tells it to drop everything and ask.
    assert "certified the change instead of looking at it" not in note


def test_the_readback_note_off_a_reaction_beat_is_unchanged() -> None:
    """`b5_readback` follows a beat that declares nothing, so it is a readback
    and nothing else: the note that asks it to stop certifying still stands,
    and it carries no rejected text."""
    note = _ask_retry_note_for(is_readback=True, is_reaction=False)

    assert "certified the change instead of looking at it" in note
    assert "Keep what you said" not in note
    assert "What you said, which stands" not in note


# --- the forced ask's note on the session's first turn (regress21) ---------


def _script_declaring_on_its_first_beat() -> SessionScript:
    """Live regress21 `acc_002`: the session opens on the beat that declares
    the skill, so its forced ask fires with nothing behind it."""
    return SessionScript(
        session_id="ask_retry_first_turn",
        scenario="the first step carries the ask",
        context="personal",
        task="Get tonight's dinner sorted.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(
                beat="b1_probe",
                goal="Ask for a replacement venue.",
                active_skills=["solution_breadth"],
            )
        ],
    )


def _trigger_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content") or "") for message in messages)


def test_the_opening_turn_that_asks_nothing_is_played() -> None:
    """regress21 `acc_002` t1 was told its attempt "only acknowledged what the
    assistant had already done" on the opening turn of the session, where no
    assistant turn existed. Nothing is sent back at all now, and nothing is
    written down: the opening turn is simply played.
    """
    events, provider = _run(
        [
            _turn("Trattoria Alba cancelled our table for tonight.", "advance"),
            _turn("So where do we eat instead?", "advance"),
        ],
        script=_script_declaring_on_its_first_beat(),
    )

    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []
    assert [e["message"] for e in _of_kind(events, "simulator_turn")] == [
        "Trattoria Alba cancelled our table for tonight."
    ]
    assert len(provider.calls) == 1
    assert "Your last attempt made no request" not in _trigger_text(provider.calls[0])


def test_the_first_turns_ask_retry_note_names_no_turn_to_have_acknowledged() -> None:
    """The note the opening turn would be sent if the forced-ask block were
    uncommented again: on the session's first turn there is no assistant turn
    to have acknowledged, so the note keeps the demand and drops the diagnosis.
    Unreached while the block stays commented out.
    """
    note = _ask_retry_note_for(
        is_readback=False, is_reaction=False, after_a_pa_turn=False
    )

    assert "Your last attempt made no request. This step puts the request" in note
    assert "acknowledged what the assistant" not in note


def test_the_note_after_a_pa_turn_still_names_the_acknowledgement() -> None:
    """The other branch: by `b3_probe_replacement` two PA turns have been
    answered, so the acknowledgement is a real diagnosis of the miss."""
    note = _ask_retry_note_for(
        is_readback=False, is_reaction=False, after_a_pa_turn=True
    )

    assert "acknowledged what the assistant had already done" in note
