"""Jump-on-limit: a stalled beat skips ahead to the next preference beat.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_beat_jump.py
"""

from __future__ import annotations

import asyncio

from simulator.loop import SimulatorLoop
from simulator.output import SimulatorOutput, TurnAssessment
from simulator.prompting import _render_user_trigger
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

from pa.bus.events import OutboundMessage


class _ScriptedProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "scripted-model"

    async def chat(self, *args, **kwargs) -> LLMResponse:
        raise AssertionError("the scripted backend never calls the provider")


class _RecordingPA:
    def __init__(self) -> None:
        self.received: list[str] = []

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
    ) -> OutboundMessage:
        self.received.append(user_message)
        return OutboundMessage(channel="cli", chat_id="direct", content="PA reply")


def _identity() -> PersonaIdentity:
    return PersonaIdentity(name="User A", speech_style={})


def _matrix() -> PreferenceMatrix:
    return PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")


class _TriggerRecordingBackend:
    """Scripted backend that also records the trigger each turn was given."""

    name = "trigger-recording"

    def __init__(self, messages: list[str], next_beats: list[str] | None = None):
        self.messages = list(messages)
        self.next_beats = list(next_beats or [])
        self.triggers: list[tuple[str, str]] = []

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        history = kwargs.get("history")
        self.triggers.append(
            (beat.beat, _render_user_trigger(session, beat, history))
        )
        next_beat = self.next_beats.pop(0) if self.next_beats else "advance"
        return SimulatorOutput(
            message=f"{self.messages.pop(0)}, can you check?",
            turn_assessment=TurnAssessment(next_beat=next_beat),
            raw_response="scripted",
        )

    async def drain_memory_tasks(self) -> None:
        return None


def _script(beats: list[Beat]) -> SessionScript:
    return SessionScript(
        session_id="jump_on_limit",
        scenario="jump on limit",
        context="personal",
        task="Get the thing resolved.",
        life_context=LifeContext(),
        task_facts={},
        beats=beats,
    )


def _beats() -> list[Beat]:
    return [
        Beat(beat="open", goal="Get the check started.", active_skills=[]),
        Beat(beat="middle_a", goal="Dictate the new note.", active_skills=[]),
        Beat(beat="middle_b", goal="Have the note read back.", active_skills=[]),
        Beat(
            beat="wrap",
            goal="Have the resolution summarised.",
            active_skills=["verbosity"],
            if_not_met="If nothing was written, ask for the comparison instead.",
        ),
    ]


def test_turn_limit_skips_to_the_next_preference_beat() -> None:
    async def run():
        backend = _TriggerRecordingBackend(
            ["open 1", "open 2", "open 3", "middle_b 1", "wrap 1"],
            next_beats=["stay", "stay", "stay", "advance", "advance"],
        )
        pa = _RecordingPA()
        script = _script(_beats())

        loop = SimulatorLoop(
            _ScriptedProvider(), _identity(), _matrix(), script, pa, backend=backend
        )
        events = await loop.run_session()

        # The jump stops one beat short of the preference beat, so `middle_b`
        # still runs and only `middle_a` is given up on.
        assert [e["beat"] for e in events if e["kind"] == "beat_enter"] == [
            "open",
            "middle_b",
            "wrap",
        ]
        assert [e["beat"] for e in events if e["kind"] == "simulator_turn"] == [
            "open",
            "open",
            "open",
            "middle_b",
            "wrap",
        ]

        skipped = [e for e in events if e["kind"] == "beat_skipped"]
        assert [e["beat"] for e in skipped] == ["middle_a"]
        assert [e["goal"] for e in skipped] == ["Dictate the new note."]
        assert {e["reason"] for e in skipped} == {
            "jump_to_preference_beat_after_turn_limit"
        }

        # The turn-limit event is still emitted, before the skips.
        kinds = [e["kind"] for e in events]
        assert kinds.index("beat_turn_limit") < kinds.index("beat_skipped")

        # The give-up sentence lands on the beat the jump landed on, and only
        # on its first turn.
        middle_b_triggers = [t for beat, t in backend.triggers if beat == "middle_b"]
        assert (
            "You gave up on earlier steps: Dictate the new note."
            in middle_b_triggers[0]
        )
        assert "'If that has not happened' line of this step" in middle_b_triggers[0]

        assert script.skipped_beats == ["middle_a"]
        assert script.pending_jump_note == ""

    asyncio.run(run())


def test_the_beat_before_the_preference_beat_is_never_skipped() -> None:
    """Live `acc_001`: beat_03 read the paper and beat_04 probed on what it
    read. The jump ate beat_03, so the PA answered the probe about a source it
    had never opened. The beat immediately before a preference beat is what
    that beat's `entry_condition` is written against, so the jump stops short
    of it — here there is nothing left between, and nothing is skipped.
    """

    async def run():
        beats = _beats()
        del beats[1]  # open, middle_b, wrap
        backend = _TriggerRecordingBackend(
            ["open 1", "open 2", "open 3", "middle_b 1", "wrap 1"],
            next_beats=["stay", "stay", "stay", "advance", "advance"],
        )
        script = _script(beats)
        loop = SimulatorLoop(
            _ScriptedProvider(),
            _identity(),
            _matrix(),
            script,
            _RecordingPA(),
            backend=backend,
        )
        events = await loop.run_session()

        assert not any(e["kind"] == "beat_skipped" for e in events)
        assert [e["beat"] for e in events if e["kind"] == "beat_enter"] == [
            "open",
            "middle_b",
            "wrap",
        ]
        assert script.skipped_beats == []

    asyncio.run(run())


def test_the_give_up_note_is_not_repeated_on_later_turns_of_the_landing_beat() -> None:
    async def run():
        backend = _TriggerRecordingBackend(
            ["open 1", "open 2", "open 3", "middle_b 1", "middle_b 2", "wrap 1"],
            next_beats=["stay", "stay", "stay", "stay", "advance", "advance"],
        )
        script = _script(_beats())
        loop = SimulatorLoop(
            _ScriptedProvider(),
            _identity(),
            _matrix(),
            script,
            _RecordingPA(),
            backend=backend,
        )
        await loop.run_session()

        landing_triggers = [t for beat, t in backend.triggers if beat == "middle_b"]
        assert len(landing_triggers) == 2
        assert "You gave up on earlier steps" in landing_triggers[0]
        assert "You gave up on earlier steps" not in landing_triggers[1]

    asyncio.run(run())


def test_no_jump_when_no_later_beat_carries_a_preference() -> None:
    async def run():
        beats = _beats()
        beats[3] = Beat(beat="wrap", goal="Close it out.", active_skills=[])
        backend = _TriggerRecordingBackend(
            ["open 1", "open 2", "open 3", "a", "b", "c"],
            next_beats=["stay", "stay", "stay", "advance", "advance", "advance"],
        )
        script = _script(beats)
        loop = SimulatorLoop(
            _ScriptedProvider(),
            _identity(),
            _matrix(),
            script,
            _RecordingPA(),
            backend=backend,
        )
        events = await loop.run_session()

        assert not any(e["kind"] == "beat_skipped" for e in events)
        assert [e["beat"] for e in events if e["kind"] == "beat_enter"] == [
            "open",
            "middle_a",
            "middle_b",
            "wrap",
        ]
        assert script.skipped_beats == []

    asyncio.run(run())


def test_no_jump_when_the_preference_beat_is_simply_the_next_one() -> None:
    async def run():
        beats = [
            Beat(beat="open", goal="Get the check started.", active_skills=[]),
            Beat(
                beat="wrap",
                goal="Have the resolution summarised.",
                active_skills=["verbosity"],
                if_not_met="Ask for the comparison instead.",
            ),
        ]
        backend = _TriggerRecordingBackend(
            ["open 1", "open 2", "open 3", "wrap 1"],
            next_beats=["stay", "stay", "stay", "advance"],
        )
        script = _script(beats)
        loop = SimulatorLoop(
            _ScriptedProvider(),
            _identity(),
            _matrix(),
            script,
            _RecordingPA(),
            backend=backend,
        )
        events = await loop.run_session()

        assert not any(e["kind"] == "beat_skipped" for e in events)
        wrap_trigger = next(t for beat, t in backend.triggers if beat == "wrap")
        assert "You gave up on earlier steps" not in wrap_trigger

    asyncio.run(run())


def test_the_jump_lands_on_a_reveal_beat_instead_of_skipping_it() -> None:
    """Live `acc_039`: with `reveal_at` beats held out of the collapse path,
    beat_04 then died to this jump instead (`transcript.jsonl` t=15). The
    November 20 slot was never spoken and beat_05 went on to "verify" a booking
    that did not exist.
    """

    async def run():
        beats = _beats()
        beats.insert(2, Beat(beat="reveal", goal="Name the November slot."))
        backend = _TriggerRecordingBackend(
            ["open 1", "open 2", "open 3", "reveal 1", "middle_b 1", "wrap 1"],
            next_beats=["stay", "stay", "stay", "advance", "advance", "advance"],
        )
        script = _script(beats)
        script.reveal_at = {"uncalendared_constraint": "reveal"}

        loop = SimulatorLoop(
            _ScriptedProvider(),
            _identity(),
            _matrix(),
            script,
            _RecordingPA(),
            backend=backend,
        )
        events = await loop.run_session()

        # `middle_a` is still given up on; the reveal beat is where the jump
        # lands, and it plays.
        assert [e["beat"] for e in events if e["kind"] == "beat_skipped"] == [
            "middle_a"
        ]
        assert [e["beat"] for e in events if e["kind"] == "beat_enter"] == [
            "open",
            "reveal",
            "middle_b",
            "wrap",
        ]
        assert [e["beat"] for e in events if e["kind"] == "simulator_turn"] == [
            "open",
            "open",
            "open",
            "reveal",
            "middle_b",
            "wrap",
        ]
        # The give-up note lands on the reveal beat, which is now the landing.
        reveal_triggers = [t for beat, t in backend.triggers if beat == "reveal"]
        assert "You gave up on earlier steps: Dictate the new note." in (
            reveal_triggers[0]
        )

    asyncio.run(run())


def test_a_reveal_beat_that_is_already_the_landing_changes_nothing() -> None:
    async def run():
        beats = _beats()
        backend = _TriggerRecordingBackend(
            ["open 1", "open 2", "open 3", "middle_b 1", "wrap 1"],
            next_beats=["stay", "stay", "stay", "advance", "advance"],
        )
        script = _script(beats)
        script.reveal_at = {"uncalendared_constraint": "middle_b"}

        loop = SimulatorLoop(
            _ScriptedProvider(),
            _identity(),
            _matrix(),
            script,
            _RecordingPA(),
            backend=backend,
        )
        events = await loop.run_session()

        assert [e["beat"] for e in events if e["kind"] == "beat_skipped"] == [
            "middle_a"
        ]

    asyncio.run(run())
