"""The reaction beat inherits the reception rules of the beat that asked.

Live run `acc_002` (persona user_a, `solution_breadth` Low): beat `b4` asked for
a recommendation with `active_skills: [solution_breadth]`, the PA answered with a
ranked shortlist, and the user's reaction landed in `b5`, whose `active_skills`
is empty — so the actor judged the answer with no spec at all and accepted it.
An asked-under attribute is now loaded in reception-only form on exactly that
one beat, the first beat after the one that asked.

It used to stay loaded for the rest of the session, and so did the complaint:
live `acc_035` objected to the confidence tags on its reaction beat
(`t=13`, correct) and again at `t=16`, 19, 21, 24 and 29. The IX selection is
made once, at the declaring beat, so the actor reacts once and then proceeds
with the task under the surface it got.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_reception_carry.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from pa.bus.events import OutboundMessage

from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.loop import SimulatorLoop
from simulator.prompting import (
    POST_REACTION_NOTE,
    assemble_prompt,
    reception_carry_attributes,
)
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

BEAT_BLOCK_ASK_LINE = "This is the step where you ask for what this conversation is about."
CARRY_HEADING = "## Reacting to what you asked for earlier in this conversation"
TAKE_A_POSITION = "this message takes a position on that surface"


def _script() -> SessionScript:
    """The acc_002 shape: a probe beat, its reaction, then two beats past it."""
    return SessionScript(
        session_id="reception_carry",
        scenario="dinner replacement",
        context="personal",
        task="Get a replacement dinner venue for tonight.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(
                beat="b3_review_calendar_commitments",
                goal="Have the assistant check the timing of the walk.",
            ),
            Beat(
                beat="b4_probe_find_replacement",
                goal="Ask for a replacement dinner venue.",
                active_skills=["solution_breadth"],
            ),
            Beat(
                beat="b5_evaluate_recommendation",
                goal="Approve the recommended booking.",
            ),
            Beat(
                beat="b6_confirm_booking",
                goal="Have the assistant message Marcus.",
            ),
        ],
    )


def _fixtures() -> tuple[PersonaIdentity, PreferenceMatrix, SessionScript]:
    return (
        PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml"),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script(),
    )


def _system_for(beat_index: int, session: SessionScript | None = None) -> str:
    identity, matrix, default_session = _fixtures()
    session = session or default_session
    return assemble_prompt(identity, matrix, session, session.beats[beat_index])[0][
        "content"
    ]


def test_the_beat_after_the_asking_beat_gets_the_reception_rules_only() -> None:
    system = _system_for(2)

    assert CARRY_HEADING in system
    assert "### Skill: solution_breadth = Low (reception only)" in system
    assert "## Reception skeleton" in system
    assert "## Trigger signals" in system
    assert "## Production skeleton" not in system
    assert "## Anti-behaviors" not in system
    assert "## Base Def" not in system


def test_a_beat_two_steps_after_the_asking_beat_carries_nothing() -> None:
    """The reaction has been taken on `b5`; `b6` gets on with the task."""
    system = _system_for(3)

    assert CARRY_HEADING not in system
    assert "solution_breadth" not in system
    assert "## Reception skeleton" not in system
    assert POST_REACTION_NOTE in system


def test_a_beat_before_any_asking_beat_carries_nothing() -> None:
    system = _system_for(0)

    assert CARRY_HEADING not in system
    assert "solution_breadth" not in system


def test_the_carried_rules_come_with_the_order_to_take_a_position() -> None:
    """Live, `reception_skills` was loaded on every reaction beat and
    the actor still answered the task only ("The structure is acceptable",
    "Reviews the updates and responds as this person would") — acc_004,
    acc_006, acc_011. The carried block now says the turn has to land on the
    surface, either way it went.
    """
    system = _system_for(2)

    assert TAKE_A_POSITION in system
    assert 'a general "fine" or "acceptable" is not it' in system
    assert "push back, correct it, ask again pointedly, or go short" in system
    assert "is not available on the first turn after it appears" in system
    # The ban on naming the preference stays.
    assert "never the name of the preference or the attribute" in system


def test_the_order_to_take_a_position_binds_on_the_beats_own_skills_too() -> None:
    """Live `acc_007`: the beat that declares the attribute carries no
    reception-only skills — the carry subtracts its own — so the order was
    switched off at the one beat where the surface had already appeared (the
    PA's unprompted ping is the `proactive_outreach` surface, delivered before
    the opening beat). It now binds on whichever reception rules are loaded.
    """
    # The declaring beat: the reception section of its own spec is right there.
    assert TAKE_A_POSITION in _system_for(1)
    # The carried case, unchanged.
    assert TAKE_A_POSITION in _system_for(2)


def test_the_order_to_take_a_position_is_absent_without_any_skills() -> None:
    assert TAKE_A_POSITION not in _system_for(0)


def test_a_beat_that_declares_the_attribute_itself_renders_it_once_in_full() -> None:
    session = _script()
    session.beats[2].active_skills = ["solution_breadth"]
    system = _system_for(2, session)

    assert system.count("Skill: solution_breadth = Low") == 1
    assert CARRY_HEADING not in system
    assert "## Production skeleton" in system
    assert "## Anti-behaviors" in system


def _turn(message: str) -> str:
    return (
        f"<message>\n{message}\n</message>\n\n"
        "<factual_check>\n</factual_check>\n\n"
        "<turn_assessment>\nnext_beat: advance\n</turn_assessment>"
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


def _run() -> tuple[list[dict], _ScriptedProvider, SimulatorLoop]:
    async def run():
        provider = _ScriptedProvider([_turn(f"line {i}, can you check?") for i in range(4)])
        loop = SimulatorLoop(
            provider,
            PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml"),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            _script(),
            _RecordingPA(),
            backend=NanobotRunnerSimulatorBackend(provider),
            # A budget that would grow if the carried attribute were counted.
            turn_budget_fn=lambda active_skills: 2 + len(active_skills),
        )
        events = await loop.run_session()
        return events, provider, loop

    return asyncio.run(run())


def test_the_reception_beat_keeps_its_ask_cue_ix_cues_and_turn_budget() -> None:
    events, provider, loop = _run()
    systems = [messages[0]["content"] for messages in provider.calls]
    beat_enters = [e for e in events if e["kind"] == "beat_enter"]

    # The asking beat is unchanged; the beat after it carries the rules but is
    # still not the step that asks.
    assert BEAT_BLOCK_ASK_LINE in systems[1]
    assert CARRY_HEADING not in systems[1]
    assert BEAT_BLOCK_ASK_LINE not in systems[2]
    assert CARRY_HEADING in systems[2]

    # The carried attribute cues no IX tool and does not widen the budget.
    assert beat_enters[1]["cued_ix_tools"] == ["IX_solution_breadth"]
    assert beat_enters[2]["cued_ix_tools"] == []
    assert [loop._beat_turn_budget(b) for b in loop.script.beats] == [2, 3, 2, 2]


def test_beat_enter_records_the_carried_reception_skills() -> None:
    events, _, _ = _run()
    beat_enters = [e for e in events if e["kind"] == "beat_enter"]

    assert [e["reception_skills"] for e in beat_enters] == [
        [],
        [],
        ["solution_breadth"],
        [],
    ]


def test_only_the_beat_right_after_the_probe_is_the_reaction_beat() -> None:
    """The two grading checks are exempted on exactly this beat, so which beat
    it is has to be decided the same way the collapse guard decides it.
    """
    from simulator.agent.loop import _is_reaction_beat

    script = _script()

    assert [_is_reaction_beat(script, beat) for beat in script.beats] == [
        False,  # b3, before the probe
        False,  # b4, the probe itself
        True,  # b5, the reaction
        False,  # b6, past it
    ]


# ---------------------------------------------------------------------------
# The reaction is taken once: `acc_035`
# ---------------------------------------------------------------------------

SPEND_IT_ON_THE_TASK = "The main content of this step is the task"


def _script_with_two_beats_past_the_reaction() -> SessionScript:
    script = _script()
    script.beats.append(
        Beat(beat="b7_close", goal="Close the thread.")
    )
    return script


def test_the_reception_rules_load_on_the_reaction_beat_and_no_beat_after() -> None:
    """Declaring beat at index 1, so index 2 reacts and 3 and 4 do not."""
    script = _script_with_two_beats_past_the_reaction()

    assert [
        reception_carry_attributes(script, beat) for beat in script.beats
    ] == [[], [], ["solution_breadth"], [], []]


def test_a_post_reaction_prompt_spends_the_step_on_the_task_and_drops_the_rules() -> None:
    script = _script_with_two_beats_past_the_reaction()
    reaction = _system_for(2, script)
    after = [_system_for(3, script), _system_for(4, script)]

    # The reaction beat reads the rules and is not told to let them go.
    assert "## Reception skeleton" in reaction
    assert SPEND_IT_ON_THE_TASK not in reaction
    # Every beat after it: the instruction, and no skeleton to grade against.
    assert SPEND_IT_ON_THE_TASK in POST_REACTION_NOTE
    for system in after:
        assert POST_REACTION_NOTE in system
        assert "## Reception skeleton" not in system
        assert CARRY_HEADING not in system
