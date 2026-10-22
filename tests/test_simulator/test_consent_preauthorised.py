"""The pre-authorisation entry on the consent sideband.

The floor must not ask again for an act the user handed over in the turn
immediately before. Live regress19 `acc_039`: t40 gives the go-ahead for the
booking, t41 holds it and asks again. A spoken go-ahead answers no held id, so
nothing in the actor's own `<consent_decisions>` carries it; the loop appends
one synthetic entry naming the tool instead, on the authorising beat only.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_consent_preauthorised.py
"""

from __future__ import annotations

import asyncio

from simulator.loop import SimulatorLoop
from simulator.output import parse_simulator_output
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

from pa.bus.events import OutboundMessage

# Live `acc_039` beat_5, the beat the go-ahead belongs to.
AUTHORISING_INSTRUCTION = (
    "Goes over the clinic appointment details put in front of him and gives "
    "the go-ahead for the Dr. Patel booking at the clinic."
)
AUTHORISING_GOAL = "Authorize the clinic booking for the agreed checkup date and time."
# The go-ahead carries an ask of its own so that the declaring beat's
# `require_ask` retry never fires and both tests can speak the same line.
GO_AHEAD = (
    "Proceed with booking the appointment for November 10 at 08:30 and place "
    "the corresponding hold on my calendar. Which room did they put us in?"
)
# An ask, and no hand-over: "first" is a withholding frame.
ASKS_ONLY = "What did the clinic say about the 08:30 slot, before anything else?"
GATE_TOOL = "appointment_create"


def _raw(message: str, consent_body: str | None = None) -> str:
    blocks = [
        f"<message>\n{message}\n</message>",
        "<factual_check>\n</factual_check>",
        "<turn_assessment>\nnext_beat: advance\n</turn_assessment>",
    ]
    if consent_body is not None:
        blocks.append(f"<consent_decisions>\n{consent_body}\n</consent_decisions>")
    return "\n\n".join(blocks)


class _ScriptedProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "scripted-model"

    async def chat(self, *args, **kwargs) -> LLMResponse:
        raise AssertionError("the scripted backend never calls the provider")


class _ScriptedBackend:
    name = "preauth"

    def __init__(self, raws: list[str]) -> None:
        self.raws = list(raws)

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        return parse_simulator_output(self.raws.pop(0))

    async def drain_memory_tasks(self) -> None:
        return None


class _ScriptedHeldPA:
    """PA whose held items are whatever the script says for that turn."""

    def __init__(self, held_by_turn: list[list[dict]]) -> None:
        self.held_by_turn = list(held_by_turn)
        self.calls: list[dict] = []

    async def process_direct_message(self, user_message, session_key, **kwargs):
        self.calls.append(
            {"user_message": user_message, "metadata": dict(kwargs.get("metadata") or {})}
        )
        held = self.held_by_turn.pop(0) if self.held_by_turn else []
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content=(
                "Item i3: book the Dr. Patel appointment now?"
                if held
                else "Nothing waiting on you."
            ),
            metadata={"tool_events": [], "held_items": held},
        )


def _loop(
    backend,
    pa,
    *,
    authorised_tool: str | None = GATE_TOOL,
    authorising: bool = True,
    with_middle: bool = False,
) -> SimulatorLoop:
    last = (
        Beat(
            beat="authorise",
            actor_instruction=AUTHORISING_INSTRUCTION,
            goal=AUTHORISING_GOAL,
        )
        if authorising
        else Beat(
            beat="authorise",
            actor_instruction="Asks what the clinic said.",
            goal="Have the PA report what the clinic said.",
        )
    )
    script = SessionScript(
        session_id="preauth_test",
        scenario="clinic booking",
        context="personal",
        task="Get the checkup booked.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(
                beat="open",
                actor_instruction="Asks the PA to look the clinic slots up.",
                goal="Have the PA report the open slots.",
                active_skills=["autonomy_level"],
            ),
            *(
                [
                    Beat(
                        beat="settle",
                        actor_instruction="Settles which slot to take.",
                        goal="Have the PA state the slot it will use.",
                    )
                ]
                if with_middle
                else []
            ),
            last,
        ],
        authorised_tool=authorised_tool,
    )
    return SimulatorLoop(
        _ScriptedProvider(),
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        script,
        pa,
        backend=backend,
    )


def _run(loop) -> list[dict]:
    return asyncio.run(loop.run_session())


def _consent(pa) -> list[list[dict]]:
    return [call["metadata"]["consent_decisions"] for call in pa.calls]


PREAUTH = {"interaction_item_id": "", "decision": "allow", "tool": GATE_TOOL}


def test_a_spoken_go_ahead_on_the_authorising_beat_preauthorises_the_tool() -> None:
    backend = _ScriptedBackend([_raw(ASKS_ONLY), _raw(GO_AHEAD)])
    pa = _ScriptedHeldPA([[], []])
    loop = _loop(backend, pa)
    events = _run(loop)

    assert _consent(pa) == [[], [PREAUTH]]
    # And it is on the record the reads work from.
    preauth = [e for e in events if e["kind"] == "consent_preauthorised"]
    assert [(e["beat"], e["tool"]) for e in preauth] == [("authorise", GATE_TOOL)]
    turns = [e for e in events if e["kind"] == "simulator_turn"]
    assert [t["consent_decisions"] for t in turns] == [[], [PREAUTH]]


def test_nothing_is_preauthorised_off_the_authorising_beat() -> None:
    """The go-ahead spoken on the opening beat is not this beat's to give."""
    backend = _ScriptedBackend([_raw(GO_AHEAD), _raw(ASKS_ONLY)])
    pa = _ScriptedHeldPA([[], []])
    loop = _loop(backend, pa)
    events = _run(loop)

    assert _consent(pa) == [[], []]
    assert not [e for e in events if e["kind"] == "consent_preauthorised"]


def test_a_turn_that_does_not_hand_the_work_over_preauthorises_nothing() -> None:
    backend = _ScriptedBackend([_raw(ASKS_ONLY), _raw(ASKS_ONLY)])
    pa = _ScriptedHeldPA([[], []])
    loop = _loop(backend, pa)
    events = _run(loop)

    assert _consent(pa) == [[], []]
    assert not [e for e in events if e["kind"] == "consent_preauthorised"]


def test_a_shown_held_id_leaves_the_answer_to_the_actors_own_decisions() -> None:
    """The PA asked about `i3` in so many words, so the actor answered about
    `i3`. Adding a synthetic allow would answer the same ask twice."""
    backend = _ScriptedBackend(
        [
            _raw(ASKS_ONLY),
            _raw(GO_AHEAD, "- interaction_item_id: i3\n  decision: allow"),
        ]
    )
    pa = _ScriptedHeldPA(
        [[{"interaction_item_id": "i3", "intent": "book the appointment"}], []]
    )
    loop = _loop(backend, pa)
    events = _run(loop)

    assert _consent(pa) == [
        [],
        [{"interaction_item_id": "i3", "decision": "allow"}],
    ]
    assert not [e for e in events if e["kind"] == "consent_preauthorised"]


def test_an_unarmed_script_preauthorises_nothing() -> None:
    backend = _ScriptedBackend([_raw(ASKS_ONLY), _raw(GO_AHEAD)])
    pa = _ScriptedHeldPA([[], []])
    loop = _loop(backend, pa, authorised_tool=None)
    events = _run(loop)

    assert _consent(pa) == [[], []]
    assert not [e for e in events if e["kind"] == "consent_preauthorised"]


def test_a_session_with_no_authorising_beat_preauthorises_nothing() -> None:
    backend = _ScriptedBackend([_raw(ASKS_ONLY), _raw(GO_AHEAD)])
    pa = _ScriptedHeldPA([[], []])
    loop = _loop(backend, pa, authorising=False)
    events = _run(loop)

    assert _consent(pa) == [[], []]
    assert not [e for e in events if e["kind"] == "consent_preauthorised"]


def test_the_actors_own_decisions_are_untouched_by_the_synthetic_one() -> None:
    """One session, both kinds of entry: the actor answers a held id on the
    beat where the PA showed it, and the synthetic entry lands on the
    authorising beat. Real entries keep their two keys; only the synthetic one
    carries a tool, and there is exactly one of it."""
    backend = _ScriptedBackend(
        [
            _raw(ASKS_ONLY),
            _raw(ASKS_ONLY, "- interaction_item_id: i3\n  decision: allow"),
            _raw(GO_AHEAD),
        ]
    )
    pa = _ScriptedHeldPA(
        [[{"interaction_item_id": "i3", "intent": "book the appointment"}], [], []]
    )
    loop = _loop(backend, pa, with_middle=True)
    _run(loop)

    assert _consent(pa) == [
        [],
        [{"interaction_item_id": "i3", "decision": "allow"}],
        [PREAUTH],
    ]
    assert [len([e for e in call if "tool" in e]) for call in _consent(pa)] == [0, 0, 1]
