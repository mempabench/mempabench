"""The beat-block ask cue renders only on the first turn of a beat.

"This is the step where you ask for what this conversation is about." is a
prompt cue for the turn that poses the request. Re-rendering it on turns 2 and
3 pushed the actor to re-send the same probe instead of reacting to the PA's
answer (run 010, beat `request_resolution`). The nanobot backend rebuilds a
SimulatorAgentLoop for every turn, so the flag is owned by the session loop,
which holds the per-beat turn counter.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_beat_first_turn.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from pa.bus.events import OutboundMessage

from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.loop import SimulatorLoop
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

BEAT_BLOCK_ASK_LINE = "This is the step where you ask for what this conversation is about."


def _turn(message: str, next_beat: str) -> str:
    return (
        f"<message>\n{message}\n</message>\n\n"
        "<factual_check>\n</factual_check>\n\n"
        f"<turn_assessment>\nnext_beat: {next_beat}\n</turn_assessment>"
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
        session_id="first_turn_in_beat",
        scenario="probe then react",
        context="personal",
        task="Get the question answered.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(
                beat="request_resolution",
                goal="Get the question answered.",
                active_skills=["verbosity"],
            )
        ],
    )


def _run(responses: list[str]) -> tuple[list[dict], _ScriptedProvider]:
    async def run():
        provider = _ScriptedProvider(responses)
        loop = SimulatorLoop(
            provider,
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml(
                "data/personas/user_a_preferences.yaml"
            ),
            _script(),
            _RecordingPA(),
            backend=NanobotRunnerSimulatorBackend(provider),
        )
        events = await loop.run_session()
        return events, provider

    return asyncio.run(run())


def test_the_ask_cue_renders_only_on_the_first_turn_of_the_beat() -> None:
    events, provider = _run(
        [
            _turn("Can you tell me which of the two options holds up?", "stay"),
            _turn("That answers it.", "advance"),
        ]
    )

    systems = [messages[0]["content"] for messages in provider.calls]
    assert len(systems) == 2
    assert BEAT_BLOCK_ASK_LINE in systems[0]
    assert BEAT_BLOCK_ASK_LINE not in systems[1]
    assert [
        e["message"] for e in events if e["kind"] == "simulator_turn"
    ] == [
        "Can you tell me which of the two options holds up?",
        "That answers it.",
    ]


def test_a_reaction_turn_is_accepted_without_a_retry() -> None:
    # A second turn that closes instead of asking is a normal reaction now that
    # the runtime probe check is gone; a retry would show as a third call.
    _, provider = _run(
        [
            _turn("Can you tell me which of the two options holds up?", "stay"),
            _turn("That answers it.", "advance"),
        ]
    )

    assert len(provider.calls) == 2
