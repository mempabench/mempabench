"""A PA turn can switch on reception rules no beat cued.

Some attributes only come into play because of what the PA does: a tool result
carrying `capability_limit` puts `capability_boundary` on the table whether or
not the script planned for it. The harness reports those on the PA turn as
`triggered_skills`, sticky for the rest of the session. They join the session's
reception set from that turn on — the same set the reception-only block runs
on — but they are not session active IX: no IX tool is cued, no turn budget
grows, no asking line appears.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_triggered_skills.py
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
# user_a's own setting for the attribute, resolved from the matrix, not a default.
CARRIED_BLOCK = "### Skill: capability_boundary = Find and hand off (reception only)"


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


class _TriggeringPA:
    """A PA that reports `triggered_skills` on every turn once it hit the limit."""

    def __init__(self, triggered: list[str] | None) -> None:
        self.triggered = triggered
        self.calls: list[dict] = []

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
        **kwargs,
    ) -> OutboundMessage:
        self.calls.append(metadata or {})
        pa_metadata: dict[str, Any] = {}
        if self.triggered is not None:
            pa_metadata["triggered_skills"] = list(self.triggered)
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="I cannot book that myself, so here is who can.",
            metadata=pa_metadata,
        )


def _script() -> SessionScript:
    return SessionScript(
        session_id="triggered_skills",
        scenario="the assistant hits its limit",
        context="personal",
        task="Get the booking sorted.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(beat="b1_ask", goal="Have the assistant make the booking."),
            Beat(beat="b2_react", goal="Settle what happens instead."),
        ],
    )


def _run(triggered: list[str] | None) -> tuple[list[dict], _ScriptedProvider, _TriggeringPA]:
    async def run():
        provider = _ScriptedProvider([_turn("Book it, then."), _turn("Fine, do that.")])
        pa = _TriggeringPA(triggered)
        loop = SimulatorLoop(
            provider,
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            _script(),
            pa,
            backend=NanobotRunnerSimulatorBackend(provider),
            # A budget that would grow if a triggered attribute were counted.
            turn_budget_fn=lambda active_skills: 2 + len(active_skills),
        )
        return await loop.run_session(), provider, pa

    return asyncio.run(run())


def test_a_triggered_attribute_loads_its_reception_rules_from_the_next_turn() -> None:
    events, provider, pa = _run(["capability_boundary"])
    systems = [messages[0]["content"] for messages in provider.calls]
    triggered_events = [e for e in events if e["kind"] == "skill_triggered"]
    beat_enters = [e for e in events if e["kind"] == "beat_enter"]

    # Nothing before the PA turn that triggered it; the reception rules only,
    # from the next turn on.
    assert CARRIED_BLOCK not in systems[0]
    assert CARRIED_BLOCK in systems[1]
    assert "## Reception skeleton" in systems[1]
    assert "## Production skeleton" not in systems[1]
    assert BEAT_BLOCK_ASK_LINE not in systems[1]

    # Recorded once, however many turns repeat the field.
    assert triggered_events == [
        {
            "t": triggered_events[0]["t"],
            "kind": "skill_triggered",
            "attribute": "capability_boundary",
            "source": "triggered_skills",
        }
    ]
    assert beat_enters[1]["reception_skills"] == ["capability_boundary"]

    # It is reception only: no IX tool, no wider budget, no active IX on the
    # PA call.
    assert beat_enters[1]["session_active_skills"] == []
    assert beat_enters[1]["cued_ix_tools"] == []
    assert beat_enters[1]["session_ix_tools"] == []
    assert pa.calls[1]["active_ix"] == []
    assert pa.calls[1]["active_preference_attributes"] == []
    assert len([e for e in events if e["kind"] == "simulator_turn"]) == 2


def test_a_pa_turn_without_the_field_changes_nothing() -> None:
    events, provider, _ = _run(None)
    systems = [messages[0]["content"] for messages in provider.calls]

    assert [e for e in events if e["kind"] == "skill_triggered"] == []
    assert all("capability_boundary" not in system for system in systems)
    assert [
        e["reception_skills"] for e in events if e["kind"] == "beat_enter"
    ] == [[], []]
