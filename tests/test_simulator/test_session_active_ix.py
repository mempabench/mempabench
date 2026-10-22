from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from pa.bus.events import OutboundMessage
from simulator.loop import SimulatorLoop
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    PersonaIdentity,
    PreferenceMatrix,
    ProactiveTrigger,
    SessionScript,
)
from simulator.skills import load_actor_skill_spec


class _ScriptedSimulatorProvider(LLMProvider):
    def __init__(self, messages: list[str]):
        super().__init__()
        self.messages = list(messages)

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self.messages:
            raise AssertionError("no scripted simulator response left")
        text = self.messages.pop(0)
        # "advance" keeps the scripted 1-turn-per-beat flow; "stay" (the
        # default) would make SimulatorLoop repeat the beat up to 3 times.
        return LLMResponse(content=(
            f"<message>{text}</message>\n"
            "<factual_check>[]</factual_check>\n"
            "<turn_assessment>{\"next_beat\":\"advance\"}</turn_assessment>"
        ))


def test_proactive_actor_skills_use_environment_trigger_definition() -> None:
    for setting in ("Low", "High"):
        body = load_actor_skill_spec("proactive_outreach", setting)
        normalized = " ".join(body.split())
        assert "This covers initiating contact, and nothing else" in normalized
        assert "state change" in normalized


class _RecordingPALoop:
    def __init__(self, setting: str = "Self-directed") -> None:
        self.setting = setting
        self.active_ix_by_turn: list[list[str]] = []
        self.metadata_by_turn: list[dict[str, Any]] = []
        self.timestamps: list[datetime | None] = []

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> OutboundMessage:
        metadata = metadata or {}
        self.timestamps.append(timestamp)
        self.metadata_by_turn.append(dict(metadata))
        self.active_ix_by_turn.append(list(metadata.get("active_ix") or []))
        tool_events = []
        if self.active_ix_by_turn[-1] == ["autonomy_level"]:
            tool_events.append({
                "ix": True,
                "name": "IX_autonomy_level",
                "attribute": "autonomy_level",
                "setting": self.setting,
            })
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="PA response",
            metadata={"tool_events": tool_events},
        )


class _ProactivePALoop:
    def __init__(self, decision: str) -> None:
        self.decision = decision
        self.events: list[dict[str, Any]] = []
        self.direct_metadata: list[dict[str, Any]] = []

    async def process_environment_event(self, **kwargs: Any) -> OutboundMessage:
        self.events.append(kwargs)
        message = "The pickup window closes at 6 PM today." if self.decision == "send" else ""
        return OutboundMessage(
            channel="benchmark",
            chat_id="direct",
            content=message,
            metadata={
                "outreach_decision": self.decision,
                "outreach_delivered": self.decision == "send",
                "tool_events": [
                    {
                        "ix": True,
                        "name": "IX_proactive_outreach",
                        "attribute": "proactive_outreach",
                        "setting": "Low" if self.decision == "suppress" else "High",
                        "outreach_action": self.decision,
                        "outreach_message": message,
                    }
                ],
            },
        )

    async def process_direct_message(
        self, *args: Any, **kwargs: Any
    ) -> OutboundMessage:
        self.direct_metadata.append(dict(kwargs.get("metadata") or {}))
        return OutboundMessage(channel="benchmark", chat_id="direct", content="Got it.")


def test_active_ix_is_exposed_once_per_session_selection() -> None:
    async def run():
        identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
        matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
        session = SessionScript(
            session_id="session_test",
            scenario="test autonomy preference",
            context="personal",
            beats=[
                Beat(beat="open", active_skills=["autonomy_level"]),
                Beat(beat="react", active_skills=[]),
            ],
        )
        pa_loop = _RecordingPALoop()

        benchmark_session_timestamp = datetime(
            2026, 10, 1, 9, 0, tzinfo=timezone.utc
        )
        loop = SimulatorLoop(
            _ScriptedSimulatorProvider(["Walk me through the plan.", "react"]),
            identity,
            matrix,
            session,
            pa_loop,
            benchmark_session_timestamp=benchmark_session_timestamp,
            memory_retrieval_max_tokens=1234,
            memory_retrieval_max_items=6,
        )
        events = await loop.run_session()

        assert pa_loop.active_ix_by_turn == [
            ["autonomy_level"],
            [],
        ]
        assert pa_loop.timestamps == [
            benchmark_session_timestamp,
            benchmark_session_timestamp + timedelta(minutes=1),
        ]
        assert pa_loop.metadata_by_turn[0]["benchmark_turn_timestamp"] == (
            "2026-10-01T09:00:00+00:00"
        )
        beat_enters = [event for event in events if event["kind"] == "beat_enter"]
        pa_turns = [event for event in events if event["kind"] == "pa_turn"]
        assert [event["benchmark_turn_timestamp"] for event in pa_turns] == [
            "2026-10-01T09:00:00+00:00",
            "2026-10-01T09:01:00+00:00",
        ]
        assert beat_enters[0]["active_skills"] == ["autonomy_level"]
        assert beat_enters[1]["active_skills"] == []
        assert beat_enters[1]["session_active_skills"] == ["autonomy_level"]
        assert beat_enters[1]["active_ix_tools"] == []
        assert pa_loop.metadata_by_turn[0]["run_id"] == "session_test"
        assert pa_loop.metadata_by_turn[0]["persona_id"] == identity.name
        assert pa_loop.metadata_by_turn[0]["script_session_id"] == "session_test"
        assert (
            pa_loop.metadata_by_turn[0]["interaction_context"]
            == "personal"
        )
        assert pa_loop.metadata_by_turn[0]["active_preference_attributes"] == [
            "autonomy_level"
        ]
        assert pa_loop.metadata_by_turn[0]["memory_retrieval_query"] == (
            "Walk me through the plan."
        )
        assert pa_loop.metadata_by_turn[0]["memory_retrieval_budget_tokens"] == 1234
        assert pa_loop.metadata_by_turn[0]["memory_retrieval_budget_items"] == 6
        # ix_context carries elicitation + session_active_ix + the beat and
        # turn index the wrapper's own records name (no autonomy_decision_scope
        # since decision_scope was deleted). The turn index counts up: it
        # reaches `AgentRunSpec.turn_index`, and every record stamped with it
        # read 0 for the whole run while it was missing.
        assert pa_loop.metadata_by_turn[0]["ix_context"] == {
            "capability_limit_attempts": [],
            "elicitation": {},
            "session_active_ix": ["autonomy_level"],
            "beat": "open",
            "turn_index": 0,
        }
        assert pa_loop.metadata_by_turn[1]["ix_context"] == {
            "capability_limit_attempts": [],
            "elicitation": {},
            "session_active_ix": ["autonomy_level"],
            "beat": "react",
            "turn_index": 1,
        }

    asyncio.run(run())


def test_one_ix_selection_retires_the_tool_for_the_session() -> None:
    """A selection satisfies the required IX call and the tool is not re-offered
    later in the same session, whichever setting was chosen."""
    async def run():
        identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
        matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
        session = SessionScript(
            session_id="session_one_selection",
            scenario="test autonomy preference",
            context="personal",
            beats=[
                Beat(beat="open", active_skills=["autonomy_level"]),
                Beat(beat="react", active_skills=["autonomy_level"]),
            ],
        )
        pa_loop = _RecordingPALoop(setting="Self-directed")
        loop = SimulatorLoop(
            _ScriptedSimulatorProvider(
                ["Walk me through the plan.", "Now show me the other option."]
            ),
            identity,
            matrix,
            session,
            pa_loop,
        )
        events = await loop.run_session()
        pa_turns = [event for event in events if event["kind"] == "pa_turn"]
        assert pa_turns[0]["ix_called"] == ["IX_autonomy_level"]
        assert pa_turns[0]["ix_missing"] == []
        assert pa_loop.active_ix_by_turn == [["autonomy_level"], []]
        assert pa_turns[1]["ix_required"] == []

    asyncio.run(run())


def _proactive_session(*, beats: list[Beat] | None = None, **overrides) -> SessionScript:
    if beats is None:
        beats = [
            Beat(beat="opening_after_send", active_skills=["proactive_outreach"],
                 entry_condition="The PA's contact has arrived."),
            Beat(beat="opening_after_suppress", active_skills=["proactive_outreach"],
                 entry_condition="No contact came; the user brings the pickup up on their own."),
            Beat(beat="shared_followup", entry_condition="The pickup plan is on the table."),
        ]
    return SessionScript(**{
        "session_id": "proactive_test",
        "scenario": "A pickup deadline occurs after the earlier conversation ended.",
        "context": "personal",
        "runtime_contract_class": "triggered_interaction",
        "proactive_trigger": ProactiveTrigger(
            trigger_id="pickup_deadline",
            activation_type="deadline",
            background="The comic order is ready for pickup.",
            detail="The store closes in two hours.",
            user_impact="User A may want time to collect the order today.",
        ),
        "beats": beats,
        **overrides,
    })


START = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def _clocked_loop(provider, pa_loop, session, *, deadline_minutes: int = 60):
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    return SimulatorLoop(
        provider, identity, matrix, session, pa_loop,
        benchmark_session_timestamp=START,
        benchmark_session_deadline=START + timedelta(minutes=deadline_minutes),
        suppress_resume_delay=timedelta(minutes=20),
    )


def test_proactive_suppress_resumes_with_the_user_after_the_harness_delay() -> None:
    async def run():
        provider = _ScriptedSimulatorProvider(["Did the comic order come in?", "Okay."])
        pa_loop = _ProactivePALoop("suppress")
        loop = _clocked_loop(provider, pa_loop, _proactive_session())

        events = await loop.run_session()

        kinds = [event["kind"] for event in events]
        assert kinds[:5] == [
            "session_start", "environment_trigger", "pa_turn", "outreach_path", "beat_enter",
        ]
        assert events[2]["outreach_decision"] == "suppress"
        assert events[2]["outbound"] == ""
        assert events[3] == {
            "t": 3, "kind": "outreach_path", "path": "suppress",
            "opening_beat": "opening_after_suppress", "skipped_beat": "opening_after_send",
            "resume_at": "2026-10-01T09:20:00+00:00",
        }
        entered = [event["beat"] for event in events if event["kind"] == "beat_enter"]
        assert entered == ["opening_after_suppress", "shared_followup"]
        # The user's first message lands trigger + 20 min; later turns add a minute.
        assert [m["benchmark_turn_timestamp"] for m in pa_loop.direct_metadata] == [
            "2026-10-01T09:20:00+00:00", "2026-10-01T09:21:00+00:00",
        ]
        assert events[-1]["beat_count"] == 2
        assert pa_loop.events[0]["metadata"]["active_preference_attributes"] == [
            "proactive_outreach"
        ]
        # PA-side keys keep the legacy names but carry the renamed trigger
        # fields (activation_type -> source_type, background -> prior_thread).
        assert events[1]["source_type"] == "deadline"
        assert pa_loop.events[0]["prior_thread"] == "The comic order is ready for pickup."

    asyncio.run(run())


def test_proactive_send_plays_the_send_opening_and_skips_the_other() -> None:
    async def run():
        provider = _ScriptedSimulatorProvider(["Why are you messaging me about that?", "Fine."])
        pa_loop = _ProactivePALoop("send")
        loop = _clocked_loop(provider, pa_loop, _proactive_session())

        events = await loop.run_session()

        proactive_turn = next(event for event in events if event.get("proactive_event"))
        assert proactive_turn["outreach_decision"] == "send"
        assert proactive_turn["outbound"] == "The pickup window closes at 6 PM today."
        path = next(event for event in events if event["kind"] == "outreach_path")
        assert (path["path"], path["opening_beat"], path["skipped_beat"], path["resume_at"]) == (
            "send", "opening_after_send", "opening_after_suppress", None,
        )
        entered = [event["beat"] for event in events if event["kind"] == "beat_enter"]
        assert entered == ["opening_after_send", "shared_followup"]
        assert pa_loop.direct_metadata[0]["benchmark_turn_timestamp"] == "2026-10-01T09:01:00+00:00"
        assert pa_loop.direct_metadata[0]["active_ix"] == []
        assert events[-1]["beat_count"] == 2

    asyncio.run(run())


def test_a_session_that_cannot_fit_its_envelope_is_refused_before_the_first_turn() -> None:
    # §3.3 part 6: the worst case is costed at construction, so an envelope that
    # cannot hold it fails here rather than halfway through the session.
    with pytest.raises(RuntimeError, match="cannot fit its in-world envelope"):
        _clocked_loop(
            _ScriptedSimulatorProvider(["Did the comic order come in?"]),
            _ProactivePALoop("suppress"),
            _proactive_session(),
            deadline_minutes=15,
        )


def test_session_fails_loudly_when_a_turn_reaches_the_deadline() -> None:
    # The runtime guard stays: the pre-check trusts the injected budget, so a
    # turn that still lands on the deadline has to blow up rather than drift.
    async def run():
        provider = _ScriptedSimulatorProvider(["Did the comic order come in?"])
        loop = _clocked_loop(provider, _ProactivePALoop("suppress"), _proactive_session())
        loop.benchmark_session_deadline = START + timedelta(minutes=15)
        with pytest.raises(RuntimeError, match="ran past its envelope"):
            await loop.run_session()

    asyncio.run(run())


def test_triggered_session_on_the_clock_needs_the_harness_resume_delay() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    with pytest.raises(ValueError, match="suppress_resume_delay"):
        SimulatorLoop(
            _ScriptedSimulatorProvider([]), identity, matrix, _proactive_session(),
            _ProactivePALoop("suppress"), benchmark_session_timestamp=START,
        )
    with pytest.raises(ValueError, match="deadline must follow"):
        SimulatorLoop(
            _ScriptedSimulatorProvider([]), identity, matrix, _proactive_session(),
            _ProactivePALoop("suppress"), benchmark_session_timestamp=START,
            benchmark_session_deadline=START, suppress_resume_delay=timedelta(minutes=20),
        )


def test_opening_beats_are_required_for_triggers_and_forbidden_elsewhere() -> None:
    with pytest.raises(ValueError, match="must open with the beats"):
        _proactive_session(beats=[Beat(beat="would_react")])
    with pytest.raises(ValueError, match="must open with the beats"):
        _proactive_session(beats=[
            Beat(beat="opening_after_suppress"), Beat(beat="opening_after_send"),
        ])
    with pytest.raises(ValueError, match="belong to sessions with proactive_trigger"):
        SessionScript(session_id="s", context="work",
                      beats=[Beat(beat="opening_after_send")])
