from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from simulator.memory import MemoryConsolidationError
from pa.bus.events import OutboundMessage

from test_simulator.judge_fake import ScriptedJudgeClient
from harness.transcript import TranscriptWriter, load_events
from simulator.backends import (
    NanobotRunnerSimulatorBackend,
    create_simulator_backend,
)
from simulator.config import SimulatorConfig
from simulator.judge.session import JudgeSession
from simulator.loop import SimulatorLoop, generate_beat_turn
from simulator.output import (
    SESSION_INVALID_RULES,
    SimulatorOutput,
    SimulatorOutputContractError,
    TurnAssessment,
)
from simulator.profiles import SimulatorProfile
from simulator.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from simulator.schemas import Beat, LifeContext, PersonaIdentity, PreferenceMatrix, SessionScript
from simulator.self_report import (
    EmotionEvent,
    SelfReportAnswers,
    SessionEndReflectionResult,
)


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[str | LLMResponse] | None = None):
        super().__init__()
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.block_memory = asyncio.Event()

    def get_default_model(self) -> str:
        return "scripted-model"

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
        self.calls.append({"messages": messages, "tools": tools, "model": model})
        if not self.responses:
            raise AssertionError("no scripted response left")
        response = self.responses.pop(0)
        if isinstance(response, LLMResponse):
            if tools:
                await self.block_memory.wait()
            return response
        return LLMResponse(content=response)


class _ScriptedBackend:
    """Scripted backend. `next_beats` aligns 1:1 with `messages`; when it runs
    out, turns default to "advance" to keep the old 1-turn-per-beat flow —
    "stay" (the SimulatorOutput default) now makes the loop repeat the beat.
    """

    name = "scripted"

    def __init__(self, messages: list[str], next_beats: list[str] | None = None):
        self.messages = list(messages)
        self.next_beats = list(next_beats or [])

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, *args, **kwargs):
        next_beat = self.next_beats.pop(0) if self.next_beats else "advance"
        return SimulatorOutput(
            message=self.messages.pop(0),
            turn_assessment=TurnAssessment(next_beat=next_beat),
            raw_response="scripted",
        )

    async def drain_memory_tasks(self) -> None:
        return None


class _PreferenceRecordingBackend:
    name = "preference-recording"

    def __init__(self):
        self.verbosity_settings: list[str | None] = []

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        self.verbosity_settings.append(matrix.get(session.context, "verbosity").setting)
        return SimulatorOutput(
            message=f"sim {beat.beat}, can you check?",
            turn_assessment=TurnAssessment(next_beat="advance"),
            raw_response="scripted",
        )

    async def drain_memory_tasks(self) -> None:
        return None


class _RecordingPA:
    def __init__(self) -> None:
        self.received: list[str] = []

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> OutboundMessage:
        self.received.append(user_message)
        return OutboundMessage(channel="cli", chat_id="direct", content="PA reply")


class _SelfReporter:
    async def generate(self, **_kwargs):
        return SessionEndReflectionResult(
            reflection="The result was acceptable after one correction.",
            final_emotion=EmotionEvent(
                trigger="PA delivered the final reply",
                reaction="narrow satisfaction",
            ),
            self_report=SelfReportAnswers(
                felt_understood_in_this_context=4,
                shift_perception="pa_finally_got_it",
                felt_effort=2,
                noticed_contradictions="no",
                overall_experience=4,
            ),
        )


def _identity() -> PersonaIdentity:
    return PersonaIdentity(
        name="User A",
        speech_style={},
    )


def _matrix() -> PreferenceMatrix:
    return PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")


def _session(beat: Beat | None = None) -> SessionScript:
    return SessionScript(
        session_id="session_backend_test",
        scenario="backend test",
        context="personal",
        task="Say hello.",
        life_context=LifeContext(),
        task_facts={},
        beats=[beat or Beat(beat="open", active_skills=[])],
    )


def _three_block(message: str, next_beat: str = "advance") -> str:
    # next_beat defaults to "advance": under dynamic advancement, "stay" makes
    # SimulatorLoop repeat the beat (up to MAX_TURNS_PER_BEAT exchanges).
    return (
        f"<message>{message}</message>\n"
        "<factual_check></factual_check>\n"
        f"<turn_assessment>next_beat: {next_beat}</turn_assessment>"
    )


def _memory_response(history_entry: str, memory_update: str = "Updated memory") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_save_memory",
                name="save_memory",
                arguments={
                    "history_entry": history_entry,
                    "memory_update": memory_update,
                },
            )
        ],
    )


def test_nanobot_backend_rejects_forced_line_that_bypasses_contract(
    tmp_path: Path,
) -> None:
    async def run():
        root = tmp_path / "sim_profile"
        profile = SimulatorProfile(
            root=root,
            id="profile_user",
            display_name="Profile User",
            identity_path=Path("data/personas/user_a_identity.yaml"),
        )
        provider = _ScriptedProvider()
        backend = NanobotRunnerSimulatorBackend(provider, profile=profile)
        beat = Beat(beat="forced", line="Forced hello", active_skills=[])

        with pytest.raises(
            SimulatorOutputContractError,
            match="fixed_beat_line_bypasses_contract",
        ):
            await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)
        assert provider.calls == []

    asyncio.run(run())


def test_generate_beat_turn_uses_nanobot_backend_wrapper() -> None:
    async def run():
        provider = _ScriptedProvider([_three_block("Hello from runner")])
        beat = Beat(beat="open", active_skills=[])

        out = await generate_beat_turn(
            _identity(),
            _matrix(),
            _session(beat),
            beat,
            provider,
            model="override-model",
        )

        assert out.message == "Hello from runner"
        assert provider.calls[0]["model"] == "override-model"

    asyncio.run(run())


def test_simulator_loop_uses_injected_backend_and_records_backend_name() -> None:
    async def run():
        provider = _ScriptedProvider()
        backend = _ScriptedBackend(["sim says hi"])
        pa = _RecordingPA()
        beat = Beat(beat="open", active_skills=[])

        loop = SimulatorLoop(
            provider,
            _identity(),
            _matrix(),
            _session(beat),
            pa,
            backend=backend,
        )
        events = await loop.run_session()

        assert provider.calls == []
        assert pa.received == ["sim says hi"]
        assert events[0]["simulator_backend"] == "scripted"
        assert events[0]["model"] == "backend-model"
        assert events[0]["pa_model"] is None
        assert events[0]["simulator_model"] == "backend-model"

    asyncio.run(run())


def test_simulator_loop_records_pa_and_simulator_models_separately() -> None:
    async def run():
        provider = _ScriptedProvider()
        backend = _ScriptedBackend(["sim says hi"])
        pa = _RecordingPA()
        beat = Beat(beat="open", active_skills=[])

        loop = SimulatorLoop(
            provider,
            _identity(),
            _matrix(),
            _session(beat),
            pa,
            backend=backend,
            pa_model="pa-model",
        )
        events = await loop.run_session()

        assert events[0]["model"] == "pa-model"
        assert events[0]["pa_model"] == "pa-model"
        assert events[0]["simulator_model"] == "backend-model"

    asyncio.run(run())


def test_simulator_loop_applies_dynamic_preference_updates() -> None:
    async def run():
        provider = _ScriptedProvider()
        backend = _PreferenceRecordingBackend()
        pa = _RecordingPA()
        script = _session()
        script.beats = [
            Beat(beat="open", active_skills=["verbosity"]),
            Beat(
                beat="after_change",
                active_skills=["verbosity"],
                preference_updates={
                    "personal": {
                        "verbosity": {
                            "setting": "Terse",
                            "source": "scripted_dynamic",
                            "confidence": "High",
                        }
                    }
                },
            ),
        ]

        loop = SimulatorLoop(provider, _identity(), _matrix(), script, pa, backend=backend)
        events = await loop.run_session()

        assert backend.verbosity_settings == ["Detailed", "Terse"]
        updates = [e for e in events if e["kind"] == "preference_update"]
        assert updates[0]["patches"]["personal"]["verbosity"]["setting"] == "Terse"

    asyncio.run(run())


def test_beat_staying_three_turns_hits_cap_then_next_beat_runs() -> None:
    """A beat whose output keeps saying "stay" consumes exactly
    MAX_TURNS_PER_BEAT (3) simulator turns, emits `beat_turn_limit`
    {beat, turns: 3}, and only then does the next beat run."""

    async def run():
        provider = _ScriptedProvider()
        backend = _ScriptedBackend(
            ["open 1", "open 2", "open 3", "wrap 1"],
            next_beats=["stay", "stay", "stay", "advance"],
        )
        pa = _RecordingPA()
        script = _session()
        script.beats = [
            Beat(beat="open", active_skills=[]),
            Beat(beat="wrap", active_skills=[]),
        ]

        loop = SimulatorLoop(
            provider, _identity(), _matrix(), script, pa, backend=backend
        )
        events = await loop.run_session()

        sim_beats = [e["beat"] for e in events if e["kind"] == "simulator_turn"]
        assert sim_beats == ["open", "open", "open", "wrap"]
        assert pa.received == ["open 1", "open 2", "open 3", "wrap 1"]

        limits = [e for e in events if e["kind"] == "beat_turn_limit"]
        assert len(limits) == 1
        assert limits[0]["beat"] == "open"
        assert limits[0]["turns"] == 3

        # beat_enter is emitted once per beat even when the beat repeats, and
        # the cap event lands before the next beat is entered.
        kinds_beats = [(e["kind"], e.get("beat")) for e in events]
        assert [b for k, b in kinds_beats if k == "beat_enter"] == ["open", "wrap"]
        assert kinds_beats.index(("beat_turn_limit", "open")) < kinds_beats.index(
            ("beat_enter", "wrap")
        )

        # session_end counts beats entered, not simulator turns.
        end = events[-1]
        assert end["kind"] == "session_end"
        assert end["simulator_turns"] == 4
        assert end["beat_count"] == 2

    asyncio.run(run())


def test_branch_next_beat_stays_on_current_beat() -> None:
    """"branch:<id>" keeps the conversation on the current beat (branch
    prompts live inside the current beat block) — no re-enter, no cap event
    when the beat advances before MAX_TURNS_PER_BEAT."""

    async def run():
        provider = _ScriptedProvider()
        backend = _ScriptedBackend(
            ["open probe", "open branch reply", "wrap 1"],
            next_beats=["branch:if_pa_wrong", "advance", "advance"],
        )
        pa = _RecordingPA()
        script = _session()
        script.beats = [
            Beat(beat="open", active_skills=[]),
            Beat(beat="wrap", active_skills=[]),
        ]

        loop = SimulatorLoop(
            provider, _identity(), _matrix(), script, pa, backend=backend
        )
        events = await loop.run_session()

        sim_turns = [e for e in events if e["kind"] == "simulator_turn"]
        assert [e["beat"] for e in sim_turns] == ["open", "open", "wrap"]
        assert sim_turns[0]["turn_assessment"]["next_beat"] == "branch:if_pa_wrong"
        assert [e["beat"] for e in events if e["kind"] == "beat_enter"] == [
            "open",
            "wrap",
        ]
        assert not any(e["kind"] == "beat_turn_limit" for e in events)

        end = events[-1]
        assert end["kind"] == "session_end"
        assert end["simulator_turns"] == 3
        assert end["beat_count"] == 2

    asyncio.run(run())


def test_generate_beat_turn_applies_beat_preference_update_to_prompt() -> None:
    async def run():
        provider = _ScriptedProvider([_three_block("Walk me through it after that change.")])
        beat = Beat(
            beat="open",
            active_skills=["verbosity"],
            preference_updates={
                "personal": {
                    "verbosity": {
                        "setting": "Terse",
                        "source": "scripted_dynamic",
                        "confidence": "High",
                    }
                }
            },
        )

        out = await generate_beat_turn(
            _identity(),
            _matrix(),
            _session(beat),
            beat,
            provider,
            model="override-model",
        )

        system = provider.calls[0]["messages"][0]["content"]
        assert out.message == "Walk me through it after that change."
        assert "## Skill: verbosity = Terse" in system

    asyncio.run(run())


def test_nanobot_runner_backend_parses_final_content() -> None:
    async def run():
        provider = _ScriptedProvider([_three_block("Hello from runner")])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(
            _identity(),
            _matrix(),
            _session(beat),
            beat,
            model="runner-model",
        )

        assert out.message == "Hello from runner"
        assert provider.calls[0]["tools"] == []
        assert provider.calls[0]["model"] == "runner-model"

    asyncio.run(run())


def test_nanobot_runner_retries_malformed_blocks_before_accepting() -> None:
    async def run():
        provider = _ScriptedProvider([
            "<message>missing routing</message>",
            _three_block("accepted"),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

        assert out.message == "accepted"
        assert len(out.rejected_attempts) == 1
        assert out.rejected_attempts[0]["error_codes"] == [
            "simulator_output_format_invalid"
        ]
        assert out.rejected_attempts[0]["raw_response"].startswith(
            "<message>missing routing"
        )
        assert len(provider.calls) == 2
        assert any(
            "three-block output format was malformed" in message["content"]
            for message in provider.calls[1]["messages"]
        )

    asyncio.run(run())


def _tool_call_response() -> LLMResponse:
    """What gemini answered memos `acc_034` with: a call to a tool that does
    not exist, on a request that registers no tools at all."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="call_1", name="test_tool", arguments={})],
    )


def test_nanobot_runner_retries_a_hallucinated_tool_call_before_accepting() -> None:
    """The runner used to raise, and the raise happened before this retry loop,
    so one invented tool call killed the whole orchestrator process. It is a
    format failure like an unparseable message: one rejected attempt, then the
    next attempt plays."""
    async def run():
        provider = _ScriptedProvider([
            _tool_call_response(),
            _three_block("accepted"),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

        assert out.message == "accepted"
        assert len(out.rejected_attempts) == 1
        rejected = out.rejected_attempts[0]
        assert "simulator_output_format_invalid" in rejected["error_codes"]
        assert any(
            "test_tool" in warning for warning in rejected["parse_warnings"]
        )
        assert "test_tool" in rejected["retry_note"]
        assert len(provider.calls) == 2
        assert any(
            "you have no tools — never call one" in message["content"]
            for message in provider.calls[1]["messages"]
        )

    asyncio.run(run())


def test_a_tool_call_on_every_attempt_ends_on_the_normal_contract_error() -> None:
    """Past the budget it is refused like any other format failure, and the
    finding it is refused on is one the session loop voids the session for."""
    async def run():
        provider = _ScriptedProvider([_tool_call_response()])
        backend = NanobotRunnerSimulatorBackend(provider, output_contract_retries=0)
        beat = Beat(beat="open", active_skills=[])

        with pytest.raises(SimulatorOutputContractError) as exc_info:
            await backend.generate_turn(
                _identity(), _matrix(), _session(beat), beat
            )

        assert [
            rule for rule in exc_info.value.findings if rule in SESSION_INVALID_RULES
        ] == ["simulator_output_format_invalid"]
        assert any(
            "test_tool" in warning
            for warning in exc_info.value.rejected_attempts[-1]["parse_warnings"]
        )
        assert len(provider.calls) == 1

    asyncio.run(run())


def test_nanobot_runner_retries_bot_like_message_before_accepting() -> None:
    async def run():
        provider = _ScriptedProvider([
            _three_block("Regarding the current beat: the validator expects this."),
            _three_block("Wait, so do I need to bring it or not?"),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

        assert out.message == "Wait, so do I need to bring it or not?"
        assert len(out.rejected_attempts) == 1
        assert out.rejected_attempts[0]["error_codes"] == [
            "authoring_metadata_in_message",
        ]
        assert len(provider.calls) == 2

    asyncio.run(run())


def test_nanobot_runner_fails_after_three_invalid_attempts() -> None:
    async def run():
        # Blank <message> is a hard finding — never accepted, even at the
        # exhausted-retries fallback that tolerates format-only findings.
        invalid = "<message></message><turn_assessment>next_beat: stay</turn_assessment>"
        provider = _ScriptedProvider([invalid, invalid, invalid])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        with pytest.raises(SimulatorOutputContractError) as exc_info:
            await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

        assert len(exc_info.value.rejected_attempts) == 3
        assert len(provider.calls) == 3

    asyncio.run(run())


def test_loop_records_rejected_raw_attempt_outside_normal_transcript(
    tmp_path: Path,
) -> None:
    async def run():
        invalid = "<message>rejected private draft</message>"
        provider = _ScriptedProvider([invalid, _three_block("accepted")])
        backend = NanobotRunnerSimulatorBackend(provider)
        pa = _RecordingPA()
        beat = Beat(beat="open", active_skills=[])
        with TranscriptWriter(tmp_path) as writer:
            loop = SimulatorLoop(
                provider,
                _identity(),
                _matrix(),
                _session(beat),
                pa,
                backend=backend,
                transcript_writer=writer,
            )
            events = await loop.run_session()

        attempts = load_events(tmp_path / "simulator_validation_attempts.jsonl")
        assert attempts[0]["raw_response"] == invalid
        transcript_text = (tmp_path / "transcript.jsonl").read_text(encoding="utf-8")
        assert "rejected private draft" not in transcript_text
        assert pa.received == ["accepted"]
        assert any(
            event["kind"] == "simulator_validation_failure" for event in events
        )

    asyncio.run(run())


def test_loop_writes_self_report_only_as_transcript_event(tmp_path: Path) -> None:
    async def run():
        provider = _ScriptedProvider()
        backend = _ScriptedBackend(["hello"])
        with TranscriptWriter(tmp_path) as writer:
            loop = SimulatorLoop(
                provider,
                _identity(),
                _matrix(),
                _session(),
                _RecordingPA(),
                backend=backend,
                transcript_writer=writer,
                self_reporter=_SelfReporter(),
            )
            events = await loop.run_session()

        kinds = [event["kind"] for event in events]
        assert kinds[-2:] == [
            "session_end_reflection",
            "session_end",
        ]
        assert not (tmp_path / "self_report.yaml").exists()
        reflection = next(
            event for event in events if event["kind"] == "session_end_reflection"
        )
        assert reflection["reflection"] == (
            "The result was acceptable after one correction."
        )
        assert reflection["final_emotion"] == {
            "trigger": "PA delivered the final reply",
            "reaction": "narrow satisfaction",
        }
        assert reflection["self_report"]["felt_understood_in_this_context"] == 4
        assert "model" not in reflection

    asyncio.run(run())


def test_nanobot_runner_backend_uses_profile_prompt_and_nanobot_memory(tmp_path: Path) -> None:
    async def run():
        root = tmp_path / "sim_profile"
        (root / "memory").mkdir(parents=True)
        (root / "TOOLS.md").write_text("Profile tools marker.", encoding="utf-8")
        (root / "memory" / "MEMORY.md").write_text("Profile memory marker.", encoding="utf-8")
        profile = SimulatorProfile(
            root=root,
            id="profile_user",
            display_name="Profile User",
            identity_path=Path("data/personas/user_a_identity.yaml"),
            preference_matrix_path=Path("data/personas/user_a_preferences.yaml"),
        )
        provider = _ScriptedProvider([
            _three_block("Hello from profile"),
            _memory_response(
                "[2026-05-17 12:00] PA asked who the simulated user is; Profile User replied.",
                memory_update="Profile memory marker.\n\n- Profile User answered identity question.",
            ),
        ])
        backend = NanobotRunnerSimulatorBackend(provider, profile=profile)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(
            _identity(),
            _matrix(),
            _session(beat),
            beat,
            history=[{"role": "user", "content": "who are you"}],
            model="runner-model",
            benchmark_timestamp="2026-10-01 09:00",
        )

        system = provider.calls[0]["messages"][0]["content"]
        assert out.message == "Hello from profile"
        assert "Profile tools marker." in system
        assert "Profile memory marker." in system
        assert provider.calls[0]["tools"] == []
        provider.block_memory.set()
        await backend.drain_memory_tasks()
        assert provider.calls[1]["tools"][0]["function"]["name"] == "save_memory"

        history = (root / "memory" / "HISTORY.md").read_text(encoding="utf-8")
        assert "PA asked who the simulated user is" in history
        memory = (root / "memory" / "MEMORY.md").read_text(encoding="utf-8")
        assert "Profile User answered identity question" in memory

    asyncio.run(run())


def test_nanobot_runner_backend_returns_before_memory_consolidation(tmp_path: Path) -> None:
    async def run():
        root = tmp_path / "sim_profile"
        root.mkdir()
        (root / "TOOLS.md").write_text("Profile tools marker.", encoding="utf-8")
        profile = SimulatorProfile(
            root=root,
            id="profile_user",
            display_name="Profile User",
            identity_path=Path("data/personas/user_a_identity.yaml"),
            preference_matrix_path=Path("data/personas/user_a_preferences.yaml"),
        )
        provider = _ScriptedProvider([
            _three_block("Fast reply"),
            _memory_response("[2026-05-17 12:00] Archived later."),
        ])
        backend = NanobotRunnerSimulatorBackend(provider, profile=profile)
        beat = Beat(beat="open", active_skills=[])

        out = await asyncio.wait_for(
            backend.generate_turn(
                _identity(),
                _matrix(),
                _session(beat),
                beat,
                history=[{"role": "user", "content": "hello"}],
                model="runner-model",
                benchmark_timestamp="2026-10-01 09:00",
            ),
            timeout=0.5,
        )

        assert out.message == "Fast reply"
        assert not (root / "memory" / "HISTORY.md").exists()
        provider.block_memory.set()
        await backend.drain_memory_tasks()
        assert "Archived later" in (root / "memory" / "HISTORY.md").read_text(encoding="utf-8")

    asyncio.run(run())


def test_nanobot_runner_backend_writes_one_entry_per_session(tmp_path: Path) -> None:
    """A whole session -> exactly one save_memory call, one HISTORY entry."""

    async def run():
        root = tmp_path / "sim_profile"
        (root / "memory").mkdir(parents=True)
        (root / "memory" / "MEMORY.md").write_text("Old durable note.", encoding="utf-8")
        profile = SimulatorProfile(
            root=root,
            id="profile_user",
            display_name="Profile User",
            identity_path=Path("data/personas/user_a_identity.yaml"),
            preference_matrix_path=Path("data/personas/user_a_preferences.yaml"),
        )
        provider = _ScriptedProvider([
            _three_block("First reply"),
            _three_block("Second reply"),
            _three_block("Third reply"),
            _memory_response(
                "[2026-05-17 12:00] I asked it for help and had to correct it once.",
                memory_update="New durable note.",
            ),
        ])
        backend = NanobotRunnerSimulatorBackend(provider, profile=profile)
        beat = Beat(beat="open", active_skills=[])
        session = _session(beat)

        history: list[dict] = []
        for pa_text in ("first ask", "second ask", "third ask"):
            history.append({"role": "user", "content": pa_text})
            out = await backend.generate_turn(
                _identity(), _matrix(), session, beat,
                history=list(history), model="runner-model",
                benchmark_timestamp="2026-10-01 09:00",
            )
            history.append({"role": "assistant", "content": out.message})

        # Nothing is written until the session closes.
        assert not (root / "memory" / "HISTORY.md").exists()

        provider.block_memory.set()
        await backend.drain_memory_tasks()

        memory_calls = [c for c in provider.calls if c["tools"]]
        assert len(memory_calls) == 1
        # The one call carries every turn of the session, user-side labels.
        prompt = memory_calls[0]["messages"][0]["content"]
        for line in (
            "Assistant: first ask", "Me: First reply",
            "Assistant: second ask", "Me: Second reply",
            "Assistant: third ask", "Me: Third reply",
        ):
            assert line in prompt
        assert "Old durable note." in prompt

        history_text = (root / "memory" / "HISTORY.md").read_text(encoding="utf-8")
        assert history_text.count("[2026-05-17 12:00]") == 1
        assert (root / "memory" / "MEMORY.md").read_text(encoding="utf-8") == "New durable note."

        # Draining again is a no-op: buffers are empty, no second LLM call.
        await backend.drain_memory_tasks()
        assert len([c for c in provider.calls if c["tools"]]) == 1
        assert (root / "memory" / "HISTORY.md").read_text(
            encoding="utf-8"
        ) == history_text

    asyncio.run(run())


def test_nanobot_runner_backend_stamps_history_with_benchmark_time(
    tmp_path: Path,
) -> None:
    """HISTORY carries in-world benchmark time, never the wall clock."""

    class _EchoTimestampProvider(_ScriptedProvider):
        """Writes back whatever session timestamp the prompt carried."""

        async def chat(self, messages, tools=None, **kwargs):
            if not tools:
                return await super().chat(messages, tools=tools, **kwargs)
            prompt = messages[0]["content"]
            self.calls.append({"messages": messages, "tools": tools, "model": None})
            stamp = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]", prompt).group(1)
            return _memory_response(f"[{stamp}] I asked it for the venue list.")

    async def run():
        root = tmp_path / "sim_profile"
        (root / "memory").mkdir(parents=True)
        profile = SimulatorProfile(
            root=root,
            id="profile_user",
            display_name="Profile User",
            identity_path=Path("data/personas/user_a_identity.yaml"),
        )
        provider = _EchoTimestampProvider([_three_block("Reply")])
        backend = NanobotRunnerSimulatorBackend(provider, profile=profile)
        beat = Beat(beat="open", active_skills=[])

        await backend.generate_turn(
            _identity(),
            _matrix(),
            _session(beat),
            beat,
            history=[{"role": "user", "content": "hello"}],
            model="runner-model",
            benchmark_timestamp="2026-10-01 09:00",
        )
        # No timestamp at drain time: the last buffered turn's benchmark
        # timestamp is used.
        await backend.drain_memory_tasks()

        history = (root / "memory" / "HISTORY.md").read_text(encoding="utf-8")
        assert "[2026-10-01 09:00]" in history

    asyncio.run(run())


def test_nanobot_runner_backend_refuses_memory_write_without_benchmark_time(
    tmp_path: Path,
) -> None:
    async def run():
        profile = SimulatorProfile(
            root=tmp_path / "sim_profile",
            id="profile_user",
            display_name="Profile User",
            identity_path=Path("data/personas/user_a_identity.yaml"),
        )
        provider = _ScriptedProvider([_three_block("Reply")])
        beat = Beat(beat="open", active_skills=[])
        backend = NanobotRunnerSimulatorBackend(provider, profile=profile)

        await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)
        with pytest.raises(
            MemoryConsolidationError, match="MISSING_BENCHMARK_TIMESTAMP"
        ):
            await backend.drain_memory_tasks()

    asyncio.run(run())


def test_nanobot_runner_backend_surfaces_completed_memory_failure(tmp_path: Path) -> None:
    async def run():
        profile = SimulatorProfile(
            root=tmp_path / "sim_profile",
            id="profile_user",
            display_name="Profile User",
            identity_path=Path("data/personas/user_a_identity.yaml"),
        )
        provider = _ScriptedProvider([
            _three_block("Fast reply"),
            LLMResponse(content="", finish_reason="error"),
            LLMResponse(content="", finish_reason="error"),
            LLMResponse(content="", finish_reason="error"),
        ])
        provider.block_memory.set()
        backend = NanobotRunnerSimulatorBackend(provider, profile=profile)
        beat = Beat(beat="open", active_skills=[])

        await backend.generate_turn(
            _identity(),
            _matrix(),
            _session(beat),
            beat,
            benchmark_timestamp="2026-10-01 09:00",
        )
        await asyncio.sleep(0)

        with pytest.raises(MemoryConsolidationError, match="INVALID_CONSOLIDATION"):
            await backend.drain_memory_tasks()

    asyncio.run(run())


def test_simulator_config_creates_nanobot_backend() -> None:
    config = SimulatorConfig()
    backend = config.create_backend(_ScriptedProvider())

    assert backend.name == "nanobot"
    assert create_simulator_backend(_ScriptedProvider()).name == "nanobot"


_LONG_SENTENCE = (
    "I need the confirmed venue list today because the committee meets tomorrow morning and every one of them has asked me twice already about which rooms are actually booked and which are still only pencilled in."
)


def test_nanobot_runner_retries_style_finding_before_accepting() -> None:
    async def run():
        provider = _ScriptedProvider([
            _three_block(_LONG_SENTENCE),
            _three_block("Send me the confirmed venue list today."),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

        assert out.message == "Send me the confirmed venue list today."
        assert out.rejected_attempts[0]["error_codes"] == ["sentence_over_35_words"]
        assert out.rejected_attempts[0]["severity"] == "style"
        assert any(
            "over 35 words" in message["content"]
            for message in provider.calls[1]["messages"]
        )

    asyncio.run(run())


def test_nanobot_runner_accepts_style_finding_on_last_attempt() -> None:
    async def run():
        provider = _ScriptedProvider([_three_block(_LONG_SENTENCE)])
        backend = NanobotRunnerSimulatorBackend(provider, output_contract_retries=0)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

        assert out.message == _LONG_SENTENCE
        assert len(provider.calls) == 1
        assert out.rejected_attempts[0]["error_codes"] == ["sentence_over_35_words"]
        assert out.rejected_attempts[0]["accepted_after_retries"] is True

    asyncio.run(run())


def test_nanobot_runner_still_fails_on_hard_finding_at_last_attempt() -> None:
    async def run():
        invalid = "<message></message><turn_assessment>next_beat: stay</turn_assessment>"
        provider = _ScriptedProvider([invalid])
        backend = NanobotRunnerSimulatorBackend(provider, output_contract_retries=0)
        beat = Beat(beat="open", active_skills=[])

        with pytest.raises(SimulatorOutputContractError):
            await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

    asyncio.run(run())


def test_nanobot_runner_records_an_invented_path_and_plays_the_turn() -> None:
    """`names_a_file_you_do_not_know` is a
    judgement about the turn, so it is recorded and the turn is played."""

    async def run():
        provider = _ScriptedProvider([
            _three_block("Look in documents/research/citation_notes.md."),
            _three_block("Look in my citation notes."),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

        assert out.message == "Look in documents/research/citation_notes.md."
        assert len(provider.calls) == 1
        assert out.rejected_attempts == []
        assert out.diagnostics == ["names_a_file_you_do_not_know"]

    asyncio.run(run())


def test_nanobot_runner_accepts_path_present_in_task_facts() -> None:
    async def run():
        provider = _ScriptedProvider([
            _three_block("Look in documents/research/citation_notes.md."),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])
        session = _session(beat)
        session.task_facts = {
            "notes": {"path": "documents/research/citation_notes.md"},
        }

        out = await backend.generate_turn(
            _identity(),
            _matrix(),
            session,
            beat,
            history=[{"role": "user", "content": "Where should I look?"}],
        )

        assert out.message == "Look in documents/research/citation_notes.md."
        assert out.rejected_attempts == []

    asyncio.run(run())


def test_nanobot_runner_records_an_unknown_path_instead_of_voiding() -> None:
    """It used to spend the whole budget and then raise. Record-only, so the
    first attempt plays and no session is voided for it."""

    async def run():
        invalid = _three_block("Look in documents/research/citation_notes.md.")
        provider = _ScriptedProvider([invalid, invalid, invalid])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

        assert out.message == "Look in documents/research/citation_notes.md."
        assert len(provider.calls) == 1
        assert out.rejected_attempts == []
        assert out.diagnostics == ["names_a_file_you_do_not_know"]

    asyncio.run(run())


def test_nanobot_runner_records_a_volunteered_identifier_instead_of_voiding() -> None:
    """Same shape for `identifier_volunteered_unprompted`."""

    async def run():
        invalid = _three_block("Your confirmation code is ABC123XY.")
        provider = _ScriptedProvider([invalid, invalid, invalid])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])
        session = _session(beat)
        session.task_facts = {"confirmation_code": "ABC123XY"}

        out = await backend.generate_turn(_identity(), _matrix(), session, beat)

        assert out.message == "Your confirmation code is ABC123XY."
        assert len(provider.calls) == 1
        assert out.rejected_attempts == []
        assert out.diagnostics == ["identifier_volunteered_unprompted"]

    asyncio.run(run())


def test_nanobot_runner_records_a_soft_only_finding_on_the_first_attempt() -> None:
    """`grades_pa_before_acting` is a style finding, so it is recorded too."""

    async def run():
        invalid = _three_block("Acceptable. That is correct.")
        provider = _ScriptedProvider([invalid, invalid, invalid])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(_identity(), _matrix(), _session(beat), beat)

        assert out.message == "Acceptable. That is correct."
        assert len(provider.calls) == 1
        assert out.rejected_attempts == []
        assert out.diagnostics == ["grades_pa_before_acting"]

    asyncio.run(run())


# The turns below quote the assistant on words it did not write, which arms
# `quoted_span_not_in_pa_turn`. It is diagnostics only and never a retry
# trigger, so each of these turns is PLAYED — and its diagnostic has to
# survive to the record anyway.
_PA_OFFERED_THE_RESERVATION = "Casa Marisol is booked for two at 18:30."
_MISQUOTE_ONE_LINER = 'You said "the terrace table is held for us".'
_MISQUOTE_WITH_A_LONG_SENTENCE = (
    'You said "the terrace table is held for us" in your reply. That is not '
    "what I asked for at all, and I would rather you gave me the short "
    "summary of what is actually held tonight instead of repeating the part "
    "of the booking that was never in question in the first place."
)


def test_accepted_turn_carries_its_diagnostics_on_the_clean_path() -> None:
    """Accept path 1 of 3: no findings at all, returned on the first pass."""
    async def run():
        provider = _ScriptedProvider([_three_block(_MISQUOTE_ONE_LINER)])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(
            _identity(),
            _matrix(),
            _session(beat),
            beat,
            history=[{"role": "user", "content": _PA_OFFERED_THE_RESERVATION}],
        )

        assert out.message == _MISQUOTE_ONE_LINER
        assert out.rejected_attempts == []
        assert out.diagnostics == ["quoted_span_not_in_pa_turn"]

    asyncio.run(run())


def test_accepted_turn_carries_its_diagnostics_when_the_style_budget_runs_out() -> None:
    """Accept path 2 of 3: a style finding is played on the last attempt.

    This is the live acc_003 t15 shape — the retry was for a style
    finding and the diagnostic rode through it.
    """
    async def run():
        provider = _ScriptedProvider([_three_block(_MISQUOTE_WITH_A_LONG_SENTENCE)])
        backend = NanobotRunnerSimulatorBackend(provider, output_contract_retries=0)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(
            _identity(),
            _matrix(),
            _session(beat),
            beat,
            history=[{"role": "user", "content": _PA_OFFERED_THE_RESERVATION}],
        )

        assert out.message == _MISQUOTE_WITH_A_LONG_SENTENCE
        assert out.rejected_attempts[-1]["error_codes"] == ["sentence_over_35_words"]
        assert out.rejected_attempts[-1]["accepted_after_retries"] is True
        assert out.rejected_attempts[-1]["diagnostics"] == ["quoted_span_not_in_pa_turn"]
        assert out.diagnostics == ["quoted_span_not_in_pa_turn"]

    asyncio.run(run())


def test_a_format_invalid_turn_is_refused_once_the_budget_is_spent() -> None:
    """What used to be accept path 3 of 3.

    A three-block output that never parsed was played on its parse fallback:
    the recovered message, and a `next_beat` the actor never wrote. The
    simulator is the instrument, so it is refused and the session loop turns
    the raise into an invalid session.
    """
    async def run():
        malformed = f"<message>{_MISQUOTE_ONE_LINER}</message>"
        provider = _ScriptedProvider([malformed])
        backend = NanobotRunnerSimulatorBackend(provider, output_contract_retries=0)
        beat = Beat(beat="open", active_skills=[])

        with pytest.raises(SimulatorOutputContractError) as exc_info:
            await backend.generate_turn(
                _identity(),
                _matrix(),
                _session(beat),
                beat,
                history=[{"role": "user", "content": _PA_OFFERED_THE_RESERVATION}],
            )

        assert exc_info.value.findings == ["simulator_output_format_invalid"]
        assert exc_info.value.rejected_attempts[-1]["error_codes"] == [
            "simulator_output_format_invalid"
        ]

    asyncio.run(run())


def test_loop_persists_the_accepted_turns_diagnostics(tmp_path: Path) -> None:
    """Same artifact and same `diagnostics` shape as a rejected attempt, so a
    reader can union the two; `accepted` is what tells them apart."""
    async def run():
        provider = _ScriptedProvider([
            _three_block("Find me a venue near the office.", next_beat="stay"),
            _three_block(
                f"{_MISQUOTE_ONE_LINER} What is the venue address?"
            ),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])
        with TranscriptWriter(tmp_path) as writer:
            loop = SimulatorLoop(
                provider,
                _identity(),
                _matrix(),
                _session(beat),
                _RecordingPA(),
                backend=backend,
                transcript_writer=writer,
            )
            events = await loop.run_session()

        rows = load_events(tmp_path / "simulator_validation_attempts.jsonl")
        accepted = [row for row in rows if row.get("accepted")]
        assert len(accepted) == 1
        assert accepted[0]["diagnostics"] == ["quoted_span_not_in_pa_turn"]
        assert accepted[0]["beat"] == "open"
        assert "error_codes" not in accepted[0]
        assert any(
            event["kind"] == "simulator_turn_diagnostics"
            and event["diagnostics"] == ["quoted_span_not_in_pa_turn"]
            for event in events
        )

    asyncio.run(run())


def _fact_session(beat: Beat) -> SessionScript:
    """A session whose one beat reveals a fact it therefore has to say."""
    return SessionScript(
        session_id="session_fact_test",
        scenario="backend test",
        context="personal",
        task="Resolve the quiz challenge.",
        life_context=LifeContext(),
        task_facts={"student_name": "Farid Rahimi"},
        reveal_at={"student_name": "open"},
        beats=[beat],
    )


def test_a_style_retry_note_names_the_facts_the_step_still_owes() -> None:
    """`acc_002`: attempt 3 bought its way under the style cap by
    cutting the facts the beat existed to deliver. The style note now says which
    ones the step still owes, and it comes after the style hint, not instead."""

    async def run():
        provider = _ScriptedProvider([
            _three_block(_LONG_SENTENCE),
            _three_block("Farid Rahimi is disputing question 2."),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(
            _identity(), _matrix(), _fact_session(beat), beat
        )

        assert out.message == "Farid Rahimi is disputing question 2."
        assert out.rejected_attempts[0]["error_codes"] == ["sentence_over_35_words"]
        note = next(
            message["content"]
            for message in provider.calls[1]["messages"]
            if "private output contract" in str(message["content"])
        )
        assert "over 35 words" in note
        # The value, not the bare key: live `acc_002` had the key
        # `marcus_available_from` and wrote "you are available from 18:00".
        assert note.endswith("This step must still say student_name: Farid Rahimi.")
        assert "still say student_name." not in note
        assert note.index("over 35 words") < note.index("still say")

    asyncio.run(run())


_LONG_SENTENCE_WITH_THE_FACT = (
    "Farid Rahimi is disputing question 2. He wants the marking scheme sent "
    "over today because he says the wording of the second part was ambiguous "
    "and he has already spoken to two other students who read it the same way "
    "he did."
)


def test_a_style_retry_note_quotes_the_sentence_that_already_carries_the_fact():
    """Naming the value was not enough. Live `acc_002` t4 had the note with
    `marcus_available_from: Marcus is free from 18:00` spelled out and the
    shortened rewrite still turned it into "you are available from 18:00". So
    the note quotes back the rejected attempt's own sentence and says the cut
    comes from the rest."""

    async def run():
        provider = _ScriptedProvider([
            _three_block(_LONG_SENTENCE_WITH_THE_FACT),
            _three_block("Farid Rahimi is disputing question 2."),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])

        out = await backend.generate_turn(
            _identity(), _matrix(), _fact_session(beat), beat
        )

        note = next(
            message["content"]
            for message in provider.calls[1]["messages"]
            if "private output contract" in str(message["content"])
        )
        assert note.endswith(
            "This step must still say student_name: Farid Rahimi. Keep these "
            'sentences word for word: "Farid Rahimi is disputing question 2." '
            "Cut elsewhere."
        )
        # Only the sentence that carries it: the rest is the cut.
        assert "He wants the marking scheme sent" not in note
        # And on the attempt's own row, so a read can see what the actor was
        # told instead of inferring it from the rewrite.
        # The whole note, style hint first, the way the actor was sent it.
        assert out.rejected_attempts[0]["retry_note"] == (
            "One sentence is over 35 words. Break it up or drop the clauses "
            "you do not need right now. This step must still say "
            "student_name: Farid Rahimi. Keep these sentences word for word: "
            '"Farid Rahimi is disputing question 2." Cut elsewhere.'
        )

    asyncio.run(run())


#: `probe_fact_missing` is the judge's alone: with the judge off
#: the rule is not checked, so a test about it runs one. Every fact answers
#: `no` — the turn does not say it — which is what composes the missing list.
def _fact_judge() -> JudgeSession:
    return JudgeSession(mode="live", client=ScriptedJudgeClient())


def test_a_turn_that_leaves_out_a_revealed_fact_is_recorded_not_said_again() -> None:
    """A diagnostic, not a hard finding.

    Record-only, so the first attempt plays and the key the judge named stays
    on the row: live `acc_027` recorded the diagnostic twice with no key
    and both reads re-derived them by hand.
    """

    async def run():
        provider = _ScriptedProvider([
            _three_block("A student is disputing question 2."),
            _three_block("Farid Rahimi is disputing question 2."),
        ])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])
        judge = JudgeSession(
            mode="live",
            client=ScriptedJudgeClient(
                {"probe_fact_missing:student_name": [("no", "")]}
            ),
        )

        out = await backend.generate_turn(
            _identity(),
            _matrix(),
            _fact_session(beat),
            beat,
            history=[{"role": "user", "content": "Who is the student?"}],
            judge=judge,
        )

        assert len(provider.calls) == 1
        assert out.message == "A student is disputing question 2."
        assert out.rejected_attempts == []
        assert out.diagnostics == ["probe_fact_missing"]
        assert out.diagnostics_detail == {"probe_fact_missing": ["student_name"]}

    asyncio.run(run())


def test_a_fact_still_missing_is_recorded_and_the_session_plays(
    tmp_path: Path,
) -> None:
    """It used to void the session. Record-only:
    the turn reaches the assistant and the rule reaches the artefact the reader
    opens, with the keys it fired on."""

    async def run():
        # Three attempts scripted; only the first is ever asked for now.
        provider = _ScriptedProvider(
            [_three_block("A student is disputing question 2.")] * 3
        )
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])
        with TranscriptWriter(tmp_path) as writer:
            loop = SimulatorLoop(
                provider,
                _identity(),
                _matrix(),
                _fact_session(beat),
                _RecordingPA(),
                backend=backend,
                transcript_writer=writer,
                judge=_fact_judge(),
            )
            await loop.run_session()

        rows = load_events(tmp_path / "simulator_validation_attempts.jsonl")
        assert rows, "the recorded finding has to reach the artefact"
        assert all(row["accepted"] is True for row in rows)
        assert all("error_codes" not in row for row in rows)
        assert rows[0]["diagnostics"] == ["probe_fact_missing"]
        assert rows[0]["diagnostics_detail"] == {
            "probe_fact_missing": ["student_name"]
        }
        # And the turn was handed to the assistant rather than thrown away.
        assert "A student is disputing question 2." in [
            event.get("message")
            for event in loop._events
            if event["kind"] == "simulator_turn"
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    "make_judge",
    [lambda: None, lambda: JudgeSession(mode="off")],
    ids=["no_judge", "judge_off"],
)
def test_with_the_judge_off_a_missing_probe_fact_is_not_checked(
    make_judge, tmp_path: Path
) -> None:
    """The rule is the judge's alone, and making it a hard
    finding did not change that: with no judge, or one that is off, the same
    turn that the judge sends back is played as it stands on its first attempt.
    """

    async def run():
        provider = _ScriptedProvider(
            [_three_block("A student is disputing question 2.")]
        )
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(beat="open", active_skills=[])
        pa = _RecordingPA()
        with TranscriptWriter(tmp_path) as writer:
            loop = SimulatorLoop(
                provider,
                _identity(),
                _matrix(),
                _fact_session(beat),
                pa,
                backend=backend,
                transcript_writer=writer,
                judge=make_judge(),
            )
            events = await loop.run_session()

        # One generation, and the turn that leaves the fact out reaches the
        # assistant.
        assert len(provider.calls) == 1
        assert pa.received == ["A student is disputing question 2."]
        # No finding and no diagnostic: a turn with neither writes no attempt
        # row at all, and nothing names the rule in the events.
        rows = load_events(tmp_path / "simulator_validation_attempts.jsonl")
        assert rows == []
        assert "probe_fact_missing" not in str(events)
        # And no rewrite was asked for: the fixed note never left the backend.
        assert not any(
            "This turn still has to state these facts" in str(message["content"])
            for call in provider.calls
            for message in call["messages"]
        )

    asyncio.run(run())


def test_a_revealed_fact_is_not_owed_until_the_pa_asks_for_it() -> None:
    """A probe beat hands its `reveal_at` facts over on request only, so the
    same turn before any PA question is the draft working, not a defect —
    live `acc_009` fired five times on turns whose `avoid` forbade
    volunteering the fact."""

    async def run():
        provider = _ScriptedProvider([_three_block("A student is disputing question 2.")])
        backend = NanobotRunnerSimulatorBackend(provider)
        beat = Beat(
            beat="open",
            active_skills=[],
            actor_instruction="hands the reason over only when the PA asks for it",
        )

        out = await backend.generate_turn(
            _identity(), _matrix(), _fact_session(beat), beat
        )

        assert out.diagnostics == []

    asyncio.run(run())


# --- history entries carry the transcript `t` ------------------------------
#
# An evidence row indexes `history`; a reader holds `transcript.jsonl`, whose
# `t` counts events. Six live reads took one for the other. The loop
# stamps each history entry with the `t` of the event that carried it, and the
# reflection event says which of the two its `turn` counts.


def test_history_entries_carry_the_transcript_t_of_their_event() -> None:
    class _CapturingSelfReporter(_SelfReporter):
        def __init__(self) -> None:
            self.history: list[dict[str, Any]] = []

        async def generate(self, **kwargs):
            self.history = list(kwargs["history"])
            return await super().generate(**kwargs)

    async def run():
        provider = _ScriptedProvider()
        backend = _ScriptedBackend(["hello"])
        reporter = _CapturingSelfReporter()
        loop = SimulatorLoop(
            provider,
            _identity(),
            _matrix(),
            _session(),
            _RecordingPA(),
            backend=backend,
            self_reporter=reporter,
        )
        events = await loop.run_session()

        by_t = {event["t"]: event for event in events}
        assert [entry["role"] for entry in reporter.history] == [
            "assistant",
            "user",
        ]
        assert by_t[reporter.history[0]["t"]]["kind"] == "simulator_turn"
        assert by_t[reporter.history[1]["t"]]["kind"] == "pa_turn"
        reflection = next(
            event for event in events if event["kind"] == "session_end_reflection"
        )
        assert reflection["turn_index"] == "history"

    asyncio.run(run())


def test_the_transcript_t_stays_out_of_the_provider_payload() -> None:
    """`build_messages` sends the chat pair alone: an extra key on a message
    is a provider-side schema error."""
    from simulator.prompting import chat_pairs

    assert chat_pairs(
        [{"role": "assistant", "content": "said", "t": 4}]
    ) == [{"role": "assistant", "content": "said"}]
