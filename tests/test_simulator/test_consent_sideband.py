"""The PA-blind `<consent_decisions>` sideband — simulator half of T18.

The simulated user knows whether this turn agreed or refused; the wrapper reads
that as gate state. It is a fourth private block, parsed like `<factual_check>`
and `<turn_assessment>`, forwarded as a typed metadata key, and never shown to
the PA in any form.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_consent_sideband.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.loop import SimulatorLoop
from simulator.message_quality import simulator_message_quality_findings
from simulator.output import SimulatorOutput, TurnAssessment, parse_simulator_output
from simulator.profiles import SimulatorProfile
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

GATE_ASK = (
    "The extension request is drafted, addressed to dale@example.com. "
    "Item i3: send it now?"
)


def _raw(message: str, consent_body: str | None) -> str:
    blocks = [
        f"<message>\n{message}\n</message>",
        "<factual_check>\n</factual_check>",
        "<turn_assessment>\nnext_beat: advance\n</turn_assessment>",
    ]
    if consent_body is not None:
        blocks.append(f"<consent_decisions>\n{consent_body}\n</consent_decisions>")
    return "\n\n".join(blocks)


# -- parsing ----------------------------------------------------------------

def test_the_block_parses_into_entries_and_stays_out_of_the_message() -> None:
    out = parse_simulator_output(
        _raw(
            "Yes, send it.",
            "- interaction_item_id: i3\n  decision: allow",
        )
    )
    assert out.message == "Yes, send it."
    assert [(d.interaction_item_id, d.decision) for d in out.consent_decisions] == [
        ("i3", "allow")
    ]
    assert "consent_decisions" not in out.message
    assert "i3" not in out.message


def test_three_ids_can_carry_three_different_decisions() -> None:
    out = parse_simulator_output(
        _raw(
            "Go ahead with the booking, leave the email, I have not decided on the rest.",
            "- interaction_item_id: i1\n  decision: allow\n"
            "- interaction_item_id: i2\n  decision: deny\n"
            "- interaction_item_id: i3\n  decision: unclear",
        )
    )
    assert [(d.interaction_item_id, d.decision) for d in out.consent_decisions] == [
        ("i1", "allow"),
        ("i2", "deny"),
        ("i3", "unclear"),
    ]


def test_missing_is_never_accepted_from_the_simulator() -> None:
    """`missing` is a wrapper-side audit value, not something the simulated user
    can claim: accepting it would hand the model a way to say "I did not decide"
    that is indistinguishable from an absent entry."""
    out = parse_simulator_output(
        _raw("Not now.", "- interaction_item_id: i3\n  decision: missing")
    )
    assert out.consent_decisions == []
    assert any("not in ['allow', 'deny', 'unclear']" in w for w in out.parse_warnings)


def test_malformed_entries_are_dropped_with_a_warning() -> None:
    out = parse_simulator_output(
        _raw(
            "Fine.",
            "- decision: allow\n"
            "- interaction_item_id: i2\n  decision: maybe\n"
            "- interaction_item_id: i4\n  decision: allow",
        )
    )
    assert [(d.interaction_item_id, d.decision) for d in out.consent_decisions] == [
        ("i4", "allow")
    ]
    assert len([w for w in out.parse_warnings if w.startswith("consent_decisions")]) == 2


def test_an_absent_block_is_the_normal_case_and_not_a_format_finding() -> None:
    out = parse_simulator_output(_raw("Where did you look?", None))
    assert out.consent_decisions == []
    assert not [w for w in out.parse_warnings if w.startswith("consent_decisions")]
    assert "simulator_output_format_invalid" not in simulator_message_quality_findings(
        out.message, out.parse_warnings
    )


def test_an_empty_block_is_an_empty_list() -> None:
    out = parse_simulator_output(_raw("Nothing to approve here.", ""))
    assert out.consent_decisions == []
    assert not [w for w in out.parse_warnings if w.startswith("consent_decisions")]


def test_the_block_is_stripped_when_the_message_wrapper_is_missing() -> None:
    """Access-boundary fallback: a model that forgot `<message>` must not have
    its private blocks handed to the PA as dialogue."""
    raw = (
        "Send it.\n\n<factual_check>\n</factual_check>\n\n"
        "<turn_assessment>\nnext_beat: advance\n</turn_assessment>\n\n"
        "<consent_decisions>\n- interaction_item_id: i3\n  decision: allow\n"
        "</consent_decisions>"
    )
    out = parse_simulator_output(raw)
    assert out.message == "Send it."
    assert "consent_decisions" not in out.message
    assert "i3" not in out.message
    assert [d.decision for d in out.consent_decisions] == ["allow"]


# -- the leak check ---------------------------------------------------------

def test_a_leaked_consent_block_in_the_message_is_a_hard_finding() -> None:
    leaked = (
        "Yes, send it. <consent_decisions>- interaction_item_id: i3\n"
        "  decision: allow</consent_decisions>"
    )
    findings = simulator_message_quality_findings(leaked)
    assert "consent_block_in_message" in findings


def test_a_clean_message_does_not_trip_the_leak_check() -> None:
    assert "consent_block_in_message" not in simulator_message_quality_findings(
        "Yes, send it."
    )


# -- the metadata channel and PA blindness ----------------------------------

class _ScriptedProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "scripted-model"

    async def chat(self, *args, **kwargs) -> LLMResponse:
        raise AssertionError("the scripted backend never calls the provider")


class _ConsentBackend:
    """Replays one raw response per turn through the real parser."""

    name = "consent"

    def __init__(self, raws: list[str]) -> None:
        self.raws = list(raws)
        self.histories: list[list[dict]] = []
        self.triggers: list[str] = []

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        self.histories.append(list(kwargs.get("history") or []))
        self.triggers.append(
            _render_user_trigger(
                session, beat, kwargs.get("history"), kwargs.get("held_items") or ()
            )
        )
        return parse_simulator_output(self.raws.pop(0))

    async def drain_memory_tasks(self) -> None:
        return None


class _HeldItemsPA:
    """A PA turn that carries held items: rendered prose plus IX metadata.

    It holds on the first turn and lets go once the user has answered: a PA
    that held forever would (correctly) never let the beat advance, because a
    beat is not over while the PA is waiting on this user.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def process_direct_message(self, user_message, session_key, **kwargs):
        self.calls.append(
            {"user_message": user_message, "metadata": dict(kwargs.get("metadata") or {})}
        )
        holding = len(self.calls) == 1
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content=GATE_ASK,
            metadata={
                "tool_events": [],
                # What the harness (P4) translates out of the render result.
                "held_items": (
                    [{"interaction_item_id": "i3", "intent": "send the extension request"}]
                    if holding else []
                ),
                "ix_render": {
                    "schema_version": "ix_render/1.0.0",
                    "held": {
                        "plan_remaining": [
                            {"id": "a3", "intent": "send the extension request"}
                        ],
                        "outreach_message": "SECRET_HELD_BODY",
                    },
                    "episodes": {
                        "autonomy_level": {"episode_id": "p1", "rounds_open": 1}
                    },
                },
            },
        )


def _script(beats: list[Beat]) -> SessionScript:
    return SessionScript(
        session_id="consent_test",
        scenario="consent sideband",
        context="personal",
        task="Get the extension request out.",
        life_context=LifeContext(),
        task_facts={},
        beats=beats,
    )


def _loop(backend, pa) -> SimulatorLoop:
    return SimulatorLoop(
        _ScriptedProvider(),
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script(
            [
                Beat(beat="open", active_skills=["autonomy_level"]),
                Beat(beat="close", active_skills=[]),
            ]
        ),
        pa,
        backend=backend,
    )


def test_the_sideband_reaches_the_pa_call_as_metadata_and_never_as_text() -> None:
    async def run():
        backend = _ConsentBackend(
            [
                _raw("Draft it and check with me first.", None),
                _raw("No, do not send that.", "- interaction_item_id: i3\n  decision: deny"),
                _raw("That is everything.", None),
            ]
        )
        pa = _HeldItemsPA()
        loop = _loop(backend, pa)
        events = await loop.run_session()

        # It travels on the same typed channel as active_ix.
        assert pa.calls[0]["metadata"]["consent_decisions"] == []
        assert pa.calls[1]["metadata"]["consent_decisions"] == [
            {"interaction_item_id": "i3", "decision": "deny"}
        ]
        # And never as anything the PA reads as text.
        for call in pa.calls:
            assert "consent_decisions" not in call["user_message"]
            assert "i3" not in call["user_message"]
        assert pa.calls[1]["user_message"] == "No, do not send that."

        # It is on record for the judge. Two turns, not three: `open` is the
        # declaring beat and this script names no authorising beat, so the
        # `advance` its first turn routed is honoured and the i3 hold is
        # carried into `close` (`beat_advanced_with_hold`). The answer lands
        # there instead, on the same typed channel — which is what this test is
        # about; the beat it lands on is the beat-pacing rule's business.
        turns = [e for e in events if e["kind"] == "simulator_turn"]
        assert [t["consent_decisions"] for t in turns] == [
            [],
            [{"interaction_item_id": "i3", "decision": "deny"}],
        ]
        assert [e["beat"] for e in turns] == ["open", "close"]

    asyncio.run(run())


def test_the_sideband_never_enters_the_simulators_own_history_or_prompt() -> None:
    async def run():
        backend = _ConsentBackend(
            [
                _raw("Draft it and check with me first.", None),
                _raw("No, do not send that.", "- interaction_item_id: i3\n  decision: deny"),
                _raw("That is everything.", None),
            ]
        )
        loop = _loop(backend, _HeldItemsPA())
        await loop.run_session()

        # The id itself is public: the PA's rendered ask names it, which is how
        # the simulator knows what to decide about. The DECISION is the private
        # half, and it never joins the conversation.
        history_text = json.dumps(loop._history)
        assert "consent_decisions" not in history_text
        assert "deny" not in history_text
        assert "decision" not in history_text
        # The history the model is re-fed each turn carries the same guarantee.
        for history in backend.histories:
            text = json.dumps(history)
            assert "consent_decisions" not in text
            assert "deny" not in text
            assert "decision" not in text

    asyncio.run(run())


def test_a_pa_turn_with_held_items_reaches_the_simulator_as_prose_only() -> None:
    """The wrapper already rendered the ask into the PA's text. The simulator
    must react to that text and must never be handed `ix_render`: the held
    bodies, the episode state and the plan ids are evaluation-side data."""

    async def run():
        backend = _ConsentBackend(
            [
                _raw("Draft it and check with me first.", None),
                _raw("No, do not send that.", "- interaction_item_id: i3\n  decision: deny"),
                _raw("That is everything.", None),
            ]
        )
        loop = _loop(backend, _HeldItemsPA())
        events = await loop.run_session()

        # What the simulator got back is the rendered prose, nothing else.
        entry = dict(loop._history[1])
        # The transcript stamp is the loop's own bookkeeping (regress21), and
        # `build_messages` leaves it behind.
        assert isinstance(entry.pop("t"), int)
        assert entry == {"role": "user", "content": GATE_ASK}
        second_turn_history = json.dumps(backend.histories[1])
        for banned in ("ix_render", "SECRET_HELD_BODY", "episode_id", "rounds_open", "a3"):
            assert banned not in second_turn_history, banned

        # The transcript event does not carry it either: harness/ owns the IX
        # face of the transcript, the simulator does not read PA IX metadata.
        pa_turns = [e for e in events if e["kind"] == "pa_turn"]
        assert pa_turns
        for event in pa_turns:
            assert "ix_render" not in event
            assert "SECRET_HELD_BODY" not in json.dumps(event)

    asyncio.run(run())


# -- held_items: the harness tells the simulator which ids are waiting --------

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
            content=GATE_ASK if held else "Done, nothing waiting on you.",
            metadata={"tool_events": [], "held_items": held},
        )


def test_held_items_put_the_ids_and_intents_into_the_next_turns_prompt() -> None:
    async def run():
        backend = _ConsentBackend(
            [
                _raw("Draft it and check with me first.", None),
                _raw(
                    "Send the Dale one, leave the invoice alone.",
                    "- interaction_item_id: i3\n  decision: allow\n"
                    "- interaction_item_id: i4\n  decision: deny",
                ),
                _raw("That is everything.", None),
            ]
        )
        pa = _ScriptedHeldPA(
            [
                [
                    {"interaction_item_id": "i3", "intent": "send the extension request"},
                    {"interaction_item_id": "i4", "intent": "pay the invoice"},
                ],
                [],
            ]
        )
        loop = _loop(backend, pa)
        await loop.run_session()

        # Turn 1: nothing was held yet, so no instruction.
        assert "holding these items" not in backend.triggers[0]
        # Turn 2: the ids and their intents are stage direction for the actor.
        second = backend.triggers[1]
        assert "The assistant is holding these items until you say" in second
        assert "- i3 (send the extension request)" in second
        assert "- i4 (pay the invoice)" in second
        assert "Never read these ids out loud" in second

        # Both ids were known, so both decisions survive and reach the wrapper.
        assert pa.calls[1]["metadata"]["consent_decisions"] == [
            {"interaction_item_id": "i3", "decision": "allow"},
            {"interaction_item_id": "i4", "decision": "deny"},
        ]

    asyncio.run(run())


def test_an_id_the_harness_never_held_is_dropped_with_a_warning() -> None:
    async def run():
        backend = _ConsentBackend(
            [
                _raw("Draft it and check with me first.", None),
                _raw(
                    "Send the Dale one.",
                    "- interaction_item_id: i3\n  decision: allow\n"
                    "- interaction_item_id: i9\n  decision: allow",
                ),
                _raw("That is everything.", None),
            ]
        )
        pa = _ScriptedHeldPA(
            [[{"interaction_item_id": "i3", "intent": "send the extension request"}], []]
        )
        loop = _loop(backend, pa)
        events = await loop.run_session()

        # A model-invented id can never become consent for a real action.
        assert pa.calls[1]["metadata"]["consent_decisions"] == [
            {"interaction_item_id": "i3", "decision": "allow"}
        ]
        turns = [e for e in events if e["kind"] == "simulator_turn"]
        assert any(
            "dropped unknown interaction_item_id='i9'" in w
            for w in turns[1]["parse_warnings"]
        )

    asyncio.run(run())


def test_with_nothing_held_there_is_no_instruction_and_no_block_survives() -> None:
    async def run():
        backend = _ConsentBackend(
            [
                _raw("Draft it and check with me first.", None),
                # A block with nothing holding it open is consent to nothing.
                _raw("Sure, go ahead.", "- interaction_item_id: i3\n  decision: allow"),
            ]
        )
        pa = _ScriptedHeldPA([[], []])
        loop = _loop(backend, pa)
        events = await loop.run_session()

        for trigger in backend.triggers:
            assert "holding these items" not in trigger
        assert [c["metadata"]["consent_decisions"] for c in pa.calls] == [[], []]
        turns = [e for e in events if e["kind"] == "simulator_turn"]
        assert any(
            "dropped unknown interaction_item_id='i3'" in w
            for w in turns[1]["parse_warnings"]
        )

    asyncio.run(run())


def test_a_malformed_held_items_payload_fails_loudly() -> None:
    async def run():
        backend = _ConsentBackend([_raw("Draft it.", None), _raw("ok", None)])
        pa = _ScriptedHeldPA([[{"intent": "no id here"}], []])
        loop = _loop(backend, pa)
        with pytest.raises(ValueError, match="needs an interaction_item_id"):
            await loop.run_session()

    asyncio.run(run())


def test_the_held_instruction_never_reaches_the_pa() -> None:
    async def run():
        backend = _ConsentBackend(
            [
                _raw("Draft it and check with me first.", None),
                _raw("Send it.", "- interaction_item_id: i3\n  decision: allow"),
                _raw("That is everything.", None),
            ]
        )
        pa = _ScriptedHeldPA(
            [[{"interaction_item_id": "i3", "intent": "send the extension request"}], []]
        )
        loop = _loop(backend, pa)
        await loop.run_session()

        # The stage direction is prompt-side only: it is not in what the PA
        # receives, and not in the history the conversation carries forward.
        for call in pa.calls:
            assert "holding these items" not in call["user_message"]
            assert "consent_decisions" not in call["user_message"]
        assert "holding these items" not in json.dumps(loop._history)

    asyncio.run(run())


def test_the_block_reaches_the_real_provider_request_through_the_backend(
    tmp_path,
) -> None:
    """End-to-end on the production prompt path: SimulatorLoop -> backend ->
    SimulatorAgentLoop -> SimulatorContextBuilder -> the provider request."""

    class _RecordingProvider(LLMProvider):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[list[dict]] = []

        def get_default_model(self) -> str:
            return "recording-model"

        async def chat(self, messages, *args, **kwargs) -> LLMResponse:
            self.requests.append(list(messages))
            turn = len(self.requests)
            body = (
                "- interaction_item_id: i3\n  decision: allow" if turn == 2 else None
            )
            text = "Draft it first." if turn == 1 else "Yes, send it."
            return LLMResponse(content=_raw(text, body))

    async def run():
        provider = _RecordingProvider()
        backend = NanobotRunnerSimulatorBackend(
            provider,
            profile=SimulatorProfile(
                root=tmp_path / "prof",
                id="user_a",
                display_name="User A",
                identity_path=Path("data/personas/user_a_identity.yaml"),
            ),
        )
        pa = _ScriptedHeldPA(
            [[{"interaction_item_id": "i3", "intent": "send the extension request"}], []]
        )
        loop = _loop(backend, pa)
        await loop.run_session()

        first = json.dumps(provider.requests[0])
        second = json.dumps(provider.requests[1])
        assert "holding these items" not in first
        assert "The assistant is holding these items until you say" in second
        assert "i3 (send the extension request)" in second
        # And the decision it produced survived id validation.
        assert pa.calls[1]["metadata"]["consent_decisions"] == [
            {"interaction_item_id": "i3", "decision": "allow"}
        ]

    asyncio.run(run())
