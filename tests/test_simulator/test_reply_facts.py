"""The actor's private record of what the assistant's latest reply did.

The simulated user picks a reaction beat's conditional branch from the PA's
visible reply plus exactly four facts the harness hands over — which
state-changing tools ran (catalog name and catalog line, never arguments), how
many task calls the turn made, how many progress lines and whether a closing
line were delivered, and whether bracketed confidence tags appeared. Never a
hold flag, never what the wrapper held back, never a gate's parameters, never a
setting name.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_reply_facts.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from simulator.loop import SimulatorLoop
from simulator.output import parse_simulator_output
from simulator.prompting import _render_user_trigger
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.reply_facts import (
    ReplyFacts,
    normalize_reply_facts,
    render_reply_facts_block,
    reply_facts_from_pa_turn,
)
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

from pa.bus.events import OutboundMessage

SEND_LABEL = "email_message_send — send an email from the current MemPABench run state"
UPDATE_LABEL = "document_update — edit an existing document in the current MemPABench run state"
CREATE_LABEL = "calendar_event_create — create one calendar event or recurring series"

# A PA turn's tool events as the runner records them: one ok read, one ok
# write, one write the gate refused, one blocked read, one IX selection and
# one wrapper control record. Every argument carries a marker that must never
# reach the actor.
TOOL_EVENTS = [
    {"name": "mcp_state_document_read", "status": "ok", "args": {"path": "SECRET_PATH"}},
    {
        "name": "mcp_state_email_message_send",
        "status": "ok",
        "args": {"to": "SECRET_ADDRESS", "body": "SECRET_BODY"},
    },
    {
        "name": "mcp_state_document_update",
        "status": "error",
        "args": {"path": "SECRET_REFUSED"},
        "error": "Error executing tool mcp_state_document_update",
    },
    {
        "name": "mcp_state_contact_read",
        "status": "capability_limit",
        "args": {"contact_id": "SECRET_BLOCKED"},
        "error": {"ok": False, "error_code": "capability_limit"},
    },
    {"name": "IX_process_visibility", "status": "ok", "ix": True, "args": {"setting": "Bookend"}},
    {"name": "ix_empty_reply", "ix_event": True},
]

WRAPPER_FACTS = {"notices_released": 1, "completion_released": True, "confidence_brackets": False}


# -- what the record holds ---------------------------------------------------


def test_state_changes_are_labelled_from_the_catalog_and_carry_no_arguments() -> None:
    facts = reply_facts_from_pa_turn(
        TOOL_EVENTS,
        WRAPPER_FACTS,
        earlier_tools=["mcp_state_calendar_event_create", "mcp_state_document_read"],
        earlier_outreach_sent=False,
    )

    # The refused write changed nothing and is not an act; the reads never are.
    assert facts.executed_acts == (SEND_LABEL,)
    assert facts.earlier_acts == (CREATE_LABEL,)
    # Every task call the PA made, blocked and failed ones included; IX
    # selections and wrapper control records are not calls on the world.
    assert facts.task_calls == 4
    assert facts.notices_released == 1
    assert facts.completion_released is True
    assert facts.confidence_brackets is False

    text = render_reply_facts_block(facts)
    assert SEND_LABEL in text
    assert CREATE_LABEL in text
    assert "SECRET" not in text
    assert "Bookend" not in text
    assert "ix_empty_reply" not in text
    assert "IX_process_visibility" not in text


def test_a_repeated_write_is_listed_each_time_it_ran() -> None:
    """Unlike the judge's deduplicated `executed_acts`, the actor's record keeps
    the count: a second `document_update` is exactly what tells a task-expansion
    reaction that a change beyond the requested one was carried out."""
    facts = reply_facts_from_pa_turn(
        [
            {"name": "mcp_state_document_update", "status": "ok", "args": {}},
            {"name": "mcp_state_document_update", "status": "ok", "args": {}},
        ],
        None,
        earlier_tools=[],
        earlier_outreach_sent=False,
    )
    assert facts.executed_acts == (UPDATE_LABEL, UPDATE_LABEL)
    assert render_reply_facts_block(facts).count(UPDATE_LABEL) == 2


def test_a_delivered_outreach_is_an_act_on_the_turn_that_sent_it_and_earlier_afterwards() -> None:
    sent = reply_facts_from_pa_turn(
        [], None, earlier_tools=[], earlier_outreach_sent=False, outreach_delivered=True
    )
    assert sent.executed_acts == ("proactive_outreach — send the user an unprompted message",)
    later = reply_facts_from_pa_turn(
        [], None, earlier_tools=[], earlier_outreach_sent=True
    )
    assert later.executed_acts == ()
    assert later.earlier_acts == ("proactive_outreach — send the user an unprompted message",)


def test_the_three_wrapper_facts_render_only_when_the_harness_wrote_them() -> None:
    with_them = render_reply_facts_block(
        reply_facts_from_pa_turn([], WRAPPER_FACTS, earlier_tools=[], earlier_outreach_sent=False)
    )
    assert "Progress lines it delivered about its own steps: 1" in with_them
    assert "closing line saying the work was done: delivered" in with_them
    assert "confidence tags beside its claims: none" in with_them

    without = render_reply_facts_block(
        reply_facts_from_pa_turn([], None, earlier_tools=[], earlier_outreach_sent=False)
    )
    assert "Changes it carried out in that reply: none" in without
    assert "Tool calls it made for the task in that reply" in without
    assert "Progress lines" not in without
    assert "closing line" not in without
    assert "confidence tags" not in without


@pytest.mark.parametrize(
    "raw",
    [
        ["notices_released", 1],
        {"notices_released": 1, "completion_released": True},
        {"notices_released": "1", "completion_released": True, "confidence_brackets": False},
        {"notices_released": 1, "completion_released": "yes", "confidence_brackets": False},
        {"notices_released": 1, "completion_released": True, "confidence_brackets": False, "held": []},
    ],
)
def test_a_malformed_harness_key_is_a_wiring_bug_and_fails_loudly(raw) -> None:
    """Silently reading a malformed key as "nothing delivered" would look exactly
    like a Silent turn, the same failure mode `_normalize_held_items` refuses."""
    with pytest.raises(ValueError, match="reply_facts"):
        normalize_reply_facts(raw)


def test_an_absent_harness_key_is_the_prompt_only_path_and_reads_as_no_record() -> None:
    assert normalize_reply_facts(None) is None
    assert render_reply_facts_block(None) == ""


# -- where the record goes ---------------------------------------------------


def _script() -> SessionScript:
    return SessionScript(
        session_id="reply_facts_test",
        scenario="the assistant sends the note and the user reacts",
        context="personal",
        task="Get the note out.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(beat="ask", active_skills=["task_expansion"]),
            Beat(beat="react", active_skills=[]),
        ],
    )


def test_the_block_waits_for_a_reply_to_exist() -> None:
    script = _script()
    facts = ReplyFacts(executed_acts=(SEND_LABEL,), earlier_acts=(), task_calls=1)
    opening = _render_user_trigger(script, script.beats[0], None, reply_facts=facts)
    assert SEND_LABEL not in opening
    after_reply = _render_user_trigger(
        script,
        script.beats[1],
        [{"role": "user", "content": "Sent."}],
        reply_facts=facts,
    )
    assert SEND_LABEL in after_reply
    assert "private record of what it did" in after_reply


def _raw(message: str) -> str:
    return (
        f"<message>\n{message}\n</message>\n\n<factual_check>\n</factual_check>\n\n"
        "<turn_assessment>\nnext_beat: advance\n</turn_assessment>"
    )


class _ScriptedProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "scripted-model"

    async def chat(self, *args, **kwargs) -> LLMResponse:
        raise AssertionError("the scripted backend never calls the provider")


class _CapturingBackend:
    """Replays one raw response per turn and keeps every trigger it was handed."""

    name = "capturing"

    def __init__(self, raws: list[str]) -> None:
        self.raws = list(raws)
        self.triggers: list[str] = []
        self.kwargs: list[dict] = []

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        self.kwargs.append(dict(kwargs))
        self.triggers.append(
            _render_user_trigger(
                session,
                beat,
                kwargs.get("history"),
                kwargs.get("held_items") or (),
                reply_facts=kwargs.get("reply_facts"),
            )
        )
        return parse_simulator_output(self.raws.pop(0))

    async def drain_memory_tasks(self) -> None:
        return None


class _LedgerPA:
    """A PA turn whose metadata carries everything the harness passes along —
    the wrapper's full `ix_render` ledger included, as it does live. It is
    mid-interaction on its first turn only, so the beats still advance."""

    def __init__(self) -> None:
        self.calls = 0

    async def process_direct_message(self, user_message, session_key, **kwargs):
        self.calls += 1
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="Sent the note. I also looked at the other thing.",
            metadata={
                "tool_events": TOOL_EVENTS,
                "held_items": [],
                "has_pending_interaction": self.calls == 1,
                "triggered_skills": [],
                "reply_facts": WRAPPER_FACTS,
                "ix_render": {
                    "held": {
                        "questions": [{"text": "SECRET_HELD_QUESTION"}],
                        "batches": [{"batch_id": "SECRET_BATCH"}],
                        "plan_remaining": [],
                        "calls": [],
                        "cancelled": [],
                        "outreach_message": "",
                    },
                    "gate": {
                        "attribute": "autonomy_level",
                        "item_id": "SECRET_GATE_ITEM",
                        "reason": "external_action_floor",
                        "action_type": "external_action",
                    },
                    "episodes": {
                        "topic_management": {"episode_id": "SECRET_EPISODE", "closed": False}
                    },
                    "notices": [{"order": 0, "tool": "x", "text": "SECRET_NOTICE"}],
                    "completion": {"released": True, "text": "SECRET_CLOSING"},
                    "certainties": [{"ref": "c1", "level": "low", "rendered": ""}],
                    "trace": ["process_visibility=Bookend -> notices 1/2"],
                },
            },
        )


def _loop(backend, pa) -> SimulatorLoop:
    return SimulatorLoop(
        _ScriptedProvider(),
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script(),
        pa,
        backend=backend,
    )


def test_the_actor_prompt_carries_the_four_facts_and_never_the_held_ledger() -> None:
    async def run():
        backend = _CapturingBackend([_raw("Send the note.")] + [_raw("Fine.")] * 7)
        loop = _loop(backend, _LedgerPA())
        await loop.run_session()

        assert backend.triggers, "no simulator turn was produced"
        opening, *after = backend.triggers
        assert "private record of what it did" not in opening
        assert after, "no turn after the PA's reply"
        for trigger in after:
            # The four facts, as the module docstring lists them.
            assert SEND_LABEL in trigger
            assert "Tool calls it made for the task in that reply" in trigger
            assert ": 4" in trigger
            assert "Progress lines it delivered about its own steps: 1" in trigger
            assert "closing line saying the work was done: delivered" in trigger
            assert "confidence tags beside its claims: none" in trigger
            # Nothing the wrapper held back, no hold flag, no gate, no setting.
            for marker in (
                "SECRET_",
                "has_pending_interaction",
                "pending",
                "autonomy_level",
                "topic_management",
                "external_action_floor",
                "Bookend",
                "ix_render",
            ):
                assert marker not in trigger, marker
        # The loop never hands the backend the ledger either. (`pa_tool_log`,
        # with arguments, does travel to the backend — that is the judge's
        # retry-check input, and the judge is not the actor.)
        for kwargs in backend.kwargs:
            assert "ix_render" not in json.dumps(kwargs, default=str)
            assert "SECRET_HELD_QUESTION" not in json.dumps(kwargs, default=str)

    asyncio.run(run())


# -- the actor is told the same boundary -------------------------------------


def test_the_actor_templates_name_the_four_facts() -> None:
    """Both actor templates name the boundary the runtime keeps: the private
    record of what the assistant's latest reply actually did."""
    def flat(path: str) -> str:
        # The prompts are hard-wrapped; compare with the line breaks folded.
        return " ".join(Path(path).read_text(encoding="utf-8").split())

    for template in (
        "simulator/prompts/generate_message.md",
        "simulator/prompts/generate_message_probe.md",
    ):
        assert "private record of what the assistant's latest reply actually did" in flat(
            template
        ), template
