"""acc_039 — the double ask, and the nudge that routes the ask through the id path.

Live shape (`pre_run_test_39`, `beat_04`): under `autonomy_level = Self-directed`
the PA wrote *"Shall I go ahead and book this Friday, November 20 at 10:00 with
Dr. Patel for your routine checkup?"* as plain prose with no call and no plan
item; the user answered *"That slot is acceptable. Proceed with booking…"*; the
next turn's `mcp_state_appointment_create` hit the external-action floor and the
wrapper delivered the PA's own `_ix_confirmation_message` — the same question —
so the user authorised the booking twice, word for word.

The gate is what makes consent real, so it does not move: it stays fail-closed
here (`test_the_gate_still_asks_after_an_unbound_prose_ask`). What moves is the
ask, nudged once per turn towards the call or the plan item.
"""

from __future__ import annotations

from typing import Any

import pytest

from pa.agent.runner import IX_PROSE_ASK_REPAIR
from pa.providers.base import LLMResponse

from .conftest import (
    ScriptProvider,
    plan_call,
    run,
    run_turn,
    selection,
    task_call,
)

ACTION_TYPES = {"task_read": "read", "task_send": "external_action"}
PROSE_ASK = "Friday 10:00 is free with Dr. Patel.\n\nShall I go ahead and book it?"
GATE_ASK = "Shall I book Friday 10:00 with Dr. Patel?"
#: regress19 acc_039 t41, verbatim: the floor held `plan_s1_book_nov10` and this
#: is the whole turn the user read — an announcement of an act that was blocked.
ANNOUNCEMENT = (
    "I'll book the November 10, 08:30 appointment with Dr. Patel at Glenmont "
    "Family Clinic (patient: User A) for the chest tightness."
)


def _event_names(outbound: Any) -> list[str]:
    return [
        str(event.get("name") or "")
        for event in outbound.metadata.get("tool_events") or []
        if event.get("ix_event")
    ]


def _nudges(request: list[dict[str, Any]]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in request
        if str(message.get("content") or "") == IX_PROSE_ASK_REPAIR
    ]


def test_an_unbound_prose_ask_is_nudged_once_and_the_retry_is_delivered(
    loop_factory,
) -> None:
    """(a) One nudge, one event, and the model keeps the last word.

    The retry here still asks in prose. That is deliberate: the nudge is a
    nudge, not a contract — a prose question is sometimes exactly right (a
    missing fact, a material ambiguity) and the wrapper cannot tell which this
    is, so whatever comes back is delivered untouched.
    """
    retry = "Friday 10:00 is still open. Shall I book it?"

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Self-directed"),
            ]),
            LLMResponse(content=PROSE_ASK),
            LLMResponse(content=retry),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Book the checkup for the 20th.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert _event_names(outbound).count("ix_prose_ask_unbound") == 1
        event = next(
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_prose_ask_unbound"
        )
        assert event["attribute"] == "autonomy_level"
        assert event["setting"] == "Self-directed"

        # Nudged once, and the draft it replaced never reaches the user.
        assert len(provider.requests) == 3
        assert _nudges(provider.requests[-1]) == [IX_PROSE_ASK_REPAIR]
        assert retry in outbound.content
        assert "Shall I go ahead and book it?" not in outbound.content

    run(scenario())


def test_the_nudged_turn_can_bind_the_proposal_and_the_gate_asks_with_an_id(
    loop_factory,
) -> None:
    """(b) The intended repair: the call goes out, the floor stops it, the ask carries an id."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Self-directed"),
            ]),
            LLMResponse(content=PROSE_ASK),
            LLMResponse(content="Booking it now.", tool_calls=[
                plan_call({"id": "a1", "bound_item_ids": ["i1"], "is_key": True}),
                task_call("send_1", "task_send", "i1", ask=GATE_ASK),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Book the checkup for the 20th.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        gate = outbound.metadata["ix_render"]["gate"]
        assert gate["item_id"] == "i1"
        assert gate["reason"] == "external_action_floor"
        assert outbound.content.rstrip().endswith(GATE_ASK)
        assert loop.tools.get("task_send").calls == []
        # The nudge fired once; the delivered turn adds only the gate's own row.
        assert _event_names(outbound) == ["ix_prose_ask_unbound", "beat_gate_turn"]

    run(scenario())


@pytest.mark.parametrize(
    "setting,active_ix",
    [
        # No floor: `Autonomous` stops nothing, so no gate can repeat the ask.
        ("Autonomous", ["autonomy_level"]),
        # Not armed at all: the attribute governs nothing this session.
        ("", []),
    ],
    ids=["autonomous", "unarmed"],
)
def test_no_nudge_when_the_floor_cannot_repeat_the_ask(
    loop_factory, setting: str, active_ix: list[str]
) -> None:
    """(c) part 1 — the nudge only exists to prevent a *duplicate* ask."""
    async def scenario() -> None:
        responses = [LLMResponse(content=PROSE_ASK)]
        if setting:
            responses.insert(0, LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", setting),
            ]))
        provider = ScriptProvider(responses)
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Book the checkup for the 20th.",
            active_ix=active_ix, action_types=ACTION_TYPES,
        )

        assert "ix_prose_ask_unbound" not in _event_names(outbound)
        assert len(provider.requests) == len(responses)
        assert _nudges(provider.requests[-1]) == []
        assert "Shall I go ahead and book it?" in outbound.content

    run(scenario())


def test_no_nudge_when_the_turn_attempted_the_call_and_the_gate_stopped_it(
    loop_factory,
) -> None:
    """(c) part 2 — a turn that made the call has already routed its ask.

    The gate's exit leaves the loop before the delivery branch, so the ask the
    user reads is the PA's `_ix_confirmation_message` and nothing is nudged.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Self-directed"),
            ]),
            LLMResponse(content="Booking it now.", tool_calls=[
                plan_call({"id": "a1", "bound_item_ids": ["i1"], "is_key": True}),
                task_call("send_1", "task_send", "i1", ask=GATE_ASK),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Book the checkup for the 20th.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert "ix_prose_ask_unbound" not in _event_names(outbound)
        assert outbound.content.rstrip().endswith(GATE_ASK)

    run(scenario())


def test_the_gate_still_asks_after_an_unbound_prose_ask(loop_factory) -> None:
    """(d) Fail-closed, unchanged — nobody may "fix" the double ask by skipping the gate.

    Turn one ends on the prose ask the nudge could not talk the model out of;
    the user authorises it in plain words; turn two issues the booking. The
    floor stops it and asks again. That second ask is the wrapper working: the
    consent sideband, not the transcript, is what releases a held call.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Self-directed"),
            ]),
            LLMResponse(content=PROSE_ASK),
            LLMResponse(content=PROSE_ASK),
            LLMResponse(content="Booking it now.", tool_calls=[
                plan_call({"id": "a1", "bound_item_ids": ["i1"], "is_key": True}),
                task_call("send_1", "task_send", "i1", ask=GATE_ASK),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        first = await run_turn(
            loop, "Book the checkup for the 20th.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )
        assert _event_names(first).count("ix_prose_ask_unbound") == 1

        second = await run_turn(
            loop, "That slot is acceptable. Proceed with booking.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert second.metadata["ix_render"]["gate"] is not None
        assert second.metadata["ix_render"]["gate"]["reason"] == "external_action_floor"
        assert second.content.rstrip().endswith(GATE_ASK)
        assert loop.tools.get("task_send").calls == []

    run(scenario())


def test_a_gate_ask_that_only_announces_the_act_is_no_ask_at_all(
    loop_factory,
) -> None:
    """regress19 acc_039 t41 — "I'll book …" is not a confirmation.

    The floor held the booking and the PA's `_ix_confirmation_message` was a
    flat declarative. The wrapper shipped it verbatim and traced "ask
    delivered", so the whole user-visible turn announced an act that was in
    fact blocked — the worst possible surface for a held outward step, and the
    simulator answered it as if consent had been sought. A sentence the user
    cannot recognise as a request is no ask, so it takes the same
    `ix_gate_no_message` path an empty one takes: repaired once, then delivered.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Self-directed"),
            ]),
            LLMResponse(content="Booking it now.", tool_calls=[
                plan_call({"id": "a1", "bound_item_ids": ["i1"], "is_key": True}),
                task_call("send_1", "task_send", "i1", ask=ANNOUNCEMENT),
            ]),
            LLMResponse(content="Booking it now.", tool_calls=[
                task_call("send_2", "task_send", "i1", ask=GATE_ASK),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "You may proceed with booking that slot at 08:30.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert _event_names(outbound).count("ix_gate_no_message") == 1
        # The announcement never reaches the user; the repaired question does.
        assert ANNOUNCEMENT not in outbound.content
        assert outbound.content.rstrip().endswith(GATE_ASK)
        assert loop.tools.get("task_send").calls == []

    run(scenario())


@pytest.mark.parametrize(
    "ask",
    [
        "Shall I book the 08:30 slot?",
        "I'll book the 08:30 slot — let me know if that works.",
        "Booking the 08:30 slot; say the word and it goes in.",
        "Please confirm and I'll book the 08:30 slot.",
    ],
    ids=["question", "let-me-know", "say-the-word", "please-confirm"],
)
def test_an_ask_that_puts_the_decision_to_the_user_is_delivered(
    loop_factory, ask: str
) -> None:
    """The other side of the same rule: no question mark is not the test.

    A gate that fails twice is a contract failure, so the check has to accept
    the ways a PA legitimately asks without a question mark. What it refuses is
    the announcement above, which invites nothing.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Self-directed"),
            ]),
            LLMResponse(content="Booking it now.", tool_calls=[
                plan_call({"id": "a1", "bound_item_ids": ["i1"], "is_key": True}),
                task_call("send_1", "task_send", "i1", ask=ask),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "You may proceed with booking that slot at 08:30.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert "ix_gate_no_message" not in _event_names(outbound)
        assert outbound.content.rstrip().endswith(ask)

    run(scenario())


def test_the_nudge_never_pins_a_tool(loop_factory) -> None:
    """(e) It shares no budget with the render-payload ladder, so it spends no pin."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Self-directed"),
            ]),
            LLMResponse(content=PROSE_ASK),
            LLMResponse(content="Nothing is booked yet."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Book the checkup for the 20th.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert "ix_prose_ask_unbound" in _event_names(outbound)
        assert provider.tool_choices == [None, None, None]

    run(scenario())
