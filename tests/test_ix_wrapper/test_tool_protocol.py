"""Every request the PA sends answers every call the history carries.

`acc_001` / `acc_004`: `openai/gpt-5.6-sol` returned
`400 No tool output found for function call <id>` and the run died. The calls
were the ones `_execute_tools_ix_first` withholds when a selection is refused —
they are recorded as held and deliberately never run, and the loop then
re-requests, so their `tool_call_id`s went out unanswered. deepseek, gemini and
qwen accept that history; OpenAI rejects it.

The assertion here is the protocol invariant itself, checked against what the
provider actually received, so it holds for every path that withholds a call —
not only the refused-selection one that happened to be hit live.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pa.providers.base import LLMProvider, LLMResponse

from .conftest import ScriptProvider, plan_call, run_turn, selection, task_call

ACTION_TYPES = {"task_read": "read", "task_send": "external_action"}


def unanswered_call_ids(request: list[dict[str, Any]]) -> list[str]:
    """Call ids in `request` that no `role: "tool"` message answers."""
    answered = {
        str(message.get("tool_call_id"))
        for message in request
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    return [
        str(tool_call.get("id"))
        for message in request
        if message.get("role") == "assistant"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict) and str(tool_call.get("id")) not in answered
    ]


def misplaced_tool_messages(request: list[dict[str, Any]]) -> list[str]:
    """Tool messages that do not follow the assistant message that called them.

    Appending the repairs at the tail of the history instead of beside their own
    assistant message would satisfy the id check and still be rejected: the
    provider reads the replies positionally.
    """
    misplaced: list[str] = []
    open_ids: set[str] = set()
    for message in request:
        role = message.get("role")
        if role == "assistant":
            open_ids = {
                str(tool_call.get("id"))
                for tool_call in (message.get("tool_calls") or [])
                if isinstance(tool_call, dict)
            }
        elif role == "tool":
            call_id = str(message.get("tool_call_id"))
            if call_id not in open_ids:
                misplaced.append(call_id)
            open_ids.discard(call_id)
        elif role == "user" and open_ids:
            # An internal repair note lands after the replies, never between an
            # assistant message and the answers it is still owed.
            misplaced.extend(sorted(open_ids))
            open_ids = set()
    return misplaced


def _refused_selection_responses() -> list[LLMResponse]:
    """A refused selection carrying task calls, then a valid one.

    `Suggest` was retired when `autonomy_level` went to three settings, so the
    registry refuses it and `_execute_tools_ix_first` withholds the `task_send`
    in the same response. The next request is the one that used to 400.
    """
    return [
        LLMResponse(content="Sending it now.", tool_calls=[
            selection("autonomy_level", "Suggest"),
            plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
            task_call("send_1", "task_send", "i1", ask="Shall I mail Dale?"),
        ]),
        LLMResponse(content="Sending it now.", tool_calls=[
            selection("autonomy_level", "Reactive"),
            plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
            task_call("send_2", "task_send", "i1", ask="Shall I mail Dale?"),
        ]),
        LLMResponse(content="The extension request is on its way."),
    ]


def test_a_withheld_call_still_gets_a_tool_result(loop_factory) -> None:
    """The turn that killed `openai/gpt-5.6-sol`, checked at the request."""
    async def scenario() -> None:
        provider = ScriptProvider(_refused_selection_responses())
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert len(provider.requests) > 1, (
            "the refusal has to be followed by a re-request for this to test "
            "anything"
        )
        for index, request in enumerate(provider.requests):
            assert unanswered_call_ids(request) == [], (
                f"request {index} carries a tool call with no result"
            )
            assert misplaced_tool_messages(request) == [], (
                f"request {index} answers a call out of position"
            )
        # The withheld call is answered, and answered as *not done* — a result
        # that read as success would tell the model the mail went out.
        withheld = [
            message for request in provider.requests for message in request
            if message.get("role") == "tool"
            and str(message.get("tool_call_id")) == "send_1"
        ]
        assert withheld, "the withheld task call was never answered"
        assert "withheld" in str(withheld[0]["content"])
        # …and it is still only an answer on the transcript: nothing ran.
        assert [
            call for call in loop.tools.get("task_send").calls
        ] == []

    asyncio.run(scenario())


def test_every_scripted_turn_leaves_a_well_formed_history(loop_factory) -> None:
    """The invariant over a turn with no refusal at all, as the control.

    A plain executed turn already answered every call; this pins that the repair
    did not start inventing duplicate results for calls that have one.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Reading it.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                task_call("read_1", "task_read", value="inbox"),
            ]),
            LLMResponse(content="Nothing new in the inbox."),
        ])
        loop, _ = loop_factory(provider, "task_read")

        await run_turn(
            loop, "Anything in my inbox?",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        for request in provider.requests:
            assert unanswered_call_ids(request) == []
            assert misplaced_tool_messages(request) == []
            ids = [
                str(message.get("tool_call_id"))
                for message in request if message.get("role") == "tool"
            ]
            assert len(ids) == len(set(ids)), "a call was answered twice"

    asyncio.run(scenario())
