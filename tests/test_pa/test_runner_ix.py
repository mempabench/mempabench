"""AgentRunner IX-tool enforcement tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from memory.adapter import PromptHistoryEvictionPolicy
from memory.results import RetrieveResult, WriteReceipt, WriteStatus
from pa.agent.ix.events import IXSelection
from pa.agent.ix.render import released_ids
from pa.agent.loop import AgentLoop
from pa.bus.queue import MessageBus
from pa.agent.runner import (
    CALENDAR_RECURRENCE_INVALID_ERROR_CODE,
    TOOL_FAILURE_STRIKE_LIMIT,
    TOOL_RETRY_CAP,
    AgentRunSpec,
    AgentRunner,
)
from pa.agent.ix import TurnToolRegistry
from pa.agent.tools.base import Tool
from pa.agent.tools.registry import ToolRegistry
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from state_server.store import (
    CALENDAR_RECURRENCE_INVALID_ERROR_CODE as STORE_CALENDAR_RECURRENCE_INVALID_ERROR_CODE,
)
from state_server.store import _parse_recurrence


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "task_echo"

    @property
    def description(self) -> str:
        return "Echo a value."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    async def execute(self, value: str) -> dict[str, str]:
        return {"value": value}


class _NamedEchoTool(EchoTool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class StateErrorTool(EchoTool):
    async def execute(self, value: str) -> str:
        return json.dumps({
            "ok": False,
            "error": {
                "code": "invalid_type",
                "message": f"Invalid value: {value!r}",
            },
        })


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.tool_choices: list[Any] = []

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
        self.calls.append(list(messages))
        self.tool_choices.append(tool_choice)
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)


def test_runner_marks_structured_tool_failure_as_error() -> None:
    async def run():
        provider = ScriptedProvider([
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="task_echo",
                        arguments={"value": "wrong"},
                    )
                ],
            ),
            LLMResponse(content="Recovered."),
        ])
        tools = ToolRegistry()
        tools.register(StateErrorTool())

        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Run the task."}],
            tools=tools,
            model="test-model",
            max_iterations=2,
        ))

        assert result.final_content == "Recovered."
        assert result.tool_events == [{
            "name": "task_echo",
            "status": "error",
            "args": {"value": "wrong"},
            "detail": (
                '{"ok": false, "error": {"code": "invalid_type", '
                '"message": "Invalid value: \'wrong\'"}}'
            ),
            "error": {
                "code": "invalid_type",
                "message": "Invalid value: 'wrong'",
            },
        }]

    asyncio.run(run())


def test_runner_repairs_missing_required_ix_tool() -> None:
    async def run():
        provider = ScriptedProvider([
            LLMResponse(content="Draft response without IX."),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="IX_guidance_level",
                        arguments={
                            "setting": "Assumed",
                            "evidence": "The user asks for peer review.",
                            "application": "Skip basic explanations.",
                        },
                    )
                ],
            ),
            LLMResponse(content="Final response after IX."),
        ])
        tools = TurnToolRegistry(ToolRegistry(), ["guidance_level"])
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Review this draft."}],
            tools=tools,
            model="test-model",
            max_iterations=4,
            required_tool_names=tools.required_ix_tool_names,
        ))
        assert result.final_content == "Final response after IX."
        assert result.tools_used == ["IX_guidance_level"]
        assert result.tool_events[0]["ix"] is True
        assert result.tool_events[0]["attribute"] == "guidance_level"
        assert result.tool_events[0]["setting"] == "Assumed"

    asyncio.run(run())


def test_process_visibility_is_passive_required_ix_tool() -> None:
    async def run():
        provider = ScriptedProvider([
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="IX_process_visibility",
                        arguments={
                            "setting": "Full narration",
                            "evidence": "The task has multiple tool steps.",
                            "application": "Narrate each step as it completes.",
                        },
                    )
                ],
            ),
            LLMResponse(content="Final response after process visibility selection."),
        ])
        tools = TurnToolRegistry(ToolRegistry(), ["process_visibility"])
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Do this multi-step task."}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            required_tool_names=tools.required_ix_tool_names,
        ))
        assert result.final_content == "Final response after process visibility selection."
        assert result.tools_used == ["IX_process_visibility"]
        assert result.tool_events[0]["attribute"] == "process_visibility"
        assert result.tool_events[0]["setting"] == "Full narration"

    asyncio.run(run())


def test_runner_executes_required_ix_before_task_tools_and_emits_process_events() -> None:
    async def run():
        provider = ScriptedProvider([
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="task_1",
                        name="task_echo",
                        arguments={"value": "first", "_ix_notice": "Running the task."},
                    ),
                    ToolCallRequest(
                        id="ix_1",
                        name="IX_process_visibility",
                        arguments={
                            "setting": "Full narration",
                            "evidence": "The user wants each step visible.",
                            "application": "Show progress around task tools.",
                        },
                    ),
                ],
            ),
            LLMResponse(content="Final response."),
        ])
        base = ToolRegistry()
        base.register(EchoTool())
        tools = TurnToolRegistry(base, ["process_visibility"])
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Run the task visibly."}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            required_tool_names=tools.required_ix_tool_names,
        ))

        assert result.final_content == "Final response."
        assert [event.get("name") for event in result.tool_events if not event.get("process")] == [
            "IX_process_visibility",
            "task_echo",
        ]
        assert [event.get("stage") for event in result.tool_events if event.get("process")] == [
            "start",
            "progress",
            "completion",
        ]

        tool_messages = [msg for msg in result.messages if msg.get("role") == "tool"]
        assert [msg["tool_call_id"] for msg in tool_messages] == ["task_1", "ix_1"]

    asyncio.run(run())


def test_runner_blocks_task_tools_until_required_ix_is_selected() -> None:
    async def run():
        provider = ScriptedProvider([
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="task_early",
                        name="task_echo",
                        arguments={"value": "too early"},
                    ),
                ],
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="ix_1",
                        name="IX_process_visibility",
                        arguments={
                            "setting": "Full narration",
                            "evidence": "The user asked to see each step.",
                            "application": "Emit process progress for task tools.",
                        },
                    ),
                ],
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="task_after_ix",
                        name="task_echo",
                        arguments={"value": "after ix", "_ix_notice": "Running the task."},
                    ),
                ],
            ),
            LLMResponse(content="Final response."),
        ])
        base = ToolRegistry()
        base.register(EchoTool())
        tools = TurnToolRegistry(base, ["process_visibility"])
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Run the task visibly."}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            required_tool_names=tools.required_ix_tool_names,
        ))

        assert result.final_content == "Final response."
        internal_user_messages = [
            message
            for message in result.messages
            if message.get("role") == "user"
            and message.get("content") != "Run the task visibly."
        ]
        assert internal_user_messages
        assert all(
            message.get("_mempa_internal") is True
            for message in internal_user_messages
        )
        executed_names = [
            event.get("name")
            for event in result.tool_events
            if not event.get("process")
        ]
        assert executed_names == ["IX_process_visibility", "task_echo"]
        assert result.tool_events[-2]["args"] == {"value": "after ix"}
        assert [event.get("stage") for event in result.tool_events if event.get("process")] == [
            "start",
            "progress",
            "completion",
        ]

        tool_messages = [msg for msg in result.messages if msg.get("role") == "tool"]
        assert [msg["tool_call_id"] for msg in tool_messages] == ["ix_1", "task_after_ix"]

    asyncio.run(run())


def test_b1_control_affecting_ix_tools_are_passive_required_tools() -> None:
    async def run():
        provider = ScriptedProvider([
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="IX_topic_management",
                        arguments={"setting": "One-at-a-time"},
                    ),
                    ToolCallRequest(
                        id="call_2",
                        name="IX_autonomy_level",
                        arguments={"setting": "Self-directed"},
                    ),
                    ToolCallRequest(
                        id="call_3",
                        name="IX_information_elicitation",
                        arguments={"setting": "Structured"},
                    ),
                ],
            ),
            LLMResponse(content="Final response after B1 IX selections."),
        ])
        tools = TurnToolRegistry(ToolRegistry(), [
            "topic_management",
            "autonomy_level",
            "information_elicitation",
        ])
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Handle this carefully."}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            required_tool_names=tools.required_ix_tool_names,
        ))
        assert result.final_content == "Final response after B1 IX selections."
        assert result.tools_used == [
            "IX_topic_management",
            "IX_autonomy_level",
            "IX_information_elicitation",
        ]
        selections = {
            event["attribute"]: event["setting"]
            for event in result.tool_events
            if event.get("ix")
        }
        assert selections == {
            "topic_management": "One-at-a-time",
            "autonomy_level": "Self-directed",
            "information_elicitation": "Structured",
        }

    asyncio.run(run())


def test_inherited_process_visibility_directive_applies_without_reselecting_ix() -> None:
    async def run():
        provider = ScriptedProvider([
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="task_1",
                        name="task_echo",
                        arguments={"value": "draft", "_ix_notice": "Drafting it now."},
                    ),
                ],
            ),
            LLMResponse(content="Final response."),
        ])
        base = ToolRegistry()
        base.register(EchoTool())
        tools = TurnToolRegistry(base, [])
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Run the task."}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            inherited_ix_selections=[
                IXSelection(
                    attribute="process_visibility",
                    tool_name="IX_process_visibility",
                    setting="Full narration",
                )
            ],
        ))

        process_events = [event for event in result.tool_events if event.get("process")]
        assert [event["stage"] for event in process_events] == [
            "start",
            "progress",
            "completion",
        ]
        assert all(event.get("name") != "IX_process_visibility" for event in result.tool_events)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# IX wrapper gates (plan §3.3 (part 2), §3.4 (parts 2/3), §3.5 (part 6))
# ---------------------------------------------------------------------------


def _ix_call(name: str, **arguments: Any) -> ToolCallRequest:
    return ToolCallRequest(id=f"{name}_1", name=name, arguments=arguments)


def _plan(*items: dict[str, Any]) -> ToolCallRequest:
    return _ix_call(
        "IX_render_autonomy_level",
        plan_id="p1",
        plan=[
            {
                "id": item["id"],
                "intent": item.get("intent", "do the thing"),
                "is_key": item.get("is_key", False),
                "bound_item_ids": item["bound_item_ids"],
            }
            for item in items
        ],
        plan_done=False,
    )


def _task(call_id: str, name: str, item_id: str, ask: str = "") -> ToolCallRequest:
    arguments: dict[str, Any] = {"value": item_id, "interaction_item_id": item_id}
    if ask:
        arguments["_ix_confirmation_message"] = ask
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _autonomy_run(
    *,
    setting: str,
    tool_calls: list[ToolCallRequest],
    action_types: dict[str, str],
    tool_names: tuple[str, ...] = ("task_echo",),
    final: str = "Turn done.",
):
    provider = ScriptedProvider([
        LLMResponse(content="Working on it.", tool_calls=tool_calls),
        LLMResponse(content=final),
    ])
    base = ToolRegistry()
    for name in tool_names:
        base.register(EchoTool() if name == "task_echo" else _NamedEchoTool(name))
    ix_context = {
        "session_active_ix": ["autonomy_level"],
        "tool_action_types": action_types,
    }
    tools = TurnToolRegistry(base, ["autonomy_level"], ix_context=ix_context)
    return provider, AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Handle it."}],
        tools=tools,
        model="test-model",
        max_iterations=4,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
    ))


def _executed(result) -> list[str]:
    return [
        call.tool_name for call in result.render_input.calls
        if call.executed and not call.tool_name.startswith("IX_")
    ]


def test_reactive_gates_an_internal_write_and_delivers_the_pa_ask_verbatim() -> None:
    async def run():
        _, coro = _autonomy_run(
            setting="Reactive",
            tool_calls=[
                _ix_call("IX_autonomy_level", setting="Reactive"),
                _plan({"id": "a1", "bound_item_ids": ["i1"]}),
                _task("write_1", "task_echo", "i1", ask="Shall I save the draft?"),
            ],
            action_types={"task_echo": "internal_write"},
        )
        result = await coro

        assert result.stop_reason == "ix_gate"
        # The PA's own sentence, not one the wrapper wrote for it.
        assert result.final_content == "Shall I save the draft?"
        assert _executed(result) == []
        held = result.collector.held
        assert [item.status for item in held] == ["held_for_confirmation"]
        assert held[0].interaction_item_id == "i1"
        assert held[0].plan_item_id == "a1"

    asyncio.run(run())


def test_reactive_lets_a_read_through() -> None:
    async def run():
        _, coro = _autonomy_run(
            setting="Reactive",
            tool_calls=[
                _ix_call("IX_autonomy_level", setting="Reactive"),
                _plan({"id": "a1", "bound_item_ids": ["i1"]}),
                _task("read_1", "task_echo", "i1"),
            ],
            action_types={"task_echo": "read"},
            final="The calendar is clear.",
        )
        result = await coro

        assert result.stop_reason == "completed"
        assert result.final_content == "The calendar is clear."
        assert _executed(result) == ["task_echo"]
        assert result.collector.held == []

    asyncio.run(run())


def test_self_directed_never_gates_a_draft_but_always_gates_an_external_action() -> None:
    async def run():
        _, coro = _autonomy_run(
            setting="Self-directed",
            tool_calls=[
                _ix_call("IX_autonomy_level", setting="Self-directed"),
                # The model calls the outward action "not key"; the floor holds.
                _plan(
                    {"id": "a1", "is_key": True, "bound_item_ids": ["i1"]},
                    {"id": "a2", "is_key": False, "bound_item_ids": ["i2"]},
                ),
                _task("draft_1", "task_echo", "i1", ask="Draft it?"),
                _task("send_1", "task_send", "i2", ask="Send it to dale@example.com?"),
            ],
            action_types={"task_echo": "draft", "task_send": "external_action"},
            tool_names=("task_echo", "task_send"),
        )
        result = await coro

        assert _executed(result) == ["task_echo"]
        assert result.stop_reason == "ix_gate"
        assert result.final_content == "Send it to dale@example.com?"
        assert result.collector.gate.reason == "external_action_floor"

    asyncio.run(run())


def test_self_directed_gates_an_internal_write_only_when_it_is_the_key_step() -> None:
    async def run():
        for is_key, expect_executed in ((True, []), (False, ["task_echo"])):
            _, coro = _autonomy_run(
                setting="Self-directed",
                tool_calls=[
                    _ix_call("IX_autonomy_level", setting="Self-directed"),
                    _plan({"id": "a1", "is_key": is_key, "bound_item_ids": ["i1"]}),
                    _task("write_1", "task_echo", "i1", ask="Save it?"),
                ],
                action_types={"task_echo": "internal_write"},
            )
            result = await coro
            assert _executed(result) == expect_executed, is_key

    asyncio.run(run())


def test_autonomous_gates_nothing() -> None:
    async def run():
        for setting in ("Autonomous",):
            _, coro = _autonomy_run(
                setting=setting,
                tool_calls=[
                    _ix_call("IX_autonomy_level", setting=setting),
                    _plan(
                        {"id": "a1", "bound_item_ids": ["i1"]},
                        {"id": "a2", "bound_item_ids": ["i2"]},
                    ),
                    _task("note_1", "task_echo", "i1"),
                    _task("send_1", "task_send", "i2"),
                ],
                action_types={
                    "task_echo": "internal_write",
                    "task_send": "external_action",
                },
                tool_names=("task_echo", "task_send"),
            )
            result = await coro
            assert _executed(result) == ["task_echo", "task_send"], setting
            assert result.stop_reason == "completed"

    asyncio.run(run())


def test_the_gate_stops_at_the_first_open_item(  # T8
) -> None:
    async def run():
        _, coro = _autonomy_run(
            setting="Reactive",
            tool_calls=[
                _ix_call("IX_autonomy_level", setting="Reactive"),
                _plan(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                    {"id": "a3", "bound_item_ids": ["i3"]},
                ),
                _task("read_1", "task_read", "i1"),
                _task("send_1", "task_send", "i2", ask="Send it?"),
                _task("read_2", "task_read2", "i3"),
            ],
            action_types={
                "task_read": "read",
                "task_send": "external_action",
                "task_read2": "read",
            },
            tool_names=("task_read", "task_send", "task_read2"),
        )
        result = await coro

        # The read after the gate is not executed either: the turn stops here.
        assert _executed(result) == ["task_read"]
        assert [item.tool for item in result.collector.held] == ["task_send", "task_read2"]

    asyncio.run(run())


def test_a_missing_interaction_item_id_stops_the_whole_batch() -> None:  # T7
    async def run():
        bad = ToolCallRequest(id="w_1", name="task_echo", arguments={"value": "x"})
        provider = ScriptedProvider([
            LLMResponse(content="Doing three topics.", tool_calls=[
                _ix_call("IX_topic_management", setting="Organize"),
                _ix_call(
                    "IX_render_topic_management",
                    batch_set_id="bs1",
                    item_count=1,
                    batches=[{"batch_id": "b1", "label": "hotel", "body": "…",
                              "call_binding": ["i1"]}],
                ),
                bad,
            ]),
            LLMResponse(content="Still doing three topics.", tool_calls=[
                _ix_call(
                    "IX_render_topic_management",
                    batch_set_id="bs1",
                    item_count=1,
                    batches=[{"batch_id": "b1", "label": "hotel", "body": "…",
                              "call_binding": ["i1"]}],
                ),
                bad,
            ]),
            # §3.4 (part 5) runs on this exit too; this model has nothing to add.
            LLMResponse(content=None),
        ])
        base = ToolRegistry()
        base.register(EchoTool())
        ix_context = {
            "session_active_ix": ["topic_management"],
            "tool_action_types": {"task_echo": "internal_write"},
        }
        tools = TurnToolRegistry(base, ["topic_management"], ix_context=ix_context)
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Three things please."}],
            tools=tools,
            model="test-model",
            max_iterations=4,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
        ))

        # No autonomy gate is active here; the binding check alone must hold it.
        assert _executed(result) == []
        repair = [
            message for message in result.messages
            if message.get("_mempa_internal") and "interaction item" in str(message["content"])
        ]
        assert len(repair) == 1
        assert any(
            event.get("name") == "ix_binding_invalid" for event in result.tool_events
        )
        # The declaration that bound ids this turn was `topic_management`'s, so
        # that is what the held rows say. Live `v7r06` t18 stamped six rows
        # `autonomy_level` — an attribute the turn never mentioned — while they
        # carried `batch_id: "b1"`, so anything keying held rows by attribute
        # looked in the wrong place. And a fail-closed refusal opens no gate and
        # asks nothing, so it must not wear the name of the gate that does.
        assert {
            (item.attribute, item.status, item.reason)
            for item in result.collector.held
        } == {("topic_management", "held_for_binding", "binding_invalid")}

    asyncio.run(run())


def test_an_unbound_read_runs_and_an_unbound_write_is_still_refused() -> None:
    """A binding governs the calls a gate can stop; a read is not one of them.

    Live `acc_039` P10: four calls arrived under item ids no plan item declared,
    all four were lookups, the whole batch was refused `binding_invalid`, and
    the user's turn was the template "I could not carry out those steps".
    """
    async def run():
        for tool_name, action_type, executed, held in (
            ("task_read", "read", ["task_read"], []),
            ("task_echo", "internal_write", [], ["task_echo"]),
        ):
            unbound = ToolCallRequest(
                id="c_1", name=tool_name, arguments={"value": "x"},
            )
            provider = ScriptedProvider([
                LLMResponse(content="Looking it up.", tool_calls=[
                    _ix_call("IX_autonomy_level", setting="Self-directed"),
                    _plan({"id": "a1", "bound_item_ids": ["i1"]}),
                    unbound,
                ]),
                LLMResponse(content="Turn done."),
                LLMResponse(content="Turn done."),
            ])
            base = ToolRegistry()
            base.register(_NamedEchoTool(tool_name))
            ix_context = {
                "session_active_ix": ["autonomy_level"],
                "tool_action_types": {tool_name: action_type},
            }
            tools = TurnToolRegistry(base, ["autonomy_level"], ix_context=ix_context)
            result = await AgentRunner(provider).run(AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Look it up."}],
                tools=tools,
                model="test-model",
                max_iterations=4,
                required_tool_names=tools.required_ix_tool_names,
                ix_context=ix_context,
            ))

            assert _executed(result) == executed, action_type
            assert [item.tool for item in result.collector.held] == held, action_type
            assert [item.reason for item in result.collector.held] == (
                [] if action_type == "read" else ["binding_invalid"]
            ), action_type

    asyncio.run(run())


def test_the_wrapper_fields_are_not_part_of_the_task_call() -> None:
    """§5.3 carrier C — stripped from the call *and* from what the run records.

    Live `acc_039` P12 logged `calendar_event_create` with an empty
    `_ix_confirmation_message` sitting among its real arguments: the registry
    strips the field before the tool, but the transcript kept the raw call.
    """
    async def run():
        _, coro = _autonomy_run(
            setting="Self-directed",
            tool_calls=[
                _ix_call("IX_autonomy_level", setting="Self-directed"),
                _plan({"id": "a1", "bound_item_ids": ["i1"]}),
                _task("write_1", "task_echo", "i1", ask="Shall I save it?"),
            ],
            action_types={"task_echo": "internal_write"},
        )
        result = await coro

        assert _executed(result) == ["task_echo"]
        args = [
            event["args"] for event in result.tool_events
            if event.get("name") == "task_echo"
        ]
        assert args == [{"value": "i1"}]

    asyncio.run(run())


def test_a_call_needs_no_binding_where_no_payload_can_declare_one() -> None:
    """acc_009: `information_elicitation` has no carrier for a call id.

    Only an `IX_render_autonomy_level` plan step and an
    `IX_render_topic_management` batch declare call ids. On a turn governed by
    `information_elicitation` (or `capability_boundary`) the field could only be
    filled by inventing an id, which the same check then refuses as unknown —
    so the PA's first attempt, which correctly omitted it, must run.
    """
    async def run():
        provider = ScriptedProvider([
            LLMResponse(content="Checking.", tool_calls=[
                _ix_call("IX_information_elicitation", setting="Infer"),
                _ix_call(
                    "IX_render_information_elicitation",
                    elicitation_id="e1",
                    questions=[],
                    assumptions=[{"about": "date", "assumed": "this Monday"}],
                ),
                ToolCallRequest(
                    id="w_1", name="task_echo", arguments={"value": "x"},
                ),
            ]),
            LLMResponse(content="Checked."),
        ])
        base = ToolRegistry()
        base.register(EchoTool())
        ix_context = {
            "session_active_ix": ["information_elicitation"],
            "tool_action_types": {"task_echo": "internal_write"},
        }
        tools = TurnToolRegistry(base, ["information_elicitation"], ix_context=ix_context)
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Check it."}],
            tools=tools,
            model="test-model",
            max_iterations=4,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
        ))

        assert _executed(result) == ["task_echo"]
        assert result.collector.held == []

    asyncio.run(run())


def test_a_plan_turn_still_refuses_a_call_with_no_binding() -> None:
    """The other half: where a plan *is* the carrier, an absent id is a refusal.

    Both shapes are the same refusal — an omitted id and an id no plan step
    claimed — because both mean the gate has nothing to hold the call by.
    """
    async def run():
        for arguments in (
            {"value": "x"},
            {"value": "x", "interaction_item_id": "i9"},
        ):
            bad = ToolCallRequest(id="w_1", name="task_echo", arguments=arguments)
            provider = ScriptedProvider([
                LLMResponse(content="Doing it.", tool_calls=[
                    _ix_call("IX_autonomy_level", setting="Self-directed"),
                    _plan({"id": "a1", "bound_item_ids": ["i1"]}),
                    bad,
                ]),
                LLMResponse(content="Doing it.", tool_calls=[
                    _plan({"id": "a1", "bound_item_ids": ["i1"]}),
                    bad,
                ]),
                LLMResponse(content=None),
            ])
            base = ToolRegistry()
            base.register(EchoTool())
            ix_context = {
                "session_active_ix": ["autonomy_level"],
                "tool_action_types": {"task_echo": "internal_write"},
            }
            tools = TurnToolRegistry(base, ["autonomy_level"], ix_context=ix_context)
            result = await AgentRunner(provider).run(AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Do it."}],
                tools=tools,
                model="test-model",
                max_iterations=4,
                required_tool_names=tools.required_ix_tool_names,
                ix_context=ix_context,
            ))

            # Refused on both attempts — the retry resends the same call — so
            # the rows are the two refusals, not a repaired one.
            assert _executed(result) == [], arguments
            assert {item.reason for item in result.collector.held} == {
                "binding_invalid"
            }, arguments

    asyncio.run(run())


def test_a_selection_repeated_in_one_response_decides_once() -> None:
    """One decision, one selection event — and the first call is the decision.

    Live `acc_026` t14: a pinned response carried `IX_capability_boundary`
    twice with byte-identical arguments and both ran, so one decision was
    recorded as two selections. A repeat naming a *different* setting was worse
    than redundant — `directive_for` reads the latest selection, so the second
    value won silently.
    """
    async def run():
        for repeated in ("Self-directed", "Reactive"):
            _, coro = _autonomy_run(
                setting="Self-directed",
                tool_calls=[
                    _ix_call("IX_autonomy_level", setting="Self-directed"),
                    ToolCallRequest(
                        id="IX_autonomy_level_2",
                        name="IX_autonomy_level",
                        arguments={"setting": repeated},
                    ),
                    _plan({"id": "a1", "bound_item_ids": ["i1"]}),
                    _task("write_1", "task_echo", "i1"),
                ],
                action_types={"task_echo": "internal_write"},
            )
            result = await coro

            selections = [
                event for event in result.tool_events
                if event.get("ix") and event.get("name") == "IX_autonomy_level"
            ]
            assert len(selections) == 1, repeated
            assert selections[0]["setting"] == "Self-directed", repeated
            duplicates = [
                event for event in result.tool_events
                if event.get("name") == "ix_selection_duplicate"
            ]
            assert [
                (event["tool"], event["ignored_setting"]) for event in duplicates
            ] == [("IX_autonomy_level", repeated)], repeated
            # First wins where it counts: `Reactive` would have gated the
            # internal write, and it does not.
            assert _executed(result) == ["task_echo"], repeated
            assert result.stop_reason == "completed", repeated

    asyncio.run(run())


def test_releasing_questions_stops_this_turn_s_task_calls() -> None:  # T10
    async def run():
        provider = ScriptedProvider([
            LLMResponse(content="Two things first. [[q:q1]]", tool_calls=[
                _ix_call("IX_information_elicitation", setting="Structured"),
                _ix_call(
                    "IX_render_information_elicitation",
                    elicitation_id="e1",
                    questions=[{"id": "q1", "text": "Which date?", "about": "date"}],
                    assumptions=[],
                ),
                _task("book_1", "task_echo", "i1"),
            ]),
            LLMResponse(content="Waiting on the date."),
        ])
        base = ToolRegistry()
        base.register(EchoTool())
        ix_context = {
            "session_active_ix": ["information_elicitation"],
            "tool_action_types": {"task_echo": "external_action"},
        }
        tools = TurnToolRegistry(base, ["information_elicitation"], ix_context=ix_context)
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Book something."}],
            tools=tools,
            model="test-model",
            max_iterations=4,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
        ))

        assert _executed(result) == []
        assert [item.status for item in result.collector.held] == ["held_for_questions"]

    asyncio.run(run())


class _CapabilityLimitTool(EchoTool):
    async def execute(self, value: str) -> str:
        return json.dumps({
            "ok": False,
            "error_code": "capability_limit",
            "boundary_kind": "missing_data",
        })


class _NamedCapabilityLimitTool(_CapabilityLimitTool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name


def _capability_run(second: LLMResponse, provider_box: list | None = None):
    provider = ScriptedProvider([
        LLMResponse(content="Fetching it.", tool_calls=[
            _ix_call("IX_capability_boundary", setting="Find and hand off"),
            ToolCallRequest(
                id="read_1", name="task_echo",
                arguments={"value": "paper", "interaction_item_id": "i1"},
            ),
        ]),
        # Reminder, forced `tool_choice`, and the response that still owes the
        # payload: §3.5 (part 4) spends a third attempt before the contract failure.
        second,
        second,
        second,
    ])
    if provider_box is not None:
        provider_box.append(provider)
    base = ToolRegistry()
    base.register(_CapabilityLimitTool())
    ix_context = {"session_active_ix": ["capability_boundary"]}
    tools = TurnToolRegistry(base, ["capability_boundary"], ix_context=ix_context)
    return AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Get me the full text."}],
        tools=tools,
        model="test-model",
        max_iterations=5,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
        tools_factory=lambda context: TurnToolRegistry(base, [], ix_context=context),
    ))


def _selection_then_resolve_run(*, forced_selection: bool, resolve_responses=None, selection_response=None):
    """One scene, with a real scripted side effect before a capability boundary."""
    writes = []

    class RecordedWrite(EchoTool):
        @property
        def name(self):
            return "task_write"

        async def execute(self, value):
            writes.append(value)
            return {"saved": value}

    def response(*calls):
        return LLMResponse(content=None, tool_calls=list(calls), finish_reason="tool_calls")

    def read():
        return ToolCallRequest(id="read", name="task_echo", arguments={"value": "read"})

    selection = response(_ix_call(
        "IX_capability_boundary", setting="Suggest alternatives",
        evidence="The source is unavailable.", application="Try another source.",
    ))
    resolve = response(_ix_call(
        "IX_capability_resolve", boundary_id="b1", blocked_goal="get full text",
        boundary_kind="missing_data",
        recovery_options=[{"id": "r1", "intent": "find another copy"}],
        handoff_message="The full text is unavailable.",
    ))
    responses = [
        # Exercise a previously repaired operation in this same PA turn. Its
        # counters must not exhaust the later capability operation.
        *([response(read()), response(read())] if forced_selection else []),
        response(_ix_call("IX_verbosity", setting="Detailed")),
        response(ToolCallRequest(id="write", name="task_write", arguments={"value": "keep-once"})),
        response(ToolCallRequest(id="blocked", name="task_blocked", arguments={"value": "paper"})),
        # A fresh operation earns its own reminder before the forced pin;
        # the earlier verbosity repair cannot lend it a spent ladder count.
        *([response(read()), response(read())] if forced_selection else []),
        selection_response if selection_response is not None else selection,
        *(resolve_responses if resolve_responses is not None else [resolve]),
        LLMResponse(content="The source is unavailable; its citation cannot be verified."),
    ]
    provider = ScriptedProvider(responses)
    base = ToolRegistry()
    base.register(EchoTool())
    base.register(RecordedWrite())
    base.register(_NamedCapabilityLimitTool("task_blocked"))
    context = {
        "session_active_ix": ["verbosity"],
        "tool_action_types": {"task_echo": "read", "task_write": "internal_write", "task_blocked": "read"},
    }

    def factory(context):
        return TurnToolRegistry(
            base,
            ["verbosity", "capability_boundary"] if context.get("capability_limit_seen") else ["verbosity"],
            ix_context=context,
        )

    tools = factory(context)
    result = asyncio.run(AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "scene acc_001: verify the citation"}],
        tools=tools, model="scripted-no-network", max_iterations=15,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=context, tools_factory=factory,
    )))
    return result, provider, writes


def test_capability_selection_is_banked_before_separate_resolve():
    """Both spontaneous and pinned selections must survive the resolve gate."""
    for forced in (False, True):
        result, provider, writes = _selection_then_resolve_run(forced_selection=forced)
        assert result.stop_reason == "completed"
        assert result.ix_contract_exhaustion is None
        assert [e["name"] for e in result.tool_events if e.get("ix")].count("IX_capability_boundary") == 1
        assert result.collector.boundary_declared
        assert writes == ["keep-once"]
        assert all(call[0] == provider.calls[0][0] for call in provider.calls)
        assert [e["name"] for e in result.tool_events].count("task_blocked") == 1
        pins = [choice["function"]["name"] for choice in provider.tool_choices if isinstance(choice, dict)]
        assert ("IX_capability_boundary" in pins) is forced


def test_missing_resolve_still_exhausts_after_three_responses_and_keeps_actions():
    for forced in (False, True):
        result, provider, writes = _selection_then_resolve_run(
            forced_selection=forced,
            resolve_responses=[LLMResponse(content="No declaration.")] * 3,
        )
        assert result.stop_reason == "ix_contract_failure"
        assert result.ix_contract_exhaustion == {
            "attempts": 3, "contract": "capability", "operation": "capability_resolve",
        }
        assert writes == ["keep-once"]
        assert [e["name"] for e in result.tool_events if e.get("ix")].count("IX_capability_boundary") == 1
        assert provider.responses[-1].content.startswith("The source is unavailable")
        assert len(provider.responses) == 1  # No fourth attempt or scene replay.


def test_invalid_resolve_still_exhausts_after_three_responses():
    invalid = LLMResponse(content=None, tool_calls=[_ix_call(
        "IX_capability_resolve", boundary_id="b1", blocked_goal="get full text",
        boundary_kind="invalid_kind", recovery_options=[], handoff_message="Missing.",
    )])
    result, provider, writes = _selection_then_resolve_run(
        forced_selection=True, resolve_responses=[invalid] * 3,
    )
    assert result.stop_reason == "ix_contract_failure"
    assert result.ix_contract_exhaustion == {
        "attempts": 3, "contract": "capability", "operation": "capability_resolve",
    }
    assert writes == ["keep-once"]
    assert len(provider.responses) == 1


def test_selection_does_not_bypass_a_submitted_invalid_resolve():
    mixed = LLMResponse(content=None, tool_calls=[
        _ix_call("IX_capability_boundary", setting="Suggest alternatives"),
        _ix_call(
            "IX_capability_resolve", boundary_id="b1", blocked_goal="get full text",
            boundary_kind="invalid_kind", recovery_options=[], handoff_message="Missing.",
        ),
    ])
    result, provider, writes = _selection_then_resolve_run(
        forced_selection=False, selection_response=mixed, resolve_responses=[mixed] * 2,
    )
    assert result.stop_reason == "ix_contract_failure"
    assert result.ix_contract_exhaustion == {
        "attempts": 3, "contract": "capability", "operation": "capability_resolve",
    }
    assert writes == ["keep-once"]
    assert len(provider.responses) == 1
    assert not any(e.get("name") == "IX_capability_boundary" and e.get("ix") for e in result.tool_events)


def test_selection_with_business_calls_stays_blocked_until_resolve():
    mixed = LLMResponse(content=None, tool_calls=[
        _ix_call("IX_capability_boundary", setting="Suggest alternatives"),
        ToolCallRequest(id="unsafe-write", name="task_write", arguments={"value": "must-not-execute"}),
    ])
    result, provider, writes = _selection_then_resolve_run(
        forced_selection=False, selection_response=mixed, resolve_responses=[mixed] * 2,
    )
    assert result.stop_reason == "ix_contract_failure"
    assert result.ix_contract_exhaustion == {
        "attempts": 3, "contract": "capability", "operation": "capability_resolve",
    }
    assert writes == ["keep-once"]
    # user_d's deferral: the first mixed response banks its legal selection and
    # withholds the write, so it is progress, not a charged miss; the closing
    # text is the third missing resolve.
    assert len(provider.responses) == 0


def test_selection_only_still_requires_valid_arguments_with_three_attempt_limit():
    invalid = LLMResponse(content=None, tool_calls=[
        _ix_call("IX_capability_boundary", setting="not-a-legal-setting"),
    ])
    result, provider, writes = _selection_then_resolve_run(
        forced_selection=False, selection_response=invalid, resolve_responses=[invalid] * 2,
    )
    assert result.stop_reason == "ix_contract_failure"
    assert result.ix_contract_exhaustion == {
        "attempts": 3, "contract": "selection_args", "operation": "ix_tool_arguments",
    }
    assert writes == ["keep-once"]
    assert len(provider.responses) == 1
    assert not any(e.get("name") == "IX_capability_boundary" and e.get("ix") for e in result.tool_events)


def test_capability_resolve_missing_after_a_forced_call_is_a_contract_failure() -> None:  # T21a
    """A reminder, then `tool_choice`, and only then the failure (§3.5 (part 4))."""
    async def run():
        box: list[ScriptedProvider] = []
        result = await _capability_run(
            LLMResponse(content="Sorry, I could not get it."), box,
        )

        assert result.stop_reason == "ix_contract_failure"
        # No scorable answer at all -- the step fails and is retried instead.
        assert result.final_content is None
        event = next(
            event for event in result.tool_events
            if event.get("name") == "ix_boundary_unresolved"
        )
        assert event["attempts"] == 3
        assert event["reason"] == "missing"
        forced = next(
            event for event in result.tool_events
            if event.get("name") == "ix_render_payload_forced"
        )
        assert forced["tool"] == "IX_capability_resolve"
        # The third response is the total budget's terminal attempt. The retry
        # after the second miss pins the tool; an ignored force is not repeated.
        assert box[0].tool_choices == [
            None,
            None,
            None,
            {"type": "function", "function": {"name": "IX_capability_resolve"}},
        ]

    asyncio.run(run())


def test_capability_resolve_invalid_payload_three_times_is_a_contract_failure() -> None:  # T21b
    async def run():
        result = await _capability_run(LLMResponse(content="Declaring.", tool_calls=[
            _ix_call(
                "IX_capability_resolve",
                boundary_id="b1",
                blocked_goal="get the full text",
                boundary_kind="not_a_real_kind",
                recovery_options=[],
                handoff_message="Ask Imran for the manuscript?",
            ),
        ]))

        assert result.stop_reason == "ix_contract_failure"
        assert result.final_content is None
        event = next(
            event for event in result.tool_events
            if event.get("name") == "ix_boundary_unresolved"
        )
        assert event["reason"] == "invalid_payload"

    asyncio.run(run())


def test_find_and_hand_off_ends_the_turn_with_the_pa_handoff_line() -> None:
    async def run():
        result = await _capability_run(LLMResponse(content="Declaring.", tool_calls=[
            _ix_call(
                "IX_capability_resolve",
                boundary_id="b1",
                blocked_goal="get the full text",
                boundary_kind="missing_data",
                recovery_options=[{"id": "r1", "intent": "email the author"}],
                handoff_message="It is not in the catalogue. Shall I write to Imran?",
            ),
        ]))

        assert result.stop_reason == "ix_gate"
        assert result.final_content == (
            "It is not in the catalogue. Shall I write to Imran?"
        )
        assert result.collector.gate.attribute == "capability_boundary"

    asyncio.run(run())


def test_find_and_hand_off_stops_later_world_changes_in_the_trigger_response() -> None:
    """A capability limit takes effect between ordered sibling task calls."""

    async def run():
        provider = ScriptedProvider([
            LLMResponse(content="Checking the route.", tool_calls=[
                _ix_call("IX_capability_boundary", setting="Find and hand off"),
                ToolCallRequest(
                    id="before", name="task_before",
                    arguments={"value": "already done"},
                ),
                ToolCallRequest(
                    id="blocked", name="task_blocked",
                    arguments={"value": "restricted record"},
                ),
                ToolCallRequest(
                    id="after_write", name="task_after_write",
                    arguments={"value": "must wait"},
                ),
                ToolCallRequest(
                    id="after_read", name="task_after_read",
                    arguments={"value": "diagnostic evidence"},
                ),
            ]),
            LLMResponse(content="Declaring.", tool_calls=[
                _ix_call(
                    "IX_capability_resolve",
                    boundary_id="b1",
                    blocked_goal="complete the restricted action",
                    boundary_kind="missing_capability",
                    recovery_options=[{"id": "r1", "intent": "hand it to the user"}],
                    handoff_message="The restricted action needs you to take over.",
                ),
            ]),
        ])
        base = ToolRegistry()
        base.register(_NamedEchoTool("task_before"))
        base.register(_NamedCapabilityLimitTool("task_blocked"))
        base.register(_NamedEchoTool("task_after_write"))
        base.register(_NamedEchoTool("task_after_read"))
        ix_context = {
            "session_active_ix": ["capability_boundary"],
            "tool_action_types": {
                "task_before": "internal_write",
                "task_blocked": "read",
                "task_after_write": "external_action",
                "task_after_read": "read",
            },
        }
        tools = TurnToolRegistry(base, ["capability_boundary"], ix_context=ix_context)
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Handle the restricted task."}],
            tools=tools,
            model="test-model",
            max_iterations=4,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
            tools_factory=lambda context: TurnToolRegistry(base, [], ix_context=context),
        ))

        assert _executed(result) == ["task_before", "task_blocked", "task_after_read"]
        assert [item.tool for item in result.collector.held] == ["task_after_write"]
        assert result.collector.held[0].reason == "capability_handoff"
        assert result.stop_reason == "ix_gate"
        assert result.final_content == "The restricted action needs you to take over."

    asyncio.run(run())


def test_find_and_hand_off_keeps_sibling_calls_when_no_limit_is_hit() -> None:
    """The ordered sibling rule is inert until a real capability-limit result."""

    async def run():
        provider = ScriptedProvider([
            LLMResponse(content="Checking the route.", tool_calls=[
                _ix_call("IX_capability_boundary", setting="Find and hand off"),
                ToolCallRequest(
                    id="before", name="task_before", arguments={"value": "first"},
                ),
                ToolCallRequest(
                    id="ordinary", name="task_ordinary",
                    arguments={"value": "available"},
                ),
                ToolCallRequest(
                    id="after_write", name="task_after_write",
                    arguments={"value": "still allowed"},
                ),
                ToolCallRequest(
                    id="after_read", name="task_after_read",
                    arguments={"value": "more evidence"},
                ),
            ]),
            LLMResponse(content="Done."),
        ])
        base = ToolRegistry()
        for name in ("task_before", "task_ordinary", "task_after_write", "task_after_read"):
            base.register(_NamedEchoTool(name))
        ix_context = {
            "session_active_ix": ["capability_boundary"],
            "tool_action_types": {
                "task_before": "internal_write",
                "task_ordinary": "read",
                "task_after_write": "external_action",
                "task_after_read": "read",
            },
        }
        tools = TurnToolRegistry(base, ["capability_boundary"], ix_context=ix_context)
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Handle the task."}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
        ))

        assert _executed(result) == [
            "task_before", "task_ordinary", "task_after_write", "task_after_read",
        ]
        assert result.collector.held == []
        assert result.stop_reason == "completed"

    asyncio.run(run())


def test_task_expansion_does_not_gate_task_tools() -> None:
    async def run():
        provider = ScriptedProvider([
            LLMResponse(content=None, tool_calls=[
                _ix_call(
                    "IX_task_expansion",
                    setting="High",
                    evidence="The related note append is necessary and low risk.",
                    application="Do the bounded current-task expansion.",
                ),
                ToolCallRequest(
                    id="task_1", name="task_echo",
                    arguments={"value": "planning note saved"},
                ),
            ]),
            LLMResponse(content="Planning note done."),
        ])
        base = ToolRegistry()
        base.register(EchoTool())
        tools = TurnToolRegistry(base, ["task_expansion"])
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Make the planning note."}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            required_tool_names=tools.required_ix_tool_names,
        ))

        assert result.final_content == "Planning note done."
        assert [event.get("name") for event in result.tool_events] == [
            "IX_task_expansion",
            "task_echo",
        ]

    asyncio.run(run())


def test_proactive_outreach_event_records_send_decision_and_message() -> None:
    async def run():
        provider = ScriptedProvider([
            LLMResponse(content=None, tool_calls=[
                _ix_call(
                    "IX_proactive_outreach",
                    setting="High",
                    evidence="One relevant follow-up is acceptable.",
                    application="Send one material update.",
                    outreach_action="send",
                    outreach_message="The delivery is now arriving tomorrow morning.",
                ),
            ]),
            LLMResponse(content="Archive done with one optional next step."),
        ])
        ix_context = {"proactive_event": True, "active_request": False}
        tools = TurnToolRegistry(
            ToolRegistry(), ["proactive_outreach"], ix_context=ix_context
        )
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "system", "content": "A later environment event occurred."}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
        ))

        selection = next(event for event in result.tool_events if event.get("ix"))
        assert selection["outreach_action"] == "send"
        assert selection["outreach_message"] == (
            "The delivery is now arriving tomorrow morning."
        )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# regress12 acc_004 — the same tool, the same error, one guessed value
# ---------------------------------------------------------------------------


class _UnknownAddressTool(EchoTool):
    """acc_004's shape: one stable error with the guessed id inside it."""

    def __init__(self, name: str = "task_order_create") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, value: str) -> str:
        return f"Error executing tool {self._name}: unknown delivery address: {value}"


def _strike_run(
    tool_calls: list[ToolCallRequest],
    *,
    tool_names: tuple[str, ...] = ("task_order_create",),
    final: str = "Turn done.",
):
    provider = ScriptedProvider([
        LLMResponse(content="Placing the order.", tool_calls=tool_calls),
        LLMResponse(content=final),
    ])
    base = ToolRegistry()
    for name in tool_names:
        base.register(
            _UnknownAddressTool(name) if name == "task_order_create"
            else _NamedEchoTool(name)
        )
    ix_context = {
        "session_active_ix": ["autonomy_level"],
        "tool_action_types": {name: "read" for name in tool_names},
    }
    tools = TurnToolRegistry(base, ["autonomy_level"], ix_context=ix_context)
    return provider, AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Order the aerator."}],
        tools=tools,
        model="test-model",
        max_iterations=4,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
    ))


def _ran(result, name: str) -> int:
    return len([
        event for event in result.tool_events if event.get("name") == name
    ])


def _order_calls(*values: str) -> list[ToolCallRequest]:
    return [
        _task(f"o{index}", "task_order_create", value)
        for index, value in enumerate(values, start=1)
    ]


def test_three_identical_tool_failures_stop_the_fourth_and_ask_the_user() -> None:
    """Live acc_004: 46 `shopping_order_create` calls, 46 guessed address ids.

    Each answered "unknown delivery address" and the PA tried the next id
    instead of asking; the session cost 274 tool calls and 1531 s and ended on
    an invented order number. The fourth attempt does not run any more.
    """
    async def run():
        provider, coro = _strike_run([
            _ix_call("IX_autonomy_level", setting="Autonomous"),
            _plan({"id": "a1", "bound_item_ids": ["addr_home"]}),
            *_order_calls("addr_home", "addr_001", "addr_4b", "addr_002"),
        ])
        result = await coro

        assert result.final_content == "Turn done."
        # Three failures, and the fourth call never reached the tool.
        assert _ran(result, "task_order_create") == 3
        strike = next(
            event for event in result.tool_events
            if event.get("name") == "tool_failure_strike"
        )
        assert strike["tool"] == "task_order_create"
        assert strike["count"] == 3
        # The guessed id is out of the class; the sentence that repeats is in.
        assert "unknown delivery address" in strike["error_code"]
        assert "addr_4b" not in strike["error_code"]
        # The fourth call still gets a result, so the tool protocol holds.
        struck = next(
            message for message in result.messages
            if message.get("role") == "tool" and message.get("tool_call_id") == "o4"
        )
        assert "will not be called again on this turn" in struck["content"]
        # And the model is told what to do instead of guessing again.
        assert any(
            message.get("_mempa_internal")
            and "ask the user for the one it needs" in message["content"]
            for message in provider.calls[-1]
        )

    asyncio.run(run())


def test_a_strike_stops_one_tool_and_leaves_the_others_running() -> None:
    """Two tools never share a strike: the guard is per tool, per error."""
    async def run():
        _, coro = _strike_run(
            [
                _ix_call("IX_autonomy_level", setting="Autonomous"),
                _plan({"id": "a1", "bound_item_ids": ["addr_home"]}),
                *_order_calls("addr_home", "addr_001", "addr_4b", "addr_002"),
                _task("e1", "task_echo", "i1"),
            ],
            tool_names=("task_order_create", "task_echo"),
        )
        result = await coro

        assert _ran(result, "task_order_create") == 3
        assert _ran(result, "task_echo") == 1

    asyncio.run(run())


def test_a_capability_limit_is_never_counted_as_a_strike() -> None:
    """§3.5 owns the blocked-capability case; the strike must not take it."""
    async def run():
        provider = ScriptedProvider([
            LLMResponse(content="Fetching them.", tool_calls=[
                _ix_call("IX_capability_boundary", setting="Find and hand off"),
                *[
                    ToolCallRequest(
                        id=f"read_{index}", name="task_echo",
                        arguments={
                            "value": f"paper_{index}", "interaction_item_id": "i1",
                        },
                    )
                    for index in range(1, 5)
                ],
            ]),
            *[LLMResponse(content="Sorry, I could not get them.")] * 3,
        ])
        base = ToolRegistry()
        base.register(_CapabilityLimitTool())
        ix_context = {
            "session_active_ix": ["capability_boundary"],
            "tool_action_types": {"task_echo": "read"},
        }
        tools = TurnToolRegistry(base, ["capability_boundary"], ix_context=ix_context)

        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Get me the full texts."}],
            tools=tools,
            model="test-model",
            max_iterations=4,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
        ))

        assert _ran(result, "task_echo") == 4
        assert not [
            event for event in result.tool_events
            if event.get("name") == "tool_failure_strike"
        ]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Owner ruling 2026-09-08 — a selection re-sent after its beat, and the
# max-steps backstop for a tool call that keeps coming back an error.
# ---------------------------------------------------------------------------


def _reselection_run(
    responses: list[LLMResponse],
    *,
    selected: tuple[str, ...] = ("emotional_engagement",),
    max_iterations: int = 6,
):
    provider = ScriptedProvider(responses)
    base = ToolRegistry()
    base.register(EchoTool())
    ix_context = {
        "session_active_ix": ["emotional_engagement"],
        "session_ix_selected": selected,
        "beat": "b2",
        "tool_action_types": {"task_echo": "read"},
    }
    # The reaction beat: no selection tool is offered at all (live acc_027).
    tools = TurnToolRegistry(base, [], ix_context=ix_context)
    return provider, AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "That worked, thanks."}],
        tools=tools,
        model="test-model",
        max_iterations=max_iterations,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
        turn_index=7,
    ))


def test_a_selection_resent_after_its_beat_is_ignored_not_refused() -> None:
    """regress20 acc_027: `IX_emotional_engagement` on the reaction beat.

    The tool is offered on beat 1 and withdrawn once the session has its
    setting. The registry answered "not found", the PA read that as a broken
    call and re-sent it, and the third refusal ended the whole session. The
    setting is already recorded, so the call is answered and ignored.
    """
    async def run():
        _, coro = _reselection_run([
            LLMResponse(content="Glad it helped.", tool_calls=[
                _ix_call("IX_emotional_engagement", setting="Task-focused"),
            ]),
            LLMResponse(content="Anything else?"),
        ])
        result = await coro

        assert result.stop_reason == "completed"
        assert result.final_content == "Anything else?"
        # An ordinary tool result, not an error.
        answer = next(
            message for message in result.messages
            if message.get("role") == "tool"
            and message.get("name") == "IX_emotional_engagement"
        )
        assert answer["content"] == (
            "Selection for emotional_engagement is already recorded for this "
            "session; this call is ignored. Continue your reply."
        )
        event = next(
            event for event in result.tool_events
            if event.get("name") == "ix_reselection_ignored"
        )
        assert event["attribute"] == "emotional_engagement"
        assert event["setting"] == "Task-focused"
        assert event["turn_index"] == 7
        assert event["beat"] == "b2"
        assert event["call_index"] == 0
        # The refusal ladder never saw it.
        assert not [
            item for item in result.tool_events
            if item.get("name") == "ix_selection_invalid"
        ]

    asyncio.run(run())


def test_three_ignored_reselections_still_finish_the_turn() -> None:
    """The ignored call spends no repair budget: three of them, one turn."""
    async def run():
        _, coro = _reselection_run([
            *[
                LLMResponse(content="Working.", tool_calls=[
                    _ix_call("IX_emotional_engagement", setting="Task-focused"),
                ])
            ] * 3,
            LLMResponse(content="Done."),
        ])
        result = await coro

        assert result.stop_reason == "completed"
        assert result.final_content == "Done."
        assert result.ix_contract_failure_kind is None
        assert len([
            event for event in result.tool_events
            if event.get("name") == "ix_reselection_ignored"
        ]) == 3
        # And nothing was recorded as a selection for the session.
        assert not [
            event for event in result.tool_events
            if event.get("name") == "IX_emotional_engagement" and event.get("ix")
        ]

    asyncio.run(run())


def test_an_ix_tool_that_was_never_offered_is_still_refused() -> None:
    """Only a *recorded* selection is ignored; an unknown name keeps its error."""
    async def run():
        # Nothing settled this session, so there is no selection to point at.
        _, coro = _reselection_run(
            [
                *[
                    LLMResponse(content="Choosing.", tool_calls=[
                        _ix_call("IX_emotional_engagement", setting="Task-focused"),
                    ])
                ] * 3,
                LLMResponse(content="Done."),
            ],
            selected=(),
        )
        result = await coro

        assert not [
            event for event in result.tool_events
            if event.get("name") == "ix_reselection_ignored"
        ]
        assert [
            event["attempt"] for event in result.tool_events
            if event.get("name") == "ix_selection_invalid"
        ] == [1, 2, 3]
        assert result.stop_reason == "ix_contract_failure"
        assert result.ix_contract_failure_kind == "selection"

    asyncio.run(run())


def test_unavailable_ix_repair_allows_a_reply_without_a_replacement_tool() -> None:
    """A historic tool name must not become a demand to call it again."""
    async def run():
        provider = ScriptedProvider([
            LLMResponse(content="", tool_calls=[
                _ix_call("IX_solution_breadth", setting="Low"),
            ]),
            LLMResponse(content="You're welcome."),
        ])
        tools = TurnToolRegistry(ToolRegistry(), [], ix_context={})
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Thanks, that works."}],
            tools=tools, model="test-model", max_iterations=4,
        ))
        repair = provider.calls[1][-1]["content"]
        assert "IX_solution_breadth" in repair
        assert "not available on this turn" in repair
        assert "Do not call them again" in repair
        assert "history" in repair
        assert "If no interaction tool is applicable or required" in repair
        assert "Call: IX_solution_breadth" not in repair
        assert result.stop_reason == "completed"
        assert result.final_content == "You're welcome."
        assert len(provider.calls) == 2

    asyncio.run(run())


def test_a_mixed_refusal_batch_separates_the_two_kinds_of_refusal() -> None:
    """One batch, two causes: bad arguments on an offered tool and a tool that
    is not offered at all.

    memos x qwen `acc_047` carried only the second kind; `acc_056`'s render
    tool the same. This is the batch that has both, so each name has to be told
    the right thing -- the offered one is owed its arguments again, the
    unavailable one must not be called again -- and only the offered one may
    become a payload demand.
    """
    async def run():
        ix_context = {
            "session_active_ix": ["verbosity"],
            "beat": "b3",
            "tool_action_types": {},
        }
        tools = TurnToolRegistry(ToolRegistry(), ["verbosity"], ix_context=ix_context)
        assert "IX_verbosity" in tools.tool_names
        assert "IX_solution_breadth" not in tools.tool_names
        provider = ScriptedProvider([
            LLMResponse(content="", tool_calls=[
                # Offered, refused on its arguments (`setting` is required).
                _ix_call("IX_verbosity"),
                # Seen in history, not offered on this turn.
                _ix_call("IX_solution_breadth", setting="Low"),
            ]),
            LLMResponse(content="", tool_calls=[
                _ix_call("IX_verbosity", setting="Terse"),
            ]),
            LLMResponse(content="Two lemons ordered."),
        ])
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Order the lemons."}],
            tools=tools,
            model="test-model",
            max_iterations=6,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
            turn_index=4,
        ))

        repair = provider.calls[1][-1]["content"]
        # The unavailable half: named as unavailable, and never as a `Call:`.
        assert (
            "The following interaction tool(s) are not available on this turn: "
            "IX_solution_breadth." in repair
        )
        assert "A tool appearing in history does not mean it is available now." in repair
        assert "Call: IX_solution_breadth" not in repair
        # The offered half keeps the existing argument repair, on its own name.
        assert "Call: IX_verbosity." in repair
        # Both refusals were charged to the same budget on the same attempt.
        invalid = [
            event for event in result.tool_events
            if event.get("name") == "ix_selection_invalid"
        ]
        assert [event["attempt"] for event in invalid] == [1]
        assert set(invalid[0]["tools"]) == {"IX_verbosity", "IX_solution_breadth"}
        assert invalid[0]["budget"] == "ix_argument_retries"
        # A name outside this turn's tool list keeps the not-offered cause.
        assert invalid[0]["cause"] == "selection"
        # Only the offered tool became a demand, so satisfying it alone ends
        # the turn and nothing is ever acknowledged for the unavailable name.
        assert result.stop_reason == "completed"
        assert result.final_content == "Two lemons ordered."
        recorded = [
            message["content"]
            for request in provider.calls
            for message in request
            if message.get("_mempa_internal")
            and "is recorded for this turn" in message["content"]
        ]
        assert any("IX_verbosity" in note for note in recorded)
        assert not any("IX_solution_breadth" in note for note in recorded)

    asyncio.run(run())


def _retry_cap_run(responses: list[LLMResponse], *, max_iterations: int = 8):
    provider = ScriptedProvider(responses)
    base = ToolRegistry()
    base.register(StateErrorTool())
    ix_context = {
        "session_active_ix": ["verbosity"],
        "beat": "b3",
        "tool_action_types": {"task_echo": "read"},
    }
    tools = TurnToolRegistry(base, ["verbosity"], ix_context=ix_context)
    return provider, AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Log the reading."}],
        tools=tools,
        model="test-model",
        max_iterations=max_iterations,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
        turn_index=4,
    ))


def _failing_call(call_id: str) -> ToolCallRequest:
    return ToolCallRequest(
        id=call_id, name="task_echo", arguments={"value": "nope"},
    )


def test_a_tool_that_keeps_failing_ends_the_turn_normally() -> None:
    """Owner ruling 2026-09-08 — the max-steps backstop.

    The store refuses the same call on four successive iterations. The runner
    stops re-prompting and ships what the model has written; the turn is not a
    contract failure and the session continues.
    """
    async def run():
        provider, coro = _retry_cap_run([
            LLMResponse(content="First try.", tool_calls=[
                _ix_call("IX_verbosity", setting="Terse"),
                _failing_call("e1"),
            ]),
            LLMResponse(content="Second try.", tool_calls=[_failing_call("e2")]),
            LLMResponse(content="Third try.", tool_calls=[_failing_call("e3")]),
            LLMResponse(content="Fourth try.", tool_calls=[_failing_call("e4")]),
            LLMResponse(content="Never reached."),
        ])
        result = await coro

        assert result.stop_reason == "completed"
        assert result.ix_contract_failure_kind is None
        assert result.final_content == "Fourth try."
        event = next(
            event for event in result.tool_events
            if event.get("name") == "tool_retry_cap_reached"
        )
        assert event["tool"] == "task_echo"
        assert event["count"] == 4
        assert event["turn_index"] == 4
        assert event["beat"] == "b3"
        # The fifth request was never made.
        assert len(provider.calls) == 4

    asyncio.run(run())


def test_the_strike_prompt_is_delivered_once_before_the_cap() -> None:
    """Item 4 of the 2026-09-09 review — the two limits must not collide.

    At `TOOL_FAILURE_STRIKE_LIMIT` the runner appends "ask the user for the one
    it needs"; with both limits at 3 that message was written and the turn ended
    before any request carried it. The cap sits one iteration higher, so the
    strike gets exactly one delivery.
    """
    async def run():
        provider, coro = _retry_cap_run([
            LLMResponse(content="First try.", tool_calls=[
                _ix_call("IX_verbosity", setting="Terse"),
                _failing_call("e1"),
            ]),
            LLMResponse(content="Second try.", tool_calls=[_failing_call("e2")]),
            LLMResponse(content="Third try.", tool_calls=[_failing_call("e3")]),
            LLMResponse(content="Fourth try.", tool_calls=[_failing_call("e4")]),
            LLMResponse(content="Never reached."),
        ])
        result = await coro

        assert TOOL_RETRY_CAP == TOOL_FAILURE_STRIKE_LIMIT + 1
        strike = next(
            event for event in result.tool_events
            if event.get("name") == "tool_failure_strike"
        )
        assert strike["count"] == TOOL_FAILURE_STRIKE_LIMIT
        delivered = [
            index for index, request in enumerate(provider.calls)
            if any(
                message.get("_mempa_internal")
                and "ask the user for the one it needs" in message["content"]
                for message in request
            )
        ]
        # Written after the third failure, carried by the fourth request, and
        # the turn ends there.
        assert delivered == [3]

    asyncio.run(run())


def test_the_retry_cap_asks_once_more_when_the_model_wrote_nothing() -> None:
    """No content at all when the cap lands: one text-only request, then ship."""
    async def run():
        provider, coro = _retry_cap_run([
            LLMResponse(content="", tool_calls=[
                _ix_call("IX_verbosity", setting="Terse"),
                _failing_call("e1"),
            ]),
            LLMResponse(content="", tool_calls=[_failing_call("e2")]),
            LLMResponse(content="", tool_calls=[_failing_call("e3")]),
            LLMResponse(content="", tool_calls=[_failing_call("e4")]),
            LLMResponse(content="I could not log the reading."),
        ])
        result = await coro

        assert result.stop_reason == "completed"
        assert result.final_content == "I could not log the reading."
        assert [
            event["count"] for event in result.tool_events
            if event.get("name") == "tool_retry_cap_reached"
        ] == [4]
        # Exactly one extra request: the text-only one.
        assert len(provider.calls) == 5

    asyncio.run(run())


def _cap_ladder_run(
    responses: list[LLMResponse],
    *,
    active: list[str],
    extra_tools: dict[str, Tool] | None = None,
    action_types: dict[str, str] | None = None,
    max_iterations: int = 8,
):
    """A retry-cap scenario that also exercises one rule below it in the ladder."""
    provider = ScriptedProvider(responses)
    base = ToolRegistry()
    base.register(StateErrorTool())
    for tool in (extra_tools or {}).values():
        base.register(tool)
    ix_context = {
        "session_active_ix": active,
        "beat": "b4",
        "tool_action_types": {
            "task_echo": "read", **(action_types or {}),
        },
    }
    tools = TurnToolRegistry(base, active, ix_context=ix_context)
    return provider, AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Handle it."}],
        tools=tools,
        model="test-model",
        max_iterations=max_iterations,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
        tools_factory=lambda context: TurnToolRegistry(
            base, active, ix_context=context
        ),
    ))


def test_a_capability_limit_in_the_capped_iteration_keeps_the_boundary_contract() -> None:
    """Item 1 of the 2026-09-09 review.

    A `capability_limit` landing on the iteration the cap would end declares no
    boundary and emits no `ix_boundary_unresolved`: the turn shipped
    `completed` with the block unresolved. §3.5 outranks the cap.
    """
    async def run():
        _, coro = _cap_ladder_run(
            [
                LLMResponse(content="First try.", tool_calls=[
                    _ix_call("IX_capability_boundary", setting="Suggest alternatives"),
                    _failing_call("e1"),
                ]),
                LLMResponse(content="Second try.", tool_calls=[_failing_call("e2")]),
                LLMResponse(content="Third try.", tool_calls=[_failing_call("e3")]),
                # The fourth failure *and* the block, in one response.
                LLMResponse(content="Fourth try.", tool_calls=[
                    _failing_call("e4"),
                    ToolCallRequest(
                        id="b1", name="task_blocked", arguments={"value": "paper"},
                    ),
                ]),
                *[LLMResponse(content="Sorry, I could not.")] * 3,
            ],
            active=["capability_boundary"],
            extra_tools={"task_blocked": _NamedCapabilityLimitTool("task_blocked")},
            action_types={"task_blocked": "read"},
        )
        result = await coro

        assert not [
            event for event in result.tool_events
            if event.get("name") == "tool_retry_cap_reached"
        ]
        assert [
            event["reason"] for event in result.tool_events
            if event.get("name") == "ix_boundary_unresolved"
        ] == ["missing"]
        assert result.ix_contract_failure_kind == "capability"

    asyncio.run(run())


def test_the_capped_exit_still_owes_the_render_payload() -> None:
    """Item 2 of the 2026-09-09 review.

    The capped exit ships `last_segment` to the user, so it is a visible-message
    turn and owes `IX_render_reasoning_visibility` like any other. It shipped the
    prose with no payload and no `ix_render_payload_missing`.
    """
    async def run():
        _, coro = _cap_ladder_run(
            [
                LLMResponse(content="First try.", tool_calls=[
                    _ix_call("IX_reasoning_visibility", setting="Show"),
                    _failing_call("e1"),
                ]),
                LLMResponse(content="Second try.", tool_calls=[_failing_call("e2")]),
                LLMResponse(content="Third try.", tool_calls=[_failing_call("e3")]),
                LLMResponse(content="Fourth try.", tool_calls=[_failing_call("e4")]),
                LLMResponse(content="Never reached."),
            ],
            active=["reasoning_visibility"],
        )
        result = await coro

        # The cap is still what stopped the re-prompting...
        assert [
            event["count"] for event in result.tool_events
            if event.get("name") == "tool_retry_cap_reached"
        ] == [4]
        # ... and the turn is still unrenderable without the payload.
        missing = next(
            event for event in result.tool_events
            if event.get("name") == "ix_render_payload_missing"
        )
        assert missing["tools"] == ["IX_render_reasoning_visibility"]
        assert result.stop_reason == "ix_contract_failure"
        assert result.ix_contract_failure_kind == "render"
        assert result.final_content is None

    asyncio.run(run())


def test_the_capped_exit_ships_when_the_render_payload_is_in_hand() -> None:
    """The other half of item 2: payload present, so the cap ships as before."""
    async def run():
        _, coro = _cap_ladder_run(
            [
                LLMResponse(content="First try.", tool_calls=[
                    _ix_call("IX_reasoning_visibility", setting="Show"),
                    _ix_call(
                        "IX_render_reasoning_visibility",
                        user_facing_rationale="Checking the log first.",
                    ),
                    _failing_call("e1"),
                ]),
                LLMResponse(content="Second try.", tool_calls=[_failing_call("e2")]),
                LLMResponse(content="Third try.", tool_calls=[_failing_call("e3")]),
                LLMResponse(content="Fourth try.", tool_calls=[_failing_call("e4")]),
                LLMResponse(content="Never reached."),
            ],
            active=["reasoning_visibility"],
        )
        result = await coro

        assert result.stop_reason == "completed"
        assert result.ix_contract_failure_kind is None
        assert result.final_content == "Fourth try."
        assert not [
            event for event in result.tool_events
            if event.get("name") == "ix_render_payload_missing"
        ]

    asyncio.run(run())


def test_a_gate_in_the_capped_iteration_still_ends_at_the_gate() -> None:
    """Item 3 of the 2026-09-09 review.

    A gate opening on the iteration the cap would end lost its `beat_gate_turn`
    record, which `judge_pack` reads. The gate ends the turn; the cap does not.
    """
    async def run():
        _, coro = _cap_ladder_run(
            [
                LLMResponse(content="First try.", tool_calls=[
                    _ix_call("IX_autonomy_level", setting="Reactive"),
                    _plan({"id": "a1", "bound_item_ids": ["i1"]}),
                    _failing_call("e1"),
                ]),
                LLMResponse(content="Second try.", tool_calls=[_failing_call("e2")]),
                LLMResponse(content="Third try.", tool_calls=[_failing_call("e3")]),
                LLMResponse(content="Fourth try.", tool_calls=[
                    _failing_call("e4"),
                    _task("w1", "task_write", "i1", ask="Shall I save the draft?"),
                ]),
                LLMResponse(content="Never reached."),
            ],
            active=["autonomy_level"],
            extra_tools={"task_write": _NamedEchoTool("task_write")},
            action_types={"task_write": "internal_write"},
        )
        result = await coro

        assert result.stop_reason == "ix_gate"
        assert result.final_content == "Shall I save the draft?"
        assert [
            event["attribute"] for event in result.tool_events
            if event.get("name") == "beat_gate_turn"
        ] == ["autonomy_level"]
        assert not [
            event for event in result.tool_events
            if event.get("name") == "tool_retry_cap_reached"
        ]

    asyncio.run(run())


def test_a_repeated_capability_limit_never_reaches_the_retry_cap() -> None:
    """§3.5 owns the blocked capability; the cap must not cut its repair short."""
    async def run():
        provider = ScriptedProvider([
            LLMResponse(content="Fetching it.", tool_calls=[
                _ix_call("IX_capability_boundary", setting="Find and hand off"),
                ToolCallRequest(
                    id="r1", name="task_echo", arguments={"value": "paper"},
                ),
            ]),
            *[
                LLMResponse(content="Trying again.", tool_calls=[
                    ToolCallRequest(
                        id=f"r{index}", name="task_echo",
                        arguments={"value": "paper"},
                    ),
                ])
                for index in range(2, 5)
            ],
        ])
        base = ToolRegistry()
        base.register(_CapabilityLimitTool())
        ix_context = {
            "session_active_ix": ["capability_boundary"],
            "tool_action_types": {"task_echo": "read"},
        }
        tools = TurnToolRegistry(base, ["capability_boundary"], ix_context=ix_context)

        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Get me the full text."}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
            tools_factory=lambda context: TurnToolRegistry(
                base, [], ix_context=context
            ),
        ))

        assert not [
            event for event in result.tool_events
            if event.get("name") == "tool_retry_cap_reached"
        ]
        assert result.ix_contract_failure_kind == "capability"

    asyncio.run(run())


def test_one_batch_of_identical_failures_does_not_reach_the_retry_cap() -> None:
    """The cap is about re-prompting: acc_004's one-response burst is the strike's."""
    async def run():
        _, coro = _retry_cap_run([
            LLMResponse(content="Trying them all.", tool_calls=[
                _ix_call("IX_verbosity", setting="Terse"),
                _failing_call("e1"),
                _failing_call("e2"),
                _failing_call("e3"),
            ]),
            LLMResponse(content="That did not work."),
        ])
        result = await coro

        assert result.stop_reason == "completed"
        assert result.final_content == "That did not work."
        assert not [
            event for event in result.tool_events
            if event.get("name") == "tool_retry_cap_reached"
        ]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# hy4 / acc_031 — the render payload refused on its own arguments
# ---------------------------------------------------------------------------


def _render_refusal_run(
    responses: list[LLMResponse], *, max_iterations: int = 8
):
    """The hy4 shape: the setting is inherited, only the render is owed.

    `session_active_ix` carries `process_visibility` and nothing is owed at the
    `required == "always"` tier, so the ladder asks for
    `IX_render_process_visibility` alone -- reminder, pin, then whatever the
    model sends back.
    """
    provider = ScriptedProvider(responses)
    base = ToolRegistry()
    base.register(EchoTool())
    ix_context = {
        "session_active_ix": ["process_visibility"],
        "beat": "b2",
        "tool_action_types": {"task_echo": "read"},
    }
    tools = TurnToolRegistry(base, [], ix_context=ix_context)
    return provider, AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Put the order through."}],
        tools=tools,
        model="test-model",
        max_iterations=max_iterations,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
        inherited_ix_selections=[
            IXSelection(
                attribute="process_visibility",
                tool_name="IX_process_visibility",
                setting="Bookend",
            )
        ],
        turn_index=6,
    ))


#: The hy4 fourth response, verbatim from
#: `scratchpad/evidence/hy4_acc_031_ixselection_0603/failed_turn_events.json`:
#: the selection tool's three fields handed to the render tool, with the render
#: tool's own required field left out.
_HY4_SELECTION_ARGS_ON_RENDER = {
    "setting": "Bookend",
    "evidence": (
        "The user said the step-by-step play-by-play read wrong and asked for "
        "results with a single closing line instead."
    ),
    "application": (
        "Give results and one closing line; do not announce each lookup as it "
        "happens."
    ),
}


def test_a_refused_render_payload_is_recorded_as_a_render_failure() -> None:
    """hy4 / acc_031: `render` arguments refused, filed as a selection failure.

    The classification read "was the tool offered", which both refusals answer
    the same way, so a render payload missing `completion_message` was recorded
    `cause: selection_args` -- a choice of setting the model had in fact made
    correctly one response earlier.
    """
    async def run():
        _, coro = _render_refusal_run([
            *[
                LLMResponse(content="Working on it.", tool_calls=[
                    _ix_call(
                        "IX_render_process_visibility",
                        **_HY4_SELECTION_ARGS_ON_RENDER,
                    ),
                ])
            ] * 3,
            LLMResponse(content="Done."),
        ])
        result = await coro

        refusals = [
            event for event in result.tool_events
            if event.get("name") == "ix_selection_invalid"
        ]
        assert [event["tools"] for event in refusals] == [
            ["IX_render_process_visibility"]
        ] * 3
        assert [event["stage"] for event in refusals] == ["render"] * 3
        assert [event["missing_fields"] for event in refusals] == [
            ["completion_message"]
        ] * 3
        assert [event["cause"] for event in refusals] == ["render_args"] * 3
        assert result.stop_reason == "ix_contract_failure"
        assert result.ix_contract_failure_kind == "render_args"

    asyncio.run(run())


def test_a_refused_selection_still_reads_as_a_selection_failure() -> None:
    """The other half of the same classification: an offered selection tool."""
    async def run():
        provider = ScriptedProvider([
            *[
                LLMResponse(content="Choosing.", tool_calls=[
                    _ix_call("IX_verbosity"),
                ])
            ] * 3,
            LLMResponse(content="Done."),
        ])
        base = ToolRegistry()
        base.register(EchoTool())
        ix_context = {
            "session_active_ix": ["verbosity"],
            "beat": "b1",
            "tool_action_types": {"task_echo": "read"},
        }
        tools = TurnToolRegistry(base, ["verbosity"], ix_context=ix_context)
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Lay it out."}],
            tools=tools,
            model="test-model",
            max_iterations=8,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
            turn_index=3,
        ))

        refusals = [
            event for event in result.tool_events
            if event.get("name") == "ix_selection_invalid"
        ]
        assert [event["stage"] for event in refusals] == ["selection"] * 3
        assert [event["cause"] for event in refusals] == ["selection_args"] * 3
        # `setting` is the selection schema's only required field
        # (`pa/agent/ix/tools.py`: `required = ["setting"]`).
        assert [event["missing_fields"] for event in refusals] == [
            ["setting"]
        ] * 3
        assert result.ix_contract_failure_kind == "selection_args"

    asyncio.run(run())


def _unknown_ix_selection_run(responses: list[LLMResponse]):
    provider = ScriptedProvider(responses)
    base = ToolRegistry()
    base.register(EchoTool())
    ix_context = {
        "session_active_ix": ["verbosity"],
        "beat": "b1",
        "tool_action_types": {"task_echo": "read"},
    }
    tools = TurnToolRegistry(base, ["verbosity"], ix_context=ix_context)
    return provider, AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Lay it out."}],
        tools=tools,
        model="test-model",
        max_iterations=8,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
        turn_index=3,
    ))


def test_unknown_ix_argument_gets_one_repair_then_the_selection_can_succeed() -> None:
    async def run():
        provider, coro = _unknown_ix_selection_run([
            LLMResponse(content="Choosing.", tool_calls=[
                _ix_call("IX_verbosity", setting="Terse", unexpected="invented"),
            ]),
            LLMResponse(content="Choosing.", tool_calls=[
                _ix_call("IX_verbosity", setting="Terse"),
            ]),
            LLMResponse(content="Done."),
        ])
        result = await coro

        refusals = [event for event in result.tool_events if event.get("name") == "ix_selection_invalid"]
        assert [(event["attempt"], event["cause"]) for event in refusals] == [
            (1, "selection_args")
        ]
        assert result.stop_reason == "completed"
        assert any(event.get("name") == "IX_verbosity" for event in result.tool_events)
        assert not provider.responses

    asyncio.run(run())


def test_mixed_ix_argument_errors_share_one_three_response_pending_budget() -> None:
    async def run():
        provider, coro = _unknown_ix_selection_run([
            LLMResponse(content="Choosing.", tool_calls=[
                _ix_call("IX_verbosity", setting="Terse", unexpected="invented"),
            ]),
            LLMResponse(content="gateway unavailable", finish_reason="error"),
            LLMResponse(content="Choosing.", tool_calls=[_ix_call("IX_verbosity")]),
            LLMResponse(content="Choosing.", tool_calls=[
                _ix_call("IX_verbosity", setting="not-a-setting"),
            ]),
            LLMResponse(content="Never reached."),
        ])
        result = await coro

        refusals = [event for event in result.tool_events if event.get("name") == "ix_selection_invalid"]
        assert [event["attempt"] for event in refusals] == [1, 2, 3]
        assert [event["budget"] for event in refusals] == ["ix_argument_retries"] * 3
        assert any(event.get("name") == "provider_error_retried" for event in result.tool_events)
        assert result.stop_reason == "ix_contract_failure"
        assert result.ix_contract_failure_kind == "selection_args"
        assert len(provider.responses) == 1

    asyncio.run(run())


def test_provider_error_does_not_spend_the_ix_argument_budget() -> None:
    async def run():
        provider, coro = _unknown_ix_selection_run([
            LLMResponse(content="gateway unavailable", finish_reason="error"),
            *[
                LLMResponse(content="Choosing.", tool_calls=[
                    _ix_call("IX_verbosity", setting="Terse", unexpected="invented"),
                ])
                for _ in range(3)
            ],
            LLMResponse(content="Never reached."),
        ])
        result = await coro

        refusals = [event for event in result.tool_events if event.get("name") == "ix_selection_invalid"]
        assert [event["attempt"] for event in refusals] == [1, 2, 3]
        assert result.stop_reason == "ix_contract_failure"
        assert any(event.get("name") == "provider_error_retried" for event in result.tool_events)
        assert len(provider.responses) == 1

    asyncio.run(run())


def test_ds_nano_unknown_numeric_ix_argument_exhausts_as_typed_failure() -> None:
    """The live ``"13"`` key is refused without reaching the tool body."""
    async def run():
        provider = ScriptedProvider([
            *[
                LLMResponse(content="Choosing.", tool_calls=[
                    _ix_call(
                        "IX_information_elicitation",
                        setting="Structured",
                        **{"13": "unexpected"},
                    ),
                ])
                for _ in range(3)
            ],
            LLMResponse(content="Never reached."),
        ])
        base = ToolRegistry()
        base.register(EchoTool())
        ix_context = {
            "session_active_ix": ["information_elicitation"],
            "beat": "b1",
            "tool_action_types": {"task_echo": "read"},
        }
        tools = TurnToolRegistry(
            base, ["information_elicitation"], ix_context=ix_context
        )
        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Check it."}],
            tools=tools,
            model="test-model",
            max_iterations=8,
            required_tool_names=tools.required_ix_tool_names,
            ix_context=ix_context,
            turn_index=3,
        ))

        refusals = [
            event for event in result.tool_events
            if event.get("name") == "ix_selection_invalid"
        ]
        assert [event["attempt"] for event in refusals] == [1, 2, 3]
        assert [event["cause"] for event in refusals] == ["selection_args"] * 3
        errors = [
            event for event in result.tool_events
            if event.get("name") == "IX_information_elicitation"
            and event.get("status") == "error"
        ]
        assert len(errors) == 3
        assert all(event["args"]["13"] == "unexpected" for event in errors)
        assert all("unknown parameter: 13" in event["detail"] for event in errors)
        assert result.stop_reason == "ix_contract_failure"
        assert result.ix_contract_failure_kind == "selection_args"
        assert len(provider.responses) == 1

    asyncio.run(run())


def test_a_missed_payload_does_not_spend_the_argument_repair_budget() -> None:
    """hy4 / acc_031, the whole sequence: miss, miss, valid render, bad args.

    The two misses and the argument refusal shared one counter, so the *first*
    invalid argument arrived with the budget already at 2 and ended the turn
    with no repair asked for -- although the model had sent a correct payload
    one response earlier. The scripted responses below are test constructions;
    only the fourth one's arguments are the ones the live run recorded.
    """
    async def run():
        provider, coro = _render_refusal_run([
            LLMResponse(content="Confirmed on the debit."),
            LLMResponse(content="The order stays with you to put through."),
            LLMResponse(content=None, tool_calls=[
                _ix_call(
                    "IX_render_process_visibility",
                    completion_message="Confirmed on the debit.",
                ),
            ]),
            LLMResponse(content=None, tool_calls=[
                _ix_call(
                    "IX_render_process_visibility",
                    **_HY4_SELECTION_ARGS_ON_RENDER,
                ),
            ]),
            LLMResponse(content=None, tool_calls=[
                _ix_call(
                    "IX_render_process_visibility",
                    completion_message="The order is with you; the note is held.",
                ),
            ]),
            LLMResponse(content="Anything else?"),
        ], max_iterations=10)
        result = await coro

        assert result.stop_reason == "completed"
        assert result.final_content == "Anything else?"
        refusals = [
            event for event in result.tool_events
            if event.get("name") == "ix_selection_invalid"
        ]
        # One refusal, charged to its own budget and reported as its first
        # attempt -- not the third of a counter two misses had already spent.
        assert [event["attempt"] for event in refusals] == [1]
        assert [event["budget"] for event in refusals] == ["ix_argument_retries"]
        assert not provider.responses

    asyncio.run(run())


def test_repeated_invalid_payload_arguments_still_end_the_turn() -> None:
    """Carrier misses and bad arguments share the unresolved-operation budget."""
    async def run():
        provider, coro = _render_refusal_run([
            LLMResponse(content="Confirmed on the debit."),
            LLMResponse(content="The order stays with you to put through."),
            *[
                LLMResponse(content=None, tool_calls=[
                    _ix_call(
                        "IX_render_process_visibility",
                        **_HY4_SELECTION_ARGS_ON_RENDER,
                    ),
                ])
            ] * 3,
            LLMResponse(content="Never reached."),
        ], max_iterations=10)
        result = await coro

        assert [
            event["attempt"] for event in result.tool_events
            if event.get("name") == "ix_selection_invalid"
        ] == [1]
        assert result.stop_reason == "ix_contract_failure"
        assert result.ix_contract_failure_kind == "render_args"
        # Two carrier misses plus this rejected payload are the three invalid
        # responses; the separate local ladders cannot buy more calls.
        assert len(provider.responses) == 3

    asyncio.run(run())


def test_a_landed_payload_carrier_is_said_to_be_recorded_once() -> None:
    """hy4 table row 5 — the standing demand nothing ever answered.

    The reminder and the pin stay in `messages` for the rest of the turn, and
    the successful receipt is the payload echoed back, so after complying the
    model still reads two live instructions to call the same tool. hy4's fourth
    response re-sent it. One line, once per carrier per turn, and nothing is
    ignored: a later payload for the same attribute still overrides this one.
    """
    async def run():
        _, coro = _render_refusal_run([
            LLMResponse(content="Confirmed on the debit."),
            LLMResponse(content=None, tool_calls=[
                _ix_call(
                    "IX_render_process_visibility",
                    completion_message="Confirmed on the debit.",
                ),
            ]),
            LLMResponse(content="Anything else?"),
        ], max_iterations=6)
        result = await coro

        assert result.stop_reason == "completed"
        landed = [
            message["content"] for message in result.messages
            if message.get("_mempa_internal")
            and "IX_render_process_visibility is recorded" in message["content"]
        ]
        assert len(landed) == 1
        assert "Continue your reply" in landed[0]

    asyncio.run(run())


def test_an_undemanded_payload_carrier_is_not_announced() -> None:
    """No demand, no answer to it: the line is only for a carrier that was asked
    for, so an ordinary turn that sends its payload first time adds nothing."""
    async def run():
        _, coro = _render_refusal_run([
            LLMResponse(content=None, tool_calls=[
                _ix_call(
                    "IX_render_process_visibility",
                    completion_message="Confirmed on the debit.",
                ),
            ]),
            LLMResponse(content="Anything else?"),
        ], max_iterations=6)
        result = await coro

        assert result.stop_reason == "completed"
        assert not [
            message for message in result.messages
            if message.get("_mempa_internal")
            and "is recorded for this turn" in message["content"]
        ]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# qwen / acc_082 — an unknown business tool reached the gate before the
# registry was asked whether it exists
# ---------------------------------------------------------------------------

#: The tools the qwen beat actually offered, with the action types
#: `DEFAULT_TOOL_ACTION_TYPES` already gives them. `mcp_state_calendar_list` is
#: deliberately absent: that is the name the model invented.
_QWEN_TOOL_NAMES = (
    "mcp_state_calendar_event_search",
    "mcp_state_calendar_event_create",
)
_UNKNOWN_CALENDAR_TOOL = "mcp_state_calendar_list"


def _qwen_plan() -> ToolCallRequest:
    """The plan `IX_render_autonomy_level` carried on the failed turn.

    `plan_id`, both step ids, both intents and both `bound_item_ids` are taken
    from `scratchpad/evidence/qwen_acc082_fail3/acc_082/failed_turn_events.json`.
    """
    return _ix_call(
        "IX_render_autonomy_level",
        plan_id="sun_dinner_cal",
        plan=[
            {
                "id": "read_sun",
                "intent": (
                    "Read the Sunday, December 13 calendar to find the call "
                    "home and any conflicts"
                ),
                "is_key": False,
                "bound_item_ids": ["sun_cal_read"],
            },
            {
                "id": "create_dinner",
                "intent": (
                    "Create the calendar entry 'Jason dinner, green coat, cab "
                    "home' for Sunday after the call home"
                ),
                "is_key": False,
                "bound_item_ids": ["sun_cal_create"],
            },
        ],
        plan_done=False,
    )


def _qwen_run(responses: list[LLMResponse], *, max_iterations: int = 6):
    """The qwen beat: `autonomy_level` active, Self-directed chosen on it."""
    provider = ScriptedProvider(responses)
    base = ToolRegistry()
    for name in _QWEN_TOOL_NAMES:
        base.register(_NamedEchoTool(name))
    ix_context = {"session_active_ix": ["autonomy_level"]}
    tools = TurnToolRegistry(base, ["autonomy_level"], ix_context=ix_context)
    return provider, AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": (
            "Put it down as Jason dinner, green coat, cab home."
        )}],
        tools=tools,
        model="test-model",
        max_iterations=max_iterations,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
        turn_index=2,
    ))


def _qwen_selection() -> ToolCallRequest:
    return _ix_call(
        "IX_autonomy_level",
        setting="Self-directed",
        evidence="Direct execution instructions this turn.",
        application="The calendar read and the instructed entry proceed.",
    )


def _unknown_call(call_id: str, **arguments: Any) -> ToolCallRequest:
    """The invented name. Its arguments are a test construction: the failure
    file did not save the business calls' parameters."""
    return ToolCallRequest(
        id=call_id, name=_UNKNOWN_CALENDAR_TOOL, arguments=arguments,
    )


def test_an_unknown_task_tool_is_named_wrong_not_withheld() -> None:
    """qwen / acc_082: `mcp_state_calendar_list` does not exist.

    The wrapper put the invented name in the business-call list, `tool_action_type`
    read it as `external_action` (§3.8's fail-closed floor) and Self-directed
    withheld it for a confirmation -- so the registry's "Tool ... not found"
    never reached the model and the turn ended with no ask on the record. The
    name is resolved first now: not found, not run, not gated.
    """
    async def run():
        _, coro = _qwen_run([
            LLMResponse(content="Checking Sunday.", tool_calls=[
                _qwen_selection(),
                _qwen_plan(),
                _unknown_call("call_list", value="sun_cal_read",
                              interaction_item_id="sun_cal_read"),
            ]),
            LLMResponse(content="Sunday is free after the call home."),
        ])
        result = await coro

        assert result.stop_reason == "completed"
        answers = [
            message["content"] for message in result.messages
            if message.get("role") == "tool"
            and message.get("tool_call_id") == "call_list"
        ]
        assert len(answers) == 1
        assert answers[0].startswith(
            f"Error: Tool '{_UNKNOWN_CALENDAR_TOOL}' not found. Available: "
        )
        # Not run, not gated, and nothing to confirm.
        records = [
            (call.executed, call.status) for call in result.render_input.calls
            if call.tool_name == _UNKNOWN_CALENDAR_TOOL
        ]
        assert records == [(False, "error")]
        assert result.collector.held == []
        assert result.collector.consent == []
        assert not [
            event for event in result.tool_events
            if event.get("name") == "ix_gate_no_message"
        ]

    asyncio.run(run())


def test_an_unknown_task_tool_with_no_binding_is_also_just_not_found() -> None:
    """The §3.3 binding refusal is for calls that could have run: an unknown
    name has nothing to bind to, and the whole batch must not go with it."""
    async def run():
        _, coro = _qwen_run([
            LLMResponse(content="Checking Sunday.", tool_calls=[
                _qwen_selection(),
                _qwen_plan(),
                _unknown_call("call_list", value="whatever"),
            ]),
            LLMResponse(content="Sunday is free after the call home."),
        ])
        result = await coro

        assert result.stop_reason == "completed"
        answers = [
            message["content"] for message in result.messages
            if message.get("tool_call_id") == "call_list"
        ]
        assert len(answers) == 1 and "not found" in answers[0]
        assert result.collector.held == []

    asyncio.run(run())


def test_a_mixed_batch_answers_each_call_on_its_own_merits() -> None:
    """The unknown name errors, the registered read runs, the registered write
    keeps the gate it always had. Only the broken call is asked for again."""
    async def run():
        _, coro = _qwen_run([
            LLMResponse(content="Checking Sunday.", tool_calls=[
                _qwen_selection(),
                _plan(
                    {"id": "read_sun", "bound_item_ids": ["sun_cal_read"]},
                    {
                        "id": "create_dinner",
                        "is_key": True,
                        "bound_item_ids": ["sun_cal_create"],
                    },
                ),
                _unknown_call("call_list", value="sun_cal_read",
                              interaction_item_id="sun_cal_read"),
                _task("call_search", "mcp_state_calendar_event_search",
                      "sun_cal_read"),
                _task("call_create", "mcp_state_calendar_event_create",
                      "sun_cal_create", ask="Shall I put the dinner on?"),
            ]),
            LLMResponse(content="Sunday is free after the call home."),
        ])
        result = await coro

        results_by_id = {
            message["tool_call_id"]: message["content"]
            for message in result.messages
            if message.get("role") == "tool"
        }
        assert "not found" in results_by_id["call_list"]
        assert "not found" not in results_by_id["call_search"]
        assert _executed(result) == ["mcp_state_calendar_event_search"]
        # The key internal write is still held for the ask it always was.
        held = result.collector.held
        assert [item.interaction_item_id for item in held] == ["sun_cal_create"]
        assert [item.status for item in held] == ["held_for_confirmation"]

    asyncio.run(run())


def test_a_plan_step_is_not_executed_by_an_unknown_tool() -> None:
    """`plan_executed` is computed from the bound calls' `executed` flags, so an
    unknown attempt marked only `status: error` would still have closed out
    `read_sun`. The model's own `plan_done` is not the evidence."""
    async def run():
        _, coro = _qwen_run([
            LLMResponse(content="Checking Sunday.", tool_calls=[
                _qwen_selection(),
                _qwen_plan(),
                _unknown_call("call_list", value="sun_cal_read",
                              interaction_item_id="sun_cal_read"),
            ]),
            LLMResponse(content="I have read Sunday.", tool_calls=[
                _ix_call(
                    "IX_render_autonomy_level",
                    plan_id="sun_dinner_cal",
                    plan=[{
                        "id": "read_sun",
                        "intent": "Read Sunday",
                        "is_key": False,
                        "bound_item_ids": ["sun_cal_read"],
                    }],
                    plan_done=True,
                ),
            ]),
            LLMResponse(content="Sunday is free."),
        ])
        result = await coro

        assert result.stop_reason == "completed"
        assert released_ids(result.render_input)["plan_executed"] == []
        assert result.collector.held == []

    asyncio.run(run())


def test_the_same_binding_runs_once_when_the_model_uses_the_real_tool() -> None:
    """The repair path: not-found, then the registered search on the same id."""
    async def run():
        _, coro = _qwen_run([
            LLMResponse(content="Checking Sunday.", tool_calls=[
                _qwen_selection(),
                _qwen_plan(),
                _unknown_call("call_list", value="sun_cal_read",
                              interaction_item_id="sun_cal_read"),
            ]),
            LLMResponse(content="Trying the real one.", tool_calls=[
                _ix_call(
                    "IX_render_autonomy_level",
                    plan_id="sun_dinner_cal",
                    plan=[{
                        "id": "read_sun",
                        "intent": "Read Sunday",
                        "is_key": False,
                        "bound_item_ids": ["sun_cal_read"],
                    }],
                    plan_done=False,
                ),
                _task("call_search", "mcp_state_calendar_event_search",
                      "sun_cal_read"),
            ]),
            LLMResponse(content="Sunday is free after the call home."),
        ])
        result = await coro

        assert result.stop_reason == "completed"
        assert _executed(result) == ["mcp_state_calendar_event_search"]
        assert released_ids(result.render_input)["plan_executed"] == ["read_sun"]
        assert [
            (call.executed, call.status) for call in result.render_input.calls
            if call.tool_name == _UNKNOWN_CALENDAR_TOOL
        ] == [(False, "error")]
        assert result.collector.held == []

    asyncio.run(run())


# ---------------------------------------------------------------------------
# gpt / acc_082 — a struck calendar call gets one limited parameter repair
# ---------------------------------------------------------------------------

#: The arguments the six live `calendar_event_create` calls carried, from
#: `scratchpad/evidence/gpt_acc082_fail3/acc_082/pa_toolcalls.json`. `>TBD` in
#: `location` is the model's own placeholder and the store accepts it; only
#: `recurrence` is refused.
_GPT_DINNER_ARGS = {
    "title": "Jason dinner, green coat, cab home",
    "date": "2026-12-13",
    "start_time": "18:00",
    "end_time": "22:00",
    "location": ">TBD",
    "participants": ["Jason Reed"],
}


class _CalendarCreateTool(Tool):
    """`calendar_event_create` as the store answers it, for `recurrence` only.

    The refusal body is the store's: `ok: False` plus the classified
    `error_code` and the same sentence it has always produced.
    """

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mcp_state_calendar_event_create"

    @property
    def description(self) -> str:
        return "Create one calendar event or recurring series."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "location": {"type": "string"},
                "participants": {"type": "array", "items": {"type": "string"}},
                "recurrence": {"type": ["string", "null"]},
            },
            "required": ["title", "date"],
        }

    async def execute(self, **arguments: Any) -> Any:
        recurrence = arguments.get("recurrence")
        if recurrence:
            try:
                _parse_recurrence(recurrence, "ev_0022")
            except ValueError as exc:
                return {
                    "ok": False,
                    "error_code": CALENDAR_RECURRENCE_INVALID_ERROR_CODE,
                    "detail": str(exc),
                }
        self.created.append(dict(arguments))
        return {"event_id": f"ev_00{21 + len(self.created)}", "status": "created"}


def test_the_runner_and_the_store_agree_on_the_recurrence_error_code() -> None:
    """`pa/` imports nothing from `state_server/`, so the literal has two homes."""
    assert (
        CALENDAR_RECURRENCE_INVALID_ERROR_CODE
        == STORE_CALENDAR_RECURRENCE_INVALID_ERROR_CODE
    )


def _dinner_run(
    responses: list[LLMResponse], *, max_iterations: int = 8
) -> tuple[ScriptedProvider, _CalendarCreateTool, Any]:
    """The gpt beat: Self-directed, one non-key calendar write in the plan."""
    provider = ScriptedProvider(responses)
    calendar = _CalendarCreateTool()
    base = ToolRegistry()
    base.register(calendar)
    ix_context = {"session_active_ix": ["autonomy_level"]}
    tools = TurnToolRegistry(base, ["autonomy_level"], ix_context=ix_context)
    return provider, calendar, AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": (
            "Yes, add it. Nothing is on the calendar yet, so put it on with "
            "the title Jason dinner - green coat, cab home."
        )}],
        tools=tools,
        model="test-model",
        max_iterations=max_iterations,
        required_tool_names=tools.required_ix_tool_names,
        ix_context=ix_context,
        turn_index=28,
    ))


def _dinner_plan(*item_ids: str) -> ToolCallRequest:
    return _plan(*[
        {"id": f"create_{index}", "is_key": False, "bound_item_ids": [item_id]}
        for index, item_id in enumerate(item_ids)
    ])


def _dinner_call(
    call_id: str, item_id: str = "jason_dinner", **overrides: Any
) -> ToolCallRequest:
    arguments: dict[str, Any] = {
        **_GPT_DINNER_ARGS,
        "interaction_item_id": item_id,
        **overrides,
    }
    return ToolCallRequest(
        id=call_id, name="mcp_state_calendar_event_create", arguments=arguments,
    )


def _failing_dinner_iterations(count: int) -> list[LLMResponse]:
    """`count` responses, one refused `recurrence: '>TBD'` call each.

    One call per response, because the binding contract refuses two calls
    carrying the same `interaction_item_id` inside one response (§3.3 (part 2)·A)
    -- so a burst of identical calls on one authorised item cannot reach the
    store in the first place.
    """
    return [
        LLMResponse(content="Putting it on.", tool_calls=[
            _ix_call("IX_autonomy_level", setting="Self-directed"),
            _dinner_plan("jason_dinner"),
            _dinner_call(f"c{index}", recurrence=">TBD"),
        ])
        for index in range(1, count + 1)
    ]


def test_a_struck_calendar_call_gets_one_repair_and_creates_one_event() -> None:
    """gpt acc_082, the whole chain: authorised, refused three times on
    `recurrence`, struck, corrected under the same id, executed once.

    The strike prompt used to send the PA to ask the user for a value the user
    had never been asked about, and the struck tool then refused the corrected
    call as well -- so an evening the user had authorised in as many words
    never reached the calendar.
    """
    async def run():
        provider, calendar, coro = _dinner_run([
            *_failing_dinner_iterations(3),
            LLMResponse(content="Correcting it.", tool_calls=[
                _dinner_call("c4"),
            ]),
            LLMResponse(content="Sunday evening is on the calendar."),
        ], max_iterations=10)
        result = await coro

        assert result.stop_reason == "completed"
        assert result.final_content == "Sunday evening is on the calendar."
        # One event, from the corrected call, with no recurrence rule.
        assert len(calendar.created) == 1
        assert calendar.created[0].get("recurrence") is None
        assert calendar.created[0]["title"] == _GPT_DINNER_ARGS["title"]
        events = [event.get("name") for event in result.tool_events]
        assert events.count("tool_failure_strike") == 1
        assert events.count("tool_recurrence_repair_allowed") == 1
        # §3 step 5: the repair prompt is reachable -- the per-iteration cap
        # (one above the strike limit) exists so the extra request happens.
        assert TOOL_RETRY_CAP == TOOL_FAILURE_STRIKE_LIMIT + 1
        assert "tool_retry_cap_reached" not in events
        # The strike prompt named the field and the item, and did not send the
        # PA to the user.
        prompts = [
            message["content"] for message in result.messages
            if message.get("_mempa_internal")
            and "failed 3 times on this turn" in message["content"]
        ]
        assert len(prompts) == 1
        assert "`recurrence` omitted or null" in prompts[0]
        assert "jason_dinner" in prompts[0]
        assert "ask the user for the one it needs" not in prompts[0]
        assert not provider.responses

    asyncio.run(run())


def test_nothing_is_created_twice_once_the_repair_has_run() -> None:
    """§5: one confirmation, one record. The repair spends its one attempt, and
    the id it ran on is a spent release, so a second corrected call is held
    before the strike is even consulted -- the duplicate guard that already
    existed, reached because the repair was allowed to run at all."""
    async def run():
        _, calendar, coro = _dinner_run([
            *_failing_dinner_iterations(3),
            LLMResponse(content="Correcting it.", tool_calls=[
                _dinner_call("c4"),
            ]),
            LLMResponse(content="Making sure.", tool_calls=[
                _dinner_call("c5"),
            ]),
            LLMResponse(content="Sunday evening is on the calendar."),
        ], max_iterations=10)
        result = await coro

        assert len(calendar.created) == 1
        events = [event.get("name") for event in result.tool_events]
        assert events.count("tool_recurrence_repair_allowed") == 1
        assert events.count("ix_release_spent") == 1
        assert [item.interaction_item_id for item in result.collector.held] == [
            "jason_dinner"
        ]

    asyncio.run(run())


def test_a_still_illegal_value_is_not_a_repair_and_ends_the_turn() -> None:
    """Another invalid recurrence is another parameter error, not a correction.

    It never runs, and the iteration it spent is counted: the per-iteration
    retry cap ends the turn there rather than letting the model go round again.
    That is the bound §3's acceptance asks for -- the one attempt is an
    exception, not an open loop.
    """
    async def run():
        provider, calendar, coro = _dinner_run([
            *_failing_dinner_iterations(3),
            LLMResponse(content="Correcting it.", tool_calls=[
                _dinner_call("c4", recurrence="whenever"),
            ]),
            LLMResponse(content="Correcting it properly.", tool_calls=[
                _dinner_call("c5"),
            ]),
            LLMResponse(content="Sunday evening is on the calendar."),
        ], max_iterations=10)
        result = await coro

        assert calendar.created == []
        assert [
            event.get("name") for event in result.tool_events
        ].count("tool_retry_cap_reached") == 1
        # The proper correction was never asked for: the turn stopped first.
        assert len(provider.responses) == 2

    asyncio.run(run())


def test_a_changed_arrangement_under_the_same_id_is_not_the_repair() -> None:
    """§4: a different evening is a different act, id or no id."""
    async def run():
        _, calendar, coro = _dinner_run([
            *_failing_dinner_iterations(3),
            LLMResponse(content="Moving it to Monday.", tool_calls=[
                _dinner_call("c4", date="2026-12-14"),
            ]),
            LLMResponse(content="I could not put it on."),
        ], max_iterations=10)
        result = await coro

        assert calendar.created == []
        assert not [
            event for event in result.tool_events
            if event.get("name") == "tool_recurrence_repair_allowed"
        ]

    asyncio.run(run())


def test_the_repair_under_a_new_id_is_sent_back_to_the_authorised_one() -> None:
    """§4: the model minted `create-corrected-jason-dinner` live. The binding is
    the model's to repair -- rewriting the id here would move the user's
    authorisation onto a call they never saw -- and the budget stays unspent."""
    async def run():
        _, calendar, coro = _dinner_run([
            LLMResponse(content="Putting it on.", tool_calls=[
                _ix_call("IX_autonomy_level", setting="Self-directed"),
                _dinner_plan("jason_dinner", "create-corrected-jason-dinner"),
                _dinner_call("c1", recurrence=">TBD"),
            ]),
            LLMResponse(content="Again.", tool_calls=[
                _dinner_call("c2", recurrence=">TBD"),
            ]),
            LLMResponse(content="Again.", tool_calls=[
                _dinner_call("c3", recurrence=">TBD"),
            ]),
            LLMResponse(content="Correcting it.", tool_calls=[
                _dinner_call("c4", item_id="create-corrected-jason-dinner"),
            ]),
            LLMResponse(content="Correcting it again.", tool_calls=[
                _dinner_call("c5"),
            ]),
            LLMResponse(content="Sunday evening is on the calendar."),
        ], max_iterations=10)
        result = await coro

        rebinds = [
            event for event in result.tool_events
            if event.get("name") == "tool_recurrence_repair_rebind"
        ]
        assert [event["item_id"] for event in rebinds] == ["jason_dinner"]
        # The budget survived the misbound attempt and the re-bound call ran.
        assert len(calendar.created) == 1
        assert [
            event.get("name") for event in result.tool_events
        ].count("tool_recurrence_repair_allowed") == 1

    asyncio.run(run())


def test_a_refused_recurrence_leaves_the_authorisation_in_place() -> None:
    """§4: nothing was written, so the item is still "authorised, not done".

    Checked here rather than assumed: the consent and release paths only
    discard a grant for a call that executed with `status == "ok"`.
    """
    async def run():
        _, calendar, coro = _dinner_run([
            *_failing_dinner_iterations(3),
            LLMResponse(content="Correcting it.", tool_calls=[
                _dinner_call("c4"),
            ]),
            LLMResponse(content="Sunday evening is on the calendar."),
        ], max_iterations=10)
        result = await coro

        # The three refusals neither held the item nor closed it out.
        assert result.collector.held == []
        assert len(calendar.created) == 1
        refused = [
            call for call in result.render_input.calls
            if call.tool_name == "mcp_state_calendar_event_create"
            and call.status == "error"
        ]
        assert len(refused) == 3
        assert all(call.interaction_item_id == "jason_dinner" for call in refused)

    asyncio.run(run())
