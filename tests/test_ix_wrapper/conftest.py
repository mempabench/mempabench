"""Shared fakes for the cross-package IX wrapper tests.

Everything here is a fake provider, a fake tool or a recording memory adapter:
no LLM call, no network, no state server. The shapes follow
`tests/test_pa/test_ix_wrapper_loop.py`, which is the loop-level half of the
same contract set — this package is the cross-package half.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from memory.adapter import PromptHistoryEvictionPolicy
from memory.results import RetrieveResult, WriteReceipt, WriteStatus
from pa.agent.ix.render_types import CONSENT_DECISIONS_BLOCK
from pa.agent.loop import AgentLoop
from pa.agent.tools.base import Tool
from pa.bus.queue import MessageBus
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest

SESSION_KEY = "benchmark:main"
SCRIPT_SESSION = "acc_001"


class EchoTool(Tool):
    """A task tool with a free-form argument list, so any per-call field fits."""

    def __init__(self, name: str = "task_echo") -> None:
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

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

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"value": kwargs.get("value", "")}


class CapabilityLimitTool(EchoTool):
    """The `capability_limit` result shape a fixture-blocked read produces."""

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        import json

        return json.dumps({
            "ok": False,
            "error_code": "capability_limit",
            "boundary_kind": "missing_data",
        })


class RefusingTool(EchoTool):
    """A tool the world refuses: the call runs, the act does not happen.

    The shape live `appointment_create` returns when the store gate rejects an
    argument (regress15/16 acc_039) -- `ok: false`, which `_tool_result_error`
    reads as `status="error"`.
    """

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        import json

        return json.dumps({"ok": False, "error": "the clinic refused the reason"})


class RecordingAdapter:
    backend_name = "null"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.after_adapter_commit

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def setup_scope(self, scope) -> None:
        return None

    async def retrieve(self, request):
        return RetrieveResult(raw=[])

    def format_context(self, result) -> str:
        return ""

    async def record_event(self, event) -> WriteReceipt:
        self.events.append(event)
        return WriteReceipt(status=WriteStatus.committed)


class RecordingMemoryIO:
    """A `MemoryIOAuditSink` that keeps every record for T12."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, operation: dict[str, Any]) -> None:
        self.records.append(operation)


class ScriptProvider(LLMProvider):
    """Replays scripted responses; keeps every request and tool list it saw."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.requests: list[list[dict[str, Any]]] = []
        self.tool_definitions: list[list[dict[str, Any]]] = []
        self.tool_choices: list[Any] = []

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self.requests.append([dict(message) for message in messages])
        self.tool_definitions.append(list(tools or []))
        self.tool_choices.append(tool_choice)
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)

    def offered(self, index: int = -1) -> list[str]:
        return [
            definition["function"]["name"]
            for definition in self.tool_definitions[index]
        ]

    def definition(self, name: str, index: int = -1) -> dict[str, Any]:
        return next(
            definition for definition in self.tool_definitions[index]
            if definition["function"]["name"] == name
        )


def build_loop(
    workspace: Path,
    provider: LLMProvider,
    *tool_names: str,
    memory_io: RecordingMemoryIO | None = None,
) -> tuple[AgentLoop, RecordingAdapter]:
    adapter = RecordingAdapter()
    loop = AgentLoop(
        bus=MessageBus(), pa_llm_provider=provider, workspace=workspace,
        pa_model="test-model", memory_adapter=adapter,
        require_committed_adapter_writes=True,
        **({"memory_io_recorder": memory_io} if memory_io is not None else {}),
    )
    for name in tool_names or ("task_echo",):
        loop.tools.register(EchoTool(name))
    loop.sessions.create(SESSION_KEY)
    return loop, adapter


def turn_metadata(
    *,
    active_ix: list[str],
    action_types: dict[str, str] | None = None,
    consent: list[dict[str, str]] | None = None,
    script_session_id: str = SCRIPT_SESSION,
    extra_ix_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run_id": "run_001",
        "persona_id": "user_a",
        "script_session_id": script_session_id,
        "active_ix": active_ix,
        "ix_context": {
            "session_active_ix": active_ix,
            "tool_action_types": action_types or {},
            **(extra_ix_context or {}),
        },
    }
    if consent is not None:
        metadata[CONSENT_DECISIONS_BLOCK] = consent
    return metadata


async def run_turn(
    loop: AgentLoop,
    text: str,
    *,
    active_ix: list[str],
    action_types: dict[str, str] | None = None,
    consent: list[dict[str, str]] | None = None,
    script_session_id: str = SCRIPT_SESSION,
    extra_ix_context: dict[str, Any] | None = None,
):
    return await loop.process_direct_message(
        text,
        session_key=SESSION_KEY,
        metadata=turn_metadata(
            active_ix=active_ix,
            action_types=action_types,
            consent=consent,
            script_session_id=script_session_id,
            extra_ix_context=extra_ix_context,
        ),
    )


def ix(name: str, call_id: str | None = None, **arguments: Any) -> ToolCallRequest:
    return ToolCallRequest(id=call_id or f"{name}_1", name=name, arguments=arguments)


def selection(attribute: str, setting: str) -> ToolCallRequest:
    return ix(
        f"IX_{attribute}",
        setting=setting,
        evidence="the user said so last time",
        application="apply it to this turn",
    )


def plan_call(*items: dict[str, Any], plan_id: str = "p1", done: bool = False) -> ToolCallRequest:
    return ix(
        "IX_render_autonomy_level",
        plan_id=plan_id,
        plan=[
            {
                "id": item["id"],
                "intent": item.get("intent", "do the step"),
                "is_key": item.get("is_key", False),
                "bound_item_ids": item["bound_item_ids"],
            }
            for item in items
        ],
        plan_done=done,
    )


def task_call(
    call_id: str,
    name: str,
    item_id: str = "",
    *,
    ask: str = "",
    notice: str = "",
    value: str = "",
) -> ToolCallRequest:
    arguments: dict[str, Any] = {"value": value or item_id or call_id}
    if item_id:
        arguments["interaction_item_id"] = item_id
    if ask:
        arguments["_ix_confirmation_message"] = ask
    if notice:
        arguments["_ix_notice"] = notice
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def executed_tools(outbound: Any) -> list[str]:
    """Task tools that really ran this turn, read off the tool log."""
    return [
        event.get("name")
        for event in outbound.metadata.get("tool_events") or []
        if not event.get("ix_event")
        and not event.get("process")
        and not str(event.get("name") or "").startswith("IX_")
    ]


def run(coro) -> Any:
    return asyncio.run(coro)


@pytest.fixture
def loop_factory(tmp_path: Path):
    """Give each scripted scenario its own workspace under one tmp_path."""
    counter = {"n": 0}

    def make(provider: LLMProvider, *tool_names: str, **kwargs: Any):
        counter["n"] += 1
        workspace = tmp_path / f"ws_{counter['n']}"
        workspace.mkdir()
        return build_loop(workspace, provider, *tool_names, **kwargs)

    return make
