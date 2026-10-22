"""Loop-level IX wrapper contracts (plan §2.4, §3.6, §3.7, §4.2).

These are not "does the renderer format nicely" tests. Each one asks whether a
semantic boundary can be walked around: can held content reach history or
memory, can a gated action run without the user's consent, can a stale hold
survive a retry, can the wrapper's own internal prompts leak into the record.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from memory.adapter import PromptHistoryEvictionPolicy
from memory.turn_experience import parse_blocks
from memory.results import RetrieveResult, WriteReceipt, WriteStatus
from pa.agent.ix.collector import TURN_REMINDER_PREFIX
from pa.agent.ix.render_types import CONSENT_DECISIONS_BLOCK
from pa.agent.loop import AgentLoop
from pa.agent.tools.base import Tool
from pa.bus.queue import MessageBus
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest

SESSION_KEY = "benchmark:main"
SCRIPT_SESSION = "acc_001"


class _Echo(Tool):
    def __init__(self, name: str = "task_echo") -> None:
        self._name = name

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

    async def execute(self, value: str = "") -> dict[str, str]:
        return {"value": value}


class _RecordingAdapter:
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


class _ScriptProvider(LLMProvider):
    """Replays a fixed list of responses and keeps every request it saw."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.requests: list[list[dict[str, Any]]] = []

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self.requests.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)


def _loop(tmp_path: Path, provider: LLMProvider, *tool_names: str):
    adapter = _RecordingAdapter()
    loop = AgentLoop(
        bus=MessageBus(), pa_llm_provider=provider, workspace=tmp_path,
        pa_model="test-model", memory_adapter=adapter,
        require_committed_adapter_writes=True,
    )
    for name in tool_names or ("task_echo",):
        loop.tools.register(_Echo(name))
    loop.sessions.create(SESSION_KEY)
    return loop, adapter


async def _turn(
    loop: AgentLoop,
    text: str,
    *,
    active_ix: list[str],
    action_types: dict[str, str] | None = None,
    consent: list[dict[str, str]] | None = None,
    script_session_id: str = SCRIPT_SESSION,
):
    metadata: dict[str, Any] = {
        "run_id": "run_001",
        "persona_id": "user_a",
        "script_session_id": script_session_id,
        "active_ix": active_ix,
        "ix_context": {
            "session_active_ix": active_ix,
            "tool_action_types": action_types or {},
        },
    }
    if consent is not None:
        metadata[CONSENT_DECISIONS_BLOCK] = consent
    return await loop.process_direct_message(
        text, session_key=SESSION_KEY, metadata=metadata,
    )


def _ix(name: str, **arguments: Any) -> ToolCallRequest:
    return ToolCallRequest(id=f"{name}_1", name=name, arguments=arguments)


def _plan(*items: dict[str, Any]) -> ToolCallRequest:
    return _ix(
        "IX_render_autonomy_level",
        plan_id="p1",
        plan=[
            {
                "id": item["id"],
                "intent": item.get("intent", "do the step"),
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


# ---------------------------------------------------------------------------
# T3 · turn envelope invariants
# ---------------------------------------------------------------------------

def test_a_plain_turn_persists_the_user_message_once_and_the_public_turn(tmp_path) -> None:
    async def run() -> None:
        provider = _ScriptProvider([LLMResponse(content="Here is the answer.")])
        loop, adapter = _loop(tmp_path, provider)

        outbound = await _turn(loop, "What is on Monday?", active_ix=[])

        assert outbound.content == "Here is the answer."
        persisted = loop.sessions.get(SESSION_KEY).messages
        assert [(m["role"], m["content"]) for m in persisted] == [
            ("user", "What is on Monday?"),
            ("assistant", "Here is the answer."),
        ]
        assert adapter.events[0].messages == [
            {"role": "user", "content": "What is on Monday?"},
            {"role": "assistant", "content": "Here is the answer."},
        ]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# T4 / T9 · mixed executed / held tool protocol
# ---------------------------------------------------------------------------

def _gate_turn_provider(second_turn: list[LLMResponse] | None = None) -> _ScriptProvider:
    """Turn 1: one read runs, one external action is gated by Reactive."""
    return _ScriptProvider([
        LLMResponse(content="Checking first.", tool_calls=[
            _ix("IX_autonomy_level", setting="Reactive", evidence="asked to be consulted",
                application="confirm before acting"),
            _plan(
                {"id": "a1", "bound_item_ids": ["i1"]},
                {"id": "a2", "bound_item_ids": ["i2"]},
            ),
            _task("read_1", "task_read", "i1"),
            _task("send_1", "task_send", "i2", ask="Shall I send it to Dale?"),
        ]),
        *(second_turn or []),
    ])


_ACTION_TYPES = {"task_read": "read", "task_send": "external_action"}


def test_a_gate_turn_keeps_only_the_executed_half_of_the_tool_protocol(tmp_path) -> None:
    async def run() -> None:
        provider = _gate_turn_provider()
        loop, adapter = _loop(tmp_path, provider, "task_read", "task_send")

        outbound = await _turn(
            loop, "Check the stock and email Dale.",
            active_ix=["autonomy_level"], action_types=_ACTION_TYPES,
        )

        # The prose of the turn plus the ask; the ask is the closing line.
        assert outbound.content.endswith("Shall I send it to Dale?")
        persisted = loop.sessions.get(SESSION_KEY).messages
        declared = [
            call["id"]
            for message in persisted
            for call in message.get("tool_calls") or []
        ]
        results = [
            message["tool_call_id"] for message in persisted if message["role"] == "tool"
        ]
        # T4: declared ids and result ids match in order, so the history stays a
        # legal tool protocol. The held call left no declaration behind.
        assert declared == results
        assert "send_1" not in declared
        assert {"IX_autonomy_level_1", "IX_render_autonomy_level_1", "read_1"} <= set(declared)

        # T9: the last assistant message is exactly the rendered ask, and the
        # held call left no declaration in the *history* — projection 2 keeps
        # declared ids and result ids equal in order.
        assert persisted[-1]["role"] == "assistant"
        assert persisted[-1]["content"] == outbound.content
        memory = adapter.events[0].messages
        assert memory[0] == {
            "role": "user", "content": "Check the stock and email Dale.",
        }
        assert memory[-1] == {"role": "assistant", "content": outbound.content}
        # Memory is the other projection. The PA *did* declare the gated call,
        # so item #5 puts the declaration in — with no result, because it never
        # ran — while carrier C, which is wrapper state rather than an argument
        # the PA chose, is stripped out of it.
        blob = json.dumps(memory, default=str)
        assert '[tool_call name=task_send id=send_1]\\n{\\"value\\":\\"i2\\"}' in blob
        assert "[tool_result name=task_send" not in blob
        for carrier in ("interaction_item_id", "_ix_confirmation_message"):
            assert carrier not in blob

        held = outbound.metadata["ix_render"]["held"]["calls"]
        assert [item["tool"] for item in held] == ["task_send"]
        assert held[0]["status"] == "held_for_confirmation"

    asyncio.run(run())


def test_the_history_after_a_gate_turn_is_accepted_by_the_next_provider_call(tmp_path) -> None:
    async def run() -> None:
        provider = _gate_turn_provider([
            LLMResponse(content="Understood."),
            LLMResponse(content="Understood."),
        ])
        loop, _ = _loop(tmp_path, provider, "task_read", "task_send")

        await _turn(
            loop, "Check the stock and email Dale.",
            active_ix=["autonomy_level"], action_types=_ACTION_TYPES,
        )
        await _turn(loop, "Not yet.", active_ix=["autonomy_level"], action_types=_ACTION_TYPES)

        # The next real request carries the previous turn's protocol; assert it
        # is well formed rather than only inspecting the stored structure.
        sent = provider.requests[-1]
        declared = [
            call["id"] for message in sent for call in message.get("tool_calls") or []
        ]
        results = [
            message["tool_call_id"] for message in sent if message.get("role") == "tool"
        ]
        assert declared == results
        assert "send_1" not in json.dumps(sent, default=str)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# T12 · memory keeps the PA's whole first-person experience of the turn
# ---------------------------------------------------------------------------

def test_memory_keeps_the_turn_experience_and_no_internal_prompt(tmp_path) -> None:
    """The pool is the PA's own experience, in two roles and tagged blocks.

    What the PA produced or perceived as the assistant goes in — including its
    own `IX_<attribute>` choice, which is the half of the (choice, reaction)
    pair preference learning needs. What exists only because a benchmark is
    running stays out.
    """
    async def run() -> None:
        provider = _gate_turn_provider()
        loop, adapter = _loop(tmp_path, provider, "task_read", "task_send")

        outbound = await _turn(
            loop, "Check the stock and email Dale.",
            active_ix=["autonomy_level"], action_types=_ACTION_TYPES,
        )

        messages = adapter.events[0].messages
        assert {message["role"] for message in messages} == {"user", "assistant"}
        # Nothing that is not the person's own speech is ever `role: "user"`.
        assert [m["content"] for m in messages if m["role"] == "user"] == [
            "Check the stock and email Dale."
        ]
        assert messages[-1]["content"] == outbound.content

        kinds = [
            item.kind
            for message in messages[1:-1]
            for item in parse_blocks(message["content"])
        ]
        assert "ix_selection" in kinds and "tool_call" in kinds
        assert {"tool_result", "ix_result"} & set(kinds)

        blob = json.dumps(messages, default=str)
        assert (
            "[ix_selection attribute=autonomy_level setting=Reactive outcome=executed]"
        ) in blob
        for banned in (TURN_REMINDER_PREFIX, CONSENT_DECISIONS_BLOCK):
            assert banned not in blob

    asyncio.run(run())


# ---------------------------------------------------------------------------
# T13 · the per-turn restatement of a pass-through setting
# ---------------------------------------------------------------------------

def test_a_pass_through_setting_is_restated_on_a_later_turn_without_tool_calls(tmp_path) -> None:
    async def run() -> None:
        provider = _ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                _ix("IX_verbosity", setting="Detailed", evidence="asked for depth",
                    application="give the full picture"),
            ]),
            LLMResponse(content="Turn one answer."),
            LLMResponse(content="Turn two answer."),
        ])
        loop, adapter = _loop(tmp_path, provider)

        await _turn(loop, "Explain the plan.", active_ix=["verbosity"])
        outbound = await _turn(loop, "And the budget?", active_ix=["verbosity"])

        # Turn two calls no tool at all -- the old injection point never fired
        # there, so the chosen setting silently lapsed from turn two onwards.
        reminders = [
            message for message in provider.requests[-1]
            if str(message.get("content") or "").startswith(TURN_REMINDER_PREFIX)
        ]
        assert len(reminders) == 1
        assert "verbosity: Detailed" in reminders[0]["content"]

        # It is an internal message: not in the record, not in memory, not shown.
        assert TURN_REMINDER_PREFIX not in outbound.content
        persisted = json.dumps(loop.sessions.get(SESSION_KEY).messages, default=str)
        assert TURN_REMINDER_PREFIX not in persisted
        assert TURN_REMINDER_PREFIX not in json.dumps(
            [event.messages for event in adapter.events], default=str
        )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# T18 · the consent sideband is the gate for an external action
# ---------------------------------------------------------------------------

def _release_turn(decision: str = "release") -> list[LLMResponse]:
    return [
        LLMResponse(content="Sending now.", tool_calls=[
            _ix("IX_continue_autonomy_level", decisions=[
                {"id": "i2", "decision": decision, "user_evidence": "go ahead"},
            ]),
            _task("send_2", "task_send", "i2", ask="Shall I send it to Dale?"),
        ]),
        LLMResponse(content="Sent."),
    ]


async def _gate_then_reply(tmp_path, *, consent, decision="release"):
    provider = _gate_turn_provider(_release_turn(decision))
    loop, adapter = _loop(tmp_path, provider, "task_read", "task_send")
    await _turn(
        loop, "Check the stock and email Dale.",
        active_ix=["autonomy_level"], action_types=_ACTION_TYPES,
    )
    second = await _turn(
        loop, "Sure, go ahead.", active_ix=["autonomy_level"],
        action_types=_ACTION_TYPES, consent=consent,
    )
    return loop, provider, adapter, second


def _sent(provider: _ScriptProvider, loop: AgentLoop) -> bool:
    return any(
        message.get("role") == "tool" and message.get("name") == "task_send"
        for message in loop.sessions.get(SESSION_KEY).messages
    )


def test_an_allowed_external_action_runs_on_the_next_turn(tmp_path) -> None:
    async def run() -> None:
        loop, provider, _, second = await _gate_then_reply(
            tmp_path, consent=[{"interaction_item_id": "i2", "decision": "allow"}],
        )
        assert _sent(provider, loop)
        consent = second.metadata["ix_render"]["consent"]
        assert consent[0]["sideband_decision"] == "allow"
        assert consent[0]["released"] is True

    asyncio.run(run())


def test_a_denied_external_action_does_not_run_however_the_pa_reads_the_reply(tmp_path) -> None:
    async def run() -> None:
        # The PA says "release" and quotes the user; the sideband says no.
        loop, provider, _, second = await _gate_then_reply(
            tmp_path, consent=[{"interaction_item_id": "i2", "decision": "deny"}],
        )
        assert not _sent(provider, loop)
        consent = second.metadata["ix_render"]["consent"]
        assert consent[0]["pa_decision"] == "release"
        assert consent[0]["sideband_decision"] == "deny"
        assert consent[0]["released"] is False
        assert any(
            event.get("name") == "ix_consent_denied"
            for event in second.metadata["tool_events"]
        )

    asyncio.run(run())


def test_an_unclear_or_absent_verdict_fails_closed(tmp_path) -> None:
    async def run() -> None:
        variants = (
            [{"interaction_item_id": "i2", "decision": "unclear"}],
            # A duplicated id is not de-duplicated upstream: one dissenting
            # entry is enough to keep the item held.
            [{"interaction_item_id": "i2", "decision": "allow"},
             {"interaction_item_id": "i2", "decision": "unclear"}],
            [],
            None,
        )
        for index, consent in enumerate(variants):
            workspace = tmp_path / f"variant_{index}"
            workspace.mkdir()
            loop, provider, _, second = await _gate_then_reply(workspace, consent=consent)
            assert not _sent(provider, loop), consent
            assert any(
                event.get("name") == "ix_consent_unresolved"
                for event in second.metadata["tool_events"]
            ), consent

    asyncio.run(run())


def test_two_verdicts_for_one_id_resolve_against_the_action(tmp_path) -> None:
    async def run() -> None:
        # The sideband is not de-duplicated upstream. If the two entries
        # disagree, the outward action is the half that cannot be undone.
        loop, provider, _, second = await _gate_then_reply(
            tmp_path,
            consent=[
                {"interaction_item_id": "i2", "decision": "allow"},
                {"interaction_item_id": "i2", "decision": "deny"},
            ],
        )
        assert not _sent(provider, loop)
        consent = second.metadata["ix_render"]["consent"]
        assert consent[0]["sideband_decision"] == "deny"
        assert consent[0]["released"] is False

    asyncio.run(run())


def test_a_task_call_on_a_held_item_without_a_release_does_not_run(tmp_path) -> None:
    async def run() -> None:
        loop, provider, _, second = await _gate_then_reply(
            tmp_path,
            consent=[{"interaction_item_id": "i2", "decision": "allow"}],
            decision="keep",
        )
        assert not _sent(provider, loop)
        assert any(
            event.get("name") == "ix_release_missing"
            for event in second.metadata["tool_events"]
        )

    asyncio.run(run())


def test_the_consent_sideband_never_reaches_the_model(tmp_path) -> None:
    async def run() -> None:
        _, provider, _, second = await _gate_then_reply(
            tmp_path, consent=[{"interaction_item_id": "i2", "decision": "deny"}],
        )
        for request in provider.requests:
            assert CONSENT_DECISIONS_BLOCK not in json.dumps(request, default=str)
        assert CONSENT_DECISIONS_BLOCK not in json.dumps(
            second.metadata.get("tool_events"), default=str
        )
        assert CONSENT_DECISIONS_BLOCK not in second.metadata

    asyncio.run(run())


# ---------------------------------------------------------------------------
# T17 · step boundaries and retries
# ---------------------------------------------------------------------------

def test_an_open_hold_is_reported_at_the_step_boundary_and_cleared_with_the_settings(
    tmp_path,
) -> None:
    async def run() -> None:
        provider = _gate_turn_provider()
        loop, _ = _loop(tmp_path, provider, "task_read", "task_send")

        await _turn(
            loop, "Check the stock and email Dale.",
            active_ix=["autonomy_level"], action_types=_ACTION_TYPES,
        )
        key = (SESSION_KEY, SCRIPT_SESSION)
        assert loop._ix_hold_cache[key].open_episodes["autonomy_level"].calls_held

        still_open = loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION)
        assert still_open == [("autonomy_level", "p1")]

        # A retry must not inherit last attempt's hold *or* its chosen setting:
        # keeping one without the other leaves two mismatched state planes.
        assert key not in loop._ix_hold_cache
        assert key not in loop._ix_selection_cache

    asyncio.run(run())


# ---------------------------------------------------------------------------
# §3.7 (part 3) · a verdict is consent only while the wrapper is holding the id
# ---------------------------------------------------------------------------

def test_a_stale_allow_does_not_release_anything_after_the_episode_closed(
    tmp_path,
) -> None:
    """The sideband keeps answering ids nobody is holding any more.

    Live, the episode closed at t24 and the simulator still returned
    `{"call_003": "allow"}` at t27, t30 and t33. An `allow` that no open
    episode anchors is not evidence of anything, and leaving it in play means
    whatever next reuses that id is released without being asked about.
    """
    async def run() -> None:
        provider = _gate_turn_provider([
            *_release_turn(),
            # A third turn reuses the same id, with the same stale verdict.
            LLMResponse(content="Sending it again.", tool_calls=[
                _plan({"id": "a2", "bound_item_ids": ["i2"]}),
                _task("send_3", "task_send", "i2", ask="Shall I send it again?"),
            ]),
        ])
        loop, _ = _loop(tmp_path, provider, "task_read", "task_send")

        await _turn(
            loop, "Check the stock and email Dale.",
            active_ix=["autonomy_level"], action_types=_ACTION_TYPES,
        )
        second = await _turn(
            loop, "Sure, go ahead.", active_ix=["autonomy_level"],
            action_types=_ACTION_TYPES,
            consent=[{"interaction_item_id": "i2", "decision": "allow"}],
        )
        assert second.metadata["ix_render"]["episodes"] == {}
        sends_after_release = sum(
            1 for message in loop.sessions.get(SESSION_KEY).messages
            if message.get("role") == "tool" and message.get("name") == "task_send"
        )

        third = await _turn(
            loop, "Anything else?", active_ix=["autonomy_level"],
            action_types=_ACTION_TYPES,
            consent=[{"interaction_item_id": "i2", "decision": "allow"}],
        )

        # The stale verdict released nothing: the gate opened as it would have
        # on a first ask, and no consent row was written for an id no episode
        # was holding.
        assert sum(
            1 for message in loop.sessions.get(SESSION_KEY).messages
            if message.get("role") == "tool" and message.get("name") == "task_send"
        ) == sends_after_release
        assert third.metadata["ix_render"]["consent"] == []
        assert third.metadata["ix_render"]["gate"]["reason"] == "external_action_floor"
        assert third.content.endswith("Shall I send it again?")

    asyncio.run(run())
