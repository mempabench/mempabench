"""The pool the loop hands the memory adapter.

Steps 1–3 rendered whatever `session.messages` had kept. These tests ask the
harder question: does the adapter receive the PA's *whole* first-person
experience of the turn — the prose of an intermediate iteration, the reasoning
of the final one, the PA's own IX choice with the outcome the registry gave it,
and the environment trigger it stayed silent about.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import memory.event_writer as event_writer
from memory.adapter import PromptHistoryEvictionPolicy
from memory.results import RetrieveResult, WriteReceipt, WriteStatus
from memory.turn_experience import parse_blocks
from pa.agent.loop import AgentLoop
from pa.agent.tools.base import Tool
from pa.bus.queue import MessageBus
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest

SESSION_KEY = "benchmark:main"


class _Echo(Tool):
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

    async def execute(self, value: str = "") -> dict[str, str]:
        return {"value": value}


class _PaperSearch(Tool):
    """A world read that returns one long string, as the MCP state tools do."""

    def __init__(self, body: str) -> None:
        self._body = body

    @property
    def name(self) -> str:
        return "task_paper_search"

    @property
    def description(self) -> str:
        return "Search papers."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    async def execute(self, query: str = "") -> str:
        return self._body


class _RecordingAdapter:
    backend_name = "recording"
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


def _loop(tmp_path: Path, provider: LLMProvider) -> tuple[AgentLoop, _RecordingAdapter]:
    adapter = _RecordingAdapter()
    loop = AgentLoop(
        bus=MessageBus(), pa_llm_provider=provider, workspace=tmp_path,
        pa_model="test-model", memory_adapter=adapter,
        require_committed_adapter_writes=True,
    )
    loop.tools.register(_Echo())
    loop.sessions.create(SESSION_KEY)
    return loop, adapter


async def _turn(loop: AgentLoop, text: str, **ix_context: Any):
    active_ix = list(ix_context.pop("active_ix", []))
    context: dict[str, Any] = {
        "session_active_ix": active_ix,
        "tool_action_types": {"task_echo": "read"},
    }
    context.update(ix_context)
    return await loop.process_direct_message(
        text,
        session_key=SESSION_KEY,
        metadata={
            "run_id": "run_001",
            "persona_id": "user_a",
            "script_session_id": "acc_001",
            "active_ix": active_ix,
            "ix_context": context,
        },
    )


def _items(event) -> list[tuple[str, str]]:
    """(kind, body) for every block in the event, dialogue included."""
    out: list[tuple[str, str]] = []
    for message in event.messages:
        blocks = _blocks_of(message)
        if blocks:
            out.extend((block.kind, block.body) for block in blocks)
        else:
            out.append((message["role"], message["content"]))
    return out


def _blocks_of(message: dict[str, str]):
    if not message["content"].startswith("["):
        return []
    try:
        return parse_blocks(message["content"])
    except ValueError:
        return []


def _selections(event):
    return [
        block
        for message in event.messages
        for block in _blocks_of(message)
        if block.kind == "ix_selection"
    ]


# ---------------------------------------------------------------------------
# step 4 · the pool is the raw loop slice, not what the next prompt kept
# ---------------------------------------------------------------------------

def test_turn_pool_contains_intermediate_prose_and_reasoning(tmp_path: Path) -> None:
    """`build_runtime_history` blanks both; the pool is not that projection.

    Projection 2 exists so held prose never returns to the PA's own prompt, and
    it pays for that by blanking the prose of an intermediate iteration
    (`pa/agent/ix/collector.py:1404`) and rebuilding the final assistant message
    without its reasoning (`:1434`). The pool is projection 3 and reads the raw
    provider slice instead.
    """
    async def run() -> None:
        provider = _ScriptProvider([
            LLMResponse(
                content="Let me check the mailbox first.",
                reasoning_content="The order id is MB-55021.",
                tool_calls=[
                    ToolCallRequest(id="echo_1", name="task_echo",
                                    arguments={"value": "MB-55021"}),
                ],
            ),
            LLMResponse(
                content="It is out for delivery.",
                reasoning_content="The search returned one match.",
            ),
        ])
        loop, adapter = _loop(tmp_path, provider)

        outbound = await _turn(loop, "Any news on my order?")

        # What projection 2 keeps: the declaration and the result only.
        persisted = loop.sessions.get(SESSION_KEY).messages
        assert [message.get("content") for message in persisted] == [
            "Any news on my order?", "", {"value": "MB-55021"},
            "It is out for delivery.",
        ]

        assert len(adapter.events) == 1
        assert _items(adapter.events[0]) == [
            ("user", "Any news on my order?"),
            ("reasoning", "The order id is MB-55021."),
            ("prose", "Let me check the mailbox first."),
            ("tool_call", '{"value":"MB-55021"}'),
            ("tool_result", '{"value": "MB-55021"}'),
            ("reasoning", "The search returned one match."),
            ("assistant", "It is out for delivery."),
        ]
        # Q3: the delivered text appears exactly once, as `public_turn`.
        blob = json.dumps(adapter.events[0].messages, default=str)
        assert blob.count("It is out for delivery.") == 1
        assert outbound.content == "It is out for delivery."

    asyncio.run(run())


def test_the_pool_keeps_the_runtime_context_prefix_off_the_person_text(
    tmp_path: Path,
) -> None:
    """Item #2: the scripted clock and channel framing is scaffolding.

    The raw slice carries it (`ContextBuilder` merges it into the user message
    so providers do not see two user turns in a row), so the pool has to take
    it off exactly as `_save_turn` does.
    """
    async def run() -> None:
        provider = _ScriptProvider([LLMResponse(content="Nothing yet.")])
        loop, adapter = _loop(tmp_path, provider)

        await _turn(loop, "Any news on my order?")

        assert provider.requests[0][-1]["content"] != "Any news on my order?"
        assert adapter.events[0].messages[0] == {
            "role": "user", "content": "Any news on my order?",
        }

    asyncio.run(run())


def test_the_acc_slice_reaches_the_pool_already_cut(tmp_path: Path) -> None:
    """The pool and the session file hold one tool result, cut once.

    Found live in run `live2_pool_acc_001`: `_save_turn` cut the persisted
    history at `TOOL_RESULT_MAX_CHARS` while the pool was built from the raw
    provider slice, so a 25 348-character `mcp_state_scholarly_paper_search`
    result went into Qdrant whole and — never carrying the marker — stamped
    `truncated=false`. That contradicts `memory/event_writer.py:20-34` and
    owner ruling Q2: the pool carries the persisted form. Both writers now go
    through `AgentLoop._cut_tool_result`.
    """
    async def run() -> None:
        oversized = json.dumps(
            {"papers": ["x" * (event_writer.TOOL_RESULT_MAX_CHARS + 9_348)]}
        )
        provider = _ScriptProvider([
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="paper_1", name="task_paper_search",
                                    arguments={"query": "einselection"}),
                ],
            ),
            LLMResponse(content="Three papers cover it."),
        ])
        loop, adapter = _loop(tmp_path, provider)
        loop.tools.register(_PaperSearch(oversized))

        await _turn(
            loop, "What does the einselection work say?",
            tool_action_types={"task_echo": "read", "task_paper_search": "read"},
        )

        persisted = [
            message for message in loop.sessions.get(SESSION_KEY).messages
            if message.get("role") == "tool"
        ]
        assert len(persisted) == 1
        result = next(
            item for item in adapter.events[0].items if item.kind == "tool_result"
        )
        # Byte for byte the same string in both places, and the stamp is the
        # truth about it.
        assert result.body == persisted[0]["content"]
        assert len(result.body) == (
            event_writer.TOOL_RESULT_MAX_CHARS + len("\n... (truncated)")
        )
        assert result.attrs["truncated"] == "true"

        # The journal record is what a resume replays, so it carries the
        # cut form too.
        record = loop.turn_journal(SESSION_KEY).records()[-1]
        journalled = next(
            item for item in record.items if item.kind == "tool_result"
        )
        assert journalled.body == persisted[0]["content"]
        assert journalled.attrs["truncated"] == "true"

    asyncio.run(run())


ELICITATION = "information_elicitation"


def _elicitation(call_id: str, setting: str, *, reasoned: bool = True) -> ToolCallRequest:
    """A selection, either fully reasoned or the bare setting a pin buys back."""
    arguments = {"setting": setting}
    if reasoned:
        arguments["evidence"] = f"the user asked for {setting}"
        arguments["application"] = f"apply {setting} to this turn"
    return ToolCallRequest(id=call_id, name=f"IX_{ELICITATION}", arguments=arguments)


def _render(call_id: str) -> ToolCallRequest:
    """The `Infer`-shaped payload: no question released, one assumption named."""
    return ToolCallRequest(
        id=call_id, name=f"IX_render_{ELICITATION}",
        arguments={
            "elicitation_id": "el1", "questions": [],
            "assumptions": [{"about": "the order", "assumed": "the most recent one"}],
        },
    )


def test_ix_selection_outcomes_cover_executed_rejected_and_duplicate(
    tmp_path: Path,
) -> None:
    """The block is the executed request, not the model's serialized copy.

    `outcome=` comes from the call's own `_run_tool` event status, and the body
    from the arguments the registry received. The assistant message's
    serialized `tool_calls` can supply neither. A valid selection in a mixed
    response now executes immediately while the task is deferred; later
    re-selections do not replace that first decision.
    """
    async def run() -> None:
        provider = _ScriptProvider([
            # Iteration 1: a setting the enum does not allow — the registry
            # refuses it, so the attribute is owed again and the turn retries.
            LLMResponse(content=None, tool_calls=[_elicitation("ix_bad", "Nonsense")]),
            # Iteration 2: the reasoned selection lands while its task waits for
            # the render payload.
            LLMResponse(content=None, tool_calls=[
                _elicitation("ix_voided", "Infer"),
                ToolCallRequest(id="echo_1", name="task_echo",
                                arguments={"value": "MB-55021"}),
            ]),
            # Iteration 3: both re-selections are duplicates; render is the only
            # new protocol progress.
            LLMResponse(content=None, tool_calls=[
                _elicitation("ix_ok", "Infer", reasoned=False),
                _elicitation("ix_dup", "Iterative"),
                _render("ix_payload"),
            ]),
            LLMResponse(content="Here is what I know."),
            LLMResponse(content=None, tool_calls=[_elicitation("ix_again", "Iterative")]),
            LLMResponse(content="Still the same answer."),
        ])
        loop, adapter = _loop(tmp_path, provider)

        await _turn(loop, "Any news on my order?", active_ix=[ELICITATION])
        # Turn 2 re-selects an attribute the session already banked and this
        # turn no longer offers: `_reselected_attribute` drops it unexecuted,
        # and it emits no tool event at all.
        await _turn(loop, "And the other order?", session_ix_selected=[ELICITATION])

        first = [
            (block.attrs["attribute"], block.attrs["setting"], block.attrs["outcome"])
            for block in _selections(adapter.events[0])
        ]
        assert first == [
            (ELICITATION, "Nonsense", "rejected"),
            (ELICITATION, "Infer", "executed"),
        ]
        second = [
            (block.attrs["attribute"], block.attrs["outcome"])
            for block in _selections(adapter.events[1])
        ]
        assert second == [(ELICITATION, "ignored_duplicate")]

        # The executed body is the original reasoned mixed-response selection.
        executed = next(
            block for block in _selections(adapter.events[0])
            if block.attrs["outcome"] == "executed"
        )
        assert json.loads(executed.body) == {
            "setting": "Infer",
            "evidence": "the user asked for Infer",
            "application": "apply Infer to this turn",
        }

    asyncio.run(run())


# ---------------------------------------------------------------------------
# step 5 · the environment branch is a write path, not a rendering
# ---------------------------------------------------------------------------

def _outreach_provider(action: str, message: str) -> _ScriptProvider:
    return _ScriptProvider([
        LLMResponse(
            content=None,
            reasoning_content="The store closes at six; the user hates nudges.",
            tool_calls=[ToolCallRequest(
                id="ix_proactive", name="IX_proactive_outreach",
                arguments={
                    "setting": "Low" if action == "suppress" else "High",
                    "evidence": "Retrieved preference evidence.",
                    "application": f"{action} this contact.",
                    "outreach_action": action,
                    "outreach_message": message,
                },
            )],
        ),
        LLMResponse(content="Internal completion text."),
    ])


async def _environment_event(loop: AgentLoop):
    return await loop.process_environment_event(
        trigger_id="pickup_deadline",
        source_type="deadline",
        prior_thread="The comic order is ready for pickup.",
        detail="The store closes in two hours.",
        user_impact="User A may want time to collect the order today.",
        session_key=SESSION_KEY,
        metadata={"run_id": "run_001", "persona_id": "user_a"},
        timestamp=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
    )


def test_suppressed_outreach_writes_one_pool_with_environment_event_and_no_reply(
    tmp_path: Path, monkeypatch,
) -> None:
    """Silence is an interaction choice, and `proactive_outreach` scores it.

    Today the whole commit window sits behind `if decision == "send":`, so a
    suppressed turn writes nothing at all and the one attribute scored on
    staying silent has no record of having done so.
    """
    async def run() -> None:
        journal: list[dict[str, Any]] = []
        monkeypatch.setattr(
            event_writer, "record_turn_journal",
            lambda **kwargs: journal.append(kwargs),
        )
        loop, adapter = _loop(tmp_path, _outreach_provider(
            "suppress", "The comic store closes at 6 PM today.",
        ))

        outbound = await _environment_event(loop)

        assert outbound.content == ""
        # Q2 = A: journal-only. Nothing synthetic re-enters the PA's prompt.
        assert loop.sessions.get(SESSION_KEY).messages == []

        assert len(adapter.events) == 1
        kinds = [kind for kind, _ in _items(adapter.events[0])]
        assert kinds[0] == "environment_event"
        assert kinds[-1] == "no_reply"
        assert kinds.count("environment_event") == 1
        assert kinds.count("no_reply") == 1
        assert "user" not in kinds

        # Q4: the fixed IX instruction suffix is contract enforcement, and the
        # split is exact because the clean body never carried it.
        blob = json.dumps(adapter.events[0].messages, default=str)
        assert "Use IX_proactive_outreach" not in blob
        assert "The store closes in two hours." in blob

        # One record, two states, and a `[n, n)` span — the resume authority.
        assert [entry["state"] for entry in journal] == ["pending", "written"]
        assert {entry["turn_id"] for entry in journal} == {journal[0]["turn_id"]}
        assert [entry["span"] for entry in journal] == [(0, 0), (0, 0)]

    asyncio.run(run())


def test_a_sent_outreach_writes_one_pool_and_advances_the_cursor_once(
    tmp_path: Path, monkeypatch,
) -> None:
    async def run() -> None:
        journal: list[dict[str, Any]] = []
        monkeypatch.setattr(
            event_writer, "record_turn_journal",
            lambda **kwargs: journal.append(kwargs),
        )
        loop, adapter = _loop(tmp_path, _outreach_provider(
            "send", "The comic store closes at 6 PM today.",
        ))

        outbound = await _environment_event(loop)

        assert outbound.content == "The comic store closes at 6 PM today."
        session = loop.sessions.get(SESSION_KEY)
        assert [m["content"] for m in session.messages] == [
            "The comic store closes at 6 PM today.",
        ]
        assert session.adapter_committed_end_index == 1
        assert len(adapter.events) == 1
        kinds = [kind for kind, _ in _items(adapter.events[0])]
        assert kinds[0] == "environment_event"
        assert "no_reply" not in kinds
        assert adapter.events[0].messages[-1] == {
            "role": "assistant", "content": "The comic store closes at 6 PM today.",
        }
        assert [entry["state"] for entry in journal] == ["pending", "written"]
        assert [entry["span"] for entry in journal] == [(0, 1), (0, 1)]

    asyncio.run(run())


def test_environment_branch_pool_carries_the_ix_selection_and_result(
    tmp_path: Path,
) -> None:
    """Both decisions carry the turn's kept loop experience.

    The event plus the reply is not the experience — the PA's own
    `IX_proactive_outreach` call is the half of the (choice, reaction) pair
    preference learning needs, and it is the one the old projection dropped.
    """
    async def run() -> None:
        for action in ("send", "suppress"):
            loop, adapter = _loop(tmp_path / action, _outreach_provider(
                action, "The comic store closes at 6 PM today.",
            ))

            await _environment_event(loop)

            kinds = [kind for kind, _ in _items(adapter.events[0])]
            assert "ix_selection" in kinds, action
            assert "ix_result" in kinds, action
            assert "reasoning" in kinds, action
            selection = _selections(adapter.events[0])[0]
            assert selection.attrs["attribute"] == "proactive_outreach"
            assert selection.attrs["outcome"] == "executed"
            # Q3 again: the internal completion text was never delivered.
            blob = json.dumps(adapter.events[0].messages, default=str)
            assert "Internal completion text." not in blob

    asyncio.run(run())
