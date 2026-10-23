from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from pa.agent.ix.verification import ENV_VAR, verify_outreach_action
from pa.agent.loop import AgentLoop
from pa.bus.queue import MessageBus
from memory.adapter import PromptHistoryEvictionPolicy
from memory.results import WriteReceipt, WriteStatus
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _ScriptedProvider(LLMProvider):
    def __init__(self, *, action: str, message: str):
        super().__init__()
        self.calls: list[list[dict[str, Any]]] = []
        self.responses = [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="ix_proactive",
                        name="IX_proactive_outreach",
                        arguments={
                            "setting": "Low" if action == "suppress" else "High",
                            "evidence": "Retrieved preference evidence.",
                            "application": f"{action} this contact.",
                            "outreach_action": action,
                            "outreach_message": message,
                        },
                    )
                ],
            ),
            LLMResponse(content="Internal completion text."),
        ]

    def add_decision(self, *, action: str, message: str) -> None:
        self.responses.extend(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id=f"ix_proactive_{action}",
                            name="IX_proactive_outreach",
                            arguments={
                                "setting": "Low" if action == "suppress" else "High",
                                "evidence": "Retrieved preference evidence.",
                                "application": f"{action} this contact.",
                                "outreach_action": action,
                                "outreach_message": message,
                            },
                        )
                    ],
                ),
                LLMResponse(content="Internal completion text."),
            ]
        )

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
        return self.responses.pop(0)


class _RecordingAdapter:
    backend_name = "recording"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.after_adapter_commit

    def __init__(self) -> None:
        self.events = []

    async def setup_scope(self, scope) -> None:
        return None

    async def retrieve(self, request):
        from memory.results import RetrieveResult

        return RetrieveResult(raw=[])

    def format_context(self, result) -> str:
        return ""

    async def record_event(self, event) -> WriteReceipt:
        self.events.append(event)
        return WriteReceipt(status=WriteStatus.committed)


async def _run_event(tmp_path: Path, *, action: str, message: str):
    provider = _ScriptedProvider(action=action, message=message)
    adapter = _RecordingAdapter()
    loop = AgentLoop(
        bus=MessageBus(),
        pa_llm_provider=provider,
        workspace=tmp_path,
        pa_model="test-model",
        memory_adapter=adapter,
        require_committed_adapter_writes=True,
    )
    loop.sessions.create("benchmark:proactive")
    outbound = await loop.process_environment_event(
        trigger_id="pickup_deadline",
        source_type="deadline",
        prior_thread="The comic order is ready for pickup.",
        detail="The store closes in two hours.",
        user_impact="User A may want time to collect the order today.",
        session_key="benchmark:proactive",
        metadata={"run_id": "run_001", "persona_id": "user_a"},
        timestamp=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
    )
    return loop, provider, adapter, outbound


def test_environment_event_can_be_suppressed_without_user_message(tmp_path: Path) -> None:
    async def run() -> None:
        # The PA always writes the reminder; Low decides whether it is released.
        loop, provider, adapter, outbound = await _run_event(
            tmp_path,
            action="suppress",
            message="The comic store closes at 6 PM today.",
        )

        assert outbound.content == ""
        assert outbound.metadata["outreach_decision"] == "suppress"
        assert outbound.metadata["outreach_delivered"] is False
        assert provider.calls[0][-1]["role"] == "system"
        assert "no user request is currently active" in provider.calls[0][-1]["content"]
        persisted = loop.sessions.get("benchmark:proactive").messages
        assert not any(
            message.get("role") == "user"
            and "Environment event" in str(message.get("content"))
            for message in persisted
        )
        assert persisted == []
        # Journal-only: the suppressed turn writes one pool and no
        # session message, so nothing synthetic re-enters the PA's prompt --
        # but the decision to stay silent is recorded, because
        # `proactive_outreach` is scored on exactly that decision.
        assert len(adapter.events) == 1
        kinds = [item.kind for item in adapter.events[0].items]
        assert kinds[0] == "environment_event" and kinds[-1] == "no_reply"

        # T19: the candidate reminder exists in exactly one place -- the audit
        # plane. The *text* of the contact the PA did not send is still nowhere
        # in history or memory.
        assert "The comic store closes at 6 PM today." not in json.dumps(
            adapter.events[0].messages, default=str
        )
        ix_render = outbound.metadata["ix_render"]
        assert ix_render["proactive_decision"] == {
            "action": "suppress",
            "setting": "Low",
            "message": "The comic store closes at 6 PM today.",
        }
        assert ix_render["held"]["outreach_message"] == (
            "The comic store closes at 6 PM today."
        )

    asyncio.run(run())


def test_environment_event_send_uses_decision_message_as_outbound(tmp_path: Path) -> None:
    async def run() -> None:
        loop, _, adapter, outbound = await _run_event(
            tmp_path,
            action="send",
            message="The comic store closes at 6 PM today.",
        )

        assert outbound.content == "The comic store closes at 6 PM today."
        assert outbound.metadata["outreach_decision"] == "send"
        assert outbound.metadata["outreach_delivered"] is True
        assert loop.sessions.get("benchmark:proactive").messages == [
            {
                "role": "assistant",
                "content": "The comic store closes at 6 PM today.",
                "timestamp": "2026-10-01T09:00:00+00:00",
            }
        ]
        # The session stores only what was delivered; the pool carries the
        # whole turn -- the trigger the PA perceived, its own selection and
        # that selection's result, then the delivered contact.
        assert [item.kind for item in adapter.events[0].items] == [
            "environment_event", "ix_selection", "ix_result", "assistant",
        ]
        assert adapter.events[0].messages[-1] == {
            "role": "assistant", "content": "The comic store closes at 6 PM today.",
        }
        assert "Use IX_proactive_outreach" not in json.dumps(
            adapter.events[0].messages, default=str
        )

    asyncio.run(run())


def test_each_environment_trigger_requires_a_fresh_outreach_decision(tmp_path: Path) -> None:
    async def run() -> None:
        loop, provider, adapter, first = await _run_event(
            tmp_path,
            action="send",
            message="The comic is ready for pickup.",
        )
        provider.add_decision(
            action="suppress", message="The pickup window is still open.",
        )

        second = await loop.process_environment_event(
            trigger_id="pickup_reminder",
            source_type="scheduled_time",
            prior_thread="The comic order is ready for pickup.",
            detail="The scheduled pickup reminder fired.",
            user_impact="User A may already know the pickup status.",
            session_key="benchmark:proactive",
            metadata={"run_id": "run_001", "persona_id": "user_a"},
            timestamp=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        )

        assert first.metadata["outreach_decision"] == "send"
        assert second.metadata["outreach_decision"] == "suppress"
        assert len(provider.calls) == 4
        # One pool per trigger, whichever way each one was decided.
        assert len(adapter.events) == 2
        assert [item.kind for item in adapter.events[1].items][-1] == "no_reply"
        assert loop.sessions.get("benchmark:proactive").messages[-1]["content"] == (
            "The comic is ready for pickup."
        )

    asyncio.run(run())


# --- MEMPA_VERIFY_OUTREACH ---------------------------------------------------
# A verification aid: acc_007 has run nine rounds and
# the PA chose `send` every time, so `opening_after_suppress` has never been
# exercised. The knob forces the branch; these tests are what keep it from
# being usable by accident.


def _verification_events(outbound) -> list[dict[str, Any]]:
    return [
        event
        for event in outbound.metadata["tool_events"]
        if event.get("name") == "verification_override"
    ]


def test_verify_outreach_overrides_the_send_the_model_chose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PA picks `send`; the wrapper delivers a suppressed turn anyway.

    Everything downstream of the render has to follow, because that is the
    whole point: the outbound is empty and `outreach_delivered` is False, which
    is what routes the simulator into `opening_after_suppress`.
    """
    monkeypatch.setenv(ENV_VAR, "suppress")

    async def run() -> None:
        loop, _, adapter, outbound = await _run_event(
            tmp_path,
            action="send",
            message="The comic store closes at 6 PM today.",
        )

        assert outbound.content == ""
        assert outbound.metadata["outreach_decision"] == "suppress"
        assert outbound.metadata["outreach_delivered"] is False
        assert outbound.metadata["ix_render"]["proactive_decision"] == {
            "action": "suppress",
            "setting": "Low",
            "message": "The comic store closes at 6 PM today.",
        }
        # A suppressed turn persists no session message and remembers the
        # silence, exactly as a model-chosen suppress does.
        assert loop.sessions.get("benchmark:proactive").messages == []
        assert len(adapter.events) == 1
        assert [item.kind for item in adapter.events[0].items][-1] == "no_reply"
        assert "The comic store closes at 6 PM today." not in json.dumps(
            adapter.events[0].messages, default=str
        )

    asyncio.run(run())


def test_verify_outreach_records_the_model_choice_beside_the_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The IX evidence keeps the selection the model actually made.

    The override is a second record, never a rewrite of the first: a read of
    this step can still say what the PA would have done unforced.
    """
    monkeypatch.setenv(ENV_VAR, "suppress")

    async def run() -> None:
        _, _, _, outbound = await _run_event(
            tmp_path,
            action="send",
            message="The comic store closes at 6 PM today.",
        )

        selections = [
            event
            for event in outbound.metadata["tool_events"]
            if event.get("ix") and event.get("attribute") == "proactive_outreach"
        ]
        assert [
            (event["setting"], event["outreach_action"]) for event in selections
        ] == [("High", "send")]

        assert _verification_events(outbound) == [
            {
                "ix_event": True,
                "name": "verification_override",
                "attribute": "proactive_outreach",
                "model_outreach_action": "send",
                "model_setting": "High",
                "outreach_action": "suppress",
                "setting": "Low",
                "source": ENV_VAR,
            }
        ]

    asyncio.run(run())


def test_verify_outreach_can_force_the_send_branch_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, "send")

    async def run() -> None:
        _, _, _, outbound = await _run_event(
            tmp_path,
            action="suppress",
            message="The comic store closes at 6 PM today.",
        )

        assert outbound.content == "The comic store closes at 6 PM today."
        assert outbound.metadata["outreach_decision"] == "send"
        assert outbound.metadata["outreach_delivered"] is True
        assert _verification_events(outbound)[0]["model_outreach_action"] == (
            "suppress"
        )

    asyncio.run(run())


@pytest.mark.parametrize("value", ["", "Suppress", "low", "1", "send,suppress"])
def test_verify_outreach_refuses_any_other_value(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed. A knob that invalidates its run gets no default reading."""
    monkeypatch.setenv(ENV_VAR, value)

    with pytest.raises(RuntimeError, match="must be 'send' or 'suppress'"):
        verify_outreach_action()


@pytest.mark.parametrize("action", ["send", "suppress"])
def test_unset_verify_outreach_leaves_the_turn_exactly_as_it_was(
    action: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-identical to the un-knobbed wrapper when the variable is unset.

    The expected payload is the one the two decision tests above pin, so this
    fails the moment the override changes an ordinary turn — and the dump is
    searched for the knob's own vocabulary, which must not appear anywhere in
    what the turn produced.
    """
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert verify_outreach_action() is None

    async def run() -> None:
        _, _, _, outbound = await _run_event(
            tmp_path,
            action=action,
            message="The comic store closes at 6 PM today.",
        )

        message = "The comic store closes at 6 PM today."
        assert {
            "content": outbound.content,
            "outreach_decision": outbound.metadata["outreach_decision"],
            "outreach_delivered": outbound.metadata["outreach_delivered"],
            "proactive_decision": (
                outbound.metadata["ix_render"]["proactive_decision"]
            ),
        } == {
            "content": message if action == "send" else "",
            "outreach_decision": action,
            "outreach_delivered": action == "send",
            "proactive_decision": {
                "action": action,
                "setting": "High" if action == "send" else "Low",
                "message": message,
            },
        }
        dumped = json.dumps(outbound.metadata, default=str)
        assert "verification_override" not in dumped
        assert ENV_VAR not in dumped

    asyncio.run(run())
