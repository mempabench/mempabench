"""An IX selection is inherited for the rest of one benchmark session, and only that one."""

from __future__ import annotations

import asyncio
from pathlib import Path

from memory.adapter import PromptHistoryEvictionPolicy
from memory.results import RetrieveResult, WriteReceipt, WriteStatus
from pa.agent.loop import AgentLoop
from pa.bus.queue import MessageBus
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest

SETTING = "Terse"


class _ToolRecordingProvider(LLMProvider):
    """Answers each turn with an IX call when the tool is offered, and records
    which tools were offered on every call."""

    def __init__(self) -> None:
        super().__init__()
        self.offered: list[list[str]] = []

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None) -> LLMResponse:
        names = [tool["function"]["name"] for tool in tools or []]
        # The IX call of this turn is in. The wrapper's own internal messages
        # are skipped first: it appends them *after* the tool results (the
        # in-turn restatement of the setting just chosen, the strike prompt),
        # so the last message being a `user` one no longer means nothing has
        # run yet.
        trailing = [
            message for message in messages if not message.get("_mempa_internal")
        ]
        if trailing and trailing[-1].get("role") == "tool":
            return LLMResponse(content="Done.")
        self.offered.append(names)
        if "IX_verbosity" in names:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id=f"ix_{len(self.offered)}", name="IX_verbosity",
                arguments={
                    "setting": SETTING,
                    "evidence": "",
                    "application": "one-line answers",
                },
            )])
        return LLMResponse(content="Done.")


class _NullAdapter:
    backend_name = "null"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.after_adapter_commit

    async def setup_scope(self, scope) -> None:
        return None

    async def retrieve(self, request):
        return RetrieveResult(raw=[])

    def format_context(self, result) -> str:
        return ""

    async def record_event(self, event) -> WriteReceipt:
        return WriteReceipt(status=WriteStatus.committed)


def test_ix_selection_is_inherited_only_within_one_script_session(tmp_path: Path) -> None:
    async def run() -> None:
        provider = _ToolRecordingProvider()
        loop = AgentLoop(
            bus=MessageBus(), pa_llm_provider=provider, workspace=tmp_path,
            pa_model="test-model", memory_adapter=_NullAdapter(),
            require_committed_adapter_writes=True,
        )
        loop.sessions.create("benchmark:main")

        async def step(script_session_id: str, text: str) -> None:
            await loop.process_direct_message(
                text, session_key="benchmark:main",
                metadata={
                    "run_id": "run_001", "persona_id": "user_a",
                    "script_session_id": script_session_id,
                    "active_ix": ["verbosity"],
                    "ix_context": {"session_active_ix": ["verbosity"]},
                },
            )

        await step("acc_001", "first turn")
        await step("acc_001", "second turn, same session")
        await step("acc_002", "next benchmark session")

        offered_ix = [[n for n in names if n.startswith("IX_")] for names in provider.offered]
        # acc_001 turn 1: offered + selected. acc_001 turn 2: the selection is
        # inherited, so no IX tool. acc_002: same session_key, new benchmark
        # session -> offered again.
        assert offered_ix == [["IX_verbosity"], [], ["IX_verbosity"]]
        assert set(loop._ix_selection_cache) == {
            ("benchmark:main", "acc_001"), ("benchmark:main", "acc_002"),
        }

    asyncio.run(run())
