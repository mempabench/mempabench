"""The slash commands, and what `/new` owes the turn journal.

`/new` was the odd one out among the three consolidation entry points: it
archived the raw `session.messages` tail with no projection at all, and it
archived it *after* clearing the session, so a failing consolidation lost the
conversation it had just refused to summarise. After step 7 it reads the same
journal the other two read, archives first, and clears only on success.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from memory.event_writer import records_owed_to_files, turn_journal_for
from pa.agent.loop import AgentLoop
from pa.bus.queue import MessageBus
from pa.bus.events import InboundMessage
from pa.command.builtin import cmd_new
from pa.command.router import CommandContext
from pa.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)

SESSION_KEY = "cli:local"


class _ScriptProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.generation = GenerationSettings(max_tokens=256)
        self.responses = list(responses)

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)


class _SaveMemoryProvider(_ScriptProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.archived_prompts: list[str] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self.archived_prompts.append(
            "\n".join(str(message.get("content") or "") for message in messages)
        )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="save_1", name="save_memory",
                arguments={"history_entry": "H", "memory_update": "M"},
            )],
            finish_reason="tool_calls",
        )


def _context(loop: AgentLoop) -> CommandContext:
    session = loop.sessions.get(SESSION_KEY)
    message = InboundMessage(
        channel="cli", chat_id="local", sender_id="user", content="/new",
    )
    return CommandContext(
        msg=message, session=session, key=SESSION_KEY, raw="/new", loop=loop,
    )


async def _one_turn(tmp_path, memory_provider: LLMProvider) -> AgentLoop:
    provider = _ScriptProvider([
        LLMResponse(content="Noted.", reasoning_content="Nothing to look up."),
    ])
    loop = AgentLoop(
        bus=MessageBus(), pa_llm_provider=provider, workspace=tmp_path,
        pa_model="test-model",
        consolidation_llm_provider=memory_provider,
        consolidation_model="fake-memory-model",
    )
    loop.sessions.create(SESSION_KEY)
    await loop.process_direct_message(
        "Remember the Braithwaite entry.",
        session_key=SESSION_KEY,
        metadata={"run_id": "run_001", "persona_id": "user_a"},
    )
    return loop


@pytest.mark.asyncio
async def test_new_keeps_session_and_journal_when_archive_fails(tmp_path) -> None:
    """A refused consolidation must leave a retriable command.

    The old order cleared the session first, so an archive that raised took the
    conversation with it. Archiving first means the failure costs nothing.
    """
    loop = await _one_turn(tmp_path, _SaveMemoryProvider())
    before = [dict(message) for message in loop.sessions.get(SESSION_KEY).messages]
    assert before

    async def _refuse(messages: list[dict[str, object]]) -> bool:
        raise RuntimeError("INVALID_CONSOLIDATION: contract not met")

    loop.memory_consolidator.archive_messages_to_files = _refuse

    with pytest.raises(RuntimeError, match="INVALID_CONSOLIDATION"):
        await cmd_new(_context(loop))

    # The session is exactly as it was, cache included.
    loop.sessions.invalidate(SESSION_KEY)
    assert [
        dict(message) for message in loop.sessions.get(SESSION_KEY).messages
    ] == before
    # And so is the journal: the record is still owed to the memory files.
    journal = turn_journal_for(loop.sessions, SESSION_KEY)
    assert len(records_owed_to_files(journal.records())) == 1


@pytest.mark.asyncio
async def test_new_archives_the_journal_before_it_clears(tmp_path) -> None:
    """The pool, not the raw tail: `/new` reads what the other two read."""
    memory_provider = _SaveMemoryProvider()
    loop = await _one_turn(tmp_path, memory_provider)

    archived: list[list[dict[str, Any]]] = []
    unwrapped = loop.memory_consolidator.archive_messages_to_files

    async def _capture(messages: list[dict[str, object]]) -> bool:
        archived.append([dict(message) for message in messages])
        return await unwrapped(messages)

    loop.memory_consolidator.archive_messages_to_files = _capture

    await cmd_new(_context(loop))

    assert len(archived) == 1
    blob = "\n".join(
        str(record["content"]) for record in archived[0]
    )
    assert "Remember the Braithwaite entry." in blob
    assert "[reasoning source=reasoning_content]" in blob
    assert "Noted." in blob

    loop.sessions.invalidate(SESSION_KEY)
    cleared = loop.sessions.get(SESSION_KEY)
    assert cleared.messages == []
    assert cleared.file_consolidated_end_index == 0

    # A new journal boundary: the next turn's span cannot collide with the
    # cleared session's, and nothing owed is left behind.
    journal = turn_journal_for(loop.sessions, SESSION_KEY)
    assert records_owed_to_files(journal.records()) == []
