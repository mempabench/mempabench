"""A journal that keeps the span and a hash, and loses the pool.

This is the ruled-out shortcut -- "persist the pool itself", not a
pointer to a slice that has already been stripped -- and it is here to be
refuted. `test_resume_pool_stub_proof.py` runs the step-9 suite against it and
asserts every test fails on its own assertions, which is what makes "these
tests assert behaviour, not the module's presence" checkable rather than
claimed.

It is the landed `TurnJournal` with one change: `append_pending` drops the
rendered pool on the way to disk and keeps only the span, the timestamp and the
payload digest the record already carries. It also stands in for the other
shortcut: `replay_pending_turns` rewrites the turn but logs nothing, so the
verification check ("the resume log lists rewritten turn ids") has
something concrete to fail against.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from memory.events import MemoryEvent
from memory.turn_journal import JournalRecord  # noqa: F401 (re-export)
from memory.turn_journal import journal_path  # noqa: F401 (re-export)
from memory.turn_journal import PENDING, WRITTEN  # noqa: F401 (re-export)
from memory.turn_journal import TurnJournal as _LandedTurnJournal


class TurnJournal(_LandedTurnJournal):
    """The landed journal, persisting a span and a hash instead of the pool."""

    def append_pending(self, record: JournalRecord) -> JournalRecord:
        return super().append_pending(replace(record, messages=[], items=[]))


async def replay_pending_turns(
    *,
    journal: TurnJournal,
    adapter: Any,
    scope: Any,
    session: Any,
    require_committed: bool = True,
) -> list[str]:
    rewritten: list[str] = []
    for record in journal.resume_pending().records:
        await adapter.setup_scope(scope)
        receipt = await adapter.record_event(MemoryEvent(
            event_id=f"{scope.run_id}:{session.key}:{record.turn_id}",
            run_id=scope.run_id,
            persona_id=scope.persona_id,
            agent_id=scope.agent_id or "nanobot",
            session_id=session.key,
            turn_id=record.turn_id,
            timestamp=datetime.fromisoformat(record.timestamp),
            messages=record.pool.messages,
            items=record.pool.items,
        ))
        journal.mark_written(record.turn_id, receipt)
        session.adapter_committed_end_index = max(
            session.adapter_committed_end_index, record.span[1]
        )
        rewritten.append(record.turn_id)
    return rewritten
