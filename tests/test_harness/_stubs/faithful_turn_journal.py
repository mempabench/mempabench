"""The landed journal, plus the smallest resume entry point still owed.

It exists to prove one thing: that `test_resume_pool.py`'s assertions are
satisfiable -- that none of them is an equality the design rejects, and that a
correct implementation can turn all three green.

`TurnJournal` and `JournalRecord` here are **the real `memory.turn_journal`**,
re-exported, not a reimplementation: the module has landed, so the faithful
journal is simply the journal. What is stubbed is the part that has *not*
landed, `replay_pending_turns` -- the at-least-once resume, reading
`resume_pending()`, re-writing each pending pool, advancing the cursor and
logging every turn id it rewrote. It is a test fixture, not a proposal: whoever
lands the real entry point picks its home and its name, and the behaviour
asserted by the suite is what it has to satisfy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from memory.events import MemoryEvent
from memory.results import WriteStatus
from memory.turn_journal import (  # noqa: F401 (re-export)
    JournalRecord,
    TurnJournal,
    journal_path,
)
from memory.turn_journal import PENDING, WRITTEN  # noqa: F401 (re-export)

_COMMITTED = {WriteStatus.committed, WriteStatus.accepted_no_change}


async def replay_pending_turns(
    *,
    journal: TurnJournal,
    adapter: Any,
    scope: Any,
    session: Any,
    require_committed: bool = True,
) -> list[str]:
    """Rewrite every pending turn and return the ids, in order.

    At-least-once: a turn whose adapter write was acknowledged but whose
    state never reached disk is written a second time, and the duplicate is
    left for the backend to absorb. Nothing here dedupes, and the id of every
    rewritten turn is logged.
    """
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
        if require_committed and receipt.status not in _COMMITTED:
            raise RuntimeError(
                f"resume rewrite refused: turn_id={record.turn_id!r} "
                f"status={receipt.status.value}"
            )
        journal.mark_written(record.turn_id, receipt)
        session.adapter_committed_end_index = max(
            session.adapter_committed_end_index, record.span[1]
        )
        logger.info("resume rewrote turn {}", record.turn_id)
        rewritten.append(record.turn_id)
    return rewritten
