"""Append-only per-session journal of pool writes (plan §6.1 (b), (c), (e)).

The pool a turn writes cannot be re-derived from `session.messages`: the
persisted slice drops intermediate prose, held call declarations and the final
iteration's reasoning (`pa/agent/ix/collector.py:1394-1404`, `:1434`). So the
rendered pool is *persisted* instead, one record per pool write, beside the
session file. A resume re-reads the artefact rather than re-deriving one.

Two completion tracks, because two backends commit differently:

* `state` — `pending` on append, `written` once the adapter receipt is
  recorded (`mark_written`). This is the resume authority: a `pending` record
  is replayed under Q5's at-least-once rule, a `written` one is skipped. It is
  never a message index, which is what makes a zero-span turn (a suppressed
  outreach, span `[n,n)`) resumable at all.
* `file_consolidated` — the second completion mark, for the file condition,
  whose `record_event` is a documented no-op returning
  `deferred_to_file_consolidation` and whose real write happens later, at
  consolidation time (§6.1 (e)). `mark_file_consolidated` moves this flag only;
  it does not touch `state`.

One more mark, for the file condition only: `dropped_from_files` names a turn
the memory files are never to have. A skipped step under
`failure_policy.partial_skip_pending_memory: drop` sets it on its own turns
the files had not yet had (`mark_dropped_from_files`), and consolidation then
leaves them out. It is not `file_consolidated`: nothing was written. Journals
of runs before 7aae87cc6 name such turns with `file_unconsolidated_skipped` or
`whole_scene_file_unconsolidated`; both are read as `dropped_from_files`.

The file is append-only: `append_pending` writes one record line, and each mark
is its own line folded onto the record when the journal is read. A crash
between the two therefore loses nothing — the record is durable from the moment
it is appended, before the turn's first `sessions.save`.

Wiring this module is NOT done here. Step 7 wires three consumers, all reading
this same journal from their own cursor and advancing it only after
`require_file_consolidation` returns:

1. mid-session compaction — `pa/agent/memory.py:531-570`, which consumes
   exactly the journal prefix whose turns end at or before its chosen
   `compaction_end_index`;
2. the pinned end-of-run flush — `harness/orchestrator.py:947-965`, which
   consumes every remaining record, zero-span ones included, and must run even
   when `session.messages` has no tail left;
3. `/new` — `pa/command/builtin.py:67-76`, which archives the outstanding
   records first and clears, saves, invalidates and opens the new journal
   boundary only on success.

Steps 4 and 5 wire the producer: `append_pending` inside the turn's commit
window before the first `sessions.save`, `mark_written` after the adapter
acknowledges and before the cursor advances (`pa/agent/loop.py:1596-1613`).

Step 9 reads `resume_pending()`: its `.records` are the pools to re-write and
its `.turn_ids` are what the resume log must list. Two further step-9 duties
live outside this module: the checkpoint must carry the journal file alongside
`session.jsonl` (§6.1 (b), "The journal is saved and restored with the
checkpoint" — `pa/session/manager.py:360-378` copies only the session file
today).

The FE line's realized artefacts (tool results, world ids, the `$call_*`
reference map, §6.1 (d)) ride in the optional `realized` field. Its producer is
`harness/forced_event_executor.py:_prepare_exchange`, which writes the pending
record once the exchange's world calls have run and before the first
`session_manager.save`; `_prepare_exchange` reads the field back on a resume and
skips `registry.execute` for an exchange that already has one. Q5 authorises
replaying a memory write, never a world action.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from loguru import logger

from memory.adapter import PromptHistoryEvictionPolicy
from memory.event_writer import (
    TurnPool,
    record_adapter_write_commit,
    write_visible_memory_event,
)
from memory.results import WriteReceipt
from memory.turn_experience import ExperienceItem
from pa.utils.helpers import safe_filename

PENDING = "pending"
WRITTEN = "written"


class UnknownTurnError(KeyError):
    """Raised when a mark names a turn the journal has no record of."""


class JournalLoadError(RuntimeError):
    """Raised when a journal line cannot be decoded."""


def journal_path(sessions_dir: Path, session_key: str) -> Path:
    """The journal beside the session file `SessionManager` writes.

    `pa/session/manager.py:181-184` names the session `<safe_key>.jsonl`; the
    journal is `<safe_key>.journal.jsonl` in the same directory, so a copy of
    the sessions directory carries it (the checkpoint branch case).
    """
    safe_key = safe_filename(session_key.replace(":", "_"))
    return Path(sessions_dir) / f"{safe_key}.journal.jsonl"


@dataclass(frozen=True)
class JournalCursor:
    """How far a consumer has read, in records — its own track.

    Independent of `prompt_history_start_index` and of
    `adapter_committed_end_index`, which answer different questions.
    """

    index: int = 0

    def to_json(self) -> dict[str, int]:
        return {"index": self.index}

    @classmethod
    def from_json(cls, data: dict[str, int]) -> JournalCursor:
        return cls(index=int(data["index"]))


@dataclass(frozen=True)
class JournalRecord:
    """One pool write."""

    turn_id: str
    span: tuple[int, int]
    timestamp: str
    messages: list[dict[str, str]]
    items: list[ExperienceItem]
    state: str = PENDING
    file_consolidated: bool = False
    dropped_from_files: bool = False
    receipt: dict[str, Any] | None = None
    realized: dict[str, Any] | None = None
    sequence: int = -1

    @property
    def is_zero_span(self) -> bool:
        return self.span[0] == self.span[1]

    @classmethod
    def for_pool(
        cls,
        *,
        turn_id: str,
        span: tuple[int, int],
        pool: TurnPool,
        timestamp: str,
        realized: dict[str, Any] | None = None,
    ) -> JournalRecord:
        return cls(
            turn_id=turn_id,
            span=(int(span[0]), int(span[1])),
            timestamp=timestamp,
            messages=[dict(message) for message in pool.messages],
            items=list(pool.items),
            realized=realized,
        )

    @property
    def pool(self) -> TurnPool:
        """The persisted pool, as the adapter received it."""
        return TurnPool(
            messages=[dict(message) for message in self.messages],
            items=list(self.items),
        )

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": "record",
            "turn_id": self.turn_id,
            "span": [self.span[0], self.span[1]],
            "timestamp": self.timestamp,
            "payload": {
                "messages": self.messages,
                "items": [_item_to_json(item) for item in self.items],
            },
            "sequence": self.sequence,
        }
        if self.realized is not None:
            data["realized"] = self.realized
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> JournalRecord:
        payload = data["payload"]
        span = data["span"]
        return cls(
            turn_id=data["turn_id"],
            span=(int(span[0]), int(span[1])),
            timestamp=data["timestamp"],
            messages=[dict(message) for message in payload["messages"]],
            items=[_item_from_json(item) for item in payload["items"]],
            realized=data.get("realized"),
            sequence=int(data["sequence"]),
        )


@dataclass(frozen=True)
class ResumePlan:
    """What a resume will re-write, and what its log must name."""

    records: list[JournalRecord] = field(default_factory=list)
    turn_ids: list[str] = field(default_factory=list)


def _item_to_json(item: ExperienceItem) -> dict[str, Any]:
    return {
        "kind": item.kind,
        "role": item.role,
        "order": item.order,
        "offset_seconds": item.offset_seconds,
        "attrs": dict(item.attrs),
        "body": item.body,
    }


def _item_from_json(data: dict[str, Any]) -> ExperienceItem:
    offset = data["offset_seconds"]
    return ExperienceItem(
        kind=data["kind"],
        role=data["role"],
        order=int(data["order"]),
        offset_seconds=None if offset is None else int(offset),
        attrs=dict(data["attrs"]),
        body=data["body"],
    )


def _receipt_summary(receipt: WriteReceipt) -> dict[str, Any]:
    status = receipt.status
    return {
        "status": getattr(status, "value", status),
        "native_ids": list(receipt.native_ids),
        "native_events": list(receipt.native_events),
        "chunk_count": receipt.chunk_count,
    }


class TurnJournal:
    """An append-only journal of one session's pool writes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: list[JournalRecord] = []
        self._by_turn: dict[str, int] = {}
        self._load()

    # ── reading ────────────────────────────────────────────────────

    def records(self) -> list[JournalRecord]:
        """Every record, in append order."""
        return list(self._records)

    def __iter__(self) -> Iterator[JournalRecord]:
        return iter(self.records())

    def __len__(self) -> int:
        return len(self._records)

    def record_for(self, turn_id: str) -> JournalRecord:
        try:
            return self._records[self._by_turn[turn_id]]
        except KeyError:
            raise UnknownTurnError(turn_id) from None

    def pending_records(self) -> list[JournalRecord]:
        return [record for record in self._records if record.state == PENDING]

    def resume_pending(self) -> ResumePlan:
        """The records a resume re-writes, under Q5's at-least-once rule."""
        pending = self.pending_records()
        return ResumePlan(
            records=pending,
            turn_ids=[record.turn_id for record in pending],
        )

    def end_cursor(self) -> JournalCursor:
        return JournalCursor(index=len(self._records))

    def cursor_after(self, record: JournalRecord) -> JournalCursor:
        return JournalCursor(index=record.sequence + 1)

    def records_after_cursor(self, cursor: JournalCursor) -> list[JournalRecord]:
        return self._records[cursor.index :]

    def records_through_message_index(self, end_index: int) -> list[JournalRecord]:
        """The journal prefix whose turns end at or before `end_index` (§6.1 (e)).

        Zero-span records belong to the turn that produced them, so a `[n,n)`
        record is carried with that prefix: the prefix ends at the first record
        that starts after `end_index` or whose span reaches past it.
        """
        boundary = len(self._records)
        for position, record in enumerate(self._records):
            starts_after = record.span[0] > end_index
            reaches_past = not record.is_zero_span and record.span[1] > end_index
            if starts_after or reaches_past:
                boundary = position
                break
        return self._records[:boundary]

    def reload(self) -> None:
        """Re-read the file, replacing the in-memory records.

        Two objects can hold the same path (the `/new` command and the loop, a
        resume and a consumer); this is how the stale one catches up.
        """
        self._records = []
        self._by_turn = {}
        self._load()

    # ── writing ────────────────────────────────────────────────────

    def append_pending(self, record: JournalRecord) -> JournalRecord:
        """Append one pending record and make it durable before returning."""
        if record.state != PENDING:
            raise ValueError(f"append_pending needs a pending record: {record.state!r}")
        if record.turn_id in self._by_turn:
            raise ValueError(f"turn already journalled: {record.turn_id!r}")

        stored = replace(record, sequence=len(self._records))
        self._append_line(stored.to_json())
        self._by_turn[stored.turn_id] = len(self._records)
        self._records.append(stored)
        return stored

    def mark_written(self, turn_id: str, receipt: WriteReceipt) -> JournalRecord:
        """Record the adapter receipt. Calling it twice changes nothing."""
        index = self._index_of(turn_id)
        if self._records[index].state == WRITTEN:
            return self._records[index]

        summary = _receipt_summary(receipt)
        self._append_line(
            {"kind": "mark", "mark": WRITTEN, "turn_id": turn_id, "receipt": summary}
        )
        self._records[index] = replace(
            self._records[index], state=WRITTEN, receipt=summary
        )
        return self._records[index]

    def mark_file_consolidated(self, turn_ids: Iterable[str]) -> None:
        """Mark the second completion track for the file condition (§6.1 (e))."""
        ids = [turn_id for turn_id in turn_ids]
        indices = [self._index_of(turn_id) for turn_id in ids]
        unmarked = [
            turn_id
            for turn_id, index in zip(ids, indices, strict=True)
            if not self._records[index].file_consolidated
        ]
        if not unmarked:
            return

        self._append_line(
            {"kind": "mark", "mark": "file_consolidated", "turn_ids": unmarked}
        )
        for turn_id in unmarked:
            index = self._by_turn[turn_id]
            self._records[index] = replace(
                self._records[index], file_consolidated=True
            )

    def mark_dropped_from_files(self, turn_ids: Iterable[str]) -> None:
        """Keep these turns out of the memory files for good (file condition)."""
        ids = [turn_id for turn_id in turn_ids]
        indices = [self._index_of(turn_id) for turn_id in ids]
        unmarked = [
            turn_id
            for turn_id, index in zip(ids, indices, strict=True)
            if not self._records[index].dropped_from_files
        ]
        if not unmarked:
            return

        self._append_line(
            {"kind": "mark", "mark": "dropped_from_files", "turn_ids": unmarked}
        )
        for turn_id in unmarked:
            index = self._by_turn[turn_id]
            self._records[index] = replace(
                self._records[index], dropped_from_files=True
            )

    def open_boundary(self) -> Path | None:
        """Rotate the journal aside and open an empty one (`/new`, §9 step 7).

        `<name>.journal.jsonl` becomes `<name>.journal.<n>.jsonl` with `n` the
        next free integer, so the cleared session's turn ids and spans cannot
        collide with the new session's. Returns the rotated path, or `None`
        when there was no journal file to rotate.
        """
        rotated: Path | None = None
        if self.path.exists():
            rotated = self._next_rotation_path()
            self.path.rename(rotated)
        self._records = []
        self._by_turn = {}
        return rotated

    # ── storage ────────────────────────────────────────────────────

    def _next_rotation_path(self) -> Path:
        stem = self.path.name[: -len(self.path.suffix)] if self.path.suffix else self.path.name
        number = 1
        while True:
            candidate = self.path.parent / f"{stem}.{number}{self.path.suffix}"
            if not candidate.exists():
                return candidate
            number += 1

    def _index_of(self, turn_id: str) -> int:
        try:
            return self._by_turn[turn_id]
        except KeyError:
            raise UnknownTurnError(turn_id) from None

    def _append_line(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        if created:
            # The first append creates the file; the entry naming it in the
            # parent directory needs its own fsync, or a crash can leave the
            # durable bytes with no name.
            self._fsync_parent()

    def _fsync_parent(self) -> None:
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as handle:
            lines = handle.readlines()
        for number, raw in enumerate(lines, start=1):
            is_tail = number == len(lines)
            torn_tail = is_tail and not raw.endswith("\n")
            line = raw.strip()
            if not line:
                continue
            if torn_tail:
                # A crash mid-append leaves a partial trailing line. Every
                # completed write ends in a newline, so a tail without one was
                # never finished: discard it. Only the tail — a bad line
                # anywhere before it is real corruption and still raises.
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JournalLoadError(
                    f"Undecodable journal line: path={self.path} line={number}"
                ) from exc
            self._apply(entry, number)

    def _apply(self, entry: dict[str, Any], number: int) -> None:
        kind = entry.get("kind")
        if kind == "record":
            record = JournalRecord.from_json(entry)
            self._by_turn[record.turn_id] = len(self._records)
            self._records.append(record)
            return
        if kind == "mark" and entry["mark"] == WRITTEN:
            index = self._index_of(entry["turn_id"])
            self._records[index] = replace(
                self._records[index], state=WRITTEN, receipt=entry["receipt"]
            )
            return
        if kind == "mark" and entry["mark"] == "file_consolidated":
            for turn_id in entry["turn_ids"]:
                index = self._index_of(turn_id)
                self._records[index] = replace(
                    self._records[index], file_consolidated=True
                )
            return
        if kind == "mark" and entry["mark"] == "dropped_from_files":
            for turn_id in entry["turn_ids"]:
                index = self._index_of(turn_id)
                self._records[index] = replace(
                    self._records[index], dropped_from_files=True
                )
            return
        if kind == "mark" and entry["mark"] in (
            "file_unconsolidated_skipped",
            "whole_scene_file_unconsolidated",
        ):
            # Marks written by runs before 7aae87cc6, read as dropped_from_files; the
            # old code's keeping of a failed interval in the prompt is not reproduced.
            if entry["mark"] == "file_unconsolidated_skipped":
                rows = entry.get("records")
                valid = isinstance(rows, list) and bool(rows) and all(
                    isinstance(row, dict) and isinstance(row.get("turn_id"), str)
                    for row in rows
                )
                turn_ids = [row["turn_id"] for row in rows] if valid else []
            else:
                valid = isinstance(entry.get("turn_id"), str)
                turn_ids = [entry["turn_id"]] if valid else []
            if not valid:
                raise JournalLoadError(
                    f"Invalid {entry['mark']} mark: path={self.path} line={number}"
                )
            for turn_id in turn_ids:
                index = self._index_of(turn_id)
                self._records[index] = replace(
                    self._records[index], dropped_from_files=True
                )
            return
        raise JournalLoadError(
            f"Unknown journal entry: path={self.path} line={number} kind={kind!r}"
        )


async def replay_pending_turns(
    *,
    journal: TurnJournal,
    adapter: Any,
    scope: Any,
    session: Any,
    require_committed: bool = False,
    records: Iterable[JournalRecord] | None = None,
) -> list[str]:
    """Rewrite every `pending` record's pool and return the ids, in order.

    §6.1 (a), Q5: **at-least-once, declared.** A turn whose adapter write was
    acknowledged but whose state never reached disk is written a second time,
    and the duplicate is left for the backend to absorb — no adapter carries an
    idempotency key. Nothing here dedupes, and no exactly-once claim is made.

    The authority for the skip is the record's `state`, never a message index
    (§6.1 (c)), which is what makes a zero-span turn resumable: a suppressed
    outreach's span is `[n, n)` and no index can tell "sent" from "never sent".

    The pool is the one the interrupted turn rendered, read back off the record
    — not re-derived from `session.messages`, which cannot carry it — and it
    goes to the adapter through `write_visible_memory_event`, the same path an
    uninterrupted turn uses. The cursor advances only behind the receipt, as it
    does live, and neither file cursor is touched.

    One condition is exempt, and it is the whole of the exemption: an adapter
    declaring `after_file_consolidation` does not do the write here. The file
    condition's `record_event` is a documented no-op returning
    `deferred_to_file_consolidation` and its real write happens later, at
    consolidation time, so a replay would put nothing in the memory files and
    still stamp the record `written` — a receipt for a write that never
    happened. Its completion track is `file_consolidated`, which the
    consolidation entry points own; here the resume replays nothing, marks
    nothing and mutates no file. The authority is the adapter's declared
    `prompt_history_eviction_policy`, never its class.
    """
    policy = getattr(adapter, "prompt_history_eviction_policy", None)
    if policy == PromptHistoryEvictionPolicy.after_file_consolidation:
        return []

    rewritten: list[str] = []
    # `records`: a subset of the pending records (a failure policy can leave a
    # skipped step's unwritten pools out of memory); default all of them.
    selected = list(records) if records is not None else journal.resume_pending().records
    for record in selected:
        if record.state != PENDING or journal.record_for(record.turn_id) != record:
            raise ValueError("pending replay selection differs from the journal")
        receipt = await write_visible_memory_event(
            adapter=adapter,
            pool=record.pool,
            run_id=scope.run_id,
            persona_id=scope.persona_id,
            agent_id=scope.agent_id or "nanobot",
            session_id=session.key,
            turn_id=record.turn_id,
            event_id=f"{scope.run_id}:{session.key}:{record.turn_id}",
            timestamp=datetime.fromisoformat(record.timestamp),
            require_committed=require_committed,
            eviction_policy=getattr(
                adapter, "prompt_history_eviction_policy", None
            ),
        )
        if receipt is None:
            continue
        journal.mark_written(record.turn_id, receipt)
        session.adapter_committed_end_index = record_adapter_write_commit(
            receipt=receipt,
            session_id=session.key,
            current_index=record.span[1],
            committed_index=session.adapter_committed_end_index,
            eviction_policy=getattr(
                adapter, "prompt_history_eviction_policy", None
            ),
            require_committed=require_committed,
        )
        logger.info(
            "Resume rewrote turn {} of session {} (at-least-once, §6.1 (a))",
            record.turn_id,
            session.key,
        )
        rewritten.append(record.turn_id)
    return rewritten
