"""The append-only per-session journal of pool writes."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import stat

import pytest

from memory.event_writer import TurnPool, records_owed_to_files
from memory.results import WriteReceipt, WriteStatus
from memory.turn_experience import ExperienceItem
from memory.turn_journal import (
    JournalCursor,
    JournalLoadError,
    JournalRecord,
    TurnJournal,
    UnknownTurnError,
    journal_path,
)


def _pool(text: str = "On it.") -> TurnPool:
    return TurnPool(
        messages=[
            {"role": "user", "content": "Where is the package?"},
            {"role": "assistant", "content": f"[prose]\nChecking.\n[/prose]\n{text}"},
        ],
        items=[
            ExperienceItem(
                kind="user",
                role="user",
                order=0,
                offset_seconds=None,
                attrs={},
                body="Where is the package?",
            ),
            ExperienceItem(
                kind="prose",
                role="assistant",
                order=1,
                offset_seconds=120,
                attrs={},
                body="Checking.",
            ),
            ExperienceItem(
                kind="assistant",
                role="assistant",
                order=2,
                offset_seconds=None,
                attrs={},
                body=text,
            ),
        ],
    )


def _receipt() -> WriteReceipt:
    return WriteReceipt(
        status=WriteStatus.committed,
        native_ids=["mem-1"],
        native_events=["add"],
    )


def _record(turn_id: str, span: tuple[int, int], pool: TurnPool | None = None) -> JournalRecord:
    return JournalRecord.for_pool(
        turn_id=turn_id,
        span=span,
        pool=pool if pool is not None else _pool(),
        timestamp="2026-09-13T10:00:00",
    )


def test_journal_sits_beside_the_session_file(tmp_path):
    from pa.session.manager import SessionManager

    manager = SessionManager(tmp_path)
    session = manager.create("cli:local")
    session_file = manager._get_session_path(session.key)

    path = journal_path(manager.sessions_dir, session.key)

    assert path.parent == session_file.parent
    assert path.name == f"{session_file.stem}.journal.jsonl"


def test_append_pending_is_durable_before_any_further_write(tmp_path, monkeypatch):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)

    fsyncs: list[tuple[bool, str]] = []
    real_fsync = os.fsync

    def spy(fd: int):
        result = real_fsync(fd)
        is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
        fsyncs.append((is_dir, "" if is_dir else path.read_text(encoding="utf-8")))
        return result

    monkeypatch.setattr(os, "fsync", spy)

    stored = journal.append_pending(_record("turn-1", (0, 2)))

    # The record line is on disk before append_pending returns, and the first
    # append also fsyncs the parent directory that now names the file.
    assert [is_dir for is_dir, _ in fsyncs] == [False, True]
    assert '"turn_id": "turn-1"' in fsyncs[0][1]

    fsyncs.clear()
    journal.append_pending(_record("turn-2", (2, 4)))
    assert [is_dir for is_dir, _ in fsyncs] == [False]

    assert stored.state == "pending"
    reopened = TurnJournal(path)
    assert [r.turn_id for r in reopened] == ["turn-1", "turn-2"]
    assert reopened.records()[0].state == "pending"


def test_records_iterate_in_append_order(tmp_path):
    journal = TurnJournal(journal_path(tmp_path, "cli:local"))

    journal.append_pending(_record("turn-1", (0, 2)))
    journal.append_pending(_record("turn-2", (2, 4)))
    journal.append_pending(_record("turn-3", (4, 6)))

    assert [r.turn_id for r in journal] == ["turn-1", "turn-2", "turn-3"]


def test_cursor_returns_only_the_records_after_it(tmp_path):
    journal = TurnJournal(journal_path(tmp_path, "cli:local"))
    first = journal.append_pending(_record("turn-1", (0, 2)))
    journal.append_pending(_record("turn-2", (2, 4)))

    after_first = journal.cursor_after(first)
    assert [r.turn_id for r in journal.records_after_cursor(after_first)] == ["turn-2"]
    assert [r.turn_id for r in journal.records_after_cursor(JournalCursor())] == [
        "turn-1",
        "turn-2",
    ]
    assert journal.records_after_cursor(journal.end_cursor()) == []


def test_cursor_survives_a_round_trip_through_json(tmp_path):
    journal = TurnJournal(journal_path(tmp_path, "cli:local"))
    first = journal.append_pending(_record("turn-1", (0, 2)))
    journal.append_pending(_record("turn-2", (2, 4)))

    restored = JournalCursor.from_json(journal.cursor_after(first).to_json())

    assert [r.turn_id for r in journal.records_after_cursor(restored)] == ["turn-2"]


def test_empty_span_record_round_trips(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)

    journal.append_pending(_record("turn-suppressed", (7, 7)))

    record = TurnJournal(path).records()[0]
    assert record.span == (7, 7)
    assert record.is_zero_span


def test_crash_after_append_leaves_the_record_pending_for_resume(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2)))
    journal.mark_written("turn-1", _receipt())
    journal.append_pending(_record("turn-2", (2, 4)))
    # The process dies here: no mark_written for turn-2.
    del journal

    plan = TurnJournal(path).resume_pending()

    assert plan.turn_ids == ["turn-2"]
    assert [r.turn_id for r in plan.records] == ["turn-2"]


def test_mark_written_is_idempotent(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2)))

    journal.mark_written("turn-1", _receipt())
    journal.mark_written("turn-1", _receipt())

    reopened = TurnJournal(path)
    assert [r.state for r in reopened] == ["written"]
    assert reopened.pending_records() == []
    assert reopened.records()[0].receipt == {
        "status": "committed",
        "native_ids": ["mem-1"],
        "native_events": ["add"],
        "chunk_count": 1,
    }


def test_mark_file_consolidated_marks_only_the_named_turns(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2)))
    journal.append_pending(_record("turn-2", (2, 4)))

    journal.mark_file_consolidated(["turn-1"])

    reopened = TurnJournal(path)
    assert [r.file_consolidated for r in reopened] == [True, False]
    # The file mark is the second completion track: it does not touch state.
    assert [r.state for r in reopened] == ["pending", "pending"]


def test_marking_an_unknown_turn_is_an_error(tmp_path):
    journal = TurnJournal(journal_path(tmp_path, "cli:local"))
    journal.append_pending(_record("turn-1", (0, 2)))

    with pytest.raises(UnknownTurnError):
        journal.mark_written("turn-9", _receipt())
    with pytest.raises(UnknownTurnError):
        journal.mark_file_consolidated(["turn-9"])


def test_appending_the_same_turn_id_twice_is_an_error(tmp_path):
    journal = TurnJournal(journal_path(tmp_path, "cli:local"))
    journal.append_pending(_record("turn-1", (0, 2)))

    with pytest.raises(ValueError):
        journal.append_pending(_record("turn-1", (2, 4)))


def test_persisted_pool_round_trips_byte_equal(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    pool = _pool("Delivered Tuesday.")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2), pool))

    record = TurnJournal(path).records()[0]

    assert record.messages == pool.messages
    for stored, original in zip(record.messages, pool.messages, strict=True):
        assert stored["content"].encode("utf-8") == original["content"].encode("utf-8")
    assert record.items == pool.items


def test_journal_survives_a_copy_of_the_session_directory(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    journal = TurnJournal(journal_path(sessions_dir, "cli:local"))
    journal.append_pending(_record("turn-1", (0, 2)))
    journal.mark_written("turn-1", _receipt())
    journal.append_pending(_record("turn-2", (2, 2)))

    branch = tmp_path / "branch" / "sessions"
    shutil.copytree(sessions_dir, branch)

    restored = TurnJournal(journal_path(branch, "cli:local"))
    assert [(r.turn_id, r.state) for r in restored] == [
        ("turn-1", "written"),
        ("turn-2", "pending"),
    ]
    assert restored.resume_pending().turn_ids == ["turn-2"]


def test_a_torn_trailing_line_is_discarded_and_the_rest_still_loads(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2), _pool("x" * 5000)))
    journal.mark_written("turn-1", _receipt())
    journal.append_pending(_record("turn-2", (2, 4), _pool("y" * 5000)))

    # The process dies mid-append: the last line is cut in half.
    raw = path.read_bytes()
    last_newline = raw.rindex(b"\n", 0, len(raw) - 1)
    path.write_bytes(raw[: last_newline + 1 + (len(raw) - last_newline) // 2])

    reopened = TurnJournal(path)

    assert [(r.turn_id, r.state) for r in reopened] == [("turn-1", "written")]
    assert reopened.resume_pending().turn_ids == []
    assert reopened.records()[0].messages == _pool("x" * 5000).messages


def test_a_torn_trailing_pending_record_leaves_the_written_turn_resumable(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2), _pool("a" * 5000)))
    journal.append_pending(_record("turn-2", (2, 4), _pool("b" * 5000)))

    raw = path.read_bytes()
    last_newline = raw.rindex(b"\n", 0, len(raw) - 1)
    path.write_bytes(raw[: last_newline + 200])

    reopened = TurnJournal(path)

    assert reopened.resume_pending().turn_ids == ["turn-1"]


def test_a_malformed_line_before_the_tail_still_raises(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2)))
    journal.append_pending(_record("turn-2", (2, 4)))

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0] = "{not json at all\n"
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(JournalLoadError):
        TurnJournal(path)


def test_a_newline_terminated_corrupt_tail_still_raises(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2)))
    journal.append_pending(_record("turn-2", (2, 4)))

    # The last line ends in a newline, so the write completed — a corrupt
    # completed record is real corruption, not a torn tail.
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[-1] = "{not json at all\n"
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(JournalLoadError):
        TurnJournal(path)


def test_open_boundary_rotates_the_journal_aside_and_starts_empty(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2)))

    rotated = journal.open_boundary()

    assert rotated is not None
    assert rotated.name == path.name.replace(".journal.jsonl", ".journal.1.jsonl")
    assert not path.exists()
    assert journal.records() == []
    assert [r.turn_id for r in TurnJournal(rotated)] == ["turn-1"]

    # The new session reuses turn ids and spans without colliding.
    journal.append_pending(_record("turn-1", (0, 2)))
    second = journal.open_boundary()
    assert second is not None and second.name.endswith(".journal.2.jsonl")


def test_reload_sees_an_append_made_through_another_instance(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    reader = TurnJournal(path)
    writer = TurnJournal(path)

    writer.append_pending(_record("turn-1", (0, 2)))
    assert reader.records() == []

    reader.reload()

    assert [r.turn_id for r in reader] == ["turn-1"]
    assert reader.resume_pending().turn_ids == ["turn-1"]


def test_records_through_message_index_carries_the_zero_span_record(tmp_path):
    journal = TurnJournal(journal_path(tmp_path, "cli:local"))
    journal.append_pending(_record("turn-1", (0, 2)))
    journal.append_pending(_record("turn-suppressed", (2, 2)))
    journal.append_pending(_record("turn-2", (2, 4)))

    through_two = journal.records_through_message_index(2)

    assert [r.turn_id for r in through_two] == ["turn-1", "turn-suppressed"]
    assert [r.turn_id for r in journal.records_through_message_index(4)] == [
        "turn-1",
        "turn-suppressed",
        "turn-2",
    ]
    assert journal.records_through_message_index(0) == []


def test_realized_artefacts_round_trip(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    realized = {
        "world_ids": ["evt_9f2"],
        "call_references": {"$call_1": "order_7731"},
    }

    journal.append_pending(
        dataclasses.replace(_record("turn-1", (0, 2)), realized=realized)
    )
    journal.append_pending(_record("turn-2", (2, 4)))

    reopened = TurnJournal(path).records()
    assert reopened[0].realized == realized
    assert reopened[1].realized is None
    # The slot is absent from the line when nothing realized it.
    second_line = path.read_text(encoding="utf-8").splitlines()[1]
    assert "realized" not in second_line


# Journals written by older versions carry two other marks for a turn the
# memory files are never to have. The lines below are synthetic, shaped like
# those journals' lines.


def _append_raw_line(path, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _skipped_interval_mark(rows) -> dict:
    return {
        "kind": "mark",
        "mark": "file_unconsolidated_skipped",
        "step_id": "acc_002",
        "interval": [2, 4],
        "validated_reason_sha256": "a" * 64,
        "receipt": {"run_relative_path": "memory_partial_skip/acc_002/x.json", "sha256": "b" * 64},
        "validated_reason_refs": {},
        "records": rows,
    }


def _whole_scene_mark(mark: str, turn_id) -> dict:
    return {
        "kind": "mark",
        "mark": mark,
        "turn_id": turn_id,
        "proof": {"schema": "mempa_whole_scene_failed_terminal_journal_mark/1", "step_id": "acc_002"},
    }


def _two_turn_journal(tmp_path):
    path = journal_path(tmp_path, "cli:local")
    journal = TurnJournal(path)
    journal.append_pending(_record("turn-1", (0, 2)))
    journal.append_pending(_record("turn-2", (2, 4)))
    journal.mark_written("turn-2", _receipt())
    return path


@pytest.mark.parametrize(
    "entry",
    [
        _skipped_interval_mark([{"turn_id": "turn-2", "span": [2, 4]}]),
        _whole_scene_mark("whole_scene_file_unconsolidated", "turn-2"),
    ],
    ids=["file_unconsolidated_skipped", "whole_scene_file_unconsolidated"],
)
def test_older_journal_marks_read_as_dropped_from_files(tmp_path, entry):
    # An old run must load as a reference-arm source: its turn stays out of
    # the memory files exactly as `dropped_from_files` keeps it out.
    path = _two_turn_journal(tmp_path)
    _append_raw_line(path, entry)

    reopened = TurnJournal(path)

    turn = reopened.record_for("turn-2")
    assert turn.dropped_from_files and not turn.file_consolidated
    assert turn.state == "written"
    assert not reopened.record_for("turn-1").dropped_from_files
    assert [r.turn_id for r in records_owed_to_files(reopened.records())] == ["turn-1"]


@pytest.mark.parametrize(
    "entry",
    [
        {k: v for k, v in _skipped_interval_mark([]).items() if k != "records"},
        _skipped_interval_mark([]),
        _skipped_interval_mark({"turn_id": "turn-2"}),
        _skipped_interval_mark([{"span": [2, 4]}]),
        _skipped_interval_mark([{"turn_id": 2}]),
        _skipped_interval_mark(["turn-2"]),
        {k: v for k, v in _whole_scene_mark("whole_scene_file_unconsolidated", "x").items() if k != "turn_id"},
        _whole_scene_mark("whole_scene_file_unconsolidated", ["turn-2"]),
    ],
    ids=[
        "records-missing",
        "records-empty",
        "records-not-a-list",
        "row-without-turn-id",
        "row-turn-id-not-str",
        "row-not-an-object",
        "whole-scene-turn-id-missing",
        "whole-scene-turn-id-not-str",
    ],
)
def test_an_older_journal_mark_without_its_turn_ids_raises_naming_the_line(tmp_path, entry):
    # Nothing is defaulted: a mark that cannot say which turns it names is
    # corruption, and the error points at the file and line.
    path = _two_turn_journal(tmp_path)
    _append_raw_line(path, entry)

    with pytest.raises(JournalLoadError, match=rf"path={path} line=4$"):
        TurnJournal(path)


def test_an_older_journal_mark_naming_an_unknown_turn_raises(tmp_path):
    path = _two_turn_journal(tmp_path)
    _append_raw_line(path, _whole_scene_mark("whole_scene_file_unconsolidated", "turn-9"))

    with pytest.raises(UnknownTurnError):
        TurnJournal(path)


@pytest.mark.parametrize(
    "mark", ["whole_scene_failed_terminal", "failed_terminal", "not_a_mark"]
)
def test_other_unknown_marks_still_raise(tmp_path, mark):
    # `whole_scene_failed_terminal` set a state this code has no counterpart
    # for, so it is not read as anything.
    path = _two_turn_journal(tmp_path)
    _append_raw_line(path, _whole_scene_mark(mark, "turn-1"))

    with pytest.raises(JournalLoadError, match="Unknown journal entry"):
        TurnJournal(path)
