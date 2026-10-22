import json

import pytest

from memory.event_writer import TurnPool
from memory.turn_experience import ExperienceItem
from memory.turn_journal import JournalRecord, TurnJournal, journal_path
from pa.session.manager import (
    InvalidSessionIndexError,
    Session,
    SessionAlreadyExistsError,
    SessionLoadError,
    SessionManager,
    SessionNotFoundError,
    UnsupportedSessionSchemaError,
)


def test_session_creation_is_explicit_and_cannot_overwrite(tmp_path):
    manager = SessionManager(tmp_path)
    created = manager.create("run_001_main")

    assert manager.get("run_001_main") is created
    with pytest.raises(SessionAlreadyExistsError, match="run_001_main"):
        manager.create("run_001_main")


def test_missing_required_session_fails_with_key_and_path(tmp_path):
    manager = SessionManager(tmp_path)

    with pytest.raises(SessionNotFoundError, match="missing_main") as exc_info:
        manager.get("missing_main")

    assert "sessions/missing_main.jsonl" in str(exc_info.value)


def test_corrupt_session_reports_line_and_value(tmp_path):
    manager = SessionManager(tmp_path)
    path = manager._get_session_path("run_001_main")
    path.write_text(
        '{"_type":"metadata","session_schema_version":3}\n{broken\n',
        encoding="utf-8",
    )

    with pytest.raises(SessionLoadError, match="line=2") as exc_info:
        manager.get("run_001_main")

    assert "value='{broken'" in str(exc_info.value)


def test_session_round_trips_separate_memory_indices(tmp_path):
    manager = SessionManager(tmp_path)
    session = Session(
        key="test:indices",
        file_consolidated_end_index=2,
        adapter_committed_end_index=4,
        prompt_history_start_index=2,
    )
    session.messages = [{"role": "user", "content": str(i)} for i in range(6)]

    manager.save(session)
    manager.invalidate(session.key)
    loaded = manager.get(session.key)

    assert loaded.file_consolidated_end_index == 2
    assert loaded.adapter_committed_end_index == 4
    assert loaded.prompt_history_start_index == 2
    assert [m["content"] for m in loaded.get_history()] == ["2", "3", "4", "5"]


def test_obsolete_session_schema_fails_instead_of_resetting_indices(tmp_path):
    manager = SessionManager(tmp_path)
    session_path = manager._get_session_path("test:obsolete")
    rows = [
        {
            "_type": "metadata",
            "key": "test:obsolete",
            "last_consolidated": 2,
        },
        {"role": "user", "content": "old message"},
    ]
    session_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        UnsupportedSessionSchemaError,
        match="expected 3",
    ):
        manager.get("test:obsolete")


def test_out_of_range_session_index_is_rejected(tmp_path):
    manager = SessionManager(tmp_path)
    session_path = manager._get_session_path("test:invalid")
    rows = [
        {
            "_type": "metadata",
            "key": "test:invalid",
            "session_schema_version": 3,
            "file_consolidated_end_index": 2,
            "adapter_committed_end_index": 0,
            "prompt_history_start_index": 0,
        },
        {"role": "user", "content": "only message"},
    ]
    session_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidSessionIndexError,
        match="outside 0..1",
    ):
        manager.get("test:invalid")


def _journal_with_one_pending(manager, key):
    """One pending pool write, beside the session file `key` names."""
    journal = TurnJournal(journal_path(manager.sessions_dir, key))
    journal.append_pending(JournalRecord.for_pool(
        turn_id="turn-1",
        span=(0, 2),
        pool=TurnPool(
            messages=[{"role": "user", "content": "Where is the package?"}],
            items=[ExperienceItem(
                kind="user", role="user", order=0, offset_seconds=None,
                attrs={}, body="Where is the package?",
            )],
        ),
        timestamp="2026-09-13T10:00:00",
    ))
    return journal.path


def test_the_journal_is_saved_and_restored_with_the_checkpoint(tmp_path):
    """A checkpoint that drops the journal drops the pool writes.

    The benchmark harness snapshots the whole `pa_workspace`, so it carries
    the journal today; a caller of this manager-level round trip got a valid
    session with no journal at all and lost the turns still owed to the
    adapter.
    """
    manager = SessionManager(tmp_path)
    session = manager.create("bench:001")
    session.add_message("user", "Where is the package?")
    manager.save(session)
    source_journal = _journal_with_one_pending(manager, session.key)
    written = source_journal.read_bytes()

    checkpoint_dir = tmp_path / "checkpoints" / "s1"
    manager.checkpoint(session, checkpoint_dir)
    source_journal.unlink()

    manager.restore_checkpoint("bench:001_eval", checkpoint_dir)

    restored = journal_path(manager.sessions_dir, "bench:001_eval")
    assert restored.read_bytes() == written
    assert [record.turn_id for record in TurnJournal(restored).records()] == ["turn-1"]


def test_restoring_a_checkpoint_without_a_journal_leaves_none_behind(tmp_path):
    """A journal written after the checkpoint records turns it never took."""
    manager = SessionManager(tmp_path)
    session = manager.create("bench:001")
    session.add_message("user", "Where is the package?")
    manager.save(session)

    checkpoint_dir = tmp_path / "checkpoints" / "s1"
    manager.checkpoint(session, checkpoint_dir)
    assert not (checkpoint_dir / "session.journal.jsonl").exists()
    later = _journal_with_one_pending(manager, "bench:001_eval")

    manager.restore_checkpoint("bench:001_eval", checkpoint_dir)

    assert not later.exists()


def test_list_sessions_reads_past_the_journal_beside_a_session(tmp_path):
    """Live `run_forced_event_gate_c.py:166`, after the FE executor journals.

    `<safe_key>.journal.jsonl` and the `<safe_key>.journal.<n>.jsonl` `/new`
    rotates aside both match `*.jsonl` and hold no session metadata line, so
    enumerating them raised `SessionLoadError` on a run that had written one.
    """
    manager = SessionManager(tmp_path)
    session = manager.create("benchmark:main")
    _journal_with_one_pending(manager, session.key)
    TurnJournal(journal_path(manager.sessions_dir, session.key)).open_boundary()
    _journal_with_one_pending(manager, session.key)
    assert len(list(manager.sessions_dir.glob("*.jsonl"))) == 3

    listed = manager.list_sessions()

    assert [record["key"] for record in listed] == ["benchmark:main"]
