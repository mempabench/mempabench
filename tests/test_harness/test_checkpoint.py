from __future__ import annotations

from contextlib import contextmanager
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from harness.checkpoint import (
    CheckpointManager,
    LEDGER_LAYER,
    RecoveryInProgressError,
    checkpoint_restorable,
    IncompleteCheckpointError,
    partition_checkpoints_by_target,
    read_pending_recovery,
    rebind_memos_cube_to_branch,
    run_restorable,
    verify_layers,
)
from harness.run_ledger import RunLedger
from memory.turn_experience import MEMORY_POOL_VERSION
from pa.agent.ix.render_types import (
    IX_CONTRACT_VERSION,
    IX_RENDER_SCHEMA_VERSION,
)
from pa.session.manager import Session, SessionManager


PERSONA = "user_a"
RUN_ID = "run_001"
SESSION_KEY = f"{RUN_ID}_main"


def _build_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / RUN_ID
    workspace = run_dir / "pa_workspace" / PERSONA
    (workspace / "memory").mkdir(parents=True)
    (workspace / "memory" / "MEMORY.md").write_text("pa memory\n", encoding="utf-8")
    (workspace / "SOUL.md").write_text("soul\n", encoding="utf-8")

    sim = run_dir / "simulator_workspace" / PERSONA / "memory"
    sim.mkdir(parents=True)
    (sim / "MEMORY.md").write_text("simulator memory\n", encoding="utf-8")

    stage = run_dir / "canonical_stage" / "email"
    stage.mkdir(parents=True)
    (stage / "inbox.json").write_text("[]", encoding="utf-8")

    (run_dir / "memory_io.jsonl").write_text('{"seq": 1}\n', encoding="utf-8")
    (run_dir / "run_plan.yaml").write_text("run_id: run_001\n", encoding="utf-8")
    return run_dir


def _write_session(run_dir: Path, *, cursors: tuple[int, int, int]) -> Session:
    manager = SessionManager(run_dir / "pa_workspace" / PERSONA)
    session = Session(
        key=SESSION_KEY,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
        file_consolidated_end_index=cursors[0],
        adapter_committed_end_index=cursors[1],
        prompt_history_start_index=cursors[2],
    )
    manager.save(session)
    return session


def _manager(
    run_dir: Path, ledger: RunLedger, *, rolling_window: int = 100
) -> CheckpointManager:
    return CheckpointManager(
        run_dir,
        ledger=ledger,
        memory_condition="nanobot_file_memory",
        memory_backend_namespace="ns_001",
        rolling_window=rolling_window,
    )


def _set_checkpoint_timestamp(
    run_dir: Path, checkpoint_id: str, timestamp: str
) -> None:
    """What a host clock that stepped backwards leaves in the metadata."""
    path = run_dir / "checkpoints" / checkpoint_id / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["timestamp"] = timestamp
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _session_metadata_line(path: Path) -> dict:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(first)


# ── step 3 ────────────────────────────────────────────────────────


def test_verify_layers_reports_missing_layers(tmp_path: Path) -> None:
    cp_dir = tmp_path / "cp"
    (cp_dir / "pa_workspace").mkdir(parents=True)
    (cp_dir / "simulator_workspace").mkdir(parents=True)
    (cp_dir / "ledger.json").write_text('{"steps": {}}', encoding="utf-8")

    missing = verify_layers(
        cp_dir, memory_condition="nanobot_file_memory", pinned=False
    )
    assert "canonical_stage" in missing
    assert "run_plan.yaml" not in missing

    pinned_missing = verify_layers(
        cp_dir, memory_condition="nanobot_file_memory", pinned=True
    )
    assert "run_plan.yaml" in pinned_missing


# ── step 4 ────────────────────────────────────────────────────────


def test_incomplete_checkpoint_is_not_published(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_001")
    manager = _manager(run_dir, ledger)

    # The simulator workspace never existed, so no layer can stand in for it.
    import shutil

    shutil.rmtree(run_dir / "simulator_workspace")

    with pytest.raises(IncompleteCheckpointError):
        manager.save("acc_001", pinned=False)

    assert not (run_dir / "checkpoints" / "acc_001").exists()
    assert manager.list_checkpoints() == []
    assert not list((run_dir / "checkpoints").glob(".staging-*"))


# ── step 5 ────────────────────────────────────────────────────────


def test_save_preserves_session_cursors(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(2, 3, 1))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_001")
    cp_dir = _manager(run_dir, ledger).save("acc_001", pinned=False)

    source = (
        run_dir / "pa_workspace" / PERSONA / "sessions" / f"{SESSION_KEY}.jsonl"
    )
    snapshot = (
        cp_dir / "pa_workspace" / PERSONA / "sessions" / f"{SESSION_KEY}.jsonl"
    )
    assert snapshot.is_file()
    snapshot_meta = _session_metadata_line(snapshot)
    assert snapshot_meta == _session_metadata_line(source)
    assert snapshot_meta["file_consolidated_end_index"] == 2
    assert snapshot_meta["adapter_committed_end_index"] == 3
    assert snapshot_meta["prompt_history_start_index"] == 1
    assert (
        cp_dir / "pa_workspace" / PERSONA / "memory" / "MEMORY.md"
    ).read_text(encoding="utf-8") == "pa memory\n"


# ── step 6 ────────────────────────────────────────────────────────


def test_save_refuses_when_stage_work_not_empty(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_002")
    (run_dir / "stage_work" / "acc_002").mkdir(parents=True)

    with pytest.raises(IncompleteCheckpointError):
        _manager(run_dir, ledger).save("acc_002", pinned=False)

    assert not (run_dir / "checkpoints" / "acc_002").exists()


# ── step 7 ────────────────────────────────────────────────────────


def test_save_includes_simulator_memory(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_001")
    cp_dir = _manager(run_dir, ledger).save("acc_001", pinned=False)

    assert (
        cp_dir / "simulator_workspace" / PERSONA / "memory" / "MEMORY.md"
    ).read_text(encoding="utf-8") == "simulator memory\n"


# ── step 8 ────────────────────────────────────────────────────────


def test_snapshot_ledger_marks_current_step_done(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_done("acc_001")
    ledger.mark_running("acc_002")
    manager = _manager(run_dir, ledger)

    rolling = manager.save("acc_002", pinned=False)
    snapshot = json.loads((rolling / "ledger.json").read_text(encoding="utf-8"))
    assert snapshot["steps"]["acc_002"]["status"] == "done"
    assert snapshot["current_step"] is None

    mainline = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))
    assert mainline["steps"]["acc_002"]["status"] == "running"
    assert mainline["current_step"] == "acc_002"

    assert not (rolling / "run_plan.yaml").exists()

    ledger.mark_running("acc_003")
    pinned = manager.save("acc_003", pinned=True)
    assert (pinned / "run_plan.yaml").is_file()
    assert manager.metadata("acc_003")["memory_backend_namespace"] == "ns_001"
    assert manager.metadata("acc_003")["pinned"] is True


# ── step 9 ────────────────────────────────────────────────────────


def test_memory_io_is_captured_and_verified(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    source = run_dir / "memory_io.jsonl"
    source.write_text('{"seq": 1}\n{"seq": 2}\n', encoding="utf-8")
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_001")
    manager = _manager(run_dir, ledger)

    cp_dir = manager.save("acc_001", pinned=False)
    assert (cp_dir / "memory_io.jsonl").read_text(
        encoding="utf-8"
    ) == source.read_text(encoding="utf-8")

    source.unlink()
    ledger.mark_running("acc_002")
    with pytest.raises(IncompleteCheckpointError):
        manager.save("acc_002", pinned=False)
    assert not (run_dir / "checkpoints" / "acc_002").exists()


# ── step 11 ───────────────────────────────────────────────────────


def test_prune_keeps_pinned_and_last_n_rolling(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger, rolling_window=100)

    order = [
        ("acc_001", False),
        ("acc_002", True),
        ("acc_003", False),
        ("acc_004", True),
        ("acc_005", False),
    ]
    for step_id, pinned in order:
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=pinned)
        ledger.mark_done(step_id)

    assert manager.list_checkpoints() == [
        "acc_001",
        "acc_002",
        "acc_003",
        "acc_004",
        "acc_005",
    ]

    removed = manager.prune(3)
    remaining = manager.list_checkpoints()

    # Both pinned survive; only the oldest rolling one is dropped, because
    # three rolling snapshots (acc_001/003/005) fit the window exactly.
    assert removed == []
    assert remaining == ["acc_001", "acc_002", "acc_003", "acc_004", "acc_005"]

    for step_id in ("acc_006", "acc_007"):
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=False)
        ledger.mark_done(step_id)
    manager.prune(3)
    remaining = manager.list_checkpoints()

    assert "acc_002" in remaining and "acc_004" in remaining
    rolling = [c for c in remaining if c not in {"acc_002", "acc_004"}]
    assert rolling == ["acc_005", "acc_006", "acc_007"]


def test_retained_checkpoint_survives_pruning_without_becoming_a_probe_pin(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = CheckpointManager(
        run_dir, ledger=ledger, memory_condition="nanobot_file_memory",
        rolling_window=2, retained_step_ids=frozenset({"evolv_01", "acc_010"}),
    )
    for number in range(1, 13):
        step_id = "evolv_01" if number == 5 else f"acc_{number:03d}"
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=number == 7)
        ledger.mark_done(step_id)
    assert set(manager.list_checkpoints()) == {
        "evolv_01", "acc_007", "acc_010", "acc_011", "acc_012",
    }
    for step_id in ("evolv_01", "acc_010"):
        snapshot = run_dir / "checkpoints" / step_id
        assert manager.metadata(step_id)["retained"] is True
        assert manager.metadata(step_id)["pinned"] is False
        assert (snapshot / "run_plan.yaml").read_bytes() == (run_dir / "run_plan.yaml").read_bytes()
        assert checkpoint_restorable(snapshot, memory_condition="nanobot_file_memory") == (True, [])
        with pytest.raises(ValueError, match="Only a pinned checkpoint"):
            manager.branch_from(step_id, tmp_path / "not_a_probe_branch")
    # A resumed manager with no retention selection still preserves old marks.
    _manager(run_dir, ledger).prune(1)
    assert set(manager.list_checkpoints()) == {"evolv_01", "acc_007", "acc_010", "acc_012"}
    (run_dir / "checkpoints" / "evolv_01" / "run_plan.yaml").unlink()
    assert checkpoint_restorable(
        run_dir / "checkpoints" / "evolv_01", memory_condition="nanobot_file_memory"
    ) == (False, ["missing layer: run_plan.yaml"])


def test_save_prunes_after_publishing(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger, rolling_window=2)

    for step_id, pinned in [
        ("acc_001", True),
        ("acc_002", False),
        ("acc_003", False),
        ("acc_004", False),
    ]:
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=pinned)
        ledger.mark_done(step_id)

    assert manager.list_checkpoints() == ["acc_001", "acc_003", "acc_004"]


# ── step 12 ───────────────────────────────────────────────────────


def test_restore_rolls_back_all_layers_and_ledger(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(1, 1, 1))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_001")
    manager = _manager(run_dir, ledger)
    manager.save("acc_001", pinned=False)
    ledger.mark_done("acc_001")

    pa_memory = run_dir / "pa_workspace" / PERSONA / "memory" / "MEMORY.md"
    sim_memory = (
        run_dir / "simulator_workspace" / PERSONA / "memory" / "MEMORY.md"
    )
    world = run_dir / "canonical_stage" / "email" / "inbox.json"
    audit = run_dir / "memory_io.jsonl"
    session_path = (
        run_dir / "pa_workspace" / PERSONA / "sessions" / f"{SESSION_KEY}.jsonl"
    )

    pa_memory.write_text("drifted\n", encoding="utf-8")
    sim_memory.write_text("drifted\n", encoding="utf-8")
    world.write_text('["later"]', encoding="utf-8")
    audit.write_text('{"seq": 1}\n{"seq": 99}\n', encoding="utf-8")
    _write_session(run_dir, cursors=(3, 3, 3))
    ledger.mark_running("acc_002")

    manager.restore("acc_001", ledger=ledger)

    assert pa_memory.read_text(encoding="utf-8") == "pa memory\n"
    assert sim_memory.read_text(encoding="utf-8") == "simulator memory\n"
    assert world.read_text(encoding="utf-8") == "[]"
    assert audit.read_text(encoding="utf-8") == '{"seq": 1}\n'
    assert _session_metadata_line(session_path)["file_consolidated_end_index"] == 1
    assert ledger.steps["acc_001"].status == "done"
    assert "acc_002" not in ledger.steps
    assert ledger.current_step is None
    assert any((run_dir / "crash_scene").iterdir())


def test_restore_refuses_an_incomplete_checkpoint_before_anything_moves(
    tmp_path: Path,
) -> None:
    """`save` refuses to publish a checkpoint missing a layer; so must restore.

    Codex review 2026-09-14, finding 3. A checkpoint whose `simulator_workspace`
    was lost after publication still carried its `metadata.json`, and restore
    read nothing else: it moved the live simulator workspace into the crash
    scene, put nothing back in its place, and installed the snapshot's ledger
    anyway — a cut point the ledger and the memory claim but the simulator
    never reached.
    """
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(1, 1, 1))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_001")
    manager = _manager(run_dir, ledger)
    manager.save("acc_001", pinned=False)
    ledger.mark_done("acc_001")

    shutil.rmtree(run_dir / "checkpoints" / "acc_001" / "simulator_workspace")

    sim_memory = (
        run_dir / "simulator_workspace" / PERSONA / "memory" / "MEMORY.md"
    )
    sim_memory.write_text("later\n", encoding="utf-8")
    ledger.mark_running("acc_002")

    with pytest.raises(IncompleteCheckpointError, match="simulator_workspace"):
        manager.restore("acc_001", ledger=ledger)

    # Nothing was touched: the live run is exactly as the refusal found it.
    assert sim_memory.read_text(encoding="utf-8") == "later\n"
    assert not (run_dir / "crash_scene").exists()
    assert ledger.steps["acc_002"].status == "running"


def test_prune_keeps_the_furthest_cut_point_when_the_clock_steps_back(
    tmp_path: Path,
) -> None:
    """Retention drops the oldest cut point, not the smallest clock reading.

    Codex review 2026-09-14, finding 4. Timestamps come from `datetime.now`, so
    a host clock that steps backwards between two saves made the newer cut
    point look like the oldest one — and with a full rolling window it was the
    one retention deleted.
    """
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger, rolling_window=2)
    for step_id in ("acc_001", "acc_002"):
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=False)
        ledger.mark_done(step_id)

    _set_checkpoint_timestamp(run_dir, "acc_002", "2000-01-01T00:00:00+00:00")

    ledger.mark_running("acc_003")
    manager.save("acc_003", pinned=False)
    ledger.mark_done("acc_003")

    # acc_001 is the one behind; acc_002 is a completed, paid session.
    assert manager.list_checkpoints() == ["acc_002", "acc_003"]


# ── step 13 ───────────────────────────────────────────────────────


def test_restore_moves_post_checkpoint_steps_to_crash_scene(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_done("acc_000")
    ledger.mark_running("acc_001")
    manager = _manager(run_dir, ledger)
    manager.save("acc_001", pinned=False)
    ledger.mark_done("acc_001")

    for step_id in ("acc_000", "acc_001", "acc_002"):
        step_dir = run_dir / "steps" / step_id
        step_dir.mkdir(parents=True)
        (step_dir / "transcript.jsonl").write_text(
            f'{{"step": "{step_id}"}}\n', encoding="utf-8"
        )

    ledger.mark_running("acc_002")
    crash_scene = manager.restore("acc_001", ledger=ledger)

    assert not (run_dir / "steps" / "acc_002").exists()
    assert (crash_scene / "steps" / "acc_002" / "transcript.jsonl").is_file()
    assert (run_dir / "steps" / "acc_000" / "transcript.jsonl").is_file()
    assert (run_dir / "steps" / "acc_001" / "transcript.jsonl").is_file()

    # A rerun of acc_002 appends into an empty directory.
    rerun = run_dir / "steps" / "acc_002"
    rerun.mkdir(parents=True)
    with (rerun / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"step": "acc_002_rerun"}\n')
    assert (rerun / "transcript.jsonl").read_text(
        encoding="utf-8"
    ) == '{"step": "acc_002_rerun"}\n'


def test_restore_clears_a_leftover_stage_work(tmp_path: Path) -> None:
    """A step that died inside the wrapper must not block the next checkpoint.

    `ix_contract_failure` is raised from the executor, so `finish_step` never
    runs and `stage_work/<step>` stays on disk — and `save()` refuses to
    publish while one is there. Restore moves it into the crash scene, so the
    save right after the resume goes through and the evidence is still kept.
    """
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_001")
    manager = _manager(run_dir, ledger)
    manager.save("acc_001", pinned=False)
    ledger.mark_done("acc_001")

    leftover = run_dir / "stage_work" / "acc_002" / "email"
    leftover.mkdir(parents=True)
    (leftover / "inbox.json").write_text('["mid-step"]', encoding="utf-8")
    ledger.mark_running("acc_002")

    crash_scene = manager.restore("acc_001", ledger=ledger)

    assert list((run_dir / "stage_work").iterdir()) == []
    assert (
        crash_scene / "stage_work" / "acc_002" / "email" / "inbox.json"
    ).read_text(encoding="utf-8") == '["mid-step"]'

    ledger.mark_running("acc_002")
    assert manager.save("acc_002", pinned=False).is_dir()


# ── steps 19 / 21 ─────────────────────────────────────────────────


def _pinned_manager(tmp_path: Path) -> tuple[Path, CheckpointManager]:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(3, 3, 3))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_108")
    manager = _manager(run_dir, ledger)
    manager.save("acc_108", pinned=True)
    ledger.mark_done("acc_108")
    return run_dir, manager


def _branch_session(branch_dir: Path) -> Path:
    return branch_dir / "pa_workspace" / PERSONA / "sessions" / f"{SESSION_KEY}.jsonl"


def test_branch_is_independent_of_source_and_mainline(tmp_path: Path) -> None:
    run_dir, manager = _pinned_manager(tmp_path)

    branch_a = manager.branch_from("acc_108", tmp_path / "branches" / "a")
    branch_b = manager.branch_from("acc_108", tmp_path / "branches" / "b")

    # Each branch continues the mainline session; no second session file.
    for branch in (branch_a, branch_b):
        sessions = sorted(
            (branch / "pa_workspace" / PERSONA / "sessions").glob("*.jsonl")
        )
        assert [path.name for path in sessions] == [f"{SESSION_KEY}.jsonl"]

    (branch_a / "pa_workspace" / PERSONA / "memory" / "MEMORY.md").write_text(
        "branch a memory\n", encoding="utf-8"
    )
    (branch_a / "canonical_stage" / "email" / "inbox.json").write_text(
        '["a"]', encoding="utf-8"
    )
    _branch_session(branch_a).write_text("clobbered\n", encoding="utf-8")

    assert (
        branch_b / "pa_workspace" / PERSONA / "memory" / "MEMORY.md"
    ).read_text(encoding="utf-8") == "pa memory\n"
    assert (branch_b / "canonical_stage" / "email" / "inbox.json").read_text(
        encoding="utf-8"
    ) == "[]"
    assert _session_metadata_line(_branch_session(branch_b))["key"] == SESSION_KEY

    # Neither the source snapshot nor the mainline run directory moved.
    snapshot = run_dir / "checkpoints" / "acc_108"
    assert (
        snapshot / "pa_workspace" / PERSONA / "memory" / "MEMORY.md"
    ).read_text(encoding="utf-8") == "pa memory\n"
    assert (run_dir / "canonical_stage" / "email" / "inbox.json").read_text(
        encoding="utf-8"
    ) == "[]"

    # Only a pinned snapshot may be branched from.
    manager.ledger.mark_running("acc_109")
    manager.save("acc_109", pinned=False)
    with pytest.raises(ValueError, match="Only a pinned checkpoint"):
        manager.branch_from("acc_109", tmp_path / "branches" / "c")


def test_branch_moves_prompt_start_and_keeps_rest(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    # A non-file backend leaves the prompt cursor behind, so the branch is the
    # one place it gets moved; the other two cursors must not follow it.
    _write_session(run_dir, cursors=(1, 2, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_108")
    manager = _manager(run_dir, ledger)
    manager.save("acc_108", pinned=True)

    branch = manager.branch_from("acc_108", tmp_path / "branches" / "a")
    metadata = _session_metadata_line(_branch_session(branch))
    snapshot_metadata = _session_metadata_line(
        run_dir
        / "checkpoints"
        / "acc_108"
        / "pa_workspace"
        / PERSONA
        / "sessions"
        / f"{SESSION_KEY}.jsonl"
    )

    assert metadata["prompt_history_start_index"] == 3
    assert metadata["file_consolidated_end_index"] == 1
    assert metadata["adapter_committed_end_index"] == 2
    assert {
        key: value
        for key, value in metadata.items()
        if key != "prompt_history_start_index"
    } == {
        key: value
        for key, value in snapshot_metadata.items()
        if key != "prompt_history_start_index"
    }
    assert SessionManager(
        branch / "pa_workspace" / PERSONA
    ).get(SESSION_KEY).messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]


# ── steps 30-38: external backends ────────────────────────────────


class FakeQdrant:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.exists = False
        #: Snapshot names the server holds, per collection. A restore confirms
        #: its snapshot is here before it drops the live collection, so this
        #: has to answer like the real endpoint: emptying it is how a test
        #: says "the snapshot is gone from the server".
        self.snapshots: dict[str, list[str]] = {}
        #: Raised by `list_snapshots` to fake an unreachable server.
        self.unreachable: Exception | None = None

    def collection_exists(self, collection_name):
        return self.exists

    def create_snapshot(self, *, collection_name: str):
        self.calls.append(("create_snapshot", collection_name))
        self.snapshots.setdefault(collection_name, []).append("snap-1")
        return {"name": "snap-1"}

    def list_snapshots(self, *, collection_name: str) -> list:
        if self.unreachable is not None:
            raise self.unreachable
        return [
            {"name": name}
            for name in self.snapshots.get(collection_name, [])
        ]

    def delete_collection(self, *, collection_name: str) -> None:
        self.calls.append(("delete_collection", collection_name))

    def recover_snapshot(self, *, collection_name: str, location: str) -> None:
        self.calls.append(("recover_snapshot", collection_name, location))


class FakeMem0Client:
    def __init__(self) -> None:
        self.qdrant_admin = FakeQdrant()


class RetentionQdrant(FakeQdrant):
    """Distinct snapshots, with deletion failures at the real API boundary."""

    def __init__(self):
        super().__init__()
        self.exists = True
        self.serial = 0
        self.fail_delete = None

    def create_snapshot(self, *, collection_name):
        self.serial += 1
        name = f"snap-{self.serial}"
        self.snapshots.setdefault(collection_name, []).append(name)
        return {"name": name}

    def delete_snapshot(self, *, collection_name, snapshot_name, wait=True):
        if self.fail_delete == collection_name:
            raise RuntimeError("snapshot delete interrupted")
        self.calls.append(("delete_snapshot", collection_name, snapshot_name))
        self.snapshots[collection_name].remove(snapshot_name)


def _mem0_retention_run(tmp_path):
    run_dir, ledger, client = _mem0_run(tmp_path)
    client.qdrant_admin = RetentionQdrant()
    manager = _external_manager(run_dir, ledger, condition="mem0", namespace="owned", client=client)
    for number in range(1, 7):
        step_id = f"acc_{number:03}"
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=number == 2)
        ledger.mark_done(step_id)
    _restamp(manager, "acc_003", retained=True)
    return run_dir, ledger, client, manager


def test_prune_removes_only_expired_mem0_and_entity_snapshots(tmp_path):
    _run_dir, _ledger, client, manager = _mem0_retention_run(tmp_path)
    client.qdrant_admin.snapshots["unrelated"] = ["keep-me"]
    assert manager.prune(2) == ["acc_001", "acc_004"]
    assert manager.list_checkpoints() == ["acc_002", "acc_003", "acc_005", "acc_006"]
    assert client.qdrant_admin.snapshots == {
        "owned": ["snap-3", "snap-5", "snap-9", "snap-11"],
        "owned_entities": ["snap-4", "snap-6", "snap-10", "snap-12"],
        "unrelated": ["keep-me"],
    }


def test_simple_rag_prunes_remote_snapshots_without_an_entity_index(tmp_path):
    run_dir, ledger, client = _mem0_run(tmp_path)
    client.qdrant_admin = RetentionQdrant()
    manager = _external_manager(run_dir, ledger, condition="simple_rag", namespace="owned", client=client)
    for number in range(1, 5):
        step_id = f"acc_{number:03}"
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=False)
        ledger.mark_done(step_id)
    assert manager.prune(1) == ["acc_001", "acc_002", "acc_003"]
    assert client.qdrant_admin.snapshots == {"owned": ["snap-4"]}


def test_prune_preserves_refs_shared_with_live_or_archived_checkpoints(tmp_path):
    run_dir, _ledger, client, manager = _mem0_retention_run(tmp_path)
    archive = run_dir / "checkpoint_archive/recovery_001/acc_001"
    shutil.copytree(run_dir / "checkpoints/acc_001", archive)
    shared = manager.metadata("acc_004")
    _restamp(manager, "acc_006", **{
        key: shared[key] for key in ("mem0_snapshot_ref", "mem0_entity_snapshot_ref")
    })
    before = {key: list(value) for key, value in client.qdrant_admin.snapshots.items()}
    manager.prune(2)
    assert client.qdrant_admin.snapshots == before


def test_interrupted_remote_prune_is_hidden_and_retried_by_new_manager(tmp_path):
    run_dir, ledger, client, manager = _mem0_retention_run(tmp_path)
    client.qdrant_admin.fail_delete = "owned_entities"
    with pytest.raises(RuntimeError, match="snapshot delete interrupted"):
        manager.prune(2)
    assert "acc_001" not in manager.list_checkpoints()
    assert "snap-1" not in client.qdrant_admin.snapshots["owned"]
    assert list((run_dir / "checkpoints").glob(".staging-prune-*/metadata.json"))
    client.qdrant_admin.fail_delete = None
    resumed = _external_manager(run_dir, ledger, condition="mem0", namespace="owned", client=client)
    resumed.prune(2)
    assert client.qdrant_admin.snapshots["owned_entities"] == ["snap-4", "snap-6", "snap-10", "snap-12"]
    assert not list((run_dir / "checkpoints").glob(".staging-prune-*"))


@pytest.mark.parametrize("bad_ref", ["foreign/snap-2", "owned_entities/../snap-2", "https://host/snap-2"])
def test_remote_prune_refuses_foreign_or_malformed_snapshot_refs(tmp_path, bad_ref):
    _run_dir, _ledger, client, manager = _mem0_retention_run(tmp_path)
    _restamp(manager, "acc_001", mem0_entity_snapshot_ref=bad_ref)
    before = {key: list(value) for key, value in client.qdrant_admin.snapshots.items()}
    with pytest.raises(IncompleteCheckpointError, match="snapshot reference"):
        manager.prune(2)
    assert client.qdrant_admin.snapshots == before


MEMOS_NATIVE_IDS = (
    "3a6d8fa0-2248-4f48-9bfe-0f20d497cf0c",
    "9c1f4b27-5d8e-4a10-8f33-7b2c6e0d41aa",
)


def _memos_snapshot(cube_id: str) -> dict:
    """One valid MemOS cube export, owned by `cube_id`.

    The shape `memos.mempa_checkpoint.validate_snapshot` accepts, carrying the
    same id-bearing surface a real export has — node `properties.id`, node
    `properties.working_binding`, point `id`, point `payload.working_binding`
    and both endpoints of an edge, all four scoped by `user_name` — verified
    against the pinned 72-node smoke snapshot. The cube id is a parameter
    because the branch path's whole question is which cube id it restores under.
    """
    from neo4j.time import DateTime
    from memos.mempa_checkpoint import _pack

    created_at = _pack(
        DateTime.from_iso_format("2026-09-06T12:00:00.123456789+00:00")
    )
    return {
        "schema_version": 1,
        "storage": "neo4j-community-qdrant",
        "cube_id": cube_id,
        "nodes": [
            {
                "labels": ["Memory"],
                "properties": {
                    "id": native_id,
                    "user_name": cube_id,
                    "user_id": "user_a",
                    "working_binding": native_id,
                    "evolve_to": [],
                    "memory": memory,
                    "created_at": created_at,
                },
            }
            for native_id, memory in zip(
                MEMOS_NATIVE_IDS, ("Keep answers concise", "Cite page numbers")
            )
        ],
        "edges": [
            {
                "source": MEMOS_NATIVE_IDS[0],
                "target": MEMOS_NATIVE_IDS[1],
                "type": "FOLLOWS",
                "properties": {},
            }
        ],
        "points": [
            {
                "id": native_id,
                "vector": [0.1, 0.2, 0.3],
                "payload": {"user_name": cube_id, "working_binding": native_id},
            }
            for native_id in MEMOS_NATIVE_IDS
        ],
    }


class FakeMemosClient:
    """A MemOS client that validates an import the way the service does.

    The service runs every snapshot through
    `memos.mempa_checkpoint.validate_snapshot` before it mutates anything, so a
    fake that accepts any payload can go green exactly where the deployment
    refuses — which is how the cross-cube branch defect reached a live run.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def export_cube(self, cube_id: str) -> dict:
        self.calls.append(("export", cube_id))
        return _memos_snapshot(cube_id)

    def delete_cube(self, cube_id: str) -> dict:
        self.calls.append(("delete", cube_id))
        return {"status": "success"}

    def import_cube_as(self, payload: dict, target_cube_id: str) -> dict:
        from memos.mempa_checkpoint import validate_snapshot

        validate_snapshot(payload, target_cube_id, 3)
        self.calls.append(("import", payload["cube_id"], target_cube_id))
        return {
            "status": "success",
            "node_count": len(payload["nodes"]),
            "point_count": len(payload["points"]),
        }


def _external_manager(
    run_dir: Path,
    ledger: RunLedger,
    *,
    condition: str,
    namespace: str,
    client: object,
) -> CheckpointManager:
    return CheckpointManager(
        run_dir,
        ledger=ledger,
        memory_condition=condition,
        memory_backend_namespace=namespace,
        backend_client=client,
        rolling_window=100,
    )


def _mem0_run(tmp_path: Path) -> tuple[Path, RunLedger, FakeMem0Client]:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(3, 3, 3))
    history = run_dir / "backend_state" / "mem0_history.sqlite3"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_bytes(b"history-v1")
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_108")
    return run_dir, ledger, FakeMem0Client()


# ── step 30 ───────────────────────────────────────────────────────


def test_mem0_save_records_snapshot_ref(tmp_path: Path) -> None:
    run_dir, ledger, client = _mem0_run(tmp_path)
    manager = _external_manager(
        run_dir, ledger, condition="mem0", namespace="mempa_run_001", client=client
    )

    cp_dir = manager.save("acc_108", pinned=True)

    metadata = manager.metadata("acc_108")
    assert metadata["mem0_collection_name"] == "mempa_run_001"
    assert metadata["mem0_snapshot_ref"] == "mempa_run_001/snap-1"
    assert (
        cp_dir / "backend_state" / "mem0_history.sqlite3"
    ).read_bytes() == b"history-v1"
    assert client.qdrant_admin.calls == [("create_snapshot", "mempa_run_001")]

    # Without the local history database the snapshot would be partial.
    (run_dir / "backend_state" / "mem0_history.sqlite3").unlink()
    with pytest.raises(IncompleteCheckpointError):
        manager.save("acc_109", pinned=False)
    assert not (run_dir / "checkpoints" / "acc_109").exists()


# ── step 31 ───────────────────────────────────────────────────────


def test_mem0_restore_deletes_then_reloads_same_name(tmp_path: Path) -> None:
    run_dir, ledger, client = _mem0_run(tmp_path)
    manager = _external_manager(
        run_dir, ledger, condition="mem0", namespace="mempa_run_001", client=client
    )
    manager.save("acc_108", pinned=True)
    client.qdrant_admin.calls.clear()
    (run_dir / "backend_state" / "mem0_history.sqlite3").write_bytes(b"history-v2")

    manager.restore("acc_108", ledger=ledger)

    assert client.qdrant_admin.calls == [
        ("delete_collection", "mempa_run_001"),
        (
            "recover_snapshot",
            "mempa_run_001",
            "file:///qdrant/snapshots/mempa_run_001/snap-1",
        ),
    ]
    assert (
        run_dir / "backend_state" / "mem0_history.sqlite3"
    ).read_bytes() == b"history-v1"

    # A checkpoint that does not name its snapshot must not drop anything.
    metadata_path = run_dir / "checkpoints" / "acc_108" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["mem0_snapshot_ref"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    client.qdrant_admin.calls.clear()
    with pytest.raises(IncompleteCheckpointError):
        manager.restore("acc_108", ledger=ledger)
    assert client.qdrant_admin.calls == []


# ── step 32 ───────────────────────────────────────────────────────


def test_mem0_branch_binds_new_namespace(tmp_path: Path) -> None:
    run_dir, ledger, client = _mem0_run(tmp_path)
    manager = _external_manager(
        run_dir, ledger, condition="mem0", namespace="mempa_run_001", client=client
    )
    manager.save("acc_108", pinned=True)
    client.qdrant_admin.calls.clear()

    branch_dir = run_dir / "branches" / "fin_01" / "01"
    manager.branch_from("acc_108", branch_dir)

    record = json.loads(
        (branch_dir / "backend_state" / "branch_namespace.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["namespace"] == "mempa_run_001_br_fin_01_01"
    assert record["namespace"] != "mempa_run_001"
    assert client.qdrant_admin.calls == [
        (
            "recover_snapshot",
            "mempa_run_001_br_fin_01_01",
            "file:///qdrant/snapshots/mempa_run_001/snap-1",
        )
    ]
    # The snapshot is read, never consumed: a second branch reuses it.
    manager.branch_from("acc_108", run_dir / "branches" / "fin_01" / "02")
    assert (
        run_dir / "checkpoints" / "acc_108" / "backend_state" / "mem0_history.sqlite3"
    ).is_file()
    # The branch works on a copy of the history database, not the mainline one.
    branch_history = branch_dir / "backend_state" / "mem0_history.sqlite3"
    assert branch_history.is_file()
    assert branch_history != run_dir / "backend_state" / "mem0_history.sqlite3"

    # Discarding the branch drops the collection it restored.
    assert manager.release_branch_namespace(branch_dir) == (
        "mempa_run_001_br_fin_01_01"
    )
    assert (
        "delete_collection",
        "mempa_run_001_br_fin_01_01",
    ) in client.qdrant_admin.calls


# ── step 37 ───────────────────────────────────────────────────────


def test_memos_restore_delegates_validated_replacement_to_service(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(3, 3, 3))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_108")
    client = FakeMemosClient()
    manager = _external_manager(
        run_dir, ledger, condition="memos", namespace="cube_001", client=client
    )
    cp_dir = manager.save("acc_108", pinned=True)
    assert client.calls == [("export", "cube_001")]
    assert (cp_dir / "backend_state" / "memos_cube.json").is_file()
    client.calls.clear()

    manager.restore("acc_108", ledger=ledger)

    assert client.calls == [
        ("import", "cube_001", "cube_001"),
    ]

    metadata_path = cp_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["memos_cube_export_path"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    client.calls.clear()
    with pytest.raises(IncompleteCheckpointError):
        manager.restore("acc_108", ledger=ledger)
    assert client.calls == []


def test_memos_restore_refuses_a_half_restored_cube(tmp_path: Path) -> None:
    """A cube that came back short must stop the run, not continue."""
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(3, 3, 3))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_108")
    client = FakeMemosClient()
    manager = _external_manager(
        run_dir, ledger, condition="memos", namespace="cube_001", client=client
    )
    manager.save("acc_108", pinned=True)
    client.import_cube_as = lambda payload, target_cube_id: {
        "status": "success",
        "node_count": len(payload["nodes"]),
        "point_count": len(payload["points"]) - 1,
    }

    with pytest.raises(IncompleteCheckpointError, match="number of vectors"):
        manager.restore("acc_108", ledger=ledger)


# ── step 38 ───────────────────────────────────────────────────────


def test_memos_branch_binds_new_cube(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(3, 3, 3))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_108")
    client = FakeMemosClient()
    manager = _external_manager(
        run_dir, ledger, condition="memos", namespace="cube_001", client=client
    )
    manager.save("acc_108", pinned=True)
    client.calls.clear()

    branch_dir = run_dir / "branches" / "fin_01" / "01"
    manager.branch_from("acc_108", branch_dir)

    # The payload's own cube id is the branch's, not the mainline's: the
    # service validates the two against each other before it writes anything.
    assert client.calls == [
        ("import", "cube_001_br_fin_01_01", "cube_001_br_fin_01_01")
    ]
    assert not any(call[0] == "delete" for call in client.calls)

    assert manager.release_branch_namespace(branch_dir) == "cube_001_br_fin_01_01"
    assert ("delete", "cube_001_br_fin_01_01") in client.calls


# ── step 27 ───────────────────────────────────────────────────────


def _complete_checkpoint_dir(tmp_path: Path, metadata: dict) -> Path:
    cp_dir = tmp_path / "cp"
    for layer in ("pa_workspace", "canonical_stage", "simulator_workspace"):
        (cp_dir / layer).mkdir(parents=True)
    (cp_dir / "ledger.json").write_text('{"steps": {}}', encoding="utf-8")
    (cp_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return cp_dir


def test_checkpoint_restorable_requires_layers_capability_and_external_ref(
    tmp_path: Path,
) -> None:
    local = _complete_checkpoint_dir(
        tmp_path / "local",
        {"checkpoint_id": "acc_001", "pinned": False},
    )
    assert checkpoint_restorable(
        local, memory_condition="nanobot_file_memory"
    ) == (True, [])

    # A layer that is gone makes the whole snapshot unusable.
    shutil.rmtree(local / "canonical_stage")
    ok, reasons = checkpoint_restorable(
        local, memory_condition="nanobot_file_memory"
    )
    assert ok is False
    assert any("canonical_stage" in reason for reason in reasons)

    # Layers complete, but the condition has no restore path of its own: one
    # absent from the capability table keeps the default of no support.
    unlisted = _complete_checkpoint_dir(
        tmp_path / "unlisted",
        {"checkpoint_id": "acc_001", "pinned": False},
    )
    ok, reasons = checkpoint_restorable(unlisted, memory_condition="unlisted_backend")
    assert ok is False
    assert any("no restore support" in reason for reason in reasons)

    # An external snapshot the metadata names but that is no longer on disk.
    memos = _complete_checkpoint_dir(
        tmp_path / "memos",
        {
            "checkpoint_id": "acc_001",
            "pinned": False,
            "memos_cube_id": "cube_001",
            "memos_cube_export_path": "backend_state/memos_cube.json",
        },
    )
    export = memos / "backend_state" / "memos_cube.json"
    export.parent.mkdir(parents=True)
    export.write_text("{}", encoding="utf-8")
    assert checkpoint_restorable(memos, memory_condition="memos") == (True, [])

    export.unlink()
    ok, reasons = checkpoint_restorable(memos, memory_condition="memos")
    assert ok is False
    assert any("backend_state/memos_cube.json" in reason for reason in reasons)

    # A checkpoint that never names its external snapshot at all.
    ok, reasons = checkpoint_restorable(local, memory_condition="memos")
    assert ok is False
    assert any("memos_cube_export_path" in reason for reason in reasons)

    # No metadata at all: a staging directory is never restorable.
    assert checkpoint_restorable(
        tmp_path / "nowhere", memory_condition="no_memory"
    )[0] is False


def _run_dir_with_checkpoint(
    run_dir: Path,
    *,
    plan: dict | None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if plan is not None:
        (run_dir / "run_plan.yaml").write_text(
            yaml.safe_dump(plan, sort_keys=False), encoding="utf-8"
        )
    cp_dir = run_dir / "checkpoints" / "acc_001"
    for layer in ("pa_workspace", "canonical_stage", "simulator_workspace"):
        (cp_dir / layer).mkdir(parents=True)
    (cp_dir / "ledger.json").write_text('{"steps": {}}', encoding="utf-8")
    (cp_dir / "metadata.json").write_text(
        json.dumps({"checkpoint_id": "acc_001", "pinned": False}),
        encoding="utf-8",
    )
    return run_dir


def test_run_restorable_folds_the_plan_contract_into_the_snapshot_verdict(
    tmp_path: Path,
) -> None:
    frozen_now = {
        "run_id": "run_ok",
        "ix_contract_version": IX_CONTRACT_VERSION,
        "memory_pool_version": MEMORY_POOL_VERSION,
    }

    # A complete snapshot under the running wrapper: resumable.
    good = _run_dir_with_checkpoint(tmp_path / "good", plan=frozen_now)
    assert run_restorable(
        good, memory_condition="nanobot_file_memory"
    ) == (True, [])

    # A run with a sound plan but nothing to roll back to.
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "run_plan.yaml").write_text(
        yaml.safe_dump(frozen_now, sort_keys=False), encoding="utf-8"
    )
    ok, reasons = run_restorable(empty, memory_condition="nanobot_file_memory")
    assert ok is False
    assert reasons == ["no checkpoint to roll back to"]

    # A run whose plan is not on disk at all.
    gone = _run_dir_with_checkpoint(tmp_path / "gone", plan=None)
    ok, reasons = run_restorable(gone, memory_condition="nanobot_file_memory")
    assert ok is False
    assert any("unreadable run_plan.yaml" in reason for reason in reasons)


# ── §7 #51 · IX contract version boundary ─────────────────────────


def test_save_stamps_the_running_ix_wrapper(tmp_path: Path) -> None:
    _run_dir, manager = _pinned_manager(tmp_path)

    metadata = manager.metadata("acc_108")
    assert metadata["ix_contract_version"] == IX_CONTRACT_VERSION
    assert metadata["ix_render_schema_version"] == IX_RENDER_SCHEMA_VERSION


def _restamp(manager: CheckpointManager, checkpoint_id: str, **changes) -> None:
    path = manager._checkpoint_path(checkpoint_id) / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize("entity_exists", [True, False])
def test_mem0_checkpoint_restores_entity_index_or_its_absence(tmp_path, entity_exists):
    run_dir, ledger, client = _mem0_run(tmp_path)
    client._enable_entity_memory = True
    client.qdrant_admin.exists = entity_exists
    manager = _external_manager(run_dir, ledger, condition="mem0", namespace="owned", client=client)
    manager.save("acc_108", pinned=True)
    metadata = manager.metadata("acc_108")
    assert metadata["mem0_entity_collection_name"] == "owned_entities"
    assert metadata["mem0_entity_snapshot_ref"] == ("owned_entities/snap-1" if entity_exists else None)
    client.qdrant_admin.calls.clear()
    client.qdrant_admin.exists = True  # An index may have appeared after the cut.
    manager.restore("acc_108", ledger=ledger)
    calls = client.qdrant_admin.calls
    assert ("delete_collection", "owned_entities") in calls
    restored = [call for call in calls if call[0] == "recover_snapshot" and call[1] == "owned_entities"]
    assert bool(restored) == entity_exists


def test_mem0_restore_lease_covers_primary_and_entity_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir, ledger, client = _mem0_run(tmp_path)
    client._enable_entity_memory = True
    client.qdrant_admin.exists = True
    manager = _external_manager(
        run_dir,
        ledger,
        condition="mem0",
        namespace="owned",
        client=client,
    )
    manager.save("acc_108", pinned=True)
    events: list[tuple] = []

    @contextmanager
    def lease():
        events.append(("lease_enter",))
        try:
            yield Path("/shared/qdrant-restore.lock")
        finally:
            events.append(("lease_exit",))

    admin = client.qdrant_admin
    original_delete = admin.delete_collection
    original_recover = admin.recover_snapshot

    def delete_collection(*, collection_name: str) -> None:
        events.append(("delete", collection_name))
        original_delete(collection_name=collection_name)

    def recover_snapshot(*, collection_name: str, location: str) -> None:
        events.append(("recover", collection_name))
        original_recover(collection_name=collection_name, location=location)

    monkeypatch.setattr(
        "memory.backends.qdrant_runtime.qdrant_restore_lease",
        lease,
    )
    monkeypatch.setattr(admin, "delete_collection", delete_collection)
    monkeypatch.setattr(admin, "recover_snapshot", recover_snapshot)

    manager.restore("acc_108", ledger=ledger)
    assert events == [
        ("lease_enter",),
        ("delete", "owned"),
        ("recover", "owned"),
        ("delete", "owned_entities"),
        ("recover", "owned_entities"),
        ("lease_exit",),
    ]

    events.clear()
    manager.branch_from("acc_108", run_dir / "branches" / "fin_01" / "01")
    assert events == [
        ("lease_enter",),
        ("recover", "owned_br_fin_01_01"),
        ("recover", "owned_br_fin_01_01_entities"),
        ("lease_exit",),
    ]


def test_mem0_restore_missing_service_lock_fails_before_backend_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from memory.backends.qdrant_runtime import QDRANT_RESTORE_LOCK_PATH_ENV

    run_dir, ledger, client = _mem0_run(tmp_path)
    manager = _external_manager(
        run_dir,
        ledger,
        condition="mem0",
        namespace="owned",
        client=client,
    )
    manager.save("acc_108", pinned=True)
    client.qdrant_admin.calls.clear()
    monkeypatch.delenv(QDRANT_RESTORE_LOCK_PATH_ENV)

    with pytest.raises(RuntimeError, match="shared service lock path"):
        manager.restore("acc_108", ledger=ledger)

    assert client.qdrant_admin.calls == []


def test_simple_rag_restore_uses_shared_service_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir, ledger, client = _mem0_run(tmp_path)
    manager = _external_manager(
        run_dir,
        ledger,
        condition="simple_rag",
        namespace="owned",
        client=client,
    )
    manager.save("acc_108", pinned=True)
    events: list[str] = []

    @contextmanager
    def lease():
        events.append("enter")
        try:
            yield Path("/shared/qdrant-restore.lock")
        finally:
            events.append("exit")

    monkeypatch.setattr(
        "memory.backends.qdrant_runtime.qdrant_restore_lease",
        lease,
    )

    manager.restore("acc_108", ledger=ledger)

    assert events == ["enter", "exit"]
    assert client.qdrant_admin.calls[-2:] == [
        ("delete_collection", "owned"),
        (
            "recover_snapshot",
            "owned",
            "file:///qdrant/snapshots/owned/snap-1",
        ),
    ]


def test_mem0_incomplete_entity_checkpoint_cannot_delete_live_data(tmp_path):
    run_dir, ledger, client = _mem0_run(tmp_path)
    client._enable_entity_memory = True
    manager = _external_manager(run_dir, ledger, condition="mem0", namespace="owned", client=client)
    cp = manager.save("acc_108", pinned=True)
    metadata = manager.metadata("acc_108")
    del metadata["mem0_entity_snapshot_ref"]
    (cp / "metadata.json").write_text(json.dumps(metadata))
    client.qdrant_admin.calls.clear()
    with pytest.raises(IncompleteCheckpointError, match="entity index"):
        manager.restore("acc_108", ledger=ledger)
    assert client.qdrant_admin.calls == []


def test_mem0_branch_copies_and_releases_its_own_entity_index(tmp_path):
    run_dir, ledger, client = _mem0_run(tmp_path)
    client.qdrant_admin.exists = True
    client._enable_entity_memory = True
    manager = _external_manager(run_dir, ledger, condition="mem0", namespace="owned", client=client)
    manager.save("acc_108", pinned=True)
    client.qdrant_admin.calls.clear()
    branch = run_dir / "branches" / "fin_01" / "01"
    manager.branch_from("acc_108", branch)
    assert ("recover_snapshot", "owned_br_fin_01_01_entities", "file:///qdrant/snapshots/owned_entities/snap-1") in client.qdrant_admin.calls
    manager.release_branch_namespace(branch)
    assert ("delete_collection", "owned_br_fin_01_01_entities") in client.qdrant_admin.calls
    assert ("delete_collection", "owned_entities") not in client.qdrant_admin.calls


# ── memos branch vs the service's own snapshot validation ─────────


def test_memos_branch_survives_the_services_own_snapshot_validation(
    tmp_path: Path,
) -> None:
    """The live smoke's blocker, reproduced hermetically, now green.

    `branch_from` restores the pinned export under a NEW branch cube id and the
    MemOS Community service refuses a cross-cube restore outright
    (`memory_raw_repos/MemOS/src/memos/mempa_checkpoint.py:51`), so no probe
    could run on that condition at all. `FakeMemosClient.import_cube_as` runs
    the service's own `validate_snapshot`, so this passes only while the
    payload the harness sends is one the deployment would accept.
    """
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(3, 3, 3))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("acc_108")
    client = FakeMemosClient()
    manager = _external_manager(
        run_dir, ledger, condition="memos", namespace="cube_001", client=client
    )
    manager.save("acc_108", pinned=True)

    branch_dir = run_dir / "branches" / "fin_01" / "01"
    manager.branch_from("acc_108", branch_dir)


def test_memos_rebind_moves_every_native_id_off_the_mainline() -> None:
    payload = _memos_snapshot("cube_001")
    before = json.dumps(payload, sort_keys=True)

    branched = rebind_memos_cube_to_branch(payload, "cube_001_br", "cube_001")

    assert branched["cube_id"] == "cube_001_br"
    assert (
        len(branched["nodes"]),
        len(branched["edges"]),
        len(branched["points"]),
    ) == (len(payload["nodes"]), len(payload["edges"]), len(payload["points"]))
    new_ids = {node["properties"]["id"] for node in branched["nodes"]}
    assert len(new_ids) == len(branched["nodes"])
    assert new_ids.isdisjoint(MEMOS_NATIVE_IDS)
    # Every id-bearing field, not just the primary key: a stale mainline id in
    # `working_binding` is a live cross-cube reference, because MemOS deletes by
    # it (`memos/mem_feedback/feedback.py:411-416`).
    assert {node["properties"]["working_binding"] for node in branched["nodes"]} == new_ids
    assert {point["id"] for point in branched["points"]} == new_ids
    assert {
        point["payload"]["working_binding"] for point in branched["points"]
    } == new_ids
    assert all(
        edge["source"] in new_ids and edge["target"] in new_ids
        for edge in branched["edges"]
    )
    assert all(
        node["properties"]["user_name"] == "cube_001_br"
        for node in branched["nodes"]
    )
    assert all(
        point["payload"]["user_name"] == "cube_001_br"
        for point in branched["points"]
    )
    # No mainline id survives anywhere in the payload, and the snapshot the
    # branch was cut from is untouched, because it is read once per branch.
    dumped = json.dumps(branched)
    assert not any(native_id in dumped for native_id in MEMOS_NATIVE_IDS)
    assert json.dumps(payload, sort_keys=True) == before

    from memos.mempa_checkpoint import validate_snapshot

    validate_snapshot(branched, "cube_001_br", 3)


def test_memos_rebind_ids_are_deterministic_per_branch() -> None:
    payload = _memos_snapshot("cube_001")

    first = rebind_memos_cube_to_branch(payload, "cube_001_br_fin_01_01", "cube_001")
    again = rebind_memos_cube_to_branch(payload, "cube_001_br_fin_01_01", "cube_001")
    other = rebind_memos_cube_to_branch(payload, "cube_001_br_fin_01_02", "cube_001")

    # Re-branching the same directory reproduces the same cube exactly, so a
    # resumed probe rewrites its own rows instead of doubling them.
    assert first == again
    # A different branch gets a different id space, or two probes of the same
    # cut point would collide in the one shared database.
    assert {node["properties"]["id"] for node in first["nodes"]}.isdisjoint(
        node["properties"]["id"] for node in other["nodes"]
    )


def test_memos_rebind_refuses_to_target_the_mainline_cube() -> None:
    payload = _memos_snapshot("cube_001")

    # `import_cube` clears the target cube before it writes
    # (`memos/mempa_checkpoint.py:152`), so a branch restore aimed at the
    # mainline would delete the run's own memory.
    with pytest.raises(ValueError, match="never the mainline"):
        rebind_memos_cube_to_branch(payload, "cube_001", "cube_001")
    with pytest.raises(ValueError, match="never the mainline"):
        rebind_memos_cube_to_branch(payload, "other_cube", "other_cube")


def test_snapshot_ledger_carries_the_done_step_summary(tmp_path: Path) -> None:
    """A rollback onto a forced event's own cut point must keep its flip.

    `mark_done` runs after the checkpoint is published, so at save time the
    step is still `running` and its summary — for a forced event, the
    `preference_overlay` flip the simulator's later steps are rebuilt from —
    exists nowhere on disk. Restoring that snapshot replaces the mainline
    ledger with it, and the flip is gone: live on
    `ub_nanobot_file_memory_deepseek_v4_flash_20260918`, the resume after the
    rollback onto `evolv_01` died at `acc_017` with the overlay empty.
    """
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ledger.mark_running("evolv_01")
    manager = _manager(run_dir, ledger)

    summary = {"beats": 7, "preference_overlay": {"work_tone_formality": "Casual"}}
    rolling = manager.save(
        "evolv_01", pinned=False, summary=summary, summary_schema=1
    )

    snapshot = json.loads((rolling / "ledger.json").read_text(encoding="utf-8"))
    record = snapshot["steps"]["evolv_01"]
    assert record["status"] == "done"
    assert record["summary"] == summary
    assert record["summary_schema"] == 1
    # The mainline is still untouched, exactly as before.
    mainline = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))
    assert mainline["steps"]["evolv_01"]["status"] == "running"
    assert "summary" not in mainline["steps"]["evolv_01"]


# ── rollback recovery (checkpoint_rollback_recovery_plan_20260918) ─


def _history(
    run_dir: Path, ledger: RunLedger, manager: CheckpointManager, order: list
) -> None:
    """Publish a run's cut points in plan order, as a real run does."""
    for step_id, pinned in order:
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=pinned)
        ledger.mark_done(step_id)


def test_restore_archives_the_candidates_the_rollback_revokes(
    tmp_path: Path,
) -> None:
    """After a rollback, a later snapshot is no longer part of the history.

    The v4.1 line: restoring `acc_017` left the old `acc_018` on disk, and
    `progress_key` reads its done count as the larger one, so the next
    automatic recovery picked the snapshot the rollback had just revoked
    (hy4, 2026-09-18).
    """
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger)
    _history(
        run_dir,
        ledger,
        manager,
        [
            ("acc_016", True),
            ("acc_017", True),
            ("evolv_01", False),
            ("acc_018", True),
        ],
    )

    manager.restore("acc_017", ledger=ledger)

    # Only the target and the history behind it are still candidates. The
    # pinned `acc_018` is archived too: `pinned` protects a snapshot from
    # retention, it does not make it part of a history that was rolled back.
    assert manager.list_checkpoints() == ["acc_016", "acc_017"]
    archive = run_dir / "checkpoint_archive"
    archived = sorted(
        path.name
        for recovery in archive.iterdir()
        for path in recovery.iterdir()
    )
    assert archived == ["acc_018", "evolv_01"]
    # Archived, not deleted: the evidence is still readable.
    moved = next(archive.iterdir()) / "acc_018"
    assert json.loads((moved / "metadata.json").read_text(encoding="utf-8"))[
        "checkpoint_id"
    ] == "acc_018"
    # And the recovery is finished, so the run may continue.
    assert manager.pending_recovery() is None


def test_automatic_recovery_after_a_rollback_reselects_the_target(
    tmp_path: Path,
) -> None:
    """The rerun fails again: the target wins, not the revoked snapshot."""
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger)
    _history(
        run_dir, ledger, manager, [("acc_017", True), ("acc_018", False)]
    )
    manager.restore("acc_017", ledger=ledger)

    # The rerun of acc_018 dies before it publishes anything.
    ledger.mark_running("acc_018")
    ledger.mark_failed("acc_018", "died again")

    assert max(manager.list_checkpoints(), key=manager.progress_key) == "acc_017"


def _revise_snapshot_ledger(
    run_dir: Path, checkpoint_id: str, *, add_done: list[str]
) -> None:
    """Give one snapshot a history that diverged from the mainline's.

    What a rerun after an earlier rollback leaves behind: the directory keeps
    its name, and its own ledger now records steps the target's ledger has
    never heard of.
    """
    path = run_dir / "checkpoints" / checkpoint_id / LEDGER_LAYER
    data = json.loads(path.read_text(encoding="utf-8"))
    for step_id in add_done:
        data["steps"][step_id] = {"step_id": step_id, "status": "done"}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_revoked_candidates_are_judged_by_history_not_clock_or_name(
    tmp_path: Path,
) -> None:
    """Three ways of guessing that the done set does not need.

    A backwards host clock, a gap in the session numbering, and a snapshot
    whose done *count* matches the target's while its history does not: none
    of them is decidable from a timestamp or from a name, and the target's own
    done set decides all three.
    """
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger)
    # acc_018 is missing from this plan: `--acc-list` has gaps, so the snapshot
    # after acc_017 is acc_019 and nothing named acc_018 ever exists.
    _history(
        run_dir,
        ledger,
        manager,
        [("acc_016", True), ("acc_017", True), ("acc_019", False)],
    )
    # The revoked snapshot reads as the *oldest* by wall clock.
    _set_checkpoint_timestamp(run_dir, "acc_019", "2000-01-01T00:00:00+00:00")
    # And acc_016 is a snapshot a divergent rerun republished: same name, same
    # number of done steps as the target's ledger, a history the target never had.
    _revise_snapshot_ledger(run_dir, "acc_016", add_done=["evolv_08"])

    kept, revoked = partition_checkpoints_by_target(run_dir, "acc_017")

    assert kept == ["acc_017"]
    assert {entry["checkpoint_id"] for entry in revoked} == {"acc_016", "acc_019"}
    reasons = {entry["checkpoint_id"]: entry["reason"] for entry in revoked}
    # The gap-numbered one is not a step of the target's ledger at all.
    assert "not a done step" in reasons["acc_019"]
    # The same-count one is, and is still excluded — by containment.
    assert "not contained" in reasons["acc_016"]
    assert "evolv_08" in reasons["acc_016"]


def test_an_unparseable_candidate_is_archived_not_silently_kept(
    tmp_path: Path,
) -> None:
    """A candidate whose own history cannot be read is not a candidate."""
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger)
    _history(run_dir, ledger, manager, [("acc_016", True), ("acc_017", True)])
    (run_dir / "checkpoints" / "acc_016" / LEDGER_LAYER).write_text(
        "{not json", encoding="utf-8"
    )

    kept, revoked = partition_checkpoints_by_target(run_dir, "acc_017")

    assert kept == ["acc_017"]
    assert revoked == [
        {
            "checkpoint_id": "acc_016",
            "reason": "its own metadata or ledger cannot be read",
        }
    ]


def test_a_republished_snapshot_is_selected_and_the_archived_one_is_not(
    tmp_path: Path,
) -> None:
    """The rerun of acc_018 succeeds, then the run dies again.

    The new acc_018 is a cut point of the *current* history and wins on its
    own merits; the old one it replaced is in the archive and takes no part.
    """
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger)
    _history(
        run_dir, ledger, manager, [("acc_017", True), ("acc_018", False)]
    )
    archived_ref = manager.metadata("acc_018")["timestamp"]
    manager.restore("acc_017", ledger=ledger)

    # The rerun of acc_018 completes and publishes a new snapshot.
    ledger.mark_running("acc_018")
    manager.save("acc_018", pinned=False)
    ledger.mark_done("acc_018")
    # Then the run dies in the step after it.
    ledger.mark_running("acc_019")
    ledger.mark_failed("acc_019", "died after the rerun")

    available = manager.list_checkpoints()
    assert available == ["acc_017", "acc_018"]
    assert max(available, key=manager.progress_key) == "acc_018"
    # The selected acc_018 is the new one, and the old one is still readable
    # in the archive under its recovery id.
    assert manager.metadata("acc_018")["timestamp"] != archived_ref
    archive = run_dir / "checkpoint_archive"
    old = next(archive.iterdir()) / "acc_018"
    assert json.loads((old / "metadata.json").read_text(encoding="utf-8"))[
        "timestamp"
    ] == archived_ref


def test_pins_before_the_target_branch_and_revoked_pins_cannot(
    tmp_path: Path,
) -> None:
    """A probe's start point survives a rollback only if the target's does."""
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger)
    _history(
        run_dir,
        ledger,
        manager,
        [("acc_016", True), ("acc_017", True), ("acc_018", True)],
    )
    manager.restore("acc_017", ledger=ledger)

    # A pin from before the target is untouched, so its probe still runs.
    assert manager.branch_from("acc_016", tmp_path / "br_ok").is_dir()
    # The pin the rollback revoked is not a cut point of this run any more.
    with pytest.raises(FileNotFoundError):
        manager.branch_from("acc_018", tmp_path / "br_gone")


def _saved_mem0_run(tmp_path: Path):
    run_dir, ledger, client = _mem0_run(tmp_path)
    manager = _external_manager(
        run_dir, ledger, condition="mem0", namespace="mempa_run_001", client=client
    )
    manager.save("acc_108", pinned=True)
    ledger.mark_done("acc_108")
    client.qdrant_admin.calls.clear()
    return run_dir, ledger, client, manager


def test_restore_refuses_when_the_remote_snapshot_is_gone(tmp_path: Path) -> None:
    """The live collection is dropped before the snapshot is recovered.

    `checkpoint_restorable` never opens a connection, so a Mem0 rollback used
    to discover a missing snapshot only after `delete_collection` had already
    destroyed the state it was replacing.
    """
    run_dir, ledger, client, manager = _saved_mem0_run(tmp_path)
    client.qdrant_admin.snapshots.clear()

    with pytest.raises(IncompleteCheckpointError, match="no longer holds"):
        manager.restore("acc_108", ledger=ledger)

    # Nothing was dropped, and nothing was begun.
    assert client.qdrant_admin.calls == []
    assert manager.pending_recovery() is None
    assert not (run_dir / "crash_scene").exists()


def test_restore_refuses_when_the_snapshot_server_is_unreachable(
    tmp_path: Path,
) -> None:
    run_dir, ledger, client, manager = _saved_mem0_run(tmp_path)
    client.qdrant_admin.unreachable = RuntimeError("connection refused")

    with pytest.raises(IncompleteCheckpointError, match="cannot be reached"):
        manager.restore("acc_108", ledger=ledger)

    assert client.qdrant_admin.calls == []
    assert manager.pending_recovery() is None


def test_a_backend_failure_mid_restore_keeps_the_marker_and_retries(
    tmp_path: Path,
) -> None:
    """The check passes, then the network fails anyway.

    No atomicity is claimed across the remote call: what makes this
    recoverable is the marker staying put and the snapshot staying on the
    server, so the same recovery simply runs again.
    """
    run_dir, ledger, client, manager = _saved_mem0_run(tmp_path)
    admin = client.qdrant_admin
    working_recover = admin.recover_snapshot
    calls: list[int] = []

    def failing_recover(*, collection_name: str, location: str) -> None:
        calls.append(1)
        raise RuntimeError("the server went away mid-recovery")

    admin.recover_snapshot = failing_recover
    with pytest.raises(RuntimeError, match="went away"):
        manager.restore("acc_108", ledger=ledger)

    pending = manager.pending_recovery()
    assert pending is not None
    assert pending["target_checkpoint_id"] == "acc_108"
    assert calls == [1]

    # The snapshot is still on the server, so the retry rebuilds from it.
    admin.recover_snapshot = working_recover
    admin.calls.clear()
    fresh_ledger = RunLedger(run_dir, RUN_ID)
    fresh = _external_manager(
        run_dir,
        fresh_ledger,
        condition="mem0",
        namespace="mempa_run_001",
        client=client,
    )
    fresh.restore("acc_108", ledger=fresh_ledger)

    assert fresh.pending_recovery() is None
    # The retry ran the whole backend layer again, from the same snapshot.
    assert admin.calls == [
        ("delete_collection", "mempa_run_001"),
        (
            "recover_snapshot",
            "mempa_run_001",
            "file:///qdrant/snapshots/mempa_run_001/snap-1",
        ),
    ]


# ── process termination, not a raised exception ───────────────────

#: A recovery driver that runs in a process of its own, so the interruption
#: under test is a real SIGKILL rather than an exception. An exception unwinds:
#: `finally` blocks run, buffers flush, objects get a chance to tidy up. A
#: killed process gets none of that, which is the situation the marker exists
#: for, so it is the situation that is tested.
_RECOVERY_DRIVER = '''
import json
import os
import signal
import sys
from pathlib import Path

from harness.checkpoint import CheckpointManager
from harness.orchestrator import resume_or_restore
from harness.run_ledger import RunLedger

RUN_ID = "run_001"


class PlanMustNotBeConsulted:
    """The startup path must read the marker before it asks anything else."""

    def __getattr__(self, name):
        raise AssertionError(
            "the plan was consulted before the recovery marker: " + name
        )


def die(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)


def main():
    mode, run_dir, target = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    ledger = RunLedger(run_dir, RUN_ID)
    manager = CheckpointManager(
        run_dir,
        ledger=ledger,
        memory_condition="nanobot_file_memory",
        memory_backend_namespace="ns_001",
        rolling_window=100,
    )
    if mode == "kill-after-pending":
        # Published immediately before archiving begins.
        CheckpointManager._archive_revoked_checkpoints = die
        manager.restore(target, ledger=ledger)
    elif mode == "kill-after-ledger":
        # ledger.replace_from has run by the time the marker is cleared.
        CheckpointManager._clear_pending_recovery = die
        manager.restore(target, ledger=ledger)
    elif mode == "finish":
        finished = resume_or_restore(ledger, manager, PlanMustNotBeConsulted())
        print(json.dumps({
            "finished": finished,
            "pending": manager.pending_recovery(),
            "checkpoints": manager.list_checkpoints(),
            "done": sorted(ledger.steps),
        }))
        return 0
    print("SURVIVED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
'''


def _driver(tmp_path: Path) -> Path:
    path = tmp_path / "recovery_driver.py"
    path.write_text(_RECOVERY_DRIVER, encoding="utf-8")
    return path


def _run_driver(driver: Path, mode: str, run_dir: Path, target: str):
    return subprocess.run(
        [sys.executable, str(driver), mode, str(run_dir), target],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        timeout=120,
    )


def _killable_run(tmp_path: Path) -> Path:
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger)
    memory = run_dir / "pa_workspace" / PERSONA / "memory" / "MEMORY.md"
    for step_id, pinned in [
        ("acc_001", True),
        ("acc_002", True),
        ("acc_003", False),
        ("acc_004", True),
    ]:
        memory.write_text(f"standing on {step_id}\n", encoding="utf-8")
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=pinned)
        ledger.mark_done(step_id)
    memory.write_text("standing on the crash\n", encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("mode", ["kill-after-pending", "kill-after-ledger"])
def test_a_killed_recovery_is_finished_by_the_next_process(
    tmp_path: Path, mode: str
) -> None:
    """A process destroyed mid-rollback, continued by a process that follows.

    The plan requires this of a real subprocess rather than a raised
    exception, at both of the two points where the run's own files have
    already changed: after the marker is published, and after the ledger has
    been replaced from the snapshot.
    """
    run_dir = _killable_run(tmp_path)
    driver = _driver(tmp_path)

    killed = _run_driver(driver, mode, run_dir, "acc_002")

    # Destroyed by the signal, with no chance to unwind.
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    assert "SURVIVED" not in killed.stderr

    # The marker survived the kill and names the target.
    pending = read_pending_recovery(run_dir)
    assert pending is not None
    assert pending["target_checkpoint_id"] == "acc_002"

    if mode == "kill-after-ledger":
        # The trap: the ledger left on disk is the target's own, so it reads as
        # a clean run with nothing to roll back.
        on_disk = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))
        assert sorted(on_disk["steps"]) == ["acc_001", "acc_002"]

    finished = _run_driver(driver, "finish", run_dir, "acc_002")

    assert finished.returncode == 0, finished.stderr
    report = json.loads(finished.stdout)
    # A new process, told only where the run is, went back to the same target.
    assert report["finished"] == "acc_002"
    assert report["pending"] is None
    assert report["checkpoints"] == ["acc_001", "acc_002"]
    assert report["done"] == ["acc_001", "acc_002"]
    memory = run_dir / "pa_workspace" / PERSONA / "memory" / "MEMORY.md"
    assert memory.read_text(encoding="utf-8") == "standing on acc_002\n"
    archive = run_dir / "checkpoint_archive" / pending["recovery_id"]
    assert sorted(path.name for path in archive.iterdir()) == ["acc_003", "acc_004"]


def test_an_unreadable_marker_is_a_verdict_not_a_crash(tmp_path: Path) -> None:
    """`run_restorable` stays total when the marker cannot be read.

    Admin asks this function and prints its reasons; a marker it cannot parse
    has to come back as "this run is not ready", not as an exception out of a
    function documented to return a verdict. The read is refused everywhere
    that would *act* on it, which is a different job from reporting.
    """
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    manager = _manager(run_dir, ledger)
    _history(run_dir, ledger, manager, [("acc_017", True)])
    marker = run_dir / "recovery" / "pending.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{truncated", encoding="utf-8")

    ok, reasons = run_restorable(run_dir, memory_condition="nanobot_file_memory")

    assert ok is False
    assert any("cannot be read" in reason for reason in reasons)
    # The acting entry points still refuse outright.
    with pytest.raises(RecoveryInProgressError):
        manager.save("acc_018", pinned=False)
    with pytest.raises(RecoveryInProgressError):
        manager.list_checkpoints()


@pytest.mark.parametrize('skips,expected', [
    (set(), {'acc_005', 'acc_010'}),
    ({'acc_002'}, {'acc_006', 'acc_011'}),
    ({'acc_002', 'acc_008'}, {'acc_006', 'acc_012'}),
])
def test_success_retention_survives_prune_and_restart(tmp_path, skips, expected):
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ids = frozenset(f'acc_{i:03d}' for i in range(1, 15))
    def manager():
        return CheckpointManager(run_dir, ledger=ledger, memory_condition='nanobot_file_memory',
                                 rolling_window=1, retention_interval=5, retention_mainline_ids=ids)
    current = manager()
    # A completed probe and a placeholder-shaped ledger row are not mainline successes.
    for sid in ('pre_01', 'gap_001'):
        ledger.mark_running(sid); ledger.mark_done(sid)
    for i in range(1, 15):
        sid = f'acc_{i:03d}'
        if sid in skips:
            ledger.mark_skipped(sid)
            continue
        if i == 8:
            ledger = RunLedger(run_dir, RUN_ID)  # real reload, not an in-memory counter
            current = manager()
        ledger.mark_running(sid)
        current.save(sid, pinned=(sid == 'acc_003'))
        ledger.mark_done(sid)
    assert {sid for sid in current.list_checkpoints() if current.metadata(sid).get('retained')} == expected
    assert 'acc_003' in current.list_checkpoints()  # existing semantic pin survives
    for sid in expected:
        assert current.metadata(sid)['pinned'] is False
        assert checkpoint_restorable(run_dir/'checkpoints'/sid, memory_condition='nanobot_file_memory') == (True, [])
        assert (run_dir/'checkpoints'/sid/'run_plan.yaml').exists()


def test_success_retention_rollback_uses_restored_ledger_not_old_counter(tmp_path):
    run_dir = _build_run_dir(tmp_path)
    _write_session(run_dir, cursors=(0, 0, 0))
    ledger = RunLedger(run_dir, RUN_ID)
    ids = frozenset(f'acc_{i:03d}' for i in range(1, 13))
    manager = CheckpointManager(run_dir, ledger=ledger, memory_condition='nanobot_file_memory',
                                rolling_window=1, retention_interval=5, retention_mainline_ids=ids,
                                retained_step_ids=frozenset({'acc_001'}))
    for i in range(1, 12):
        sid = f'acc_{i:03d}'; ledger.mark_running(sid)
        manager.save(sid, pinned=False); ledger.mark_done(sid)
    manager.restore('acc_005', ledger=ledger)
    assert 'acc_010' not in manager.list_checkpoints()  # revoked suffix is not active history
    ledger.mark_skipped('acc_006')
    for i in range(7, 13):
        sid=f'acc_{i:03d}';ledger.mark_running(sid)
        manager.save(sid,pinned=False);ledger.mark_done(sid)
    assert manager.metadata('acc_011')['retained'] is True  # tenth actual success
    assert manager.metadata('acc_001')['retained'] is True  # explicit legacy retention preserved
    assert checkpoint_restorable(run_dir/'checkpoints/acc_011',memory_condition='nanobot_file_memory')==(True,[])
    # Re-publishing the same done cut point does not increment the success set.
    manager.save('acc_012',pinned=False)
    assert not manager.metadata('acc_012').get('retained',False)
