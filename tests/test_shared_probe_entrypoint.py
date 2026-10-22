"""Repeat entrypoint source binding, stopped before provider/backend creation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from harness import probe_repeat_source
from harness.run_ledger import RunLedger
from scripts import run_orchestrator


class StopBeforeRuntime(Exception):
    pass


@pytest.mark.asyncio
async def test_shared_entrypoint_reads_original_ledger_without_mutating_source(tmp_path, monkeypatch):
    source = tmp_path / "accepted-original"
    original = RunLedger(source, "original-run")
    original.mark_done(
        "evolv_01", summary={"preference_overlay": {"work_autonomy_level": "Reactive"}}
    )
    source_plan = source / "run_plan.yaml"
    source_plan.write_text("run_id: original-run\n")
    original_bytes = original.path.read_bytes()
    repeat = tmp_path / "repeat02"
    repeat.mkdir()
    plan_path = repeat / "run_plan.yaml"
    plan_path.write_text("run_id: independent-repeat02\n")
    verifier = Mock(return_value={"schema_version": 1, "repeat": 2, "source_run_dir": str(source)})
    monkeypatch.setattr(probe_repeat_source, "load_repeat_source", verifier)
    read_ledgers = []

    def ledger_reader(directory, run_id):
        ledger = RunLedger(directory, run_id)
        read_ledgers.append(ledger)
        return ledger

    monkeypatch.setattr(run_orchestrator, "RunLedger", ledger_reader)
    monkeypatch.setattr(run_orchestrator, "_parse_force_ix", Mock(side_effect=StopBeforeRuntime))
    memos = Mock(side_effect=AssertionError("must not open a backend"))
    monkeypatch.setattr(run_orchestrator, "MemOSHttpClient", memos)
    args = SimpleNamespace(plan=str(plan_path), overwrite=False, regenerate_plan=False, verbose=True)

    with pytest.raises(StopBeforeRuntime):
        await run_orchestrator._execute_benchmark_run(args, shared_probe_memory_owner=object())

    verifier.assert_called_once_with(repeat)
    assert len(read_ledgers) == 1
    loaded = read_ledgers[0]
    assert loaded.run_id == "original-run"
    assert loaded.run_dir == source
    assert loaded.steps["evolv_01"].summary["preference_overlay"] == {"work_autonomy_level": "Reactive"}
    assert original.path.read_bytes() == original_bytes
    assert not (repeat / "ledger.json").exists()
    memos.assert_not_called()


@pytest.mark.asyncio
async def test_shared_entrypoint_rejects_missing_source_before_clients_or_ledgers(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_repeat_source, "load_repeat_source", Mock(return_value=None))
    ledger = Mock(side_effect=AssertionError("must verify source before creating a ledger"))
    backend = Mock(side_effect=AssertionError("must not open a backend"))
    monkeypatch.setattr(run_orchestrator, "RunLedger", ledger)
    monkeypatch.setattr(run_orchestrator, "MemOSHttpClient", backend)
    args = SimpleNamespace(plan=str(tmp_path / "run_plan.yaml"), overwrite=False, regenerate_plan=False)

    with pytest.raises(ValueError, match="requires a probe_repeat_source"):
        await run_orchestrator._execute_benchmark_run(args, shared_probe_memory_owner=object())

    ledger.assert_not_called()
    backend.assert_not_called()
