"""A plain run records every probe attempt, and its launch, as it happens.

`score --not-applicable-exhausted` floors a probe only when the attempts its
last launch recorded are three failures with a retried kind
(`harness.probe_na.source_exhausted_failure`). These tests drive the real
retry loop (`Orchestrator._run_one_probe`) through whole launches of one run
directory, so the records under test are the ones a run writes. The ledger's
`started_at` and the `steps_superseded/` stamps are whole seconds of a clock
that can be set back; the records must decide alone (Codex round 5, A-1/A-2).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from harness import orchestrator as orchestrator_module
from harness import run_layout, run_ledger
from test_harness.test_orchestrator import (
    _IXRuntimeProbeExecutor,
    _VoidAfterAStopExecutor,
    _batch_orchestrator,
    _many_probe_plan,
)

UTC = timezone.utc


class Clock:
    """The wall clock behind the ledger's `started_at` and the set-aside stamps."""

    def __init__(self, monkeypatch, now: datetime) -> None:
        self.now = now
        clock = self

        class Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.now

        monkeypatch.setattr(run_layout, "datetime", Frozen)
        monkeypatch.setattr(run_ledger, "now_iso", lambda: clock.now.isoformat(timespec="seconds"))


async def launch(tmp_path: Path, plan, executor, **kwargs):
    """One start or resume of the same run directory."""
    orchestrator, ledger, run_dir = _batch_orchestrator(tmp_path, plan, executor, concurrency=1, **kwargs)
    await orchestrator.run()
    return orchestrator, ledger, run_dir


def rows(run_dir: Path, pid: str) -> list[tuple]:
    record = json.loads((run_dir / "probe_launch_attempts" / f"{pid}.json").read_text())
    return [(row["launch_id"], row["attempt"], row["status"], row.get("failure_kind")) for row in record["attempts"]]


def ledger_of(run_dir: Path) -> dict:
    return json.loads((run_dir / "ledger.json").read_text())


@pytest.mark.asyncio
async def test_each_attempt_is_recorded_with_its_launch_before_the_run_acts_on_it(tmp_path, monkeypatch):
    plan = _many_probe_plan(2)
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, _IXRuntimeProbeExecutor("fin_01", voids=99), concurrency=1)
    seen: list[tuple] = []

    def last_row(pid):
        path = run_dir / "probe_launch_attempts" / f"{pid}.json"
        row = json.loads(path.read_text())["attempts"][-1] if path.exists() else {}
        return row.get("attempt"), row.get("status")

    set_aside = orchestrator_module.supersede_step_dirs

    def supersede(run, step_ids):
        seen.append(("set aside", step_ids[0], last_row(step_ids[0])))
        return set_aside(run, step_ids)

    monkeypatch.setattr(orchestrator_module, "supersede_step_dirs", supersede)
    for name in ("mark_failed", "mark_done"):
        def wrapped(step_id, *args, _name=name, _inner=getattr(ledger, name), **kwargs):
            if step_id.startswith("fin_"):
                seen.append((_name, step_id, last_row(step_id)))
            return _inner(step_id, *args, **kwargs)
        monkeypatch.setattr(ledger, name, wrapped)

    await orchestrator.run()

    # The outcome is on disk before the attempt is set aside and before the ledger moves.
    assert seen == [("set aside", "fin_01", (1, "failed")), ("set aside", "fin_01", (2, "failed")),
                    ("mark_failed", "fin_01", (3, "failed")), ("mark_done", "fin_02", (1, "succeeded"))]
    record = json.loads((run_dir / "probe_launch_attempts" / "fin_01.json").read_text())
    assert record["schema"] == "mempa_probe_launch_attempts/1" and record["step_id"] == "fin_01"
    launch_id = orchestrator.launch_id
    assert isinstance(launch_id, str) and launch_id
    assert rows(run_dir, "fin_01") == [(launch_id, n, "failed", "ix_runtime") for n in (1, 2, 3)]
    assert all(row["error"].startswith("AgentProviderError: PA turn failed the IX render contract")
               for row in record["attempts"])
    succeeded = json.loads((run_dir / "probe_launch_attempts" / "fin_02.json").read_text())["attempts"]
    assert succeeded == [{"launch_id": launch_id, "attempt": 1, "status": "succeeded"}]
    from harness.probe_na import source_exhausted_failures
    assert set(source_exhausted_failures(run_dir, ["fin_01", "fin_02"], ledger_of(run_dir))) == {"fin_01"}


@pytest.mark.asyncio
async def test_a_resume_in_the_same_second_is_still_exhausted(tmp_path, monkeypatch):
    """A-1: both launches and every set-aside fall in one second of the ledger's
    clock. The resumed probe failed all three new attempts, so it is exhausted."""
    Clock(monkeypatch, datetime(2026, 10, 18, 10, 0, 0, tzinfo=UTC))
    plan = _many_probe_plan(1)
    first, _, run_dir = await launch(tmp_path, plan, _IXRuntimeProbeExecutor("fin_01", voids=99))
    second, ledger, _ = await launch(tmp_path, plan, _IXRuntimeProbeExecutor("fin_01", voids=99))
    assert ledger.steps["fin_01"].status == "failed"
    assert ledger.steps["fin_01"].started_at == "2026-10-18T10:00:00+00:00"
    from harness.probe_na import source_exhausted_failure
    evidence = source_exhausted_failure(run_dir, "fin_01", ledger_of(run_dir))
    assert first.launch_id != second.launch_id
    assert evidence["launch_id"] == second.launch_id and len(evidence["errors"]) == 3
    assert rows(run_dir, "fin_01") == [(launch_id, n, "failed", "ix_runtime")
                                       for launch_id in (first.launch_id, second.launch_id) for n in (1, 2, 3)]


@pytest.mark.asyncio
async def test_a_clock_moved_back_never_floors_a_probe_that_failed_once(tmp_path, monkeypatch):
    """A-2: launch 1 failed all three attempts; the clock then went back an hour,
    and launch 2 failed once before a stop. Launch 1's set-asides now read as
    later than launch 2's `started_at`; the probe still has attempts left."""
    clock = Clock(monkeypatch, datetime(2026, 10, 18, 11, 0, 0, tzinfo=UTC))
    plan = _many_probe_plan(1)
    first, _, run_dir = await launch(tmp_path, plan, _IXRuntimeProbeExecutor("fin_01", voids=99))
    clock.now = datetime(2026, 10, 18, 10, 0, 0, tzinfo=UTC)
    stopping = {"requested": False}
    executor = _VoidAfterAStopExecutor("fin_01", take_the_stop=lambda: stopping.update(requested=True))
    second, ledger, _ = await launch(tmp_path, plan, executor, stop_requested=lambda: stopping["requested"])
    assert ledger.steps["fin_01"].status == "failed"
    assert ledger.steps["fin_01"].failure_kind == "judge_timeout"  # a retried kind
    from harness.probe_na import source_exhausted_failure
    with pytest.raises(ValueError, match="fin_01: its last launch .* recorded 1 attempt"):
        source_exhausted_failure(run_dir, "fin_01", ledger_of(run_dir))
    assert rows(run_dir, "fin_01") == [(first.launch_id, n, "failed", "ix_runtime") for n in (1, 2, 3)] + [
        (second.launch_id, 1, "failed", "judge_timeout")]


@pytest.mark.asyncio
async def test_a_probe_that_failed_every_attempt_and_then_passed_on_resume_is_never_floored(tmp_path):
    plan = _many_probe_plan(1)
    first, _, run_dir = await launch(tmp_path, plan, _IXRuntimeProbeExecutor("fin_01", voids=99))
    second, ledger, _ = await launch(tmp_path, plan, _IXRuntimeProbeExecutor("fin_01", voids=0))
    assert ledger.steps["fin_01"].status == "done"
    assert rows(run_dir, "fin_01") == [(first.launch_id, n, "failed", "ix_runtime") for n in (1, 2, 3)] + [
        (second.launch_id, 1, "succeeded", None)]
    from harness.probe_na import source_exhausted_failure, source_exhausted_failures
    assert source_exhausted_failures(run_dir, ["fin_01"], ledger_of(run_dir)) == {}
    with pytest.raises(ValueError, match="fin_01: only a failed"):
        source_exhausted_failure(run_dir, "fin_01", ledger_of(run_dir))


@pytest.mark.asyncio
async def test_an_attempt_that_cannot_be_recorded_stops_the_run_before_the_state_moves(tmp_path, monkeypatch):
    from harness import probe_launch_attempts

    def disk_full(self, attempt, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(probe_launch_attempts.ProbeLaunchAttempts, "fail", disk_full)
    plan = _many_probe_plan(2)
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, _IXRuntimeProbeExecutor("fin_01", voids=99), concurrency=1)
    with pytest.raises(OSError, match="No space left on device"):
        await orchestrator.run()
    # No retry, no set-aside and no ledger outcome without the record.
    assert ledger.steps["fin_01"].status == "running"
    assert ledger_of(run_dir)["steps"]["fin_01"]["status"] == "running"
    assert not (run_dir / "steps_superseded").exists()
    assert rows(run_dir, "fin_01") == [(orchestrator.launch_id, 1, "started", None)]
    assert "fin_02" not in ledger.steps
