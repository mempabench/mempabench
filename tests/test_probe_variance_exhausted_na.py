"""Exhausted-attempt Probes: not judged, scored at the floor (owner 2026-10-16)."""
import fcntl
import json
from pathlib import Path

import pytest
import yaml

from harness import metrics
from harness.probe_na import exhausted_failure, exhausted_failures
from test_harness.test_metrics import (
    FINAL_CELLS, JUDGE_ID, PRE_CELLS, _gt_setting, build_run,
)


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def attempts(run: Path, pid: str, statuses):
    save(run / "probe_attempts" / f"{pid}.json",
         {"step_id": pid, "attempts": [{"attempt": i, "status": s, "error": "x"} for i, s in enumerate(statuses, 1)]})


def ledger(pid="fin_27", status="failed"):
    return {"steps": {pid: {"status": status, "failure_kind": "ix_runtime"}}}


def test_only_three_failed_attempts_are_exhausted(tmp_path):
    attempts(tmp_path, "fin_27", ["failed"] * 3)
    evidence = exhausted_failure(tmp_path, "fin_27", ledger())
    assert evidence["attempts"]["path"] == "run/probe_attempts/fin_27.json" and len(evidence["errors"]) == 3
    assert exhausted_failures(tmp_path, metrics.PROBE_IDS, ledger()) == {"fin_27": evidence}
    for statuses in (["failed"], ["failed", "failed"], ["failed", "failed", "started"], ["failed", "failed", "succeeded"]):
        attempts(tmp_path, "fin_27", statuses)
        with pytest.raises(ValueError):
            exhausted_failure(tmp_path, "fin_27", ledger())
    attempts(tmp_path, "fin_27", ["failed"] * 3)
    with pytest.raises(ValueError, match="only a failed"):
        exhausted_failure(tmp_path, "fin_27", ledger(status="done"))


def test_held_attempt_lock_is_not_exhaustion(tmp_path):
    attempts(tmp_path, "fin_27", ["failed"] * 3)
    with (tmp_path / "probe_attempts" / "fin_27.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="still owned"):
            exhausted_failure(tmp_path, "fin_27", ledger())


def write_plan(run: Path) -> None:
    """The probe steps the floor takes its GT from (an exhausted Probe is never packed)."""
    cells = dict(zip(metrics.FINAL_PROBE_IDS, FINAL_CELLS)) | dict(zip(metrics.PRE_EVENT_PROBE_IDS, PRE_CELLS))
    steps = [{"step_id": pid, "context": context, "target_attribute": attribute,
              "target_setting": _gt_setting(context, attribute)} for pid, (context, attribute) in cells.items()]
    (run / "run_plan.yaml").write_text(yaml.safe_dump({"steps": steps}))


def test_an_exhausted_probe_gets_the_floor_and_every_denominator_stays(tmp_path):
    # fin_01 is work_attr_01: a differing attribute (E3) and pre_01's final (E4).
    run = build_run(tmp_path / "run", ix_match={"fin_02": False}, skip_probe="fin_01")
    write_plan(run)
    with pytest.raises(FileNotFoundError):
        metrics.compute_metrics(run, JUDGE_ID)
    result = metrics.compute_metrics(run, JUDGE_ID, not_applicable=frozenset({"fin_01"}))
    assert result["rule_assigned_scores"] == {
        "fin_01": {"source": "retain_scores_model_failure_floor_v1", "ix": 0, "text": 1}}
    e1 = result["E1_consistency"]
    assert e1["n"] == 28 and e1["ix_mean"] == 26 / 28
    assert e1["text_percent"] == pytest.approx(100.0 * (27 * 5 + 1) / (5 * 28))
    assert result["E2_cross_tab"]["n"] == 28
    e3 = result["E3_cross_scenario"]
    assert e3["differing_attributes"] == ["attr_01", "attr_02"]
    assert e3["consistency"]["n"] == 4 and e3["consistency"]["ix_mean"] == 2 / 4
    assert e3["cross_scenario_error"]["count"] == 0  # a floor selects nothing, so it never crosses
    e4 = result["E4_pet"]
    assert [pair["pre"] for pair in e4["pairs"]] == list(metrics.PRE_EVENT_PROBE_IDS)
    assert e4["n"] == 10 and e4["ix_mean"] == 8 / 10
    assert "Floored (exhausted attempts, retain_scores_model_failure_floor_v1: IX 0, text 1): fin_01" \
        in metrics.render_markdown(result)
    complete = metrics.compute_metrics(build_run(tmp_path / "full"), JUDGE_ID)
    assert "rule_assigned_scores" not in complete and "Floored" not in metrics.render_markdown(complete)


def test_the_metrics_cli_floors_the_probes_the_judge_audit_lists(tmp_path):
    """`python -m harness.metrics` scores an N/A round as score_completed_run does."""
    run = build_run(tmp_path / "run", skip_probe="fin_01")
    write_plan(run)
    save(run / "judge_audit" / "manifest.json", {"not_applicable": {"fin_01": {"errors": ["x"] * 3}}})
    assert metrics.main(["--run-dir", str(run), "--judge-id", JUDGE_ID]) == 0
    written = json.loads((run / "metrics" / f"{JUDGE_ID}.json").read_text())
    expected = metrics.compute_metrics(run, JUDGE_ID, not_applicable=frozenset({"fin_01"}))
    assert written == json.loads(json.dumps(expected))
    assert written["E1_consistency"]["n"] == 28 and written["rule_assigned_scores"] == metrics.floor_assignments({"fin_01"})
    # Without the audit's list the probe's missing score is a missing input, not a floor.
    (run / "judge_audit" / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        metrics.main(["--run-dir", str(run), "--judge-id", JUDGE_ID])


def test_a_failure_with_attempts_left_keeps_the_round_unscored(tmp_path, monkeypatch):
    from scripts import score_completed_run as scorer
    monkeypatch.setattr(scorer, "pack_run", lambda *a, **k: pytest.fail("an incomplete round reached packing"))
    save(tmp_path / "ledger.json", {"steps": {"fin_27": {"status": "failed", "failure_kind": "ix_runtime"}}})
    attempts(tmp_path, "fin_27", ["failed", "failed"])
    with pytest.raises(ValueError, match="attempts are not 3"):
        scorer.score(tmp_path, variance_retries=True, not_applicable_exhausted=True)
    assert not (tmp_path / "metrics").exists() and not (tmp_path / "judge_audit").exists()


def test_original_exhaustion_needs_retryable_kind_and_two_superseded_attempts(tmp_path):
    from harness.probe_na import original_exhausted_failure
    record = {"steps": {"fin_27": {"status": "failed", "failure_kind": "ix_runtime"}}}
    with pytest.raises(ValueError, match="only 1 recorded"):
        original_exhausted_failure(tmp_path, "fin_27", record)
    for stamp in ("a", "b"):
        (tmp_path / "steps_superseded" / f"fin_27__{stamp}").mkdir(parents=True)
    assert len(original_exhausted_failure(tmp_path, "fin_27", record)["superseded_attempts"]) == 2
    record["steps"]["fin_27"]["failure_kind"] = "provider"
    with pytest.raises(ValueError, match="not retried"):
        original_exhausted_failure(tmp_path, "fin_27", record)


def test_raised_pa_cap_is_per_probe_bounded_and_recorded(tmp_path):
    from types import SimpleNamespace as NS
    from scripts.run_orchestrator import _probe_pa_max_tokens
    plan = NS(pa_max_tokens=8192)
    args = NS(pa_max_tokens_for_probe=["fin_10=24576"])
    probe, other = NS(step_id="fin_10", kind="final_probe"), NS(step_id="fin_11", kind="final_probe")
    assert _probe_pa_max_tokens(args, plan, other, tmp_path) == 8192
    assert _probe_pa_max_tokens(args, plan, probe, tmp_path) == 24576
    [record] = (tmp_path / "pa_output_cap_override").glob("fin_10-*.json")
    assert json.loads(record.read_text())["effective_pa_max_tokens"] == 24576
    with pytest.raises(ValueError):
        _probe_pa_max_tokens(NS(pa_max_tokens_for_probe=["fin_10=40000"]), plan, probe, tmp_path)


def test_rerun_only_failed_probes_with_attempts_left(tmp_path):
    from types import SimpleNamespace as NS
    from scripts.run_probe_variance import require_retryable
    save(tmp_path / "ledger.json", {"steps": {"fin_10": {"status": "failed"}, "fin_13": {"status": "failed"},
                                              "fin_01": {"status": "done"}}})
    attempts(tmp_path, "fin_10", ["failed"])
    attempts(tmp_path, "fin_13", ["failed"] * 3)
    require_retryable(tmp_path, NS(rerun_probe_steps="fin_10"))
    for step in ("fin_13", "fin_01"):
        with pytest.raises(ValueError, match="attempts left"):
            require_retryable(tmp_path, NS(rerun_probe_steps=step))


def test_network_failures_do_not_count_toward_the_three(tmp_path):
    from harness.probe_attempts import ProbeAttempts
    net, ix = "APIConnectionError: Connection error.", "PA turn failed the IX render contract"
    def write(rows):
        save(tmp_path / "probe_attempts" / "fin_27.json",
             {"step_id": "fin_27", "attempts": [{"attempt": i, "status": "failed", "error": e} for i, e in enumerate(rows, 1)]})
    write([net, ix, net, ix])
    assert ProbeAttempts(tmp_path, "fin_27").next_number() == 5  # only 2 counted
    with pytest.raises(ValueError, match="non-network"):
        exhausted_failure(tmp_path, "fin_27", ledger())
    write([net, ix, net, ix, ix])
    assert exhausted_failure(tmp_path, "fin_27", ledger())["errors"][-1] == ix
    with pytest.raises(RuntimeError, match="three counted"):
        ProbeAttempts(tmp_path, "fin_27").next_number()
    write([net] * 6)
    with pytest.raises(RuntimeError, match="six total"):
        ProbeAttempts(tmp_path, "fin_27").next_number()
    with pytest.raises(ValueError):
        exhausted_failure(tmp_path, "fin_27", ledger())


def test_user_authorized_reopen_gives_one_more_budget_of_three(tmp_path):
    from types import SimpleNamespace as NS
    from harness.probe_attempts import ProbeAttempts
    from scripts.run_probe_variance import require_retryable
    save(tmp_path / "ledger.json", {"steps": {"fin_13": {"status": "failed"}}})
    attempts(tmp_path, "fin_13", ["failed"] * 2)
    with pytest.raises(RuntimeError, match="exhausted"):
        ProbeAttempts(tmp_path, "fin_13").reopen("C/USER_X.json")  # not exhausted yet
    attempts(tmp_path, "fin_13", ["failed"] * 3)
    record = ProbeAttempts(tmp_path, "fin_13")
    record.reopen("C/USER_X.json")
    with pytest.raises(RuntimeError, match="reopened"):
        record.reopen("C/USER_X.json")
    assert (record.next_number(), record.ceiling()) == (4, 6)
    require_retryable(tmp_path, NS(rerun_probe_steps="fin_13"))
    with pytest.raises(ValueError, match="non-network"):  # the three since the reopen are not used up
        exhausted_failure(tmp_path, "fin_13", ledger("fin_13"))
    value = json.loads(record.path.read_text())
    value["attempts"] += [{"attempt": i, "status": "failed", "error": "x"} for i in (4, 5, 6)]
    save(record.path, value)
    assert len(exhausted_failure(tmp_path, "fin_13", ledger("fin_13"))["errors"]) == 6
    with pytest.raises(RuntimeError):
        ProbeAttempts(tmp_path, "fin_13").next_number()
    with pytest.raises(ValueError, match="attempts left"):
        require_retryable(tmp_path, NS(rerun_probe_steps="fin_13"))


def test_settled_round_is_left_alone(tmp_path):
    from scripts.run_probe_variance import production_settled
    ids = metrics.PROBE_IDS
    save(tmp_path / "run_plan.yaml", {"steps": [{"step_id": pid, "kind": "final_probe" if pid.startswith("fin") else "pre_event_probe"} for pid in ids]})
    steps = {pid: {"status": "done"} for pid in ids}
    steps["fin_28"] = {"status": "failed"}
    save(tmp_path / "ledger.json", {"steps": steps})
    attempts(tmp_path, "fin_28", ["failed"] * 2)
    assert not production_settled(tmp_path)  # one attempt left
    attempts(tmp_path, "fin_28", ["failed"] * 3)
    assert production_settled(tmp_path)
    steps["fin_01"] = {"status": "running"}
    save(tmp_path / "ledger.json", {"steps": steps})
    assert not production_settled(tmp_path)


# A source run (a plain `mempabench run`) records every attempt of a Probe, and
# the launch it belongs to, in `probe_launch_attempts/<id>.json`
# (`harness.probe_launch_attempts`). Exhaustion is read from those records and
# the ledger row only; no clock is consulted.
IX = "ix_runtime"


def launch(run: Path, launch_id: str, outcomes, pid="fin_27") -> None:
    """Record one launch's attempts through the run's writer: each outcome is a
    failure kind, "succeeded", or "started" (an attempt that never reported back)."""
    from harness.probe_launch_attempts import ProbeLaunchAttempts
    record = ProbeLaunchAttempts(run, pid, launch_id)
    for attempt, outcome in enumerate(outcomes, 1):
        record.start(attempt)
        if outcome == "succeeded":
            record.succeed(attempt)
        elif outcome != "started":
            record.fail(attempt, failure_kind=outcome, error=RuntimeError(f"attempt {attempt}"))


def source_ledger(status="failed", kind=IX, pid="fin_27") -> dict:
    record = {"status": status, "failure_kind": kind if status == "failed" else None,
              "started_at": "2026-10-18T10:00:00+00:00", "ended_at": "2026-10-18T10:06:00+00:00",
              "error": "PA turn failed the IX render contract: x" if status == "failed" else None}
    return {"run_id": "r", "current_step": pid if status == "failed" else None, "steps": {pid: record}}


def test_a_source_probe_whose_last_launch_failed_three_times_is_exhausted(tmp_path):
    from harness.probe_na import na_evidence, source_exhausted_failure, source_exhausted_failures
    launch(tmp_path, "launch-1", [IX] * 3)
    record = source_ledger()
    evidence = source_exhausted_failure(tmp_path, "fin_27", record)
    assert evidence["launch_id"] == "launch-1" and evidence["failure_kind"] == IX
    assert evidence["attempts"] == {"path": "run/probe_launch_attempts/fin_27.json"}
    assert evidence["errors"] == [f"RuntimeError: attempt {n}" for n in (1, 2, 3)]
    assert source_exhausted_failures(tmp_path, metrics.PROBE_IDS, record) == {"fin_27": evidence}
    assert na_evidence(tmp_path, "fin_27", record) == evidence  # what require_complete checks


def test_rows_of_an_earlier_launch_never_count(tmp_path):
    from harness.probe_na import source_exhausted_failure
    # Exhausted in launch 1; the resume (launch 2) failed once and was stopped.
    launch(tmp_path / "a", "launch-1", [IX] * 3)
    launch(tmp_path / "a", "launch-2", [IX])
    with pytest.raises(ValueError, match="fin_27: its last launch launch-2 recorded 1 attempt"):
        source_exhausted_failure(tmp_path / "a", "fin_27", source_ledger())
    # Stopped after two in launch 1; launch 2 failed all three: only its own count.
    launch(tmp_path / "b", "launch-1", [IX] * 2)
    launch(tmp_path / "b", "launch-2", [IX] * 3)
    evidence = source_exhausted_failure(tmp_path / "b", "fin_27", source_ledger())
    assert evidence["launch_id"] == "launch-2" and len(evidence["errors"]) == 3


def test_a_launch_with_two_of_three_failed_attempts_is_not_exhaustion(tmp_path):
    """A stop or signal ends a launch's attempts early; its last failure still has a retried kind."""
    from harness.probe_na import source_exhausted_failure
    launch(tmp_path, "launch-1", [IX] * 2)
    with pytest.raises(ValueError, match="fin_27: its last launch launch-1 recorded 2 attempt"):
        source_exhausted_failure(tmp_path, "fin_27", source_ledger())


def test_a_failure_kind_that_is_not_retried_is_not_exhaustion(tmp_path):
    from harness.probe_na import source_exhausted_failure
    launch(tmp_path / "a", "launch-1", ["provider"])
    with pytest.raises(ValueError, match="fin_27: its last launch launch-1 recorded 1 attempt"):
        source_exhausted_failure(tmp_path / "a", "fin_27", source_ledger(kind="provider"))
    launch(tmp_path / "b", "launch-1", [IX, IX, "provider"])
    with pytest.raises(ValueError, match="not 3 failed attempts with a retried failure kind"):
        source_exhausted_failure(tmp_path / "b", "fin_27", source_ledger(kind="provider"))


def test_a_running_or_interrupted_source_probe_is_not_exhaustion(tmp_path):
    from harness.probe_na import source_exhausted_failure, source_exhausted_failures
    launch(tmp_path, "launch-1", [IX, IX, "started"])
    record = source_ledger(status="running")
    with pytest.raises(ValueError, match="fin_27: only a failed"):
        source_exhausted_failure(tmp_path, "fin_27", record)
    assert source_exhausted_failures(tmp_path, metrics.PROBE_IDS, record) == {}  # left to require_complete


def test_a_failed_probe_without_an_attempt_record_names_the_resume_path(tmp_path):
    """A run whose attempts were not recorded cannot show exhaustion, whatever
    its set-aside attempts and timestamps look like."""
    from harness.probe_na import source_exhausted_failure, source_exhausted_failures
    for stamp in ("20261018T100200Z", "20261018T100400Z"):
        (tmp_path / "steps_superseded" / f"fin_27__{stamp}").mkdir(parents=True)
    (tmp_path / "steps" / "fin_27").mkdir(parents=True)
    expected = (r"fin_27: no attempt record probe_launch_attempts/fin_27\.json.*"
                r"Resume the run with the same `mempabench run` command and run ID: fin_27 gets 3 new attempts")
    with pytest.raises(ValueError, match=expected):
        source_exhausted_failure(tmp_path, "fin_27", source_ledger())
    with pytest.raises(ValueError, match=expected):
        source_exhausted_failures(tmp_path, metrics.PROBE_IDS, source_ledger())


def _rows(*rows):
    return {"schema": "mempa_probe_launch_attempts/1", "step_id": "fin_27", "attempts": list(rows)}


def _failed(attempt, launch_id="launch-1", kind=IX):
    return {"launch_id": launch_id, "attempt": attempt, "status": "failed", "failure_kind": kind, "error": "x"}


@pytest.mark.parametrize("content", [
    "{not json",
    json.dumps({"step_id": "fin_27", "attempts": [_failed(1), _failed(2), _failed(3)]}),  # no schema
    json.dumps(_rows()),
    json.dumps(_rows(_failed(1), _failed(3), _failed(4))),  # attempt 2 missing
    json.dumps(_rows(_failed(1), _failed(2), _failed(3, kind="exploded"))),
    json.dumps(_rows(_failed(1), {"launch_id": "launch-1", "attempt": 2, "status": "failed"}, _failed(3))),
    json.dumps(_rows(_failed(1), _failed(1, "launch-2"), _failed(2), _failed(3))),  # launch-1 split
], ids=["not-json", "no-schema", "no-attempts", "numbering", "kind", "missing-fields", "split-launch"])
def test_a_corrupt_attempt_record_is_an_error_not_a_refusal_to_floor(tmp_path, content):
    from harness.probe_na import source_exhausted_failure
    path = tmp_path / "probe_launch_attempts" / "fin_27.json"
    path.parent.mkdir()
    path.write_text(content)
    message = r"fin_27: attempt record probe_launch_attempts/fin_27\.json is invalid"
    with pytest.raises(ValueError, match=message):
        source_exhausted_failure(tmp_path, "fin_27", source_ledger())
    # The run never writes past a corrupt record either.
    with pytest.raises(ValueError, match=message):
        launch(tmp_path, "launch-2", [IX])
    assert path.read_text() == content


def test_a_repeat_round_still_needs_its_attempt_ledger(tmp_path):
    """Probe repeats keep their own evidence, even where a source run's record would pass."""
    from harness.probe_na import na_evidence
    launch(tmp_path, "launch-1", [IX] * 3)
    (tmp_path / "probe_repeat_source.json").write_text("{}")
    with pytest.raises(ValueError, match="fin_27: no durable attempt record"):
        na_evidence(tmp_path, "fin_27", source_ledger())
