import hashlib
import json

from scripts.report_probe_variance import retrieval_summary, sample, summarize


def test_complete_wall_includes_preflight_and_survives_round_switch(tmp_path):
    state, run = tmp_path / "state", tmp_path / "repeat2"
    state.mkdir()
    run.mkdir()
    source = str(tmp_path / "source")
    identity = hashlib.sha256(source.encode()).hexdigest()
    worker = state / "settings" / identity / "worker.json"
    worker.parent.mkdir(parents=True)
    worker.write_text(json.dumps({"current_repeat": 2, "updated_at": "2026-09-22T16:59:00+00:00"}))
    manifest = {"state_root": str(state), "settings": [{"id": "s", "source": {"run_dir": source},
        "repeats": [{"repeat": 2, "argv": ["--plan", str(run / "run_plan.yaml")]}]}]}
    steps = {f"{kind}_{i:02d}": {"status": "done", "started_at": "2026-09-22T17:00:00+00:00",
        "ended_at": "2026-09-22T17:01:00+00:00"} for kind, total in [("pre", 5), ("fin", 28)]
        for i in range(1, total + 1)}
    (run / "ledger.json").write_text(json.dumps({"steps": steps}))
    starts = {}
    row = sample(manifest, starts)["rounds"][0]
    assert row["production_wall_seconds"] == 120
    assert row["probes_per_hour"] == 990
    assert row["inflight_model_requests"] is None
    worker.write_text(json.dumps({"current_repeat": 3, "updated_at": "2026-09-22T17:02:00+00:00"}))
    assert sample(manifest, starts)["rounds"][0]["production_wall_seconds"] == 120
    assert sample(manifest)["rounds"][0]["production_wall_seconds"] is None
    steps["fin_28"]["status"] = "running"
    (run / "ledger.json").write_text(json.dumps({"steps": steps}))
    row = sample(manifest, starts)["rounds"][0]
    assert not row["production_complete"]
    assert row["production_wall_seconds"] is None
    assert row["probes_per_hour"] is None


def test_observed_concurrency_is_not_declared_verified_capacity():
    rows = [{"active_rounds": 7, "observed_at": "now", "resident_instances": 42,
             "resources": {"memory": {"MemAvailable_bytes": value}}} for value in (0, 1000)]
    last = {**rows[-1], "rounds": [{"production_complete": True}]}
    result = summarize(rows, last)
    assert result["highest_observed_G"] == 7
    assert result["highest_complete_validated_G"] is None
    assert result["concurrency_measurements"][0]["min_available_memory_bytes"] == 0
    assert result["completed_rounds"] == 1


def test_retrieval_percentile_uses_only_actual_completed_logical_searches(tmp_path):
    assert retrieval_summary(tmp_path)["p95_seconds"] is None
    rows = [{"boundary": "native_logical_search", "run_dir": str(tmp_path),
             "elapsed_ms": number * 1000, "success": number != 20} for number in range(1, 21)]
    rows.append({"boundary": "native_logical_search", "run_dir": "another-run",
                 "elapsed_ms": 1000000, "success": True})
    (tmp_path / "retrieval_activity.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + '{"incomplete":')
    result = retrieval_summary(tmp_path)
    assert result["p95_seconds"] == 19
    assert result["samples"] == 20 and result["failures"] == 1


def test_a_complete_repeat_is_the_probe_count_the_run_config_names(tmp_path, monkeypatch):
    """`probe_variance.pre_event_probes` / `final_probes` decide what complete means;
    a repeat with the default 33 probes is not complete under a 1 + 2 scope."""
    config = tmp_path / "run_config.yaml"
    config.write_text("probe_variance:\n  pre_event_probes: 1\n  final_probes: 2\n")
    monkeypatch.setenv("MEMPA_RUN_CONFIG", str(config))
    state, run = tmp_path / "state", tmp_path / "repeat2"
    run.mkdir()
    source = str(tmp_path / "source")
    worker = state / "settings" / hashlib.sha256(source.encode()).hexdigest() / "worker.json"
    worker.parent.mkdir(parents=True)
    worker.write_text(json.dumps({"current_repeat": 2, "updated_at": "2026-09-22T16:59:00+00:00"}))
    manifest = {"state_root": str(state), "settings": [{"id": "s", "source": {"run_dir": source},
        "repeats": [{"repeat": 2, "argv": ["--plan", str(run / "run_plan.yaml")]}]}]}
    done = {"status": "done", "started_at": "2026-09-22T17:00:00+00:00", "ended_at": "2026-09-22T17:01:00+00:00"}
    (run / "ledger.json").write_text(json.dumps({"steps": {
        sid: done for sid in ("pre_01", "fin_01", "fin_02")}}))
    row = sample(manifest)["rounds"][0]
    assert row["production_complete"] and row["expected_probes"] == 3
    assert row["probes_per_hour"] == 3 * 3600 / row["production_wall_seconds"]
    assert "delivery" not in row and "issue" not in row
