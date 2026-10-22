"""Variance scheduling tests use local fake runners and never invoke a model."""
import asyncio
import copy
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import yaml

from scripts import run_probe_variance as variance
from test_harness.test_orchestrator import _plan as _probe_template_plan
from test_harness.test_run_plan import _plan as _frozen_plan


@pytest.fixture
def manifest_case(tmp_path, monkeypatch):
    # The repeat source record has its own tests; this scheduler fixture
    # writes the record and reads it back as is.
    from harness import probe_repeat_source
    monkeypatch.setattr(probe_repeat_source, "load_repeat_source",
                        lambda run: variance.read(Path(run) / "probe_repeat_source.json"))
    workspace = tmp_path / "workspace"
    source = workspace / "source"
    source.mkdir(parents=True)
    # A frozen plan whose one accumulation step, acc_001, is every Probe's pin.
    plan = _frozen_plan()
    plan["run_id"] = "source"
    probe = _probe_template_plan().steps[1].model_dump(mode="json")
    for prefix, kind, count in (("pre", "pre_event_probe", 5), ("fin", "final_probe", 28)):
        plan["steps"].extend({**probe, "step_id": f"{prefix}_{index:02d}", "kind": kind,
                              "script_path": f"{prefix}_{index:02d}.yaml",
                              "before_acc_num": 2 if kind == "pre_event_probe" else None,
                              "probe_overlay": {"owner_probe_id": f"{prefix}_{index:02d}"}}
                             for index in range(1, count + 1))
    (source / "run_plan.yaml").write_text(yaml.safe_dump(plan))
    pin = source / "checkpoints" / "acc_001"
    pin.mkdir(parents=True)
    (pin / "metadata.json").write_text('{"pinned":true}\n')
    reference = {"run_dir": str(source)}
    finish_production(source)
    (source / "resolved_gt.yaml").write_text("frozen: original\n")
    setting = {"id": "a-model-rag", "source": reference, "repeats": []}
    for number in (2, 3):
        run = workspace / f"repeat{number:02d}"
        run.mkdir()
        repeated = copy.deepcopy(plan)
        repeated["run_id"] = run.name
        repeated["pa_session_key"] = "source_main"
        for step in repeated["steps"]:
            if step["kind"] != "accumulation":
                step["borrow_start_from"] = "source/acc_001"
        path = run / "run_plan.yaml"
        path.write_text(yaml.safe_dump(repeated))
        (run / "resolved_gt.yaml").write_bytes((source / "resolved_gt.yaml").read_bytes())
        variance.save(run / "probe_repeat_source.json", {"repeat": number,
                      "source_run_dir": str(source)})
        setting["repeats"].append({"repeat": number, "source": copy.deepcopy(reference),
                                   "argv": ["--plan", str(path), "--persona-id", "user_a",
                                            "--pa-model", "openai/gpt-5.5", "--memory-condition", "mem0"]})
    data = {"schema": variance.SCHEMA, "workspace_root": str(workspace),
            "state_root": str(tmp_path / "state"), "settings": [setting]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path, data, setting


def finish_production(run):
    plan = yaml.safe_load((run / "run_plan.yaml").read_text())
    variance.save(run / "ledger.json", {"run_id": plan["run_id"],
                  "steps": {step["step_id"]: {"status": "done"} for step in plan["steps"]}})


class Owner:
    instances = []

    def __init__(self, root, **kwargs):
        self.root, self.options = root, kwargs
        self.stopping = False
        self.closed = False
        self.calls = []
        Owner.instances.append(self)

    def request_stop(self):
        self.stopping = True

    def is_stop_requested(self):
        return self.stopping

    async def close(self):
        self.closed = True
        return True


def options():
    return SimpleNamespace(restore_concurrency=1, score_concurrency=2, probe_concurrency=4)


def test_manifest_binds_source_and_rejects_scientific_changes(manifest_case):
    path, data, setting = manifest_case
    assert variance.validate_manifest(path) == data
    repeat = setting["repeats"][0]
    plan_path = Path(repeat["argv"][1])
    plan = yaml.safe_load(plan_path.read_text())
    plan["steps"][0]["target_setting"] = "changed"
    plan_path.write_text(yaml.safe_dump(plan))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="scientific"):
        variance.validate_manifest(path)


def test_manifest_refuses_source_run_reuse_and_reset_flags(manifest_case):
    path, data, setting = manifest_case
    repeat = setting["repeats"][0]
    repeat["argv"].append("--overwrite")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="without reset"):
        variance.validate_manifest(path)
    repeat["argv"].pop()
    repeat["argv"][1] = str(Path(setting["source"]["run_dir"]) / "run_plan.yaml")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="new run identity"):
        variance.validate_manifest(path)


def test_claim_rejects_duplicate_and_never_unlinks_guard(tmp_path):
    path = tmp_path / "guard.lock"
    with variance.claim(path):
        inode = path.stat().st_ino
        assert variance.occupied(path)
        with pytest.raises(BlockingIOError), variance.claim(path):
            pass
    assert not variance.occupied(path)
    with variance.claim(path):
        assert path.stat().st_ino == inode


@pytest.mark.asyncio
async def test_two_rounds_use_one_owner_and_release_run_lock_before_handoff(manifest_case, monkeypatch):
    _, data, setting = manifest_case
    calls = []

    async def execute(args, *, shared_probe_memory_owner, shared_probe_stop):
        assert not shared_probe_memory_owner.closed
        calls.append(shared_probe_memory_owner)
        finish_production(Path(args.plan).parent)
        return 0

    original = variance.scoring_handoff

    def handoff(directory, setting, repeat, run, limit):
        with variance.producer_lock(run, workspace_root=Path(data["workspace_root"])):
            pass
        assert not calls[-1].closed
        original(directory, setting, repeat, run, limit)

    monkeypatch.setattr(variance, "scoring_handoff", handoff)
    state = await variance.run_setting(data, setting, options(), owner_factory=Owner, execute=execute)
    assert state["status"] == "production_complete"
    assert len(calls) == 2 and calls[0] is calls[1] and calls[0].closed
    directory = variance.setting_dir(data, setting)
    assert len(list(directory.glob("repeat*/round_complete.json"))) == 2
    assert not (directory / "delivery_event.json").exists()


@pytest.mark.asyncio
async def test_explicit_stop_and_existing_success_do_not_launch_models(manifest_case):
    _, data, setting = manifest_case
    finish_production(Path(setting["repeats"][0]["argv"][1]).parent)
    variance.save(Path(data["state_root"]) / "stop.json", {"reason": "user_requested"})

    async def forbidden(*args, **kwargs):
        pytest.fail("A complete or explicitly stopped round must not execute")

    state = await variance.run_setting(data, setting, options(), owner_factory=Owner, execute=forbidden)
    assert state["status"] == "stopped"
    assert state["rounds"]["2"] == "production_complete"
    assert Owner.instances[-1].closed


@pytest.mark.asyncio
async def test_ordinary_incomplete_round_does_not_block_independent_round(manifest_case):
    _, data, setting = manifest_case
    count = 0

    async def execute(args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            finish_production(Path(args.plan).parent)
        return 0

    state = await variance.run_setting(data, setting, options(), owner_factory=Owner, execute=execute)
    assert count == 2 and state["status"] == "incomplete"
    assert state["rounds"] == {"2": "incomplete", "3": "production_complete"}


@pytest.mark.asyncio
async def test_cancellation_drains_owner_and_preserves_failure_state(manifest_case):
    _, data, setting = manifest_case

    async def execute(*args, **kwargs):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await variance.run_setting(data, setting, options(), owner_factory=Owner, execute=execute)
    assert Owner.instances[-1].closed
    assert variance.read(variance.setting_dir(data, setting) / "worker.json")["status"] == "failed"


@pytest.mark.asyncio
async def test_unresolved_cleanup_blocks_new_consumers(manifest_case):
    _, data, setting = manifest_case
    variance.save(Path(data["state_root"]) / "stop.json", {"reason": "user_requested"})

    class PendingOwner(Owner):
        async def close(self):
            return False

    state = await variance.run_setting(data, setting, options(), owner_factory=PendingOwner, execute=None)
    assert state["status"] == "cleanup_pending"
    with pytest.raises(RuntimeError, match="cleanup remains unresolved"):
        await variance.run_setting(data, setting, options(), owner_factory=Owner, execute=None)


def test_snapshot_counts_held_guards_and_running_probes_only(manifest_case):
    _, data, setting = manifest_case
    run = Path(setting["repeats"][0]["argv"][1]).parent
    variance.save(run / "ledger.json", {"steps": {"fin_01": {"status": "running"}, "pre_01": {"status": "done"}}})
    variance.save(variance.setting_dir(data, setting) / "worker.json", {
        "pid": 9999, "status": "running", "current_run_dir": str(run)})
    assert variance.snapshot(data)["active_settings"] == 0
    with variance.claim(variance.guard_path(data, setting)):
        state = variance.snapshot(data)
        assert state["active_settings"] == 1 and state["active_probes"] == 1


def test_parallel_round_option_rejects_widths_above_two(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["run_probe_variance", "--manifest", str(tmp_path / "missing"), "--rounds-per-setting", "3"])
    with pytest.raises(ValueError, match="W must be 1 or 2"):
        variance.main()


def test_dispatcher_reaches_eight_setting_workers_without_duplicating_claims(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = [{"id": f"setting-{i}", "source": {"run_dir": str(workspace / f"source-{i}")}}
                for i in range(10)]
    data = {"workspace_root": str(workspace), "state_root": str(tmp_path / "state"), "settings": settings}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    active = maximum = 0
    launched = []

    class Child:
        def __init__(self, argv, **kwargs):
            nonlocal active, maximum
            label = argv[argv.index("--worker-setting") + 1]
            assert argv[argv.index("--rounds-per-setting") + 1] == "2"
            self.setting = next(row for row in settings if row["id"] == label)
            self.lock = variance.claim(variance.guard_path(data, self.setting))
            self.lock.__enter__()
            self.calls = 0
            self.returncode = None
            active += 1
            maximum = max(maximum, active)
            launched.append(label)
            variance.save(variance.setting_dir(data, self.setting) / "worker.json",
                          {"status": "running", "pid": 10000 + len(launched)})

        def poll(self):
            nonlocal active
            self.calls += 1
            if self.calls > 1 and self.returncode is None:
                self.returncode = 0
                active -= 1
                self.lock.__exit__(None, None, None)
                variance.save(variance.setting_dir(data, self.setting) / "worker.json", {"status": "production_complete"})
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(variance.subprocess, "Popen", Child)
    monkeypatch.setattr(variance.time, "sleep", lambda _: None)
    monkeypatch.setattr(variance, "resources", lambda *args: {
        "memory": {"MemAvailable_bytes": 64 * 1024 ** 3}, "swap_pages": {"pswpout": 0}, "disk_free_bytes": 100 * 1024 ** 3})
    config = variance.parser().parse_args(["--manifest", str(path), "--execute", "--rounds-per-setting", "2"])
    assert variance.run_dispatcher(path, data, config) == 0
    assert variance.batch_capacity(workspace / ".probe-batch-slots") == 8
    assert maximum == 8 and active == 0 and len(launched) == len(set(launched)) == 10
    events = [json.loads(line) for line in (Path(data["state_root"]) / "activity.jsonl").read_text().splitlines()]
    assert max(row["active_settings"] for row in events) == 8


def test_restart_preserves_explicit_stop_without_launching(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = {"workspace_root": str(workspace), "state_root": str(tmp_path / "state"),
            "settings": [{"id": "pending", "source": {"run_dir": str(workspace / "source")}}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    variance.save(Path(data["state_root"]) / "dispatch.json",
                  {"manifest": str(path)})
    variance.save(Path(data["state_root"]) / "stop.json", {"reason": "user_requested"})
    monkeypatch.setattr(variance.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Stopped batch launched a worker"))
    monkeypatch.setattr(variance, "resources", lambda *args: {
        "memory": {"MemAvailable_bytes": 64 * 1024 ** 3}, "swap_pages": {"pswpout": 0}, "disk_free_bytes": 100 * 1024 ** 3})
    config = variance.parser().parse_args(["--manifest", str(path), "--execute"])
    assert variance.run_dispatcher(path, data, config) == 1
    assert variance.read(Path(data["state_root"]) / "stop.json")["reason"] == "user_requested"


@pytest.mark.asyncio
async def test_first_sigterm_finishes_current_round_but_never_starts_next(manifest_case):
    _, data, setting = manifest_case
    from scripts.run_orchestrator import GracefulStop
    executed = []

    async def execute(args, *, shared_probe_memory_owner, shared_probe_stop):
        executed.append(args.plan)
        stop = GracefulStop(SimpleNamespace(current_step="fin_28"), on_stop=shared_probe_memory_owner.request_stop)
        # Invoke the actual handler without delivering a signal to pytest.
        stop.handle(signal.SIGTERM, None)
        finish_production(Path(args.plan).parent)
        return 0

    state = await variance.run_setting(data, setting, options(), owner_factory=Owner, execute=execute)
    assert len(executed) == 1
    assert state["status"] == "stopped" and state["rounds"] == {"2": "production_complete"}
    assert Owner.instances[-1].closed


@pytest.mark.asyncio
async def test_explicit_stop_while_current_round_runs_drains_and_stops_next(manifest_case):
    _, data, setting = manifest_case
    executed = []

    async def execute(args, *, shared_probe_memory_owner, shared_probe_stop):
        executed.append(args.plan)
        variance.save(Path(data["state_root"]) / "stop.json", {"reason": "user_requested"})
        for _ in range(10):
            if shared_probe_memory_owner.is_stop_requested():
                break
            await asyncio.sleep(.05)
        assert shared_probe_memory_owner.is_stop_requested()
        finish_production(Path(args.plan).parent)
        return 0

    state = await variance.run_setting(data, setting, options(), owner_factory=Owner, execute=execute)
    assert len(executed) == 1 and state["status"] == "stopped"
    assert Owner.instances[-1].closed


def test_process_inventory_limits_output_to_safe_fields(monkeypatch):
    argv = ["python", "-m", "scripts.run_locked", "--run-id", "accepted-source",
            "--probe-concurrency=4", "--api-key", "must-not-appear-in-report"]
    listing = f"  123 {' '.join(argv)}\n  124 python -m http.server\n"
    calls = []
    monkeypatch.setattr(variance.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(stdout=listing))
    monkeypatch.setattr("harness.runtime_activity.process_start", lambda pid: f"started {pid}")
    rows = variance.existing_producers()
    assert calls == [["ps", "-A", "-ww", "-o", "pid=,args="]]
    assert len(rows) == 1 and rows[0]["role"] == "producer" and rows[0]["pid"] == 123
    assert rows[0]["arguments"] == {"run-id": "accepted-source", "probe-concurrency": "4"}
    assert rows[0]["process_start"] == "started 123"
    assert "must-not-appear" not in json.dumps(rows)


def test_macos_counters_give_the_pressure_gate_available_memory_and_swap_outs(monkeypatch):
    # Without these two readings the dispatcher would pause admission forever on macOS.
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              1000.
Pages active:                            9000.
Pages inactive:                          3000.
Pages speculative:                        200.
"Translation faults":                   12345.
Swapins:                                    7.
Swapouts:                                  11.
"""
    outputs = {"vm_stat": vm_stat, "ps": "  2048\n"}
    monkeypatch.setattr(variance.subprocess, "run", lambda command, **kwargs:
                        SimpleNamespace(stdout=outputs[command[0]]))
    memory, swap, cpu, processes = variance.darwin_counters({42})
    assert memory["MemAvailable_bytes"] == (1000 + 200 + 3000) * 16384
    assert swap == {"pswpin": 7, "pswpout": 11}
    assert cpu == [] and processes == {"42": {"VmRSS_bytes": 2048 * 1024}}


def test_capacity_report_is_read_only_and_never_claims_hardware_validation(tmp_path, monkeypatch):
    workspace = tmp_path / "not-created"
    monkeypatch.setattr(variance, "existing_producers", lambda: [])
    monkeypatch.setattr(variance, "cgroup_limits", lambda: [{"path": "test", "memory.max": 1000,
                        "memory.current": 300, "cpu.max": "200000 100000"}])
    config = variance.parser().parse_args(["--capacity-report", "--workspace-root", str(workspace), "--sample-seconds", "0"])
    before = list(tmp_path.iterdir())
    report = variance.capacity_report(workspace, config)
    assert list(tmp_path.iterdir()) == before
    assert report["candidate"] == {"S": 8, "W": 1, "P": 4, "R": 1, "J_requested": 2,
                                   "batch_slots": 8, "probe_task_upper_bound": 32}
    assert report["capacity_status"] == "candidate_not_validated_by_real_production"
    assert report["highest_verified_stable_S"] is None
    assert report["effective_memory_headroom_bytes"] == 700 and report["effective_cpu_quota"] == 2
    assert report["execution"] == "not_started" and report["swap_page_delta"] is None


def test_capacity_cli_works_before_manifest_preparation(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "not-created"
    monkeypatch.setattr("sys.argv", ["run_probe_variance", "--capacity-report", "--workspace-root", str(workspace), "--sample-seconds", "0"])
    monkeypatch.setattr(variance, "existing_producers", lambda: [])
    assert variance.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["workspace_exists"] is False and not workspace.exists()


def test_pressure_uses_consecutive_swap_activity_not_historical_occupancy(tmp_path):
    config = variance.parser().parse_args(["--capacity-report"])
    measured = {"memory": {"MemAvailable_bytes": 32 * 1024 ** 3, "SwapFree_bytes": 0},
                "disk_free_bytes": 100 * 1024 ** 3, "swap_pages": {"pswpout": 100}}
    assert not variance.update_pressure(tmp_path, measured, config)["paused"]
    for index in (1, 2):
        measured["swap_pages"]["pswpout"] += 1
        value = variance.update_pressure(tmp_path, measured, config)
        assert value["consecutive_swap_out_samples"] == index and not value["paused"]
    measured["swap_pages"]["pswpout"] += 1
    assert variance.update_pressure(tmp_path, measured, config)["reasons"] == ["sustained_swap_out"]
    assert not variance.update_pressure(tmp_path, measured, config)["paused"]
    measured["memory"]["MemAvailable_bytes"] = 7 * 1024 ** 3
    measured["disk_free_bytes"] = 19 * 1024 ** 3
    assert variance.update_pressure(tmp_path, measured, config)["reasons"] == [
        "memory_below_admission_threshold", "disk_below_admission_threshold"]


def fake_process_identity(monkeypatch):
    """Fake Popen children are not processes, so `ps` cannot identify them.

    A negative PID stands for a process that is gone.
    """
    monkeypatch.setattr("harness.runtime_activity.process_start",
                        lambda pid: None if pid < 0 else f"fake start {pid}")


def ready_score_rounds(data, setting):
    for repeat in setting["repeats"]:
        run = Path(repeat["argv"][1]).parent
        finish_production(run)
        variance.scoring_handoff(variance.setting_dir(data, setting), setting, repeat, run, 2)


def test_score_jobs_obey_J_and_verify_outputs_before_completion(manifest_case, monkeypatch):
    _, data, setting = manifest_case
    ready_score_rounds(data, setting)
    config = variance.parser().parse_args(["--capacity-report", "--score-concurrency", "1", "--score"])
    completed = set()
    monkeypatch.setattr(variance, "verified_scores", lambda run: run in completed)
    launched = []

    class Child:
        def __init__(self, argv, **kwargs):
            import os
            self.pid = os.getpid()
            self.returncode = None
            self.run = Path(argv[argv.index("--run-dir") + 1])
            assert argv[-1] == "--variance-retries"
            launched.append(self)

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(variance.subprocess, "Popen", Child)
    fake_process_identity(monkeypatch)
    children = {}
    report = variance.score_ready_rounds(data, config, children, stopped=False)
    assert report["active"] == len(children) == len(launched) == 1
    launched[0].returncode = 0
    completed.add(launched[0].run)
    report = variance.score_ready_rounds(data, config, children, stopped=False)
    assert len(launched) == 2 and report["active"] == 1
    state = variance.read(variance.setting_dir(data, setting) / "repeat02" / "score.json")
    assert state["status"] == "complete" and state["outputs_verified"]
    # Exit 0 with missing scores is still a failure and never an automatic retry.
    launched[1].returncode = 0
    variance.score_ready_rounds(data, config, children, stopped=False)
    variance.score_ready_rounds(data, config, children, stopped=False)
    assert len(launched) == 2
    assert variance.read(variance.setting_dir(data, setting) / "repeat03" / "score.json")["status"] == "failed"


def test_a_reference_arm_scores_its_round_one_with_the_others(manifest_case, monkeypatch):
    """An arm's round 1 is a new run, not its source, so `--score` scores it too."""
    _, data, setting = manifest_case
    setting["arm"] = {"memory_condition": "no_memory", "pa_model": "test/model"}
    second = setting["repeats"][0]
    run = Path(second["argv"][1]).parent.with_name("repeat01")
    run.mkdir()
    plan = yaml.safe_load(Path(second["argv"][1]).read_text())
    (run / "run_plan.yaml").write_text(yaml.safe_dump({**plan, "run_id": run.name}))
    setting["repeats"].insert(0, {**copy.deepcopy(second), "repeat": 1,
                                  "argv": ["--plan", str(run / "run_plan.yaml"), *second["argv"][2:]]})
    ready_score_rounds(data, setting)
    config = variance.parser().parse_args(["--capacity-report", "--score-concurrency", "3", "--score"])
    monkeypatch.setattr(variance, "verified_scores", lambda _: False)
    launched = []

    class Child:
        def __init__(self, argv, **kwargs):
            import os
            self.pid, self.returncode = os.getpid(), None
            launched.append(Path(argv[argv.index("--run-dir") + 1]).name)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(variance.subprocess, "Popen", Child)
    fake_process_identity(monkeypatch)
    variance.score_ready_rounds(data, config, {}, stopped=False)
    assert sorted(launched) == ["repeat01", "repeat02", "repeat03"]


def test_unknown_scorer_crash_is_not_a_new_paid_opportunity(manifest_case, monkeypatch):
    _, data, setting = manifest_case
    ready_score_rounds(data, setting)
    config = variance.parser().parse_args(["--capacity-report", "--score"])
    for number in (2, 3):
        variance.save(variance.setting_dir(data, setting) / f"repeat{number:02d}" / "score.json",
                      {"status": "running", "pid": -1, "process_start": "lost"})
    monkeypatch.setattr(variance, "verified_scores", lambda _: False)
    monkeypatch.setattr(variance.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Unknown paid job was retried"))
    fake_process_identity(monkeypatch)
    report = variance.score_ready_rounds(data, config, {}, stopped=False)
    assert report["active"] == 0 and not report["pending"]
    assert variance.read(variance.setting_dir(data, setting) / "repeat02" / "score.json")["status"] == "interrupted"


def test_explicit_stop_blocks_new_score_jobs(manifest_case, monkeypatch):
    _, data, setting = manifest_case
    ready_score_rounds(data, setting)
    config = variance.parser().parse_args(["--capacity-report", "--score"])
    monkeypatch.setattr(variance, "verified_scores", lambda _: False)
    monkeypatch.setattr(variance.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Explicit stop started a score job"))
    report = variance.score_ready_rounds(data, config, {}, stopped=True)
    assert report["pending"] and report["active"] == 0


def release_setting(data, setting):
    directory = variance.setting_dir(data, setting)
    variance.save(directory / "worker.json", {"status": "production_complete",
                  "rounds": {"2": "production_complete", "3": "production_complete"}})
    source = Path(setting["source"]["run_dir"])
    record = {"source_run_dir": str(source), "checkpoint_id": "acc_001",
              "state": "released", "restore_count": 1,
              "consumers": {str(Path(repeat["argv"][1]).parent / "branches" / "fin_01" / "01"): "finished"
                            for repeat in setting["repeats"]}}
    variance.save(directory / "shared_memory" / "pin.json", record)
    return directory / "shared_memory" / "pin.json"


def test_interrupted_scorer_reenters_only_once_per_dispatch_with_ledger_policy(manifest_case, monkeypatch):
    _, data, setting = manifest_case
    ready_score_rounds(data, setting)
    config = variance.parser().parse_args(["--capacity-report", "--score"])
    config.scoring_dispatch_id = "dispatcher-1"
    monkeypatch.setattr(variance, "verified_scores", lambda _: False)
    launched = []

    class Child:
        pid = -1

        def __init__(self, argv, **kwargs):
            launched.append(argv)

        def poll(self):
            return None

    monkeypatch.setattr(variance.subprocess, "Popen", Child)
    fake_process_identity(monkeypatch)
    for number in (2, 3):
        variance.save(variance.setting_dir(data, setting) / f"repeat{number:02d}" / "score.json",
                      {"status": "interrupted", "pid": -1, "process_start": "lost"})
    children = {}
    variance.score_ready_rounds(data, config, children, stopped=False)
    assert len(launched) == 2
    # Simulate another lost worker in this same dispatch; no repeated loop.
    children.clear()
    variance.score_ready_rounds(data, config, children, stopped=False)
    assert len(launched) == 2
    state = variance.read(variance.setting_dir(data, setting) / "repeat02" / "score.json")
    assert state["status"] == "interrupted" and state["resume_dispatch_id"] == "dispatcher-1"


@pytest.mark.asyncio
async def test_parallel_rounds_overlap_restore_one_pin_and_release_after_both(manifest_case, tmp_path, monkeypatch):
    from harness.shared_probe_memory import SharedProbeMemoryOwner
    from test_harness.test_shared_probe_memory import _setting, _calls, _branch_session

    _, data, setting = manifest_case
    manager, admin = _setting(tmp_path / "real_checkpoint")
    config = options()
    config.rounds_per_setting = 2
    entered = {}
    both = asyncio.Event()
    first_finished = asyncio.Event()
    stops = []
    owners = []

    def factory(*args, **kwargs):
        owner = SharedProbeMemoryOwner(*args, **kwargs)
        owners.append(owner)
        return owner

    handoff = variance.scoring_handoff

    def after_round(directory, setting, repeat, run, limit):
        handoff(directory, setting, repeat, run, limit)
        if repeat["repeat"] == 2:
            first_finished.set()

    monkeypatch.setattr(variance, "scoring_handoff", after_round)
    original_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    async def execute(args, *, shared_probe_memory_owner, shared_probe_stop):
        run = Path(args.plan).parent
        branch = run / "private_probe"
        adapter = await shared_probe_memory_owner.branch(manager, "acc_108", branch)
        entered[run.name] = adapter
        stops.append(shared_probe_stop)
        if len(entered) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 5)
        assert len(_calls(admin, "import")) == 1
        live = variance.read(variance.setting_dir(data, setting) / "worker.json")
        if run.name == "repeat02":
            assert set(live["active_rounds"]) == {"2", "3"}
            _branch_session(branch).write_text("private repeat02 conversation")
        else:
            await asyncio.wait_for(first_finished.wait(), 5)
            assert not shared_probe_memory_owner._closed
            assert not _calls(admin, "delete")
            assert "private repeat02 conversation" not in _branch_session(branch).read_text()
        shared_probe_memory_owner.release_consumer(branch)
        finish_production(run)
        return 0

    state = await variance.run_setting(data, setting, config, owner_factory=factory, execute=execute)
    assert len(owners) == 1 and len(entered) == 2
    assert entered["repeat02"] is not entered["repeat03"]
    assert stops[0] is stops[1]
    assert len(_calls(admin, "import")) == len(_calls(admin, "delete")) == 1
    assert state["status"] == "production_complete" and state["active_rounds"] == {}
    assert state["rounds"] == {"2": "production_complete", "3": "production_complete"}
    details = state["round_details"]
    assert details["2"]["started_at"] <= details["3"]["started_at"] < details["2"]["finished_at"] <= details["3"]["finished_at"]
    assert all(signal.getsignal(sig) == handler for sig, handler in original_signals.items())


@pytest.mark.asyncio
async def test_parallel_round_failure_waits_for_independent_round(manifest_case):
    _, data, setting = manifest_case
    config = options()
    config.rounds_per_setting = 2
    both = asyncio.Event()
    release = asyncio.Event()
    entered = []

    async def execute(args, *, shared_probe_memory_owner, **kwargs):
        run = Path(args.plan).parent
        entered.append(run.name)
        if len(entered) == 2:
            both.set()
        await both.wait()
        if run.name == "repeat02":
            raise ValueError("independent failure")
        await release.wait()
        assert not shared_probe_memory_owner.closed
        finish_production(run)
        return 0

    task = asyncio.create_task(variance.run_setting(data, setting, config, owner_factory=Owner, execute=execute))
    await asyncio.wait_for(both.wait(), 5)
    await asyncio.sleep(.02)
    assert not task.done() and not Owner.instances[-1].closed
    release.set()
    with pytest.raises(ValueError, match="independent failure"):
        await task
    state = variance.read(variance.setting_dir(data, setting) / "worker.json")
    assert state["rounds"] == {"2": "failed", "3": "production_complete"}
    assert Owner.instances[-1].closed
    assert "finished_at" not in state["round_details"]["2"]
    assert state["round_details"]["3"]["finished_at"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["signal", "stop_file", "cancel"])
async def test_parallel_stop_preserves_both_inflight_rounds(manifest_case, reason):
    _, data, setting = manifest_case
    config = options()
    config.rounds_per_setting = 2
    both, release = asyncio.Event(), asyncio.Event()
    entered, stops = [], []

    async def execute(args, *, shared_probe_memory_owner, shared_probe_stop):
        entered.append(Path(args.plan).parent.name)
        stops.append(shared_probe_stop)
        if len(entered) == 2:
            both.set()
        await release.wait()
        assert shared_probe_memory_owner.is_stop_requested()
        assert not shared_probe_memory_owner.closed
        return 0  # Incomplete rounds remain resumable; no fabricated completion.

    task = asyncio.create_task(variance.run_setting(data, setting, config, owner_factory=Owner, execute=execute))
    await asyncio.wait_for(both.wait(), 5)
    if reason == "signal":
        handler = signal.getsignal(signal.SIGTERM)
        assert handler.__self__ is stops[0] is stops[1]
        handler(signal.SIGTERM, None)
    elif reason == "stop_file":
        variance.save(Path(data["state_root"]) / "stop.json", {"reason": "user_requested"})
    else:
        task.cancel()
    for _ in range(20):
        if Owner.instances[-1].is_stop_requested():
            break
        await asyncio.sleep(.02)
    assert Owner.instances[-1].is_stop_requested()
    assert not task.done() and not Owner.instances[-1].closed
    release.set()
    if reason == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert (await task)["status"] == "stopped"
    state = variance.read(variance.setting_dir(data, setting) / "worker.json")
    assert state["rounds"] == {"2": "stopped", "3": "stopped"}
    assert state["active_rounds"] == {} and Owner.instances[-1].closed


def test_snapshot_counts_both_rounds_without_double_counting_legacy_field(manifest_case, monkeypatch):
    _, data, setting = manifest_case
    runs = [Path(row["argv"][1]).parent for row in setting["repeats"]]
    for index, run in enumerate(runs):
        variance.save(run / "ledger.json", {"steps": {
            "fin_01": {"status": "running"}, "pre_01": {"status": "running" if index else "done"}}})
    variance.save(variance.setting_dir(data, setting) / "worker.json", {
        "pid": 9999, "status": "running", "current_run_dir": str(runs[0]),
        "active_rounds": {str(index + 2): {"run_dir": str(run)} for index, run in enumerate(runs)}})
    monkeypatch.setattr(variance, "active_batch_slots", lambda _: [{"pid": 9999}, {"pid": 9999}])
    with variance.claim(variance.guard_path(data, setting)):
        state = variance.snapshot(data)
    assert state["active_settings"] == 1 and state["active_rounds"] == 2 and state["active_probes"] == 3


def test_w2_capacity_keeps_global_slots_default_and_allows_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setattr(variance, "existing_producers", lambda: [])
    monkeypatch.setattr(variance, "cgroup_limits", lambda: [])
    config = variance.parser().parse_args(["--capacity-report", "--rounds-per-setting", "2", "--sample-seconds", "0"])
    report = variance.capacity_report(tmp_path, config)["candidate"]
    assert report["W"] == 2 and report["batch_slots"] == 8 and report["probe_task_upper_bound"] == 32
    config.batch_slots = 16
    report = variance.capacity_report(tmp_path, config)["candidate"]
    assert report["batch_slots"] == 16 and report["probe_task_upper_bound"] == 64


def test_shared_signal_controller_reports_both_independent_inflight_sets(capsys):
    from scripts.run_orchestrator import GracefulStop

    notified = []
    controller = GracefulStop(SimpleNamespace(current_step=None), on_stop=lambda: notified.append(True))
    first, second = {"pre_01"}, {"fin_02"}
    controller.round_in_flight.update(repeat02=first, repeat03=second)
    first.add("pre_02")
    controller.handle(signal.SIGTERM, None)
    message = capsys.readouterr().out
    assert notified == [True] and controller()
    assert "repeat02: pre_01, pre_02" in message and "repeat03: fin_02" in message


