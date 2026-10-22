"""Wall-clock duration never stops production or scoring; explicit stops do."""

import asyncio
from contextlib import nullcontext
import json
import os
from pathlib import Path

import pytest

from scripts import run_probe_variance as variance
from test_probe_variance import (Owner, fake_process_identity, finish_production, manifest_case, options,
                                 ready_score_rounds)


@pytest.mark.parametrize("legacy_dispatch", [False, True])
def test_dispatcher_keeps_production_and_scoring_after_thirteen_hours(
    manifest_case, monkeypatch, legacy_dispatch,
):
    path, data, setting = manifest_case
    root = Path(data["state_root"])
    if legacy_dispatch:
        # A historical timestamp is no longer a scheduling control.
        variance.save(root / "dispatch.json", {
            "manifest": str(path), "deadline_epoch": 1,
        })
    clock = [1790000000.0]
    monkeypatch.setattr(variance.time, "time", lambda: clock[0])
    monkeypatch.setattr(variance.time, "sleep", lambda _: None)
    monkeypatch.setattr(variance, "resources", lambda *args: {
        "memory": {"MemAvailable_bytes": 64 * 1024 ** 3},
        "swap_pages": {"pswpout": 0}, "disk_free_bytes": 100 * 1024 ** 3,
    })
    scored = set()
    monkeypatch.setattr(variance, "verified_scores", lambda run: run in scored)
    launched = []

    class Child:
        pid = os.getpid()
        returncode = 0

        def __init__(self, command, **kwargs):
            launched.append(command)
            assert not any(arg.startswith("--deadline") for arg in command)
            if "--worker-setting" in command:
                self.run = None
                clock[0] += 13 * 3600
                ready_score_rounds(data, setting)
                variance.save(variance.setting_dir(data, setting) / "worker.json", {"status": "production_complete"})
            else:
                self.run = Path(command[command.index("--run-dir") + 1])

        def poll(self):
            if self.run is not None:
                scored.add(self.run)
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(variance.subprocess, "Popen", Child)
    fake_process_identity(monkeypatch)
    config = variance.parser().parse_args(["--manifest", str(path), "--execute", "--score"])
    assert variance.run_dispatcher(path, data, config) == 0
    assert len(launched) == 3 and len(scored) == 2
    assert not (root / "stop.json").exists()
    for name in ("dispatch.json", "status.json"):
        assert "deadline" not in json.dumps(variance.read(root / name))
    for number in (2, 3):
        assert variance.read(variance.setting_dir(data, setting) / f"repeat{number:02}/score.json")["status"] == "complete"


@pytest.mark.asyncio
async def test_worker_keeps_both_rounds_after_thirteen_hours(manifest_case, monkeypatch):
    _, data, setting = manifest_case
    clock = [1790000000.0]
    monkeypatch.setattr(variance.time, "time", lambda: clock[0])
    executed = []

    async def execute(args, *, shared_probe_memory_owner, **kwargs):
        executed.append(Path(args.plan).parent.name)
        clock[0] += 13 * 3600
        await asyncio.sleep(.3)  # Allow the stop-file watcher to run.
        assert not shared_probe_memory_owner.is_stop_requested()
        finish_production(Path(args.plan).parent)
        return 0

    state = await variance.run_setting(data, setting, options(), owner_factory=Owner, execute=execute)
    assert executed == ["repeat02", "repeat03"]
    assert state["status"] == "production_complete" and Owner.instances[-1].closed


def test_worker_cli_requires_no_clock_parameter(manifest_case, monkeypatch):
    path, data, setting = manifest_case
    called = []

    async def run(manifest, selected, config):
        called.append(selected["id"])
        return {"status": "production_complete"}

    monkeypatch.setenv("MEMPA_WORKSPACE_ROOT", data["workspace_root"])
    monkeypatch.setattr("sys.argv", [
        "run_probe_variance", "--manifest", str(path), "--worker-setting", setting["id"],
    ])
    monkeypatch.setattr(variance, "run_setting", run)
    monkeypatch.setattr("harness.runtime_activity.observe_model_calls", lambda _: nullcontext())
    assert variance.main() == 0
    assert called == [setting["id"]]


def test_preview_has_no_time_limit(manifest_case, monkeypatch, capsys):
    path, data, _ = manifest_case
    monkeypatch.setenv("MEMPA_WORKSPACE_ROOT", data["workspace_root"])
    monkeypatch.setattr(variance, "capacity_report", lambda *a, **k: {"execution": "not_started"})
    monkeypatch.setattr("sys.argv", ["run_probe_variance", "--manifest", str(path)])
    assert variance.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert "deadline" not in json.dumps(report)
    assert not (Path(data["state_root"]) / "dispatch.json").exists()


@pytest.mark.parametrize("option", ["--deadline-hours=12", "--deadline-epoch=1790140929.3274586"])
def test_removed_clock_options_are_rejected(option):
    with pytest.raises(SystemExit) as exc:
        variance.parser().parse_args([option])
    assert exc.value.code == 2


def test_score_admission_obeys_stop_requested_during_validation(manifest_case, monkeypatch):
    _, data, setting = manifest_case
    ready_score_rounds(data, setting)

    def finish_validation(run):
        variance.save(Path(data["state_root"]) / "stop.json", {"reason": "user_requested"})
        return False

    monkeypatch.setattr(variance, "verified_scores", finish_validation)
    monkeypatch.setattr(variance.subprocess, "Popen", lambda *a, **k: pytest.fail("Stopped scorer started"))
    report = variance.score_ready_rounds(data, options(), {}, stopped=False)
    assert report["pending"] and report["active"] == 0
    assert not list(variance.setting_dir(data, setting).glob("repeat*/score.json"))
