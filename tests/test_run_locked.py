"""Real process exclusion and kernel release after an abnormal producer exit."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_locked import producer_lock, workspace_lock


def test_duplicate_process_is_rejected_and_killed_owner_releases_lock(tmp_path):
    run = tmp_path / "run"
    script = (
        "from pathlib import Path; import sys,time; "
        "from scripts.run_locked import producer_lock\n"
        "with producer_lock(Path(sys.argv[1])):\n"
        " print('locked',flush=True)\n"
        " time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(run)], stdout=subprocess.PIPE,
        text=True, cwd=Path(__file__).resolve().parents[1],
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="already owns"):
            with producer_lock(run):
                pytest.fail("duplicate producer acquired the run")
        with producer_lock(tmp_path / "independent-run"):
            pass
        child.kill()
        child.wait(timeout=10)
        with producer_lock(run):
            pass
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        child.stdout.close()


def test_lock_survives_run_directory_recreation_and_resolves_aliases(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(run, target_is_directory=True)
    with producer_lock(run):
        with pytest.raises(RuntimeError, match="already owns"):
            with producer_lock(alias):
                pass
        run.rmdir()
        run.mkdir()
        with pytest.raises(RuntimeError, match="already owns"):
            with producer_lock(run):
                pass


def test_maintenance_and_production_exclude_each_other(tmp_path):
    with workspace_lock(exclusive=True):
        with pytest.raises(RuntimeError, match="Workspace is busy"):
            with producer_lock(tmp_path / "run"):
                pass
    with producer_lock(tmp_path / "run"):
        with pytest.raises(RuntimeError, match="Workspace is busy"):
            with workspace_lock(exclusive=True):
                pass


def test_explicit_workspace_environment_uses_same_maintenance_lock(monkeypatch, tmp_path):
    shared = tmp_path / "actual-workspace"
    monkeypatch.setenv("MEMPA_WORKSPACE_ROOT", str(shared))
    with workspace_lock(exclusive=True, workspace_root=shared):
        with pytest.raises(RuntimeError, match="Workspace is busy"):
            with producer_lock(shared / "runs" / "one"):
                pass
    assert (shared / ".maintenance.lock").is_file()
