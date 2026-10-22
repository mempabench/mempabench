"""`scripts/diagnostics/mint_run_plan_only.py` mints a frozen plan and starts
nothing: no run directory, no socket, no judge preflight, no ledger, no
memory adapter."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from harness.run_plan import load_run_plan
from scripts import run_orchestrator
from scripts.diagnostics import mint_run_plan_only
from test_harness.test_run_plan_builder_probes import (  # noqa: F401 -- fixture
    ACC_COUNT,
    PERSONA,
    _write_probe_line,
    synthetic_root,
)


def _boom(what: str):
    def raise_(*args, **kwargs):
        raise AssertionError(f"{what} must not be touched by a plan-only mint")

    return raise_


@pytest.fixture
def sealed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every road out of the process is a trap: any connection attempt, the
    judge preflight, the ledger and the memory registry all fail the test."""
    monkeypatch.setattr(socket.socket, "connect", _boom("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", _boom("socket.create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", _boom("socket.getaddrinfo"))
    monkeypatch.setattr("simulator.judge.session.preflight", _boom("judge_preflight"))
    monkeypatch.setattr("harness.run_ledger.RunLedger.__init__", _boom("RunLedger"))
    monkeypatch.setattr("memory.registry.default_registry", _boom("memory registry"))


def _runs_listing(root: Path | str) -> list[str]:
    runs = Path(root) / "workspace/runs"
    return sorted(entry.name for entry in runs.iterdir()) if runs.exists() else []


def test_mint_writes_the_plan_and_starts_nothing(
    synthetic_root: Path,
    sealed: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_probe_line(synthetic_root)
    out_dir = tmp_path / "mint_out"
    real_runs_before = _runs_listing(run_orchestrator.PROJECT_ROOT)
    fixture_runs_before = _runs_listing(synthetic_root)

    rc = mint_run_plan_only.main(
        [
            "--run-id", "mint_test",
            "--persona-id", PERSONA,
            "--acc-range", f"1-{ACC_COUNT}",
            "--memory-condition", "simple_rag",
            "--pa-model", "deepseek/deepseek-v4-flash",
            "--simulator-model", "google/gemini-3.8-flash",
            "--pin-all-accumulations",
            "--out-dir", str(out_dir),
        ]
    )

    assert rc == 0
    plan_path = out_dir / "mint_test" / "run_plan.yaml"
    assert plan_path.is_file()
    assert (out_dir / "mint_test" / "resolved_gt.yaml").is_file()
    plan = load_run_plan(plan_path, project_root=synthetic_root)
    kinds = [step.kind for step in plan.steps]
    assert kinds.count("accumulation") == ACC_COUNT
    assert kinds.count("final_probe") == 28
    assert kinds.count("pre_event_probe") == 5
    assert plan.memory_condition == "simple_rag"
    assert plan.mem0_runtime_config is not None
    assert plan.simulator_runtime.simulator_model == "google/gemini-3.8-flash"
    # Nothing started: no run directory anywhere, in the real tree or the fixture.
    assert _runs_listing(run_orchestrator.PROJECT_ROOT) == real_runs_before
    assert _runs_listing(synthetic_root) == fixture_runs_before
    out = capsys.readouterr().out
    assert f"ix_contract_version={plan.ix_contract_version}" in out
    assert "probe_steps=33 acc_steps=99" in out


def test_a_new_plan_freezes_the_shipped_24576_token_pa_output_cap(
    synthetic_root: Path,
    sealed: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PA and File memory consolidation both answer within the frozen
    `pa_max_tokens`; a new plan must carry the shipped `budgets.pa_max_tokens`
    so that a long consolidation is not cut off at the cap."""
    monkeypatch.delenv("MEMPA_RUN_CONFIG", raising=False)
    _write_probe_line(synthetic_root)
    out_dir = tmp_path / "mint_out"

    rc = mint_run_plan_only.main(
        [
            "--run-id", "cap_test",
            "--persona-id", PERSONA,
            "--acc-range", f"1-{ACC_COUNT}",
            "--memory-condition", "nanobot_file_memory",
            "--pa-model", "deepseek/deepseek-v4-flash",
            "--consolidation-model", "deepseek/deepseek-v4-flash",
            "--out-dir", str(out_dir),
        ]
    )

    assert rc == 0
    plan_path = out_dir / "cap_test" / "run_plan.yaml"
    assert load_run_plan(plan_path, project_root=synthetic_root).pa_max_tokens == 24576
    assert "pa_max_tokens: 24576\n" in plan_path.read_text(encoding="utf-8")


def test_an_out_dir_under_workspace_runs_is_refused(
    synthetic_root: Path, sealed: None
) -> None:
    with pytest.raises(ValueError, match="never writes into a formal runs root"):
        mint_run_plan_only.main(
            [
                "--run-id", "mint_test",
                "--persona-id", PERSONA,
                "--acc-range", "1-3",
                "--out-dir", str(synthetic_root / "workspace/runs/mint_test"),
            ]
        )
    assert not (synthetic_root / "workspace").exists()


def test_the_guard_follows_the_configured_runs_root(
    synthetic_root: Path,
    sealed: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard has to hold the tree a launch actually writes into.

    Left as a literal `workspace/runs`, it would keep standing over the old
    location after the runs root moved and leave the new one -- the one real
    runs are in -- unguarded.
    """
    moved = tmp_path / "MemPA-runs" / "runs"
    monkeypatch.setenv("MEMPA_RUNS_ROOT", str(moved))

    assert moved in mint_run_plan_only._guarded_runs_roots()
    with pytest.raises(ValueError, match="never writes into a formal runs root"):
        mint_run_plan_only.main(
            [
                "--run-id", "mint_test",
                "--persona-id", PERSONA,
                "--acc-range", "1-3",
                "--out-dir", str(moved / "model" / "cond" / PERSONA / "mint_test"),
            ]
        )
    assert not moved.exists()
