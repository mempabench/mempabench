"""`harness/run_layout.py::runs_root`: one configurable answer, one default.

The default is nailed down first and hardest. Seven literals over five modules
were collapsed into this function, so if it ever answers something other than
`workspace/runs` for an unconfigured tree, the benchmark writes one place and
Experiment Admin looks in another.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import run_layout
from harness.run_layout import archive_root, runs_root

#: The repository root, reached from this test file rather than from the module
#: under test, so the pinned default below cannot follow a typo in
#: `run_layout.PROJECT_ROOT`.
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def unconfigured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Neither source says anything, which is the state of the repo today."""
    monkeypatch.delenv(run_layout.RUNS_ROOT_ENV, raising=False)
    monkeypatch.setattr(
        run_layout, "RUNS_LAYOUT_CONFIG", tmp_path / "absent.yaml"
    )


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "runs_layout.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_unconfigured_root_is_the_pre_configuration_path(
    unconfigured: None,
) -> None:
    """Spelled out rather than built from the module's own constants: these are
    the literals the seven call sites carried before they were collapsed, and a
    test that recomputed them from `DEFAULT_RUNS_ROOT` would follow a typo."""
    assert runs_root() == REPO_ROOT / "workspace" / "runs"
    assert archive_root() == REPO_ROOT / "workspace" / "runs_archive"


def test_the_unconfigured_root_is_the_repo_root_and_not_the_cwd(
    unconfigured: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert runs_root() == run_layout.PROJECT_ROOT / "workspace" / "runs"


def test_the_environment_variable_moves_the_root(
    unconfigured: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(run_layout.RUNS_ROOT_ENV, str(tmp_path / "elsewhere"))

    assert runs_root() == tmp_path / "elsewhere"
    assert archive_root() == tmp_path / "runs_archive"


def test_the_config_file_moves_the_root(
    unconfigured: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_layout,
        "RUNS_LAYOUT_CONFIG",
        _config(tmp_path, f"runs_root: {tmp_path / 'from_file'}\n"),
    )

    assert runs_root() == tmp_path / "from_file"
    assert archive_root() == tmp_path / "runs_archive"


def test_the_environment_wins_over_the_config_file(
    unconfigured: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_layout,
        "RUNS_LAYOUT_CONFIG",
        _config(tmp_path, f"runs_root: {tmp_path / 'from_file'}\n"),
    )
    monkeypatch.setenv(run_layout.RUNS_ROOT_ENV, str(tmp_path / "from_env"))

    assert runs_root() == tmp_path / "from_env"


@pytest.mark.parametrize("body", ["", "# comments only\n", "not_a_mapping\n"])
def test_a_config_file_that_configures_nothing_leaves_the_default(
    unconfigured: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    """The shipped `configs/runs_layout.yaml` holds comments only, so this is
    the state of a checkout that has not been pointed anywhere."""
    monkeypatch.setattr(
        run_layout, "RUNS_LAYOUT_CONFIG", _config(tmp_path, body)
    )

    assert runs_root() == run_layout.PROJECT_ROOT / "workspace" / "runs"


def test_the_shipped_config_file_configures_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(run_layout.RUNS_ROOT_ENV, raising=False)
    assert run_layout.RUNS_LAYOUT_CONFIG.is_file()
    assert run_layout._configured_runs_root() is None


def test_a_relative_configured_value_is_read_against_the_repo_root(
    unconfigured: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative value used to mean "wherever the process is standing", which
    is how `harness/state_runtime.py`'s default could land a run's state under
    whatever directory the operator typed the command in."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(run_layout.RUNS_ROOT_ENV, "somewhere/runs")

    assert runs_root() == run_layout.PROJECT_ROOT / "somewhere" / "runs"
    assert runs_root() != (tmp_path / "somewhere" / "runs")


def test_a_configured_value_goes_through_expanduser(
    unconfigured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(run_layout.RUNS_ROOT_ENV, "~/MemPA-runs/runs")

    assert runs_root() == (Path.home() / "MemPA-runs" / "runs").resolve()


def test_expanduser_applies_to_the_config_file_too(
    unconfigured: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_layout,
        "RUNS_LAYOUT_CONFIG",
        _config(tmp_path, "runs_root: ~/MemPA-runs/runs\n"),
    )

    assert runs_root() == (Path.home() / "MemPA-runs" / "runs").resolve()


def test_an_empty_environment_value_is_not_a_configured_root(
    unconfigured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported-but-blank variable is how a shell hands over "unset"."""
    monkeypatch.setenv(run_layout.RUNS_ROOT_ENV, "")

    assert runs_root() == run_layout.PROJECT_ROOT / "workspace" / "runs"


def test_the_root_is_read_at_call_time_and_not_at_import(
    unconfigured: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = runs_root()
    monkeypatch.setenv(run_layout.RUNS_ROOT_ENV, str(tmp_path / "late"))

    assert runs_root() == tmp_path / "late" != before
