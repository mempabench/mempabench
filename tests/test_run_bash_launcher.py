from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "Run_bash" / "run_real_pa_session.sh"


def _run_launcher(
    tmp_path: Path,
    *,
    overrides: dict[str, str] | None = None,
    arguments: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_path = tmp_path / "uv-arguments.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_FILE": str(capture_path),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        }
    )
    for name in ("PERSONA_ID", "ACC_RANGE", "MEMORY_CONDITION", "PA_MODEL"):
        env.pop(name, None)
    env.update(overrides or {})

    completed = subprocess.run(
        [str(LAUNCHER), *arguments],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    captured_arguments = (
        capture_path.read_text(encoding="utf-8").splitlines()
        if capture_path.exists()
        else []
    )
    return completed, captured_arguments


def test_launcher_uses_documented_defaults(tmp_path: Path) -> None:
    completed, arguments = _run_launcher(tmp_path)
    model_config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "pa_models.yaml").read_text(encoding="utf-8")
    )

    assert completed.returncode == 0, completed.stderr
    assert arguments == [
        "run",
        "python",
        "scripts/run_orchestrator.py",
        "--persona-id",
        "user_a",
        "--acc-range",
        "1-10",
        "--memory-condition",
        "nanobot_file_memory",
        "--pa-model",
        model_config["default"],
    ]


def test_launcher_applies_environment_overrides_and_forwards_arguments(
    tmp_path: Path,
) -> None:
    completed, arguments = _run_launcher(
        tmp_path,
        overrides={
            "PERSONA_ID": "user_b",
            "ACC_RANGE": "7-8",
            "MEMORY_CONDITION": "no_memory",
            "PA_MODEL": "example/model",
        },
        arguments=("--run-id", "launcher-test", "--overwrite"),
    )

    assert completed.returncode == 0, completed.stderr
    assert arguments == [
        "run",
        "python",
        "scripts/run_orchestrator.py",
        "--persona-id",
        "user_b",
        "--acc-range",
        "7-8",
        "--memory-condition",
        "no_memory",
        "--pa-model",
        "example/model",
        "--run-id",
        "launcher-test",
        "--overwrite",
    ]


@pytest.mark.parametrize("acc_range", ["bad", "0-2", "3-2", "1", "1-"])
def test_launcher_rejects_invalid_accumulation_ranges(
    tmp_path: Path,
    acc_range: str,
) -> None:
    completed, arguments = _run_launcher(
        tmp_path,
        overrides={"ACC_RANGE": acc_range, "PA_MODEL": "example/model"},
    )

    assert completed.returncode == 2
    assert "ACC_RANGE" in completed.stderr
    assert arguments == []
