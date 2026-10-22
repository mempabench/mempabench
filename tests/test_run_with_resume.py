"""`scripts/run_with_resume.sh` reruns the identical `mempabench run` only while it exits 75.

A fake `uv` on PATH stands in for the run: try N plays the Nth entry of
FAKE_TRIES ("<exit code>[:<kind>:<step>]"), printing the stop line
`mempabench-run-stopped kind=<kind> step=<step>` as its last stderr line when a
kind is given, and records its arguments. A fake `sleep` records each wait
instead of waiting, so the tests see every try and every wait without sleeping.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess
import time

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_with_resume.sh"

_FAKE_UV = """\
#!/usr/bin/env bash
count="$FAKE_DIR/count"
n=$(( $(cat "$count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$count"
printf '%s\\0' "$@" > "$FAKE_DIR/call_$n"
echo "fake run output $n"
echo "fake run log $n" >&2
set -- $FAKE_TRIES
shift $(( n - 1 ))
IFS=: read -r code kind step <<< "$1"
if [ -n "$kind" ]; then
  echo "mempabench-run-stopped kind=$kind step=$step" >&2
fi
exit "$code"
"""

_FAKE_SLEEP = """\
#!/usr/bin/env bash
echo "$1" >> "$FAKE_DIR/sleeps"
"""

#: Spaces, a glob character and an empty argument must reach the run unchanged;
#: `--retain-checkpoints-every` shares `--re` with `--regenerate-plan` and must
#: not be refused.
RUN_ARGUMENTS = (
    "--persona-id", "user_a", "--run-id", "my run 01", "--retain-checkpoints-every", "10",
    "--note=*", "",
)
EXPECTED_CALL = ["run", "mempabench", "run", *RUN_ARGUMENTS]

_TIME = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4}"
_TRY_LINE = re.compile(rf"^run_with_resume: try (\d+) ended at {_TIME} with exit (\d+)")
_SUMMARY_ROW = re.compile(
    rf"^ +(\d+)  ({_TIME})  ({_TIME})  +(\d+)  (\S+) +(\S+) +(\S+)$"
)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _environment(
    tmp_path: Path, tries: list[str], env: dict[str, str | None] | None
) -> dict[str, str]:
    """The script's environment; a None in `env` leaves that variable unset."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "uv", _FAKE_UV)
    _write_executable(fake_bin / "sleep", _FAKE_SLEEP)
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    full_env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("MEMPABENCH_")
    }
    full_env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{full_env['PATH']}",
            "FAKE_DIR": str(fake_dir),
            "FAKE_TRIES": " ".join(tries),
            "TMPDIR": str(temp_dir),
            "MEMPABENCH_RESUME_WAIT_SECONDS": "1",
        }
    )
    full_env.update(env or {})
    return {name: value for name, value in full_env.items() if value is not None}


def _run_script(
    tmp_path: Path,
    tries: list[str],
    env: dict[str, str | None] | None = None,
    arguments: tuple[str, ...] = RUN_ARGUMENTS,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], list[int]]:
    completed = subprocess.run(
        [str(SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        env=_environment(tmp_path, tries, env),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    fake_dir = tmp_path / "fake"
    calls = []
    while (call := fake_dir / f"call_{len(calls) + 1}").exists():
        calls.append(call.read_text(encoding="utf-8").split("\0")[:-1])
    sleeps_file = fake_dir / "sleeps"
    sleeps = [int(line) for line in sleeps_file.read_text().split()] if sleeps_file.exists() else []
    # The stderr copy of every try is gone.
    assert list((tmp_path / "tmp").iterdir()) == []
    return completed, calls, sleeps


def _tries(stderr: str) -> list[tuple[int, int]]:
    """(try number, exit code) of every try line the script printed."""
    return [
        (int(match[1]), int(match[2]))
        for line in stderr.splitlines()
        if (match := _TRY_LINE.match(line))
    ]


def _summary(stderr: str) -> tuple[list[tuple[str, ...]], str]:
    """The summary rows as (try, exit, kind, step, next wait), and the final status line."""
    lines = stderr.splitlines()
    start = lines.index("run_with_resume summary:")
    assert lines[start + 1].split() == [
        "try", "started", "ended", "exit", "kind", "step", "next_wait_s"
    ]
    rows = []
    for line in lines[start + 2 : -1]:
        match = _SUMMARY_ROW.match(line)
        assert match, line
        assert match[2] <= match[3]
        rows.append((match[1], match[4], match[5], match[6], match[7]))
    return rows, lines[-1]


def test_two_resumable_failures_then_success_resume_twice_with_identical_arguments_and_exit_0(
    tmp_path: Path,
) -> None:
    completed, calls, sleeps = _run_script(
        tmp_path, ["75:provider:acc_001", "75:provider:acc_002", "0"]
    )

    assert completed.returncode == 0, completed.stderr
    assert calls == [EXPECTED_CALL] * 3
    assert _tries(completed.stderr) == [(1, 75), (2, 75), (3, 0)]
    # The run's own output is not swallowed.
    assert completed.stdout.splitlines() == [f"fake run output {n}" for n in (1, 2, 3)]
    run_lines = [line for line in completed.stderr.splitlines() if line.startswith(("fake", "mempabench-"))]
    assert run_lines == [
        "fake run log 1",
        "mempabench-run-stopped kind=provider step=acc_001",
        "fake run log 2",
        "mempabench-run-stopped kind=provider step=acc_002",
        "fake run log 3",
    ]


def test_any_other_exit_code_after_a_resume_stops_and_is_passed_through(tmp_path: Path) -> None:
    completed, calls, _ = _run_script(tmp_path, ["75:provider:acc_001", "1"])

    assert completed.returncode == 1, completed.stderr
    assert calls == [EXPECTED_CALL] * 2
    assert _tries(completed.stderr) == [(1, 75), (2, 1)]


def test_a_run_that_keeps_failing_stops_after_the_resume_limit_with_exit_75(tmp_path: Path) -> None:
    completed, calls, _ = _run_script(
        tmp_path,
        ["75:provider:acc_001", "75:provider:acc_002", "75:provider:acc_003", "0"],
        env={"MEMPABENCH_MAX_RESUMES": "2"},
    )

    assert completed.returncode == 75, completed.stderr
    assert calls == [EXPECTED_CALL] * 3
    assert _tries(completed.stderr) == [(1, 75), (2, 75), (3, 75)]
    assert "no resumes left (MEMPABENCH_MAX_RESUMES=2)" in completed.stderr.splitlines()[-1]


def test_by_default_the_wait_is_300_s_doubling_at_one_step_up_to_3600(tmp_path: Path) -> None:
    completed, _, sleeps = _run_script(
        tmp_path,
        ["75:provider:acc_001"] * 6 + ["0"],
        env={"MEMPABENCH_RESUME_WAIT_SECONDS": None, "MEMPABENCH_SAME_STEP_LIMIT": "10"},
    )

    assert completed.returncode == 0, completed.stderr
    assert sleeps == [300, 600, 1200, 2400, 3600, 3600]


def test_the_wait_doubles_only_while_the_same_step_keeps_failing(tmp_path: Path) -> None:
    completed, _, sleeps = _run_script(
        tmp_path,
        ["75:provider:acc_001"] * 3 + ["0"],
        env={"MEMPABENCH_SAME_STEP_LIMIT": "10"},
    )

    assert completed.returncode == 0, completed.stderr
    assert sleeps == [1, 2, 4]


def test_the_wait_is_capped(tmp_path: Path) -> None:
    completed, _, sleeps = _run_script(
        tmp_path,
        ["75:provider:acc_001"] * 4 + ["0"],
        env={"MEMPABENCH_SAME_STEP_LIMIT": "10", "MEMPABENCH_RESUME_WAIT_MAX_SECONDS": "3"},
    )

    assert completed.returncode == 0, completed.stderr
    assert sleeps == [1, 2, 3, 3]


def test_a_different_step_resets_the_wait_and_the_same_step_count(tmp_path: Path) -> None:
    completed, calls, sleeps = _run_script(
        tmp_path,
        [
            "75:provider:acc_005",
            "75:provider:acc_005",
            "75:judge_timeout:acc_006",
            "75:judge_timeout:acc_006",
            "0",
        ],
    )

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 5
    assert sleeps == [1, 2, 1, 2]


def test_a_failure_outside_a_step_resets_the_wait_and_never_stops_on_the_same_step(
    tmp_path: Path,
) -> None:
    completed, calls, sleeps = _run_script(
        tmp_path,
        ["75:provider:acc_005", "75:provider:acc_005"] + ["75:memory_backend:-"] * 4 + ["0"],
    )

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 7
    assert sleeps == [1, 2, 1, 1, 1, 1]


def test_the_same_step_failing_three_tries_in_a_row_stops_with_exit_75(tmp_path: Path) -> None:
    completed, calls, sleeps = _run_script(tmp_path, ["75:ix_runtime:acc_005"] * 3 + ["0"])

    assert completed.returncode == 75, completed.stderr
    assert len(calls) == 3
    assert sleeps == [1, 2]
    last_line = completed.stderr.splitlines()[-1]
    assert "acc_005" in last_line
    assert "ix_runtime" in last_line
    assert "retrying is not helping" in last_line


def test_exit_75_without_a_stop_line_is_not_resumed(tmp_path: Path) -> None:
    completed, calls, sleeps = _run_script(tmp_path, ["75", "0"])

    assert completed.returncode == 75, completed.stderr
    assert len(calls) == 1
    assert sleeps == []
    assert "mempabench-run-stopped" in completed.stderr.splitlines()[-1]


def test_the_summary_has_one_row_per_try_and_ends_with_the_final_status(tmp_path: Path) -> None:
    completed, _, _ = _run_script(
        tmp_path,
        ["75:provider:acc_001", "75:provider:acc_001", "75:judge_unavailable:-", "0"],
    )

    rows, final_line = _summary(completed.stderr)
    assert rows == [
        ("1", "75", "provider", "acc_001", "1"),
        ("2", "75", "provider", "acc_001", "2"),
        ("3", "75", "judge_unavailable", "-", "1"),
        ("4", "0", "-", "-", "-"),
    ]
    assert final_line == "run_with_resume: finished at try 4 with exit 0"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MEMPABENCH_MAX_RESUMES", "-1"),
        ("MEMPABENCH_MAX_RESUMES", "ten"),
        ("MEMPABENCH_RESUME_WAIT_SECONDS", "1.5"),
        ("MEMPABENCH_RESUME_WAIT_SECONDS", ""),
        ("MEMPABENCH_RESUME_WAIT_MAX_SECONDS", "1h"),
        ("MEMPABENCH_SAME_STEP_LIMIT", "0"),
    ],
)
def test_a_bad_setting_is_named_and_nothing_runs(tmp_path: Path, name: str, value: str) -> None:
    completed, calls, _ = _run_script(tmp_path, ["0"], env={name: value})

    assert completed.returncode == 2
    assert name in completed.stderr
    assert calls == []


@pytest.mark.parametrize(
    "restarting_argument",
    [
        "--overwrite",
        "--regenerate-plan",
        "--over",
        "--regen",
        "--overwrite=1",
        "--regenerate-plan=yes",
        "--overw=",
    ],
)
def test_an_argument_that_would_restart_the_run_on_every_resume_is_refused_and_nothing_runs(
    tmp_path: Path, restarting_argument: str
) -> None:
    completed, calls, _ = _run_script(
        tmp_path, ["0"], arguments=(*RUN_ARGUMENTS, restarting_argument)
    )

    assert completed.returncode == 2
    assert calls == []
    assert restarting_argument in completed.stderr
    assert "a resume would restart the run" in completed.stderr
    assert "new run id" in completed.stderr


_GRACEFUL_FAKE_UV = """\
#!/usr/bin/env bash
/bin/sleep 30 > /dev/null 2>&1 &
trap 'kill $!; echo "fake graceful stop done" >&2; exit 0' INT
echo ready > "$FAKE_DIR/ready"
wait
"""


def test_ctrl_c_lets_the_run_stop_gracefully_and_its_stderr_still_arrives(tmp_path: Path) -> None:
    """Ctrl-C reaches every process of the foreground group, the stderr copier too.

    The run's graceful-stop output is written after the signal, so the copier
    must outlive it; the script then ends with the run's own exit code.
    """
    environment = _environment(tmp_path, ["0"], None)
    _write_executable(tmp_path / "bin" / "uv", _GRACEFUL_FAKE_UV)
    script = subprocess.Popen(
        [str(SCRIPT)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "fake" / "ready").exists():
            assert time.monotonic() < deadline, "the fake run never started"
            time.sleep(0.05)
        os.killpg(script.pid, signal.SIGINT)
        _, stderr = script.communicate(timeout=30)
    finally:
        if script.poll() is None:
            os.killpg(script.pid, signal.SIGKILL)
            script.wait()

    assert script.returncode == 0, stderr
    assert "fake graceful stop done" in stderr.splitlines()
    assert stderr.splitlines()[-1] == "run_with_resume: finished at try 1 with exit 0"
