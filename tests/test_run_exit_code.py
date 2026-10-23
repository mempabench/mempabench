"""`mempabench run` exits 75 when it stopped on a failure a rerun resumes from.

`scripts/run_with_resume.sh` reruns the command on exactly that code, and reads
where the run stopped from the stop line that then ends stderr, so a failure
that a rerun cannot fix must keep the interpreter's exit status 1, and the
traceback is printed either way. Each case raises a real exception class from
the run's own code and checks the kind `classify_failure` gives it, so the exit
code is tied to the run's failure classification.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from harness.failure_kind import FAILURE_KINDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The `mempabench` console script (`sys.exit(main())`) with the run entry
#: point replaced by one that raises `{exception}`, after `{setup}`. It first
#: prints the kind `classify_failure` gives that exception and the exception's
#: own last line.
_PROGRAM = """\
import sys
import traceback
import scripts.run_locked
from harness.failure_kind import classify_failure
from mempabench.cli import main
{imports}

error = {exception}
{setup}
print(classify_failure(error))
print("".join(traceback.format_exception_only(error)).splitlines()[-1], flush=True)

def stopped_run():
    raise error

scripts.run_locked.main = stopped_run
sys.exit(main(["run", "--pa-model", "test/model"]))
"""

#: (kind, imports, exception, exit code): one case per failure kind, plus a
#: memory-service failure next to the File memory consolidation one.
CASES = [
    ("provider", "from pa.agent.loop import AgentProviderError",
     "AgentProviderError(\"PA provider failed to produce a response: model='m' error='length'\")", 75),
    ("ix_runtime", "from pa.agent.loop import AgentProviderError",
     "AgentProviderError('PA turn failed the IX render contract: test detail')", 75),
    ("memory_backend", "from pa.agent.memory import MemoryConsolidationError",
     "MemoryConsolidationError('INVALID_CONSOLIDATION: session=s failed after 3 attempts')", 75),
    ("memory_backend", "from memory.backends.memos_client import MemOSTransportError",
     "MemOSTransportError('MemOS request failed: connection refused')", 75),
    ("session_invalid", "from simulator.output import SessionInvalidError",
     "SessionInvalidError('beat_ask_unfulfilled', beat='b01', attempts=3)", 75),
    ("output_contract", "from simulator.output import SimulatorOutputContractError",
     "SimulatorOutputContractError(['missing <reply> block'])", 75),
    ("judge_timeout", "from simulator.output import SessionInvalidError",
     "SessionInvalidError('judge_timeout', beat='b01', attempts=1)", 75),
    ("judge_unparseable", "from simulator.output import SessionInvalidError",
     "SessionInvalidError('judge_unparseable', beat='b01', attempts=1)", 75),
    ("judge_evidence_invalid", "from simulator.output import SessionInvalidError",
     "SessionInvalidError('judge_evidence_invalid', beat='b01', attempts=1)", 75),
    ("judge_unavailable", "from simulator.judge.client import JudgeUnavailable",
     "JudgeUnavailable('judge endpoint is not answering')", 75),
    ("ix_open_hold", "from harness.pa_factory import OpenHoldEpisodeError",
     "OpenHoldEpisodeError('acc_001', [('autonomy_level', 'ep_1')])", 1),
    ("ix_contract", "from pa.agent.ix.render_types import IXContractVersionError",
     "IXContractVersionError('ix_render written under another contract version')", 1),
    ("other", "", "RuntimeError('checkpoint store unavailable')", 1),
]


def test_the_cases_cover_every_failure_kind() -> None:
    assert {kind for kind, *_ in CASES} == set(FAILURE_KINDS)


@pytest.mark.parametrize(
    ("kind", "imports", "exception", "exit_code"),
    CASES,
    ids=[f"{kind}-{exception.partition('(')[0]}" for kind, _, exception, _ in CASES],
)
def test_a_run_exits_75_with_a_stop_line_only_for_a_resumable_failure_kind(
    kind: str, imports: str, exception: str, exit_code: int
) -> None:
    completed = _run_stopped_by(imports, exception)

    classified_kind, exception_line = completed.stdout.splitlines()
    assert classified_kind == kind
    assert completed.returncode == exit_code, completed.stderr
    assert completed.stderr.startswith("Traceback (most recent call last):\n")
    assert ", in stopped_run\n" in completed.stderr
    stderr_lines = completed.stderr.splitlines()
    if exit_code == 75:
        # Raised outside any step, so no step is known.
        assert stderr_lines[-2:] == [exception_line, f"mempabench-run-stopped kind={kind} step=-"]
    else:
        assert stderr_lines[-1] == exception_line
        assert "mempabench-run-stopped" not in completed.stderr


def test_a_failure_inside_a_step_names_that_step_on_the_stop_line() -> None:
    completed = _run_stopped_by(
        "from pa.agent.loop import AgentProviderError\n"
        "from harness.failure_kind import FAILED_STEP_ID_ATTR",
        "AgentProviderError('PA turn failed the IX render contract: test detail')",
        setup="setattr(error, FAILED_STEP_ID_ATTR, 'acc_012')",
    )

    assert completed.returncode == 75, completed.stderr
    assert completed.stderr.splitlines()[-1] == "mempabench-run-stopped kind=ix_runtime step=acc_012"


def test_a_proactive_event_left_undecided_exits_75() -> None:
    """acc_007: the run stopped with exit 1, so
    `scripts/run_with_resume.sh` did not rerun it. The exception comes from the
    real raise site, `AgentLoop._proactive_decision`."""
    completed = _run_stopped_by(
        "from pa.agent.loop import AgentLoop\n"
        "from pa.agent.runner import AgentRunResult\n"
        "def undecided():\n"
        "    try:\n"
        "        AgentLoop._proactive_decision(AgentRunResult(final_content=None, messages=[]))\n"
        "    except RuntimeError as error:\n"
        "        return error",
        "undecided()",
    )

    classified_kind, _ = completed.stdout.splitlines()
    assert classified_kind == "ix_runtime"
    assert completed.returncode == 75, completed.stderr
    assert completed.stderr.splitlines()[-1] == "mempabench-run-stopped kind=ix_runtime step=-"


def _run_stopped_by(imports: str, exception: str, setup: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PROGRAM.format(imports=imports, exception=exception, setup=setup)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
