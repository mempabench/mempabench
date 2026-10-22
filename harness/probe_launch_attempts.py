"""Every attempt of a probe in a plain `mempabench run`, recorded as it happens.

`probe_launch_attempts/<probe_id>.json` lists a probe's attempts in the order
they ran. Each launch of the run (a start or a resume) gives the probe up to
`probe.attempt_limit` attempts, numbered from 1 (`Orchestrator._run_one_probe`),
and every row names its launch by the `launch_id` the orchestrator made for
that launch, which the launch's `run_info.yaml` record also carries:

    {"schema": "mempa_probe_launch_attempts/1", "step_id": "fin_27",
     "attempts": [{"launch_id": "<hex>", "attempt": 1, "status": "failed",
                   "failure_kind": "ix_runtime", "error": "AgentProviderError: ..."}]}

A row is written `started` before its attempt runs, and becomes `failed` (with
the failure kind and the error) or `succeeded` before the run acts on the
outcome: before a retry sets the attempt aside and before the ledger records
the probe. A row left `started` is an attempt that never reported back. A probe
repeat keeps its own attempt budget instead (`harness.probe_attempts`).
"""
from __future__ import annotations

import json
from pathlib import Path

from harness.failure_kind import FAILURE_KINDS
from harness.shared_probe_memory import _atomic

SCHEMA = "mempa_probe_launch_attempts/1"
RECORD_DIR = "probe_launch_attempts"
STATUSES = ("started", "failed", "succeeded")
_ROW_KEYS = {"launch_id", "attempt", "status"}
_FAILED_ROW_KEYS = _ROW_KEYS | {"failure_kind", "error"}


def record_path(run_dir: Path | str, step_id: str) -> Path:
    return Path(run_dir) / RECORD_DIR / f"{step_id}.json"


def load_attempts(run_dir: Path | str, step_id: str) -> list[dict]:
    """The probe's attempt rows, in the order they ran.

    Raises FileNotFoundError when the run has no record for the probe, and
    ValueError, naming the probe and the fault, when the record is not valid.
    """
    text = record_path(run_dir, step_id).read_text(encoding="utf-8")

    def invalid(detail: str) -> ValueError:
        return ValueError(f"{step_id}: attempt record {RECORD_DIR}/{step_id}.json is invalid: {detail}")

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise invalid(f"not JSON ({exc})") from exc
    if (not isinstance(value, dict) or set(value) != {"schema", "step_id", "attempts"}
            or value["schema"] != SCHEMA or value["step_id"] != step_id):
        raise invalid(f"expected the keys schema ({SCHEMA}), step_id ({step_id}) and attempts")
    rows = value["attempts"]
    if not isinstance(rows, list) or not rows:
        raise invalid("attempts is not a non-empty list")
    previous: dict | None = None
    launches: set[str] = set()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get("status") not in STATUSES:
            raise invalid(f"row {number} has no status out of {STATUSES}")
        if set(row) != (_FAILED_ROW_KEYS if row["status"] == "failed" else _ROW_KEYS):
            raise invalid(f"row {number} has the keys {sorted(row)}")
        launch_id, attempt = row["launch_id"], row["attempt"]
        if not isinstance(launch_id, str) or not launch_id:
            raise invalid(f"row {number} has the launch_id {launch_id!r}")
        same_launch = previous is not None and previous["launch_id"] == launch_id
        if not same_launch and launch_id in launches:
            raise invalid(f"row {number}: the rows of launch {launch_id} are not together")
        if same_launch and previous["status"] != "failed":
            raise invalid(f"row {number} follows a {previous['status']} attempt of the same launch")
        expected = previous["attempt"] + 1 if same_launch else 1
        if type(attempt) is not int or attempt != expected:
            raise invalid(f"row {number} is attempt {attempt!r}, not {expected}")
        if row["status"] == "failed" and (row["failure_kind"] not in FAILURE_KINDS
                                          or not isinstance(row["error"], str)):
            raise invalid(f"row {number} has the failure_kind {row['failure_kind']!r} and error {row['error']!r}")
        launches.add(launch_id)
        previous = row
    return rows


class ProbeLaunchAttempts:
    """The writer one launch uses for one probe's attempts."""

    def __init__(self, run_dir: Path | str, step_id: str, launch_id: str) -> None:
        self.run_dir, self.step_id, self.launch_id = Path(run_dir), step_id, launch_id
        self.path = record_path(run_dir, step_id)

    def start(self, attempt: int) -> None:
        # The probe's first attempt in this run creates the record.
        rows = load_attempts(self.run_dir, self.step_id) if self.path.exists() else []
        rows.append({"launch_id": self.launch_id, "attempt": attempt, "status": "started"})
        self._write(rows)

    def fail(self, attempt: int, *, failure_kind: str, error: BaseException) -> None:
        self._finish(attempt, status="failed", failure_kind=failure_kind, error=f"{type(error).__name__}: {error}")

    def succeed(self, attempt: int) -> None:
        self._finish(attempt, status="succeeded")

    def _finish(self, attempt: int, **outcome) -> None:
        rows = load_attempts(self.run_dir, self.step_id)
        row = rows[-1]
        if (row["launch_id"], row["attempt"], row["status"]) != (self.launch_id, attempt, "started"):
            raise RuntimeError(f"{self.step_id}: attempt {attempt} of launch {self.launch_id} is not the attempt "
                               f"in progress in {RECORD_DIR}/{self.step_id}.json: {row}")
        row.update(outcome)
        self._write(rows)

    def _write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic(self.path, {"schema": SCHEMA, "step_id": self.step_id, "attempts": rows})
