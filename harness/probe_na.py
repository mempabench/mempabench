"""Probes that exhausted their whole-Probe attempt budget (`not_applicable`).

A Probe whose three counted attempts all failed is not judged in that round;
the judge audit lists it under `not_applicable`. The run-level metrics and the
paper export both score it at the floor (`harness.metrics.FLOOR_RULE`: IX 0,
Text 1) and keep every denominator, so a CS/PET pair with such an endpoint
counts and fails. Every other Probe of the round is judged normally. Nothing
here grants an extra attempt or reinterprets a trajectory.

Only the exact exhausted state qualifies: ledger status ``failed`` and, in a
repeat round, a ``probe_attempts/<id>.json`` with attempts 1, 2, 3 each
``failed``; in a source run, three failed attempts with a retried failure kind
in the last launch its ``probe_launch_attempts/<id>.json`` records
(`source_exhausted_failure`). A single non-retryable failure, a started/unknown
attempt, or a held attempt lock is not exhaustion and keeps the round
incomplete.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from harness.run_config import load_run_config

_PROBE_POLICY = load_run_config().probe
#: The whole-Probe budget (`probe.attempt_limit` in configs/run_config.yaml),
#: also ``harness.orchestrator.PROBE_ATTEMPTS``.
ATTEMPT_LIMIT = _PROBE_POLICY.attempt_limit
#: Network/infrastructure failures do
#: not count toward the three (`probe.network_failures_count`); a Probe gets at
#: most six attempts in total (`probe.total_attempt_limit`).
TOTAL_ATTEMPT_LIMIT = _PROBE_POLICY.total_attempt_limit
NETWORK_FAILURES_COUNT = _PROBE_POLICY.network_failures_count
NA_POLICY = "exhausted_three_attempts_na/1"

import re

_NETWORK = re.compile(r"connecterror|connection error|connection reset|connection refused|connection aborted"
                      r"|apiconnectionerror|timed out|timeout|name resolution|dns|server disconnected"
                      r"|remoteprotocolerror|temporarily unavailable|network is unreachable", re.I)
#: Classified as model behaviour by the user even when raised by a backend.
_MODEL_EVEN_IF_BACKEND = re.compile(r"memory list|native processing failed|invalid_request_error"
                                    r"|provider_truncation_exhausted|IX render contract", re.I)


def is_network_failure(error: str | None) -> bool:
    """Only a real outage: connection errors, timeouts, DNS, server disconnects."""
    text = error or ""
    return bool(_NETWORK.search(text)) and not _MODEL_EVEN_IF_BACKEND.search(text)


def counted_failures(attempts: list[dict]) -> int:
    """Failed attempts that count toward the three (non-network, unless configured to count)."""
    return sum(1 for row in attempts if row.get("status") == "failed"
               and (NETWORK_FAILURES_COUNT or not is_network_failure(row.get("error"))))


def _lock_free(path: Path) -> bool:
    if not path.exists():
        return True
    with path.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream, fcntl.LOCK_UN)
    return True


def exhausted_failure(run: Path, probe_id: str, ledger: dict) -> dict:
    """Return the N/A evidence for one Probe, or raise if it is not exhausted."""
    record = ledger.get("steps", {}).get(probe_id, {})
    if record.get("status") != "failed":
        raise ValueError(f"{probe_id}: only a failed Probe can be N/A")
    path = Path(run) / "probe_attempts" / f"{probe_id}.json"
    if not path.is_file():
        raise ValueError(f"{probe_id}: no durable attempt record")
    if not _lock_free(path.with_suffix(".lock")):
        raise ValueError(f"{probe_id}: attempt record is still owned by a producer")
    record_value = json.loads(path.read_text())
    attempts = record_value.get("attempts")
    start = record_value.get("reopened", {}).get("after_attempt", 0)
    if (not isinstance(attempts, list) or not attempts
            or counted_failures(attempts[start:]) < ATTEMPT_LIMIT
            or [row.get("attempt") for row in attempts] != list(range(1, len(attempts) + 1))
            or any(row.get("status") != "failed" for row in attempts)
            or counted_failures(attempts) < ATTEMPT_LIMIT):
        raise ValueError(f"{probe_id}: attempts are not {ATTEMPT_LIMIT} failed non-network attempts")
    return {"policy": NA_POLICY, "ledger_status": "failed",
            "failure_kind": record.get("failure_kind"),
            "attempts": {"path": f"run/probe_attempts/{probe_id}.json"},
            "errors": [row.get("error") for row in attempts]}


def exhausted_failures(run: Path, probe_ids, ledger: dict | None = None) -> dict[str, dict]:
    """N/A evidence for every failed Probe; raise if any failure is not exhausted."""
    run = Path(run)
    ledger = ledger if ledger is not None else json.loads((run / "ledger.json").read_text())
    failed = [pid for pid in probe_ids if ledger.get("steps", {}).get(pid, {}).get("status") == "failed"]
    return {pid: exhausted_failure(run, pid, ledger) for pid in failed}


def original_exhausted_failure(run: Path, probe_id: str, ledger: dict) -> dict:
    """Whether a repeat source's ``current_step`` may rest on this failed Probe
    (`harness.probe_repeat_source`); scoring uses `source_exhausted_failure`.

    The orchestrator retries a retryable Probe in-batch, setting each failed
    attempt aside under ``steps_superseded/<id>__<stamp>/``; the final failure
    stays in ``steps/<id>/``. Exhaustion therefore needs a retryable failure kind
    and at least two superseded attempts. Such original Probes are N/A as
    well.
    """
    from harness.failure_kind import PROBE_RETRY_KINDS
    record = ledger.get("steps", {}).get(probe_id, {})
    if record.get("status") != "failed":
        raise ValueError(f"{probe_id}: only a failed Probe can be N/A")
    if record.get("failure_kind") not in PROBE_RETRY_KINDS:
        raise ValueError(f"{probe_id}: failure kind {record.get('failure_kind')!r} is not retried, so its attempts were not exhausted")
    superseded = sorted(p.name for p in (Path(run) / "steps_superseded").glob(f"{probe_id}__*") if p.is_dir())
    if len(superseded) < ATTEMPT_LIMIT - 1:
        raise ValueError(f"{probe_id}: only {len(superseded) + 1} recorded attempts")
    return {"policy": NA_POLICY + "+original_superseded", "ledger_status": "failed",
            "failure_kind": record["failure_kind"], "superseded_attempts": superseded,
            "ledger_error": record.get("error")}


def source_exhausted_failure(run: Path, probe_id: str, ledger: dict) -> dict:
    """N/A evidence for a Probe of a source run (a plain `mempabench run`).

    The run records every attempt with the launch it belongs to
    (`harness.probe_launch_attempts`); each launch gives the Probe up to
    ATTEMPT_LIMIT attempts. Exhaustion is the ledger row ``failed`` and the
    launch of the record's last row holding exactly ATTEMPT_LIMIT attempts,
    each ``failed`` with a retried failure kind. Rows of earlier launches never
    count. A run without the record cannot show exhaustion; a record that is
    not valid is an error.
    """
    from harness.failure_kind import PROBE_RETRY_KINDS
    from harness.probe_launch_attempts import RECORD_DIR, load_attempts
    record = ledger.get("steps", {}).get(probe_id, {})
    if record.get("status") != "failed":
        raise ValueError(f"{probe_id}: only a failed Probe can be N/A")
    try:
        rows = load_attempts(run, probe_id)
    except FileNotFoundError as exc:
        raise ValueError(
            f"{probe_id}: no attempt record {RECORD_DIR}/{probe_id}.json, so this run does not show that "
            f"{probe_id} failed all {ATTEMPT_LIMIT} attempts of a launch. Resume the run with the same "
            f"`mempabench run` command and run ID: {probe_id} gets {ATTEMPT_LIMIT} new attempts, each "
            "recorded as it happens") from exc
    launch_id = rows[-1]["launch_id"]
    latest = [row for row in rows if row["launch_id"] == launch_id]
    if len(latest) != ATTEMPT_LIMIT or any(
            row["status"] != "failed" or row["failure_kind"] not in PROBE_RETRY_KINDS for row in latest):
        outcomes = ", ".join(" ".join(filter(None, (row["status"], row.get("failure_kind")))) for row in latest)
        raise ValueError(f"{probe_id}: its last launch {launch_id} recorded {len(latest)} attempt(s) ({outcomes}), "
                         f"not {ATTEMPT_LIMIT} failed attempts with a retried failure kind")
    return {"policy": NA_POLICY + "+launch_attempts", "ledger_status": "failed",
            "failure_kind": record.get("failure_kind"), "launch_id": launch_id,
            "attempts": {"path": f"run/{RECORD_DIR}/{probe_id}.json"},
            "errors": [row["error"] for row in latest]}


def source_exhausted_failures(run: Path, probe_ids, ledger: dict | None = None) -> dict[str, dict]:
    """N/A evidence for every failed Probe of a source run; raise if any failure is not exhausted."""
    run = Path(run)
    ledger = ledger if ledger is not None else json.loads((run / "ledger.json").read_text())
    failed = [pid for pid in probe_ids if ledger.get("steps", {}).get(pid, {}).get("status") == "failed"]
    return {pid: source_exhausted_failure(run, pid, ledger) for pid in failed}


def na_evidence(run: Path, probe_id: str, ledger: dict) -> dict:
    """Repeat rounds use their attempt ledger; a source run its per-launch attempt record."""
    if (Path(run) / "probe_repeat_source.json").exists():
        return exhausted_failure(run, probe_id, ledger)
    return source_exhausted_failure(run, probe_id, ledger)
