"""Score a completed run with the approved judge.

Usage: python -m scripts.score_completed_run --run-dir <run>
Uses the existing packer, judge/reask policy and E1–E4 aggregator. No E5 baseline.
The metrics are raw: a run whose accumulation steps were skipped is scored, and
which skipped or exposed results to leave out is decided where results are
aggregated, not here. A Probe that exhausted its attempts is not judged; the
metrics score it at the floor (`harness.metrics.FLOOR_RULE`).

`judge_input` is packed only while `judge_output/<judge>/` is empty, so old and
new score files never mix; a run that already has score files resumes with
them. After a failed judgement, rerun the same command: finished judgements are
kept and the failed ones are judged again. To rescore, remove
`judge_output/<judge>/` and `judge_audit/` first.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

import httpx

from harness.judge_audit import AUDIT_PATH, JUDGE_ID, MODEL, checked_outputs, require_complete, write_json
from harness.metrics import PROBE_IDS, compute_metrics, render_markdown
from harness.run_plan import load_run_plan
from judge.evaluator import JudgeOutputInvalid, JudgeReply, OpenRouterJudgeClient, run_judge
from judge.pack import pack_run
from scripts.run_locked import producer_lock
from harness.runtime_activity import observe_model_calls

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def scoring_lock(run: Path):
    folder = run / "judge_audit"
    folder.mkdir(exist_ok=True)
    with (folder / "scoring.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


#: The paid variance-request ledger: request intent, outcome and chances per
#: Probe. It lives apart from the scoring manifest so that nothing about the
#: manifest's lifetime can take it along; a lost ledger re-sends requests whose
#: outcome is unknown, i.e. pays twice.
REQUESTS_PATH = Path("judge_audit") / "requests.json"


def _load_variance_requests(run: Path) -> dict:
    """The run's request ledger; a new, empty one before the first request."""
    path = run / REQUESTS_PATH
    if path.exists():
        return json.loads(path.read_text())
    return {"schema_version": 1, "probes": {}}


def _save_variance_audit(path: Path, audit: dict) -> None:
    """Durably record request intent before the single provider call begins."""
    temporary = path.with_suffix(".variance.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


#: Judge chances per Probe in variance scoring (user instruction 2026-09-22):
#: one chance is one complete judge evaluation, including its native re-ask.
#: The first chance that ends in HTTP429 is not counted (``extra_used``): a
#: rate limit gets one retry beyond the three; a later 429 counts normally.
MAX_JUDGE_CHANCES = 3


class _VarianceJudgeClient:
    """Journal every real provider call of one judge chance before it is sent."""

    def __init__(self, client, record: dict, save):
        self.client, self.record, self.save = client, record, save

    def complete(self, *, system: str, user: str) -> JudgeReply:
        payload = self.client.request_payload(system=system, user=user)
        request = {"kind": "initial" if self.record["chances_used"] == 1 else "retry",
                   "chance": self.record["chances_used"], "payload": deepcopy(payload), "status": "started",
                   "started_at": datetime.now(timezone.utc).isoformat()}
        self.record["requests"].append(request)
        self.record["status"] = "running"
        self.save()
        try:
            reply = self.client.complete_payload(deepcopy(payload))
        except Exception as exc:
            status = "http429" if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429 else "error"
            request.update(status=status, error={"type": type(exc).__name__, "message": str(exc)})
            self.save()
            raise
        request.update(status="replied", reply=asdict(reply))
        self.save()
        return reply


def _variance_scores(run: Path, audit: dict, valid: set[str], manifest: Path,
                     not_applicable: frozenset[str] = frozenset()) -> None:
    state = _load_variance_requests(run)
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported variance judge request ledger")
    records = state["probes"]

    def save():
        # The request ledger first and durably: it is what prevents paying twice.
        _save_variance_audit(run / REQUESTS_PATH, state)
        write_json(manifest, audit)
    client = None

    def fail(pid, record, error, *, status):
        record["status"] = status
        detail = {"type": type(error).__name__, "message": str(error)}
        audit.setdefault("failure_history", []).append({"probe_id": pid, "error": detail,
            "at": datetime.now(timezone.utc).isoformat(), "chances_used": record["chances_used"]})
        audit["errors"][pid] = detail
        save()
        print(f"judge_failed {pid} chance={record['chances_used']}/{MAX_JUDGE_CHANCES}: {detail['type']}: {detail['message']}", flush=True)

    def evaluate(pid, record):
        nonlocal client
        if (run / "judge_output" / JUDGE_ID / f"{pid}.json").exists():
            fail(pid, record, ValueError("Invalid score file requires inspection; it will not be overwritten"),
                 status="needs_inspection")
            return
        if client is None:
            client = OpenRouterJudgeClient(MODEL)
        # The chance is spent durably before any request can be sent.
        record["chances_used"] += 1
        save()
        try:
            run_judge(run / "judge_input" / pid, _VarianceJudgeClient(client, record, save),
                      model=MODEL, run_dir=run)
        except Exception as exc:
            if record["requests"] and record["requests"][-1].get("status") == "http429" and not record["extra_used"]:
                # The one rate-limit retry beyond the three chances.
                record.update(extra_used=True, chances_used=record["chances_used"] - 1)
                fail(pid, record, exc, status="http429")
                return
            fail(pid, record, exc, status="failed" if record["chances_used"] < MAX_JUDGE_CHANCES else "exhausted")
            return
        audit["errors"].pop(pid, None)
        record["status"] = "succeeded"
        save()

    def retryable(pid) -> bool:
        record = records.get(pid)
        return bool(record and pid not in valid and record.get("status") in {"failed", "http429"}
                    and not any(r.get("status") == "started" for r in record["requests"])
                    and record["chances_used"] < MAX_JUDGE_CHANCES)

    # Finish every never-requested item before spending any further chance.
    for pid in PROBE_IDS:
        if pid in valid or pid in not_applicable:
            continue
        if pid in records:
            record = records[pid]
            if any(r.get("status") == "started" for r in record["requests"]):
                # A request with no recorded outcome may have been billed; hold it.
                fail(pid, record, RuntimeError("Prior provider request outcome is unknown; per-item reconciliation is required"), status="unknown")
            elif record.get("status") in {"running", "unrequested"}:
                # Interrupted after every request of its chance had an outcome.
                last = record["requests"][-1] if record["requests"] else {}
                if last.get("status") == "http429" and not record["extra_used"]:
                    record.update(extra_used=True, chances_used=max(record["chances_used"] - 1, 0))
                    status = "http429"
                else:
                    status = "failed" if record["chances_used"] < MAX_JUDGE_CHANCES else "exhausted"
                fail(pid, record, RuntimeError("Interrupted chance without an audited output"), status=status)
            continue
        record = records[pid] = {"requests": [], "chances_used": 0, "extra_used": False, "status": "unrequested"}
        evaluate(pid, record)
    # Then give each known failure its remaining chances, one pass per chance
    # (plus the single uncounted rate-limit retry).
    for _ in range(MAX_JUDGE_CHANCES):
        for pid in PROBE_IDS:
            if pid not in not_applicable and retryable(pid):
                evaluate(pid, records[pid])


def score(run: Path, *, variance_retries: bool = False, not_applicable_exhausted: bool = False) -> int:
    run = run.resolve()
    manifest = run / AUDIT_PATH
    na_evidence = {}
    if not_applicable_exhausted:
        from harness.probe_na import exhausted_failures, source_exhausted_failures
        if variance_retries:
            na_evidence = exhausted_failures(run, PROBE_IDS)
        elif (run / "probe_repeat_source.json").exists():
            raise ValueError("Exhausted-attempt N/A scoring of a variance repeat needs --variance-retries")
        else:
            na_evidence = source_exhausted_failures(run, PROBE_IDS)
    not_applicable = frozenset(na_evidence)
    scored_ids = set(PROBE_IDS) - not_applicable
    plan = require_complete(run, not_applicable)
    load_run_plan(run / "run_plan.yaml", project_root=PROJECT_ROOT)
    outputs = run / "judge_output" / JUDGE_ID
    with producer_lock(run), scoring_lock(run), observe_model_calls(run):
        audit = json.loads(manifest.read_text()) if manifest.exists() else {
            "schema_version": 1, "run_id": plan["run_id"], "model": MODEL,
            "judge_id": JUDGE_ID, "project_root": str(PROJECT_ROOT), "errors": {}}
        if any(outputs.glob("*.json")):
            # Resume: the score files on disk count; judge_input is not repacked.
            valid, _, errors = checked_outputs(run, not_applicable)
            if errors:
                raise ValueError(f"invalid score files require inspection: {errors}")
        else:
            # Pack only into an empty judge_output.
            # An N/A Probe keeps its failed step artifacts; it is never packed.
            index = (pack_run(run, project_root=PROJECT_ROOT, only=sorted(scored_ids)) if not_applicable
                     else pack_run(run, project_root=PROJECT_ROOT))
            if {p["probe_id"] for p in json.loads(index.read_text())["probes"]} != scored_ids:
                raise ValueError("packer did not produce every scored probe")
            valid = set()
        # What the statistics need to apply their own exclusions (D2).
        ledger = json.loads((run / "ledger.json").read_text())
        audit["skipped_steps"] = {
            sid: (row.get("summary") or {}).get("skip_reason")
            for sid, row in ledger["steps"].items() if row.get("status") == "skipped"}
        audit["metrics_scope"] = ("raw_all_33_exhausted_attempts_floored" if not_applicable
                                  else "raw_all_33")
        if not_applicable:
            audit["not_applicable"] = na_evidence
        else:
            audit.pop("not_applicable", None)
        write_json(manifest, audit)
        if (valid == scored_ids and not audit.get("errors")
                and (run / "metrics" / f"{JUDGE_ID}.json").is_file()):
            print(f"scoring_already_complete run={plan['run_id']}", flush=True)
            return 0
        if (run / REQUESTS_PATH).exists() and not variance_retries:
            raise ValueError("Resume this request budget with --variance-retries; generic retries are refused")
        if audit.get("errors") and not variance_retries:
            # A rerun judges the failed probes again; their errors move to the history.
            failed = set(audit["errors"])
            if not failed <= set(PROBE_IDS) or failed & valid:
                raise ValueError("inconsistent failed/output identities; inspect the audit")
            audit.setdefault("failure_history", []).append({
                "errors": audit["errors"], "retry_at": datetime.now(timezone.utc).isoformat(),
            })
            audit["errors"] = {}
            write_json(manifest, audit)
        if variance_retries:
            audit.setdefault("errors", {})
            _variance_scores(run, audit, valid, manifest, not_applicable)
        client = OpenRouterJudgeClient(MODEL) if not variance_retries else None
        for pid in (() if variance_retries else PROBE_IDS):
            if pid in valid or pid in not_applicable:
                continue
            try:
                run_judge(run / "judge_input" / pid, client, model=MODEL, run_dir=run)
            except Exception as exc:
                audit["errors"][pid] = {"type": type(exc).__name__, "message": str(exc)}
                print(f"judge_failed {pid}: {type(exc).__name__}: {exc}", flush=True)
            audit["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(manifest, audit)
            if audit["errors"]:
                return 1  # Inspect a failure before spending on further requests.
        valid, _, errors = checked_outputs(run, not_applicable)
        if audit["errors"] or errors or valid != scored_ids:
            return 1
        metrics = compute_metrics(run, JUDGE_ID, not_applicable=not_applicable)
        output = run / "metrics" / f"{JUDGE_ID}.json"
        write_json(output, metrics)
        output.with_suffix(".md").write_text(render_markdown(metrics))
        suffix = f" not_applicable={','.join(sorted(not_applicable))}" if not_applicable else ""
        print(f"scoring_complete run={plan['run_id']} probes={len(scored_ids)}{suffix} metrics={output}", flush=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variance-retries", action="store_true", help="continue untouched items, then replay each explicit HTTP429 once with a durable request budget")
    parser.add_argument("--not-applicable-exhausted", action="store_true",
                        help="score a run whose failed Probes each failed all three attempts; they are not judged and the metrics give them the floor score (setting check missed, BRS 1). A variance repeat also needs --variance-retries")
    args = parser.parse_args()
    return score(args.run_dir, variance_retries=args.variance_retries,
                 not_applicable_exhausted=args.not_applicable_exhausted)


if __name__ == "__main__":
    raise SystemExit(main())
