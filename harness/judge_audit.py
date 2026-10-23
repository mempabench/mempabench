"""Which completed runs can be scored, and which score files count.

No provenance: a score file counts when it parses, names its probe, run and
judge, and carries a valid score. Rescoring is explicit: `score_completed_run`
packs only into an empty `judge_output`, so old and new outputs never mix.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from harness.metrics import PROBE_IDS

MODEL = "google/gemini-3.8-flash"
JUDGE_ID = "llm__google_gemini-3.8-flash"
AUDIT_PATH = Path("judge_audit") / "manifest.json"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def not_applicable_ids(audit: dict) -> frozenset[str]:
    """Probes an audit explicitly scored as N/A (exhausted attempts); usually none."""
    value = audit.get("not_applicable") or {}
    if not isinstance(value, dict) or not set(value) <= set(PROBE_IDS):
        raise ValueError("invalid not_applicable record")
    return frozenset(value)


def require_complete(run: Path, not_applicable: frozenset[str] = frozenset()) -> dict:
    """Every Probe and forced event done, except the named Probes that exhausted
    all attempts. An accumulation step may be done or skipped (a run with
    skipped steps is scored; the statistics decide what to exclude)."""
    if (run / "recovery/pending.json").exists():
        raise ValueError("run has an unfinished checkpoint restore")
    plan = yaml.safe_load((run / "run_plan.yaml").read_text())
    ledger = json.loads((run / "ledger.json").read_text())
    steps = [s for s in plan["steps"] if not s.get("placeholder")]
    probes = {s["step_id"] for s in steps if s["kind"] in ("pre_event_probe", "final_probe")}
    if probes != set(PROBE_IDS):
        raise ValueError("plan does not contain exactly the required 33 probes")
    not_applicable = frozenset(not_applicable)
    if not_applicable:
        if not not_applicable <= probes:
            raise ValueError("N/A Probes are not in the plan")
        from harness.probe_na import na_evidence
        for pid in not_applicable:
            na_evidence(run, pid, ledger)
    from harness.probe_repeat_source import load_repeat_source
    repeat_source = load_repeat_source(run)
    timeline_ledger = (json.loads((Path(repeat_source["source_run_dir"]) / "ledger.json").read_text())
                       if repeat_source is not None else ledger)
    pending = [s["step_id"] for s in steps
               if s["step_id"] not in not_applicable
               and (ledger if s["kind"] in ("pre_event_probe", "final_probe") else timeline_ledger)["steps"].get(s["step_id"], {}).get("status") not in
               ({"done", "skipped"} if s["kind"] == "accumulation" else {"done"})]
    # A producer can leave ``current_step`` on its last Probe after that Probe
    # exhausted its attempts; only a verified N/A Probe is accepted there.
    if (ledger.get("current_step") and ledger["current_step"] not in not_applicable) or pending:
        raise ValueError(f"run is not complete: {pending}")
    if ledger["run_id"] != plan["run_id"]:
        raise ValueError("ledger/plan run identity mismatch")
    return plan


def checked_outputs(run: Path, not_applicable: frozenset[str] | None = None,
                    ) -> tuple[set[str], bool, list[str]]:
    """Read-only coverage of the score files on disk: (valid probe ids, metrics
    present for all of them, errors). A run not yet scored has nothing to count;
    an incomplete run counts nothing. The N/A Probes are the caller's, or else
    the ones the last scoring recorded."""
    if not (run / AUDIT_PATH).exists() and not any((run / "judge_output" / JUDGE_ID).glob("*.json")):
        return set(), False, []
    try:
        audit = json.loads((run / AUDIT_PATH).read_text()) if (run / AUDIT_PATH).exists() else {}
        if not_applicable is None:
            not_applicable = not_applicable_ids(audit)
        not_applicable = frozenset(not_applicable)
        run_id = require_complete(run, not_applicable)["run_id"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return set(), False, [str(exc)]
    scored_ids = set(PROBE_IDS) - not_applicable
    valid, errors = set(), []
    for pid in sorted(scored_ids):
        path = run / "judge_output" / JUDGE_ID / f"{pid}.json"
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text())
            if (record["probe_id"], record["run_id"], record["judge_id"]) != (pid, run_id, JUDGE_ID):
                raise ValueError("output identity differs")
            if type(record["text_face"]["score"]) is not int or record["text_face"]["score"] not in range(1, 6):
                raise ValueError("invalid score")
            if type(record["ix_face"]["exact_match"]) is not bool:
                raise ValueError("invalid IX match")
            valid.add(pid)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{pid}: {exc}")
    metrics_ok = (not audit.get("errors") and valid == scored_ids
                  and (run / "metrics" / f"{JUDGE_ID}.json").is_file())
    return valid, metrics_ok, errors
