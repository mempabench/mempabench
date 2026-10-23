"""Rate a scored run's final probes as its simulated user: PRS.

Usage: python -m scripts.self_report_completed_run --run-dir <run>

After a finished final probe, the same simulated user (the run's persona card,
the simulator model and parameters its RunPlan froze, none of the memory it
built up) re-reads its own side of the conversation and rates the one
attribute the probe tests, 1-5, with a first-person reason
(`simulator/dimension_self_report.py`). PRS (Perceived Realization Score) is
the mean over the run's 28 final probes.

The run must be scored first (`score_completed_run`), because PRS takes the
probes BRS scores: a final BRS floors (all attempts failed,
`harness.metrics.FLOOR_RULE`) gets the same floor here, status `rule_assigned`,
and is never asked. Every other final is rebuilt from what the run recorded:
its transcript, its step's `meta.yaml` (including the recorded
`preference_overlay`), the persona files of the checkpoint it branched from
(a probe-repeat round's are its source run's, as `probe_repeat_source.json`
names it) and the run's preference matrix. A rebuild that does not agree with
the run's own records is status `rebuild_mismatch`: no call, and it leaves the
PRS denominator.

One JSON per probe goes to `metrics/self_report/<probe>.json` and the summary
to `metrics/self_report.json`. A failed call is asked again after 30, 60, 120
and 240 s; if it still fails, or every reply breaks the output contract, the
command stops with an error and writes nothing for that probe. Rerunning the
same command keeps every `scored` result whose prompt and inputs are
unchanged and asks only the rest; it refuses to run when a kept result's
inputs changed. To self-report again from scratch, remove `metrics/self_report/`
first.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

import harness.pa_factory as pa_factory
import simulator.dimension_self_report as dsr
from harness.judge_audit import JUDGE_ID, write_json
from harness.metrics import FINAL_PROBE_IDS, FLOOR_RULE, FLOOR_TEXT, floor_assignments
from harness.probe_repeat_source import load_repeat_source
from harness.run_plan import RunPlan, RunStep, load_run_plan
from harness.run_plan_builder import _load_script
from simulator.config import SimulatorConfig
from simulator.profiles import load_simulator_profile
from simulator.schemas import PreferenceMatrix
from simulator.self_report import SELF_REPORT_MAX_TOKENS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIR = Path("metrics") / "self_report"
SUMMARY_PATH = Path("metrics") / "self_report.json"
#: Waits before asking a probe again after a provider error.
PROVIDER_ERROR_BACKOFF_S = (30, 60, 120, 240)
#: A kept `scored` result is reused only while these are unchanged.
RESUME_KEYS = ("prompt_sha256", "input_sha256")
#: Material the actor is not given here, as fixed literals: profile memory,
#: the message-writing template, the naturalness guide, the per-turn
#: instructions, the reply-facts frame lines, the held-items instruction and
#: IX tool names. One in the prompt is a rebuild mismatch.
NOT_GIVEN_LITERALS = (
    "## Profile Memory",
    "# GLOBAL RULES",
    "# USER DIALOGUE NATURALNESS",
    "# User Dialogue Naturalness",
    "Write the user's visible message as something this person would say",
    "Produce only the current",
    "Output exactly three XML-like blocks",
    "Private record of what the assistant's latest reply",
    "A change the reply describes but this record does not list",
    "Answer them in <message>",
    "Then add a <consent_decisions> block",
    "IX_",
)


class SelfReportError(RuntimeError):
    """The run cannot be self-reported as it stands, or a probe's rating failed."""


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clock(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _only(events: list[dict[str, Any]], kind: str, path: Path) -> dict[str, Any]:
    found = [event for event in events if event.get("kind") == kind]
    if len(found) != 1:
        raise SelfReportError(f"{path}: {len(found)} {kind} events, expected 1")
    return found[0]


# ------------------------------------------------------------------ inputs


def floored_finals(run_dir: Path) -> frozenset[str]:
    """The finals BRS scored at the floor, from the metrics `score` wrote."""
    path = run_dir / "metrics" / f"{JUDGE_ID}.json"
    if not path.is_file():
        raise SelfReportError(f"{path} is missing: score the run first (`mempabench score --run-dir {run_dir}`)")
    assigned = json.loads(path.read_text(encoding="utf-8")).get("rule_assigned_scores") or {}
    if assigned != floor_assignments(assigned):
        raise SelfReportError(f"{path}: rule_assigned_scores is not the {FLOOR_RULE} floor: {assigned}")
    return frozenset(assigned) & frozenset(FINAL_PROBE_IDS)


def checkpoint_run_dir(run_dir: Path) -> Path:
    """The run whose checkpoints the probes branched from: a probe-repeat
    round's source run, or the run itself."""
    source = load_repeat_source(run_dir)
    return Path(source["source_run_dir"]) if source is not None else run_dir


def final_steps(plan: RunPlan, path: Path) -> dict[str, RunStep]:
    steps = {step.step_id: step for step in plan.steps if step.kind == "final_probe"}
    if sorted(steps) != sorted(FINAL_PROBE_IDS):
        raise SelfReportError(f"{path}: final probes are {sorted(steps)}, not {FINAL_PROBE_IDS[0]}..{FINAL_PROBE_IDS[-1]}")
    return steps


# ----------------------------------------------------------------- rebuild


def prepare(run_dir: Path, plan: RunPlan, step: RunStep, *, floored: bool, checkpoint_run: Path) -> dict[str, Any]:
    """The probe's record before any call. A floored final gets the floor
    score and no rebuild; any other is rebuilt and checked against the run's
    records, and a failed check makes it `rebuild_mismatch`. `status` stays
    None for a probe to ask."""
    runtime = plan.simulator_runtime
    generation = SimulatorConfig(model=runtime.simulator_model, api_key="unused").generation
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "probe_id": step.step_id,
        "status": None,
        "attribute": None, "context": None, "setting": None, "setting_source": None,
        "score": None, "reason": None,
        "model": runtime.simulator_model,
        "params": {
            "temperature": generation.temperature, "max_tokens": SELF_REPORT_MAX_TOKENS, "tools": [],
            "reasoning_effort": generation.reasoning_effort,
            "contract_retries": runtime.self_report_contract_retries, "deadline_s": dsr.CALL_DEADLINE_S,
        },
        "prompt_sha256": None, "prompt_files_sha256": dsr.prompt_files_sha256(), "prompt": None,
        "block_sha256": {}, "input_sha256": {},
        "source": {
            "run_dir": str(run_dir),
            "step_dir": str(run_dir / "steps" / step.step_id),
            "checkpoint_run_dir": None, "source_checkpoint_id": None,
        },
        "rebuild_verification": None,
        "attempts": [], "usage_total": {"prompt_tokens": 0, "completion_tokens": 0},
        "cost_usd_total": 0.0, "cost_missing_calls": 0, "raw_output": None, "created_at": None,
    }
    if floored:
        record.update(
            status="rule_assigned", rule=FLOOR_RULE, score=FLOOR_TEXT,
            attribute=step.target_attribute, context=step.context, setting=step.target_setting,
            setting_source="run_plan", rebuild_verification={"status": "not_rebuilt", "failures": []},
        )
        return record
    failures: list[str] = []
    try:
        _rebuild(run_dir, plan, step, checkpoint_run, record, failures)
    except dsr.RebuildError as exc:
        failures.append(f"rebuild_error:{exc}")
    record["rebuild_verification"] = {"status": "fail" if failures else "pass", "failures": failures}
    if failures:
        record["status"] = "rebuild_mismatch"
    return record


def _rebuild(
    run_dir: Path, plan: RunPlan, step: RunStep, checkpoint_run: Path,
    record: dict[str, Any], failures: list[str],
) -> None:
    def check(ok: bool, code: str) -> None:
        if not ok:
            failures.append(code)

    step_dir = run_dir / "steps" / step.step_id
    transcript = step_dir / "transcript.jsonl"
    meta = _yaml(step_dir / "meta.yaml")
    events = _events(transcript)
    attribute, context, setting = meta["target_attribute"], meta["context"], meta["target_setting"]
    overlay = meta.get("preference_overlay") or {}
    borrow = meta.get("borrow_start_from")
    source_run = checkpoint_run if borrow else run_dir
    checkpoint = (source_run / "checkpoints" / str(meta["source_checkpoint_id"])
                  / "simulator_workspace" / plan.persona_id)
    run_ws = run_dir / "simulator_workspace" / plan.persona_id
    draft = PROJECT_ROOT / step.script_path
    record.update(
        attribute=attribute, context=context, setting=setting,
        setting_source="overlay" if attribute in (overlay.get(context) or {}) else "matrix",
        input_sha256={
            "transcript.jsonl": _sha256(transcript), "meta.yaml": _sha256(step_dir / "meta.yaml"),
            "run_plan.yaml": _sha256(run_dir / "run_plan.yaml"), "draft": _sha256(draft),
            **{name: _sha256(checkpoint / name) for name in ("identity.yaml", "TOOLS.md", "profile.yaml")},
            "preferences.yaml": _sha256(run_ws / "preferences.yaml"),
        },
    )
    record["source"].update(checkpoint_run_dir=str(source_run), source_checkpoint_id=meta["source_checkpoint_id"])

    # One target everywhere; a borrowed checkpoint from the run the repeat
    # names; the same simulator, clock and setting as the plan.
    check((step.target_attribute, step.context, step.target_setting) == (attribute, context, setting),
          "plan_vs_meta_target")
    check(not borrow or borrow.split("/")[0] == checkpoint_run.name, "borrowed_from_another_run")
    start = _only(events, "session_start", transcript)
    runtime = meta["simulator_runtime"]
    check(
        runtime == plan.simulator_runtime.model_dump()
        and start.get("simulator_model") == meta.get("simulator_model") == runtime["simulator_model"],
        "simulator_runtime",
    )
    check(_clock(meta["benchmark_session_timestamp"]) == step.benchmark_session_timestamp, "session_clock")
    check((meta["pa_model"], meta["memory_condition"]) == (plan.pa_model, plan.memory_condition), "combo")

    # The transcript's structure.
    pa_turns = [event for event in events if event["kind"] == "pa_turn"]
    renders = {event["t"]: (event.get("metadata") or {}).get("ix_render") for event in pa_turns}
    check(
        all((render or {}).get("schema_version") == meta["ix_render_schema_version"] for render in renders.values()),
        "ix_render_schema",
    )
    check("ix_render_merge" not in meta and "ix_render_version_error" not in meta, "ix_render_merge")
    enters = [event for event in events if event["kind"] == "beat_enter"]
    check(all(not event.get("reception_skills") for event in enters), "reception_skills")
    check(bool(enters) and enters[-1].get("active_skills") == [attribute], "last_beat_declares_attribute")
    check(
        all(
            bool(event.get("outreach_delivered")) == bool(str(event.get("outbound") or "").strip())
            for event in pa_turns if event.get("proactive_event")
        ),
        "outreach_delivered",
    )

    # The rebuild, with the harness keys exactly as pa_factory derived them live.
    script = _load_script(draft)
    check(not any(beat.preference_updates for beat in script.beats), "beat_preference_updates")
    derived = {
        t: {
            "reply_facts": pa_factory.reply_facts_from_ix_render(render) if render else None,
            "held_items": pa_factory.held_items_from_ix_render(render) if render else [],
        }
        for t, render in renders.items()
    }
    rebuilt = dsr.rebuild_blocks(
        events,
        script=script,
        session_clock=_clock(meta["benchmark_session_timestamp"]),
        profile=load_simulator_profile(checkpoint),
        preferences=PreferenceMatrix.from_yaml(run_ws / "preferences.yaml"),
        recorded_overlay=overlay,
        derived=derived,
        attribute=attribute,
        specs_root=PROJECT_ROOT / "data/actor_skill_specs",
    )
    check(rebuilt.setting == setting, "rebuilt_setting")
    prompt = dsr.build_prompt(rebuilt.blocks, attribute, setting)

    # The conversation numbering; nothing not given; consent ids were held.
    history = rebuilt.history
    rows = [
        row for event in events if event["kind"] == "session_end_reflection"
        for row in event.get("evidence") or [] if row.get("turn") is not None
    ]
    check(all(0 <= row["turn"] < len(history) and history[row["turn"]]["t"] == row["t"] for row in rows), "turn_t")
    end = _only(events, "session_end", transcript)
    check(
        end.get("simulator_turns") in (None, sum(item["role"] == "assistant" for item in history)),
        "simulator_turns",
    )
    failures.extend(f"not_given:{literal}" for literal in NOT_GIVEN_LITERALS if literal in prompt)
    held_ids: set[str] = set()
    for event in events:
        if event["kind"] == "pa_turn":
            held_ids = {item["interaction_item_id"] for item in derived[event["t"]]["held_items"]}
        elif event["kind"] == "simulator_turn":
            decided = {str(item.get("interaction_item_id") or "") for item in event.get("consent_decisions") or []}
            check(decided - {""} <= held_ids, "consent_ids_not_held")
    record.update(
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        block_sha256={name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in rebuilt.blocks.items()},
    )


# --------------------------------------------------------------------- run


def make_provider(model: str) -> Any:
    """One provider per probe, built the way the run built the actor's."""
    return SimulatorConfig(model=model).create_provider()


async def ask(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The probe's tries: one, and another after each provider error, after
    the waits in `PROVIDER_ERROR_BACKOFF_S`."""
    reporter = dsr.DimensionSelfReporter(
        make_provider(record["model"]),
        model=record["model"],
        output_contract_retries=record["params"]["contract_retries"],
    )
    tries: list[dict[str, Any]] = []
    for number, wait_s in enumerate((0, *PROVIDER_ERROR_BACKOFF_S), start=1):
        if wait_s:
            print(f"self_report_retry probe={record['probe_id']} after=provider_error wait_s={wait_s} try={number}",
                  flush=True)
            await asyncio.sleep(wait_s)
        tries.append(await reporter.generate(record["prompt"]))
        for call in tries[-1]["calls"]:
            call["try"] = number  # each try numbers its attempts from 1 again
        if tries[-1]["status"] != "provider_error":
            break
    return tries


async def ask_each(queue: list[dict[str, Any]], out: Path, records: dict[str, dict[str, Any]]) -> None:
    """Ask the queued probes one after another, in one event loop, writing
    each result as soon as it is in."""
    for record in queue:
        _score(record, await ask(record))
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / f"{record['probe_id']}.json", record)
        records[record["probe_id"]] = record
        print(_line(record), flush=True)


def _score(record: dict[str, Any], tries: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [call for one in tries for call in one["calls"]]
    outcome = tries[-1]
    if outcome["status"] != "ok":
        findings = sorted({finding for call in calls for finding in call["findings"]})
        finishes = sorted({str(call["finish_reason"]) for call in calls})
        cost = sum(one["cost_usd_total"] for one in tries)
        raise SelfReportError(
            f"{record['probe_id']}: no rating after {len(tries)} tries and {len(calls)} calls "
            f"(status={outcome['status']} findings={findings} finish_reasons={finishes} "
            f"cost_usd={cost:.6f}); nothing was written for it, rerun the same command to ask it again"
        )
    record.update(
        status="scored", score=outcome["score"], reason=outcome["reason"], attempts=calls,
        usage_total={key: sum(one["usage_total"][key] for one in tries) for key in outcome["usage_total"]},
        cost_usd_total=sum(one["cost_usd_total"] for one in tries),
        cost_missing_calls=sum(one["cost_missing_calls"] for one in tries),
        raw_output=calls[-1]["raw"],
    )
    return record


def summarize(plan: RunPlan, run_dir: Path, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """PRS: the mean score over the scored and rule-assigned finals; a
    rebuild mismatch leaves the denominator."""
    counts = Counter(record["status"] for record in records.values())
    counted = [records[pid]["score"] for pid in FINAL_PROBE_IDS if records[pid]["status"] in ("scored", "rule_assigned")]
    return {
        "schema_version": 1,
        "run_id": plan.run_id,
        "run_dir": str(run_dir),
        "prs": sum(counted) / len(counted) if counted else None,
        "denominator": len(counted),
        "counts": {status: counts[status] for status in ("scored", "rule_assigned", "rebuild_mismatch")},
        "rule": FLOOR_RULE,
        "model": plan.simulator_runtime.simulator_model,
        "prompt_files_sha256": dsr.prompt_files_sha256(),
        "cost_usd_total": sum(record["cost_usd_total"] for record in records.values()),
        "cost_missing_calls": sum(record["cost_missing_calls"] for record in records.values()),
        "probes": {
            pid: {
                "status": records[pid]["status"],
                "score": records[pid]["score"],
                "attribute": records[pid]["attribute"],
                "context": records[pid]["context"],
                **({"failures": records[pid]["rebuild_verification"]["failures"]}
                   if records[pid]["status"] == "rebuild_mismatch" else {}),
            }
            for pid in FINAL_PROBE_IDS
        },
    }


@contextmanager
def self_report_lock(run_dir: Path):
    folder = run_dir / RECORDS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SelfReportError(f"another self-report holds {folder / 'run.lock'}") from None
        yield


def _line(record: dict[str, Any]) -> str:
    parts = [f"self_report probe={record['probe_id']}", f"status={record['status']}", f"score={record['score']}"]
    if record["status"] == "scored":
        parts += [f"calls={len(record['attempts'])}", f"cost_usd={record['cost_usd_total']:.6f}"]
    if record["status"] == "rebuild_mismatch":
        parts.append("failures=" + ",".join(record["rebuild_verification"]["failures"]))
    return " ".join(parts)


def self_report(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not (run_dir / "run_plan.yaml").is_file():
        raise SelfReportError(f"{run_dir} is not a run directory: no run_plan.yaml")
    plan = load_run_plan(run_dir / "run_plan.yaml", project_root=PROJECT_ROOT)
    steps = final_steps(plan, run_dir / "run_plan.yaml")
    floored = floored_finals(run_dir)
    checkpoint_run = checkpoint_run_dir(run_dir)
    ledger = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))
    unfinished = [pid for pid in FINAL_PROBE_IDS
                  if pid not in floored and (ledger["steps"].get(pid) or {}).get("status") != "done"]
    if unfinished:
        raise SelfReportError(f"{run_dir}: finals neither done nor floored by BRS: {unfinished}")
    out = run_dir / RECORDS_DIR
    with self_report_lock(run_dir):
        # A summary exists only for a complete self-report of the run as it is now.
        (run_dir / SUMMARY_PATH).unlink(missing_ok=True)
        records: dict[str, dict[str, Any]] = {}
        queue: list[dict[str, Any]] = []
        conflicts: list[str] = []
        for pid in FINAL_PROBE_IDS:
            record = prepare(run_dir, plan, steps[pid], floored=pid in floored, checkpoint_run=checkpoint_run)
            path = out / f"{pid}.json"
            prior = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            if prior is not None and prior.get("status") == "scored":
                if record["status"] is not None or any(prior.get(key) != record[key] for key in RESUME_KEYS):
                    conflicts.append(pid)
                records[pid] = prior
                continue
            if prior is not None and {**prior, "created_at": None} == record:
                records[pid] = prior  # the same floor or mismatch as before
                continue
            if record["status"] is None:
                queue.append(record)
                continue
            record["created_at"] = datetime.now(timezone.utc).isoformat()
            write_json(path, record)
            records[pid] = record
            print(_line(record), flush=True)
        if conflicts:
            raise SelfReportError(
                f"{run_dir}: the inputs of these scored results changed since they were written: {conflicts}; "
                f"remove {out} to self-report the run again"
            )
        if queue and not os.environ.get("OPENROUTER_API_KEY"):
            raise SelfReportError("OPENROUTER_API_KEY is not set (export .env first)")
        if queue:
            asyncio.run(ask_each(queue, out, records))
        summary = summarize(plan, run_dir, records)
        write_json(run_dir / SUMMARY_PATH, summary)
    counts = summary["counts"]
    prs = "n/a" if summary["prs"] is None else f"{summary['prs']:.3f}"
    print(
        f"self_report_complete run={plan.run_id} prs={prs} scored={counts['scored']} "
        f"rule_assigned={counts['rule_assigned']} rebuild_mismatch={counts['rebuild_mismatch']} "
        f"cost_usd={summary['cost_usd_total']:.4f} summary={run_dir / SUMMARY_PATH}",
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    self_report(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
