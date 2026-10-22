"""Run ledger and concise progress formatting."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Literal

from harness.run_plan import RunPlan, RunStep


StepStatus = Literal["running", "done", "failed", "skipped"]
ResumeDecision = Literal["run", "skip"]
_STEP_STATUSES = frozenset({"running", "done", "failed", "skipped"})
PROBE_KINDS = frozenset({"pre_event_probe", "final_probe"})


class RunLedgerLoadError(ValueError):
    """The persisted ledger does not match the current run contract."""


class UnsafeRunResumeError(RuntimeError):
    """The run stopped inside a step whose side effects cannot be replayed."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StepRecord:
    status: StepStatus
    started_at: str | None = None
    ended_at: str | None = None
    error: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    #: One of `harness.failure_kind.FAILURE_KINDS` on a step that failed. None
    #: on every other step and on every ledger written before R3, which is why
    #: it is optional here and shown as `-` rather than guessed at.
    failure_kind: str | None = None
    #: Which counting rules `summary` was produced under. `tool_calls` already
    #: changed meaning once — the wrapper's own tools stopped counting — and the
    #: archive kept no trace of it; a bump is how the next change leaves one.
    summary_schema: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, step_id: str) -> StepRecord:
        if not isinstance(data, dict):
            raise RunLedgerLoadError(
                "Run ledger step record must be an object: "
                f"step_id={step_id!r} value={data!r}"
            )
        status = data.get("status")
        if status not in _STEP_STATUSES:
            raise RunLedgerLoadError(
                "Run ledger step has an invalid status: "
                f"step_id={step_id!r} value={status!r}"
            )
        artifacts = data.get("artifacts") or {}
        summary = data.get("summary") or {}
        if not isinstance(artifacts, dict) or not isinstance(summary, dict):
            raise RunLedgerLoadError(
                "Run ledger step artifacts and summary must be objects: "
                f"step_id={step_id!r} artifacts={artifacts!r} "
                f"summary={summary!r}"
            )
        return cls(
            status=status,
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            error=data.get("error"),
            artifacts=dict(artifacts),
            summary=dict(summary),
            failure_kind=data.get("failure_kind"),
            summary_schema=data.get("summary_schema"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }
        if self.error:
            data["error"] = self.error
        if self.artifacts:
            data["artifacts"] = self.artifacts
        if self.summary:
            data["summary"] = self.summary
        if self.failure_kind:
            data["failure_kind"] = self.failure_kind
        if self.summary_schema is not None:
            data["summary_schema"] = self.summary_schema
        return data


class RunLedger:
    """Durable `ledger.json` for one frozen benchmark run."""

    def __init__(self, run_dir: Path | str, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.path = self.run_dir / "ledger.json"
        self.current_step: str | None = None
        self.steps: dict[str, StepRecord] = {}
        if self.path.exists():
            self._load()
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.save()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunLedgerLoadError(
                "Failed to load run ledger: "
                f"path={str(self.path)!r} error={type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise RunLedgerLoadError(
                "Run ledger root must be an object: "
                f"path={str(self.path)!r} value={data!r}"
            )
        stored_run_id = data.get("run_id")
        if stored_run_id != self.run_id:
            raise RunLedgerLoadError(
                "Run ledger belongs to a different run: "
                f"path={str(self.path)!r} expected_run_id={self.run_id!r} "
                f"value={stored_run_id!r}"
            )
        self.current_step = data.get("current_step")
        if self.current_step is not None and not isinstance(
            self.current_step,
            str,
        ):
            raise RunLedgerLoadError(
                "Run ledger current_step must be a string or null: "
                f"path={str(self.path)!r} value={self.current_step!r}"
            )
        raw_steps = data.get("steps") or {}
        if not isinstance(raw_steps, dict):
            raise RunLedgerLoadError(
                "Run ledger steps must be an object: "
                f"path={str(self.path)!r} value={raw_steps!r}"
            )
        self.steps = {
            step_id: StepRecord.from_dict(record, step_id=step_id)
            for step_id, record in raw_steps.items()
        }
        if self.current_step is not None and self.current_step not in self.steps:
            raise RunLedgerLoadError(
                "Run ledger current_step has no matching step record: "
                f"path={str(self.path)!r} value={self.current_step!r}"
            )

    def dump_to(
        self,
        path: Path | str,
        *,
        override_done: str | None = None,
        override_status: tuple[str, str] | None = None,
        override_summary: dict[str, Any] | None = None,
        override_summary_schema: int | None = None,
        override_artifacts: dict[str, str] | None = None,
        override_error: str | None = None,
        override_failure_kind: str | None = None,
    ) -> Path:
        """Serialize the ledger to `path` atomically, without touching memory.

        `override_done` rewrites that one step as `mark_done` would (status
        done, fresh `ended_at`, no error) and clears `current_step` in the
        written file only. The in-memory ledger and the mainline file stay
        exactly as they are.

        `override_summary` is the summary `mark_done` is about to write for
        that step, because at snapshot time the step is still `running` and
        has none. It must be carried: a forced event's summary holds the
        `preference_overlay` flip the simulator's later steps are rebuilt
        from, and a rollback onto that event's own cut point replaces the
        mainline ledger with this file. Without it the flip exists nowhere
        and the resume replays the pre-event matrix.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        steps = {k: v.to_dict() for k, v in self.steps.items()}
        current_step = self.current_step
        if override_done is not None and override_status is not None:
            raise ValueError("ledger snapshot accepts one terminal override")
        terminal = override_status or (
            (override_done, "done") if override_done is not None else None
        )
        if terminal is not None:
            step_id, status = terminal
            if status not in {"done", "skipped"}:
                raise ValueError(f"unsupported terminal ledger status: {status!r}")
            record = dict(steps.get(step_id) or {"started_at": None})
            record["status"] = status
            record["ended_at"] = now_iso()
            record.pop("error", None)
            record.pop("failure_kind", None)
            if override_summary is not None:
                record["summary"] = override_summary
            if override_summary_schema is not None:
                record["summary_schema"] = override_summary_schema
            if override_artifacts is not None:
                record["artifacts"] = dict(override_artifacts)
            if override_error is not None:
                record["error"] = override_error
            if override_failure_kind is not None:
                record["failure_kind"] = override_failure_kind
            steps[step_id] = record
            current_step = None
        payload = json.dumps(
            {
                "run_id": self.run_id,
                "current_step": current_step,
                "steps": steps,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)
        return target

    def replace_from(self, source: Path | str) -> None:
        """Atomically replace `ledger.json` with `source` and reload it."""
        source_path = Path(source)
        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunLedgerLoadError(
                "Failed to read replacement run ledger: "
                f"path={str(source_path)!r} error={type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(data, dict) or data.get("run_id") != self.run_id:
            raise RunLedgerLoadError(
                "Replacement run ledger belongs to a different run: "
                f"path={str(source_path)!r} expected_run_id={self.run_id!r} "
                f"value={(data.get('run_id') if isinstance(data, dict) else data)!r}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        shutil.copyfile(source_path, tmp)
        os.replace(tmp, self.path)
        self._load()

    def save(self) -> None:
        self.dump_to(self.path)

    def _run_relative(self, artifacts: dict[str, str] | None) -> dict[str, str]:
        """Store artifact paths as paths *within* this run.

        An absolute path under `workspace/runs/` names where the run was on
        the day it ran, and goes stale the moment the run is archived or a
        snapshot of this ledger is restored somewhere else — which forced
        every reader to re-root it before use. What the run owns is the path
        inside its own directory, so that is what is written down. A path from
        outside this run is stored exactly as it came in.
        """
        stored: dict[str, str] = {}
        for name, raw in (artifacts or {}).items():
            try:
                stored[name] = str(PurePath(raw).relative_to(self.run_dir))
            except ValueError:
                stored[name] = raw
        return stored

    def forget(self, step_id: str) -> None:
        """Drop one step's record, so the next resume runs it again."""
        self.steps.pop(step_id, None)
        if self.current_step == step_id:
            self.current_step = None
        self.save()

    def decision_for(self, step_id: str) -> ResumeDecision:
        record = self.steps.get(step_id)
        if record and record.status in {"done", "skipped"}:
            return "skip"
        return "run"

    def assert_resumable(self, plan: RunPlan) -> None:
        """Refuse to resume past a timeline step that stopped mid-way.

        A probe step is never unsafe: its start point is an immutable pinned
        checkpoint and nothing of it reaches the mainline, so one stopped at
        `running` / `failed` is simply rerun (`decision_for` already says
        "run"). A run made only of probes — the memory-removed baseline — is
        therefore always resumable, checkpoint or not.
        """
        probe_step_ids = {
            step.step_id for step in plan.steps if step.kind in PROBE_KINDS
        }
        unsafe_steps = [
            {"step_id": step_id, "status": record.status}
            for step_id, record in self.steps.items()
            if record.status in {"running", "failed"}
            and step_id not in probe_step_ids
        ]
        if unsafe_steps:
            raise UnsafeRunResumeError(
                "Refusing to resume a run that stopped inside a step because "
                "PA-session or memory side effects may already exist: "
                f"run_id={self.run_id!r} unsafe_steps={unsafe_steps!r}. "
                "Start a clean run with a new run_id."
            )

    def assert_probe_attach_ready(self, plan: RunPlan) -> None:
        """Refuse to attach a probe plan to a timeline that is not finished.

        A probe branches a *pinned cut point* and is scored as the state of the
        run at that point, so attaching one to a run that stopped part-way
        measures a world the timeline never finished making. Worse, the
        pre-flight `assert_resumable` below is skipped as soon as the run has
        checkpoints, so such an attach would take the rollback path instead:
        the mainline would be rewound to the last snapshot and the probes would
        then read the rewound world without anything saying so.

        Checked whatever the checkpoints say, and only for an attach — a plan
        (re)frozen onto a ledger that already holds records. Continuing a
        timeline, probes in the plan or not, keeps the rollback path it was
        built for.
        """
        probe_step_ids = {
            step.step_id for step in plan.steps if step.kind in PROBE_KINDS
        }
        if not any(
            self.decision_for(step_id) == "run" for step_id in probe_step_ids
        ):
            return
        self.assert_resumable(plan)
        unfinished = [
            {
                "step_id": step.step_id,
                "status": (
                    record.status
                    if (record := self.steps.get(step.step_id))
                    else "never ran"
                ),
            }
            for step in plan.steps
            # A placeholder is a gap day: it keeps the acc numbering and runs
            # nothing, so it never reaches the ledger at all.
            if step.step_id not in probe_step_ids and not step.placeholder
            and self.decision_for(step.step_id) == "run"
        ]
        if unfinished:
            raise UnsafeRunResumeError(
                "Refusing to attach a probe plan to a run whose timeline has "
                "not finished, because every probe would branch a cut point "
                "the timeline never reached: "
                f"run_id={self.run_id!r} unfinished_steps={unfinished!r}. "
                "Finish the timeline first, then attach the probes."
            )

    def done_count(self) -> int:
        return sum(1 for record in self.steps.values() if record.status == "done")

    def next_step(self, plan: RunPlan) -> str | None:
        for step in plan.steps:
            if self.decision_for(step.step_id) == "run":
                return step.step_id
        return None

    def mark_running(self, step_id: str) -> None:
        self.current_step = step_id
        self.steps[step_id] = StepRecord(status="running", started_at=now_iso())
        self.save()

    def mark_done(
        self,
        step_id: str,
        *,
        artifacts: dict[str, str] | None = None,
        summary: dict[str, Any] | None = None,
        summary_schema: int | None = None,
    ) -> None:
        record = self.steps.get(step_id) or StepRecord(status="running")
        record.status = "done"
        record.ended_at = now_iso()
        record.error = None
        record.failure_kind = None
        record.artifacts = self._run_relative(artifacts)
        record.summary = summary or {}
        record.summary_schema = summary_schema
        self.steps[step_id] = record
        self.current_step = None
        self.save()

    def mark_failed(
        self,
        step_id: str,
        error: str,
        *,
        artifacts: dict[str, str] | None = None,
        kind: str | None = None,
    ) -> None:
        # Artifacts are recorded the same way `mark_done` records them: the
        # executor writes a failed step's transcript, tool calls and meta
        # before it re-raises, and a failed step is the one whose audit trail
        # `inspect` most needs to be able to point at.
        #
        # `kind` is one word out of `harness.failure_kind.FAILURE_KINDS`, which
        # is what separates "look at this by hand" from "archive it" from
        # "restore the last cut point" — a distinction `error` only ever made
        # in English prose.
        record = self.steps.get(step_id) or StepRecord(status="running")
        record.status = "failed"
        record.ended_at = now_iso()
        record.error = error
        record.failure_kind = kind
        record.artifacts = self._run_relative(artifacts)
        self.steps[step_id] = record
        self.current_step = step_id
        self.save()

    def mark_skipped(
        self,
        step_id: str,
        *,
        artifacts: dict[str, str] | None = None,
        error: str | None = None,
        kind: str | None = None,
        summary: dict[str, Any] | None = None,
        summary_schema: int | None = None,
    ) -> None:
        record = self.steps.get(step_id) or StepRecord(status="running")
        record.status = "skipped"
        record.ended_at = now_iso()
        record.error = error
        record.failure_kind = kind
        record.artifacts = self._run_relative(artifacts)
        record.summary = summary or {}
        record.summary_schema = summary_schema
        self.steps[step_id] = record
        self.current_step = None
        self.save()


def format_start(plan: RunPlan, *, start_time: datetime | None = None) -> str:
    start = start_time or datetime.now().astimezone()
    return (
        f"start {start.strftime('%Y-%m-%d %H:%M:%S %z')} "
        f"run={plan.run_id} persona={plan.persona_id} "
        f"memory={plan.memory_condition} steps={len(plan.steps)}"
    )


def format_resume(plan: RunPlan, ledger: RunLedger) -> str:
    next_step = ledger.next_step(plan) or "none"
    return f"resume {plan.run_id}: {ledger.done_count()} done, next {next_step}"


def format_step_running(index: int, total: int, step: RunStep, memory_condition: str) -> str:
    target = step.target_cell or "-"
    context = step.context or "-"
    return (
        f"[{index:03d}/{total:03d}] {step.step_id} {step.kind} {step.persona_id} "
        f"{context} {target} {memory_condition} {step.memory_abbrev} running"
    )


def format_step_done(index: int, total: int, step: RunStep, *, beats: int, tool_calls: int, elapsed_s: float) -> str:
    # The timestamp is appended, never prefixed: a long run is read back with
    # the positional `awk '{print $2 … $8}'` in the launch runbook and with
    # `grep -E "^\[[0-9]{3}/[0-9]{3}\] .* done "`, and both keep working only
    # while the leading fields stay where they are.
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return (
        f"[{index:03d}/{total:03d}] {step.step_id} done "
        f"{beats} beats {tool_calls} tool_calls {elapsed_s:.1f}s "
        f"at {finished_at}"
    )
