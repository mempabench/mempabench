"""Benchmark-level orchestrator for frozen MemPABench run plans."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from loguru import logger

from harness.probe_na import ATTEMPT_LIMIT
from harness.run_config import FailurePolicy
from harness.probe_slots import probe_batch_slot
from memory.adapter import PromptHistoryEvictionPolicy
from memory.audit import MemoryIOAuditSink
from harness.checkpoint import (
    LOCAL_MEMORY_CONDITIONS,
    SIMULATOR_WORKSPACE_LAYER,
    CheckpointManager,
    IncompleteCheckpointError,
    read_branch_namespace,
)
from harness.failure_kind import FAILED_STEP_ID_ATTR, PROBE_RETRY_KINDS, classify_failure
from harness.gate import resolve_compressed_reaction_index, resolve_session_gate
from harness.ix_seed import seed_ix_selection
from harness.memory_io import MemoryIORecorder
from harness.memory_runtime import MemoryRuntime
from harness.pa_factory import OpenHoldEpisodeError
from harness.run_plan import (
    RunPlan,
    RunStep,
    load_run_plan,
    resolve_probe_overlay,
    split_target_cell,
)
from harness.run_layout import orphan_branch_dirs, supersede_step_dirs
from harness.run_ledger import (
    PROBE_KINDS,
    RunLedger,
    UnsafeRunResumeError,
    format_resume,
    format_start,
    format_step_done,
    format_step_running,
)
from harness.session_executor import (
    FAILED_STEP_ARTIFACTS_ATTR,
    SUMMARY_SCHEMA,
    SessionExecutor,
    failed_step_artifacts,
)
from harness.state_runtime import StageRuntime
from memory.conditions.no_memory import NoMemoryAdapter
from memory.conditions.oracle_preference import OraclePreferenceAdapter
from memory.event_writer import (
    consolidation_records,
    records_owed_to_files,
)
from memory.results import HealthStatus
from pa.agent.ix.verification import (
    ENV_VAR as VERIFY_OUTREACH_ENV_VAR,
    verify_outreach_action,
)
from pa.session.manager import _JOURNAL_FILE_PATTERNS, SessionManager, SessionNotFoundError
from simulator.judge.client import JudgeUnavailable
from simulator.profiles import load_simulator_profile


def _three_miss_ix_exhaustion(error: BaseException) -> dict[str, Any] | None:
    """Typed proof for the one ordinary-ACC continue-and-record path."""
    if classify_failure(error) != "ix_runtime":
        return None
    value = getattr(error, "ix_contract_exhaustion", None)
    if not isinstance(value, dict) or value.get("attempts") != 3:
        return None
    if not value.get("contract") or not value.get("operation"):
        return None
    return dict(value)


def _record_abandoned_ix_holds(
    run_dir: Path, step_id: str, exhaustion: dict[str, Any],
    open_episodes: list[tuple[str, str]],
) -> Path:
    """Record holds returned by the already-completed session cleanup.

    These holds were never approved or executed. Abandoning their cleared
    session cache is not resolving them, and must not release banked actions.
    The file is listed in the skipped step's ledger artifacts.
    """
    if any(
        not isinstance(pair, (list, tuple)) or len(pair) != 2
        or any(not isinstance(value, str) or not value for value in pair)
        for pair in open_episodes
    ):
        raise ValueError("IX cleanup returned malformed unresolved episodes")
    path = run_dir / "steps" / step_id / "ix_session_cleanup.json"
    payload = {
        "schema": "mempa_ix_exhausted_session_cleanup/1",
        "step_id": step_id,
        "ix_contract_exhaustion": dict(exhaustion),
        "session_cleanup_completed": True,
        "disposition": "abandoned_unresolved_after_ix_exhaustion",
        "unresolved_hold_episodes": [
            {"attribute": attribute, "episode_id": episode_id}
            for attribute, episode_id in open_episodes
        ],
        "banked_actions_released": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _snapshot_memory_condition(cp_dir: Path, *, default: str) -> str:
    """The condition a snapshot was written under, per its own metadata."""
    try:
        metadata = json.loads(
            (cp_dir / "metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return default
    condition = metadata.get("memory_condition")
    return condition if isinstance(condition, str) and condition else default


ProgressSink = Callable[[str], None]
LEDGER_OVERLAY_KEY = "preference_overlay"
CHECKPOINT_READY_TIMEOUT_S = 300.0
#: How many times one probe is run before its cell is written off as a hole.
#: The retried kinds are
#: `failure_kind.PROBE_RETRY_KINDS`, every attempt is the same worker's own
#: work inside the slot the probe already holds, and the attempts before the
#: last are set aside as superseded cohorts rather than discarded.
#: Configured as `probe.attempt_limit` in configs/run_config.yaml.
PROBE_ATTEMPTS = ATTEMPT_LIMIT
#: The memory-removed baseline. Its probes are the only ones whose adapter is
#: replaced, and that follows from the condition alone: borrowing a start point
#: from another run says where the probe begins, not whether it has memory.
NO_MEMORY_CONDITION = "no_memory"
#: The realization ceiling:
#: the no_memory arm, with the probe cell's true setting as its only memory.
ORACLE_PREFERENCE_CONDITION = "oracle_preference"
SCENE_START_DIRNAME = "scene_start"
#: Ledger summary key of a skipped step: the turn ids whose memory write never
#: landed and that `partial_skip_pending_memory: drop` keeps out of memory.
DROPPED_PENDING_TURNS_KEY = "dropped_pending_turns"
#: Whole-scene attempts of each accumulation step (`max_scene_attempts`).
SCENE_ATTEMPTS_FILE = "scene_attempts.json"
SCENE_START_INDEX = "scene_start.json"
#: `meta.yaml` key a probe writes when its step boundary found a hold still
#: open. A probe never fails for it: the finding is recorded here.
OPEN_IX_EPISODES_KEY = "open_ix_episodes"


class ProbeRunConfigurationError(RuntimeError):
    """The run cannot branch a probe at all, whichever probe asked.

    The line `_run_probe_batch` draws: a probe's own failure is recorded and
    the batch moves on, but a defect in what every branch is cut from
    — no checkpoint manager, no adapter on an external backend — would fail all
    33 of them identically. Recording that 33 times would report a probe line
    that ran and measured nothing, so it stops the run instead.
    """


#: What ends the probe batch instead of failing one probe. The first two are
#: the branch machinery every probe shares; `JudgeUnavailable` is the third —
#: the probe that met the dead judge is recorded
#: `failed` like any other, and then the batch stops, because the judge will
#: still be gone for the next one. Everything else is one probe's own failure.
PROBE_BATCH_STOPPERS: tuple[type[BaseException], ...] = (
    ProbeRunConfigurationError,
    IncompleteCheckpointError,
    JudgeUnavailable,
)


@runtime_checkable
class PACheckpointHandle(Protocol):
    """The PA-side pieces a cut point needs, injected by the run entrypoint.

    The orchestrator never reaches into `pa/` itself: the PA and its session
    are built per step by the loop factory, so the entrypoint that owns them
    hands this handle over.
    """

    prompt_history_eviction_policy: PromptHistoryEvictionPolicy

    async def drain_background_tasks(self) -> None: ...
    def checkpoint_session(self) -> Any: ...
    def turn_journal(self) -> Any:
        """This session's pool journal, freshly read.

        The pinned flush consumes journal records, not `session.messages`, so
        the handle has to hand one over: the orchestrator holds a `Session`
        and never a path.
        """
        ...

    async def archive_messages_to_files(
        self, messages: list[dict[str, object]]
    ) -> bool: ...
    def save_session(self, session: Any) -> None: ...

    def close_ix_session(
        self, script_session_id: str
    ) -> list[tuple[str, str]]:
        """Clear this step's IX state; return the episodes still open.

        Optional: a handle that predates the wrapper simply does not define it,
        and the orchestrator then closes nothing. Probe branches never bind
        this mainline handle — each has its own `ProbePAHandle`, which dies
        with the branch directory.
        """
        ...


def resume_or_restore(
    ledger: RunLedger,
    checkpoint_manager: CheckpointManager | None,
    plan: RunPlan,
) -> str | None:
    """Make a crashed run resumable by rolling back, not by refusing.

    A run that died mid-step is unsafe to continue as it stands. If a snapshot
    exists, the honest move is to go back to it — every layer plus the ledger —
    and rerun from there. With no snapshot there is nothing to go back to, so
    the original refusal stands.

    Only a timeline step counts as "mid-step": the ledger classifies
    by the plan's step kinds, so a probe left at `running` / `failed` is rerun
    from its immutable pin with no rollback, and a probe-only baseline run
    passes with no local checkpoint at all.

    An unfinished rollback is handled *before* the ledger is consulted at all.
    A restore replaces `ledger.json` from the snapshot, so from the moment that
    lands the ledger reads as cleanly resumable while the layers around it are
    still half rolled back — asking the ledger first is what let such a run
    start the PA. The marker names the target, and the same target is what gets
    finished, however many times this has already been tried.
    """
    if checkpoint_manager is not None:
        pending = checkpoint_manager.pending_recovery()
        if pending is not None:
            target = str(pending["target_checkpoint_id"])
            logger.warning(
                "A rollback recovery to checkpoint {} is unfinished; finishing "
                "it before this run goes any further (recovery {})",
                target,
                pending.get("recovery_id"),
            )
            checkpoint_manager.restore(target, ledger=ledger)
            return target
    try:
        ledger.assert_resumable(plan)
    except UnsafeRunResumeError:
        if checkpoint_manager is None:
            raise
        available = checkpoint_manager.list_checkpoints()
        if not available:
            raise
        latest = max(available, key=checkpoint_manager.progress_key)
        logger.warning(
            "Run is not safely resumable as it stands; rolling back to "
            "checkpoint {}",
            latest,
        )
        checkpoint_manager.restore(latest, ledger=ledger)
        return latest
    return None


@dataclass
class OrchestratorConfig:
    run_dir: Path
    project_root: Path = Path(".")
    pa_workspace: Path | None = None
    emit_start: bool = True
    checkpoint_manager: CheckpointManager | None = None
    # Where the run directories a baseline plan borrows its start points from
    # live; defaults to the parent of this run's own directory, which assumes
    # the source run sits beside this one under the same
    # model/condition/persona. Set by `--borrowed-runs-root`
    # (`scripts/run_orchestrator.py`): until that flag existed nothing could
    # reach this field, so the same-parent assumption was the only behaviour
    # the field ever had.
    borrowed_runs_root: Path | None = None
    #: Cut points to pin beyond the ones this plan's own probes imply, so the
    #: accumulations can run now and a probe plan attach later off the same
    #: checkpoints. Named by the operator, because nothing in a
    #: probeless plan says which cut points a later plan will want. There is
    #: only one kind of pin: these take the full pinned save path.
    extra_pinned_step_ids: frozenset[str] = frozenset()
    #: The run's simulator profile. A probe runs in a throwaway branch with its
    #: own workspace, adapter and recorder, and the simulator identity used to
    #: ride along only because `loop_factory` closed over it — nothing in the
    #: branch's own `loop_context` or `meta.yaml` said which persona spoke. It
    #: is passed explicitly so the branch carries the same identity as an
    #: inline step and says so on its face.
    sim_profile: Any = None
    #: `--force-ix` seeds keyed by step id: `{step_id: {attribute: setting}}`,
    #: attribute one of `pa.agent.ix.specs.IX_PASSIVE_SPECS`, setting one of
    #: that spec's settings. A probe's seeds reach its branch PA through the
    #: probe-local handle; a timeline step's go through the entrypoint's own
    #: binder, which is why the orchestrator only reads probe ids from here.
    forced_ix: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Probe step ids to run this time; every other probe is left untouched in
    #: the ledger. None runs every probe the ledger has not marked done. The
    #: timeline phase ignores it (`--only-steps` is probe-phase only).
    only_steps: frozenset[str] | None = None
    #: Asks the loop to stop after the step in flight. Read at a step
    #: boundary only — never mid-step — so a stop
    #: leaves the ledger with no `running` record and the next resume has
    #: nothing to roll back. The runner's signal handler is what sets it.
    stop_requested: Callable[[], bool] | None = None
    #: Probe batch workers; 1 = the serial batch.
    probe_concurrency: int = 1
    #: Shared across every managed run; None is for isolated unit executors.
    probe_slot_root: Path | None = None
    shared_probe_memory_owner: Any | None = None
    probe_only: bool = False
    probe_source_ledger: Any | None = None
    #: What a failed timeline step does (`failure_policy` in
    #: configs/run_config.yaml, or the run's own `failure_policy.yaml`). The
    #: default stops the run on the first failure.
    failure_policy: FailurePolicy = field(default_factory=FailurePolicy)


@dataclass
class OrchestratorResult:
    run_id: str
    executed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


@dataclass
class RunPreferenceOverlay:
    """Run-level simulator preference state, flipped by forced events.

    A forced event moves its target cell to the event's `post_setting` for the
    rest of the run, in every memory condition: the simulated user is not the
    system under test, so what she has lived through must not vary with the
    PA's memory backend. The flip is written inside the forced-event step, so
    an event that never replayed leaves no entry and a later accumulation step
    fails fast instead of silently replaying the base matrix.
    """

    cells: dict[str, str] = field(default_factory=dict)

    def record(self, target_cell: str, post_setting: str) -> None:
        split_target_cell(target_cell)
        self.cells[target_cell] = post_setting

    def preference_overrides(self) -> dict[str, dict[str, dict[str, str]]]:
        """Shape the overlay as `SessionScript.preference_overrides` patches."""
        overrides: dict[str, dict[str, dict[str, str]]] = {}
        for target_cell, setting in self.cells.items():
            context, attribute = split_target_cell(target_cell)
            overrides.setdefault(context, {})[attribute] = {
                "setting": setting,
                "source": "forced_event",
                "confidence": "High",
            }
        return overrides

    def ledger_summary(self, step: RunStep) -> dict[str, dict[str, str]]:
        """The flip this forced-event step contributed, for `mark_done`."""
        if step.kind != "forced_event" or step.target_cell is None:
            return {}
        setting = self.cells.get(step.target_cell)
        if setting is None:
            return {}
        return {LEDGER_OVERLAY_KEY: {step.target_cell: setting}}

    def rebuild_from_ledger(
        self,
        plan: RunPlan,
        ledger: RunLedger | None,
        *,
        before_acc_num: int | None = None,
    ) -> None:
        """Replay the flips of every forced event the ledger already ran.

        `before_acc_num` cuts the replay at a point on the timeline: only the
        forced events that sit strictly before that acc number are replayed,
        which is what a `pre_event_probe` has to face. The mainline is a
        timeline and reaches its steps in order, so replaying everything is
        right for it; a probe is not, it branches a cut point in the middle of
        the run, and the run-level object handed to it carried flips from
        events that had not happened yet at that cut point. Live on user_b,
        every pre probe's simulator ran on the POST setting of its own cell —
        `pre_01`'s `meta.yaml` records `work.tone_formality: Consultative`
        while the probe was written to read `Casual`. Omitted, nothing is cut
        and the run-level behaviour is unchanged.
        """
        self.cells.clear()
        records = ledger.steps if ledger is not None else {}
        for step in plan.steps:
            if step.kind != "forced_event":
                continue
            if (
                before_acc_num is not None
                and step.acc_num is not None
                and step.acc_num >= before_acc_num
            ):
                continue
            record = records.get(step.step_id)
            if record is None or record.status != "done":
                continue
            for target_cell, setting in (
                record.summary.get(LEDGER_OVERLAY_KEY) or {}
            ).items():
                self.record(target_cell, setting)

    def require_forced_event_cells(self, plan: RunPlan, step: RunStep) -> None:
        """Fail before a step that depends on a flip the run never made."""
        assert step.acc_num is not None
        missing = [
            forced.target_cell
            for forced in plan.steps
            if forced.kind == "forced_event"
            and forced.acc_num is not None
            and forced.acc_num < step.acc_num
            and forced.target_cell
            and forced.target_cell not in self.cells
        ]
        if missing:
            raise RuntimeError(
                "Simulator preference overlay is missing forced-event cells "
                "that precede this step, so it would replay the pre-event "
                "matrix: "
                f"step_id={step.step_id!r} acc_num={step.acc_num!r} "
                f"missing_cells={missing!r}. The forced event must run before "
                "any later session; do not skip it in the ledger."
            )


def _branch_session_files(branch_dir: Path) -> list[Path]:
    """The branch's session files as `SessionManager.list_sessions` sees them: turn journals excluded."""
    return [
        path
        for path in sorted((branch_dir / "pa_workspace").glob("*/sessions/*.jsonl"))
        if not any(path.match(pattern) for pattern in _JOURNAL_FILE_PATTERNS)
    ]


def _prompt_history_at_start(branch_dir: Path) -> int:
    """History messages the branch sessions would hand the PA's first call."""
    total = 0
    for path in _branch_session_files(branch_dir):
        session = SessionManager(path.parent.parent)._load_from_path(path, path.stem)
        total += len(session.get_history(max_messages=0)) if session else 0
    return total


def probe_preference_overlay(
    plan: RunPlan, ledger: RunLedger | None, step: RunStep
) -> RunPreferenceOverlay:
    """The overlay this one probe's simulator runs under.

    A probe is not a point on the timeline, it is a branch off a cut point, so
    it gets its own object rather than the run-level one: a `pre_event_probe`
    as of its `before_acc_num` (its own cell must still be unflipped — that is
    the whole question it asks), a `final_probe` at the end of the run, where
    every flip the run made applies.
    """
    overlay = RunPreferenceOverlay()
    overlay.rebuild_from_ledger(
        plan, ledger, before_acc_num=step.before_acc_num
    )
    return overlay


class ProbePAHandle:
    """The narrow handle a probe branch keeps on its own PA.

    Three jobs and no more: seed the `--force-ix` selections into the branch
    PA when the loop factory binds it, close the step's IX session at the
    boundary so an open hold can be written into the probe's `meta.yaml`, and
    let go of the PA before the branch is discarded. The mainline cut-point
    handle is never bound to a probe PA — this one lives and dies with the
    branch directory.
    """

    def __init__(
        self,
        *,
        session_key: str,
        script_session_id: str,
        forced_ix: dict[str, str],
    ) -> None:
        self.session_key = session_key
        self.script_session_id = script_session_id
        self.forced_ix = dict(forced_ix)
        self._pa: Any | None = None

    def bind(self, pa: Any) -> None:
        """Called by the loop factory once the branch PA exists (`pa_binder`)."""
        self._pa = pa
        for attribute, setting in self.forced_ix.items():
            seed_ix_selection(
                pa,
                session_key=self.session_key,
                script_session_id=self.script_session_id,
                attribute=attribute,
                setting=setting,
            )
            logger.info(
                "forced_ix step={} attribute={} setting={}",
                self.script_session_id,
                attribute,
                setting,
            )

    def close_ix_session(self) -> list[tuple[str, str]]:
        """Clear the branch PA's IX state; return the episodes still open."""
        if self._pa is None:
            return []
        return list(
            self._pa.close_ix_session(self.session_key, self.script_session_id)
            or []
        )

    def release(self) -> None:
        self._pa = None


class Orchestrator:
    """Outer loop that coordinates a frozen benchmark plan."""

    def __init__(
        self,
        *,
        plan: RunPlan,
        config: OrchestratorConfig,
        executor: SessionExecutor,
        stage_runtime: StageRuntime,
        memory_runtime: MemoryRuntime,
        memory_io_recorder: MemoryIOAuditSink | None = None,
        ledger: RunLedger | None = None,
        progress: ProgressSink | None = None,
        preference_overlay: RunPreferenceOverlay | None = None,
        pa_checkpoint: PACheckpointHandle | None = None,
    ) -> None:
        self.pa_checkpoint = pa_checkpoint
        self.plan = plan
        self.config = config
        self.executor = executor
        self.stage_runtime = stage_runtime
        self.memory_runtime = memory_runtime
        self.memory_io_recorder = memory_io_recorder
        self.ledger = ledger or RunLedger(config.run_dir, plan.run_id)
        self.progress = progress or print
        self.preference_overlay = preference_overlay or RunPreferenceOverlay()
        #: The probe ids the batch runner has marked `running` and not yet
        #: recorded. `ledger.current_step` is last-writer-wins once more than
        #: one probe is in flight, so this set — not the ledger — is what names
        #: them; it is what a stop message and `_reap_orphan_branches(active=)`
        #: have to be given.
        self.in_flight: set[str] = set()
        #: This launch of the run (one start or resume). A plain run's probe
        #: attempt records name it (`harness.probe_launch_attempts`), and the
        #: entrypoint writes it into the launch's `run_info.yaml` record.
        self.launch_id = uuid.uuid4().hex

    async def run(self) -> OrchestratorResult:
        result = OrchestratorResult(run_id=self.plan.run_id)
        total = len(self.plan.steps)
        # Before the world is touched at all: fail closed on a bad value before
        # a single provider call is paid for. A good value is announced,
        # because the run it produces is a verification run and not a scored
        # one (`pa/agent/ix/verification.py`).
        if (forced_outreach := verify_outreach_action()) is not None:
            self.progress(
                f"{VERIFY_OUTREACH_ENV_VAR}={forced_outreach} — VERIFICATION RUN, "
                "the PA's outreach decision is overridden and the run is not scored"
            )
        checkpoint_manager = self.config.checkpoint_manager
        policy = self.config.failure_policy
        skippable = {
            step.step_id for step in self.plan.steps
            if step.kind == "accumulation" and not step.placeholder
        }
        if set(policy.skip_steps) - skippable:
            raise ValueError(
                "failure_policy.skip_steps may name only accumulation steps of "
                f"this plan: {sorted(set(policy.skip_steps) - skippable)}"
            )
        # A rollback rewrites the ledger from the snapshot, so the overlay
        # below must be rebuilt from what the ledger says afterwards.
        if not self.config.probe_only:
            resume_or_restore(self.ledger, self.config.checkpoint_manager, self.plan)
        # The rollback puts the run back on its snapshot; the journal is what
        # says which turns died with a pool the adapter never received. The
        # replay is at-least-once and declared, so this is the one place a run
        # replays them, before any step of it runs again.
        if not self.config.probe_only:
            await self._replay_pending_turns()
        self._reap_orphan_branches()
        # Resume rebuilds the overlay from the forced events the ledger already
        # ran; their flips are state, not artifacts, so they do not survive the
        # process on their own.
        self.preference_overlay.rebuild_from_ledger(
            self.plan, self.config.probe_source_ledger or self.ledger
        )
        health_request = {
            "scope": {
                "run_id": self.memory_runtime.scope.run_id,
                "persona_id": self.memory_runtime.scope.persona_id,
                "agent_id": self.memory_runtime.scope.agent_id,
                "backend_namespace": (
                    self.memory_runtime.scope.backend_namespace
                ),
            },
            "qdrant_endpoint": (
                {
                    "host": self.plan.qdrant_host,
                    "port": self.plan.qdrant_port,
                }
                if self.plan.qdrant_host
                else None
            ),
        }
        try:
            await self.memory_runtime.setup()
            health = await self.memory_runtime.health()
        except Exception as exc:
            self._record_memory_health(
                health_request,
                status="failed",
                error=exc,
            )
            raise
        if health is not None:
            self._record_memory_health(
                health_request,
                status=health.status.value,
                health=health,
            )
        if health is not None and health.status != HealthStatus.ok:
            raise RuntimeError(
                "Memory backend is not ready at run start: "
                f"backend={health.backend_name!r} status={health.status.value!r} "
                f"warnings={health.warnings!r}"
            )
        if self.config.pa_workspace is not None:
            session_manager = SessionManager(self.config.pa_workspace)
            main_session_key = self.plan.main_pa_session_key()
            if self.config.probe_only:
                try:
                    session_manager.get(main_session_key)
                except SessionNotFoundError:
                    session_manager.create(main_session_key)
            elif self.ledger.steps:
                session_manager.get(main_session_key)
            else:
                session_manager.create(main_session_key)
        if self.config.emit_start:
            self.progress(format_start(self.plan))
        if self.ledger.done_count():
            self.progress(format_resume(self.plan, self.ledger))

        pinned = self.plan.pinned_step_ids() | set(
            self.config.extra_pinned_step_ids
        )
        self._require_only_steps_are_probes()

        # The timeline runs to the end first; every probe is a job of
        # its own, collected here and run as one batch afterwards.
        probes: list[tuple[int, RunStep]] = []
        stopped = False
        for index, step in enumerate[RunStep](self.plan.steps, start=1):
            # The only place a stop is read: between two steps, after the
            # previous one published its cut point and its ledger record. A
            # stop taken anywhere else would leave the ledger `running` and
            # cost the session it interrupted.
            if self._stop_requested():
                stopped = True
                break
            if step.kind in PROBE_KINDS:
                probes.append((index, step))
                continue
            if self.config.probe_only:
                result.skipped.append(step.step_id)
                continue
            if step.placeholder:
                # A gap day (`run_plan_builder._placeholder_step`): the slot
                # keeps the acc numbering and runs nothing. No script is read,
                # no stage is opened, no cut point is published, and neither
                # the PA nor the simulator is ever called for it.
                result.skipped.append(step.step_id)
                continue
            if self.ledger.decision_for(step.step_id) == "skip":
                result.skipped.append(step.step_id)
                continue
            if policy.skip_steps.get(step.step_id) == "rollback":
                # The failed attempt was rolled back with the rest of the run
                # (`resume_or_restore` above): record the step skipped and
                # keep nothing of it.
                self.ledger.mark_skipped(
                    step.step_id,
                    summary={"skip_reason": "failure_policy.skip_steps"},
                )
                result.skipped.append(step.step_id)
                self.progress(f"skip {step.step_id}: failure_policy.skip_steps rollback")
                continue

            if step.kind == "accumulation":
                self.preference_overlay.require_forced_event_cells(
                    self.plan,
                    step,
                )

            self.progress(format_step_running(index, total, step, self.plan.memory_condition))
            self.ledger.mark_running(step.step_id)
            self._begin_memory_side_usage(step.step_id)
            step_stage = None
            stage_finished = False
            execution = None
            open_episodes: list[tuple[str, str]] = []
            # Where this step's own journal records begin: a skip that drops
            # its unwritten memory leaves exactly those out of the memory files.
            journal_records_at_start = (
                len(self.pa_checkpoint.turn_journal().records())
                if self.pa_checkpoint is not None
                else 0
            )
            try:
                await self.memory_runtime.apply_mode(step.memory_mode, reason=step.step_id)
                step_stage = self.stage_runtime.begin_step(
                    step_id=step.step_id,
                    # step_id is the artifact's session identity (acc_001 /
                    # evolv_05); the script filename is not (a forced script is
                    # .../evolv_05/transcript.yaml, whose stem is "transcript").
                    session_id=step.step_id,
                    runtime_artifact_path=(
                        self.config.project_root / step.script_path
                    ),
                    session_date=(
                        step.benchmark_session_timestamp.date().isoformat()
                    ),
                    session_timestamp=(
                        step.benchmark_session_timestamp.isoformat()
                    ),
                    gate=resolve_session_gate(
                        self.config.project_root / step.script_path,
                        self.config.project_root,
                    ),
                    compressed_reaction_index=resolve_compressed_reaction_index(
                        self.config.project_root / step.script_path
                    ),
                )
                extra_meta = {
                    "run_id": self.plan.run_id,
                    "memory_condition": self.plan.memory_condition,
                    "script_session_id": step.step_id,
                    "stage_root": str(step_stage.state_root),
                    "simulator_runtime": (
                        self.plan.simulator_runtime.model_dump(mode="json")
                    ),
                }
                if step_stage.gate:
                    # The step directory has to say the session ran gated;
                    # `gate_loaded` in tool_log.jsonl says the server agreed.
                    extra_meta["gate"] = dict(step_stage.gate)
                if self.memory_runtime.scope.backend_namespace:
                    extra_meta["memory_backend_namespace"] = (
                        self.memory_runtime.scope.backend_namespace
                    )
                if step.kind in {"accumulation", "forced_event"}:
                    extra_meta["pa_session_key"] = self.plan.pa_session_key_for_step(step)
                if self.config.pa_workspace is not None:
                    extra_meta["pa_workspace"] = str(self.config.pa_workspace)
                try:
                    execution = await self.executor.execute(
                        step,
                        steps_dir=self.config.run_dir / "steps",
                        stage_runtime=step_stage,
                        extra_meta=extra_meta,
                    )
                finally:
                    # Success, exception, cancel and same-step retry all
                    # clear the IX state. The close itself never masks the
                    # step's own error — what it found is acted on below.
                    open_episodes = self._close_step_ix_session(step)
                # A step that left a hold open is a failed step, so its world
                # writes are discarded with the working stage instead of being
                # committed: `canonical_stage/` stays byte-identical to what it
                # was before the step, the same guarantee the executor's own
                # exceptions get (contract 7, T17).
                self.stage_runtime.finish_step(
                    step_stage,
                    "discard" if open_episodes else step.stage_policy,
                )
                stage_finished = True
                if open_episodes:
                    # Raised before the checkpoint block, so a step that left a
                    # hold open publishes no cut point (contract 7, T17).
                    raise OpenHoldEpisodeError(step.step_id, open_episodes)
                # Built before the checkpoint block: the snapshot's own ledger
                # records this step as done, so it has to record the same
                # summary the `mark_done` below writes — a forced event's
                # `preference_overlay` flip is state that lives nowhere else,
                # and a rollback onto this cut point is what reads it back.
                step_summary = {
                    "beats": execution.beats,
                    "tool_calls": execution.tool_calls,
                    "elapsed_s": execution.elapsed_s,
                    **self.preference_overlay.ledger_summary(step),
                }
                if (
                    checkpoint_manager is not None
                    and step.kind in {"accumulation", "forced_event"}
                ):
                    # Every cut point waits for in-flight work first; only
                    # pinned ones then flush the never-compacted turns. A step
                    # counts as done once its checkpoint is published, so the
                    # mainline ledger and the snapshot's own ledger agree.
                    await self._await_checkpoint_ready()
                    is_pinned = step.step_id in pinned
                    if is_pinned:
                        await self._flush_uncompacted_turns()
                    checkpoint_manager.save(
                        step.step_id,
                        pinned=is_pinned,
                        summary=step_summary,
                        summary_schema=SUMMARY_SCHEMA,
                    )
                self.ledger.mark_done(
                    step.step_id,
                    artifacts=execution.artifacts,
                    summary=step_summary,
                    summary_schema=SUMMARY_SCHEMA,
                )
                result.executed.append(step.step_id)
                self._flush_memory_side_usage()
                self.progress(
                    format_step_done(
                        index,
                        total,
                        step,
                        beats=execution.beats,
                        tool_calls=execution.tool_calls,
                        elapsed_s=execution.elapsed_s,
                    )
                )
            except Exception as exc:
                skip_reason = self._failure_policy_skip(step, exc)
                if skip_reason is not None:
                    await self._skip_retaining_partial(
                        step,
                        exc,
                        reason=skip_reason,
                        step_stage=None if stage_finished else step_stage,
                        open_episodes=open_episodes,
                        execution=execution,
                        pinned=step.step_id in pinned,
                        journal_records_at_start=journal_records_at_start,
                    )
                    result.skipped.append(step.step_id)
                    self._flush_memory_side_usage()
                    continue
                # The executor writes a failed step's artifacts before it
                # re-raises; record their paths so `inspect` can point at the
                # transcript of the session that died. When the step got past
                # the executor and died later (an open hold, a checkpoint),
                # `execution` already holds them.
                self.ledger.mark_failed(
                    step.step_id,
                    str(exc),
                    artifacts=(
                        execution.artifacts
                        if execution is not None
                        else failed_step_artifacts(exc)
                    ),
                    # The same exception, said in one word: an open hold, a
                    # contract boundary, the provider and a memory backend all
                    # read as `non_resumable(<step>)` without it, and they are
                    # handled four different ways.
                    kind=classify_failure(exc),
                )
                # Read by `mempabench run` for its stop line.
                setattr(exc, FAILED_STEP_ID_ATTR, step.step_id)
                result.failed.append(step.step_id)
                raise
        # A stop taken at a timeline boundary ends the run there. The probe
        # batch is a phase of its own and belongs to the resume, not
        # to the stop: `probes` stops being the whole list the moment the loop
        # breaks, and every probe left unrecorded is picked up next time.
        if not stopped:
            pending = any(
                self.ledger.decision_for(step.step_id) != "skip"
                and (self.config.only_steps is None or step.step_id in self.config.only_steps)
                for _, step in probes
            )
            async with probe_batch_slot(
                self.config.probe_slot_root if pending else None,
                stop_requested=self._stop_requested,
                workers=self.config.probe_concurrency,
                progress=self.progress,
            ) as acquired:
                if acquired:
                    await self._run_probe_batch(
                        probes,
                        result=result,
                        total=total,
                        checkpoint_manager=checkpoint_manager,
                        pinned=pinned,
                    )
        return result

    def _require_only_steps_are_probes(self) -> None:
        only = self.config.only_steps
        if only is None:
            return
        probe_ids = {
            step.step_id for step in self.plan.steps if step.kind in PROBE_KINDS
        }
        unknown = sorted(only - probe_ids)
        if unknown:
            raise ValueError(
                "--only-steps names steps that are not probe steps of this "
                f"plan: {unknown!r} (probe steps: {sorted(probe_ids)!r})"
            )

    def _reap_orphan_branches(self, active: frozenset[str] = frozenset()) -> None:
        """Remove `branches/<step>/NN` a hard death left behind.

        A branch is deleted by the probe that made it, so at run start any
        that exists belongs to no live probe. `active` names the probes whose
        branches are in use — empty at run start, and the hook for a parallel
        batch runner that reaps between jobs.

        The remote namespace goes first, through the same release the normal
        finish uses. A branch that died
        hard still owns the Mem0 collection or MemOS cube it restored for
        itself, and the only thing that still names it is the
        `branch_namespace.json` inside the directory — so an `rmtree` on its
        own leaks the namespace and throws away the name that could have
        cleaned it up later. A hard kill under a parallel batch leaks one per
        in-flight probe. When the release fails the directory is kept, so the
        name survives for the next run to retry.
        """
        for relative in orphan_branch_dirs(self.config.run_dir):
            step_id = relative.split("/")[1]
            if step_id in active:
                continue
            branch_dir = self.config.run_dir / relative
            error = self._release_branch_namespaces(
                branch_dir, self.config.checkpoint_manager
            )
            if error is not None:
                kept = f"orphan branch kept: {branch_dir} release failed: {error}"
                logger.warning(kept)
                self.progress(kept)
                continue
            logger.warning("Removing orphan probe branch {}", relative)
            self.progress(f"reap orphan branch {relative}")
            shutil.rmtree(branch_dir, ignore_errors=True)

    def _failure_policy_skip(self, step: RunStep, exc: BaseException) -> str | None:
        """Why `failure_policy` skips this failed step, or None: the run stops.

        Only an accumulation step is ever skipped. `max_scene_attempts` counts
        whole-scene attempts across launches in `scene_attempts.json`: until
        the last one the run stops (the next launch rolls back and runs the
        scene again), and the last one failing is skipped.
        """
        policy = self.config.failure_policy
        if step.kind != "accumulation":
            return None
        if policy.skip_steps.get(step.step_id) == "retain_partial":
            return "failure_policy.skip_steps"
        if policy.auto_skip_on_ix_exhaustion and _three_miss_ix_exhaustion(exc) is not None:
            return "failure_policy.auto_skip_on_ix_exhaustion"
        if policy.max_scene_attempts > 1:
            path = self.config.run_dir / SCENE_ATTEMPTS_FILE
            counts = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            counts[step.step_id] = int(counts.get(step.step_id, 0)) + 1
            path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if counts[step.step_id] >= policy.max_scene_attempts:
                return "failure_policy.max_scene_attempts"
        return None

    async def _skip_retaining_partial(
        self,
        step: RunStep,
        exc: BaseException,
        *,
        reason: str,
        step_stage: Any,
        open_episodes: list[tuple[str, str]],
        execution: Any,
        pinned: bool,
        journal_records_at_start: int,
    ) -> None:
        """Keep what the failed scene left and record the step skipped.

        The scene is not run again: its working stage is committed (unless
        the failure already discarded it), holds it left open are recorded as
        abandoned (never approved or released), and its turns whose memory
        write never landed are written now (`partial_skip_pending_memory:
        replay`) or left out of memory (`drop`, the default: their turn ids go
        into the ledger summary; an adapter's cursor moves past them, and for
        the file condition the step's own journal records the memory files
        have not had yet, from `journal_records_at_start` on, are marked
        `dropped_from_files` so no later consolidation writes them). Then a
        retained skipped checkpoint is published and the ledger marks the
        step skipped.
        """
        run_dir = self.config.run_dir
        artifacts = dict(
            execution.artifacts if execution is not None else failed_step_artifacts(exc)
        )
        failed_events = run_dir / "steps" / step.step_id / "failed_turn_events.json"
        if failed_events.is_file():
            artifacts["failed_turn_events"] = str(failed_events)
        exhaustion = _three_miss_ix_exhaustion(exc)
        if open_episodes:
            artifacts["ix_session_cleanup"] = str(
                _record_abandoned_ix_holds(
                    run_dir, step.step_id, exhaustion or {}, open_episodes
                )
            )
        if step_stage is not None:
            self.stage_runtime.finish_step(step_stage, step.stage_policy)
        summary: dict[str, Any] = {"beats": 0, "tool_calls": 0, "skip_reason": reason}
        if exhaustion is not None:
            summary["ix_in_scene_exhaustion"] = exhaustion
        handle = self.pa_checkpoint
        drop = self.config.failure_policy.partial_skip_pending_memory == "drop"
        if (
            handle is not None
            and self.memory_runtime.adapter is not None
            and handle.prompt_history_eviction_policy
            == PromptHistoryEvictionPolicy.after_file_consolidation
        ):
            # The file condition writes at consolidation time, so what this
            # step's turns owe the files is still owed: `replay` leaves them
            # to the next consolidation, `drop` keeps them out for good.
            if drop:
                journal = handle.turn_journal()
                owed = records_owed_to_files(
                    journal.records()[journal_records_at_start:]
                )
                if owed:
                    dropped = [record.turn_id for record in owed]
                    journal.mark_dropped_from_files(dropped)
                    summary[DROPPED_PENDING_TURNS_KEY] = dropped
        elif handle is not None and self.memory_runtime.adapter is not None:
            pending = handle.turn_journal().pending_records()
            if pending and not drop:
                await self._replay_pending_turns()
            elif pending:
                summary[DROPPED_PENDING_TURNS_KEY] = [record.turn_id for record in pending]
                if (
                    handle.prompt_history_eviction_policy
                    == PromptHistoryEvictionPolicy.after_adapter_commit
                ):
                    session = handle.checkpoint_session()
                    session.adapter_committed_end_index = max(
                        session.adapter_committed_end_index,
                        max(record.span[1] for record in pending),
                    )
                    handle.save_session(session)
        kind = classify_failure(exc)
        if self.config.checkpoint_manager is not None:
            await self._await_checkpoint_ready()
            self.config.checkpoint_manager.save_skipped(
                step.step_id,
                pinned=pinned,
                summary=summary,
                summary_schema=SUMMARY_SCHEMA,
                artifacts=artifacts,
                error=str(exc),
                failure_kind=kind,
            )
        self.ledger.mark_skipped(
            step.step_id,
            artifacts=artifacts,
            error=str(exc),
            kind=kind,
            summary=summary,
            summary_schema=SUMMARY_SCHEMA,
        )
        self.progress(f"skip {step.step_id}: {reason}; partial state retained")

    def _stop_requested(self) -> bool:
        """Whether the operator has asked the run to stop."""
        return (
            self.config.stop_requested is not None
            and self.config.stop_requested()
        )

    async def _run_probe_batch(
        self,
        probes: list[tuple[int, RunStep]],
        *,
        result: OrchestratorResult,
        total: int,
        checkpoint_manager: CheckpointManager | None,
        pinned: set[str],
    ) -> None:
        """Dispatch every pending probe over `probe_concurrency` workers.

        Each probe is self-contained (its own pin, branch and evidence), so the
        batch is a queue and N workers draining it; `_run_one_probe` is the one
        body, entered the same way at every N, and N=1 runs the probes in plan
        order exactly as the serial loop did.

        A probe that fails is marked `failed` with its `failure_kind` and the
        batch moves on — the run never stops on a probe, and a probe the judge
        voided is rerun by its own worker first (`_run_one_probe`).
        Three exceptions stop
        it: a defect in what every branch is cut from
        (`ProbeRunConfigurationError`, `IncompleteCheckpointError`), which is
        not this probe failing but the probe line not being runnable, and a
        dead judge (`JudgeUnavailable`), which the
        next probe would meet again. A stopper sets `abort`: no queued probe is
        started after it, the probes already in flight run to their end and are
        recorded, and the first stopper is then re-raised so the process exits
        non-zero.
        """
        if not probes:
            return
        # The ledger's verdict and `--only-steps` are applied here, as the
        # queue is filled, rather than inside the probe body: `result.skipped`
        # is then in plan order whatever order the workers finish in, and a
        # probe that is not going to run never occupies a worker slot.
        runnable: list[tuple[int, RunStep]] = []
        for index, step in probes:
            if self.ledger.decision_for(step.step_id) == "skip":
                result.skipped.append(step.step_id)
                continue
            if (
                self.config.only_steps is not None
                and step.step_id not in self.config.only_steps
            ):
                result.skipped.append(step.step_id)
                continue
            if self._exhausted_repeat_probe(step.step_id):
                # Its failure record is N/A evidence; a resume leaves it as is.
                result.failed.append(step.step_id)
                continue
            runnable.append((index, step))
        if not runnable:
            return
        first_step_id = runnable[0][1].step_id
        if checkpoint_manager is None:
            # A probe runs only in a branch cut from a pin. There is no
            # mainline fallback, and it is one defect for the whole batch, not
            # 33 failures, so it is raised once here — before the mainline
            # runtime below is touched.
            raise ProbeRunConfigurationError(
                "A probe runs only in a branch cut from a pinned "
                "checkpoint, and this run has no checkpoint manager: "
                f"step_id={first_step_id!r} probe_count={len(runnable)}"
            )
        # Hoisted out of the per-probe body: every probe is `read_only` by plan
        # validation (`run_plan.py`), and the runtime this touches is the
        # shared mainline one, so it is set once for the batch instead of once
        # per probe from N workers at the same time. A backend that will not go
        # read_only is re-raised as a stopper rather than left to fall out of
        # `run()` untyped: it is the probe line being unrunnable, not one probe
        # failing, and no probe may write to the memory the line is measured
        # against.
        try:
            await self.memory_runtime.apply_mode(
                "read_only", reason="probe_batch"
            )
        except Exception as exc:
            raise ProbeRunConfigurationError(
                "The mainline memory runtime could not be put in read_only "
                f"for the probe batch: step_id={first_step_id!r} "
                f"probe_count={len(runnable)}: {exc}"
            ) from exc
        queue: asyncio.Queue[tuple[int, RunStep]] = asyncio.Queue()
        for job in runnable:
            queue.put_nowait(job)
        workers = min(max(1, int(self.config.probe_concurrency)), len(runnable))
        abort = asyncio.Event()
        stopper: BaseException | None = None

        async def worker() -> None:
            nonlocal stopper
            while True:
                # Both stop conditions are read here, before a probe is taken
                # off the queue: what is left on it is left unrecorded, and the
                # next resume runs it.
                if abort.is_set() or self._stop_requested():
                    return
                try:
                    index, step = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await self._run_one_probe(
                        index,
                        step,
                        result=result,
                        total=total,
                        checkpoint_manager=checkpoint_manager,
                        pinned=pinned,
                        abort=abort,
                        workers=workers,
                    )
                except PROBE_BATCH_STOPPERS as exc:
                    # First one wins, and nothing awaits between the two lines,
                    # so the losers see `abort` already set.
                    if stopper is None:
                        stopper = exc
                        abort.set()
                    return

        tasks = [asyncio.create_task(worker()) for _ in range(workers)]
        # `return_exceptions` so every worker is awaited even when one dies:
        # the contract is that no probe is abandoned mid-flight.
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        if stopper is not None:
            raise stopper
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome

    def _exhausted_repeat_probe(self, step_id: str) -> bool:
        """A shared-memory repeat Probe already failed with no attempt left."""
        record = self.ledger.steps.get(step_id)
        if self.config.shared_probe_memory_owner is None or record is None or record.status != "failed":
            return False
        from harness.probe_attempts import ProbeAttempts
        try:
            ProbeAttempts(self.config.run_dir, step_id).next_number()
        except RuntimeError as exc:
            return "exhausted" in str(exc) or "six total" in str(exc)
        return False

    async def _run_one_probe(
        self,
        index: int,
        step: RunStep,
        *,
        result: OrchestratorResult,
        total: int,
        checkpoint_manager: CheckpointManager,
        pinned: set[str],
        abort: asyncio.Event,
        workers: int,
    ) -> None:
        """One probe, from the ledger's verdict on it to its ledger record."""
        # Both stop conditions are read again here, with nothing awaited
        # between them and `mark_running`: a stopper raised, or a signal taken,
        # while this probe sat in the queue must leave it unrecorded, exactly
        # as if it had never been dequeued, because unrecorded is what the next
        # resume runs.
        if abort.is_set() or self._stop_requested():
            return
        first_attempt = 1
        self.progress(
            format_step_running(index, total, step, self.plan.memory_condition)
        )
        self.ledger.mark_running(step.step_id)
        self.in_flight.add(step.step_id)
        # A separate line kind on purpose: the two step lines above and below
        # are read positionally by the runbook's `awk`/`grep`, so the slot the
        # dispatcher opened says so on a line of its own.
        self.progress(
            f"probe slot open id={step.step_id} "
            f"inflight={len(self.in_flight)}/{workers}"
        )
        status: str | None = None
        try:
            # A voided probe is rerun immediately.
            # A rerun after the batch is the wrong shape at N>1, so the retry
            # is this worker's own work: a fresh branch off the same pin,
            # inside the slot this probe already holds.
            #
            # The cap is `PROBE_ATTEMPTS` and `ix_runtime` is one of the retried
            # kinds (`PROBE_RETRY_KINDS`), so the PA gets three chances at a probe
            # before the cell is written off, and the failure the report surfaces
            # is the third.
            # `first_attempt` is 1 unless the shared-memory attempt record
            # resumes this Probe further on.
            last_attempt = PROBE_ATTEMPTS
            if self.config.shared_probe_memory_owner is not None:
                from harness.probe_attempts import ProbeAttempts
                try:
                    record = ProbeAttempts(self.config.run_dir, step.step_id)
                    first_attempt, last_attempt = record.next_number(), record.ceiling()
                except RuntimeError as exc:
                    self.ledger.mark_failed(step.step_id, str(exc), kind="probe_attempt_budget")
                    result.failed.append(step.step_id)
                    status = "failed"
                    return
                launch_attempts = None
            else:
                # A plain run records each attempt with this launch's id: the
                # evidence `score --not-applicable-exhausted` counts. A record
                # that cannot be written raises, and the run stops there.
                from harness.probe_launch_attempts import ProbeLaunchAttempts
                launch_attempts = ProbeLaunchAttempts(self.config.run_dir, step.step_id, self.launch_id)
            # A resumed Probe whose earlier network failures did not count may
            # start past PROBE_ATTEMPTS; it still gets this attempt.
            for attempt in range(first_attempt, max(last_attempt, first_attempt) + 1):
                if launch_attempts is not None:
                    launch_attempts.start(attempt)
                try:
                    execution = await self._run_probe_in_branch(
                        step,
                        checkpoint_manager=checkpoint_manager,
                        pinned=pinned,
                    )
                except (ProbeRunConfigurationError, IncompleteCheckpointError):
                    # Not this probe failing: the branch machinery every probe
                    # shares is broken, so the next 32 would die the same way.
                    raise
                except Exception as exc:
                    kind = classify_failure(exc)
                    retryable = kind in PROBE_RETRY_KINDS
                    if launch_attempts is not None:
                        # On disk before the run acts on it: the set-aside and
                        # retry below, or the ledger record.
                        launch_attempts.fail(attempt, failure_kind=kind, error=exc)
                    if (
                        retryable
                        and attempt < last_attempt
                        # The two start gates once more: a retry is a new start,
                        # and nothing new starts after a stopper or a signal.
                        # The failure is then recorded without a retry.
                        and not (abort.is_set() or self._stop_requested())
                    ):
                        # The failure path has already copied this attempt's
                        # evidence into `steps/<probe_id>/`; it is set aside as
                        # a superseded cohort of its own, both so the retry
                        # writes onto a clean directory and so the void is
                        # still readable (`--list-cohorts`). No ledger record
                        # is written for it: the step stays `running` from
                        # `mark_running` above until the retry ends it, so the
                        # one record per step keeps naming the outcome.
                        supersede_step_dirs(self.config.run_dir, [step.step_id])
                        logger.warning(
                            "Probe {} failed ({}); rerunning it now, "
                            "attempt {} of {}",
                            step.step_id,
                            kind,
                            attempt + 1,
                            last_attempt,
                        )
                        # Its own line kind: the retry prints this probe's real
                        # step line, so this one must not read as either of the
                        # two the runbook greps. The `probe void`
                        # prefix predates `ix_runtime` joining the retried
                        # kinds and is kept so those greps keep matching; the
                        # `kind=` field is what says which failure it was.
                        self.progress(
                            f"probe void id={step.step_id} kind={kind} "
                            f"attempt={attempt} retrying"
                        )
                        continue
                    self.ledger.mark_failed(
                        step.step_id,
                        str(exc),
                        artifacts=failed_step_artifacts(exc),
                        kind=kind,
                    )
                    # Read by `mempabench run` for its stop line when this
                    # failure stops the batch (`JudgeUnavailable` below).
                    setattr(exc, FAILED_STEP_ID_ATTR, step.step_id)
                    result.failed.append(step.step_id)
                    status = "failed"
                    logger.exception("Probe {} failed; continuing", step.step_id)
                    if retryable and attempt == last_attempt:
                        # The batch still goes on: a probe that failed all
                        # three attempts is the report's problem, not the
                        # runner's.
                        self.progress(
                            f"probe void id={step.step_id} kind={kind} "
                            f"attempt={attempt} failed all attempts"
                        )
                    self.progress(
                        f"[{index:03d}/{total:03d}] {step.step_id} failed "
                        f"{kind}: {exc}"
                    )
                    if isinstance(exc, JudgeUnavailable):
                        # The probe is recorded like any other failure —
                        # the branch is already cleaned and its evidence copied
                        # out by `_run_probe_in_branch` — and then the batch
                        # stops, because a judge that is gone is gone for the
                        # next probe too. Never a void: nothing is retried.
                        logger.error(
                            "The judge is gone; the probe batch stops once the "
                            "probes in flight have finished"
                        )
                        raise
                    return
                if launch_attempts is not None:
                    launch_attempts.succeed(attempt)
                self.ledger.mark_done(
                    step.step_id,
                    artifacts=execution.artifacts,
                    summary={
                        "beats": execution.beats,
                        "tool_calls": execution.tool_calls,
                        "elapsed_s": execution.elapsed_s,
                        # Only on a probe that was voided and rerun, so every
                        # other step's summary keeps the shape it has always
                        # had. The counting rules `SUMMARY_SCHEMA` names are
                        # unchanged: this is a new key, not a new meaning.
                        **(
                            {"void_retries": attempt - 1}
                            if attempt > 1
                            else {}
                        ),
                    },
                    summary_schema=SUMMARY_SCHEMA,
                )
                result.executed.append(step.step_id)
                status = "done"
                self.progress(
                    format_step_done(
                        index,
                        total,
                        step,
                        beats=execution.beats,
                        tool_calls=execution.tool_calls,
                        elapsed_s=execution.elapsed_s,
                    )
                )
                return
        finally:
            self.in_flight.discard(step.step_id)
            if status is not None:
                # Only the two recorded outcomes close a slot. A probe the
                # branch machinery could not start is left `running` for the
                # resume, and saying `failed` about it would contradict the
                # ledger.
                self.progress(
                    f"probe slot close id={step.step_id} status={status}"
                )

    def _close_step_ix_session(self, step: RunStep) -> list[tuple[str, str]]:
        """Close this step's IX session; return the episodes still open.

        Never raises: it runs in the step's `finally`, where an exception would
        replace the step's real error with this one. The caller decides what to
        do with what it found.
        """
        closer = getattr(self.pa_checkpoint, "close_ix_session", None)
        if closer is None:
            return []
        return list(closer(step.step_id) or [])

    # ── probe branches ────────────────────────────────────────────

    async def _run_probe_in_branch(
        self, step: RunStep, *, checkpoint_manager: CheckpointManager, pinned: set[str]
    ) -> Any:
        owner = self.config.shared_probe_memory_owner
        failed = True
        try:
            result = await self._run_probe_in_branch_owned(
                step, checkpoint_manager=checkpoint_manager, pinned=pinned
            )
            failed = False
            return result
        finally:
            if owner is not None:
                root = self.config.run_dir / "branches" / step.step_id
                if root.exists():
                    for branch in root.iterdir():
                        if branch.is_dir():
                            owner.release_consumer(branch, failed=failed)

    async def _run_probe_in_branch_owned(
        self,
        step: RunStep,
        *,
        checkpoint_manager: CheckpointManager,
        pinned: set[str],
    ) -> Any:
        """Run one probe on a throwaway copy of a pinned cut point.

        The probe's mutable world, session and memory audit live inside the
        branch directory. External memory belongs to this branch or to the
        probe-variance owner's immutable checkpoint copy. Only
        the evidence listed in `_collect_probe_evidence` comes back, and the
        branch is then discarded.
        """
        source_manager, source_checkpoint_id = self._probe_source_checkpoint(
            step,
            checkpoint_manager,
            pinned,
        )
        branch_dir = self._next_branch_dir(step)
        # Off the event loop: `branch_from` is three `copytree`s and, on an
        # external backend, a blocking restore call. Left inline it would hold
        # the loop for its whole length, and the other probes' in-flight HTTP
        # requests — whose timeouts are wall clock — would be reading nothing
        # while it ran.
        owner = self.config.shared_probe_memory_owner
        shared_adapter = None
        if owner is None:
            await asyncio.to_thread(source_manager.branch_from, source_checkpoint_id, branch_dir)
        else:
            try:
                shared_adapter = await owner.branch(source_manager, source_checkpoint_id, branch_dir)
            except Exception as exc:
                raise ProbeRunConfigurationError(f"Shared memory preparation failed: {exc}") from exc

        branch_namespace = (read_branch_namespace(branch_dir) or {}).get(
            "namespace"
        )

        memory_adapter: Any | None = None
        if self.plan.memory_condition == NO_MEMORY_CONDITION:
            # The memory-removed baseline lives in the system's own world with
            # its memory taken away: clearing the local files is not enough,
            # the adapter itself has to be replaced or an external backend
            # would keep answering from the remote store.
            #
            # The key is this run's condition, never `borrow_start_from`. A
            # probe plan attached later to a finished accumulation run borrows
            # its start point under the *memory* condition that wrote it, and
            # handing it `NoMemoryAdapter` would silently turn every such probe
            # into the baseline.
            self._clear_branch_pa_memory(branch_dir)
            memory_adapter = NoMemoryAdapter()
        elif self.plan.memory_condition == ORACLE_PREFERENCE_CONDITION:
            # The same memory-removed branch, whatever condition wrote the
            # borrowed start point; its one retrieved entry is this probe's
            # frozen ground truth. The accumulation dialogue is already out of
            # the prompt: `branch_from` moved `prompt_history_start_index` to
            # the end of the session (`_move_branch_prompt_start_to_end`).
            self._clear_branch_pa_memory(branch_dir)
            memory_adapter = OraclePreferenceAdapter(
                step.target_cell, step.target_setting
            )
        else:
            # An external backend keeps its memory outside the
            # branch directory, so the directory copy leaves the probe with
            # nothing to read. `branch_from` has just restored the snapshot
            # into a namespace of this branch's own; the adapter that reads it
            # is built here, before the probe PA exists, and set read-only
            # through the same path a mainline step uses, so the probe
            # retrieves the cut point's memory and writes none of its own.
            memory_adapter = shared_adapter if owner is not None else source_manager.build_branch_adapter(branch_dir)
            if inspect.isawaitable(memory_adapter):
                memory_adapter = await memory_adapter
            # The snapshot's own condition, not the plan's: a borrowed start
            # point is branched under the condition that wrote it.
            branch_condition = (
                getattr(source_manager, "memory_condition", None)
                or self.plan.memory_condition
            )
            if memory_adapter is None and (
                branch_condition
                and branch_condition not in LOCAL_MEMORY_CONDITIONS
            ):
                # Silence here would be a defect: a probe with no
                # adapter retrieves nothing and reads exactly like the
                # no_memory baseline, and every memory-vs-no-memory comparison
                # on this condition would be null without saying so.
                raise ProbeRunConfigurationError(
                    "A probe branch on an external memory backend was built "
                    "with no adapter, so it would retrieve nothing: "
                    f"memory_condition={branch_condition!r} "
                    f"branch_dir={str(branch_dir)!r} step_id={step.step_id!r}"
                )
            if memory_adapter is not None:
                await MemoryRuntime(
                    memory_adapter,
                    run_id=self.memory_runtime.scope.run_id,
                    persona_id=self.memory_runtime.scope.persona_id,
                    agent_id=self.memory_runtime.scope.agent_id or "nanobot",
                    backend_namespace=branch_namespace,
                ).apply_mode("read_only", reason=step.step_id)

        branch_workspace = self._branch_workspace(branch_dir)
        branch_audit_path = branch_dir / "memory_io.jsonl"
        # Created up front so "this probe wrote nothing" is evidence rather
        # than a missing file.
        branch_audit_path.touch()
        branch_recorder = MemoryIORecorder(
            branch_audit_path,
            run_id=self.plan.run_id,
        )
        branch_stage = StageRuntime(
            run_id=self.plan.run_id,
            user_id=step.persona_id,
            fixture_root=self.stage_runtime.fixture_root,
            project_root=self.config.project_root,
            run_dir=branch_dir,
        )
        step_stage = branch_stage.begin_step(
            step_id=step.step_id,
            session_id=step.step_id,
            runtime_artifact_path=self.config.project_root / step.script_path,
            session_date=step.benchmark_session_timestamp.date().isoformat(),
            session_timestamp=step.benchmark_session_timestamp.isoformat(),
            fixture_overlay=self._probe_fixture_overlay(step),
            # Probe steps only: the branch world is thrown away afterwards.
            merge_records=True,
            gate=resolve_session_gate(
                self.config.project_root / step.script_path,
                self.config.project_root,
            ),
            compressed_reaction_index=resolve_compressed_reaction_index(
                self.config.project_root / step.script_path
            ),
        )
        branch_steps_dir = branch_dir / "steps"
        # The scene's starting facts are captured after the overlay and before
        # the first PA call, because the branch is gone by scoring time.
        self._capture_scene_start(step, step_stage, branch_steps_dir)

        extra_meta: dict[str, Any] = {
            "run_id": self.plan.run_id,
            "memory_condition": self.plan.memory_condition,
            "script_session_id": step.step_id,
            "stage_root": str(step_stage.state_root),
            "simulator_runtime": (
                self.plan.simulator_runtime.model_dump(mode="json")
            ),
            "pa_session_key": self.plan.pa_session_key_for_step(step),
            "pa_workspace": str(branch_workspace),
            "branch_dir": str(branch_dir),
            "source_checkpoint_id": source_checkpoint_id,
        }
        if self.plan.memory_condition in (
            NO_MEMORY_CONDITION, ORACLE_PREFERENCE_CONDITION
        ):
            # The branch receipt: history messages the first PA call carries.
            extra_meta["prompt_history_messages_at_start"] = (
                _prompt_history_at_start(branch_dir)
            )
        if step_stage.gate:
            extra_meta["gate"] = dict(step_stage.gate)
        if step.borrow_start_from is not None:
            extra_meta["borrow_start_from"] = step.borrow_start_from
        # This probe's own overlay, as of its own cut point — never the
        # run-level object, which carries every flip the run has made by the
        # time the probe batch runs (the timeline finishes first).
        # The mainline object is left untouched.
        probe_overlay = probe_preference_overlay(
            self.plan,
            self.config.probe_source_ledger or self.ledger,
            step,
        )
        # The branch's own record of the simulator identity it ran under, so a
        # probe transcript can be read without reconstructing the closure the
        # loop factory was built with.
        extra_meta["preference_overlay"] = probe_overlay.preference_overrides()
        branch_sim_profile = self._branch_sim_profile(branch_dir)
        if branch_sim_profile is not None:
            extra_meta["sim_profile_root"] = str(branch_sim_profile.root)
        pa_handle = ProbePAHandle(
            session_key=self.plan.pa_session_key_for_step(step),
            script_session_id=step.step_id,
            forced_ix=self.config.forced_ix.get(step.step_id) or {},
        )
        probe_succeeded = False
        attempt_record = None
        attempt_started = False
        try:
            if owner is not None:
                from harness.probe_attempts import ProbeAttempts
                attempt_record = ProbeAttempts(self.config.run_dir, step.step_id)
                attempt_record.begin()
                attempt_started = True
            execution = await self.executor.execute(
                step,
                steps_dir=branch_steps_dir,
                stage_runtime=step_stage,
                extra_meta=extra_meta,
                loop_context={
                    "pa_binder": pa_handle.bind,
                    "preference_overlay": probe_overlay,
                    **(
                        {"sim_profile": branch_sim_profile}
                        if branch_sim_profile is not None
                        else {}
                    ),
                    "workspace": branch_workspace,
                    "session_manager": SessionManager(branch_workspace),
                    "memory_adapter": memory_adapter,
                    "memory_backend_namespace": branch_namespace,
                    "memory_io_recorder": branch_recorder,
                    # The branch skips compaction end to end, not just
                    # the LM write inside it.
                    "skip_consolidation": True,
                },
            )
            branch_stage.finish_step(step_stage, step.stage_policy)
            self._record_probe_open_holds(step, pa_handle, branch_steps_dir)
            artifacts = self._collect_probe_evidence(step, branch_dir)
            if attempt_record is not None:
                attempt_record.finish()
                attempt_started = False
            probe_succeeded = True
        except BaseException as probe_error:
            if attempt_started and not isinstance(probe_error, asyncio.CancelledError):
                attempt_record.finish(error=probe_error)
            # The probe that failed is the one that most needs an audit
            # trail. `session_executor` already wrote the transcript, the
            # `pa_toolcalls.json` and a `meta.yaml` carrying `failure_reason`
            # into the branch, but the `finally` below removes the branch — so
            # whatever exists is copied out first. Best effort by construction:
            # a probe that died before writing anything leaves nothing to copy,
            # and this must never replace the probe's own error.
            try:
                self._record_probe_open_holds(step, pa_handle, branch_steps_dir)
                # Overwrite the branch paths the executor attached: the branch
                # is about to be deleted, so only the copies survive.
                setattr(
                    probe_error,
                    FAILED_STEP_ARTIFACTS_ATTR,
                    self._collect_probe_evidence(step, branch_dir),
                )
            except Exception as copy_error:  # pragma: no cover - best effort
                logger.warning(
                    "Could not preserve the failed probe's evidence for {}: {}",
                    step.step_id,
                    copy_error,
                )
            raise
        finally:
            pa_handle.release()
            if owner is not None:
                owner.release_consumer(branch_dir, failed=not probe_succeeded)
            self._discard_branch(branch_dir, source_manager)
        return replace(execution, artifacts=artifacts)

    def _record_probe_open_holds(
        self,
        step: RunStep,
        pa_handle: ProbePAHandle,
        branch_steps_dir: Path,
    ) -> None:
        """Close the probe's IX session; an open hold goes into `meta.yaml`.

        The mainline fails a step that leaves a hold open (contract 7); a
        probe is a job whose only output is evidence, so the finding is
        recorded beside the transcript and the probe is still done.
        Runs before the evidence is copied out, on success and on failure.
        """
        import yaml

        open_episodes = pa_handle.close_ix_session()
        if not open_episodes:
            return
        meta_path = branch_steps_dir / step.step_id / "meta.yaml"
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            meta = {}
        meta[OPEN_IX_EPISODES_KEY] = [
            {"attribute": attribute, "episode_id": episode_id}
            for attribute, episode_id in open_episodes
        ]
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.warning(
            "Probe {} ended with open IX holds, recorded in meta.yaml: {}",
            step.step_id,
            open_episodes,
        )

    def _probe_source_checkpoint(
        self,
        step: RunStep,
        checkpoint_manager: CheckpointManager,
        pinned: set[str],
    ) -> tuple[CheckpointManager, str]:
        if step.borrow_start_from is not None:
            source_run_id, checkpoint_id = step.borrowed_checkpoint()
            runs_root = (
                self.config.borrowed_runs_root
                or self.config.run_dir.parent
            )
            source_dir = Path(runs_root) / source_run_id
            # The snapshot is read back under the condition that wrote it, not
            # under this run's own — except for the memory-removed baseline.
            # That arm binds `NoMemoryAdapter` below and reads none of the
            # source's external store, so restoring the source's collection
            # into a branch namespace would be pure waste; worse, the
            # entrypoint leaves `backend_client` unset for `no_memory`, so
            # `branch_from` would demand a client this run cannot have and the
            # baseline arm of an external-backend cell could not run at all.
            # The condition here is the whole restore decision:
            # `_restore_backend_state_into_branch` short-circuits on the
            # local conditions, and `no_memory` is one of them. So is
            # `oracle_preference`, which binds its own adapter the same way.
            branch_condition = (
                self.plan.memory_condition
                if self.plan.memory_condition
                in (NO_MEMORY_CONDITION, ORACLE_PREFERENCE_CONDITION)
                else _snapshot_memory_condition(
                    source_dir / "checkpoints" / checkpoint_id,
                    default=self.plan.memory_condition,
                )
            )
            source_manager = CheckpointManager(
                source_dir,
                ledger=self.ledger,
                memory_condition=branch_condition,
                rolling_window=self.plan.checkpoint_rolling_window,
                # A borrowed snapshot on an external backend still has to be
                # restored into a namespace of the branch's own and read
                # through an adapter built on it, exactly as a same-run cut
                # point is. Both come from this run's entrypoint.
                backend_client=checkpoint_manager.backend_client,
                branch_adapter_factory=checkpoint_manager.branch_adapter_factory,
            )
            source_manager.shared_query_factory = checkpoint_manager.shared_query_factory
            source_manager.shared_query_identity = checkpoint_manager.shared_query_identity
            return source_manager, checkpoint_id
        latest = self.plan.probe_source_step_id(step, pinned)
        if latest is None:
            # Not this probe failing: no probe in the batch has a pin in front
            # of it either, so recording it per probe would report 33 failures
            # for one missing timeline run.
            raise ProbeRunConfigurationError(
                "A probe has no pinned checkpoint to branch from and does not "
                f"borrow one: step_id={step.step_id!r}"
            )
        return checkpoint_manager, latest

    def _next_branch_dir(self, step: RunStep) -> Path:
        """One directory per attempt, so a retry never reuses a dirty branch."""
        root = self.config.run_dir / "branches" / step.step_id
        root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while True:
            # Claimed with `mkdir`, not chosen with `exists()`: the attempt
            # directory is reserved by the call that picks it, so two probes
            # cannot pick the same number even if they ever share a `root`.
            branch_dir = root / f"{attempt:02d}"
            try:
                branch_dir.mkdir()
            except FileExistsError:
                attempt += 1
                continue
            return branch_dir

    def _branch_workspace(self, branch_dir: Path) -> Path:
        workspace = branch_dir / "pa_workspace" / self.plan.persona_id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _branch_sim_profile(self, branch_dir: Path) -> Any:
        """The branch's own copy of the run's simulator profile.

        Closing a session writes the simulated user's `MEMORY.md` and
        `HISTORY.md` under the profile root (`simulator/memory/store.py`), so a
        probe handed the run-level object writes into the run directory and the
        next probe reads what the previous one wrote. `branch_from` has already
        copied `simulator_workspace/` into the branch
        (`harness/checkpoint.py`), so the profile is loaded from that copy —
        the same loader the run-level profile is built with, rooted at the
        branch — and the writes die with the branch.
        """
        run_profile = self.config.sim_profile
        if run_profile is None:
            return None
        return load_simulator_profile(
            branch_dir
            / SIMULATOR_WORKSPACE_LAYER
            / Path(run_profile.root).name
        )

    def _probe_fixture_overlay(self, step: RunStep) -> Path | None:
        """The probe's effective `fixture/` overlay, resolved from the drafts now.

        This is called immediately before ``StageRuntime.begin_step``. The path
        it returns is the path StageRuntime consumes.
        """
        return resolve_probe_overlay(
            self.plan,
            step,
            project_root=self.config.project_root,
        )

    def _clear_branch_pa_memory(self, branch_dir: Path) -> None:
        for memory_dir in sorted(
            (branch_dir / "pa_workspace").glob("*/memory")
        ):
            shutil.rmtree(memory_dir)
            memory_dir.mkdir(parents=True)

    def _capture_scene_start(
        self,
        step: RunStep,
        step_stage: Any,
        branch_steps_dir: Path,
    ) -> None:
        """Copy the world this probe starts from, plus an index of it.

        The branch is discarded once the evidence is out, so this copy is the
        only record of what the probe faced. Deriving `feasible_options` and
        `pending_items` from it is the packer's job.
        """
        step_dir = branch_steps_dir / step.step_id
        target = step_dir / SCENE_START_DIRNAME
        target.mkdir(parents=True, exist_ok=True)
        world = target / "world"
        if world.exists():
            shutil.rmtree(world)
        shutil.copytree(step_stage.state_root, world)
        entries = []
        for path in sorted(world.rglob("*")):
            if not path.is_file():
                continue
            entries.append(
                {
                    "path": str(path.relative_to(world)),
                    "bytes": path.stat().st_size,
                }
            )
        (step_dir / SCENE_START_INDEX).write_text(
            json.dumps(
                {
                    "step_id": step.step_id,
                    "kind": step.kind,
                    "target_cell": step.target_cell,
                    "world_root": SCENE_START_DIRNAME + "/world",
                    "world_state": entries,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _collect_probe_evidence(
        self,
        step: RunStep,
        branch_dir: Path,
    ) -> dict[str, str]:
        """Hand back the probe's artifacts; the branch's state stays behind.

        The branch's `memory_io.jsonl` comes across as this step's own audit
        file. It is deliberately *not* merged into the mainline audit: the
        mainline record must keep matching the mainline database one for one.
        """
        source = branch_dir / "steps" / step.step_id
        target = self.config.run_dir / "steps" / step.step_id
        target.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        branch_audit = branch_dir / "memory_io.jsonl"
        if branch_audit.is_file():
            shutil.copy2(branch_audit, target / "memory_io.jsonl")
        artifacts: dict[str, str] = {}
        for name, relative in (
            ("transcript_jsonl", "transcript.jsonl"),
            ("transcript_md", "transcript.md"),
            ("pa_toolcalls", "pa_toolcalls.json"),
            ("text_judge_pack", "text_judge_pack.json"),
            ("meta", "meta.yaml"),
            ("tool_log", "state/tool_log.jsonl"),
            ("state_diff", "state/state_diff.jsonl"),
            ("memory_io", "memory_io.jsonl"),
            ("scene_start", SCENE_START_INDEX),
        ):
            path = target / relative
            if path.exists():
                artifacts[name] = str(path)
        return artifacts

    def _discard_branch(
        self,
        branch_dir: Path,
        source_manager: CheckpointManager | None = None,
    ) -> None:
        self._release_branch_namespaces(branch_dir, source_manager)
        shutil.rmtree(branch_dir, ignore_errors=True)

    def _release_branch_namespaces(
        self,
        branch_dir: Path,
        source_manager: CheckpointManager | None = None,
    ) -> Exception | None:
        """Drop the remote namespace this branch restored for itself.

        Local conditions keep everything inside `branch_dir`, so removing the
        directory is the whole cleanup. External backends would otherwise leave
        33 collections or cubes per run behind, so the branch's own namespace
        is deleted here.

        Returns the error that stopped the release, or None when it succeeded
        or there was nothing to release. `_discard_branch` ignores it — that
        probe already has its evidence and its cleanup is best effort — but the
        orphan reaper keeps the directory on an error, because the directory is
        the only thing that still names the namespace.
        """
        if read_branch_namespace(branch_dir) is None:
            return None
        if source_manager is None:
            return RuntimeError(
                "no checkpoint manager to release the branch namespace through"
            )
        try:
            source_manager.release_branch_namespace(branch_dir)
        except Exception as error:
            logger.warning(
                "Could not release the remote namespace for branch {}: {}",
                branch_dir,
                error,
            )
            return error
        return None

    async def _await_checkpoint_ready(self) -> None:
        """Wait only — no consolidation, no cursor moves, no backend split.

        A snapshot must not stop halfway through a write, so background
        archives are drained and the adapter is asked to settle. Turns that
        were never compacted stay exactly as they are; putting them into the
        memory files is the pinned-only flush, and it happens after this.
        """
        session = None
        if self.pa_checkpoint is not None:
            await self.pa_checkpoint.drain_background_tasks()
            session = self.pa_checkpoint.checkpoint_session()
        await self.memory_runtime.await_checkpoint_ready(
            CHECKPOINT_READY_TIMEOUT_S,
            session=session,
        )

    def _begin_memory_side_usage(self, session_id: str) -> None:
        """Open the session's memory-side token line, where there is one.

        Only memos has one: its extraction LLM runs in a container and cannot
        print an `openrouter_call` line of its own, so the adapter tails the
        container's log instead (`memory/backends/memos_usage_log.py`). The
        step id has to come from here — a `MemoryEvent` carries the PA session
        key, which is the whole run, not this session.
        """
        begin = getattr(
            self.memory_runtime.adapter, "begin_memory_side_usage", None
        )
        if begin is not None:
            begin(session_id)

    def _flush_memory_side_usage(self) -> None:
        """Close it: drain the last calls, then print the session's totals."""
        flush = getattr(
            self.memory_runtime.adapter, "flush_memory_side_usage", None
        )
        if flush is not None:
            flush()

    async def _replay_pending_turns(self) -> None:
        """Rewrite the pools of turns that died before their adapter write.

        Run-start only, and once: a `pending` journal record is a turn whose
        pool never reached the backend, and no message index can tell one from
        a turn that never happened — a suppressed outreach's span is `[n, n)`.
        The replay logs every turn id it rewrites, which is the
        at-least-once declaration.
        """
        from memory.turn_journal import replay_pending_turns

        handle = self.pa_checkpoint
        adapter = self.memory_runtime.adapter
        if handle is None or adapter is None:
            return
        try:
            session = handle.checkpoint_session()
        except SessionNotFoundError:
            # A fresh run has no session file yet, and `SessionManager.get`
            # raises rather than inventing one. No session means no turn, so
            # there is nothing to replay and the journal beside it does not
            # exist either. Reading it unguarded killed every fresh run at
            # start-up (live run #3, zero API calls).
            return
        journal = handle.turn_journal()
        # `failure_policy.partial_skip_pending_memory: drop`: a skipped step's
        # unwritten pools stay out of memory; the skip listed their turn ids.
        dropped = {
            turn_id
            for record in self.ledger.steps.values()
            if record.status == "skipped"
            for turn_id in (record.summary or {}).get(DROPPED_PENDING_TURNS_KEY, [])
        }
        pending_records = (
            [r for r in journal.pending_records() if r.turn_id not in dropped]
            if dropped
            else None
        )
        rewritten = await replay_pending_turns(
            journal=journal,
            adapter=adapter,
            scope=self.memory_runtime.scope,
            session=session,
            # Fail closed, and a no-op where it cannot apply: both
            # `write_visible_memory_event` and `record_adapter_write_commit`
            # enforce this only under `after_adapter_commit`, the one policy
            # whose cursor a refused write would otherwise advance past.
            require_committed=True,
            records=pending_records,
        )
        if rewritten:
            handle.save_session(session)
            self.progress(
                f"resume rewrote {len(rewritten)} pool(s): {', '.join(rewritten)}"
            )

    async def _flush_uncompacted_turns(self) -> None:
        """Pinned-only: force one consolidation so the snapshot is complete.

        Pinned checkpoints are the only copies a probe branches from, so every
        turn must already be in the memory files. Non-file backends go through
        the same eviction-policy gate and do nothing here, cursors included.
        """
        handle = self.pa_checkpoint
        if handle is None:
            return
        if (
            handle.prompt_history_eviction_policy
            != PromptHistoryEvictionPolicy.after_file_consolidation
        ):
            return
        session = handle.checkpoint_session()
        # Every record the journal still owes the memory files, zero-span ones
        # included -- so this runs even when `session.messages` has no tail
        # left, which is exactly the shape a suppressed outreach leaves behind.
        # The old message-slice version archived `[]` there and the
        # probe branched from a memory that had never heard of the turn.
        journal = handle.turn_journal()
        owed = records_owed_to_files(journal.records())
        records = consolidation_records(owed)
        if records:
            await handle.archive_messages_to_files(records)
        journal.mark_file_consolidated([record.turn_id for record in owed])
        end_index = len(session.messages)
        session.file_consolidated_end_index = end_index
        session.prompt_history_start_index = end_index
        handle.save_session(session)

    def _record_memory_health(
        self,
        backend_request: dict[str, Any],
        *,
        status: str,
        health: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        if self.memory_io_recorder is None or self.memory_runtime.adapter is None:
            return
        self.memory_io_recorder.record(
            {
                "run_id": self.plan.run_id,
                "step_id": "",
                "session_id": self.plan.main_pa_session_key(),
                "turn_id": "",
                "backend": self.memory_runtime.adapter.backend_name,
                "operation": "health",
                "adapter_request": backend_request,
                "backend_request": backend_request,
                "backend_response": (
                    {
                        "status": health.status.value,
                        "latency_ms": health.latency_ms,
                        "warnings": health.warnings,
                    }
                    if health is not None
                    else None
                ),
                "status": status,
                "native_ids": [],
                "native_events": [],
                "result_count": None,
                "top_score": None,
                "latency_ms": health.latency_ms if health is not None else None,
                "audit_only_labels": {},
                "error": (
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    if error is not None
                    else None
                ),
            }
        )


def load_frozen_orchestrator_plan(
    plan_path: Path | str,
    *,
    project_root: Path | str = ".",
    validate_paths: bool = True,
) -> RunPlan:
    return load_run_plan(plan_path, project_root=project_root, validate_paths=validate_paths)
