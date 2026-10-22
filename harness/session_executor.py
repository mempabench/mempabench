"""Thin one-step executor around `SimulatorLoop.run_session()`."""

from __future__ import annotations

import time
from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Protocol

from harness.gate import gate_marker_hook, session_closing_hook
from harness.judge_pack import build_text_judge_pack
from harness.pa_factory import ix_render_records_from_loop
from harness.run_plan import RunStep
from harness.state_runtime import StageStepRuntime, preserve_step_state_artifacts
from harness.transcript import (
    TranscriptWriter,
    declared_plan_arguments,
    declared_plan_tools,
    load_events,
    merge_ix_render_into_transcript,
    read_ix_render,
    task_tool_events,
    tool_outside_plan_events,
    verification_override_events,
    write_markdown,
    write_meta,
    write_pa_toolcalls,
)
from pa.agent.ix.verification import VERIFICATION_OVERRIDE


class SessionLoop(Protocol):
    async def run_session(self) -> list[dict]: ...


# The trailing keyword arguments are the per-step run context a probe branch
# injects (workspace, session manager, memory adapter, namespace, recorder);
# a mainline step passes none and the factory falls back to its own values.
LoopFactory = Callable[..., SessionLoop]


@dataclass(frozen=True)
class SessionExecutionResult:
    artifacts: dict[str, str]
    beats: int
    tool_calls: int
    elapsed_s: float


#: A step that fails still has its artifacts written (see `execute`), but the
#: result object carrying their paths is lost with the raised exception. The
#: paths ride on the exception instead, so the ledger can record them.
FAILED_STEP_ARTIFACTS_ATTR = "mempa_session_artifacts"


def failed_step_artifacts(error: BaseException) -> dict[str, str]:
    """Paths of the artifacts written before `error` ended a step; {} if none."""
    return dict(getattr(error, FAILED_STEP_ARTIFACTS_ATTR, None) or {})


class SessionExecutor:
    """Execute one `RunStep` and write session artifacts."""

    def __init__(self, loop_factory: LoopFactory) -> None:
        self.loop_factory = loop_factory

    async def execute(
        self,
        step: RunStep,
        *,
        steps_dir: Path | str,
        stage_runtime: Any | None = None,
        extra_meta: dict[str, Any] | None = None,
        loop_context: dict[str, Any] | None = None,
    ) -> SessionExecutionResult:
        step_dir = Path(steps_dir) / step.step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        state_artifacts: dict[str, str] = {}
        loop: Any = None
        events: list[dict] = []
        failure: BaseException | None = None
        result: SessionExecutionResult | None = None
        try:
            with TranscriptWriter(
                step_dir, event_hook=gate_marker_hook(stage_runtime)
            ) as writer:
                # The other state-root marker, and never on the same step as
                # `gate_open`: a compressed session is exactly the one with no
                # gate to open. It joins here rather than at construction only
                # because the writer takes one hook.
                closing_hook = session_closing_hook(stage_runtime)
                if closing_hook is not None:
                    writer.add_event_hook(closing_hook)
                loop = self.loop_factory(
                    step,
                    writer,
                    stage_runtime,
                    **(loop_context or {}),
                )
                events = await loop.run_session()
        except BaseException as error:  # noqa: BLE001 — re-raised below
            failure = error
            raise
        finally:
            if isinstance(stage_runtime, StageStepRuntime):
                state_artifacts = preserve_step_state_artifacts(
                    stage_runtime,
                    step_dir,
                )
            # A failed step is the one that most needs an audit trail: the turns
            # that ran before the failure rendered fine, and their `ix_render`
            # lives only inside the PA until the merge below puts it back. So
            # the artifacts are written either way — best effort when the step
            # failed, and never allowed to replace the step's own error.
            try:
                result = self._write_artifacts(
                    step,
                    step_dir,
                    loop=loop,
                    events=events,
                    extra_meta=extra_meta,
                    state_artifacts=state_artifacts,
                    start=start,
                    failure=failure,
                )
            except Exception:
                if failure is None:
                    raise
            if failure is not None and result is not None:
                setattr(failure, FAILED_STEP_ARTIFACTS_ATTR, result.artifacts)
            failed_turn_events = getattr(failure, "failed_turn_events", None)
            if failed_turn_events:
                # The turn that failed the IX contract has no transcript entry;
                # its tool events are the only trace of the ladder.
                try:
                    (Path(step_dir) / "failed_turn_events.json").write_text(
                        json.dumps(failed_turn_events, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception:  # noqa: BLE001 — never mask the step's own error
                    pass
        if result is None:  # pragma: no cover — the failure path re-raises first
            raise RuntimeError(f"no session artifacts were written for {step.step_id}")
        return result

    def _write_artifacts(
        self,
        step: RunStep,
        step_dir: Path,
        *,
        loop: Any,
        events: list[dict],
        extra_meta: dict[str, Any] | None,
        state_artifacts: dict[str, str],
        start: float,
        failure: BaseException | None,
    ) -> SessionExecutionResult:
        transcript = step_dir / "transcript.jsonl"
        if failure is not None and transcript.exists():
            # `run_session` never returned its event list, so read back what the
            # writer already put on disk.
            events = load_events(transcript)
        # The simulator emits `pa_turn` without the PA's metadata, so the
        # wrapper's render block reaches no artifact on its own. Put it back
        # before anything reads the transcript (§4.4).
        merge_report = merge_ix_render_into_transcript(
            transcript,
            ix_render_records_from_loop(loop),
        )
        if merge_report["merged"]:
            events = load_events(transcript)
        # Only here do the draft and the executed calls sit side by side: the
        # simulator never reads `atomic_steps`, and the PA never sees the plan
        # at all. Diagnostic only — the call already ran and nothing refuses
        # it; the record is so a batch review can find the turn.
        outside_plan = tool_outside_plan_events(
            events,
            declared_plan_tools(step.script_path),
            declared_plan_arguments(step.script_path),
        )
        if outside_plan:
            with TranscriptWriter(step_dir) as writer:
                for record in outside_plan:
                    writer.log_event(record)
            events.extend(outside_plan)
        # Owner-approved verification aid (2026-09-08): a step the wrapper
        # steered under `MEMPA_VERIFY_OUTREACH` stamps its own transcript and,
        # below, its own `meta.yaml`, so neither artifact can be picked up
        # later and read as a scored session.
        overrides = verification_override_events(events)
        if overrides:
            with TranscriptWriter(step_dir) as writer:
                for record in overrides:
                    writer.log_event(record)
            events.extend(overrides)
        validation_attempts = step_dir / "simulator_validation_attempts.jsonl"
        step_meta = {
            "step_id": step.step_id,
            "kind": step.kind,
            "persona_id": step.persona_id,
            "context": step.context,
            "target_cell": step.target_cell,
            "target_attribute": step.target_attribute,
            "target_setting": step.target_setting,
            "acc_num": step.acc_num,
            "before_acc_num": step.before_acc_num,
            "session_type": step.session_type,
            "domain": step.domain,
            "story_id": step.story_id,
            "timeline_slot": step.timeline_slot,
            "time_position": step.time_position,
            "benchmark_session_timestamp": (
                step.benchmark_session_timestamp.isoformat()
            ),
            "memory_mode": step.memory_mode,
            "stage_policy": step.stage_policy,
            **(extra_meta or {}),
        }
        if failure is not None:
            # The ledger records the same error; this is the copy that sits next
            # to the transcript, so a step directory explains itself.
            step_meta["failure_reason"] = f"{type(failure).__name__}: {failure}"
        invalid = next(
            (event for event in events if event.get("kind") == "session_invalid"),
            None,
        )
        if invalid is not None:
            # The simulator invalidated this session (owner ruling 2026-09-08).
            # Read off the transcript rather than the exception, so `harness`
            # stays free of `simulator` imports and a re-read of the directory
            # says the same thing the live run did.
            step_meta["session_invalid"] = {
                "rule": invalid.get("rule"),
                "beat": invalid.get("beat"),
                "attempts": invalid.get("attempts"),
            }
        if overrides:
            step_meta[VERIFICATION_OVERRIDE] = {
                "outreach_action": overrides[0].get("outreach_action"),
                "source": overrides[0].get("source"),
            }
        start_event = next(
            (event for event in events if event.get("kind") == "session_start"),
            {},
        )
        for field in ("pa_model", "simulator_model"):
            if start_event.get(field):
                step_meta[field] = start_event[field]
        try:
            step_meta.update(_ix_render_versions(events))
        except Exception as error:
            if failure is None:
                raise
            # A step that died mid-render can have left a malformed blob; that
            # is a finding to record, not a reason to lose the whole directory.
            step_meta["ix_render_version_error"] = f"{type(error).__name__}: {error}"
        if merge_report["records"] != merge_report["pa_turns"]:
            # Positional alignment is the whole contract here, so a count
            # mismatch is recorded rather than left to look like a PA that
            # simply never rendered.
            step_meta["ix_render_merge"] = merge_report
        if validation_attempts.exists():
            step_meta["simulator_validation_attempts"] = (
                "simulator_validation_attempts.jsonl"
            )
        # The text face of this session, written once beside the transcript
        # rather than rebuilt by every reader. `render_session_html` used to be
        # the only thing that ever built it, in-process, so nothing on disk
        # said what the judge would actually have seen.
        judge_pack_path = _best_effort(
            failure,
            lambda: _write_text_judge_pack(step_dir, events, step_meta),
        )
        md_path = _best_effort(failure, lambda: write_markdown(transcript))
        toolcalls_path = _best_effort(
            failure, lambda: write_pa_toolcalls(transcript, extra_meta=step_meta)
        )
        beats = sum(1 for event in events if event.get("kind") == "beat_enter")
        tool_calls = _count_tool_calls(events)
        meta_path = write_meta(step_dir, step_meta)
        elapsed = time.monotonic() - start
        artifacts = {
            "transcript_jsonl": str(transcript),
            **({"transcript_md": str(md_path)} if md_path else {}),
            **({"pa_toolcalls": str(toolcalls_path)} if toolcalls_path else {}),
            **(
                {"text_judge_pack": str(judge_pack_path)}
                if judge_pack_path
                else {}
            ),
            "meta": str(meta_path),
            **state_artifacts,
        }
        if validation_attempts.exists():
            artifacts["simulator_validation_attempts"] = str(validation_attempts)
        return SessionExecutionResult(
            artifacts=artifacts,
            beats=beats,
            tool_calls=tool_calls,
            elapsed_s=elapsed,
        )


def _write_text_judge_pack(
    step_dir: Path,
    events: list[dict],
    step_meta: Mapping[str, Any],
) -> Path:
    """Write `text_judge_pack.json` beside the transcript it packs."""
    path = Path(step_dir) / "text_judge_pack.json"
    path.write_text(
        json.dumps(
            build_text_judge_pack(events, meta=step_meta),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _best_effort(failure: BaseException | None, action: Callable[[], Any]) -> Any:
    """Run `action`, tolerating its failure only on a step that already failed.

    One unreadable artifact must not cost the rest of the step directory; on a
    step that succeeded there is nothing to tolerate, so the error propagates.
    """
    try:
        return action()
    except Exception:
        if failure is None:
            raise
        return None


def _ix_render_versions(events: list[dict]) -> dict[str, str]:
    """Land the contract-12 version stamps next to the transcript.

    Every `ix_render` record is re-read through `from_ix_render_metadata`
    (inside `read_ix_render`), so a mismatched blob raises before the step
    artifacts are written. What is recorded in `meta.yaml` is the single pair
    of versions the step actually produced; a step with no rendered turn
    records nothing.
    """
    versions: dict[str, str] = {}
    for event in events:
        if event.get("kind") != "pa_turn":
            continue
        blob = read_ix_render(event)
        if blob is None:
            continue
        versions = {
            "ix_render_schema_version": blob["schema_version"],
            "ix_contract_version": blob["ix_contract_version"],
        }
    return versions


#: Which counting rules the `beats` / `tool_calls` / `elapsed_s` summary was
#: produced under. `tool_calls` already changed meaning once — the wrapper's
#: own tools stopped counting, see `_count_tool_calls` below — and the archive
#: kept no trace of which rule a given run used. Bump this whenever one of the
#: three changes, so the ledger says so instead of the number silently drifting.
SUMMARY_SCHEMA = 1


def _count_tool_calls(events: list[dict]) -> int:
    """Count real task tool calls only.

    Excluded: every `IX_*` tool (the 14 selection tools, the six
    `IX_render_*`, the four `IX_continue_*` and `IX_capability_resolve`) and
    every internal wrapper control event. `task_tool_events` owns that rule so
    the count and the transcript never drift apart.
    """
    return sum(
        len(task_tool_events(event))
        for event in events
        if event.get("kind") == "pa_turn"
    )
