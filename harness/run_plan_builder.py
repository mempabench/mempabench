"""Build and freeze benchmark RunPlans from canonical authoring artifacts."""

from __future__ import annotations

import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from harness.benchmark_clock import benchmark_session_timestamps
from harness.forced_event_schema import load_forced_event_transcript
from harness.run_config import load_run_config
from harness.run_ledger import PROBE_KINDS
from harness.run_plan import (
    EMBEDDING_MEMORY_CONDITIONS,
    FILE_MEMORY_CONDITION,
    FULL_TIMELINE_ACC_COUNT,
    FrozenProbeOverlay,
    FrozenResolvedGT,
    MEM0_CONDITION,
    Mem0RuntimeConfig,
    MEMORY_SIDE_LM_CONDITIONS,
    MEMOS_CONDITION,
    MemOSRuntimeConfig,
    RESOLVED_GT_COPY_NAME,
    RunPlan,
    RunStep,
    SimulatorRuntimeConfig,
    validate_full_timeline_shape,
)
from pa.agent.ix.render_types import IX_CONTRACT_VERSION
from memory.backends.runtime_contracts import (
    MEM0_OPENAI_EMBEDDING_DIMENSIONS,
    MEM0_OPENAI_EMBEDDING_MODEL,
    MEM0_OPENAI_EMBEDDING_PROVIDER,
)
from pa.providers.request_policy import resolve_model_request_policy
from scripts.resolved_gt import PreEvent, pre_events, setting_for
from scripts.session_authoring_schema import FORCED_EVENT_ID_BY_ACC
from scripts.probe_overlay import (
    _declared_pairing,
    effective_overlay,
    overlay_owner,
)
from simulator.schemas import SessionScript


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
#: `meta.artifact_kind` values loaded through `from_authored_session_yaml`.
AUTHORED_ARTIFACT_KINDS = frozenset({"normal_session", "probe_session"})


def _validate_forced_transcript(path: Path) -> None:
    """D5: the forced-event transcript passes its schema and source checks now, at freeze.

    `harness.forced_event_schema.validate_forced_event_transcript` in process:
    schema and the sources the transcript names. A failing transcript is not
    frozen. Replaying every tool call against a scratch copy of the persona's
    world is the authoring validator's check; it runs on the published persona
    data before release. At run time each call's `ok` and pinned
    `expected_contains` are still enforced (`harness/forced_event_executor.py`).
    """
    from harness.forced_event_schema import (
        ForcedEventTranscriptLoadError,
        validate_forced_event_transcript,
    )

    try:
        validate_forced_event_transcript(path, project_root=PROJECT_ROOT)
    except ForcedEventTranscriptLoadError as exc:
        raise ValueError(
            f"the forced-event transcript does not pass its validator: path={path}\n{exc}"
        ) from exc


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _embedding_settings(memory_condition: str) -> tuple[str, str, int]:
    if memory_condition in EMBEDDING_MEMORY_CONDITIONS:
        return (
            MEM0_OPENAI_EMBEDDING_PROVIDER,
            MEM0_OPENAI_EMBEDDING_MODEL,
            MEM0_OPENAI_EMBEDDING_DIMENSIONS,
        )
    return "", "", 0


def _new_memos_cube_id(run_id: str) -> str:
    return f"mempa_{_slug(run_id)}_memos_{uuid.uuid4().hex}"


def _script_path(
    persona_id: str,
    script_name: str,
    *,
    outline_entry: dict[str, Any],
) -> Path:
    if outline_entry.get("session_type") == "evolving_event":
        acc_num = outline_entry.get("acc_num")
        event_id = FORCED_EVENT_ID_BY_ACC.get(acc_num)
        if event_id is None:
            raise ValueError(
                f"No forced event id is defined for acc_num={acc_num!r}"
            )
        path = (
            PROJECT_ROOT
            / "Authoring/sessions/forced/transcripts"
            / persona_id
            / event_id
            / "transcript.yaml"
        )
    else:
        path = (
            PROJECT_ROOT
            / "Authoring/sessions/normal/drafts"
            / persona_id
            / f"{script_name}.yaml"
        )
    if not path.exists():
        raise FileNotFoundError(f"runtime artifact not found: path={path}")
    return path


def _placeholder_step(
    persona_id: str, acc_num: int, *, timestamp: datetime
) -> RunStep:
    """A gap day: an acc slot a complete Outline authored no session for.

    The acc skeleton runs 1..108 while a complete Outline holds 99 sessions
    (`meta.active_session_count`), so nine slots per persona carry nothing.
    The plan keeps the slot so the numbering and every `before_acc` reference
    still line up, and marks it `placeholder`: the orchestrator runs nothing
    for it, no script is read, no cut point is published, and neither the PA
    nor the simulator ever sees it. Its `benchmark_session_timestamp` is the
    previous authored session's -- the day sequence is carried by the authored
    clocks, and a slot that never runs has no clock of its own.
    """
    return RunStep(
        step_id=f"acc_{acc_num:03d}",
        kind="accumulation",
        persona_id=persona_id,
        script_path=(
            Path("Authoring/sessions/normal/drafts")
            / persona_id
            / f"acc_{acc_num:03d}.yaml"
        ),
        acc_num=acc_num,
        benchmark_session_timestamp=timestamp,
        memory_mode="read_write",
        stage_policy="commit",
        placeholder=True,
    )


def _load_script(path: Path) -> SessionScript:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    meta = raw.get("meta") if isinstance(raw, dict) else None
    if isinstance(meta, dict) and meta.get("artifact_kind") in AUTHORED_ARTIFACT_KINDS:
        return SessionScript.from_authored_session_yaml(path)
    return SessionScript.from_yaml(path)


def _outline_error_value_preview(value: Any) -> str:
    preview = repr(value)
    return preview if len(preview) <= 160 else preview[:157] + "..."


def _load_outline_sessions_by_acc_num(
    path: Path,
    *,
    requested_acc_nums: Iterable[int] | None = None,
) -> dict[int, dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"Failed to read session outline: path={path} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    try:
        outline = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid session outline YAML: path={path} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(outline, dict):
        raise ValueError(
            f"Session outline root must be a mapping: path={path} "
            f"root_type={type(outline).__name__} "
            f"value={_outline_error_value_preview(outline)}"
        )
    sessions = outline.get("sessions", [])
    if not isinstance(sessions, list):
        raise ValueError(
            f"Session outline 'sessions' must be a list: path={path} "
            f"value={_outline_error_value_preview(sessions)}"
        )

    sessions_by_acc_num: dict[int, dict[str, Any]] = {}
    for index, session in enumerate(sessions):
        if not isinstance(session, dict):
            raise ValueError(
                f"Session outline entry must be a mapping: path={path} "
                f"sessions_index={index} "
                f"value={_outline_error_value_preview(session)}"
            )
        identity = session.get("identity")
        story = session.get("story")
        if not isinstance(identity, dict) or not isinstance(story, dict):
            raise ValueError(
                f"Session outline entry requires identity and story groups: "
                f"path={path} sessions_index={index} "
                f"value={_outline_error_value_preview(session)}"
            )
        if "acc_num" not in identity:
            raise ValueError(
                f"Session outline entry requires identity.acc_num: path={path} "
                f"sessions_index={index} "
                f"value={_outline_error_value_preview(session)}"
            )
        raw_acc_num = identity["acc_num"]
        if type(raw_acc_num) is not int or raw_acc_num <= 0:
            raise ValueError(
                f"Session outline acc_num must be a positive integer: path={path} "
                f"sessions_index={index} value={raw_acc_num!r}"
            )
        acc_num = raw_acc_num
        if acc_num in sessions_by_acc_num:
            raise ValueError(
                f"Duplicate session outline acc_num: path={path} "
                f"sessions_index={index} value={acc_num!r}"
            )
        target_cell = identity.get("target_cell")
        sessions_by_acc_num[acc_num] = {
            "acc_num": acc_num,
            "session_type": identity.get("session_type"),
            "timeline_slot": identity.get("timeline_slot"),
            "target_cell": target_cell,
            "domain": story.get("domain"),
            "story_id": story.get("story_id"),
            "time_position": story.get("time_position"),
            "date": story.get("date"),
            "hour": story.get("hour"),
        }
    meta = outline.get("meta")
    status = meta.get("status") if isinstance(meta, dict) else None
    if status == "complete":
        return sessions_by_acc_num
    if status == "in_progress" and requested_acc_nums is not None:
        pending = meta.get("pending_acc_nums")
        if not isinstance(pending, list) or any(
            type(item) is not int for item in pending
        ):
            raise ValueError(
                "Authoring Outline is in_progress without an integer "
                f"meta.pending_acc_nums list: path={path} "
                f"value={_outline_error_value_preview(pending)}"
            )
        requested = list(requested_acc_nums)
        pending_hits = sorted({acc for acc in requested if acc in set(pending)})
        missing = sorted(
            {
                acc
                for acc in requested
                if acc not in sessions_by_acc_num and acc not in set(pending)
            }
        )
        if pending_hits or missing:
            raise ValueError(
                "Authoring Outline is in_progress and does not cover every "
                f"requested session: path={path} "
                f"pending_acc_nums={pending_hits} missing_acc_nums={missing}"
            )
        return sessions_by_acc_num
    raise ValueError(
        f"Authoring Outline is not complete: path={path} status={status!r}"
    )


def _load_probe_outline(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The probe outline's `meta` and its `probes`, sorted by id."""
    if not path.exists():
        raise FileNotFoundError(f"Probe outline not found: path={path}")
    outline = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(outline, dict):
        raise ValueError(
            f"Probe outline root must be a mapping: path={path} "
            f"root_type={type(outline).__name__}"
        )
    meta = outline.get("meta")
    probes = outline.get("probes")
    if not isinstance(meta, dict) or not isinstance(probes, list):
        raise ValueError(
            f"Probe outline requires meta and a probes list: path={path}"
        )
    ids: set[str] = set()
    for index, entry in enumerate(probes):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Probe outline entry must be a mapping: path={path} "
                f"probes_index={index} "
                f"value={_outline_error_value_preview(entry)}"
            )
        for key in ("id", "kind", "cell", "clock"):
            if key not in entry:
                raise ValueError(
                    f"Probe outline entry requires {key}: path={path} "
                    f"probes_index={index} "
                    f"value={_outline_error_value_preview(entry)}"
                )
        if entry["kind"] not in {"final", "pre_event"}:
            raise ValueError(
                "Probe outline kind must be final or pre_event: "
                f"path={path} id={entry['id']!r} value={entry['kind']!r}"
            )
        if (entry["kind"] == "pre_event") != ("before_acc" in entry):
            raise ValueError(
                "before_acc is present exactly on a pre_event probe: "
                f"path={path} id={entry['id']!r} kind={entry['kind']!r}"
            )
        if entry["id"] in ids:
            raise ValueError(
                f"Duplicate probe outline id: path={path} id={entry['id']!r}"
            )
        ids.add(entry["id"])
    return meta, sorted(probes, key=lambda entry: entry["id"])


def _load_resolved_gt(path: Path, *, persona_id: str) -> dict[str, dict[str, Any]]:
    """The `cells` mapping of the persona's `resolved_gt.yaml`."""
    if not path.exists():
        raise FileNotFoundError(f"resolved_gt.yaml not found: path={path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cells"), dict):
        raise ValueError(
            f"resolved_gt.yaml requires persona and a cells mapping: path={path}"
        )
    if data.get("persona") != persona_id:
        raise ValueError(
            "resolved_gt.yaml persona does not match the plan: "
            f"path={path} value={data.get('persona')!r} expected={persona_id!r}"
        )
    return data["cells"]


def _clock_to_timestamp(clock: Any, *, probe_id: str) -> datetime:
    """A probe outline `clock` (`{date: YYYY-MM-DD, hour: 0-23}`) as UTC."""
    try:
        day = date.fromisoformat(str(clock["date"]))
        hour = int(clock["hour"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Probe clock must carry date (YYYY-MM-DD) and hour (0-23): "
            f"id={probe_id!r} value={clock!r}"
        ) from exc
    if not 0 <= hour <= 23:
        raise ValueError(f"Probe clock hour must be 0-23: id={probe_id!r} value={hour!r}")
    return datetime(day.year, day.month, day.day, hour, 0, tzinfo=timezone.utc)


def _probe_outline_dir(persona_id: str) -> Path:
    return PROJECT_ROOT / "Authoring/probe/outline" / persona_id


def _resolved_gt_source(persona_id: str) -> Path:
    return _probe_outline_dir(persona_id) / RESOLVED_GT_COPY_NAME


def persona_has_probe_line(persona_id: str) -> bool:
    """True when the persona's probe outline and resolved GT both exist.

    One without the other is an authoring error, not "no probes": the plan
    must not silently freeze as a timeline alone.
    """
    outline_dir = _probe_outline_dir(persona_id)
    has_outline = (outline_dir / "outline.yaml").is_file()
    has_gt = _resolved_gt_source(persona_id).is_file()
    if has_outline != has_gt:
        raise FileNotFoundError(
            "A probe line needs both outline.yaml and resolved_gt.yaml: "
            f"dir={outline_dir} outline={has_outline} resolved_gt={has_gt}"
        )
    return has_outline


def freeze_resolved_gt_copy(plan: RunPlan, run_dir: Path) -> Path:
    """Copy the persona's `resolved_gt.yaml`, as it is now, into the run
    directory (§7 row 8)."""
    if plan.resolved_gt is None:
        raise ValueError(
            f"plan records no resolved_gt to freeze: run_id={plan.run_id!r}"
        )
    source = _resolved_gt_source(plan.persona_id)
    target = run_dir / plan.resolved_gt.path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _probe_steps(
    persona_id: str, *, acc_nums: set[int], timeline_end_acc: int
) -> list[RunStep]:
    """Construct the probe steps whose cut point lies in the plan's accs.

    A pre_event probe's cut point is its `before_acc`: it is in the plan when
    `acc_nums` holds that acc. The finals' cut point is the end of the
    timeline, after the Outline's last session (`timeline_end_acc`): they are
    in the plan when `acc_nums` holds that acc. A probe outside is neither
    constructed nor validated, so a full-timeline plan carries every probe and
    a shorter range carries only its own.

    Construction only: where each step goes in the plan is decided by
    `_merge_probe_steps`. The clock is the outline's `clock` (a final's must
    be the frozen `meta.final_clock`); the target setting comes from
    `resolved_gt.yaml` through `setting_for`, never re-derived from the
    persona matrix. A draft is frozen only once its overlay owner is settled,
    checked here (`_validate_probe_draft`).
    """
    outline_dir = _probe_outline_dir(persona_id)
    meta, probes = _load_probe_outline(outline_dir / "outline.yaml")
    cells = _load_resolved_gt(_resolved_gt_source(persona_id), persona_id=persona_id)
    # The pairing the frozen GT declares, handed to the overlay resolution so a
    # draft whose own `meta.pairs_with` says something else is refused rather than
    # signed off over whichever directory the caller happened to look at.
    gt_pairs = pre_events({"cells": cells})
    final_clock = _clock_to_timestamp(meta.get("final_clock"), probe_id="meta.final_clock")
    steps: list[RunStep] = []
    for entry in probes:
        cut_point_acc = (
            entry["before_acc"] if entry["kind"] == "pre_event" else timeline_end_acc
        )
        if cut_point_acc not in acc_nums:
            continue
        probe_id = entry["id"]
        context, attribute = entry["cell"].split("_", 1)
        record = cells.get(entry["cell"])
        if record is None:
            raise ValueError(
                "resolved_gt.yaml has no record for the probe's cell: "
                f"id={probe_id!r} cell={entry['cell']!r}"
            )
        timestamp = _clock_to_timestamp(entry["clock"], probe_id=probe_id)
        if entry["kind"] == "final" and timestamp != final_clock:
            raise ValueError(
                "a final probe's clock must be the outline's frozen "
                f"meta.final_clock: id={probe_id!r} "
                f"value={timestamp.isoformat()!r} "
                f"final_clock={final_clock.isoformat()!r}"
            )
        path = (
            PROJECT_ROOT / "Authoring/probe/drafts" / persona_id / probe_id
            / f"{probe_id}.yaml"
        )
        if not path.exists():
            raise FileNotFoundError(f"runtime artifact not found: path={path}")
        frozen_overlay = _validate_probe_draft(probe_id, path, gt_pairs=gt_pairs)
        steps.append(
            RunStep(
                step_id=probe_id,
                kind=(
                    "pre_event_probe" if entry["kind"] == "pre_event"
                    else "final_probe"
                ),
                persona_id=persona_id,
                script_path=path.relative_to(PROJECT_ROOT),
                probe_overlay=frozen_overlay,
                context=context,
                target_cell=entry["cell"],
                target_attribute=attribute,
                target_setting=setting_for(entry["kind"], record),
                before_acc_num=entry.get("before_acc"),
                benchmark_session_timestamp=timestamp,
                memory_mode="read_only",
                skip_consolidation=True,
                stage_policy="discard",
            )
        )
    return steps


def _validate_probe_draft(
    probe_id: str,
    draft_path: Path,
    *,
    gt_pairs: Mapping[str, PreEvent] | None = None,
) -> FrozenProbeOverlay:
    """D5: the probe's fixture overlay is settled now, at freeze.

    The overlay the probe will run over is resolved the one shared way (for a
    final a pre_event probe pairs with, the pre's `fixture/`), and the drafts'
    own declared pairing has to name the same owner, because that is what the
    run follows. The owner is recorded in the step; nothing reads it back. The draft's authoring validator runs on the published persona data
    before release, not here.
    """
    persona_id = draft_path.parent.parent.name
    overlay = effective_overlay(
        persona_id,
        probe_id,
        drafts_root=draft_path.parent.parent.parent,
        gt_pairs=gt_pairs,
    )
    declared_kind, declared_pairs_with = _declared_pairing(draft_path)
    runtime_owner = overlay_owner(
        probe_id,
        persona=persona_id,
        declared_kind=declared_kind,
        declared_pairs_with=declared_pairs_with,
    )
    if runtime_owner != overlay.owner_probe_id:
        raise ValueError(
            "the run would overlay a different directory than the resolved GT "
            f"pairs this probe with: probe_id={probe_id!r} "
            f"resolved_gt_owner={overlay.owner_probe_id!r} "
            f"runtime_owner={runtime_owner!r}; the drafts of a pair each declare it "
            "(`meta.probe_kind`, `meta.pairs_with`) and these do not"
        )
    return FrozenProbeOverlay(owner_probe_id=overlay.owner_probe_id)


def _merge_probe_steps(steps: list[RunStep], probe_steps: list[RunStep]) -> list[RunStep]:
    """Place probes by cut point: each pre-event probe immediately before the
    timeline step its `before_acc_num` names, the finals after the last
    timeline step. Appending everything would collapse the five pre-event pins
    onto the final pin."""
    merged = list(steps)
    for probe in probe_steps:
        if probe.kind != "pre_event_probe":
            continue
        position = next(
            (
                index
                for index, step in enumerate(merged)
                if step.kind in {"accumulation", "forced_event"}
                and step.acc_num == probe.before_acc_num
            ),
            None,
        )
        if position is None:
            raise ValueError(
                "a pre_event probe's before_acc step is not in the plan: "
                f"step_id={probe.step_id!r} "
                f"before_acc_num={probe.before_acc_num!r}"
            )
        merged.insert(position, probe)
    merged.extend(probe for probe in probe_steps if probe.kind == "final_probe")
    return merged


def _build_frozen_run_plan(
    *,
    run_id: str,
    persona_id: str,
    memory_condition: str,
    pa_model: str,
    script_names: list[str],
    with_probes: bool = False,
    assistant_llm_provider_name: str = "openrouter",
    assistant_llm_api_base: str = OPENROUTER_API_BASE,
    consolidation_model: str = "",
    memory_side_model: str = "",
    qdrant_host: str = "",
    qdrant_port: int = 0,
    mem0_runtime_config: Mem0RuntimeConfig | None = None,
    memos_runtime_config: MemOSRuntimeConfig | None = None,
    simulator_runtime: SimulatorRuntimeConfig | None = None,
    existing_plan: RunPlan | None = None,
    done_step_ids: Iterable[str] = (),
) -> RunPlan:
    """Freeze a run plan; `existing_plan` is what a re-freeze may not change.

    `--regenerate-plan` on a finished run re-freezes the plan so a probe line
    can be attached to it (§2.4.6), and everything it rebuilds from today's
    artifacts it rebuilds over steps that have already run. So the caller hands
    back the plan those steps ran under together with the ledger's done step
    ids, and two things are carried over instead of re-minted: the mainline
    `memos_cube_id`, which the checkpoints already name, and every done
    non-probe step, verbatim. Probe steps are always
    rebuilt -- that is what regenerating is for.
    """
    simulator_runtime = simulator_runtime or SimulatorRuntimeConfig()
    uses_file_memory = memory_condition == FILE_MEMORY_CONDITION
    if uses_file_memory and not consolidation_model:
        raise ValueError(
            "nanobot_file_memory requires consolidation_model: "
            f"value={consolidation_model!r}"
        )
    if not uses_file_memory and consolidation_model:
        raise ValueError(
            "consolidation_model is only valid for nanobot_file_memory: "
            f"memory_condition={memory_condition!r} "
            f"consolidation_model={consolidation_model!r}"
        )
    uses_memory_side_lm = memory_condition in MEMORY_SIDE_LM_CONDITIONS
    if uses_memory_side_lm and not memory_side_model:
        raise ValueError(
            f"{memory_condition} requires memory_side_model: "
            f"value={memory_side_model!r}"
        )
    if not uses_memory_side_lm and memory_side_model:
        raise ValueError(
            "memory_side_model is only valid for a condition that uses a "
            "memory-side LM: "
            f"memory_condition={memory_condition!r} "
            f"memory_side_model={memory_side_model!r}"
        )
    uses_mem0_runtime = memory_condition in {MEM0_CONDITION, "simple_rag"}
    if uses_mem0_runtime != (mem0_runtime_config is not None):
        raise ValueError(
            "mem0_runtime_config must be set exactly for Mem0-backed "
            f"conditions: memory_condition={memory_condition!r} "
            f"mem0_runtime_config_set={mem0_runtime_config is not None!r}"
        )
    uses_memos = memory_condition == MEMOS_CONDITION
    if uses_memos != (memos_runtime_config is not None):
        raise ValueError(
            "memos_runtime_config must be set exactly for memos: "
            f"memory_condition={memory_condition!r} "
            f"memos_runtime_config_set={memos_runtime_config is not None!r}"
        )
    pa_request_policy = resolve_model_request_policy(
        assistant_llm_provider_name,
        pa_model,
    )
    pa_request_policy.require_exact_role("pa")
    consolidation_request_policy = None
    memory_side_request_policy = None
    if uses_file_memory:
        consolidation_request_policy = resolve_model_request_policy(
            assistant_llm_provider_name,
            consolidation_model,
        )
        consolidation_request_policy.require_exact_role("consolidation")
    if uses_memory_side_lm:
        memory_side_request_policy = resolve_model_request_policy(
            assistant_llm_provider_name,
            memory_side_model,
        )
        memory_side_request_policy.require_exact_role("memory_extraction")

    steps: list[RunStep] = []
    # Keyed by acc_num because that is what a timeline step is addressed by
    # here before its file is read: a forced step's id lives inside the
    # transcript this rebuild is not going to open again.
    carried_steps_by_acc: dict[int, RunStep] = {}
    if existing_plan is not None:
        carried_ids = set(done_step_ids)
        carried_steps_by_acc = {
            step.acc_num: step
            for step in existing_plan.steps
            if step.kind not in PROBE_KINDS
            and step.acc_num is not None
            and step.step_id in carried_ids
        }
    outline_path = (
        PROJECT_ROOT / f"Authoring/outline/drafts/{persona_id}/outline.yaml"
    )
    if not outline_path.exists():
        raise FileNotFoundError(
            f"Authoring Outline not found: path={outline_path}"
        )
    requested_acc_nums = [
        int(name.removeprefix("acc_")) if name.startswith("acc_") else position
        for position, name in enumerate(script_names, start=1)
    ]
    outline_by_acc = _load_outline_sessions_by_acc_num(
        outline_path,
        requested_acc_nums=requested_acc_nums,
    )
    benchmark_timestamps = benchmark_session_timestamps(outline_by_acc)
    for index, script_name in enumerate(script_names, start=1):
        acc_num = requested_acc_nums[index - 1]
        if acc_num not in outline_by_acc:
            # Inside the skeleton a complete Outline's silence is a gap day,
            # not a hole: `_load_outline_sessions_by_acc_num` has already
            # refused an in-progress Outline that does not cover the request.
            # Outside it there is nothing to be a gap in.
            if acc_num > FULL_TIMELINE_ACC_COUNT:
                raise ValueError(
                    "Session is missing from the Authoring Outline: "
                    f"script_name={script_name!r} acc_num={acc_num!r} "
                    f"outline_path={outline_path}"
                )
            steps.append(
                _placeholder_step(
                    persona_id,
                    acc_num,
                    timestamp=(
                        steps[-1].benchmark_session_timestamp
                        if steps
                        else benchmark_timestamps[min(benchmark_timestamps)]
                    ),
                )
            )
            continue
        outline_meta = outline_by_acc[acc_num]
        path = _script_path(
            persona_id,
            script_name,
            outline_entry=outline_meta,
        )
        is_forced = outline_meta.get("session_type") == "evolving_event"
        carried = carried_steps_by_acc.get(acc_num)
        if carried is not None:
            rebuilt_kind = "forced_event" if is_forced else "accumulation"
            rebuilt_script_path = path.relative_to(PROJECT_ROOT)
            if (
                carried.kind != rebuilt_kind
                or carried.script_path != rebuilt_script_path
                or carried.acc_num != acc_num
            ):
                raise ValueError(
                    "re-freezing this plan would change a step the run has "
                    "already finished, so the plan would stop describing what "
                    f"ran: step_id={carried.step_id!r} "
                    f"done_kind={carried.kind!r} rebuilt_kind={rebuilt_kind!r} "
                    f"done_script_path={str(carried.script_path)!r} "
                    f"rebuilt_script_path={str(rebuilt_script_path)!r} "
                    f"done_acc_num={carried.acc_num!r} "
                    f"rebuilt_acc_num={acc_num!r}"
                )
            steps.append(carried)
            continue
        if is_forced:
            _validate_forced_transcript(path)
            transcript = load_forced_event_transcript(path)
            runtime_meta = {
                "target_cell": (
                    f"{transcript.event.context}_"
                    f"{transcript.event.target_attribute}"
                ),
                "target_attribute": transcript.event.target_attribute,
                "target_setting": transcript.event.pre_setting,
            }
            context = transcript.event.context
            step_id = transcript.event.event_id
        else:
            script = _load_script(path)
            raw_session = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            runtime_meta = raw_session.get("meta") or script.meta or {}
            context = script.context
            step_id = (
                script_name
                if script_name.startswith("acc_")
                else f"acc_{index:03d}"
            )

        def meta_value(key: str) -> Any:
            value = runtime_meta.get(key)
            return value if value is not None else outline_meta.get(key)

        target_cell = meta_value("target_cell")
        target_attribute = meta_value("target_attribute")
        if target_attribute is None and isinstance(target_cell, str):
            target_attribute = target_cell.split("_", 1)[-1]

        steps.append(
            RunStep(
                step_id=step_id,
                kind="forced_event" if is_forced else "accumulation",
                persona_id=persona_id,
                script_path=path.relative_to(PROJECT_ROOT),
                context=context,
                target_cell=target_cell,
                target_attribute=target_attribute,
                target_setting=meta_value("target_setting"),
                acc_num=acc_num,
                session_type=meta_value("session_type"),
                domain=meta_value("domain"),
                story_id=meta_value("story_id"),
                timeline_slot=meta_value("timeline_slot"),
                time_position=meta_value("time_position"),
                benchmark_session_timestamp=benchmark_timestamps[acc_num],
                memory_mode="read_write",
                stage_policy="commit",
            )
        )
    resolved_gt: FrozenResolvedGT | None = None
    if with_probes:
        steps = _merge_probe_steps(
            steps,
            _probe_steps(
                persona_id,
                acc_nums=set(requested_acc_nums),
                timeline_end_acc=max(outline_by_acc),
            ),
        )
        resolved_gt = FrozenResolvedGT(path=RESOLVED_GT_COPY_NAME)
    embedding_provider, embedding_model, embedding_dimensions = (
        _embedding_settings(memory_condition)
    )
    # PA / retrieval / checkpoint budgets and the provider avoidance threshold
    # come from configs/run_config.yaml and are frozen into the plan; a resumed
    # run keeps its plan's values.
    config = load_run_config()
    budgets = config.budgets
    plan = RunPlan(
        run_id=run_id,
        persona_id=persona_id,
        memory_condition=memory_condition,
        pa_session_key=f"{run_id}_main",
        assistant_llm_provider_name=assistant_llm_provider_name,
        assistant_llm_api_base=assistant_llm_api_base,
        pa_model=pa_model,
        pa_request_policy=pa_request_policy.snapshot(),
        consolidation_model=consolidation_model,
        consolidation_request_policy=(
            consolidation_request_policy.snapshot()
            if consolidation_request_policy
            else None
        ),
        memory_side_model=memory_side_model,
        memory_side_request_policy=(
            memory_side_request_policy.snapshot()
            if memory_side_request_policy
            else None
        ),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        mem0_runtime_config=mem0_runtime_config,
        memos_runtime_config=memos_runtime_config,
        memos_cube_id=(
            # A re-freeze keeps the namespace the finished steps wrote into.
            (
                (existing_plan.memos_cube_id if existing_plan else "")
                or _new_memos_cube_id(run_id)
            )
            if uses_memos
            else ""
        ),
        simulator_runtime=simulator_runtime,
        pa_max_tokens=budgets.pa_max_tokens,
        provider_avoid_after_failures=config.providers.avoid_after_failures,
        memory_retrieval_max_tokens=budgets.memory_retrieval_max_tokens,
        memory_retrieval_max_items=budgets.memory_retrieval_max_items,
        checkpoint_rolling_window=budgets.checkpoint_rolling_window,
        generated_from={
            "timeline_path": "manual real PA run",
            "outline_path": str(outline_path.relative_to(PROJECT_ROOT)),
        },
        resolved_gt=resolved_gt,
        ix_contract_version=IX_CONTRACT_VERSION,
        steps=steps,
    )
    # A plan over the whole skeleton is a full benchmark run, so it has to have
    # the full shape: 1..108 with 99 active steps, 5 forced, 5 pre-event and 28
    # final probes. An `--acc-range` sub-range is a slice of that run and meets
    # none of those counts, so it is not checked.
    if requested_acc_nums == list(range(1, FULL_TIMELINE_ACC_COUNT + 1)):
        validate_full_timeline_shape(plan)
    return plan
