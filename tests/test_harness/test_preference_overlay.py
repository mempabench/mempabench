"""Run-level simulator preference state written by forced events."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.forced_event_executor import ForcedEventExecutor
from harness.memory_runtime import MemoryRuntime
from harness.orchestrator import (
    LEDGER_OVERLAY_KEY,
    Orchestrator,
    OrchestratorConfig,
    RunPreferenceOverlay,
    probe_preference_overlay,
)
from harness.run_ledger import RunLedger
from harness.run_plan import RunPlan
from harness.session_executor import SessionExecutionResult
from harness.state_runtime import StageRuntime
from pa.providers.request_policy import resolve_model_request_policy


RUN_ID = "user_a_NanoBotFileMemory_openai_gpt55_cfgaaaaaaaaaa_r01"
#: The fixture Outline every step of these plans resolves its gate through.
#: 15 is the acc `evolv_01` sits on: a forced event resolves its gate through
#: that entry, the same as an accumulation session.
OUTLINE_TEXT = "sessions:\n" + "".join(
    f"  - identity:\n"
    f"      acc_num: {acc_num}\n"
    f"    probe:\n"
    f"      why_both_viable: compressed, so no gate\n"
    for acc_num in (13, 15, 40)
)

BENCHMARK_TIMESTAMP = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)


class RecordingExecutor:
    """Stand-in for `SessionExecutor` that reports what each step saw."""

    def __init__(
        self,
        *,
        overlay: RunPreferenceOverlay,
        fail_step: str | None = None,
        history: list[str] | None = None,
    ):
        self.overlay = overlay
        self.fail_step = fail_step
        self.history = history if history is not None else []
        self.seen: list[tuple[str, dict[str, str]]] = []

    async def execute(self, step, *, steps_dir, stage_runtime=None, extra_meta=None):
        self.seen.append((step.step_id, dict(self.overlay.cells)))
        if step.step_id == self.fail_step:
            raise RuntimeError("failed step")
        if step.kind == "forced_event":
            # What the real ForcedEventExecutor does at the end of its replay.
            self.overlay.record("work_autonomy_level", "Reactive")
            self.history.append(_Transcript.user_memory_seed)
        step_dir = Path(steps_dir) / step.step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        artifact = step_dir / "meta.yaml"
        artifact.write_text("ok: true\n", encoding="utf-8")
        return SessionExecutionResult(
            artifacts={"meta": str(artifact)},
            beats=1,
            tool_calls=0,
            elapsed_s=0.1,
        )


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "fixtures"
    (root / "user_a").mkdir(parents=True, exist_ok=True)
    (root / "user_a" / "base.txt").write_text("base", encoding="utf-8")
    catalog = tmp_path / "data/state_tools/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "tools:\n  document_read:\n    status: implemented\n",
        encoding="utf-8",
    )
    for name in ("acc_013", "evolv_01", "acc_040"):
        (tmp_path / f"{name}.yaml").write_text(
            (
                "meta:\n"
                "  artifact_kind: normal_session\n"
                "  tool_status: world_toolbox\n"
                "  persona_id: user_a\n"
                f"  session_id: {name}\n"
                "  target_cell: work_verbosity\n"
            ),
            encoding="utf-8",
        )
    # `resolve_session_gate` fails closed on an acc session with no outline
    # entry. These steps exercise the overlay, not gating, so they are
    # `verbosity` — a compressed attribute, which is the one shape that may
    # legitimately carry no gate.
    outline = tmp_path / "Authoring/outline/drafts/user_a/outline.yaml"
    outline.parent.mkdir(parents=True, exist_ok=True)
    outline.write_text(OUTLINE_TEXT, encoding="utf-8")
    return root


def _step(
    step_id: str,
    kind: str,
    acc_num: int,
    hour: int,
    *,
    session_type: str,
) -> dict:
    return {
        "step_id": step_id,
        "kind": kind,
        "persona_id": "user_a",
        "script_path": f"{step_id}.yaml",
        "context": "work",
        "target_cell": "work_autonomy_level",
        "target_attribute": "autonomy_level",
        "session_type": session_type,
        "acc_num": acc_num,
        "benchmark_session_timestamp": f"2026-10-15T{hour:02d}:00:00+00:00",
        "memory_mode": "read_write",
        "stage_policy": "commit",
    }


def _plan(steps: list[dict]) -> RunPlan:
    policy = resolve_model_request_policy("openrouter", "test-pa").model_copy(
        update={"policy_source": "model", "validated_roles": ["pa"]}
    )
    consolidation_policy = resolve_model_request_policy(
        "openrouter",
        "test-consolidation",
    ).model_copy(
        update={
            "policy_source": "model",
            "validated_roles": ["consolidation"],
        }
    )
    return RunPlan(
        run_id=RUN_ID,
        persona_id="user_a",
        memory_condition="nanobot_file_memory",
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base="https://openrouter.ai/api/v1",
        pa_model="test-pa",
        pa_request_policy=policy.snapshot(),
        consolidation_model="test-consolidation",
        consolidation_request_policy=consolidation_policy.snapshot(),
        steps=steps,
    )


def _timeline_plan() -> RunPlan:
    return _plan(
        [
            _step("acc_013", "accumulation", 13, 9, session_type="evolving_pre"),
            _step("evolv_01", "forced_event", 15, 11, session_type="evolving_event"),
            _step("acc_040", "accumulation", 40, 13, session_type="evolving_post"),
        ]
    )


def _orchestrator(
    plan: RunPlan,
    tmp_path: Path,
    *,
    overlay: RunPreferenceOverlay,
    executor,
) -> Orchestrator:
    return Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=tmp_path / "runs" / plan.run_id,
            project_root=tmp_path,
        ),
        executor=executor,
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            None,
            run_id=plan.run_id,
            persona_id=plan.persona_id,
        ),
        progress=lambda _line: None,
        preference_overlay=overlay,
    )


class _Event:
    context = "work"
    target_attribute = "autonomy_level"
    post_setting = "Reactive"


class _Transcript:
    event = _Event()
    user_memory_seed = (
        "I asked for the file and it went ahead and sent it. "
        "From now on I want to see it before it goes out."
    )


def _executor_under_test(
    *,
    overlay: RunPreferenceOverlay | None,
    history: list[str],
) -> ForcedEventExecutor:
    return ForcedEventExecutor(
        transcript_path=Path("unused.yaml"),
        project_root=Path("."),
        benchmark_session_timestamp=BENCHMARK_TIMESTAMP,
        stage_runtime=None,
        session_manager=None,
        pa_session_key="unused",
        memory_adapter=None,
        memory_io_recorder=None,
        writer=None,
        preference_overlay=overlay,
        history_appender=history.append,
    )


def test_forced_event_success_flips_the_cell_and_appends_the_seed() -> None:
    overlay = RunPreferenceOverlay()
    history: list[str] = []

    _executor_under_test(overlay=overlay, history=history)._commit_simulator_state(
        _Transcript()
    )

    assert overlay.cells == {"work_autonomy_level": "Reactive"}
    assert history == [f"[2026-10-15 09:00] {_Transcript.user_memory_seed}"]


def test_overlay_becomes_simulator_preference_overrides() -> None:
    overlay = RunPreferenceOverlay()
    overlay.record("work_autonomy_level", "Reactive")
    overlay.record("personal_topic_management", "Organize")

    assert overlay.preference_overrides() == {
        "work": {
            "autonomy_level": {
                "setting": "Reactive",
                "source": "forced_event",
                "confidence": "High",
            }
        },
        "personal": {
            "topic_management": {
                "setting": "Organize",
                "source": "forced_event",
                "confidence": "High",
            }
        },
    }


@pytest.mark.asyncio
async def test_overlay_reaches_every_later_accumulation_step(tmp_path: Path) -> None:
    plan = _timeline_plan()
    overlay = RunPreferenceOverlay()
    executor = RecordingExecutor(overlay=overlay)

    await _orchestrator(
        plan,
        tmp_path,
        overlay=overlay,
        executor=executor,
    ).run()

    assert executor.seen == [
        ("acc_013", {}),
        ("evolv_01", {}),
        ("acc_040", {"work_autonomy_level": "Reactive"}),
    ]


@pytest.mark.asyncio
async def test_failed_forced_event_leaves_no_overlay_entry(tmp_path: Path) -> None:
    plan = _timeline_plan()
    overlay = RunPreferenceOverlay()
    executor = RecordingExecutor(overlay=overlay, fail_step="evolv_01")
    orchestrator = _orchestrator(
        plan,
        tmp_path,
        overlay=overlay,
        executor=executor,
    )

    with pytest.raises(RuntimeError, match="failed step"):
        await orchestrator.run()

    assert overlay.cells == {}
    assert orchestrator.ledger.steps["evolv_01"].status == "failed"
    assert LEDGER_OVERLAY_KEY not in orchestrator.ledger.steps["acc_013"].summary


@pytest.mark.asyncio
async def test_resume_rebuilds_the_overlay_from_done_forced_events(
    tmp_path: Path,
) -> None:
    plan = _timeline_plan()
    history: list[str] = []
    first_overlay = RunPreferenceOverlay()
    first = _orchestrator(
        plan,
        tmp_path,
        overlay=first_overlay,
        executor=RecordingExecutor(
            overlay=first_overlay,
            fail_step="acc_040",
            history=history,
        ),
    )
    with pytest.raises(RuntimeError, match="failed step"):
        await first.run()
    first.ledger.steps["acc_040"].status = "skipped"
    first.ledger.save()

    assert first.ledger.steps["evolv_01"].summary[LEDGER_OVERLAY_KEY] == {
        "work_autonomy_level": "Reactive"
    }

    resumed_overlay = RunPreferenceOverlay()
    resumed = _orchestrator(
        plan,
        tmp_path,
        overlay=resumed_overlay,
        executor=RecordingExecutor(overlay=resumed_overlay, history=history),
    )
    result = await resumed.run()

    # The forced event does not replay, so its HISTORY.md seed is not appended
    # a second time; the flip comes back from the ledger instead.
    assert result.executed == []
    assert result.skipped == ["acc_013", "evolv_01", "acc_040"]
    assert resumed_overlay.cells == {"work_autonomy_level": "Reactive"}
    assert history == [_Transcript.user_memory_seed]


@pytest.mark.asyncio
async def test_accumulation_fails_fast_when_its_forced_event_was_skipped(
    tmp_path: Path,
) -> None:
    plan = _timeline_plan()
    overlay = RunPreferenceOverlay()
    orchestrator = _orchestrator(
        plan,
        tmp_path,
        overlay=overlay,
        executor=RecordingExecutor(overlay=overlay),
    )
    orchestrator.ledger.mark_done("evolv_01")

    with pytest.raises(
        RuntimeError,
        match=r"missing_cells=\['work_autonomy_level'\]",
    ):
        await orchestrator.run()


def test_freeze_rejects_a_post_step_without_its_forced_event(tmp_path: Path) -> None:
    plan = _plan(
        [
            _step("acc_013", "accumulation", 13, 9, session_type="evolving_pre"),
            _step("acc_040", "accumulation", 40, 11, session_type="evolving_post"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="no forced_event for its target cell",
    ):
        plan.dump_yaml(tmp_path / "run_plan.yaml")


def test_freeze_accepts_a_post_step_after_its_forced_event(tmp_path: Path) -> None:
    path = _timeline_plan().dump_yaml(tmp_path / "run_plan.yaml")

    assert path.is_file()


# ── per-probe overlay, as of the probe's own cut point ────────────────


def _probe_step(
    step_id: str,
    kind: str,
    hour: int,
    *,
    target_cell: str,
    target_setting: str,
    before_acc_num: int | None = None,
) -> dict:
    return {
        "step_id": step_id,
        "kind": kind,
        "persona_id": "user_a",
        "script_path": f"{step_id}.yaml",
        "context": target_cell.split("_", 1)[0],
        "target_cell": target_cell,
        "target_attribute": target_cell.split("_", 1)[1],
        "target_setting": target_setting,
        "before_acc_num": before_acc_num,
        "benchmark_session_timestamp": f"2026-10-15T{hour:02d}:00:00+00:00",
        "memory_mode": "read_only",
        "stage_policy": "discard",
    }


def _two_event_plan() -> RunPlan:
    """acc_013 → pre_01 → evolv_01(15) → acc_020 → pre_02 → evolv_02(30) → fin_01.

    Two forced events on two cells, a pre-event probe before each and one
    final. `pre_02` reads the cell `evolv_01` already flipped — the shape the
    attach-time gate exists to catch.
    """
    work = _step("evolv_01", "forced_event", 15, 11, session_type="evolving_event")
    personal = {
        **_step("evolv_02", "forced_event", 30, 14, session_type="evolving_event"),
        "target_cell": "personal_topic_management",
        "target_attribute": "topic_management",
        "context": "personal",
    }
    return _plan(
        [
            _step("acc_013", "accumulation", 13, 9, session_type="evolving_pre"),
            _probe_step(
                "pre_01",
                "pre_event_probe",
                10,
                target_cell="work_autonomy_level",
                target_setting="Self-directed",
                before_acc_num=15,
            ),
            work,
            _step("acc_020", "accumulation", 20, 12, session_type="evolving_pre"),
            _probe_step(
                "pre_02",
                "pre_event_probe",
                13,
                target_cell="work_autonomy_level",
                target_setting="Self-directed",
                before_acc_num=30,
            ),
            personal,
            _probe_step(
                "fin_01",
                "final_probe",
                15,
                target_cell="work_autonomy_level",
                target_setting="Reactive",
            ),
        ]
    )


def _ran_both_events(tmp_path: Path, plan: RunPlan) -> RunLedger:
    ledger = RunLedger(tmp_path / "runs" / plan.run_id, plan.run_id)
    ledger.mark_done(
        "evolv_01",
        summary={LEDGER_OVERLAY_KEY: {"work_autonomy_level": "Reactive"}},
    )
    ledger.mark_done(
        "evolv_02",
        summary={LEDGER_OVERLAY_KEY: {"personal_topic_management": "One-at-a-time"}},
    )
    return ledger


def test_rebuild_as_of_replays_only_the_flips_before_that_acc(
    tmp_path: Path,
) -> None:
    plan = _two_event_plan()
    ledger = _ran_both_events(tmp_path, plan)
    overlay = RunPreferenceOverlay()

    overlay.rebuild_from_ledger(plan, ledger, before_acc_num=15)
    assert overlay.cells == {}

    overlay.rebuild_from_ledger(plan, ledger, before_acc_num=30)
    assert overlay.cells == {"work_autonomy_level": "Reactive"}

    overlay.rebuild_from_ledger(plan, ledger)
    assert overlay.cells == {
        "work_autonomy_level": "Reactive",
        "personal_topic_management": "One-at-a-time",
    }


def test_each_probe_gets_its_own_overlay_as_of_its_cut_point(
    tmp_path: Path,
) -> None:
    plan = _two_event_plan()
    ledger = _ran_both_events(tmp_path, plan)

    cells = {
        step.step_id: probe_preference_overlay(plan, ledger, step).cells
        for step in plan.steps
        if step.kind in {"pre_event_probe", "final_probe"}
    }

    assert cells == {
        "pre_01": {},
        "pre_02": {"work_autonomy_level": "Reactive"},
        "fin_01": {
            "work_autonomy_level": "Reactive",
            "personal_topic_management": "One-at-a-time",
        },
    }


@pytest.mark.parametrize("persona_id", ["user_a", "user_b", "user_c", "user_d"])
def test_each_shipped_probe_runs_under_its_own_preference_state(
    tmp_path: Path, persona_id: str
) -> None:
    """On the shipped data, with every forced event done: a pre-event probe
    asks under its cell's pre setting, which is the resolved GT's, and a final
    runs with every flip."""
    from harness import run_plan_builder

    plan = run_plan_builder._build_frozen_run_plan(
        run_id=f"overlay_{persona_id}",
        persona_id=persona_id,
        memory_condition="no_memory",
        pa_model="deepseek/deepseek-v4-flash",
        script_names=[f"acc_{num:03d}" for num in range(1, 109)],
        with_probes=True,
    )
    gt_cells = run_plan_builder._load_resolved_gt(
        run_plan_builder._resolved_gt_source(persona_id), persona_id=persona_id
    )
    ledger = RunLedger(tmp_path / plan.run_id, plan.run_id)
    flips = {}
    for step in plan.steps:
        if step.kind == "forced_event":
            flips[step.target_cell] = gt_cells[step.target_cell]["post_setting"]
            ledger.mark_done(
                step.step_id,
                summary={LEDGER_OVERLAY_KEY: {step.target_cell: flips[step.target_cell]}},
            )
    assert len(flips) == 5

    for step in plan.steps:
        overlay = probe_preference_overlay(plan, ledger, step).cells
        if step.kind == "pre_event_probe":
            assert step.target_cell not in overlay, step.step_id
            assert step.target_setting == gt_cells[step.target_cell]["pre_setting"]
        elif step.kind == "final_probe":
            assert overlay == flips, step.step_id
