"""Independent repeats borrow accepted history, never its Probe results."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from harness import judge_audit
from harness import probe_repeat_source as source_refs
from harness.metrics import PROBE_IDS
from harness.run_ledger import RunLedger, StepRecord
from harness.run_plan import RunPlan, validate_full_timeline_shape
from test_harness import test_run_plan_builder_probes as plan_support


ROOT = Path(__file__).resolve().parents[2]
PROBE_KINDS = {"pre_event_probe", "final_probe"}


@pytest.fixture(scope="module")
def frozen_project(tmp_path_factory):
    """Reuse the full 108-slot/33-Probe fixture and its real frozen builder."""
    project = tmp_path_factory.mktemp("repeat-source-project")
    with pytest.MonkeyPatch.context() as patch:
        plan_support.full_skeleton_root.__wrapped__(project, patch)
        plan = plan_support._build(script_names=[f"acc_{index:03d}" for index in range(1, 109)])
    validate_full_timeline_shape(plan)
    assert isinstance(plan, RunPlan)
    assert len([step for step in plan.steps if step.kind in PROBE_KINDS]) == 33
    # These are the actual code files bound by the existing scorer. The
    # synthetic persona/drafts/GT come from the established builder fixture.
    for relative in (
        "harness/metrics.py", "harness/judge_pack.py", "harness/transcript.py",
        "data/personas/coverage.py",
    ):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    preferences = project / f"data/personas/{plan.persona_id}_preferences.yaml"
    preferences.write_text("fixture_persona: true\n")
    return project, plan


@pytest.fixture
def repeat_case(tmp_path, frozen_project):
    project, plan = frozen_project
    source = tmp_path / plan.run_id
    repeat = tmp_path / f"{plan.run_id}_repeat02"
    source.mkdir()
    repeat.mkdir()
    plan.dump_yaml(source / "run_plan.yaml")
    gt = project / f"Authoring/probe/outline/{plan.persona_id}/resolved_gt.yaml"
    shutil.copy2(gt, source / "resolved_gt.yaml")
    source_ledger = RunLedger(source, plan.run_id)
    source_ledger.steps = {
        step.step_id: StepRecord(status="done")
        for step in plan.steps if not step.placeholder
    }
    source_ledger.save()

    repeated = yaml.safe_load((source / "run_plan.yaml").read_text())
    repeated["run_id"] = repeat.name
    repeated["pa_session_key"] = plan.main_pa_session_key()
    pins = plan.pinned_step_ids()
    expected_pins = {}
    for step in plan.steps:
        if step.kind in PROBE_KINDS:
            expected_pins[step.step_id] = plan.probe_source_step_id(step, pins)
    for pin in set(expected_pins.values()):
        judge_audit.write_json(source / "checkpoints" / pin / "metadata.json", {
            "checkpoint_id": pin, "step_id": pin, "pinned": True, "terminal_status": "done",
        })
    for step in repeated["steps"]:
        if step["kind"] in PROBE_KINDS:
            step["borrow_start_from"] = f"{plan.run_id}/{expected_pins[step['step_id']]}"
    repeat_plan = RunPlan.model_validate(repeated)
    repeat_plan.dump_yaml(repeat / "run_plan.yaml")
    shutil.copy2(gt, repeat / "resolved_gt.yaml")
    new_ledger = RunLedger(repeat, repeat_plan.run_id)
    new_ledger.steps = {pid: StepRecord(status="done") for pid in PROBE_IDS}
    new_ledger.save()
    source_refs.write_repeat_source(repeat, source, 2)
    return SimpleNamespace(
        project=project, source=source, repeat=repeat, original=plan,
        plan=repeat_plan, expected_pins=expected_pins,
    )


def _read(path):
    return json.loads(path.read_text())


def _mutate_yaml(path, mutate):
    value = yaml.safe_load(path.read_text())
    mutate(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_full_plan_borrows_exact_pins_without_fabricating_new_timeline(repeat_case):
    case = repeat_case
    before = (case.repeat / "ledger.json").read_bytes()
    reference = source_refs.load_repeat_source(case.repeat)
    assert reference["repeat"] == 2
    assert reference["source_run_dir"] == str(case.source)
    assert case.plan.run_id != case.original.run_id
    assert case.plan.main_pa_session_key() == case.original.main_pa_session_key()
    assert len(set(case.expected_pins.values())) == 6
    assert len(case.plan.steps) == len(case.original.steps)
    assert {step.step_id: step.borrowed_checkpoint()[1] for step in case.plan.steps
            if step.kind in PROBE_KINDS} == case.expected_pins
    assert set(_read(case.repeat / "ledger.json")["steps"]) == set(PROBE_IDS)
    assert (case.repeat / "ledger.json").read_bytes() == before
    # Plan C1: the record is the source pointer only.
    assert reference == {"schema_version": 1, "repeat": 2, "source_run_dir": str(case.source)}
    assert source_refs.write_repeat_source(case.repeat, case.source, 2)
    assert (case.repeat / "ledger.json").read_bytes() == before


def test_original_scorer_completeness_uses_new_probes_and_source_timeline(repeat_case):
    from scripts.score_completed_run import require_complete

    case = repeat_case
    before = (case.repeat / "ledger.json").read_bytes()
    assert require_complete(case.repeat)["run_id"] == case.plan.run_id
    assert require_complete(case.source)["run_id"] == case.original.run_id
    assert (case.repeat / "ledger.json").read_bytes() == before
    assert not set(_read(case.repeat / "ledger.json")["steps"]) - set(PROBE_IDS)


@pytest.mark.parametrize("probe", ["pre_01", "fin_28"])
def test_original_finished_probes_cannot_fill_a_missing_new_probe(repeat_case, probe):
    case = repeat_case
    ledger_path = case.repeat / "ledger.json"
    ledger = _read(ledger_path)
    del ledger["steps"][probe]
    judge_audit.write_json(ledger_path, ledger)
    assert _read(case.source / "ledger.json")["steps"][probe]["status"] == "done"
    with pytest.raises(ValueError, match="not complete"):
        judge_audit.require_complete(case.repeat)


@pytest.mark.parametrize("kind", ["accumulation", "forced_event"])
def test_complete_new_probes_cannot_hide_failed_original_timeline(repeat_case, kind):
    case = repeat_case
    step = next(step for step in case.original.steps if step.kind == kind and not step.placeholder)
    path = case.source / "ledger.json"
    ledger = _read(path)
    ledger["steps"][step.step_id]["status"] = "failed"
    judge_audit.write_json(path, ledger)
    # Even a fabricated done marker on the new run must not override source history.
    new_ledger = _read(case.repeat / "ledger.json")
    new_ledger["steps"][step.step_id] = {"status": "done"}
    judge_audit.write_json(case.repeat / "ledger.json", new_ledger)
    with pytest.raises(ValueError, match="not complete"):
        judge_audit.require_complete(case.repeat)


def test_an_older_record_with_file_digests_and_a_rebind_still_loads(repeat_case):
    """D7: an older repeat's record also carries file digests, an acceptance
    reference and maybe a rebind; none of it is read or compared."""
    case = repeat_case
    stale = {"path": "/elsewhere/file", "sha256": "0" * 64}
    record = case.repeat / source_refs.NAME
    judge_audit.write_json(record, {
        "schema_version": 1, "repeat": 2, "source_run_dir": str(case.source),
        "acceptance": stale, "plan": stale, "ledger": stale, "gt": stale,
        "checkpoints": {"acc_014": stale},
    })
    judge_audit.write_json(case.repeat / "probe_repeat_source_rebind.json", {"to": {"ledger": stale}})
    reference = source_refs.load_repeat_source(case.repeat)
    assert reference["source_run_dir"] == str(case.source)
    assert judge_audit.require_complete(case.repeat)["run_id"] == case.plan.run_id


def _audited_new_scores(case):
    run = case.repeat
    judge_audit.write_json(run / "judge_input/index.json", {
        "run_id": case.plan.run_id, "probes": [{"probe_id": pid} for pid in PROBE_IDS],
    })
    for pid in PROBE_IDS:
        judge_audit.write_json(run / "steps" / pid / "transcript.json", {
            "run_id": case.plan.run_id, "probe_id": pid, "trajectory": "new independent run",
        })
        judge_audit.write_json(run / "judge_input" / pid / "text_judge_pack.json", {
            "run_id": case.plan.run_id, "probe_id": pid, "text": "new independent answer",
        })
    for pid in PROBE_IDS:
        target = run / "judge_output" / judge_audit.JUDGE_ID / f"{pid}.json"
        judge_audit.write_json(target, {
            "run_id": case.plan.run_id, "probe_id": pid, "judge_id": judge_audit.JUDGE_ID,
            "text_face": {"score": 4}, "ix_face": {"exact_match": True},
        })
    metrics = run / "metrics" / f"{judge_audit.JUDGE_ID}.json"
    judge_audit.write_json(metrics, {"E1_consistency": {"n": 28}})
    judge_audit.write_json(run / judge_audit.AUDIT_PATH, {
        "run_id": case.plan.run_id, "model": judge_audit.MODEL, "judge_id": judge_audit.JUDGE_ID,
    })


def test_all_new_probe_scores_count_for_the_repeat(repeat_case):
    _audited_new_scores(repeat_case)
    assert judge_audit.checked_outputs(repeat_case.repeat) == (set(PROBE_IDS), True, [])


def test_a_copied_original_probe_score_does_not_count(repeat_case):
    case = repeat_case
    _audited_new_scores(case)
    pid = "fin_01"
    old = case.source / "judge_output" / judge_audit.JUDGE_ID / f"{pid}.json"
    judge_audit.write_json(old, {
        "run_id": case.original.run_id, "probe_id": pid, "judge_id": judge_audit.JUDGE_ID,
        "text_face": {"score": 4}, "ix_face": {"exact_match": True},
    })
    copied = case.repeat / "judge_output" / judge_audit.JUDGE_ID / f"{pid}.json"
    shutil.copy2(old, copied)
    valid, metrics_ok, errors = judge_audit.checked_outputs(case.repeat)
    assert valid == set(PROBE_IDS) - {pid}
    assert not metrics_ok and any("identity differs" in error for error in errors)


@pytest.mark.parametrize("change", ["source_current_step", "source_recovery_pending"])
def test_an_unfinished_source_is_rejected(repeat_case, change):
    case = repeat_case
    if change == "source_current_step":
        path = case.source / "ledger.json"
        ledger = _read(path)
        ledger["current_step"] = "acc_001"
        judge_audit.write_json(path, ledger)
    else:
        judge_audit.write_json(case.source / "recovery/pending.json", {"incomplete": True})
    with pytest.raises(ValueError):
        judge_audit.require_complete(case.repeat)


def test_reference_conditions_are_the_two_the_orchestrator_binds_itself():
    from harness import orchestrator
    assert source_refs.REFERENCE_CONDITIONS == {
        orchestrator.NO_MEMORY_CONDITION, orchestrator.ORACLE_PREFERENCE_CONDITION}


@pytest.fixture(params=["no_memory", "oracle_preference"])
def arm_case(request, repeat_case):
    """Round 1 of a reference arm on the same source and pins, prepared the way
    `prepare_probe_variance` writes it."""
    case = repeat_case
    arm = {"memory_condition": request.param, "pa_model": "qwen/qwen3.8-27b"}
    original = yaml.safe_load((case.source / "run_plan.yaml").read_text())
    planned = source_refs.repeat_base_plan(original, arm)
    run = case.source.parent / source_refs.repeat_run_id(case.original.run_id, arm, 1)
    run.mkdir()
    planned["run_id"] = run.name
    planned["pa_session_key"] = case.original.main_pa_session_key()
    for step in planned["steps"]:
        if step["kind"] in PROBE_KINDS:
            step["borrow_start_from"] = f"{case.original.run_id}/{case.expected_pins[step['step_id']]}"
    RunPlan.model_validate(planned).dump_yaml(run / "run_plan.yaml")
    shutil.copy2(case.source / "resolved_gt.yaml", run / "resolved_gt.yaml")
    ledger = RunLedger(run, planned["run_id"])
    ledger.steps = {pid: StepRecord(status="done") for pid in PROBE_IDS}
    ledger.save()
    source_refs.write_repeat_source(run, case.source, 1, arm)
    return SimpleNamespace(case=case, arm=arm, run=run, plan=planned)


def test_a_reference_arm_round_loads_and_reaches_the_scorer(arm_case):
    reference = source_refs.load_repeat_source(arm_case.run)
    assert reference == {"schema_version": 1, "repeat": 1, "source_run_dir": str(arm_case.case.source),
                         "arm": arm_case.arm}
    assert judge_audit.require_complete(arm_case.run)["memory_condition"] == arm_case.arm["memory_condition"]
    assert arm_case.case.original.consolidation_model and not arm_case.plan["consolidation_model"]


@pytest.mark.parametrize("change", ["drop_arm", "repeat_04"])
def test_a_reference_arm_record_must_name_its_arm_and_a_round_of_it(arm_case, change):
    record = arm_case.run / source_refs.NAME
    value = _read(record)
    if change == "drop_arm":
        del value["arm"]  # round 1 of a memory repeat is its source, never a repeat
    else:
        value["repeat"] = 4
    judge_audit.write_json(record, value)
    with pytest.raises(ValueError):
        source_refs.load_repeat_source(arm_case.run)
