"""Preparation uses real frozen RunPlans, without services."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
import yaml

from harness.probe_repeat_source import load_repeat_source
from harness.run_ledger import RunLedger
from harness.run_plan import RunPlan
from scripts.prepare_probe_variance import prepare
from scripts.run_probe_variance import parse_repeat, validate_manifest
from test_harness.test_run_plan import _plan, _memos_plan
from test_harness.test_orchestrator import _plan as _probe_template_plan


def _source(tmp_path, monkeypatch, backend="mem0"):
    raw = _memos_plan() if backend == "memos" else _plan()
    raw["run_id"] = f"original_{backend}"
    source = tmp_path / raw["run_id"]
    source.mkdir(parents=True)
    acc = raw["steps"][0]
    probe = _probe_template_plan().steps[1].model_dump(mode="json")
    steps = []
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for index, pin in enumerate((1, 20, 40, 60, 80)):
        steps.append({**acc, "step_id": f"acc_{pin:03d}", "acc_num": pin,
                      "benchmark_session_timestamp": (start + timedelta(days=index * 2)).isoformat()})
        pid = f"pre_{index + 1:02d}"
        steps.append({**probe, "step_id": pid, "script_path": f"{pid}.yaml",
                      "before_acc_num": pin + 1,
                      "probe_overlay": {"owner_probe_id": pid},
                      "benchmark_session_timestamp": (start + timedelta(days=index * 2 + 1)).isoformat()})
    steps.append({**acc, "step_id": "acc_108", "acc_num": 108,
                  "benchmark_session_timestamp": (start + timedelta(days=10)).isoformat()})
    for index in range(1, 29):
        pid = f"fin_{index:02d}"
        steps.append({**probe, "step_id": pid, "kind": "final_probe", "before_acc_num": None,
                      "script_path": f"{pid}.yaml", "probe_overlay": {"owner_probe_id": pid},
                      "benchmark_session_timestamp": (start + timedelta(days=11)).isoformat()})
    raw["steps"] = steps
    plan = RunPlan.model_validate(raw)
    plan.dump_yaml(source / "run_plan.yaml")
    ledger = RunLedger(source, plan.run_id)
    for step in plan.steps:
        ledger.mark_done(step.step_id)
    (source / "resolved_gt.yaml").write_text("frozen: 原始GT\n")
    for pin in plan.pinned_step_ids():
        path = source / "checkpoints" / pin
        path.mkdir(parents=True)
        (path / "metadata.json").write_text(json.dumps({"pinned": True, "checkpoint_id": pin}))
    return {"id": backend, "source": {"run_dir": str(source)}}


def _prepare(tmp_path, settings):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"settings": settings}))
    return prepare(path, workspace_root=tmp_path / "workspace", state_root=tmp_path / "state",
                   output=tmp_path / "dispatch.json")


@pytest.mark.parametrize("backend", ["mem0", "memos"])
def test_prepare_preserves_original_plan_gt_and_exact_six_pins(tmp_path, monkeypatch, backend):
    setting = _source(tmp_path, monkeypatch, backend)
    source = Path(setting["source"]["run_dir"])
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    result = _prepare(tmp_path, [setting])
    original = yaml.safe_load((source / "run_plan.yaml").read_text())
    plans = []
    for row in result["settings"][0]["repeats"]:
        args, path, repeated = parse_repeat(row)
        plans.append(repeated)
        assert repeated["pa_session_key"] == f"{original['run_id']}_main"
        assert Path(args.borrowed_runs_root) == source.parent
        expected = deepcopy(original)
        expected["run_id"] = repeated["run_id"]
        expected["pa_session_key"] = repeated["pa_session_key"]
        expected_pins = dict(zip((f"pre_{i:02d}" for i in range(1, 6)), ("acc_001", "acc_020", "acc_040", "acc_060", "acc_080")))
        for old, new in zip(expected["steps"], repeated["steps"]):
            if old["kind"] in {"pre_event_probe", "final_probe"}:
                pin = expected_pins.get(old["step_id"], "acc_108")
                old["borrow_start_from"] = f"{original['run_id']}/{pin}"
                assert new["borrow_start_from"] == old["borrow_start_from"]
        assert repeated == expected
        assert (path.parent / "resolved_gt.yaml").read_bytes() == before[source / "resolved_gt.yaml"]
        proof = load_repeat_source(path.parent)
        assert proof == {"schema_version": 1, "repeat": row["repeat"], "source_run_dir": str(source)}
    assert len({plan["run_id"] for plan in plans}) == 2
    assert all(path.read_bytes() == content for path, content in before.items())
    assert _prepare(tmp_path, [setting]) == result


def test_mem0_and_memos_settings_keep_shared_service_endpoints_and_distinct_runs(tmp_path, monkeypatch):
    settings = [_source(tmp_path, monkeypatch, backend) for backend in ("mem0", "memos")]
    result = _prepare(tmp_path, settings)
    runs = []
    for setting in result["settings"]:
        original = yaml.safe_load((Path(setting["source"]["run_dir"]) / "run_plan.yaml").read_text())
        for repeat in setting["repeats"]:
            _, path, repeated = parse_repeat(repeat)
            runs.append(path.parent)
            for name in ("qdrant_host", "qdrant_port", "memos_runtime_config", "mem0_runtime_config"):
                assert repeated[name] == original[name]
    assert len(set(runs)) == 4


@pytest.mark.parametrize("changed", ["source_gt", "source_ledger", "repeat_gt", "pin"])
def test_a_repeat_that_cannot_run_from_its_source_is_refused(tmp_path, monkeypatch, changed):
    """GT copies that differ, an unreadable source ledger or a lost pin; file
    digests themselves are not compared (see the next test)."""
    setting = _source(tmp_path, monkeypatch)
    result = _prepare(tmp_path, [setting])
    source = Path(setting["source"]["run_dir"])
    _, repeat, _ = parse_repeat(result["settings"][0]["repeats"][0])
    if changed == "pin":
        (source / "checkpoints/acc_001/metadata.json").write_text('{"pinned":false}')
    else:
        path = {"source_gt": source / "resolved_gt.yaml", "source_ledger": source / "ledger.json",
                "repeat_gt": repeat.parent / "resolved_gt.yaml"}[changed]
        path.write_text(path.read_text() + "\n# changed evidence\n")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        validate_manifest(tmp_path / "dispatch.json")


def test_source_and_repeat_plan_bytes_are_not_compared(tmp_path, monkeypatch):
    setting = _source(tmp_path, monkeypatch)
    result = _prepare(tmp_path, [setting])
    source = Path(setting["source"]["run_dir"])
    _, repeat, _ = parse_repeat(result["settings"][0]["repeats"][0])
    for path in (source / "run_plan.yaml", repeat):
        path.write_text(path.read_text() + "\n# an edit that changes bytes, not content\n")
    assert validate_manifest(tmp_path / "dispatch.json")["settings"][0]["id"] == setting["id"]


def test_existing_different_repeat_plan_is_never_overwritten(tmp_path, monkeypatch):
    setting = _source(tmp_path, monkeypatch)
    result = _prepare(tmp_path, [setting])
    _, path, plan = parse_repeat(result["settings"][0]["repeats"][0])
    plan["pa_max_tokens"] += 1
    path.write_text(yaml.safe_dump(plan))
    changed = path.read_bytes()
    with pytest.raises(ValueError, match="Repeat changed frozen scientific fields"):
        _prepare(tmp_path, [setting])
    assert path.read_bytes() == changed


@pytest.mark.parametrize("change", ["wrong_pin", "changed_target", "changed_session", "same_run_id"])
def test_validation_refuses_a_repeat_plan_that_departs_from_its_source(tmp_path, monkeypatch, change):
    setting = _source(tmp_path, monkeypatch)
    result = _prepare(tmp_path, [setting])
    source = Path(setting["source"]["run_dir"])
    _, path, plan = parse_repeat(result["settings"][0]["repeats"][0])
    probe = next(step for step in plan["steps"] if step["step_id"] == "pre_01")
    if change == "wrong_pin":
        probe["borrow_start_from"] = f"{source.name}/acc_108"  # a pinned checkpoint, not pre_01's
    elif change == "changed_target":
        probe["target_setting"] = "Different scientific GT setting"
    elif change == "changed_session":
        plan["pa_session_key"] = f"{plan['run_id']}_main"
    else:
        plan["run_id"] = source.name
    path.write_text(yaml.safe_dump(plan, sort_keys=False))
    with pytest.raises(ValueError):
        validate_manifest(tmp_path / "dispatch.json")


# ── reference arms: no_memory / oracle_preference on a memory source's pins ──

ARM_CLEARED = {"consolidation_model": "", "consolidation_request_policy": None,
               "memory_side_model": "", "memory_side_request_policy": None,
               "embedding_provider": "", "embedding_model": "", "embedding_dimensions": 0,
               "qdrant_host": "", "qdrant_port": 0, "mem0_runtime_config": None,
               "memos_runtime_config": None, "memos_cube_id": ""}


@pytest.mark.parametrize("backend", ["mem0", "memos"])
@pytest.mark.parametrize("condition", ["no_memory", "oracle_preference"])
def test_a_reference_arm_runs_its_own_condition_and_model_on_the_source_pins(tmp_path, monkeypatch, backend, condition):
    from pa.providers.request_policy import resolve_model_request_policy
    setting = _source(tmp_path, monkeypatch, backend)
    arm = {"memory_condition": condition, "pa_model": "qwen/qwen3.8-27b"}
    source = Path(setting["source"]["run_dir"])
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    result = _prepare(tmp_path, [{**setting, "arm": arm}])
    row = result["settings"][0]
    assert row["arm"] == arm and "repeat01" not in row
    original = yaml.safe_load((source / "run_plan.yaml").read_text())
    policy = resolve_model_request_policy("openrouter", arm["pa_model"])
    run_ids = []
    for repeat in row["repeats"]:
        args, path, plan = parse_repeat(repeat)
        run_ids.append(plan["run_id"])
        assert plan["run_id"] == f"{original['run_id']}_{condition}_qwen_qwen3.8-27b_probe_repeat{repeat['repeat']:02d}"
        assert (args.memory_condition, args.pa_model) == (condition, arm["pa_model"])
        assert not {"--consolidation-model", "--memory-side-model", "--memos-api-base"} & set(repeat["argv"])
        assert (plan["memory_condition"], plan["pa_model"]) == (condition, arm["pa_model"])
        assert plan["pa_request_policy"] == policy.snapshot()
        assert {name: plan[name] for name in ARM_CLEARED} == ARM_CLEARED
        assert plan["pa_session_key"] == f"{original['run_id']}_main"
        kept = {name for name in original if name not in ARM_CLEARED} - {
            "run_id", "memory_condition", "pa_model", "pa_request_policy", "steps"}
        assert {name: plan[name] for name in kept} == {name: original[name] for name in kept}
        probes = [step for step in plan["steps"] if step["kind"] in {"pre_event_probe", "final_probe"}]
        assert len(probes) == 33 and all(step["borrow_start_from"].startswith(original["run_id"] + "/")
                                         for step in probes)
        assert len(plan["steps"]) == len(original["steps"])
        assert load_repeat_source(path.parent) == {"schema_version": 1, "repeat": repeat["repeat"],
                                                   "source_run_dir": str(source), "arm": arm}
    # Every round is new: the source is another condition's run, so it is no round of the arm.
    assert [r["repeat"] for r in row["repeats"]] == [1, 2, 3] and len(set(run_ids)) == 3
    assert all(path.read_bytes() == content for path, content in before.items())
    assert _prepare(tmp_path, [{**setting, "arm": arm}]) == result


def test_both_arms_and_the_memory_repeat_of_one_source_are_distinct_settings(tmp_path, monkeypatch):
    setting = _source(tmp_path, monkeypatch, "mem0")
    settings = [setting] + [{**setting, "id": f"{condition}_{model}", "arm": {"memory_condition": condition, "pa_model": model}}
                            for condition in ("no_memory", "oracle_preference")
                            for model in ("qwen/qwen3.8-27b", "tencent/hy4-preview")]
    result = _prepare(tmp_path, settings)
    from scripts.run_probe_variance import key
    assert len({key(row) for row in result["settings"]}) == 5
    plans = [parse_repeat(repeat)[1] for row in result["settings"] for repeat in row["repeats"]]
    assert len(set(plans)) == 2 + 4 * 3


@pytest.mark.parametrize("change", ["memory_side_model", "memory_condition", "target_setting", "record_arm"])
def test_validation_accepts_only_the_rewritten_arm_plan(tmp_path, monkeypatch, change):
    setting = _source(tmp_path, monkeypatch, "mem0")
    result = _prepare(tmp_path, [{**setting, "arm": {"memory_condition": "no_memory", "pa_model": "qwen/qwen3.8-27b"}}])
    _, path, plan = parse_repeat(result["settings"][0]["repeats"][0])
    if change == "record_arm":
        record = json.loads((path.parent / "probe_repeat_source.json").read_text())
        record["arm"]["memory_condition"] = "oracle_preference"
        (path.parent / "probe_repeat_source.json").write_text(json.dumps(record))
    else:
        if change == "target_setting":
            next(step for step in plan["steps"] if step["step_id"] == "pre_01")["target_setting"] = "Other"
        else:
            plan[change] = {"memory_side_model": "openai/gpt-5.5",
                            "memory_condition": "oracle_preference"}[change]
        path.write_text(yaml.safe_dump(plan, sort_keys=False))
    with pytest.raises(ValueError):
        validate_manifest(tmp_path / "dispatch.json")


@pytest.mark.parametrize("arm", [{"memory_condition": "mem0", "pa_model": "qwen/qwen3.8-27b"},
                                 {"memory_condition": "no_memory"},
                                 {"memory_condition": "no_memory", "pa_model": "qwen/qwen3.8-27b", "rounds": [1]}])
def test_an_arm_is_exactly_a_reference_condition_and_a_pa_model(tmp_path, monkeypatch, arm):
    setting = _source(tmp_path, monkeypatch, "mem0")
    with pytest.raises(ValueError, match="reference arm"):
        _prepare(tmp_path, [{**setting, "arm": arm}])


def test_an_hy_arm_runs_on_siliconflow(tmp_path, monkeypatch):
    from harness.pa_factory import AssistantLLMProviderProfile, build_pa_from_provider_profile
    from harness.run_plan import load_run_plan
    setting = _source(tmp_path, monkeypatch, "mem0")
    result = _prepare(tmp_path, [{**setting, "arm": {"memory_condition": "no_memory", "pa_model": "tencent/hy4-preview"}}])
    for repeat in result["settings"][0]["repeats"]:
        args, path, _ = parse_repeat(repeat)
        plan = load_run_plan(path, validate_paths=False)
        workspace = path.parent / "pa_workspace" / plan.persona_id
        workspace.mkdir(parents=True)
        loop = build_pa_from_provider_profile(
            provider_profile=AssistantLLMProviderProfile(provider_name="openrouter", api_key="test-key"),
            workspace=workspace, pa_model=args.pa_model, pa_max_tokens=plan.pa_max_tokens)
        assert plan.pa_model == args.pa_model == "tencent/hy4-preview"
        assert loop.pa_llm_provider.extra_body == {"provider": {"order": ["SiliconFlow"], "allow_fallbacks": False}}
