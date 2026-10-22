"""`run_info.yaml` gets one record per event and is a single YAML list."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from harness.run_info import RUN_INFO_NAME, memos_health_record, plan_scope_record, write_run_info

REPO = Path(__file__).resolve().parents[2]


class _Plan(SimpleNamespace):
    def model_dump(self, *, mode, exclude):
        return {k: v for k, v in vars(self).items() if k not in exclude}


def _plan(tmp_path: Path) -> _Plan:
    script = tmp_path / "acc_001.yaml"
    script.write_text("turns: []\n")
    return _Plan(
        run_id="r1", persona_id="user_x", pa_model="m", memory_condition="no_memory",
        memos_cube_id="",
        steps=[
            SimpleNamespace(placeholder=False, script_path=str(script)),
            SimpleNamespace(placeholder=True, script_path=""),
        ],
    )


def test_each_launch_appends_one_record(tmp_path):
    plan = _plan(tmp_path)
    run_dir = tmp_path / "run"
    write_run_info(plan, run_dir, REPO, event="freeze")
    path = write_run_info(plan, run_dir, REPO, event="launch", extra={"launch_id": "abc"})
    records = yaml.safe_load(path.read_text())
    assert path.name == RUN_INFO_NAME
    assert [r["event"] for r in records] == ["freeze", "launch"]
    last = records[-1]
    assert len(last["code"]["commit"]) == 40
    assert isinstance(last["code"]["dirty"], bool)
    assert last["config"]["pa_model"] == "m"
    assert "memos_cube_id" not in last["config"]
    assert last["launch_id"] == "abc"


def test_the_pa_model_provider_routing_rows_are_recorded(tmp_path):
    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    (root / "configs/model_request_policies.yaml").write_text(yaml.safe_dump({"provider_routing": [
        {"provider_name": "openrouter", "model": "m", "order": ["SiliconFlow"], "allow_fallbacks": False},
        {"provider_name": "openrouter", "model": "other", "order": ["X"]},
    ]}))
    path = write_run_info(_plan(tmp_path), tmp_path / "run", root, event="launch")
    record = yaml.safe_load(path.read_text())[0]
    assert record["pa_provider_routing"] == [
        {"provider_name": "openrouter", "model": "m", "order": ["SiliconFlow"], "allow_fallbacks": False}]


def test_the_memos_health_record_keeps_version_patch_and_contract():
    health = {"status": "healthy", "service": "memos", "version": "2.0.3",
              "mempa_patch_version": "p7", "runtime_contract": {"qdrant": "v1.15.3"}, "dependencies": {}}
    assert memos_health_record(health) == {
        "version": "2.0.3", "mempa_patch_version": "p7", "runtime_contract": {"qdrant": "v1.15.3"}}


def test_outside_a_checkout_the_commit_is_unknown(tmp_path):
    path = write_run_info(_plan(tmp_path), tmp_path / "run", tmp_path, event="launch")
    record = yaml.safe_load(path.read_text())[0]
    assert record["code"] == {"commit": None, "dirty": None}


def test_plan_scope_record_names_the_accs_and_the_probes():
    """The freeze record says which accs ran and which probes came in with them;
    a gap in the accs shows as a gap."""
    def step(step_id, kind, acc_num=None):
        return SimpleNamespace(step_id=step_id, kind=kind, acc_num=acc_num)

    plan = SimpleNamespace(steps=[
        step("acc_013", "accumulation", 13),
        step("acc_014", "accumulation", 14),
        step("pre_01", "pre_event_probe"),
        step("evolv_01", "forced_event", 15),
        step("acc_020", "accumulation", 20),
        step("fin_01", "final_probe"),
    ])
    assert plan_scope_record(plan) == {
        "acc_range": "13-15,20",
        "probe_step_ids": ["pre_01", "fin_01"],
    }
    assert plan_scope_record(SimpleNamespace(steps=[step("acc_001", "accumulation", 1)])) == {
        "acc_range": "1", "probe_step_ids": [],
    }
