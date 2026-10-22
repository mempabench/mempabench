"""Within one run, an OpenRouter upstream that failed one model
`providers.avoid_after_failures` times gets no more of that model's requests.

The state is `<run_dir>/provider_avoidance.json` (`harness/provider_avoidance.py`);
the requests and the failure hooks are covered in
`tests/test_pa/test_provider_avoidance_requests.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from harness import run_config, run_plan_builder
from harness.pa_factory import (
    AssistantLLMProviderProfile,
    build_pa_from_provider_profile,
)
from harness.provider_avoidance import STATE_FILENAME, ProviderAvoidance
from harness.run_config import RunConfig, load_run_config
from harness.run_plan import load_run_plan
from scripts import run_orchestrator

MODEL = "deepseek/deepseek-v4-flash"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: user_a x nanobot_file_memory, acc 1, frozen by `scripts.diagnostics.mint_run_plan_only`
#: at ce1de67, before the setting existed.
OLD_PLAN = PROJECT_ROOT / "tests/fixtures/run_plan_ce1de67_user_a_file_memory.yaml"


def _fail(avoidance: ProviderAvoidance, provider: str, model: str = MODEL, *,
          kind: str = "contract_rejected", failure: str = "file_consolidation: Missing save_memory fields") -> None:
    avoidance.record_failure(model, provider, kind=kind, failure=failure)


def test_the_third_failure_of_one_model_on_one_provider_ignores_that_provider(tmp_path, capsys):
    avoidance = ProviderAvoidance(tmp_path, threshold=3)
    _fail(avoidance, "DigitalOcean")
    _fail(avoidance, "DigitalOcean", kind="provider_error", failure="InternalServerError status=502")
    # Other providers and other models keep their own counts.
    _fail(avoidance, "Baidu")
    _fail(avoidance, "DigitalOcean", "z-ai/glm-5.3-flash")
    assert avoidance.ignored_providers(MODEL) == []
    assert "provider_avoided" not in capsys.readouterr().err

    _fail(avoidance, "DigitalOcean", kind="provider_truncation", failure="finish_reason=length, empty")

    assert avoidance.ignored_providers(MODEL) == ["DigitalOcean"]
    assert avoidance.ignored_providers("z-ai/glm-5.3-flash") == []
    assert capsys.readouterr().err.splitlines() == [
        f"provider_avoided model={MODEL} provider=DigitalOcean failures=3"
    ]
    # A request already in flight can still fail there: it is counted, and the
    # provider is not announced a second time.
    _fail(avoidance, "DigitalOcean")
    assert "provider_avoided" not in capsys.readouterr().err
    state = json.loads((tmp_path / STATE_FILENAME).read_text())
    assert state["models"][MODEL]["failures"] == {"Baidu": 1, "DigitalOcean": 4}


def test_a_resume_loads_the_counts_and_the_ignored_providers(tmp_path):
    first_launch = ProviderAvoidance(tmp_path, threshold=3)
    for _ in range(3):
        _fail(first_launch, "DigitalOcean")
    _fail(first_launch, "Baidu", kind="provider_error", failure="RateLimitError status=429")

    resumed = ProviderAvoidance(tmp_path, threshold=3)

    assert resumed.ignored_providers(MODEL) == ["DigitalOcean"]
    _fail(resumed, "Baidu")
    _fail(resumed, "Baidu")
    assert resumed.ignored_providers(MODEL) == ["Baidu", "DigitalOcean"]
    state = json.loads((tmp_path / STATE_FILENAME).read_text())
    assert state["schema"] == "mempa_provider_avoidance/1"
    assert state["models"][MODEL]["failures"] == {"Baidu": 3, "DigitalOcean": 3}
    ignored = state["models"][MODEL]["ignored"]
    assert ignored["DigitalOcean"]["failure_kind"] == "contract_rejected"
    assert ignored["DigitalOcean"]["failure"] == "file_consolidation: Missing save_memory fields"
    assert datetime.fromisoformat(ignored["DigitalOcean"]["at"]).tzinfo is not None
    assert ignored["Baidu"]["failure_kind"] == "contract_rejected"


def _valid_state() -> dict:
    return {"schema": "mempa_provider_avoidance/1", "models": {MODEL: {
        "failures": {"DigitalOcean": 3},
        "ignored": {"DigitalOcean": {"at": "2026-10-19T08:00:00+00:00",
                                     "failure_kind": "provider_error", "failure": "x"}},
    }}}


def _broken(**changes) -> str:
    state = _valid_state()
    entry = state["models"][MODEL]
    for key, value in changes.items():
        if key == "schema":
            state["schema"] = value
        elif key == "count":
            entry["failures"]["DigitalOcean"] = value
        elif key == "ignored_without_count":
            entry["failures"] = {}
        elif key == "kind":
            entry["ignored"]["DigitalOcean"]["failure_kind"] = value
        elif key == "extra":
            entry["note"] = value
    return json.dumps(state)


@pytest.mark.parametrize("text", [
    "{not json",
    "[]",
    _broken(schema="mempa_provider_avoidance/0"),
    _broken(count="3"),
    _broken(count=0),
    _broken(ignored_without_count=True),
    _broken(kind="timeout"),
    _broken(extra="unknown key"),
])
def test_an_invalid_state_file_stops_the_launch_and_stays_as_it_is(tmp_path, text):
    path = tmp_path / STATE_FILENAME
    path.write_text(text)

    with pytest.raises(ValueError, match=r"provider_avoidance\.json is invalid"):
        ProviderAvoidance(tmp_path, threshold=3)

    assert path.read_text() == text


def test_a_state_file_broken_mid_run_fails_the_next_request(tmp_path):
    avoidance = ProviderAvoidance(tmp_path, threshold=3)
    (tmp_path / STATE_FILENAME).write_text(json.dumps(_valid_state()))
    assert avoidance.ignored_providers(MODEL) == ["DigitalOcean"]
    (tmp_path / STATE_FILENAME).write_text("{")
    with pytest.raises(ValueError, match="is invalid"):
        avoidance.ignored_providers(MODEL)
    with pytest.raises(ValueError, match="is invalid"):
        _fail(avoidance, "Baidu")


def test_threshold_zero_turns_the_feature_off(tmp_path):
    (tmp_path / STATE_FILENAME).write_text(json.dumps(_valid_state()))
    avoidance = ProviderAvoidance(tmp_path, threshold=0)
    for _ in range(5):
        _fail(avoidance, "Baidu")
    assert avoidance.ignored_providers(MODEL) == []
    assert json.loads((tmp_path / STATE_FILENAME).read_text()) == _valid_state()

    other_run = tmp_path / "other_run"
    other_run.mkdir()
    off = ProviderAvoidance(other_run, threshold=0)
    _fail(off, "Baidu")
    assert not (other_run / STATE_FILENAME).exists()


@pytest.mark.parametrize("threshold", [-1, True, 2.0])
def test_the_threshold_is_a_non_negative_integer(tmp_path, threshold):
    with pytest.raises(ValueError, match="threshold"):
        ProviderAvoidance(tmp_path, threshold=threshold)


_WORKER = """
import sys, time
from pathlib import Path
from harness.provider_avoidance import ProviderAvoidance
run_dir, name, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
avoidance = ProviderAvoidance(run_dir, threshold=100000)
(run_dir / f"ready_{name}").touch()
while not (run_dir / "go").exists():
    time.sleep(0.001)
for _ in range(count):
    avoidance.record_failure("m", "DigitalOcean", kind="provider_error", failure=name)
"""


def test_two_processes_counting_at_once_lose_no_failure(tmp_path):
    """Concurrent probes share the file: every update is read-modify-write under the lock."""
    workers = [
        subprocess.Popen([sys.executable, "-c", _WORKER, str(tmp_path), name, "150"], cwd=PROJECT_ROOT)
        for name in ("a", "b")
    ]
    deadline = time.monotonic() + 120
    while not all((tmp_path / f"ready_{name}").exists() for name in ("a", "b")):
        assert time.monotonic() < deadline and all(w.poll() is None for w in workers)
        time.sleep(0.01)
    (tmp_path / "go").touch()
    assert [worker.wait(timeout=120) for worker in workers] == [0, 0]

    state = json.loads((tmp_path / STATE_FILENAME).read_text())
    assert state["models"]["m"]["failures"] == {"DigitalOcean": 300}


def test_two_threads_counting_at_once_lose_no_failure(tmp_path):
    """Mem0 extraction counts from its own thread while the PA counts from the event loop."""
    avoidance = ProviderAvoidance(tmp_path, threshold=100_000)
    start = threading.Barrier(2)

    def count(name: str) -> None:
        start.wait()
        for _ in range(150):
            avoidance.record_failure("m", "DigitalOcean", kind="provider_error", failure=name)

    threads = [threading.Thread(target=count, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    state = json.loads((tmp_path / STATE_FILENAME).read_text())
    assert state["models"]["m"]["failures"] == {"DigitalOcean": 300}


def test_deleting_the_state_file_before_a_resume_clears_the_skipped_providers(tmp_path):
    """A missing file is a fresh start with nothing counted or ignored."""
    first_launch = ProviderAvoidance(tmp_path, threshold=3)
    for _ in range(3):
        _fail(first_launch, "DigitalOcean")
    assert first_launch.ignored_providers(MODEL) == ["DigitalOcean"]

    (tmp_path / STATE_FILENAME).unlink()
    resumed = ProviderAvoidance(tmp_path, threshold=3)

    assert resumed.ignored_providers(MODEL) == []
    _fail(resumed, "DigitalOcean")
    state = json.loads((tmp_path / STATE_FILENAME).read_text())
    assert state["models"][MODEL] == {"failures": {"DigitalOcean": 1}, "ignored": {}}


# ── run config and plan ───────────────────────────────────────────────────


def test_the_threshold_defaults_to_three_and_zero_is_allowed(tmp_path):
    assert RunConfig().providers.avoid_after_failures == 3
    path = tmp_path / "run_config.yaml"
    path.write_text(yaml.safe_dump({"providers": {"avoid_after_failures": 0}}))
    assert load_run_config(path).providers.avoid_after_failures == 0
    path.write_text(yaml.safe_dump({"providers": {"avoid_after_failures": -1}}))
    with pytest.raises(ValueError):
        load_run_config(path)


def _file_memory_plan(tmp_path: Path, monkeypatch, threshold: int | None):
    if threshold is not None:
        path = tmp_path / f"run_config_{threshold}.yaml"
        path.write_text(yaml.safe_dump({"providers": {"avoid_after_failures": threshold}}))
        monkeypatch.setenv(run_config.RUN_CONFIG_ENV, str(path))
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="avoid_plan", persona_id="user_a", memory_condition="nanobot_file_memory",
        pa_model=MODEL, script_names=["acc_001"], consolidation_model=MODEL,
    )
    monkeypatch.delenv(run_config.RUN_CONFIG_ENV, raising=False)
    return plan


def test_a_new_plan_freezes_the_threshold(tmp_path, monkeypatch):
    default_plan = _file_memory_plan(tmp_path, monkeypatch, None)
    off_plan = _file_memory_plan(tmp_path, monkeypatch, 0)

    assert default_plan.provider_avoid_after_failures == 3
    assert off_plan.provider_avoid_after_failures == 0
    dumped = yaml.safe_load(off_plan.dump_yaml(tmp_path / "off" / "run_plan.yaml").read_text())
    assert dumped["provider_avoid_after_failures"] == 0


def test_a_plan_frozen_by_ce1de67_loads_with_avoidance_off():
    assert "provider_avoid_after_failures" not in OLD_PLAN.read_text()

    plan = load_run_plan(OLD_PLAN, project_root=PROJECT_ROOT, validate_paths=False)

    assert plan.provider_avoid_after_failures is None


# ── wiring ────────────────────────────────────────────────────────────────


def test_a_probe_branch_pa_and_the_consolidation_provider_share_the_run_state(tmp_path):
    run_dir = tmp_path / "run"
    branch_workspace = run_dir / "branches/fin_01/1/pa_workspace/user_a"
    branch_workspace.mkdir(parents=True)
    avoidance = ProviderAvoidance(run_dir, threshold=3)
    profile = AssistantLLMProviderProfile(
        provider_name="openrouter", api_key="test-key", provider_avoidance=avoidance,
    )

    branch_pa = build_pa_from_provider_profile(
        provider_profile=profile, workspace=branch_workspace, pa_model=MODEL, pa_max_tokens=1024,
    ).pa_llm_provider
    consolidation = profile.create_provider(model=MODEL, role="consolidation")

    assert branch_pa.provider_avoidance is avoidance
    assert consolidation.provider_avoidance is avoidance
    assert avoidance.path == run_dir / STATE_FILENAME


class _StopAtConsolidationProvider(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_a_launch_binds_the_run_state_with_the_plans_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    plan = _file_memory_plan(tmp_path, monkeypatch, 5)
    run_dir = tmp_path / "workspace/runs/m/c/user_a/avoid_plan"
    plan_path = plan.dump_yaml(run_dir / "run_plan.yaml")
    seen: list[ProviderAvoidance | None] = []

    def create_provider(self, *, model, role, **_kwargs):
        seen.append(self.provider_avoidance)
        raise _StopAtConsolidationProvider(role)

    monkeypatch.setattr(AssistantLLMProviderProfile, "create_provider", create_provider)
    args = SimpleNamespace(
        verbose=True, plan=str(plan_path), persona_id="user_a",
        memory_condition="nanobot_file_memory", pa_model=MODEL,
        consolidation_model=MODEL, memory_side_model=None, memos_api_base=None,
    )

    with pytest.raises(_StopAtConsolidationProvider, match="consolidation"):
        await run_orchestrator._execute_benchmark_run(args)

    [avoidance] = seen
    assert avoidance.path == run_dir / STATE_FILENAME
    assert avoidance.threshold == 5


@pytest.mark.asyncio
async def test_a_launch_of_a_plan_frozen_without_the_setting_never_touches_the_state(tmp_path, monkeypatch):
    """Off for that run: no state object, and the file is neither read nor written."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    plan = _file_memory_plan(tmp_path, monkeypatch, None)
    run_dir = tmp_path / "workspace/runs/m/c/user_a/avoid_plan"
    plan_path = plan.dump_yaml(run_dir / "run_plan.yaml")
    data = yaml.safe_load(plan_path.read_text())
    del data["provider_avoid_after_failures"]
    plan_path.write_text(yaml.safe_dump(data, sort_keys=False))
    # Would stop any launch that read it.
    (run_dir / STATE_FILENAME).write_text("{not json")
    seen: list[ProviderAvoidance | None] = []

    def create_provider(self, *, model, role, **_kwargs):
        seen.append(self.provider_avoidance)
        raise _StopAtConsolidationProvider(role)

    monkeypatch.setattr(AssistantLLMProviderProfile, "create_provider", create_provider)
    args = SimpleNamespace(
        verbose=True, plan=str(plan_path), persona_id="user_a",
        memory_condition="nanobot_file_memory", pa_model=MODEL,
        consolidation_model=MODEL, memory_side_model=None, memos_api_base=None,
    )

    with pytest.raises(_StopAtConsolidationProvider, match="consolidation"):
        await run_orchestrator._execute_benchmark_run(args)

    assert seen == [None]
    assert (run_dir / STATE_FILENAME).read_text() == "{not json"
    assert not (run_dir / "provider_avoidance.lock").exists()
