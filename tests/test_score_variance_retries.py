"""Real judge/HTTP/provenance tests for the durable variance request allowance."""
from copy import deepcopy
import json
import re
import shutil
from types import SimpleNamespace

import httpx
import pytest
import yaml

from data.personas.coverage import TAXONOMY
from harness import judge_audit as audit
from harness.metrics import PROBE_IDS, compute_metrics, floor_assignments
from judge import evaluator
from scripts import score_completed_run as scorer
from test_harness.test_judge_audit import scored_run  # noqa: F401 -- shared fixture
from test_judge.test_evaluator import valid_reply, write_judge_input


@pytest.fixture
def variance_run(scored_run, monkeypatch):
    for name in ("judge_input", "judge_output", "judge_audit", "metrics"):
        shutil.rmtree(scored_run / name)

    def pack(run, *, project_root, only=None):
        # Scoring packs whenever judge_output is still empty, so a resume
        # before the first score file packs again, over the same inputs.
        shutil.rmtree(run / "judge_input", ignore_errors=True)
        for i, pid in enumerate(PROBE_IDS):
            if only is not None and pid not in only:
                continue
            attribute = list(TAXONOMY)[i % 14] if i < 28 else list(TAXONOMY)[i - 28]
            probe = write_judge_input(run, pid, attribute)
            for filename in ("meta.json", "ground_truth.json"):
                path = probe / filename
                data = json.loads(path.read_text())
                data.update(run_id="r", context="personal" if 14 <= i < 28 else "work")
                audit.write_json(path, data)
        index = run / "judge_input/index.json"
        audit.write_json(index, {"run_id": "r", "probes": [{"probe_id": pid} for pid in (only or PROBE_IDS)]})
        return index

    monkeypatch.setattr(scorer, "pack_run", pack)
    monkeypatch.setattr(scorer, "load_run_plan", lambda *args, **kwargs: None)
    return scored_run


class LocalProvider:
    """Only the HTTP server is replaced; evaluator and accounting stay real."""

    def __init__(self, run, monkeypatch, actions=None):
        self.run = run
        self.actions = deepcopy(actions or {})
        self.calls = []
        original_client = httpx.Client
        monkeypatch.setenv("OPENROUTER_API_KEY", "local-test-only")
        monkeypatch.setattr(evaluator, "_supported_parameters", lambda *args: {"temperature", "response_format"})
        monkeypatch.setattr(evaluator, "httpx", SimpleNamespace(Client=lambda **kwargs: original_client(
            **kwargs, transport=httpx.MockTransport(self.handle))))

    def handle(self, request):
        payload = json.loads(request.content)
        assert request.method == "POST"
        assert payload["model"] == audit.MODEL
        assert payload["max_tokens"] == evaluator.JUDGE_MAX_TOKENS == 8192
        assert payload["temperature"] == 0.0
        assert payload["response_format"] == {"type": "json_object"}
        pid = re.search(r"- probe_id: `([^`]+)`", payload["messages"][1]["content"]).group(1)
        state = self.state()["variance_requests"]["probes"][pid]
        # The persisted intent must precede even the first byte of the POST.
        assert state["requests"][-1]["status"] == "started"
        assert state["requests"][-1]["payload"] == payload
        activity = json.loads((self.run / "runtime_activity.json").read_text())
        assert [item["role"] for item in activity["requests"]] == ["final_probe_judge"]
        self.calls.append((pid, payload))
        options = self.actions.get(pid, [])
        action = options.pop(0) if options else valid_reply()
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, int):
            return httpx.Response(action, json={"error": "injected provider response"})
        return httpx.Response(200, json={"model": audit.MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": action}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001}})

    def state(self):
        state = json.loads((self.run / audit.AUDIT_PATH).read_text())
        requests = self.run / scorer.REQUESTS_PATH
        if requests.exists():
            state["variance_requests"] = json.loads(requests.read_text())
        return state

    def calls_for(self, pid):
        return [body for probe, body in self.calls if probe == pid]

    def scored(self):
        return {path.stem for path in (self.run / "judge_output" / audit.JUDGE_ID).glob("*.json")}


def test_finishes_untouched_before_retries_preserves_native_reask_and_resumes(variance_run, monkeypatch):
    rate_limited, reasked = PROBE_IDS[:2]
    provider = LocalProvider(variance_run, monkeypatch, {
        rate_limited: [429, valid_reply()], reasked: ["not JSON", valid_reply()]})
    assert scorer.score(variance_run, variance_retries=True) == 0
    assert [pid for pid, _ in provider.calls] == [rate_limited, reasked, reasked, *PROBE_IDS[2:], rate_limited]
    assert provider.calls_for(rate_limited)[0] == provider.calls_for(rate_limited)[1]
    assert len(provider.calls) == 35
    assert audit.checked_outputs(variance_run) == (set(PROBE_IDS), True, [])
    state = provider.state()
    assert not state["errors"]
    record = state["variance_requests"]["probes"][rate_limited]
    assert record["extra_used"] and record["chances_used"] == 1
    assert state["variance_requests"]["probes"][reasked]["chances_used"] == 1
    assert len(state["variance_requests"]["probes"][reasked]["requests"]) == 2
    assert scorer.score(variance_run, variance_retries=True) == 0
    assert len(provider.calls) == 35


def test_429_during_native_reask_ends_that_chance_and_a_fresh_chance_is_scored(variance_run, monkeypatch):
    pid = PROBE_IDS[0]
    provider = LocalProvider(variance_run, monkeypatch, {pid: ["not JSON", 429, valid_reply()]})
    assert scorer.score(variance_run, variance_retries=True) == 0
    bodies = provider.calls_for(pid)
    assert len(bodies) == 3 and bodies[0] != bodies[1] and bodies[2] == bodies[0]
    assert provider.calls[-1][0] == pid
    record = provider.state()["variance_requests"]["probes"][pid]
    assert record["extra_used"] and record["chances_used"] == 1
    output = json.loads((variance_run / "judge_output" / audit.JUDGE_ID / f"{pid}.json").read_text())
    assert output["cost"]["prompt_tokens"] == 100


def test_rate_limit_gets_one_uncounted_retry_beyond_three_counted_chances(variance_run, monkeypatch):
    pid = PROBE_IDS[0]
    provider = LocalProvider(variance_run, monkeypatch, {pid: [429, 429, 500, httpx.ReadTimeout("timeout")]})
    assert scorer.score(variance_run, variance_retries=True) == 1
    assert len(provider.calls_for(pid)) == 4
    assert len(provider.calls) == 36
    state = provider.state()
    assert provider.scored() == set(PROBE_IDS[1:])
    assert set(state["errors"]) == {pid}
    record = state["variance_requests"]["probes"][pid]
    assert record["extra_used"] and record["chances_used"] == 3 and record["status"] == "exhausted"
    assert scorer.score(variance_run, variance_retries=True) == 1
    assert len(provider.calls) == 36
    with pytest.raises(ValueError, match="generic retries are refused"):
        scorer.score(variance_run)
    assert len(provider.calls) == 36


@pytest.mark.parametrize("failure", [500, httpx.ReadTimeout("timeout"), "truncated"])
def test_other_failure_is_retried_after_every_untouched_item(variance_run, monkeypatch, failure):
    pid = PROBE_IDS[0]
    if failure == "truncated":
        failure = "not JSON"  # invalid twice (native re-ask included) ends one chance
        actions = [failure, failure, valid_reply()]
    else:
        actions = [failure, valid_reply()]
    provider = LocalProvider(variance_run, monkeypatch, {pid: actions})
    assert scorer.score(variance_run, variance_retries=True) == 0
    assert [item for item, _ in provider.calls][:len(actions) - 1] == [pid] * (len(actions) - 1)
    assert provider.calls[-1][0] == pid and len(provider.calls_for(pid)) == len(actions)
    assert provider.state()["variance_requests"]["probes"][pid]["chances_used"] == 2


def test_three_counted_failures_exhaust_the_probe_across_restarts(variance_run, monkeypatch):
    pid = PROBE_IDS[0]
    provider = LocalProvider(variance_run, monkeypatch, {pid: [500, KeyboardInterrupt("stop"), ]})
    with pytest.raises(KeyboardInterrupt):
        scorer.score(variance_run, variance_retries=True)
    assert provider.state()["variance_requests"]["probes"][pid]["requests"][-1]["status"] == "started"
    assert scorer.score(variance_run, variance_retries=True) == 1  # the unknown request is held
    assert provider.state()["variance_requests"]["probes"][pid]["status"] == "unknown"
    provider.actions[pid] = [500, 500, 500]
    state = provider.state()
    record = state["variance_requests"]["probes"][pid]
    record["requests"][-1]["status"] = "error"  # reconciled: the provider reported a failure
    record["status"] = "failed"
    scorer._save_variance_audit(variance_run / scorer.REQUESTS_PATH, state["variance_requests"])
    assert scorer.score(variance_run, variance_retries=True) == 1
    assert len(provider.calls_for(pid)) == 3
    assert provider.state()["variance_requests"]["probes"][pid]["status"] == "exhausted"


def test_pending_provider_call_freezes_only_that_item_across_restart(variance_run, monkeypatch):
    pid = PROBE_IDS[0]
    provider = LocalProvider(variance_run, monkeypatch, {pid: [KeyboardInterrupt("injected process stop")]})
    with pytest.raises(KeyboardInterrupt, match="process stop"):
        scorer.score(variance_run, variance_retries=True)
    assert len(provider.calls) == 1
    assert provider.state()["variance_requests"]["probes"][pid]["requests"][0]["status"] == "started"
    assert scorer.score(variance_run, variance_retries=True) == 1
    assert len(provider.calls) == 33
    state = provider.state()
    assert state["variance_requests"]["probes"][pid]["status"] == "unknown"
    assert provider.scored() == set(PROBE_IDS[1:])
    assert scorer.score(variance_run, variance_retries=True) == 1
    assert len(provider.calls) == 33


def test_known_429_budget_survives_restart_before_extra(variance_run, monkeypatch):
    limited, stopped = PROBE_IDS[:2]
    provider = LocalProvider(variance_run, monkeypatch, {
        limited: [429, valid_reply()], stopped: [KeyboardInterrupt("stop after 429")]})
    with pytest.raises(KeyboardInterrupt):
        scorer.score(variance_run, variance_retries=True)
    assert provider.state()["variance_requests"]["probes"][limited]["status"] == "http429"
    assert scorer.score(variance_run, variance_retries=True) == 1
    assert [pid for pid, _ in provider.calls] == [limited, stopped, *PROBE_IDS[2:], limited]
    assert len(provider.calls_for(stopped)) == 1
    assert len(provider.calls_for(limited)) == 2
    assert provider.state()["variance_requests"]["probes"][limited]["status"] == "succeeded"


def test_interrupted_extra_consumes_its_only_allowance_before_post(variance_run, monkeypatch):
    pid = PROBE_IDS[0]
    provider = LocalProvider(variance_run, monkeypatch, {pid: [429, KeyboardInterrupt("stop during extra")]})
    with pytest.raises(KeyboardInterrupt, match="during extra"):
        scorer.score(variance_run, variance_retries=True)
    record = provider.state()["variance_requests"]["probes"][pid]
    assert record["extra_used"] and record["requests"][-1]["status"] == "started"
    assert len(provider.calls) == 34
    assert scorer.score(variance_run, variance_retries=True) == 1
    assert len(provider.calls) == 34


def test_known_429_response_is_durable_before_failure_summary(variance_run, monkeypatch):
    pid = PROBE_IDS[0]
    provider = LocalProvider(variance_run, monkeypatch, {pid: [429, valid_reply()]})
    save = scorer._save_variance_audit
    stopped = False

    def stop_after_response(path, state):
        nonlocal stopped
        save(path, state)
        record = state["probes"].get(pid, {}) if "probes" in state else {}
        if not stopped and record.get("requests") and record["requests"][-1].get("status") == "http429":
            stopped = True
            raise KeyboardInterrupt("saved 429 but not outer failure summary")

    monkeypatch.setattr(scorer, "_save_variance_audit", stop_after_response)
    with pytest.raises(KeyboardInterrupt):
        scorer.score(variance_run, variance_retries=True)
    assert pid not in provider.state()["errors"]
    assert scorer.score(variance_run, variance_retries=True) == 0
    assert [probe for probe, _ in provider.calls] == [pid, *PROBE_IDS[1:], pid]
    record = provider.state()["variance_requests"]["probes"][pid]
    assert record["extra_used"] and record["chances_used"] == 1


def record_launch(run, launch_id, attempts):
    """Append one launch's failed attempts of fin_27 to its attempt record
    (`probe_launch_attempts/fin_27.json`, the file the run writes)."""
    path = run / "probe_launch_attempts" / "fin_27.json"
    record = (json.loads(path.read_text()) if path.exists()
              else {"schema": "mempa_probe_launch_attempts/1", "step_id": "fin_27", "attempts": []})
    record["attempts"] += [{"launch_id": launch_id, "attempt": n, "status": "failed", "failure_kind": "ix_runtime",
                            "error": "AgentProviderError: PA turn failed the IX render contract: x"}
                           for n in range(1, attempts + 1)]
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record))


@pytest.fixture
def source_run(variance_run):
    """A plain `mempabench run` whose fin_27 failed all three attempts of its last launch."""
    plan = yaml.safe_load((variance_run / "run_plan.yaml").read_text())
    attribute = list(TAXONOMY)[PROBE_IDS.index("fin_27") % 14]
    for step in plan["steps"]:
        if step["step_id"] == "fin_27":
            step.update(context="personal", target_attribute=attribute, target_setting=TAXONOMY[attribute][-1])
    (variance_run / "run_plan.yaml").write_text(yaml.safe_dump(plan))
    ledger = json.loads((variance_run / "ledger.json").read_text())
    ledger["current_step"] = "fin_27"
    ledger["steps"]["fin_27"] = {"status": "failed", "failure_kind": "ix_runtime",
                                 "started_at": "2026-10-18T10:00:00+00:00", "ended_at": "2026-10-18T10:06:00+00:00",
                                 "error": "PA turn failed the IX render contract: x"}
    audit.write_json(variance_run / "ledger.json", ledger)
    # Launch 1 was stopped after two failed attempts; the resume (launch 2) failed all three.
    record_launch(variance_run, "launch-1", 2)
    record_launch(variance_run, "launch-2", 3)
    return variance_run


class StandInJudge:
    """Replaces only the OpenRouter client; the evaluator, score files and metrics stay real."""

    def __init__(self):
        self.probes = []

    def complete(self, *, system, user):
        self.probes.append(re.search(r"- probe_id: `([^`]+)`", user).group(1))
        return evaluator.JudgeReply(content=valid_reply(), model=audit.MODEL,
                                    prompt_tokens=1, completion_tokens=1, usd=0.0)


def test_a_source_run_with_an_exhausted_probe_is_scored_at_the_floor(source_run, monkeypatch):
    judge = StandInJudge()
    monkeypatch.setattr(scorer, "OpenRouterJudgeClient", lambda model: judge)
    assert scorer.score(source_run, not_applicable_exhausted=True) == 0
    scored = [pid for pid in PROBE_IDS if pid != "fin_27"]
    assert judge.probes == scored and not (source_run / "judge_input" / "fin_27").exists()
    written = json.loads((source_run / "metrics" / f"{audit.JUDGE_ID}.json").read_text())
    assert written["rule_assigned_scores"] == floor_assignments({"fin_27"})
    assert written["E1_consistency"]["n"] == 28 and written["E2_cross_tab"]["n"] == 28
    assert written["E4_pet"]["n"] == 10
    expected = compute_metrics(source_run.resolve(), audit.JUDGE_ID, not_applicable=frozenset({"fin_27"}))
    assert written == json.loads(json.dumps(expected))
    manifest = json.loads((source_run / audit.AUDIT_PATH).read_text())
    assert manifest["metrics_scope"] == "raw_all_33_exhausted_attempts_floored"
    evidence = manifest["not_applicable"]["fin_27"]
    assert evidence["launch_id"] == "launch-2" and len(evidence["errors"]) == 3
    assert evidence["attempts"] == {"path": "run/probe_launch_attempts/fin_27.json"}
    assert audit.checked_outputs(source_run) == (set(scored), True, [])


def test_a_source_run_whose_failed_probe_has_an_attempt_left_is_not_scored(source_run, monkeypatch):
    monkeypatch.setattr(scorer, "pack_run", lambda *a, **k: pytest.fail("an incomplete run reached packing"))
    record_launch(source_run, "launch-3", 1)  # a resume that failed once and was stopped
    with pytest.raises(ValueError, match="fin_27: its last launch launch-3 recorded 1 attempt"):
        scorer.score(source_run, not_applicable_exhausted=True)
    assert not (source_run / "judge_audit").exists() and not (source_run / "metrics").exists()


def test_a_source_run_without_attempt_records_is_refused_with_the_resume_path(source_run, monkeypatch):
    monkeypatch.setattr(scorer, "pack_run", lambda *a, **k: pytest.fail("an unproven floor reached packing"))
    shutil.rmtree(source_run / "probe_launch_attempts")
    with pytest.raises(ValueError, match="fin_27: no attempt record .*Resume the run with the same "
                                         "`mempabench run` command and run ID"):
        scorer.score(source_run, not_applicable_exhausted=True)
    assert not (source_run / "judge_audit").exists() and not (source_run / "metrics").exists()


def test_scoring_a_source_run_refuses_while_its_producer_holds_it(source_run, monkeypatch):
    from scripts.run_locked import producer_lock
    monkeypatch.setattr(scorer, "pack_run", lambda *a, **k: pytest.fail("a run in production reached packing"))
    with producer_lock(source_run):
        with pytest.raises(RuntimeError, match="A producer already owns this run"):
            scorer.score(source_run, not_applicable_exhausted=True)
    assert not (source_run / "judge_audit").exists()


def test_without_the_flag_a_source_run_with_a_failed_probe_stays_incomplete(source_run, monkeypatch):
    monkeypatch.setattr(scorer, "pack_run", lambda *a, **k: pytest.fail("an incomplete run reached packing"))
    with pytest.raises(ValueError, match=r"run is not complete: \['fin_27'\]"):
        scorer.score(source_run)


def test_the_repeat_evidence_and_its_flag_are_unchanged(source_run, monkeypatch):
    monkeypatch.setattr(scorer, "pack_run", lambda *a, **k: pytest.fail("a refused run reached packing"))
    # --variance-retries still reads the per-Probe attempt ledger, which a source run does not keep.
    with pytest.raises(ValueError, match="fin_27: no durable attempt record"):
        scorer.score(source_run, variance_retries=True, not_applicable_exhausted=True)
    # A variance repeat is still refused N/A scoring without --variance-retries.
    (source_run / "probe_repeat_source.json").write_text("{}")
    with pytest.raises(ValueError, match="variance repeat"):
        scorer.score(source_run, not_applicable_exhausted=True)
