"""Stale transcripts, restored ledgers and altered scores must not look complete."""
import json
import importlib.util
from pathlib import Path
import shutil

import pytest
import yaml

from harness import judge_audit as audit
from harness.metrics import PROBE_IDS

ROOT = Path(__file__).resolve().parents[2]


def _load_test_support(relative):
    spec = importlib.util.spec_from_file_location("audit_support", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scored_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    draft = tmp_path / "draft.yaml"
    draft.write_text("fixture: true\n")
    steps = [{"step_id": "acc_108", "kind": "accumulation"}] + [
        {"step_id": pid, "kind": "final_probe" if pid.startswith("fin") else "pre_event_probe",
         "script_path": str(draft)} for pid in PROBE_IDS
    ]
    (run / "run_plan.yaml").write_text(yaml.safe_dump({"run_id": "r", "persona_id": "user_b", "steps": steps}))
    audit.write_json(run / "ledger.json", {"run_id": "r", "current_step": None,
                     "steps": {s["step_id"]: {"status": "done"} for s in steps}})
    (run / "resolved_gt.yaml").write_text("fixture: true\n")
    audit.write_json(run / "judge_input/index.json", {"probes": list(PROBE_IDS)})
    for pid in PROBE_IDS:
        audit.write_json(run / "steps" / pid / "text_judge_pack.json", {"text": "actual run"})
        audit.write_json(run / "judge_input" / pid / "text_judge_pack.json", {"text": "actual run"})
        audit.write_json(run / "judge_output" / audit.JUDGE_ID / f"{pid}.json",
                         {"run_id": "r", "probe_id": pid, "judge_id": audit.JUDGE_ID,
                          "text_face": {"score": 4}, "ix_face": {"exact_match": True}})
    audit.write_json(run / "metrics" / f"{audit.JUDGE_ID}.json",
                     {"E1_consistency": {"n": 28, "ix_mean": 1, "text_percent": 80}})
    audit.write_json(run / audit.AUDIT_PATH, {"model": audit.MODEL, "judge_id": audit.JUDGE_ID,
                     "outputs": {}, "errors": {}})
    return run


def _record_skip(run, step="acc_108"):
    ledger = json.loads((run / "ledger.json").read_text())
    ledger["steps"][step] = {"status": "skipped", "summary": {"skip_reason": "failure_policy.skip_steps"}}
    audit.write_json(run / "ledger.json", ledger)


def test_validated_outputs_count(scored_run):
    assert audit.checked_outputs(scored_run) == (set(PROBE_IDS), True, [])


def test_an_unfinished_restore_counts_nothing(scored_run):
    ledger_before = (scored_run / "ledger.json").read_bytes()
    marker = scored_run / "recovery/pending.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("invalid JSON is still a pending marker\n")
    with pytest.raises(ValueError, match="unfinished checkpoint restore"):
        audit.require_complete(scored_run)
    valid, metrics, errors = audit.checked_outputs(scored_run)
    assert not valid and not metrics and errors
    assert (scored_run / "ledger.json").read_bytes() == ledger_before


def test_a_copied_run_checks_on_its_own(scored_run):
    copied = scored_run.parent / "formal_workspace" / "run"
    shutil.copytree(scored_run, copied)
    shutil.rmtree(scored_run)
    assert audit.checked_outputs(copied) == (set(PROBE_IDS), True, [])


def test_changed_inputs_do_not_revoke_scores(scored_run):
    """No provenance: score files count whatever the packs and sources say now."""
    for relative in ("steps/fin_01/text_judge_pack.json", "judge_input/fin_01/text_judge_pack.json",
                     "resolved_gt.yaml"):
        (scored_run / relative).write_text("changed\n")
    assert audit.checked_outputs(scored_run) == (set(PROBE_IDS), True, [])


def test_a_skipped_accumulation_step_is_scored(scored_run):
    _record_skip(scored_run)
    assert audit.checked_outputs(scored_run) == (set(PROBE_IDS), True, [])


@pytest.mark.parametrize("step", ["fin_01", "acc_108"])
def test_an_unfinished_or_skipped_probe_step_counts_nothing(scored_run, step):
    p = scored_run / "ledger.json"
    ledger = json.loads(p.read_text())
    if step == "fin_01":
        del ledger["steps"]["fin_01"]
    else:
        ledger["steps"]["acc_108"]["status"] = "failed"
    audit.write_json(p, ledger)
    valid, metrics, errors = audit.checked_outputs(scored_run)
    assert not valid and not metrics and errors


@pytest.mark.parametrize("change", ["score", "probe_id", "run_id", "exact_match"])
def test_an_invalid_score_file_does_not_count(scored_run, change):
    p = scored_run / "judge_output" / audit.JUDGE_ID / "fin_01.json"
    record = json.loads(p.read_text())
    if change == "score":
        record["text_face"]["score"] = 7
    elif change == "exact_match":
        record["ix_face"]["exact_match"] = "yes"
    else:
        record[change] = "other"
    audit.write_json(p, record)
    valid, metrics, errors = audit.checked_outputs(scored_run)
    assert len(valid) == 32 and "fin_01" not in valid and not metrics and errors


def test_metrics_without_score_files_is_not_completion(scored_run):
    shutil.rmtree(scored_run / "judge_output")
    assert audit.checked_outputs(scored_run) == (set(), False, [])


def test_score_files_without_metrics_are_not_completion(scored_run):
    (scored_run / "metrics" / f"{audit.JUDGE_ID}.json").unlink()
    assert audit.checked_outputs(scored_run) == (set(PROBE_IDS), False, [])


def test_incomplete_run_never_calls_packer_or_judge(scored_run, monkeypatch):
    from scripts import score_completed_run as scorer
    def forbidden(*args, **kwargs):
        pytest.fail("incomplete run reached scoring")
    monkeypatch.setattr(scorer, "pack_run", forbidden)
    monkeypatch.setattr(scorer, "run_judge", forbidden)
    p = scored_run / "ledger.json"
    ledger = json.loads(p.read_text())
    ledger["steps"]["fin_01"]["status"] = "failed"
    audit.write_json(p, ledger)
    with pytest.raises(ValueError, match="not complete"):
        scorer.score(scored_run)


def test_existing_score_files_are_never_judged_again(scored_run, monkeypatch):
    from scripts import score_completed_run as scorer
    original = (scored_run / "judge_output" / audit.JUDGE_ID / "fin_01.json").read_bytes()
    monkeypatch.setattr(scorer, "load_run_plan", lambda *args, **kwargs: None)
    def forbidden(*args, **kwargs):
        pytest.fail("an existing score reached a new judge request or a repack")
    monkeypatch.setattr(scorer, "OpenRouterJudgeClient", forbidden)
    monkeypatch.setattr(scorer, "pack_run", forbidden)
    (scored_run / audit.AUDIT_PATH).unlink()  # outputs from an earlier scorer
    assert scorer.score(scored_run) == 0
    assert (scored_run / "judge_output" / audit.JUDGE_ID / "fin_01.json").read_bytes() == original


def test_an_invalid_score_file_stops_scoring(scored_run, monkeypatch):
    from scripts import score_completed_run as scorer
    monkeypatch.setattr(scorer, "load_run_plan", lambda *args, **kwargs: None)
    p = scored_run / "judge_output" / audit.JUDGE_ID / "fin_01.json"
    p.write_text("{not json")
    with pytest.raises(ValueError, match="invalid score files"):
        scorer.score(scored_run)
    assert p.read_text() == "{not json"


@pytest.mark.parametrize("mainline_skipped", [False, True])
@pytest.mark.parametrize("interrupt_after", [None, 4])
def test_completed_cell_scores_and_aggregates_then_resumes_without_new_calls(scored_run, monkeypatch, mainline_skipped, interrupt_after):
    """Exercise the real evaluator and aggregator with a local fake provider."""
    import shutil
    from data.personas.coverage import TAXONOMY
    from scripts import score_completed_run as scorer
    support = _load_test_support("tests/test_judge/test_evaluator.py")
    for name in ("judge_input", "judge_output", "judge_audit", "metrics"):
        shutil.rmtree(scored_run / name)
    if mainline_skipped:
        _record_skip(scored_run)
    def pack(run, *, project_root):
        for i, pid in enumerate(PROBE_IDS):
            attribute = list(TAXONOMY)[i % 14] if i < 28 else list(TAXONOMY)[i - 28]
            probe = support.write_judge_input(run, pid, attribute)
            for filename in ("meta.json", "ground_truth.json"):
                p = probe / filename
                data = json.loads(p.read_text())
                data["run_id"] = "r"
                data["context"] = "personal" if 14 <= i < 28 else "work"
                audit.write_json(p, data)
        index = run / "judge_input/index.json"
        audit.write_json(index, {"run_id": "r", "probes": [{"probe_id": pid} for pid in PROBE_IDS]})
        return index
    client = support.FakeClient([support.valid_reply()] * 33)
    models = []
    def make_client(model):
        models.append(model)
        return client
    monkeypatch.setattr(scorer, "pack_run", pack)
    monkeypatch.setattr(scorer, "load_run_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(scorer, "OpenRouterJudgeClient", make_client)
    if interrupt_after is not None:
        real_judge = scorer.run_judge
        failed_once = False
        def interrupt(probe, *args, **kwargs):
            nonlocal failed_once
            if probe.name == PROBE_IDS[interrupt_after] and not failed_once:
                failed_once = True
                raise TimeoutError("injected connection interruption")
            return real_judge(probe, *args, **kwargs)
        monkeypatch.setattr(scorer, "run_judge", interrupt)
        assert scorer.score(scored_run) == 1
        outputs = scored_run / "judge_output" / audit.JUDGE_ID
        prior = {path.name: path.read_bytes() for path in outputs.glob("*.json")}
        assert len(prior) == interrupt_after
        assert len(client.calls) == interrupt_after
        # Rerunning the same command judges the failed probe again and keeps the rest.
        assert scorer.score(scored_run) == 0
        assert all((outputs / name).read_bytes() == data for name, data in prior.items())
        recovered = json.loads((scored_run / audit.AUDIT_PATH).read_text())
        assert recovered["failure_history"][0]["errors"][PROBE_IDS[interrupt_after]]["type"] == "TimeoutError"
    else:
        assert scorer.score(scored_run) == 0
    assert audit.checked_outputs(scored_run) == (set(PROBE_IDS), True, [])
    assert len(client.calls) == 33 and models == [audit.MODEL] * (2 if interrupt_after is not None else 1)
    metrics = json.loads((scored_run / "metrics" / f"{audit.JUDGE_ID}.json").read_text())
    assert metrics["E1_consistency"]["n"] == 28
    assert metrics["E5_iqi"]["baseline"] is None
    recorded = json.loads((scored_run / audit.AUDIT_PATH).read_text())
    assert recorded["skipped_steps"] == ({"acc_108": "failure_policy.skip_steps"} if mainline_skipped else {})
    assert scorer.score(scored_run) == 0
    assert len(client.calls) == 33 and models == [audit.MODEL] * (2 if interrupt_after is not None else 1)

    # A missing score file resumes with exactly one request and no repack.
    monkeypatch.setattr(scorer, "pack_run", lambda *a, **k: pytest.fail("judge_output is not empty"))
    (scored_run / "judge_output" / audit.JUDGE_ID / "fin_01.json").unlink()
    client.replies.append(support.valid_reply())
    assert scorer.score(scored_run) == 0
    assert len(client.calls) == 34 and models == [audit.MODEL] * (3 if interrupt_after is not None else 2)
    assert audit.checked_outputs(scored_run) == (set(PROBE_IDS), True, [])


def test_retry_refuses_a_failure_that_already_has_a_score(scored_run, monkeypatch):
    from scripts import score_completed_run as scorer
    monkeypatch.setattr(scorer, "load_run_plan", lambda *args, **kwargs: None)
    (scored_run / "metrics" / f"{audit.JUDGE_ID}.json").unlink()
    manifest = scored_run / audit.AUDIT_PATH
    record = json.loads(manifest.read_text())
    pid = PROBE_IDS[0]
    record["errors"] = {pid: {"type": "TimeoutError", "message": "interrupted"}}
    audit.write_json(manifest, record)
    output = scored_run / "judge_output" / audit.JUDGE_ID / f"{pid}.json"
    before = output.read_bytes()
    with pytest.raises(ValueError, match="inconsistent failed/output"):
        scorer.score(scored_run)
    assert output.read_bytes() == before
