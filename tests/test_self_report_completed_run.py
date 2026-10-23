"""`mempabench self-report` (scripts/self_report_completed_run.py) on a small
synthetic run of user_a: three finals with a recorded conversation, the rest
floored by BRS. Covers the rebuild from the run's records, the floor rule,
resume, a rebuild mismatch, a probe-repeat round and the refusals. No
network: the provider is scripted."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import scripts.self_report_completed_run as sr
from harness.judge_audit import JUDGE_ID, write_json
from harness.metrics import FINAL_PROBE_IDS, FLOOR_RULE, FLOOR_TEXT, floor_assignments
from harness.run_plan_builder import _load_script
from mempabench.cli import ENTRY_POINTS
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import PreferenceMatrix

ROOT = Path(__file__).resolve().parents[1]
PERSONA = "user_a"
PLAN_HEADER = ROOT / "tests/fixtures/run_plan_before_provider_avoidance_user_a_file_memory.yaml"
PREFERENCES = ROOT / "data/personas/user_a_preferences.yaml"
CHECKPOINT = "acc_108"
#: Finals with a recorded conversation; every other final is floored by BRS.
LIVE = ("fin_01", "fin_02", "fin_15")
SCORES = {"fin_01": 4, "fin_02": 2, "fin_15": 5}


def _reply(score: int) -> str:
    return (f"<dimension_self_report>\nscore: {score}\n"
            "reason: In turn 1 it said 'Here it is' and that is how I like it.\n</dimension_self_report>")


class _Provider(LLMProvider):
    """Scripted; reports a cost through `_extract_cost` as the real one does."""

    def __init__(self, responses: list[Any], log: list[dict[str, Any]]) -> None:
        super().__init__()
        self.responses, self.log = list(responses), log

    def get_default_model(self) -> str:
        return "default"

    @classmethod
    def _extract_cost(cls, provider_response: Any) -> Any:
        return provider_response

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.log.append({"messages": messages, "tools": tools, "model": model, **kwargs})
        response = self.responses.pop(0)
        response = response if isinstance(response, LLMResponse) else LLMResponse(content=response)
        if response.finish_reason != "error":
            self._extract_cost(0.01)
        return response


@pytest.fixture
def calls(monkeypatch) -> list[dict[str, Any]]:
    """Every provider call; the probe's own prompt picks its scripted score."""
    log: list[dict[str, Any]] = []

    class Scripted(_Provider):
        async def chat(self, messages, tools=None, model=None, **kwargs):
            pid = next(p for p in LIVE if f"conversation {p}" in messages[0]["content"])
            self.responses = [_reply(SCORES[pid])]
            return await super().chat(messages, tools, model, **kwargs)

    monkeypatch.setattr(sr, "make_provider", lambda model: Scripted([], log))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return log


# ---------------------------------------------------------------- the run


def _draft(pid: str) -> Path:
    return ROOT / "Authoring/probe/drafts" / PERSONA / pid / f"{pid}.yaml"


def _target(pid: str) -> tuple[str, str, str]:
    script = _load_script(_draft(pid))
    (attribute,) = script.beats[-1].active_skills
    setting = PreferenceMatrix.from_yaml(PREFERENCES).matrix[script.context][attribute].setting
    return attribute, script.context, setting


def _persona_files(folder: Path, memory: str | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "data/personas/user_a_identity.yaml", folder / "identity.yaml")
    shutil.copy(PREFERENCES, folder / "preferences.yaml")
    shutil.copy(ROOT / "simulator/profiles/user_a/TOOLS.md", folder / "TOOLS.md")
    (folder / "profile.yaml").write_text(
        "identity: identity.yaml\npreference_matrix: preferences.yaml\nmemory:\n  mode: nanobot\n", encoding="utf-8")
    if memory is not None:
        (folder / "memory").mkdir()
        (folder / "memory" / "MEMORY.md").write_text(memory, encoding="utf-8")


def _plan(run_id: str) -> dict[str, Any]:
    plan = yaml.safe_load(PLAN_HEADER.read_text(encoding="utf-8"))
    plan["run_id"] = run_id
    plan["pa_session_key"] = f"{run_id}_main"
    plan["steps"] = plan["steps"][:1]  # the fixture's acc_001, which the probes follow
    for pid in FINAL_PROBE_IDS:
        attribute, context, setting = _target(pid)
        plan["steps"].append({
            "step_id": pid, "kind": "final_probe", "persona_id": PERSONA,
            "script_path": str(_draft(pid).relative_to(ROOT)),
            "context": context, "target_cell": f"{context}_{attribute}",
            "target_attribute": attribute, "target_setting": setting,
            "benchmark_session_timestamp": "2027-01-31T10:00:00Z",
            "memory_mode": "read_only", "stage_policy": "discard",
        })
    return plan


def _transcript(pid: str, attribute: str, model: str) -> list[dict[str, Any]]:
    render = {"schema_version": "ix_render/1.1.0", "episodes": {}, "held": {}, "notices": [],
              "completion": {"released": False}, "certainties": []}
    return [
        {"t": 0, "kind": "session_start", "simulator_model": model},
        {"t": 1, "kind": "beat_enter", "beat": "ask", "active_skills": [attribute], "reception_skills": []},
        {"t": 2, "kind": "simulator_turn", "beat": "ask", "message": f"Can you help with conversation {pid}?",
         "consent_decisions": []},
        {"t": 3, "kind": "pa_turn", "beat": "ask", "outbound": "Here it is.",
         "tool_events": [{"name": "mcp_state_document_read"}], "metadata": {"ix_render": render}},
        {"t": 4, "kind": "session_end_reflection", "evidence": [{"sentence": "s", "turn": 1, "t": 3}]},
        {"t": 5, "kind": "session_end", "simulator_turns": 1},
    ]


def _step(run: Path, pid: str, plan: dict[str, Any], *, borrow: str | None = None) -> None:
    attribute, context, setting = _target(pid)
    runtime = plan["simulator_runtime"]
    folder = run / "steps" / pid
    folder.mkdir(parents=True)
    (folder / "meta.yaml").write_text(yaml.safe_dump({
        "step_id": pid, "kind": "final_probe", "context": context, "target_attribute": attribute,
        "target_setting": setting, "benchmark_session_timestamp": "2027-01-31T10:00:00+00:00",
        "run_id": plan["run_id"], "memory_condition": plan["memory_condition"], "pa_model": plan["pa_model"],
        "simulator_runtime": runtime, "simulator_model": runtime["simulator_model"],
        "source_checkpoint_id": CHECKPOINT, "preference_overlay": {},
        "ix_render_schema_version": "ix_render/1.1.0",
        **({"borrow_start_from": borrow} if borrow else {}),
    }, sort_keys=False), encoding="utf-8")
    (folder / "transcript.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in _transcript(pid, attribute, runtime["simulator_model"])),
        encoding="utf-8")


def _run(root: Path, run_id: str = "self_report_run", *, floored=None, checkpoints=True, borrow_from=None) -> Path:
    """A scored run: plan, ledger, BRS metrics with its floors, the run's
    matrix, the checkpoint persona files (with memory the rating must not
    see) and the recorded steps of the finals that are not floored."""
    run = root / run_id
    plan = _plan(run_id)
    floored = [pid for pid in FINAL_PROBE_IDS if pid not in LIVE] if floored is None else floored
    run.mkdir(parents=True)
    (run / "run_plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    write_json(run / "ledger.json", {"run_id": run_id, "current_step": None, "steps": {
        pid: {"status": "failed" if pid in floored else "done"} for pid in FINAL_PROBE_IDS}})
    write_json(run / "metrics" / f"{JUDGE_ID}.json", {
        "E1_consistency": {"n": 28}, **({"rule_assigned_scores": floor_assignments(floored)} if floored else {})})
    _persona_files(run / "simulator_workspace" / PERSONA)
    if checkpoints:
        _persona_files(run / "checkpoints" / CHECKPOINT / "simulator_workspace" / PERSONA, memory="MEMORY-SENTINEL")
    for pid in FINAL_PROBE_IDS:
        if pid not in floored:
            _step(run, pid, plan, borrow=f"{borrow_from}/{CHECKPOINT}" if borrow_from else None)
    return run


def _record(run: Path, pid: str) -> dict[str, Any]:
    return json.loads((run / "metrics" / "self_report" / f"{pid}.json").read_text(encoding="utf-8"))


def _summary(run: Path) -> dict[str, Any]:
    return json.loads((run / "metrics" / "self_report.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- tests


def test_the_command_is_registered() -> None:
    assert ENTRY_POINTS["self-report"] == "scripts.self_report_completed_run"


def test_each_live_final_is_rebuilt_from_the_run_and_rated_once(tmp_path: Path, calls) -> None:
    run = _run(tmp_path)
    summary = sr.self_report(run)

    assert len(calls) == len(LIVE)
    plan = yaml.safe_load((run / "run_plan.yaml").read_text())
    assert {call["model"] for call in calls} == {plan["simulator_runtime"]["simulator_model"]}
    assert all(call["tools"] == [] and call["max_tokens"] == sr.SELF_REPORT_MAX_TOKENS for call in calls)
    for pid in LIVE:
        record = _record(run, pid)
        attribute, context, setting = _target(pid)
        assert (record["status"], record["score"]) == ("scored", SCORES[pid])
        assert (record["attribute"], record["context"], record["setting"]) == (attribute, context, setting)
        assert record["rebuild_verification"] == {"status": "pass", "failures": []}
        prompt = record["prompt"]
        assert f"turn 0 — me:\nCan you help with conversation {pid}?\n\nturn 1 — assistant:\nHere it is." in prompt
        assert f"## Skill: {attribute} = {setting}\n" in prompt
        assert "private notes after turn 1:" in prompt
        # The checkpoint's persona card, never the memory the actor built up.
        assert "MEMORY-SENTINEL" not in prompt and "## Profile Memory" not in prompt
        assert record["source"]["checkpoint_run_dir"] == str(run)
    assert summary["counts"] == {"scored": 3, "rule_assigned": 25, "rebuild_mismatch": 0}
    assert summary["denominator"] == 28
    assert summary["prs"] == pytest.approx((sum(SCORES.values()) + 25 * FLOOR_TEXT) / 28)
    assert summary["cost_usd_total"] == pytest.approx(0.03)
    assert _summary(run) == summary


def test_a_final_brs_floors_gets_the_floor_and_is_never_asked(tmp_path: Path, calls) -> None:
    run = _run(tmp_path)
    sr.self_report(run)

    record = _record(run, "fin_03")
    assert (record["status"], record["score"], record["rule"]) == ("rule_assigned", FLOOR_TEXT, FLOOR_RULE)
    assert record["prompt"] is None and record["attempts"] == []
    assert record["rebuild_verification"]["status"] == "not_rebuilt"
    assert record["attribute"] == _target("fin_03")[0]
    assert not any("conversation fin_03" in call["messages"][0]["content"] for call in calls)


def test_a_rerun_keeps_scored_results_and_asks_nothing(tmp_path: Path, calls) -> None:
    run = _run(tmp_path)
    first = sr.self_report(run)
    kept = {path.name: path.read_bytes() for path in (run / "metrics" / "self_report").glob("*.json")}
    assert len(kept) == 28

    second = sr.self_report(run)
    assert len(calls) == len(LIVE)
    assert {path.name: path.read_bytes() for path in (run / "metrics" / "self_report").glob("*.json")} == kept
    assert second == first


def test_a_failed_probe_writes_nothing_and_the_rerun_asks_only_what_is_missing(
    tmp_path: Path, monkeypatch, calls,
) -> None:
    run = _run(tmp_path)
    monkeypatch.setattr(sr, "PROVIDER_ERROR_BACKOFF_S", (0,))
    good = sr.make_provider

    def failing_on_fin_02(model: str):
        provider = good(model)
        chat = provider.chat

        async def maybe_fail(messages, tools=None, model=None, **kwargs):
            if "conversation fin_02" in messages[0]["content"]:
                calls.append({"messages": messages, "model": model, "failed": True})
                return LLMResponse(content="Error calling LLM: boom", finish_reason="error")
            return await chat(messages, tools, model, **kwargs)

        provider.chat = maybe_fail
        return provider

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(sr, "make_provider", failing_on_fin_02)
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    with pytest.raises(sr.SelfReportError, match="fin_02: no rating after 2 tries"):
        sr.self_report(run)
    assert _record(run, "fin_01")["status"] == "scored"
    assert not (run / "metrics" / "self_report" / "fin_02.json").exists()
    assert not (run / "metrics" / "self_report" / "fin_15.json").exists()
    assert not (run / "metrics" / "self_report.json").exists()

    monkeypatch.setattr(sr, "make_provider", good)
    calls.clear()
    summary = sr.self_report(run)
    asked = {pid for call in calls for pid in LIVE if f"conversation {pid}" in call["messages"][0]["content"]}
    assert asked == {"fin_02", "fin_15"}
    assert summary["counts"]["scored"] == 3


def test_a_rebuild_that_disagrees_with_the_run_is_a_mismatch_outside_the_denominator(
    tmp_path: Path, calls,
) -> None:
    run = _run(tmp_path)
    meta_path = run / "steps" / "fin_02" / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta["simulator_model"] = "another/model"
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    summary = sr.self_report(run)
    record = _record(run, "fin_02")
    assert record["status"] == "rebuild_mismatch" and record["score"] is None
    assert record["rebuild_verification"]["failures"] == ["simulator_runtime"]
    assert not any("conversation fin_02" in call["messages"][0]["content"] for call in calls)
    assert summary["counts"] == {"scored": 2, "rule_assigned": 25, "rebuild_mismatch": 1}
    assert summary["denominator"] == 27
    assert summary["prs"] == pytest.approx((SCORES["fin_01"] + SCORES["fin_15"] + 25 * FLOOR_TEXT) / 27)
    assert summary["probes"]["fin_02"]["failures"] == ["simulator_runtime"]


def test_a_transcript_the_rebuild_cannot_replay_is_a_mismatch(tmp_path: Path, calls) -> None:
    run = _run(tmp_path)
    path = run / "steps" / "fin_15" / "transcript.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events.insert(3, {"t": 9, "kind": "beat_skipped", "beat": "ask"})
    path.write_text("".join(json.dumps(e) + "\n" for e in events))

    sr.self_report(run)
    record = _record(run, "fin_15")
    assert record["status"] == "rebuild_mismatch"
    assert record["rebuild_verification"]["failures"] == ["rebuild_error:beat_skipped in transcript"]


def test_the_recorded_overlay_sets_the_rated_setting(tmp_path: Path, calls) -> None:
    run = _run(tmp_path)
    attribute, context, setting = _target("fin_01")
    other = next(s for s in sr.dsr.load_prompt_files()[1][attribute]["who"] if s != setting)
    meta_path = run / "steps" / "fin_01" / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta["target_setting"] = other
    meta["preference_overlay"] = {context: {attribute: {"setting": other, "source": "forced_event",
                                                        "confidence": "High"}}}
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    plan_path = run / "run_plan.yaml"
    plan = yaml.safe_load(plan_path.read_text())
    next(s for s in plan["steps"] if s["step_id"] == "fin_01")["target_setting"] = other
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False))

    sr.self_report(run)
    record = _record(run, "fin_01")
    assert (record["status"], record["setting"], record["setting_source"]) == ("scored", other, "overlay")
    assert f"## Skill: {attribute} = {other}\n" in record["prompt"]


def test_changed_inputs_under_a_scored_result_are_refused(tmp_path: Path, calls) -> None:
    run = _run(tmp_path)
    sr.self_report(run)
    path = run / "steps" / "fin_01" / "transcript.jsonl"
    path.write_text(path.read_text().replace("Here it is.", "Here it is, done."))

    with pytest.raises(sr.SelfReportError, match=r"changed since they were written: \['fin_01'\]"):
        sr.self_report(run)
    assert len(calls) == len(LIVE)


def test_a_repeat_round_reads_the_checkpoint_of_the_run_it_names(tmp_path: Path, calls) -> None:
    source = _run(tmp_path / "runs", "source_run", floored=list(FINAL_PROBE_IDS))
    repeat = _run(tmp_path / "probe_variance", "source_run_probe_repeat02", checkpoints=False,
                  borrow_from="source_run")
    (repeat / "probe_repeat_source.json").write_text(json.dumps(
        {"schema_version": 1, "repeat": 2, "source_run_dir": str(source)}))

    summary = sr.self_report(repeat)
    record = _record(repeat, "fin_01")
    assert record["status"] == "scored"
    assert record["source"]["checkpoint_run_dir"] == str(source)
    assert summary["counts"]["scored"] == 3

    # A step that borrowed from another run than the one the round names.
    other = _run(tmp_path / "probe_variance", "other_probe_repeat02", checkpoints=False, borrow_from="elsewhere")
    (other / "probe_repeat_source.json").write_text(json.dumps(
        {"schema_version": 1, "repeat": 2, "source_run_dir": str(source)}))
    sr.self_report(other)
    assert _record(other, "fin_01")["rebuild_verification"]["failures"] == ["borrowed_from_another_run"]


def test_an_unscored_run_is_refused(tmp_path: Path, calls) -> None:
    run = _run(tmp_path)
    (run / "metrics" / f"{JUDGE_ID}.json").unlink()
    with pytest.raises(sr.SelfReportError, match="score the run first"):
        sr.self_report(run)
    assert not (run / "metrics" / "self_report").exists()


def test_a_final_neither_done_nor_floored_is_refused(tmp_path: Path, calls) -> None:
    run = _run(tmp_path)
    ledger = json.loads((run / "ledger.json").read_text())
    ledger["steps"]["fin_01"]["status"] = "failed"
    write_json(run / "ledger.json", ledger)
    with pytest.raises(sr.SelfReportError, match=r"neither done nor floored by BRS: \['fin_01'\]"):
        sr.self_report(run)


def test_no_call_starts_without_an_api_key(tmp_path: Path, calls, monkeypatch) -> None:
    run = _run(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(sr.SelfReportError, match="OPENROUTER_API_KEY is not set"):
        sr.self_report(run)
    assert calls == []
