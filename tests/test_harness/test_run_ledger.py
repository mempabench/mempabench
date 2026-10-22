import json
from datetime import datetime
from pathlib import Path

import pytest

from harness.run_plan import (
    Mem0RuntimeConfig,
    RunPlan,
    RunStep,
)
from harness.run_ledger import (
    RunLedger,
    RunLedgerLoadError,
    UnsafeRunResumeError,
    format_start,
    format_step_running,
)
from pa.providers.request_policy import resolve_model_request_policy


def _plan() -> RunPlan:
    policy = resolve_model_request_policy(
        "openrouter", "test-pa"
    ).model_copy(
        update={"policy_source": "model", "validated_roles": ["pa"]}
    )
    memory_policy = policy.model_copy(
        update={"validated_roles": ["pa", "memory_extraction"]}
    )
    mem0_runtime_config = Mem0RuntimeConfig(
        mem0_version="2.0.2",
        mem0_submodule_commit="70bc9e51d57fe005d02b7b6d81b56476bade3cb3",
        embedding_api_base="https://api.openai.com/v1",
        embedding_max_input_tokens=8191,
        embedding_chunk_safety_margin_tokens=256,
        embedding_tokenizer_encoding="cl100k_base",
        tiktoken_version="0.12.0",
        extraction_max_output_tokens=16000,
        enable_entity_memory=True,
        entity_model="en_core_web_sm@3.8.0",
        spacy_version="3.8.16",
    )
    return RunPlan(
        run_id="run_001",
        persona_id="user_a",
        memory_condition="mem0",
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base="https://openrouter.ai/api/v1",
        pa_model="test-pa",
        pa_request_policy=policy.snapshot(),
        memory_side_model="test-pa",
        memory_side_request_policy=memory_policy.snapshot(),
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        mem0_runtime_config=mem0_runtime_config,
        steps=[
            {
                "step_id": "acc_001",
                "kind": "accumulation",
                "persona_id": "user_a",
                "script_path": "missing.yaml",
                "context": "work",
                "target_cell": "cell_a",
                    "acc_num": 1,
                    "benchmark_session_timestamp": (
                        "2026-10-01T09:00:00+00:00"
                    ),
                    "memory_mode": "read_write",
                "stage_policy": "commit",
                "placeholder": True,
            },
            {
                "step_id": "acc_002",
                "kind": "accumulation",
                "persona_id": "user_a",
                "script_path": "missing.yaml",
                "context": "work",
                "target_cell": "cell_b",
                    "acc_num": 2,
                    "benchmark_session_timestamp": (
                        "2026-10-01T15:00:00+00:00"
                    ),
                    "memory_mode": "read_write",
                "stage_policy": "commit",
                "placeholder": True,
            },
        ],
    )


def test_ledger_persists_status(tmp_path: Path):
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_running("acc_001")
    ledger.mark_done("acc_001", artifacts={"meta": "meta.yaml"}, summary={"beats": 1})

    loaded = RunLedger(tmp_path / "run", "run_001")

    assert loaded.steps["acc_001"].status == "done"
    assert loaded.steps["acc_001"].artifacts == {"meta": "meta.yaml"}


def test_a_failed_step_records_its_artifacts(tmp_path: Path):
    """A failed step's audit trail exists on disk; the ledger has to name it."""
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_running("acc_001")
    ledger.mark_failed(
        "acc_001",
        "RuntimeError: boom",
        artifacts={"transcript_jsonl": "transcript.jsonl", "meta": "meta.yaml"},
    )

    loaded = RunLedger(tmp_path / "run", "run_001")

    assert loaded.steps["acc_001"].status == "failed"
    assert loaded.steps["acc_001"].error == "RuntimeError: boom"
    assert loaded.steps["acc_001"].artifacts == {
        "transcript_jsonl": "transcript.jsonl",
        "meta": "meta.yaml",
    }


def test_artifact_paths_are_stored_relative_to_the_run(tmp_path: Path):
    """What the run owns is the path inside its own directory.

    The absolute path names where the run was on the day it ran; archiving it
    or restoring a snapshot of this ledger elsewhere makes it a lie, so it is
    never what gets written down.
    """
    run_dir = tmp_path / "runs" / "m" / "c" / "p" / "run_001"
    ledger = RunLedger(run_dir, "run_001")
    ledger.mark_done(
        "acc_001",
        artifacts={
            "meta": str(run_dir / "steps" / "acc_001" / "meta.yaml"),
            "elsewhere": "/outside/the/run/notes.txt",
        },
    )

    assert ledger.steps["acc_001"].artifacts == {
        "meta": "steps/acc_001/meta.yaml",
        # A path from outside this run is left exactly as it came in.
        "elsewhere": "/outside/the/run/notes.txt",
    }
    stored = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))
    assert stored["steps"]["acc_001"]["artifacts"]["meta"] == (
        "steps/acc_001/meta.yaml"
    )


def test_the_typed_kind_and_summary_schema_survive_a_reload(tmp_path: Path):
    """Two facts the free-text `error` and the bare `summary` cannot carry.

    The kind says how the step died in one word; the schema says which counting
    rules produced the numbers next to it. Both have to be on disk, because the
    reader is Admin looking at an archived run months later.
    """
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_done("acc_001", summary={"tool_calls": 3}, summary_schema=1)
    ledger.mark_failed(
        "acc_002",
        "OpenHoldEpisodeError: acc_002 left a hold open",
        kind="ix_open_hold",
    )

    stored = json.loads(
        (tmp_path / "run" / "ledger.json").read_text(encoding="utf-8")
    )
    assert stored["steps"]["acc_001"]["summary_schema"] == 1
    assert "failure_kind" not in stored["steps"]["acc_001"]
    assert stored["steps"]["acc_002"]["failure_kind"] == "ix_open_hold"

    loaded = RunLedger(tmp_path / "run", "run_001")
    assert loaded.steps["acc_001"].summary_schema == 1
    assert loaded.steps["acc_001"].failure_kind is None
    assert loaded.steps["acc_002"].failure_kind == "ix_open_hold"


def test_a_ledger_written_before_the_typed_fields_still_loads(tmp_path: Path):
    """Every ledger already on disk lacks both fields; none of them may break."""
    run_dir = tmp_path / "run"
    RunLedger(run_dir, "run_001")
    path = run_dir / "ledger.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["steps"] = {
        "acc_001": {
            "status": "failed",
            "started_at": "2026-08-30T10:00:00+00:00",
            "ended_at": "2026-08-30T10:05:00+00:00",
            "error": "RuntimeError: boom",
            "summary": {"tool_calls": 2},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = RunLedger(run_dir, "run_001")

    assert loaded.steps["acc_001"].status == "failed"
    assert loaded.steps["acc_001"].failure_kind is None
    assert loaded.steps["acc_001"].summary_schema is None


def test_a_retried_step_that_succeeds_drops_its_old_failure_kind(tmp_path: Path):
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_failed("acc_001", "boom", kind="provider")
    ledger.mark_done("acc_001", summary={"beats": 1}, summary_schema=1)

    assert ledger.steps["acc_001"].failure_kind is None
    # The snapshot's own ledger rewrites a step as done; it must not carry the
    # kind either (`dump_to(override_done=...)`).
    ledger.mark_failed("acc_002", "boom", kind="provider")
    ledger.dump_to(tmp_path / "snapshot" / "ledger.json", override_done="acc_002")
    snapshot = json.loads(
        (tmp_path / "snapshot" / "ledger.json").read_text(encoding="utf-8")
    )
    assert "failure_kind" not in snapshot["steps"]["acc_002"]


def test_a_failed_step_without_artifacts_stays_empty(tmp_path: Path):
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_running("acc_001")
    ledger.mark_failed("acc_001", "RuntimeError: boom")

    assert ledger.steps["acc_001"].artifacts == {}


def test_resume_decisions(tmp_path: Path):
    plan = _plan()
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_done("acc_001")
    ledger.mark_running("acc_002")

    assert ledger.decision_for("acc_001") == "skip"
    assert ledger.decision_for("acc_002") == "run"
    with pytest.raises(
        UnsafeRunResumeError,
        match="run_id='run_001'.*acc_002.*running",
    ):
        ledger.assert_resumable(plan)


def test_forget_drops_a_step_and_reloads(tmp_path: Path):
    """`--rerun-probe-steps` forgets a finished probe so the resume runs it
    again; the reload proves `current_step` went with it, which is what the
    loader refuses to see pointing at a record that is gone."""
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_done("acc_001")
    ledger.mark_running("fin_09")

    ledger.forget("fin_09")

    assert ledger.current_step is None
    assert ledger.decision_for("fin_09") == "run"
    reloaded = RunLedger(tmp_path / "run", "run_001")
    assert set(reloaded.steps) == {"acc_001"}


def test_a_stopped_probe_is_rerun_not_refused(tmp_path: Path):
    """A probe's start point is an immutable pin, so a probe left at
    `running` / `failed` is nothing to roll back — it is simply run again.
    Only a timeline step makes the ledger refuse.
    """
    base = _plan()
    acc = base.steps[1].model_dump(mode="json")
    plan = base.model_copy(
        update={
            "steps": [
                *base.steps,
                RunStep(
                    **{
                        **acc,
                        "step_id": "pretest_W_A",
                        "kind": "pre_event_probe",
                        "acc_num": None,
                        "before_acc_num": 2,
                        "benchmark_session_timestamp": "2026-10-01T18:00:00+00:00",
                        "memory_mode": "read_only",
                        "stage_policy": "discard",
                    }
                ),
                RunStep(
                    **{
                        **acc,
                        "step_id": "fin_01",
                        "kind": "final_probe",
                        "acc_num": None,
                        "benchmark_session_timestamp": "2026-10-01T21:00:00+00:00",
                        "memory_mode": "read_only",
                        "stage_policy": "discard",
                    }
                ),
            ]
        }
    )
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_done("acc_001")
    ledger.mark_done("acc_002")
    ledger.mark_failed("pretest_W_A", "ix contract failure", kind="ix_contract")
    ledger.mark_running("fin_01")

    ledger.assert_resumable(plan)
    assert ledger.decision_for("pretest_W_A") == "run"
    assert ledger.decision_for("fin_01") == "run"

    # A timeline step in the same state is still refused.
    ledger.mark_failed("acc_002", "died mid-step")
    with pytest.raises(UnsafeRunResumeError, match="acc_002"):
        ledger.assert_resumable(plan)


def test_skipped_step_stays_skipped_on_resume(tmp_path: Path):
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_skipped("acc_001")

    assert ledger.decision_for("acc_001") == "skip"


def test_ledger_rejects_different_run_id(tmp_path: Path):
    run_dir = tmp_path / "run"
    RunLedger(run_dir, "run_001")
    path = run_dir / "ledger.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "run_other"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        RunLedgerLoadError,
        match="expected_run_id='run_001'.*value='run_other'",
    ):
        RunLedger(run_dir, "run_001")


def test_ledger_rejects_unknown_step_status(tmp_path: Path):
    run_dir = tmp_path / "run"
    RunLedger(run_dir, "run_001")
    path = run_dir / "ledger.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["steps"] = {"acc_001": {"status": "maybe"}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        RunLedgerLoadError,
        match="step_id='acc_001'.*value='maybe'",
    ):
        RunLedger(run_dir, "run_001")


def test_progress_lines_are_concise():
    plan = _plan()
    start = format_start(plan, start_time=datetime(2026, 5, 14, 10, 0, 0).astimezone())
    running = format_step_running(1, 2, plan.steps[0], plan.memory_condition)

    assert "run=run_001" in start
    assert "persona=user_a" in start
    assert "memory=mem0" in start
    assert "hello" not in running
    assert running.endswith("mem0 rw running")


def test_dump_to_override_done_does_not_mutate_self(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path, "run_x")
    ledger.mark_running("acc_001")

    assert not list(tmp_path.glob("*.tmp"))
    reloaded = RunLedger(tmp_path, "run_x")
    assert reloaded.steps["acc_001"].status == "running"

    other = tmp_path / "snapshot" / "ledger.json"
    ledger.dump_to(other, override_done="acc_001")

    snapshot = json.loads(other.read_text(encoding="utf-8"))
    assert snapshot["steps"]["acc_001"]["status"] == "done"
    assert snapshot["steps"]["acc_001"]["ended_at"]
    assert snapshot["current_step"] is None

    assert ledger.steps["acc_001"].status == "running"
    assert ledger.current_step == "acc_001"
    mainline = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert mainline["steps"]["acc_001"]["status"] == "running"
    assert mainline["current_step"] == "acc_001"


def test_replace_from_rejects_foreign_run_id(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "mine", "run_x")
    ledger.mark_running("acc_001")
    before = (tmp_path / "mine" / "ledger.json").read_text(encoding="utf-8")

    foreign = RunLedger(tmp_path / "theirs", "run_y")
    foreign.mark_done("acc_001")

    with pytest.raises(RunLedgerLoadError):
        ledger.replace_from(tmp_path / "theirs" / "ledger.json")

    assert (tmp_path / "mine" / "ledger.json").read_text(encoding="utf-8") == before
    assert ledger.steps["acc_001"].status == "running"


def _plan_with_probes() -> RunPlan:
    """acc_001 acc_002 + a pre-event probe and a final, as an attach freezes.

    Sessions the run really executes, not the gap days `_plan` builds: a
    placeholder holds an acc slot and runs nothing, so it never reaches the
    ledger and cannot be what an attach is waiting for.
    """
    base = _plan()
    acc = base.steps[1].model_dump(mode="json")
    probe = {
        **acc,
        "acc_num": None,
        "placeholder": False,
        "memory_mode": "read_only",
        "stage_policy": "discard",
    }
    return base.model_copy(
        update={
            "steps": [
                *(
                    RunStep(
                        **{**step.model_dump(mode="json"), "placeholder": False}
                    )
                    for step in base.steps
                ),
                RunStep(
                    **{
                        **probe,
                        "step_id": "pre_01",
                        "kind": "pre_event_probe",
                        "before_acc_num": 2,
                        "benchmark_session_timestamp": "2026-10-01T18:00:00+00:00",
                    }
                ),
                RunStep(
                    **{
                        **probe,
                        "step_id": "fin_01",
                        "kind": "final_probe",
                        "benchmark_session_timestamp": "2026-10-01T21:00:00+00:00",
                    }
                ),
            ]
        }
    )


def test_attaching_probes_to_an_unfinished_timeline_is_refused(tmp_path: Path):
    """A probe is scored as the state of the run at its cut point, so a
    timeline that never reached the end has no such state to measure. The
    pre-flight `assert_resumable` is skipped as soon as the run has
    checkpoints, and an attach would then quietly take the rollback path
    instead."""
    plan = _plan_with_probes()
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_done("acc_001")

    with pytest.raises(UnsafeRunResumeError, match="acc_002.*never ran"):
        ledger.assert_probe_attach_ready(plan)

    ledger.mark_done("acc_002")
    ledger.assert_probe_attach_ready(plan)


def test_attaching_probes_over_a_step_that_died_mid_way_is_refused(
    tmp_path: Path,
):
    plan = _plan_with_probes()
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_done("acc_001")
    ledger.mark_failed("acc_002", "died mid-step")

    with pytest.raises(UnsafeRunResumeError, match="acc_002"):
        ledger.assert_probe_attach_ready(plan)


def test_continuing_a_timeline_keeps_the_rollback_path(tmp_path: Path):
    """No probe left to run: the invocation is not an attach, and the refusal
    above must not take the rollback path's place."""
    plan = _plan_with_probes()
    ledger = RunLedger(tmp_path / "run", "run_001")
    ledger.mark_done("pre_01")
    ledger.mark_done("fin_01")
    ledger.mark_failed("acc_002", "died mid-step")

    ledger.assert_probe_attach_ready(plan)
