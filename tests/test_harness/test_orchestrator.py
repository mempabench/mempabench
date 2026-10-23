import asyncio
import hashlib
import json
import os
import re
import signal
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from harness.memory_runtime import MemoryRuntime
from harness.memory_io import MemoryIORecorder
from harness import orchestrator as orchestrator_module
from harness.orchestrator import (
    LEDGER_OVERLAY_KEY,
    Orchestrator,
    OrchestratorConfig,
)
from harness.pa_factory import (
    OpenHoldEpisodeError,
    assert_benchmark_action_type_coverage,
)
from harness import checkpoint as checkpoint_module
from harness.checkpoint import CheckpointManager
from harness.run_config import FailurePolicy
from harness.run_plan import (
    MEMORY_POOL_VERSION,
    Mem0RuntimeConfig,
    RunPlan,
    RunStep,
)
from harness.run_ledger import (
    RunLedger,
    UnsafeRunResumeError,
    format_step_running,
)
from harness.session_executor import (
    FAILED_STEP_ARTIFACTS_ATTR,
    SUMMARY_SCHEMA,
    SessionExecutionResult,
    failed_step_artifacts,
)
from harness.state_runtime import StageRuntime
from memory.results import BackendHealth, ConsistencyModel, HealthStatus
from pa.agent.loop import AgentProviderError
from pa.session.manager import SessionManager, SessionNotFoundError
from pa.providers.request_policy import resolve_model_request_policy
from simulator.memory import UserMemoryStore
from simulator.output import SessionInvalidError, SimulatorOutputContractError
from simulator.profiles import load_simulator_profile
from simulator.judge.client import JudgeUnavailable
from scripts.run_orchestrator import GracefulStop


class FakeExecutor:
    def __init__(self, fail_step: str | None = None):
        self.calls = []
        self.loop_contexts = []
        self.fail_step = fail_step

    async def execute(
        self,
        step,
        *,
        steps_dir,
        stage_runtime=None,
        extra_meta=None,
        loop_context=None,
    ):
        self.calls.append((step.step_id, stage_runtime.state_root, extra_meta or {}))
        self.loop_contexts.append(loop_context)
        (stage_runtime.state_root / "marker.txt").write_text(step.step_id, encoding="utf-8")
        step_dir = Path(steps_dir) / step.step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        artifact = step_dir / "meta.yaml"
        artifact.write_text("ok: true\n", encoding="utf-8")
        if step.step_id == self.fail_step:
            # The real executor writes the artifacts before it re-raises and
            # hands their paths over on the exception.
            error = RuntimeError("failed step")
            setattr(error, FAILED_STEP_ARTIFACTS_ATTR, {"meta": str(artifact)})
            raise error
        return SessionExecutionResult(
            artifacts={"meta": str(artifact)},
            beats=1,
            tool_calls=0,
            elapsed_s=0.1,
        )


class InSceneIXExhaustionExecutor(FakeExecutor):
    """One executor call: the wrapper already spent all three responses."""

    async def execute(self, step, *, steps_dir, stage_runtime=None, **kwargs):
        self.calls.append((step.step_id, stage_runtime.state_root, kwargs))
        step_dir = Path(steps_dir) / step.step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        artifact = step_dir / "failed_turn_events.json"
        artifact.write_text(
            json.dumps({
                "ix_contract_exhaustion": {
                    "attempts": 3,
                    "contract": "gate",
                    "operation": "gate_confirmation_message",
                }
            }),
            encoding="utf-8",
        )
        (stage_runtime.state_root / "from_skipped.txt").write_text(
            step.step_id, encoding="utf-8"
        )
        error = AgentProviderError("PA turn failed the IX gate contract")
        error.ix_contract_exhaustion = {
            "attempts": 3,
            "contract": "gate",
            "operation": "gate_confirmation_message",
        }
        setattr(
            error,
            FAILED_STEP_ARTIFACTS_ATTR,
            {"failed_turn_events": str(artifact)},
        )
        raise error


class FakeMemoryAdapter:
    backend_name = "fake"

    def __init__(self):
        self.modes = []
        self.setup_calls = []

    async def setup_scope(self, scope):
        self.setup_calls.append(scope)
        return None

    async def set_mode(self, mode):
        self.modes.append(mode)
        return None

    async def health(self):
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=1.0,
            consistency_model=ConsistencyModel.immediate,
        )


class UnavailableMemoryAdapter(FakeMemoryAdapter):
    async def health(self):
        return BackendHealth(
            status=HealthStatus.unavailable,
            backend_name=self.backend_name,
            latency_ms=1.0,
            consistency_model=ConsistencyModel.immediate,
            warnings=["connection refused"],
        )


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "fixtures"
    user = root / "user_a"
    user.mkdir(parents=True, exist_ok=True)
    (user / "base.txt").write_text("base", encoding="utf-8")
    catalog = tmp_path / "data/state_tools/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "tools:\n  document_read:\n    status: implemented\n",
        encoding="utf-8",
    )
    for script_name in ("acc_001", "acc_002", "pretest_W_A"):
        probe_kind = (
            "  probe_kind: pre_event\n"
            if script_name == "pretest_W_A"
            else ""
        )
        (tmp_path / f"{script_name}.yaml").write_text(
            (
                "meta:\n"
                "  artifact_kind: normal_session\n"
                "  tool_status: world_toolbox\n"
                "  persona_id: user_a\n"
                f"  session_id: {script_name}\n"
                f"{probe_kind}"
                "  target_cell: work_verbosity\n"
            ),
            encoding="utf-8",
        )
    # `resolve_session_gate` runs on every accumulation step and fails closed:
    # an acc session whose outline is not on disk is an error, not a gateless
    # session. These steps exercise checkpointing, not gating, so they are
    # `verbosity` — a compressed attribute, the one shape that is legitimately
    # gateless now that `probe.no_gate_reason` is abolished.
    outline = tmp_path / "Authoring/outline/drafts/user_a/outline.yaml"
    outline.parent.mkdir(parents=True, exist_ok=True)
    outline.write_text(
        "sessions:\n"
        + "".join(
            f"  - identity:\n"
            f"      acc_num: {acc_num}\n"
            f"    probe:\n"
            f"      why_both_viable: compressed, so no gate\n"
            for acc_num in (1, 2)
        ),
        encoding="utf-8",
    )
    return root


def _write_probe_script(tmp_path: Path, step: RunStep) -> None:
    """Write the authored pairing declared by a test's frozen probe step."""
    assert step.probe_overlay is not None
    meta = {
        # These orchestrator fixtures deliberately use the existing gateless
        # normal-session shell; only the probe pairing metadata is relevant to
        # the frozen-overlay boundary tested here.
        "artifact_kind": "normal_session",
        "tool_status": "world_toolbox",
        "persona_id": step.persona_id,
        # Keep these orchestration-only fixtures out of the probe gate path;
        # dedicated gate tests author real ``pre_*``/``fin_*`` sessions and
        # their Outline entries.
        "session_id": f"fixture_{step.step_id}",
        "probe_kind": (
            "pre_event" if step.kind == "pre_event_probe" else "final"
        ),
        "target_cell": "work_verbosity",
    }
    if step.probe_overlay.owner_probe_id != step.step_id:
        meta["pairs_with"] = step.probe_overlay.owner_probe_id
    path = tmp_path / step.script_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"meta": meta}, sort_keys=False), encoding="utf-8")


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
                "script_path": "acc_001.yaml",
                "context": "work",
                "target_cell": "cell_a",
                    "acc_num": 1,
                    "benchmark_session_timestamp": (
                        "2026-10-01T09:00:00+00:00"
                    ),
                    "memory_mode": "read_write",
                "stage_policy": "commit",
            },
            {
                "step_id": "pretest_W_A",
                "kind": "pre_event_probe",
                "persona_id": "user_a",
                "script_path": "pretest_W_A.yaml",
                "probe_overlay": {"owner_probe_id": "pretest_W_A"},
                "context": "work",
                "target_cell": "cell_a",
                    "before_acc_num": 2,
                    "benchmark_session_timestamp": (
                        "2026-10-01T15:00:00+00:00"
                    ),
                    "memory_mode": "read_only",
                "stage_policy": "discard",
            },
        ],
    )


@pytest.mark.asyncio
async def test_orchestrator_runs_steps_and_applies_policies(tmp_path: Path):
    plan = _plan()
    adapter = FakeMemoryAdapter()
    lines = []
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=tmp_path / "runs" / plan.run_id,
            project_root=tmp_path,
            checkpoint_manager=manager,
        ),
        ledger=ledger,
        executor=FakeExecutor(),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            adapter,
            run_id=plan.run_id,
            persona_id=plan.persona_id,
            backend_namespace="mempa_run_001_mem0",
        ),
        progress=lines.append,
    )

    result = await orchestrator.run()

    assert result.executed == ["acc_001", "pretest_W_A"]
    assert len(adapter.setup_calls) == 3
    assert [mode.read_only for mode in adapter.modes] == [False, True]
    canonical_marker = tmp_path / "runs" / plan.run_id / "canonical_stage" / "marker.txt"
    assert canonical_marker.read_text(encoding="utf-8") == "acc_001"
    assert lines[0].startswith("start ")
    assert any("acc_001 accumulation" in line for line in lines)
    assert all("hello" not in line for line in lines)
    first_meta = orchestrator.executor.calls[0][2]
    assert first_meta["run_id"] == plan.run_id
    assert first_meta["pa_session_key"] == "run_001_main"
    assert first_meta["script_session_id"] == "acc_001"
    assert (
        first_meta["memory_backend_namespace"]
        == "mempa_run_001_mem0"
    )


def _verify_outreach_orchestrator(tmp_path: Path, lines: list[str]) -> Orchestrator:
    plan = _plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    return Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=tmp_path / "runs" / plan.run_id,
            project_root=tmp_path,
            checkpoint_manager=manager,
        ),
        ledger=ledger,
        executor=FakeExecutor(),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            FakeMemoryAdapter(),
            run_id=plan.run_id,
            persona_id=plan.persona_id,
        ),
        progress=lines.append,
    )


@pytest.mark.asyncio
async def test_a_bad_verify_outreach_value_stops_the_run_at_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed, beside the action-type check and for the same reason:
    before the world is touched and before a provider call is paid for."""
    monkeypatch.setenv("MEMPA_VERIFY_OUTREACH", "yes")
    lines: list[str] = []
    orchestrator = _verify_outreach_orchestrator(tmp_path, lines)

    with pytest.raises(RuntimeError, match="must be 'send' or 'suppress'"):
        await orchestrator.run()

    assert orchestrator.executor.calls == []


@pytest.mark.asyncio
async def test_a_verification_run_announces_itself_before_the_first_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMPA_VERIFY_OUTREACH", "suppress")
    lines: list[str] = []
    orchestrator = _verify_outreach_orchestrator(tmp_path, lines)

    await orchestrator.run()

    assert lines[0] == (
        "MEMPA_VERIFY_OUTREACH=suppress — VERIFICATION RUN, the PA's outreach "
        "decision is overridden and the run is not scored"
    )


@pytest.mark.asyncio
async def test_orchestrator_stops_before_first_step_when_memory_is_unavailable(
    tmp_path: Path,
) -> None:
    plan = _plan()
    executor = FakeExecutor()
    audit_path = tmp_path / "runs" / plan.run_id / "memory_io.jsonl"
    orchestrator = Orchestrator(
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
            UnavailableMemoryAdapter(),
            run_id=plan.run_id,
            persona_id=plan.persona_id,
        ),
        memory_io_recorder=MemoryIORecorder(
            audit_path,
            run_id=plan.run_id,
        ),
        progress=lambda _line: None,
    )

    with pytest.raises(
        RuntimeError,
        match="backend='fake'.*status='unavailable'.*connection refused",
    ):
        await orchestrator.run()

    assert executor.calls == []
    assert orchestrator.ledger.steps == {}
    health_record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert health_record["operation"] == "health"
    assert health_record["status"] == "unavailable"
    assert health_record["backend_response"]["warnings"] == [
        "connection refused"
    ]


@pytest.mark.asyncio
async def test_orchestrator_uses_plan_run_id_and_one_main_pa_session_for_accumulation(tmp_path: Path):
    policy = resolve_model_request_policy(
        "openrouter", "test-pa"
    ).model_copy(
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
    plan = RunPlan(
        run_id="user_a_NanoBotFileMemory_openai_gpt55_cfgaaaaaaaaaa_r01",
        persona_id="user_a",
        memory_condition="nanobot_file_memory",
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base="https://openrouter.ai/api/v1",
        pa_model="test-pa",
        pa_request_policy=policy.snapshot(),
        consolidation_model="test-consolidation",
        consolidation_request_policy=consolidation_policy.snapshot(),
        steps=[
            {
                "step_id": "acc_001",
                "kind": "accumulation",
                "persona_id": "user_a",
                "script_path": "acc_001.yaml",
                "context": "work",
                "target_cell": "cell_a",
                    "acc_num": 1,
                    "benchmark_session_timestamp": (
                        "2026-10-01T09:00:00+00:00"
                    ),
                    "memory_mode": "read_write",
                "stage_policy": "commit",
            },
            {
                "step_id": "acc_002",
                "kind": "accumulation",
                "persona_id": "user_a",
                "script_path": "acc_002.yaml",
                "context": "personal",
                "target_cell": "cell_b",
                    "acc_num": 2,
                    "benchmark_session_timestamp": (
                        "2026-10-01T15:00:00+00:00"
                    ),
                    "memory_mode": "read_write",
                "stage_policy": "commit",
            },
        ],
    )
    executor = FakeExecutor()
    orchestrator = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=tmp_path / "runs" / plan.run_id,
            pa_workspace=tmp_path / "runs" / plan.run_id / "pa_workspace" / plan.persona_id,
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
        memory_runtime=MemoryRuntime(None, run_id=plan.run_id, persona_id=plan.persona_id),
        progress=lambda _line: None,
    )

    await orchestrator.run()

    metas = [call[2] for call in executor.calls]
    assert [meta["run_id"] for meta in metas] == [plan.run_id, plan.run_id]
    assert [meta["pa_session_key"] for meta in metas] == [
        "user_a_NanoBotFileMemory_openai_gpt55_cfgaaaaaaaaaa_r01_main",
        "user_a_NanoBotFileMemory_openai_gpt55_cfgaaaaaaaaaa_r01_main",
    ]
    assert [meta["script_session_id"] for meta in metas] == ["acc_001", "acc_002"]
    assert all(meta["pa_workspace"].endswith("pa_workspace/user_a") for meta in metas)
    session_files = list(
        (tmp_path / "runs" / plan.run_id / "pa_workspace" / plan.persona_id / "sessions").glob("*.jsonl")
    )
    assert len(session_files) == 1
    assert SessionManager(
        tmp_path / "runs" / plan.run_id / "pa_workspace" / plan.persona_id
    ).get(plan.main_pa_session_key()).key == plan.main_pa_session_key()


@pytest.mark.asyncio
async def test_orchestrator_resume_fails_when_main_session_is_missing(tmp_path: Path):
    plan = _plan()
    run_dir = tmp_path / "runs" / plan.run_id
    workspace = run_dir / "pa_workspace" / plan.persona_id
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    # The first run creates the main session itself.
    SessionManager(workspace)._get_session_path(plan.main_pa_session_key()).unlink()
    config = OrchestratorConfig(
        run_dir=run_dir,
        pa_workspace=workspace,
        project_root=tmp_path,
        checkpoint_manager=manager,
    )
    first = Orchestrator(
        plan=plan,
        config=config,
        ledger=ledger,
        executor=FakeExecutor(),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(None, run_id=plan.run_id, persona_id=plan.persona_id),
        progress=lambda _line: None,
    )
    await first.run()
    SessionManager(workspace)._get_session_path(plan.main_pa_session_key()).unlink()

    resumed = Orchestrator(
        plan=plan,
        config=config,
        executor=FakeExecutor(),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(None, run_id=plan.run_id, persona_id=plan.persona_id),
        progress=lambda _line: None,
    )

    with pytest.raises(SessionNotFoundError, match=plan.main_pa_session_key()):
        await resumed.run()


@pytest.mark.asyncio
async def test_orchestrator_skips_done_steps_on_resume(tmp_path: Path):
    plan = _plan()
    run_dir = tmp_path / "runs" / plan.run_id
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    first = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=run_dir, project_root=tmp_path, checkpoint_manager=manager
        ),
        ledger=ledger,
        executor=FakeExecutor(),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(None, run_id=plan.run_id, persona_id=plan.persona_id),
        progress=lambda _line: None,
    )
    await first.run()
    second_executor = FakeExecutor()
    second_ledger = RunLedger(run_dir, plan.run_id)
    second = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=run_dir,
            project_root=tmp_path,
            checkpoint_manager=_file_memory_manager(run_dir, second_ledger),
        ),
        ledger=second_ledger,
        executor=second_executor,
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(None, run_id=plan.run_id, persona_id=plan.persona_id),
        progress=lambda _line: None,
    )

    result = await second.run()

    assert result.skipped == ["acc_001", "pretest_W_A"]
    assert second_executor.calls == []


@pytest.mark.asyncio
async def test_orchestrator_marks_failed_step(tmp_path: Path):
    plan = _plan()
    run_dir = tmp_path / "runs" / plan.run_id
    orchestrator = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(run_dir=run_dir, project_root=tmp_path),
        executor=FakeExecutor(fail_step="acc_001"),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(None, run_id=plan.run_id, persona_id=plan.persona_id),
        progress=lambda _line: None,
    )

    with pytest.raises(RuntimeError, match="failed step"):
        await orchestrator.run()

    assert orchestrator.ledger.steps["acc_001"].status == "failed"
    # The failed step's audit trail is named in the ledger, so `inspect` can
    # point a reviewer at it instead of only reporting the error string — as a
    # path within the run, which survives the run being archived.
    assert orchestrator.ledger.steps["acc_001"].artifacts == {
        "meta": "steps/acc_001/meta.yaml"
    }
    # A bare `RuntimeError` is nothing the vocabulary names, so it lands as the
    # word that means "a human has to read this".
    assert orchestrator.ledger.steps["acc_001"].failure_kind == "other"

    resumed_executor = FakeExecutor()
    resumed = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(run_dir=run_dir, project_root=tmp_path),
        executor=resumed_executor,
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
    )

    with pytest.raises(
        UnsafeRunResumeError,
        match="acc_001.*failed.*new run_id",
    ):
        await resumed.run()
    assert resumed_executor.calls == []


class InvalidSessionExecutor(FakeExecutor):
    """The executor of a step the simulator invalidated."""

    async def execute(self, step, **kwargs):
        try:
            return await super().execute(step, **kwargs)
        except RuntimeError as error:
            invalid = SessionInvalidError(
                "beat_ask_unfulfilled", beat="b3_probe", attempts=3
            )
            setattr(
                invalid,
                FAILED_STEP_ARTIFACTS_ATTR,
                failed_step_artifacts(error),
            )
            raise invalid from None


@pytest.mark.asyncio
async def test_an_invalid_session_stops_the_run_and_names_the_rule(tmp_path: Path):
    """The simulator failed as an instrument, so the step is dead and the run
    stops. The raise is what gives the process a non-zero rc; the ledger is
    what a batch read counts, so it carries the typed kind and the rule.
    """
    plan = _plan()
    run_dir = tmp_path / "runs" / plan.run_id
    orchestrator = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(run_dir=run_dir, project_root=tmp_path),
        executor=InvalidSessionExecutor(fail_step="acc_001"),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            None, run_id=plan.run_id, persona_id=plan.persona_id
        ),
        progress=lambda _line: None,
    )

    with pytest.raises(SessionInvalidError, match="beat_ask_unfulfilled"):
        await orchestrator.run()

    record = orchestrator.ledger.steps["acc_001"]
    assert record.status == "failed"
    assert record.failure_kind == "session_invalid"
    assert "rule=beat_ask_unfulfilled" in record.error
    assert "beat=b3_probe" in record.error
    # The discarded session's audit trail is still reachable.
    assert record.artifacts == {"meta": "steps/acc_001/meta.yaml"}


class FakeCheckpointManager:
    """Records the publication calls the orchestrator makes at a cut point."""

    def __init__(self, calls: list, *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail
        self.published: list[str] = []
        #: The summary each cut point was published with, so a test can check
        #: the snapshot's own ledger would carry the step's flip.
        self.saved_summaries: dict[str, dict | None] = {}

    def save(
        self,
        checkpoint_id: str,
        *,
        pinned: bool,
        summary: dict | None = None,
        summary_schema: int | None = None,
    ):
        self.calls.append(("save", checkpoint_id, pinned))
        self.saved_summaries[checkpoint_id] = summary
        if self.fail:
            raise RuntimeError("checkpoint disk full")
        self.published.append(checkpoint_id)
        return Path(checkpoint_id)

    def list_checkpoints(self) -> list[str]:
        return list(self.published)

    def pending_recovery(self) -> dict | None:
        """No rollback recovery is open: this fake never restores."""
        return None

    def metadata(self, checkpoint_id: str) -> dict:
        return {
            "checkpoint_id": checkpoint_id,
            "pinned": True,
            "timestamp": checkpoint_id,
        }

    def branch_from(self, checkpoint_id: str, branch_dir: Path) -> Path:
        self.calls.append(("branch", checkpoint_id))
        Path(branch_dir).mkdir(parents=True, exist_ok=True)
        return Path(branch_dir)

    def build_branch_adapter(self, branch_dir: Path):
        # Stands in for a manager that restored this branch's namespace and
        # built an adapter on it. Returning None here would be the missing-adapter
        # defect itself, and the orchestrator now refuses it for an external
        # condition — which is what these plans declare.
        return FakeMemoryAdapter()


class _FakeTurnJournal:
    """The records a real journal holds for this fake session's turns.

    Built through the same renderer the loop uses, so a pinned flush sees what
    a live run would have left it. The flush reads the journal, not
    `session.messages`, so a fake session needs one.
    """

    def __init__(self, turns: list[list[dict]]) -> None:
        from memory.event_writer import build_turn_pool

        self._records: list = []
        index = 0
        for position, turn in enumerate(turns):
            pool = build_turn_pool([dict(message) for message in turn])
            self._records.append(SimpleNamespace(
                turn_id=f"turn_{position}",
                span=(index, index + len(turn)),
                timestamp=str(turn[0].get("timestamp") or ""),
                messages=pool.messages,
                state="committed",
                file_consolidated=False,
                dropped_from_files=False,
            ))
            index += len(turn)

    def records(self) -> list:
        return list(self._records)

    def resume_pending(self):
        """Nothing to replay: these stand for turns the adapter already had."""
        return SimpleNamespace(records=[], turn_ids=[])

    def pending_records(self) -> list:
        return []

    def mark_file_consolidated(self, turn_ids) -> None:
        for record in self._records:
            if record.turn_id in set(turn_ids):
                record.file_consolidated = True


class FakePACheckpoint:
    def __init__(
        self,
        calls: list,
        *,
        policy=None,
        message_count: int = 4,
        memory_file: Path | None = None,
    ) -> None:
        from memory.adapter import PromptHistoryEvictionPolicy

        self.prompt_history_eviction_policy = (
            policy or PromptHistoryEvictionPolicy.after_file_consolidation
        )
        self.calls = calls
        self.archived: list[list] = []
        self.saved = 0
        self.memory_file = memory_file
        self._session = SimpleNamespace(
            messages=[
                {"role": "user", "content": f"turn {i}"}
                for i in range(message_count)
            ],
            file_consolidated_end_index=0,
            adapter_committed_end_index=0,
            prompt_history_start_index=0,
        )

    #: How the fake session's messages group into turns. One message per turn
    #: unless a test sets it, which is what a bare `turn N` session looks like.
    turn_groups: list[list[dict]] | None = None

    async def drain_background_tasks(self) -> None:
        self.calls.append(("drain",))

    def checkpoint_session(self):
        return self._session

    def turn_journal(self):
        if getattr(self, "_journal", None) is None:
            self._journal = _FakeTurnJournal(
                self.turn_groups
                or [[message] for message in self._session.messages]
            )
        return self._journal

    async def archive_messages_to_files(self, messages) -> bool:
        self.calls.append(("flush", len(messages)))
        self.archived.append(list(messages))
        if self.memory_file is not None:
            with self.memory_file.open("a", encoding="utf-8") as handle:
                for message in messages:
                    handle.write(f"{message['content']}\n")
        return True

    def save_session(self, session) -> None:
        self.saved += 1


def _two_acc_plan() -> RunPlan:
    """acc_001 (rolling) → acc_002 (pinned) → pretest_W_A probe."""
    plan = _plan()
    steps = [
        plan.steps[0].model_dump(mode="json"),
        {
            **plan.steps[0].model_dump(mode="json"),
            "step_id": "acc_002",
            "script_path": "acc_002.yaml",
            "acc_num": 2,
            "benchmark_session_timestamp": "2026-10-01T11:00:00+00:00",
        },
        plan.steps[1].model_dump(mode="json"),
    ]
    return plan.model_copy(update={"steps": [RunStep(**s) for s in steps]})


def _orchestrator(
    tmp_path: Path,
    plan: RunPlan,
    *,
    checkpoint_manager,
    pa_checkpoint,
    stop_requested=None,
) -> Orchestrator:
    return Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=tmp_path / "runs" / plan.run_id,
            project_root=tmp_path,
            checkpoint_manager=checkpoint_manager,
            stop_requested=stop_requested,
        ),
        executor=FakeExecutor(),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            FakeMemoryAdapter(),
            run_id=plan.run_id,
            persona_id=plan.persona_id,
        ),
        pa_checkpoint=pa_checkpoint,
        progress=lambda _line: None,
    )


@pytest.mark.asyncio
async def test_step_is_marked_done_only_after_checkpoint_published(
    tmp_path: Path,
):
    plan = _two_acc_plan()
    calls: list = []
    pa = FakePACheckpoint(calls)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=FakeCheckpointManager(calls, fail=True),
        pa_checkpoint=pa,
    )

    with pytest.raises(RuntimeError, match="checkpoint disk full"):
        await orchestrator.run()

    # The publication failed, so the step must not read as done anywhere.
    assert orchestrator.ledger.steps["acc_001"].status == "failed"
    assert orchestrator.config.checkpoint_manager.list_checkpoints() == []
    # A rolling cut point waits but never consolidates.
    assert calls == [("drain",), ("save", "acc_001", False)]
    assert pa.archived == []
    assert pa._session.file_consolidated_end_index == 0
    assert pa._session.prompt_history_start_index == 0
    assert pa._session.adapter_committed_end_index == 0

    # Happy path: wait → (pinned only) flush → save, then done.
    plan = _two_acc_plan()
    calls = []
    pa = FakePACheckpoint(calls)
    manager = FakeCheckpointManager(calls)
    ok = _orchestrator(
        tmp_path / "ok",
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=pa,
    )
    result = await ok.run()

    assert result.executed == ["acc_001", "acc_002", "pretest_W_A"]
    assert calls == [
        ("drain",),
        ("save", "acc_001", False),
        ("drain",),
        ("flush", 4),
        ("save", "acc_002", True),
        ("branch", "acc_002"),
    ]
    assert manager.list_checkpoints() == ["acc_001", "acc_002"]
    assert ok.ledger.steps["acc_002"].status == "done"
    # Which counting rules produced `tool_calls`, on the mainline and on the
    # probe branch alike — the number changed meaning once already and left
    # nothing behind that said so.
    assert ok.ledger.steps["acc_002"].summary_schema == SUMMARY_SCHEMA
    assert ok.ledger.steps["pretest_W_A"].summary_schema == SUMMARY_SCHEMA


@pytest.mark.asyncio
async def test_pinned_checkpoint_flushes_all_uncompacted_turns(tmp_path: Path):
    from memory.adapter import PromptHistoryEvictionPolicy

    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    memory_dir = run_dir / "pa_workspace" / plan.persona_id / "memory"
    memory_dir.mkdir(parents=True)
    memory_file = memory_dir / "HISTORY.md"
    memory_file.write_text("", encoding="utf-8")
    sim_dir = run_dir / "simulator_workspace" / plan.persona_id
    sim_dir.mkdir(parents=True)
    (sim_dir / "MEMORY.md").write_text("sim\n", encoding="utf-8")
    (run_dir / "memory_io.jsonl").write_text("", encoding="utf-8")
    (run_dir / "run_plan.yaml").write_text("run_id: run_001\n", encoding="utf-8")

    ledger = RunLedger(run_dir, plan.run_id)
    manager = CheckpointManager(
        run_dir,
        ledger=ledger,
        memory_condition="nanobot_file_memory",
        rolling_window=10,
    )
    calls: list = []
    pa = FakePACheckpoint(calls, message_count=3, memory_file=memory_file)

    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=pa,
    )
    orchestrator.ledger = ledger
    await orchestrator.run()

    # Exactly one forced consolidation, at the pinned step only.
    assert [c for c in calls if c[0] == "flush"] == [("flush", 3)]
    assert pa._session.file_consolidated_end_index == 3
    assert pa._session.prompt_history_start_index == 3
    assert pa.saved == 1

    snapshot_memory = (
        run_dir
        / "checkpoints"
        / "acc_002"
        / "pa_workspace"
        / plan.persona_id
        / "memory"
        / "HISTORY.md"
    )
    assert snapshot_memory.read_text(encoding="utf-8") == "turn 0\nturn 1\nturn 2\n"

    # A non-file backend is a no-op here, cursors included.
    plan_b = _two_acc_plan()
    calls_b: list = []
    pa_b = FakePACheckpoint(
        calls_b,
        policy=PromptHistoryEvictionPolicy.without_persistence,
        message_count=3,
    )
    other = _orchestrator(
        tmp_path / "nofile",
        plan_b,
        checkpoint_manager=FakeCheckpointManager(calls_b),
        pa_checkpoint=pa_b,
    )
    await other.run()

    assert [c for c in calls_b if c[0] == "flush"] == []
    assert pa_b._session.file_consolidated_end_index == 0
    assert pa_b._session.prompt_history_start_index == 0
    assert pa_b.saved == 0


# ── the pinned flush projects the same face as the per-turn path (T12) ──


IX_SELECTION_RESULT = json.dumps(
    {
        "type": "ix_preference_selection",
        "attribute": "response_length",
        "setting": "Brief",
        "evidence": "the user cut me off twice",
        "application": "kept the reply to one line",
    },
    ensure_ascii=False,
)


def _wrapper_turn_messages() -> list[dict]:
    """One turn shaped the way the live wrapper stores it in the session."""
    return [
        {
            "role": "user",
            "content": "Can you check the venue note?",
            "timestamp": "2026-10-01T09:00:00+00:00",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_ix",
                    "type": "function",
                    "function": {
                        "name": "IX_response_length",
                        "arguments": '{"setting": "Brief"}',
                    },
                },
                {
                    "id": "call_task",
                    "type": "function",
                    "function": {
                        "name": "mcp_state_document_read",
                        "arguments": '{"path": "venue.md"}',
                    },
                },
            ],
        },
        # The IX declaration's own result: attribute, setting, evidence.
        {
            "role": "tool",
            "tool_call_id": "call_ix",
            "name": "IX_response_length",
            "content": IX_SELECTION_RESULT,
            "interaction_item_id": "ix_item_7",
        },
        {
            "role": "tool",
            "tool_call_id": "call_task",
            "name": "mcp_state_document_read",
            "content": "Venue: the Blue Door, 7pm.",
        },
        {
            "role": "assistant",
            "content": "The Blue Door at 7pm.",
            "tools_used": ["IX_response_length", "mcp_state_document_read"],
            "timestamp": "2026-10-01T09:00:05+00:00",
        },
        {
            "role": "user",
            "content": "<runtime-context>hidden scaffold</runtime-context>",
            "_mempa_internal": True,
        },
    ]


async def _flush_one_wrapper_turn(tmp_path: Path) -> tuple[FakePACheckpoint, str]:
    """Run to the pinned cut point with a wrapper turn in the session."""
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    memory_dir = run_dir / "pa_workspace" / plan.persona_id / "memory"
    memory_dir.mkdir(parents=True)
    history = memory_dir / "HISTORY.md"
    history.write_text("", encoding="utf-8")

    calls: list = []
    pa = FakePACheckpoint(calls, memory_file=history)
    pa._session.messages = _wrapper_turn_messages()
    pa.turn_groups = [list(pa._session.messages)]
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=FakeCheckpointManager(calls),
        pa_checkpoint=pa,
    )
    await orchestrator.run()
    return pa, history.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_pinned_flush_hands_the_files_the_whole_turn(
    tmp_path: Path,
) -> None:
    """The flush reads the journal, so the files get the pool.

    It used to archive the two-message projection, and MEMORY.md never learned
    that the PA had looked the venue up or what it chose to do about response
    length. The PA's own IX call and its world traffic are ruled IN -- they are
    the PA's experience, and the ground truth reaches the PA through
    `metadata`, never through a message.
    """
    pa, history = await _flush_one_wrapper_turn(tmp_path)

    assert len(pa.archived) == 1
    blob = "\n".join(str(record["content"]) for record in pa.archived[0])
    assert "Can you check the venue note?" in blob
    assert "The Blue Door at 7pm." in blob
    assert "mcp_state_document_read" in blob
    assert "Venue: the Blue Door, 7pm." in blob
    # The cursors still cover every message.
    assert pa._session.file_consolidated_end_index == 6
    assert pa._session.prompt_history_start_index == 6


@pytest.mark.asyncio
async def test_no_wrapper_scaffolding_survives_the_pinned_flush(
    tmp_path: Path,
) -> None:
    """The OUT column, regrepped: carrier C and the injected scaffold.

    The IN column moved, so the old blanket `ix_` grep would now ban
    the PA's own choice. What stays banned is what exists only because a
    benchmark is running.
    """
    pa, history = await _flush_one_wrapper_turn(tmp_path)
    written = f"{history}\n{json.dumps(pa.archived, ensure_ascii=False)}".lower()

    for key in (
        "interaction_item_id",
        "_mempa_internal",
        "hidden scaffold",
        "runtime-context",
        "tools_used",
    ):
        assert key not in written, key
    # Not vacuous: the half ruled IN did survive the same flush.
    assert "mcp_state_document_read" in written


def _seed_run_dir(run_dir: Path, persona_id: str, session_key: str) -> Path:
    workspace = run_dir / "pa_workspace" / persona_id
    (workspace / "memory").mkdir(parents=True, exist_ok=True)
    (workspace / "memory" / "MEMORY.md").write_text("pa memory\n", encoding="utf-8")
    SessionManager(workspace).create(session_key)
    sim = run_dir / "simulator_workspace" / persona_id
    sim.mkdir(parents=True, exist_ok=True)
    (sim / "MEMORY.md").write_text("sim\n", encoding="utf-8")
    (run_dir / "canonical_stage").mkdir(parents=True, exist_ok=True)
    (run_dir / "canonical_stage" / "base.txt").write_text("base", encoding="utf-8")
    (run_dir / "memory_io.jsonl").write_text('{"seq": 1}\n', encoding="utf-8")
    # A borrowed start point is checked against the memory-pool grammar the
    # source run froze, so the seeded plan carries one as a real plan does.
    (run_dir / "run_plan.yaml").write_text(
        f"run_id: run_001\nmemory_pool_version: {MEMORY_POOL_VERSION}\n",
        encoding="utf-8",
    )
    return workspace


def _file_memory_manager(run_dir: Path, ledger: RunLedger) -> CheckpointManager:
    return CheckpointManager(
        run_dir,
        ledger=ledger,
        memory_condition="nanobot_file_memory",
        rolling_window=10,
    )


def _seeded_checkpoint_manager(
    tmp_path: Path, plan: RunPlan
) -> tuple[RunLedger, CheckpointManager]:
    """A run directory with everything a pinned cut point snapshots, and the
    manager that saves it; the ledger is shared with the orchestrator."""
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    return ledger, _file_memory_manager(run_dir, ledger)


@pytest.mark.asyncio
async def test_in_scene_ix_exhaustion_keeps_partial_state_and_runs_next_step(
    tmp_path: Path,
) -> None:
    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = _orchestrator(
        tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=FakePACheckpoint([])
    )
    orchestrator.config.extra_pinned_step_ids = frozenset({"acc_001"})
    orchestrator.config.failure_policy = FailurePolicy(auto_skip_on_ix_exhaustion=True)
    first = InSceneIXExhaustionExecutor()
    normal = FakeExecutor()

    class SequenceExecutor:
        async def execute(self, step, **kwargs):
            if step.step_id == "acc_001":
                return await first.execute(step, **kwargs)
            return await normal.execute(step, **kwargs)

    orchestrator.executor = SequenceExecutor()
    orchestrator.ledger = ledger

    result = await orchestrator.run()

    assert len(first.calls) == 1  # no outer scene retry
    assert normal.calls[0][0] == "acc_002"
    assert [call[0] for call in normal.calls].count("acc_002") == 1
    assert result.skipped == ["acc_001"]
    assert result.executed[0] == "acc_002"
    assert result.executed.count("acc_002") == 1
    assert ledger.steps["acc_001"].status == "skipped"
    assert ledger.steps["acc_002"].status == "done"
    assert sum(
        state.status == "done"
        for step_id, state in ledger.steps.items()
        if step_id.startswith("acc_")
    ) == 1
    assert (manager.run_dir / "canonical_stage/from_skipped.txt").read_text() == "acc_001"
    summary = ledger.steps["acc_001"].summary
    assert summary["skip_reason"] == "failure_policy.auto_skip_on_ix_exhaustion"
    assert summary["ix_in_scene_exhaustion"]["attempts"] == 3
    assert ledger.steps["acc_001"].failure_kind == "ix_runtime"
    assert (manager.run_dir / "checkpoints/acc_001").is_dir()
    assert manager.metadata("acc_001")["retained"] is True
    assert manager.metadata("acc_001")["pinned"] is True
    branch = manager.branch_from("acc_001", tmp_path / "skip_donor_branch")
    assert (branch / "canonical_stage/from_skipped.txt").read_text() == "acc_001"


@pytest.mark.asyncio
async def test_in_scene_ix_exhaustion_stops_the_run_when_auto_skip_is_off(tmp_path: Path) -> None:
    """With `auto_skip_on_ix_exhaustion: false` the run stops there."""
    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = _orchestrator(
        tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=FakePACheckpoint([])
    )
    orchestrator.config.failure_policy = FailurePolicy(auto_skip_on_ix_exhaustion=False)
    first = InSceneIXExhaustionExecutor()
    orchestrator.executor = first
    orchestrator.ledger = ledger

    with pytest.raises(AgentProviderError):
        await orchestrator.run()

    assert [call[0] for call in first.calls] == ["acc_001"]
    assert ledger.steps["acc_001"].status == "failed"
    assert "acc_002" not in ledger.steps
    assert not (manager.run_dir / "checkpoints/acc_001").exists()


@pytest.mark.asyncio
async def test_three_miss_acc_abandons_closed_hold_and_keeps_all_partial_state(
    tmp_path: Path,
) -> None:
    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    pa = HoldingPACheckpoint([], open_by_step={"acc_001": [("autonomy_level", "ix_send")]})
    orchestrator = _orchestrator(tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=pa)
    orchestrator.config.extra_pinned_step_ids = frozenset({"acc_001"})
    orchestrator.config.failure_policy = FailurePolicy(auto_skip_on_ix_exhaustion=True)
    first, normal = InSceneIXExhaustionExecutor(), FakeExecutor()

    class SequenceExecutor:
        async def execute(self, step, **kwargs):
            if step.step_id == "acc_001":
                return await first.execute(step, **kwargs)
            assert not pa.open_by_step
            return await normal.execute(step, **kwargs)

    orchestrator.executor = SequenceExecutor()
    orchestrator.ledger = ledger
    result = await orchestrator.run()
    assert len(first.calls) == 1
    assert pa.closed[:2] == ["acc_001", "acc_002"]
    assert result.skipped == ["acc_001"]
    assert ledger.steps["acc_001"].status == "skipped"
    assert ledger.steps["acc_002"].status == "done"
    assert (manager.run_dir / "canonical_stage/from_skipped.txt").read_text() == "acc_001"
    cleanup = json.loads((manager.run_dir / ledger.steps["acc_001"].artifacts["ix_session_cleanup"]).read_text())
    assert cleanup["session_cleanup_completed"] is True
    assert cleanup["banked_actions_released"] is False
    assert cleanup["disposition"] == "abandoned_unresolved_after_ix_exhaustion"
    assert cleanup["unresolved_hold_episodes"] == [{"attribute": "autonomy_level", "episode_id": "ix_send"}]
    assert manager.metadata("acc_001")["retained"] is True
    assert manager.metadata("acc_001")["terminal_status"] == "skipped"
    branch = manager.branch_from("acc_001", tmp_path / "open_hold_skip_branch")
    assert (branch / "canonical_stage/from_skipped.txt").read_text() == "acc_001"


@pytest.mark.asyncio
@pytest.mark.parametrize("proof", [None, {"attempts": 2, "contract": "gate", "operation": "gate_confirmation_message"}])
async def test_open_hold_without_exact_three_miss_proof_still_fails(tmp_path: Path, proof) -> None:
    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    pa = HoldingPACheckpoint([], open_by_step={"acc_001": [("autonomy_level", "ix_send")]})
    orchestrator = _orchestrator(tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=pa)
    orchestrator.config.failure_policy = FailurePolicy(auto_skip_on_ix_exhaustion=True)

    class NotExhausted(FakeExecutor):
        async def execute(self, step, **kwargs):
            error = AgentProviderError("PA turn failed the IX gate contract")
            if proof is not None:
                error.ix_contract_exhaustion = proof
            raise error

    orchestrator.executor = NotExhausted()
    orchestrator.ledger = ledger
    with pytest.raises(AgentProviderError):
        await orchestrator.run()
    assert pa.closed == ["acc_001"]
    assert ledger.steps["acc_001"].status == "failed"
    assert not (manager.run_dir / "checkpoints/acc_001").exists()


@pytest.mark.asyncio
async def test_forced_event_three_miss_open_hold_does_not_use_acc_skip(tmp_path: Path) -> None:
    _forced_artifacts(tmp_path)
    plan = _forced_plan()
    orchestrator = _forced_orchestrator(tmp_path, plan, InSceneIXExhaustionExecutor())
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    pa = HoldingPACheckpoint([], open_by_step={"evolv_01": [("autonomy_level", "ix_send")]})
    orchestrator.ledger = ledger
    orchestrator.config.checkpoint_manager = manager
    orchestrator.config.failure_policy = FailurePolicy(auto_skip_on_ix_exhaustion=True)
    orchestrator.pa_checkpoint = pa
    with pytest.raises(AgentProviderError):
        await orchestrator.run()
    assert ledger.steps["evolv_01"].status == "failed"
    assert pa.closed == ["evolv_01"]
    assert not (manager.run_dir / "checkpoints/evolv_01").exists()


@pytest.mark.asyncio
async def test_real_gate_hold_three_misses_closes_and_skips_without_replay(
    tmp_path: Path,
) -> None:
    from pa.providers.base import LLMResponse
    from test_ix_wrapper.conftest import (
        SCRIPT_SESSION,
        SESSION_KEY,
        ScriptProvider,
        build_loop,
        plan_call,
        run_turn,
        selection,
        task_call,
    )

    first = LLMResponse(content="Planning.", tool_calls=[
        selection("autonomy_level", "Reactive"),
        plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
        task_call("send_1", "task_send", "i1"),
    ])
    invalid = LLMResponse(content="Planning.", tool_calls=[
        task_call("send_2", "task_send", "i1"),
    ])
    provider = ScriptProvider([first, invalid, invalid])
    loop_workspace = tmp_path / "actual_loop"
    loop_workspace.mkdir()
    loop, _adapter = build_loop(loop_workspace, provider, "task_send")

    class ActualLoopCheckpoint(FakePACheckpoint):
        def close_ix_session(self, script_session_id: str) -> list:
            return loop.close_ix_session(SESSION_KEY, script_session_id)

    class ActualLoopExecutor(FakeExecutor):
        async def execute(self, step, *, steps_dir, stage_runtime=None, **kwargs):
            self.calls.append((step.step_id, stage_runtime.state_root, kwargs))
            (stage_runtime.state_root / "from_skipped.txt").write_text(
                step.step_id, encoding="utf-8"
            )
            try:
                return await run_turn(
                    loop,
                    "Mail Dale the extension request.",
                    active_ix=["autonomy_level"],
                    action_types={"task_send": "external_action"},
                    script_session_id=step.step_id,
                )
            except AgentProviderError as error:
                step_dir = Path(steps_dir) / step.step_id
                step_dir.mkdir(parents=True, exist_ok=True)
                artifact = step_dir / "failed_turn_events.json"
                artifact.write_text(
                    json.dumps(error.failed_turn_events), encoding="utf-8"
                )
                setattr(
                    error,
                    FAILED_STEP_ARTIFACTS_ATTR,
                    {"failed_turn_events": str(artifact)},
                )
                raise

    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    pa = ActualLoopCheckpoint([])
    orchestrator = _orchestrator(
        tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=pa,
    )
    orchestrator.config.failure_policy = FailurePolicy(auto_skip_on_ix_exhaustion=True)
    actual = ActualLoopExecutor()
    normal = FakeExecutor()

    class SequenceExecutor:
        async def execute(self, step, **kwargs):
            if step.step_id == SCRIPT_SESSION:
                return await actual.execute(step, **kwargs)
            return await normal.execute(step, **kwargs)

    orchestrator.executor = SequenceExecutor()
    orchestrator.ledger = ledger
    result = await orchestrator.run()

    assert len(provider.requests) == 3
    assert len(actual.calls) == 1
    assert result.skipped == [SCRIPT_SESSION]
    assert normal.calls[0][0] == "acc_002"
    summary = ledger.steps[SCRIPT_SESSION].summary
    assert summary["skip_reason"] == "failure_policy.auto_skip_on_ix_exhaustion"
    cleanup = json.loads((manager.run_dir / ledger.steps[SCRIPT_SESSION].artifacts["ix_session_cleanup"]).read_text())
    assert cleanup["banked_actions_released"] is False
    assert cleanup["unresolved_hold_episodes"]
    assert loop.sessions.get(SESSION_KEY).messages == []


@pytest.mark.asyncio
async def test_ix_runtime_without_three_miss_proof_still_fails(tmp_path: Path) -> None:
    plan = _plan().model_copy(update={"steps": [_plan().steps[0]]})
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = _orchestrator(
        tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=FakePACheckpoint([])
    )
    orchestrator.config.failure_policy = FailurePolicy(auto_skip_on_ix_exhaustion=True)

    class UntypedIX(FakeExecutor):
        async def execute(self, step, *, steps_dir, stage_runtime=None, **kwargs):
            error = AgentProviderError("PA turn failed the IX gate contract")
            raise error

    orchestrator.executor = UntypedIX()
    orchestrator.ledger = ledger
    with pytest.raises(AgentProviderError):
        await orchestrator.run()
    assert ledger.steps["acc_001"].status == "failed"


# ── failure_policy: one test per key ──────────────────────────────


class _PendingTurnsCheckpoint(FakePACheckpoint):
    """A mainline whose last turn's memory write never landed."""

    def __init__(self, calls: list, *, policy=None) -> None:
        super().__init__(calls, policy=policy)
        self.pending = [SimpleNamespace(turn_id="turn_9", span=(4, 6))]

    def turn_journal(self):
        journal = super().turn_journal()
        journal.pending_records = lambda: list(self.pending)
        return journal


async def _skip_acc_001(tmp_path: Path, policy: FailurePolicy, pa_checkpoint=None):
    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = _orchestrator(
        tmp_path, plan, checkpoint_manager=manager,
        pa_checkpoint=pa_checkpoint or FakePACheckpoint([]),
    )
    orchestrator.config.failure_policy = policy
    executor = FakeExecutor(fail_step="acc_001")
    orchestrator.executor = executor
    orchestrator.ledger = ledger
    result = await orchestrator.run()
    return orchestrator, ledger, manager, executor, result


@pytest.mark.asyncio
async def test_failure_policy_skip_steps_rollback_never_runs_the_step(tmp_path: Path) -> None:
    _orchestrator_, ledger, _manager, executor, result = await _skip_acc_001(
        tmp_path, FailurePolicy(skip_steps={"acc_001": "rollback"})
    )
    assert "acc_001" not in [call[0] for call in executor.calls]
    assert result.skipped[0] == "acc_001"
    assert ledger.steps["acc_001"].status == "skipped"
    assert ledger.steps["acc_001"].summary == {"skip_reason": "failure_policy.skip_steps"}
    assert ledger.steps["acc_002"].status == "done"


@pytest.mark.asyncio
async def test_failure_policy_skip_steps_retain_partial_keeps_the_failed_scene(tmp_path: Path) -> None:
    _orchestrator_, ledger, manager, executor, result = await _skip_acc_001(
        tmp_path, FailurePolicy(skip_steps={"acc_001": "retain_partial"})
    )
    assert [call[0] for call in executor.calls][:2] == ["acc_001", "acc_002"]
    assert result.skipped[0] == "acc_001"
    assert ledger.steps["acc_001"].status == "skipped"
    assert ledger.steps["acc_001"].summary["skip_reason"] == "failure_policy.skip_steps"
    assert ledger.steps["acc_002"].status == "done"
    # What the failed scene wrote stays in the world the next step ran in.
    assert manager.metadata("acc_001")["terminal_status"] == "skipped"
    assert (manager.run_dir / "checkpoints/acc_001/canonical_stage/marker.txt").read_text() == "acc_001"


@pytest.mark.asyncio
async def test_failure_policy_skip_steps_names_only_accumulation_steps(tmp_path: Path) -> None:
    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = _orchestrator(
        tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=FakePACheckpoint([])
    )
    orchestrator.config.failure_policy = FailurePolicy(skip_steps={plan.steps[-1].step_id: "rollback"})
    orchestrator.executor = FakeExecutor()
    orchestrator.ledger = ledger
    with pytest.raises(ValueError, match="only accumulation steps"):
        await orchestrator.run()
    assert orchestrator.executor.calls == []


@pytest.mark.asyncio
async def test_failure_policy_max_scene_attempts_skips_the_last_failed_attempt(tmp_path: Path) -> None:
    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = _orchestrator(
        tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=FakePACheckpoint([])
    )
    orchestrator.config.failure_policy = FailurePolicy(max_scene_attempts=2)
    orchestrator.executor = FakeExecutor(fail_step="acc_001")
    orchestrator.ledger = ledger
    # First attempt: the run stops, as it would with the default policy.
    with pytest.raises(RuntimeError, match="failed step"):
        await orchestrator.run()
    assert ledger.steps["acc_001"].status == "failed"
    attempts = manager.run_dir / orchestrator_module.SCENE_ATTEMPTS_FILE
    assert json.loads(attempts.read_text()) == {"acc_001": 1}
    # The second (last) attempt fails too: skipped, partial state kept, run goes on.
    assert orchestrator._failure_policy_skip(plan.steps[0], RuntimeError("again")) == (
        "failure_policy.max_scene_attempts"
    )
    assert json.loads(attempts.read_text()) == {"acc_001": 2}
    # Never for a step that is not an accumulation.
    assert orchestrator._failure_policy_skip(plan.steps[-1], RuntimeError("x")) is None


@pytest.mark.asyncio
async def test_failure_policy_pending_memory_drop_leaves_unwritten_turns_out(tmp_path: Path) -> None:
    from memory.adapter import PromptHistoryEvictionPolicy

    pa = _PendingTurnsCheckpoint([], policy=PromptHistoryEvictionPolicy.after_adapter_commit)
    replays = []

    async def recorded_replay():
        replays.append(1)

    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = _orchestrator(tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=pa)
    orchestrator.config.failure_policy = FailurePolicy(skip_steps={"acc_001": "retain_partial"})
    orchestrator._replay_pending_turns = recorded_replay
    orchestrator.executor = FakeExecutor(fail_step="acc_001")
    orchestrator.ledger = ledger
    await orchestrator.run()

    summary = ledger.steps["acc_001"].summary
    assert summary[orchestrator_module.DROPPED_PENDING_TURNS_KEY] == ["turn_9"]
    assert pa.checkpoint_session().adapter_committed_end_index == 6
    assert replays == [1]  # run start only


@pytest.mark.asyncio
async def test_failure_policy_pending_memory_replay_writes_them_before_going_on(tmp_path: Path) -> None:
    from memory.adapter import PromptHistoryEvictionPolicy

    pa = _PendingTurnsCheckpoint([], policy=PromptHistoryEvictionPolicy.after_adapter_commit)
    replays = []

    async def recorded_replay():
        replays.append(1)

    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = _orchestrator(tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=pa)
    orchestrator.config.failure_policy = FailurePolicy(
        skip_steps={"acc_001": "retain_partial"}, partial_skip_pending_memory="replay",
    )
    orchestrator._replay_pending_turns = recorded_replay
    orchestrator.executor = FakeExecutor(fail_step="acc_001")
    orchestrator.ledger = ledger
    await orchestrator.run()

    assert orchestrator_module.DROPPED_PENDING_TURNS_KEY not in ledger.steps["acc_001"].summary
    assert replays == [1, 1]  # run start, then the skip
    assert ledger.steps["acc_002"].status == "done"


class _JournalCheckpoint(FakePACheckpoint):
    """A file-condition mainline over a real turn journal on disk."""

    def __init__(self, path: Path) -> None:
        super().__init__([])
        self.path = path

    def turn_journal(self):
        from memory.turn_journal import TurnJournal

        return TurnJournal(self.path)


def _journal_turn(path: Path, step_id: str) -> None:
    """One turn of `step_id`, journalled as the PA loop does for the file
    condition (pending, then written with the deferred receipt), owed to the files."""
    from memory.event_writer import TurnPool
    from memory.results import WriteReceipt, WriteStatus
    from memory.turn_journal import JournalRecord, TurnJournal

    journal = TurnJournal(path)
    start = 2 * len(journal.records())
    turn_id = f"{step_id}:opening:t1"
    journal.append_pending(JournalRecord.for_pool(
        turn_id=turn_id,
        span=(start, start + 2),
        timestamp="2026-10-01T10:00:00+00:00",
        pool=TurnPool(
            messages=[
                {"role": "user", "content": f"{step_id} user turn"},
                {"role": "assistant", "content": f"{step_id} reply"},
            ],
            items=[],
        ),
    ))
    journal.mark_written(turn_id, WriteReceipt(status=WriteStatus.deferred_to_file_consolidation))


class _JournallingExecutor(FakeExecutor):
    """Each accumulation step leaves one journalled turn, then runs as usual."""

    def __init__(self, path: Path, fail_step: str) -> None:
        super().__init__(fail_step=fail_step)
        self.path = path

    async def execute(self, step, **kwargs):
        if step.kind == "accumulation":
            _journal_turn(self.path, step.step_id)
        return await super().execute(step, **kwargs)


async def _skip_acc_001_over_file_memory(tmp_path: Path, pending_memory: str):
    journal_path = tmp_path / "mainline.journal.jsonl"
    _journal_turn(journal_path, "acc_000")  # an earlier turn the files are still owed
    pa = _JournalCheckpoint(journal_path)
    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    orchestrator = _orchestrator(tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=pa)
    orchestrator.config.failure_policy = FailurePolicy(
        skip_steps={"acc_001": "retain_partial"}, partial_skip_pending_memory=pending_memory,
    )
    orchestrator.executor = _JournallingExecutor(journal_path, fail_step="acc_001")
    orchestrator.ledger = ledger
    await orchestrator.run()
    written = [message["content"] for batch in pa.archived for message in batch]
    return ledger, written, pa.turn_journal()


@pytest.mark.asyncio
async def test_drop_keeps_a_skipped_steps_turns_out_of_file_memory(tmp_path: Path) -> None:
    """The file condition writes at consolidation, after the skip: `drop` has to
    mark the skipped step's own turns, or the next flush writes them anyway.
    Turns of earlier steps the files are still owed are not touched."""
    ledger, written, journal = await _skip_acc_001_over_file_memory(tmp_path, "drop")

    assert ledger.steps["acc_001"].status == "skipped"
    assert ledger.steps["acc_001"].summary[orchestrator_module.DROPPED_PENDING_TURNS_KEY] == [
        "acc_001:opening:t1"
    ]
    assert "acc_001 user turn" not in written and "acc_001 reply" not in written
    assert "acc_000 user turn" in written and "acc_002 user turn" in written
    dropped = journal.record_for("acc_001:opening:t1")
    assert dropped.dropped_from_files and not dropped.file_consolidated
    assert journal.record_for("acc_000:opening:t1").file_consolidated


@pytest.mark.asyncio
async def test_replay_leaves_a_skipped_steps_turns_to_file_consolidation(tmp_path: Path) -> None:
    ledger, written, journal = await _skip_acc_001_over_file_memory(tmp_path, "replay")

    assert orchestrator_module.DROPPED_PENDING_TURNS_KEY not in ledger.steps["acc_001"].summary
    assert "acc_001 user turn" in written
    assert not journal.record_for("acc_001:opening:t1").dropped_from_files


@pytest.mark.asyncio
async def test_a_later_start_does_not_replay_dropped_turns(tmp_path: Path, monkeypatch) -> None:
    import memory.turn_journal as turn_journal_module

    pa = _PendingTurnsCheckpoint([])
    pa.pending = [SimpleNamespace(turn_id="turn_9", span=(4, 6)),
                  SimpleNamespace(turn_id="turn_10", span=(6, 8))]
    plan = _two_acc_plan()
    ledger, manager = _seeded_checkpoint_manager(tmp_path, plan)
    ledger.mark_skipped("acc_001", summary={
        "skip_reason": "failure_policy.skip_steps",
        orchestrator_module.DROPPED_PENDING_TURNS_KEY: ["turn_9"],
    })
    orchestrator = _orchestrator(tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=pa)
    orchestrator.ledger = ledger
    seen = []

    async def fake_replay(**kwargs):
        seen.append([record.turn_id for record in kwargs["records"]])
        return []

    monkeypatch.setattr(turn_journal_module, "replay_pending_turns", fake_replay)
    await orchestrator._replay_pending_turns()
    assert seen == [["turn_10"]]


@pytest.mark.asyncio
async def test_a_probe_with_no_checkpoint_manager_stops_the_run(tmp_path: Path):
    """A probe runs only in a branch cut from a pin. With no
    checkpoint manager there is no mainline fallback: the run stops at the
    first probe, the probe is never marked, and the timeline stays done."""
    plan = _plan()
    orchestrator = _orchestrator(
        tmp_path, plan, checkpoint_manager=None, pa_checkpoint=None
    )

    with pytest.raises(RuntimeError, match="no checkpoint manager: step_id='pretest_W_A'"):
        await orchestrator.run()

    assert [call[0] for call in orchestrator.executor.calls] == ["acc_001"]
    assert orchestrator.ledger.steps["acc_001"].status == "done"
    assert "pretest_W_A" not in orchestrator.ledger.steps
    assert not hasattr(orchestrator, "_run_probe_on_mainline")


class ProbeRecordingExecutor(FakeExecutor):
    """Records what a probe actually faced, before its branch is discarded."""

    def __init__(self):
        super().__init__()
        self.probe_views: list[dict] = []

    async def execute(
        self,
        step,
        *,
        steps_dir,
        stage_runtime=None,
        extra_meta=None,
        loop_context=None,
    ):
        if loop_context is not None:
            workspace = loop_context["workspace"]
            self.probe_views.append(
                {
                    "step_id": step.step_id,
                    "adapter": loop_context["memory_adapter"],
                    "recorder_path": loop_context["memory_io_recorder"].path,
                    "skip_consolidation": loop_context["skip_consolidation"],
                    "preference_overlay": loop_context.get("preference_overlay"),
                    "sim_profile": loop_context.get("sim_profile"),
                    "extra_meta": dict(extra_meta or {}),
                    "session_keys": sorted(
                        path.stem
                        for path in (workspace / "sessions").glob("*.jsonl")
                    ),
                    "memory_files": sorted(
                        path.name for path in (workspace / "memory").iterdir()
                    ),
                    "world": (stage_runtime.state_root / "base.txt").read_text(
                        encoding="utf-8"
                    ),
                    "steps_dir": Path(steps_dir),
                }
            )
        return await super().execute(
            step,
            steps_dir=steps_dir,
            stage_runtime=stage_runtime,
            extra_meta=extra_meta,
            loop_context=loop_context,
        )


class RaisingExecutor(FakeExecutor):
    """Raises one prepared session-level failure on `fail_step`."""

    def __init__(self, fail_step: str, error: BaseException) -> None:
        super().__init__(fail_step=fail_step)
        self.error = error

    async def execute(self, step, **kwargs):
        try:
            return await super().execute(step, **kwargs)
        except RuntimeError as written:
            setattr(
                self.error,
                FAILED_STEP_ARTIFACTS_ATTR,
                failed_step_artifacts(written),
            )
            raise self.error from None


#: One case per session-level failure kind. The three
#: judge voids and the two simulator ones reach the step as the class named
#: here; `judge_unavailable` arrives as the bare `JudgeUnavailable`, because
#: `SimulatorLoop.run_session` re-raises the cause rather than the void.
SESSION_LEVEL_FAILURES = [
    (
        SessionInvalidError("beat_ask_unfulfilled", beat="b3_probe", attempts=3),
        "session_invalid",
    ),
    (
        SimulatorOutputContractError(["simulator_output_format_invalid"]),
        "output_contract",
    ),
    (
        SessionInvalidError("judge_timeout", beat="b3_probe", attempts=1),
        "judge_timeout",
    ),
    (
        SessionInvalidError("judge_unparseable", beat="b3_probe", attempts=2),
        "judge_unparseable",
    ),
    (
        SessionInvalidError("judge_evidence_invalid", beat="b3_probe", attempts=2),
        "judge_evidence_invalid",
    ),
    (
        JudgeUnavailable("ollama is not reachable at http://127.0.0.1:11434"),
        "judge_unavailable",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "kind"),
    SESSION_LEVEL_FAILURES,
    ids=[kind for _error, kind in SESSION_LEVEL_FAILURES],
)
async def test_every_session_level_failure_stops_the_round(
    tmp_path: Path, error: BaseException, kind: str
) -> None:
    """A round is an accumulation run, so a session that measured nothing may
    not be skipped past — the hole would break the chain every later session
    reads from. Every one of the six goes out the same door `judge_unavailable`
    always used: the exception leaves `run()` (that is the process's non-zero
    rc), the ledger names the kind, and no later session is started.
    """
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    executor = RaisingExecutor("acc_002", error)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=_file_memory_manager(run_dir, ledger),
        pa_checkpoint=FakePACheckpoint([]),
    )
    orchestrator.ledger = ledger
    orchestrator.executor = executor

    with pytest.raises(type(error)):
        await orchestrator.run()

    assert ledger.steps["acc_002"].status == "failed"
    assert ledger.steps["acc_002"].failure_kind == kind
    # The round ended where it stood: the probe after it was never attempted.
    assert [call[0] for call in executor.calls] == ["acc_001", "acc_002"]
    assert "pretest_W_A" not in ledger.steps


@pytest.mark.asyncio
async def test_the_exception_that_stops_the_round_names_the_failed_step(
    tmp_path: Path,
) -> None:
    """`mempabench run` reads the step for its stop line (`step=<id>`) from the
    exception that leaves `run()`, named where the ledger marks the step failed.
    """
    from harness.failure_kind import FAILED_STEP_ID_ATTR

    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    error = AgentProviderError(
        "PA provider failed to produce a response: model='m' error='length'"
    )
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=_file_memory_manager(run_dir, ledger),
        pa_checkpoint=FakePACheckpoint([]),
    )
    orchestrator.ledger = ledger
    orchestrator.executor = RaisingExecutor("acc_002", error)

    with pytest.raises(AgentProviderError) as raised:
        await orchestrator.run()

    assert raised.value is error
    assert getattr(error, FAILED_STEP_ID_ATTR) == "acc_002"
    assert ledger.steps["acc_002"].failure_kind == "provider"


@pytest.mark.asyncio
async def test_the_same_run_id_resumes_at_the_session_that_stopped_the_round(
    tmp_path: Path,
) -> None:
    """The other half of the rule: stopping must not mean the run is over.

    Rerunning the SAME run id with no `--overwrite` rolls back to the
    checkpoint taken before the failed session — the failed record goes with
    the ledger — and that session runs again, followed by the rest.
    """
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)

    first = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint([]),
    )
    first.ledger = ledger
    first.executor = RaisingExecutor(
        "acc_002",
        SessionInvalidError("judge_timeout", beat="b3_probe", attempts=1),
    )
    with pytest.raises(SessionInvalidError):
        await first.run()
    assert ledger.steps["acc_002"].failure_kind == "judge_timeout"
    # acc_001 got its cut point, so there is somewhere to come back to.
    assert manager.list_checkpoints() == ["acc_001"]

    # Same run id, same directory, nothing overwritten.
    resumed_ledger = RunLedger(run_dir, plan.run_id)
    resumed = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=_file_memory_manager(run_dir, resumed_ledger),
        pa_checkpoint=FakePACheckpoint([]),
    )
    resumed.ledger = resumed_ledger
    executor = FakeExecutor()
    resumed.executor = executor
    result = await resumed.run()

    # acc_001 is behind the cut point and is not repeated; the session that
    # stopped the round is the one the rerun starts at, and the rest follows.
    assert result.skipped == ["acc_001"]
    assert result.executed == ["acc_002", "pretest_W_A"]
    assert [call[0] for call in executor.calls] == ["acc_002", "pretest_W_A"]
    assert resumed_ledger.steps["acc_002"].status == "done"
    assert resumed_ledger.steps["acc_002"].failure_kind is None


@pytest.mark.asyncio
async def test_unsafe_resume_without_checkpoint_still_raises(tmp_path: Path):
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    ledger.mark_running("acc_001")

    # No snapshot to fall back to: the original refusal stands.
    bare = _orchestrator(tmp_path, plan, checkpoint_manager=None, pa_checkpoint=None)
    bare.ledger = RunLedger(run_dir, plan.run_id)
    with pytest.raises(UnsafeRunResumeError):
        await bare.run()

    # With a snapshot, the run rolls back to it and carries on.
    manager = _file_memory_manager(run_dir, ledger)
    manager.save("acc_001", pinned=False)
    ledger.mark_running("acc_002")
    resumed = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint([]),
    )
    resumed.ledger = ledger
    resumed.config.checkpoint_manager = manager
    await resumed.run()

    assert (run_dir / "crash_scene").exists()
    assert ledger.steps["acc_002"].status == "done"


@pytest.mark.asyncio
async def test_resume_reruns_steps_after_checkpoint(tmp_path: Path):
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)

    # Crash *before* publication: the snapshot's ledger has no record of the
    # step at all, so it is rerun.
    ledger.mark_done("acc_001")
    manager = _file_memory_manager(run_dir, ledger)
    manager.save("acc_001", pinned=False)
    ledger.mark_running("acc_002")

    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint([]),
    )
    orchestrator.ledger = ledger
    result = await orchestrator.run()
    assert result.skipped == ["acc_001"]
    assert result.executed == ["acc_002", "pretest_W_A"]

    # Crash *after* publication: the snapshot's own ledger already marks the
    # step done, so the run continues from the next one.
    after = _two_acc_plan()
    run_dir_b = tmp_path / "after" / "runs" / after.run_id
    _seed_run_dir(run_dir_b, after.persona_id, after.main_pa_session_key())
    ledger_b = RunLedger(run_dir_b, after.run_id)
    ledger_b.mark_running("acc_001")
    manager_b = _file_memory_manager(run_dir_b, ledger_b)
    manager_b.save("acc_001", pinned=False)
    ledger_b.mark_running("acc_002")

    second = _orchestrator(
        tmp_path / "after",
        after,
        checkpoint_manager=manager_b,
        pa_checkpoint=FakePACheckpoint([]),
    )
    second.ledger = ledger_b
    result_b = await second.run()
    assert "acc_001" not in result_b.executed
    assert result_b.executed == ["acc_002", "pretest_W_A"]


def test_rollback_goes_to_the_furthest_cut_point_not_the_latest_clock_reading(
    tmp_path: Path,
) -> None:
    """A resume goes back to the last cut point the run reached.

    The cut point was chosen by the largest
    `timestamp` in `metadata.json`, and those come from `datetime.now`: a host
    clock that steps backwards between two saves makes an earlier cut point
    read as the newest one, and the resume then reruns a completed, paid
    session. How far a checkpoint is into the run is what its own ledger says,
    and that only ever grows.
    """
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)
    for step_id in ("acc_001", "acc_002"):
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=False)
        ledger.mark_done(step_id)

    metadata_path = run_dir / "checkpoints" / "acc_002" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["timestamp"] = "2000-01-01T00:00:00+00:00"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # The run then died in a timeline step, which is what makes it unsafe to
    # resume as it stands. It has to be a timeline step: this plan's probe is
    # rerun from its own immutable pin, so a probe left `running`
    # forces no rollback and there would be nothing here to measure.
    ledger.mark_failed("acc_002", "died after the cut point was published")

    assert orchestrator_module.resume_or_restore(ledger, manager, plan) == "acc_002"
    assert ledger.decision_for("acc_002") == "skip"


@pytest.fixture
def _signal_handlers_restored():
    """Put this process's SIGTERM/SIGINT handlers back after the test."""
    previous = {
        number: signal.getsignal(number)
        for number in (signal.SIGTERM, signal.SIGINT)
    }
    yield
    for number, handler in previous.items():
        signal.signal(number, handler)


class _SignallingExecutor(FakeExecutor):
    """Delivers real signals to this process from inside one step."""

    def __init__(self, step_id: str, *, count: int = 1) -> None:
        super().__init__()
        self.step_id = step_id
        self.count = count

    async def execute(self, step, **kwargs):
        if step.step_id == self.step_id:
            for _ in range(self.count):
                os.kill(os.getpid(), signal.SIGTERM)
        return await super().execute(step, **kwargs)


@pytest.mark.asyncio
async def test_a_signal_mid_step_finishes_the_step_and_stops_clean(
    tmp_path: Path,
    capsys,
    _signal_handlers_restored,
) -> None:
    """SIGTERM lands between steps, never inside one.

    The step in flight when the signal arrives runs to its end, publishes its
    cut point and is marked done; the loop then stops, and the ledger has no
    `running` record for the next resume to roll back.
    """
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)
    stop = GracefulStop(ledger)
    stop.install()

    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint([]),
        stop_requested=stop,
    )
    orchestrator.ledger = ledger
    executor = _SignallingExecutor("acc_001")
    orchestrator.executor = executor

    result = await orchestrator.run()

    # The interrupted step finished, and nothing after it started.
    assert stop.requested is True
    assert result.executed == ["acc_001"]
    assert [call[0] for call in executor.calls] == ["acc_001"]
    # Its cut point was published before the loop read the flag.
    assert manager.list_checkpoints() == ["acc_001"]
    assert ledger.steps["acc_001"].status == "done"
    assert ledger.current_step is None
    # The whole point: the next resume rolls nothing back.
    ledger.assert_resumable(plan)

    stop.report(result)
    printed = capsys.readouterr().out
    assert "stop requested: finishing acc_001 then exiting" in printed
    assert (
        "stopped cleanly: 1 step(s) done, 0 failed; resume with the same "
        "--run-id" in printed
    )


@pytest.mark.asyncio
async def test_a_second_signal_aborts_and_leaves_the_step_running(
    tmp_path: Path,
    capsys,
    monkeypatch,
    _signal_handlers_restored,
) -> None:
    """The second signal is the old behaviour: die now, roll back later."""
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)
    stop = GracefulStop(ledger)
    stop.install()

    # Stand in for the default disposition the handler hands the signal back
    # to, which would kill this process for real.
    def _die(number: int) -> None:
        raise SystemExit(f"killed by {number}")

    monkeypatch.setattr(signal, "raise_signal", _die)

    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint([]),
        stop_requested=stop,
    )
    orchestrator.ledger = ledger
    orchestrator.executor = _SignallingExecutor("acc_001", count=2)

    with pytest.raises(SystemExit):
        await orchestrator.run()

    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    assert ledger.steps["acc_001"].status == "running"
    assert manager.list_checkpoints() == []
    with pytest.raises(UnsafeRunResumeError):
        ledger.assert_resumable(plan)
    printed = capsys.readouterr().out
    assert "stop requested: finishing acc_001 then exiting" in printed
    assert "stop forced: acc_001 left running, resume will roll back" in printed


@pytest.mark.asyncio
async def test_probe_runs_in_branch_and_returns_only_artifacts(tmp_path: Path):
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    workspace = _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)
    executor = ProbeRecordingExecutor()
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint([]),
    )
    orchestrator.ledger = ledger
    orchestrator.executor = executor

    await orchestrator.run()

    view = executor.probe_views[0]
    assert view["step_id"] == "pretest_W_A"
    # The probe continues the mainline session inside its own branch.
    assert view["session_keys"] == [plan.main_pa_session_key()]
    assert view["adapter"] is None
    # The whole compaction step is skipped inside the branch.
    assert view["skip_consolidation"] is True
    assert Path(view["recorder_path"]).parent != run_dir
    assert "branches" in str(view["steps_dir"])

    # The mainline is untouched: world, memory, session and audit.
    assert (run_dir / "canonical_stage" / "marker.txt").read_text(
        encoding="utf-8"
    ) == "acc_002"
    assert (workspace / "memory" / "MEMORY.md").read_text(
        encoding="utf-8"
    ) == "pa memory\n"
    assert (run_dir / "memory_io.jsonl").read_text(
        encoding="utf-8"
    ) == '{"seq": 1}\n'

    # Only the evidence comes back, and the branch is gone.
    step_dir = run_dir / "steps" / "pretest_W_A"
    assert (step_dir / "meta.yaml").is_file()
    assert (step_dir / "memory_io.jsonl").read_text(encoding="utf-8") == ""
    scene = json.loads((step_dir / "scene_start.json").read_text(encoding="utf-8"))
    assert scene["step_id"] == "pretest_W_A"
    assert any(entry["path"] == "base.txt" for entry in scene["world_state"])
    assert list((run_dir / "branches" / "pretest_W_A").iterdir()) == []

    # Each retry gets a directory of its own, and the call that picks a
    # number reserves it, so a second caller cannot pick the same one.
    first = orchestrator._next_branch_dir(plan.steps[-1])
    assert first.is_dir()
    second = orchestrator._next_branch_dir(plan.steps[-1])
    assert first != second


class FailingProbeExecutor(FakeExecutor):
    """A probe that dies after `session_executor` already wrote its artifacts."""

    async def execute(
        self,
        step,
        *,
        steps_dir,
        stage_runtime=None,
        extra_meta=None,
        loop_context=None,
    ):
        if loop_context is None:
            return await super().execute(
                step,
                steps_dir=steps_dir,
                stage_runtime=stage_runtime,
                extra_meta=extra_meta,
                loop_context=loop_context,
            )
        step_dir = Path(steps_dir) / step.step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "transcript.jsonl").write_text(
            '{"t": 0, "kind": "session_start"}\n', encoding="utf-8"
        )
        (step_dir / "pa_toolcalls.json").write_text(
            '{"beats": []}\n', encoding="utf-8"
        )
        (step_dir / "meta.yaml").write_text(
            f"step_id: {step.step_id}\n"
            "failure_reason: 'AgentProviderError: ix contract failure'\n",
            encoding="utf-8",
        )
        raise RuntimeError("ix contract failure")


@pytest.mark.asyncio
async def test_a_failed_probe_leaves_its_evidence_behind(tmp_path: Path):
    """The probe most worth auditing must not die with its branch."""
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint([]),
    )
    orchestrator.ledger = ledger
    orchestrator.executor = FailingProbeExecutor()

    # A probe is a job of its own, so its failure ends the probe, not
    # the run — it is recorded and the batch goes on.
    result = await orchestrator.run()
    assert result.failed == ["pretest_W_A"]
    assert ledger.steps["pretest_W_A"].error == "ix contract failure"

    step_dir = run_dir / "steps" / "pretest_W_A"
    assert (step_dir / "transcript.jsonl").is_file()
    assert (step_dir / "pa_toolcalls.json").is_file()
    assert "failure_reason" in (step_dir / "meta.yaml").read_text(encoding="utf-8")
    # The scene snapshot was taken before the first PA call, so it survives too.
    assert (step_dir / "scene_start.json").is_file()
    # The branch is still discarded, and the step is still failed.
    assert list((run_dir / "branches" / "pretest_W_A").iterdir()) == []
    assert ledger.steps["pretest_W_A"].status == "failed"
    # The ledger names the copies under the run dir, never the branch paths
    # the executor attached — the branch is gone by now.
    recorded = ledger.steps["pretest_W_A"].artifacts
    assert recorded["transcript_jsonl"] == "steps/pretest_W_A/transcript.jsonl"
    assert recorded["meta"] == "steps/pretest_W_A/meta.yaml"


def test_paired_final_probe_overlays_its_pre_event_partner_fixture(tmp_path: Path):
    plan = _two_acc_plan()
    steps = [step.model_dump(mode="json") for step in plan.steps]
    steps.append(
        {
            **steps[-1],
            "step_id": "fin_03",
            "kind": "final_probe",
            "script_path": "probe/fin_03.yaml",
            "before_acc_num": None,
            "benchmark_session_timestamp": "2026-10-01T18:00:00+00:00",
        }
    )
    steps[-2]["script_path"] = "probe/pretest_W_A.yaml"
    plan = plan.model_copy(update={"steps": [RunStep(**s) for s in steps]})

    probe_dir = tmp_path / "probe"
    (probe_dir / "fixture").mkdir(parents=True)
    (probe_dir / "fin_03.yaml").write_text(
        "meta:\n  probe_kind: final\n  pairs_with: pretest_W_A\n", encoding="utf-8"
    )
    (probe_dir / "pretest_W_A.yaml").write_text(
        "meta:\n  probe_kind: pre_event\n", encoding="utf-8"
    )

    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=None,
        pa_checkpoint=None,
    )
    # Both halves of the pair resolve to the pre-event probe's fixture dir.
    assert orchestrator._probe_fixture_overlay(plan.steps[-1]) == (
        probe_dir / "fixture"
    )
    assert orchestrator._probe_fixture_overlay(plan.steps[-2]) == (
        probe_dir / "fixture"
    )


@pytest.mark.asyncio
async def test_memory_removed_baseline_binds_no_memory_adapter(tmp_path: Path):
    from memory.conditions.no_memory import NoMemoryAdapter

    # The tested run leaves behind two pinned cut points.
    source = _two_acc_plan()
    source_dir = tmp_path / "runs" / source.run_id
    _seed_run_dir(source_dir, source.persona_id, source.main_pa_session_key())
    (source_dir / "canonical_stage").mkdir(parents=True, exist_ok=True)
    source_ledger = RunLedger(source_dir, source.run_id)
    source_manager = _file_memory_manager(source_dir, source_ledger)
    for step_id, marker in (("acc_001", "pre"), ("acc_002", "final")):
        (source_dir / "canonical_stage" / "base.txt").write_text(
            marker, encoding="utf-8"
        )
        source_manager.save(step_id, pinned=True)
        source_ledger.mark_done(step_id)

    # The baseline run has only probes, each borrowing its own cut point.
    base = source.steps[-1].model_dump(mode="json")
    baseline = source.model_copy(
        update={
            # What makes this run the baseline is its condition, not the fact
            # that its probes borrow: a probe plan attached later to a finished
            # accumulation run borrows too, and it has memory.
            "memory_condition": "no_memory",
            "pa_session_key": source.main_pa_session_key(),
            "steps": [
                RunStep(
                    **{
                        **base,
                        "step_id": "pre_01",
                        "script_path": "pre_01.yaml",
                        "probe_overlay": {"owner_probe_id": "pre_01"},
                        "borrow_start_from": f"{source.run_id}/acc_001",
                    }
                ),
                RunStep(
                    **{
                        **base,
                        "step_id": "fin_03",
                        "kind": "final_probe",
                        "script_path": "fin_03.yaml",
                        "probe_overlay": {"owner_probe_id": "pre_01"},
                        "before_acc_num": None,
                        "borrow_start_from": f"{source.run_id}/acc_002",
                        "benchmark_session_timestamp": (
                            "2026-10-01T18:00:00+00:00"
                        ),
                    }
                ),
            ],
        }
    )
    assert baseline.main_pa_session_key() == f"{source.run_id}_main"
    _fixture(tmp_path)
    for step in baseline.steps:
        _write_probe_script(tmp_path, step)

    baseline_dir = tmp_path / "runs" / "baseline_run"
    executor = ProbeRecordingExecutor()
    orchestrator = Orchestrator(
        plan=baseline,
        config=OrchestratorConfig(
            run_dir=baseline_dir,
            project_root=tmp_path,
            checkpoint_manager=_file_memory_manager(
                baseline_dir, RunLedger(baseline_dir, baseline.run_id)
            ),
            borrowed_runs_root=tmp_path / "runs",
        ),
        executor=executor,
        stage_runtime=StageRuntime(
            run_id=baseline.run_id,
            user_id=baseline.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            FakeMemoryAdapter(),
            run_id=baseline.run_id,
            persona_id=baseline.persona_id,
        ),
        progress=lambda _line: None,
    )

    await orchestrator.run()

    views = {view["step_id"]: view for view in executor.probe_views}
    assert set(views) == {"pre_01", "fin_03"}
    for view in views.values():
        assert isinstance(view["adapter"], NoMemoryAdapter)
        # The PA memory directory was emptied inside the branch.
        assert view["memory_files"] == []
        # The borrowed session file keeps the *source* run's key.
        assert view["session_keys"] == [f"{source.run_id}_main"]
    # A pre-event probe borrows its own pinned cut point, not the final one.
    assert views["pre_01"]["world"] == "pre"
    assert views["fin_03"]["world"] == "final"

    # Ledger and artifacts stay in the baseline run's own directory.
    assert (baseline_dir / "ledger.json").is_file()
    assert (baseline_dir / "steps" / "fin_03" / "meta.yaml").is_file()
    assert not (source_dir / "steps" / "fin_03").exists()


class _PromptWindowExecutor(ProbeRecordingExecutor):
    """Also records the branch session's prompt window as the probe PA reads it."""

    def __init__(self):
        super().__init__()
        self.prompt_windows: list[tuple[int, int, list]] = []

    async def execute(
        self, step, *, steps_dir, stage_runtime=None, extra_meta=None, loop_context=None
    ):
        if loop_context is not None:
            session = loop_context["session_manager"].get(extra_meta["pa_session_key"])
            self.prompt_windows.append(
                (
                    session.prompt_history_start_index,
                    len(session.messages),
                    session.get_history(),
                )
            )
        return await super().execute(
            step,
            steps_dir=steps_dir,
            stage_runtime=stage_runtime,
            extra_meta=extra_meta,
            loop_context=loop_context,
        )


@pytest.mark.asyncio
async def test_oracle_preference_probe_reads_only_its_ground_truth(tmp_path: Path):
    """The no_memory arm with the probe's frozen setting as its one memory:
    the File-memory source's MEMORY.md is emptied in the branch, and the
    accumulation dialogue is out of the prompt window."""
    from memory.conditions.oracle_preference import OraclePreferenceAdapter
    from memory.scope import MemoryRetrievalBudget, MemoryScope, RetrieveRequest

    source = _two_acc_plan()
    source_dir = tmp_path / "runs" / source.run_id
    workspace = _seed_run_dir(
        source_dir, source.persona_id, source.main_pa_session_key()
    )
    sessions = SessionManager(workspace)
    session = sessions.get(source.main_pa_session_key())
    session.add_message("user", "acc question")
    session.add_message("assistant", "acc answer")
    sessions.save(session)
    source_ledger = RunLedger(source_dir, source.run_id)
    source_manager = _file_memory_manager(source_dir, source_ledger)
    source_manager.save("acc_002", pinned=True)
    source_ledger.mark_done("acc_002")
    snapshot = SessionManager(
        source_dir / "checkpoints" / "acc_002" / "pa_workspace" / "user_a"
    ).get(source.main_pa_session_key())
    # The pinned session still has the dialogue in its prompt window.
    assert (snapshot.prompt_history_start_index, len(snapshot.messages)) == (0, 2)

    base = source.steps[-1].model_dump(mode="json")
    oracle = source.model_copy(
        update={
            "memory_condition": "oracle_preference",
            "pa_session_key": source.main_pa_session_key(),
            "steps": [
                RunStep(
                    **{
                        **base,
                        "step_id": "fin_01",
                        "kind": "final_probe",
                        "script_path": "fin_01.yaml",
                        "probe_overlay": {"owner_probe_id": "fin_01"},
                        "target_cell": "work_verbosity",
                        "target_attribute": "verbosity",
                        "target_setting": "Detailed",
                        "before_acc_num": None,
                        "borrow_start_from": f"{source.run_id}/acc_002",
                        "benchmark_session_timestamp": "2026-10-01T18:00:00+00:00",
                    }
                )
            ],
        }
    )
    _fixture(tmp_path)
    _write_probe_script(tmp_path, oracle.steps[0])

    oracle_dir = tmp_path / "runs" / "oracle_run"
    executor = _PromptWindowExecutor()
    orchestrator = Orchestrator(
        plan=oracle,
        config=OrchestratorConfig(
            run_dir=oracle_dir,
            project_root=tmp_path,
            checkpoint_manager=CheckpointManager(
                oracle_dir,
                ledger=RunLedger(oracle_dir, oracle.run_id),
                memory_condition="oracle_preference",
            ),
            borrowed_runs_root=tmp_path / "runs",
        ),
        executor=executor,
        stage_runtime=StageRuntime(
            run_id=oracle.run_id,
            user_id=oracle.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            FakeMemoryAdapter(),
            run_id=oracle.run_id,
            persona_id=oracle.persona_id,
        ),
        progress=lambda _line: None,
    )

    await orchestrator.run()

    [view] = executor.probe_views
    adapter = view["adapter"]
    assert isinstance(adapter, OraclePreferenceAdapter)
    result = await adapter.retrieve(
        RetrieveRequest(
            query="q",
            scope=MemoryScope(run_id=oracle.run_id, persona_id="user_a"),
            session_id="s",
            budget=MemoryRetrievalBudget(max_tokens=4000),
        )
    )
    assert adapter.format_context(result).startswith(
        "In work situations like this one, the user prefers Verbosity: Detailed. "
    )
    # (b) The PA memory directory was emptied inside the branch only.
    assert view["memory_files"] == []
    assert (
        source_dir / "checkpoints" / "acc_002" / "pa_workspace" / "user_a"
        / "memory" / "MEMORY.md"
    ).is_file()
    # (a) The probe PA's prompt window starts past the accumulation dialogue.
    assert executor.prompt_windows == [(2, 2, [])]
    assert view["session_keys"] == [f"{source.run_id}_main"]


class _FirstPACallExecutor(ProbeRecordingExecutor):
    """Runs the real PA loop once on the branch session and records its first model call."""

    def __init__(self):
        super().__init__()
        self.first_calls: list[list[dict]] = []

    async def execute(
        self, step, *, steps_dir, stage_runtime=None, extra_meta=None, loop_context=None
    ):
        if loop_context is not None:
            from pa.agent.loop import AgentLoop
            from pa.bus.queue import MessageBus
            from test_pa.test_memory_adapter_context import RecordingProvider

            provider = RecordingProvider()
            loop = AgentLoop(
                bus=MessageBus(),
                pa_llm_provider=provider,
                workspace=loop_context["workspace"],
                pa_model="test-model",
                memory_adapter=loop_context["memory_adapter"],
            )
            await loop.process_direct_message(
                "probe question",
                session_key=extra_meta["pa_session_key"],
                metadata={"run_id": "oracle_run", "persona_id": "user_a"},
            )
            self.first_calls.append(provider.calls[0])
        return await super().execute(
            step,
            steps_dir=steps_dir,
            stage_runtime=stage_runtime,
            extra_meta=extra_meta,
            loop_context=loop_context,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["oracle_preference", "no_memory"])
async def test_oracle_probe_starts_with_no_prompt_history(tmp_path: Path, condition: str):
    """A source session with dialogue reaches the oracle PA's first model call
    with no history at all; no_memory is the same branch
    with the ground truth left out."""
    source = _two_acc_plan()
    source_dir = tmp_path / "runs" / source.run_id
    workspace = _seed_run_dir(source_dir, source.persona_id, source.main_pa_session_key())
    sessions = SessionManager(workspace)
    session = sessions.get(source.main_pa_session_key())
    session.add_message("user", "acc question")
    session.add_message("assistant", "acc answer")
    sessions.save(session)
    source_ledger = RunLedger(source_dir, source.run_id)
    _file_memory_manager(source_dir, source_ledger).save("acc_002", pinned=True)
    source_ledger.mark_done("acc_002")

    base = source.steps[-1].model_dump(mode="json")
    oracle = source.model_copy(update={
        "memory_condition": condition,
        "pa_session_key": source.main_pa_session_key(),
        "steps": [RunStep(**{
            **base, "step_id": "fin_01", "kind": "final_probe",
            "script_path": "fin_01.yaml",
            "probe_overlay": {"owner_probe_id": "fin_01"},
            "target_cell": "work_verbosity", "target_attribute": "verbosity",
            "target_setting": "Detailed", "before_acc_num": None,
            "borrow_start_from": f"{source.run_id}/acc_002",
            "benchmark_session_timestamp": "2026-10-01T18:00:00+00:00",
        })],
    })
    _fixture(tmp_path)
    _write_probe_script(tmp_path, oracle.steps[0])
    oracle_dir = tmp_path / "runs" / "oracle_run"
    executor = _FirstPACallExecutor()
    orchestrator = Orchestrator(
        plan=oracle,
        config=OrchestratorConfig(
            run_dir=oracle_dir,
            project_root=tmp_path,
            checkpoint_manager=CheckpointManager(
                oracle_dir,
                ledger=RunLedger(oracle_dir, oracle.run_id),
                memory_condition=condition,
            ),
            borrowed_runs_root=tmp_path / "runs",
        ),
        executor=executor,
        stage_runtime=StageRuntime(
            run_id=oracle.run_id,
            user_id=oracle.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            FakeMemoryAdapter(), run_id=oracle.run_id, persona_id=oracle.persona_id
        ),
        progress=lambda _line: None,
    )

    await orchestrator.run()

    [first_call] = executor.first_calls
    # System prompt and the probe's own message only: 0 history messages.
    assert [m["role"] for m in first_call] == ["system", "user"]
    assert "acc question" not in json.dumps(first_call)
    system = first_call[0]["content"]
    if condition == "oracle_preference":
        assert "# Retrieved Memory\n\nIn work situations like this one, the user prefers Verbosity: Detailed" in system
    else:
        assert "# Retrieved Memory" not in system and "the user prefers" not in system
    assert "# Memory\n" not in system
    [view] = executor.probe_views
    # What the session executor writes into the probe's meta.yaml.
    assert view["extra_meta"]["prompt_history_messages_at_start"] == 0


def test_prompt_history_at_start_counts_past_the_start_index(tmp_path: Path):
    from harness.orchestrator import _prompt_history_at_start

    sessions = tmp_path / "pa_workspace" / "user_a" / "sessions"
    sessions.mkdir(parents=True)
    header = {"_type": "metadata", "key": "k", "metadata": {"other": 1},
              "session_schema_version": SessionManager._SESSION_SCHEMA_VERSION,
              "prompt_history_start_index": 1}
    body = [json.dumps({"role": "user", "content": "old"}), json.dumps({"role": "user", "content": "new"})]
    path = sessions / "k.jsonl"
    path.write_text("\n".join([json.dumps(header), *body]) + "\n", encoding="utf-8")
    # This line's history is the window past prompt_history_start_index.
    assert _prompt_history_at_start(tmp_path) == 1


# ---------------------------------------------------------------------------
# IX wrapper step boundary — contract 10 and contract 7 (T17)
# ---------------------------------------------------------------------------


class HoldingPACheckpoint(FakePACheckpoint):
    """A PA whose IX state still has open hold episodes when a step ends."""

    def __init__(self, calls: list, *, open_by_step: dict[str, list] | None = None):
        super().__init__(calls)
        self.open_by_step = open_by_step or {}
        self.closed: list[str] = []

    def close_ix_session(self, script_session_id: str) -> list[tuple[str, str]]:
        self.closed.append(script_session_id)
        # Closing is destructive whatever it finds: a retry must not inherit
        # the previous attempt's holds.
        return self.open_by_step.pop(script_session_id, [])


def test_every_task_tool_has_an_action_type() -> None:
    from pa.agent.ix.collector import assert_action_type_coverage

    # The shipped catalog is fully mapped.
    assert assert_benchmark_action_type_coverage(".")

    with pytest.raises(ValueError, match="brand_new_tool"):
        assert_action_type_coverage(["document_read", "brand_new_tool"])


def _stage_fingerprint(canonical: Path) -> list[tuple[str, str]]:
    """Every file under `canonical_stage/`, by path and content hash."""
    return sorted(
        (
            str(path.relative_to(canonical)),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in canonical.rglob("*")
        if path.is_file()
    )


@pytest.mark.asyncio
async def test_step_with_an_open_hold_fails_and_publishes_no_checkpoint(
    tmp_path: Path,
) -> None:
    plan = _two_acc_plan()
    calls: list = []
    manager = FakeCheckpointManager(calls)
    pa = HoldingPACheckpoint(
        calls,
        open_by_step={"acc_001": [("autonomy_level", "p1")]},
    )
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=pa,
    )
    canonical = tmp_path / "runs" / plan.run_id / "canonical_stage"
    before = _stage_fingerprint(canonical)

    with pytest.raises(OpenHoldEpisodeError, match="autonomy_level=p1"):
        await orchestrator.run()

    assert orchestrator.ledger.steps["acc_001"].status == "failed"
    # An open hold is its own kind: rerunning it reproduces it, so the ledger
    # says "human look" rather than leaving that in the exception text.
    assert orchestrator.ledger.steps["acc_001"].failure_kind == "ix_open_hold"
    assert manager.list_checkpoints() == []
    assert calls == []  # no drain, no save
    assert pa.closed == ["acc_001"]
    # The world this step wrote is discarded with its working stage: a failed
    # step publishes no cut point, so a committed world would have no snapshot
    # to roll it back (contract 7, T17). `marker.txt` is what the executor
    # wrote inside the step.
    assert _stage_fingerprint(canonical) == before
    assert not (canonical / "marker.txt").exists()
    assert not (tmp_path / "runs" / plan.run_id / "stage_work" / "acc_001").exists()


@pytest.mark.asyncio
async def test_a_clean_step_closes_its_ix_session_and_still_publishes(
    tmp_path: Path,
) -> None:
    plan = _two_acc_plan()
    calls: list = []
    manager = FakeCheckpointManager(calls)
    pa = HoldingPACheckpoint(calls)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=pa,
    )

    result = await orchestrator.run()

    assert result.executed == ["acc_001", "acc_002", "pretest_W_A"]
    assert manager.list_checkpoints() == ["acc_001", "acc_002"]
    # Both mainline step boundaries closed. The probe is not here on purpose:
    # it runs on a throwaway branch whose PA is never bound to this handle.
    assert pa.closed == ["acc_001", "acc_002"]


@pytest.mark.asyncio
async def test_a_failing_step_still_closes_its_ix_session(tmp_path: Path) -> None:
    """Success, exception, cancel and retry all clear the state."""
    plan = _two_acc_plan()
    calls: list = []
    pa = HoldingPACheckpoint(calls)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=FakeCheckpointManager(calls),
        pa_checkpoint=pa,
    )
    orchestrator.executor = FakeExecutor(fail_step="acc_001")

    with pytest.raises(RuntimeError, match="failed step"):
        await orchestrator.run()

    # The close ran, and it did not mask the step's own error.
    assert pa.closed == ["acc_001"]
    assert orchestrator.ledger.steps["acc_001"].status == "failed"


class _StubMainlinePA:
    """Only the pieces `MainlinePACheckpointHandle` reaches on a real PA.

    The `SessionManager` is real, because the cache it keeps is exactly what
    the forced-step tests below are about.
    """

    def __init__(self, workspace: Path) -> None:
        from memory.adapter import PromptHistoryEvictionPolicy

        self.prompt_history_eviction_policy = (
            PromptHistoryEvictionPolicy.after_file_consolidation
        )
        self.sessions = SessionManager(workspace)
        self.archived: list[list] = []
        self.memory_consolidator = SimpleNamespace(
            archive_messages_to_files=self._archive
        )

    async def _archive(self, messages) -> bool:
        self.archived.append(list(messages))
        return True

    async def drain_background_tasks(self) -> None:
        return None


async def _forced_step_writes(workspace: Path, session_key: str) -> None:
    """One forced exchange, committed by the executor's own two methods.

    The journal record is no longer seeded by hand: `_prepare_exchange` writes
    the pending one and `_commit_session_and_memory` extends the session and
    marks it, which is exactly what the pinned flush then consumes. The point
    the forced-step tests below are about is unchanged -- the manager: the
    forced executor is handed one of its own, so the PA bound during the
    previous acc step never sees these messages.
    """
    from harness.forced_event_executor import ForcedEventExecutor
    from harness.forced_event_replay import GatedReplay
    from harness.forced_event_schema import MessageItem
    from pa.agent.tools.registry import ToolRegistry

    sessions = SessionManager(workspace)
    # `_prepare_exchange` and `_commit_session_and_memory` read exactly two
    # fields off the transcript -- the event id the turn id is built from and
    # the persona id the adapter event carries -- and this exchange is pure
    # dialogue, so no tool is ever reached through the registry.
    transcript = SimpleNamespace(
        event=SimpleNamespace(event_id="evolv_01", persona_id="user_a")
    )
    executor = ForcedEventExecutor(
        transcript_path=workspace / "evolv_01.yaml",
        project_root=workspace,
        benchmark_session_timestamp=datetime(2026, 12, 8, 18, 30, tzinfo=timezone.utc),
        stage_runtime=SimpleNamespace(run_id="run_main", state_root=workspace),
        session_manager=sessions,
        pa_session_key=session_key,
        memory_adapter=None,
        memory_io_recorder=None,
        writer=None,
        require_committed_adapter_writes=False,
    )
    executor._replay = GatedReplay(gate=None)
    executor._open_journal()
    prepared = await executor._prepare_exchange(
        transcript,
        exchange_index=0,
        items=[
            MessageItem(
                type="message", offset_seconds=0, source="user",
                visibility="both", content="forced inbound",
            ),
            MessageItem(
                type="message", offset_seconds=5, source="pa",
                visibility="both", content="forced reply",
            ),
        ],
        registry=ToolRegistry(),
        actual_call_ids=[],
        actual_action_ids=[],
    )
    await executor._commit_session_and_memory(transcript, [prepared])


@pytest.mark.asyncio
async def test_a_pinned_flush_after_a_forced_step_keeps_the_forced_dialogue(
    tmp_path: Path,
) -> None:
    """The cut point must not save the acc PA's pre-forced session back."""
    from scripts.run_orchestrator import MainlinePACheckpointHandle

    workspace = tmp_path / "pa_workspace" / "user_a"
    session_key = "run_main"

    from memory.event_writer import build_turn_pool, turn_journal_for
    from memory.turn_journal import JournalRecord

    acc_pa = _StubMainlinePA(workspace)
    session = acc_pa.sessions.create(session_key)
    session.add_message("user", "acc question")
    session.add_message("assistant", "acc answer")
    acc_pa.sessions.save(session)
    acc_journal = turn_journal_for(acc_pa.sessions, session_key)
    acc_journal.append_pending(JournalRecord.for_pool(
        turn_id="acc_001:turn_1", span=(0, 2), timestamp="",
        pool=build_turn_pool([dict(m) for m in session.messages]),
    ))

    handle = MainlinePACheckpointHandle(session_key, _StubMainlinePA(workspace))
    handle.bind(acc_pa)
    await _forced_step_writes(workspace, session_key)

    orchestrator = _orchestrator(
        tmp_path,
        _two_acc_plan(),
        checkpoint_manager=None,
        pa_checkpoint=handle,
    )
    await orchestrator._flush_uncompacted_turns()

    saved = SessionManager(workspace).get(session_key)
    assert [message["content"] for message in saved.messages] == [
        "acc question",
        "acc answer",
        "forced inbound",
        "forced reply",
    ]
    assert saved.file_consolidated_end_index == 4
    assert saved.prompt_history_start_index == 4
    # The snapshot a probe branches from has the forced dialogue in memory.
    assert [record["content"] for record in acc_pa.archived[0]] == [
        "acc question",
        "acc answer",
        "forced inbound",
        "forced reply",
    ]


@pytest.mark.asyncio
async def test_a_forced_first_run_still_has_a_usable_cut_point(
    tmp_path: Path,
) -> None:
    """Starting or resuming at a forced step binds no session PA."""
    from scripts.run_orchestrator import MainlinePACheckpointHandle

    workspace = tmp_path / "pa_workspace" / "user_a"
    session_key = "run_main"

    mainline_pa = _StubMainlinePA(workspace)
    mainline_pa.sessions.create(session_key)
    handle = MainlinePACheckpointHandle(session_key, mainline_pa)  # never bound
    await _forced_step_writes(workspace, session_key)

    orchestrator = _orchestrator(
        tmp_path,
        _two_acc_plan(),
        checkpoint_manager=None,
        pa_checkpoint=handle,
    )
    await orchestrator._await_checkpoint_ready()
    await orchestrator._flush_uncompacted_turns()

    assert [record["content"] for record in mainline_pa.archived[0]] == [
        "forced inbound",
        "forced reply",
    ]
    saved = SessionManager(workspace).get(session_key)
    assert saved.file_consolidated_end_index == 2


AUTHORING_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/authoring"
FORCED_SCRIPT_PATH = (
    "Authoring/sessions/forced/transcripts/user_a/evolv_01/transcript.yaml"
)


def _forced_artifacts(tmp_path: Path) -> None:
    """The three artifacts a forced-event step is frozen from.

    Laid down under the orchestrator's own project root, so the run reads them
    the way a real run does. Same worked fixtures as
    `test_forced_event_executor.py:_fixture_project_root`.
    """
    files = {
        "outline": (
            tmp_path / "Authoring/outline/drafts/user_a/outline.yaml",
            AUTHORING_FIXTURES / "outline_acc_015.yaml",
        ),
        "draft": (
            tmp_path / "Authoring/sessions/forced/drafts/user_a/evolv_01.yaml",
            AUTHORING_FIXTURES / "forced_event_evolv_01.yaml",
        ),
        "transcript": (
            tmp_path / FORCED_SCRIPT_PATH,
            AUTHORING_FIXTURES / "forced_event_transcript_evolv_01_gated.yaml",
        ),
    }
    for target, source in files.values():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _forced_fixture_root(tmp_path: Path) -> Path:
    """`_fixture` writes its own user_a outline; put the FE one back."""
    root = _fixture(tmp_path)
    _forced_artifacts(tmp_path)
    return root


def _forced_plan() -> RunPlan:
    step = {
        "step_id": "evolv_01",
        "kind": "forced_event",
        "persona_id": "user_a",
        "script_path": FORCED_SCRIPT_PATH,
        "context": "work",
        "target_cell": "work_verbosity",
        "target_attribute": "verbosity",
        "target_setting": "Brief",
        "acc_num": 15,
        "benchmark_session_timestamp": "2026-12-08T18:30:00+00:00",
        "memory_mode": "read_write",
        "stage_policy": "commit",
    }
    return _plan().model_copy(update={"steps": [RunStep.model_validate(step)]})


def _forced_orchestrator(tmp_path: Path, plan: RunPlan, executor) -> Orchestrator:
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
            fixture_root=_forced_fixture_root(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            FakeMemoryAdapter(),
            run_id=plan.run_id,
            persona_id=plan.persona_id,
        ),
        progress=lambda line: None,
    )


@pytest.mark.asyncio
async def test_a_probe_branch_carries_the_runs_simulator_identity(tmp_path: Path):
    """A probe runs in a throwaway branch with its own workspace, adapter and
    recorder. The simulator identity used to reach it only because the real
    `loop_factory` closed over it, so nothing in the branch's `loop_context` or
    its `meta.yaml` said which persona spoke or which cells a forced event had
    already flipped. Both are passed explicitly now, as they are inline — and
    the identity arrives as the branch's own copy of the profile, rooted in the
    branch, because closing a session writes through it."""
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)
    executor = ProbeRecordingExecutor()
    sim_profile = load_simulator_profile(
        run_dir / "simulator_workspace" / plan.persona_id
    )
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint([]),
    )
    orchestrator.config.sim_profile = sim_profile
    orchestrator.ledger = ledger
    orchestrator.executor = executor

    await orchestrator.run()

    view = executor.probe_views[0]
    # Its own object, as of its own cut point — never the run-level one, which
    # by the time the probe batch runs carries every flip the run made.
    assert view["preference_overlay"] is not orchestrator.preference_overlay
    branch_profile = view["sim_profile"]
    assert branch_profile is not sim_profile
    assert branch_profile.id == sim_profile.id
    # The same persona, read out of the branch: `branch_from` copied the
    # workspace in, and the probe's own writes go nowhere else.
    assert Path(branch_profile.root) == (
        Path(view["extra_meta"]["branch_dir"])
        / "simulator_workspace"
        / plan.persona_id
    )
    # And the branch's own record says so, so the probe transcript can be read
    # without reconstructing the closure the loop factory was built with.
    assert view["extra_meta"]["sim_profile_root"] == str(branch_profile.root)
    assert view["extra_meta"]["preference_overlay"] == (
        view["preference_overlay"].preference_overrides()
    )


# ── probe batch runner ─────────────


def _two_probe_plan() -> RunPlan:
    """acc_001 → acc_002 (pinned) → pretest_W_A, fin_01: two probes, one pin."""
    plan = _two_acc_plan()
    steps = [step.model_dump(mode="json") for step in plan.steps]
    steps.append(
        {
            **steps[-1],
            "step_id": "fin_01",
            "kind": "final_probe",
            "script_path": "fin_01.yaml",
            "before_acc_num": None,
            "benchmark_session_timestamp": "2026-10-01T18:00:00+00:00",
        }
    )
    return plan.model_copy(update={"steps": [RunStep(**s) for s in steps]})


class FailOneProbeExecutor(FailingProbeExecutor):
    """Dies on `fail_step` the way `FailingProbeExecutor` does; the rest run."""

    def __init__(self, fail_step: str) -> None:
        super().__init__()
        self.fail_probe = fail_step

    async def execute(self, step, **kwargs):
        if step.step_id != self.fail_probe:
            return await FakeExecutor.execute(self, step, **kwargs)
        return await super().execute(step, **kwargs)


def _probe_orchestrator(tmp_path: Path, plan: RunPlan, executor) -> tuple:
    run_dir = tmp_path / "runs" / plan.run_id
    if not (run_dir / "ledger.json").exists():
        _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint([]),
    )
    orchestrator.ledger = ledger
    orchestrator.executor = executor
    for step in plan.steps:
        if step.kind in orchestrator_module.PROBE_KINDS:
            _write_probe_script(tmp_path, step)
    return orchestrator, ledger, run_dir


@pytest.mark.asyncio
async def test_probes_run_as_a_batch_after_the_timeline(tmp_path: Path):
    """The timeline loop runs only acc/forced; every probe follows, in order."""
    plan = _two_probe_plan()
    executor = ProbeRecordingExecutor()
    orchestrator, ledger, _run_dir = _probe_orchestrator(tmp_path, plan, executor)

    result = await orchestrator.run()

    assert result.executed == ["acc_001", "acc_002", "pretest_W_A", "fin_01"]
    assert [view["step_id"] for view in executor.probe_views] == [
        "pretest_W_A",
        "fin_01",
    ]
    # Both probes branched from the one pin, each in a branch of its own.
    assert {
        view["extra_meta"]["source_checkpoint_id"] for view in executor.probe_views
    } == {"acc_002"}
    assert len({view["extra_meta"]["branch_dir"] for view in executor.probe_views}) == 2
    assert all(ledger.steps[s].status == "done" for s in ("pretest_W_A", "fin_01"))


class SimMemoryProbeExecutor(ProbeRecordingExecutor):
    """Writes the simulated user's memory the way a session close does.

    Every probe reads first and is then held until `width` of them have read,
    so the writes really do overlap. Sharing one profile would make the reads
    order-dependent instead of a race — and with the write held back, a shared
    profile could not even be caught by reading after the fact.
    """

    def __init__(self, width: int = 1) -> None:
        super().__init__()
        self.memory_read: list[tuple[str, str]] = []
        self.width = width
        self.arrived = 0
        self._full = asyncio.Event()

    async def execute(self, step, **kwargs):
        profile = (kwargs.get("loop_context") or {}).get("sim_profile")
        if profile is not None:
            store = UserMemoryStore(
                Path(profile.root), user_name=profile.display_name
            )
            self.memory_read.append((step.step_id, store.read_long_term()))
            self.arrived += 1
            if self.arrived >= self.width:
                self._full.set()
            await self._full.wait()
            store.memory_file.write_text(
                f"{step.step_id} wrote this\n", encoding="utf-8"
            )
            store.append_history(f"{step.step_id} closed a session")
        return await super().execute(step, **kwargs)


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.asyncio
async def test_a_probes_simulator_memory_write_stays_in_its_branch(
    tmp_path: Path,
):
    """Closing a session writes `MEMORY.md`/`HISTORY.md` through the profile.
    Each probe is handed the branch's copy, so both probes read the cut point's
    memory rather than each other's write — at N=2, where the two writes are
    concurrent and one shared profile would be a race, not just an ordering —
    and the run-level profile comes out of the batch byte for byte as it went
    in."""
    plan = _two_probe_plan()
    executor = SimMemoryProbeExecutor(width=2)
    orchestrator, _ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=2
    )
    sim_root = run_dir / "simulator_workspace" / plan.persona_id
    (sim_root / "memory").mkdir(parents=True, exist_ok=True)
    (sim_root / "memory" / "MEMORY.md").write_text(
        "cut point memory\n", encoding="utf-8"
    )
    orchestrator.config.sim_profile = load_simulator_profile(sim_root)
    before = _tree_digest(sim_root)

    result = await orchestrator.run()

    assert set(result.executed) == {
        "acc_001",
        "acc_002",
        "pretest_W_A",
        "fin_01",
    }
    assert sorted(executor.memory_read) == [
        ("fin_01", "cut point memory\n"),
        ("pretest_W_A", "cut point memory\n"),
    ]
    assert _tree_digest(sim_root) == before


@pytest.mark.asyncio
async def test_a_failed_probe_does_not_stop_the_later_probes(tmp_path: Path):
    """`failed` + `failure_kind`, evidence kept, next probe runs."""
    plan = _two_probe_plan()
    executor = FailOneProbeExecutor("pretest_W_A")
    orchestrator, ledger, run_dir = _probe_orchestrator(tmp_path, plan, executor)

    result = await orchestrator.run()

    assert result.failed == ["pretest_W_A"]
    assert result.executed == ["acc_001", "acc_002", "fin_01"]
    assert ledger.steps["pretest_W_A"].status == "failed"
    assert ledger.steps["pretest_W_A"].failure_kind == "other"
    assert ledger.steps["fin_01"].status == "done"
    # #49: the failed probe's evidence was copied out before its branch went.
    assert (run_dir / "steps" / "pretest_W_A" / "transcript.jsonl").is_file()
    assert list((run_dir / "branches" / "pretest_W_A").iterdir()) == []
    assert list((run_dir / "branches" / "fin_01").iterdir()) == []


@pytest.mark.asyncio
async def test_restart_after_a_failed_probe_reruns_only_it_without_rollback(
    tmp_path: Path,
):
    plan = _two_probe_plan()
    first, ledger, run_dir = _probe_orchestrator(
        tmp_path, plan, FailOneProbeExecutor("pretest_W_A")
    )
    await first.run()
    assert ledger.steps["pretest_W_A"].status == "failed"

    # A probe can also be left at `running` by a hard death; same treatment.
    ledger.mark_running("pretest_W_A")

    second_executor = ProbeRecordingExecutor()
    second, ledger2, _ = _probe_orchestrator(tmp_path, plan, second_executor)
    result = await second.run()

    assert result.executed == ["pretest_W_A"]
    assert sorted(result.skipped) == ["acc_001", "acc_002", "fin_01"]
    assert [view["step_id"] for view in second_executor.probe_views] == ["pretest_W_A"]
    assert ledger2.steps["pretest_W_A"].status == "done"
    assert ledger2.steps["pretest_W_A"].failure_kind is None
    # Nothing was rolled back: no crash scene, the mainline world untouched.
    assert not (run_dir / "crash_scene").exists()
    assert (run_dir / "canonical_stage" / "marker.txt").read_text(
        encoding="utf-8"
    ) == "acc_002"


@pytest.mark.asyncio
async def test_a_probe_only_run_with_no_checkpoint_resumes(tmp_path: Path):
    """The memory-removed baseline has only probes and no local snapshot; a
    probe left `failed` must not make `resume_or_restore` raise."""
    from harness.orchestrator import resume_or_restore

    plan = _two_probe_plan()
    probes_only = plan.model_copy(
        update={
            "steps": [
                s for s in plan.steps if s.kind in orchestrator_module.PROBE_KINDS
            ]
        }
    )
    ledger = RunLedger(tmp_path / "baseline", plan.run_id)
    ledger.mark_failed("pretest_W_A", "died", kind="other")
    ledger.mark_running("fin_01")

    assert resume_or_restore(ledger, None, probes_only) is None
    assert ledger.decision_for("pretest_W_A") == "run"
    assert ledger.decision_for("fin_01") == "run"


@pytest.mark.asyncio
async def test_orphan_probe_branches_are_reaped_at_run_start(tmp_path: Path):
    plan = _two_probe_plan()
    lines: list[str] = []
    orchestrator, _ledger, run_dir = _probe_orchestrator(
        tmp_path, plan, ProbeRecordingExecutor()
    )
    orchestrator.progress = lines.append
    orphan = run_dir / "branches" / "fin_01" / "01"
    (orphan / "steps").mkdir(parents=True)
    (run_dir / "branches" / "fin_01" / "notes.txt").write_text("x", encoding="utf-8")

    await orchestrator.run()

    assert not orphan.exists()
    assert "reap orphan branch branches/fin_01/01" in lines
    # Only the attempt directories are reaped; a stray file is left alone.
    assert (run_dir / "branches" / "fin_01" / "notes.txt").is_file()


class _RecordingQdrantAdmin:
    """Records whether the branch record was still on disk at delete time.

    The name of a branch's collection lives nowhere but that branch's
    `branch_namespace.json`, so a delete that fired after the directory was
    gone could not have been handed the name at all: the file's presence at
    delete time is what proves the release ran before the `rmtree`.
    """

    def __init__(self, branch_dir: Path, *, fail: bool = False) -> None:
        self._record = branch_dir / "backend_state" / "branch_namespace.json"
        self._fail = fail
        self.deleted: list[tuple[str, bool]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def delete_collection(self, *, collection_name: str) -> None:
        self.deleted.append((collection_name, self._record.is_file()))
        if self._fail:
            raise RuntimeError("backend unreachable")


class _RecordingMem0Client:
    def __init__(self, branch_dir: Path, *, fail: bool = False) -> None:
        self.qdrant_admin = _RecordingQdrantAdmin(branch_dir, fail=fail)

    @property
    def deleted(self) -> list[tuple[str, bool]]:
        return self.qdrant_admin.deleted


class _RecordingMemosClient:
    def __init__(self, branch_dir: Path, *, fail: bool = False) -> None:
        self._record = branch_dir / "backend_state" / "branch_namespace.json"
        self._fail = fail
        self.deleted: list[tuple[str, bool]] = []

    def delete_cube(self, cube_id: str) -> None:
        self.deleted.append((cube_id, self._record.is_file()))
        if self._fail:
            raise RuntimeError("backend unreachable")


def _orphan_branch_reaper(
    tmp_path: Path, condition: str, client_for, *, namespace: str | None
) -> tuple:
    """A run whose one probe branch died hard, and the reaper that finds it.

    `namespace` None is a file-backed condition: `branch_from` writes no
    `branch_namespace.json` because the whole state is the directory copy.
    """
    plan = _two_probe_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    branch_dir = run_dir / "branches" / "fin_01" / "01"
    (branch_dir / "backend_state").mkdir(parents=True)
    if namespace is not None:
        (branch_dir / "backend_state" / "branch_namespace.json").write_text(
            json.dumps(
                {
                    "memory_condition": condition,
                    "namespace": namespace,
                    "source_namespace": namespace.split("_br_")[0],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    client = client_for(branch_dir)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=CheckpointManager(
            run_dir,
            ledger=ledger,
            memory_condition=condition,
            backend_client=client,
            rolling_window=10,
        ),
        pa_checkpoint=None,
    )
    lines: list[str] = []
    orchestrator.progress = lines.append
    return orchestrator, branch_dir, client, lines


def test_reaping_an_orphan_branch_releases_its_mem0_collection_first(
    tmp_path: Path,
) -> None:
    """The reaper used to `rmtree` only.

    A branch that died hard still owns the collection it restored, and the
    directory is the only thing that names it — so removing the directory
    first leaks the collection and throws the name away with it.
    """
    orchestrator, branch_dir, client, lines = _orphan_branch_reaper(
        tmp_path,
        "mem0",
        _RecordingMem0Client,
        namespace="mempa_run_001_br_fin_01_01",
    )

    orchestrator._reap_orphan_branches()

    # The entity sibling goes with it, and both deletes fired while the record
    # that names them was still on disk.
    assert client.deleted == [
        ("mempa_run_001_br_fin_01_01_entities", True),
        ("mempa_run_001_br_fin_01_01", True),
    ]
    assert not branch_dir.exists()
    assert "reap orphan branch branches/fin_01/01" in lines


def test_reaping_an_orphan_branch_releases_its_memos_cube_first(
    tmp_path: Path,
) -> None:
    orchestrator, branch_dir, client, lines = _orphan_branch_reaper(
        tmp_path, "memos", _RecordingMemosClient, namespace="cube_001_br_fin_01_01"
    )

    orchestrator._reap_orphan_branches()

    assert client.deleted == [("cube_001_br_fin_01_01", True)]
    assert not branch_dir.exists()
    assert "reap orphan branch branches/fin_01/01" in lines


@pytest.mark.parametrize(
    "condition,client_for,namespace",
    [
        (
            "mem0",
            lambda branch_dir: _RecordingMem0Client(branch_dir, fail=True),
            "mempa_run_001_br_fin_01_01",
        ),
        (
            "memos",
            lambda branch_dir: _RecordingMemosClient(branch_dir, fail=True),
            "cube_001_br_fin_01_01",
        ),
    ],
)
def test_an_orphan_branch_whose_release_fails_is_kept(
    tmp_path: Path, condition: str, client_for, namespace: str
) -> None:
    """A failed release must not take the name down with the directory."""
    orchestrator, branch_dir, _client, lines = _orphan_branch_reaper(
        tmp_path, condition, client_for, namespace=namespace
    )

    orchestrator._reap_orphan_branches()

    assert (branch_dir / "backend_state" / "branch_namespace.json").is_file()
    assert lines == [
        f"orphan branch kept: {branch_dir} release failed: backend unreachable"
    ]


def test_an_orphan_branch_is_kept_when_no_manager_can_release_it(
    tmp_path: Path,
) -> None:
    """No manager is no backend: the namespace cannot be released at all, so
    the directory stays and the name with it."""
    orchestrator, branch_dir, _client, lines = _orphan_branch_reaper(
        tmp_path,
        "mem0",
        _RecordingMem0Client,
        namespace="mempa_run_001_br_fin_01_01",
    )
    orchestrator.config.checkpoint_manager = None

    orchestrator._reap_orphan_branches()

    assert (branch_dir / "backend_state" / "branch_namespace.json").is_file()
    assert lines == [
        f"orphan branch kept: {branch_dir} release failed: "
        "no checkpoint manager to release the branch namespace through"
    ]


def test_reaping_a_branch_with_no_namespace_record_only_removes_it(
    tmp_path: Path,
) -> None:
    """`nanobot_file_memory` and `no_memory` keep every byte inside the
    directory, so there is nothing to release and nothing to keep it for."""
    orchestrator, branch_dir, client, lines = _orphan_branch_reaper(
        tmp_path, "nanobot_file_memory", _RecordingMem0Client, namespace=None
    )

    orchestrator._reap_orphan_branches()

    assert client.deleted == []
    assert not branch_dir.exists()
    assert "reap orphan branch branches/fin_01/01" in lines


class FakeBranchPA:
    """The PA the loop factory would build inside a probe branch."""

    def __init__(self, open_episodes: list[tuple[str, str]] | None = None):
        self._ix_selection_cache: dict = {}
        self.open_episodes = list(open_episodes or [])
        self.closed: list[tuple[str, str | None]] = []

    def close_ix_session(self, session_key, script_session_id=None):
        self.closed.append((session_key, script_session_id))
        return list(self.open_episodes)


class BindingProbeExecutor(ProbeRecordingExecutor):
    """Binds a fake branch PA through `loop_context["pa_binder"]`, as the real
    `RealSessionLoop` does once it has built the PA."""

    def __init__(self, pa: FakeBranchPA) -> None:
        super().__init__()
        self.pa = pa
        self.bound = 0

    async def execute(self, step, **kwargs):
        loop_context = kwargs.get("loop_context")
        if loop_context is not None:
            loop_context["pa_binder"](self.pa)
            self.bound += 1
        return await super().execute(step, **kwargs)


@pytest.mark.asyncio
async def test_force_ix_reaches_the_probe_pa_through_its_own_handle(
    tmp_path: Path,
):
    """#52: the seed lands on the branch PA; the mainline handle is never
    bound to it (`FakePACheckpoint` has no `bind`, so binding would raise)."""
    plan = _two_probe_plan()
    pa = FakeBranchPA()
    executor = BindingProbeExecutor(pa)
    orchestrator, ledger, _ = _probe_orchestrator(tmp_path, plan, executor)
    orchestrator.config.forced_ix = {
        "fin_01": {"process_visibility": "Full narration"},
    }

    await orchestrator.run()

    assert executor.bound == 2
    session_key = plan.pa_session_key_for_step(plan.steps[-1])
    seeded = pa._ix_selection_cache[(session_key, "fin_01")]["process_visibility"]
    assert seeded.setting == "Full narration"
    assert seeded.forced is True
    # The other probe carried no seed, and both closed their IX session.
    assert (session_key, "pretest_W_A") not in pa._ix_selection_cache
    assert pa.closed == [(session_key, "pretest_W_A"), (session_key, "fin_01")]
    assert ledger.steps["fin_01"].status == "done"


@pytest.mark.asyncio
async def test_an_open_hold_at_a_probe_boundary_lands_in_meta_yaml(
    tmp_path: Path,
):
    """#50: recorded in `steps/<probe>/meta.yaml`, the probe is still done."""
    import yaml

    plan = _two_probe_plan()
    pa = FakeBranchPA(open_episodes=[("autonomy_level", "ep_0007")])
    orchestrator, ledger, run_dir = _probe_orchestrator(
        tmp_path, plan, BindingProbeExecutor(pa)
    )

    result = await orchestrator.run()

    assert result.failed == []
    assert ledger.steps["pretest_W_A"].status == "done"
    meta = yaml.safe_load(
        (run_dir / "steps" / "pretest_W_A" / "meta.yaml").read_text(encoding="utf-8")
    )
    assert meta["ok"] is True  # what the executor wrote is kept
    assert meta[orchestrator_module.OPEN_IX_EPISODES_KEY] == [
        {"attribute": "autonomy_level", "episode_id": "ep_0007"}
    ]


@pytest.mark.asyncio
async def test_only_steps_runs_just_the_named_probes(tmp_path: Path):
    plan = _two_probe_plan()
    executor = ProbeRecordingExecutor()
    orchestrator, ledger, _ = _probe_orchestrator(tmp_path, plan, executor)
    orchestrator.config.only_steps = frozenset({"fin_01"})

    result = await orchestrator.run()

    # The timeline phase is unaffected; the other probe is left untouched.
    assert result.executed == ["acc_001", "acc_002", "fin_01"]
    assert result.skipped == ["pretest_W_A"]
    assert "pretest_W_A" not in ledger.steps

    orchestrator.config.only_steps = frozenset({"acc_001", "fin_99"})
    with pytest.raises(ValueError, match=r"\['acc_001', 'fin_99'\]"):
        await orchestrator.run()


def _unpinned_probe_plan() -> RunPlan:
    """One forced event, then three probes, and no pin in front of any of them.

    `pinned_step_ids` derives its cut points from accumulation steps alone, so
    a timeline that has not reached one yet leaves every probe without a
    checkpoint to branch from — the same shape as pointing a probe phase at a
    run whose timeline never ran.
    """
    plan = _two_acc_plan()
    forced = plan.steps[0].model_dump(mode="json")
    forced.update(
        {
            "step_id": "evolv_01",
            "kind": "forced_event",
        }
    )
    probe = plan.steps[2].model_dump(mode="json")
    probes = [
        {
            **probe,
            "step_id": step_id,
            "kind": "final_probe",
            "script_path": f"{step_id}.yaml",
            "probe_overlay": {"owner_probe_id": step_id},
            "before_acc_num": None,
            "benchmark_session_timestamp": "2026-10-01T18:00:00+00:00",
        }
        for step_id in ("fin_01", "fin_02", "fin_03")
    ]
    return plan.model_copy(
        update={"steps": [RunStep(**step) for step in (forced, *probes)]}
    )


@pytest.mark.asyncio
async def test_a_missing_pin_stops_the_run_instead_of_failing_every_probe(
    tmp_path: Path,
) -> None:
    """A defect in what every branch is cut from is not one probe failing.

    The plain `RuntimeError` this used to raise fell past the batch's re-raise
    arm, so the first probe was recorded as failed and the next N died the same
    way — N failure records for one misconfiguration.
    """
    fixture_root = _fixture(tmp_path)
    plan = _unpinned_probe_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=_file_memory_manager(run_dir, ledger),
        pa_checkpoint=FakePACheckpoint([]),
    )
    orchestrator.ledger = ledger
    assert fixture_root.is_dir()

    with pytest.raises(
        orchestrator_module.ProbeRunConfigurationError,
        match="no pinned checkpoint to branch from and does not borrow one: "
        "step_id='fin_01'",
    ):
        await orchestrator.run()

    # No probe is recorded as failed, and the batch stopped at the first one:
    # the other two were never started.
    assert [
        step_id
        for step_id, record in ledger.steps.items()
        if record.status == "failed"
    ] == []
    assert ledger.steps["fin_01"].status == "running"
    assert not {"fin_02", "fin_03"} & set(ledger.steps)


@pytest.mark.asyncio
async def test_a_placeholder_step_is_skipped_without_touching_the_pa(
    tmp_path: Path,
) -> None:
    """A gap day holds its acc slot and runs nothing.

    Its script is not on disk (the Outline authored no session for it), so a
    placeholder that reached the executor would fail the run at a slot that
    was never meant to run.
    """
    plan = _two_acc_plan()
    steps = [step.model_dump(mode="json") for step in plan.steps]
    gap = {
        **steps[0],
        "step_id": "acc_003",
        "script_path": "acc_003.yaml",
        "acc_num": 3,
        "placeholder": True,
    }
    # The timeline alone: a probe with no checkpoint manager stops the run.
    plan = plan.model_copy(
        update={"steps": [RunStep(**step) for step in (*steps[:2], gap)]}
    )
    orchestrator = _orchestrator(
        tmp_path, plan, checkpoint_manager=None, pa_checkpoint=None
    )
    assert not (tmp_path / "acc_003.yaml").exists()

    result = await orchestrator.run()

    assert result.executed == ["acc_001", "acc_002"]
    assert result.skipped == ["acc_003"]
    assert [call[0] for call in orchestrator.executor.calls] == ["acc_001", "acc_002"]
    assert "acc_003" not in orchestrator.ledger.steps


def _pre_event_probe_plan() -> RunPlan:
    """acc_001 (pinned) → pre_01 → evolv_01, the shape a run really has.

    The probe sits before the event on the timeline but *runs* after it: the
    timeline finishes first and every probe is a job of its own, so by
    the time `pre_01` branches acc_001, the ledger already holds evolv_01's
    flip.
    """
    plan = _plan()
    steps = [
        plan.steps[0].model_dump(mode="json"),
        {
            **plan.steps[1].model_dump(mode="json"),
            "step_id": "pre_01",
            "script_path": "pre_01.yaml",
            "probe_overlay": {"owner_probe_id": "pre_01"},
            "target_cell": "work_autonomy_level",
            "target_attribute": "autonomy_level",
            "target_setting": "Self-directed",
            "before_acc_num": 2,
            "benchmark_session_timestamp": "2026-10-01T10:00:00+00:00",
        },
        {
            **plan.steps[0].model_dump(mode="json"),
            "step_id": "evolv_01",
            "kind": "forced_event",
            "script_path": "evolv_01.yaml",
            "target_cell": "work_autonomy_level",
            "target_attribute": "autonomy_level",
            "acc_num": 2,
            "benchmark_session_timestamp": "2026-10-01T11:00:00+00:00",
        },
    ]
    return plan.model_copy(update={"steps": [RunStep(**s) for s in steps]})


@pytest.mark.asyncio
async def test_a_pre_event_probe_meta_omits_the_cell_its_own_event_flips(
    tmp_path: Path,
):
    """The probe reads the pre-event setting, so its simulator must hold it.

    Live on user_b every pre probe ran the other way round: the run-level
    overlay object was handed to the branch, and `steps/pre_01/meta.yaml`
    recorded `work.tone_formality: Consultative` — the POST setting of the very
    cell the probe was cut to read as `Casual`.
    """
    plan = _pre_event_probe_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    (tmp_path / "pre_01.yaml").write_text(
        "meta:\n"
        "  artifact_kind: normal_session\n"
        "  tool_status: world_toolbox\n"
        "  persona_id: user_a\n"
        "  session_id: pre_01\n"
        "  probe_kind: pre_event\n"
        "  target_cell: work_verbosity\n",
        encoding="utf-8",
    )
    # `harness.gate.resolve_session_gate` resolves a probe's gate out of the
    # probe outline, so the branch needs one on disk the way a real run has
    # one. `work_verbosity` is compressed, so the entry carries no gate and the
    # probe resolves to None -- which is what this test's step wants.
    probe_outline = tmp_path / "Authoring/probe/outline/user_a/outline.yaml"
    probe_outline.parent.mkdir(parents=True, exist_ok=True)
    probe_outline.write_text(
        "probes:\n"
        "- id: pre_01\n"
        "  kind: pre_event\n"
        "  cell: work_verbosity\n"
        "  world: []\n",
        encoding="utf-8",
    )
    ledger = RunLedger(run_dir, plan.run_id)
    # The event has replayed and written its flip, as it has by the time any
    # probe runs.
    ledger.mark_done(
        "evolv_01",
        summary={LEDGER_OVERLAY_KEY: {"work_autonomy_level": "Reactive"}},
    )
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=_file_memory_manager(run_dir, ledger),
        pa_checkpoint=FakePACheckpoint([]),
    )
    executor = ProbeRecordingExecutor()
    orchestrator.ledger = ledger
    orchestrator.executor = executor

    await orchestrator.run()

    view = executor.probe_views[0]
    assert view["step_id"] == "pre_01"
    assert view["preference_overlay"].cells == {}
    assert view["extra_meta"]["preference_overlay"] == {}
    # The run-level object still holds the flip: only the probe's view is cut.
    assert orchestrator.preference_overlay.cells == {
        "work_autonomy_level": "Reactive"
    }


# ── probe batch dispatcher ──────────────────

#: How a finished step is read off the tee'd log in the launch runbook. Every
#: line the dispatcher adds has to fall outside it.
_DONE_LINE = re.compile(r"^\[\d{3}/\d{3}\] \S+ done ")


def _many_probe_plan(count: int) -> RunPlan:
    """acc_001 → acc_002 (pinned) → `count` final probes off the one pin."""
    plan = _two_acc_plan()
    timeline = [step.model_dump(mode="json") for step in plan.steps[:-1]]
    probe = plan.steps[-1].model_dump(mode="json")
    probes = [
        {
            **probe,
            "step_id": f"fin_{number:02d}",
            "kind": "final_probe",
            "script_path": f"fin_{number:02d}.yaml",
            "probe_overlay": {"owner_probe_id": f"fin_{number:02d}"},
            "before_acc_num": None,
            "benchmark_session_timestamp": "2026-10-01T18:00:00+00:00",
        }
        for number in range(1, count + 1)
    ]
    return plan.model_copy(
        update={"steps": [RunStep(**step) for step in (*timeline, *probes)]}
    )


def _probe_ids(plan: RunPlan) -> list[str]:
    return [
        step.step_id
        for step in plan.steps
        if step.kind in orchestrator_module.PROBE_KINDS
    ]


def _batch_orchestrator(
    tmp_path: Path,
    plan: RunPlan,
    executor,
    *,
    concurrency: int,
    lines: list[str] | None = None,
    stop_requested=None,
) -> tuple:
    run_dir = tmp_path / "runs" / plan.run_id
    if not (run_dir / "ledger.json").exists():
        _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=_file_memory_manager(run_dir, ledger),
        pa_checkpoint=FakePACheckpoint([]),
        stop_requested=stop_requested,
    )
    orchestrator.ledger = ledger
    orchestrator.executor = executor
    orchestrator.config.probe_concurrency = concurrency
    if lines is not None:
        orchestrator.progress = lines.append
    for step in plan.steps:
        if step.kind in orchestrator_module.PROBE_KINDS:
            _write_probe_script(tmp_path, step)
    return orchestrator, ledger, run_dir


class _WidthGateExecutor(FakeExecutor):
    """Holds every probe until `width` of them are inside `execute` at once.

    Deadlocks rather than passing if the dispatcher never runs `width` probes
    together, and keeps the high-water mark so the other half — never more
    than N of them — is asserted on the same run. Timeline steps go straight
    through: they run before the batch and would never fill the gate.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.live = 0
        self.max_live = 0
        self._full = asyncio.Event()

    async def execute(self, step, **kwargs):
        if step.kind not in orchestrator_module.PROBE_KINDS:
            return await super().execute(step, **kwargs)
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        if self.live >= self.width:
            self._full.set()
        try:
            await self._full.wait()
            return await super().execute(step, **kwargs)
        finally:
            self.live -= 1


class _ReverseFinishExecutor(FakeExecutor):
    """Every probe waits for the rest, then they finish in reverse order."""

    def __init__(self, probe_ids: list[str]) -> None:
        super().__init__()
        self.probe_ids = list(probe_ids)
        self.turn = {step_id: asyncio.Event() for step_id in self.probe_ids}
        self.arrived = 0
        self.finished: list[str] = []

    async def execute(self, step, **kwargs):
        if step.kind not in orchestrator_module.PROBE_KINDS:
            return await super().execute(step, **kwargs)
        self.arrived += 1
        if self.arrived == len(self.probe_ids):
            self.turn[self.probe_ids[-1]].set()
        await self.turn[step.step_id].wait()
        self.finished.append(step.step_id)
        position = self.probe_ids.index(step.step_id)
        if position:
            self.turn[self.probe_ids[position - 1]].set()
        return await super().execute(step, **kwargs)


class _StopperBarrierExecutor(FakeExecutor):
    """`width` probes enter together; one raises, the rest finish after it.

    The others are held until the stopper has been raised, so a worker that
    comes back for more work finds the queue already closed — which is the
    whole claim: no probe is STARTED after a stopper.
    """

    def __init__(self, width: int, stop_step: str, error: BaseException) -> None:
        super().__init__()
        self.width = width
        self.stop_step = stop_step
        self.error = error
        self.entered: list[str] = []
        self._full = asyncio.Event()
        self._stopped = asyncio.Event()

    async def execute(self, step, **kwargs):
        if step.kind not in orchestrator_module.PROBE_KINDS:
            return await super().execute(step, **kwargs)
        self.entered.append(step.step_id)
        if len(self.entered) >= self.width:
            self._full.set()
        await self._full.wait()
        if step.step_id == self.stop_step:
            self._stopped.set()
            raise self.error
        await self._stopped.wait()
        return await super().execute(step, **kwargs)


class _StopWhenFullExecutor(FakeExecutor):
    """The stop flag is set while `width` probes are inside the executor."""

    def __init__(self, width: int, flag: dict) -> None:
        super().__init__()
        self.width = width
        self.flag = flag
        self.live = 0
        self._full = asyncio.Event()

    async def execute(self, step, **kwargs):
        if step.kind not in orchestrator_module.PROBE_KINDS:
            return await super().execute(step, **kwargs)
        self.live += 1
        if self.live >= self.width:
            self.flag["requested"] = True
            self._full.set()
        try:
            await self._full.wait()
            return await super().execute(step, **kwargs)
        finally:
            self.live -= 1


class _DeadJudgeProbeExecutor(FailingProbeExecutor):
    """Writes the probe's evidence, then dies the way a dead judge does.

    `SimulatorLoop.run_session` re-raises the cause rather than the void, so
    `JudgeUnavailable` is what the step actually sees.

    `width` probes are held together before the judge dies, and the survivors
    are released only after it has: a worker coming back for more work then
    finds the abort already up, which is what makes "nothing queued behind it
    started" a fact of the dispatcher rather than of the scheduling order.
    """

    def __init__(self, stop_step: str, width: int = 1) -> None:
        super().__init__()
        self.stop_step = stop_step
        self.width = width
        self.entered: list[str] = []
        self._full = asyncio.Event()
        self._stopped = asyncio.Event()

    async def execute(self, step, **kwargs):
        if step.kind not in orchestrator_module.PROBE_KINDS:
            return await FakeExecutor.execute(self, step, **kwargs)
        self.entered.append(step.step_id)
        if len(self.entered) >= self.width:
            self._full.set()
        await self._full.wait()
        if step.step_id != self.stop_step:
            await self._stopped.wait()
            return await FakeExecutor.execute(self, step, **kwargs)
        try:
            return await super().execute(step, **kwargs)
        except RuntimeError:
            self._stopped.set()
            raise JudgeUnavailable(
                "ollama is not reachable at http://127.0.0.1:11434"
            ) from None


@pytest.mark.asyncio
async def test_one_worker_is_the_serial_batch_it_replaced(tmp_path: Path):
    """N=1: same order, same ledger, and the two step lines untouched.

    The dispatcher's own lines are a separate kind that falls outside the
    runbook's `grep`, so a log written at any N still reads back the same way.
    """
    plan = _many_probe_plan(3)
    total = len(plan.steps)
    lines: list[str] = []
    executor = ProbeRecordingExecutor()
    orchestrator, ledger, _ = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=1, lines=lines
    )

    result = await orchestrator.run()

    probe_ids = _probe_ids(plan)
    assert result.executed == ["acc_001", "acc_002", *probe_ids]
    assert [view["step_id"] for view in executor.probe_views] == probe_ids
    assert all(ledger.steps[step_id].status == "done" for step_id in probe_ids)

    running = format_step_running(
        3, total, plan.steps[2], plan.memory_condition
    )
    first = lines.index(running)
    assert lines[first + 1] == "probe slot open id=fin_01 inflight=1/1"
    assert lines[first + 2].startswith(f"[003/{total:03d}] fin_01 done ")
    assert lines[first + 3] == "probe slot close id=fin_01 status=done"
    # Every step still prints exactly one greppable done line, and no slot
    # line is mistaken for one.
    assert sum(1 for line in lines if _DONE_LINE.match(line)) == total
    assert not [
        line
        for line in lines
        if line.startswith("probe slot ") and _DONE_LINE.match(line)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [3, 4])
async def test_workers_overlap_requested_probes_and_never_more(
    tmp_path: Path, width: int,
):
    """N probes really run together, and N+1 never do."""
    plan = _many_probe_plan(width * 2)
    lines: list[str] = []
    executor = _WidthGateExecutor(width)
    orchestrator, ledger, _ = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=width, lines=lines
    )

    result = await orchestrator.run()

    probe_ids = _probe_ids(plan)
    assert executor.max_live == width
    assert set(result.executed) == {"acc_001", "acc_002", *probe_ids}
    assert all(ledger.steps[step_id].status == "done" for step_id in probe_ids)
    # The slot lines say how full the batch was, and stay out of the way of
    # the runbook's done-line grep.
    assert f"probe slot open id=fin_{width:02d} inflight={width}/{width}" in lines
    assert not [
        line
        for line in lines
        if line.startswith("probe slot ") and _DONE_LINE.match(line)
    ]
    assert sum(1 for line in lines if _DONE_LINE.match(line)) == len(plan.steps)
    assert not orchestrator.in_flight


@pytest.mark.asyncio
async def test_the_ledger_does_not_depend_on_the_completion_order(
    tmp_path: Path,
):
    """Probes finish backwards; every one of them still lands `done`."""
    plan = _many_probe_plan(3)
    probe_ids = _probe_ids(plan)
    executor = _ReverseFinishExecutor(probe_ids)
    orchestrator, ledger, _ = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=3
    )

    result = await orchestrator.run()

    assert executor.finished == list(reversed(probe_ids))
    assert set(result.executed) == {"acc_001", "acc_002", *probe_ids}
    assert {
        step_id: ledger.steps[step_id].status for step_id in probe_ids
    } == {step_id: "done" for step_id in probe_ids}
    assert all(ledger.steps[step_id].artifacts for step_id in probe_ids)


@pytest.mark.asyncio
async def test_a_plain_probe_failure_at_three_workers_fails_only_that_probe(
    tmp_path: Path,
):
    """A probe's own failure is still one record and the batch goes on."""
    plan = _many_probe_plan(3)
    executor = FailOneProbeExecutor("fin_02")
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=3
    )

    result = await orchestrator.run()

    assert result.failed == ["fin_02"]
    assert set(result.executed) == {"acc_001", "acc_002", "fin_01", "fin_03"}
    assert ledger.steps["fin_02"].status == "failed"
    assert ledger.steps["fin_02"].failure_kind == "other"
    assert (run_dir / "steps" / "fin_02" / "transcript.jsonl").is_file()
    assert ledger.steps["fin_01"].status == "done"
    assert ledger.steps["fin_03"].status == "done"


@pytest.mark.parametrize(
    "stopper_class",
    [
        orchestrator_module.ProbeRunConfigurationError,
        orchestrator_module.IncompleteCheckpointError,
    ],
    ids=["probe_run_configuration", "incomplete_checkpoint"],
)
@pytest.mark.asyncio
async def test_a_stopper_starts_nothing_new_and_lets_the_in_flight_finish(
    tmp_path: Path,
    stopper_class,
):
    """N+2 probes, one stopper, N-1 in flight, 2 queued.

    A semaphore over tasks created up front cannot promise this — the queued
    tasks were already scheduled. The queue can: what is still on it when the
    abort flag goes up is never dequeued, so it is never marked `running` and
    the next resume runs it.

    Once per stopper class that is not the judge: `IncompleteCheckpointError`
    is reachable from `branch_from`, i.e. after `mark_running` and after the
    slot opened, which is the leg that leaves a probe `running` with its slot
    unclosed. `JudgeUnavailable` is the test below, because it gets a
    ledger record of its own to assert.
    """
    plan = _many_probe_plan(5)
    error = stopper_class(
        "the branch machinery every probe shares is broken"
    )
    executor = _StopperBarrierExecutor(3, "fin_02", error)
    orchestrator, ledger, _ = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=3
    )

    with pytest.raises(stopper_class) as raised:
        await orchestrator.run()

    # The first stopper, unwrapped and unreplaced, is what leaves `run()`.
    assert raised.value is error
    assert sorted(executor.entered) == ["fin_01", "fin_02", "fin_03"]
    # The two that were in flight ran to their end and were recorded.
    assert ledger.steps["fin_01"].status == "done"
    assert ledger.steps["fin_03"].status == "done"
    # The stopper is not this probe failing, so it is left for the resume.
    assert ledger.steps["fin_02"].status == "running"
    # The queued two were never marked at all.
    assert not {"fin_04", "fin_05"} & set(ledger.steps)
    assert not orchestrator.in_flight


@pytest.mark.asyncio
async def test_a_dead_judge_records_the_probe_and_then_stops_the_batch(
    tmp_path: Path,
):
    """The probe that met the dead judge cleans its branch, is recorded `failed`
    with `judge_unavailable` and the evidence it managed to write, and joins
    `result.failed`; the batch then stops — nothing queued is started, and the
    exception leaves the batch, which is the process's non-zero rc.
    """
    plan = _many_probe_plan(5)
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, ProbeRecordingExecutor(), concurrency=3
    )
    # Run the timeline alone first, so the pin every probe branches from
    # exists and the batch below is the only thing under test.
    orchestrator.config.only_steps = frozenset()
    await orchestrator.run()
    orchestrator.config.only_steps = None
    orchestrator.executor = _DeadJudgeProbeExecutor("fin_02", width=3)
    branch_runs = _count_probe_branch_runs(orchestrator)

    batch = orchestrator_module.OrchestratorResult(run_id=plan.run_id)
    with pytest.raises(JudgeUnavailable) as raised:
        await orchestrator._run_probe_batch(
            [
                (index, step)
                for index, step in enumerate(plan.steps, start=1)
                if step.kind in orchestrator_module.PROBE_KINDS
            ],
            result=batch,
            total=len(plan.steps),
            checkpoint_manager=orchestrator.config.checkpoint_manager,
            pinned=plan.pinned_step_ids(),
        )

    # The two in flight beside it ran to their end and were recorded.
    assert set(batch.executed) == {"fin_01", "fin_03"}
    assert batch.failed == ["fin_02"]
    assert ledger.steps["fin_02"].status == "failed"
    assert ledger.steps["fin_02"].failure_kind == "judge_unavailable"
    # `mempabench run` names this probe on its stop line.
    from harness.failure_kind import FAILED_STEP_ID_ATTR

    assert getattr(raised.value, FAILED_STEP_ID_ATTR) == "fin_02"
    # Its evidence was copied out of the branch before the branch went.
    assert ledger.steps["fin_02"].artifacts["transcript_jsonl"] == (
        "steps/fin_02/transcript.jsonl"
    )
    assert (run_dir / "steps" / "fin_02" / "meta.yaml").is_file()
    assert list((run_dir / "branches" / "fin_02").iterdir()) == []
    # Nothing queued behind it was started.
    assert not {"fin_04", "fin_05"} & set(ledger.steps)
    assert not orchestrator.in_flight
    # A dead judge is not a void: the probe is not rerun, so it has one
    # branch and nothing of it was set aside.
    assert branch_runs.count("fin_02") == 1
    assert not (run_dir / "steps_superseded").exists()


@pytest.mark.asyncio
async def test_a_stop_mid_batch_finishes_the_in_flight_and_leaves_the_queue(
    tmp_path: Path,
):
    """The signal lands with two probes inside the executor: both finish, the
    two still queued stay unrecorded, and the resume runs exactly those."""
    plan = _many_probe_plan(4)
    flag = {"requested": False}
    orchestrator, ledger, _ = _batch_orchestrator(
        tmp_path,
        plan,
        _StopWhenFullExecutor(2, flag),
        concurrency=2,
        stop_requested=lambda: flag["requested"],
    )

    result = await orchestrator.run()

    assert flag["requested"] is True
    assert set(result.executed) == {"acc_001", "acc_002", "fin_01", "fin_02"}
    assert ledger.steps["fin_01"].status == "done"
    assert ledger.steps["fin_02"].status == "done"
    assert not {"fin_03", "fin_04"} & set(ledger.steps)
    # Nothing is left `running`, so the resume rolls nothing back.
    assert ledger.current_step is None
    ledger.assert_resumable(plan)

    resumed_executor = ProbeRecordingExecutor()
    resumed, ledger2, _ = _batch_orchestrator(
        tmp_path, plan, resumed_executor, concurrency=2
    )
    outcome = await resumed.run()

    assert set(outcome.executed) == {"fin_03", "fin_04"}
    assert {view["step_id"] for view in resumed_executor.probe_views} == {
        "fin_03",
        "fin_04",
    }
    assert all(
        ledger2.steps[step_id].status == "done"
        for step_id in _probe_ids(plan)
    )


@pytest.mark.parametrize("gate", ["abort", "stop_requested"])
@pytest.mark.asyncio
async def test_a_probe_dequeued_as_the_run_stops_is_left_unrecorded(
    tmp_path: Path, gate: str
):
    """*Both* stop conditions are read again before `mark_running`.

    The stop flag is set by a plain `signal.signal` handler, so it can flip at
    any bytecode boundary in the main thread — including between the worker's
    read of it and the ledger write. Reading it again here, with nothing
    awaited in between, is what keeps that probe unrecorded, and unrecorded is
    what the next resume runs.
    """
    plan = _two_probe_plan()
    executor = ProbeRecordingExecutor()
    stopping = {"requested": gate == "stop_requested"}
    orchestrator, ledger, _ = _batch_orchestrator(
        tmp_path,
        plan,
        executor,
        concurrency=1,
        stop_requested=lambda: stopping["requested"],
    )
    abort = asyncio.Event()
    if gate == "abort":
        abort.set()
    step = next(
        step
        for step in plan.steps
        if step.kind in orchestrator_module.PROBE_KINDS
    )
    result = orchestrator_module.OrchestratorResult(run_id=plan.run_id)

    await orchestrator._run_one_probe(
        1,
        step,
        result=result,
        total=len(plan.steps),
        checkpoint_manager=orchestrator.config.checkpoint_manager,
        pinned=plan.pinned_step_ids(),
        abort=abort,
        workers=1,
    )

    assert step.step_id not in ledger.steps
    assert (result.executed, result.failed, result.skipped) == ([], [], [])
    assert executor.calls == []


@pytest.mark.asyncio
async def test_a_runtime_that_will_not_go_read_only_stops_the_batch(
    tmp_path: Path,
):
    """`apply_mode` is hoisted out of the probe body, so it needs its own
    stopper: a mainline runtime that will not go read_only is the probe line
    being unrunnable, and leaving the raw backend error to fall out of `run()`
    would end the run with nothing in the ledger to read it off.
    """
    plan = _many_probe_plan(2)
    executor = ProbeRecordingExecutor()
    orchestrator, ledger, _ = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=2
    )
    original = orchestrator.memory_runtime.apply_mode

    async def refuse(memory_mode: str, *, reason: str = "") -> None:
        if reason == "probe_batch":
            raise RuntimeError("the memory backend is unreachable")
        await original(memory_mode, reason=reason)

    orchestrator.memory_runtime.apply_mode = refuse

    with pytest.raises(
        orchestrator_module.ProbeRunConfigurationError,
        match="could not be put in read_only",
    ):
        await orchestrator.run()

    # The timeline is untouched, and no probe was started or recorded.
    assert ledger.steps["acc_002"].status == "done"
    assert not {"fin_01", "fin_02"} & set(ledger.steps)
    assert executor.probe_views == []


@pytest.mark.asyncio
async def test_a_thirty_three_probe_batch_schedules_and_then_resumes(
    tmp_path: Path,
):
    """The formal round's shape, with a fake executor and no API calls.

    33 probes over 4 workers: four run together and never five, all 33 land
    `done`; a hard kill then leaves three of them `running`, and the rerun
    runs exactly those three off the same pin and skips the rest.
    """
    plan = _many_probe_plan(33)
    probe_ids = _probe_ids(plan)
    executor = _WidthGateExecutor(4)
    orchestrator, ledger, _ = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=4
    )

    result = await orchestrator.run()

    assert executor.max_live == 4
    assert set(result.executed) == {"acc_001", "acc_002", *probe_ids}
    assert result.failed == []
    assert all(ledger.steps[step_id].status == "done" for step_id in probe_ids)

    killed = ["fin_07", "fin_19", "fin_33"]
    for step_id in killed:
        ledger.mark_running(step_id)

    second_executor = ProbeRecordingExecutor()
    second, ledger2, run_dir = _batch_orchestrator(
        tmp_path, plan, second_executor, concurrency=4
    )
    resumed = await second.run()

    assert set(resumed.executed) == set(killed)
    # Skipped in plan order at N=4: the filters run as the queue is filled, so
    # the list does not depend on which worker picked what up.
    assert resumed.skipped == [
        step.step_id for step in plan.steps if step.step_id not in killed
    ]
    assert {view["step_id"] for view in second_executor.probe_views} == set(
        killed
    )
    assert all(ledger2.steps[step_id].status == "done" for step_id in probe_ids)
    # A rerun branches from the immutable pin; nothing was rolled back.
    assert not (run_dir / "crash_scene").exists()


# ── a voided probe is rerun at once ───────────────────────────────────

#: The directory name `harness.run_layout.supersede_step_dirs` writes.
_SUPERSEDED_DIR = re.compile(r"^(?P<step_id>.+)__\d{8}T\d{6}Z$")


class _VoidingProbeExecutor(FailingProbeExecutor):
    """Voids the named probe on its first `voids` attempts, then lets it pass.

    A void is how a probe session ends when the local judge fails on it:
    `SimulatorLoop.run_session` raises `SessionInvalidError` whose `rule` is
    the judge's own event name, and `classify_failure` maps exactly those three
    rules to a judge void kind. Every attempt writes its own number into
    `transcript.jsonl`, so which attempt's evidence a directory holds is
    readable off the disk rather than inferred.
    """

    def __init__(
        self,
        void_step: str,
        *,
        voids: int = 1,
        rule: str = "judge_timeout",
    ) -> None:
        super().__init__()
        self.void_step = void_step
        self.voids = voids
        self.rule = rule
        self.attempts: list[int] = []

    async def execute(self, step, *, steps_dir, **kwargs):
        if (
            step.step_id != self.void_step
            or step.kind not in orchestrator_module.PROBE_KINDS
        ):
            return await FakeExecutor.execute(
                self, step, steps_dir=steps_dir, **kwargs
            )
        attempt = len(self.attempts) + 1
        self.attempts.append(attempt)
        step_dir = Path(steps_dir) / step.step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "transcript.jsonl").write_text(
            json.dumps({"attempt": attempt}) + "\n", encoding="utf-8"
        )
        if attempt > self.voids:
            return await FakeExecutor.execute(
                self, step, steps_dir=steps_dir, **kwargs
            )
        raise self.failure()

    def failure(self) -> BaseException:
        """What this attempt dies of. Overridden to test the other kinds."""
        return SessionInvalidError(self.rule, beat="b3_probe", attempts=3)


class _IXRuntimeProbeExecutor(_VoidingProbeExecutor):
    """Fails the named probe the way three live holes failed.

    `AgentProviderError` whose message carries one of the IX runtime prefixes,
    which `classify_failure` reads as `ix_runtime` — a kind in
    `PROBE_RETRY_KINDS`.
    """

    def failure(self) -> BaseException:
        return AgentProviderError(
            "PA turn failed the IX render contract: a required "
            "IX_render_<attribute> payload was missing twice, so this turn "
            "had no per-turn render data"
        )


class _VoidAfterAStopExecutor(_VoidingProbeExecutor):
    """Takes the stop the caller hands it, then voids: a stop already up."""

    def __init__(self, void_step: str, *, take_the_stop) -> None:
        super().__init__(void_step)
        self.take_the_stop = take_the_stop

    async def execute(self, step, **kwargs):
        if step.step_id == self.void_step and not self.attempts:
            self.take_the_stop()
        return await super().execute(step, **kwargs)


class _VoidRetryHeldExecutor(_VoidingProbeExecutor):
    """A two-sided barrier between the retry and the other probes.

    Holding the retry until the others are done is only half of
    it — the others could finish while the voiding worker copied evidence,
    superseded the void's cohort and cut the replacement branch, and then the
    retry would wait on an already-set event. So neither side may pass alone:

    * no other probe leaves `execute` until `_retry_entered` is set, which
      attempt 2 of the voided probe and nothing else sets;
    * the retry does not leave `execute` until `_others_done` is set, which
      only the last of `others` other probes sets.

    A run therefore only gets through here if attempt 2 was inside the
    executor *while* the other workers were still draining the queue. A runner
    that drained them first, or that serialised the retry behind them, has one
    side waiting for an event the other side can no longer set, and both
    bounded waits raise `TimeoutError` — the suite fails instead of hanging.
    """

    def __init__(self, void_step: str, *, others: int) -> None:
        super().__init__(void_step)
        self.others = others
        self.finished_others = 0
        self._retry_entered = asyncio.Event()
        self._others_done = asyncio.Event()

    async def execute(self, step, **kwargs):
        if step.kind not in orchestrator_module.PROBE_KINDS:
            return await FakeExecutor.execute(self, step, **kwargs)
        if step.step_id != self.void_step:
            await asyncio.wait_for(self._retry_entered.wait(), 5)
            outcome = await FakeExecutor.execute(self, step, **kwargs)
            self.finished_others += 1
            if self.finished_others >= self.others:
                self._others_done.set()
            return outcome
        if self.attempts:  # the retry
            self._retry_entered.set()
            await asyncio.wait_for(self._others_done.wait(), 5)
        return await super().execute(step, **kwargs)


def _count_probe_branch_runs(orchestrator) -> list[str]:
    """Record every `_run_probe_in_branch` call, in order, and pass it on."""
    calls: list[str] = []
    inner = orchestrator._run_probe_in_branch

    async def counted(step, **kwargs):
        calls.append(step.step_id)
        return await inner(step, **kwargs)

    orchestrator._run_probe_in_branch = counted
    return calls


@pytest.mark.asyncio
async def test_a_voided_probe_is_rerun_at_once_and_its_void_set_aside(
    tmp_path: Path,
):
    """The runner does the rerun itself, inside the batch.

    The void's evidence becomes a superseded cohort of its own — the layout
    `--rerun-probe-steps` writes, so `--list-cohorts` and the manifest script
    read it like any other — and the retry's evidence is the live one.
    """
    plan = _many_probe_plan(2)
    total = len(plan.steps)
    lines: list[str] = []
    executor = _VoidingProbeExecutor("fin_01")
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=1, lines=lines
    )
    branch_runs = _count_probe_branch_runs(orchestrator)

    result = await orchestrator.run()

    assert branch_runs == ["fin_01", "fin_01", "fin_02"]
    assert result.failed == []
    assert set(result.executed) == {"acc_001", "acc_002", "fin_01", "fin_02"}
    record = ledger.steps["fin_01"]
    assert record.status == "done"
    assert record.failure_kind is None
    assert record.summary["void_retries"] == 1
    assert record.summary_schema == SUMMARY_SCHEMA
    # Only the probe that was rerun says so.
    assert "void_retries" not in ledger.steps["fin_02"].summary

    live = run_dir / "steps" / "fin_01" / "transcript.jsonl"
    assert json.loads(live.read_text(encoding="utf-8"))["attempt"] == 2
    superseded = sorted((run_dir / "steps_superseded").iterdir())
    assert [
        _SUPERSEDED_DIR.match(path.name).group("step_id") for path in superseded
    ] == ["fin_01"]
    assert (
        json.loads(
            (superseded[0] / "transcript.jsonl").read_text(encoding="utf-8")
        )["attempt"]
        == 1
    )

    # One void line, of a kind neither runbook readback can see, and the
    # probe's own done line printed exactly once.
    assert [line for line in lines if line.startswith("probe void ")] == [
        "probe void id=fin_01 kind=judge_timeout attempt=1 retrying"
    ]
    assert not [
        line
        for line in lines
        if line.startswith("probe void ")
        and (_DONE_LINE.match(line) or re.match(r"^\[\d{3}/\d{3}\] ", line))
    ]
    assert (
        sum(
            1
            for line in lines
            if _DONE_LINE.match(line) and " fin_01 done " in line
        )
        == 1
    )
    assert sum(1 for line in lines if _DONE_LINE.match(line)) == total


@pytest.mark.asyncio
async def test_a_probe_that_fails_every_attempt_is_failed_and_the_batch_goes_on(
    tmp_path: Path,
):
    """The last of `PROBE_ATTEMPTS` is recorded, never retried again, and the
    probe queued behind it still runs. Surfacing a probe that failed them all
    is the report's job; the batch does not stop on it."""
    plan = _many_probe_plan(2)
    total = len(plan.steps)
    lines: list[str] = []
    executor = _VoidingProbeExecutor(
        "fin_01", voids=orchestrator_module.PROBE_ATTEMPTS
    )
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=1, lines=lines
    )
    branch_runs = _count_probe_branch_runs(orchestrator)

    result = await orchestrator.run()

    attempts = orchestrator_module.PROBE_ATTEMPTS
    assert branch_runs == ["fin_01"] * attempts + ["fin_02"]
    assert result.failed == ["fin_01"]
    assert ledger.steps["fin_01"].status == "failed"
    assert ledger.steps["fin_01"].failure_kind == "judge_timeout"
    # The next queued probe ran: a probe that failed them all stops nothing.
    assert ledger.steps["fin_02"].status == "done"
    assert "fin_02" in result.executed
    # Every attempt but the last is set aside, in order; the last one's
    # evidence is the live directory.
    superseded = sorted((run_dir / "steps_superseded").iterdir())
    assert len(superseded) == attempts - 1
    assert [
        json.loads((d / "transcript.jsonl").read_text(encoding="utf-8"))[
            "attempt"
        ]
        for d in superseded
    ] == list(range(1, attempts))
    assert (
        json.loads(
            (run_dir / "steps" / "fin_01" / "transcript.jsonl").read_text(
                encoding="utf-8"
            )
        )["attempt"]
        == attempts
    )
    assert [line for line in lines if line.startswith("probe void ")] == [
        f"probe void id=fin_01 kind=judge_timeout attempt={n} retrying"
        for n in range(1, attempts)
    ] + [
        "probe void id=fin_01 kind=judge_timeout "
        f"attempt={attempts} failed all attempts"
    ]
    # The retry still prints the probe's ordinary failed line, unchanged.
    assert [line for line in lines if " fin_01 failed " in line] == [
        f"[003/{total:03d}] fin_01 failed judge_timeout: "
        "SESSION_INVALID: rule=judge_timeout beat=b3_probe attempts=3"
    ]


@pytest.mark.asyncio
async def test_an_other_probe_failure_is_retried_three_times(
    tmp_path: Path,
):
    """An unclassified (`other`) probe failure gets the same three attempts as
    a judge void or `ix_runtime`; the two earlier ones are set aside as
    superseded cohorts and the one record names the third."""
    plan = _many_probe_plan(2)
    executor = FailOneProbeExecutor("fin_01")
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=1
    )
    branch_runs = _count_probe_branch_runs(orchestrator)

    result = await orchestrator.run()

    assert branch_runs == ["fin_01", "fin_01", "fin_01", "fin_02"]
    assert result.failed == ["fin_01"]
    assert ledger.steps["fin_01"].status == "failed"
    assert ledger.steps["fin_01"].failure_kind == "other"
    assert len(list((run_dir / "steps_superseded").glob("fin_01__*"))) == 2


@pytest.mark.asyncio
async def test_an_ix_runtime_probe_is_rerun_and_can_pass_on_a_later_attempt(
    tmp_path: Path,
):
    """The PA gets three attempts at a probe.

    Three live holes (`fin_06`, `fin_01`, `fin_22` across the
    nanobot, hy4 and gpt lines) were all `ix_runtime` — the PA breaking an IX
    contract mid-turn — and under the old set they were never retried, so one
    bad turn wrote the cell off. Here the same failure happens twice and the
    third attempt lands `done`.
    """
    plan = _many_probe_plan(2)
    lines: list[str] = []
    executor = _IXRuntimeProbeExecutor("fin_01", voids=2)
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=1, lines=lines
    )
    branch_runs = _count_probe_branch_runs(orchestrator)

    result = await orchestrator.run()

    assert branch_runs == ["fin_01", "fin_01", "fin_01", "fin_02"]
    assert result.failed == []
    record = ledger.steps["fin_01"]
    assert record.status == "done"
    assert record.failure_kind is None
    # The summary key keeps its name: it counts reruns, and every reader takes
    # the summary with `.get`.
    assert record.summary["void_retries"] == 2
    # The two failed attempts are set aside in order; the winner is live.
    superseded = sorted((run_dir / "steps_superseded").iterdir())
    assert len(superseded) == 2
    assert [
        json.loads((d / "transcript.jsonl").read_text(encoding="utf-8"))[
            "attempt"
        ]
        for d in superseded
    ] == [1, 2]
    assert json.loads(
        (run_dir / "steps" / "fin_01" / "transcript.jsonl").read_text(
            encoding="utf-8"
        )
    )["attempt"] == 3
    assert [line for line in lines if line.startswith("probe void ")] == [
        "probe void id=fin_01 kind=ix_runtime attempt=1 retrying",
        "probe void id=fin_01 kind=ix_runtime attempt=2 retrying",
    ]


@pytest.mark.parametrize("gate", ["abort", "stop_requested"])
@pytest.mark.asyncio
async def test_a_void_with_a_stop_already_up_is_failed_without_a_retry(
    tmp_path: Path, gate: str
):
    """A retry is a new start, and nothing new starts after a stopper or a
    signal. The void is then recorded `failed`, with no retry."""
    plan = _two_probe_plan()
    lines: list[str] = []
    stopping = {"requested": False}
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path,
        plan,
        ProbeRecordingExecutor(),
        concurrency=1,
        lines=lines,
        stop_requested=lambda: stopping["requested"],
    )
    # The timeline alone first, so the pin the probe branches from exists.
    orchestrator.config.only_steps = frozenset()
    await orchestrator.run()
    orchestrator.config.only_steps = None
    abort = asyncio.Event()

    def take_the_stop() -> None:
        if gate == "abort":
            abort.set()
        else:
            stopping["requested"] = True

    step = next(
        step
        for step in plan.steps
        if step.kind in orchestrator_module.PROBE_KINDS
    )
    orchestrator.executor = _VoidAfterAStopExecutor(
        step.step_id, take_the_stop=take_the_stop
    )
    branch_runs = _count_probe_branch_runs(orchestrator)
    result = orchestrator_module.OrchestratorResult(run_id=plan.run_id)

    await orchestrator._run_one_probe(
        1,
        step,
        result=result,
        total=len(plan.steps),
        checkpoint_manager=orchestrator.config.checkpoint_manager,
        pinned=plan.pinned_step_ids(),
        abort=abort,
        workers=1,
    )

    assert branch_runs == [step.step_id]
    assert ledger.steps[step.step_id].status == "failed"
    assert ledger.steps[step.step_id].failure_kind == "judge_timeout"
    assert result.failed == [step.step_id]
    assert not (run_dir / "steps_superseded").exists()
    assert not [line for line in lines if line.startswith("probe void ")]


@pytest.mark.asyncio
async def test_a_void_retry_does_not_block_the_other_workers(tmp_path: Path):
    """N=3: the retry is the one worker's own work.

    Its branch is cut, run and recorded inside the slot that probe already
    holds, so the other two workers drain the queue while it happens. The
    executor's barrier is two-sided, so passing it is proof of
    real overlap rather than of a lucky order: neither of the other two probes
    may finish before attempt 2 of the voided probe has *entered* the
    executor, and attempt 2 may not finish before both of them have. Overlap
    is therefore the only way through — a runner that drained the other
    workers first, or that serialised the retry behind them, leaves one side
    of the barrier waiting on an event the other can no longer set, and the
    bounded waits turn that into a failure instead of a hang.
    """
    plan = _many_probe_plan(3)
    probe_ids = _probe_ids(plan)
    executor = _VoidRetryHeldExecutor("fin_01", others=2)
    orchestrator, ledger, run_dir = _batch_orchestrator(
        tmp_path, plan, executor, concurrency=3
    )

    result = await orchestrator.run()

    assert executor.finished_others == 2
    assert set(result.executed) == {"acc_001", "acc_002", *probe_ids}
    assert result.failed == []
    assert all(ledger.steps[step_id].status == "done" for step_id in probe_ids)
    assert ledger.steps["fin_01"].summary["void_retries"] == 1
    assert len(list((run_dir / "steps_superseded").iterdir())) == 1
    assert not orchestrator.in_flight


@pytest.mark.asyncio
async def test_a_cut_point_is_published_with_the_summary_the_ledger_records(
    tmp_path: Path,
) -> None:
    """The snapshot's own ledger has to agree with the mainline's, summary too.

    The checkpoint is published before `mark_done`, so unless the summary is
    handed to `save`, the snapshot records the step as done with none. For a
    forced event that summary is the only copy of its `preference_overlay`
    flip, and a rollback onto its own cut point is what reads it back: a live
    run resumed with an empty overlay and died at `acc_017`.
    """
    plan = _two_acc_plan()
    calls: list = []
    manager = FakeCheckpointManager(calls)
    orchestrator = _orchestrator(
        tmp_path,
        plan,
        checkpoint_manager=manager,
        pa_checkpoint=FakePACheckpoint(calls),
    )

    await orchestrator.run()

    for step_id in ("acc_001", "acc_002"):
        assert manager.saved_summaries[step_id] is not None
        assert (
            manager.saved_summaries[step_id]
            == orchestrator.ledger.steps[step_id].summary
        )


# ── rollback recovery ─────────────────────────────────────────────


def test_a_pending_recovery_is_not_bypassed_by_a_clean_ledger(
    monkeypatch, tmp_path: Path
) -> None:
    """A recovery that died after replacing the ledger is finished, not skipped.

    `restore` replaces `ledger.json` from the snapshot, so the moment that
    lands the ledger on disk is *cleanly resumable* — and the old
    `resume_or_restore` rolled back only when `assert_resumable` raised. A
    recovery interrupted at exactly that point therefore looked like a healthy
    run, and the PA started on half-restored state.
    """
    plan = _two_acc_plan()
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)
    for step_id in ("acc_001", "acc_002"):
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=True)
        ledger.mark_done(step_id)

    # A real recovery to acc_001, killed right after the ledger replacement
    # and before the marker was cleared.
    boom = RuntimeError("killed after the ledger replacement")

    def _die(self, run_dir_arg) -> None:
        raise boom

    monkeypatch.setattr(
        checkpoint_module.CheckpointManager, "_clear_pending_recovery", _die
    )
    with pytest.raises(RuntimeError, match="killed after"):
        manager.restore("acc_001", ledger=ledger)
    monkeypatch.undo()

    # This is the trap: the ledger left on disk is acc_001's own, which is
    # perfectly resumable, and acc_002 is still on disk reading one step
    # further along.
    fresh_ledger = RunLedger(run_dir, plan.run_id)
    fresh_ledger.assert_resumable(plan)
    fresh_manager = _file_memory_manager(run_dir, fresh_ledger)
    assert fresh_manager.pending_recovery() is not None

    # A new process must finish that recovery, on the same target.
    assert (
        orchestrator_module.resume_or_restore(fresh_ledger, fresh_manager, plan)
        == "acc_001"
    )
    assert fresh_manager.pending_recovery() is None
    assert fresh_manager.list_checkpoints() == ["acc_001"]


def _four_cut_points(tmp_path: Path, plan) -> tuple[Path, RunLedger, CheckpointManager]:
    """A run whose PA memory says which cut point it is standing on.

    acc_003 and acc_004 are cut points this plan does not list: the ledger is
    what publishes a snapshot, and a rollback has to judge whatever is on disk
    rather than whatever the plan happens to mention.
    """
    run_dir = tmp_path / "runs" / plan.run_id
    workspace = _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    memory = workspace / "memory" / "MEMORY.md"
    ledger = RunLedger(run_dir, plan.run_id)
    manager = _file_memory_manager(run_dir, ledger)
    for step_id, pinned in [
        ("acc_001", True),
        ("acc_002", True),
        ("acc_003", False),
        ("acc_004", True),
    ]:
        memory.write_text(f"standing on {step_id}\n", encoding="utf-8")
        ledger.mark_running(step_id)
        manager.save(step_id, pinned=pinned)
        ledger.mark_done(step_id)
    # What the run was doing when it died, which is what a rollback replaces.
    memory.write_text("standing on the crash\n", encoding="utf-8")
    return run_dir, ledger, manager


def _fail_on_nth(monkeypatch, name: str, nth: int, message: str) -> None:
    """Make the nth call to `shutil.<name>` inside a restore blow up."""
    real = getattr(checkpoint_module.shutil, name)
    seen: list[int] = []

    def flaky(*args, **kwargs):
        seen.append(1)
        if len(seen) == nth:
            raise RuntimeError(message)
        return real(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.shutil, name, flaky)


@pytest.mark.parametrize(
    "injection",
    ["marker", "archiving", "local_copy", "backend", "ledger_replacement"],
)
def test_an_interrupted_recovery_continues_the_same_target(
    monkeypatch, tmp_path: Path, injection: str
) -> None:
    """Every way of dying mid-rollback leads back to the same target.

    Inject a failure after the marker lands, during archiving, during the local
    copy, after the backend layer and after the ledger replacement; rebuild the
    manager each time. None of them may start the PA, re-select a revoked
    snapshot, or overwrite the evidence of the original crash.
    """
    plan = _two_acc_plan()
    run_dir, ledger, manager = _four_cut_points(tmp_path, plan)

    if injection == "marker":
        # The marker is published immediately before archiving begins.
        monkeypatch.setattr(
            checkpoint_module.CheckpointManager,
            "_archive_revoked_checkpoints",
            lambda self, pending: (_ for _ in ()).throw(RuntimeError("died at marker")),
        )
    elif injection == "archiving":
        # Two snapshots are revoked, so the second archive move is mid-way.
        _fail_on_nth(monkeypatch, "move", 2, "died while archiving")
    elif injection == "local_copy":
        _fail_on_nth(monkeypatch, "copytree", 2, "died while copying layers")
    elif injection == "backend":
        monkeypatch.setattr(
            checkpoint_module.CheckpointManager,
            "_move_post_checkpoint_steps",
            lambda self, cp_dir, attempt: (_ for _ in ()).throw(
                RuntimeError("died after the backend layer")
            ),
        )
    else:
        monkeypatch.setattr(
            checkpoint_module.CheckpointManager,
            "_clear_pending_recovery",
            lambda self, run_dir_arg: (_ for _ in ()).throw(
                RuntimeError("died after the ledger replacement")
            ),
        )

    with pytest.raises(RuntimeError, match="died"):
        manager.restore("acc_002", ledger=ledger)
    monkeypatch.undo()

    # Whatever was reached, the run is on record as going back to acc_002.
    interrupted = RunLedger(run_dir, plan.run_id)
    resumed = _file_memory_manager(run_dir, interrupted)
    pending = resumed.pending_recovery()
    assert pending is not None
    assert pending["target_checkpoint_id"] == "acc_002"
    # No revoked snapshot is offered as a cut point while this is open, so
    # nothing can pick acc_004 even by asking for the furthest one.
    assert resumed.list_checkpoints() == ["acc_001", "acc_002"]
    # And a rollback to anywhere else is refused outright.
    with pytest.raises(checkpoint_module.RecoveryInProgressError):
        resumed.restore("acc_001", ledger=interrupted)

    # A new process finishes it, and says which target it finished.
    assert (
        orchestrator_module.resume_or_restore(interrupted, resumed, plan) == "acc_002"
    )

    assert resumed.pending_recovery() is None
    assert resumed.list_checkpoints() == ["acc_001", "acc_002"]
    assert sorted(interrupted.steps) == ["acc_001", "acc_002"]
    memory = run_dir / "pa_workspace" / plan.persona_id / "memory" / "MEMORY.md"
    assert memory.read_text(encoding="utf-8") == "standing on acc_002\n"
    archive = run_dir / "checkpoint_archive" / pending["recovery_id"]
    assert sorted(path.name for path in archive.iterdir()) == ["acc_003", "acc_004"]

    # The original crash is still readable, and no attempt wrote over another.
    # The injections that died before the layer stage never moved a live layer,
    # so the retry's own first attempt is what holds the crash. The ones that
    # died inside or after that stage left attempt_01 holding it, and the retry
    # had to take attempt_02 instead of copying over it.
    evidence = run_dir / pending["evidence_dir"]
    attempts = sorted(path.name for path in evidence.iterdir())
    reached_layers = injection in {"local_copy", "backend", "ledger_replacement"}
    assert attempts == (
        ["attempt_01", "attempt_02"] if reached_layers else ["attempt_01"]
    )
    crashed = evidence / "attempt_01" / "pa_workspace" / plan.persona_id
    assert (crashed / "memory" / "MEMORY.md").read_text(
        encoding="utf-8"
    ) == "standing on the crash\n"


@pytest.mark.asyncio
async def test_dynamic_retained_forced_checkpoint_does_not_become_pin_or_flush(tmp_path):
    from harness.checkpoint import CheckpointManager, checkpoint_restorable
    _fixture(tmp_path)
    plan = _unpinned_probe_plan()
    plan = plan.model_copy(update={'steps': plan.steps[:1]})
    run_dir = tmp_path/'runs'/plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)
    manager = CheckpointManager(run_dir, ledger=ledger, memory_condition='nanobot_file_memory',
                                rolling_window=1, retention_interval=1,
                                retention_mainline_ids=frozenset({'evolv_01'}))
    calls = []; pa = FakePACheckpoint(calls)
    orchestrator = _orchestrator(tmp_path, plan, checkpoint_manager=manager, pa_checkpoint=pa)
    orchestrator.ledger = ledger
    result = await orchestrator.run()
    assert result.executed == ['evolv_01'] and ledger.steps['evolv_01'].status == 'done'
    assert calls == [('drain',)] and pa.archived == [] and pa.saved == 0
    assert pa._session.file_consolidated_end_index == 0
    assert pa._session.prompt_history_start_index == 0
    assert all(not record.file_consolidated for record in pa.turn_journal().records())
    assert manager.metadata('evolv_01')['retained'] is True
    assert manager.metadata('evolv_01')['pinned'] is False
    assert checkpoint_restorable(run_dir/'checkpoints/evolv_01', memory_condition='nanobot_file_memory') == (True, [])
    assert manager.prune(0) == []
    with pytest.raises(ValueError, match='Only a pinned checkpoint'):
        manager.branch_from('evolv_01', tmp_path/'forbidden_branch')


@pytest.mark.asyncio
async def test_global_batch_wait_finishes_mainline_and_leaves_all_33_pending(tmp_path):
    from harness.probe_slots import probe_batch_slot
    plan = _many_probe_plan(33)
    executor = ProbeRecordingExecutor()
    lines = []
    orchestrator, ledger, run_dir = _batch_orchestrator(tmp_path, plan, executor, concurrency=4, lines=lines)
    slot_root = tmp_path / "global-slots"
    orchestrator.config.probe_slot_root = slot_root
    stopped = False
    orchestrator.config.stop_requested = lambda: stopped
    async with probe_batch_slot(slot_root, stop_requested=lambda: False, workers=4):
        async with probe_batch_slot(slot_root, stop_requested=lambda: False, workers=4):
            task = asyncio.create_task(orchestrator.run())
            for _ in range(100):
                if any("waiting for a global" in line for line in lines):
                    break
                await asyncio.sleep(.01)
            assert any("waiting for a global" in line for line in lines)
            assert ledger.steps["acc_001"].status == "done"
            assert ledger.steps["acc_002"].status == "done"
            assert not ledger.current_step
            assert not set(_probe_ids(plan)) & set(ledger.steps)
            assert executor.probe_views == []
            stopped = True
            await asyncio.wait_for(task, 2)
    second_executor = ProbeRecordingExecutor()
    second, ledger2, _ = _batch_orchestrator(tmp_path, plan, second_executor, concurrency=4)
    second.config.probe_slot_root = slot_root
    result = await second.run()
    assert set(result.executed) == set(_probe_ids(plan))
    assert all(ledger2.steps[sid].status == "done" for sid in _probe_ids(plan))
