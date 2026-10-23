from pathlib import Path

import pytest
import yaml

from harness.run_plan import (
    Mem0RuntimeConfig,
    MemOSRuntimeConfig,
    RunPlan,
    RunPlanLoadError,
    SimulatorRuntimeConfig,
    load_run_plan,
    validate_full_timeline_shape,
)
from memory.turn_experience import MEMORY_POOL_VERSION
from pa.agent.ix.render_types import IX_CONTRACT_VERSION
from pa.agent.ix.specs import IX_PASSIVE_SPECS
from pa.providers.request_policy import resolve_model_request_policy


def _mem0_runtime_config() -> Mem0RuntimeConfig:
    return Mem0RuntimeConfig(
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


def _plan(script_path: str = "script.yaml", *, placeholder: bool = False) -> dict:
    pa_policy = resolve_model_request_policy(
        "openrouter",
        "openai/gpt-5.5",
    ).model_copy(
        update={"policy_source": "model", "validated_roles": ["pa"]}
    )
    memory_side_policy = pa_policy.model_copy(
        update={"validated_roles": ["pa", "memory_extraction"]}
    )
    mem0_runtime_config = _mem0_runtime_config()
    return {
        "run_id": "run_001",
        "persona_id": "user_a",
        "memory_condition": "mem0",
        "assistant_llm_provider_name": "openrouter",
        "assistant_llm_api_base": "https://openrouter.ai/api/v1",
        "pa_model": "openai/gpt-5.5",
        "pa_request_policy": pa_policy.snapshot(),
        "memory_side_model": "openai/gpt-5.5",
        "memory_side_request_policy": memory_side_policy.snapshot(),
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "qdrant_host": "127.0.0.1",
        "qdrant_port": 6333,
        "mem0_runtime_config": mem0_runtime_config.model_dump(mode="json"),
        "generated_from": {"timeline_path": "Timeline.md"},
        "steps": [
            {
                "step_id": "acc_001",
                "kind": "accumulation",
                "persona_id": "user_a",
                "script_path": script_path,
                "context": "work",
                "target_cell": "personal_uncertainty_expression",
                "acc_num": 1,
                "benchmark_session_timestamp": "2026-10-01T09:00:00+00:00",
                "memory_mode": "read_write",
                "stage_policy": "commit",
                "placeholder": placeholder,
            }
        ],
    }


def test_simulator_runtime_defaults_are_frozen_in_round_trip(
    tmp_path: Path,
) -> None:
    plan = RunPlan(**_plan())
    path = tmp_path / "run_plan.yaml"
    plan.dump_yaml(path)

    loaded = load_run_plan(path, project_root=tmp_path, validate_paths=False)

    assert loaded.simulator_runtime == SimulatorRuntimeConfig()
    dumped = path.read_text(encoding="utf-8")
    assert "simulator_runtime:" in dumped
    assert "output_contract_retries: 3" in dumped
    assert "self_report_contract_retries: 5" in dumped
    assert "simulator_model: google/gemini-3.8-flash" in dumped


@pytest.mark.parametrize(
    "config",
    [
        {"output_contract_retries": -1},
        {"self_report_contract_retries": -1},
        {"simulator_model": ""},
        {"simulator_model": "  "},
        {"emotion_window_size": 15},
        {"self_report_model": "google/gemini-2.5-pro"},
        {"calibration_model": "google/gemini-2.5-pro"},
    ],
)
def test_simulator_runtime_rejects_invalid_values(config: dict) -> None:
    data = _plan()
    data["simulator_runtime"] = config
    with pytest.raises(ValueError):
        RunPlan(**data)


def _memos_plan() -> dict:
    data = _plan()
    memory_side_policy = data["memory_side_request_policy"]
    runtime_config = MemOSRuntimeConfig(
        api_base="http://127.0.0.1:8100",
        memos_version="2.0.15",
        memos_submodule_commit="1cf8a78fda66ebb529f2b599a66741fbf5c9a220",
        mempa_patch_version="20260803.6",
        memory_side_provider_name="openrouter",
        memory_side_model=data["memory_side_model"],
        memory_side_api_base="https://openrouter.ai/api/v1",
        memory_side_model_roles={
            "chat": data["memory_side_model"],
            "memory_reader": data["memory_side_model"],
            "general_reader": data["memory_side_model"],
        },
        memory_side_temperature_policy=memory_side_policy["temperature"],
        memory_side_completion_tokens_parameter=memory_side_policy[
            "completion_tokens_parameter"
        ],
        memory_side_reasoning_effort_policy=memory_side_policy[
            "reasoning_effort"
        ],
        memory_side_task_affinity_required=False,
        memory_side_task_affinity_namespace="",
        embedding_provider=data["embedding_provider"],
        embedding_model=data["embedding_model"],
        embedding_api_base="https://api.openai.com/v1",
        embedding_dimensions=data["embedding_dimensions"],
        reranker_backend="cosine_local",
        qdrant_host="qdrant",
        qdrant_port=6333,
        qdrant_image="qdrant/qdrant:v1.15.3",
        neo4j_backend="neo4j-community",
        neo4j_uri="bolt://neo4j:7687",
        neo4j_database="neo4j",
        neo4j_image="neo4j:5.26.6",
    )
    data.update(
        {
            "memory_condition": "memos",
            "qdrant_host": "",
            "qdrant_port": 0,
            "mem0_runtime_config": None,
            "memos_runtime_config": runtime_config.model_dump(mode="json"),
            "memos_cube_id": "mempa_run_001_memos_test",
        }
    )
    return data


@pytest.mark.parametrize(
    "timeout",
    [0.0, float("nan"), float("inf"), -float("inf")],
)
def test_memos_runtime_rejects_non_finite_or_non_positive_timeout(
    timeout: float,
) -> None:
    data = _memos_plan()["memos_runtime_config"]
    data["api_timeout_seconds"] = timeout

    with pytest.raises(ValueError, match="finite and positive"):
        MemOSRuntimeConfig.model_validate(data)


def test_run_plan_freezes_shared_memory_retrieval_budget() -> None:
    plan = RunPlan(**_plan())

    assert plan.memory_retrieval_max_tokens == 4000
    assert plan.memory_retrieval_max_items == 10


def test_run_plan_freezes_the_pa_completion_token_limit(tmp_path: Path) -> None:
    """acc_006 hit the provider dataclass's 4096 six times in one
    session, because nothing on the PA path ever set a limit. The plan sets it,
    and the frozen file records it so a run's meta shows the value."""
    plan = RunPlan(**_plan())
    assert plan.pa_max_tokens == 8192

    path = tmp_path / "run_plan.yaml"
    plan.dump_yaml(path)
    assert "pa_max_tokens: 8192" in path.read_text(encoding="utf-8")

    loaded = load_run_plan(path, project_root=tmp_path, validate_paths=False)
    assert loaded.pa_max_tokens == 8192


def test_run_plan_carries_a_configured_pa_completion_token_limit() -> None:
    data = _plan()
    data["pa_max_tokens"] = 16384

    assert RunPlan(**data).pa_max_tokens == 16384


@pytest.mark.parametrize("value", [0, -1])
def test_run_plan_rejects_a_non_positive_pa_completion_token_limit(
    value: int,
) -> None:
    data = _plan()
    data["pa_max_tokens"] = value

    with pytest.raises(ValueError, match="pa_max_tokens must be positive"):
        RunPlan(**data)


def test_mem0_plan_freezes_complete_runtime_configuration() -> None:
    plan = RunPlan(**_plan())

    assert plan.mem0_runtime_config is not None
    assert plan.mem0_runtime_config.mem0_version == "2.0.2"
    assert (
        plan.mem0_runtime_config.embedding_api_base
        == "https://api.openai.com/v1"
    )
    assert plan.mem0_runtime_config.retrieval_similarity_threshold == 0.1
    assert plan.mem0_runtime_config.retrieval_rerank is False
    assert plan.mem0_runtime_config.qdrant_distance == "Cosine"
    assert plan.mem0_runtime_config.qdrant_on_disk is True
    assert plan.mem0_runtime_config.qdrant_bm25_enabled is False
    assert plan.mem0_runtime_config.enable_entity_memory is True
    assert (
        plan.mem0_runtime_config.history_db_relative_path
        == "backend_state/mem0_history.sqlite3"
    )


def test_mem0_plan_requires_runtime_configuration() -> None:
    data = _plan()
    data["mem0_runtime_config"] = None

    with pytest.raises(ValueError, match="requires mem0_runtime_config"):
        RunPlan(**data)


def test_non_mem0_plan_rejects_mem0_runtime_configuration() -> None:
    data = _memos_plan()
    data["mem0_runtime_config"] = _mem0_runtime_config().model_dump(mode="json")

    with pytest.raises(
        ValueError,
        match="mem0_runtime_config is only valid",
    ):
        RunPlan(**data)


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), float("-inf"), -0.01, 1.01],
)
def test_mem0_runtime_rejects_invalid_similarity_threshold(
    threshold: float,
) -> None:
    data = _mem0_runtime_config().model_dump(mode="json")
    data["retrieval_similarity_threshold"] = threshold

    with pytest.raises(ValueError, match="finite and between 0 and 1"):
        Mem0RuntimeConfig.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "embedding_max_input_tokens",
        "embedding_chunk_safety_margin_tokens",
        "embedding_tokenizer_encoding",
        "tiktoken_version",
        "extraction_max_output_tokens",
    ],
)
def test_mem0_plan_without_the_chunk_policy_is_rejected(field: str) -> None:
    """A frozen plan that cannot say where it cut a pool is not loadable.

    Nor one that cannot say how much its extractor was allowed to say.
    """
    data = _plan()
    mem0_runtime_config = dict(data["mem0_runtime_config"])
    del mem0_runtime_config[field]
    data["mem0_runtime_config"] = mem0_runtime_config

    with pytest.raises(ValueError, match=field):
        RunPlan(**data)


def test_mem0_runtime_rejects_a_margin_that_exceeds_the_cap() -> None:
    data = _mem0_runtime_config().model_dump(mode="json")
    data["embedding_chunk_safety_margin_tokens"] = data[
        "embedding_max_input_tokens"
    ]

    with pytest.raises(ValueError, match="embedding chunk budget is invalid"):
        Mem0RuntimeConfig.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("memory_retrieval_max_tokens", 0),
        ("memory_retrieval_max_items", -1),
    ],
)
def test_run_plan_rejects_invalid_memory_retrieval_budget(
    field: str,
    value: int,
) -> None:
    data = _plan()
    data[field] = value

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        RunPlan(**data)


@pytest.mark.parametrize(
    "relativity",
    [float("nan"), float("inf"), float("-inf"), -0.01, 1.01],
)
def test_memos_runtime_rejects_invalid_relativity(relativity: float) -> None:
    data = _memos_plan()
    data["memos_runtime_config"]["relativity"] = relativity

    with pytest.raises(ValueError, match="finite and between 0 and 1"):
        RunPlan(**data)


def test_valid_small_plan_loads(tmp_path: Path):
    script = tmp_path / "script.yaml"
    script.write_text("ok: true\n", encoding="utf-8")
    plan_path = tmp_path / "run_plan.yaml"
    RunPlan(**_plan()).dump_yaml(plan_path)

    plan = load_run_plan(plan_path, project_root=tmp_path)

    assert plan.run_id == "run_001"
    assert plan.memory_condition == "mem0"
    assert plan.steps[0].step_id == "acc_001"


def test_embedding_run_plan_requires_qdrant_endpoint():
    data = _plan()
    data["qdrant_host"] = ""
    data["qdrant_port"] = 0

    with pytest.raises(
        ValueError,
        match="qdrant_host='' qdrant_port=0",
    ):
        RunPlan(**data)


def test_embedding_run_plan_rejects_invalid_qdrant_port():
    data = _plan()
    data["qdrant_port"] = 70000

    with pytest.raises(ValueError, match="qdrant_port=70000"):
        RunPlan(**data)


def test_non_embedding_run_plan_rejects_qdrant_endpoint():
    data = _plan()
    data.update(
        {
            "memory_condition": "no_memory",
            "memory_side_model": "",
            "memory_side_request_policy": None,
            "embedding_provider": "",
            "embedding_model": "",
            "embedding_dimensions": 0,
            "mem0_runtime_config": None,
        }
    )

    with pytest.raises(
        ValueError,
        match="only valid for a direct-Qdrant.*qdrant_host='127.0.0.1'",
    ):
        RunPlan(**data)


def test_mem0_rejects_missing_memory_side_configuration():
    data = _plan()
    data.update(
        {
            "memory_side_model": "",
            "memory_side_request_policy": None,
        }
    )

    with pytest.raises(
        ValueError,
        match="mem0 requires frozen memory-side LM configuration",
    ):
        RunPlan(**data)


def test_memos_plan_freezes_validated_runtime_configuration():
    plan = RunPlan(**_memos_plan())

    assert plan.memos_runtime_config is not None
    assert plan.memos_runtime_config.api_base == "http://127.0.0.1:8100"
    assert plan.memos_runtime_config.memory_profile == "text_memory"
    assert plan.memos_runtime_config.activation_memory_enabled is False
    assert plan.memos_runtime_config.preference_memory_enabled is False
    assert set(plan.memos_runtime_config.memory_side_model_roles) == {
        "chat",
        "memory_reader",
        "general_reader",
    }
    assert plan.qdrant_host == ""
    assert plan.qdrant_port == 0
    assert plan.memos_runtime_config.embedding_model == "text-embedding-3-small"


def test_memos_plan_requires_runtime_configuration():
    data = _memos_plan()
    data["memos_runtime_config"] = None

    with pytest.raises(ValueError, match="memos requires memos_runtime_config"):
        RunPlan(**data)


def test_memos_plan_requires_frozen_cube_id():
    data = _memos_plan()
    data["memos_cube_id"] = ""

    with pytest.raises(ValueError, match="requires a frozen memos_cube_id"):
        RunPlan(**data)


def test_non_memos_plan_rejects_memos_cube_id():
    data = _plan()
    data["memos_cube_id"] = "unexpected_cube"

    with pytest.raises(ValueError, match="memos_cube_id is only valid for memos"):
        RunPlan(**data)


def test_non_memos_plan_rejects_memos_runtime_configuration():
    data = _plan()
    data["memos_runtime_config"] = _memos_plan()["memos_runtime_config"]

    with pytest.raises(
        ValueError,
        match="memos_runtime_config is only valid for memos",
    ):
        RunPlan(**data)


def test_memos_runtime_configuration_must_match_shared_models():
    data = _memos_plan()
    data["memos_runtime_config"]["embedding_dimensions"] = 3072

    with pytest.raises(
        ValueError,
        match="does not match frozen shared memory fields",
    ):
        RunPlan(**data)


def test_memos_runtime_request_fields_must_match_frozen_memory_policy():
    data = _memos_plan()
    data["memos_runtime_config"]["memory_side_temperature_policy"] = "omit"

    with pytest.raises(
        ValueError,
        match="does not match frozen shared memory fields",
    ):
        RunPlan(**data)


def test_memos_runtime_affinity_must_match_provider_contract():
    data = _memos_plan()
    data["memos_runtime_config"]["memory_side_task_affinity_required"] = True
    data["memos_runtime_config"][
        "memory_side_task_affinity_namespace"
    ] = "memos_memory_extraction"

    with pytest.raises(
        ValueError,
        match="does not match frozen shared memory fields",
    ):
        RunPlan(**data)


def test_non_file_run_plan_rejects_consolidation_fields():
    data = _plan()
    consolidation_policy = resolve_model_request_policy(
        "openrouter",
        "fixed-memory-model",
    )
    data.update(
        {
            "consolidation_model": "fixed-memory-model",
            "consolidation_request_policy": consolidation_policy.snapshot(),
        }
    )

    with pytest.raises(
        ValueError,
        match="only valid for nanobot_file_memory",
    ):
        RunPlan(**data)


def test_run_plan_defaults_to_main_pa_session_key(tmp_path: Path):
    plan_path = tmp_path / "run_plan.yaml"
    RunPlan(**_plan()).dump_yaml(plan_path)

    plan = load_run_plan(plan_path, project_root=tmp_path, validate_paths=False)

    assert plan.main_pa_session_key() == "run_001_main"
    assert plan.pa_session_key_for_step(plan.steps[0]) == "run_001_main"
    assert "pa_session_key: run_001_main" in plan_path.read_text(encoding="utf-8")


def test_accumulation_steps_share_main_pa_session_key():
    data = _plan()
    data["steps"].append(
        {
            **data["steps"][0],
            "step_id": "acc_002",
            "script_path": "acc_002.yaml",
            "acc_num": 2,
            "benchmark_session_timestamp": "2026-10-01T15:00:00+00:00",
        }
    )

    plan = RunPlan(**data)

    assert plan.pa_session_key_for_step(plan.steps[0]) == "run_001_main"
    assert plan.pa_session_key_for_step(plan.steps[1]) == "run_001_main"


def test_steps_must_start_at_least_one_session_envelope_apart():
    data = _plan()
    data["steps"].append(
        {
            **data["steps"][0],
            "step_id": "acc_002",
            "script_path": "acc_002.yaml",
            "acc_num": 2,
            "benchmark_session_timestamp": "2026-10-01T09:30:00+00:00",
        }
    )
    with pytest.raises(ValueError, match=r"session envelope \(60 minutes\).*step_id='acc_002'"):
        RunPlan(**data)

    data["steps"][1]["benchmark_session_timestamp"] = "2026-10-01T10:00:00+00:00"
    assert len(RunPlan(**data).steps) == 2


def test_the_envelope_increase_only_binds_timeline_steps():
    """Finals share one cut point; a pre sits one envelope before its acc."""
    data = _plan()
    data["steps"] = _timeline_and_probe_steps()
    plan = RunPlan(**data)
    finals = [step for step in plan.steps if step.kind == "final_probe"]
    assert len({step.benchmark_session_timestamp for step in finals}) == 1

    # A second final at a different clock is not "the" final cut point.
    drifted = _plan()
    drifted["steps"] = _timeline_and_probe_steps()
    drifted["steps"][-1]["benchmark_session_timestamp"] = (
        "2026-10-05T10:00:00+00:00"
    )
    with pytest.raises(
        ValueError,
        match="share the one frozen final cut point: step_id='fin_02'",
    ):
        RunPlan(**drifted)

    # A final inside the last timeline step's envelope is refused too.
    early = _plan()
    early["steps"] = _timeline_and_probe_steps()
    for step in early["steps"]:
        if step["kind"] == "final_probe":
            step["benchmark_session_timestamp"] = "2026-10-04T09:30:00+00:00"
    with pytest.raises(ValueError, match="after the last timeline step: step_id='fin_01'"):
        RunPlan(**early)


def test_a_pre_event_probe_at_the_wrong_time_fails():
    data = _plan()
    data["steps"] = _timeline_and_probe_steps()
    data["steps"][2]["benchmark_session_timestamp"] = "2026-10-03T07:00:00+00:00"
    with pytest.raises(
        ValueError,
        match=(
            r"exactly one session envelope \(60 minutes\) before its before_acc "
            r"step: step_id='pre_01'.*before_acc_step_id='acc_003'.*"
            r"expected='2026-10-03T08:00:00\+00:00'"
        ),
    ):
        RunPlan(**data)

    # Right clock, wrong place: appended after its acc it would pin acc_003.
    misplaced = _plan()
    misplaced["steps"] = _timeline_and_probe_steps()
    pre = misplaced["steps"].pop(2)
    misplaced["steps"].insert(3, pre)
    with pytest.raises(
        ValueError,
        match="must sit before its before_acc step in the plan: step_id='pre_01'",
    ):
        RunPlan(**misplaced)


def test_explicit_pa_session_key_must_be_nonempty_and_match_run_id():
    data = _plan()
    data["pa_session_key"] = " "
    with pytest.raises(ValueError, match=r"value=' '"):
        RunPlan(**data)

    data["pa_session_key"] = "other_main"
    with pytest.raises(
        ValueError,
        match="pa_session_key='other_main'.*run_id='run_001'",
    ):
        RunPlan(**data)


def _probe_step(base: dict, **overrides) -> dict:
    return {
        **base,
        "step_id": "pretest_W_A",
        "kind": "pre_event_probe",
        "acc_num": None,
        "before_acc_num": 15,
        "memory_mode": "read_only",
        "stage_policy": "discard",
        "benchmark_session_timestamp": "2026-10-01T12:00:00+00:00",
        **overrides,
    }


def test_probe_requires_preceding_acc_step_or_borrow():
    # The read-only rule for probes was never relaxed.
    data = _plan()
    data["steps"].append(_probe_step(data["steps"][0], memory_mode="read_write"))
    with pytest.raises(ValueError, match="pre_event_probe must be read_only/discard"):
        RunPlan(**data)

    # A probe with nothing in front of it has no cut point to branch from.
    lonely = _plan()
    lonely["steps"] = [_probe_step(lonely["steps"][0])]
    with pytest.raises(
        ValueError,
        match="must follow an accumulation/forced_event step or declare",
    ):
        RunPlan(**lonely)

    # A baseline run borrows the tested run's pinned cut point instead, and
    # carries that run's mainline session key so the copied session is found.
    baseline = _plan()
    baseline["steps"] = [
        _probe_step(
            baseline["steps"][0],
            borrow_start_from="source_run/acc_108",
        )
    ]
    baseline["pa_session_key"] = "source_run_main"
    with pytest.raises(ValueError, match="pa_session_key='run_001_main'"):
        RunPlan(**{**baseline, "pa_session_key": None})
    plan = RunPlan(**baseline)
    assert plan.steps[0].borrowed_checkpoint() == ("source_run", "acc_108")
    # Probes continue the mainline session; the probe-specific key is gone.
    assert plan.pa_session_key_for_step(plan.steps[0]) == "source_run_main"

    malformed = {**baseline}
    malformed["steps"] = [
        _probe_step(baseline["steps"][0], borrow_start_from="source_run")
    ]
    with pytest.raises(ValueError, match="<run_id>/<pinned_checkpoint_id>"):
        RunPlan(**malformed)


def test_duplicate_step_id_fails():
    data = _plan()
    data["steps"].append({**data["steps"][0], "acc_num": 2})

    with pytest.raises(ValueError, match="duplicate step_id"):
        RunPlan(**data)


def test_missing_non_placeholder_script_fails(tmp_path: Path):
    plan_path = tmp_path / "run_plan.yaml"
    RunPlan(**_plan("missing.yaml")).dump_yaml(plan_path)

    with pytest.raises(FileNotFoundError):
        load_run_plan(plan_path, project_root=tmp_path)


def test_missing_placeholder_script_passes(tmp_path: Path):
    plan_path = tmp_path / "run_plan.yaml"
    RunPlan(**_plan("missing.yaml", placeholder=True)).dump_yaml(plan_path)

    plan = load_run_plan(plan_path, project_root=tmp_path)

    assert plan.steps[0].placeholder is True


def test_probe_field_policy_is_validated():
    data = _plan()
    data["steps"][0] = {
        **data["steps"][0],
        "step_id": "pretest_W_A",
        "kind": "pre_event_probe",
        "acc_num": None,
        "before_acc_num": 15,
        "memory_mode": "read_write",
        "stage_policy": "discard",
    }

    with pytest.raises(
        ValueError,
        match="step_id='pretest_W_A'.*memory_mode='read_write'.*stage_policy='discard'",
    ):
        RunPlan(**data)


def test_non_increasing_acc_num_error_contains_step_and_values():
    data = _plan()
    data["steps"].append(
        {
            **data["steps"][0],
            "step_id": "acc_002",
            "acc_num": 1,
            "benchmark_session_timestamp": "2026-10-01T15:00:00+00:00",
        }
    )

    with pytest.raises(
        ValueError,
        match="step_id='acc_002'.*previous_acc_num=1.*acc_num=1",
    ):
        RunPlan(**data)


@pytest.mark.parametrize(
    ("content", "root_type", "value"),
    [
        ("", "NoneType", "None"),
        ("- acc_001\n", "list", "['acc_001']"),
    ],
)
def test_run_plan_requires_mapping_root(
    tmp_path: Path,
    content: str,
    root_type: str,
    value: str,
):
    path = tmp_path / "run_plan.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(RunPlanLoadError) as exc_info:
        load_run_plan(path, validate_paths=False)

    message = str(exc_info.value)
    assert str(path) in message
    assert f"root_type={root_type}" in message
    assert f"value={value}" in message


def _timeline_and_probe_steps() -> list[dict]:
    """acc_001 acc_002 pre_01 acc_003 forced_01 fin_01 fin_02.

    pre_01 sits one envelope before acc_003 (its before_acc step); the two
    finals share the one final cut point after evolv_01.
    """
    shape = [
        ("acc_001", "accumulation", 1, None, "2026-10-01T09:00:00+00:00"),
        ("acc_002", "accumulation", 2, None, "2026-10-02T09:00:00+00:00"),
        ("pre_01", "pre_event_probe", None, 3, "2026-10-03T08:00:00+00:00"),
        ("acc_003", "accumulation", 3, None, "2026-10-03T09:00:00+00:00"),
        ("evolv_01", "forced_event", 4, None, "2026-10-04T09:00:00+00:00"),
        ("fin_01", "final_probe", None, None, "2026-10-05T09:00:00+00:00"),
        ("fin_02", "final_probe", None, None, "2026-10-05T09:00:00+00:00"),
    ]
    steps = []
    for step_id, kind, acc_num, before_acc_num, timestamp in shape:
        is_probe = kind in {"pre_event_probe", "final_probe"}
        step = {
            "step_id": step_id,
            "kind": kind,
            "persona_id": "user_a",
            "script_path": "script.yaml",
            "context": "work",
            "benchmark_session_timestamp": timestamp,
            "memory_mode": "read_only" if is_probe else "read_write",
            "stage_policy": "discard" if is_probe else "commit",
        }
        if acc_num is not None:
            step["acc_num"] = acc_num
        if before_acc_num is not None:
            step["before_acc_num"] = before_acc_num
        steps.append(step)
    return steps


def test_pinned_step_ids_derive_from_step_order() -> None:
    data = _plan()
    data["steps"] = _timeline_and_probe_steps()
    plan = RunPlan(**data)

    # pre_01's nearest preceding accumulation step is acc_002; both final
    # probes share acc_003, so it is not pinned twice.
    assert plan.pinned_step_ids() == {"acc_002", "acc_003"}
    assert "acc_001" not in plan.pinned_step_ids()
    assert plan.checkpoint_rolling_window == 3


def test_a_forced_event_step_is_never_pinned() -> None:
    """No consolidation runs after a forced event.

    A pinned cut point flushes the never-compacted turns
    (`harness/orchestrator.py:_flush_uncompacted_turns`), so pinning a forced
    event would consolidate right after it. The pin falls back to the last
    accumulation step before the event instead.
    """
    data = _plan()
    data["steps"] = _timeline_and_probe_steps()
    plan = RunPlan(**data)

    forced = {step.step_id for step in plan.steps if step.kind == "forced_event"}
    assert forced, "the fixture plan has a forced event to pin or not pin"
    assert not (forced & plan.pinned_step_ids())
    # The probes right after evolv_01 pin the accumulation step before it.
    assert "acc_003" in plan.pinned_step_ids()


def test_checkpoint_rolling_window_must_be_positive() -> None:
    data = _plan()
    data["checkpoint_rolling_window"] = 0
    with pytest.raises(ValueError):
        RunPlan(**data)


def _full_timeline_plan(*, active: int = 99) -> dict:
    """A full-shape plan: 108 acc_num slots, 5 forced, 5 pre, 28 final.

    `active` is the Outline's own count — every authored session, the five
    forced events included — so the default leaves the nine gap days a real
    persona's Outline leaves.
    """
    from datetime import datetime, timedelta, timezone

    data = _plan()
    base = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    template = data["steps"][0]
    forced_at = {10, 30, 50, 70, 90}
    steps: list[dict] = []
    slot = 0
    authored = 0
    for acc_num in range(1, 109):
        is_forced = acc_num in forced_at
        # Slots beyond the authored budget stay placeholders.
        placeholder = authored >= active
        if not placeholder:
            authored += 1
        steps.append(
            {
                **template,
                "step_id": f"acc_{acc_num:03d}",
                "kind": "forced_event" if is_forced else "accumulation",
                "acc_num": acc_num,
                "placeholder": placeholder,
                "benchmark_session_timestamp": (
                    base + timedelta(hours=slot)
                ).isoformat(),
            }
        )
        slot += 1
    # Each pre-event probe sits right before its forced event, one envelope
    # earlier on the clock; the 28 finals share the one cut point after acc_108.
    for index, acc_num in enumerate(sorted(forced_at)):
        position = next(
            i for i, step in enumerate(steps) if step["acc_num"] == acc_num
        )
        steps.insert(
            position,
            {
                **template,
                "step_id": f"pre_{index:02d}",
                "kind": "pre_event_probe",
                "acc_num": None,
                "before_acc_num": acc_num,
                "memory_mode": "read_only",
                "stage_policy": "discard",
                "benchmark_session_timestamp": (
                    base + timedelta(hours=acc_num - 2)
                ).isoformat(),
            },
        )
    for index in range(28):
        steps.append(
            {
                **template,
                "step_id": f"fin_{index:02d}",
                "kind": "final_probe",
                "acc_num": None,
                "memory_mode": "read_only",
                "stage_policy": "discard",
                "benchmark_session_timestamp": (
                    base + timedelta(hours=slot)
                ).isoformat(),
            }
        )
    data["steps"] = steps
    return data


def test_full_timeline_requires_99_active_timeline_steps():
    """The 99 the Outline counts: every authored session, forced ones included.

    A complete Outline holds 99 entries of which five are `evolving_event`
    (`meta.active_session_count`, held equal to the entry count by
    `scripts/validate_outline.py`), so the 1..108 skeleton has nine gap days.
    """
    plan = RunPlan(**_full_timeline_plan())
    # The acc_num skeleton is unchanged: 1..108, 5 of them forced events.
    assert [step.acc_num for step in plan.steps if step.acc_num] == list(
        range(1, 109)
    )
    assert sum(step.placeholder for step in plan.steps) == 9
    validate_full_timeline_shape(plan)

    short = RunPlan(**_full_timeline_plan(active=98))
    with pytest.raises(
        ValueError,
        match="must contain 99 active timeline steps: value=98",
    ):
        validate_full_timeline_shape(short)


def test_frozen_plan_records_the_contract_version(tmp_path: Path) -> None:
    path = tmp_path / "run_plan.yaml"
    RunPlan(**_plan()).dump_yaml(path)

    dumped = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert dumped["ix_contract_version"] == IX_CONTRACT_VERSION
    assert load_run_plan(path, project_root=tmp_path, validate_paths=False)


def test_no_spec_carries_a_neutral_setting() -> None:
    for spec in IX_PASSIVE_SPECS.values():
        assert not hasattr(spec, "neutral_setting")
        assert not hasattr(spec, "neutral_description")


def test_frozen_plan_records_the_memory_pool_version(tmp_path: Path) -> None:
    path = tmp_path / "run_plan.yaml"
    RunPlan(**_plan()).dump_yaml(path)

    dumped = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert dumped["memory_pool_version"] == MEMORY_POOL_VERSION
    assert load_run_plan(path, project_root=tmp_path, validate_paths=False)


def test_frozen_resolved_gt_round_trips_as_a_record(tmp_path: Path):
    (tmp_path / "script.yaml").write_text("ok: true\n", encoding="utf-8")
    plan = RunPlan(**_plan(), resolved_gt={"path": "resolved_gt.yaml"})
    plan_path = plan.dump_yaml(tmp_path / "run_plan.yaml")
    (tmp_path / "resolved_gt.yaml").write_bytes(b"persona: user_a\ncells: {}\n# edited\n")
    assert load_run_plan(plan_path, project_root=tmp_path).resolved_gt == plan.resolved_gt


def test_frozen_resolved_gt_rejects_a_foreign_path() -> None:
    with pytest.raises(ValueError, match="resolved_gt"):
        RunPlan(**_plan(), resolved_gt={"path": "Authoring/probe/outline/user_a/resolved_gt.yaml"})


# ── The mem0 extraction chunk strategy, and the plans frozen without it ─
#
# `extraction_chunk_strategy` decides whether one over-budget pool message is
# cut into sub-messages before mem0 extracts from it, so it changes what mem0
# infers over and belongs in the frozen configuration. It arrived on
# 2026-09-18, after the user_b runs were frozen and while they were still
# being resumed, so it is the one chunk-policy field with a default: absent
# means the legacy strategy, which is the one those plans ran.


def test_a_plan_frozen_without_the_chunk_strategy_loads_as_legacy() -> None:
    """The 2026-09-18 user_b plans predate the field and must keep loading."""
    data = _plan()
    mem0_runtime_config = dict(data["mem0_runtime_config"])
    del mem0_runtime_config["extraction_chunk_strategy"]
    data["mem0_runtime_config"] = mem0_runtime_config

    plan = RunPlan(**data)

    assert plan.mem0_runtime_config is not None
    assert (
        plan.mem0_runtime_config.extraction_chunk_strategy
        == "legacy_no_message_split"
    )


def test_an_unknown_chunk_strategy_is_rejected() -> None:
    data = _mem0_runtime_config().model_dump(mode="json")
    data["extraction_chunk_strategy"] = "message_split_v2"

    with pytest.raises(ValueError, match="extraction_chunk_strategy"):
        Mem0RuntimeConfig.model_validate(data)
