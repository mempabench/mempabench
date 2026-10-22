"""Run real PA sessions through the benchmark orchestrator loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from harness import run_plan_builder
from harness.benchmark_clock import (
    SESSION_ENVELOPE_MINUTES,
    SUPPRESS_RESUME_DELAY_MINUTES,
)
from harness.memory_runtime import MemoryRuntime
from harness.memory_io import MemoryIORecorder
from harness.forced_event_executor import ForcedEventExecutor
from harness.gate import gate_cue_carry_hook, resolve_authorised_tool
from harness.checkpoint import (
    BACKEND_STATE_LAYER,
    LOCAL_MEMORY_CONDITIONS,
    MEM0_HISTORY_FILENAME,
    CheckpointManager,
    read_pending_recovery,
)
from harness.ix_seed import seed_ix_selection
from harness.orchestrator import (
    PROBE_KINDS,
    Orchestrator,
    OrchestratorConfig,
    RunPreferenceOverlay,
    resume_or_restore,
)
from harness.pa_factory import (
    AssistantLLMProviderProfile,
    RecordingPA,
    build_pa_from_provider_profile,
)
from harness.run_plan import (
    EMBEDDING_MEMORY_CONDITIONS,
    FILE_MEMORY_CONDITION,
    MEM0_CONDITION,
    Mem0RuntimeConfig,
    MEMORY_SIDE_LM_CONDITIONS,
    MEMOS_CONDITION,
    MemOSRuntimeConfig,
    RunPlan,
    RunPlanLoadError,
    RunStep,
    DEFAULT_SIMULATOR_MODEL,
    SimulatorRuntimeConfig,
    load_run_plan,
)
from harness.run_config import effective_failure_policy, load_run_config
from harness.run_info import memos_health_record, plan_scope_record, write_run_info
from harness.run_layout import (
    run_dir_for,
    runs_root,
    slug,
    supersede_step_dirs,
)
from harness.run_ledger import RunLedger, UnsafeRunResumeError
from harness.provider_avoidance import ProviderAvoidance
from harness.session_executor import SessionExecutor
from harness.state_runtime import StageRuntime
from memory.adapter import MemoryAdapter
from memory.event_writer import turn_journal_for
from harness.probe_slots import workspace_root

from memory.backends.runtime_contracts import (
    MEM0_HISTORY_DB_RELATIVE_PATH,
    MEM0_OPENAI_EMBEDDING_DIMENSIONS,
    MEM0_OPENAI_EMBEDDING_MODEL,
    MEM0_OPENAI_EMBEDDING_PROVIDER,
    MEM0_QDRANT_BM25_ENABLED,
    MEM0_QDRANT_DISTANCE,
    MEM0_QDRANT_ON_DISK,
    MEM0_RETRIEVAL_RERANK,
    MEM0_RETRIEVAL_SIMILARITY_THRESHOLD,
    MEM0_SUBMODULE_COMMIT,
    MEM0_VERSION,
    MEMOS_MEMPA_PATCH_VERSION,
    MEMOS_SUBMODULE_COMMIT,
    MEMOS_VERSION,
)
from memory.registry import default_registry
from pa.agent.ix.render_types import turn_budget
from pa.providers.registry import find_by_name
from pa.providers.request_policy import resolve_model_request_policy
from pa.session.manager import SessionManager
from simulator.config import SimulatorConfig
from simulator.loop import SimulatorLoop
from simulator.memory import MemoryConsolidationError, UserMemoryStore
from simulator.profiles import load_simulator_profile
from simulator.schemas import PersonaIdentity, PreferenceMatrix, PreferencePatch
from simulator.self_report import SessionEndSelfReporter
from state_server.store import NUMBER_WORDS


DEFAULT_PA_MODEL = "deepseek/deepseek-v4-flash"
ASSISTANT_LLM_PROVIDER_NAME = "openrouter"
PROMPT_WORKSPACE_ENV = "NANOBOT_PROMPT_WORKSPACE"

#: The PA's persona file, written once per run into the workspace. The second
#: sentence is round-1's acc_006 finding: a turn with no tool call at all
#: reported three citation merges as done, and the claim was then consolidated
#: into the memory artefact as fact.
SOUL_TEXT = (
    "You are a helpful personal assistant.\n"
    "Say a file changed only when a writing tool returned status \"updated\" in "
    "the same turn; otherwise say what you would change and ask.\n"
)


def create_mem0_client(*args: Any, **kwargs: Any) -> Any:
    from memory.backends.mem0_client import create_mem0_client as create

    return create(*args, **kwargs)


def create_simple_rag_mem0_client(*args: Any, **kwargs: Any) -> Any:
    from memory.backends.mem0_client import create_simple_rag_mem0_client as create

    return create(*args, **kwargs)


def MemOSHttpClient(*args: Any, **kwargs: Any) -> Any:
    from memory.backends.memos_client import MemOSHttpClient as client_type

    return client_type(*args, **kwargs)


def _memos_usage_reporter(run_id: str) -> Any:
    from memory.backends.memos_usage_log import (
        MEMOS_CONTAINER_NAME,
        MemosUsageReporter,
        MemosUsageTail,
    )

    raw_container = os.environ.get("MEMOS_CONTAINER_NAME", MEMOS_CONTAINER_NAME)
    container = raw_container.strip()
    if not container:
        raise ValueError(
            f"MEMOS_CONTAINER_NAME must not be empty: value={raw_container!r}"
        )
    return MemosUsageReporter(
        tail=MemosUsageTail(container=container), run_id=run_id
    )


def _new_run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _effective_consolidation_model(
    memory_condition: str,
    configured_model: str | None,
) -> str:
    if memory_condition != FILE_MEMORY_CONDITION:
        return ""
    return configured_model or ""


def _effective_memory_side_model(
    memory_condition: str,
    configured_model: str | None,
) -> str:
    if memory_condition not in MEMORY_SIDE_LM_CONDITIONS:
        return ""
    return configured_model or ""


def _mem0_collection_name(run_id: str) -> str:
    return f"mempa_{slug(run_id)}_mem0"


def _mem0_branch_client_kwargs(namespace: str, branch_dir: Path) -> dict[str, Any]:
    """What a probe branch changes about the mainline mem0 client arguments.

    `branch_from` has already restored the pinned collection under `namespace`
    and copied the history database into the branch, so both exist and neither
    may be created afresh.
    """
    return {
        "collection_name": namespace,
        "allow_existing_collection": True,
        # Never under `--overwrite` either: the branch namespace was restored a
        # moment ago and dropping it would empty the probe's own memory.
        "drop_existing_collection": False,
        "history_db_path": (
            Path(branch_dir) / BACKEND_STATE_LAYER / MEM0_HISTORY_FILENAME
        ),
        "allow_existing_history_db": True,
        "require_existing_history_db": True,
    }


def _requires_committed_adapter_writes(step: RunStep) -> bool:
    """Whether this step's PA may refuse a turn whose adapter write is not committed.

    Plan §4.1: a test probe (`pre_event_probe`, `final_probe`) is read-only by
    design — `run_plan.py` pins it to `read_only`/`discard` — so its adapter
    answers `skipped_read_only` on every turn and `record_adapter_write_commit`
    would raise `AdapterWriteCommitError` on the first one. The steps whose
    writes the run actually depends on, accumulation and forced events, keep
    the requirement; widening the accepted status set instead would weaken it
    for them too.
    """
    return step.kind not in PROBE_KINDS


async def _build_branch_memory_adapter(
    *,
    memory_condition: str,
    namespace: str,
    branch_dir: Path,
    mainline_options: dict[str, Any],
    mem0_client_kwargs: dict[str, Any] | None = None,
    mem0_llm_kwargs: dict[str, Any] | None = None,
) -> MemoryAdapter:
    """The adapter a probe reads its branch's restored namespace through.

    Same registry entry and same options as the mainline adapter; only the
    namespace moves, to the copy `branch_from` restored for this branch alone.
    What "the namespace" means differs per backend, and getting that wrong is
    silent: memos names a cube and mem0 a Qdrant collection.
    """
    options = dict(mainline_options)
    if memory_condition == MEM0_CONDITION:
        options["client"] = create_mem0_client(
            **(mem0_llm_kwargs or {}),
            **{
                **(mem0_client_kwargs or {}),
                **_mem0_branch_client_kwargs(namespace, branch_dir),
            },
        )
    elif memory_condition == "simple_rag":
        options["client"] = create_simple_rag_mem0_client(
            **{
                **(mem0_client_kwargs or {}),
                **_mem0_branch_client_kwargs(namespace, branch_dir),
            },
        )
    elif memory_condition == MEMOS_CONDITION:
        options["cube_id"] = namespace
    else:
        raise NotImplementedError(
            "A probe branch on this memory condition has no adapter of "
            "its own, so it would retrieve nothing: "
            f"memory_condition={memory_condition!r}"
        )
    return default_registry().create(memory_condition, **options)


_MEMOS_MEMORY_GROUPS = (
    "text_mem",
    "act_mem",
    "para_mem",
    "pref_mem",
    "tool_mem",
    "skill_mem",
)


def _committed_memos_native_ids(
    records: list[dict[str, Any]],
) -> set[str]:
    native_ids: set[str] = set()
    for record in records:
        if record.get("backend") != MEMOS_CONDITION:
            continue
        if record.get("operation") != "add_memories":
            continue
        status = record.get("status")
        record_native_ids = record.get("native_ids")
        if not isinstance(record_native_ids, list) or not all(
            isinstance(memory_id, str) and memory_id.strip()
            for memory_id in record_native_ids
        ):
            raise ValueError(
                "MemOS audit add_memories record has invalid native_ids: "
                f"sequence={record.get('sequence')!r} "
                f"value={record_native_ids!r}"
            )
        if status == "committed":
            if not record_native_ids:
                raise ValueError(
                    "Committed MemOS audit record must contain native_ids: "
                    f"sequence={record.get('sequence')!r}"
                )
            native_ids.update(record_native_ids)
        elif status == "accepted_no_change":
            if record_native_ids:
                raise ValueError(
                    "MemOS accepted_no_change audit record must not contain "
                    "native_ids: "
                    f"sequence={record.get('sequence')!r} "
                    f"value={record_native_ids!r}"
                )
        else:
            raise ValueError(
                "MemOS add_memories audit record has invalid status: "
                f"sequence={record.get('sequence')!r} value={status!r}"
            )
    return native_ids


def _stored_memos_native_ids(
    grouped_memories: dict[str, Any],
    *,
    expected_cube_id: str,
) -> set[str]:
    native_ids: set[str] = set()
    for group_name in _MEMOS_MEMORY_GROUPS:
        buckets = grouped_memories.get(group_name, [])
        if not isinstance(buckets, list):
            raise TypeError(
                "MemOS backend state group must be a list: "
                f"group={group_name!r} value={buckets!r}"
            )
        for bucket_index, bucket in enumerate(buckets):
            if not isinstance(bucket, dict):
                raise TypeError(
                    "MemOS backend state bucket must be an object: "
                    f"group={group_name!r} index={bucket_index!r} "
                    f"value={bucket!r}"
                )
            cube_id = bucket.get("cube_id")
            if cube_id != expected_cube_id:
                raise ValueError(
                    "MemOS backend returned a different cube: "
                    f"expected_cube_id={expected_cube_id!r} "
                    f"group={group_name!r} index={bucket_index!r} "
                    f"value={cube_id!r}"
                )
            memories = bucket.get("memories")
            if not isinstance(memories, list):
                raise TypeError(
                    "MemOS backend state memories must be a list: "
                    f"group={group_name!r} index={bucket_index!r} "
                    f"value={memories!r}"
                )
            for memory_index, memory in enumerate(memories):
                if not isinstance(memory, dict):
                    raise TypeError(
                        "MemOS backend state memory must be an object: "
                        f"group={group_name!r} bucket_index={bucket_index!r} "
                        f"memory_index={memory_index!r} value={memory!r}"
                    )
                memory_id = memory.get("id") or memory.get("memory_id")
                if not isinstance(memory_id, str) or not memory_id.strip():
                    raise ValueError(
                        "MemOS backend state memory requires a non-empty id: "
                        f"group={group_name!r} bucket_index={bucket_index!r} "
                        f"memory_index={memory_index!r} value={memory_id!r}"
                    )
                native_ids.add(memory_id)
    return native_ids


def _validate_memos_backend_state(
    *,
    client: MemOSHttpClient,
    plan: RunPlan,
    memory_io_recorder: MemoryIORecorder,
    run_ledger: RunLedger,
) -> None:
    if run_ledger.steps and not memory_io_recorder.path.exists():
        raise RuntimeError(
            "Cannot resume MemOS run with recorded steps but no memory audit: "
            f"run_id={plan.run_id!r} "
            f"ledger_path={str(run_ledger.path)!r} "
            f"audit_path={str(memory_io_recorder.path)!r}"
        )

    expected_ids = _committed_memos_native_ids(
        memory_io_recorder.read_records()
    )
    grouped_memories = client.get_memories(
        {
            "mem_cube_id": plan.memos_cube_id,
            "user_id": plan.persona_id,
            "include_preference": False,
            "include_tool_memory": False,
            "include_skill_memory": False,
        }
    )
    actual_ids = _stored_memos_native_ids(
        grouped_memories,
        expected_cube_id=plan.memos_cube_id,
    )
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        raise RuntimeError(
            "MemOS backend state does not match committed memory audit: "
            f"run_id={plan.run_id!r} cube_id={plan.memos_cube_id!r} "
            f"expected_count={len(expected_ids)!r} "
            f"actual_count={len(actual_ids)!r} "
            f"missing_ids={missing_ids!r} extra_ids={extra_ids!r}"
        )


def _audited_mem0_native_ids(
    records: list[dict[str, Any]],
    *,
    memory_condition: str,
) -> tuple[set[str], set[str]]:
    active_ids: set[str] = set()
    history_ids: set[str] = set()
    for record in records:
        if record.get("backend") != memory_condition:
            continue
        if record.get("operation") != "add":
            continue
        status = record.get("status")
        native_ids = record.get("native_ids")
        native_events = record.get("native_events")
        if status == "accepted_no_change":
            if native_ids or native_events:
                raise ValueError(
                    "Mem0 accepted_no_change audit record must not contain "
                    f"native mutations: sequence={record.get('sequence')!r} "
                    f"native_ids={native_ids!r} native_events={native_events!r}"
                )
            continue
        if status != "committed":
            raise ValueError(
                "Mem0 audit ADD record has an invalid terminal status: "
                f"sequence={record.get('sequence')!r} value={status!r}"
            )
        if (
            not isinstance(native_ids, list)
            or not native_ids
            or not all(
                isinstance(native_id, str) and native_id.strip()
                for native_id in native_ids
            )
        ):
            raise ValueError(
                "Committed Mem0 audit ADD record requires non-empty native_ids: "
                f"sequence={record.get('sequence')!r} value={native_ids!r}"
            )
        if (
            not isinstance(native_events, list)
            or len(native_events) != len(native_ids)
            or any(
                event not in {"ADD", "UPDATE", "DELETE"}
                for event in native_events
            )
        ):
            raise ValueError(
                "Committed Mem0 audit ADD record has invalid native_events: "
                f"sequence={record.get('sequence')!r} "
                f"native_ids={native_ids!r} native_events={native_events!r}"
            )
        for native_id, event in zip(native_ids, native_events):
            history_ids.add(native_id)
            if event in {"ADD", "UPDATE"}:
                active_ids.add(native_id)
            else:
                active_ids.discard(native_id)
    return active_ids, history_ids


def _validate_mem0_backend_state(
    *,
    client: Mem0Client,
    plan: RunPlan,
    memory_io_recorder: MemoryIORecorder,
    run_ledger: RunLedger,
) -> None:
    if run_ledger.steps and not memory_io_recorder.path.exists():
        raise RuntimeError(
            "Cannot resume Mem0-backed run with recorded steps but no memory "
            f"audit: run_id={plan.run_id!r} "
            f"ledger_path={str(run_ledger.path)!r} "
            f"audit_path={str(memory_io_recorder.path)!r}"
        )
    expected_active_ids, expected_history_ids = _audited_mem0_native_ids(
        memory_io_recorder.read_records(),
        memory_condition=plan.memory_condition,
    )
    if not run_ledger.steps and (expected_active_ids or expected_history_ids):
        raise RuntimeError(
            "Mem0-backed run has committed memory audit records but no ledger "
            f"steps: run_id={plan.run_id!r} "
            f"active_ids={sorted(expected_active_ids)!r} "
            f"history_ids={sorted(expected_history_ids)!r}"
        )
    actual_active_ids = client.native_ids()
    actual_history_ids = client.history_native_ids()
    missing_active_ids = sorted(expected_active_ids - actual_active_ids)
    extra_active_ids = sorted(actual_active_ids - expected_active_ids)
    missing_history_ids = sorted(expected_history_ids - actual_history_ids)
    extra_history_ids = sorted(actual_history_ids - expected_history_ids)
    if (
        missing_active_ids
        or extra_active_ids
        or missing_history_ids
        or extra_history_ids
    ):
        raise RuntimeError(
            "Mem0 backend state does not match committed memory audit: "
            f"run_id={plan.run_id!r} "
            f"missing_active_ids={missing_active_ids!r} "
            f"extra_active_ids={extra_active_ids!r} "
            f"missing_history_ids={missing_history_ids!r} "
            f"extra_history_ids={extra_history_ids!r}"
        )


def _effective_memos_runtime_config(
    *,
    memory_condition: str,
    api_base: str | None,
    memory_side_provider_name: str,
    memory_side_model: str,
    memory_side_api_base: str,
) -> MemOSRuntimeConfig | None:
    if memory_condition != MEMOS_CONDITION:
        return None
    if not api_base or not api_base.strip():
        raise ValueError(
            f"memos requires memos_api_base: value={api_base!r}"
        )
    if not memory_side_model:
        raise ValueError(
            "memos requires memory_side_model: "
            f"value={memory_side_model!r}"
        )
    if memory_side_provider_name != ASSISTANT_LLM_PROVIDER_NAME:
        raise ValueError(
            "MemOS memory-side provider must be OpenRouter: "
            f"value={memory_side_provider_name!r}"
        )
    if (
        memory_side_api_base.rstrip("/")
        != run_plan_builder.OPENROUTER_API_BASE
    ):
        raise ValueError(
            "MemOS memory-side API base must use the standard OpenRouter "
            f"endpoint: value={memory_side_api_base!r}"
        )
    memory_side_request_policy = resolve_model_request_policy(
        memory_side_provider_name,
        memory_side_model,
    )
    memory_side_request_policy.require_exact_role("memory_extraction")
    provider_spec = find_by_name(memory_side_provider_name)
    if provider_spec is None:
        raise ValueError(
            "memos requires a registered memory-side provider: "
            f"provider_name={memory_side_provider_name!r}"
        )
    return MemOSRuntimeConfig(
        api_base=api_base,
        memos_version=MEMOS_VERSION,
        memos_submodule_commit=MEMOS_SUBMODULE_COMMIT,
        mempa_patch_version=MEMOS_MEMPA_PATCH_VERSION,
        memory_side_provider_name=memory_side_provider_name,
        memory_side_model=memory_side_model,
        memory_side_api_base=memory_side_api_base,
        memory_side_model_roles={
            "chat": memory_side_model,
            "memory_reader": memory_side_model,
            "general_reader": memory_side_model,
        },
        memory_side_temperature_policy=memory_side_request_policy.temperature,
        memory_side_completion_tokens_parameter=(
            memory_side_request_policy.completion_tokens_parameter
        ),
        memory_side_reasoning_effort_policy=(
            memory_side_request_policy.reasoning_effort
        ),
        memory_side_task_affinity_required=False,
        memory_side_task_affinity_namespace="",
        embedding_provider=MEM0_OPENAI_EMBEDDING_PROVIDER,
        embedding_model=MEM0_OPENAI_EMBEDDING_MODEL,
        embedding_api_base=os.getenv(
            "OPENAI_EMB_API_BASE", "https://api.openai.com/v1"
        ),
        embedding_dimensions=MEM0_OPENAI_EMBEDDING_DIMENSIONS,
        # Deployment values: `memory_backends.memos` in configs/run_config.yaml.
        **load_run_config().memory_backends.memos.model_dump(),
    )


def _effective_mem0_runtime_config(
    memory_condition: str,
    *,
    extraction_chunk_strategy: str | None = None,
    extraction_max_output_tokens: int | None = None,
) -> Mem0RuntimeConfig | None:
    if memory_condition not in {MEM0_CONDITION, "simple_rag"}:
        return None
    # Imported here, the way `memory/registry.py` imports the adapter: a run
    # on another backend never pays for the mem0 package.
    from memory.backends.mem0_adapter import mem0_embedding_chunk_policy

    chunk_policy = mem0_embedding_chunk_policy()
    if extraction_chunk_strategy is not None:
        chunk_policy["extraction_chunk_strategy"] = extraction_chunk_strategy
    run_config = load_run_config()
    # A new run's budget is `budgets.mem0_extraction_max_output_tokens`; a
    # resumed run passes its plan's value.
    chunk_policy["extraction_max_output_tokens"] = (
        extraction_max_output_tokens
        if extraction_max_output_tokens is not None
        else run_config.budgets.mem0_extraction_max_output_tokens
    )
    mem0_deployment = run_config.memory_backends.mem0
    return Mem0RuntimeConfig(
        **chunk_policy,
        mem0_version=MEM0_VERSION,
        mem0_submodule_commit=MEM0_SUBMODULE_COMMIT,
        embedding_api_base=os.environ.get(
            "OPENAI_EMB_API_BASE",
            "https://api.openai.com/v1",
        ),
        retrieval_similarity_threshold=(
            MEM0_RETRIEVAL_SIMILARITY_THRESHOLD
        ),
        retrieval_rerank=MEM0_RETRIEVAL_RERANK,
        qdrant_distance=MEM0_QDRANT_DISTANCE,
        qdrant_on_disk=MEM0_QDRANT_ON_DISK,
        qdrant_bm25_enabled=MEM0_QDRANT_BM25_ENABLED,
        enable_entity_memory=memory_condition == MEM0_CONDITION,
        entity_model=(
            mem0_deployment.entity_model if memory_condition == MEM0_CONDITION else None
        ),
        spacy_version=(
            mem0_deployment.spacy_version if memory_condition == MEM0_CONDITION else None
        ),
        history_db_relative_path=MEM0_HISTORY_DB_RELATIVE_PATH,
    )


def _qdrant_endpoint_from_environment(memory_condition: str) -> tuple[str, int]:
    if memory_condition not in {MEM0_CONDITION, "simple_rag"}:
        return "", 0
    qdrant_host = os.environ.get("QDRANT_HOST", "127.0.0.1").strip()
    raw_qdrant_port = os.environ.get("QDRANT_PORT", "6333")
    if not qdrant_host:
        raise ValueError(f"QDRANT_HOST must not be empty: value={qdrant_host!r}")
    try:
        qdrant_port = int(raw_qdrant_port)
    except ValueError as exc:
        raise ValueError(
            f"QDRANT_PORT must be an integer: value={raw_qdrant_port!r}"
        ) from exc
    if not 1 <= qdrant_port <= 65535:
        raise ValueError(
            "QDRANT_PORT must be between 1 and 65535: "
            f"value={qdrant_port!r}"
        )
    return qdrant_host, qdrant_port


def _run_dir(
    *,
    pa_model: str,
    memory_condition: str,
    persona_id: str,
    run_id: str,
) -> Path:
    return run_dir_for(
        runs_root(),
        pa_model=pa_model,
        memory_condition=memory_condition,
        persona_id=persona_id,
        run_id=run_id,
    )


def _timeline_half_of_plan(plan_path: Path) -> RunPlan:
    """A finished run's plan, read as a record with its probe steps dropped.

    Only done non-probe steps and `memos_cube_id` are ever carried over, so
    the probe half is not read: the builder rebuilds every probe step. The
    script paths go unchecked for the same reason: this plan is not the one
    that runs, and the builder resolves and checks the artifacts behind every
    step it keeps.
    """
    data = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RunPlanLoadError(
            f"Run-plan YAML root must be a mapping: path={plan_path} "
            f"root_type={type(data).__name__}"
        )
    data["steps"] = [
        step
        for step in (data.get("steps") or [])
        if not (isinstance(step, dict) and step.get("kind") in PROBE_KINDS)
    ]
    try:
        return RunPlan.model_validate(data)
    except ValidationError as exc:
        raise RunPlanLoadError(
            f"Invalid run plan: path={plan_path} error={exc}"
        ) from exc


def _finished_plan_carry_over(
    *,
    enabled: bool,
    plan_path: Path,
    run_dir: Path,
    run_id: str,
) -> tuple[RunPlan | None, frozenset[str]]:
    """What a re-freeze of a finished run's plan may not re-mint (§2.4.6).

    `--regenerate-plan` rebuilds the plan from today's artifacts, which is the
    point when a probe line is being attached and wrong for everything the run
    has already run: a re-minted `memos_cube_id` points the mainline at an
    empty cube while the checkpoints still name the old one, and a re-frozen
    forced step drops the digests its finished step actually ran under. So a
    directory that holds both a plan and a ledger with done steps hands both
    back to the builder.
    """
    if not enabled or not plan_path.exists():
        return None, frozenset()
    # Read, never create: `RunLedger` writes an empty ledger for a directory
    # that has none, and a run with no ledger has nothing to carry over.
    if not (run_dir / "ledger.json").is_file():
        return None, frozenset()
    existing_plan = _timeline_half_of_plan(plan_path)
    ledger = RunLedger(run_dir, run_id)
    done_step_ids = frozenset(
        step_id
        for step_id, record in ledger.steps.items()
        if record.status == "done"
    )
    if not done_step_ids:
        return None, frozenset()
    return existing_plan, done_step_ids


DEFAULT_SIMULATOR_MEMORY_MODE = "nanobot"


def _prepare_simulator_workspace(*, persona_id: str, run_dir: Path) -> Path:
    source = PROJECT_ROOT / "simulator/profiles" / persona_id
    if not source.exists():
        raise FileNotFoundError(f"simulator profile not found: {source}")
    dest = run_dir / "simulator_workspace" / persona_id
    if not dest.exists():
        shutil.copytree(source, dest)
    config_path = dest / "profile.yaml"
    config: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(
                "Simulator profile config must be a mapping: "
                f"path={config_path} value={loaded!r}"
            )
        config = loaded or {}
    # A run stays self-contained: the persona files the profile renders from
    # are copied in and declared profile-relative, so an archived run keeps the
    # identity and matrix it actually ran with even if `data/personas` moves on.
    source_profile = load_simulator_profile(source)
    for key, filename, persona_path in (
        ("identity", "identity.yaml", source_profile.identity_path),
        (
            "preference_matrix",
            "preferences.yaml",
            source_profile.preference_matrix_path,
        ),
    ):
        if persona_path is None:
            # Missing files stay missing: the profile loader is what reports it.
            continue
        if Path(persona_path).resolve().is_relative_to(source.resolve()):
            # The profile workspace already carries it; copytree brought it over.
            continue
        local_path = dest / filename
        if not local_path.exists():
            shutil.copy2(persona_path, local_path)
        config[key] = filename
    memory_config = config.get("memory")
    if memory_config is not None and not isinstance(memory_config, dict):
        raise ValueError(
            "Simulator profile memory config must be a mapping: "
            f"path={config_path} value={memory_config!r}"
        )
    # The simulated user's own profile memory is writable: it is what carries
    # the persona across sessions of a run. Keep whatever mode the profile
    # config already declares; with none declared, use the store's normal
    # writable mode.
    config["memory"] = {
        **(memory_config or {}),
        "mode": (memory_config or {}).get("mode") or DEFAULT_SIMULATOR_MEMORY_MODE,
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return dest


def _parse_force_ix(values: list[str] | None) -> dict[str, dict[str, str]]:
    """`--force-ix STEP:ATTRIBUTE=SETTING` -> `{step_id: {attribute: setting}}`."""
    forced: dict[str, dict[str, str]] = {}
    for raw in values or []:
        step_id, colon, rest = raw.partition(":")
        attribute, equals, setting = rest.partition("=")
        if not (colon and equals and step_id and attribute and setting):
            raise ValueError(
                f"invalid --force-ix value: {raw!r} (want STEP:ATTRIBUTE=SETTING)"
            )
        forced.setdefault(step_id, {})[attribute] = setting
    return forced


def _parse_only_steps(value: str | None) -> frozenset[str] | None:
    """`--only-steps fin_02,pre_01` -> `{"fin_02", "pre_01"}`; None when unset.

    Whether every id names a probe step of the plan is checked by the
    orchestrator once the plan is loaded (`--only-steps` is probe-phase only).
    """
    if value is None:
        return None
    step_ids = frozenset(part.strip() for part in value.split(",") if part.strip())
    if not step_ids:
        raise ValueError(
            f"invalid --only-steps value: {value!r} (want a comma-separated "
            "list of probe step ids, e.g. fin_02,pre_01)"
        )
    return step_ids


#: Every condition, MemOS included (owner ruling 2026-09-20), accepts four
#: independent probe branches. Mainline steps remain sequential.
PROBE_CONCURRENCY_CAP = 4


def _probe_concurrency_arg(value: str) -> int:
    """`--probe-concurrency`: a worker count, so a positive integer.

    A non-integer lands here as a `ValueError` and takes the same refusal as 0
    and the negatives, so one mistake gets one message.
    """
    try:
        workers = int(value)
    except ValueError:
        workers = 0
    if workers < 1:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a positive integer (1 is the serial batch)"
        )
    return workers


def _validated_probe_concurrency(workers: int, memory_condition: str) -> int:
    """Refuse a worker count the memory condition cannot take."""
    cap = PROBE_CONCURRENCY_CAP
    if workers > cap:
        # `SystemExit`, not `ValueError`: this is a misuse of a flag, and the
        # operator should read a line, not a traceback. The parser is gone by
        # the time the condition is known, so this takes the same shape as the
        # other post-parse flag refusals in this file (`--pin-steps`).
        raise SystemExit(
            f"--probe-concurrency {workers} is above the cap for "
            f"memory_condition={memory_condition!r}: at most {cap}"
        )
    return workers


def _probe_pa_max_tokens(args: argparse.Namespace, plan: RunPlan, step: Any, run_dir: Path) -> int:
    """The PA output cap for one step: the plan's, or a recorded per-probe raise."""
    overrides = {}
    for item in getattr(args, "pa_max_tokens_for_probe", None) or []:
        step_id, _, tokens = item.partition("=")
        overrides[step_id.strip()] = int(tokens)
    tokens = overrides.get(step.step_id)
    if tokens is None:
        return plan.pa_max_tokens
    if step.kind not in PROBE_KINDS or not plan.pa_max_tokens < tokens <= 3 * plan.pa_max_tokens:
        raise ValueError(f"invalid --pa-max-tokens-for-probe for {step.step_id}: {tokens}")
    record = run_dir / "pa_output_cap_override" / f"{step.step_id}-{os.getpid()}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({
        "schema": "mempa_pa_output_cap_override/1", "field": "pa_max_tokens", "step_id": step.step_id,
        "base_pa_max_tokens": plan.pa_max_tokens, "effective_pa_max_tokens": tokens, "pid": os.getpid(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "rule": "user 2026-09-23: plan cap by default; a probe stuck at it may be retried with a higher cap, at most 3x the plan cap",
    }, indent=2) + "\n")
    return tokens


def _parse_rerun_probe_steps(value: str | None, plan: RunPlan) -> list[str]:
    """`--rerun-probe-steps fin_09,fin_12` -> those ids; `[]` when unset.

    Probe steps only, checked here against the plan: the flag throws a step's
    recorded run away, and doing that to an accumulation step would leave the
    mainline world and the memory it wrote behind a ledger that says the step
    never ran.
    """
    if value is None:
        return []
    step_ids = [part.strip() for part in value.split(",") if part.strip()]
    probes = {step.step_id for step in plan.steps if step.kind in PROBE_KINDS}
    unknown = [step_id for step_id in step_ids if step_id not in probes]
    if not step_ids or unknown:
        raise ValueError(
            f"invalid --rerun-probe-steps value: {value!r} "
            f"(not probe steps of this plan: {unknown or step_ids})"
        )
    return step_ids


def _supersede_probe_steps(
    run_dir: Path, ledger: RunLedger, step_ids: list[str]
) -> list[Path]:
    """Set each probe's finished step directory aside and forget its ledger
    record, so the resume below runs the probe again from its pinned cut point.

    Nothing else moves: the checkpoints the probes branch from, the mainline
    world and the memory pool are all untouched, which is what makes a probe
    rerun safe where an accumulation rerun is not.

    The move itself is `harness.run_layout.supersede_step_dirs`, shared with
    the in-batch void retry (`Orchestrator._run_one_probe`, plan §6b) so both
    write the one layout the cohort manifest reads.
    """
    moved = supersede_step_dirs(run_dir, step_ids)
    for step_id in step_ids:
        ledger.forget(step_id)
    return moved


DEFAULT_ACC_RANGE = "1-10"


def _acc_names(acc_range: str) -> list[str]:
    start_text, end_text = acc_range.split("-", 1)
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError(f"invalid acc range: {acc_range}")
    return [f"acc_{i:03d}" for i in range(start, end + 1)]


def _acc_names_from_list(acc_list: str) -> list[str]:
    """`--acc-list "1-15,17-31,33"` -> acc names, in order, de-duplicated.

    Each comma-separated item is an integer or an inclusive `a-b` range. A
    persona's timeline is not contiguous -- its 99 sessions sit inside
    `acc_001`..`acc_108` with nine probe-slot gaps -- so `--acc-range` alone
    cannot name one. An acc this names but the Outline lacks still fails at
    plan build; nothing here skips a missing session.

    Raises `argparse.ArgumentTypeError` so the reason reaches the operator:
    argparse reports a plain `ValueError` as an unhelpful "invalid value".
    """

    if not acc_list.strip():
        raise argparse.ArgumentTypeError(
            "--acc-list must name at least one acc"
        )
    names: list[str] = []
    seen: set[str] = set()
    for item in acc_list.split(","):
        token = item.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        start = int(match.group(1)) if match else 0
        end = int(match.group(2) or start) if match else 0
        if not match or start < 1 or end < start:
            raise argparse.ArgumentTypeError(
                f"invalid --acc-list item: {token!r} (want N or A-B, 1<=A<=B)"
            )
        for num in range(start, end + 1):
            name = f"acc_{num:03d}"
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


class GracefulStop:
    """SIGTERM / SIGINT: finish the step in flight, then stop (owner ruling
    2026-09-14).

    A mid-run code roll used to mean watching the log and killing the process
    in the gap between two sessions — miss the gap and the step that was
    running is lost, because the ledger keeps it `running` and the next resume
    rolls it back. The first signal now only raises a flag; the step loop reads
    it at a step boundary, once the checkpoint and the ledger record are
    written, so the session in flight is paid for once and kept.

    A second signal is the old behaviour and says so: the default disposition
    is put back and re-raised at once, the step stays `running`, and the resume
    that follows rolls it back.
    """

    def __init__(self, ledger: RunLedger, *, on_stop: Any | None = None) -> None:
        self._ledger = ledger
        self._on_stop = on_stop
        self.requested = False
        #: The probe ids a concurrent batch is waiting on, kept by the batch
        #: runner. `current_step` names whichever record the ledger wrote
        #: last, so with several probes in flight it names one of them at
        #: best; the set names all of them. Empty for the serial batch and
        #: the timeline, where `current_step` is exactly the step in flight.
        self.in_flight: set[str] = set()
        # A setting worker owns the signals when two independent rounds share
        # memory. Each runner registers its live set under its own run identity.
        self.round_in_flight: dict[str, set[str]] = {}

    def __call__(self) -> bool:
        """The step loop's read of the flag."""
        return self.requested

    def _in_flight_label(self) -> str | None:
        if self.round_in_flight:
            return ", ".join(
                f"{run}: {', '.join(sorted(steps)) or 'waiting'}"
                for run, steps in sorted(self.round_in_flight.items())
            )
        if self.in_flight:
            return ", ".join(sorted(self.in_flight))
        return self._ledger.current_step

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self.handle)
        signal.signal(signal.SIGINT, self.handle)

    def handle(self, signum: int, _frame: Any) -> None:
        step = self._in_flight_label()
        if self.requested:
            print(
                f"stop forced: {step} left running, resume will roll back"
                if step
                else "stop forced: no step running",
                flush=True,
            )
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
            return
        self.requested = True
        if self._on_stop is not None:
            self._on_stop()
        print(
            f"stop requested: finishing {step} then exiting"
            if step
            else "stop requested: no step running, exiting",
            flush=True,
        )

    def report(self, result: Any) -> None:
        """Say what the stop kept, so the operator knows what to resume.

        Counts, never "after <step>": a concurrent probe batch appends to
        `result.executed` in completion order, so its last entry is whichever
        probe finished last and not a point the run got past. The ledger is
        what the resume reads, and it holds exactly these records.
        """
        if not self.requested:
            return
        print(
            f"stopped cleanly: {len(result.executed)} step(s) done, "
            f"{len(result.failed)} failed; resume with the same --run-id",
            flush=True,
        )


class MainlinePACheckpointHandle:
    """The PA-side pieces a cut point needs, taken from the mainline PA.

    The PA is built per step by the loop factory, so the entrypoint that owns
    it binds the most recent mainline one here. Probe branches never bind: a
    branch PA lives in a directory that is deleted right after the probe, and a
    cut point must act on the mainline session.

    A forced step has no PA of its own: it writes the same session file through
    a `SessionManager` of its own, and a run that starts or resumes at one
    never binds anything. So the handle is created around a mainline PA over
    the mainline workspace, and `checkpoint_session()` re-reads the file rather
    than trusting whichever cache it is holding (R8).
    """

    def __init__(self, session_key: str, mainline_pa: Any) -> None:
        self.session_key = session_key
        self._pa: Any | None = mainline_pa

    def bind(self, pa: Any) -> None:
        self._pa = pa

    def _require(self) -> Any:
        if self._pa is None:
            raise RuntimeError(
                "No PA has been built yet, so this run has no cut-point handle"
            )
        return self._pa

    @property
    def prompt_history_eviction_policy(self):
        return self._require().prompt_history_eviction_policy

    async def drain_background_tasks(self) -> None:
        await self._require().drain_background_tasks()

    def checkpoint_session(self) -> Any:
        # Every mainline writer persists on write -- the session PA per turn,
        # the forced executor per exchange -- but each holds its own
        # `SessionManager`. Reading the bound PA's cache would hand the cut
        # point a session from before the last forced event and then save it
        # back over the forced dialogue, so re-read the file instead (R8).
        sessions = self._require().sessions
        sessions.invalidate(self.session_key)
        return sessions.get(self.session_key)

    def turn_journal(self) -> Any:
        """The journal beside this session's file, freshly read.

        Same reason `checkpoint_session` re-reads: the forced executor holds
        its own `SessionManager` for the same key and appends its own records
        between this PA's turns.
        """
        journal = turn_journal_for(self._require().sessions, self.session_key)
        journal.reload()
        return journal

    async def archive_messages_to_files(
        self, messages: list[dict[str, object]]
    ) -> bool:
        return await self._require().memory_consolidator.archive_messages_to_files(
            messages
        )

    def save_session(self, session: Any) -> None:
        self._require().sessions.save(session)

    def close_ix_session(self, script_session_id: str) -> list[tuple[str, str]]:
        """Clear the mainline PA's IX state for this step (§4.2, contract 7).

        Unlike the cut-point methods this tolerates an unbound PA: the step
        boundary runs in a `finally`, and a step that died before its PA was
        ever built has no IX state to clear.
        """
        if self._pa is None:
            return []
        return self._pa.close_ix_session(self.session_key, script_session_id)


def _as_draft_beat(beat):
    """One loaded beat in the shape the validator's window helpers read.

    They take an authored draft's beat, whose `user_instruction` the loaded
    script carries as `actor_instruction`; they are duck-typed on `.goal` /
    `.user_instruction`, so this rename is the whole of the adaptation.
    """
    return SimpleNamespace(goal=beat.goal, user_instruction=beat.actor_instruction)


def _declared_cue_beat(script_path: Path) -> str | None:
    """The draft's own `gate_cue_beat`, or None when it declares none.

    Read off the file rather than the loaded script: the authored draft carries
    the field, and `authored_session_to_session_script` builds the runtime
    script by allowlist, so it never reaches this side otherwise. A forced
    (`evolv_*`) draft has no such field and answers None, which is the fallback.
    """
    try:
        data = yaml.safe_load(script_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    value = data.get("gate_cue_beat") if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else None


# The window `gate_cue_absent_before_reveal` accepted the draft on, so the beat
# the cue is armed to is the beat that rule read it in. The helpers below are
# verbatim copies of that rule's own, from the authoring validator
# `validate_normal_event`.
_TIME_NUM_TOKEN = r"[0-9]{1,2}(?::[0-9]{2})?"
_TIME_RANGE_DASH_RE = re.compile(
    rf"({_TIME_NUM_TOKEN})\s*[-–—]\s*({_TIME_NUM_TOKEN})"
)
_TIME_RANGE_TO_RE = re.compile(
    rf"({_TIME_NUM_TOKEN})\s+to\s+({_TIME_NUM_TOKEN})"
)


def _normalised_fact(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip().casefold()
    # A time/number range written with a hyphen, en dash, em dash, or the word
    # "to" is the same range either way: canonicalise all of them to "-" so
    # e.g. "10:00 to 11:50" matches a self_view entry written "10:00–11:50".
    text = _TIME_RANGE_DASH_RE.sub(r"\1-\2", text)
    text = _TIME_RANGE_TO_RE.sub(r"\1-\2", text)
    return text


# The authoring convention for handing over a gated value: the beat refers to
# `task_facts` rather than quoting it ("gives the note text from task_facts when
# the PA is ready to write" — session_gen_system.md).
_TASK_FACTS_REFERENCE_RE = re.compile(r"(?<!\w)task_facts(?!\w)", re.IGNORECASE)


#: One standalone alphabetic word of an already normalised (case-folded) text.
_STANDALONE_WORD_RE = re.compile(r"(?<!\w)[a-z]+(?!\w)")


def _number_folded(text: str) -> str:
    """`_normalised_fact(text)`, a standalone number word read as its digits.

    The store folds them on both sides of its own key-term check, so a record
    that writes "3 weeks" already holds a term of "three weeks" and this rule
    has to see it the same way.
    """
    return _STANDALONE_WORD_RE.sub(
        lambda match: NUMBER_WORDS.get(match.group(0), match.group(0)),
        _normalised_fact(text),
    )


def _beat_instruction_text(beat: Any) -> str:
    """What one beat has the user want and say: its goal and its instruction.

    Narrower than `_beat_text` on purpose. `avoid` and `if_not_met` are what the
    actor must NOT say, and the reveal beat's `avoid` names the withheld fact
    almost word for word ("do not volunteer the sentence about what the review
    should concentrate on"), so counting them would let the ban on speaking a
    fact stand in for the reason to ask about it.
    """
    return " ".join(part for part in (beat.goal, beat.user_instruction) if part)


def _instruction_before_dictation(beat: Any) -> str:
    """The reveal beat's own instruction up to the sentence that hands the fact over.

    Only used when the reveal beat is the session's first beat, which is the one
    shape with no earlier beat to carry the cue (acc_009: the opening beat IS the
    probe beat). The cut is the sentence carrying the first `task_facts`
    reference — the authoring convention for the hand-over — because everything
    from there on describes the withheld fact rather than what the user says
    before the PA has to ask.
    """
    text = beat.user_instruction or ""
    match = _TASK_FACTS_REFERENCE_RE.search(text)
    if match is not None:
        cut = max(text.rfind(mark, 0, match.start()) for mark in ".?!")
        text = text[: cut + 1] if cut >= 0 else ""
    return " ".join(part for part in (beat.goal, text) if part)


def _gate_cue_beat(script, cue: str, reveal_beat: str, declared: str | None = None) -> str | None:
    """The beat on which the actor has to speak the gate's cue, or None.

    `declared` is the draft's own `gate_cue_beat` and wins when it is a beat of
    this script: the author pinned the beat, and `validate_normal_event`
    (`gate_cue_beat_unresolved`) held it to sitting before the reveal. The
    search below is the fallback for a draft that declares nothing — the forced
    line, whose drafts carry no such field.

    Read off the same window `validate_normal_event`'s
    `gate_cue_absent_before_reveal` accepted the draft on, through verbatim
    copies of that rule's own helpers (above): the first beat strictly
    before the reveal whose goal or instruction carries the cue, or the reveal
    beat itself when it is the session's first (acc_009's opening IS the probe
    beat) and its instruction carries the cue ahead of the sentence that hands
    the fact over. None when the cue lives only in the framing prose — the
    scenario and the task are handed to the PA, not said by the actor, so there
    is no beat to hold to.

    A reveal beat the script does not have resolves to None rather than raising:
    that hole is `SessionScript._validate_gate_key_terms`'s to report, one step
    on, and reporting it here would name the wrong field.
    """
    beat_ids = [beat.beat for beat in script.beats]
    if declared is not None and declared in beat_ids:
        return declared
    if reveal_beat not in beat_ids:
        return None
    reveal_index = beat_ids.index(reveal_beat)
    folded_cue = _number_folded(cue)
    for beat in script.beats[:reveal_index]:
        if folded_cue in _number_folded(_beat_instruction_text(_as_draft_beat(beat))):
            return beat.beat
    if reveal_index == 0 and folded_cue in _number_folded(
        _instruction_before_dictation(_as_draft_beat(script.beats[0]))
    ):
        return beat_ids[0]
    return None


def _arm_dictation_check(script, gate, authorised_tool=None, declared_cue_beat=None):
    """The simulator's dictation check runs off the same resolved gate the state
    server enforces, so the two can never disagree: the user's dictation must
    carry every key term the store will look for. No gate (compressed or
    forced session) leaves the check unarmed.

    The cue rides along for the simulator's sake (peer request, 2026-09-08): it
    is the reason the PA has an ask to make, so the actor has to keep saying it
    however the style retry rewrites the turn, and `gate_cue_beat` is the beat
    that owes it. Read like the other three rather than defaulted: a resolved
    gate always carries a cue, because `harness.gate._gate_cue` refuses one
    that does not.

    `authorised_tool` is armed for every session, gated or not: it says which
    tool the user has to release, which is a fact about the task rather than
    about the dictation, and a compressed session answers it with None the same
    way a draft that marks no step does."""
    update = {"authorised_tool": authorised_tool}
    if gate:
        cue = gate["gate_cue"]
        update |= {
            "gate_key_terms": list(gate["key_terms"]),
            "gate_dictated_text": gate["dictated_text"],
            "gate_reveal_beat": gate["reveal_beat"],
            "gate_cue": cue,
            "gate_cue_beat": _gate_cue_beat(
                script, cue, gate["reveal_beat"], declared_cue_beat
            ),
        }
    armed = script.model_copy(update=update)
    # model_copy skips validation; re-validate so a gate naming a beat the
    # script lacks fails here rather than arming nothing.
    return type(script).model_validate(armed.model_dump())


class RealSessionLoop:
    def __init__(
        self,
        *,
        step: RunStep,
        transcript_writer: Any,
        stage_runtime: Any,
        identity: PersonaIdentity,
        matrix: PreferenceMatrix,
        sim_config: SimulatorConfig,
        sim_profile: Any,
        workspace: Path,
        pa_session_key: str,
        benchmark_run_id: str,
        assistant_llm_provider_profile: AssistantLLMProviderProfile,
        pa_model: str,
        pa_max_tokens: int,
        memory_adapter: MemoryAdapter | None,
        consolidation_llm_provider: Any | None,
        consolidation_model: str | None,
        memory_io_recorder: MemoryIORecorder,
        memory_retrieval_max_tokens: int,
        memory_retrieval_max_items: int,
        simulator_runtime: SimulatorRuntimeConfig | None = None,
        preference_overlay: RunPreferenceOverlay | None = None,
        pa_binder: Any | None = None,
        skip_consolidation: bool = False,
    ) -> None:
        self.skip_consolidation = skip_consolidation
        self.pa_binder = pa_binder
        self.step = step
        self.transcript_writer = transcript_writer
        self.stage_runtime = stage_runtime
        self.identity = identity
        self.matrix = matrix
        self.sim_config = sim_config
        self.sim_profile = sim_profile
        self.workspace = workspace
        self.pa_session_key = pa_session_key
        self.benchmark_run_id = benchmark_run_id
        self.assistant_llm_provider_profile = assistant_llm_provider_profile
        self.pa_model = pa_model
        self.pa_max_tokens = pa_max_tokens
        self.memory_adapter = memory_adapter
        self.consolidation_llm_provider = consolidation_llm_provider
        self.consolidation_model = consolidation_model
        self.memory_io_recorder = memory_io_recorder
        self.memory_retrieval_max_tokens = memory_retrieval_max_tokens
        self.memory_retrieval_max_items = memory_retrieval_max_items
        self.simulator_runtime = simulator_runtime or SimulatorRuntimeConfig()
        self.preference_overlay = preference_overlay

    async def run_session(self) -> list[dict]:
        script = run_plan_builder._load_script(
            PROJECT_ROOT / self.step.script_path
        )
        # Cells a forced event already flipped stay flipped for the rest of the
        # run, whatever this session's own target attribute is.
        overrides = (
            self.preference_overlay.preference_overrides()
            if self.preference_overlay is not None
            else {}
        )
        if overrides:
            # model_copy skips validation, so the overlay's plain dicts have to
            # be built into the `PreferencePatch` the field declares; a raw dict
            # here makes every later `model_dump` warn.
            script = script.model_copy(
                update={
                    "preference_overrides": {
                        context: {
                            attribute: PreferencePatch(**patch)
                            for attribute, patch in patches.items()
                        }
                        for context, patches in overrides.items()
                    }
                }
            )
        from harness.capability_attempt import prepare
        capability_limit_attempts = prepare(
            PROJECT_ROOT / self.step.script_path, self.stage_runtime.state_root,
            session_date=self.stage_runtime.session_date,
            session_timestamp=self.stage_runtime.session_timestamp,
        )
        script = _arm_dictation_check(
            script,
            getattr(self.stage_runtime, "gate", None),
            resolve_authorised_tool(
                PROJECT_ROOT / self.step.script_path, PROJECT_ROOT,
                capability_limit_attempts=capability_limit_attempts,
            ),
            _declared_cue_beat(PROJECT_ROOT / self.step.script_path),
        )
        # The cue beat is pinned above, but the simulator can skip it after a
        # turn limit; this moves the obligation onto the beat it jumps to. The
        # writer is opened by the executor, before this script exists, which is
        # why the hook joins it here rather than at construction.
        cue_hook = gate_cue_carry_hook(script)
        if cue_hook is not None:
            self.transcript_writer.add_event_hook(cue_hook)
        sim_provider = self.sim_config.create_provider()
        sim_backend = self.sim_config.create_backend(
            sim_provider,
            profile=self.sim_profile,
            output_contract_retries=(
                self.simulator_runtime.output_contract_retries
            ),
        )
        self_reporter = SessionEndSelfReporter(
            sim_provider,
            profile_context=self.sim_profile.build_context(),
            model=self.simulator_runtime.simulator_model,
            output_contract_retries=(
                self.simulator_runtime.self_report_contract_retries
            ),
        )
        pa_inner = build_pa_from_provider_profile(
            provider_profile=self.assistant_llm_provider_profile,
            workspace=self.workspace,
            pa_model=self.pa_model,
            pa_max_tokens=self.pa_max_tokens,
            mcp_servers=self.stage_runtime.mcp_server_config(),
            memory_adapter=self.memory_adapter,
            consolidation_llm_provider=self.consolidation_llm_provider,
            consolidation_model=self.consolidation_model,
            require_committed_adapter_writes=(
                _requires_committed_adapter_writes(self.step)
            ),
            memory_io_recorder=self.memory_io_recorder,
            skip_consolidation=self.skip_consolidation,
        )
        # `providers.probe_phase` in configs/run_config.yaml: OpenRouter
        # provider routing for probe steps only (None keeps the model's own).
        probe_routing = load_run_config().providers.probe_phase
        if probe_routing and self.step.kind in PROBE_KINDS:
            pa_llm = pa_inner.pa_llm_provider
            pa_llm.extra_body = {**(getattr(pa_llm, "extra_body", None) or {}), "provider": probe_routing}
        if self.pa_binder is not None:
            self.pa_binder(pa_inner)
        pa_recorder = RecordingPA(pa_inner)
        # The executor reaches the per-call `ix_render` records through this:
        # the PA is built here, but the transcript is assembled after the
        # session ends (§4.4 merge site in harness/session_executor.py).
        self.pa_recorder = pa_recorder
        try:
            loop = SimulatorLoop(
                provider=sim_provider,
                backend=sim_backend,
                identity=self.identity,
                matrix=self.matrix,
                script=script,
                pa_loop=pa_recorder,
                capability_limit_attempts=capability_limit_attempts,
                transcript_writer=self.transcript_writer,
                model=self.sim_config.model,
                pa_model=self.pa_model,
                session_key=self.pa_session_key,
                run_id=self.benchmark_run_id,
                persona_id=self.step.persona_id,
                agent_id="nanobot",
                benchmark_session_timestamp=(
                    self.step.benchmark_session_timestamp
                ),
                benchmark_session_deadline=(
                    self.step.benchmark_session_timestamp
                    + timedelta(minutes=SESSION_ENVELOPE_MINUTES)
                ),
                suppress_resume_delay=timedelta(
                    minutes=SUPPRESS_RESUME_DELAY_MINUTES
                ),
                memory_retrieval_max_tokens=(
                    self.memory_retrieval_max_tokens
                ),
                memory_retrieval_max_items=self.memory_retrieval_max_items,
                # Contract 9: the formula and its constants live once, in
                # pa/agent/ix/render_types.py. simulator/ never imports pa/, so
                # the harness injects the callable here, at construction.
                turn_budget_fn=turn_budget,
                self_reporter=self_reporter,
            )
            return await loop.run_session()
        finally:
            try:
                await sim_backend.drain_memory_tasks()
            except MemoryConsolidationError as exc:
                # Raising here would mask the session's own exception, and a
                # lost memory write must not kill the run: log and continue.
                # The message carries the attempt count, the finish reason of
                # each attempt and the budget history, so a read of the run log
                # can tell straight away whether the output cap was the cause.
                print(
                    f"[memory] simulator memory write FAILED: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                await pa_inner.close_mcp()


def _resolve_assistant_llm_provider_profile(
    args: argparse.Namespace,
) -> AssistantLLMProviderProfile:
    from pa.providers.registry import find_by_name

    provider_name = ASSISTANT_LLM_PROVIDER_NAME
    spec = find_by_name(provider_name)
    if spec is None:
        raise ValueError(
            f"Unknown assistant LLM provider: value={provider_name!r}"
        )
    api_key = os.environ.get(spec.env_key) if spec.env_key else None
    if not api_key:
        raise ValueError(
            "Assistant LLM provider credential is missing: "
            f"provider_name={provider_name!r} env_key={spec.env_key!r}"
        )
    return AssistantLLMProviderProfile(
        provider_name=provider_name,
        api_key=api_key,
        api_base=run_plan_builder.OPENROUTER_API_BASE,
    )


def _required_environment_value(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(
            f"Required environment variable is unset: name={name!r}"
        )
    return value


def _checkpoint_retention_interval(value: str) -> int:
    try:
        interval = int(value)
    except ValueError:
        interval = 0
    if interval < 1:
        raise argparse.ArgumentTypeError("checkpoint retention interval must be positive")
    return interval


def _retention_mainline_ids(plan: RunPlan) -> frozenset[str]:
    """The frozen scope for counting successfully published mainline steps."""
    return frozenset(
        step.step_id for step in plan.steps
        if step.kind in {"accumulation", "forced_event"} and not step.placeholder
    )


def _resolve_extra_pins(
    plan: RunPlan,
    *,
    pin_steps: str | None,
    pin_all_accumulations: bool,
) -> frozenset[str]:
    """Cut points to pin beyond the ones this plan's own probes imply.

    Owner ruling 2026-09-13: a run may execute its accumulation timeline now
    and have a probe plan attached later, off the same checkpoints. Nothing in
    a probeless plan says which cut points that later plan will want, so the
    operator names them. Only accumulation steps qualify, for the same reason
    `RunPlan.pinned_step_ids` pins only those: no consolidation pass runs after
    a forced event, so an event's snapshot is not a complete memory state.
    """
    accumulations = {
        step.step_id for step in plan.steps if step.kind == "accumulation"
    }
    requested: set[str] = set(accumulations) if pin_all_accumulations else set()
    for raw in (pin_steps or "").split(","):
        step_id = raw.strip()
        if step_id:
            requested.add(step_id)
    unknown = sorted(requested - accumulations)
    if unknown:
        raise SystemExit(
            "--pin-steps names step ids that are not accumulation steps of "
            f"this plan: unknown={unknown!r} known={sorted(accumulations)!r}"
        )
    return frozenset(requested)


async def _execute_benchmark_run(args: argparse.Namespace, *, shared_probe_memory_owner: Any | None = None,
                                 shared_probe_stop: GracefulStop | None = None) -> int:
    if shared_probe_memory_owner is not None and (not args.plan or args.overwrite or args.regenerate_plan):
        raise ValueError("Shared Probe repeats require an existing frozen plan and cannot overwrite it")
    repeat_source = None
    probe_source_ledger = None
    if shared_probe_memory_owner is not None:
        from harness.probe_repeat_source import load_repeat_source
        repeat_source = load_repeat_source(Path(args.plan).parent)
        if repeat_source is None:
            raise ValueError("Shared Probe repeat requires a probe_repeat_source.json")
        source_plan = yaml.safe_load((Path(repeat_source["source_run_dir"]) / "run_plan.yaml").read_text())
        probe_source_ledger = RunLedger(Path(repeat_source["source_run_dir"]), source_plan["run_id"])
    if not args.verbose:
        logger.remove()

    forced_ix = _parse_force_ix(getattr(args, "force_ix", None))
    only_steps = _parse_only_steps(getattr(args, "only_steps", None))
    probe_concurrency = _validated_probe_concurrency(
        getattr(args, "probe_concurrency", 1), args.memory_condition
    )

    effective_consolidation_model = _effective_consolidation_model(
        args.memory_condition,
        args.consolidation_model,
    )
    effective_memory_side_model = _effective_memory_side_model(
        args.memory_condition,
        args.memory_side_model,
    )
    effective_assistant_llm_api_base = run_plan_builder.OPENROUTER_API_BASE
    effective_mem0_runtime_config = _effective_mem0_runtime_config(
        args.memory_condition
    )
    effective_memos_runtime_config = _effective_memos_runtime_config(
        memory_condition=args.memory_condition,
        api_base=args.memos_api_base,
        memory_side_provider_name=ASSISTANT_LLM_PROVIDER_NAME,
        memory_side_model=effective_memory_side_model,
        memory_side_api_base=effective_assistant_llm_api_base,
    )

    resuming_existing_run = False
    pending_overwrite = False
    plan_needs_write = False
    if args.plan:
        plan_path = Path(args.plan)
        plan = load_run_plan(plan_path, project_root=PROJECT_ROOT)
        run_dir = plan_path.parent
        resuming_existing_run = True
    else:
        script_names = (
            args.acc_list
            if args.acc_list
            else _acc_names(args.acc_range or DEFAULT_ACC_RANGE)
        )
        qdrant_host, qdrant_port = _qdrant_endpoint_from_environment(
            args.memory_condition
        )
        run_id = args.run_id or _new_run_id()
        run_dir = _run_dir(
            pa_model=args.pa_model,
            memory_condition=args.memory_condition,
            persona_id=args.persona_id,
            run_id=run_id,
        )
        pending_overwrite = bool(args.overwrite and run_dir.exists())
        plan_path = run_dir / "run_plan.yaml"
        if (
            plan_path.exists()
            and not args.regenerate_plan
            and not pending_overwrite
        ):
            plan = load_run_plan(plan_path, project_root=PROJECT_ROOT)
            resuming_existing_run = True
        else:
            existing_plan, done_step_ids = _finished_plan_carry_over(
                # `--overwrite` deletes the directory further down, so nothing
                # of the old run survives for the new plan to carry.
                enabled=bool(args.regenerate_plan) and not pending_overwrite,
                plan_path=plan_path,
                run_dir=run_dir,
                run_id=run_id,
            )
            plan = run_plan_builder._build_frozen_run_plan(
                run_id=run_id,
                persona_id=args.persona_id,
                memory_condition=args.memory_condition,
                pa_model=args.pa_model,
                script_names=script_names,
                # A persona with a probe line freezes into the plan every
                # probe whose cut point lies in the requested accs (§2.4.1):
                # all of them for the full timeline. There is no flag to
                # leave them out.
                with_probes=run_plan_builder.persona_has_probe_line(
                    args.persona_id
                ),
                assistant_llm_provider_name=ASSISTANT_LLM_PROVIDER_NAME,
                assistant_llm_api_base=effective_assistant_llm_api_base,
                consolidation_model=effective_consolidation_model,
                memory_side_model=effective_memory_side_model,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                mem0_runtime_config=effective_mem0_runtime_config,
                memos_runtime_config=effective_memos_runtime_config,
                simulator_runtime=SimulatorRuntimeConfig(
                    simulator_model=getattr(args, "simulator_model", DEFAULT_SIMULATOR_MODEL)
                ),
                existing_plan=existing_plan,
                done_step_ids=done_step_ids,
            )
            plan_needs_write = True

    if plan.persona_id != args.persona_id:
        raise ValueError(f"plan persona_id={plan.persona_id!r} does not match CLI persona_id={args.persona_id!r}")
    if plan.memory_condition != args.memory_condition:
        raise ValueError(
            f"plan memory_condition={plan.memory_condition!r} does not match "
            f"CLI memory_condition={args.memory_condition!r}"
        )
    if plan.pa_model and plan.pa_model != args.pa_model:
        raise ValueError(f"plan pa_model={plan.pa_model!r} does not match CLI pa_model={args.pa_model!r}")
    requested_simulator_model = getattr(args, "simulator_model", DEFAULT_SIMULATOR_MODEL)
    if plan.simulator_runtime.simulator_model != requested_simulator_model:
        raise ValueError(
            f"plan simulator_model={plan.simulator_runtime.simulator_model!r} does not "
            f"match CLI --simulator-model={requested_simulator_model!r}"
        )
    if plan.consolidation_model != effective_consolidation_model:
        raise ValueError(
            f"plan consolidation_model={plan.consolidation_model!r} does not match "
            "effective CLI consolidation_model="
            f"{effective_consolidation_model!r}"
        )
    if plan.memory_side_model != effective_memory_side_model:
        raise ValueError(
            "plan memory_side_model does not match effective CLI value: "
            f"plan={plan.memory_side_model!r} "
            f"effective_cli={effective_memory_side_model!r}"
        )
    # A resumed run runs on its plan's frozen runtime configuration
    # (`plan.*_runtime_config`); today's defaults apply to new plans only.
    if (
        plan.assistant_llm_provider_name
        != ASSISTANT_LLM_PROVIDER_NAME
    ):
        raise ValueError(
            "plan assistant_llm_provider_name does not match CLI: "
            f"plan={plan.assistant_llm_provider_name!r} "
            f"required={ASSISTANT_LLM_PROVIDER_NAME!r}"
        )
    if plan.assistant_llm_api_base != effective_assistant_llm_api_base:
        raise ValueError(
            "plan assistant_llm_api_base does not match effective CLI value: "
            f"plan={plan.assistant_llm_api_base!r} "
            f"effective_cli={effective_assistant_llm_api_base!r}"
        )
    current_pa_policy = resolve_model_request_policy(
        plan.assistant_llm_provider_name,
        plan.pa_model,
    )
    current_pa_policy.require_exact_role("pa")
    if plan.consolidation_model:
        current_consolidation_policy = resolve_model_request_policy(
            plan.assistant_llm_provider_name,
            plan.consolidation_model,
        )
        current_consolidation_policy.require_exact_role("consolidation")
    if plan.memory_side_model:
        current_memory_side_policy = resolve_model_request_policy(
            plan.assistant_llm_provider_name,
            plan.memory_side_model,
        )
        current_memory_side_policy.require_exact_role("memory_extraction")

    assistant_llm_provider_profile = _resolve_assistant_llm_provider_profile(
        args
    )
    if args.memory_condition in EMBEDDING_MEMORY_CONDITIONS:
        _required_environment_value("OPENAI_EMB_API_KEY")

    rerun_probe_steps = _parse_rerun_probe_steps(
        getattr(args, "rerun_probe_steps", None), plan
    )

    # Before the run: with the local judge armed (`MEMPA_JUDGE=live`) a round
    # that starts without the endpoint answering on the pinned model would
    # record its sessions against another standard. No fallback model, so a
    # failure here refuses the round rather than degrading it (local judge
    # contract §6).
    from simulator.judge.session import preflight as judge_preflight

    judge_preflight()

    if pending_overwrite:
        shutil.rmtree(run_dir)
        resuming_existing_run = False
    run_dir.mkdir(parents=True, exist_ok=True)
    if plan_needs_write:
        plan.dump_yaml(plan_path)
        if plan.resolved_gt is not None:
            run_plan_builder.freeze_resolved_gt_copy(plan, run_dir)
        write_run_info(
            plan, run_dir, PROJECT_ROOT, event="freeze",
            extra=plan_scope_record(plan),
        )

    # One avoidance state per run directory, with the plan's threshold: the PA
    # of every step and probe branch, File memory consolidation and Mem0
    # extraction all count into it. Loaded here, so an invalid state file
    # stops the launch first. A plan frozen without the threshold runs
    # without it.
    if plan.provider_avoid_after_failures is not None:
        assistant_llm_provider_profile = replace(
            assistant_llm_provider_profile,
            provider_avoidance=ProviderAvoidance(
                run_dir, threshold=plan.provider_avoid_after_failures
            ),
        )

    run_ledger = RunLedger(run_dir, plan.run_id)
    # Read before anything below draws a conclusion from this ledger. A
    # rollback replaces `ledger.json` from the snapshot, so from that moment
    # until the recovery finishes the ledger reads as a clean, resumable run
    # while the layers around it are still half rolled back -- every pre-flight
    # that asks the ledger "what state is this run in" would be answering about
    # that half state. What the run is actually in is established first, and
    # `resume_or_restore` below finishes the recovery on the recorded target
    # before a step runs.
    pending_recovery = read_pending_recovery(run_dir)
    if pending_recovery is not None:
        print(
            "  rollback recovery unfinished: target "
            f"{pending_recovery['target_checkpoint_id']} "
            f"(recovery {pending_recovery.get('recovery_id')}); finishing it "
            "before this run does anything else"
        )
    # A run may execute its accumulation timeline now and have a probe plan
    # attached later, off the same checkpoints (§2.4): `--regenerate-plan`
    # rebuilds and re-freezes the plan on a directory whose ledger already
    # records steps. That is a resume of that run, not a new one -- the Qdrant
    # collection and history db those steps wrote are expected to be there --
    # so the ledger decides it, the same predicate the two `require_existing_*`
    # arguments below already use. `--overwrite` deleted the directory above,
    # so its fresh ledger is empty and its meaning is unchanged.
    if run_ledger.steps:
        resuming_existing_run = True
    for superseded in _supersede_probe_steps(
        run_dir, run_ledger, rerun_probe_steps
    ):
        print(f"  superseded {superseded.relative_to(run_dir)}")
    checkpoint_manager = CheckpointManager(
        run_dir,
        ledger=run_ledger,
        memory_condition=plan.memory_condition,
        rolling_window=plan.checkpoint_rolling_window,
        retention_interval=getattr(args, "retain_checkpoints_every", None),
        retention_mainline_ids=_retention_mainline_ids(plan),
    )
    # With no snapshot there is nothing to roll back to, and the refusal is
    # final. It is made here, before any backend client exists, so a run that
    # cannot be resumed never opens an external backend. The rollback itself
    # needs the client and so waits until the client is bound, below.
    #
    # An unfinished recovery already names the snapshot this run is going back
    # to, so there is nothing for this to decide: the target exists by
    # construction, and the ledger it would judge is the one mid-replacement.
    if shared_probe_memory_owner is None and pending_recovery is None and not checkpoint_manager.list_checkpoints():
        run_ledger.assert_resumable(plan)
    # An attach — a plan (re)frozen onto a ledger that already holds records —
    # is checked whatever the checkpoints say. The refusal above is skipped as
    # soon as a checkpoint exists, and for an attach that silently turns into a
    # rollback: the mainline is rewound and the probes then branch cut points
    # the timeline never finished making.
    if plan_needs_write and run_ledger.steps:
        # Not on a run that is half rolled back: this judges whether the
        # timeline finished, and the ledger it would read is the snapshot's,
        # mid-recovery. Re-freezing a plan onto that is a configuration change
        # made against a state that does not exist yet.
        if pending_recovery is not None:
            raise UnsafeRunResumeError(
                "Refusing to (re)freeze a plan onto a run whose rollback "
                "recovery is unfinished, because this run's live state is "
                "part snapshot and part crash: run_id="
                f"{plan.run_id!r} target_checkpoint_id="
                f"{pending_recovery['target_checkpoint_id']!r}. Resume the run "
                "without a plan-refreezing flag to finish the recovery first."
            )
        run_ledger.assert_probe_attach_ready(plan)

    memory_adapter = None
    memory_backend_namespace = None
    memos_health = None
    if args.memory_condition:
        memory_adapter_options: dict[str, Any] = {}
        # Named once here so a probe branch's own client is the same client on
        # another namespace, never a second spelling of these arguments. Each
        # stays None for every condition but the one that fills it.
        mem0_client_kwargs: dict[str, Any] | None = None
        mem0_llm_kwargs: dict[str, Any] | None = None
        if args.memory_condition in {MEM0_CONDITION, "simple_rag"}:
            memory_backend_namespace = _mem0_collection_name(plan.run_id)
            mem0_config = plan.mem0_runtime_config
            if mem0_config is None:
                raise ValueError(
                    "Mem0-backed RunPlan is missing mem0_runtime_config: "
                    f"run_id={plan.run_id!r}"
                )
            history_db_path = (
                run_dir / mem0_config.history_db_relative_path
            ).resolve()
            # Named once so a probe branch's own client is the same client on
            # another collection, never a second spelling of these arguments.
            mem0_client_kwargs = dict(
                collection_name=memory_backend_namespace,
                qdrant_host=plan.qdrant_host,
                qdrant_port=plan.qdrant_port,
                expected_run_id=plan.run_id,
                allow_existing_collection=resuming_existing_run,
                # `--overwrite` has already deleted this run id's directory;
                # the collection it left in Qdrant would otherwise refuse the
                # retry under the same run id.
                drop_existing_collection=pending_overwrite,
                embedding_api_base=mem0_config.embedding_api_base,
                embedding_provider=plan.embedding_provider,
                embedding_model=plan.embedding_model,
                embedding_dimensions=plan.embedding_dimensions,
                qdrant_distance=mem0_config.qdrant_distance,
                qdrant_on_disk=mem0_config.qdrant_on_disk,
                qdrant_bm25_enabled=mem0_config.qdrant_bm25_enabled,
                enable_entity_memory=mem0_config.enable_entity_memory,
                history_db_path=history_db_path,
                allow_existing_history_db=resuming_existing_run,
                # The client that performs a rollback must not require the live
                # state the rollback exists to rebuild. A recovery interrupted
                # inside the backend layer can leave the history database not
                # yet copied back from the snapshot, and demanding it here would
                # refuse to build the very client that would restore it.
                require_existing_history_db=(
                    bool(run_ledger.steps) and pending_recovery is None
                ),
            )
            mem0_llm_kwargs = dict(
                provider_name=assistant_llm_provider_profile.provider_name,
                api_key=assistant_llm_provider_profile.api_key,
                api_base=(
                    assistant_llm_provider_profile.api_base or ""
                ),
                memory_side_model=plan.memory_side_model,
                extraction_max_output_tokens=(
                    mem0_config.extraction_max_output_tokens
                ),
                provider_avoidance=assistant_llm_provider_profile.provider_avoidance,
            )
        if args.memory_condition == MEM0_CONDITION:
            # The exact same kwargs create mainline and independent Probe clients.
            memory_adapter_options["extraction_chunk_strategy"] = (
                mem0_config.extraction_chunk_strategy
            )
            memory_adapter_options["client"] = (
                create_mem0_client(**mem0_llm_kwargs, **mem0_client_kwargs)
                if shared_probe_memory_owner is None else None
            )
        if args.memory_condition == "simple_rag":
            memory_adapter_options["client"] = (
                create_simple_rag_mem0_client(**mem0_client_kwargs)
                if shared_probe_memory_owner is None else None
            )
        if shared_probe_memory_owner is not None and args.memory_condition in {MEM0_CONDITION, "simple_rag"}:
            from types import SimpleNamespace
            memory_adapter_options["client"] = SimpleNamespace(
                _qdrant_host=plan.qdrant_host, _qdrant_port=plan.qdrant_port
            )
        if args.memory_condition == MEMOS_CONDITION:
            memos_config = plan.memos_runtime_config
            if memos_config is None:
                raise ValueError(
                    "memos RunPlan is missing memos_runtime_config: "
                    f"run_id={plan.run_id!r}"
                )
            client = MemOSHttpClient(
                api_base=memos_config.api_base,
                timeout_seconds=memos_config.api_timeout_seconds,
            )
            memos_health = memos_health_record(client.health())
            # Into the options here rather than with the rest of the MemOS
            # settings below, because the binding under this block reads the
            # client back out of them: a rollback has to find it too.
            memory_adapter_options["client"] = client
            memory_backend_namespace = plan.memos_cube_id
    checkpoint_manager.memory_backend_namespace = memory_backend_namespace
    # External backends keep their memory outside the run directory, so the
    # checkpoint needs the live client to snapshot, restore and branch it. A
    # rollback is a restore, so the client is bound before the resume decision
    # rather than after it: a run stopped mid-step on an external backend has
    # to be able to go back to its last snapshot exactly like a file-backed one.
    checkpoint_manager.backend_client = (
        memory_adapter_options.get("client")
        if args.memory_condition
        else None
    )
    # A run that died mid-step rolls back to its last snapshot instead of
    # being refused outright; with no snapshot the refusal stands.
    if shared_probe_memory_owner is None:
        resume_or_restore(run_ledger, checkpoint_manager, plan)

    consolidation_llm_provider = None
    if args.memory_condition == FILE_MEMORY_CONDITION:
        consolidation_llm_provider = (
            assistant_llm_provider_profile.create_provider(
                model=plan.consolidation_model,
                role="consolidation",
            )
        )

    memory_io_recorder = MemoryIORecorder(
        run_dir / "memory_io.jsonl",
        run_id=plan.run_id,
    )

    outline_path = (
        PROJECT_ROOT
        / "Authoring/outline/drafts"
        / plan.persona_id
        / "outline.yaml"
    )

    workspace = run_dir / "pa_workspace" / plan.persona_id
    workspace.mkdir(parents=True, exist_ok=True)
    soul_path = workspace / "SOUL.md"
    if not soul_path.exists():
        soul_path.write_text(SOUL_TEXT, encoding="utf-8")
    os.environ[PROMPT_WORKSPACE_ENV] = f"/workspace/{plan.persona_id}"

    # Backend state is checked against the committed audit only once the
    # rollback above has settled it, so a resumed run validates what it will
    # actually run on.
    if args.memory_condition:
        if args.memory_condition == FILE_MEMORY_CONDITION:
            memory_dir = workspace / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            memory_adapter_options["memory_dir"] = memory_dir
        if args.memory_condition in {MEM0_CONDITION, "simple_rag"}:
            if shared_probe_memory_owner is None:
                _validate_mem0_backend_state(
                    client=memory_adapter_options["client"], plan=plan,
                    memory_io_recorder=memory_io_recorder, run_ledger=run_ledger,
                )
            memory_adapter_options.update(
                {
                    "top_k": plan.memory_retrieval_max_items,
                    "threshold": mem0_config.retrieval_similarity_threshold,
                    "rerank": mem0_config.retrieval_rerank,
                }
            )
        if args.memory_condition == MEMOS_CONDITION:
            if shared_probe_memory_owner is None:
                _validate_memos_backend_state(
                    client=client, plan=plan,
                    memory_io_recorder=memory_io_recorder, run_ledger=run_ledger,
                )
            memory_adapter_options.update(
                {
                    "cube_id": memory_backend_namespace,
                    "memory_session_id": plan.main_pa_session_key(),
                    "add_mode": memos_config.add_mode,
                    "search_mode": memos_config.search_mode,
                    "top_k": memos_config.top_k,
                    "relativity": memos_config.relativity,
                    "dedup": memos_config.dedup,
                    # The container's extraction LLM cannot print an
                    # `openrouter_call` line from inside its own process.
                    "usage_reporter": _memos_usage_reporter(plan.run_id),
                }
            )
        memory_adapter = default_registry().create(
            args.memory_condition, **memory_adapter_options,
        ) if shared_probe_memory_owner is None else None

    sim_config = SimulatorConfig(
        model=plan.simulator_runtime.simulator_model,
    )
    sim_profile_root = _prepare_simulator_workspace(persona_id=plan.persona_id, run_dir=run_dir)
    sim_profile = load_simulator_profile(sim_profile_root)
    # The run-local copies are the single source: the loop and the profile
    # render from the same files, and a resume keeps the persona it began with.
    identity = sim_profile.load_identity()
    matrix = sim_profile.load_matrix()

    # Run-level simulator state written by forced events: the preference flip
    # and the event's own HISTORY.md line. Both are independent of
    # memory_condition; the simulated user is not the system under test.
    preference_overlay = RunPreferenceOverlay()

    def _branch_memory_adapter(namespace: str, branch_dir: Path) -> Any:
        """The adapter a probe reads the branch's restored namespace through.

        Returns an awaitable, and the orchestrator awaits whatever the factory
        hands back. The orchestrator sets the adapter read-only before the
        probe PA is built, so a probe reads the cut point and writes nothing.
        """
        return _build_branch_memory_adapter(
            memory_condition=args.memory_condition,
            namespace=namespace,
            branch_dir=branch_dir,
            mainline_options=memory_adapter_options,
            mem0_client_kwargs=mem0_client_kwargs,
            mem0_llm_kwargs=mem0_llm_kwargs,
        )

    if shared_probe_memory_owner is not None:
        from harness.shared_probe_memory import SharedQueryResource
        async def shared_query_factory(namespace: str, directory: Path, source_run_id: str):
            options = dict(memory_adapter_options)
            if args.memory_condition in {MEM0_CONDITION, "simple_rag"}:
                from memory.backends.mem0_client import create_read_only_mem0_client
                kwargs = {**(mem0_llm_kwargs or {}), **(mem0_client_kwargs or {}),
                          **_mem0_branch_client_kwargs(namespace, directory),
                          "expected_run_id": source_run_id,
                          "allow_missing_entity_collection": (
                              json.loads((directory / "checkpoint_metadata.json").read_text()).get("mem0_entity_snapshot_ref", "missing") is None)}
                query_client = await shared_probe_memory_owner._thread(
                    create_read_only_mem0_client, infer=args.memory_condition == MEM0_CONDITION, **kwargs)
                options["client"] = query_client
                options["retrieval_run_id"] = source_run_id
            elif args.memory_condition == MEMOS_CONDITION:
                query_client = memory_adapter_options["client"].read_only_for_cube(namespace)
                options.update(client=query_client, cube_id=namespace, usage_reporter=None)
            else:
                raise ValueError(f"No shared query factory for {args.memory_condition}")
            return SharedQueryResource(query_client,
                lambda: default_registry().create(args.memory_condition, **options))
        checkpoint_manager.shared_query_factory = shared_query_factory
        checkpoint_manager.shared_query_identity = json.dumps({
            "condition": plan.memory_condition,
            "mem0": plan.mem0_runtime_config.model_dump(mode="json") if plan.mem0_runtime_config else None,
            "memos": plan.memos_runtime_config.model_dump(mode="json") if plan.memos_runtime_config else None,
            "max_items": plan.memory_retrieval_max_items,
            "max_tokens": plan.memory_retrieval_max_tokens,
            "session_key": plan.main_pa_session_key(),
        }, sort_keys=True)

    # File-backed conditions keep every byte inside the branch directory, so
    # the branch PA's own adapter is enough and nothing is rebuilt for them.
    if args.memory_condition and args.memory_condition not in LOCAL_MEMORY_CONDITIONS:
        checkpoint_manager.branch_adapter_factory = _branch_memory_adapter
    # The cut point needs a mainline consolidator and session even before the
    # first session PA exists: a run that starts or resumes at a forced step
    # binds nothing, and the forced executor is not a PA. Each session step
    # rebinds its own PA over this same workspace.
    pa_checkpoint = MainlinePACheckpointHandle(
        plan.main_pa_session_key(),
        build_pa_from_provider_profile(
            provider_profile=assistant_llm_provider_profile,
            workspace=workspace,
            pa_model=args.pa_model,
            pa_max_tokens=plan.pa_max_tokens,
            memory_adapter=memory_adapter,
            consolidation_llm_provider=consolidation_llm_provider,
            consolidation_model=plan.consolidation_model or None,
            require_committed_adapter_writes=True,
            memory_io_recorder=memory_io_recorder,
        ),
    )
    sim_memory_store = UserMemoryStore(
        sim_profile.root,
        user_name=sim_profile.display_name,
    )

    def loop_factory(
        step: RunStep,
        writer: Any,
        stage_runtime: Any,
        *,
        workspace: Path = workspace,
        session_manager: SessionManager | None = None,
        memory_adapter: MemoryAdapter | None = memory_adapter,
        memory_backend_namespace: str | None = memory_backend_namespace,
        memory_io_recorder: MemoryIORecorder = memory_io_recorder,
        skip_consolidation: bool = False,
        preference_overlay: RunPreferenceOverlay = preference_overlay,
        sim_profile: Any = sim_profile,
        pa_binder: Any | None = None,
    ) -> Any:
        # A probe branch hands in its own workspace, adapter and recorder; a
        # mainline step passes nothing and the run-level values above apply.
        # The overlay and the simulator profile are run-level either way — the
        # branch names them so its own artifacts record the identity it spoke
        # with rather than inheriting it silently from this closure.
        session_manager = session_manager or SessionManager(workspace)
        is_probe = step.kind in {"pre_event_probe", "final_probe"}
        if step.kind == "forced_event":
            return ForcedEventExecutor(
                transcript_path=PROJECT_ROOT / step.script_path,
                project_root=PROJECT_ROOT,
                benchmark_session_timestamp=step.benchmark_session_timestamp,
                stage_runtime=stage_runtime,
                session_manager=session_manager,
                pa_session_key=plan.pa_session_key_for_step(step),
                memory_adapter=memory_adapter,
                memory_io_recorder=memory_io_recorder,
                writer=writer,
                require_committed_adapter_writes=True,
                preference_overlay=preference_overlay,
                history_appender=sim_memory_store.append_history,
            )

        def bind_pa(pa: Any) -> None:
            pa_checkpoint.bind(pa)
            # Validation-only: pre-select a setting the persona matrix never
            # produces, so §7.3 rows 1b/5 can be run at all.
            for attribute, setting in (forced_ix.get(step.step_id) or {}).items():
                seed_ix_selection(
                    pa,
                    session_key=plan.pa_session_key_for_step(step),
                    script_session_id=step.step_id,
                    attribute=attribute,
                    setting=setting,
                )
                print(
                    f"forced_ix step={step.step_id} "
                    f"attribute={attribute} setting={setting}"
                )

        return RealSessionLoop(
            step=step,
            transcript_writer=writer,
            stage_runtime=stage_runtime,
            identity=identity,
            matrix=matrix,
            sim_config=sim_config,
            sim_profile=sim_profile,
            workspace=workspace,
            pa_session_key=plan.pa_session_key_for_step(step),
            benchmark_run_id=plan.run_id,
            assistant_llm_provider_profile=(
                assistant_llm_provider_profile
            ),
            pa_model=args.pa_model,
            pa_max_tokens=_probe_pa_max_tokens(args, plan, step, run_dir),
            memory_adapter=memory_adapter,
            consolidation_llm_provider=consolidation_llm_provider,
            consolidation_model=plan.consolidation_model or None,
            memory_io_recorder=memory_io_recorder,
            memory_retrieval_max_tokens=plan.memory_retrieval_max_tokens,
            memory_retrieval_max_items=plan.memory_retrieval_max_items,
            simulator_runtime=plan.simulator_runtime,
            preference_overlay=preference_overlay,
            # A probe's PA belongs to a branch that is deleted right after, so
            # it must never become the run's cut-point handle: the mainline
            # binder is withheld, and the branch's own probe-local handle
            # (`loop_context["pa_binder"]`, harness/orchestrator.py) binds it
            # instead — that is where a probe's `--force-ix` seed lands.
            pa_binder=pa_binder if is_probe else bind_pa,
            skip_consolidation=skip_consolidation or step.skip_consolidation,
        )

    failure_policy = effective_failure_policy(run_dir, load_run_config())
    stop = shared_probe_stop or (GracefulStop(run_ledger, on_stop=shared_probe_memory_owner.request_stop)
                                if shared_probe_memory_owner is not None else GracefulStop(run_ledger))
    if shared_probe_stop is None:
        stop.install()
    orchestrator = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=run_dir,
            project_root=PROJECT_ROOT,
            pa_workspace=workspace,
            checkpoint_manager=checkpoint_manager,
            sim_profile=sim_profile,
            forced_ix=forced_ix,
            only_steps=only_steps,
            stop_requested=(lambda: stop() or shared_probe_memory_owner.is_stop_requested()) if shared_probe_memory_owner is not None else stop,
            shared_probe_memory_owner=shared_probe_memory_owner,
            probe_only=shared_probe_memory_owner is not None,
            probe_source_ledger=probe_source_ledger,
            probe_concurrency=probe_concurrency,
            failure_policy=failure_policy,
            probe_slot_root=workspace_root(PROJECT_ROOT) / ".probe-batch-slots",
            extra_pinned_step_ids=_resolve_extra_pins(
                plan,
                pin_steps=getattr(args, "pin_steps", None),
                pin_all_accumulations=getattr(
                    args, "pin_all_accumulations", False
                ),
            ),
            borrowed_runs_root=getattr(args, "borrowed_runs_root", None),
        ),
        executor=SessionExecutor(loop_factory),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=PROJECT_ROOT / "data/state_fixtures",
            runs_root=run_dir.parent,
            project_root=PROJECT_ROOT,
        ),
        memory_runtime=MemoryRuntime(
            memory_adapter,
            run_id=plan.run_id,
            persona_id=plan.persona_id,
            backend_namespace=memory_backend_namespace,
        ),
        memory_io_recorder=memory_io_recorder,
        ledger=run_ledger,
        preference_overlay=preference_overlay,
        pa_checkpoint=pa_checkpoint,
    )
    # One set, two readers: the probe batch adds and discards ids as its
    # workers open and close slots, and the signal handler reads it to name
    # every probe a stop is waiting on. Sharing the object rather than copying
    # is what makes the handler's read current at the moment the signal lands.
    if shared_probe_stop is None:
        stop.in_flight = orchestrator.in_flight
    else:
        stop.round_in_flight[str(run_dir)] = orchestrator.in_flight
    write_run_info(
        plan,
        run_dir,
        PROJECT_ROOT,
        event="launch",
        extra={
            # The id this launch's probe attempt records carry.
            "launch_id": orchestrator.launch_id,
            "resumed": resuming_existing_run,
            "argv": sys.argv[1:],
            "run_config": load_run_config().model_dump(mode="json"),
            "failure_policy": failure_policy.model_dump(mode="json"),
            "memos_health": memos_health,
        },
    )
    try:
        result = await orchestrator.run()
    finally:
        if shared_probe_stop is not None:
            stop.round_in_flight.pop(str(run_dir), None)
    stop.report(result)
    print()
    print(f"done run_dir={run_dir}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="optional run instance id; defaults to timestamp")
    parser.add_argument("--persona-id", default="user_a")
    acc_selection = parser.add_mutually_exclusive_group()
    acc_selection.add_argument(
        "--acc-range",
        default=None,
        help=(
            "inclusive acc range, e.g. 1-10 maps to acc_001..acc_010; "
            f"default {DEFAULT_ACC_RANGE}"
        ),
    )
    acc_selection.add_argument(
        "--acc-list",
        default=None,
        type=_acc_names_from_list,
        metavar="N|A-B,...",
        help=(
            "explicit acc selection, e.g. 1-15,17-31,33-34; comma-separated "
            "integers and inclusive ranges, expanded in order. Use it for a "
            "whole persona timeline, which the Outline leaves non-contiguous"
        ),
    )
    parser.add_argument("--memory-condition", default="nanobot_file_memory")
    parser.add_argument("--pa-model", default=os.environ.get("PA_MODEL", DEFAULT_PA_MODEL))
    parser.add_argument("--consolidation-model", default=os.environ.get("CONSOLIDATION_MODEL"))
    parser.add_argument(
        "--simulator-model",
        default=os.environ.get("SIMULATOR_MODEL", DEFAULT_SIMULATOR_MODEL),
        help="simulated-user model (OpenRouter id); frozen into the run plan",
    )
    parser.add_argument(
        "--memory-side-model",
        default=os.environ.get("MEMORY_SIDE_MODEL"),
    )
    parser.add_argument(
        "--memos-api-base",
        default=os.environ.get("MEMOS_API_BASE"),
    )
    parser.add_argument(
        "--force-ix",
        action="append",
        default=[],
        metavar="STEP:ATTRIBUTE=SETTING",
        help=(
            "validation only: pre-select one IX setting for that step's PA, "
            "e.g. acc_003:process_visibility=Full narration; repeatable"
        ),
    )
    parser.add_argument(
        "--only-steps",
        default=None,
        metavar="STEP_ID[,STEP_ID...]",
        help=(
            "probe phase only: run just these probe step ids (fin_NN / pre_NN "
            "as in run_plan.yaml, comma-separated, e.g. fin_02,pre_01); other "
            "probes are left as the ledger has them; the timeline phase is "
            "unaffected"
        ),
    )
    parser.add_argument(
        "--probe-concurrency",
        default=1,
        type=_probe_concurrency_arg,
        metavar="N",
        help=(
            "probe phase only: run up to N probes at once (default 1, the "
            "serial batch); capped at 4 for every memory condition"
        ),
    )
    parser.add_argument(
        "--pa-max-tokens-for-probe",
        action="append",
        default=[],
        metavar="STEP_ID=TOKENS",
        help=(
            "one probe step only: run its PA at a higher output cap than the "
            "frozen plan (at most 3x), recorded under pa_output_cap_override/. "
            "User rule 2026-09-23: only for a probe that keeps getting stuck "
            "at the plan cap"
        ),
    )
    parser.add_argument(
        "--rerun-probe-steps",
        default=None,
        metavar="STEP_ID[,STEP_ID...]",
        help=(
            "probe steps only: set these finished probes' step directories "
            "aside under steps_superseded/ and forget their ledger records, so "
            "the resume runs them again from their pinned cut points; "
            "checkpoints, the mainline world and memory are untouched"
        ),
    )
    parser.add_argument(
        "--pin-steps",
        default=None,
        metavar="STEP_ID,...",
        help=(
            "also pin these accumulation checkpoints, so a probe plan frozen "
            "later can branch from them; comma-separated step ids"
        ),
    )
    parser.add_argument(
        "--pin-all-accumulations",
        action="store_true",
        help=(
            "pin every accumulation checkpoint in this plan; the whole-run "
            "form of --pin-steps"
        ),
    )
    parser.add_argument(
        "--retain-checkpoints-every",
        type=_checkpoint_retention_interval,
        default=None,
        metavar="N",
        help=(
            "keep every Nth completed timeline checkpoint beyond the rolling "
            "window, including forced events; does not change probe pins or "
            "file-memory consolidation; repeat the same option on resume"
        ),
    )
    parser.add_argument(
        "--borrowed-runs-root",
        default=None,
        type=Path,
        metavar="DIR",
        help=(
            "where a probe's borrow_start_from looks for its source run; "
            "defaults to this run's own parent directory, which assumes the "
            "source sits beside it under the same model/condition/persona"
        ),
    )
    parser.add_argument("--plan")
    parser.add_argument("--regenerate-plan", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    return asyncio.run(_execute_benchmark_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
