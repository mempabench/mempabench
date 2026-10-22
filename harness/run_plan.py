"""Frozen run plan models and validation."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from harness.benchmark_clock import SESSION_ENVELOPE_MINUTES
from memory.backends.runtime_contracts import (
    MEM0_EXTRACTION_CHUNK_STRATEGY_LEGACY,
    MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1,
)
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from pa.agent.ix.render_types import IX_CONTRACT_VERSION
from memory.turn_experience import MEMORY_POOL_VERSION
from pa.providers.request_policy import validate_policy_snapshot


StepKind = Literal[
    "accumulation", "forced_event", "pre_event_probe", "final_probe"
]
#: The kinds that make up the mainline timeline, the only ones that publish a
#: checkpoint a probe can branch from.
TIMELINE_KINDS = frozenset({"accumulation", "forced_event"})
MemoryMode = Literal["read_write", "read_only"]
StagePolicy = Literal["commit", "discard"]
FILE_MEMORY_CONDITION = "nanobot_file_memory"
MEM0_CONDITION = "mem0"
MEMOS_CONDITION = "memos"
MEMORY_SIDE_LM_CONDITIONS = frozenset(
    {MEM0_CONDITION, MEMOS_CONDITION}
)
EMBEDDING_MEMORY_CONDITIONS = frozenset(
    {MEM0_CONDITION, "simple_rag", MEMOS_CONDITION}
)
DIRECT_QDRANT_CONDITIONS = frozenset({MEM0_CONDITION, "simple_rag"})
#: File name of the frozen `resolved_gt.yaml` copy inside a run directory.
RESOLVED_GT_COPY_NAME = "resolved_gt.yaml"
#: The benchmark's acc-number skeleton: every full plan carries slots 1..108.
#: A complete Outline authors 99 of them (`meta.active_session_count`); the
#: rest are gap days, carried as placeholder steps.
FULL_TIMELINE_ACC_COUNT = 108
FULL_TIMELINE_ACTIVE_COUNT = 99
DEFAULT_MEMORY_RETRIEVAL_MAX_TOKENS = 4000
DEFAULT_MEMORY_RETRIEVAL_MAX_ITEMS = 10
#: Completion-token ceiling for every PA request in a run. The provider
#: dataclass default (4096) cut six replies short in one regress16 session, so
#: the limit is a frozen plan field rather than a library default nobody sets.
#: A new plan freezes `budgets.pa_max_tokens` from configs/run_config.yaml;
#: this value is what a plan file without the key loads with.
DEFAULT_PA_MAX_TOKENS = 8192
DEFAULT_SIMULATOR_MODEL = "google/gemini-3.8-flash"
DEFAULT_SIMULATOR_RUNTIME = {
    "output_contract_retries": 3,
    "self_report_contract_retries": 5,
    "simulator_model": DEFAULT_SIMULATOR_MODEL,
}
RUNTIME_CONTEXTS = ("work", "personal")


def split_target_cell(target_cell: str) -> tuple[str, str]:
    """Split `work_autonomy_level` into its runtime context and attribute."""
    for context in RUNTIME_CONTEXTS:
        prefix = f"{context}_"
        if target_cell.startswith(prefix):
            attribute = target_cell[len(prefix) :]
            if attribute:
                return context, attribute
    raise ValueError(
        "target_cell must be '<context>_<attribute>' with context in "
        f"{list(RUNTIME_CONTEXTS)}: value={target_cell!r}"
    )


class RunPlanLoadError(ValueError):
    """Raised when a frozen run-plan file cannot be loaded or validated."""


class GeneratedFrom(BaseModel):
    """Source metadata for a frozen plan."""

    model_config = ConfigDict(extra="allow")

    timeline_path: str | None = None
    outline_path: str | None = None
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class FrozenResolvedGT(BaseModel):
    """The run's frozen copy of the persona's `resolved_gt.yaml` (§7 row 8).

    Copied into the run directory when the plan is frozen; the judge packer
    reads only this copy, never the authoring file.
    """

    # "ignore", as for `RunStep`: an older plan's keys are not read.
    model_config = ConfigDict(extra="ignore")

    #: Run-directory-relative path of the copy: `resolved_gt.yaml`.
    path: str

    @model_validator(mode="after")
    def _validate(self) -> "FrozenResolvedGT":
        if self.path != RESOLVED_GT_COPY_NAME:
            raise ValueError(
                "resolved_gt.path must be the run-relative copy "
                f"{RESOLVED_GT_COPY_NAME!r}: value={self.path!r}"
            )
        return self


class Mem0RuntimeConfig(BaseModel):
    """Secret-free runtime identity for Mem0-backed conditions."""

    # "ignore", as for `RunStep`: an older plan's keys are not read.
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[5] = 5
    mem0_version: str
    mem0_submodule_commit: str
    embedding_api_base: str
    #: Where a turn pool is cut into inputs mem0 can embed. The four values
    #: come from `memory/backends/mem0_adapter.py` via
    #: `mem0_embedding_chunk_policy()` and are required: a frozen plan without
    #: them cannot say which chunk boundaries produced its memories.
    embedding_max_input_tokens: int
    embedding_chunk_safety_margin_tokens: int
    embedding_tokenizer_encoding: str
    tiktoken_version: str
    #: How the `infer=True` path builds the inputs mem0 extracts from, from
    #: the same policy (`MEM0_EXTRACTION_CHUNK_STRATEGY`, which a new plan now
    #: freezes as `message_split_v1`). Unlike the four above this field has a
    #: default, and that default is a *read* rule for plans written before the
    #: field existed, not the strategy a new plan freezes: every plan frozen
    #: before 2026-09-18 was written without the field and must keep loading,
    #: under the legacy strategy it actually ran. Requiring the field would
    #: retire those plans; defaulting it to the new strategy would reinterpret
    #: them as something they never ran.
    extraction_chunk_strategy: Literal[
        MEM0_EXTRACTION_CHUNK_STRATEGY_LEGACY,
        MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1,
    ] = MEM0_EXTRACTION_CHUNK_STRATEGY_LEGACY
    #: The completion ceiling on one mem0 extraction call, from the same
    #: policy (`MEM0_EXTRACTION_MAX_OUTPUT_TOKENS`). Required for the same
    #: reason: it changes what mem0 is able to extract, so two runs on
    #: different ceilings are different runs.
    extraction_max_output_tokens: int
    retrieval_similarity_threshold: float = 0.1
    retrieval_rerank: Literal[False] = False
    qdrant_distance: Literal["Cosine"] = "Cosine"
    qdrant_on_disk: Literal[True] = True
    qdrant_bm25_enabled: Literal[False] = False
    enable_entity_memory: bool
    entity_model: str | None = None
    spacy_version: str | None = None
    history_db_relative_path: Literal[
        "backend_state/mem0_history.sqlite3"
    ] = "backend_state/mem0_history.sqlite3"

    @model_validator(mode="after")
    def _validate_runtime_values(self) -> Mem0RuntimeConfig:
        expected_nlp = (
            ("en_core_web_sm@3.8.0", "3.8.16")
            if self.enable_entity_memory else (None, None)
        )
        if (self.entity_model, self.spacy_version) != expected_nlp:
            raise ValueError("Mem0 entity NLP identity must match its enabled frozen configuration")
        self.mem0_version = self.mem0_version.strip()
        if not self.mem0_version:
            raise ValueError(
                f"Mem0 mem0_version must not be empty: value={self.mem0_version!r}"
            )
        self.mem0_submodule_commit = self.mem0_submodule_commit.strip()
        if (
            len(self.mem0_submodule_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.mem0_submodule_commit
            )
        ):
            raise ValueError(
                "Mem0 mem0_submodule_commit must contain 40 lowercase "
                f"hexadecimal characters: value={self.mem0_submodule_commit!r}"
            )
        self.embedding_api_base = self.embedding_api_base.strip().rstrip("/")
        embedding_api_base = urlsplit(self.embedding_api_base)
        if (
            embedding_api_base.scheme not in {"http", "https"}
            or not embedding_api_base.netloc
            or embedding_api_base.username is not None
            or embedding_api_base.password is not None
        ):
            raise ValueError(
                "Mem0 embedding_api_base must be an absolute HTTP(S) URL "
                f"without credentials: value={self.embedding_api_base!r}"
            )
        if (
            not math.isfinite(self.retrieval_similarity_threshold)
            or not 0 <= self.retrieval_similarity_threshold <= 1
        ):
            raise ValueError(
                "Mem0 retrieval_similarity_threshold must be finite and between "
                f"0 and 1: value={self.retrieval_similarity_threshold!r}"
            )
        if not 0 < self.embedding_chunk_safety_margin_tokens < (
            self.embedding_max_input_tokens
        ):
            raise ValueError(
                "Mem0 embedding chunk budget is invalid: "
                f"max_input_tokens={self.embedding_max_input_tokens!r} "
                "safety_margin_tokens="
                f"{self.embedding_chunk_safety_margin_tokens!r}"
            )
        if self.extraction_max_output_tokens <= 0:
            raise ValueError(
                "Mem0 extraction_max_output_tokens must be positive: "
                f"value={self.extraction_max_output_tokens!r}"
            )
        self.embedding_tokenizer_encoding = (
            self.embedding_tokenizer_encoding.strip()
        )
        self.tiktoken_version = self.tiktoken_version.strip()
        for field_name, value in (
            ("embedding_tokenizer_encoding", self.embedding_tokenizer_encoding),
            ("tiktoken_version", self.tiktoken_version),
        ):
            if not value:
                raise ValueError(
                    f"Mem0 {field_name} must not be empty: value={value!r}"
                )
        return self


class MemOSRuntimeConfig(BaseModel):
    """Secret-free runtime identity for one self-hosted MemOS condition."""

    # "ignore", as for `RunStep`: an older plan's keys are not read.
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[5] = 5
    api_base: str
    api_timeout_seconds: float = 180.0
    memos_version: str
    memos_submodule_commit: str
    mempa_patch_version: str
    memory_side_provider_name: str
    memory_side_model: str
    memory_side_api_base: str
    memory_side_model_roles: dict[str, str]
    memory_side_temperature_policy: Literal["send", "omit"]
    memory_side_completion_tokens_parameter: Literal[
        "max_tokens", "max_completion_tokens"
    ]
    memory_side_reasoning_effort_policy: Literal["send", "omit"]
    memory_side_task_affinity_required: bool
    memory_side_task_affinity_namespace: str
    memory_side_backup_enabled: Literal[False] = False
    memory_profile: Literal["text_memory"] = "text_memory"
    activation_memory_enabled: Literal[False] = False
    preference_memory_enabled: Literal[False] = False
    embedding_provider: str
    embedding_model: str
    embedding_api_base: str
    embedding_dimensions: int
    reranker_backend: str
    qdrant_host: str
    qdrant_port: int
    qdrant_image: str
    neo4j_backend: str
    neo4j_uri: str
    neo4j_database: str
    neo4j_image: str
    add_mode: Literal["fine"] = "fine"
    search_mode: Literal["fast", "fine", "mixture"] = "fast"
    top_k: int = 10
    relativity: float = 0.45
    dedup: Literal["no", "sim", "mmr"] = "mmr"
    chat_api_enabled: Literal[False] = False

    @model_validator(mode="after")
    def _validate_runtime_values(self) -> MemOSRuntimeConfig:
        self.api_base = self.api_base.strip().rstrip("/")
        parsed = urlsplit(self.api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "MemOS api_base must be an absolute HTTP(S) URL: "
                f"value={self.api_base!r}"
            )
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "MemOS api_base must not contain credentials: "
                f"value={self.api_base!r}"
            )
        self.memory_side_api_base = self.memory_side_api_base.strip().rstrip("/")
        memory_api_base = urlsplit(self.memory_side_api_base)
        if (
            memory_api_base.scheme not in {"http", "https"}
            or not memory_api_base.netloc
            or memory_api_base.username is not None
            or memory_api_base.password is not None
        ):
            raise ValueError(
                "MemOS memory_side_api_base must be an absolute HTTP(S) URL "
                "without credentials: "
                f"value={self.memory_side_api_base!r}"
            )
        self.embedding_api_base = self.embedding_api_base.strip().rstrip("/")
        embedding_api_base = urlsplit(self.embedding_api_base)
        if (
            embedding_api_base.scheme not in {"http", "https"}
            or not embedding_api_base.netloc
            or embedding_api_base.username is not None
            or embedding_api_base.password is not None
        ):
            raise ValueError(
                "MemOS embedding_api_base must be an absolute HTTP(S) URL "
                "without credentials: "
                f"value={self.embedding_api_base!r}"
            )
        self.neo4j_uri = self.neo4j_uri.strip().rstrip("/")
        neo4j_uri = urlsplit(self.neo4j_uri)
        if (
            not neo4j_uri.scheme
            or not neo4j_uri.netloc
            or neo4j_uri.username is not None
            or neo4j_uri.password is not None
        ):
            raise ValueError(
                "MemOS neo4j_uri must be an absolute URL without credentials: "
                f"value={self.neo4j_uri!r}"
            )
        required_text_fields = {
            "memos_version": self.memos_version,
            "memos_submodule_commit": self.memos_submodule_commit,
            "mempa_patch_version": self.mempa_patch_version,
            "memory_side_provider_name": self.memory_side_provider_name,
            "memory_side_model": self.memory_side_model,
            "memory_side_api_base": self.memory_side_api_base,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_api_base": self.embedding_api_base,
            "reranker_backend": self.reranker_backend,
            "qdrant_host": self.qdrant_host,
            "qdrant_image": self.qdrant_image,
            "neo4j_backend": self.neo4j_backend,
            "neo4j_uri": self.neo4j_uri,
            "neo4j_database": self.neo4j_database,
            "neo4j_image": self.neo4j_image,
        }
        for field_name, value in required_text_fields.items():
            if not value.strip():
                raise ValueError(
                    f"MemOS {field_name} must not be empty: value={value!r}"
                )
        if (
            not math.isfinite(self.api_timeout_seconds)
            or self.api_timeout_seconds <= 0
        ):
            raise ValueError(
                "MemOS api_timeout_seconds must be finite and positive: "
                f"value={self.api_timeout_seconds!r}"
            )
        if self.embedding_dimensions <= 0:
            raise ValueError(
                "MemOS embedding_dimensions must be positive: "
                f"value={self.embedding_dimensions!r}"
            )
        if not self.memory_side_model_roles or any(
            not role.strip() or not model.strip()
            for role, model in self.memory_side_model_roles.items()
        ):
            raise ValueError(
                "MemOS memory_side_model_roles requires non-empty role/model values: "
                f"value={self.memory_side_model_roles!r}"
            )
        required_model_roles = {
            "chat",
            "memory_reader",
            "general_reader",
        }
        if set(self.memory_side_model_roles) != required_model_roles:
            raise ValueError(
                "MemOS memory_side_model_roles does not match the required roles: "
                f"expected={sorted(required_model_roles)!r} "
                f"actual={sorted(self.memory_side_model_roles)!r}"
            )
        unexpected_role_models = {
            role: model
            for role, model in self.memory_side_model_roles.items()
            if model != self.memory_side_model
        }
        if unexpected_role_models:
            raise ValueError(
                "MemOS memory-side roles must use the fixed memory-side model: "
                f"memory_side_model={self.memory_side_model!r} "
                f"mismatches={unexpected_role_models!r}"
            )
        if (
            self.memory_side_task_affinity_required
            and not self.memory_side_task_affinity_namespace.strip()
        ):
            raise ValueError(
                "MemOS memory_side_task_affinity_namespace is required when "
                "task affinity is enabled: "
                f"value={self.memory_side_task_affinity_namespace!r}"
            )
        self.qdrant_host = self.qdrant_host.strip()
        if not self.qdrant_host or not 1 <= self.qdrant_port <= 65535:
            raise ValueError(
                "MemOS Qdrant endpoint is invalid: "
                f"host={self.qdrant_host!r} port={self.qdrant_port!r}"
            )
        if self.top_k <= 0:
            raise ValueError(
                f"MemOS top_k has an invalid value: value={self.top_k!r}"
            )
        if not math.isfinite(self.relativity) or not 0 <= self.relativity <= 1:
            raise ValueError(
                "MemOS relativity must be finite and between 0 and 1: "
                f"value={self.relativity!r}"
            )
        return self


class FrozenProbeOverlay(BaseModel):
    """The probe whose `fixture/` directory a probe runs over."""

    # "ignore", as for `RunStep`: an older plan's keys are not read.
    model_config = ConfigDict(extra="ignore")

    owner_probe_id: str

    @field_validator("owner_probe_id")
    @classmethod
    def _owner_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("owner_probe_id must not be empty")
        return value


class RunStep(BaseModel):
    """One executable step in a frozen benchmark run plan."""

    # "ignore": a plan frozen by an older or newer checkout still loads; a
    # field this code does not know is simply not read.
    model_config = ConfigDict(extra="ignore")

    step_id: str
    kind: StepKind
    persona_id: str
    script_path: Path
    # Recorded at freeze, read by nothing: which probe's `fixture/` this probe
    # was frozen over. The run resolves the overlay itself
    # (`resolve_probe_overlay`).
    probe_overlay: FrozenProbeOverlay | None = None
    context: str | None = None
    target_cell: str | None = None
    target_attribute: str | None = None
    target_setting: str | None = None
    acc_num: int | None = None
    before_acc_num: int | None = None
    session_type: str | None = None
    domain: str | None = None
    story_id: str | None = None
    timeline_slot: str | None = None
    time_position: str | None = None
    benchmark_session_timestamp: datetime
    memory_mode: MemoryMode
    stage_policy: StagePolicy
    placeholder: bool = False
    borrow_start_from: str | None = None
    # A probe runs on a throwaway branch, so it skips compaction outright.
    skip_consolidation: bool = False

    @field_validator("step_id")
    @classmethod
    def _step_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("step_id must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> RunStep:
        if (
            self.benchmark_session_timestamp.tzinfo is None
            or self.benchmark_session_timestamp.utcoffset() is None
        ):
            raise ValueError(
                "benchmark_session_timestamp must include a timezone: "
                f"step_id={self.step_id!r} "
                f"value={self.benchmark_session_timestamp!r}"
            )
        if self.kind in {"accumulation", "forced_event"}:
            if self.acc_num is None:
                raise ValueError(
                    f"{self.kind} step requires acc_num: "
                    f"step_id={self.step_id!r} acc_num={self.acc_num!r}"
                )
            if self.before_acc_num is not None:
                raise ValueError(
                    f"{self.kind} step must not set before_acc_num: "
                    f"step_id={self.step_id!r} "
                    f"before_acc_num={self.before_acc_num!r}"
                )
            if self.kind == "forced_event" and (
                self.memory_mode != "read_write" or self.stage_policy != "commit"
            ):
                raise ValueError(
                    "forced_event must be read_write/commit so its transcript reaches "
                    "the shared PA and memory history"
                )
        elif self.kind == "pre_event_probe":
            if self.before_acc_num is None:
                raise ValueError(
                    "pre_event_probe requires before_acc_num: "
                    f"step_id={self.step_id!r} "
                    f"before_acc_num={self.before_acc_num!r}"
                )
            if self.acc_num is not None:
                raise ValueError(
                    "pre_event_probe must not set acc_num: "
                    f"step_id={self.step_id!r} acc_num={self.acc_num!r}"
                )
            if self.memory_mode != "read_only" or self.stage_policy != "discard":
                raise ValueError(
                    "pre_event_probe must be read_only/discard: "
                    f"step_id={self.step_id!r} "
                    f"memory_mode={self.memory_mode!r} "
                    f"stage_policy={self.stage_policy!r}"
                )
        elif self.kind == "final_probe":
            if self.acc_num is not None or self.before_acc_num is not None:
                raise ValueError(
                    "final_probe must not set acc_num or before_acc_num: "
                    f"step_id={self.step_id!r} acc_num={self.acc_num!r} "
                    f"before_acc_num={self.before_acc_num!r}"
                )
            if self.memory_mode != "read_only" or self.stage_policy != "discard":
                raise ValueError(
                    "final_probe must be read_only/discard: "
                    f"step_id={self.step_id!r} "
                    f"memory_mode={self.memory_mode!r} "
                    f"stage_policy={self.stage_policy!r}"
                )
        if (
            self.kind not in {"pre_event_probe", "final_probe"}
            and self.probe_overlay is not None
        ):
            raise ValueError(
                "only a probe step carries a frozen effective overlay: "
                f"step_id={self.step_id!r} kind={self.kind!r}"
            )
        if self.borrow_start_from is not None:
            if self.kind not in {"pre_event_probe", "final_probe"}:
                raise ValueError(
                    "borrow_start_from is only meaningful for probe steps: "
                    f"step_id={self.step_id!r} kind={self.kind!r}"
                )
            self.borrowed_checkpoint()
        return self

    def borrowed_checkpoint(self) -> tuple[str, str]:
        """Split `borrow_start_from` into `(source_run_id, pinned_id)`.

        The validator only checks the shape: whether that pinned snapshot is
        actually restorable is a runtime question, not a plan-freeze one.
        """
        if self.borrow_start_from is None:
            raise ValueError(
                f"step does not borrow a start point: step_id={self.step_id!r}"
            )
        parts = self.borrow_start_from.split("/")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(
                "borrow_start_from must be '<run_id>/<pinned_checkpoint_id>': "
                f"step_id={self.step_id!r} value={self.borrow_start_from!r}"
            )
        return parts[0], parts[1]

    @property
    def memory_abbrev(self) -> str:
        return "rw" if self.memory_mode == "read_write" else "ro"


class SimulatorRuntimeConfig(BaseModel):
    """Frozen simulator model and retry settings for one run."""

    model_config = ConfigDict(extra="forbid")

    output_contract_retries: int = 3
    self_report_contract_retries: int = 5
    simulator_model: str = DEFAULT_SIMULATOR_MODEL

    @model_validator(mode="after")
    def _validate_values(self) -> "SimulatorRuntimeConfig":
        if self.output_contract_retries < 0:
            raise ValueError("output_contract_retries must be non-negative")
        if self.self_report_contract_retries < 0:
            raise ValueError("self_report_contract_retries must be non-negative")
        self.simulator_model = self.simulator_model.strip()
        if not self.simulator_model:
            raise ValueError("simulator_model must not be empty")
        return self


class RunPlan(BaseModel):
    """Frozen runtime plan consumed by the benchmark orchestrator."""

    # "ignore" for the same reason as `RunStep`: old run_plan.yaml files load.
    model_config = ConfigDict(extra="ignore")

    run_id: str
    persona_id: str
    memory_condition: str
    memory_retrieval_max_tokens: int = DEFAULT_MEMORY_RETRIEVAL_MAX_TOKENS
    memory_retrieval_max_items: int = DEFAULT_MEMORY_RETRIEVAL_MAX_ITEMS
    pa_session_key: str | None = None
    checkpoint_rolling_window: int = 3
    assistant_llm_provider_name: str
    assistant_llm_api_base: str = ""
    pa_model: str = ""
    pa_max_tokens: int = DEFAULT_PA_MAX_TOKENS
    #: Failures of one model on one OpenRouter upstream before the run stops
    #: sending that model's requests there; 0 = off (`harness/provider_avoidance.py`).
    #: None, for a plan frozen before the key existed, is off as well: no
    #: state is read, written or applied.
    provider_avoid_after_failures: int | None = Field(default=None, ge=0)
    pa_request_policy: dict[str, Any]
    consolidation_model: str = ""
    consolidation_request_policy: dict[str, Any] | None = None
    memory_side_model: str = ""
    memory_side_request_policy: dict[str, Any] | None = None
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 0
    qdrant_host: str = ""
    qdrant_port: int = 0
    mem0_runtime_config: Mem0RuntimeConfig | None = None
    memos_runtime_config: MemOSRuntimeConfig | None = None
    memos_cube_id: str = ""
    simulator_runtime: SimulatorRuntimeConfig = Field(
        default_factory=SimulatorRuntimeConfig
    )
    generated_from: GeneratedFrom = Field(default_factory=GeneratedFrom)
    #: Set when the plan carries probe steps: where the frozen GT copy lives
    #: in the run directory.
    resolved_gt: FrozenResolvedGT | None = None
    ix_contract_version: str = Field(default_factory=lambda: IX_CONTRACT_VERSION)
    memory_pool_version: str = Field(default_factory=lambda: MEMORY_POOL_VERSION)
    steps: list[RunStep]

    @field_validator("steps")
    @classmethod
    def _steps_not_empty(cls, steps: list[RunStep]) -> list[RunStep]:
        if not steps:
            raise ValueError("run plan must include at least one step")
        return steps

    @model_validator(mode="after")
    def _validate_steps(self) -> RunPlan:
        if self.memory_retrieval_max_tokens <= 0:
            raise ValueError(
                "memory_retrieval_max_tokens must be positive: "
                f"value={self.memory_retrieval_max_tokens!r}"
            )
        if self.memory_retrieval_max_items <= 0:
            raise ValueError(
                "memory_retrieval_max_items must be positive: "
                f"value={self.memory_retrieval_max_items!r}"
            )
        if self.checkpoint_rolling_window <= 0:
            raise ValueError(
                "checkpoint_rolling_window must be positive: "
                f"value={self.checkpoint_rolling_window!r}"
            )
        if self.pa_max_tokens <= 0:
            raise ValueError(
                "pa_max_tokens must be positive: "
                f"value={self.pa_max_tokens!r}"
            )
        if not self.assistant_llm_provider_name.strip():
            raise ValueError(
                "assistant_llm_provider_name must not be empty: "
                f"value={self.assistant_llm_provider_name!r}"
            )
        if self.assistant_llm_provider_name != "openrouter":
            raise ValueError(
                "Benchmark RunPlan assistant provider must be OpenRouter: "
                f"value={self.assistant_llm_provider_name!r}"
            )
        if (
            self.assistant_llm_api_base.rstrip("/")
            != "https://openrouter.ai/api/v1"
        ):
            raise ValueError(
                "Benchmark RunPlan assistant API base must use the standard "
                "OpenRouter endpoint: "
                f"value={self.assistant_llm_api_base!r}"
            )
        pa_policy = validate_policy_snapshot(
            self.pa_request_policy,
            field_name="pa_request_policy",
        )
        pa_policy.require_exact_role("pa")
        if (
            pa_policy.provider_name != self.assistant_llm_provider_name
            or pa_policy.model != self.pa_model
        ):
            raise ValueError(
                "PA request policy does not match provider/model: "
                f"policy_provider={pa_policy.provider_name!r} "
                f"provider={self.assistant_llm_provider_name!r} "
                f"policy_model={pa_policy.model!r} model={self.pa_model!r}"
            )
        uses_file_memory = self.memory_condition == FILE_MEMORY_CONDITION
        if uses_file_memory and not self.consolidation_model:
            raise ValueError(
                "nanobot_file_memory requires consolidation_model: "
                f"memory_condition={self.memory_condition!r} "
                f"consolidation_model={self.consolidation_model!r}"
            )
        if not uses_file_memory and (
            self.consolidation_model
            or self.consolidation_request_policy is not None
        ):
            raise ValueError(
                "Consolidation configuration is only valid for "
                "nanobot_file_memory: "
                f"memory_condition={self.memory_condition!r} "
                f"consolidation_model={self.consolidation_model!r} "
                f"consolidation_request_policy="
                f"{self.consolidation_request_policy!r}"
            )
        if uses_file_memory:
            if self.consolidation_request_policy is None:
                raise ValueError(
                    "consolidation_request_policy is required when "
                    f"consolidation_model={self.consolidation_model!r}"
                )
            consolidation_policy = validate_policy_snapshot(
                self.consolidation_request_policy,
                field_name="consolidation_request_policy",
            )
            consolidation_policy.require_exact_role("consolidation")
            if (
                consolidation_policy.provider_name
                != self.assistant_llm_provider_name
                or consolidation_policy.model != self.consolidation_model
            ):
                raise ValueError(
                    "Consolidation request policy does not match provider/model: "
                    f"policy_provider={consolidation_policy.provider_name!r} "
                    f"provider={self.assistant_llm_provider_name!r} "
                    f"policy_model={consolidation_policy.model!r} "
                    f"model={self.consolidation_model!r}"
                )
        uses_memory_side_lm = self.memory_condition in MEMORY_SIDE_LM_CONDITIONS
        memory_side_fields_set = bool(
            self.memory_side_model
            or self.memory_side_request_policy is not None
        )
        if uses_memory_side_lm and not memory_side_fields_set:
            raise ValueError(
                f"{self.memory_condition} requires frozen memory-side LM "
                "configuration: "
                f"memory_condition={self.memory_condition!r} "
                f"memory_side_model={self.memory_side_model!r}"
            )
        if not uses_memory_side_lm and memory_side_fields_set:
            raise ValueError(
                "Memory-side LM configuration is only valid for a condition "
                "that uses a memory-side LM: "
                f"memory_condition={self.memory_condition!r}"
            )
        if uses_memory_side_lm:
            if self.memory_side_request_policy is None:
                raise ValueError(
                    "memory_side_request_policy is required for this memory "
                    f"condition: memory_condition={self.memory_condition!r} "
                    f"memory_side_model={self.memory_side_model!r}"
                )
            memory_side_policy = validate_policy_snapshot(
                self.memory_side_request_policy,
                field_name="memory_side_request_policy",
            )
            memory_side_policy.require_exact_role("memory_extraction")
            if (
                memory_side_policy.provider_name
                != self.assistant_llm_provider_name
                or memory_side_policy.model != self.memory_side_model
            ):
                raise ValueError(
                    "Memory-side request policy does not match provider/model: "
                    f"policy_provider={memory_side_policy.provider_name!r} "
                    f"provider={self.assistant_llm_provider_name!r} "
                    f"policy_model={memory_side_policy.model!r} "
                    f"model={self.memory_side_model!r}"
                )

        uses_embedding = self.memory_condition in EMBEDDING_MEMORY_CONDITIONS
        embedding_fields_set = bool(
            self.embedding_provider
            or self.embedding_model
            or self.embedding_dimensions
        )
        if uses_embedding and not (
            self.embedding_provider
            and self.embedding_model
            and self.embedding_dimensions > 0
        ):
            raise ValueError(
                "Embedding-backed memory requires provider, model, and positive "
                "dimensions: "
                f"memory_condition={self.memory_condition!r} "
                f"embedding_provider={self.embedding_provider!r} "
                f"embedding_model={self.embedding_model!r} "
                f"embedding_dimensions={self.embedding_dimensions!r}"
            )
        if not uses_embedding and embedding_fields_set:
            raise ValueError(
                "Embedding configuration is only valid for embedding-backed "
                f"memory: memory_condition={self.memory_condition!r}"
            )
        uses_direct_qdrant = self.memory_condition in DIRECT_QDRANT_CONDITIONS
        qdrant_endpoint_fields_set = bool(self.qdrant_host or self.qdrant_port)
        if uses_direct_qdrant:
            self.qdrant_host = self.qdrant_host.strip()
            if not self.qdrant_host or not 1 <= self.qdrant_port <= 65535:
                raise ValueError(
                    "Embedding-backed memory requires a valid Qdrant endpoint: "
                    f"memory_condition={self.memory_condition!r} "
                    f"qdrant_host={self.qdrant_host!r} "
                    f"qdrant_port={self.qdrant_port!r}"
                )
        elif qdrant_endpoint_fields_set:
            raise ValueError(
                "Direct Qdrant endpoint is only valid for a direct-Qdrant "
                "memory condition: "
                f"memory_condition={self.memory_condition!r} "
                f"qdrant_host={self.qdrant_host!r} "
                f"qdrant_port={self.qdrant_port!r}"
            )
        if uses_direct_qdrant and self.mem0_runtime_config is None:
            raise ValueError(
                "Mem0-backed memory requires mem0_runtime_config: "
                f"memory_condition={self.memory_condition!r}"
            )
        if not uses_direct_qdrant and self.mem0_runtime_config is not None:
            raise ValueError(
                "mem0_runtime_config is only valid for a Mem0-backed memory "
                f"condition: memory_condition={self.memory_condition!r}"
            )
        if self.mem0_runtime_config is not None:
            expected_entity_memory = self.memory_condition == MEM0_CONDITION
            if (
                self.mem0_runtime_config.enable_entity_memory
                is not expected_entity_memory
            ):
                raise ValueError(
                    "Mem0 entity memory setting does not match the memory "
                    "condition: "
                    f"memory_condition={self.memory_condition!r} "
                    f"expected={expected_entity_memory!r} "
                    "actual="
                    f"{self.mem0_runtime_config.enable_entity_memory!r}"
                )
        uses_memos = self.memory_condition == MEMOS_CONDITION
        if uses_memos and self.memos_runtime_config is None:
            raise ValueError(
                "memos requires memos_runtime_config: "
                f"memory_condition={self.memory_condition!r}"
            )
        self.memos_cube_id = self.memos_cube_id.strip()
        if uses_memos and not self.memos_cube_id:
            raise ValueError(
                "memos requires a frozen memos_cube_id: "
                f"value={self.memos_cube_id!r}"
            )
        if not uses_memos and self.memos_cube_id:
            raise ValueError(
                "memos_cube_id is only valid for memos: "
                f"memory_condition={self.memory_condition!r} "
                f"value={self.memos_cube_id!r}"
            )
        if not uses_memos and self.memos_runtime_config is not None:
            raise ValueError(
                "memos_runtime_config is only valid for memos: "
                f"memory_condition={self.memory_condition!r}"
            )
        if self.memos_runtime_config is not None:
            memos_config = self.memos_runtime_config
            if (
                memos_config.memory_side_provider_name
                != self.assistant_llm_provider_name
                or memos_config.memory_side_model != self.memory_side_model
                or memos_config.memory_side_temperature_policy
                != memory_side_policy.temperature
                or memos_config.memory_side_completion_tokens_parameter
                != memory_side_policy.completion_tokens_parameter
                or memos_config.memory_side_reasoning_effort_policy
                != memory_side_policy.reasoning_effort
                or memos_config.memory_side_task_affinity_required
                or memos_config.memory_side_task_affinity_namespace
                or memos_config.embedding_provider != self.embedding_provider
                or memos_config.embedding_model != self.embedding_model
                or memos_config.embedding_dimensions
                != self.embedding_dimensions
            ):
                raise ValueError(
                    "MemOS runtime configuration does not match frozen shared "
                    "memory fields: "
                    f"runtime_provider={memos_config.memory_side_provider_name!r} "
                    f"provider={self.assistant_llm_provider_name!r} "
                    f"runtime_model={memos_config.memory_side_model!r} "
                    f"model={self.memory_side_model!r} "
                    "runtime_temperature_policy="
                    f"{memos_config.memory_side_temperature_policy!r} "
                    f"temperature_policy={memory_side_policy.temperature!r} "
                    "runtime_completion_tokens_parameter="
                    f"{memos_config.memory_side_completion_tokens_parameter!r} "
                    "completion_tokens_parameter="
                    f"{memory_side_policy.completion_tokens_parameter!r} "
                    "runtime_task_affinity_required="
                    f"{memos_config.memory_side_task_affinity_required!r} "
                    "runtime_task_affinity_namespace="
                    f"{memos_config.memory_side_task_affinity_namespace!r} "
                    f"runtime_embedding_model={memos_config.embedding_model!r} "
                    f"embedding_model={self.embedding_model!r} "
                    "runtime_embedding_dimensions="
                    f"{memos_config.embedding_dimensions!r} "
                    f"embedding_dimensions={self.embedding_dimensions!r}"
                )
        if self.pa_session_key is not None:
            key = self.pa_session_key.strip()
            if not key:
                raise ValueError(
                    f"pa_session_key must not be empty: "
                    f"value={self.pa_session_key!r}"
                )
            # A plan that borrows another run's start point deliberately names
            # that run's key: the session file it has to open was copied out of
            # the source run and still carries the source run's name.
            borrowed_run_ids = {
                step.borrowed_checkpoint()[0]
                for step in self.steps
                if step.borrow_start_from is not None
            }
            if self.run_id not in key and not any(
                run_id in key for run_id in borrowed_run_ids
            ):
                raise ValueError(
                    "pa_session_key must include run_id: "
                    f"pa_session_key={key!r} run_id={self.run_id!r}"
                )
            self.pa_session_key = key

        seen: set[str] = set()
        last_acc_num = 0
        previous_benchmark_timestamp: datetime | None = None
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError(f"duplicate step_id: {step.step_id}")
            seen.add(step.step_id)
            if step.persona_id != self.persona_id:
                raise ValueError(
                    f"step {step.step_id} persona_id {step.persona_id!r} "
                    f"does not match plan persona_id {self.persona_id!r}"
                )
            if step.kind not in {"accumulation", "forced_event"}:
                continue
            assert step.acc_num is not None
            if step.acc_num <= last_acc_num:
                raise ValueError(
                    "runtime acc_num values must be strictly increasing: "
                    f"step_id={step.step_id!r} "
                    f"previous_acc_num={last_acc_num!r} "
                    f"acc_num={step.acc_num!r}"
                )
            last_acc_num = step.acc_num
            if step.placeholder:
                # A gap day holds the acc slot and runs nothing
                # (`run_plan_builder._placeholder_step`), so it spends no
                # envelope and is not the clock the next authored session has
                # to clear: it carries the previous one's timestamp verbatim.
                continue
            # The envelope's strict increase is a timeline rule: probes branch
            # off a pinned cut point and share its clock (see
            # `_validate_probe_timestamps`), so they are skipped here.
            if (
                previous_benchmark_timestamp is not None
                and step.benchmark_session_timestamp
                < previous_benchmark_timestamp
                + timedelta(minutes=SESSION_ENVELOPE_MINUTES)
            ):
                raise ValueError(
                    "benchmark_session_timestamp must start at least the session "
                    f"envelope ({SESSION_ENVELOPE_MINUTES} minutes) after the "
                    f"previous step: step_id={step.step_id!r} "
                    f"value={step.benchmark_session_timestamp.isoformat()!r} "
                    "previous="
                    f"{previous_benchmark_timestamp.isoformat()!r}"
                )
            previous_benchmark_timestamp = step.benchmark_session_timestamp
        self._validate_probe_timestamps()
        self._validate_probe_start_points()
        return self

    def _validate_probe_timestamps(self) -> None:
        """A probe's clock is its cut point's clock, not a timeline slot.

        Every `final_probe` runs at the one frozen final cut point (the probe
        outline's `meta.final_clock`, one envelope after the last timeline
        step), so the finals all carry the same timestamp. A `pre_event_probe`
        runs exactly one envelope before the timeline step its `before_acc_num`
        names, and sits immediately before that step in the plan so the pin
        derivation (`pinned_step_ids`) lands on the step before it. A plan
        without that timeline step (a borrowing baseline run) has nothing to
        check the pre against here; `_validate_probe_start_points` owns that
        case.
        """
        envelope = timedelta(minutes=SESSION_ENVELOPE_MINUTES)
        timeline_index = {
            step.acc_num: index
            for index, step in enumerate(self.steps)
            if step.kind in {"accumulation", "forced_event"}
        }
        last_timeline_timestamp = max(
            (
                step.benchmark_session_timestamp
                for step in self.steps
                if step.kind in {"accumulation", "forced_event"}
            ),
            default=None,
        )
        final_cut: tuple[str, datetime] | None = None
        for index, step in enumerate(self.steps):
            if step.kind == "pre_event_probe":
                assert step.before_acc_num is not None
                acc_index = timeline_index.get(step.before_acc_num)
                if acc_index is None:
                    continue
                acc_step = self.steps[acc_index]
                expected = acc_step.benchmark_session_timestamp - envelope
                if step.benchmark_session_timestamp != expected:
                    raise ValueError(
                        "pre_event_probe must start exactly one session envelope "
                        f"({SESSION_ENVELOPE_MINUTES} minutes) before its "
                        f"before_acc step: step_id={step.step_id!r} "
                        f"value={step.benchmark_session_timestamp.isoformat()!r} "
                        f"before_acc_step_id={acc_step.step_id!r} "
                        f"expected={expected.isoformat()!r}"
                    )
                if index > acc_index:
                    raise ValueError(
                        "pre_event_probe must sit before its before_acc step in "
                        f"the plan: step_id={step.step_id!r} "
                        f"before_acc_step_id={acc_step.step_id!r}"
                    )
            elif step.kind == "final_probe":
                if final_cut is None:
                    if (
                        last_timeline_timestamp is not None
                        and step.benchmark_session_timestamp
                        < last_timeline_timestamp + envelope
                    ):
                        raise ValueError(
                            "final_probe must start at least the session envelope "
                            f"({SESSION_ENVELOPE_MINUTES} minutes) after the last "
                            f"timeline step: step_id={step.step_id!r} "
                            f"value={step.benchmark_session_timestamp.isoformat()!r} "
                            f"last_timeline={last_timeline_timestamp.isoformat()!r}"
                        )
                    final_cut = (step.step_id, step.benchmark_session_timestamp)
                elif step.benchmark_session_timestamp != final_cut[1]:
                    raise ValueError(
                        "every final_probe must share the one frozen final cut "
                        f"point: step_id={step.step_id!r} "
                        f"value={step.benchmark_session_timestamp.isoformat()!r} "
                        f"first_final_step_id={final_cut[0]!r} "
                        f"final_cut={final_cut[1].isoformat()!r}"
                    )

    def _validate_probe_start_points(self) -> None:
        """Every probe needs a cut point to branch from.

        Normally that is the nearest preceding accumulation/forced_event step
        in this same plan. A memory-removed baseline run has no timeline of its
        own, so each of its probes names the pinned snapshot it borrows from
        instead, and that borrowed session file carries the *source* run's
        mainline key.
        """
        seen_timeline_step = False
        borrowed_run_ids: set[str] = set()
        for step in self.steps:
            if step.kind in {"accumulation", "forced_event"}:
                seen_timeline_step = True
                continue
            if step.kind not in {"pre_event_probe", "final_probe"}:
                continue
            if step.borrow_start_from is not None:
                borrowed_run_ids.add(step.borrowed_checkpoint()[0])
                continue
            if not seen_timeline_step:
                raise ValueError(
                    "a probe must follow an accumulation/forced_event step or "
                    "declare borrow_start_from: "
                    f"step_id={step.step_id!r} kind={step.kind!r}"
                )
        if not borrowed_run_ids:
            return
        if len(borrowed_run_ids) > 1:
            raise ValueError(
                "a borrowing plan must borrow from a single source run, "
                "because all its probes share one session key: "
                f"source_run_ids={sorted(borrowed_run_ids)!r}"
            )
        expected_key = f"{next(iter(borrowed_run_ids))}_main"
        if self.main_pa_session_key() != expected_key:
            raise ValueError(
                "a plan that borrows a start point must set pa_session_key to "
                "the source run's mainline key, or the copied session file "
                "cannot be found in the branch workspace: "
                f"pa_session_key={self.main_pa_session_key()!r} "
                f"expected={expected_key!r}"
            )

    def pinned_step_ids(self) -> set[str]:
        """Derive the pinned checkpoint steps from step order alone.

        A pinned step is the nearest preceding *accumulation* step before each
        `pre_event_probe` / `final_probe`. Nothing is stored on the plan: the
        cut points follow from the timeline itself.

        A forced-event step is never pinned, structurally. The orchestrator
        flushes the never-compacted turns at a pinned cut point
        (`harness/orchestrator.py:_flush_uncompacted_turns`), and no
        consolidation pass runs after a forced event (owner rule): its content
        is swept up at the next accumulation turn end. So when the step before
        a probe is a forced event, the pin falls back to the last accumulation
        step before it rather than landing on the event.

        A placeholder is skipped for the same reason: a gap day runs nothing,
        so it can never publish the checkpoint a probe would branch from.
        """
        pinned: set[str] = set()
        latest_accumulation_step_id: str | None = None
        for step in self.steps:
            if step.kind == "accumulation" and not step.placeholder:
                latest_accumulation_step_id = step.step_id
            elif step.kind in {"pre_event_probe", "final_probe"}:
                if latest_accumulation_step_id is not None:
                    pinned.add(latest_accumulation_step_id)
        return pinned

    def probe_source_step_id(self, step: RunStep, pinned: set[str]) -> str | None:
        """The cut point this probe branches from: the last pinned timeline step
        in front of it, or None when the timeline has published none yet.

        Plan-only, so the orchestrator's branch
        (`harness/orchestrator.py::_probe_source_checkpoint`) and the repeat
        preparation read the same cut point rather than each deciding for
        itself.
        """
        latest: str | None = None
        for candidate in self.steps:
            if candidate.step_id == step.step_id:
                break
            if candidate.kind in TIMELINE_KINDS and candidate.step_id in pinned:
                latest = candidate.step_id
        return latest

    def main_pa_session_key(self) -> str:
        return self.pa_session_key or f"{self.run_id}_main"

    def pa_session_key_for_step(self, step: RunStep) -> str:
        """Every step shares the mainline session key.

        A probe continues the mainline PA session inside its own branch
        directory; isolation comes from the branch, not from a second session
        file. The `{run_id}_probe_{probe_id}` naming is gone.
        """
        return self.main_pa_session_key()

    def validate_script_paths(self, *, project_root: Path | str = ".") -> None:
        root = Path(project_root)
        for step in self.steps:
            if step.placeholder:
                continue
            script_path = step.script_path
            candidate = script_path if script_path.is_absolute() else root / script_path
            if not candidate.exists():
                raise FileNotFoundError(f"script_path not found for {step.step_id}: {script_path}")

    def dump_yaml(self, path: Path | str) -> Path:
        validate_forced_event_precedence(self)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        data["pa_session_key"] = self.main_pa_session_key()
        target.write_text(
            yaml.safe_dump(
                data,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return target


def resolve_probe_overlay(
    plan: RunPlan,
    step: RunStep,
    *,
    project_root: Path | str = ".",
) -> Path | None:
    """The `fixture/` overlay one probe runs over, read off the drafts as they are.

    A pre_event probe (or an unpaired final) owns its own `fixture/`; a final
    whose draft declares `meta.pairs_with` runs over that pre_event probe's.
    Returns the directory, or None when the owner has none.
    """
    if step.kind not in {"pre_event_probe", "final_probe"}:
        raise ValueError(
            f"effective probe overlay requested for non-probe: step_id={step.step_id!r}"
        )
    root = Path(project_root)
    draft_path = step.script_path
    if not draft_path.is_absolute():
        draft_path = root / draft_path
    body = yaml.safe_load(draft_path.read_text(encoding="utf-8")) or {}
    meta = body.get("meta") if isinstance(body, dict) else None
    if not isinstance(meta, dict):
        meta = {}
    declared_kind = meta.get("probe_kind")
    declared_pair = meta.get("pairs_with")
    expected_kind = "pre_event" if step.kind == "pre_event_probe" else "final"
    if declared_kind != expected_kind:
        raise RunPlanLoadError(
            "probe draft kind does not match its RunStep: "
            f"step_id={step.step_id!r} step_kind={step.kind!r} "
            f"draft_kind={declared_kind!r}"
        )
    implied_owner = (
        step.step_id
        if step.kind == "pre_event_probe" or not declared_pair
        else str(declared_pair)
    )
    owners = [candidate for candidate in plan.steps if candidate.step_id == implied_owner]
    if len(owners) != 1:
        raise RunPlanLoadError(
            "probe effective-overlay owner must exist exactly once in the plan: "
            f"step_id={step.step_id!r} owner_probe_id={implied_owner!r} "
            f"matches={len(owners)}"
        )
    owner = owners[0]
    if owner.persona_id != step.persona_id:
        raise RunPlanLoadError(
            "probe effective-overlay owner belongs to a different persona: "
            f"step_id={step.step_id!r} persona={step.persona_id!r} "
            f"owner_persona={owner.persona_id!r}"
        )
    expected_owner_kind = (
        "pre_event_probe" if implied_owner != step.step_id else step.kind
    )
    if owner.kind != expected_owner_kind:
        raise RunPlanLoadError(
            "probe effective-overlay owner has the wrong probe kind: "
            f"step_id={step.step_id!r} owner_probe_id={implied_owner!r} "
            f"owner_kind={owner.kind!r} expected={expected_owner_kind!r}"
        )
    owner_draft = owner.script_path
    if not owner_draft.is_absolute():
        owner_draft = root / owner_draft
    overlay = owner_draft.parent / "fixture"
    return overlay if overlay.is_dir() else None


def load_run_plan(
    path: Path | str,
    *,
    project_root: Path | str = ".",
    validate_paths: bool = True,
) -> RunPlan:
    """Read a frozen plan: YAML, then the model, then (by default) that every
    step's script exists under `project_root`. Nothing is compared against a
    digest; `run_info.yaml` records what a run was made of."""
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunPlanLoadError(
            f"Failed to read run plan: path={source_path} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RunPlanLoadError(
            f"Invalid run-plan YAML: path={source_path} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        preview = repr(data)
        if len(preview) > 160:
            preview = preview[:157] + "..."
        raise RunPlanLoadError(
            f"Run-plan YAML root must be a mapping: path={source_path} "
            f"root_type={type(data).__name__} value={preview}"
        )
    try:
        plan = RunPlan.model_validate(data)
    except ValidationError as exc:
        raise RunPlanLoadError(
            f"Invalid run plan: path={source_path} error={exc}"
        ) from exc
    if validate_paths:
        plan.validate_script_paths(project_root=project_root)
    return plan


def validate_forced_event_precedence(plan: RunPlan) -> None:
    """Every post-event session must sit after its own forced event.

    The simulated user only moves a cell to its post setting once the forced
    event has actually replayed, so a plan that keeps an `evolving_post`
    session without the matching `evolving_event` earlier in the same plan
    could never reach the setting that session was authored against.
    """

    forced_acc_by_cell: dict[str, int] = {}
    for step in plan.steps:
        if step.kind != "forced_event" or not step.target_cell:
            continue
        assert step.acc_num is not None
        forced_acc_by_cell.setdefault(step.target_cell, step.acc_num)

    for step in plan.steps:
        if step.kind != "accumulation" or step.session_type != "evolving_post":
            continue
        forced_acc_num = forced_acc_by_cell.get(step.target_cell or "")
        if forced_acc_num is None or forced_acc_num >= (step.acc_num or 0):
            raise ValueError(
                "evolving_post step has no forced_event for its target cell "
                "earlier in the same plan, so the simulated user would still "
                "hold the pre-event setting: "
                f"step_id={step.step_id!r} acc_num={step.acc_num!r} "
                f"target_cell={step.target_cell!r} "
                f"forced_event_acc_num={forced_acc_num!r}"
            )


def freeze_run_plan(plan: RunPlan, run_dir: Path | str) -> Path:
    return plan.dump_yaml(Path(run_dir) / "run_plan.yaml")


def validate_full_timeline_shape(plan: RunPlan) -> None:
    """Strict shape check for a full benchmark run, called by the builder
    whenever the requested range is the whole 1..108 skeleton."""

    timeline = [
        step
        for step in plan.steps
        if step.kind in {"accumulation", "forced_event"}
    ]
    forced = [step for step in timeline if step.kind == "forced_event"]
    pre_event = [step for step in plan.steps if step.kind == "pre_event_probe"]
    final = [step for step in plan.steps if step.kind == "final_probe"]
    acc_nums = [step.acc_num for step in timeline]
    if acc_nums != list(range(1, FULL_TIMELINE_ACC_COUNT + 1)):
        raise ValueError(
            "full plan runtime steps must contain acc_num 1..108: "
            f"value={acc_nums!r}"
        )
    if len(forced) != 5:
        raise ValueError(
            f"full plan must contain 5 forced_event steps: value={len(forced)}"
        )
    if len(pre_event) != 5:
        raise ValueError(
            f"full plan must contain 5 pre_event_probe steps: value={len(pre_event)}"
        )
    if len(final) != 28:
        raise ValueError(
            f"full plan must contain 28 final_probe steps: value={len(final)}"
        )
    # The acc_num skeleton still runs 1..108; what has to be authored and
    # actually executed is the 99 active sessions inside it. "Active" is the
    # Outline's own count (`meta.active_session_count`, which
    # `scripts/validate_outline.py` holds equal to the number of entries), so
    # the five forced events are among the 99 and the remaining nine slots are
    # gap days carried as placeholders.
    active = [step for step in timeline if not step.placeholder]
    if len(active) != FULL_TIMELINE_ACTIVE_COUNT:
        raise ValueError(
            "full plan must contain 99 active timeline steps: "
            f"value={len(active)}"
        )
