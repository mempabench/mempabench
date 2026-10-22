"""Run settings that used to be written into code: one YAML, defaults = today's behaviour.

`configs/run_config.yaml` holds every key (another file can be named in the
`MEMPA_RUN_CONFIG` environment variable); a missing key keeps the default
below, and the defaults equal the shipped file, so an empty file behaves
exactly like it. Nothing here checks a digest: a value is read when a run
starts and recorded in that run's `run_info.yaml`.

`failure_policy` alone can also be set per run, in
`<run_dir>/failure_policy.yaml` (same keys), because which step of which run
to skip is decided after that run failed. That file, when present, replaces
the global `failure_policy` section for that run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_CONFIG_PATH = PROJECT_ROOT / "configs" / "run_config.yaml"
RUN_CONFIG_ENV = "MEMPA_RUN_CONFIG"
FAILURE_POLICY_OVERRIDE_NAME = "failure_policy.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FailurePolicy(_Section):
    """What a failed timeline step does. By default only an accumulation step
    that exhausted its three IX repairs is skipped; any other failure stops the run."""

    #: accumulation step_id -> "rollback" (go back to the latest checkpoint, record the step
    #: skipped, keep nothing of it) or "retain_partial" (commit what the failed
    #: scene left, publish a skipped checkpoint, continue).
    skip_steps: dict[str, Literal["rollback", "retain_partial"]] = Field(default_factory=dict)
    #: An accumulation step that exhausted its three IX repairs is skipped with
    #: its partial state retained instead of stopping the run.
    auto_skip_on_ix_exhaustion: bool = True
    #: Whole-scene attempts per accumulation step. 1 = a failure stops the run.
    #: N > 1: attempts 1..N-1 stop the run (the next launch rolls back and runs
    #: the scene again); attempt N failing is skipped with partial state retained.
    max_scene_attempts: int = Field(default=1, ge=1)
    #: Turns of a skipped step whose memory write never landed: "drop" leaves
    #: them out of memory (and out of every later replay), "replay" writes
    #: them before the run goes on.
    partial_skip_pending_memory: Literal["drop", "replay"] = "drop"


class ProbePolicy(_Section):
    """Whole-probe attempt budget."""

    attempt_limit: int = Field(default=3, ge=1)
    total_attempt_limit: int = Field(default=6, ge=1)
    #: Whether network/infrastructure failures count toward `attempt_limit`.
    network_failures_count: bool = False


class RepeatPolicy(_Section):
    """Probe-only repeat rounds of an existing run (`*_probe_repeatNN`)."""

    rounds: list[int] = Field(default_factory=lambda: [2, 3])


class ProbeVarianceScope(_Section):
    """How many pre-event and final Probes a probe-variance round runs."""

    pre_event_probes: int = 5
    final_probes: int = 28


class Budgets(_Section):
    """Token and window budgets frozen into a new plan."""

    pa_max_tokens: int = Field(default=24576, gt=0)
    memory_retrieval_max_tokens: int = Field(default=4000, gt=0)
    memory_retrieval_max_items: int = Field(default=10, gt=0)
    checkpoint_rolling_window: int = Field(default=3, gt=0)
    mem0_extraction_max_output_tokens: int = Field(default=16000, gt=0)


class Providers(_Section):
    """OpenRouter `provider` routing for one phase, e.g. {"order": ["SiliconFlow"],
    "allow_fallbacks": false}. None keeps the model's own routing."""

    probe_phase: dict | None = None
    #: Failures of one model on one OpenRouter upstream after which the run
    #: sends that model's requests elsewhere (`harness/provider_avoidance.py`);
    #: 0 turns this off. Frozen into a new plan.
    avoid_after_failures: int = Field(default=3, ge=0)


class MemOSDeployment(_Section):
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_image: str = "qdrant/qdrant:v1.15.3"
    reranker_backend: str = "cosine_local"
    neo4j_backend: str = "neo4j-community"
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_database: str = "neo4j"
    neo4j_image: str = "neo4j:5.26.6"


class Mem0Deployment(_Section):
    entity_model: str = "en_core_web_sm@3.8.0"
    spacy_version: str = "3.8.16"


class MemoryBackends(_Section):
    memos: MemOSDeployment = Field(default_factory=MemOSDeployment)
    mem0: Mem0Deployment = Field(default_factory=Mem0Deployment)


class RunConfig(_Section):
    failure_policy: FailurePolicy = Field(default_factory=FailurePolicy)
    probe: ProbePolicy = Field(default_factory=ProbePolicy)
    repeat: RepeatPolicy = Field(default_factory=RepeatPolicy)
    probe_variance: ProbeVarianceScope = Field(default_factory=ProbeVarianceScope)
    budgets: Budgets = Field(default_factory=Budgets)
    providers: Providers = Field(default_factory=Providers)
    memory_backends: MemoryBackends = Field(default_factory=MemoryBackends)


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"run config must be a mapping: path={path}")
    return data


def load_run_config(path: Path | str | None = None) -> RunConfig:
    """The run config at `path`, else `$MEMPA_RUN_CONFIG`, else
    `configs/run_config.yaml`; the defaults when that last file is absent."""
    if path is None:
        path = os.environ.get(RUN_CONFIG_ENV) or None
    source = Path(path) if path is not None else DEFAULT_RUN_CONFIG_PATH
    if not source.is_file():
        if path is not None:
            raise FileNotFoundError(f"run config not found: {source}")
        return RunConfig()
    return RunConfig.model_validate(_read_yaml(source))


def effective_failure_policy(run_dir: Path | str, config: RunConfig) -> FailurePolicy:
    """The run's own `failure_policy.yaml` if it has one, else the global section."""
    override = Path(run_dir) / FAILURE_POLICY_OVERRIDE_NAME
    if override.is_file():
        return FailurePolicy.model_validate(_read_yaml(override))
    return config.failure_policy
