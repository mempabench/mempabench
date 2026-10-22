"""References to the original timeline behind an independent Probe repeat.

`probe_repeat_source.json` names the original run directory the repeat reads
its timeline from, and, for a reference arm, the arm (`memory_condition`,
`pa_model`) the repeat runs instead of the source's own. Loading checks only
that the source is a distinct original run that is not still producing. The
repeat's plan and GT copy are compared with the source's once, by
`scripts.run_probe_variance.validate_manifest`. Older records carry more keys
(file digests, an acceptance reference); they are not read.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from harness.run_config import load_run_config

NAME = "probe_repeat_source.json"
#: The conditions a reference arm runs: the two whose probe branch the
#: orchestrator empties of memory and gives its own adapter
#: (`harness/orchestrator.py` NO_MEMORY_CONDITION, ORACLE_PREFERENCE_CONDITION).
REFERENCE_CONDITIONS = frozenset({"no_memory", "oracle_preference"})


def repeat_rounds(arm: dict | None) -> list[int]:
    """`repeat.rounds` are the rounds a memory repeat adds to its source, which
    is round 1. A reference arm has no round in its source, so it also runs 1."""
    rounds = load_run_config().repeat.rounds
    return list(rounds) if arm is None else sorted({1, *rounds})


def repeat_run_id(source_run_id: str, arm: dict | None, repeat: int) -> str:
    """One run id per (source, arm condition, arm PA model, round)."""
    if arm is None:
        return f"{source_run_id}_probe_repeat{repeat:02d}"
    model = arm["pa_model"].replace("/", "_")
    return f"{source_run_id}_{arm['memory_condition']}_{model}_probe_repeat{repeat:02d}"


def repeat_base_plan(source_plan: dict, arm: dict | None) -> dict:
    """The source's frozen plan as a repeat runs it, before the repeat's own
    run_id, PA session key and Probe borrow links.

    A memory repeat runs it unchanged. A reference arm `{memory_condition,
    pa_model}` runs another condition and PA model on the same timeline and
    pins: the PA request policy is resolved for that model, and every
    consolidation, memory-side and external-backend field is cleared (none of
    them belongs to a reference condition).
    """
    plan = deepcopy(source_plan)
    if arm is None:
        return plan
    if not isinstance(arm, dict) or set(arm) != {"memory_condition", "pa_model"} \
            or arm["memory_condition"] not in REFERENCE_CONDITIONS:
        raise ValueError(f"A reference arm is {{memory_condition: one of {sorted(REFERENCE_CONDITIONS)}, "
                         f"pa_model}}; got {arm!r}")
    from pa.providers.request_policy import resolve_model_request_policy
    policy = resolve_model_request_policy(plan["assistant_llm_provider_name"], arm["pa_model"])
    policy.require_exact_role("pa")
    plan.update(
        memory_condition=arm["memory_condition"], pa_model=arm["pa_model"],
        pa_request_policy=policy.snapshot(),
        consolidation_model="", consolidation_request_policy=None,
        memory_side_model="", memory_side_request_policy=None,
        embedding_provider="", embedding_model="", embedding_dimensions=0, qdrant_host="", qdrant_port=0,
        mem0_runtime_config=None, memos_runtime_config=None, memos_cube_id="",
    )
    return plan


def write_repeat_source(run_dir: Path, source_run_dir: Path, repeat: int, arm: dict | None = None) -> Path:
    run_dir, source_run_dir = Path(run_dir).resolve(), Path(source_run_dir).resolve()
    if repeat not in repeat_rounds(arm) or run_dir == source_run_dir:
        raise ValueError(f"A Probe repeat needs its own directory and a repeat round in {repeat_rounds(arm)}")
    value = {"schema_version": 1, "repeat": repeat, "source_run_dir": str(source_run_dir)}
    if arm is not None:
        value["arm"] = dict(arm)
    path = run_dir / NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Refusing to replace an existing repeat's source record")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    load_repeat_source(run_dir)
    return path


def load_repeat_source(run_dir: Path) -> dict[str, Any] | None:
    run_dir = Path(run_dir).resolve()
    path = run_dir / NAME
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    arm = value.get("arm")
    if value.get("schema_version") != 1 or value.get("repeat") not in repeat_rounds(arm):
        raise ValueError("Unsupported Probe repeat source")
    source = Path(value["source_run_dir"])
    if not source.is_absolute() or source.resolve() == run_dir:
        raise ValueError("Probe repeat source must be a distinct absolute directory")
    if (source / NAME).exists():
        raise ValueError("A Probe repeat must reference an original timeline, not another repeat")
    if (source / "recovery/pending.json").exists():
        raise ValueError("Probe repeat source has an unfinished checkpoint restore")
    source_ledger = json.loads((source / "ledger.json").read_text())
    current = source_ledger.get("current_step")
    if current is not None:
        # The orchestrator can leave ``current_step`` on a final Probe that
        # exhausted its attempts; only that state is accepted.
        from harness.probe_na import original_exhausted_failure
        try:
            original_exhausted_failure(source, current, source_ledger)
        except ValueError as exc:
            raise ValueError(f"Probe repeat source still has an active current_step: {exc}") from exc
    return value
