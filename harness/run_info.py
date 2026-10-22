"""`run_info.yaml`: what each launch of a run was made of. Read only as a record.

One record is appended per event (`freeze` when a plan is written, `launch`
for every start or resume): the code commit and whether the checkout was
dirty, the model and memory configuration, the version strings, the PA model's
`provider_routing` rows, and whatever the caller adds (`extra`, e.g. the acc
range and probe step ids a plan froze, a launch's `launch_id` that its probe
attempt records carry, the failure policy in force or the MemOS service's
health answer). No code decides anything from this file.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

RUN_INFO_NAME = "run_info.yaml"


def _git(project_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), *args],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _config(plan: Any) -> dict[str, Any]:
    data = plan.model_dump(mode="json", exclude={"steps"})
    return {key: value for key, value in data.items() if value not in (None, "", {})}


def _pa_provider_routing(plan: Any, project_root: Path) -> list[dict[str, Any]]:
    """The `provider_routing` rows of configs/model_request_policies.yaml for the
    plan's PA model, as the file reads now."""
    try:
        data = yaml.safe_load((project_root / "configs/model_request_policies.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return []
    return [row for row in data.get("provider_routing") or []
            if isinstance(row, dict) and row.get("model") == getattr(plan, "pa_model", None)]


def plan_scope_record(plan: Any) -> dict[str, Any]:
    """The accs and probes a plan was frozen with: `acc_range` in `--acc-list`
    syntax ("14-15", "1-15,17-31"; gap-day placeholders count as accs) and the
    probe step ids in plan order."""
    spans: list[list[int]] = []
    for step in plan.steps:
        if step.kind not in {"accumulation", "forced_event"}:
            continue
        if spans and step.acc_num == spans[-1][1] + 1:
            spans[-1][1] = step.acc_num
        else:
            spans.append([step.acc_num, step.acc_num])
    return {
        "acc_range": ",".join(
            str(first) if first == last else f"{first}-{last}" for first, last in spans
        ),
        "probe_step_ids": [
            step.step_id for step in plan.steps
            if step.kind in {"pre_event_probe", "final_probe"}
        ],
    }


def memos_health_record(health: dict[str, Any]) -> dict[str, Any]:
    """What a MemOS `/health` answer says about the service that ran: its
    version, MemPA patch version and runtime contract."""
    return {key: health.get(key) for key in ("version", "mempa_patch_version", "runtime_contract")}


def write_run_info(
    plan: Any,
    run_dir: Path | str,
    project_root: Path | str,
    *,
    event: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append one record to `<run_dir>/run_info.yaml` and return its path."""
    from memory.turn_experience import MEMORY_POOL_VERSION
    from pa.agent.ix.render_types import IX_CONTRACT_VERSION

    root = Path(project_root)
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    record: dict[str, Any] = {
        "event": event,
        "time": datetime.now(timezone.utc).isoformat(),
        "code": {
            "commit": commit.strip() if commit else None,
            "dirty": bool(status.strip()) if status is not None else None,
        },
        "config": _config(plan),
        "versions": {
            "ix_contract_version": IX_CONTRACT_VERSION,
            "memory_pool_version": MEMORY_POOL_VERSION,
        },
        "pa_provider_routing": _pa_provider_routing(plan, root),
    }
    if extra:
        record.update(extra)
    path = Path(run_dir) / RUN_INFO_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    # A YAML list, one `- ` item per record, so appending keeps it one document.
    with path.open("a", encoding="utf-8") as stream:
        stream.write(yaml.safe_dump([record], sort_keys=False, allow_unicode=True))
    return path
