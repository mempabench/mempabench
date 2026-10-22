"""Read a persona's resolved probe ground truth (``Authoring/probe/outline/<persona>/resolved_gt.yaml``).

Each cell record is stable (``setting``) or evolving (``pre_setting`` /
``post_setting`` / ``event_acc``). ``setting_for(kind, record)`` is the only way
any consumer (RunPlan builder, judge packer) reads a setting back out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.probe_outline_schema import final_probe_id

PROBE_KINDS = ("pre_event", "final")


def setting_for(kind: str, record: Mapping[str, Any]) -> str:
    """The GT setting a probe of `kind` is scored against, read from one cell record.

    `kind` is `pre_event` or `final`. A stable record (`setting`) answers both kinds
    with the same value; an evolving record (`pre_setting` / `post_setting` /
    `event_acc`) answers `pre_event` with `pre_setting` and `final` with `post_setting`.
    Any other record shape, or an unknown kind, is an error — never a guess.
    """
    if kind not in PROBE_KINDS:
        raise ValueError(f"probe kind must be one of {PROBE_KINDS}, got {kind!r}")
    keys = set(record)
    if keys == {"setting"}:
        value = record["setting"]
    elif keys == {"pre_setting", "post_setting", "event_acc"}:
        value = record["pre_setting"] if kind == "pre_event" else record["post_setting"]
    else:
        raise ValueError(
            "GT record must be {setting} or {pre_setting, post_setting, event_acc}, "
            f"got keys {sorted(keys)}"
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GT setting for kind={kind!r} must be a non-empty string, got {value!r}")
    return value


def load_resolved_gt(path: Path) -> dict[str, Any]:
    """Read a resolved_gt.yaml and check every record has one of the two shapes."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"persona", "cells"}:
        raise ValueError(f"{path}: top level must be exactly {{persona, cells}}")
    for cell, record in data["cells"].items():
        for kind in PROBE_KINDS:
            try:
                setting_for(kind, record)
            except ValueError as exc:
                raise ValueError(f"{path}: cell {cell!r}: {exc}") from exc
    return data


@dataclass(frozen=True)
class PreEvent:
    probe_id: str
    cell: str
    event_acc: int
    pairs_with: str


def pre_events(resolved_gt: dict[str, Any]) -> dict[str, PreEvent]:
    """`pre_01..05` = the evolving cells of `resolved_gt.yaml` in `event_acc` order."""
    evolving = sorted(
        (record["event_acc"], cell)
        for cell, record in resolved_gt["cells"].items()
        if "event_acc" in record
    )
    return {
        f"pre_{index + 1:02d}": PreEvent(
            f"pre_{index + 1:02d}", cell, int(acc), final_probe_id(cell)
        )
        for index, (acc, cell) in enumerate(evolving)
    }
