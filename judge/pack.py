"""Judge input packer — test doc §4.2 / §4.3 (rows 18, 24–27 of §7).

Pure file movement plus extraction. No LLM, idempotent: running it twice over
the same run writes the same bytes. One finished probe step directory
(`steps/<probe_id>/`) becomes one `judge_input/<probe_id>/` directory holding
the two faces (§0-B2) and the files both faces share:

    meta.json               meta.yaml as JSON (carries the two IX version stamps)
    transcript.jsonl / .md  copied — IX face only (they carry `ix_render`)
    text_judge_pack.json    copied — the text face's only transcript
    tool_log.jsonl          copied — state/tool_log.jsonl
    ix_selection.json       the `ix_preference_selection` entries for the target attribute
    ground_truth.json       the cell's GT from the run's frozen resolved_gt copy
    scene.json              assembled from the draft, the outline and scene_start/

`index.json` at `judge_input/` lists every packed probe, says which files
belong to which reader (text face / IX face / aggregation), and carries the run
identity a score must trace back to (§4.2 "index.json").

The GT setting is read through `setting_for` and nothing else (§4.2 "取值只走
一个 helper"); the packer reads the frozen copy inside the run directory, never
the Authoring original.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from harness.judge_pack import scan_banned_keys, scan_banned_strings
from harness.orchestrator import SCENE_START_DIRNAME
from harness.transcript import load_events
from pa.agent.ix.action_types import tool_action_type
from pa.agent.ix.events import IX_SELECTION_PAYLOAD_TYPE
from scripts.resolved_gt import setting_for
from state_server.store import _CAPABILITY_MARKER_KEYS

from data.personas.coverage import TAXONOMY

JUDGE_INPUT_DIRNAME = "judge_input"
INDEX_FILENAME = "index.json"
PROBE_STEP_KINDS = ("pre_event_probe", "final_probe")
#: RunStep.kind → the `kind` argument `setting_for` takes.
PROBE_KIND_BY_STEP_KIND = {"pre_event_probe": "pre_event", "final_probe": "final"}

#: Which reader gets which file (§4.2 "输入包分两层"). The packer only writes
#: what it derives; `rubric.md`, `render.html` and the `baseline_*` files are
#: other components' outputs and are not listed until they exist.
FILE_GROUPS: dict[str, tuple[str, ...]] = {
    "text_face": (
        "meta.json", "text_judge_pack.json", "tool_log.jsonl",
        "ground_truth.json", "scene.json",
    ),
    "ix_face": (
        "meta.json", "transcript.jsonl", "transcript.md", "tool_log.jsonl",
        "ground_truth.json",
    ),
    "aggregation": ("ix_selection.json",),
}

#: Step files copied verbatim: (published name, source relative to the step dir).
COPIED_FILES: tuple[tuple[str, str], ...] = (
    ("transcript.jsonl", "transcript.jsonl"),
    ("transcript.md", "transcript.md"),
    ("text_judge_pack.json", "text_judge_pack.json"),
    ("tool_log.jsonl", "state/tool_log.jsonl"),
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return dict(data)


# --- meta.json ---------------------------------------------------------------


def build_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """meta.yaml → meta.json. The two IX version stamps must be present (§4.2)."""
    for key in ("ix_render_schema_version", "ix_contract_version"):
        if not meta.get(key):
            raise ValueError(f"meta.yaml lacks {key}; a probe without a rendered turn cannot be judged")
    return json.loads(json.dumps(dict(meta), default=str))


# --- ix_selection.json -------------------------------------------------------


def build_ix_selection(
    pa_toolcalls: Mapping[str, Any], *, probe_id: str, target_attribute: str, context: str
) -> dict[str, Any]:
    """The explicit-selection face's raw material (§4.2 "ix_selection.json").

    Filters first, then takes: only `type == "ix"` calls whose `payload_type`
    is `ix_preference_selection` and whose `attribute` is the target count. The
    per-turn payload entries (`IX_render_*` / `IX_continue_*`, empty `setting`)
    and the wrapper's `ix_event` control records never reach `selections[]`.
    """
    selections: list[dict[str, Any]] = []
    for beat in pa_toolcalls.get("beats") or []:
        for call in beat.get("calls") or []:
            if call.get("type") != "ix" or call.get("ix_event"):
                continue
            if call.get("payload_type") != IX_SELECTION_PAYLOAD_TYPE:
                continue
            if call.get("attribute") != target_attribute:
                continue
            selections.append(
                {
                    "beat": beat.get("beat"),
                    "call_index": call.get("call_index"),
                    "attribute": call.get("attribute"),
                    "setting": call.get("setting"),
                    "evidence": call.get("evidence"),
                    "application": call.get("application"),
                }
            )
    settings = {entry["setting"] for entry in selections}
    if len(settings) > 1:
        # A11: one selection per attribute per session. Two different answers
        # is a wrapper contract breach, not something to pick from.
        raise ValueError(
            f"{probe_id}: {target_attribute} selected more than once with different "
            f"settings: {sorted(settings)}"
        )
    return {
        "probe_id": probe_id,
        "target_attribute": target_attribute,
        "context": context,
        "selections": selections,
        "selected_setting": selections[0]["setting"] if selections else None,
        "selection_count": len(selections),
        "no_selection": not selections,
    }


# --- ground_truth.json -------------------------------------------------------


def build_ground_truth(
    resolved_gt: Mapping[str, Any],
    persona_matrix: Mapping[str, Any],
    *,
    persona_id: str,
    cell: str,
    step_kind: str,
    source_path: str,
) -> dict[str, Any]:
    """The cell's GT (§4.2 "ground_truth.json").

    `setting` is `setting_for(kind, record)` on the frozen record; the
    `evolving` block is read straight off the record's shape. `confidence` /
    `source` come from the persona matrix cell (the frozen file does not carry
    them); `allowed_settings` is the taxonomy's selection domain.
    """
    context, attribute = cell.split("_", 1)
    if attribute not in TAXONOMY:
        raise ValueError(f"{cell}: {attribute!r} is not an active attribute")
    cells = resolved_gt.get("cells") or {}
    if cell not in cells:
        raise ValueError(f"frozen resolved_gt has no record for cell {cell!r}")
    record = cells[cell]
    kind = PROBE_KIND_BY_STEP_KIND[step_kind]
    matrix_cell = (persona_matrix.get("matrix") or {}).get(context, {}).get(attribute) or {}
    is_evolving = "event_acc" in record
    return {
        "persona_id": persona_id,
        "context": context,
        "attribute": attribute,
        "setting": setting_for(kind, record),
        "confidence": matrix_cell.get("confidence"),
        "source": matrix_cell.get("source"),
        "source_path": source_path,
        "allowed_settings": list(TAXONOMY[attribute]),
        "evolving": {
            "is_evolving": is_evolving,
            "pre_setting": record.get("pre_setting") if is_evolving else None,
            "post_setting": record.get("post_setting") if is_evolving else None,
        },
    }


# --- scene.json --------------------------------------------------------------


def _strip_markers(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in _CAPABILITY_MARKER_KEYS}


def _records_in(store_file: Path) -> list[dict[str, Any]]:
    """Every record-shaped object in a fixture store file, whatever its layout.

    A `.jsonl` store is one record per line; a `.json` store nests its records
    under some top-level list. Walking every dict is the one rule that covers
    both without a per-store table.
    """
    if store_file.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in store_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if any(str(key).endswith("id") for key in node):
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(json.loads(store_file.read_text(encoding="utf-8")))
    return found


def _locate_record(world_root: Path, ingredient: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve one outline `world[]` ingredient to the record it names at scene start."""
    path = ingredient.get("path")
    if path:
        document = world_root / path
        if not document.is_file():
            return None
        return {"path": path, "bytes": document.stat().st_size}
    store, record_id = ingredient.get("store"), ingredient.get("record_id")
    store_file = world_root / str(store)
    if not store or not record_id or not store_file.is_file():
        return None
    for record in _records_in(store_file):
        if any(str(key).endswith("id") and value == record_id for key, value in record.items()):
            return record
    return None


def _world_records(
    world_root: Path, ingredients: Iterable[Mapping[str, Any]], role: str
) -> list[dict[str, Any]]:
    """Outline ingredients of `role`, resolved at scene start, markers stripped.

    A record marked `unavailable: true` is excluded from the list outright
    (§7 row 24: it must not be in the denominator, and its `unavailable_reason`
    is something the PA never saw).
    """
    resolved: list[dict[str, Any]] = []
    for ingredient in ingredients:
        if ingredient.get("role") != role:
            continue
        record = _locate_record(world_root, ingredient)
        if record is None:
            raise ValueError(
                f"scene_start world has no record for ingredient {ingredient.get('ref')!r} "
                f"(store={ingredient.get('store')!r} record_id={ingredient.get('record_id')!r} "
                f"path={ingredient.get('path')!r})"
            )
        if record.get("unavailable") is True:
            continue
        resolved.append(
            {
                "ref": ingredient.get("ref"),
                "store": ingredient.get("store"),
                "record_id": ingredient.get("record_id"),
                "path": ingredient.get("path"),
                "record": _strip_markers(record),
            }
        )
    return resolved


def _withheld_required_facts(ingredients: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The outline's `role: missing_field` entries — the required facts the ask withholds.

    Each carries its `$task_facts` key and the author's `note` naming the fact.
    World records hold at least two candidate values for each, so no single
    record resolves one: the entry stays a name, not a scene-start lookup. A
    probe that withholds nothing gets an empty list.
    """
    return [
        {"task_fact": ingredient.get("task_fact"), "note": ingredient.get("note")}
        for ingredient in ingredients
        if ingredient.get("role") == "missing_field"
    ]


def turn_index(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Number the `pa_turn` events by ascending `t` (§7 row 26: no `exchange` on probes)."""
    pa_turns = sorted(
        (event for event in events if event.get("kind") == "pa_turn"),
        key=lambda event: int(event["t"]),
    )
    return [
        {"turn": index, "t": event["t"], "beat": event.get("beat")}
        for index, event in enumerate(pa_turns, start=1)
    ]


def outreach_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`outreach_action` / `outreach_message` from the transcript's `tool_events[]` (§7 row 27)."""
    found: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "pa_turn":
            continue
        for tool_event in event.get("tool_events") or []:
            if not tool_event.get("ix") or tool_event.get("attribute") != "proactive_outreach":
                continue
            found.append(
                {
                    "t": event.get("t"),
                    "beat": event.get("beat"),
                    "outreach_decision": event.get("outreach_decision"),
                    "outreach_delivered": event.get("outreach_delivered"),
                    "outreach_action": tool_event.get("outreach_action"),
                    "outreach_message": tool_event.get("outreach_message"),
                }
            )
    return found


def tool_actions(tool_log: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each `tool_log` row classified with `tool_action_type` (§7 row 25)."""
    return [
        {"t": row.get("t"), "tool": row.get("tool"), "action_type": tool_action_type(str(row.get("tool")))}
        for row in tool_log
    ]


def build_scene(
    *,
    probe_id: str,
    draft: Mapping[str, Any],
    outline_entry: Mapping[str, Any],
    world_root: Path,
    events: list[dict[str, Any]],
    tool_log: list[dict[str, Any]],
) -> dict[str, Any]:
    """scene.json (§4.2 "scene.json 形状与来源").

    (a) `triggers[]` from the draft's `proactive_trigger`; (b) `core_task` from
    the active beat's `user_instruction` (the one beat carrying
    `active_skills`, §2.1.3); (c) `world_state` from the outline's
    `candidate` / `unresolved` ingredients resolved against the scene-start
    world; (d) `withheld_required_facts` from the outline's `missing_field`
    ingredients — the author's list of required facts the ask withholds (the
    records hold candidate values for each, never the answer), the coverage
    reference `information_elicitation`'s `Structured` row is judged
    against; (e) `familiarity_signals` from that same beat, `null` unless set.
    The three derived views the §7 rows ask the packer for — turn numbers,
    outreach events, tool action types — ride along under their own keys.

    A triggered probe (`proactive_outreach`, §2.1.2b) is the one shape with two
    active beats: its beats are the two openings and the runtime plays whichever
    the PA's own decision selects. There is no user question on it, so the task
    the PA acted on is the arrival — `core_task` is the trigger's `detail`.
    """
    active = [beat for beat in draft.get("beats") or [] if beat.get("active_skills")]
    trigger = draft.get("proactive_trigger")
    if trigger and len(active) != 2:
        raise ValueError(
            f"{probe_id}: a triggered probe measures on whichever opening plays, so both "
            f"are active beats; found {len(active)}"
        )
    if not trigger and len(active) != 1:
        raise ValueError(f"{probe_id}: expected exactly one active beat, found {len(active)}")
    triggers = (
        [{key: trigger[key] for key in ("trigger_id", "background", "detail", "user_impact")}]
        if trigger
        else []
    )
    ingredients = outline_entry.get("world") or []
    return {
        "probe_id": probe_id,
        "core_task": trigger["detail"] if trigger else active[0]["user_instruction"],
        "triggers": triggers,
        "world_state": {
            "feasible_options": _world_records(world_root, ingredients, "candidate"),
            "pending_items": _world_records(world_root, ingredients, "unresolved"),
        },
        "withheld_required_facts": _withheld_required_facts(ingredients),
        "familiarity_signals": active[0].get("familiarity_signals"),
        "turns": turn_index(events),
        "outreach": outreach_events(events),
        "tool_actions": tool_actions(tool_log),
    }


# --- per-probe / per-run entry points ---------------------------------------


def verify_text_judge_pack(pack: Mapping[str, Any]) -> None:
    """Re-run the two banned scans on the persisted pack before publishing it."""
    leaks = scan_banned_keys(pack) + scan_banned_strings(pack)
    if leaks:
        raise ValueError("text_judge_pack.json carries the IX face: " + "; ".join(leaks))


def _probe_steps(run_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [step for step in run_plan.get("steps") or [] if step.get("kind") in PROBE_STEP_KINDS]


def _final_pinned_checkpoint_id(run_plan: Mapping[str, Any]) -> str | None:
    """The accumulation step nearest before the first `final_probe` (the final pin)."""
    latest: str | None = None
    for step in run_plan.get("steps") or []:
        if step.get("kind") == "accumulation":
            latest = step["step_id"]
        elif step.get("kind") == "final_probe":
            return latest
    return None


def _outline_entry(outline: Mapping[str, Any], probe_id: str) -> dict[str, Any]:
    for entry in outline.get("probes") or []:
        if entry.get("id") == probe_id:
            return dict(entry)
    raise ValueError(f"probe outline has no entry {probe_id!r}")


def pack_probe(
    run_dir: Path,
    step: Mapping[str, Any],
    *,
    resolved_gt: Mapping[str, Any],
    outline: Mapping[str, Any],
    project_root: Path,
) -> Path:
    """Build `judge_input/<probe_id>/` from `steps/<probe_id>/`. Returns the directory."""
    probe_id = step["step_id"]
    step_dir = run_dir / "steps" / probe_id
    target = run_dir / JUDGE_INPUT_DIRNAME / probe_id
    target.mkdir(parents=True, exist_ok=True)

    for published, relative in COPIED_FILES:
        source = step_dir / relative
        if not source.is_file():
            raise FileNotFoundError(f"{probe_id}: missing step artifact {relative}")
        shutil.copyfile(source, target / published)
    verify_text_judge_pack(json.loads((target / "text_judge_pack.json").read_text(encoding="utf-8")))

    meta = _load_yaml(step_dir / "meta.yaml")
    _write_json(target / "meta.json", build_meta(meta))

    persona_id = step["persona_id"]
    cell = step["target_cell"]
    context, attribute = cell.split("_", 1)
    pa_toolcalls = json.loads((step_dir / "pa_toolcalls.json").read_text(encoding="utf-8"))
    _write_json(
        target / "ix_selection.json",
        build_ix_selection(pa_toolcalls, probe_id=probe_id, target_attribute=attribute, context=context),
    )

    matrix_path = Path("data/personas") / f"{persona_id}_preferences.yaml"
    _write_json(
        target / "ground_truth.json",
        build_ground_truth(
            resolved_gt,
            _load_yaml(project_root / matrix_path),
            persona_id=persona_id,
            cell=cell,
            step_kind=step["kind"],
            source_path=str(matrix_path),
        ),
    )

    script_path = Path(step["script_path"])
    draft = _load_yaml(script_path if script_path.is_absolute() else project_root / script_path)
    events = load_events(step_dir / "transcript.jsonl")
    tool_log = load_events(step_dir / "state" / "tool_log.jsonl")
    _write_json(
        target / "scene.json",
        build_scene(
            probe_id=probe_id,
            draft=draft,
            outline_entry=_outline_entry(outline, probe_id),
            world_root=step_dir / SCENE_START_DIRNAME / "world",
            events=events,
            tool_log=tool_log,
        ),
    )
    return target


def pack_run(
    run_dir: Path | str,
    *,
    resolved_gt_path: Path | None = None,
    outline_path: Path | None = None,
    project_root: Path | str = ".",
    only: Iterable[str] | None = None,
) -> Path:
    """Pack every finished probe step of a run; write `judge_input/index.json`.

    A probe whose `steps/<id>/` does not exist yet is skipped (it has not run);
    one that exists but is incomplete raises. Returns the index path.
    """
    run_dir = Path(run_dir)
    project_root = Path(project_root)
    run_plan = _load_yaml(run_dir / "run_plan.yaml")
    persona_id = run_plan["persona_id"]
    resolved_gt_path = resolved_gt_path or run_dir / "resolved_gt.yaml"
    outline_path = outline_path or project_root / "Authoring/probe/outline" / persona_id / "outline.yaml"
    resolved_gt = _load_yaml(resolved_gt_path)
    outline = _load_yaml(outline_path)
    wanted = set(only) if only is not None else None

    probes: list[dict[str, Any]] = []
    for step in _probe_steps(run_plan):
        probe_id = step["step_id"]
        if wanted is not None and probe_id not in wanted:
            continue
        step_dir = run_dir / "steps" / probe_id
        if not step_dir.is_dir():
            continue
        target = pack_probe(
            run_dir, step, resolved_gt=resolved_gt, outline=outline, project_root=project_root
        )
        meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
        probes.append(
            {
                "probe_id": probe_id,
                "kind": step["kind"],
                "cell": step["target_cell"],
                "before_acc_num": step.get("before_acc_num"),
                "ix_render_schema_version": meta["ix_render_schema_version"],
                "ix_contract_version": meta["ix_contract_version"],
                "files": {group: list(names) for group, names in FILE_GROUPS.items()},
            }
        )

    index = {
        "run_id": run_plan["run_id"],
        "persona_id": persona_id,
        "pa_model": run_plan.get("pa_model"),
        "memory_condition": run_plan.get("memory_condition"),
        "final_pinned_checkpoint_id": _final_pinned_checkpoint_id(run_plan),
        "probes": probes,
    }
    index_path = run_dir / JUDGE_INPUT_DIRNAME / INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(index_path, index)
    return index_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory holding run_plan.yaml and steps/<probe_id>/ "
        "(workspace/runs/<model>/<condition>/<persona>/<run_id>). judge_input/ is written inside it.",
    )
    parser.add_argument(
        "--resolved-gt",
        default=None,
        help="Path to the run's frozen resolved_gt.yaml copy (top level {persona, cells}). "
        "Default: <run-dir>/resolved_gt.yaml.",
    )
    parser.add_argument(
        "--outline",
        default=None,
        help="Probe outline YAML (meta + probes[] with world[] ingredients). "
        "Default: <project-root>/Authoring/probe/outline/<persona>/outline.yaml.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Directory the plan's relative script_path and data/personas/ resolve against. Default: cwd.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated probe ids to pack (fin_NN / pre_NN); default packs every finished probe.",
    )
    args = parser.parse_args(argv)
    index_path = pack_run(
        args.run_dir,
        resolved_gt_path=Path(args.resolved_gt) if args.resolved_gt else None,
        outline_path=Path(args.outline) if args.outline else None,
        project_root=args.project_root,
        only=[item.strip() for item in args.only.split(",") if item.strip()] if args.only else None,
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    print(f"wrote {index_path}: {len(index['probes'])} probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
