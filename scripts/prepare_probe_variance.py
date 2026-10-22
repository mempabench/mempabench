"""Prepare independent full-plan Probe repeats from finished source runs.

No model calls, backend restores, or changes to an original run are performed.
Input: {settings: [{id, source: {run_dir}, arm?: {memory_condition, pa_model}}]}.
Without `arm` a setting repeats its source's own condition and PA model, in the
rounds `repeat.rounds` lists. With `arm` (memory_condition no_memory or
oracle_preference) it is a reference arm: the same source timeline and pins,
run under that condition and PA model, in round 1 and those rounds. An
existing repeat plan is left as it is; validation compares it with the source.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import yaml

from harness.probe_repeat_source import (
    repeat_base_plan, repeat_rounds, repeat_run_id, write_repeat_source,
)
from harness.run_plan import RunPlan
from scripts.run_probe_variance import SCHEMA, validate_manifest


def prepare(source_manifest: Path, *, workspace_root: Path, state_root: Path,
            output: Path) -> dict:
    source_data = json.loads(source_manifest.read_text())
    result = {"schema": SCHEMA, "workspace_root": str(workspace_root.resolve()),
              "state_root": str(state_root.resolve()), "settings": []}
    for setting in source_data["settings"]:
        source = {"run_dir": setting["source"]["run_dir"]}
        arm = setting.get("arm")
        source_dir = Path(source["run_dir"])
        if not source_dir.is_absolute():
            raise ValueError("Source run_dir must be absolute")
        raw = yaml.safe_load((source_dir / "run_plan.yaml").read_text())
        plan = RunPlan.model_validate(raw)
        base = repeat_base_plan(raw, arm)
        produced = RunPlan.model_validate(base)
        row = {"id": setting["id"], "source": source, "repeats": []}
        if arm is None:
            row["repeat01"] = {"run_dir": str(source_dir)}
        else:
            row["arm"] = arm
        for repeat in repeat_rounds(arm):
            run_id = repeat_run_id(plan.run_id, arm, repeat)
            directory = workspace_root.resolve() / "probe_variance" / run_id
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "run_plan.yaml"
            new = deepcopy(base)
            new["run_id"] = run_id
            new["pa_session_key"] = plan.main_pa_session_key()
            for step, original in zip(new["steps"], plan.steps):
                if original.kind in {"pre_event_probe", "final_probe"}:
                    pin = plan.probe_source_step_id(original, plan.pinned_step_ids())
                    if pin is None:
                        raise ValueError(f"No original pin for {original.step_id}")
                    step["borrow_start_from"] = f"{plan.run_id}/{pin}"
            RunPlan.model_validate(new)
            if not target.exists():
                target.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True))
            gt = directory / "resolved_gt.yaml"
            if not gt.exists():
                shutil.copyfile(source_dir / "resolved_gt.yaml", gt)
            write_repeat_source(directory, source_dir, repeat, arm)
            argv = ["--plan", str(target), "--persona-id", plan.persona_id,
                    "--memory-condition", produced.memory_condition, "--pa-model", produced.pa_model,
                    "--borrowed-runs-root", str(source_dir.parent),
                    "--simulator-model", plan.simulator_runtime.simulator_model]
            for flag, value in (("--consolidation-model", produced.consolidation_model),
                                ("--memory-side-model", produced.memory_side_model)):
                if value:
                    argv += [flag, value]
            if produced.memos_runtime_config is not None:
                argv += ["--memos-api-base", produced.memos_runtime_config.api_base]
            row["repeats"].append({"repeat": repeat, "argv": argv, "source": source})
        result["settings"].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError("Existing dispatch manifest differs; refusing overwrite")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return validate_manifest(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare(args.source_manifest, workspace_root=args.workspace_root,
                       state_root=args.state_root, output=args.output)
    print(json.dumps({"settings": len(manifest["settings"]), "manifest": str(args.output.resolve()),
                      "execution": "not_started"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
