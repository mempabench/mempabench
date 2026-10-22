"""Mint a frozen RunPlan and stop -- nothing is started.

`scripts/run_orchestrator.py` has no plan-only switch: the moment it writes
`run_plan.yaml` (`plan.dump_yaml(plan_path)`) it has already run
`judge_preflight()` against the judge endpoint, and it goes straight on to the
ledger, the checkpoint manager and the memory adapter. This diagnostic takes
the same arguments, builds the plan through the same builder call with the
same runtime-config helpers, writes it under `--out-dir` (never the formal
runs root `harness/run_layout.py::runs_root` names), reads it back, and prints
what was frozen.

Deliberately NOT done here: `judge_preflight`, `RunLedger`, checkpoints, any
memory adapter, any Qdrant / judge / LLM connection. The mem0 runtime config a
`simple_rag` or `mem0` plan freezes is built by
`run_orchestrator._effective_mem0_runtime_config` -- constants, the vendored
mem0 source hash and the tiktoken encoding name -- and no client is created
from it. `_qdrant_endpoint_from_environment` only reads `QDRANT_HOST` /
`QDRANT_PORT`; nothing connects to them.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from harness import run_plan_builder
from harness.run_layout import runs_root
from harness.run_plan import (
    DEFAULT_SIMULATOR_MODEL,
    RunPlan,
    SimulatorRuntimeConfig,
    load_run_plan,
)
from pa.agent.ix.render_types import IX_CONTRACT_VERSION
# Importing the orchestrator module runs its `load_dotenv(PROJECT_ROOT/.env)`:
# values land in the environment the way they do for a real launch, and
# nothing here prints them.
from scripts import run_orchestrator

#: memos needs `--memos-api-base`, which is outside this diagnostic's
#: argument subset.
SUPPORTED_CONDITIONS = frozenset(
    {"nanobot_file_memory", "no_memory", "simple_rag", "mem0"}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--persona-id", default="user_a")
    acc_selection = parser.add_mutually_exclusive_group()
    acc_selection.add_argument("--acc-range", default=None)
    acc_selection.add_argument(
        "--acc-list",
        default=None,
        type=run_orchestrator._acc_names_from_list,
        metavar="N|A-B,...",
    )
    parser.add_argument("--memory-condition", default="nanobot_file_memory")
    parser.add_argument("--pa-model", default=run_orchestrator.DEFAULT_PA_MODEL)
    parser.add_argument("--simulator-model", default=DEFAULT_SIMULATOR_MODEL)
    parser.add_argument("--consolidation-model", default=None)
    parser.add_argument("--memory-side-model", default=None)
    parser.add_argument(
        "--pin-all-accumulations",
        action="store_true",
        help=(
            "accepted for command parity only: pinning is a checkpoint "
            "decision at run time and is not frozen into the plan"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="where to write <run_id>/run_plan.yaml (never a formal runs root)",
    )
    return parser


def _guarded_runs_roots() -> set[Path]:
    """Every formal runs tree this diagnostic must stay out of.

    `runs_root()` is the one that matters: it is where a real launch writes, so
    it is the tree that must never take a plan nothing is going to run. The two
    project roots are kept alongside it because a test points them at a
    synthetic tree and expects that tree's `workspace/runs` guarded too -- and
    because a checkout whose root has been moved still has the old location on
    disk until the data is migrated.
    """
    guarded = {runs_root()}
    for root in {run_orchestrator.PROJECT_ROOT, run_plan_builder.PROJECT_ROOT}:
        guarded.add((Path(root) / "workspace/runs").resolve())
    return guarded


def _refuse_workspace_runs(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    for runs in _guarded_runs_roots():
        if resolved == runs or resolved.is_relative_to(runs):
            raise ValueError(
                "this diagnostic never writes into a formal runs root: "
                f"out_dir={out_dir} runs_root={runs}"
            )


def mint(args: argparse.Namespace) -> tuple[RunPlan, Path]:
    """Build, write, and read back the plan. Nothing runs."""
    if args.memory_condition not in SUPPORTED_CONDITIONS:
        raise ValueError(
            "memory_condition outside this diagnostic's argument subset: "
            f"value={args.memory_condition!r} "
            f"supported={sorted(SUPPORTED_CONDITIONS)}"
        )
    _refuse_workspace_runs(args.out_dir)
    script_names = (
        args.acc_list
        if args.acc_list
        else run_orchestrator._acc_names(
            args.acc_range or run_orchestrator.DEFAULT_ACC_RANGE
        )
    )
    memory_condition = args.memory_condition
    qdrant_host, qdrant_port = run_orchestrator._qdrant_endpoint_from_environment(
        memory_condition
    )
    plan = run_plan_builder._build_frozen_run_plan(
        run_id=args.run_id,
        persona_id=args.persona_id,
        memory_condition=memory_condition,
        pa_model=args.pa_model,
        script_names=script_names,
        with_probes=run_plan_builder.persona_has_probe_line(args.persona_id),
        assistant_llm_provider_name=run_orchestrator.ASSISTANT_LLM_PROVIDER_NAME,
        assistant_llm_api_base=run_plan_builder.OPENROUTER_API_BASE,
        consolidation_model=run_orchestrator._effective_consolidation_model(
            memory_condition, args.consolidation_model
        ),
        memory_side_model=run_orchestrator._effective_memory_side_model(
            memory_condition, args.memory_side_model
        ),
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        mem0_runtime_config=run_orchestrator._effective_mem0_runtime_config(
            memory_condition
        ),
        simulator_runtime=SimulatorRuntimeConfig(
            simulator_model=args.simulator_model
        ),
    )
    run_dir = args.out_dir / args.run_id
    plan_path = plan.dump_yaml(run_dir / "run_plan.yaml")
    if plan.resolved_gt is not None:
        run_plan_builder.freeze_resolved_gt_copy(plan, run_dir)
    loaded = load_run_plan(plan_path, project_root=run_plan_builder.PROJECT_ROOT)
    return loaded, plan_path


def _print_summary(plan: RunPlan, plan_path: Path, args: argparse.Namespace) -> None:
    out = sys.stdout
    print("mint_run_plan_only: plan minted and read back; nothing started", file=out)
    print(
        f"run_id={plan.run_id} persona_id={plan.persona_id} "
        f"memory_condition={plan.memory_condition} pa_model={plan.pa_model} "
        f"simulator_model={plan.simulator_runtime.simulator_model}",
        file=out,
    )
    print(f"plan_path={plan_path}", file=out)
    if plan.resolved_gt is None:
        print("resolved_gt=none (persona has no probe line)", file=out)
    else:
        print(f"resolved_gt_copy={plan_path.parent / plan.resolved_gt.path}", file=out)
    print(f"ix_contract_version={plan.ix_contract_version} (code: {IX_CONTRACT_VERSION})", file=out)
    print(f"memory_pool_version={plan.memory_pool_version}", file=out)
    kinds = Counter(step.kind for step in plan.steps)
    print(
        f"steps total={len(plan.steps)} "
        + " ".join(f"{kind}={kinds.get(kind, 0)}" for kind in (
            "accumulation", "forced_event", "pre_event_probe", "final_probe"
        )),
        file=out,
    )
    print(
        f"probe_steps={kinds.get('pre_event_probe', 0) + kinds.get('final_probe', 0)} "
        f"acc_steps={kinds.get('accumulation', 0) + kinds.get('forced_event', 0)}",
        file=out,
    )
    print(f"consolidation_model={plan.consolidation_model or '-'}", file=out)
    print(f"memory_side_model={plan.memory_side_model or '-'}", file=out)
    print(
        f"qdrant_host={plan.qdrant_host or '-'} qdrant_port={plan.qdrant_port} "
        "(from environment; not connected)",
        file=out,
    )
    if plan.mem0_runtime_config is not None:
        mem0 = plan.mem0_runtime_config.model_dump(mode="json")
        print(
            "mem0_runtime_config: "
            f"mem0_version={mem0.get('mem0_version')} "
            f"mem0_submodule_commit={mem0.get('mem0_submodule_commit')} "
            f"enable_entity_memory={mem0.get('enable_entity_memory')} "
            "(constructed; no client created)",
            file=out,
        )
    if args.pin_all_accumulations:
        print(
            "pin_all_accumulations=accepted (runtime checkpoint decision; "
            "not part of the plan)",
            file=out,
        )
    print(
        "skipped on purpose: judge_preflight, RunLedger, checkpoint manager, "
        "memory adapter, Qdrant, every LLM/judge endpoint",
        file=out,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan, plan_path = mint(args)
    _print_summary(plan, plan_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
