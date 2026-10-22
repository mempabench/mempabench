# Adding a memory backend

MemPABench has no plug-in registry: a new memory system is a code change in a
fixed set of places, listed below. Pick one name for it, lowercase with
underscores (for example `my_memory`), and use exactly that string everywhere:
in the registry, in the checkpoint tables, as `--memory-condition`, and in the
run directory path. The steps refer to functions and constants by name; open
the files at the names given.

There are two shapes:

- **Shape A, external service** (recommended): the backend keeps its memory in
  a service outside the run directory, as `mem0`, `simple_rag` and `memos` do.
- **Shape B, state in files**: the backend keeps all of its state in files
  inside the run directory, as `nanobot_file_memory` does.

## What the harness requires of a backend

1. **The adapter protocol.** Implement `MemoryAdapter` (`memory/adapter.py`):
   the attributes `backend_name` (your condition name) and
   `prompt_history_eviction_policy`, and the methods `setup_scope`, `set_mode`,
   `record_event`, `retrieve`, `format_context`, `reset_scope` and `health`.
   Use `PromptHistoryEvictionPolicy.after_adapter_commit` (the PA drops a turn
   from its prompt only after your backend has committed it) or
   `without_persistence`. Do not use `after_file_consolidation`: the PA turns
   on nanobot's own file memory from that policy, not from the condition name.
2. **Read-only mode.** Probes put the adapter into read-only mode through
   `set_mode` (`MemoryRuntime.apply_mode` in `harness/memory_runtime.py`).
   In that mode `record_event` must write nothing, as
   `memory/conditions/nanobot_file_memory.py` does.
3. **Readiness at checkpoints.** Before each checkpoint the harness waits
   until every write has landed (`MemoryRuntime.await_checkpoint_ready`).
   Either report `ConsistencyModel.immediate` from `health`, or implement
   `SupportsReadiness` (`memory/adapter.py`).
4. **Checkpoint and branch.** Runs resume from checkpoints, and every probe
   reads a copy of the memory as it was at its cut point. This is where the
   condition name is wired in by hand, in the steps below.

## Shape A: external-service backend

Follow the `mem0` and `memos` code paths as the pattern.

1. **Adapter.** Write `memory/backends/<name>_adapter.py` implementing
   `MemoryAdapter`, honouring read-only mode and returning a meaningful
   `health`.
2. **Registry.** Add one `reg.register("<name>", <YourAdapter>, kind="backend")`
   line to `default_registry()` in `memory/registry.py`.
3. **Checkpoint capabilities.** Add one entry to `_BACKEND_CAPABILITIES` in
   `harness/memory_runtime.py`, for example
   `"<name>": MemoryRuntimeCapabilities(True, True, "scope_clone")`. A name
   missing from this table means "cannot restore", and resume and rollback
   are refused.
4. **Client and adapter options.** In `_execute_benchmark_run`
   (`scripts/run_orchestrator.py`), add an `if args.memory_condition == "<name>":`
   block next to the existing per-condition blocks that build the client and
   fill `memory_adapter_options`, and set `memory_backend_namespace` to the
   namespace this run owns (a collection, a cube, a database). The adapter is
   then created by `default_registry().create(args.memory_condition, **memory_adapter_options)`.
5. **Checkpoint export, restore and branch.** In `CheckpointManager`
   (`harness/checkpoint.py`) add an arm for your name to
   `_copy_backend_state` (export the namespace into the checkpoint),
   `_restore_backend_state` (put it back for a resume or rollback) and
   `_restore_backend_state_into_branch_unlocked` (restore it into a namespace
   of the branch's own, from `branch_namespace_for`), and to the module-level
   `_unreachable_external_refs` (which snapshot references a checkpoint must
   carry). Without these arms an unknown condition raises
   `IncompleteCheckpointError` or `NotImplementedError`. Also teach
   `CheckpointManager.release_branch_namespace` to delete a branch namespace,
   so finished probes do not leave one behind.
6. **Probe-branch adapter.** Add an `elif memory_condition == "<name>":` arm
   to `_build_branch_memory_adapter` (`scripts/run_orchestrator.py`) that
   points the adapter options at the branch's namespace. Without it a probe
   on your backend stops with "A probe branch on this memory condition has no
   adapter of its own".
7. **Only if the backend uses its own LLM or embeddings.** Add the name to
   `MEMORY_SIDE_LM_CONDITIONS` and/or `EMBEDDING_MEMORY_CONDITIONS` in
   `harness/run_plan.py`, so the plan freezes the memory-side model (passed as
   `--memory-side-model`, with a `memory_extraction` role validated in
   `configs/model_request_policies.yaml`) and the embedding settings.

## Shape B: state-in-files backend

1. Do steps 1–3 of Shape A.
2. Keep all state under `pa_workspace/<persona>/<name>/` in the run
   directory. That directory is already copied into every checkpoint and every
   probe branch; `backend_state/` is not copied for file-based conditions.
3. Add the name to `LOCAL_MEMORY_CONDITIONS` in `harness/checkpoint.py`. It is
   imported from there by `harness/orchestrator.py` and
   `scripts/run_orchestrator.py`.
4. In `_execute_benchmark_run` (`scripts/run_orchestrator.py`), pass the state
   directory to the adapter through `memory_adapter_options`; the
   `nanobot_file_memory` block that sets `memory_adapter_options["memory_dir"]`
   is the precedent.
5. **Build the probe-branch adapter yourself.** For a condition in
   `LOCAL_MEMORY_CONDITIONS` no branch factory is set, so a probe gets no
   adapter. In the probe set-up of `Orchestrator` (`harness/orchestrator.py`,
   where the branch adapter is chosen for `no_memory`, `oracle_preference` and
   the rest), add an arm for your name that builds your adapter over
   `branch_dir / "pa_workspace" / <persona> / "<name>"`, and extend the
   "no adapter" guard below it so your name may not run with `None`. Keep the
   arm specific to your name, so `nanobot_file_memory` still gets no adapter.

   Skipping this step fails silently: with no adapter, the probe PA falls back
   to nanobot's own file memory, every probe measures File memory instead of
   your backend, and nothing reports it.

## Smoke test: one probe, one judgement

This checks that your backend writes during a session and that a probe reads
from it. It runs `user_a`'s first everyday session, the probe `pre_01` (cut
just before slot 15) and the first preference-change event (slot 15). It makes
paid OpenRouter calls, judges one probe and computes no metrics.

1. Start your service and export `.env` (see the README's Quick start), then
   run:

   ```bash
   uv run mempabench run --persona-id user_a --run-id my_memory_smoke_01 \
     --acc-list 1,15 --memory-condition my_memory \
     --pa-model deepseek/deepseek-v4-flash
   # add --memory-side-model <model> if your backend is in MEMORY_SIDE_LM_CONDITIONS
   RUN=workspace/runs/deepseek_deepseek-v4-flash/my_memory/user_a/my_memory_smoke_01
   ```

   `mempabench plan` is not part of this test: its plan check accepts only the
   built-in conditions.

2. **Writes.** The run-level audit holds your backend's writes from the two
   sessions:

   ```bash
   jq -c 'select(.backend == "my_memory") | {step_id, operation, status, native_ids}' "$RUN/memory_io.jsonl"
   ```

   Expect write rows for the first session (its `step_id`) with `status: "succeeded"`.

3. **The probe's reads.** The probe's own audit holds what it retrieved on its
   branch:

   ```bash
   jq -c '{step_id, backend, operation, status, result_count}' "$RUN/steps/pre_01/memory_io.jsonl"
   ```

   Expect `backend: "my_memory"`, a read (your adapter's
   `trace.native_operation`, or `retrieve`) with `status: "succeeded"`, and a
   non-zero `result_count` once the first session has written something. No
   row from your backend means the probe ran without your adapter: check
   step 6 (Shape A) or step 5 (Shape B).

4. **Judge that one probe.**

   ```bash
   uv run python -m judge.pack --run-dir "$RUN" --only pre_01
   uv run python -m judge.evaluator --judge-input "$RUN/judge_input/pre_01" --model google/gemini-3.8-flash
   ```

   Run both from the repository root. The judgement is written to
   `$RUN/judge_output/llm__google_gemini-3.8-flash/pre_01.json`.

The smoke run holds one of the 33 probes, so `mempabench score` refuses it;
that is expected. Remove the run and its backend state afterwards (README,
"Cleaning up"). A full run with `--acc-range 1-108`, followed by
`mempabench score`, then gives your backend's PSA and BRS for that user.
