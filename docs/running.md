# Running MemPABench

The details behind the README's [Quick start](../README.md#quick-start).

## Getting the user data

This downloads the four users from Hugging Face ([`mempabench/mempabench-personas`](https://huggingface.co/datasets/mempabench/mempabench-personas)) at the revision pinned in the code (`DATASET_REVISION`) and places their files at the paths the harness reads (`Authoring/`, `data/personas/`, `data/state_fixtures/`, `simulator/profiles/`), all gitignored. It overwrites files already there, and stops and names any file listed in `mempabench/persona_data_files.tsv` that the download lacks.

## Memory services

| Condition (`--memory-condition`) | Paper name | Service |
|---|---|---|
| `nanobot_file_memory` | File memory | none |
| `simple_rag`, `mem0` | Simple RAG, Mem0 | Qdrant: `docker compose --env-file .env -f deploy/mem0/compose.yaml up -d --wait` |
| `memos` | MemOS | MemOS API, Neo4j, Qdrant: `docker compose --env-file .env -f deploy/memos/compose.yaml up -d --build --wait` (`--build` builds the image from `memory_raw_repos/MemOS` on first start) |
| `no_memory`, `oracle_preference` | No Memory, Oracle | none |

`docker compose … down -v` deletes the databases and the checkpoint snapshots in them.

For `mem0` and `simple_rag`, set `MEMPA_QDRANT_RESTORE_LOCK_PATH` in `.env` to an absolute lock-file path, e.g. `<repository>/workspace/locks/qdrant_6333.lock` (create the folder first), and export `.env` again.

## Checking a plan without model calls

```bash
uv run mempabench plan --persona-id user_a --run-id plan_check_user_a \
  --acc-range 1-108 --memory-condition nanobot_file_memory \
  --pa-model deepseek/deepseek-v4-flash \
  --consolidation-model deepseek/deepseek-v4-flash \
  --out-dir /tmp/mempabench_plan_check
```

This freezes the whole timeline of `user_a` (99 sessions and 33 probes) into `/tmp/mempabench_plan_check/plan_check_user_a/run_plan.yaml` without any model, judge or database call. `--out-dir` must lie outside the runs root. `plan` supports every condition except `memos`.

## Run options and the paper's models

To fit more models and tasks, each PA call and each File memory consolidation call may return up to 24,576 tokens (`budgets.pa_max_tokens` in `configs/run_config.yaml`), with a 600 s request timeout (`MEMPA_LLM_REQUEST_TIMEOUT_S`).

The paper's model flags per condition:

| Condition | Flags used in the paper |
|---|---|
| `nanobot_file_memory` | `--consolidation-model deepseek/deepseek-v4-flash` |
| `mem0` | `--memory-side-model deepseek/deepseek-v4-flash` |
| `memos` | `--memory-side-model openai/gpt-5.5`, the same model as `MEMOS_MEMORY_LM_MODEL`; the API comes from `MEMOS_API_BASE` (or `--memos-api-base`) |
| `simple_rag` | none (no memory-side model) |

The paper's PA backbones are `openai/gpt-5.6-sol`, `qwen/qwen3.8-27b`, `tencent/hy4-preview` and `deepseek/deepseek-v4-flash`. The simulated user (`google/gemini-3.8-flash`), the live judge (`google/gemma-4-31b-it`) and the offline judge (`google/gemini-3.8-flash`) are fixed; runs with another simulator or judge model are not comparable with the leaderboard.

## Scoring

Scoring needs all 33 probes; add `--not-applicable-exhausted` when a probe failed all its attempts (see [Long runs](#long-runs-interruptions-and-resume)). It judges each with `google/gemini-3.8-flash` and writes `metrics/llm__google_gemini-3.8-flash.json` and a Markdown copy into the run directory. For a single run, the `E1_consistency` block holds PSA (as a fraction of the 28 final probes) and BRS (as a percentage of the 5-point maximum). The paper averages PSA over three repeats per user, and pools the CS pairs and PET cells of all four users within a repeat. If scoring fails, rerun the same command; finished judgements are kept.

PRS comes from a separate command, run after `score`:

```bash
uv run mempabench self-report --run-dir <run directory>
```

After each final probe, the same simulated user (its persona card and the simulator model and settings frozen in the run plan, without the memory it built up) rereads its side of the conversation and rates the one dimension that probe tests from 1 to 5, with a short reason. PRS is the mean over the 28 final probes. A probe that BRS scores at the floor because all its attempts failed gets the same floor here (`rule_assigned`) and is not asked. A probe whose recorded transcript does not rebuild into what the simulated user saw is marked `rebuild_mismatch`; it is not asked and leaves the mean. The command writes one JSON per probe to `metrics/self_report/` and a summary with the PRS and these counts to `metrics/self_report.json`, and prints the PRS. It works the same on a probe-repeat round or reference arm. It asks one probe at a time, one `google/gemini-3.8-flash` call each; a full run took about 11 minutes. A call that keeps failing stops the command with an error; rerun it and the probes already rated are kept. In the paper, an extra check left out emotional_engagement probes whose simulated user voiced no feeling in its opening; it left out none of 432, so it is not part of this command.

## Your own PA model

Any OpenRouter model with tool calling can be the backbone. Add an entry under `models:` in [`configs/model_request_policies.yaml`](../configs/model_request_policies.yaml), copying an existing one (for example `deepseek/deepseek-v4-flash`): `provider_name: openrouter`, the model id, its request fields, and `validated_roles: [pa]` once you have checked that the model handles the PA's tool calls. Then pass it as `--pa-model`. Upstream routing for one model (the paper pins HY4-Preview to SiliconFlow) goes under `provider_routing:` in the same file.

## Probe repeats and reference arms

The paper runs each probe three times. Round 1 is the run itself. Rounds 2 and 3 (`repeat.rounds` in `configs/run_config.yaml`) branch the same pinned checkpoints again without replaying the sessions; each round is its own run directory under `workspace/probe_variance/`. No Memory and Oracle run the same way, as reference arms on another run's checkpoints.

```bash
uv run mempabench repeat-prepare --source-manifest /absolute/sources.json \
  --workspace-root "$PWD/workspace" \
  --state-root "$PWD/workspace/probe_variance/user_a_my_runs" \
  --output /absolute/dispatch.json
uv run mempabench repeat-run --manifest /absolute/dispatch.json            # validates and reports, starts nothing
uv run mempabench repeat-run --manifest /absolute/dispatch.json \
  --execute --score --settings 8 --probe-concurrency 4 --restore-concurrency 1
uv run mempabench repeat-report --manifest /absolute/dispatch.json \
  --output "$PWD/workspace/probe_variance/user_a_my_runs/results" --watch-pid <repeat-run pid>
```

The source manifest names finished runs by absolute path, and an `arm` entry turns a source into a No Memory or Oracle reference arm with its own PA model. [probe_variance.md](probe_variance.md) describes the manifest, the concurrency options and how stops and restarts are handled.

## Long runs: interruptions and resume

- Start long runs inside `tmux` or `screen`; an SSH disconnect then leaves them running.
- To resume after a crash or reboot, bring the services back and repeat the original command unchanged.
- `mempabench run` exits with code 75 when rerunning the same command resumes it (a model, provider, simulator, judge or memory-service failure); its last stderr line is then `mempabench-run-stopped kind=<failure kind> step=<step id, or - outside a step>`. Other failures keep their own non-zero code. `scripts/run_with_resume.sh <run arguments>` runs `uv run mempabench run` with them and reruns the identical command after each exit 75, up to `MEMPABENCH_MAX_RESUMES` times (default 10). Before each rerun it waits `MEMPABENCH_RESUME_WAIT_SECONDS` (default 300); the wait doubles while the same step keeps failing, up to `MEMPABENCH_RESUME_WAIT_MAX_SECONDS` (default 3600), and drops back once another step, or no step, fails. It stops with exit 75 once one step has failed `MEMPABENCH_SAME_STEP_LIMIT` tries in a row (default 3), and with the run's own code on any other exit. The run's output reaches the terminal unchanged (the current try's stderr is also copied to `$TMPDIR` until the try ends); each try adds one line, and a table of all tries (start, end, exit code, kind, step, wait) and a status line end the output. It refuses `--overwrite` and `--regenerate-plan` (or an abbreviation of either), since every rerun would restart the run; start a new run ID instead.
- Completed steps are skipped; an interrupted step restarts from the latest checkpoint.
- If the PA fails the same IX repair three times in one turn of an everyday session, that session is recorded as skipped and the run goes on, keeping what the session had already done; set `failure_policy.auto_skip_on_ix_exhaustion: false` in `configs/run_config.yaml` to stop the run there instead, or put a `failure_policy.yaml` with the same keys in a run directory to change it for that run only.
- Press `Ctrl-C` once for a graceful stop after the in-flight work publishes its checkpoint; a second signal forces termination.
- If the first step fails before any checkpoint exists, start a new run ID.
- A probe gets three attempts per launch. If all three fail, `score` refuses the run; repeat the original command to give that probe three new attempts. If it keeps failing, score with `--not-applicable-exhausted`: the probe gets the paper's floor (setting missed, BRS 1) and still counts.
- `--retain-checkpoints-every 10` keeps every tenth completed mainline checkpoint, the checkpoints the probes branch from and the latest three.
- If one OpenRouter upstream fails a model 3 times in a run (an error that names it, an empty reply cut off at the token limit, or an empty PA or File memory consolidation reply), the run, resumes included, stops sending that model's requests there and logs `provider_avoided model=<model> provider=<provider> failures=3`. The counts and skipped upstreams are kept in `<run_dir>/provider_avoidance.json`. `providers.avoid_after_failures` in `configs/run_config.yaml` sets the number (`0` turns it off); a new run plan freezes it. A model pinned to one provider, such as HY4-Preview on SiliconFlow, is never rerouted. Once every upstream of a model is skipped, the run clears that model's counts and skipped upstreams, logs `provider_avoidance_reset model=<model> cleared=<providers>`, and resends its requests without the skipped upstreams; any routing the model is configured with still applies.

## How long a run takes

Measured on the paper's runs (wall clock):

| Stage | Median | Spread |
|---|---|---|
| Session phase (99 sessions of one user, one backbone × memory setting) | 21.2 h | 7.9–89.6 h over the 64 runs |
| One probe round (33 probes) | 0.42 h | 90th percentile 1.7 h over 288 rounds |

One everyday session takes about 10–20 minutes and one probe under 5 minutes. The paper protocol is one session phase plus three probe rounds per setting. Settings are independent runs, so several can run side by side.

## Cleaning up

A run's files live in its run directory; external backends also keep state in their Docker volumes, which grow with every run and every checkpoint. Clean the external state first, because the run directory records its names.

- **`mem0`, `simple_rag` (Qdrant on port 6333).** The run's collections are `mempa_<run id>_mem0` and, for `mem0`, `mempa_<run id>_mem0_entities`. Delete each collection's checkpoint snapshots first, then the collection:
  ```bash
  curl -s http://127.0.0.1:6333/collections                                   # list collections
  curl -s http://127.0.0.1:6333/collections/<collection>/snapshots            # list its snapshots
  curl -X DELETE http://127.0.0.1:6333/collections/<collection>/snapshots/<snapshot>
  curl -X DELETE http://127.0.0.1:6333/collections/<collection>
  ```
- **`memos` (MemOS API on port 8100).** The run's cube is `memos_cube_id` in its `run_plan.yaml`; delete it with
  ```bash
  curl -X POST http://127.0.0.1:8100/product/delete_cube -H 'Content-Type: application/json' \
    -d '{"mem_cube_id": "<memos_cube_id>"}'
  ```
- **Probe branches.** A finished probe deletes its own branch collection or cube. One left behind by a failed probe is named in `backend_state/branch_namespace.json` under the run directory (`find <run dir> -name branch_namespace.json`); delete it the same way.
- **The run itself.** Then delete `workspace/runs/<pa model>/<condition>/<user>/<run id>/` (and its probe-repeat rounds under `workspace/probe_variance/`).

## Tests

```bash
uv sync --locked --extra dev
uv run mempabench data pull
uv run pytest -q
```
