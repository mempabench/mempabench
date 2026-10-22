# Independent A/C Probe repeats

`scripts.prepare_probe_variance` creates repeat02/03 from a completed original
run. It retains the complete frozen plan and ground truth, gives each repeat a
new identity, and explicitly borrows each Probe's original checkpoint. Original
timelines are referenced through `probe_repeat_source.json`; accumulation is
not executed again and original Probe outputs are not copied. Each repeat is
its own run directory, `<workspace-root>/probe_variance/<source run_id>_probe_repeatNN/`,
with `NN` a round in `repeat.rounds`; `--state-root` holds only the
dispatcher's own state (below).

Supply an absolute-path source manifest:

```json
{
  "settings": [{
    "id": "user_a_gpt_simple_rag",
    "source": {
      "run_dir": "/absolute/workspace/runs/original"
    }
  }]
}
```

The source must be a complete original run. A stopped process or a
partial-result decision does not make it one. The probe counts of a complete
repeat and the repeat rounds are set in `configs/run_config.yaml`
(`probe_variance`, `repeat`).

```bash
python -m scripts.prepare_probe_variance \
  --source-manifest /absolute/sources.json \
  --workspace-root /absolute/shared/workspace \
  --state-root /absolute/shared/workspace/probe_variance/<persona>_<date> \
  --output /absolute/dispatch.json
python -m scripts.run_probe_variance --manifest /absolute/dispatch.json
python -m scripts.run_probe_variance --manifest /absolute/dispatch.json \
  --execute --score --settings 8 --probe-concurrency 4 --restore-concurrency 1
```

## Reference arms

A setting with `arm` runs a reference condition, `no_memory` or
`oracle_preference`, with the named PA model on the source run's timeline and
pins:

```json
{
  "settings": [
    {"id": "user_a_no_memory_gpt",
     "source": {"run_dir": "/absolute/workspace/runs/original"},
     "arm": {"memory_condition": "no_memory", "pa_model": "openai/gpt-5.6-sol"}},
    {"id": "user_a_oracle_hy",
     "source": {"run_dir": "/absolute/workspace/runs/original"},
     "arm": {"memory_condition": "oracle_preference", "pa_model": "tencent/hy4-preview"}}
  ]
}
```

- The arm's plan is the source plan with that condition and PA model: the PA
  request policy is resolved for the model from
  `configs/model_request_policies.yaml` (HY4-Preview: SiliconFlow only, no
  fallbacks), and consolidation, memory-side and external-backend fields are
  cleared. `harness/probe_repeat_source.repeat_base_plan` is the one rewrite;
  the dispatcher's validation compares each repeat plan against it.
- The source is another condition's run, so every round is new: round 1 and
  the rounds in `repeat.rounds`. The run id is
  `<source run_id>_<condition>_<pa model, / as _>_probe_repeatNN`, so one
  source can feed both arms and every model.
- Each Probe branches the source pin directly; the branch's memory is emptied
  and the arm's adapter is bound. No shared memory is restored or held.
- Which source run an arm borrows decides the world it starts from; keep one
  source per persona for every model of a comparison.

Prepare, validate and run as above; `--score` scores arm rounds like any other.
Export them with a reference-condition source manifest (`data_analysis/README.md`).

Use the same workspace as existing producers so maintenance and global Probe
locks remain effective. Without `--execute` the dispatcher validates inputs
and reports the proposed configuration. Runs have no automatic elapsed-time
cutoff; signals and the explicit stop record control graceful stopping.

Start a later dispatcher after the current dispatcher releases its workspace
lock; active owners keep their existing lifecycle.

Each setting has one process and one memory owner for its new repeats. Each
distinct pinned checkpoint is restored once. Per-Probe adapters, scenes,
sessions, attempts, and evidence remain independent. Supported shared backends
are SimpleRAG, Mem0, MemOS, and File memory. Query transports reject backend
writes. File views also reject direct file creation, replacement, and deletion
through POSIX file and directory permissions. Memory is released after both rounds
and actual background requests have drained. A Qdrant or MemOS response timeout
that leaves remote completion unknown blocks new consumers of that checkpoint
and retains its namespace and evidence; independent checkpoints can continue.

`S` is the number of setting workers; `W=1` executes the two rounds sequentially
within a worker, and `--rounds-per-setting 2` allows both rounds to overlap under
one owner. `P` is Probe concurrency, capped at four; `R` limits concurrent restores
across workers. Worker signals stop both rounds, and request observations retain
each round's identity. The global round slot count defaults to `max(2, S)`, even
with W=2; increasing W does not silently double total active rounds. Use
`--batch-slots` only for an explicitly chosen total G. Increasing S or W does not
change P or R.

Use `--capacity-report --workspace-root /absolute/shared/workspace` without a
manifest for read-only hardware and existing-producer measurements. These
measurements report a candidate S, not a verified production limit. Admission
pauses below 8 GiB available memory, below 20 GiB free disk, or after three
consecutive swap-out increases. In-flight work continues. Linux reads these from
/proc; on macOS available memory is vm_stat's free + speculative + inactive
pages and swap-outs are its Swapouts count.

Completed rounds produce durable scoring handoffs. `--score` starts the existing
scorer with `--variance-retries`, at most `--score-concurrency` processes. It
finishes untouched questions before one exact extra request per HTTP429 failure;
request intent and allowance persist across restarts. Other ambiguous requests
are held without blocking untouched questions. Scoring uses the existing
judge and packer. The dispatcher does not calculate variance or write result
documents.

Runtime state is in `status.json`, `activity.jsonl`, and per-setting
`worker.json`; per-checkpoint manifests record restoration, consumers and
cleanup. Probe attempts persist under each new run's
`probe_attempts/`. An unfinished request is not a free retry after restart.
Reconcile a retained namespace or ambiguous request before resuming its
consumers. Successful Probes must never be regenerated.


For read-only measurements of a running dispatcher:

```bash
python -m scripts.report_probe_variance --manifest /absolute/dispatch.json \
  --output /absolute/shared/workspace/probe_variance/<persona>_<date>/results \
  --watch-pid <dispatcher-pid>
```

The collector writes an incremental concurrency table, samples, and per-round
production times without scoring or calculating scientific variance. It checks
the watched process start identity and never restarts a missing dispatcher.
`round_details` stores execution start and completion before scoring. Native
logical search timings are recorded per round in `retrieval_activity.jsonl`,
without queries, replies, or embeddings. The reported p95 includes failures and
uses nearest rank; it includes embedding and internal retries, rather than
representing a single HTTP call. Older executions without timing evidence retain
null measurements. Observed concurrency does not by itself establish verified
stable capacity.

MemOS retrieval parity and side-effect checks cover the frozen fast/text search
with MMR and the deployed service configuration. Service plugins can execute
writes inside a search request; an HTTP route allowlist alone does not certify
arbitrary hooks or other search modes. Record the actual service
code/configuration and enabled plugins with the results.
