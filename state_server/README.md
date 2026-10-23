# StateServer

StateServer exposes MemPABench's fixed world-toolbox over a run-local copy of a
persona's task state. Tool results come from real fixture files, and write tools
persist their changes in the current step's working state.

## Runtime structure

```text
data/state_fixtures/{persona}/
  -> harness/state_runtime.py
       canonical_stage/                 persistent state for one run
       stage_work/{step_id}/             isolated copy for the current step
         -> state_server/server.py       fixed MCP tool handlers
              -> state_server/store.py  file reads, writes, and audit logs
```

`data/state_tools/catalog.yaml` is the canonical tool catalog. A runtime
artifact with `tool_status: world_toolbox` receives every catalog entry whose
status is `implemented`; the artifact does not generate or narrow the tool set.
Each entry's `returns.fields` lists the result field paths the live handlers
return; when a handler's result shape changes, update its entry to match.

## Main files

| Path | Responsibility |
|------|----------------|
| `server.py` | Registers the fixed MCP tools and delegates each call to `StateStore`. |
| `store.py` | Reads and mutates run-local state while enforcing path and date visibility rules. |
| `harness/state_runtime.py` | Copies the complete persona fixture tree, creates per-step working copies, and commits or discards each step. |
| `data/state_tools/catalog.yaml` | Lists the supported tool contracts and implementation status. |
| `data/state_fixtures/{persona}/` | Provides the canonical starting world for that persona. |

## Argument formats

Every date and time a tool accepts is later compared as a string or parsed, so
the format is part of the contract and each one is stated in the tool's
parameter description (`typing.Annotated[..., pydantic.Field(description=...)]`
in `server.py`, which FastMCP renders into the schema the PA sees):

| Parameter kind | Format | Enforced by |
|----------------|--------|-------------|
| `date`, `check_in`, `check_out` | `YYYY-MM-DD` | `_require_date` |
| `time`, `start_time`, `end_time` | 24-hour `HH:MM` | `_require_time` |
| `recurrence` | `daily` / `yearly` / `weekly on <days>`, optionally `through YYYY-MM-DD` | `_parse_recurrence` |
| `period` (utility bill) | `YYYY-MM` | `_require_period` |
| `status` on an update tool | the two values that tool's guards match | `_require_choice` |
| `time_window` (weather) | `HH:MM-HH:MM` or a label such as `evening` | documented only |

A bad value raises `ValueError` naming the parameter, the value received, and
the expected format; FastMCP returns that text to the PA as a tool error, so the
PA can correct the call. Keep the descriptions and the validators in step.

## Record ids

`ID_PREFIXES` in `store.py` is the one table of id shapes (`IdScheme` per record
type), keyed by the store file each record type lives in. Handlers mint through
`StateStore._mint_id`, which reads the table, and the seeded records were
checked against the same table when the worlds were authored — so a fixture and
the handler that appends to it cannot end up in two id families. (They did, twice: seeded
reservations were `res_0001` while `restaurant_reservation_create` minted
`rsv_0001`, and seeded orders were `GM-118427` while `shopping_order_create`
minted `order_0001`.)

Most rows are `<prefix>_NNNN`. `shopping/order_store.json` is the exception:
The persona's merchant issues its own order numbers and the seeded emails quote
them verbatim (`Order GM-118427 confirmed`), so the store mints in the merchant's
scheme — a two-letter merchant code, a hyphen, six digits — continuing past the
highest number that code has already used. The merchant comes from the optional
top-level `merchant: {name, code}` in the persona's `shopping/order_store.json`
(`order_merchant()`); with no key it is GlenMart / `GM`, and a merchant without a
code falls back to the first two uppercase letters of its name.

Still not in the table: the world's other vendor ids (bills, tracking numbers,
flight numbers, room names) that the store never mints.

## Adding a tool

1. **Describe every parameter.** `Annotated[T, Field(description=...)]` on each
   one — what the value is, and its format or vocabulary when the store matches
   on it. For an id, name the tool that returns it and the prefix (`Event id as
   returned by calendar_event_search, ev_NNNN`).
2. **Enforce formats through the shared helpers**, never a fresh regex:
   `_require_date`, `_require_time`, `_require_period`, `_require_choice`,
   `_parse_recurrence`. The description and the helper must agree.
3. **Check the slot before writing.** A create/purchase whose date and time name
   a real-world slot goes through `_require_slot` (or the domain equivalent), so
   a booking for something that does not exist errors instead of persisting.
4. **Take ids from the id table.** Add an `IdScheme` row to `ID_PREFIXES` for a
   new record type and mint with `_mint_id`; seed fixture records in the same
   shape, and never format an id by hand.
5. **Run the tests.** `uv run pytest tests/test_state_server* -q`.

## Runtime environment

The MCP subprocess receives:

- `MEMPA_STATE_ROOT`: current step's mutable state directory
- `MEMPA_RUN_ID`: benchmark run identifier
- `MEMPA_USER_ID`: persona identifier
- `MEMPA_SESSION_ID`: current session or event identifier
- `MEMPA_SESSION_DATE`: optional benchmark date used to hide future records
- `PYTHONPATH`: project root used to import `state_server`

`MEMPA_STATE_ROOT` is required. The other identifiers default only when the
server is started directly outside the benchmark harness.

## Step lifecycle

1. `StageRuntime` copies `data/state_fixtures/{persona}/` to
   `{runtime_root}/{run_id}/canonical_stage/`. Formal runs pass their
   model/memory/persona parent under `workspace/runs/`; diagnostics pass their
   producer/persona parent under `workspace/diagnostics/`.
2. Before each step, it copies the canonical stage to
   `stage_work/{step_id}/` and exposes the fixed tool handlers against that copy.
3. Tool calls append to `tool_log.jsonl`; state-changing calls also append to
   `state_diff.jsonl`.
4. The harness preserves those two audit files with the step output.
5. A `commit` step replaces the canonical stage with the working copy. A
   `discard` step removes the working copy without changing canonical state.

## Focused checks

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_state_server.py \
  tests/test_state_server_mcp.py \
  tests/test_state_server_toolbox.py \
  tests/test_harness/test_state_runtime.py
```
