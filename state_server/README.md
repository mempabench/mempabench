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
Each entry's `returns.fields` lists the result field paths probed from the live
handlers (`scripts/generate_catalog_result_fields.py`); rerun that script after
any change to a handler's result shape — `tests/test_catalog_result_fields.py`
gates the match, and every `implemented` tool must be stamped (a new tool with
no corpus sample needs a probe chain in the script's `_hand_chains`).
Rerun it as `--all-personas`, not `--persona`: the union over every world is
the stamp, and only that run drops a field no world returns any more.
Verify with `--all-personas --check`, which fails both on an observed field the
stamp lacks and on a stale field the stamp still carries.

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
PA can correct the call. Keep the descriptions and the validators in step —
`tests/test_state_server_param_formats.py` gates the pairing.

## Record ids

`ID_PREFIXES` in `store.py` is the one table of id shapes (`IdScheme` per record
type), keyed by the store file each record type lives in. Handlers mint through
`StateStore._mint_id`, which reads the table, and `scripts/validate_l0_world.py`
checks seeded records against the same table — so a fixture and the handler that
appends to it cannot end up in two id families. (They did, twice: seeded
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
   new record type and mint with `_mint_id`; that row is also what the fixture
   validator enforces on seeded records, so never format an id by hand.
5. **Run the gates.** `python -m pytest tests/test_state_server* -q` (the
   parameter tests fail on any undescribed or unformatted parameter) and
   `python scripts/validate_l0_world.py` over the fixtures.

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
