"""Text-face judge packing — projection 4 of §2.4 (`text_judge_projection`).

Lives in harness/ because it spans the PA's render output and the transcript the
simulator loop produced (the no-share rule keeps pa/ and simulator/ apart).

Two judge faces, one transcript (§4.4 "two judge faces"):

- **IX face** — the full `ix_render` blob + the full tool log + the transcript.
  Nothing here filters it; `harness/transcript.py` writes it as it came.
- **text face** — this module. It keeps `public_turn` and the **executed** task
  tool events (`name` / `args` / `status` / order) and nothing else. The whole
  `ix_render` block goes, every IX and internal tool event goes, the legacy
  `ix_directives` / `process_events` / `elicitation_events` groups go, and the
  per-call injected fields are stripped out of the args that remain. Removing a
  few fields is not enough: one `trace` string leaks the setting. Two ruled
  exceptions (2026-09-17, unification plan §5.4 / §5.5): a proactive turn also
  carries its send / suppress decision and delivery flag plus the trigger it
  answered (`TEXT_JUDGE_OUTREACH_KEYS` / `TEXT_JUDGE_TRIGGER_KEYS`), and the
  judge prompt renders `tool_log.jsonl` through `text_judge_tool_log`, an
  allow-list of business fields.

`pa/agent/ix/render_types.py` (P0) freezes the field set, the banned keys and a
`text_judge_projection` stub that raises; the packing itself is P4 and lands
here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from harness.transcript import task_tool_events
from pa.agent.ix.render_types import (
    HELD_STATUS_BY_ATTRIBUTE,
    IX_CAPABILITY_RESOLVE_TOOL,
    IX_CONTINUE_TOOL_NAMES,
    IX_EVENT_NAMES,
    IX_RENDER_TOOL_NAMES,
    IX_RENDER_TOP_LEVEL_KEYS,
    IX_TOOL_NAME_PREFIX,
    IXRenderInput,
    PASS_THROUGH_ATTRIBUTES,
    PER_CALL_INJECTED_FIELDS,
    RENDERED_ATTRIBUTES,
    RenderedTurn,
    TEXT_JUDGE_BANNED_KEYS,
    TextJudgeProjection,
)
from pa.agent.ix.specs import IX_PASSIVE_SPECS

#: This pack's own schema version. The text face never carries
#: `ix_contract_version` — knowing which wrapper version ran is an IX-face fact.
TEXT_JUDGE_PACK_SCHEMA_VERSION = "text_judge_pack/1.0.0"

#: **Narrow scan.** Every attribute name and every setting label in the taxonomy
#: — the strings that say *which* attribute the wrapper selected and *which way*
#: (§2.4 projection 4 "any string a setting can be inferred from").
#:
#: Half of them are ordinary English words — `High`, `Low`, `Show`, `Hide`,
#: `Terse`, `Moderate`, `Detailed`, `Guided`, `Formal`, `Casual`, `Silent`,
#: `Structured` — so they are matched **as whole tokens** (case-sensitive, as
#: written) and **only on IX-shaped structural leaves**
#: (`IX_STRUCTURAL_KEY_NAMES`). Scanning them as substrings over every
#: non-free-text leaf is what broke the judge: an ordinary task argument
#: `priority: "High"` failed the pack, and `Highlight` / `Showcase` / `Lowell`
#: matched `High` / `Show` / `Low`.
IX_SETTING_LABELS: frozenset[str] = frozenset(
    {*RENDERED_ATTRIBUTES, *PASS_THROUGH_ATTRIBUTES}
    | {setting for spec in IX_PASSIVE_SPECS.values() for setting in spec.settings}
)

#: **Everywhere scan.** The wrapper's control surface: its tool names, the
#: per-call injected fields, the `[[…]]` position markers, the hold statuses,
#: the audit event names and the runner's phrase for the per-turn payload. None
#: of it can occur in genuine dialogue, so — unlike the labels above — it stays
#: banned in free text too, matched as a substring exactly the way the caller's
#: `needles` are.
#:
#: The audit names come from `IX_EVENT_NAMES` at import time, so an event added
#: to the wrapper is covered here without a second edit. They go in as **exact
#: names** (`ix_consent_denied`, `beat_gate_turn`, …) and never as a bare `ix_`
#: prefix: this half is a substring scan, and `ix_` would fire on ordinary words
#: like `prefix_length` or `matrix_size`.
IX_CONTROL_VOCABULARY: frozenset[str] = frozenset(
    {spec.name for spec in IX_PASSIVE_SPECS.values()}
    | set(IX_RENDER_TOOL_NAMES.values())
    | set(IX_CONTINUE_TOOL_NAMES.values())
    | {IX_CAPABILITY_RESOLVE_TOOL}
    | {f"{IX_TOOL_NAME_PREFIX}render_", f"{IX_TOOL_NAME_PREFIX}continue_"}
    | set(PER_CALL_INJECTED_FIELDS)
    | set(HELD_STATUS_BY_ATTRIBUTE.values())
    | set(IX_EVENT_NAMES)
    | {"[[batch:", "[[b:", "[[question:", "[[q:", "[[certainty:", "[[c:"}
    | {"ix_render", "ix_directives", "elicitation_events", "interaction payload"}
)

#: The key names whose leaves are wrapper structure rather than task data — the
#: only place `IX_SETTING_LABELS` is looked for. Derived from the §4.4 blob's own
#: top-level keys (`IX_RENDER_TOP_LEVEL_KEYS`) and `TEXT_JUDGE_BANNED_KEYS` (a
#: banned key's subtree is IX by definition), plus the selection-payload words
#: the wrapper writes around them. A key path containing an `ix` / `IX_`-prefixed
#: key counts as IX-shaped for everything below it, which is how
#: `ix_render.selections[0].setting` is reached even though `selections` is not
#: itself a §4.4 key. Ordinary task arguments — `priority`, `query`, `title`,
#: `path` — are deliberately absent.
IX_STRUCTURAL_KEY_NAMES: frozenset[str] = frozenset(
    {key.lower() for key in TEXT_JUDGE_BANNED_KEYS}
    | {key.lower() for key in IX_RENDER_TOP_LEVEL_KEYS}
    | {
        "setting", "settings", "selection", "selections",
        "attribute", "attributes", "render", "rationale",
        "certainty", "certainties", "hold", "held", "plan",
        "gate", "consent",
    }
)

#: `IX_SETTING_LABELS` as one word-boundary alternation, longest label first so
#: a multi-word label (`Full narration`, `One-at-a-time`) is reported whole
#: rather than as whichever of its words matched. Case matters: the labels are
#: matched as the taxonomy writes them.
_SETTING_LABEL_RE = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(label)
        for label in sorted(IX_SETTING_LABELS, key=lambda label: (-len(label), label))
    )
    + r")\b"
)

#: The forced-event visibility values a text-face pack may show, i.e. the whole
#: `Visibility` vocabulary of `harness/forced_event_schema.py` minus `god_only`.
#: A `god_only` item is world setup that reached neither party, so it is not
#: part of the conversation the text judge scores. Spelled out here rather than
#: imported — `forced_event_schema` drags in the whole authoring stack — and
#: `tests/test_harness/test_judge_pack.py` asserts the two stay in step.
TEXT_JUDGE_VISIBILITY: frozenset[str] = frozenset({"pa", "user", "both"})

#: The visibilities a forced `environment_action` has to have before the text
#: face sees it — the same set `harness/forced_event_executor.py:326` checks
#: before putting the action's content into the PA's session history. `user` is
#: not in it: that action reached neither the PA nor this conversation.
TEXT_JUDGE_PA_DELIVERED_VISIBILITY: frozenset[str] = frozenset({"pa", "both"})

#: The only `meta` keys a text-face pack may carry. Callers merge whole
#: `metadata` dicts into this pack, and that dict is where the provider's
#: `thinking_blocks` / `reasoning_content` and the simulator's
#: `consent_decisions` live — so the meta field takes an **allow-list**, not a
#: deny-list: a deny-list closes only the holes already known, and the next key
#: the PA's metadata grows would ride straight through.
#:
#: `target_cell` / `target_attribute` / `target_setting` are deliberately out:
#: they are the attribute and setting names the text face exists to be blind
#: to. The judge gets the ground truth through `ground_truth.json` (§4.2),
#: which is a different file with a different reader.
TEXT_JUDGE_META_KEYS: frozenset[str] = frozenset({
    "created_at", "run_id", "step_id", "kind",
    "persona_id", "context", "session_type", "domain",
    "story_id", "timeline_slot", "time_position",
    "acc_num", "before_acc_num",
    "benchmark_session_timestamp",
    "memory_condition", "memory_mode", "stage_policy",
    "pa_model", "simulator_model",
})

#: The proactive-outreach exception (owner ruling 2026-09-17, unification plan
#: §5.5): on a `proactive_event` turn the text face carries these three keys
#: of the transcript's `pa_turn` — the runtime's send / suppress decision and
#: whether the contact was delivered — next to `public_turn`, which is the
#: delivered content itself. The contact opportunity travels as its own entry
#: (`environment_trigger`, `TEXT_JUDGE_TRIGGER_KEYS`). Only these keys and only
#: on that turn: the `ix_render.proactive_decision` block (the chosen setting)
#: and the archived `held.outreach_message` stay on the IX face, and both scans
#: below still ban them — `tests/test_harness/test_judge_pack.py` holds this
#: opening to exactly these names.
TEXT_JUDGE_OUTREACH_KEYS: tuple[str, ...] = (
    "proactive_event", "outreach_decision", "outreach_delivered",
)

#: What a `kind == "environment_trigger"` transcript event contributes: the
#: contact opportunity as the simulator emitted it (`simulator/loop.py`
#: `_run_proactive_trigger`), i.e. what the PA was handed before it decided.
TEXT_JUDGE_TRIGGER_KEYS: tuple[str, ...] = (
    "trigger_id", "source_type", "prior_thread", "detail", "user_impact",
)

#: **`tool_log.jsonl`, §5.4 (owner ruling 2026-09-17, unification plan):** the
#: text face may show a tool's business result and error, filtered through an
#: allow-list rather than the assumption that nothing else in the row leaks.
#: The store writes every row through `state_server/store.py` `log_tool`
#: (`t` / `timestamp` / `run_id` / `user_id` / `session_id` / `tool` / `args` /
#: `result_summary` / `status`); these are the only row fields that cross.
#: The three identity fields and the fixed world clock are run plumbing, not
#: evidence.
TEXT_JUDGE_TOOL_LOG_ROW_KEYS: frozenset[str] = frozenset({
    "t", "tool", "status", "args", "result_summary",
})

#: Rows that are the store's own audit of the dictation gate, not a PA call:
#: `_log_gate_loaded` (`gate_loaded`) and `_mark_gate_satisfied`
#: (`gate_satisfied`). Their `args` name the gate tool, parameter and
#: `task_fact`; the whole row is control and is dropped. An `IX_*` row never
#: reaches the store, but it is dropped here too so the rule has one home.
TEXT_JUDGE_TOOL_LOG_CONTROL_TOOLS: frozenset[str] = frozenset({
    "gate_loaded", "gate_satisfied",
})

#: `args` keys the store adds beside a tool's own parameters, all of them
#: dictation-gate bookkeeping: `_payload_log_args` nests the gate verdict under
#: `gate` (`param` / `matched` / `satisfied` / `open`); `_enforce_write_gate`
#: logs a refusal with `gate_tool` / `gate_param`; the two audit rows above
#: carry `task_fact` / `reveal_beat` / `pre_reveal_write_tools`. Everything
#: else under `args` is what the PA passed (or the store's `<param>_chars`
#: length summary of it). This level is a named drop rather than a catalog
#: allow-list on purpose: the store's logged vocabulary is not the catalog's —
#: `appointment_search` logs `query` for the catalog's `service` — so an
#: allow-list keyed on catalog parameters would silently lose business args.
TEXT_JUDGE_TOOL_LOG_CONTROL_ARG_KEYS: frozenset[str] = frozenset({
    "gate", "gate_tool", "gate_param",
    "task_fact", "reveal_beat", "pre_reveal_write_tools",
})

#: The `result_summary` keys that may cross — every business key `log_tool` is
#: handed anywhere in `state_server/store.py` (AST walk over its call sites plus
#: the `_search_log_result` shape), and nothing the store writes about its gate.
#: A key ending in `_id` is a record identifier the tool returned
#: (`appointment_id`, `draft_id`, `event_id`, `sent_id`, …) and is allowed by
#: that suffix. Not here, hence dropped: `gate_refusal`, `closed`,
#: `session_closing` (the write-gate refusals), `loaded`, `open`, `satisfied`
#: (the two audit rows), `drops_terms` (ruling-A refusal detail).
TEXT_JUDGE_TOOL_LOG_RESULT_KEYS: frozenset[str] = frozenset({
    # search / read outcomes
    "matches", "ids", "truncated", "found", "id", "type", "status",
    # documents
    "path", "bytes", "files", "match_tier", "note",
    # domain payload summaries
    "options", "usable", "events", "bills", "methods", "matched",
    # errors and the capability boundary (§3.5)
    "error", "error_code", "boundary_kind", "detail",
})

#: `error` values that name the store's own refusal rather than a world
#: outcome. The row stays (the call was made and errored) but the value goes.
TEXT_JUDGE_TOOL_LOG_CONTROL_ERROR_VALUES: frozenset[str] = frozenset({
    "gate_refusal",
})


def text_judge_tool_log(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The text face's view of `tool_log.jsonl`: allow-listed rows, in `t` order.

    One function, three allow-lists (`TEXT_JUDGE_TOOL_LOG_ROW_KEYS`,
    `TEXT_JUDGE_TOOL_LOG_RESULT_KEYS`, the `_id` suffix) and the two named
    drops for `args` and `error`. The IX face keeps reading the copied file;
    this is the projection the judge prompt renders.
    """
    kept: list[dict[str, Any]] = []
    for row in rows:
        tool = str(row.get("tool") or "")
        if tool in TEXT_JUDGE_TOOL_LOG_CONTROL_TOOLS or tool.startswith(IX_TOOL_NAME_PREFIX):
            continue
        args = {
            key: value
            for key, value in (row.get("args") or {}).items()
            if key not in TEXT_JUDGE_TOOL_LOG_CONTROL_ARG_KEYS
            and key not in PER_CALL_INJECTED_FIELDS
            and key not in TEXT_JUDGE_BANNED_KEYS
        }
        result = {
            key: value
            for key, value in (row.get("result_summary") or {}).items()
            if (key in TEXT_JUDGE_TOOL_LOG_RESULT_KEYS or str(key).endswith("_id"))
            and not (key == "error" and value in TEXT_JUDGE_TOOL_LOG_CONTROL_ERROR_VALUES)
        }
        projected = {key: row[key] for key in TEXT_JUDGE_TOOL_LOG_ROW_KEYS if key in row}
        projected["args"] = args
        projected["result_summary"] = result
        kept.append(projected)
    kept.sort(key=lambda entry: entry.get("t", 0))
    return kept


def text_judge_projection(
    rendered: RenderedTurn,
    render_input: IXRenderInput,
) -> TextJudgeProjection:
    """Projection 4 for one turn (§2.4). Implements the P0 stub.

    `public_turn` is the rendered message verbatim — the gate's ask is already
    inside it, which is why there is no second field for it. The task events are
    the calls whose ids are in `executed_call_ids`, IX tools excluded.
    """
    executed = set(rendered.executed_call_ids)
    events: list[dict[str, Any]] = []
    for call in render_input.calls:
        if call.tool_name.startswith(IX_TOOL_NAME_PREFIX):
            continue
        if call.tool_call_id not in executed:
            continue
        events.append(
            {
                "order": call.order,
                "name": call.tool_name,
                "args": _strip_injected_fields(call.args),
                "status": call.status,
            }
        )
    events.sort(key=lambda event: event["order"])
    return TextJudgeProjection(
        public_turn=rendered.message,
        task_tool_events=events,
    )


def build_text_judge_pack(
    events: list[dict],
    *,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pack a whole `transcript.jsonl` into the text face's task-only view.

    One entry per user message and per PA turn, in transcript order. A PA turn
    contributes its `public_turn` (`outbound`) and its executed task tool
    events; everything the IX wrapper recorded about *why* the turn looks the
    way it does is dropped here, not filtered field by field.
    """
    turns: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("kind")
        if kind == "simulator_turn":
            turns.append(
                {
                    "t": event.get("t"),
                    "kind": "user_message",
                    "beat": event.get("beat"),
                    "text": event.get("message", "") or "",
                }
            )
        elif kind == "message":
            if event.get("visibility") not in TEXT_JUDGE_VISIBILITY:
                continue
            # A forced event replays both sides' authored speech through the
            # one `message` kind, so `source` decides whose turn it is — the
            # same mapping `harness/forced_event_executor.py` uses to build the
            # PA's session history ("assistant" for `source == "pa"`, "user"
            # otherwise, third-party `SourceRef`s included). A forced PA line
            # never went through the wrapper and carries no tool events of its
            # own; the step's tool calls arrive as their own `pa_turn` events.
            if event.get("source") == "pa":
                turns.append(
                    {
                        "t": event.get("t"),
                        "kind": "pa_turn",
                        "beat": event.get("beat"),
                        "public_turn": event.get("content", "") or "",
                        "task_tool_events": [],
                    }
                )
            else:
                turns.append(
                    {
                        "t": event.get("t"),
                        "kind": "user_message",
                        "beat": event.get("beat"),
                        "text": event.get("content", "") or "",
                    }
                )
        elif kind == "environment_action":
            # A forced third-party message is delivered as an environment
            # action, and `harness/forced_event_executor.py:319-337` writes its
            # content into the PA's history and memory as a **user** message —
            # the same role rule the `message` branch above follows. Skipping
            # the kind left the judge reading the PA's reply to a cancellation
            # notice with the notice itself missing, i.e. the causal input gone.
            #
            # Only `content` crosses over: `action` / `action_id` / `source` and
            # the delivery's state payload are world plumbing the conversation
            # never showed either party.
            if event.get("status") != "executed":
                continue
            if event.get("visibility") not in TEXT_JUDGE_PA_DELIVERED_VISIBILITY:
                continue
            turns.append(
                {
                    "t": event.get("t"),
                    "kind": "user_message",
                    "beat": event.get("beat"),
                    "text": event.get("content", "") or "",
                }
            )
        elif kind == "environment_trigger":
            # §5.5: the contact opportunity, as the simulator emitted it. It is
            # the arrival the proactive turn answers, so without it the judge
            # would read a send / suppress decision with nothing to decide on.
            turns.append(
                {
                    "t": event.get("t"),
                    "kind": "environment_trigger",
                    **{key: event.get(key) for key in TEXT_JUDGE_TRIGGER_KEYS},
                }
            )
        elif kind == "pa_turn":
            turn: dict[str, Any] = {
                "t": event.get("t"),
                "kind": "pa_turn",
                "beat": event.get("beat"),
                "public_turn": event.get("outbound", "") or "",
                "task_tool_events": _pack_task_tool_events(event),
            }
            if event.get("proactive_event"):
                # §5.5: the decision and the delivery, and only on this turn.
                # `public_turn` already is the delivered content (empty on a
                # suppress: the archived message stays held, on the IX face).
                turn.update({key: event.get(key) for key in TEXT_JUDGE_OUTREACH_KEYS})
            turns.append(turn)
    pack = {
        # Not `schema_version`: that key belongs to the `ix_render` blob and is
        # therefore banned here, which is exactly what `scan_banned_keys` says.
        "pack_schema_version": TEXT_JUDGE_PACK_SCHEMA_VERSION,
        "meta": {
            key: value
            for key, value in (meta or {}).items()
            if key in TEXT_JUDGE_META_KEYS
        },
        "turns": turns,
    }
    # The whitelists above are what *should* keep the audit plane out; these two
    # scans are what proves it, on the finished object, every time. Without them
    # `turns` was checked only by the tests that remembered to call a scan, and
    # `meta` was not checked at all.
    leaks = scan_banned_keys(pack) + scan_banned_strings(pack)
    if leaks:
        raise ValueError(
            "text judge pack would carry the IX face: " + "; ".join(leaks)
        )
    return pack


def _pack_task_tool_events(pa_event: dict) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    for order, tool_event in enumerate(task_tool_events(pa_event)):
        if tool_event.get("executed") is False:
            continue
        packed.append(
            {
                "order": order,
                "name": tool_event.get("name"),
                "args": _strip_injected_fields(tool_event.get("args") or {}),
                "status": tool_event.get("status"),
            }
        )
    return packed


def _strip_injected_fields(args: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the three per-call injected fields and any banned key from args.

    They are stripped before the real tool is called (§5.3 carrier C), so
    seeing one here already means something went wrong upstream — dropping it
    keeps the leak out of the pack, and `scan_banned_keys` still reports it when
    the caller scans the raw event.
    """
    return {
        key: value
        for key, value in args.items()
        if key not in PER_CALL_INJECTED_FIELDS and key not in TEXT_JUDGE_BANNED_KEYS
    }


def scan_banned_keys(pack: Any, path: str = "") -> list[str]:
    """Every occurrence of a banned key, or of an IX tool name, as a path.

    T11's banned-field half: `ix_render` and each of its top-level keys,
    `ix_directives` / `process_events` / `elicitation_events`, the IX selection
    payload fields, the per-call injected fields and the provider-reasoning
    paths — plus any tool event still named `IX_*`.
    """
    hits: list[str] = []
    if isinstance(pack, Mapping):
        for key, value in pack.items():
            child = f"{path}.{key}" if path else str(key)
            if key in TEXT_JUDGE_BANNED_KEYS:
                hits.append(child)
            if key == "name" and str(value or "").startswith(IX_TOOL_NAME_PREFIX):
                hits.append(child)
            hits.extend(scan_banned_keys(value, child))
    elif isinstance(pack, (list, tuple)):
        for index, item in enumerate(pack):
            hits.extend(scan_banned_keys(item, f"{path}[{index}]"))
    return hits


#: The dialogue keys plus the tool-argument names that carry prose the PA wrote
#: — the body of a document, an email, a message. They are free text in exactly
#: the sense `public_turn` is, so they are scanned only for the needles and
#: `IX_CONTROL_VOCABULARY`. Every other argument (ids, paths, enums, flags)
#: stays structural.
FREE_TEXT_KEYS: frozenset[str] = frozenset({
    "public_turn", "text", "content", "body", "message", "replacement",
    "subject", "note",
    # The raw-transcript spellings of the same two sides: `pa_turn.outbound` is
    # what becomes `public_turn`, `user_message` is what becomes `text`. Callers
    # scan raw events as well as finished packs, and there the dialogue arrives
    # under these names.
    "outbound", "user_message",
    # A tool result quoted back into the record — a document the user wrote, a
    # calendar entry, an inbox line. It is world text, not wrapper structure.
    "detail",
    # A document title and a search query are prose the PA (or the fixture)
    # wrote, not machine-written structure. Live, saving a paper called
    # "Structured environments and non-Markovian edge coupling" reported a
    # banned-string hit because `Structured` is also a setting name.
    "title", "query",
})


def _is_ix_structural_key(key: Any) -> bool:
    """True when everything under `key` is wrapper structure, not task data."""
    lowered = str(key).lower()
    return (
        lowered == "ix"
        or lowered.startswith("ix_")
        or lowered in IX_STRUCTURAL_KEY_NAMES
    )


def scan_banned_strings(
    pack: Any,
    *,
    needles: Iterable[str] = (),
    free_text_keys: Iterable[str] = FREE_TEXT_KEYS,
) -> list[str]:
    """T11's banned-string half. Returns `"<path>: <needle>"` for each hit.

    Three scopes, because they are three different promises:

    - `IX_CONTROL_VOCABULARY` and the caller's `needles` — the wrapper's tool
      names and control syntax, the held bodies, `gate.reason`, `trace` lines
      and provider reasoning — are matched **everywhere**, free text included,
      as substrings. None of them can occur in dialogue by accident.
    - `IX_SETTING_LABELS` is matched **only on IX-shaped structural leaves**
      (`IX_STRUCTURAL_KEY_NAMES`, inherited down the key path) and **only as
      whole tokens**. That is the leak the scan exists for: a setting reaching
      the text face through non-dialogue structure, e.g.
      `ix_render.selections[0].setting`.
    - everything else — the free text of `FREE_TEXT_KEYS` and ordinary task
      arguments alike — is not scanned for labels at all. `priority: "High"` is
      a task argument, "keep it casual" is prose, and `Highlight` is a word;
      the old whole-vocabulary substring scan failed all three.
    """
    needles = tuple(needles)
    free_text_keys = frozenset(free_text_keys)
    control = sorted(IX_CONTROL_VOCABULARY)
    hits: list[str] = []

    def walk(node: Any, path: str, free_text: bool, ix_shaped: bool) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                walk(
                    value,
                    child,
                    free_text or key in free_text_keys,
                    ix_shaped or _is_ix_structural_key(key),
                )
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]", free_text, ix_shaped)
        elif isinstance(node, str):
            for needle in (*control, *needles):
                if needle and needle in node:
                    hits.append(f"{path}: {needle}")
            if ix_shaped and not free_text:
                labels = {match.group(0) for match in _SETTING_LABEL_RE.finditer(node)}
                hits.extend(f"{path}: {label}" for label in sorted(labels))

    walk(pack, "", False, False)
    return hits
