"""Transcript logger + markdown renderer for SimulatorLoop sessions.

Lives in harness/ because it spans simulator + PA (per the no-share rule between
pa/ and simulator/). SimulatorLoop emits events through a duck-typed sink with a
single `log_event(dict)` method; this module persists those events as JSONL and
renders human-readable Markdown.

Layout produced per session:

    <run_dir>/
    ├── transcript.jsonl   — append-only, primary record
    ├── transcript.md      — human-readable rendering (write_markdown after run)
    └── meta.yaml          — config snapshot (persona / script / model / time)

Event schema (one JSON object per line):
    {"t": int, "kind": str, ...payload}

Kinds: session_start | beat_enter | simulator_turn | pa_turn | session_end |
       session_end_reflection (added later when self-report wired) |
       tool_outside_plan | tool_arguments_off_plan (both written by the
       harness after the session, see `tool_outside_plan_events`)

`pa_turn` additionally carries the IX wrapper's audit projection under
`ix_render`: the held
content, the gate, the consent rows, the notices and the trace. That blob is the
**IX face only** — the text-face judge pack is built by `harness/judge_pack.py`,
which removes it wholesale.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.outline_schema import GATE_PAYLOAD_PARAMS

from pa.agent.ix.render_types import IX_TOOL_NAME_PREFIX, from_ix_render_metadata
from pa.agent.ix.verification import VERIFICATION_OVERRIDE


def is_ix_tool_event(tool_event: dict) -> bool:
    """True for every IX/internal tool event: selection, render, continue, resolve.

    Name-prefix based so the six `IX_render_*`, the four `IX_continue_*` and
    `IX_capability_resolve` are all covered without a second list to keep in
    sync (the action-type coverage check uses the same prefix rule).

    `ix_event` covers the wrapper's own control records — `beat_gate_turn`,
    `ix_consent_denied`, `ix_release_missing`, `ix_hold_forced_close` and the
    rest of `IX_EVENT_NAMES`, stamped by `TurnCollector.emit()`. Their names
    carry no attribute or setting string, so the text pack's vocabulary scan
    cannot catch them, yet a row named `ix_consent_denied` tells the
    setting-blind judge that a gate fired and consent was refused — exactly
    what projection 4 exists to withhold. They stay on the IX face: this
    predicate keeps them out of `task_tool_events()` (text pack and tool
    count) while `extract_pa_toolcalls` still records them as `type: "ix"`.
    """
    if tool_event.get("ix") or tool_event.get("ix_event"):
        return True
    return str(tool_event.get("name") or "").startswith(IX_TOOL_NAME_PREFIX)


def is_internal_tool_event(tool_event: dict) -> bool:
    """True for wrapper control events that are not real tool calls.

    `process` / `ix_directive` / `elicitation` / `autonomy` are the legacy
    instruction-injection markers the wrapper replaces; they never counted as
    task tool calls and they never reach the text face.
    """
    return bool(
        tool_event.get("process")
        or tool_event.get("ix_directive")
        or tool_event.get("elicitation")
        or tool_event.get("autonomy")
    )


def task_tool_events(pa_event: dict) -> list[dict]:
    """The executed task tool events of one `pa_turn`, in call order."""
    return [
        tool_event
        for tool_event in pa_event.get("tool_events") or []
        if not is_ix_tool_event(tool_event) and not is_internal_tool_event(tool_event)
    ]


#: How a catalog tool is named at run time. The draft's `atomic_steps` name
#: bare catalog tools (`email_message_send`); the PA calls them through the
#: state MCP server, which prefixes them. Same rule as
#: `harness/forced_event_schema.py`.
STATE_TOOL_NAME_PREFIX = "mcp_state_"

#: Suffixes of the read-class catalog tools. Every one of the 72 tools in
#: `data/state_tools/catalog.yaml` ends in one of nine suffixes and the split
#: is exact: `_read` (27 tools) and `_search` (17) declare `allowed_effects`
#: within {read, verify}; `_book`, `_create`, `_delete`, `_dial`, `_purchase`,
#: `_send` and `_update` all write. No suffix carries both kinds, so the name alone
#: decides and this module needs no catalog at run time.
#: `_list` / `_get` / `_lookup` name no catalog tool and are therefore not
#: listed. `tests/test_harness/test_session_executor.py` re-derives the split
#: from the catalog, so a new tool cannot break the rule quietly.
READ_CLASS_TOOL_SUFFIXES = ("_read", "_search")


def declared_plan_tools(script_path: Path | str) -> set[str]:
    """Catalog tools the session draft's atomic steps name.

    Union over every `atomic_steps` list in the draft, so a forced event's
    branches both count and no caller has to know the draft's shape. A file
    that declares no atomic step at all — a probe artifact, a missing path —
    returns the empty set, which `tool_outside_plan_events` reads as "this
    step has no plan to be outside of".
    """
    path = Path(script_path)
    try:
        draft = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return set()

    tools: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "atomic_steps" and isinstance(value, list):
                    for step in value:
                        tool = step.get("tool") if isinstance(step, dict) else None
                        if isinstance(tool, str) and tool.strip():
                            tools.add(tool)
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(draft)
    return tools


def declared_plan_arguments(
    script_path: Path | str,
) -> dict[str, list[dict[str, Any]]]:
    """`tool_arguments` the draft declares, keyed by tool name.

    The companion to `declared_plan_tools`: one entry per `atomic_steps` step
    that names a tool, so a call on a declared tool can be checked against the
    arguments the plan actually wrote for it. A step with no `tool_arguments`
    contributes an empty mapping, which matches any call — the plan declared
    the tool and said nothing about how it would be pointed.
    """
    path = Path(script_path)
    try:
        draft = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}

    declared: dict[str, list[dict[str, Any]]] = {}

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "atomic_steps" and isinstance(value, list):
                    for step in value:
                        if not isinstance(step, dict):
                            continue
                        tool = step.get("tool")
                        if not isinstance(tool, str) or not tool.strip():
                            continue
                        args = step.get("tool_arguments")
                        declared.setdefault(tool, []).append(
                            dict(args) if isinstance(args, Mapping) else {}
                        )
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(draft)
    return declared


def _call_matches_declared_arguments(
    call_args: Mapping[str, Any],
    declared_args: Mapping[str, Any],
) -> bool:
    """One declared step's `tool_arguments` against one executed call.

    Every key the plan wrote has to appear on the call; keys the call adds on
    its own do not break the match (subset-equal), because a draft names the
    arguments that carry the story and the runtime fills in defaults, ids and
    pagination the author never wrote. A `GATE_PAYLOAD_PARAMS` key is the
    exception and is compared exactly: those are the payload fields a gate
    binds, so "close enough" there is the thing worth seeing.
    """
    for key, expected in declared_args.items():
        if key not in call_args:
            return False
        actual = call_args[key]
        if key in GATE_PAYLOAD_PARAMS:
            if actual != expected:
                return False
            continue
        if actual == expected:
            continue
        if isinstance(expected, Mapping) and isinstance(actual, Mapping):
            if all(actual.get(k) == v for k, v in expected.items()):
                continue
        return False
    return True


def tool_outside_plan_events(
    events: Sequence[Mapping[str, Any]],
    declared_tools: set[str],
    declared_arguments: Mapping[str, list[Mapping[str, Any]]] | None = None,
) -> list[dict]:
    """One `tool_outside_plan` record per **call** the plan never named.

    Diagnostic only — nothing refuses the call, and the PA keeps the capability
    it had. It says where the PA reached for a tool no `atomic_step` declares,
    so a batch review can find the turn without re-reading every transcript:
    regress13 `acc_023` sent a real email (`email_message_send`) and edited a
    calendar event (`calendar_event_update`) with neither step in its draft.

    Both classes are recorded and `tool_class` (`read` / `write`, by
    `READ_CLASS_TOOL_SUFFIXES`) tells them apart. Reads used to be dropped
    here, on the grounds that they are how a PA finds its way around a world it
    was not handed a map to and that the `document_read` / `place_search` /
    `map_route_search` probes were the bulk of the first regress13 pass. That
    filter also threw away the measurement: regress20 `acc_023` made roughly
    ninety out-of-plan reads and its records said none, and the per-model read
    behaviour is analysed later (owner ruling 2026-09-08). Keeping the class on
    the record leaves the write count exactly as readable as it was, since
    every consumer that wants writes only filters on `tool_class`.

    Every call is recorded, not the first per tool name. The old rule folded a
    retry loop into one record on the theory that "a PA that retries the same
    undeclared write eleven times is one finding" — but the record is what a
    batch review counts, so the fold made a loop indistinguishable from a
    single slip: regress18 `acc_011` wrote 1 record for 6 refused
    `document_update`s and `acc_039` 2 for 4 refused email writes, and both
    read as "tried once". `turn` and `beat` locate the PA turn; `call_index`
    is the call's position among that turn's task tool events, which is the
    only thing separating two identical calls inside one turn (`tool_events`
    carries no per-call timestamp of its own).

    Only `mcp_state_*` names count — `task_tool_events` has already dropped the
    `IX_*` tools and the wrapper's control records, and the world toolbox is
    the only surface an atomic step can name.

    Argument-level matching closes the gap the name-only rule left open
    (regress13 `acc_019` wrote to `draft_section3_w_notes.md`, a file no atomic
    step mentions, through a perfectly declared `document_update`). When
    `declared_arguments` is supplied, a call whose tool *is* declared is
    compared against the `tool_arguments` of every step that declares that
    tool; matching none of them is its own record, `tool_arguments_off_plan`,
    carrying under `args` the arguments the call actually used. The two kinds never both
    fire for one call: an undeclared tool has no arguments to be off-plan
    against. Omitting `declared_arguments` keeps the old name-only behaviour.
    """
    if not declared_tools:
        return []
    records: list[dict] = []
    for event in events:
        if event.get("kind") != "pa_turn":
            continue
        for call_index, tool_event in enumerate(task_tool_events(dict(event))):
            name = str(tool_event.get("name") or "")
            if not name.startswith(STATE_TOOL_NAME_PREFIX):
                continue
            tool = name[len(STATE_TOOL_NAME_PREFIX):]
            if tool in declared_tools:
                declared_steps = (declared_arguments or {}).get(tool)
                if not declared_steps:
                    continue
                call_args = tool_event.get("args")
                if not isinstance(call_args, Mapping):
                    call_args = {}
                if any(
                    _call_matches_declared_arguments(call_args, step_args)
                    for step_args in declared_steps
                ):
                    continue
                records.append(
                    {
                        "kind": "tool_arguments_off_plan",
                        "tool": tool,
                        "tool_class": (
                            "read"
                            if tool.endswith(READ_CLASS_TOOL_SUFFIXES)
                            else "write"
                        ),
                        "args": dict(call_args),
                        "turn": event.get("t"),
                        "beat": event.get("beat"),
                        "call_index": call_index,
                    }
                )
                continue
            records.append(
                {
                    "kind": "tool_outside_plan",
                    "tool": tool,
                    "tool_class": (
                        "read"
                        if tool.endswith(READ_CLASS_TOOL_SUFFIXES)
                        else "write"
                    ),
                    "turn": event.get("t"),
                    "beat": event.get("beat"),
                    "call_index": call_index,
                }
            )
    return records


def verification_override_events(
    events: Sequence[Mapping[str, Any]],
) -> list[dict]:
    """One `verification_override` record per turn the wrapper forced.

    The wrapper's stamp rides inside `pa_turn.tool_events` as an `ix_event`,
    where only a reader that already opens tool events would find it. This
    lifts it to a top-level transcript event naming the turn it belongs to —
    the same shape and the same reason as `tool_outside_plan` above — so the
    step directory says on its face that the run was steered.

    `MEMPA_VERIFY_OUTREACH` is the only thing that produces these records, and
    it is not a scored-run setting (`pa/agent/ix/verification.py`).
    """
    records: list[dict] = []
    for event in events:
        if event.get("kind") != "pa_turn":
            continue
        for tool_event in event.get("tool_events") or []:
            if tool_event.get("name") != VERIFICATION_OVERRIDE:
                continue
            records.append({
                "kind": VERIFICATION_OVERRIDE,
                "turn": event.get("t"),
                "beat": event.get("beat"),
                "attribute": tool_event.get("attribute"),
                "outreach_action": tool_event.get("outreach_action"),
                "setting": tool_event.get("setting"),
                "model_outreach_action": tool_event.get("model_outreach_action"),
                "model_setting": tool_event.get("model_setting"),
                "source": tool_event.get("source"),
            })
    return records


def read_ix_render(pa_event: dict) -> dict | None:
    """Validate and return the `ix_render` blob of one `pa_turn`, or None.

    Contract 12: the blob is re-read through `from_ix_render_metadata`, so a
    record whose `schema_version` / `ix_contract_version` does not match the
    running code fails fast instead of being reinterpreted under new semantics.

    `metadata.ix_render` is where `merge_ix_render_into_transcript` puts it —
    the PA's own `OutboundMessage.metadata` key, restored after the simulator
    dropped it. A top-level `ix_render` is also accepted so a hand-written
    fixture stays readable.
    """
    metadata = pa_event.get("metadata")
    blob = None
    if isinstance(metadata, dict):
        blob = metadata.get("ix_render")
    blob = blob or pa_event.get("ix_render")
    if not blob:
        return None
    return dict(from_ix_render_metadata(blob))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TranscriptWriter:
    """Append-only JSONL writer for one session.

    Crash-safe: each event is flushed immediately so a partial transcript survives
    interruption. Use as a context manager (`with TranscriptWriter(...) as tw:`)
    or call `close()` explicitly.
    """

    def __init__(
        self,
        run_dir: Path | str,
        event_hook: Callable[[dict], None] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "transcript.jsonl"
        self._fp = self.jsonl_path.open("a", encoding="utf-8")
        self.validation_attempts_path = (
            self.run_dir / "simulator_validation_attempts.jsonl"
        )
        # Written empty rather than lazily, so "no retries" and "retries not
        # recorded" stop looking alike. The file was absent on regress19
        # `acc_004`, `acc_007` and `acc_019` and a read could not tell whether
        # the simulator accepted every turn first time or the artifact was
        # simply never produced.
        self.validation_attempts_path.touch()
        self._validation_attempts_fp = None
        # The one place the harness sees the simulator's beats as they happen.
        # `harness.gate.gate_marker_hook` uses it to open a gated session's gate
        # on the reveal beat; every other step passes None.
        self._event_hooks = [event_hook] if event_hook is not None else []

    def add_event_hook(self, hook: Callable[[dict], None]) -> None:
        """Watch events alongside the hook this writer was built with.

        The writer is opened before the session's script is loaded, so a hook
        that needs the script -- `harness.gate.gate_cue_carry_hook` -- can only
        be installed once the loop has it. Hooks run in the order they are added.
        """
        self._event_hooks.append(hook)

    def log_event(self, event: dict) -> None:
        self._fp.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()
        for hook in self._event_hooks:
            hook(event)

    def log_validation_attempt(self, attempt: dict) -> None:
        if self._validation_attempts_fp is None:
            self._validation_attempts_fp = self.validation_attempts_path.open(
                "a", encoding="utf-8"
            )
        self._validation_attempts_fp.write(
            json.dumps(attempt, ensure_ascii=False, default=str) + "\n"
        )
        self._validation_attempts_fp.flush()

    def close(self) -> None:
        if self._fp is not None and not self._fp.closed:
            self._fp.close()
        if (
            self._validation_attempts_fp is not None
            and not self._validation_attempts_fp.closed
        ):
            self._validation_attempts_fp.close()

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def write_meta(run_dir: Path | str, meta: dict) -> Path:
    """Write meta.yaml (config snapshot) into run_dir. Idempotent — overwrites."""
    path = Path(run_dir) / "meta.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"created_at": _utc_now_iso(), **meta},
            f, sort_keys=False, allow_unicode=True,
        )
    return path


def load_events(jsonl_path: Path | str) -> list[dict]:
    """Read transcript.jsonl into a list of event dicts (one per line)."""
    events: list[dict] = []
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _load_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return load_events(path)


def extract_pa_toolcalls(jsonl_path: Path | str, extra_meta: dict | None = None) -> dict:
    """Extract PA tool calls from transcript.jsonl into beat-grouped records."""
    events = load_events(jsonl_path)
    start = next((e for e in events if e.get("kind") == "session_start"), {})
    active_by_beat: dict[str, dict] = {}
    beats: list[dict] = []

    for event in events:
        kind = event.get("kind")
        if kind == "beat_enter":
            beat_record = {
                "beat": event.get("beat"),
                "active_skills": event.get("active_skills") or [],
                "session_active_skills": event.get("session_active_skills") or event.get("active_skills") or [],
                "cued_ix_tools": event.get("cued_ix_tools") or event.get("active_ix_tools") or [],
                "session_ix_tools": event.get("session_ix_tools") or event.get("active_ix_tools") or [],
                "active_ix_tools": event.get("active_ix_tools") or [],
                "ix_required": [],
                "ix_called": [],
                "ix_missing": [],
                "ix_render": [],
                "autonomy_events": [],
                "calls": [],
            }
            beats.append(beat_record)
            active_by_beat[event.get("beat", "")] = beat_record
            continue
        if kind != "pa_turn":
            continue

        beat = event.get("beat", "")
        active = active_by_beat.get(beat)
        if active is None and (event.get("forced_event") or event.get("proactive_event")):
            active = {
                "beat": beat or ("forced_event" if event.get("forced_event") else "proactive_trigger"),
                "active_skills": [],
                "session_active_skills": [],
                "cued_ix_tools": [],
                "session_ix_tools": [],
                "active_ix_tools": [],
                "ix_required": [],
                "ix_called": [],
                "ix_missing": [],
                "ix_render": [],
                "autonomy_events": [],
                "calls": [],
            }
            active_by_beat[beat] = active
            beats.append(active)
        elif active is None:
            active = {}
        # A beat may hold several pa_turn exchanges (stay/advance pacing), so
        # accumulate across turns instead of letting the last turn overwrite.
        def _merge_unique(key: str, values: list) -> None:
            merged = list(active.get(key) or [])
            merged.extend(item for item in values if item not in merged)
            active[key] = merged

        _merge_unique("ix_required", event.get("ix_required") or [])
        _merge_unique("ix_called", event.get("ix_called") or [])
        active["ix_missing"] = [
            name
            for name in (active.get("ix_required") or [])
            if name not in (active.get("ix_called") or [])
        ]
        render_blob = read_ix_render(event)
        if render_blob is not None:
            record = {"t": event.get("t"), **render_blob}
            _merge_unique("ix_render", [record])
            gate = render_blob.get("gate")
            if gate:
                # This group records the gate, not the old interception.
                # Every hold-bearing attribute gates through the same record,
                # so `attribute` says which one fired.
                _merge_unique("autonomy_events", [{"t": event.get("t"), **gate}])
        calls = active.setdefault("calls", [])
        call_index = len(calls)
        for tool_event in event.get("tool_events") or []:
            if is_internal_tool_event(tool_event):
                continue
            call = {
                "call_index": call_index,
                "name": tool_event.get("name"),
                "status": tool_event.get("status"),
                "args": tool_event.get("args") or {},
                "detail": tool_event.get("detail"),
                "error": tool_event.get("error"),
                "type": "ix" if is_ix_tool_event(tool_event) else "task",
            }
            if tool_event.get("ix"):
                # `payload_type` is what separates a selection from a per-turn
                # payload (`pa/agent/ix/events.py:101`). Without it every
                # `IX_render_*` / `IX_continue_*` entry reads as a selection
                # whose `setting` is the empty string, and the judge packer's
                # "take only `payload_type == "ix_preference_selection"`"
                # selector has nothing to filter on.
                call.update({
                    "payload_type": tool_event.get("payload_type"),
                    "attribute": tool_event.get("attribute"),
                    "setting": tool_event.get("setting"),
                    "evidence": tool_event.get("evidence"),
                    "application": tool_event.get("application"),
                })
            elif tool_event.get("ix_event"):
                # The wrapper's own control records. Each
                # name carries its own diagnostic fields — `item_id`,
                # `item_ids`, `tool`, `tools`, `episode_id`, `cancelled`,
                # `sideband_decision`, `reason` — and the judge correlates
                # cancellations and false claims through exactly those ids, so
                # they are carried over as they came. A fixed list would drop
                # the next record's field silently, which is how the whole
                # payload went missing here in the first place.
                call["ix_event"] = True
                call.update({
                    key: value
                    for key, value in tool_event.items()
                    if key not in call
                })
            calls.append(call)
            call_index += 1

    meta = {
        "session_id": start.get("session_id"),
        "persona": start.get("persona"),
        "context": start.get("context"),
        "scenario": start.get("scenario"),
        "model": start.get("pa_model") or start.get("model"),
        "pa_model": start.get("pa_model"),
        "simulator_model": start.get("simulator_model"),
    }
    if extra_meta:
        meta.update({key: value for key, value in extra_meta.items() if value is not None})

    return {
        "meta": meta,
        "beats": beats,
    }


def merge_ix_render_into_transcript(
    jsonl_path: Path | str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Restore each PA turn's `ix_render` into the transcript it belongs to.

    The simulator emits `pa_turn` **without** the PA's `OutboundMessage.metadata`
    — deliberately, since it must not read `ix_render` — so the block would
    otherwise exist only inside the PA and reach no artifact at all. The
    harness captures it per call in `RecordingPA` and this puts it back, into
    `pa_turn.metadata.ix_render`.

    Alignment is by turn order, the same ordering `tool_events` already relies
    on: the simulator has exactly two PA call sites and each emits exactly one
    `pa_turn`, so the n-th record belongs to the n-th `pa_turn`. Records whose
    blob is empty (the PA rendered nothing that turn) still hold their place.

    A forced-event step replays authored PA text and never calls the PA LLM
    (`harness/forced_event_executor.py`), so its `pa_turn`s never pass through
    the wrapper and no record is produced for them. They are left out of
    `pa_turns` entirely: counting them would make every forced step read as a
    merge mismatch, and aligning against them would shift the records.

    Rewrites the file in place and returns the counts, so a caller can notice
    a mismatch rather than trust a silent merge.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return {"pa_turns": 0, "records": len(records), "merged": 0}

    events = load_events(jsonl_path)
    pa_turns = [
        event
        for event in events
        if event.get("kind") == "pa_turn" and not event.get("forced_event")
    ]
    if not records:
        # Nothing to merge, but still report the wrapper turns that went
        # unrendered — `0 != 0` would have hidden a real mismatch.
        return {"pa_turns": len(pa_turns), "records": 0, "merged": 0}
    merged = 0
    for event, record in zip(pa_turns, records):
        blob = record.get("ix_render")
        if not blob:
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            event["metadata"] = metadata
        metadata["ix_render"] = blob
        merged += 1

    if merged:
        # The merge runs at every step boundary, before the checkpoint is
        # published. A crash mid-write must not leave the only audit copy of
        # that step half-rewritten, so the new file is built beside the old one
        # and swapped in atomically, as `RunLedger.dump_to` does.
        payload = "".join(
            json.dumps(event, ensure_ascii=False, default=str) + "\n"
            for event in events
        )
        tmp = jsonl_path.with_name(jsonl_path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, jsonl_path)
    return {"pa_turns": len(pa_turns), "records": len(records), "merged": merged}



def write_pa_toolcalls(
    jsonl_path: Path | str,
    out_path: Path | str | None = None,
    extra_meta: dict | None = None,
) -> Path:
    """Write PA tool calls extracted from transcript.jsonl as a JSON array."""
    jsonl_path = Path(jsonl_path)
    target = Path(out_path) if out_path else jsonl_path.parent / "pa_toolcalls.json"
    target.write_text(
        json.dumps(extract_pa_toolcalls(jsonl_path, extra_meta=extra_meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _consecutive_repeats(rows: list[dict]) -> list[tuple[dict, int]]:
    """`rows` with each run of identical neighbours collapsed to (row, count).

    Rendering only: the ledger keeps every row. regress21r acc_039 t19 printed
    25 identical `held call` lines for one booking the wrapper refused over and
    over, and a reviewer reading the page could not see past them.
    """
    collapsed: list[tuple[dict, int]] = []
    for row in rows:
        if collapsed and collapsed[-1][0] == row:
            previous, count = collapsed[-1]
            collapsed[-1] = (previous, count + 1)
            continue
        collapsed.append((row, 1))
    return collapsed


def _quote(text: str) -> str:
    """Render `text` as a markdown blockquote (every line prefixed with '> ')."""
    if not text:
        return "> (empty)"
    return "\n".join("> " + line for line in text.split("\n"))


def _render_eval_block(sim_evt: dict) -> list[str]:
    """Render the eval-only fields of a simulator_turn event as a <details> section."""
    out: list[str] = ["<details><summary>eval-only (not visible to PA)</summary>", ""]

    factual_checks = sim_evt.get("factual_check") or []
    if factual_checks:
        out.append("**factual_check**:")
        for v in factual_checks:
            out.append(
                f"- `{v.get('fact')}`: expected={v.get('expected')!r}, "
                f"pa_said={v.get('pa_said')!r}"
            )
    else:
        out.append("**factual_check**: (none)")
    out.append("")

    turn_assessment = sim_evt.get("turn_assessment") or {}
    out.append(
        "**turn_assessment.next_beat**: "
        f"`{turn_assessment.get('next_beat', '?')}`"
    )
    out.append("")

    # The consent sideband is the gate's ground truth — PA-blind,
    # like `factual_check` and `turn_assessment`. Without it the IX face's
    # human review cannot tell a held item the user refused from one the user
    # was never asked about. It stays inside this eval-only block; the text
    # face's pack never carries it at all.
    consent_decisions = sim_evt.get("consent_decisions") or []
    if consent_decisions:
        out.append("**consent_decisions**:")
        for decision in consent_decisions:
            # Owner ruling 2026-09-08: an entry with no id and a `tool` is a
            # standing authorisation for that act, not an answer about a held
            # item. Rendered as the tool it covers — as an empty id it read
            # like a malformed row.
            subject = (
                decision.get("interaction_item_id")
                or f"standing: {decision.get('tool') or '?'}"
            )
            out.append(f"- `{subject}`: `{decision.get('decision')}`")
    else:
        out.append("**consent_decisions**: (none)")
    out.append("")

    warnings = sim_evt.get("parse_warnings") or []
    if warnings:
        out.append("**parse_warnings**:")
        for w in warnings:
            out.append(f"- {w}")
        out.append("")

    out.append("</details>")
    out.append("")
    return out


def _render_turn_tool_events(pa_evt: dict) -> list[str]:
    events = task_tool_events(pa_evt)
    if not events:
        return []
    out = ["**[PA task/tool calls]**", ""]
    for event in events:
        out.append(
            f"- `{event.get('name')}` status={event.get('status', '?')} "
            f"args={event.get('args', {})} result={event.get('detail', '')}"
        )
        if event.get("interaction") and event.get("message"):
            out.append(f"  - interaction_message: {event.get('message')}")
    out.append("")
    return out


def _render_ix_audit(pa_evt: dict) -> list[str]:
    required = pa_evt.get("ix_required") or []
    called = pa_evt.get("ix_called") or []
    missing = pa_evt.get("ix_missing") or []
    ix_events = [event for event in pa_evt.get("tool_events") or [] if event.get("ix")]
    if not required and not ix_events:
        return []

    out = ["**[IX audit]**", ""]
    out.append(f"- required: {', '.join(f'`{name}`' for name in required) if required else '(none)'}")
    out.append(f"- called: {', '.join(f'`{name}`' for name in called) if called else '(none)'}")
    out.append(f"- missing: {', '.join(f'`{name}`' for name in missing) if missing else '(none)'}")
    # An `IX_render_*` call carries a payload, not a setting, so it has nothing
    # to list here: acc_011 printed a phantom `reasoning_visibility: ``` row on
    # every render turn.
    selections = [event for event in ix_events if (event.get("setting") or "").strip()]
    if selections:
        out.append("")
        out.append("**[IX selections]**")
        for event in selections:
            attr = event.get("attribute", "")
            setting = event.get("setting", "")
            out.append(f"- `{attr}`: `{setting}`")
            if event.get("evidence"):
                out.append(f"  - evidence: {event.get('evidence')}")
            if event.get("application"):
                out.append(f"  - application: {event.get('application')}")
    out.append("")
    return out


def _render_ix_render(pa_evt: dict) -> list[str]:
    """Render the `ix_render` blob — the IX face, held content included.

    The held bodies are printed verbatim: a reviewer has to be able to tell
    "the PA never wrote it" from "the PA wrote it and the wrapper held it".
    None of this reaches the text-face judge pack (`harness/judge_pack.py`).
    """
    blob = read_ix_render(pa_evt)
    if blob is None:
        return []

    out = ["**[IX render]**", ""]
    episodes = blob.get("episodes") or {}
    if episodes:
        for attribute, episode in episodes.items():
            out.append(
                f"- episode `{attribute}`: `{episode.get('episode_id', '?')}` "
                f"rounds_open={episode.get('rounds_open', 0)} "
                f"closed={bool(episode.get('closed'))}"
            )

    released = blob.get("released") or {}
    released_parts = [
        f"{name}: {', '.join(f'`{item}`' for item in items)}"
        for name, items in released.items()
        if items
    ]
    out.append(
        "- released: " + ("; ".join(released_parts) if released_parts else "(none)")
    )

    held = blob.get("held") or {}
    for batch in held.get("batches") or []:
        out.append(
            f"- held batch `{batch.get('batch_id', '?')}` "
            f"({batch.get('label', '')}):"
        )
        out.extend("  " + line for line in _quote(batch.get("body", "")).split("\n"))
    for question in held.get("questions") or []:
        out.append(
            f"- held question `{question.get('id', '?')}`: {question.get('text', '')}"
        )
    for item in held.get("plan_remaining") or []:
        out.append(
            f"- held plan item `{item.get('id', '?')}`: {item.get('intent', '')}"
        )
    for call, repeat in _consecutive_repeats(held.get("calls") or []):
        line = (
            f"- held call `{call.get('tool', '?')}` "
            f"item=`{call.get('interaction_item_id', '')}` "
            f"status=`{call.get('status', '')}` reason=`{call.get('reason', '')}`"
        )
        out.append(line if repeat == 1 else f"{line} repeat={repeat}")
    for item in held.get("cancelled") or []:
        out.append(f"- cancelled: {item}")
    if held.get("outreach_message"):
        out.append("- held outreach message:")
        out.extend(
            "  " + line for line in _quote(held["outreach_message"]).split("\n")
        )

    gate = blob.get("gate")
    if gate:
        out.append(
            f"- gate: `{gate.get('attribute', '?')}` item=`{gate.get('item_id', '')}` "
            f"reason=`{gate.get('reason', '')}` "
            f"action_type=`{gate.get('action_type', '')}` "
            f"message_source=`{gate.get('message_source', '')}`"
        )

    consent = blob.get("consent") or []
    if consent:
        out.append("")
        out.append("| attribute | item | pa_decision | sideband_decision | released |")
        out.append("|---|---|---|---|---|")
        for row in consent:
            out.append(
                f"| `{row.get('attribute', '')}` | `{row.get('item_id', '')}` "
                f"| `{row.get('pa_decision', '')}` "
                f"| `{row.get('sideband_decision', '')}` "
                f"| {bool(row.get('released'))} |"
            )
        out.append("")

    for notice in blob.get("notices") or []:
        out.append(
            f"- notice #{notice.get('order', 0)} `{notice.get('tool', '')}`: "
            f"{notice.get('text', '')}"
        )

    completion = blob.get("completion") or {}
    if completion.get("text"):
        out.append(
            f"- completion (released={bool(completion.get('released'))}): "
            f"{completion.get('text')}"
        )

    reasoning = blob.get("reasoning") or {}
    out.append(
        f"- reasoning: shown={bool(reasoning.get('shown'))} "
        f"source=`{reasoning.get('source', '')}` "
        f"provider_reasoning_present="
        f"{bool(reasoning.get('provider_reasoning_present'))}"
    )

    for certainty in blob.get("certainties") or []:
        out.append(
            f"- certainty `{certainty.get('ref', '')}` "
            f"level=`{certainty.get('level', '')}` "
            f"rendered={certainty.get('rendered', '')!r}"
        )

    proactive = blob.get("proactive_decision")
    if proactive:
        out.append(
            f"- proactive: `{proactive.get('action', '')}` "
            f"setting=`{proactive.get('setting', '')}`"
        )
        if proactive.get("message"):
            out.extend(
                "  " + line for line in _quote(proactive["message"]).split("\n")
            )

    trace = blob.get("trace") or []
    if trace:
        out.append("- trace:")
        out.extend(f"  - {step}" for step in trace)

    out.append("")
    return out


def render_markdown(jsonl_path: Path | str) -> str:
    """Render a transcript.jsonl into human-readable Markdown.

    Layout: per beat, each turn shows simulator → PA (visible blockquote) and
    PA → simulator (visible blockquote), plus a collapsible <details> with the
    eval-only fields (factual_check / turn_assessment / parse_warnings),
    labeled "not visible to PA".
    """
    jsonl_path = Path(jsonl_path)
    events = load_events(jsonl_path)
    lines: list[str] = []

    start = next((e for e in events if e["kind"] == "session_start"), None)
    if start:
        lines.append(f"# Session: {start.get('session_id', '?')}")
        lines.append("")
        lines.append(f"- persona: **{start.get('persona', '?')}**")
        lines.append(f"- context: `{start.get('context', '?')}`")
        lines.append(f"- scenario: {start.get('scenario', '?')}")
        pa_model = start.get("pa_model")
        simulator_model = start.get("simulator_model")
        if pa_model:
            lines.append(f"- PA model: `{pa_model}`")
        else:
            lines.append(f"- model: `{start.get('model', '?')}`")
        if simulator_model:
            lines.append(f"- simulator model: `{simulator_model}`")
        beat_names = start.get("beat_names") or []
        if beat_names:
            lines.append(f"- beats: {', '.join(beat_names)}")
        lines.append("")

    pending_sim: dict | None = None  # simulator_turn buffered until paired pa_turn arrives

    for transcript_event in events:
        kind = transcript_event["kind"]

        if kind == "beat_enter":
            skills = transcript_event.get("active_skills") or []
            skill_str = ", ".join(skills) if skills else "(none)"
            lines.append(
                f"## Beat: `{transcript_event['beat']}` · "
                f"active_skills: [{skill_str}]"
            )
            active_ix_tools = transcript_event.get("active_ix_tools") or []
            ix_str = ", ".join(f"`{name}`" for name in active_ix_tools) if active_ix_tools else "(none)"
            lines.append("")
            lines.append(f"**[Active IX tools]** {ix_str}")
            session_ix_tools = transcript_event.get("session_ix_tools") or []
            if session_ix_tools != active_ix_tools:
                session_ix_str = ", ".join(f"`{name}`" for name in session_ix_tools) if session_ix_tools else "(none)"
                lines.append(f"**[Session IX tools]** {session_ix_str}")
            session_skills = transcript_event.get("session_active_skills") or []
            if session_skills != skills:
                session_skill_str = ", ".join(f"`{skill}`" for skill in session_skills) if session_skills else "(none)"
                lines.append(f"**[Session active skills]** {session_skill_str}")
            lines.append("")

        elif kind == "beat_turn_limit":
            lines.append(
                f"**[Beat turn limit]** beat `{transcript_event.get('beat')}` "
                f"reached {transcript_event.get('turns')} exchanges; "
                "the story moves on."
            )
            lines.append("")

        elif kind == "exchange_start":
            lines.append(f"## Exchange {transcript_event.get('exchange', '?')}")
            lines.append("")

        elif kind == "message":
            source = transcript_event.get("source")
            if isinstance(source, dict):
                source_label = source.get("id", source.get("kind", "unknown"))
            else:
                source_label = source
            lines.append(
                "**["
                f"{source_label} | "
                f"visibility={transcript_event.get('visibility', '?')}"
                "]**"
            )
            lines.append("")
            lines.append(_quote(transcript_event.get("content", "") or ""))
            lines.append("")

        elif kind == "environment_action":
            source = transcript_event.get("source") or {}
            source_label = source.get("id", source.get("kind", "unknown"))
            lines.append(
                f"**[Environment: `{transcript_event.get('action')}` from "
                f"{source_label} | "
                f"visibility={transcript_event.get('visibility', '?')} | "
                f"status={transcript_event.get('status', '?')}]**"
            )
            lines.append("")
            lines.append(_quote(transcript_event.get("content", "") or ""))
            lines.append("")

        elif kind == "outreach_path":
            if transcript_event.get("path") == "send":
                lines.append(
                    "**[Outreach path: contact sent → "
                    f"`{transcript_event.get('opening_beat', '?')}`]**"
                )
            else:
                lines.append(
                    "**[Outreach path: contact suppressed → "
                    f"`{transcript_event.get('opening_beat', '?')}`; "
                    "the user comes back on their own at "
                    f"{transcript_event.get('resume_at') or '?'}]**"
                )
            lines.append("")

        elif kind == VERIFICATION_OVERRIDE:
            lines.append(
                "**[VERIFICATION OVERRIDE — NOT A SCORED RUN: "
                f"`{transcript_event.get('attribute', '?')}` forced to "
                f"`{transcript_event.get('outreach_action', '?')}` by "
                f"{transcript_event.get('source', '?')}; the PA chose "
                f"`{transcript_event.get('model_outreach_action', '?')}`]**"
            )
            lines.append("")

        elif kind == "environment_trigger":
            lines.append(
                "## Environment trigger: "
                f"`{transcript_event.get('trigger_id', '?')}` "
                f"({transcript_event.get('source_type', '?')})"
            )
            lines.append("")
            lines.append(_quote(transcript_event.get("detail", "") or ""))
            lines.append("")
            lines.append(
                f"**Prior thread:** {transcript_event.get('prior_thread', '')}"
            )
            lines.append(
                "**Potential user impact:** "
                f"{transcript_event.get('user_impact', '')}"
            )
            lines.append("")

        elif kind == "simulator_turn":
            pending_sim = transcript_event

        elif kind == "simulator_validation_failure":
            lines.append(
                "**[Simulator output validation failed]** "
                f"phase={transcript_event.get('phase', '?')} "
                f"attempt={transcript_event.get('attempt', '?')} "
                f"codes={transcript_event.get('error_codes') or []}"
            )
            lines.append("")

        elif kind == "pa_turn":
            if transcript_event.get("forced_event"):
                # Forced events replay authored PA text and never call the PA
                # LLM (`harness/forced_event_executor.py`), so they carry no
                # `ix_render` blob and this branch stays as it was.
                lines.extend(_render_turn_tool_events(transcript_event))
                if transcript_event.get("outbound"):
                    lines.append("**[PA]**")
                    lines.append("")
                    lines.append(_quote(transcript_event.get("outbound", "")))
                    lines.append("")
                continue
            if transcript_event.get("proactive_event"):
                lines.extend(_render_ix_audit(transcript_event))
                lines.extend(_render_ix_render(transcript_event))
                lines.extend(_render_turn_tool_events(transcript_event))
                decision = transcript_event.get("outreach_decision", "?")
                lines.append(f"**[PA outreach decision: {decision}]**")
                lines.append("")
                if transcript_event.get("outreach_delivered"):
                    lines.append(_quote(transcript_event.get("outbound", "") or ""))
                    lines.append("")
                continue
            if pending_sim is None:
                # PA turn without a prior simulator turn — render bare
                lines.append("**[PA → Simulator]** _(no paired simulator_turn)_")
                lines.append("")
                lines.append(_quote(transcript_event.get("outbound", "") or ""))
                lines.append("")
                continue
            lines.append("**[Simulator → PA]**")
            lines.append("")
            lines.append(_quote(pending_sim.get("message", "") or ""))
            lines.append("")
            lines.extend(_render_ix_audit(transcript_event))
            lines.extend(_render_ix_render(transcript_event))
            lines.extend(_render_turn_tool_events(transcript_event))
            lines.append("**[PA → Simulator]**")
            lines.append("")
            lines.append(_quote(transcript_event.get("outbound", "") or ""))
            lines.append("")
            lines.extend(_render_eval_block(pending_sim))
            pending_sim = None

        elif kind == "session_end":
            lines.append("---")
            lines.append("")
            if "exchanges" in transcript_event:
                lines.append(
                    "**Session end** · "
                    f"exchanges: {transcript_event.get('exchanges', 0)} · "
                    f"tool calls: {transcript_event.get('tool_calls', 0)} · "
                    "environment actions: "
                    f"{transcript_event.get('environment_actions', 0)}"
                )
            else:
                lines.append(
                    "**Session end** · simulator turns: "
                    f"{transcript_event.get('simulator_turns', 0)} · "
                    f"beats: {transcript_event.get('beat_count', 0)}"
                )
            lines.append("")

        elif kind == "session_end_reflection":
            lines.append("## Session-end self-report")
            lines.append("")
            lines.append("**Reflection (in-character):**")
            lines.append("")
            lines.append(_quote(transcript_event.get("reflection", "") or ""))
            lines.append("")
            final_emotion = transcript_event.get("final_emotion") or {}
            lines.append("**Final emotion:**")
            lines.append(f"- trigger: {final_emotion.get('trigger', '')}")
            lines.append(f"- reaction: {final_emotion.get('reaction', '')}")
            lines.append("")
            self_report = transcript_event.get("self_report") or {}
            if self_report:
                lines.append("**Self-report:**")
                for k, v in self_report.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")

    state_dir = jsonl_path.parent / "state"
    tool_events = _load_optional_jsonl(state_dir / "tool_log.jsonl")
    state_diffs = _load_optional_jsonl(state_dir / "state_diff.jsonl")
    if (state_dir / "tool_log.jsonl").exists() or (state_dir / "state_diff.jsonl").exists():
        lines.append("## Tool And State Audit")
        lines.append("")
        if tool_events:
            lines.append("**Tool calls:**")
            for event in tool_events:
                lines.append(
                    f"- t={event.get('t')}: `{event.get('tool')}` "
                    f"status={event.get('status', '?')} "
                    f"args={event.get('args', {})} "
                    f"result={event.get('result_summary', {})}"
                )
            lines.append("")
        else:
            lines.append("**Tool calls:** (none)")
            lines.append("")
        if state_diffs:
            lines.append("**State diffs:**")
            for event in state_diffs:
                lines.append(
                    f"- t={event.get('t')}: `{event.get('namespace')}` "
                    f"{event.get('op')} `{event.get('id')}` — {event.get('summary', '')}"
                )
            lines.append("")
        else:
            lines.append("**State diffs:** (none)")
            lines.append("")

    return "\n".join(lines)


def write_markdown(jsonl_path: Path | str, md_path: Path | str | None = None) -> Path:
    """Render JSONL → Markdown and write to disk. Returns the md path written."""
    jsonl_path = Path(jsonl_path)
    out_path = Path(md_path) if md_path else jsonl_path.parent / "transcript.md"
    out_path.write_text(render_markdown(jsonl_path), encoding="utf-8")
    return out_path
