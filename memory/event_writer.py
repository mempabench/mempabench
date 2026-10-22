"""Shared writer for the PA's first-person experience of a turn."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from memory.adapter import MemoryAdapter, PromptHistoryEvictionPolicy
from memory.audit import MemoryIOAuditSink
from memory.events import MemoryEvent
from memory.results import WriteReceipt, WriteStatus
from memory.scope import MemoryScope
from memory.turn_experience import ExperienceItem, render_blocks

if TYPE_CHECKING:  # `memory.turn_journal` imports `TurnPool` from here.
    from memory.turn_journal import JournalRecord, TurnJournal

#: The length `pa/agent/loop.py` cuts a persisted tool result at, and the
#: length this module reads `truncated=` back from. One constant, imported
#: there, so the two can never drift apart.
TOOL_RESULT_MAX_CHARS = 16_000

#: `pa/agent/loop.py:1975` appends this when it cuts a tool result at
#: `TOOL_RESULT_MAX_CHARS`, and the pool carries the persisted form.  A result
#: whose own text happens to end this way is not a cut one, so the marker
#: counts only above the cut length.
_TRUNCATION_MARKER = "\n... (truncated)"

#: The prefixes of `IX_PAYLOAD_RESULT_TOOLS` (`pa/agent/ix/collector.py:1329`,
#: built from `IX_RENDER_TOOL_NAMES` and `IX_CONTINUE_TOOL_NAMES`,
#: `pa/agent/ix/render_types.py:247-260`).  The constant itself cannot be
#: imported here: `pa/agent/__init__.py` imports `ContextBuilder`, which
#: imports this module.
_IX_PAYLOAD_TOOL_PREFIXES = ("IX_render_", "IX_continue_")

#: Carrier C (`pa/agent/ix/render_types.py:303`, `PER_CALL_INJECTED_FIELDS`):
#: the three fields the IX wrapper injects onto every task tool's schema and
#: strips before the real tool runs (`pa/agent/runner.py:3739-3743`). They are
#: wrapper state that exists only because a benchmark enforces the IX contract,
#: so they are not experience and never reach a `[tool_call]` body. Spelled out
#: here for the same reason as the prefixes above — the constant cannot be
#: imported without a cycle — and a test pins the two lists together.
PER_CALL_INJECTED_FIELDS = (
    "_ix_notice",
    "interaction_item_id",
    "_ix_confirmation_message",
)

#: The three outcomes of `[ix_selection]`. There is no `held` and no
#: `voided`: IX selections are declarations that always run
#: (`pa/agent/runner.py:2864-2866`), and a voided iteration contributes nothing
#: at all (owner ruling 2026-09-13, Q1).
_IX_SELECTION_OUTCOMES = frozenset({"executed", "rejected", "ignored_duplicate"})

#: The `IX_proactive_outreach` argument carrying the contact text itself
#: (`pa/agent/ix/tools.py:131-160`). It is the one selection argument that is a
#: payload rather than a reason, so on a turn that delivered nothing it is held
#: prose (O4) and stays on the audit plane.
OUTREACH_MESSAGE_FIELD = "outreach_message"


class AdapterWriteCommitError(RuntimeError):
    """Raised when a formal adapter-backed run cannot commit a turn."""


@dataclass(frozen=True)
class TurnPool:
    """One turn's experience, in the two forms `MemoryEvent` carries."""

    messages: list[dict[str, str]] = field(default_factory=list)
    items: list[ExperienceItem] = field(default_factory=list)


def build_turn_pool(
    messages: list[dict[str, Any]],
    *,
    delivered: str | None = None,
    environment_event: Mapping[str, str] | None = None,
    ix_selections: Sequence[Mapping[str, Any]] = (),
    ix_payload_status: str = "",
) -> TurnPool:
    """Render one turn's messages as the pool every backend receives.

    The harness performs no selection: what the PA produced, or perceived in
    its role as the assistant, goes in; what exists only because a benchmark
    is running stays out. The only filter here is that second rule — an
    `_mempa_internal` injection is IX contract enforcement or harness retry
    machinery, never experience.

    Ordering is the PA's own, message by message: an iteration that declared
    tool calls contributes one assistant message carrying its reasoning, its
    prose and one block per call; each tool result is its own message; the
    delivered reply is bare assistant dialogue, preceded by the final
    iteration's reasoning. Only messages are read — never `metadata`, which is
    where every piece of ground truth reaches the PA.

    `environment_event` is the ACC trigger the PA perceived, as
    `process_environment_event` built it, before the harness appended its fixed
    `"Use IX_proactive_outreach …"` instruction (owner ruling 2026-09-13, Q4).
    It leads the pool as an `[environment_event]` block, never as `role: user`.

    `delivered` is the turn's `public_turn`. When it is given, no assistant
    message's own prose becomes bare dialogue — the pre-render text is the
    wrapper's input, not a delivered act (Q3) — and the turn ends with either
    the delivered reply or, when nothing was delivered, `[no_reply]`.

    `ix_selections` carries `{id, name, args, outcome}` per selection call as
    the registry received it, which the serialized `tool_calls` on the
    assistant message cannot: that copy is written before
    `_dedupe_ix_selections` and `_restore_voided_selection_fields` run
    (`pa/agent/runner.py:1413-1421` against `:1458-1462`).

    `ix_payload_status` is the turn's `ix_payload_status` JSON, carrying
    `released` / `held_not_released` / `cancelled`. It replaces a payload
    tool's echoed result (item #9) and decides F1: a payload call's arguments
    are the delivered text and stay, unless this turn held or cancelled a
    section, in which case O4 keeps them out. The status is **per turn**, and
    both lists count — a forced close empties the hold by cancelling it, so
    `held_not_released` alone would read a cancelled section as delivered. One
    undelivered item blanks every payload call of that turn.
    """
    pool_messages: list[dict[str, str]] = []
    items: list[ExperienceItem] = []
    order = 0
    selections = {
        str(selection["id"]): selection
        for selection in ix_selections
        if selection.get("id")
    }
    status = ix_payload_status or _payload_status_in(messages)
    holding = bool(_undelivered_payload_ids(status))
    # O4 on the one attribute whose payload rides on its selection call: a
    # suppressed `IX_proactive_outreach` carries the contact the PA wrote and
    # the wrapper did not deliver. "Produced" means produced and kept, so the
    # candidate text stays on the audit plane and only the decision is
    # remembered — the `[no_reply]` block below. The general rule reads
    # the selection body as "the arguments as the registry received them",
    # which would put an unsent contact into memory as if it had been sent.
    withheld = delivered == ""

    def new_item(kind: str, role: str, attrs: dict[str, str], body: str) -> ExperienceItem:
        nonlocal order
        item = ExperienceItem(
            kind=kind,
            role=role,
            order=order,
            offset_seconds=None,
            attrs=attrs,
            body=body,
        )
        order += 1
        return item

    def emit(role: str, group: list[ExperienceItem]) -> None:
        if not group:
            return
        pool_messages.append({"role": role, "content": render_blocks(group)})
        items.extend(group)

    if environment_event is not None:
        attrs = {
            "source_kind": "environment",
            "source_id": str(environment_event.get("source_id") or ""),
        }
        if environment_event.get("action_id"):
            attrs["action_id"] = str(environment_event["action_id"])
        emit(
            "assistant",
            [new_item(
                "environment_event", "assistant", attrs,
                str(environment_event.get("body") or ""),
            )],
        )

    for message in messages:
        if message.get("_mempa_internal") is True:
            continue
        role = message.get("role")
        content = _as_text(message.get("content"))
        if role == "user":
            if content:
                emit("user", [new_item("user", "user", {}, content)])
        elif role == "tool":
            emit(
                "assistant",
                [_result_item(message, content, status, withheld, new_item)],
            )
        elif role == "assistant":
            if message.get("source_kind") == "subagent":
                # The PA dispatched the sub-agent and perceived its return; the
                # tag comes from that data, never from a role guess.
                if content:
                    emit(
                        "assistant",
                        [new_item("subagent_result", "assistant", {}, content)],
                    )
                continue
            calls = message.get("tool_calls") or []
            group = [
                new_item("reasoning", "assistant", {"source": source}, text)
                for source, text in _reasoning_texts(message)
            ]
            if calls:
                if content:
                    group.append(new_item("prose", "assistant", {}, content))
                group.extend(
                    _call_item(call, selections, holding, withheld, new_item)
                    for call in calls
                )
                emit("assistant", group)
            else:
                # Q3: the delivered reply is `public_turn`, bare; the final
                # iteration contributes its reasoning and no `[prose]`.
                emit("assistant", group)
                if content and delivered is None:
                    emit("assistant", [new_item("assistant", "assistant", {}, content)])

    if delivered is not None:
        if delivered:
            emit("assistant", [new_item("assistant", "assistant", {}, delivered)])
        else:
            # Silence is an interaction choice, and `proactive_outreach` is
            # scored on exactly that choice, so it is written down.
            emit(
                "assistant",
                [new_item("no_reply", "assistant", {"decision": "suppress"}, "")],
            )

    return TurnPool(messages=pool_messages, items=items)


def turn_journal_for(session_manager: Any, session_key: str) -> TurnJournal:
    """The journal beside this session's file.

    Four callers need the same journal for the same session — the loop that
    writes it, the two consolidation entry points that consume it and `/new` —
    and all four hold a `SessionManager` rather than a path, so the resolution
    lives here once. Imported inside the function because `turn_journal` reads
    `TurnPool` from this module.
    """
    from memory.turn_journal import TurnJournal, journal_path

    return TurnJournal(journal_path(session_manager.sessions_dir, session_key))


def record_turn_journal(
    *,
    journal: TurnJournal,
    turn_id: str,
    session_id: str,
    span: tuple[int, int],
    timestamp: datetime,
    pool: TurnPool,
    state: str,
    receipt: WriteReceipt | None = None,
    realized: dict[str, Any] | None = None,
) -> None:
    """Write one pool's journal record, or mark it written.

    Every caller writes `state="pending"` before the turn's first
    `sessions.save`, so the record is durable before the session file it
    describes, and `state="written"` once the adapter receipt is in. The
    resume authority is the record's state, not a message index, which is what
    makes a suppressed outreach's `[n, n)` span resumable.

    A run with no adapter has no receipt to record, and `mark_written` takes a
    `WriteReceipt`, so such a turn keeps its `pending` state: the file
    condition's completion mark is `file_consolidated`, the journal's second
    track, not this one.

    `realized` is the forced-event line's: the artefacts the world
    handed back for the exchange this record describes, so a resume can restore
    them instead of calling the world a second time. The ACC loop passes none —
    it re-executes nothing, because its turn is not replayed from a script.
    """
    from memory.turn_journal import PENDING, WRITTEN, JournalRecord

    if state == PENDING:
        journal.append_pending(JournalRecord.for_pool(
            turn_id=turn_id,
            span=span,
            timestamp=timestamp.isoformat(),
            pool=pool,
            realized=realized,
        ))
        return
    if state == WRITTEN:
        if receipt is not None:
            journal.mark_written(turn_id, receipt)
        return
    raise ValueError(f"Unknown turn-journal state: {state!r}")


def records_owed_to_files(
    records: Sequence[JournalRecord],
) -> list[JournalRecord]:
    """Of these records, the ones the memory files have not had and are to
    have.

    `file_consolidated` is the journal's second completion track, separate from
    `state`: the file condition's adapter write is a documented no-op, so its
    real write happens later, at consolidation time. A record marked
    `dropped_from_files` (a skipped step's turns under
    `partial_skip_pending_memory: drop`) is never owed.
    """
    return [
        record
        for record in records
        if not record.file_consolidated and not record.dropped_from_files
    ]


def consolidation_records(
    records: Sequence[JournalRecord],
) -> list[dict[str, str]]:
    """The pool messages of these journal records, in journal order.

    This is what the file condition consolidates, and it replaces the
    two-message projection the three entry points used to build from
    `session.messages`. `MarkdownMemoryStore._format_messages` renders
    `[timestamp] ROLE: content`, and a pool message carries no timestamp of its
    own — the record carries the turn's, which is the benchmark clock.
    """
    out: list[dict[str, str]] = []
    for record in records:
        for message in record.messages:
            entry = {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            if record.timestamp:
                entry["timestamp"] = record.timestamp
            out.append(entry)
    return out


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not content:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def _reasoning_texts(message: dict[str, Any]) -> list[tuple[str, str]]:
    """The message's reasoning, one entry per source that carries any."""
    found: list[tuple[str, str]] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        found.append(("reasoning_content", reasoning))
    blocks = "\n\n".join(
        str(block.get("thinking") or "")
        for block in message.get("thinking_blocks") or []
        if isinstance(block, dict) and str(block.get("thinking") or "").strip()
    )
    if blocks:
        found.append(("thinking_blocks", blocks))
    return found


def _payload_status_in(messages: Sequence[Mapping[str, Any]]) -> str:
    """The turn's `ix_payload_status`, read off a payload tool's own result.

    The forced-event line hands in `session.messages`, where
    `build_runtime_history` has already replaced the echo
    (`pa/agent/ix/collector.py:1405-1412`). The ACC line hands the status in
    directly, because its raw loop slice still carries the echo.
    """
    for message in messages:
        if message.get("role") != "tool":
            continue
        if not str(message.get("name") or "").startswith(_IX_PAYLOAD_TOOL_PREFIXES):
            continue
        content = message.get("content")
        if isinstance(content, str) and '"status"' in content:
            return content
    return ""


def _undelivered_payload_ids(status: str) -> list[Any]:
    """The turn's payload sections nobody received: held, plus cancelled.

    F1 keeps a payload call's arguments when they are the delivered text. An
    empty `held_not_released` on its own does not prove that: a forced close
    *empties* the hold by cancelling what was in it, so a
    cancelled question would read as delivered and its wording would enter
    memory having reached nobody. Both lists decide.
    """
    try:
        parsed = json.loads(status)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return []
    return [
        item
        for key in ("held_not_released", "cancelled")
        for item in (parsed.get(key) or [])
        if isinstance(parsed.get(key), list)
    ]


def _without_undelivered_outreach(body: str) -> str:
    """Blank `outreach_message` in a JSON body the turn did not deliver."""
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        return body
    if not isinstance(parsed, dict) or OUTREACH_MESSAGE_FIELD not in parsed:
        return body
    return json.dumps(
        {**parsed, OUTREACH_MESSAGE_FIELD: ""}, ensure_ascii=False, default=str
    )


def _result_item(
    message: dict[str, Any],
    content: str,
    status: str,
    withheld: bool,
    new_item: Any,
) -> ExperienceItem:
    name = str(message.get("name") or "")
    call_id = str(message.get("tool_call_id") or "")
    if name.startswith(_IX_PAYLOAD_TOOL_PREFIXES) and status:
        # Item #9: what the PA perceived of a payload call is the hold status,
        # never the registry's echo of the payload back at it.
        content = status
    if name.startswith("IX_"):
        if withheld:
            content = _without_undelivered_outreach(content)
        # The registry's acceptance of the PA's own call, or a hold status.
        return new_item(
            "ix_result", "assistant", {"name": name, "id": call_id}, _compact_json(content)
        )
    return new_item(
        "tool_result",
        "assistant",
        {
            "name": name,
            "id": call_id,
            "truncated": (
                "true"
                if content.endswith(_TRUNCATION_MARKER)
                and len(content) > TOOL_RESULT_MAX_CHARS
                else "false"
            ),
        },
        content,
    )


def _call_item(
    call: dict[str, Any],
    selections: Mapping[str, Mapping[str, Any]],
    holding: bool,
    withheld: bool,
    new_item: Any,
) -> ExperienceItem:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    arguments = str(function.get("arguments") or "")
    call_id = str(call.get("id") or "")
    selection = selections.get(call_id)
    if selection is not None:
        outcome = str(selection.get("outcome") or "")
        if outcome not in _IX_SELECTION_OUTCOMES:
            raise ValueError(
                f"Unknown ix_selection outcome for {name!r}: {outcome!r} "
                f"(expected one of {sorted(_IX_SELECTION_OUTCOMES)})"
            )
        args = selection.get("args")
        body = json.dumps(args, ensure_ascii=False, default=str)
        if withheld:
            body = _without_undelivered_outreach(body)
        return new_item(
            "ix_selection",
            "assistant",
            {
                "attribute": name[len("IX_"):],
                "setting": str((args or {}).get("setting") or ""),
                "outcome": outcome,
            },
            _compact_json(body),
        )
    setting = _selection_setting(name, arguments)
    if setting is None:
        if name.startswith(_IX_PAYLOAD_TOOL_PREFIXES) and holding:
            # O4: held prose stays out at hold time. A payload call whose turn
            # still holds something records the act and not the text; what the
            # PA perceived of it is the `[ix_result]` hold status. F1 (owner
            # ruling 2026-09-13): a turn that held nothing keeps the arguments,
            # because they are the text the PA delivered.
            arguments = ""
        return new_item(
            "tool_call",
            "assistant",
            {"name": name, "id": call_id},
            _compact_json(_without_carrier_fields(arguments)),
        )
    # No `ix_selections` entry: the caller has no tool events for this turn
    # (the forced-event line), so the outcome has no source and is omitted
    # rather than guessed.
    return new_item(
        "ix_selection",
        "assistant",
        {"attribute": name[len("IX_"):], "setting": setting},
        _compact_json(arguments),
    )


def _without_carrier_fields(arguments: str) -> str:
    """The call's arguments with carrier C dropped (see the constant above)."""
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    if not isinstance(parsed, dict):
        return arguments
    return json.dumps(
        {
            key: value
            for key, value in parsed.items()
            if key not in PER_CALL_INJECTED_FIELDS
        },
        ensure_ascii=False,
        default=str,
    )


def _compact_json(text: str) -> str:
    """The canonical compact JSON, or the text as it stands.

    A provider serializes call arguments with `json.dumps` defaults
    (`pa/providers/base.py:24-32`), which inserts spaces and keeps the model's
    key order; the grammar fixes one form for every JSON-bodied block. A
    DSML-recovered call carries the model's raw text, which is not JSON at
    all — that goes in verbatim, because the act was real and its arguments
    are what the PA produced.
    """
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    return json.dumps(
        parsed,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _selection_setting(name: str, arguments: str) -> str | None:
    """The selected setting when this call is an `IX_<attribute>` selection."""
    if not name.startswith("IX_"):
        return None
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or "setting" not in parsed:
        return None
    return str(parsed["setting"])


async def write_visible_memory_event(
    *,
    adapter: MemoryAdapter | None,
    messages: list[dict[str, Any]] | None = None,
    pool: TurnPool | None = None,
    run_id: str,
    persona_id: str,
    agent_id: str,
    session_id: str,
    turn_id: str,
    event_id: str,
    timestamp: datetime,
    step_id: str = "",
    audit_sink: MemoryIOAuditSink | None = None,
    audit_only_labels: dict[str, Any] | None = None,
    require_committed: bool = False,
    eviction_policy: PromptHistoryEvictionPolicy | None = None,
) -> WriteReceipt | None:
    """Write one turn's pool and preserve the existing adapter audit contract.

    `pool` is for a caller that has already rendered the turn because it holds
    data the message dicts do not carry — the forced-event line renders its
    delivered world and third-party lines by their own `source.kind` and stamps
    each item's `offset=`, and the ACC loop holds the raw provider slice, the
    turn's IX call records and its `ix_payload_status`. Everything else passes
    `messages` and the pool is built here. A `pool` always wins; without one,
    `messages` is required.
    """
    if messages is None and pool is None:
        raise ValueError(
            "write_visible_memory_event needs either messages or a pool"
        )
    if adapter is None:
        return None
    if pool is None:
        pool = build_turn_pool(messages or [])
    if not pool.messages:
        return None

    policy = eviction_policy or adapter.prompt_history_eviction_policy
    scope = MemoryScope(
        run_id=run_id,
        persona_id=persona_id,
        agent_id=agent_id or None,
    )
    event = MemoryEvent(
        event_id=event_id,
        run_id=run_id,
        persona_id=persona_id,
        agent_id=agent_id,
        session_id=session_id,
        turn_id=turn_id,
        timestamp=timestamp,
        messages=pool.messages,
        items=pool.items,
    )
    adapter_request = {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "persona_id": event.persona_id,
        "agent_id": event.agent_id,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "timestamp": event.timestamp,
        "messages": event.messages,
    }
    try:
        await adapter.setup_scope(scope)
        receipt = await adapter.record_event(event)
    except Exception as exc:
        _record_memory_io(
            audit_sink=audit_sink,
            adapter=adapter,
            run_id=run_id,
            step_id=step_id,
            session_id=session_id,
            turn_id=turn_id,
            operation="record_event",
            adapter_request=adapter_request,
            status="failed",
            audit_only_labels=audit_only_labels,
            error=exc,
        )
        if require_committed and policy == PromptHistoryEvictionPolicy.after_adapter_commit:
            raise AdapterWriteCommitError(
                "INVALID_ADAPTER_WRITE: "
                f"backend={adapter.backend_name!r} run_id={run_id!r} "
                f"session={session_id!r} turn_id={turn_id!r} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        logger.opt(exception=True).warning(
            "Memory adapter write failed; continuing without adapter memory write"
        )
        return None

    trace = receipt.trace
    _record_memory_io(
        audit_sink=audit_sink,
        adapter=adapter,
        run_id=run_id,
        step_id=step_id,
        session_id=session_id,
        turn_id=turn_id,
        operation=trace.native_operation if trace else "record_event",
        adapter_request=adapter_request,
        backend_request=trace.native_request if trace else None,
        backend_response=(
            trace.native_response
            if trace and trace.native_response is not None
            else {
                "status": receipt.status.value,
                "native_ids": receipt.native_ids,
                "native_events": receipt.native_events,
            }
        ),
        status=receipt.status.value,
        trace=trace,
        audit_only_labels=audit_only_labels,
    )
    return receipt


def record_adapter_write_commit(
    *,
    receipt: WriteReceipt | None,
    session_id: str,
    current_index: int,
    committed_index: int,
    eviction_policy: PromptHistoryEvictionPolicy,
    require_committed: bool,
) -> int:
    """Return the updated exclusive adapter commit index."""
    if eviction_policy != PromptHistoryEvictionPolicy.after_adapter_commit:
        return committed_index
    if receipt is not None and receipt.status in {
        WriteStatus.committed,
        WriteStatus.accepted_no_change,
    }:
        return max(committed_index, current_index)
    status = receipt.status.value if receipt is not None else "missing"
    if require_committed:
        raise AdapterWriteCommitError(
            f"INVALID_ADAPTER_WRITE: session={session_id} status={status}"
        )
    logger.warning("Memory adapter write not committed for {}: {}", session_id, status)
    return committed_index


def _record_memory_io(
    *,
    audit_sink: MemoryIOAuditSink | None,
    adapter: MemoryAdapter,
    run_id: str,
    step_id: str,
    session_id: str,
    turn_id: str,
    operation: str,
    adapter_request: dict[str, Any],
    status: str,
    backend_request: dict[str, Any] | None = None,
    backend_response: Any | None = None,
    trace: Any | None = None,
    audit_only_labels: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> None:
    if audit_sink is None:
        return
    audit_sink.record(
        {
            "run_id": run_id,
            "step_id": step_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "backend": adapter.backend_name,
            "operation": operation,
            "adapter_request": adapter_request,
            "backend_request": backend_request,
            "backend_response": backend_response,
            "status": status,
            "native_ids": list(trace.native_ids) if trace else [],
            "native_events": list(trace.native_events) if trace else [],
            "result_count": trace.retrieved_count if trace else None,
            "token_count": trace.token_count if trace else None,
            "top_score": trace.top_score if trace else None,
            "latency_ms": trace.latency_ms if trace else None,
            "warnings": list(trace.warnings) if trace else [],
            "audit_only_labels": audit_only_labels or {},
            "error": (
                {"type": type(error).__name__, "message": str(error)}
                if error is not None
                else None
            ),
        }
    )
