"""Deterministic executor for one validated forced-event transcript."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from memory.adapter import MemoryAdapter
from memory.audit import MemoryIOAuditSink
from memory.event_writer import (
    TurnPool,
    build_turn_pool,
    record_adapter_write_commit,
    record_turn_journal,
    turn_journal_for,
    write_visible_memory_event,
)
from memory.event_writer import _compact_json as _canonical_json_body
from memory.turn_experience import DIALOGUE_KINDS, ExperienceItem, render_blocks
from memory.turn_journal import WRITTEN
from pa.agent.loop import ENVIRONMENT_EVENT_ROLE, render_environment_event
from pa.agent.runner import IX_HELD_TOOL_RESULT
from pa.agent.tools.mcp import connect_mcp_servers
from pa.agent.tools.registry import ToolRegistry
from pa.session.manager import SessionManager

from harness.forced_event_replay import (
    ForcedEventReplayError,
    GatedReplay,
    STATE_TOOL_NAME_PREFIX,
    result_contains_expected,
)
from harness.forced_event_schema import (
    EnvironmentActionItem,
    ForcedEventTranscript,
    MessageItem,
    PAToolCallItem,
    SourceRef,
    load_forced_event_transcript,
)
from harness.state_runtime import _STEP_RESET_FILES, StageStepRuntime
from harness.transcript import TranscriptWriter
from state_server.store import GATE_OPEN_MARKER

if TYPE_CHECKING:  # imported for typing only; the outer loop owns the overlay
    from harness.orchestrator import RunPreferenceOverlay


class ForcedEventExecutionError(RuntimeError):
    """Raised when a fixed event cannot be executed exactly as authored."""


def _exchange_turn_id(transcript: ForcedEventTranscript, index: int) -> str:
    """The one id an exchange's adapter event and journal record both carry.

    Receipt and record are matched by `turn_id`, which two sites deriving
    their own strings cannot do -- the defect step 7 found on the ACC line and
    fixed with `PAAgentLoop._turn_id`. This is the FE line's single id, the
    string the adapter event already carried before the journal existed; no
    second scheme is introduced.
    """
    return f"{transcript.event.event_id}:exchange_{index:02d}"


@dataclass
class _PreparedExchange:
    index: int
    timestamp: datetime
    session_messages: list[dict[str, Any]]
    transcript_events: list[dict[str, Any]]
    #: The id the journal record and the adapter event both carry.
    turn_id: str
    #: The `session.messages` slice this exchange owns. Zero-span (`[n, n)`)
    #: when the exchange added no PA-visible message.
    span: tuple[int, int]
    #: The pool journalled before the commit window, written to the adapter
    #: inside it -- one rendering, not two.
    pool: TurnPool
    #: True when a journal record for `turn_id` already existed, i.e. this is a
    #: resume and the exchange's world calls are not run a second time.
    replayed: bool


class ForcedEventExecutor:
    """Replay fixed dialogue while executing its real tool and environment actions."""

    def __init__(
        self,
        *,
        transcript_path: Path | str,
        project_root: Path | str,
        benchmark_session_timestamp: datetime,
        stage_runtime: StageStepRuntime,
        session_manager: SessionManager,
        pa_session_key: str,
        memory_adapter: MemoryAdapter | None,
        memory_io_recorder: MemoryIOAuditSink | None,
        writer: TranscriptWriter,
        require_committed_adapter_writes: bool = True,
        agent_id: str = "nanobot",
        preference_overlay: RunPreferenceOverlay | None = None,
        history_appender: Callable[[str], None] | None = None,
    ) -> None:
        if benchmark_session_timestamp.tzinfo is None:
            raise ValueError("benchmark_session_timestamp must include a timezone")
        self.transcript_path = Path(transcript_path)
        self.project_root = Path(project_root).resolve()
        self.benchmark_session_timestamp = benchmark_session_timestamp
        self.stage_runtime = stage_runtime
        self.session_manager = session_manager
        self.pa_session_key = pa_session_key
        self.memory_adapter = memory_adapter
        self.memory_io_recorder = memory_io_recorder
        self.writer = writer
        self.require_committed_adapter_writes = require_committed_adapter_writes
        self.agent_id = agent_id
        self.preference_overlay = preference_overlay
        self.history_appender = history_appender

    async def run_session(self) -> list[dict[str, Any]]:
        transcript = load_forced_event_transcript(self.transcript_path)
        self._validate_runtime_identity(transcript)
        self._validate_preconditions(transcript)
        persistent_before = _persistent_state_snapshot(self.stage_runtime.state_root)

        prepared: list[_PreparedExchange] = []
        actual_call_ids: list[str] = []
        actual_action_ids: list[str] = []
        preparation_error: Exception | None = None
        async with AsyncExitStack() as stack:
            registry = ToolRegistry()
            await connect_mcp_servers(
                self.stage_runtime.mcp_server_config(),
                registry,
                stack,
            )
            required_names = {
                item.name
                for exchange in transcript.exchanges
                for item in exchange.items
                if isinstance(item, PAToolCallItem)
            }
            missing_names = sorted(required_names.difference(registry.tool_names))
            if missing_names:
                raise ForcedEventExecutionError(
                    f"Required forced-event tools are not enabled: {missing_names}"
                )
            self._replay = self._new_replay(transcript)
            self._open_journal()
            try:
                for exchange_index, exchange in enumerate(
                    transcript.exchanges,
                    start=1,
                ):
                    prepared_exchange = await self._prepare_exchange(
                        transcript,
                        exchange_index=exchange_index,
                        items=exchange.items,
                        registry=registry,
                        actual_call_ids=actual_call_ids,
                        actual_action_ids=actual_action_ids,
                    )
                    prepared.append(prepared_exchange)
            except ForcedEventReplayError as exc:
                preparation_error = ForcedEventExecutionError(str(exc))
            except Exception as exc:
                preparation_error = exc

        if preparation_error is not None:
            raise preparation_error

        self._validate_assertions(
            transcript,
            actual_call_ids=actual_call_ids,
            actual_action_ids=actual_action_ids,
            persistent_before=persistent_before,
            replayed=any(exchange.replayed for exchange in prepared),
        )
        events = self._write_transcript_events(transcript, prepared)
        await self._commit_session_and_memory(transcript, prepared)
        self._commit_simulator_state(transcript)
        end_event = {
            "t": _event_offset(transcript) + 1,
            "kind": "session_end",
            "exchanges": len(transcript.exchanges),
            "tool_calls": len(actual_call_ids),
            "environment_actions": len(actual_action_ids),
        }
        self.writer.log_event(end_event)
        events.append(end_event)
        return events

    def _new_replay(self, transcript: ForcedEventTranscript) -> GatedReplay:
        """The shared replay routine, bound to this step's real stage.

        The gate marker is the file the state server itself watches, so the
        offline replay's in-memory flag and this one are the same decision made
        over two different stores rather than two decisions.
        """
        marker = self.stage_runtime.state_root / GATE_OPEN_MARKER
        return GatedReplay(
            gate=getattr(self.stage_runtime, "gate", None) or None,
            task_facts=dict(transcript.task_facts),
            gate_is_open=marker.exists,
            open_gate=marker.touch,
        )

    def _validate_runtime_identity(self, transcript: ForcedEventTranscript) -> None:
        if transcript.event.event_id != self.stage_runtime.step_id:
            raise ForcedEventExecutionError(
                "Transcript event_id does not match step runtime: "
                f"event_id={transcript.event.event_id!r} "
                f"step_id={self.stage_runtime.step_id!r}"
            )
        if transcript.event.persona_id != self.stage_runtime.user_id:
            raise ForcedEventExecutionError(
                "Transcript persona_id does not match step runtime: "
                f"persona_id={transcript.event.persona_id!r} "
                f"user_id={self.stage_runtime.user_id!r}"
            )

    def _validate_preconditions(self, transcript: ForcedEventTranscript) -> None:
        for precondition in transcript.preconditions.required_fixtures:
            fixture = (self.stage_runtime.state_root / precondition.path).resolve()
            try:
                fixture.relative_to(self.stage_runtime.state_root)
            except ValueError as exc:
                raise ForcedEventExecutionError(
                    f"Fixture precondition escapes stage root: {precondition.path}"
                ) from exc
            if precondition.expectation == "present" and not fixture.is_file():
                raise ForcedEventExecutionError(
                    f"Required fixture is absent from step stage: {precondition.path}"
                )

    async def _prepare_exchange(
        self,
        transcript: ForcedEventTranscript,
        *,
        exchange_index: int,
        items: list[Any],
        registry: ToolRegistry,
        actual_call_ids: list[str],
        actual_action_ids: list[str],
    ) -> _PreparedExchange:
        session_messages: list[dict[str, Any]] = []
        transcript_events: list[dict[str, Any]] = []
        first_visible_timestamp: datetime | None = None
        turn_id = _exchange_turn_id(transcript, exchange_index)
        record = self._journal_record(turn_id)
        realized = self._realized_artefacts(turn_id, record, items)
        if record is not None:
            # The exchange is realized. Its `$call_*`
            # reference map goes back into the replay before any argument is
            # resolved, so a reference this exchange's own later call makes
            # resolves to the id the first run minted -- and the registry is
            # not touched below.
            self._replay.results.update(realized["results"])
        for item in items:
            timestamp = self.benchmark_session_timestamp + timedelta(
                seconds=item.offset_seconds
            )
            timestamp_text = timestamp.isoformat()
            if isinstance(item, MessageItem):
                transcript_events.append(
                    {
                        "t": item.offset_seconds,
                        "kind": "message",
                        "exchange": exchange_index,
                        "source": _source_value(item.source),
                        "visibility": item.visibility,
                        "content": item.content,
                        "timestamp": timestamp_text,
                    }
                )
                self._replay.observe_message(
                    source=item.source,
                    visibility=item.visibility,
                    reveals=item.reveals,
                    content=item.content,
                )
                if item.visibility in {"pa", "both"}:
                    # `source` is `Literal["user", "pa"] | SourceRef`, and a
                    # `SourceRef` source -- `third_party` or `environment` --
                    # is the discriminator the pool routes on: `_experience_pool`
                    # reads this dict and tags the line by its own
                    # `source.kind`, so the PA's whole experience of the
                    # exchange reaches memory without the world's or a third
                    # party's words ever being attributed to the person. `role`
                    # here serves the PA's prompt (`Session.get_history` copies
                    # role and content and reads nothing else) and no longer
                    # decides anything about memory.
                    message: dict[str, Any] = {
                        "role": "assistant" if item.source == "pa" else "user",
                        "content": item.content,
                        "timestamp": timestamp_text,
                        "offset_seconds": item.offset_seconds,
                        "source": _source_value(item.source),
                        "visibility": item.visibility,
                    }
                    session_messages.append(message)
                    first_visible_timestamp = first_visible_timestamp or timestamp
                continue
            if isinstance(item, PAToolCallItem):
                # Ids the world assigns at run time (draft_0007, res_0003, ...)
                # cannot be known when the transcript is authored; arguments
                # name an earlier call or delivered action instead
                # (``$call_evolv_04_07.reservation_id``, see harness.tool_refs).
                arguments = self._replay.resolve_arguments(
                    call_id=item.call_id,
                    arguments=dict(item.arguments),
                    tool=item.name,
                )
                # On a realized exchange the record decides, not the gate: the
                # gate marker is a file that survives the crash, so re-asking
                # `is_held` could answer differently the second time round and
                # run a call the first run held (or refuse one it ran).
                held = (
                    item.call_id not in realized["results"]
                    if record is not None
                    else self._replay.is_held(
                        call_id=item.call_id, name=item.name, marked_held=item.held
                    )
                )
                if held:
                    session_messages.extend(
                        _tool_call_messages(
                            item,
                            arguments=arguments,
                            result=json.loads(IX_HELD_TOOL_RESULT),
                            timestamp_text=timestamp_text,
                        )
                    )
                    transcript_events.append(
                        _pa_turn_event(
                            item,
                            exchange_index=exchange_index,
                            arguments=arguments,
                            result=json.loads(IX_HELD_TOOL_RESULT),
                            status="held",
                            timestamp_text=timestamp_text,
                        )
                    )
                    first_visible_timestamp = first_visible_timestamp or timestamp
                    continue
                if record is not None:
                    # The world already answered this call; Q5 authorises
                    # replaying a memory write, never a world action.
                    actual_result = realized["results"][item.call_id]
                else:
                    actual_result = _decode_tool_result(
                        item.name, await registry.execute(item.name, arguments)
                    )
                    self._replay.results[item.call_id] = actual_result
                    realized["results"][item.call_id] = actual_result
                if isinstance(actual_result, dict) and actual_result.get("ok") is False:
                    raise ForcedEventExecutionError(
                        "Forced-event tool call failed against the world: "
                        f"call_id={item.call_id!r} tool={item.name!r} "
                        f"result={actual_result!r}"
                    )
                if item.expected_contains is not None and not result_contains_expected(
                    actual_result, item.expected_contains
                ):
                    raise ForcedEventExecutionError(
                        "Forced-event tool result is missing its pinned content: "
                        f"call_id={item.call_id!r} tool={item.name!r} "
                        f"expected_contains={item.expected_contains!r} "
                        f"actual={actual_result!r}"
                    )
                actual_call_ids.append(item.call_id)
                session_messages.extend(
                    _tool_call_messages(
                        item,
                        arguments=arguments,
                        result=actual_result,
                        timestamp_text=timestamp_text,
                    )
                )
                transcript_events.append(
                    _pa_turn_event(
                        item,
                        exchange_index=exchange_index,
                        arguments=arguments,
                        result=actual_result,
                        status="ok",
                        timestamp_text=timestamp_text,
                    )
                )
                first_visible_timestamp = first_visible_timestamp or timestamp
                continue
            if isinstance(item, EnvironmentActionItem):
                if record is not None:
                    # A delivery is a world action too: re-running it would put
                    # a second copy of the message in the inbox or the thread.
                    executed = item.action_id in realized["action_ids"]
                else:
                    executed = self._prepare_environment_action(item)
                if executed:
                    if record is None:
                        delivered = self._deliver_to_world(item)
                        if delivered is not None:
                            self._replay.results[item.action_id] = delivered
                            realized["results"][item.action_id] = delivered
                        realized["action_ids"].append(item.action_id)
                    actual_action_ids.append(item.action_id)
                    if item.visibility in {"pa", "both"}:
                        environment = item.source.kind == "environment"
                        session_messages.append(
                            {
                                # The PA has to see what was delivered -- the
                                # user's next line answers it -- so this role
                                # is the one `Session.get_history` copies into
                                # the prompt, unchanged. Memory no longer reads
                                # it: `_experience_pool` routes the line by its
                                # own `source.kind` into an `[environment_event]`
                                # or `[third_party]` block on an `assistant`
                                # message, so the delivery reaches the pool and
                                # is never attributed to the person.
                                #
                                # An `environment` delivery is the same kind of
                                # thing as an ACC environment event, so it
                                # reaches the PA the way ACC's does: the role
                                # and the header come from `pa.agent.loop`
                                # (`ENVIRONMENT_EVENT_ROLE`,
                                # `render_environment_event`), imported rather
                                # than copied, so the two paths cannot drift.
                                # `role: "user"` here used to tell the PA the
                                # *person* had said it.
                                #
                                # A `third_party` delivery keeps the old role:
                                # ACC has no injected-message path for one (a
                                # third party reaches an ACC session as a world
                                # record the PA reads), so there is no ACC
                                # format to match and inventing one is not this
                                # change's to make.
                                "role": (
                                    ENVIRONMENT_EVENT_ROLE if environment else "user"
                                ),
                                "content": (
                                    render_environment_event(item.content)
                                    if environment
                                    else item.content
                                ),
                                "timestamp": timestamp_text,
                                "offset_seconds": item.offset_seconds,
                                "source": item.source.model_dump(),
                                "visibility": item.visibility,
                                "environment_action_id": item.action_id,
                            }
                        )
                        first_visible_timestamp = first_visible_timestamp or timestamp
                transcript_events.append(
                    {
                        "t": item.offset_seconds,
                        "kind": "environment_action",
                        "exchange": exchange_index,
                        "action_id": item.action_id,
                        "action": item.action,
                        "source": item.source.model_dump(),
                        "visibility": item.visibility,
                        "content": item.content,
                        "status": "executed" if executed else "missing",
                        "timestamp": timestamp_text,
                    }
                )
                continue
            raise ForcedEventExecutionError(
                f"Unsupported transcript item at runtime: {type(item).__name__}"
            )
        _validate_tool_trajectory(session_messages)
        timestamp = first_visible_timestamp or self.benchmark_session_timestamp
        pool = _experience_pool(session_messages)
        if record is not None:
            span = record.span
        else:
            # The span the exchange will own once `_commit_session_and_memory`
            # extends the session. An exchange that added no PA-visible
            # message owns `[n, n)` -- the zero-span rule -- and is still
            # journalled, because its world actions are realized all the same.
            span = (self._span_cursor, self._span_cursor + len(session_messages))
            self._span_cursor = span[1]
            # Durable before the first `session_manager.save`
            # -- and here that is the whole preparation loop earlier, not
            # just one statement: `run_session` prepares every exchange before
            # it commits any (`:131-143`, `:159`). The record is written at the
            # end of preparation rather than at the top because `realized` does
            # not exist until `registry.execute` has run; writing it earlier
            # would journal an exchange whose world artefacts a resume could
            # not restore, which is the re-execution this is here to prevent.
            record_turn_journal(
                journal=self._journal,
                turn_id=turn_id,
                session_id=self.pa_session_key,
                span=span,
                timestamp=timestamp,
                pool=pool,
                state="pending",
                realized=realized,
            )
        return _PreparedExchange(
            index=exchange_index,
            timestamp=timestamp,
            session_messages=session_messages,
            transcript_events=transcript_events,
            turn_id=turn_id,
            span=span,
            pool=pool,
            replayed=record is not None,
        )

    def _open_journal(self) -> None:
        """Bind this step's journal and the span it will write into.

        Read fresh, so a resume sees what the crashed run left behind.
        The span cursor starts where the session already ends: on a first
        run that is the PA's own tail, on a resume it already includes any
        exchange the crashed run committed — and an exchange that already has a
        record carries its own span anyway.
        """
        self._journal = turn_journal_for(
            self.session_manager, self.pa_session_key
        )
        self._span_cursor = len(
            self.session_manager.get(self.pa_session_key).messages
        )

    def _journal_record(self, turn_id: str):
        """This exchange's journal record, or `None` if it has never run."""
        from memory.turn_journal import UnknownTurnError

        try:
            return self._journal.record_for(turn_id)
        except UnknownTurnError:
            return None

    def _realized_artefacts(
        self, turn_id: str, record: Any, items: list[Any]
    ) -> dict[str, Any]:
        """What the world handed this exchange back, restored or empty.

        No compat path: a record for an exchange that touched the world and
        carries no `realized` cannot say which calls already ran, and the only
        thing left to do with it would be to run them again. That is the one
        outcome Q5 does not authorise, so it is an error.
        """
        if record is None:
            return {"results": {}, "action_ids": []}
        if record.realized is None:
            touches_world = any(
                isinstance(item, (PAToolCallItem, EnvironmentActionItem))
                for item in items
            )
            if touches_world:
                raise ForcedEventExecutionError(
                    "Forced-event journal record carries no realized artefacts "
                    "for an exchange with world calls, so a resume cannot tell "
                    "which of them already ran: "
                    f"turn_id={turn_id!r}"
                )
            return {"results": {}, "action_ids": []}
        return {
            "results": dict(record.realized.get("results") or {}),
            "action_ids": list(record.realized.get("action_ids") or []),
        }

    def _prepare_environment_action(
        self,
        item: EnvironmentActionItem,
    ) -> bool:
        if item.action != "deliver_message":
            raise ForcedEventExecutionError(
                f"Unsupported environment action: {item.action!r}"
            )
        return True

    def _deliver_to_world(self, item: EnvironmentActionItem) -> dict[str, Any] | None:
        """Land a third-party message in the inbox or SMS store, as it would in life.

        Returns the stored record so later calls can reference it
        (``$<action_id>.id`` / ``$<action_id>.message_id``).
        """
        if item.source.kind != "third_party":
            return None
        from state_server.store import StateStore

        store = StateStore(
            self.stage_runtime.state_root,
            run_id=self.stage_runtime.run_id,
            user_id=self.stage_runtime.user_id,
            session_id=self.stage_runtime.session_id,
            session_date=self.stage_runtime.session_date,
            session_timestamp=self.stage_runtime.session_timestamp,
        )
        sender = item.sender or item.source.id
        if item.channel == "message":
            return store.message_deliver(sender=sender, body=item.content)
        subject = item.subject or item.content.strip().splitlines()[0][:80]
        return store.inbox_deliver(sender=sender, subject=subject, body=item.content)

    def _validate_assertions(
        self,
        transcript: ForcedEventTranscript,
        *,
        actual_call_ids: list[str],
        actual_action_ids: list[str],
        persistent_before: dict[str, str],
        replayed: bool = False,
    ) -> None:
        if actual_call_ids != transcript.assertions.required_tool_calls:
            raise ForcedEventExecutionError(
                "Required tool calls were not executed exactly once in order: "
                f"expected={transcript.assertions.required_tool_calls!r} "
                f"actual={actual_call_ids!r}"
            )
        if actual_action_ids != transcript.assertions.expected_environment_actions:
            raise ForcedEventExecutionError(
                "Expected environment actions were not executed exactly once in order: "
                f"expected={transcript.assertions.expected_environment_actions!r} "
                f"actual={actual_action_ids!r}"
            )
        persistent_after = _persistent_state_snapshot(self.stage_runtime.state_root)
        expectation = transcript.assertions.expected_persistent_state_diff
        # Third-party deliveries land in the world by design; they are the
        # environment's doing, not the PA's, so they do not count as the
        # event's own mutations.
        delivered_paths = {
            "email/inbox.jsonl" if item.channel == "email" else "message/threads.jsonl"
            for exchange in transcript.exchanges
            for item in exchange.items
            if isinstance(item, EnvironmentActionItem) and item.source.kind == "third_party"
        }
        changed_paths = _changed_paths(persistent_before, persistent_after)
        state_diff_records = [
            record
            for record in _load_state_diff_records(
                self.stage_runtime.state_root / "state_diff.jsonl"
            )
            if not str(record.get("summary", "")).startswith("Delivered ")
        ]
        if expectation == "none" and set(changed_paths) - delivered_paths:
            raise ForcedEventExecutionError(
                "Forced event changed persistent state despite expected_persistent_state_diff=none: "
                f"changed_paths={sorted(set(changed_paths) - delivered_paths)!r}"
            )
        if expectation == "none" and state_diff_records:
            raise ForcedEventExecutionError(
                "Forced event produced StateServer mutations despite expected_persistent_state_diff=none"
            )
        if expectation != "none":
            expected_paths = sorted(set(expectation.changed_paths) | delivered_paths)
            # A resumed run realized some of the event's world calls in the
            # process that crashed, so its own before/after snapshot cannot
            # show their paths: what it must show is that it changed nothing
            # the event does not own. `state_diff.jsonl` is the step's audit
            # file, not this process's, so the mutation count below still
            # spans both and keeps its equality.
            footprint_ok = (
                set(changed_paths) <= set(expected_paths)
                if replayed
                else changed_paths == expected_paths
            )
            if not footprint_ok:
                raise ForcedEventExecutionError(
                    "Forced event persistent changed paths do not match the transcript: "
                    f"expected={expected_paths!r} actual={changed_paths!r}"
                )
            if len(state_diff_records) != expectation.state_diff_count:
                raise ForcedEventExecutionError(
                    "Forced event StateServer mutation count does not match the transcript: "
                    f"expected={expectation.state_diff_count} "
                    f"actual={len(state_diff_records)}"
                )

    def _write_transcript_events(
        self,
        transcript: ForcedEventTranscript,
        prepared: list[_PreparedExchange],
    ) -> list[dict[str, Any]]:
        start = {
            "t": 0,
            "kind": "session_start",
            "persona": transcript.event.persona_id,
            "session_id": transcript.event.event_id,
            "context": transcript.event.context,
            "scenario": "forced_event",
            "model": "fixed-transcript",
            "beat_names": [],
        }
        self.writer.log_event(start)
        events = [start]
        for exchange in prepared:
            exchange_start = {
                "t": exchange.transcript_events[0]["t"],
                "kind": "exchange_start",
                "exchange": exchange.index,
                "timestamp": exchange.timestamp.isoformat(),
            }
            self.writer.log_event(exchange_start)
            events.append(exchange_start)
            for event in exchange.transcript_events:
                self.writer.log_event(event)
                events.append(event)
        return events

    async def _commit_session_and_memory(
        self,
        transcript: ForcedEventTranscript,
        prepared: list[_PreparedExchange],
    ) -> None:
        session = self.session_manager.get(self.pa_session_key)
        for exchange in prepared:
            record = self._journal_record(exchange.turn_id)
            if record is not None and record.state == WRITTEN:
                # The crashed run got this exchange's receipt in; its messages
                # and its cursor are on disk. Re-committing it would put a
                # second copy of the dialogue in the session.
                continue
            if len(session.messages) < exchange.span[1]:
                # Not already extended by the run that crashed.
                session.messages.extend(exchange.session_messages)
            session.updated_at = datetime.now()
            self.session_manager.save(session)
            pool = exchange.pool
            if not pool.messages:
                continue
            turn_id = exchange.turn_id
            receipt = await write_visible_memory_event(
                adapter=self.memory_adapter,
                messages=exchange.session_messages,
                run_id=self.stage_runtime.run_id,
                persona_id=transcript.event.persona_id,
                agent_id=self.agent_id,
                session_id=self.pa_session_key,
                turn_id=turn_id,
                event_id=f"{self.stage_runtime.run_id}:{turn_id}",
                timestamp=exchange.timestamp,
                step_id=transcript.event.event_id,
                audit_sink=self.memory_io_recorder,
                audit_only_labels={"script_session_id": transcript.event.event_id},
                require_committed=self.require_committed_adapter_writes,
                eviction_policy=(
                    self.memory_adapter.prompt_history_eviction_policy
                    if self.memory_adapter is not None
                    else None
                ),
                pool=pool,
            )
            if self.memory_adapter is not None:
                session.adapter_committed_end_index = record_adapter_write_commit(
                    receipt=receipt,
                    session_id=session.key,
                    current_index=len(session.messages),
                    committed_index=session.adapter_committed_end_index,
                    eviction_policy=self.memory_adapter.prompt_history_eviction_policy,
                    require_committed=self.require_committed_adapter_writes,
                )
                self.session_manager.save(session)
            # After the receipt is recorded and inside the same commit window
            # as the cursor advance, exactly as the ACC loop orders it
            # (`pa/agent/loop.py:1651-1680`). A run with no adapter has no
            # receipt, so its record stays `pending` and the file condition's
            # own mark, `file_consolidated`, is what completes it.
            record_turn_journal(
                journal=self._journal,
                turn_id=turn_id,
                session_id=self.pa_session_key,
                span=exchange.span,
                timestamp=exchange.timestamp,
                pool=pool,
                state="written",
                receipt=receipt,
            )

    def _commit_simulator_state(self, transcript: ForcedEventTranscript) -> None:
        """Move the simulated user's run-level state past the event.

        Both writes belong to the step: the orchestrator only marks it done
        after `run_session` returns, so an event that fails or is skipped
        leaves the overlay and HISTORY.md untouched. Neither write depends on
        `memory_condition` — the simulated user is not the system under test,
        and she has to have lived through the event in every condition.
        """
        event = transcript.event
        if self.preference_overlay is not None:
            self.preference_overlay.record(
                f"{event.context}_{event.target_attribute}",
                event.post_setting,
            )
        if self.history_appender is not None:
            stamp = self.benchmark_session_timestamp.strftime("%Y-%m-%d %H:%M")
            seed = transcript.user_memory_seed
            self.history_appender(f"[{stamp}] {seed.strip()}")


#: The block tag each `SourceRef.kind` routes to
#: (`harness/forced_event_schema.py` `SourceRef`). The item's own kind decides
#: the tag, for a `MessageItem` and an `EnvironmentActionItem` alike, never the
#: call site that produced it.
_SOURCE_KIND_TAGS = {"environment": "environment_event", "third_party": "third_party"}

#: The body `projected_memory_records` puts where a live `[tool_result]` would
#: carry the world's answer.  The preview is written before replay, so there is
#: no answer yet; this stands in for one.  It is a body and not an attribute
#: because the grammar declares `[tool_result]`'s keys and `status` is not one
#: of them (`memory/turn_experience.py:BLOCK_ATTRIBUTES`).
_NOT_EXECUTED = "not executed"


def _experience_pool(session_messages: list[dict[str, Any]]) -> TurnPool:
    """One exchange's pool: the PA's whole first-person experience of it.

    The harness performs no selection (owner ruling 2026-09-13). What the PA
    said, what the person said, what the PA did and learned through a tool, and
    what the world or a third party delivered into the conversation are all in
    it. Dialogue stays bare and byte-equal -- `simple_rag` compares those bytes
    (`memory/backends/mem0_client.py`) -- and every other item is a tagged block
    on an `assistant` message, so nothing that is not the person's own speech is
    ever `role: "user"`.

    Two things a forced-event message carries that the shared builder cannot
    read off a message dict are handled here: the `source` of a delivered line,
    which routes it to `[environment_event]` or `[third_party]`, and the item's
    `offset_seconds`, which reaches the wire as `offset=` on its block.
    """
    messages: list[dict[str, str]] = []
    items: list[ExperienceItem] = []
    for message in session_messages:
        offset = message.get("offset_seconds")
        source = message.get("source")
        if isinstance(source, dict):
            group = [_source_item(message, source, offset, len(items))]
        else:
            group = _dialogue_items(message, offset, len(items))
        if not group:
            continue
        role = "user" if group[0].kind == "user" else "assistant"
        messages.append({"role": role, "content": render_blocks(group)})
        items.extend(group)
    return TurnPool(messages=messages, items=items)


def _dialogue_items(
    message: dict[str, Any],
    offset: int | None,
    start_order: int,
) -> list[ExperienceItem]:
    """The items of one spoken line, or one tool call or result.

    `build_turn_pool` is the one renderer for these on both lines; a forced
    event's session message is a single item, so it is rendered one message at
    a time and the offset is stamped on whatever blocks come back. A bare
    dialogue message carries no tag and so no offset.
    """
    pool = build_turn_pool([message])
    if len(pool.messages) > 1:
        raise ForcedEventExecutionError(
            "Forced-event session message rendered as more than one pool "
            f"message: {message!r}"
        )
    return [
        replace(
            item,
            order=start_order + index,
            offset_seconds=None if item.kind in DIALOGUE_KINDS else offset,
        )
        for index, item in enumerate(pool.items)
    ]


def _source_item(
    message: dict[str, Any],
    source: dict[str, Any],
    offset: int | None,
    order: int,
) -> ExperienceItem:
    """One delivered world or third-party line, tagged by its own `source.kind`."""
    kind = str(source.get("kind") or "")
    tag = _SOURCE_KIND_TAGS.get(kind)
    if tag is None:
        raise ForcedEventExecutionError(
            f"Unsupported forced-event source kind: {kind!r}"
        )
    attrs = {"source_kind": kind, "source_id": str(source.get("id") or "")}
    action_id = message.get("environment_action_id")
    if action_id:
        # Only an `EnvironmentActionItem` has one; a `message` with a
        # `SourceRef` source has none and emits none.
        attrs["action_id"] = str(action_id)
    return ExperienceItem(
        kind=tag,
        role="assistant",
        order=order,
        offset_seconds=offset,
        attrs=attrs,
        body=str(message.get("content") or ""),
    )


def projected_memory_records(
    transcript: ForcedEventTranscript,
) -> list[dict[str, str]]:
    """A static semantic preview of the pool, for the transcript review skill.

    It renders what the authored transcript already knows, in the pool's own
    block grammar: the spoken lines bare, a delivered world or third-party line
    as its `source.kind`'s block, and each PA tool call as a `[tool_call]`
    followed by a `[tool_result]` whose body is the `not executed` placeholder.

    **It is not the resume oracle and not a prediction of the live pool.** The
    world answers a call at replay time -- ids like `draft_0007` and `res_0003`
    are assigned then, `$call_*` references resolve then, and a gated call's
    result is the hold notice -- so no result exists at authoring time, and the
    `not executed` body says so. It stands where a live result's persisted text
    would be, and carries no attribute the grammar does not declare, so the
    preview renders under the same renderer as the live pool and cannot drift
    into a spelling that renderer would refuse. Nothing asserts that this equals
    what a run writes; a reviewer reads it to see whether the event is legible
    in what the PA will carry away from it.
    """
    records: list[dict[str, str]] = []
    order = 0

    def emit(role: str, item: ExperienceItem) -> None:
        nonlocal order
        records.append({"role": role, "content": render_blocks([item])})
        order += 1

    for exchange in transcript.exchanges:
        for item in exchange.items:
            if isinstance(item, MessageItem):
                if item.visibility not in {"pa", "both"}:
                    continue
                if isinstance(item.source, SourceRef):
                    emit(
                        "assistant",
                        _source_item(
                            {"content": item.content},
                            item.source.model_dump(),
                            item.offset_seconds,
                            order,
                        ),
                    )
                    continue
                role = "assistant" if item.source == "pa" else "user"
                emit(
                    role,
                    ExperienceItem(
                        kind=role,
                        role=role,
                        order=order,
                        offset_seconds=None,
                        attrs={},
                        body=item.content,
                    ),
                )
                continue
            if isinstance(item, PAToolCallItem):
                emit(
                    "assistant",
                    ExperienceItem(
                        kind="tool_call",
                        role="assistant",
                        order=order,
                        offset_seconds=item.offset_seconds,
                        attrs={"name": item.name, "id": item.call_id},
                        body=_compact_json(item.arguments),
                    ),
                )
                emit(
                    "assistant",
                    ExperienceItem(
                        kind="tool_result",
                        role="assistant",
                        order=order,
                        offset_seconds=item.offset_seconds,
                        attrs={"name": item.name, "id": item.call_id},
                        body=_NOT_EXECUTED,
                    ),
                )
                continue
            if isinstance(item, EnvironmentActionItem):
                if item.visibility not in {"pa", "both"}:
                    continue
                emit(
                    "assistant",
                    _source_item(
                        {
                            "content": item.content,
                            "environment_action_id": item.action_id,
                        },
                        item.source.model_dump(),
                        item.offset_seconds,
                        order,
                    ),
                )
    return records


def _tool_call_messages(
    item: PAToolCallItem,
    *,
    arguments: dict[str, Any],
    result: Any,
    timestamp_text: str,
) -> list[dict[str, Any]]:
    """The PA session's record of one call: the call, then its result.

    A held call keeps this shape exactly -- the tool protocol has to stay well
    formed whether or not the call ran -- and its result is the accumulation
    line's own `IX_HELD_TOOL_RESULT`, which says in as many words that the call
    did not happen and must not be reported as done.
    """
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": _compact_json(arguments),
                    },
                }
            ],
            "timestamp": timestamp_text,
            "offset_seconds": item.offset_seconds,
            "source": "pa",
        },
        {
            "role": "tool",
            "content": _compact_json(result),
            "tool_call_id": item.call_id,
            "name": item.name,
            "timestamp": timestamp_text,
            # A `[tool_result]` inherits the offset of the item that
            # produced it, so both halves of one call carry the call's own.
            "offset_seconds": item.offset_seconds,
        },
    ]


def _pa_turn_event(
    item: PAToolCallItem,
    *,
    exchange_index: int,
    arguments: dict[str, Any],
    result: Any,
    status: str,
    timestamp_text: str,
) -> dict[str, Any]:
    """The transcript's record of one call. `status` is `ok` or `held`."""
    return {
        "t": item.offset_seconds,
        "kind": "pa_turn",
        "forced_event": True,
        "exchange": exchange_index,
        "outbound": "",
        "tool_events": [
            {
                "call_id": item.call_id,
                "name": item.name,
                "status": status,
                "args": arguments,
                "detail": result,
            }
        ],
        "timestamp": timestamp_text,
    }


def _compact_json(value: Any) -> str:
    """The canonical compact JSON, for an object the FE line holds.

    The wire form is never spelled a second time here: this serialises once and
    hands the text to `memory.event_writer`'s canonicaliser, which is the one
    definition of that form for every JSON-bodied block on both lines.  The
    inner `dumps` may escape non-ASCII; the canonicaliser re-emits with
    `ensure_ascii=False`, so the characters come back.
    """
    return _canonical_json_body(json.dumps(value, default=str))


def _decode_tool_result(tool_name: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ForcedEventExecutionError(
            f"Tool {tool_name!r} did not return a JSON value: {value!r}"
        ) from exc


def _source_value(source: str | SourceRef) -> str | dict[str, str]:
    return source if isinstance(source, str) else source.model_dump()


def _validate_tool_trajectory(messages: list[dict[str, Any]]) -> None:
    declared: list[str] = []
    results: list[str] = []
    for message in messages:
        if message.get("role") == "assistant":
            declared.extend(
                str(call.get("id"))
                for call in message.get("tool_calls") or []
                if call.get("id")
            )
        elif message.get("role") == "tool":
            results.append(str(message.get("tool_call_id") or ""))
    if declared != results:
        raise ForcedEventExecutionError(
            f"Invalid assistant tool-call trajectory: calls={declared!r} results={results!r}"
        )


def _persistent_state_snapshot(root: Path) -> dict[str, str]:
    # The audit files and the gate / session-closing markers are harness and
    # store bookkeeping sitting in the state root, not state the event changed;
    # `begin_step` and `finish_step` clear the same list.
    ignored = set(_STEP_RESET_FILES)
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in ignored:
            continue
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(
        path
        for path in set(before).union(after)
        if before.get(path) != after.get(path)
    )


def _load_state_diff_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ForcedEventExecutionError(
                f"StateServer state diff record must be an object: {value!r}"
            )
        records.append(value)
    return records


def _event_offset(transcript: ForcedEventTranscript) -> int:
    return max(
        item.offset_seconds
        for exchange in transcript.exchanges
        for item in exchange.items
    )
