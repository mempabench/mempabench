"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from memory.adapter import PromptHistoryEvictionPolicy
from memory.event_writer import (
    consolidation_records,
    records_owed_to_files,
    turn_journal_for,
)
from pa.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from memory.turn_journal import TurnJournal
    from pa.providers.base import LLMProvider, LLMTaskIdentity
    from pa.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


class MemoryConsolidationError(RuntimeError):
    """Raised when a benchmark consolidation cannot produce valid memory."""


class MemoryConsolidationContractError(MemoryConsolidationError):
    """Raised when the consolidation LM violates the save_memory contract."""


class PromptHistoryEvictionBlockedError(RuntimeError):
    """Raised when token pressure would evict uncommitted adapter input."""


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MarkdownMemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(
        self,
        workspace: Path,
        *,
        log_scope: str,
    ):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.log_scope = log_scope

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self._require_writable()
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        self._require_writable()
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def _require_writable(self) -> None:
        if (self.memory_dir / ".mempa-read-only").is_file():
            raise PermissionError("Shared Probe file memory is immutable")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] "
                f"{message['role'].upper()}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        consolidation_llm_provider: LLMProvider,
        consolidation_model: str,
        task_identity: LLMTaskIdentity | None = None,
        *,
        max_tokens: int | None = None,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md."""
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
            {"role": "user", "content": prompt},
        ]

        forced = {"type": "function", "function": {"name": "save_memory"}}
        request_kwargs = {
            "messages": chat_messages,
            "tools": _SAVE_MEMORY_TOOL,
            "model": consolidation_model,
            "tool_choice": forced,
        }
        if task_identity is not None:
            request_kwargs["task_identity"] = task_identity
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        response = await consolidation_llm_provider.chat_with_retry(
            **request_kwargs,
        )

        if response.finish_reason == "error" and _is_tool_choice_unsupported(
            response.content
        ):
            logger.warning("Forced tool_choice unsupported, retrying with auto")
            request_kwargs["tool_choice"] = "auto"
            response = await consolidation_llm_provider.chat_with_retry(
                **request_kwargs,
            )

        def contract_failure(message: str) -> MemoryConsolidationContractError:
            # The upstream that served this response is charged for it if
            # the response is empty (provider avoidance,
            # `harness/provider_avoidance.py`).
            consolidation_llm_provider.record_contract_failure(
                response, f"file_consolidation: {message}"
            )
            return MemoryConsolidationContractError(message)

        if not response.has_tool_calls:
            provider_error_detail = (
                json.dumps(
                    response.error_detail,
                    ensure_ascii=False,
                    default=str,
                )[:2_000]
                if response.error_detail
                else None
            )
            raise contract_failure(
                "No save_memory call: "
                f"finish_reason={response.finish_reason!r} "
                f"content_preview={(response.content or '')[:1_000]!r} "
                f"provider_error_detail={provider_error_detail!r}"
            )

        raw_args = response.tool_calls[0].arguments
        try:
            args = _normalize_save_memory_args(raw_args)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise contract_failure(
                "Invalid save_memory arguments: "
                f"value={str(raw_args)[:200]!r} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        if args is None:
            raise contract_failure(
                "Unexpected save_memory arguments: "
                f"value={str(raw_args)[:200]!r}"
            )

        missing_fields = [
            field for field in ("history_entry", "memory_update") if field not in args
        ]
        if missing_fields:
            raise contract_failure(
                "Missing save_memory fields: "
                f"missing_fields={missing_fields!r} value={str(args)[:200]!r}"
            )

        entry = args["history_entry"]
        update = args["memory_update"]
        if entry is None or update is None:
            raise contract_failure(
                "Null save_memory field: "
                f"value={str(args)[:200]!r}"
            )

        entry = _ensure_text(entry).strip()
        if not entry:
            raise contract_failure(
                "Empty save_memory history_entry: "
                f"value={str(args)[:200]!r}"
            )

        self.append_history(entry)
        update = _ensure_text(update)
        memory_updated = update != current_memory
        if memory_updated:
            self.write_long_term(update)

        logger.info(
            "Markdown memory consolidation committed: scope={} messages={} "
            "memory_updated={} history_appended=true",
            self.log_scope,
            len(messages),
            memory_updated,
        )
        return True


class MemoryConsolidator:
    """Compacts PA prompt history and writes NanoBot Markdown memory when selected."""

    _MAX_COMPACTION_ROUNDS = 5
    _MAX_STRICT_CONSOLIDATION_ATTEMPTS = 3
    _PROMPT_TOKEN_SAFETY_BUFFER = 1024
    _PROMPT_COMPACTION_TRIGGER_TOKENS = 20_000

    def __init__(
        self,
        workspace: Path,
        pa_llm_provider: LLMProvider,
        pa_model: str,
        consolidation_llm_provider: LLMProvider,
        consolidation_model: str,
        session_manager: SessionManager,
        pa_context_window_tokens: int,
        build_pa_prompt_messages: Callable[..., list[dict[str, Any]]],
        get_pa_tool_definitions: Callable[[], list[dict[str, Any]]],
        prompt_history_eviction_policy: PromptHistoryEvictionPolicy,
        pa_max_completion_tokens: int = 4096,
        require_committed_adapter_writes: bool = False,
    ):
        self.markdown_store = MarkdownMemoryStore(
            workspace,
            log_scope="pa_file_memory",
        )
        self.pa_llm_provider = pa_llm_provider
        self.pa_model = pa_model
        self.consolidation_llm_provider = consolidation_llm_provider
        self.consolidation_model = consolidation_model
        self.session_manager = session_manager
        self.pa_context_window_tokens = pa_context_window_tokens
        self.pa_max_completion_tokens = pa_max_completion_tokens
        self.prompt_history_eviction_policy = prompt_history_eviction_policy
        self.require_committed_adapter_writes = require_committed_adapter_writes
        self._build_pa_prompt_messages = build_pa_prompt_messages
        self._get_pa_tool_definitions = get_pa_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages_to_files(
        self,
        messages: list[dict[str, object]],
        *,
        task_identity: LLMTaskIdentity,
    ) -> bool:
        """Archive a selected message chunk into persistent memory."""
        return await self.markdown_store.consolidate(
            messages,
            self.consolidation_llm_provider,
            self.consolidation_model,
            task_identity,
            # The file tool must return the complete MEMORY.md plus one history
            # entry. Use the frozen PA output budget rather than the separately
            # constructed consolidation provider's smaller default.
            max_tokens=self.pa_max_completion_tokens,
        )

    async def require_file_consolidation(
        self,
        messages: list[dict[str, object]],
        *,
        session_key: str,
        chunk_start_index: int | None = None,
        chunk_end_index: int | None = None,
    ) -> None:
        """Require a valid summary after bounded contract retries."""
        from pa.providers.base import LLMTaskIdentity

        if (chunk_start_index is None) != (chunk_end_index is None):
            raise ValueError(
                "Consolidation chunk indices must be provided together: "
                f"chunk_start_index={chunk_start_index!r} "
                f"chunk_end_index={chunk_end_index!r}"
            )
        if chunk_start_index is None:
            serialized_messages = json.dumps(
                messages,
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            chunk_identity = hashlib.sha256(serialized_messages).hexdigest()
        else:
            chunk_identity = f"{chunk_start_index}:{chunk_end_index}"
        task_identity = LLMTaskIdentity(
            logical_task_key=(
                "file_consolidation:v1:"
                f"{session_key}:{chunk_identity}"
            )
        )
        last_error: MemoryConsolidationContractError | None = None
        for _ in range(self._MAX_STRICT_CONSOLIDATION_ATTEMPTS):
            try:
                await self.consolidate_messages_to_files(
                    messages,
                    task_identity=task_identity,
                )
                return
            except MemoryConsolidationContractError as exc:
                last_error = exc
        raise MemoryConsolidationError(
            "INVALID_CONSOLIDATION: "
            f"session={session_key} failed after "
            f"{self._MAX_STRICT_CONSOLIDATION_ATTEMPTS} attempts "
            f"last_error={last_error}"
        ) from last_error

    def select_compaction_end_index(
        self,
        session: Session,
        prompt_tokens_to_remove: int,
        *,
        safe_eviction_end_index: int | None = None,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start_index = session.prompt_history_start_index
        if start_index >= len(session.messages) or prompt_tokens_to_remove <= 0:
            return None

        selected_message_tokens = 0
        last_user_turn_boundary: tuple[int, int] | None = None
        for message_index in range(start_index, len(session.messages)):
            if (
                safe_eviction_end_index is not None
                and message_index > safe_eviction_end_index
            ):
                break
            message = session.messages[message_index]
            if message_index > start_index and message.get("role") == "user":
                last_user_turn_boundary = (
                    message_index,
                    selected_message_tokens,
                )
                if selected_message_tokens >= prompt_tokens_to_remove:
                    return last_user_turn_boundary
            selected_message_tokens += estimate_message_tokens(message)

        return last_user_turn_boundary

    @staticmethod
    def snap_end_index_to_journal_records(
        journal: TurnJournal,
        end_index: int,
    ) -> int:
        """Pull a compaction boundary back onto a whole-record boundary.

        `select_compaction_end_index` cuts at any `role: user` message, and a
        journal record can own more than one (an FE exchange pairs a delivery or
        a PA line with the user line about it), so the cut can land *inside* one
        record's span. Compaction consumes the journal prefix whose turns end at
        or before the boundary, so a straddling record is still owed: its head
        must stay in the prompt rather than fall inside a "consolidated" prefix
        the files do not hold yet. Snapping back, not forward, is the rule,
        and it can never evict past `safe_eviction_end_index`.

        Returns 0 when no whole record ends at or before `end_index`; the
        caller's empty-chunk guard then leaves the prompt untouched.
        """
        records = journal.records_through_message_index(end_index)
        if not records:
            return 0
        return max(record.span[1] for record in records)

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_pa_prompt_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.pa_llm_provider,
            self.pa_model,
            probe_messages,
            self._get_pa_tool_definitions(),
        )

    def safe_pa_prompt_limit_tokens(self) -> int:
        return (
            self.pa_context_window_tokens
            - self.pa_max_completion_tokens
            - self._PROMPT_TOKEN_SAFETY_BUFFER
        )

    def prompt_compaction_trigger_tokens(self) -> int:
        # Native NanoBot behavior: return self.safe_pa_prompt_limit_tokens()
        return min(
            self._PROMPT_COMPACTION_TRIGGER_TOKENS,
            self.safe_pa_prompt_limit_tokens(),
        )

    def preflight_archive_chunk(
        self,
        records: list[dict[str, object]],
    ) -> None:
        """Refuse a chunk that cannot fit one consolidation prompt.

        `/new` archives everything the journal still owes in a single call,
        which can be a whole session -- and the pool carries the turn's
        reasoning and tool traffic, several times what the old projection did.
        A chunk over the budget is a contract failure the LM would discover
        halfway through, so it is named here instead.
        """
        budget = self.safe_pa_prompt_limit_tokens()
        if budget <= 0:
            return
        estimated = sum(estimate_message_tokens(record) for record in records)
        if estimated > budget:
            raise MemoryConsolidationError(
                "INVALID_CONSOLIDATION_CHUNK_SIZE: "
                f"chunk_tokens={estimated} budget={budget}"
            )

    async def archive_messages_to_files(
        self,
        messages: list[dict[str, object]],
    ) -> bool:
        """Archive messages with bounded contract retries and no degraded fallback."""
        if not messages:
            return True
        await self.require_file_consolidation(
            messages,
            session_key="explicit_archive",
        )
        return True

    def _completed_adapter_turn_end(self, session: Session) -> int | None:
        """Prove a whole persisted user turn is safe for next-turn eviction.

        A cursor alone cannot prove completion. Require the append-only journal
        and committed receipts to cover the entire remaining span, with no gap,
        pending record, or straddling boundary. This never writes memory/history.
        """
        start, end = session.prompt_history_start_index, len(session.messages)
        if (self.prompt_history_eviction_policy != PromptHistoryEvictionPolicy.after_adapter_commit
                or not self.require_committed_adapter_writes
                or session.adapter_committed_end_index != end or start >= end
                or session.messages[start].get("role") != "user"):
            return None
        records = turn_journal_for(self.session_manager, session.key).records()
        if not records or any(
            not (record.state == "written" and record.receipt
                 and record.receipt.get("status") in {"committed", "accepted_no_change"})
            for record in records
        ):
            return None
        cursor = start
        for record in records:
            first, last = record.span
            if last <= start:
                continue
            if first != cursor or last < first or last > end:
                return None
            cursor = last
        return end if cursor == end else None

    async def maybe_compact_prompt_history(
        self, session: Session, *, stage: str = "legacy",
    ) -> None:
        """Compact persisted prompt history at user-turn lifecycle boundaries.

        The configured history budget reserves completion tokens and a safety
        buffer. It governs these boundary decisions; this routine does not
        enforce a token cap or truncate traffic inside an in-progress PA turn.
        """
        if stage not in ("legacy", "incoming_user_turn", "completed_user_turn"):
            raise ValueError("Unknown prompt compaction lifecycle stage")
        if not session.messages or self.pa_context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            # trigger = context_window - max_completion - safety_buffer
            prompt_compaction_trigger_tokens = (
                self.prompt_compaction_trigger_tokens()
            )
            post_compaction_target_tokens = prompt_compaction_trigger_tokens // 2
            estimated_prompt_tokens, token_estimation_source = (
                self.estimate_session_prompt_tokens(session)
            )
            if estimated_prompt_tokens <= 0:
                return
            if estimated_prompt_tokens < prompt_compaction_trigger_tokens:
                logger.debug(
                    "Prompt-history compaction idle {}: {}/{} trigger via {} "
                    "(context={}, target={})",
                    session.key,
                    estimated_prompt_tokens,
                    prompt_compaction_trigger_tokens,
                    token_estimation_source,
                    self.pa_context_window_tokens,
                    post_compaction_target_tokens,
                )
                return

            write_native_file_memory = (
                self.prompt_history_eviction_policy
                == PromptHistoryEvictionPolicy.after_file_consolidation
            )
            safe_eviction_end_index = (
                session.adapter_committed_end_index
                if self.prompt_history_eviction_policy
                == PromptHistoryEvictionPolicy.after_adapter_commit
                else None
            )
            raise_on_uncommitted_eviction = (
                self.require_committed_adapter_writes
                and self.prompt_history_eviction_policy
                == PromptHistoryEvictionPolicy.after_adapter_commit
            )

            for compaction_round in range(self._MAX_COMPACTION_ROUNDS):
                if estimated_prompt_tokens <= post_compaction_target_tokens:
                    return

                compaction_boundary = self.select_compaction_end_index(
                    session,
                    max(
                        1,
                        estimated_prompt_tokens - post_compaction_target_tokens,
                    ),
                    safe_eviction_end_index=safe_eviction_end_index,
                )
                if (compaction_boundary is None
                        and estimated_prompt_tokens >= self.safe_pa_prompt_limit_tokens()
                        and stage != "legacy"):
                    completed_end = self._completed_adapter_turn_end(session)
                    if completed_end is not None:
                        if stage == "completed_user_turn":
                            # No new model request follows this lifecycle point.
                            # Defer, do not claim it fits or evict the delivered turn.
                            return
                        # A real next user is waiting outside persisted history.
                        # Its boundary is the end of the fully committed old turn.
                        compaction_boundary = (completed_end, estimated_prompt_tokens)
                if compaction_boundary is None:
                    if (
                        raise_on_uncommitted_eviction
                        and estimated_prompt_tokens
                        >= prompt_compaction_trigger_tokens
                    ):
                        if safe_eviction_end_index == len(session.messages):
                            # Only the current user turn remains. All its messages
                            # are committed, but removing it would split the turn.
                            # The experimental trigger is not the model's hard cap.
                            if estimated_prompt_tokens >= self.safe_pa_prompt_limit_tokens():
                                raise PromptHistoryEvictionBlockedError(
                                    "INVALID_PROMPT_HISTORY_CAPACITY: "
                                    f"session={session.key} "
                                    f"prompt_tokens={estimated_prompt_tokens} "
                                    f"safe_limit={self.safe_pa_prompt_limit_tokens()}"
                                )
                            return
                        raise PromptHistoryEvictionBlockedError(
                            "INVALID_PROMPT_HISTORY_EVICTION_UNCOMMITTED: "
                            f"session={session.key} "
                            f"prompt_start={session.prompt_history_start_index} "
                            "adapter_committed_end_index="
                            f"{safe_eviction_end_index}"
                        )
                    logger.debug(
                        "Prompt-history compaction: no safe boundary for {} "
                        "(round {})",
                        session.key,
                        compaction_round,
                    )
                    return

                compaction_end_index = compaction_boundary[0]
                journal = (
                    turn_journal_for(self.session_manager, session.key)
                    if write_native_file_memory
                    else None
                )
                if journal is not None:
                    # The invariant the cursors below stand on: the consolidated
                    # prefix is a union of whole journal records. The boundary
                    # above is a bare user-turn cut and can be interior to one
                    # record's span, so snap it back before the file write, the
                    # chunk identity and the two cursors all read it.
                    compaction_end_index = self.snap_end_index_to_journal_records(
                        journal, compaction_end_index
                    )
                chunk_start_index = (
                    session.file_consolidated_end_index
                    if write_native_file_memory
                    else session.prompt_history_start_index
                )
                history_chunk = session.messages[
                    chunk_start_index:compaction_end_index
                ]
                if not history_chunk:
                    return

                logger.info(
                    "Prompt-history compaction round {} for {}: {}/{} "
                    "trigger via {} "
                    "(policy={}, context={}, target={}), chunk={} msgs",
                    compaction_round,
                    session.key,
                    estimated_prompt_tokens,
                    prompt_compaction_trigger_tokens,
                    token_estimation_source,
                    self.prompt_history_eviction_policy.value,
                    self.pa_context_window_tokens,
                    post_compaction_target_tokens,
                    len(history_chunk),
                )
                if journal is not None:
                    # Same source as the pinned flush in
                    # `harness/orchestrator.py`: the turn journal, not the
                    # message list. `session.messages` is projection 2 -- the
                    # PA's own next prompt -- and cannot re-derive the turn's
                    # prose, its held declarations or the final iteration's
                    # reasoning. Compaction consumes exactly the
                    # journal prefix whose turns end at or before the chosen
                    # boundary, zero-span records included; the
                    # cursor below keeps advancing on raw message indices.
                    consumed = records_owed_to_files(
                        journal.records_through_message_index(
                            compaction_end_index
                        )
                    )
                    records = consolidation_records(consumed)
                    if records:
                        await self.require_file_consolidation(
                            records,
                            session_key=session.key,
                            chunk_start_index=chunk_start_index,
                            chunk_end_index=compaction_end_index,
                        )
                    journal.mark_file_consolidated(
                        [record.turn_id for record in consumed]
                    )
                    session.file_consolidated_end_index = compaction_end_index
                session.prompt_history_start_index = compaction_end_index
                self.session_manager.save(session)

                estimated_prompt_tokens, token_estimation_source = (
                    self.estimate_session_prompt_tokens(session)
                )
                if estimated_prompt_tokens <= 0:
                    return
