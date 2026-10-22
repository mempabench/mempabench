"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import os
import time
from contextlib import AsyncExitStack, nullcontext
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from memory.adapter import MemoryAdapter, PromptHistoryEvictionPolicy
from memory import event_writer
from memory.event_writer import (
    TOOL_RESULT_MAX_CHARS,
    AdapterWriteCommitError,
    TurnPool,
    build_turn_pool,
    record_adapter_write_commit,
    turn_journal_for,
    write_visible_memory_event,
)
from memory.results import WriteReceipt
from pa.agent.context import ContextBuilder
from pa.agent.hook import AgentHook, AgentHookContext
from pa.agent.ix.collector import (
    WORLD_CHANGING_ACTION_TYPES,
    build_runtime_history,
    consent_decisions_from_metadata,
    hold_prompt,
    ix_payload_status,
    open_episode_ids,
    render_turn,
    turn_reminder_message,
)
from pa.agent.ix.policy import selections_from_tool_events
from pa.agent.ix.rationale import (
    certainty_block_text,
    render_certainties,
    render_rationale,
)
from pa.agent.ix.render import clean_delivered_text, released_ids, unmarked_questions
from pa.agent.ix.render_types import (
    CAPABILITY_BOUNDARY_ATTRIBUTE,
    CONSENT_DECISIONS_BLOCK,
    INTERACTION_ITEM_ID_FIELD,
    IX_SELECTION_TOOL_NAMES,
    STOP_REASON_IX_CONTRACT_FAILURE,
    HeldItem,
    IXHold,
    ProactiveDecision,
    RenderedTurn,
    held_status,
    to_ix_render_metadata,
)
from pa.agent.ix.verification import (
    ENV_VAR as VERIFY_OUTREACH_ENV_VAR,
    OVERRIDE_SETTINGS,
    VERIFICATION_OVERRIDE,
    verify_outreach_action,
)
from pa.agent.memory import MemoryConsolidator
from pa.agent.runner import (
    IX_CONTRACT_CAPABILITY,
    IX_CONTRACT_GATE_MESSAGE,
    IX_CONTRACT_NOTICE,
    IX_CONTRACT_RENDER,
    IX_CONTRACT_RENDER_ARGS,
    IX_CONTRACT_SELECTION,
    IX_CONTRACT_SELECTION_ARGS,
    AgentRunResult,
    AgentRunSpec,
    AgentRunner,
    internal_user_message,
)
from pa.agent.ix import TurnToolRegistry
from pa.agent.tools.registry import ToolRegistry
from pa.bus.events import InboundMessage, OutboundMessage
from pa.command import CommandContext, CommandRouter, register_builtin_commands
from pa.bus.queue import MessageBus
from pa.providers.base import LLMProvider, LLMTaskIdentity
from pa.session.manager import Session, SessionManager, SessionNotFoundError
from pa.utils.helpers import estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from memory.audit import MemoryIOAuditSink


class AdapterRetrievalError(RuntimeError):
    """Raised when configured adapter memory cannot be retrieved."""


class AgentProviderError(RuntimeError):
    """Raised when the PA provider cannot produce a valid turn response."""


class ProactiveDecisionMissingError(RuntimeError):
    """Raised when the PA's turn on a proactive environment event ends without
    its IX `proactive_outreach` send/suppress decision."""


#: §3.5 (part 6): `IX_capability_resolve` still missing or malformed on the third
#: response charged to the turn's shared IX repair budget ends the turn
#: with no scorable answer at all. The step fails and can be retried from a
#: clean state -- it must not degrade into an ordinary reply that a judge would
#: then score as if the boundary had been handled.
IX_CAPABILITY_CONTRACT_FAILURE_DETAIL = (
    "IX_capability_resolve was still missing or invalid after three responses "
    "once a tool reported a capability limit"
)

#: §3.0: the same three-invalid-response budget for the per-turn render
#: payload. Without it the turn is shipped missing the one field the setting
#: shapes, which reads to a judge as the PA choosing not to say it.
IX_RENDER_CONTRACT_FAILURE_DETAIL = (
    "a required IX_render_<attribute> payload was still missing after three "
    "invalid responses, so this turn had no per-turn render data"
)

#: Owner decision 2026-09-07: the same rule for the opening line. A Bookend /
#: Full narration turn that executed task calls with no `_ix_notice` on any of
#: them, and wrote none when asked twice, reaches the user as Silent -- so it
#: does not reach the user.
IX_NOTICE_CONTRACT_FAILURE_DETAIL = (
    "the process_visibility setting shows an opening line and no task call "
    "carried _ix_notice, twice after being asked"
)

#: regress20 `acc_027`: the reaction beat offers no IX tools at all, the PA
#: called `IX_emotional_engagement` anyway, the registry refused it three times
#: and the turn failed closed at `runner.py`'s third-refusal break. It was
#: reported as the capability failure below -- the mapping had a default, and
#: this path fell into it -- although no tool on that turn reported a
#: capability limit.
IX_SELECTION_CONTRACT_FAILURE_DETAIL = (
    "an IX selection call the registry refused was re-sent three times (the "
    "attribute's tool is not offered on this beat), so this turn produced no "
    "reply"
)

#: regress23m `acc_001`: the other cause of the same break. The beat *did* offer
#: `IX_verbosity`; the PA called it with `{}` and the registry refused it three
#: times for "missing required setting". The sentence above sent the reader to
#: the beat's tool list, where there was nothing wrong.
IX_SELECTION_ARGS_CONTRACT_FAILURE_DETAIL = (
    "an IX selection call the registry refused was re-sent three times (the "
    "attribute's tool is offered on this beat, but the call's arguments were "
    "invalid -- a missing or unrecognised setting), so this turn produced no "
    "reply"
)

#: hy4 `acc_031`, 2026-09-18: the third cause of the same break, one tier down.
#: `IX_render_process_visibility` was called with the *selection* tool's
#: `setting`/`evidence`/`application` and without its own required
#: `completion_message`, so the turn wore the selection-arguments sentence and
#: pointed the reader at a setting the model had already chosen correctly. Kept
#: under the "render contract" prefix `harness/failure_kind.py` matches, because
#: the tier that broke is the render payload's.
IX_RENDER_ARGS_CONTRACT_FAILURE_DETAIL = (
    "an IX_render_<attribute> call the registry refused was re-sent three "
    "times (the payload tool is offered on this beat, but the call's arguments "
    "were invalid -- a required payload field was missing), so this turn "
    "produced no reply"
)

#: The same default swallowed the gate turn that carried no confirmation
#: message: one repair, then fail closed, and the turn wore the capability
#: label too.
IX_GATE_CONTRACT_FAILURE_DETAIL = (
    "the gate turn carried no confirmation message, once after being asked, "
    "so the ask the hold exists for was never put to the user"
)

#: Every kind the runner can name, and nothing else: an unmapped kind is a new
#: fatal site that has not said what it broke, and guessing is what produced
#: the regress20 mislabel.
_IX_CONTRACT_FAILURE_DETAILS: dict[str, str] = {
    IX_CONTRACT_RENDER: f"render contract: {IX_RENDER_CONTRACT_FAILURE_DETAIL}",
    IX_CONTRACT_NOTICE: f"notice contract: {IX_NOTICE_CONTRACT_FAILURE_DETAIL}",
    IX_CONTRACT_CAPABILITY: (
        f"capability contract: {IX_CAPABILITY_CONTRACT_FAILURE_DETAIL}"
    ),
    IX_CONTRACT_SELECTION: (
        f"selection contract: {IX_SELECTION_CONTRACT_FAILURE_DETAIL}"
    ),
    IX_CONTRACT_SELECTION_ARGS: (
        f"selection contract: {IX_SELECTION_ARGS_CONTRACT_FAILURE_DETAIL}"
    ),
    IX_CONTRACT_RENDER_ARGS: (
        f"render contract: {IX_RENDER_ARGS_CONTRACT_FAILURE_DETAIL}"
    ),
    IX_CONTRACT_GATE_MESSAGE: f"gate contract: {IX_GATE_CONTRACT_FAILURE_DETAIL}",
}

#: §3.5: the attribute whose *detection* is already session-scoped. The runner
#: turns `capability_limit_seen` on from the tool result alone, whatever the
#: beat lists -- so a `capability_limit` on a beat that does not list the
#: attribute still forces `IX_capability_resolve` (stage 2). Stage 3 applies the
#: setting, and it reads the turn directive, which is built from
#: `inherited_ix_selections` alone. With
#: the two scopes mismatched, stage 3 could never run for the events stage 2
#: fires on: live, `v3g07` and `v3g08` (forced `Suggest alternatives` /
#: `Find and hand off`) both declared the boundary on beat 1 and then rendered
#: byte-identically -- `gate: null`, zero holds -- because
#: `directive.capability_boundary_policy` was empty there. The renderer already
#: reads the whole cache (`_render_turn`); this makes the directive agree.
_SESSION_SCOPED_IX = ("capability_boundary",)

#: §3.5 / pre-run-test audit §2 (acc_026): the attribute whose *activation*
#: follows the trigger rather than the beat. A capability limit can land on any
#: turn -- live, `acc_026` hit it on `b1` while the beat that listed
#: `capability_boundary` in `active_skills` was `b3`, so the PA substituted two
#: other papers with no setting chosen and by `b3` there was nothing left to
#: decide. So the first `capability_limit` result in a session activates the
#: attribute itself: `IX_capability_boundary` is offered and required from that
#: turn, and the attribute stays in the session's active set afterwards, exactly
#: as if the beat had listed it.
_TRIGGERED_IX = CAPABILITY_BOUNDARY_ATTRIBUTE

#: The metadata key carrying the environment trigger as `process_environment_event`
#: built it — `{source_id, action_id, body}` — before the fixed
#: `"Use IX_proactive_outreach …"` instruction was appended to the text the PA
#: is shown. The instruction is IX contract enforcement and stays out of the
#: pool (owner ruling 2026-09-13, Q4); the split stays exact because the clean
#: body is *constructed* here and never recovered by string-stripping the
#: `InboundMessage.content`, whose suffix is an inline f-string literal. Popped
#: off the outbound metadata like the consent sideband.
ENVIRONMENT_EVENT_BLOCK = "_mempa_environment_event"

#: The role the PA sees an environment event in, and the line every such event
#: opens with. `process_environment_event` builds the ACC body from these, and
#: `_process_message` reads the role off the same constant, so the two halves of
#: the ACC contract cannot drift apart.
#:
#: The forced-event executor imports both rather than carrying its own copy: a
#: forced environment delivery is the same kind of thing as an ACC environment
#: event and has to reach the PA the same way. It used to append the delivery as
#: `role: "user"` with the bare text, which tells the PA the *person* said it --
#: the same misattribution the memory side already fixed by routing the line on
#: `source.kind` instead of on its role.
ENVIRONMENT_EVENT_ROLE = "system"
ENVIRONMENT_EVENT_HEADER = "[Environment event; no user request is currently active]"


def render_environment_event(body: str) -> str:
    """Wrap an environment delivery in the header the PA reads it by.

    One definition for both callers. ACC passes its five trigger fields; the
    forced-event executor passes the delivered text. What they share -- the
    header line, and therefore what the PA recognises the message as -- lives
    here and not in two places.
    """
    return f"{ENVIRONMENT_EVENT_HEADER}\n{body}"

#: regress13 acc_035 — how much of a dropped claim goes on the event. Enough to
#: recognise the sentence in the reply (or to see that it is not in *this*
#: reply); the whole claim is already on the audit plane in
#: `ix_render.certainties`, so the event carries a head and not a second copy.
_CLAIM_HEAD_CHARS = 120


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = TOOL_RESULT_MAX_CHARS
    _AUDIT_ONLY_METADATA_KEYS = (
        "script_session_id",
        "scenario_description",
        "interaction_context",
        "active_preference_attributes",
    )

    def __init__(
        self,
        bus: MessageBus,
        pa_llm_provider: LLMProvider,
        workspace: Path,
        pa_model: str | None = None,
        max_iterations: int = 40,
        pa_context_window_tokens: int = 65_536,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        timezone: str | None = None,
        memory_adapter: MemoryAdapter | None = None,
        consolidation_llm_provider: LLMProvider | None = None,
        consolidation_model: str | None = None,
        require_committed_adapter_writes: bool = False,
        allow_session_creation: bool = False,
        memory_io_recorder: MemoryIOAuditSink | None = None,
        skip_consolidation: bool = False,
    ):
        self.bus = bus
        self.pa_llm_provider = pa_llm_provider
        self.workspace = workspace
        self.pa_model = pa_model or pa_llm_provider.get_default_model()
        self.max_iterations = max_iterations
        self.pa_context_window_tokens = pa_context_window_tokens
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}

        self.memory_adapter = memory_adapter
        self.prompt_history_eviction_policy = (
            PromptHistoryEvictionPolicy.after_file_consolidation
            if memory_adapter is None
            else memory_adapter.prompt_history_eviction_policy
        )
        self.native_file_memory_enabled = (
            self.prompt_history_eviction_policy
            == PromptHistoryEvictionPolicy.after_file_consolidation
        )
        self.require_committed_adapter_writes = require_committed_adapter_writes
        self.allow_session_creation = allow_session_creation
        self.memory_io_recorder = memory_io_recorder
        # A probe runs on a throwaway branch and must leave the cut point's
        # memory exactly as it found it, so the whole compaction step is
        # skipped -- not just the LM write inside it.
        self.skip_consolidation = skip_consolidation
        self.context = ContextBuilder(
            workspace,
            timezone=timezone,
            native_file_memory_enabled=self.native_file_memory_enabled,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(pa_llm_provider)

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # IX selections are inherited only inside one benchmark session:
        # keyed by (session_key, script_session_id).
        self._ix_selection_cache: dict[tuple[str, str], dict[str, Any]] = {}
        # Cross-turn IX holds, same benchmark-session scope. A held item is
        # released only by an explicit per-id decision plus, for outward-facing
        # actions, the user's own consent -- never by action-type lookalike.
        self._ix_hold_cache: dict[tuple[str, str], IXHold] = {}
        # Attributes a runtime trigger activated rather than a beat, same
        # benchmark-session scope. Sticky: once a `capability_limit` result has
        # been seen, `capability_boundary` governs the rest of the session.
        self._ix_triggered_cache: dict[tuple[str, str], set[str]] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            pa_llm_provider=pa_llm_provider,
            pa_model=self.pa_model,
            consolidation_llm_provider=(
                consolidation_llm_provider or pa_llm_provider
            ),
            consolidation_model=consolidation_model or self.pa_model,
            session_manager=self.sessions,
            pa_context_window_tokens=pa_context_window_tokens,
            build_pa_prompt_messages=self.context.build_messages,
            get_pa_tool_definitions=self.tools.get_definitions,
            prompt_history_eviction_policy=self.prompt_history_eviction_policy,
            pa_max_completion_tokens=pa_llm_provider.generation.max_tokens,
            require_committed_adapter_writes=(
                require_committed_adapter_writes
            ),
        )
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _get_message_session(self, key: str) -> Session:
        try:
            return self.sessions.get(key)
        except SessionNotFoundError:
            if not self.allow_session_creation:
                raise
            return self.sessions.create(key)

    def _record_memory_io(
        self,
        metadata: dict[str, Any],
        *,
        session_id: str,
        turn_id: str,
        operation: str,
        adapter_request: dict[str, Any],
        backend_request: dict[str, Any] | None = None,
        backend_response: Any | None = None,
        status: str,
        trace: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        if self.memory_io_recorder is None or self.memory_adapter is None:
            return
        audit_only_labels = {
            key: metadata[key]
            for key in self._AUDIT_ONLY_METADATA_KEYS
            if key in metadata
        }
        self.memory_io_recorder.record(
            {
                "run_id": metadata.get("run_id"),
                "step_id": metadata.get("script_session_id") or "",
                "session_id": session_id,
                "turn_id": turn_id,
                "backend": self.memory_adapter.backend_name,
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
                "audit_only_labels": audit_only_labels,
                "error": (
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    if error is not None
                    else None
                ),
            }
        )

    async def _retrieve_adapter_memory(
        self,
        metadata: dict[str, Any],
        current_message: str = "",
        *,
        session_key: str,
    ) -> str | None:
        """Retrieve memory from the adapter if configured and metadata has required scope."""
        if self.memory_adapter is None:
            return None
        unsupported_filters = metadata.get("memory_retrieval_filters")
        if unsupported_filters:
            raise AdapterRetrievalError(
                "INVALID_ADAPTER_RETRIEVAL: formal benchmark retrieval does not "
                "accept caller-provided filters: "
                f"value={unsupported_filters!r} session={session_key!r}"
            )
        run_id = metadata.get("run_id")
        persona_id = metadata.get("persona_id")
        if not run_id or not persona_id:
            missing = [
                name
                for name, value in (("run_id", run_id), ("persona_id", persona_id))
                if not value
            ]
            raise AdapterRetrievalError(
                "INVALID_ADAPTER_RETRIEVAL: "
                f"backend={self.memory_adapter.backend_name!r} "
                f"missing_fields={missing!r} run_id={run_id!r} "
                f"persona_id={persona_id!r} "
                f"session_id={session_key!r}"
            )
        from memory.scope import MemoryRetrievalBudget, MemoryScope, RetrieveRequest

        scope = MemoryScope(
            run_id=run_id,
            persona_id=persona_id,
            agent_id=metadata.get("agent_id"),
        )
        retrieval_budget = MemoryRetrievalBudget(
            max_tokens=metadata.get("memory_retrieval_budget_tokens"),
            max_items=metadata.get("memory_retrieval_budget_items"),
        )
        if (
            isinstance(retrieval_budget.max_tokens, bool)
            or not isinstance(retrieval_budget.max_tokens, (int, type(None)))
            or (
                retrieval_budget.max_tokens is not None
                and retrieval_budget.max_tokens <= 0
            )
        ):
            raise AdapterRetrievalError(
                "INVALID_ADAPTER_RETRIEVAL: memory retrieval token budget "
                "must be a positive integer when set: "
                f"value={retrieval_budget.max_tokens!r} session={session_key!r}"
            )
        if (
            isinstance(retrieval_budget.max_items, bool)
            or not isinstance(retrieval_budget.max_items, (int, type(None)))
            or (
                retrieval_budget.max_items is not None
                and retrieval_budget.max_items <= 0
            )
        ):
            raise AdapterRetrievalError(
                "INVALID_ADAPTER_RETRIEVAL: memory retrieval item budget "
                "must be a positive integer when set: "
                f"value={retrieval_budget.max_items!r} session={session_key!r}"
            )
        request = RetrieveRequest(
            query=metadata.get("memory_retrieval_query") or current_message,
            scope=scope,
            session_id=session_key,
            budget=retrieval_budget,
        )
        adapter_request = {
            "query": request.query,
            "scope": {
                "run_id": scope.run_id,
                "persona_id": scope.persona_id,
                "agent_id": scope.agent_id,
            },
            "session_id": request.session_id,
            "budget": {
                "max_tokens": retrieval_budget.max_tokens,
                "max_items": retrieval_budget.max_items,
            },
        }
        try:
            await self.memory_adapter.setup_scope(scope)
            result = await self.memory_adapter.retrieve(request)
            context = self.memory_adapter.format_context(result)
            context, context_tokens, token_source, truncated = (
                self._limit_adapter_memory_context(
                    context,
                    max_tokens=retrieval_budget.max_tokens,
                )
            )
        except Exception as exc:
            self._record_memory_io(
                metadata,
                session_id=session_key,
                turn_id=str(metadata.get("turn_id") or ""),
                operation="retrieve",
                adapter_request=adapter_request,
                status="failed",
                error=exc,
            )
            raise AdapterRetrievalError(
                "INVALID_ADAPTER_RETRIEVAL: "
                f"backend={self.memory_adapter.backend_name!r} "
                f"run_id={run_id!r} persona_id={persona_id!r} "
                f"session_id={session_key!r} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        trace = result.trace
        if trace is not None and context_tokens is not None:
            trace.token_count = context_tokens
            if truncated:
                trace.warnings.append(
                    "PA prompt memory context was truncated to the shared "
                    "retrieval token budget: "
                    f"max_tokens={retrieval_budget.max_tokens!r} "
                    f"token_count={context_tokens!r} source={token_source!r}"
                )
        self._record_memory_io(
            metadata,
            session_id=session_key,
            turn_id=str(metadata.get("turn_id") or ""),
            operation=(trace.native_operation if trace else "retrieve"),
            adapter_request=adapter_request,
            backend_request=(trace.native_request if trace else None),
            backend_response=(
                trace.native_response
                if trace and trace.native_response is not None
                else result.raw
            ),
            status="succeeded",
            trace=trace,
        )
        return context or None

    def _limit_adapter_memory_context(
        self,
        context: str,
        *,
        max_tokens: int | None,
    ) -> tuple[str, int | None, str | None, bool]:
        """Apply the final PA-model token cap to adapter-formatted memory."""
        if not context or max_tokens is None:
            return context, None, None, False

        def estimate(candidate: str) -> tuple[int, str]:
            return estimate_prompt_tokens_chain(
                self.pa_llm_provider,
                self.pa_model,
                [{"role": "system", "content": candidate}],
            )

        full_tokens, source = estimate(context)
        if full_tokens <= 0:
            raise AdapterRetrievalError(
                "INVALID_ADAPTER_RETRIEVAL: unable to estimate memory context "
                f"tokens for pa_model={self.pa_model!r}"
            )
        if full_tokens <= max_tokens:
            return context, full_tokens, source, False

        wrapper = re.fullmatch(
            r"(?P<open><memory-context\b[^>]*>)\n?(?P<body>.*)\n"
            r"(?P<close></memory-context>)",
            context,
            flags=re.DOTALL,
        )
        if wrapper:
            body = wrapper.group("body")

            def candidate(length: int) -> str:
                return (
                    f"{wrapper.group('open')}\n{body[:length]}\n"
                    f"{wrapper.group('close')}"
                )
        else:
            body = context

            def candidate(length: int) -> str:
                return body[:length]

        empty_candidate = candidate(0)
        empty_tokens, empty_source = estimate(empty_candidate)
        if empty_tokens <= 0 or empty_tokens > max_tokens:
            return "", 0, empty_source, True

        low, high = 0, len(body)
        best_context = empty_candidate
        best_tokens = empty_tokens
        best_source = empty_source
        while low <= high:
            middle = (low + high) // 2
            trial = candidate(middle)
            trial_tokens, trial_source = estimate(trial)
            if 0 < trial_tokens <= max_tokens:
                best_context = trial
                best_tokens = trial_tokens
                best_source = trial_source
                low = middle + 1
            else:
                high = middle - 1

        return best_context, best_tokens, best_source, True

    async def _write_turn_to_memory_adapter(
        self,
        metadata: dict[str, Any],
        pool: TurnPool,
        *,
        timestamp: Any,
        session_key: str,
        turn_id: str,
    ) -> WriteReceipt | None:
        """Write the turn's pool to the configured memory adapter.

        The pool is built by the caller, not here: only the caller holds the
        raw provider slice (`result.messages`), the turn's IX call records and
        its `ix_payload_status`, and the persisted `session.messages` slice
        carries none of the three (§6.1 (b)).
        """
        if self.memory_adapter is None:
            return None
        run_id = metadata.get("run_id")
        persona_id = metadata.get("persona_id")
        if not run_id or not persona_id:
            if (
                self.require_committed_adapter_writes
                and self.prompt_history_eviction_policy
                == PromptHistoryEvictionPolicy.after_adapter_commit
            ):
                raise AdapterWriteCommitError(
                    "INVALID_ADAPTER_WRITE: missing run_id or persona_id"
                )
            return None

        session_id = session_key
        return await write_visible_memory_event(
            adapter=self.memory_adapter,
            pool=pool,
            run_id=run_id,
            persona_id=persona_id,
            agent_id=metadata.get("agent_id") or "nanobot",
            session_id=session_id,
            turn_id=turn_id,
            event_id=(
                metadata.get("event_id")
                or f"{run_id}:{session_id}:{turn_id}"
            ),
            timestamp=timestamp,
            step_id=metadata.get("script_session_id") or "",
            audit_sink=self.memory_io_recorder,
            audit_only_labels={
                key: metadata[key]
                for key in self._AUDIT_ONLY_METADATA_KEYS
                if key in metadata
            },
            require_committed=self.require_committed_adapter_writes,
            eviction_policy=self.prompt_history_eviction_policy,
        )

    def _pool_input(self, raw_turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The turn's provider slice in the form the persisted copy keeps.

        `ContextBuilder` merges the scripted clock and channel framing into the
        user message so a provider never sees two user turns in a row. That
        framing is item #2 — scaffolding that exists only because a run is
        scripted — so it comes off here, exactly as `_save_turn` takes it off
        the persisted copy.

        A tool result goes through the same `_cut_tool_result` for the same
        reason: the pool carries the persisted form
        (`memory/event_writer.py:20-34`), so the raw slice cannot reach
        `build_turn_pool` uncut.
        """
        stripped: list[dict[str, Any]] = []
        for message in raw_turn:
            content = message.get("content")
            if message.get("role") == "tool":
                if isinstance(content, (str, list)):
                    message = {**message, "content": self._cut_tool_result(content)}
                stripped.append(message)
                continue
            if message.get("role") != "user":
                stripped.append(message)
                continue
            if isinstance(content, str):
                if content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    parts = content.split("\n\n", 1)
                    if not (len(parts) > 1 and parts[1].strip()):
                        continue
                    message = {**message, "content": parts[1]}
            elif isinstance(content, list):
                blocks = self._sanitize_persisted_blocks(content, drop_runtime=True)
                if not blocks:
                    continue
                message = {**message, "content": blocks}
            stripped.append(message)
        return stripped

    @staticmethod
    def _pool_ix_selections(result: AgentRunResult) -> list[dict[str, Any]]:
        """Every `IX_<attribute>` selection of the turn, with its `outcome=`.

        §3.3: the block is built from the executed request and its tool event,
        never from the assistant message's serialized `tool_calls` — that copy
        is written at `pa/agent/runner.py:1413-1421`, before
        `_dedupe_ix_selections` and `_restore_voided_selection_fields` run at
        `:1458-1462`, so it can carry an ignored duplicate or a blank
        `evidence` the registry never saw. The collector's call records carry
        the arguments as the registry received them, with carrier C already
        stripped (`pa/agent/ix/collector.py:995`).

        `executed` / `rejected` come from the call's `_run_tool` event status
        (`pa/agent/runner.py:3751-3762`), which `record_call` copies;
        `ignored_duplicate` comes from `_reselected_attribute` (`:2840-2860`,
        recorded unexecuted with `status="ignored"`) and from
        `_dedupe_ix_selections` (`:3637-3666`), whose drops never reach the
        collector at all and ride on `AgentRunResult` instead. Each entry
        carries its own call id, so nothing is zipped positionally.
        """
        records: list[dict[str, Any]] = []
        collector = getattr(result, "collector", None)
        for record in getattr(collector, "calls", None) or []:
            if record.tool_name not in IX_SELECTION_TOOL_NAMES:
                continue
            if record.executed:
                outcome = "executed" if record.status == "ok" else "rejected"
            elif record.status == "ignored":
                outcome = "ignored_duplicate"
            else:
                continue
            records.append({
                "id": record.tool_call_id,
                "name": record.tool_name,
                "args": dict(record.args),
                "outcome": outcome,
            })
        records.extend(result.ignored_ix_selections)
        return records

    def _build_turn_pool(
        self,
        *,
        result: AgentRunResult,
        history_length: int,
        delivered: str,
        rendered: RenderedTurn,
        environment_event: dict[str, str] | None = None,
    ) -> TurnPool:
        """The PA's whole first-person experience of one turn (§3.4)."""
        released, held_not_released, cancelled = self._ix_payload_ids(rendered, result)
        return build_turn_pool(
            self._pool_input(result.messages[1 + history_length:]),
            delivered=delivered,
            environment_event=environment_event,
            ix_selections=self._pool_ix_selections(result),
            ix_payload_status=ix_payload_status(
                released=released,
                held_not_released=held_not_released,
                cancelled=cancelled,
            ),
        )

    @staticmethod
    def _turn_id(metadata: dict[str, Any]) -> str:
        """One id per turn, shared by the adapter event and the journal record.

        §6.1 (c) makes the record the resume authority and matches it to the
        receipt by `turn_id`, so the adapter event and the journal record must
        carry the same string — which they cannot if each site derives its own.
        """
        return str(
            metadata.get("turn_id")
            or metadata.get("message_id")
            or time.time_ns()
        )

    def turn_journal(self, session_key: str):
        """This session's append-only pool journal (§6.1 (b)).

        Built fresh each time rather than cached -- `TurnJournal.__init__`
        reads the file -- because the forced-event executor holds its own
        `SessionManager` for the same session key and appends its own records
        between this PA's turns.
        """
        return turn_journal_for(self.sessions, session_key)

    def _record_adapter_write_commit(
        self,
        session: Session,
        receipt: WriteReceipt | None,
        *,
        adapter_committed_end_index: int,
    ) -> None:
        session.adapter_committed_end_index = record_adapter_write_commit(
            receipt=receipt,
            session_id=session.key,
            current_index=adapter_committed_end_index,
            committed_index=session.adapter_committed_end_index,
            eviction_policy=self.prompt_history_eviction_policy,
            require_committed=self.require_committed_adapter_writes,
        )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from pa.agent.tools.mcp import connect_mcp_servers
        try:
            async with AsyncExitStack() as stack:
                await connect_mcp_servers(self._mcp_servers, self.tools, stack)
                self._mcp_stack = stack.pop_all()
                self._mcp_connected = True
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from pa.utils.helpers import strip_think
        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'document_read("path")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        active_ix: list[str] | None = None,
        ix_context: dict[str, Any] | None = None,
        session_key: str | None = None,
        script_session_id: str | None = None,
        llm_task_identity: LLMTaskIdentity | None = None,
        consent_decisions: list[Any] | None = None,
        user_message: str = "",
    ) -> AgentRunResult:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        loop_self = self

        class _LoopHook(AgentHook):
            def __init__(self) -> None:
                self._stream_buf = ""

            def wants_streaming(self) -> bool:
                return on_stream is not None

            async def on_stream(self, context: AgentHookContext, delta: str) -> None:
                from pa.utils.helpers import strip_think

                prev_clean = strip_think(self._stream_buf)
                self._stream_buf += delta
                new_clean = strip_think(self._stream_buf)
                incremental = new_clean[len(prev_clean):]
                if incremental and on_stream:
                    await on_stream(incremental)

            async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
                if on_stream_end:
                    await on_stream_end(resuming=resuming)
                self._stream_buf = ""

            async def before_execute_tools(self, context: AgentHookContext) -> None:
                if on_progress:
                    if not on_stream:
                        thought = loop_self._strip_think(context.response.content if context.response else None)
                        if thought:
                            await on_progress(thought)
                    tool_hint = loop_self._strip_think(loop_self._tool_hint(context.tool_calls))
                    await on_progress(tool_hint, tool_hint=True)
                for tc in context.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tc.name, args_str[:200])
                loop_self._set_tool_context(channel, chat_id, message_id)

            def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
                return loop_self._strip_think(content)
        # Loop owns the product logic (session, streaming pushes, progress notifications), Runner owns the pure LLM loop logic; Hook decouples the two.
        ix_context = dict(ix_context or {})
        ix_cache_key = (session_key or "", str(script_session_id or ""))
        # An attribute a trigger activated on an earlier turn of this session
        # joins the beat's own list, so from the trigger onward it is offered,
        # rendered and gated exactly like a beat-provided entry. A proactive
        # round inherits nothing, so it is left out of that one.
        triggered_ix = (
            ()
            if ix_context.get("proactive_event")
            else tuple(sorted(self._ix_triggered_cache.get(ix_cache_key) or ()))
        )
        beat_active_ix = list(ix_context.get("session_active_ix") or active_ix or [])
        active_ix_all = [
            *beat_active_ix,
            *(attr for attr in triggered_ix if attr not in beat_active_ix),
        ]
        if triggered_ix:
            ix_context["session_active_ix"] = list(active_ix_all)
        # A proactive round inherits no selection, so `IX_proactive_outreach`
        # stays on the turn and the PA still writes the candidate contact. A
        # `--force-ix` seed survives it anyway: `_remember_ix_selections` will
        # not let the model's answer replace a forced entry, and `_render_turn`
        # reads the cache, not this list.
        inherited_ix_selections = (
            []
            if ix_context.get("proactive_event")
            else self._cached_ix_selections(ix_cache_key, active_ix_all)
        )
        selected_attrs = {selection.attribute for selection in inherited_ix_selections}
        beat_turn_ix = [*(active_ix or []), *(
            attr for attr in triggered_ix if attr not in (active_ix or [])
        )]
        turn_active_ix = [attr for attr in beat_turn_ix if attr not in selected_attrs]
        # A proactive event round inherits neither settings nor holds.
        hold = (
            IXHold()
            if ix_context.get("proactive_event")
            else self._ix_hold_cache.get(ix_cache_key) or IXHold()
        )
        ix_context.setdefault("user_message", user_message)
        ix_context["hold"] = hold
        # Which attributes this session has already settled, forced or natural.
        # An attribute can be governing the session before the current beat
        # lists it in `session_active_ix` -- a `--force-ix` seed does exactly
        # that -- and the wrapper path must not switch itself off in between.
        ix_context["session_ix_selected"] = (
            ()
            if ix_context.get("proactive_event")
            else tuple(self._ix_selection_cache.get(ix_cache_key) or ())
        )
        # §3.7 (part 3): a verdict is only consent when it answers an ask the user was
        # actually shown. Ids that no open episode holds are dropped here, so a
        # stale `allow` cannot release anything that later reuses that id.
        held_now = {
            str(call.get(INTERACTION_ITEM_ID_FIELD) or "")
            for episode in hold.open_episodes.values()
            if not episode.closed
            for call in episode.calls_held
        }
        # Owner ruling 2026-09-08: a standing authorisation is the one verdict
        # with no ask behind it -- the actor spoke the go-ahead on the beat
        # before anything was held (acc_039 t40). It names a tool and no id, so
        # the anchor above would drop it, and its own scope rule takes the place
        # of the anchor: `preauthorised_acts` reads it for this turn only and
        # nothing writes it into the hold.
        anchored_consent = [
            decision for decision in (consent_decisions or [])
            if getattr(decision, "is_standing", False)
            or getattr(decision, "interaction_item_id", "") in held_now
        ]

        def _build_tools(context: dict[str, Any]) -> TurnToolRegistry:
            # The trigger turn itself: the `capability_limit` result arrives
            # mid-turn, so the registry the turn opened with cannot carry
            # `IX_capability_boundary`. The runner rebuilds the view through
            # this factory before the next provider request of the same turn
            # (`runner.py`, `tools_factory`), which is where the selection tool
            # first appears. No duplicate if the beat listed it as well.
            selection_ix = turn_active_ix
            if (
                context.get("capability_limit_seen")
                and _TRIGGERED_IX not in turn_active_ix
                and _TRIGGERED_IX not in selected_attrs
            ):
                selection_ix = [*turn_active_ix, _TRIGGERED_IX]
            return TurnToolRegistry(self.tools, selection_ix, ix_context=context)

        turn_tools = _build_tools(ix_context)
        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=turn_tools,
            model=self.pa_model,
            max_iterations=self.max_iterations,
            hook=_LoopHook(),
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
            required_tool_names=turn_tools.required_ix_tool_names,
            ix_context=ix_context,
            inherited_ix_selections=inherited_ix_selections,
            turn_reminder=turn_reminder_message(inherited_ix_selections, active_ix_all),
            hold_prompt=hold_prompt(hold),
            hold=hold,
            consent_decisions=anchored_consent,
            turn_index=int(ix_context.get("turn_index") or 0),
            tools_factory=_build_tools,
            llm_task_identity=llm_task_identity,
        ))
        self._remember_ix_selections(ix_cache_key, result.tool_events)
        self._remember_ix_triggers(ix_cache_key, result)
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            provider_error = result.error or result.final_content or "unknown provider error"
            failure = AgentProviderError(
                "PA provider failed to produce a response: "
                f"model={self.pa_model!r} error={provider_error!r}"
            )
            failure.failed_turn_events = {
                "tool_events": list(result.tool_events),
                "iterations": list(result.iteration_log),
            }
            raise failure
        return result

    def _cached_ix_selections(self, cache_key: tuple[str, str], active_ix: list[str]) -> list[Any]:
        if not cache_key[0]:
            return []
        by_attr = self._ix_selection_cache.get(cache_key) or {}
        attrs = [
            *active_ix,
            *(attr for attr in _SESSION_SCOPED_IX if attr not in active_ix),
        ]
        return [
            selection
            for attr in attrs
            if (selection := by_attr.get(attr)) is not None
        ]

    def _remember_ix_selections(self, cache_key: tuple[str, str], tool_events: list[dict[str, Any]]) -> None:
        if not cache_key[0]:
            return
        selections = selections_from_tool_events(tool_events)
        if not selections:
            return
        by_attr = self._ix_selection_cache.setdefault(cache_key, {})
        for selection in selections:
            # A `--force-ix` seed outranks the model's own choice for the rest
            # of the session. This only bites on the environment-event turn,
            # where the selection tool is still offered because it carries the
            # outreach message; everywhere else the tool is already withdrawn.
            if getattr(by_attr.get(selection.attribute), "forced", False):
                continue
            by_attr[selection.attribute] = selection

    def _remember_ix_triggers(
        self, cache_key: tuple[str, str], result: AgentRunResult
    ) -> None:
        """Make a runtime-triggered attribute stick for the rest of the session.

        The runner sees the `capability_limit` result and opens the two-stage
        protocol inside the turn; this is what carries the activation to the
        next one, so a beat that never listed `capability_boundary` still runs
        under the setting the trigger turn chose.
        """
        if not cache_key[0]:
            return
        collector = result.collector
        if collector is not None and getattr(collector, "capability_limit_seen", False):
            self._ix_triggered_cache.setdefault(cache_key, set()).add(_TRIGGERED_IX)

    def triggered_ix_skills(
        self, session_key: str, script_session_id: str | None = None
    ) -> list[str]:
        """Which attributes a runtime trigger has activated in this session.

        The harness restates this as `metadata["triggered_skills"]`
        (`harness/pa_factory.py`), which is how the simulator learns that an
        attribute is live on a beat whose script never named it.
        """
        cache_key = (session_key, str(script_session_id or ""))
        return sorted(self._ix_triggered_cache.get(cache_key) or ())

    def _render_turn(
        self,
        result: AgentRunResult,
        *,
        cache_key: tuple[str, str],
        inherit: bool = True,
        proactive: ProactiveDecision | None = None,
    ) -> RenderedTurn:
        """Render this turn and fold the result back into the hold cache.

        This is the last stage before anything is persisted or shown: the
        message the user gets, the history the next provider call sees and the
        memory event all descend from the same rendered turn.
        """
        collector = result.collector
        render_input = result.render_input
        if collector is None or render_input is None:
            return RenderedTurn(message=result.final_content or "")

        if proactive is not None:
            render_input = replace(render_input, proactive_decision=proactive)
        collector.settings = {
            attribute: getattr(selection, "setting", "") or ""
            for attribute, selection in (self._ix_selection_cache.get(cache_key) or {}).items()
        }
        # Owner-approved verification aid (2026-09-08, `pa/agent/ix/verification.py`).
        # Here and nowhere else: the model's selection is already in
        # `result.tool_events` and in the selection cache, so the evidence keeps
        # the choice the PA actually made and this adds a second record beside
        # it. Both the decision and the cached setting are forced, because
        # `_render_proactive` re-derives the action from the setting and would
        # otherwise put the model's branch straight back.
        if proactive is not None and (forced := verify_outreach_action()) is not None:
            setting = OVERRIDE_SETTINGS[forced]
            result.tool_events.append({
                "ix_event": True,
                "name": VERIFICATION_OVERRIDE,
                "attribute": "proactive_outreach",
                "model_outreach_action": proactive.action,
                "model_setting": proactive.setting,
                "outreach_action": forced,
                "setting": setting,
                "source": VERIFY_OUTREACH_ENV_VAR,
            })
            render_input = replace(
                render_input,
                proactive_decision=replace(
                    proactive, action=forced, setting=setting
                ),
            )
            collector.settings["proactive_outreach"] = setting
        rendered = render_turn(
            render_input,
            collector.settings,
            held=collector.held,
            executed_call_ids=collector.executed_call_ids,
            held_call_ids=collector.held_call_ids,
        )
        if not rendered.message and result.final_content and rendered.proactive is None:
            # The one delivery path that never went through `_strip_markers`.
            # Contract 6 is about what the user may see, not about which
            # function produced it, so the fallback gets the same cleanup.
            rendered = replace(
                rendered, message=clean_delivered_text(result.final_content)
            )
        # §3.7 (part 4): reaching the round cap cancels; it never releases. The user
        # still has to be told, and that line is the PA's to write -- the
        # renderer's template line underneath it does not make the miss go away.
        # `cancelled_now`, not `finished_episodes`: the latter accumulates for
        # the whole session, so live `v6g01`'s two cancellations were reported
        # as nine, and anything counting cancellations off the event stream
        # over-counted 4.5 to 1.
        # acc_009 t=8 — the questions the payload declared and the prose never
        # marked. Recorded every time it happens, nudge or no nudge: the drop is
        # the fact worth counting, and the reply that survives one nudge is
        # exactly the reply nobody was told about.
        dropped_questions = unmarked_questions(render_input)
        if dropped_questions:
            result.tool_events.append({
                "ix_event": True, "name": "ix_questions_unmarked",
                "attribute": "information_elicitation",
                "question_count": dropped_questions,
            })
        # acc_001 t32/t34 — the model's own tool-call syntax arriving as prose.
        # The stripper keeps it off the surface; this is what says it happened,
        # because a stripped block is usually a call that never ran and the
        # session log alone cannot tell that from a call the PA chose not to
        # make. `chars` is the size of what was cut: t32's was 7056.
        markup = [
            item for item in rendered.withheld_narration
            if item.item_id == "dsml_markup"
        ]
        if markup:
            result.tool_events.append({
                "ix_event": True, "name": "ix_markup_stripped",
                "blocks": len(markup),
                "chars": sum(len(item.text) for item in markup),
            })
        # acc_035 — a declared claim the locator could not place gets no hedge,
        # so the turn ships flat and the attribute has no surface. It was
        # trace-only for three rounds, and that is how a `claim_end` gap ate the
        # first `low` claim in four rounds without anything counting it.
        # regress13 acc_035 — one row per dropped claim, not one row per turn
        # carrying a `refs` list: the four drops of that session sat in two
        # events, so a counter over the stream read "2" where the surface lost
        # four hedges, and neither row said *which* sentence went missing. The
        # `claim` head is what the read needs to classify the drop — three of
        # those four were claims copied out of an *earlier* turn's reply, which
        # a bare ref cannot show — and `turn` is what puts the drop beside the
        # beat that was supposed to read the surface (t=7, all three).
        claims = {
            item.ref: str(item.claim or "")
            for item in collector.payloads.certainties
        }
        for ref in rendered.claims_not_found:
            result.tool_events.append({
                "ix_event": True, "name": "ix_certainty_claim_not_found",
                "attribute": "uncertainty_expression",
                "ref": ref,
                "claim": claims.get(ref, "")[:_CLAIM_HEAD_CHARS],
                "turn": render_input.turn_index,
            })
        # regress13 acc_009 — the excision's own tidy-up, recorded because it is
        # a sentence the PA wrote and the user did not read. Three turns there
        # shipped the lead-in alone; the count is what says how often the
        # setting's release rate and the PA's prose disagreed.
        if rendered.leadin_dropped:
            result.tool_events.append({
                "ix_event": True, "name": "ix_leadin_dropped",
                "attribute": "information_elicitation",
                "turn": render_input.turn_index,
                "sentence": rendered.leadin_dropped,
            })
        for episode in rendered.cancelled_now:
            result.tool_events.append({
                "ix_event": True, "name": "ix_hold_forced_close",
                "attribute": episode.attribute, "episode_id": episode.episode_id,
                "cancelled": len(episode.cancelled),
            })
            if not collector.payloads.closing_message:
                result.tool_events.append({
                    "ix_event": True, "name": "ix_close_no_message",
                    "attribute": episode.attribute, "episode_id": episode.episode_id,
                })
        # §4.3, owner decision 2026-09-06: a carried question the PA released
        # without re-marking it is folded into this reply's prose -- released,
        # settled, nothing carried over. That is the rule the hold reminder
        # states, so it is not a failure; the record is what lets the batch
        # review flag every beat where a held question went out unmarked.
        for folded in rendered.release_folded:
            result.tool_events.append(
                {"ix_event": True, "name": "ix_release_folded", **folded}
            )
        # A world-changing call was withheld and the turn stopped for nobody:
        # the user is handed ordinary prose with no ask and no question, so
        # nothing in the delivered message says that anything is pending. Live,
        # a withheld `email_message_send` was reported to the user as sent. This
        # is a flag for the IX judge to read the prose against the tool log --
        # it is structural, and it does not claim the PA said anything false.
        #
        # What the turn delivered decides it. §3.4 (part 5) put prose back on exactly
        # the hold turns this reads, so on a batch hold the flag fired by
        # construction (`v5g01` = 3, two of them honest) until the three things
        # a hold turn may legitimately deliver were excluded: the gate's ask,
        # released questions, and a released batch -- each of them a message
        # about work that did happen, or a request the user still owes.
        #
        # A refusal the turn itself answered is the fourth exclusion, for the
        # same reason as the other three: the action did happen. The refused
        # attempt stays in `calls` as `executed=False` history, and
        # `mark_binding_repaired` is what records that the resent call ran, so
        # the row it stamped is not outstanding. Without this the flag fired on
        # the very turn that carried the action out -- `acc_009`'s shape, one
        # layer below the `held.calls` fix it already got.
        if collector.gate is None:
            repaired = {
                item.interaction_item_id for item in collector.held if item.repaired
            }
            withheld = [
                call for call in collector.calls
                if not call.executed
                and call.action_type in WORLD_CHANGING_ACTION_TYPES
                and call.interaction_item_id not in repaired
            ]
            delivered = released_ids(render_input, collector.settings)
            if (
                withheld
                and rendered.message.strip()
                and not delivered["batches"]
                and not delivered["questions"]
            ):
                result.tool_events.append({
                    "ix_event": True, "name": "ix_claims_unexecuted_action",
                    "tools": [call.tool_name for call in withheld],
                    "item_ids": [call.interaction_item_id for call in withheld],
                })
        if inherit and cache_key[0]:
            self._ix_hold_cache[cache_key] = rendered.hold_next
        return rendered

    @staticmethod
    def _proactive_decision(result: AgentRunResult) -> ProactiveDecision:
        """The PA's explicit send/suppress call for this environment event."""
        event = next(
            (
                event
                for event in reversed(result.tool_events)
                if event.get("ix")
                and event.get("attribute") == "proactive_outreach"
                and event.get("outreach_action") in {"send", "suppress"}
            ),
            None,
        )
        if event is None:
            raise ProactiveDecisionMissingError(
                "Proactive environment event completed without a valid "
                "send/suppress decision"
            )
        return ProactiveDecision(
            action=str(event["outreach_action"]),
            setting=str(event.get("setting") or ""),
            message=str(event.get("outreach_message") or ""),
        )

    def close_ix_session(
        self, session_key: str, script_session_id: str | None = None
    ) -> list[tuple[str, str]]:
        """Clear this benchmark session's IX runtime state, atomically.

        Returns the ``(attribute, episode_id)`` pairs that were still open, so
        the caller can fail the step: a step boundary must not carry an open
        hold. Selection, hold and trigger caches are cleared together in the
        ``finally`` whatever the outcome -- a retry that inherits last attempt's
        chosen setting but none of its holds would run with two mismatched state
        planes.
        """
        cache_key = (session_key, str(script_session_id or ""))
        try:
            return open_episode_ids(self._ix_hold_cache.get(cache_key) or IXHold())
        finally:
            self._ix_hold_cache.pop(cache_key, None)
            self._ix_selection_cache.pop(cache_key, None)
            self._ix_triggered_cache.pop(cache_key, None)

    @staticmethod
    def _pa_turn_task_identity(
        *,
        metadata: dict[str, Any],
        session_key: str,
    ) -> LLMTaskIdentity:
        run_id = str(metadata.get("run_id") or "interactive")
        turn_id = str(
            metadata.get("turn_id")
            or metadata.get("message_id")
            or time.time_ns()
        )
        return LLMTaskIdentity(
            logical_task_key=f"pa_turn:{run_id}:{session_key}:{turn_id}"
        )

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                # take one inbound message
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)
            # in effect: once this task is done it has to be removed

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    # Split one answer into distinct stream segments.
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def on_stream(delta: str) -> None:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content=delta,
                            metadata={
                                "_stream_delta": True,
                                "_stream_id": _current_stream_id(),
                            },
                        ))

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="",
                            metadata={
                                "_stream_end": True,
                                "_resuming": resuming,
                                "_stream_id": _current_stream_id(),
                            },
                        ))
                        stream_segment += 1

                response = await self._process_message(
                    msg, on_stream=on_stream, on_stream_end=on_stream_end,
                )
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    # Trivial branch of §2.3: nothing was rendered, so the
                    # empty CLI reply carries an empty rendered turn.
                    await self.bus.publish_outbound(self._outbound(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=dict(msg.metadata or {}),
                        rendered=RenderedTurn(message=""),
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(self._outbound(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.", metadata={},
                    rendered=RenderedTurn(message="Sorry, I encountered an error."),
                ))

    async def drain_background_tasks(self) -> None:
        """Wait for pending background archives to finish.

        Checkpoints call this on its own: `close_mcp` also tears down the MCP
        connections, which is a run-teardown action, not a cut-point action.
        """
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        await self.drain_background_tasks()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    async def _compact_prompt_history_if_needed(
        self, session: Session, *, stage: str = "legacy",
    ) -> None:
        if self.skip_consolidation:
            return
        await self.memory_consolidator.maybe_compact_prompt_history(session, stage=stage)

    async def _compact_prompt_history_after_turn(
        self, session: Session, *, stage: str = "legacy",
    ) -> None:
        await self._compact_prompt_history_if_needed(session, stage=stage)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            # the subagent result came back
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = session_key or f"{channel}:{chat_id}"
            session = self._get_message_session(key)
            await self._compact_prompt_history_if_needed(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            adapter_memory_context = None
            if msg.sender_id == "environment":
                adapter_memory_context = await self._retrieve_adapter_memory(
                    msg.metadata or {},
                    current_message=msg.content,
                    session_key=key,
                )
            history = session.get_history(max_messages=0)
            current_role = (
                "assistant"
                if msg.sender_id == "subagent"
                else ENVIRONMENT_EVENT_ROLE
                if msg.sender_id == "environment"
                else "user"
            )
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
                adapter_memory_context=adapter_memory_context,
                timestamp=(
                    msg.timestamp if msg.timestamp.tzinfo is not None else None
                ),
            )
            if msg.sender_id == "environment":
                messages[-1]["_mempa_internal"] = True
            elif msg.sender_id == "subagent":
                # `[subagent_result]` is tagged from data, never inferred from
                # a role: this message is `role: "assistant"` like any other.
                messages[-1]["source_kind"] = "subagent"
            cache_key = (key, str((msg.metadata or {}).get("script_session_id") or ""))
            result = await self._run_agent_loop(
                messages, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
                active_ix=msg.metadata.get("active_ix"),
                ix_context=msg.metadata.get("ix_context"),
                session_key=key,
                script_session_id=msg.metadata.get("script_session_id"),
                llm_task_identity=self._pa_turn_task_identity(
                    metadata=msg.metadata or {},
                    session_key=key,
                ),
                consent_decisions=consent_decisions_from_metadata(msg.metadata),
                user_message=msg.content,
            )
            if msg.sender_id == "environment":
                # Render first, then persist. The rendered decision is the only
                # source for the outbound, the history and the memory event --
                # a suppressed reminder must not reach any of the three.
                proactive = self._proactive_decision(result)
                rendered = self._render_turn(
                    result, cache_key=cache_key, inherit=False, proactive=proactive,
                )
                decision = (
                    rendered.proactive.action if rendered.proactive is not None
                    else proactive.action
                )
                outbound_content = rendered.message if decision == "send" else ""
                # Both outcomes write exactly one pool. Silence is an
                # interaction choice and `proactive_outreach` is scored on it,
                # so a suppressed turn is not a turn that never happened --
                # but it delivered nothing, so it appends no session message
                # (owner ruling 2026-09-13, round-2 Q2 = A) and its journal
                # span is `[n, n)`.
                before_len = len(session.messages)
                pool = self._build_turn_pool(
                    result=result,
                    history_length=len(history),
                    delivered=outbound_content,
                    rendered=rendered,
                    environment_event=(msg.metadata or {}).get(
                        ENVIRONMENT_EVENT_BLOCK
                    ),
                )
                turn_id = self._turn_id(msg.metadata or {})
                journal = self.turn_journal(key)
                self._save_turn(
                    session,
                    [
                        {
                            "role": "assistant",
                            "content": outbound_content,
                            "timestamp": msg.timestamp.isoformat(),
                        }
                    ],
                    0,
                )
                event_writer.record_turn_journal(
                    journal=journal,
                    turn_id=turn_id,
                    session_id=key,
                    span=(before_len, len(session.messages)),
                    timestamp=msg.timestamp,
                    pool=pool,
                    state="pending",
                )
                self.sessions.save(session)
                receipt = await self._write_turn_to_memory_adapter(
                    msg.metadata or {},
                    pool,
                    timestamp=msg.timestamp,
                    session_key=key,
                    turn_id=turn_id,
                )
                self._record_adapter_write_commit(
                    session,
                    receipt,
                    adapter_committed_end_index=len(session.messages),
                )
                event_writer.record_turn_journal(
                    journal=journal,
                    turn_id=turn_id,
                    session_id=key,
                    span=(before_len, len(session.messages)),
                    timestamp=msg.timestamp,
                    pool=pool,
                    state="written",
                    receipt=receipt,
                )
                self.sessions.save(session)
                if decision == "send":
                    await self._compact_prompt_history_after_turn(session)
                proactive_meta = dict(msg.metadata or {})
                proactive_meta.pop(CONSENT_DECISIONS_BLOCK, None)
                proactive_meta.pop(ENVIRONMENT_EVENT_BLOCK, None)
                proactive_meta.update({
                    "tool_events": result.tool_events,
                    "outreach_decision": decision,
                    "outreach_delivered": decision == "send",
                })
                return self._outbound(
                    channel=channel, chat_id=chat_id, content=outbound_content,
                    metadata=proactive_meta, rendered=rendered, result=result,
                )

            rendered = self._render_turn(result, cache_key=cache_key)
            public_turn = rendered.message or (
                result.final_content or "Background task completed."
            )
            released, held_not_released, cancelled = self._ix_payload_ids(rendered, result)
            runtime_history = build_runtime_history(
                user_message=messages[-1],
                messages=result.messages[1 + len(history):],
                executed_call_ids=rendered.executed_call_ids,
                public_turn=public_turn,
                released=released,
                held_not_released=held_not_released,
                cancelled=cancelled,
            )
            before_len = len(session.messages)
            pool = self._build_turn_pool(
                result=result,
                history_length=len(history),
                delivered=public_turn,
                rendered=rendered,
            )
            turn_id = self._turn_id(msg.metadata or {})
            journal = self.turn_journal(key)
            self._save_turn(session, runtime_history.messages, 0, msg.timestamp)
            event_writer.record_turn_journal(
                journal=journal,
                turn_id=turn_id,
                session_id=key,
                span=(before_len, len(session.messages)),
                timestamp=msg.timestamp,
                pool=pool,
                state="pending",
            )
            self.sessions.save(session)
            receipt = await self._write_turn_to_memory_adapter(
                msg.metadata or {},
                pool,
                timestamp=msg.timestamp,
                session_key=key,
                turn_id=turn_id,
            )
            self._record_adapter_write_commit(
                session,
                receipt,
                adapter_committed_end_index=len(session.messages),
            )
            event_writer.record_turn_journal(
                journal=journal,
                turn_id=turn_id,
                session_id=key,
                span=(before_len, len(session.messages)),
                timestamp=msg.timestamp,
                pool=pool,
                state="written",
                receipt=receipt,
            )
            self.sessions.save(session)
            await self._compact_prompt_history_after_turn(session)
            return self._outbound(
                channel=channel, chat_id=chat_id, content=public_turn,
                metadata={"tool_events": result.tool_events},
                rendered=rendered, result=result,
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self._get_message_session(key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        # A genuine new user request is present, but not yet in durable history.
        await self._compact_prompt_history_if_needed(session, stage="incoming_user_turn")

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        # MessageTool.start_turn() will be called here once BenchmarkChannel is added

        adapter_memory_context = await self._retrieve_adapter_memory(
            msg.metadata or {},
            current_message=msg.content,
            session_key=key,
        )

        history = session.get_history(max_messages=0)
        # assemble the messages
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            adapter_memory_context=adapter_memory_context,
            timestamp=(
                msg.timestamp if msg.timestamp.tzinfo is not None else None
            ),
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        # the llm loop runs here
        cache_key = (key, str((msg.metadata or {}).get("script_session_id") or ""))
        # Open a fresh upstream-provider window for this turn. Memory retrieval
        # above has already run, so the window holds exactly the agent loop's
        # own provider calls, and a failed turn can name the upstream that
        # served each one. Providers that do not track upstreams have no such
        # method and are left alone.
        reset_upstreams = getattr(
            self.pa_llm_provider, "reset_upstream_providers", None
        )
        if callable(reset_upstreams):
            reset_upstreams()
        result = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=msg.channel, chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            active_ix=msg.metadata.get("active_ix"),
            ix_context=msg.metadata.get("ix_context"),
            session_key=key,
            script_session_id=msg.metadata.get("script_session_id"),
            llm_task_identity=self._pa_turn_task_identity(
                metadata=msg.metadata or {},
                session_key=key,
            ),
            consent_decisions=consent_decisions_from_metadata(msg.metadata),
            user_message=msg.content,
        )

        # The wrapper is the last stage before anything is persisted or shown.
        # A contract failure produces no scorable answer at all.
        rendered = self._render_turn(result, cache_key=cache_key)
        if result.stop_reason == STOP_REASON_IX_CONTRACT_FAILURE:
            # The runner names the rule that ended the turn at the site that
            # ended it. Reading the event stream instead cannot work: the
            # non-fatal repairs emit the same events as the fatal ones, and
            # anything the list did not name fell through to the capability
            # text -- which is how a refused selection on a beat with no IX
            # tools was reported as an unresolved capability limit (regress20
            # acc_027). So: no default, and an unnamed kind says so.
            kind = result.ix_contract_failure_kind
            detail = _IX_CONTRACT_FAILURE_DETAILS.get(kind or "") or (
                f"contract: an IX rule that names no kind ({kind!r}) ended "
                "this turn"
            )
            # The failed turn is never persisted as a turn, so its tool events
            # (the ladder: nudge, pin, the calls the PA made instead) ride on
            # the exception for the harness to write beside the step.
            failure = AgentProviderError(f"PA turn failed the IX {detail}")
            failure.failed_turn_events = {
                "tool_events": list(result.tool_events),
                "iterations": list(result.iteration_log),
                # Only the final collector state survives repairs. Keep its
                # confirmation text so an empty field and a rejected statement
                # can be distinguished without saving provider reasoning.
                "unexecuted_calls": [
                    {"tool": call.tool_name, "call_id": call.tool_call_id,
                     "interaction_item_id": call.interaction_item_id,
                     "confirmation_message": call.ask_text, "status": call.status}
                    for call in (result.render_input.calls if result.render_input else [])
                    if not call.executed
                ],
            }
            exhaustion = result.ix_contract_exhaustion
            if exhaustion is not None:
                # The third response on one in-turn repair ladder; with
                # `failure_policy.auto_skip_on_ix_exhaustion` the orchestrator
                # skips an ordinary ACC that ends on it.
                failure.ix_contract_exhaustion = dict(exhaustion)
                failure.failed_turn_events["ix_contract_exhaustion"] = dict(
                    exhaustion
                )
            # The upstreams that served this turn's requests, in order, so a
            # contract failure can be attributed to one. Ids only -- no
            # response bodies, and nothing when the gateway does not report
            # them (`iterations` is built in the runner and is not touched).
            upstreams = getattr(self.pa_llm_provider, "upstream_providers", None)
            if callable(upstreams) and (seen := upstreams()):
                failure.failed_turn_events["upstream_providers"] = seen
            raise failure
        # No placeholder here any more (round 9 item 1). An empty delivered turn
        # is decided at the runner's single exit, which spends one request asking
        # for the sentence the turn owes and hands back the template line when
        # that comes back empty too -- so by the time the renderer runs there is
        # something to say, and the loop is not the last chance to invent one.
        public_turn = rendered.message or result.final_content or ""

        released, held_not_released, cancelled = self._ix_payload_ids(rendered, result)
        runtime_history = build_runtime_history(
            user_message=initial_messages[-1],
            messages=result.messages[1 + len(history):],
            executed_call_ids=rendered.executed_call_ids,
            public_turn=public_turn,
            released=released,
            held_not_released=held_not_released,
            cancelled=cancelled,
        )

        # save the conversation history
        before_len = len(session.messages)
        # The pool is the raw provider slice, not this: projection 2 blanks an
        # intermediate iteration's prose and rebuilds the final assistant
        # message without its reasoning, so the persisted slice cannot
        # re-derive the turn's experience (§6.1 (b)).
        pool = self._build_turn_pool(
            result=result,
            history_length=len(history),
            delivered=public_turn,
            rendered=rendered,
        )
        turn_id = self._turn_id(msg.metadata or {})
        journal = self.turn_journal(key)
        self._save_turn(session, runtime_history.messages, 0, msg.timestamp)
        # Durable before the session file it describes (§6.1 (b)): a crash
        # between the two loses the turn's prose and reasoning otherwise, and
        # `session.messages` cannot re-derive them.
        event_writer.record_turn_journal(
            journal=journal,
            turn_id=turn_id,
            session_id=key,
            span=(before_len, len(session.messages)),
            timestamp=msg.timestamp,
            pool=pool,
            state="pending",
        )
        self.sessions.save(session)
        receipt = await self._write_turn_to_memory_adapter(
            msg.metadata or {},
            pool,
            timestamp=msg.timestamp,
            session_key=key,
            turn_id=turn_id,
        )
        self._record_adapter_write_commit(
            session,
            receipt,
            adapter_committed_end_index=len(session.messages),
        )
        event_writer.record_turn_journal(
            journal=journal,
            turn_id=turn_id,
            session_id=key,
            span=(before_len, len(session.messages)),
            timestamp=msg.timestamp,
            pool=pool,
            state="written",
            receipt=receipt,
        )
        self.sessions.save(session)
        await self._compact_prompt_history_after_turn(session, stage="completed_user_turn")

        # When MessageTool is registered (Stage 2 BenchmarkChannel),
        # check mt._sent_in_turn here — if PA already sent via MessageTool, return None.

        preview = public_turn[:120] + "..." if len(public_turn) > 120 else public_turn
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None:
            meta["_streamed"] = True
        meta["tool_events"] = result.tool_events
        meta.pop(CONSENT_DECISIONS_BLOCK, None)
        return self._outbound(
            channel=msg.channel, chat_id=msg.chat_id, content=public_turn,
            metadata=meta, rendered=rendered, result=result,
        )

    def _outbound(
        self,
        *,
        channel: str,
        chat_id: str,
        content: str,
        metadata: dict[str, Any],
        rendered: RenderedTurn | None,
        result: AgentRunResult | None = None,
    ) -> OutboundMessage:
        """The single exit for every message the simulated user can see.

        Held content leaves the PA only here, and only into
        ``metadata["ix_render"]`` -- the audit plane. It is never dressed up as
        an assistant message in history and never reaches memory.
        """
        if rendered is None:
            raise AssertionError("_outbound requires a rendered turn")
        collector = getattr(result, "collector", None)
        metadata = dict(metadata)
        metadata["ix_render"] = to_ix_render_metadata(
            rendered,
            episodes={
                attribute: {
                    "episode_id": episode.episode_id,
                    "rounds_open": episode.rounds_open,
                    "closed": episode.closed,
                }
                for attribute, episode in rendered.hold_next.open_episodes.items()
            },
            released=(
                {"batches": [], "questions": [], "plan_executed": []}
                if result is None
                or result.render_input is None
                or rendered.proactive is not None
                else released_ids(
                    result.render_input,
                    {} if collector is None else collector.settings,
                )
            ),
            held=self._held_json(rendered, collector),
            gate=None if collector is None else collector.gate,
            consent=() if collector is None else collector.consent,
            notices=(
                ()
                if collector is None
                else [
                    {"order": call.order, "tool": call.tool_name, "text": call.notice}
                    for call in collector.calls
                    if call.notice
                ]
            ),
            # `released` is read off the renderer's own output, never
            # re-derived here from the setting: on a gate turn §4.4 gives the
            # closing line no place after the ask, and setting+message said
            # `released: true` for a reply that carried none. `text` still
            # records the sentence the PA wrote, delivered or not.
            completion=(
                None
                if collector is None
                else {
                    "released": rendered.completion_delivered,
                    "text": collector.payloads.completion_message,
                    **(
                        {"skipped": rendered.completion_skipped}
                        if rendered.completion_skipped else {}
                    ),
                }
            ),
            # `RenderedTurn` has no slot for these audit rows, so the same pure
            # functions the renderer used are re-run here rather than a second
            # copy of the rules being written.
            reasoning=(
                None
                if collector is None
                else render_rationale(
                    collector.settings.get("reasoning_visibility", ""),
                    collector.payloads.user_facing_rationale,
                    provider_reasoning_present=bool(collector.provider_reasoning),
                ).reasoning
            ),
            # §4.4 `raw_blocks`: the carrier the renderer strips out of every
            # delivered surface. Without it the audit plane has no record of
            # what the model actually committed to, which is what makes a
            # located-but-withheld claim indistinguishable from one never found.
            raw_blocks=(
                ()
                if collector is None
                else [
                    {"kind": "certainties", "text": block}
                    for block in (
                        certainty_block_text(
                            collector.segments[-1] if collector.segments else ""
                        ),
                    )
                    if block
                ]
            ),
            certainties=(
                ()
                if collector is None
                else render_certainties(
                    collector.settings.get("uncertainty_expression", ""),
                    collector.segments[-1] if collector.segments else "",
                    collector.payloads.certainties,
                ).rows
            ),
            # §4.4 — how the turn ended and whether §3.4 (part 5)'s closing request
            # ran. Neither had any record in any artefact: `ix_hold` had to be
            # derived from held statuses and provider prompt sizes.
            stop_reason="" if result is None else (result.stop_reason or ""),
            closing_iteration=None if collector is None else collector.closing_iteration,
            cancelled_items=self._cancelled_items(rendered),
        )
        return OutboundMessage(
            channel=channel, chat_id=chat_id, content=content, metadata=metadata,
        )

    @staticmethod
    def _cancelled_items(rendered: RenderedTurn) -> list[dict[str, Any]]:
        """§4.4 `cancelled_items`: every item this session cancelled, id + intent.

        Read off `finished_episodes`, which accumulates, precisely because the
        judge needs it on the turns *after* the cancellation: those turns
        withhold nothing, so no detector fires when the PA lists a cancelled
        item as done -- live, `v6g01` t35's "Confirmation email sent to
        Dr. Okafor" for an `email_message_send` that never ran. De-duplicated
        by `(attribute, id)`: one episode can be cancelled twice in a session.
        """
        items: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for episode in rendered.hold_next.finished_episodes:
            for entry in episode.cancelled:
                key = (str(entry.get("attribute") or ""), str(entry.get("id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                items.append({
                    "attribute": key[0],
                    "id": key[1],
                    "kind": str(entry.get("kind") or ""),
                    "intent": str(entry.get("intent") or ""),
                })
        return items

    @staticmethod
    def _ix_payload_ids(
        rendered: RenderedTurn, result: AgentRunResult
    ) -> tuple[list[str], list[str], list[str]]:
        """The three id lists §2.4's persisted IX payload result states.

        `released` is what this turn actually let out; `held_not_released` is
        what the hold still carries after it; `cancelled` is what this turn's
        forced close dropped for good (§3.7 (part 4)) — read off
        `cancelled_now` rather than `finished_episodes`, because the status is
        per turn and a cancellation must not repeat on every later turn. Ids
        only — the whole point of replacing the payload echo is that the held
        bodies are not in the record, so nothing here may be read off
        `Question.text` or `Batch.body`.

        `cancelled` is separate from `held_not_released` because a forced close
        *empties* the hold: after it, the turn holds nothing and yet the
        cancelled section was never delivered either. Memory reads both.
        """
        collector = result.collector
        delivered = (
            {"batches": [], "questions": [], "plan_executed": []}
            if result.render_input is None or rendered.proactive is not None
            else released_ids(
                result.render_input, {} if collector is None else collector.settings
            )
        )
        held = [
            item_id
            for episode in rendered.hold_next.open_episodes.values()
            if not episode.closed
            for item_id in (
                [batch.batch_id for batch in episode.batches_held]
                + [question.id for question in episode.questions_held]
                + [
                    str(call.get(INTERACTION_ITEM_ID_FIELD) or "")
                    for call in episode.calls_held
                ]
            )
        ]
        released = [
            *delivered["batches"],
            *delivered["questions"],
            *delivered["plan_executed"],
        ]
        cancelled = [
            str(entry.get("id") or "")
            for episode in rendered.cancelled_now
            for entry in episode.cancelled
        ]
        return released, held, cancelled

    @staticmethod
    def _held_json(rendered: RenderedTurn, collector: Any) -> dict[str, Any]:
        """§4.4 `held`: the held bodies verbatim. Audit plane only.

        Open episodes only. `finished_episodes` accumulates for the whole
        session and a closing episode is filed with its `plan_remaining`
        intact, so reading both reported a step as still held on every later
        turn -- live, a booking that had already happened was reported held
        three turns later, while `episodes` on the same blob said `{}`.
        `cancelled` is the exception: it is a terminal fact and only ever
        written on the episode that was cancelled.

        `narration` is the prose written in the same assistant message as a held
        call (§3.3 (part 2)·D addendum), plus the sentences the wrapper-vocabulary filter
        withheld (round 9 item 3, `item_id: "wrapper_vocabulary"`). The key is
        absent when there is none, so a 1.1.0 record stays a valid 1.1.x record.

        `calls` is read from the open episodes for the same reason the other
        three keys are: an episode carries its pending items across turns, while
        `rendered.held` is only what the gate stopped on *this* turn. Reading
        the latter alone made a carried call disappear from the blob -- live
        (`v11s02` t7) `calls` was `[]` on a turn where the same blob's
        `plan_remaining` still listed the step it belonged to. `episode.calls_held`
        is the pending set (`render._pending` merges what the episode carried
        with this turn's holds), so this turn's rows are added on top only for
        the holds no episode owns: a fail-closed `binding_invalid` and its
        siblings are refused for the turn alone and open no episode
        (`render.NON_EPISODE_HOLD_REASONS`).
        """
        open_episodes = list(rendered.hold_next.open_episodes.values())
        narration = [
            *([] if collector is None else collector.held_narration),
            *rendered.withheld_narration,
        ]
        this_turn = {
            (item.interaction_item_id, item.tool): item for item in rendered.held
        }
        calls: list[dict[str, Any]] = []
        carried: set[tuple[str, str]] = set()
        for episode in open_episodes:
            for call in episode.calls_held:
                item_id = str(call.get("interaction_item_id") or "")
                tool = str(call.get("tool") or "")
                carried.add((item_id, tool))
                held_now = this_turn.get((item_id, tool))
                if held_now is not None:
                    calls.append(asdict(held_now))
                    continue
                reason = str(call.get("reason") or "")
                calls.append(asdict(HeldItem(
                    status=held_status(episode.attribute, reason),
                    interaction_item_id=item_id,
                    attribute=episode.attribute,
                    tool=tool,
                    plan_item_id=str(call.get("plan_item_id") or ""),
                    episode_id=episode.episode_id,
                    batch_id=str(call.get("batch_id") or ""),
                    reason=reason,
                    asked=bool(call.get("asked")),
                )))
        calls.extend(
            asdict(item) for item in rendered.held
            if (item.interaction_item_id, item.tool) not in carried
        )
        return {
            "batches": [asdict(b) for e in open_episodes for b in e.batches_held],
            "questions": [asdict(q) for e in open_episodes for q in e.questions_held],
            "plan_remaining": [
                asdict(i) for e in open_episodes for i in e.plan_remaining
            ],
            "calls": calls,
            "cancelled": [
                dict(item)
                for e in rendered.hold_next.finished_episodes
                for item in e.cancelled
            ],
            "outreach_message": (
                rendered.proactive.message
                if rendered.proactive is not None and rendered.proactive.action == "suppress"
                else ""
            ),
            **(
                {"narration": [asdict(item) for item in narration]}
                if narration else {}
            ),
        }

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline image block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if (
                block.get("type") == "image_url"
                and block.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                filtered.append(self._image_placeholder(block))
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if truncate_text and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    text = text[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _cut_tool_result(self, content: Any) -> Any:
        """One tool result in the single form every writer of it stores.

        `_save_turn` persists it and `_pool_input` hands it to
        `build_turn_pool`, which reads `truncated=` back from this length and
        this marker (`memory/event_writer.py:20-34`). A second cut site — or
        none, which is what the pool had until live run #2 stored a 25 348
        character `mcp_state_scholarly_paper_search` result with
        `truncated=false` — makes that stamp a lie, so both writers cut here
        and nowhere else.
        """
        if isinstance(content, str):
            if len(content) > self._TOOL_RESULT_MAX_CHARS:
                return content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            return content
        if isinstance(content, list):
            return self._sanitize_persisted_blocks(content, truncate_text=True)
        return content

    def _save_turn(
        self,
        session: Session,
        messages: list[dict],
        skip: int,
        timestamp: datetime | None = None,
    ) -> None:
        """Save new-turn messages into session, truncating large tool results.

        `timestamp` is the turn's own clock — the inbound message's, which the
        benchmark sets to the scripted session time and a CLI turn leaves at
        `datetime.now()`. It matters beyond bookkeeping: these stamps are what
        `MarkdownMemoryStore._format_messages` renders in front of every line
        the consolidation LM reads, so a wall-clock stamp here is the date that
        ends up written into MEMORY.md and HISTORY.md.
        """
        stamp = (timestamp or datetime.now()).isoformat()
        for m in messages[skip:]:
            entry = dict(m)
            if entry.get("_mempa_internal") is True:
                continue
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, (str, list)):
                    cut = self._cut_tool_result(content)
                    if isinstance(content, list) and not cut:
                        continue
                    entry["content"] = cut
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", stamp)
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_environment_event(
        self,
        *,
        trigger_id: str,
        source_type: str,
        prior_thread: str,
        detail: str,
        user_impact: str,
        session_key: str = "benchmark:direct",
        channel: str = "benchmark",
        chat_id: str = "direct",
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> OutboundMessage:
        """Let the PA decide whether an environment event warrants new contact."""
        fields = {
            "trigger_id": trigger_id,
            "source_type": source_type,
            "prior_thread": prior_thread,
            "detail": detail,
            "user_impact": user_impact,
        }
        empty = [name for name, value in fields.items() if not str(value).strip()]
        if empty:
            raise ValueError(f"proactive environment event fields must not be empty: {empty}")
        allowed_source_types = {
            "scheduled_time",
            "deadline",
            "state_change",
            "external_event",
        }
        if source_type not in allowed_source_types:
            raise ValueError(
                "proactive environment event source_type must be one of "
                f"{sorted(allowed_source_types)}"
            )
        if timestamp is not None and (
            not isinstance(timestamp, datetime)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise ValueError(
                "process_environment_event timestamp must be a timezone-aware datetime"
            )

        # The trigger the PA perceives, and the IX instruction the harness
        # appends to it, are built as two strings rather than one, so the pool
        # gets the event without the instruction by construction (Q4) instead
        # of by stripping a suffix off the text.
        event_body = render_environment_event(
            f"Trigger ID: {trigger_id}\n"
            f"Source type: {source_type}\n"
            f"Prior thread: {prior_thread}\n"
            f"Event: {detail}\n"
            f"Potential user impact: {user_impact}"
        )
        event_content = (
            f"{event_body}\n\n"
            "Use IX_proactive_outreach to choose the remembered interaction setting "
            "and explicitly decide whether to send a new contact or suppress it."
        )
        event_metadata = dict(metadata or {})
        event_metadata[ENVIRONMENT_EVENT_BLOCK] = {
            "source_id": source_type,
            "action_id": trigger_id,
            "body": event_body,
        }
        event_metadata["active_ix"] = ["proactive_outreach"]
        ix_context = dict(event_metadata.get("ix_context") or {})
        ix_context.update({
            "proactive_event": True,
            "active_request": False,
            "session_active_ix": ["proactive_outreach"],
            "trigger_id": trigger_id,
            "source_type": source_type,
            "prior_thread": prior_thread,
        })
        event_metadata["ix_context"] = ix_context
        event_metadata.setdefault("memory_retrieval_query", f"{prior_thread}\n{detail}")

        await self._connect_mcp()
        msg = InboundMessage(
            channel="system",
            sender_id="environment",
            chat_id=f"{channel}:{chat_id}",
            content=event_content,
            **({"timestamp": timestamp} if timestamp is not None else {}),
            metadata=event_metadata,
        )
        outbound = await self._process_message(msg, session_key=session_key)
        if outbound is None:
            raise RuntimeError("Proactive environment event produced no decision result")
        return outbound

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        if timestamp is not None and not isinstance(timestamp, datetime):
            raise ValueError(
                "process_direct_message timestamp must be a datetime: "
                f"value_type={type(timestamp).__name__} value={timestamp!r}"
            )
        if timestamp is not None and (
            timestamp.tzinfo is None or timestamp.utcoffset() is None
        ):
            raise ValueError(
                "process_direct_message timestamp must include a timezone: "
                f"value={timestamp!r}"
            )
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=user_message,
            **({"timestamp": timestamp} if timestamp is not None else {}),
            metadata=metadata or {},
        )
        return await self._process_message(
            msg, session_key=session_key, on_progress=on_progress,
            on_stream=on_stream, on_stream_end=on_stream_end,
        )
