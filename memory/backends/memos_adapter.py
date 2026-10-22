"""MemOS adapter for one run-scoped MemCube."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import math
import time
from typing import Any

from memory.adapter import PromptHistoryEvictionPolicy
from memory.backends.memos_client import MemOSClient, MemOSClientError
from memory.backends.memos_usage_log import MemosUsageReporter
from memory.events import MemoryEvent
from memory.results import (
    BackendHealth,
    ConsistencyModel,
    HealthStatus,
    ModeReceipt,
    ResetReceipt,
    RetrieveResult,
    Trace,
    WriteReceipt,
    WriteStatus,
)
from memory.scope import AdapterMode, MemoryScope, RetrieveRequest


_GROUP_LABELS = {
    "text_mem": "Text Memories",
}
_ALL_GROUP_NAMES = (
    "text_mem",
    "act_mem",
    "para_mem",
    "pref_mem",
    "tool_mem",
    "skill_mem",
)
class MemOSAdapter:
    """Map benchmark memory events to native MemOS add/search operations."""

    backend_name: str = "memos"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.after_adapter_commit

    def __init__(
        self,
        *,
        client: MemOSClient,
        cube_id: str,
        memory_session_id: str,
        add_mode: str = "fine",
        search_mode: str = "fast",
        top_k: int = 10,
        relativity: float = 0.45,
        dedup: str = "mmr",
        usage_reporter: MemosUsageReporter | None = None,
    ) -> None:
        self._cube_id = self._require_identifier("cube_id", cube_id)
        self._memory_session_id = self._require_identifier(
            "memory_session_id", memory_session_id
        )
        if add_mode not in {"fast", "fine"}:
            raise ValueError(
                f"MemOS add_mode must be 'fast' or 'fine': value={add_mode!r}"
            )
        if search_mode not in {"fast", "fine", "mixture"}:
            raise ValueError(
                "MemOS search_mode must be fast, fine, or mixture: "
                f"value={search_mode!r}"
            )
        if top_k <= 0:
            raise ValueError(f"MemOS top_k must be positive: value={top_k!r}")
        if not math.isfinite(relativity) or not 0 <= relativity <= 1:
            raise ValueError(
                "MemOS relativity must be finite and between 0 and 1: "
                f"value={relativity!r}"
            )
        if dedup not in {"no", "sim", "mmr"}:
            raise ValueError(
                "MemOS dedup must be no, sim, or mmr: "
                f"value={dedup!r}"
            )
        self._client = client
        self._add_mode = add_mode
        self._search_mode = search_mode
        self._top_k = top_k
        self._relativity = relativity
        self._dedup = dedup
        self._usage_reporter = usage_reporter
        self._adapter_mode = AdapterMode()
        self._scope: MemoryScope | None = None

    async def setup_scope(self, scope: MemoryScope) -> None:
        self._scope = scope

    async def set_mode(self, mode: AdapterMode) -> ModeReceipt:
        previous_mode = self._adapter_mode
        self._adapter_mode = mode
        return ModeReceipt(mode=mode, previous_mode=previous_mode)

    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        scope = MemoryScope(run_id=event.run_id, persona_id=event.persona_id)
        if self._adapter_mode.read_only or getattr(self._client, "read_only", False):
            return WriteReceipt(
                status=WriteStatus.skipped_read_only,
                trace=self._trace(scope, "add_memories"),
            )

        info: dict[str, Any] = {
            "run_id": event.run_id,
            "agent_id": event.agent_id,
            "turn_id": event.turn_id,
            "timestamp": event.timestamp.isoformat(),
        }
        add_request = {
            "messages": event.messages,
            "user_id": event.persona_id,
            "session_id": self._memory_session_id,
            "task_id": event.event_id,
            "writable_cube_ids": [self._cube_id],
            "async_mode": "sync",
            "mode": self._add_mode,
            "custom_tags": None,
            "info": info,
            "chat_history": [],
        }

        started_at = time.monotonic()
        result = self._client.add_memories(add_request)
        latency_ms = (time.monotonic() - started_at) * 1000
        # `async_mode: sync`, so the extraction calls this write paid for are
        # already in the container's log by the time it returns.
        if self._usage_reporter is not None:
            self._usage_reporter.emit_calls(turn_id=event.turn_id)
        native_ids = [
            str(item.get("memory_id") or item.get("id")) for item in result
        ]
        if not native_ids:
            return WriteReceipt(
                status=WriteStatus.accepted_no_change,
                trace=Trace(
                    backend_name=self.backend_name,
                    scope=scope,
                    native_operation="add_memories",
                    native_ids=[],
                    latency_ms=latency_ms,
                    consistency_model=ConsistencyModel.immediate,
                    native_request=add_request,
                    native_response=result,
                    warnings=[
                        "MemOS extraction completed and accepted this turn "
                        "without a memory change"
                    ],
                ),
            )

        return WriteReceipt(
            status=WriteStatus.committed,
            native_ids=native_ids,
            trace=Trace(
                backend_name=self.backend_name,
                scope=scope,
                native_operation="add_memories",
                native_ids=native_ids,
                latency_ms=latency_ms,
                consistency_model=ConsistencyModel.immediate,
                native_request=add_request,
                native_response=result,
            ),
        )

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        if request.budget.max_items is not None and request.budget.max_items <= 0:
            raise ValueError(
                "MemOS retrieval max_items must be positive when set: "
                f"value={request.budget.max_items!r}"
            )
        if request.budget.max_tokens is not None and request.budget.max_tokens <= 0:
            raise ValueError(
                "MemOS retrieval max_tokens must be positive when set: "
                f"value={request.budget.max_tokens!r}"
            )
        top_k = request.budget.max_items or self._top_k
        search_request = {
            "query": request.query,
            "user_id": request.scope.persona_id,
            "session_id": self._memory_session_id,
            "readable_cube_ids": [self._cube_id],
            "mode": self._search_mode,
            "top_k": top_k,
            "relativity": self._relativity,
            "dedup": self._dedup,
            "pref_top_k": 0,
            "tool_mem_top_k": 0,
            "skill_mem_top_k": 0,
            "include_preference": False,
            "search_tool_memory": False,
            "include_skill_memory": False,
            "chat_history": [],
        }

        started_at = time.monotonic()
        # Off the event loop: `search_memories` is a synchronous client call --
        # one container, one database, a 180 s timeout and its own transport
        # retry -- and on the loop it left every other coroutine's in-flight
        # HTTP unread for its whole length. The
        # retry runs inside the thread, unchanged.
        grouped_result = await asyncio.to_thread(
            self._client.search_memories, search_request
        )
        latency_ms = (time.monotonic() - started_at) * 1000
        self._require_text_memory_profile(grouped_result)
        bounded_result, dropped_count = self._bounded_text_memories(
            grouped_result,
            max_items=top_k,
        )
        memories = list(self._iter_memories(bounded_result))

        return RetrieveResult(
            raw=[bounded_result],
            trace=Trace(
                backend_name=self.backend_name,
                scope=request.scope,
                native_operation="search_memories",
                native_ids=[
                    str(memory.get("id") or memory.get("memory_id"))
                    for memory in memories
                ],
                latency_ms=latency_ms,
                consistency_model=ConsistencyModel.immediate,
                retrieved_count=len(memories),
                top_score=self._top_score(memories),
                native_request=search_request,
                native_response=grouped_result,
                warnings=(
                    [
                        "MemOS retrieval omitted memories that exceeded the "
                        f"item budget: dropped_count={dropped_count}"
                    ]
                    if dropped_count
                    else []
                ),
            ),
        )

    def format_context(self, result: RetrieveResult) -> str:
        if not result.raw:
            return ""
        grouped = result.raw[0]
        if not isinstance(grouped, dict):
            raise TypeError(
                "MemOS retrieve result must contain a grouped object: "
                f"value_type={type(grouped).__name__}"
            )

        return self._format_grouped_context(grouped)

    @classmethod
    def _format_grouped_context(cls, grouped: dict[str, Any]) -> str:
        sections: list[str] = []
        for group_name, label in _GROUP_LABELS.items():
            lines = [
                memory["memory"]
                for memory in cls._iter_group_memories(grouped, group_name)
            ]
            if lines:
                sections.append(f"[{label}]\n" + "\n".join(lines))
        if not sections:
            return ""
        return (
            f'<memory-context backend="{cls.backend_name}">\n'
            + "\n\n".join(sections)
            + "\n</memory-context>"
        )

    async def reset_scope(self, scope: MemoryScope) -> ResetReceipt:
        if self._adapter_mode.read_only or getattr(self._client, "read_only", False):
            raise PermissionError("Cannot reset a read-only MemOS scope")
        grouped_result = self._client.get_memories(
            {
                "mem_cube_id": self._cube_id,
                "user_id": scope.persona_id,
                "include_preference": False,
                "include_tool_memory": False,
                "include_skill_memory": False,
            }
        )
        self._require_text_memory_profile(grouped_result)
        native_ids = list(
            dict.fromkeys(
                str(memory.get("id") or memory.get("memory_id"))
                for memory in self._iter_memories(grouped_result)
            )
        )
        if native_ids:
            self._client.delete_memories(
                {
                    "writable_cube_ids": [self._cube_id],
                    "memory_ids": native_ids,
                }
            )
            remaining_result = self._client.get_memories(
                {
                    "mem_cube_id": self._cube_id,
                    "user_id": scope.persona_id,
                    "include_preference": False,
                    "include_tool_memory": False,
                    "include_skill_memory": False,
                }
            )
            self._require_text_memory_profile(remaining_result)
            remaining_ids = {
                str(memory.get("id") or memory.get("memory_id"))
                for memory in self._iter_memories(remaining_result)
            }
            undeleted_ids = [memory_id for memory_id in native_ids if memory_id in remaining_ids]
            if undeleted_ids:
                raise RuntimeError(
                    "MemOS reset verification found memories that were not deleted: "
                    f"cube_id={self._cube_id!r} undeleted_ids={undeleted_ids!r}"
                )
        return ResetReceipt(scope=scope, cleared=True, warnings=[])

    def begin_memory_side_usage(self, session_id: str) -> None:
        """A benchmark session starts: its token lines carry its step id."""
        if self._usage_reporter is not None:
            self._usage_reporter.open_session(session_id)

    def flush_memory_side_usage(self) -> None:
        """It ends: drain what is left, then print the session's totals."""
        if self._usage_reporter is not None:
            self._usage_reporter.emit_session_summary()

    async def health(self) -> BackendHealth:
        started_at = time.monotonic()
        try:
            self._client.health()
        except MemOSClientError as exc:
            return BackendHealth(
                status=HealthStatus.unavailable,
                backend_name=self.backend_name,
                latency_ms=(time.monotonic() - started_at) * 1000,
                consistency_model=ConsistencyModel.immediate,
                warnings=[str(exc)],
            )
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=(time.monotonic() - started_at) * 1000,
            consistency_model=ConsistencyModel.immediate,
            warnings=[],
        )

    @staticmethod
    def _iter_group_memories(
        grouped: dict[str, Any], group_name: str
    ) -> Any:
        buckets = grouped.get(group_name, [])
        if not isinstance(buckets, list):
            raise TypeError(
                f"MemOS {group_name} must be a list: value={buckets!r}"
            )
        for bucket in buckets:
            if not isinstance(bucket, dict) or not isinstance(
                bucket.get("memories"), list
            ):
                raise TypeError(
                    f"MemOS {group_name} bucket is invalid: value={bucket!r}"
                )
            for memory in bucket["memories"]:
                if not isinstance(memory, dict):
                    raise TypeError(
                        f"MemOS {group_name} memory is invalid: value={memory!r}"
                    )
                yield memory

    @classmethod
    def _iter_memories(cls, grouped: dict[str, Any]) -> Any:
        for group_name in _GROUP_LABELS:
            yield from cls._iter_group_memories(grouped, group_name)

    @classmethod
    def _require_text_memory_profile(cls, grouped: dict[str, Any]) -> None:
        unexpected_groups = [
            group_name
            for group_name in _ALL_GROUP_NAMES
            if group_name != "text_mem"
            and any(cls._iter_group_memories(grouped, group_name))
        ]
        pref_note = grouped.get("pref_note")
        if unexpected_groups or (
            isinstance(pref_note, str) and pref_note.strip()
        ):
            raise RuntimeError(
                "MemOS text_memory profile returned typed memory: "
                f"groups={unexpected_groups!r} "
                f"pref_note_present={bool(isinstance(pref_note, str) and pref_note.strip())!r}"
            )

    @classmethod
    def _bounded_text_memories(
        cls,
        grouped: dict[str, Any],
        *,
        max_items: int,
    ) -> tuple[dict[str, Any], int]:
        bounded = deepcopy(grouped)
        for group_name in _ALL_GROUP_NAMES:
            bounded[group_name] = []
        bounded.pop("pref_note", None)

        selected_count = 0
        candidate_count = 0
        for bucket in grouped.get("text_mem", []):
            bucket_copy = deepcopy(bucket)
            bucket_copy["memories"] = []
            for memory in cls._iter_group_memories(
                {"text_mem": [bucket]}, "text_mem"
            ):
                candidate_count += 1
                if selected_count >= max_items:
                    continue
                bucket_copy["memories"].append(deepcopy(memory))
                selected_count += 1
            if bucket_copy["memories"]:
                bucket_copy["total_nodes"] = len(bucket_copy["memories"])
                bounded["text_mem"].append(bucket_copy)

        return bounded, candidate_count - selected_count

    @staticmethod
    def _top_score(memories: list[dict[str, Any]]) -> float | None:
        scores: list[float] = []
        for index, memory in enumerate(memories):
            metadata = memory.get("metadata")
            score = (
                metadata.get("relativity")
                if isinstance(metadata, dict)
                else None
            )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError(
                    "MemOS search memory requires metadata.relativity as a "
                    "finite number between 0 and 1: "
                    f"index={index!r} value={score!r}"
                )
            scores.append(float(score))
        return float(max(scores)) if scores else None

    @staticmethod
    def _require_identifier(field_name: str, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(
                f"MemOS {field_name} must not be empty: value={value!r}"
            )
        return normalized

    def _trace(self, scope: MemoryScope, operation: str) -> Trace:
        return Trace(
            backend_name=self.backend_name,
            scope=scope,
            native_operation=operation,
            native_ids=[],
            latency_ms=0.0,
            consistency_model=ConsistencyModel.immediate,
        )
