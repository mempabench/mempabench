"""Tests for the MemOS benchmark adapter."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any

import pytest

from memory.adapter import MemoryAdapter
from memory.backends.memos_adapter import MemOSAdapter
from memory.backends.memos_client import MemOSTransportError
from memory.events import MemoryEvent
from memory.registry import default_registry
from memory.results import ConsistencyModel, WriteStatus
from memory.scope import (
    AdapterMode,
    MemoryRetrievalBudget,
    MemoryScope,
    RetrieveRequest,
)


class FakeMemOSClient:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.add_result: list[dict[str, Any]] = [
            {
                "memory": "User prefers concise updates.",
                "memory_id": "memos_1",
                "memory_type": "PreferenceMemory",
                "cube_id": "cube_001",
            }
        ]
        self.search_result: dict[str, Any] = {
            "text_mem": [
                {
                    "cube_id": "cube_001",
                    "memories": [
                        {
                            "id": "text_1",
                            "memory": "User discussed travel plans.",
                            "metadata": {"relativity": 0.90},
                        }
                    ],
                    "total_nodes": 1,
                }
            ],
            "act_mem": [],
            "para_mem": [],
            "pref_mem": [],
            "tool_mem": [],
            "skill_mem": [],
        }

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "service": "memos", "version": "1.0.1"}

    def add_memories(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        self.add_calls.append(request)
        return self.add_result

    def get_memories(self, request: dict[str, Any]) -> dict[str, Any]:
        self.get_calls.append(request)
        return self.search_result

    def search_memories(self, request: dict[str, Any]) -> dict[str, Any]:
        self.search_calls.append(request)
        return self.search_result

    def delete_memories(self, request: dict[str, Any]) -> dict[str, Any]:
        self.delete_calls.append(request)
        deleted_ids = set(request["memory_ids"])
        for group_name in ("text_mem", "pref_mem", "tool_mem", "skill_mem"):
            for bucket in self.search_result.get(group_name, []):
                bucket["memories"] = [
                    memory
                    for memory in bucket.get("memories", [])
                    if str(memory.get("id") or memory.get("memory_id")) not in deleted_ids
                ]
        return {"status": "success", "deleted_count": len(deleted_ids)}


class UnavailableMemOSClient(FakeMemOSClient):
    def health(self) -> dict[str, Any]:
        raise MemOSTransportError("connection refused")


class UnverifiedDeleteMemOSClient(FakeMemOSClient):
    def delete_memories(self, request: dict[str, Any]) -> dict[str, Any]:
        self.delete_calls.append(request)
        return {"status": "success", "deleted_count": len(set(request["memory_ids"]))}


def _scope() -> MemoryScope:
    return MemoryScope(run_id="run_001", persona_id="user_a")


def _event(*, turn_label: str = "acc_001") -> MemoryEvent:
    return MemoryEvent(
        event_id=f"run_001:{turn_label}:turn_01",
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
        session_id="run_001_main",
        turn_id=f"{turn_label}:turn_01",
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        messages=[
            {"role": "user", "content": "Book a flight to NYC"},
            {"role": "assistant", "content": "Looking into flights now."},
        ],
    )


def _request() -> RetrieveRequest:
    return RetrieveRequest(
        query="travel preferences",
        scope=_scope(),
        session_id="run_001_main",
        budget=MemoryRetrievalBudget(max_tokens=4000, max_items=8),
    )


def _adapter(client: FakeMemOSClient | None = None) -> MemOSAdapter:
    return MemOSAdapter(
        client=client or FakeMemOSClient(),
        cube_id="cube_001",
        memory_session_id="run_001_main",
    )


def test_adapter_requires_explicit_client_cube_and_session() -> None:
    with pytest.raises(TypeError, match="client"):
        MemOSAdapter()  # type: ignore[call-arg]


def test_adapter_satisfies_memory_contract() -> None:
    assert isinstance(_adapter(), MemoryAdapter)


def test_registry_requires_explicit_memos_runtime() -> None:
    with pytest.raises(TypeError, match="client"):
        default_registry().create("memos")


def test_constructor_rejects_async_or_invalid_modes() -> None:
    with pytest.raises(ValueError, match="add_mode"):
        MemOSAdapter(
            client=FakeMemOSClient(),
            cube_id="cube_001",
            memory_session_id="run_001_main",
            add_mode="async",
        )


@pytest.mark.parametrize(
    "relativity",
    [float("nan"), float("inf"), float("-inf"), -0.01, 1.01],
)
def test_constructor_rejects_invalid_relativity(relativity: float) -> None:
    with pytest.raises(ValueError, match="finite and between 0 and 1"):
        MemOSAdapter(
            client=FakeMemOSClient(),
            cube_id="cube_001",
            memory_session_id="run_001_main",
            relativity=relativity,
        )


@pytest.mark.asyncio
async def test_sync_add_uses_stable_session_without_benchmark_labels() -> None:
    client = FakeMemOSClient()
    adapter = _adapter(client)

    receipt = await adapter.record_event(_event(turn_label="acc_001"))

    assert receipt.status == WriteStatus.committed
    request = client.add_calls[0]
    assert request["async_mode"] == "sync"
    assert request["mode"] == "fine"
    assert request["session_id"] == "run_001_main"
    assert request["writable_cube_ids"] == ["cube_001"]
    assert "script_session_id" not in request["info"]
    assert "active_preference_attributes" not in request["info"]
    assert "scenario_description" not in request["info"]
    assert "interaction_context" not in request["info"]
    assert request["custom_tags"] is None
    assert request["chat_history"] == []
    assert receipt.native_ids == ["memos_1"]
    assert receipt.trace is not None
    assert receipt.trace.consistency_model == ConsistencyModel.immediate


@pytest.mark.asyncio
async def test_later_acc_uses_same_memory_session() -> None:
    client = FakeMemOSClient()
    adapter = _adapter(client)

    await adapter.record_event(_event(turn_label="acc_001"))
    await adapter.record_event(_event(turn_label="acc_002"))

    assert [call["session_id"] for call in client.add_calls] == [
        "run_001_main",
        "run_001_main",
    ]


@pytest.mark.asyncio
async def test_valid_empty_sync_add_is_accepted_without_change() -> None:
    client = FakeMemOSClient()
    client.add_result = []

    receipt = await _adapter(client).record_event(_event())

    assert receipt.status == WriteStatus.accepted_no_change
    assert receipt.native_ids == []
    assert receipt.trace is not None
    assert receipt.trace.warnings == [
        "MemOS extraction completed and accepted this turn without a memory change"
    ]


@pytest.mark.asyncio
async def test_read_only_mode_does_not_call_add() -> None:
    client = FakeMemOSClient()
    adapter = _adapter(client)
    await adapter.set_mode(AdapterMode(read_only=True, reason="probe"))

    receipt = await adapter.record_event(_event())

    assert receipt.status == WriteStatus.skipped_read_only
    assert client.add_calls == []


@pytest.mark.asyncio
async def test_search_uses_stable_session_without_control_filters() -> None:
    client = FakeMemOSClient()
    adapter = _adapter(client)

    await adapter.retrieve(_request())

    request = client.search_calls[0]
    assert request["session_id"] == "run_001_main"
    assert request["readable_cube_ids"] == ["cube_001"]
    assert request["query"] == "travel preferences"
    assert request["relativity"] == 0.45
    assert request["pref_top_k"] == 0
    assert request["tool_mem_top_k"] == 0
    assert request["skill_mem_top_k"] == 0
    assert request["include_preference"] is False
    assert request["search_tool_memory"] is False
    assert request["include_skill_memory"] is False
    assert request["chat_history"] == []
    assert "filter" not in request
    assert "active_preference_attributes" not in request
    assert "scenario_description" not in request
    assert "interaction_context" not in request


@pytest.mark.asyncio
async def test_retrieve_counts_nested_memories_and_native_ids() -> None:
    result = await _adapter().retrieve(_request())

    assert result.trace is not None
    assert result.trace.retrieved_count == 1
    assert result.trace.native_ids == ["text_1"]
    assert result.trace.top_score == 0.90


@pytest.mark.asyncio
async def test_format_context_reads_nested_cube_buckets() -> None:
    adapter = _adapter()
    result = await adapter.retrieve(_request())

    context = adapter.format_context(result)

    assert 'backend="memos"' in context
    assert "[Text Memories]" in context
    assert "User discussed travel plans." in context


@pytest.mark.asyncio
async def test_retrieve_leaves_final_token_budget_to_agent_loop() -> None:
    client = FakeMemOSClient()
    client.search_result["text_mem"][0]["memories"] = [
        {
            "id": "text_1",
            "memory": "short memory",
            "metadata": {"relativity": 0.9},
        },
        {
            "id": "text_2",
            "memory": "very long memory " * 100,
            "metadata": {"relativity": 0.8},
        },
    ]
    request = RetrieveRequest(
        query="travel preferences",
        scope=_scope(),
        session_id="run_001_main",
        budget=MemoryRetrievalBudget(max_tokens=30, max_items=8),
    )

    adapter = _adapter(client)
    result = await adapter.retrieve(request)
    context = adapter.format_context(result)

    assert "short memory" in context
    assert "very long memory" in context
    assert result.trace is not None
    assert result.trace.native_ids == ["text_1", "text_2"]
    assert result.trace.retrieved_count == 2
    assert result.trace.token_count is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "score",
    [None, "bad", float("nan"), float("inf"), -0.1, 1.1],
)
async def test_retrieve_rejects_invalid_relativity_from_injected_client(
    score: object,
) -> None:
    client = FakeMemOSClient()
    memory = client.search_result["text_mem"][0]["memories"][0]
    if score is None:
        memory.pop("metadata", None)
    else:
        memory["metadata"] = {"relativity": score}

    with pytest.raises(ValueError, match="metadata.relativity"):
        await _adapter(client).retrieve(_request())


@pytest.mark.asyncio
async def test_text_memory_profile_rejects_returned_typed_memory() -> None:
    client = FakeMemOSClient()
    client.search_result["pref_mem"] = [
        {
            "cube_id": "cube_001",
            "memories": [
                {
                    "id": "pref_1",
                    "memory": "User prefers formal communication.",
                }
            ],
            "total_nodes": 1,
        }
    ]

    with pytest.raises(RuntimeError, match="text_memory profile.*pref_mem"):
        await _adapter(client).retrieve(_request())


@pytest.mark.asyncio
async def test_reset_scope_deletes_only_ids_found_in_run_cube() -> None:
    client = FakeMemOSClient()
    receipt = await _adapter(client).reset_scope(_scope())

    assert receipt.cleared is True
    assert client.get_calls == [
        {
            "mem_cube_id": "cube_001",
            "user_id": "user_a",
            "include_preference": False,
            "include_tool_memory": False,
            "include_skill_memory": False,
        },
        {
            "mem_cube_id": "cube_001",
            "user_id": "user_a",
            "include_preference": False,
            "include_tool_memory": False,
            "include_skill_memory": False,
        },
    ]
    assert client.delete_calls == [
        {
            "writable_cube_ids": ["cube_001"],
            "memory_ids": ["text_1"],
        }
    ]


@pytest.mark.asyncio
async def test_reset_empty_scope_does_not_issue_ambiguous_delete() -> None:
    client = FakeMemOSClient()
    client.search_result = {
        "text_mem": [],
        "act_mem": [],
        "para_mem": [],
        "pref_mem": [],
        "tool_mem": [],
        "skill_mem": [],
    }

    receipt = await _adapter(client).reset_scope(_scope())

    assert receipt.cleared is True
    assert client.delete_calls == []


@pytest.mark.asyncio
async def test_reset_fails_when_deleted_ids_are_still_returned() -> None:
    client = UnverifiedDeleteMemOSClient()

    with pytest.raises(RuntimeError, match="undeleted_ids=.*text_1"):
        await _adapter(client).reset_scope(_scope())


@pytest.mark.asyncio
async def test_health_contacts_real_client() -> None:
    health = await _adapter(FakeMemOSClient()).health()
    assert health.status.value == "ok"
    assert health.consistency_model == ConsistencyModel.immediate


@pytest.mark.asyncio
async def test_health_reports_known_transport_failure() -> None:
    health = await _adapter(UnavailableMemOSClient()).health()
    assert health.status.value == "unavailable"
    assert health.warnings == ["connection refused"]


#: How long a gated search waits for the loop to release it. Reached only when
#: the wrap is gone, where it is the test's red instead of a hang.
_GATE_S = 0.5


class GatedSearchMemOSClient(FakeMemOSClient):
    """A client whose `search_memories` blocks until the loop releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.released = threading.Event()
        self.order: list[str] = []

    def search_memories(self, request: dict[str, Any]) -> dict[str, Any]:
        self.entered.set()
        self.released.wait(timeout=_GATE_S)
        self.order.append("the search returned")
        return super().search_memories(request)


@pytest.mark.asyncio
async def test_a_blocking_search_leaves_the_event_loop_running() -> None:
    """`search_memories` is a synchronous client call inside an `async def` --
    one container, one database, a 180 s timeout -- so on the loop it left
    every other coroutine's in-flight HTTP unread for its whole length, which
    with more than one session at a time is another session's request budget.
    It goes out through
    `asyncio.to_thread`, so a sibling coroutine gets to run while the search is
    out, and it is the sibling that releases it: the order is the assertion and
    the clock is not. The client's own transport retry runs inside the thread.
    """
    client = GatedSearchMemOSClient()
    adapter = _adapter(client)

    async def _sibling() -> None:
        while not client.entered.is_set():
            await asyncio.sleep(0.005)
        client.order.append("the loop kept running")
        client.released.set()

    await asyncio.gather(adapter.retrieve(_request()), _sibling())

    assert client.order == ["the loop kept running", "the search returned"]
    assert len(client.search_calls) == 1
