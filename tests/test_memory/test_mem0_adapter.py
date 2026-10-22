"""Tests for the mem0 adapter using a fake client."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any

import pytest

from memory.adapter import MemoryAdapter
from memory.backends.mem0_adapter import Mem0Adapter
from memory.events import MemoryEvent
from memory.registry import default_registry
from memory.results import WriteStatus
from memory.scope import (
    AdapterMode,
    MemoryRetrievalBudget,
    MemoryScope,
    RetrieveRequest,
)


# ── Fake mem0 client ───────────────────────────────────────────────

class FakeMem0Client:
    """Records calls for assertion without real mem0 infra."""

    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.delete_all_calls: list[dict[str, Any]] = []
        self._store: list[dict[str, Any]] = []

    def add(
        self,
        messages: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> dict[str, Any]:
        call = {
            "messages": messages,
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": metadata,
            "infer": infer,
        }
        self.add_calls.append(call)
        source_messages = (
            [message for message in messages if message.get("role") != "system"]
            if not infer
            else [{"content": str(messages)}]
        )
        records = []
        for message in source_messages:
            record = {
                "id": f"mem_{len(self._store)}",
                "memory": message["content"],
                "event": "ADD",
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "metadata": metadata,
            }
            if not infer:
                record["role"] = message["role"]
            self._store.append(record)
            records.append(record)
        return {"results": records}

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        threshold: float = 0.1,
        rerank: bool = False,
    ) -> dict[str, Any]:
        call = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "threshold": threshold,
            "rerank": rerank,
        }
        self.search_calls.append(call)
        results = []
        for item in self._store:
            if filters:
                if filters.get("user_id") and item.get("user_id") != filters["user_id"]:
                    continue
                if filters.get("agent_id") and item.get("agent_id") != filters["agent_id"]:
                    continue
                if filters.get("run_id") and item.get("run_id") != filters["run_id"]:
                    continue
            result = {
                "id": item["id"],
                "memory": item["memory"],
                "score": 0.95,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "metadata": item.get("metadata"),
            }
            if "role" in item:
                result["role"] = item["role"]
            results.append(result)
        return {"results": results[:top_k]}

    def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self.delete_all_calls.append({
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
        })
        self._store = [
            s for s in self._store
            if not (
                (user_id is None or s.get("user_id") == user_id)
                and (run_id is None or s.get("run_id") == run_id)
            )
        ]
        return {"message": "ok"}

    def health(self) -> float:
        return 12.5


# ── Helpers ────────────────────────────────────────────────────────

def _scope() -> MemoryScope:
    return MemoryScope(run_id="run_001", persona_id="user_a")


def _runtime_agent_scope() -> MemoryScope:
    return MemoryScope(
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
    )


def _event() -> MemoryEvent:
    return MemoryEvent(
        event_id="evt_1",
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
        session_id="s01",
        turn_id="t01",
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        messages=[
            {"role": "user", "content": "I prefer formal tone"},
            {"role": "assistant", "content": "Understood."},
        ],
    )


def _request() -> RetrieveRequest:
    return RetrieveRequest(
        query="tone preference",
        scope=_scope(),
        session_id="s01",
        budget=MemoryRetrievalBudget(max_tokens=4000, max_items=5),
    )


# ── Tests ──────────────────────────────────────────────────────────

class TestMem0AdapterProtocol:
    def test_is_memory_adapter(self) -> None:
        adapter = Mem0Adapter(client=FakeMem0Client())
        assert isinstance(adapter, MemoryAdapter)

    def test_backend_name(self) -> None:
        adapter = Mem0Adapter(client=FakeMem0Client())
        assert adapter.backend_name == "mem0"

class TestMem0AdapterRecord:
    @pytest.fixture()
    def client(self) -> FakeMem0Client:
        return FakeMem0Client()

    @pytest.fixture()
    def adapter(self, client: FakeMem0Client) -> Mem0Adapter:
        return Mem0Adapter(client=client)

    @pytest.mark.asyncio
    async def test_record_event_calls_add_with_infer_true(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_scope())
        receipt = await adapter.record_event(_event())
        assert receipt.status == WriteStatus.committed
        assert len(client.add_calls) == 1
        call = client.add_calls[0]
        assert call["infer"] is True
        assert call["user_id"] == "user_a"
        assert call["run_id"] == "run_001"

    @pytest.mark.asyncio
    async def test_record_event_infer_false(self, client: FakeMem0Client) -> None:
        adapter = Mem0Adapter(client=client, infer=False)
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        assert client.add_calls[0]["infer"] is False

    @pytest.mark.asyncio
    async def test_metadata_contains_operation_identity_only(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        meta = client.add_calls[0]["metadata"]
        assert meta["session_id"] == "s01"
        assert meta["turn_id"] == "t01"
        assert "timestamp" in meta
        assert "scenario_description" not in meta
        assert "interaction_context" not in meta
        assert "active_preference_attributes" not in meta

    @pytest.mark.asyncio
    async def test_record_event_returns_native_ids(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_scope())
        receipt = await adapter.record_event(_event())
        assert len(receipt.native_ids) > 0
        assert receipt.trace is not None
        assert receipt.trace.native_operation == "add"

    @pytest.mark.asyncio
    async def test_successful_inference_with_no_memory_is_accepted_no_change(
        self,
    ) -> None:
        class EmptyAddClient(FakeMem0Client):
            def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
                return {"results": []}

        adapter = Mem0Adapter(client=EmptyAddClient())
        receipt = await adapter.record_event(_event())

        assert receipt.status == WriteStatus.accepted_no_change
        assert receipt.trace is not None
        assert receipt.trace.warnings == [
            "Mem0 extraction completed and accepted this turn without a memory change"
        ]

    @pytest.mark.asyncio
    async def test_simple_rag_empty_write_is_a_failure(self) -> None:
        class EmptyAddClient(FakeMem0Client):
            def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
                return {"results": []}

        adapter = Mem0Adapter(client=EmptyAddClient(), infer=False)

        with pytest.raises(
            RuntimeError,
            match="Simple RAG.*run_id='run_001'.*session_id='s01'",
        ):
            await adapter.record_event(_event())

    @pytest.mark.asyncio
    async def test_record_event_preserves_native_add_signal(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_scope())
        receipt = await adapter.record_event(_event())
        assert receipt.native_events == ["ADD"]
        assert len(receipt.native_events) == len(receipt.native_ids)
        assert receipt.trace is not None
        assert receipt.trace.native_events == ["ADD"]

    @pytest.mark.asyncio
    async def test_record_event_rejects_missing_native_id(self) -> None:
        class MissingIdClient(FakeMem0Client):
            def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
                return {
                    "results": [
                        {"id": "", "memory": "remember me", "event": "ADD"}
                    ]
                }

        adapter = Mem0Adapter(client=MissingIdClient())

        with pytest.raises(ValueError, match="non-empty string id.*value=''"):
            await adapter.record_event(_event())

    @pytest.mark.asyncio
    async def test_read_only_skips_write(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.set_mode(AdapterMode(read_only=True, reason="test_session"))
        receipt = await adapter.record_event(_event())
        assert receipt.status == WriteStatus.skipped_read_only
        assert len(client.add_calls) == 0


class TestMem0AdapterRetrieve:
    @pytest.fixture()
    def client(self) -> FakeMem0Client:
        return FakeMem0Client()

    @pytest.fixture()
    def adapter(self, client: FakeMem0Client) -> Mem0Adapter:
        return Mem0Adapter(client=client)

    @pytest.mark.asyncio
    async def test_retrieve_calls_search_with_filters(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        await adapter.retrieve(_request())
        assert len(client.search_calls) == 1
        call = client.search_calls[0]
        assert call["filters"]["user_id"] == "user_a"
        assert call["filters"]["run_id"] == "run_001"

    @pytest.mark.asyncio
    async def test_retrieve_does_not_filter_by_hidden_benchmark_labels(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_scope())
        await adapter.retrieve(_request())
        filters = client.search_calls[0]["filters"]
        assert filters == {"user_id": "user_a", "run_id": "run_001"}

    @pytest.mark.asyncio
    async def test_runtime_agent_does_not_change_unconfigured_backend_scope(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_runtime_agent_scope())
        await adapter.record_event(_event())
        await adapter.retrieve(
            RetrieveRequest(
                query="tone preference",
                scope=_runtime_agent_scope(),
                session_id="s01",
                budget=MemoryRetrievalBudget(max_items=5),
            )
        )

        assert client.add_calls[0]["agent_id"] is None
        assert client.search_calls[0]["filters"] == {
            "user_id": "user_a",
            "run_id": "run_001",
        }

    @pytest.mark.asyncio
    async def test_explicit_agent_scope_is_shared_by_add_and_search(
        self, client: FakeMem0Client
    ) -> None:
        adapter = Mem0Adapter(client=client, agent_id="nanobot")
        await adapter.setup_scope(_runtime_agent_scope())
        await adapter.record_event(_event())
        result = await adapter.retrieve(
            RetrieveRequest(
                query="tone preference",
                scope=_runtime_agent_scope(),
                session_id="s01",
                budget=MemoryRetrievalBudget(max_items=5),
            )
        )

        assert client.add_calls[0]["agent_id"] == "nanobot"
        assert client.search_calls[0]["filters"] == {
            "user_id": "user_a",
            "run_id": "run_001",
            "agent_id": "nanobot",
        }
        assert len(result.raw) == 1

    @pytest.mark.asyncio
    async def test_retrieve_passes_top_k_from_budget(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_scope())
        await adapter.retrieve(_request())
        assert client.search_calls[0]["top_k"] == 5

    @pytest.mark.asyncio
    async def test_retrieve_preserves_raw_results(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        result = await adapter.retrieve(_request())
        assert len(result.raw) == 1
        assert "memory" in result.raw[0]
        assert "score" in result.raw[0]
        assert "id" in result.raw[0]

    @pytest.mark.asyncio
    async def test_retrieve_trace_has_score(
        self, adapter: Mem0Adapter, client: FakeMem0Client
    ) -> None:
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        result = await adapter.retrieve(_request())
        assert result.trace is not None
        assert result.trace.top_score == 0.95
        assert result.trace.retrieved_count == 1


class TestMem0AdapterFormatContext:
    @pytest.mark.asyncio
    async def test_format_context_with_results(self) -> None:
        client = FakeMem0Client()
        adapter = Mem0Adapter(client=client)
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        result = await adapter.retrieve(_request())
        ctx = adapter.format_context(result)
        assert 'backend="mem0"' in ctx
        assert ctx.startswith("<memory-context")

    @pytest.mark.asyncio
    async def test_simple_rag_context_preserves_message_identity(self) -> None:
        client = FakeMem0Client()
        adapter = Mem0Adapter(client=client, infer=False)
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        result = await adapter.retrieve(_request())

        context = adapter.format_context(result)

        assert 'backend="simple_rag"' in context
        assert '"role": "user"' in context
        assert '"role": "assistant"' in context
        assert '"session_id": "s01"' in context
        assert '"turn_id": "t01"' in context
        assert "I prefer formal tone" in context
        assert "Understood." in context

    @pytest.mark.asyncio
    async def test_simple_rag_rejects_search_result_without_role(self) -> None:
        class MissingRoleClient(FakeMem0Client):
            def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {
                    "results": [
                        {
                            "id": "memory-1",
                            "memory": "hello",
                            "score": 0.9,
                            "metadata": {
                                "session_id": "s01",
                                "turn_id": "t01",
                            },
                        }
                    ]
                }

        adapter = Mem0Adapter(client=MissingRoleClient(), infer=False)

        with pytest.raises(ValueError, match="requires a user or assistant role"):
            await adapter.retrieve(_request())

    @pytest.mark.asyncio
    async def test_format_context_empty(self) -> None:
        adapter = Mem0Adapter(client=FakeMem0Client())
        from memory.results import RetrieveResult
        assert adapter.format_context(RetrieveResult(raw=[])) == ""


class TestMem0AdapterLifecycle:
    @pytest.mark.asyncio
    async def test_reset_scope_calls_delete_all(self) -> None:
        client = FakeMem0Client()
        adapter = Mem0Adapter(client=client)
        reset = await adapter.reset_scope(_scope())
        assert reset.cleared is True
        assert len(client.delete_all_calls) == 1
        assert client.delete_all_calls[0]["user_id"] == "user_a"
        assert client.delete_all_calls[0]["run_id"] == "run_001"

    @pytest.mark.asyncio
    async def test_reset_ignores_runtime_agent_when_not_explicitly_configured(
        self,
    ) -> None:
        client = FakeMem0Client()
        adapter = Mem0Adapter(client=client)

        await adapter.reset_scope(_runtime_agent_scope())

        assert client.delete_all_calls[0]["agent_id"] is None

    @pytest.mark.asyncio
    async def test_health(self) -> None:
        adapter = Mem0Adapter(client=FakeMem0Client())
        h = await adapter.health()
        assert h.status.value == "ok"
        assert h.backend_name == "mem0"
        assert h.latency_ms == 12.5

    @pytest.mark.asyncio
    async def test_set_mode(self) -> None:
        adapter = Mem0Adapter(client=FakeMem0Client())
        r = await adapter.set_mode(AdapterMode(read_only=True, reason="test"))
        assert r.mode.read_only is True
        assert r.previous_mode is not None
        assert r.previous_mode.read_only is False


class TestMem0Registry:
    def test_mem0_in_default_registry(self) -> None:
        reg = default_registry()
        assert "mem0" in reg
        assert reg.adapter_kind("mem0") == "backend"

    def test_simple_rag_in_default_registry(self) -> None:
        reg = default_registry()
        assert "simple_rag" in reg
        assert reg.adapter_kind("simple_rag") == "condition"

    def test_create_mem0_with_fake_client(self) -> None:
        reg = default_registry()
        adapter = reg.create("mem0", client=FakeMem0Client())
        assert isinstance(adapter, MemoryAdapter)
        assert adapter.backend_name == "mem0"

    def test_mem0_requires_explicit_client(self) -> None:
        reg = default_registry()
        with pytest.raises(TypeError, match="client"):
            reg.create("mem0")

    def test_simple_rag_requires_explicit_client(self) -> None:
        reg = default_registry()
        with pytest.raises(TypeError, match="client"):
            reg.create("simple_rag")

    @pytest.mark.asyncio
    async def test_simple_rag_uses_infer_false(self) -> None:
        client = FakeMem0Client()
        reg = default_registry()
        adapter = reg.create("simple_rag", client=client)
        assert isinstance(adapter, MemoryAdapter)
        assert adapter.backend_name == "simple_rag"
        await adapter.setup_scope(_scope())
        await adapter.record_event(_event())
        assert client.add_calls[0]["infer"] is False


# ── Chunked writes: a turn pool against mem0's embedding input cap ──
#
# The 2026-09-14 `mem0` smoke died on the first
# adapter write of `acc_001` with a 400 from the embeddings endpoint: on the
# `infer=True` path mem0 embeds the whole pool as ONE input
# (`mem0/memory/main.py:822-825`) and an ACC turn pool runs ~10k tokens
# against `text-embedding-3-small`'s cap. The adapter now splits the pool at
# MESSAGE boundaries into consecutive chunks that each fit, and calls `add`
# once per chunk. mem0 still sees every message, in order, and still infers.
#
# The token counts below are taken independently of the adapter — mem0's own
# `parse_messages` plus the embedder's newline flattening, tokenized with the
# embedding model's own encoding — so these assertions do not lean on the
# implementation's arithmetic.

import tiktoken
from mem0.memory.utils import parse_messages

from memory.backends.mem0_adapter import MEM0_EMBEDDING_MAX_INPUT_TOKENS
from memory.backends.mem0_client import Mem0PersistenceError
from memory.backends.runtime_contracts import MEM0_OPENAI_EMBEDDING_MODEL
from test_memory.pool_fixtures import (
    acc_sized_turn_pool,
    full_experience_pool,
    pool_event,
)


def _embed_input_tokens(messages: list[dict[str, Any]]) -> int:
    """What mem0 hands the embeddings endpoint for these messages."""
    encoding = tiktoken.encoding_for_model(MEM0_OPENAI_EMBEDDING_MODEL)
    return len(encoding.encode(parse_messages(list(messages)).replace("\n", " ")))


@pytest.mark.asyncio
async def test_an_acc_sized_pool_is_chunked_into_inputs_mem0_can_embed() -> None:
    pool = acc_sized_turn_pool()
    assert _embed_input_tokens(pool.messages) > MEM0_EMBEDDING_MAX_INPUT_TOKENS
    client = FakeMem0Client()
    adapter = Mem0Adapter(client=client)
    await adapter.setup_scope(_scope())

    receipt = await adapter.record_event(pool_event(pool))

    chunks = [call["messages"] for call in client.add_calls]
    assert len(chunks) > 1
    assert receipt.chunk_count == len(chunks)
    for chunk in chunks:
        assert chunk, "a chunk must carry at least one message"
        assert _embed_input_tokens(chunk) <= MEM0_EMBEDDING_MAX_INPUT_TOKENS
    # Consecutive, in order, and no message split across a boundary.
    assert [message for chunk in chunks for message in chunk] == pool.messages
    # Identical scope and metadata on every chunk.
    first = client.add_calls[0]
    for call in client.add_calls[1:]:
        assert {k: v for k, v in call.items() if k != "messages"} == {
            k: v for k, v in first.items() if k != "messages"
        }
    assert first["metadata"] == {
        "session_id": "s01",
        "turn_id": "t01",
        "timestamp": "2026-03-03T09:00:00+00:00",
    }
    # The receipt aggregates every chunk's ids, not just the last chunk's.
    assert receipt.status == WriteStatus.committed
    assert receipt.native_ids == [f"mem_{index}" for index in range(len(chunks))]


@pytest.mark.asyncio
async def test_a_single_message_over_the_cap_raises_instead_of_truncating() -> None:
    client = FakeMem0Client()
    adapter = Mem0Adapter(client=client, extraction_chunk_strategy="legacy_no_message_split")
    await adapter.setup_scope(_scope())
    oversized = "pointer states under non-Markovian noise. " * 2000
    assert _embed_input_tokens([{"role": "assistant", "content": oversized}]) > (
        MEM0_EMBEDDING_MAX_INPUT_TOKENS
    )
    event = MemoryEvent(
        event_id="evt_1",
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
        session_id="s01",
        turn_id="acc_001:b1:t0",
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        messages=[
            {"role": "user", "content": "Summarise the decoherence chapter."},
            {"role": "assistant", "content": f"[tool_result] {oversized}"},
        ],
    )

    with pytest.raises(Mem0PersistenceError) as excinfo:
        await adapter.record_event(event)

    message = str(excinfo.value)
    assert "turn_id='acc_001:b1:t0'" in message
    assert "message_index=1" in message
    assert "tokens=" in message
    assert client.add_calls == [], "nothing may be written, truncated or skipped"


@pytest.mark.asyncio
async def test_a_pool_under_the_cap_is_written_with_exactly_one_add() -> None:
    pool = full_experience_pool()
    assert _embed_input_tokens(pool.messages) <= MEM0_EMBEDDING_MAX_INPUT_TOKENS
    client = FakeMem0Client()
    adapter = Mem0Adapter(client=client)
    await adapter.setup_scope(_scope())

    receipt = await adapter.record_event(pool_event(pool))

    assert len(client.add_calls) == 1
    assert client.add_calls[0]["messages"] == pool.messages
    assert receipt.chunk_count == 1


# ── The search is off the event loop ───────────────────────────────

#: How long a gated search waits for the loop to release it. Reached only when
#: the wrap is gone, where it is the test's red instead of a hang.
_GATE_S = 0.5


class GatedSearchMem0Client(FakeMem0Client):
    """A client whose `search` blocks until the event loop releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.released = threading.Event()
        self.order: list[str] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.entered.set()
        self.released.wait(timeout=_GATE_S)
        self.order.append("the search returned")
        return super().search(**kwargs)


@pytest.mark.asyncio
async def test_a_blocking_search_leaves_the_event_loop_running() -> None:
    """`search` is a synchronous client call inside an `async def`, so on the
    loop it left every other coroutine's in-flight HTTP unread for its whole
    length -- and with more than one session at a time that is another
    session's request budget. It goes
    out through `asyncio.to_thread`, so a sibling coroutine gets to run while
    the search is out, and it is the sibling that releases it: the order is the
    assertion and the clock is not.
    """
    client = GatedSearchMem0Client()
    adapter = Mem0Adapter(client=client)
    await adapter.setup_scope(_scope())

    async def _sibling() -> None:
        while not client.entered.is_set():
            await asyncio.sleep(0.005)
        client.order.append("the loop kept running")
        client.released.set()

    await asyncio.gather(adapter.retrieve(_request()), _sibling())

    assert client.order == ["the loop kept running", "the search returned"]
    assert len(client.search_calls) == 1


# ── simple_rag: a pool MESSAGE against the same embedding input cap ─
#
# `infer=False` stores one raw record per message and embeds each message's
# own content (`mem0/memory/main.py:722-723`), not the `parse_messages`
# concatenation, so message-boundary chunking never applied to it and a single
# over-cap message was the one thing it could not write. That killed
# `ub_simple_rag_glm_5_3_flash_20260918` twice on 2026-09-18 — at
# `acc_014:beat_clarify_task:t1` (one 8291-token assistant message: an 8180-
# token `[reasoning]` block plus three `[tool_call]` blocks) and at
# `acc_017:request_dance_spot:t0` (8567 tokens: a 4849-token `[reasoning]`
# block plus 43 deduplicated `[ix_selection]` blocks).
#
# A naive RAG baseline's answer to a document that does not fit is to chunk
# it, so an over-budget message becomes consecutive records of the same role
# whose contents concatenate back to the original byte for byte. The token
# counts below are measured the way mem0's `infer=False` path measures — the
# content with the embedder's newline flattening, tokenized with the embedding
# model's own encoding — so they do not lean on the implementation.

from memory.backends.mem0_adapter import (
    MEM0_EMBEDDING_CHUNK_SAFETY_MARGIN_TOKENS,
    _pieces_that_fit_the_embedding_input,
)
from test_memory.pool_fixtures import (
    oversized_reasoning_turn_pool,
    oversized_selection_turn_pool,
)

_RAW_BUDGET = MEM0_EMBEDDING_MAX_INPUT_TOKENS - MEM0_EMBEDDING_CHUNK_SAFETY_MARGIN_TOKENS


def _raw_embed_input_tokens(content: str) -> int:
    """What mem0 hands the embeddings endpoint for one raw record."""
    encoding = tiktoken.encoding_for_model(MEM0_OPENAI_EMBEDDING_MODEL)
    return len(encoding.encode(content.replace("\n", " ")))


async def _simple_rag_written_messages(
    pool: Any,
) -> list[dict[str, Any]]:
    client = FakeMem0Client()
    adapter = Mem0Adapter(client=client, infer=False)
    await adapter.setup_scope(_scope())
    await adapter.record_event(pool_event(pool))
    assert len(client.add_calls) == 1, "simple_rag writes one add per turn"
    assert client.add_calls[0]["infer"] is False
    return client.add_calls[0]["messages"]


@pytest.mark.asyncio
async def test_simple_rag_cuts_an_over_cap_message_into_records_that_fit() -> None:
    pool = oversized_reasoning_turn_pool()
    oversized = [
        index
        for index, message in enumerate(pool.messages)
        if _raw_embed_input_tokens(message["content"]) > _RAW_BUDGET
    ]
    assert oversized == [1], "the fixture must carry exactly one over-cap message"

    written = await _simple_rag_written_messages(pool)

    assert len(written) > len(pool.messages)
    for message in written:
        assert _raw_embed_input_tokens(message["content"]) <= _RAW_BUDGET
        assert message["content"].strip(), "an empty record cannot be written"
    # Lossless and in order: the records of each pool message concatenate back
    # to that message's content, byte for byte, and no other message moved.
    assert [message["role"] for message in written] == (
        ["user"] + ["assistant"] * (len(written) - 1)
    )
    assert "".join(
        message["content"] for message in written[1:-3]
    ) == pool.messages[1]["content"]
    assert written[0] == pool.messages[0]
    assert written[-3:] == pool.messages[2:]


@pytest.mark.asyncio
async def test_simple_rag_cuts_on_a_block_seam_when_one_fits() -> None:
    pool = oversized_selection_turn_pool()
    assert _raw_embed_input_tokens(pool.messages[1]["content"]) > _RAW_BUDGET

    written = await _simple_rag_written_messages(pool)

    # Message 1 is the oversized one; two pool messages follow it.
    pieces = [message["content"] for message in written[1:-2]]
    assert len(pieces) > 1
    assert written[0] == pool.messages[0]
    assert written[-2:] == pool.messages[2:]
    assert "".join(pieces) == pool.messages[1]["content"]
    for piece in pieces[:-1]:
        assert piece.endswith("]\n"), f"cut did not land on a block seam: {piece[-40:]!r}"


@pytest.mark.asyncio
async def test_simple_rag_passes_a_pool_that_fits_through_untouched() -> None:
    pool = full_experience_pool()
    for message in pool.messages:
        assert _raw_embed_input_tokens(message["content"]) <= _RAW_BUDGET

    written = await _simple_rag_written_messages(pool)

    assert written == pool.messages


@pytest.mark.asyncio
async def test_simple_rag_cuts_inside_a_single_over_cap_line() -> None:
    """A one-line record that does not fit is still written, in full."""
    client = FakeMem0Client()
    adapter = Mem0Adapter(client=client, infer=False)
    await adapter.setup_scope(_scope())
    single_line = '{"rows":[' + ",".join(f'{{"sku":{n},"qty":2}}' for n in range(2200)) + "]}"
    assert "\n" not in single_line
    assert _raw_embed_input_tokens(single_line) > _RAW_BUDGET
    event = MemoryEvent(
        event_id="evt_line",
        run_id="run_001",
        persona_id="user_b",
        agent_id="nanobot",
        session_id="s01",
        turn_id="acc_014:beat_clarify_task:t1",
        timestamp=datetime(2026, 10, 5, 14, 1, tzinfo=timezone.utc),
        messages=[{"role": "assistant", "content": single_line}],
    )

    await adapter.record_event(event)

    pieces = [message["content"] for message in client.add_calls[0]["messages"]]
    assert len(pieces) > 1
    assert "".join(pieces) == single_line
    for piece in pieces:
        assert _raw_embed_input_tokens(piece) <= _RAW_BUDGET


# ── Boundary checks: measured pieces, no whitespace-only record ──
#
# Two risks the review of `3c451e0b` named, both reproduced against that
# commit before this one:
#
#   * `_longest_fitting_prefix` bisected on the assumption that a prefix's
#     token count grows with its length. No tokenizer promises that, and the
#     fallback the assumption justified — `... or 1` — handed back a
#     one-character piece nothing had measured.
#   * A pool message can open on blank lines and end on a whitespace run, and
#     the coarsest fitting seam inside such a run cut a record that was
#     nothing but whitespace: `"\n\n\n"` ahead of one over-cap line, and a
#     2000-space tail behind a 440-line body. mem0 stores that record and
#     `_validate_add_response` then refuses it as empty memory text, so the
#     turn still could not be written.


def _one_over_cap_line() -> str:
    """One line, no newline in it, that does not fit the embedding input."""
    line = "".join(f'{{"sku":{index},"qty":2}},' for index in range(2200))
    assert "\n" not in line
    assert _raw_embed_input_tokens(line) > _RAW_BUDGET
    return line


def _lines_just_under_the_budget() -> str:
    """As many whole lines as fit, so the next cut lands on the last one."""
    row = '{"row":%d,"note":"two cases of lemons for the weekend brunch service"}'
    lines: list[str] = []
    while True:
        candidate = lines + [row % len(lines)]
        if _raw_embed_input_tokens("\n".join(candidate) + "\n") > _RAW_BUDGET:
            return "\n".join(lines) + "\n"
        lines = candidate


async def _simple_rag_records(content: str) -> list[dict[str, Any]]:
    client = FakeMem0Client()
    adapter = Mem0Adapter(client=client, infer=False)
    await adapter.setup_scope(_scope())
    event = MemoryEvent(
        event_id="evt_boundary",
        run_id="run_001",
        persona_id="user_b",
        agent_id="nanobot",
        session_id="s01",
        turn_id="acc_014:beat_clarify_task:t1",
        timestamp=datetime(2026, 9, 18, 4, 51, tzinfo=timezone.utc),
        messages=[{"role": "assistant", "content": content}],
    )
    await adapter.record_event(event)
    return client.add_calls[0]["messages"]


@pytest.mark.asyncio
async def test_simple_rag_writes_no_whitespace_only_record_for_leading_blank_lines() -> None:
    content = "\n\n\n" + _one_over_cap_line()

    written = await _simple_rag_records(content)

    assert len(written) > 1
    for record in written:
        assert record["content"].strip(), "mem0 refuses empty memory text"
        assert _raw_embed_input_tokens(record["content"]) <= _RAW_BUDGET
    # The blank lines are kept, on the piece the boundary moved them to.
    assert written[0]["content"].startswith("\n\n\n")
    assert "".join(record["content"] for record in written) == content


@pytest.mark.asyncio
async def test_simple_rag_writes_no_whitespace_only_record_for_a_trailing_run() -> None:
    content = _lines_just_under_the_budget() + " " * 2000
    assert _raw_embed_input_tokens(content) > _RAW_BUDGET

    written = await _simple_rag_records(content)

    assert len(written) > 1
    for record in written:
        assert record["content"].strip(), "mem0 refuses empty memory text"
        assert _raw_embed_input_tokens(record["content"]) <= _RAW_BUDGET
    # Every one of the 2000 spaces still reaches mem0.
    assert written[-1]["content"].endswith(" " * 2000)
    assert "".join(record["content"] for record in written) == content


def test_every_piece_is_measured_rather_than_assumed_monotone() -> None:
    """A tokenizer whose prefix counts spike still yields measured pieces.

    `measure` here reports ten times the length for a middle band of prefix
    lengths, which is exactly what the bisection's monotonicity assumption
    does not allow: its first probes land in the band and fail. The pieces it
    settles on must each still satisfy the same `measure` inside the budget,
    and the cut must advance every time.
    """
    def spiky_measure(text: str) -> int:
        return len(text) * 10 if 10 < len(text) < 50 else len(text)

    content = "".join(f"line {index}\n" for index in range(40))

    pieces = _pieces_that_fit_the_embedding_input(
        content, budget=20, measure=spiky_measure
    )

    assert len(pieces) > 1
    for piece in pieces:
        assert piece, "a cut must advance"
        assert spiky_measure(piece) <= 20
    assert "".join(pieces) == content


def test_content_that_cannot_be_cut_without_a_blank_record_is_an_error() -> None:
    """No escape hatch: no truncation, no whitespace-only record either."""
    with pytest.raises(Mem0PersistenceError) as excinfo:
        _pieces_that_fit_the_embedding_input(
            "   \n\n   \n", budget=2, measure=len
        )

    assert "whitespace-only record" in str(excinfo.value)


# ── mem0 `infer=True`: one over-budget message, pre-split ────────
#
# `_chunks_that_fit_the_embedding_input` batches at message boundaries, so a
# message that alone exceeds the budget was a hard error there — the same
# shape that killed `simple_rag` twice on 2026-09-18, and the one pool the
# `mem0` condition could not write either. `message_split_v1` cuts that one
# message into consecutive, non-overlapping sub-messages first and leaves the
# batching alone.
#
# A resumed run explicitly selects its frozen strategy.

from memory.backends.runtime_contracts import (
    MEM0_EXTRACTION_CHUNK_STRATEGY,
    MEM0_EXTRACTION_CHUNK_STRATEGY_LEGACY,
    MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1,
)


def _with_message_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "memory.backends.mem0_adapter.MEM0_EXTRACTION_CHUNK_STRATEGY",
        MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1,
    )


def test_new_runs_default_to_message_split() -> None:
    assert MEM0_EXTRACTION_CHUNK_STRATEGY == MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1


def test_an_unknown_strategy_is_refused_at_construction() -> None:
    """The adapter validates what it is handed rather than falling back.

    A strategy it does not implement has to fail where the run is wired, not
    quietly cut (or not cut) a pool some other way.
    """
    with pytest.raises(ValueError, match="Unknown Mem0 extraction chunk strategy"):
        Mem0Adapter(client=FakeMem0Client(), extraction_chunk_strategy="message_split_v2")


@pytest.mark.asyncio
async def test_legacy_and_split_runs_can_share_a_process() -> None:
    pool = oversized_reasoning_turn_pool()
    legacy_client, split_client = FakeMem0Client(), FakeMem0Client()
    legacy = Mem0Adapter(client=legacy_client, extraction_chunk_strategy="legacy_no_message_split")
    split = Mem0Adapter(client=split_client, extraction_chunk_strategy="message_split_v1")
    with pytest.raises(Mem0PersistenceError):
        await legacy.record_event(pool_event(pool))
    receipt = await split.record_event(pool_event(pool))
    assert receipt.status == WriteStatus.committed
    assert not legacy_client.add_calls
    assert len(split_client.add_calls) > 1
    assert "".join(m["content"] for c in split_client.add_calls for m in c["messages"]) == "".join(m["content"] for m in pool.messages)


@pytest.mark.asyncio
async def test_mem0_splits_one_over_budget_message_and_keeps_the_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_message_split(monkeypatch)
    pool = oversized_reasoning_turn_pool()
    oversized = [
        index
        for index, message in enumerate(pool.messages)
        if _embed_input_tokens([message]) > _RAW_BUDGET
    ]
    assert oversized == [1], "the fixture must carry exactly one over-budget message"
    client = FakeMem0Client()
    adapter = Mem0Adapter(client=client)
    await adapter.setup_scope(_scope())

    receipt = await adapter.record_event(pool_event(pool))

    chunks = [call["messages"] for call in client.add_calls]
    assert receipt.chunk_count == len(chunks)
    # Every batch is measured as the concatenation mem0 really builds.
    for chunk in chunks:
        assert chunk, "a batch must carry at least one message"
        assert _embed_input_tokens(chunk) <= _RAW_BUDGET
        assert all(call["infer"] is True for call in client.add_calls)
    written = [message for chunk in chunks for message in chunk]
    # Consecutive and non-overlapping: the sub-messages of message 1 sit in
    # its place, in order, and nothing else moved or repeated.
    trailing = len(pool.messages) - 2
    assert written[0] == pool.messages[0]
    assert written[-trailing:] == pool.messages[2:]
    pieces = written[1:-trailing]
    assert len(pieces) > 1
    assert "".join(piece["content"] for piece in pieces) == pool.messages[1]["content"]
    assert {piece["role"] for piece in pieces} == {pool.messages[1]["role"]}


@pytest.mark.asyncio
async def test_mem0_splits_a_message_that_only_the_wrapping_pushes_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two paths measure differently, and this is the gap between them.

    `infer=False` embeds the content alone; `infer=True` embeds the
    `parse_messages` concatenation, which prefixes the role and closes with a
    newline. This content fits by the first measure and not by the second, so
    only a measure taken the way mem0 builds its input catches it.
    """
    _with_message_split(monkeypatch)
    sentence = "two cases of lemons for the weekend brunch service, as dictated. "
    content = sentence
    while _raw_embed_input_tokens(content) <= _RAW_BUDGET:
        content += sentence
    while _raw_embed_input_tokens(content) > _RAW_BUDGET:
        content = content[:-1]
    assert _raw_embed_input_tokens(content) <= _RAW_BUDGET
    assert _embed_input_tokens([{"role": "assistant", "content": content}]) > _RAW_BUDGET
    client = FakeMem0Client()
    adapter = Mem0Adapter(client=client)
    await adapter.setup_scope(_scope())
    event = MemoryEvent(
        event_id="evt_wrapping",
        run_id="run_001",
        persona_id="user_b",
        agent_id="nanobot",
        session_id="s01",
        turn_id="acc_017:request_dance_spot:t0",
        timestamp=datetime(2026, 9, 18, 5, 15, tzinfo=timezone.utc),
        messages=[{"role": "assistant", "content": content}],
    )

    await adapter.record_event(event)

    written = [
        message for call in client.add_calls for message in call["messages"]
    ]
    assert len(written) > 1
    for call in client.add_calls:
        assert _embed_input_tokens(call["messages"]) <= _RAW_BUDGET
    assert "".join(message["content"] for message in written) == content



# ── What a failed batch leaves behind ───────────────────────────
#
# Every piece and every batch is measured before the first `add`, so a pool
# that cannot be written is refused with nothing written
# (`test_a_single_message_over_the_cap_raises_instead_of_truncating`). That is
# not a transaction: an endpoint that fails on a later batch leaves the
# earlier ones in mem0. The journal record for the turn stays `pending`
# (`tests/test_memory_turn_journal.py::
# test_crash_after_append_leaves_the_record_pending_for_resume`), a formal
# run turns the failure into `AdapterWriteCommitError` → `memory_backend`
# (`harness/failure_kind.py`) and leaves the step unfinished, and the next
# launch rolls the whole run back to its last checkpoint — which deletes and
# reloads the collection (`tests/test_harness/test_checkpoint.py::
# test_mem0_restore_deletes_then_reloads_same_name`) and restores the journal
# and ledger with it, rather than replaying the turn onto a half-written
# backend.


@pytest.mark.asyncio
async def test_a_failing_later_batch_leaves_the_earlier_batches_written() -> None:
    class FailsOnTheSecondBatch(FakeMem0Client):
        def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
            if len(self.add_calls) == 1:
                raise Mem0PersistenceError("Mem0 add failed: batch 2")
            return super().add(messages, **kwargs)

    pool = acc_sized_turn_pool()
    client = FailsOnTheSecondBatch()
    adapter = Mem0Adapter(client=client)
    await adapter.setup_scope(_scope())

    with pytest.raises(Mem0PersistenceError):
        await adapter.record_event(pool_event(pool))

    # One batch is in mem0 and no receipt was returned, so nothing downstream
    # may treat this turn as written.
    assert len(client.add_calls) == 1
