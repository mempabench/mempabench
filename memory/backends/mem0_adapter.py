"""mem0 adapter — preserves native add/search semantics."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import re
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import tiktoken
from mem0.memory.utils import parse_messages

from memory.adapter import PromptHistoryEvictionPolicy
from memory.backends.mem0_client import (
    MEM0_EXTRACTION_MAX_OUTPUT_TOKENS,
    Mem0Client,
    Mem0PersistenceError,
    _validate_add_response,
    _validate_search_response,
)
from memory.backends.runtime_contracts import (
    MEM0_EXTRACTION_CHUNK_STRATEGY,
    MEM0_EXTRACTION_CHUNK_STRATEGY_LEGACY,
    MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1,
    MEM0_OPENAI_EMBEDDING_MODEL,
)
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


#: The single-input token ceiling the OpenAI embeddings endpoint enforces for
#: `text-embedding-3-small` — the embedder our mem0 config pins
#: (`MEM0_OPENAI_EMBEDDING_MODEL`, wired into the client's `embedder` block in
#: `memory/backends/mem0_client.py`). It is the cap that killed the 2026-09-14
#: `mem0` smoke on its first write, and this is the repo's only declaration of
#: it.
#:
#: On the `infer=True` path mem0 embeds the WHOLE turn pool as ONE input
#: (`mem0/memory/main.py:821-825`):
#:
#:     parsed_messages = parse_messages(messages)
#:     ...
#:     query_embedding = self.embedding_model.embed(parsed_messages, "search")
#:
#: A MemPA turn pool carries the whole turn experience — reasoning, every tool
#: call, every tool result — and runs ~10k tokens, so it does not fit. The
#: `infer=False` path never reaches that block: it builds one record per
#: message and embeds each on its own. That deferred the cap for `simple_rag`
#: rather than removing it — one MESSAGE over the cap is unwritable there, and
#: two of them killed `ub_simple_rag_glm_5_3_flash_20260918` on 2026-09-18
#: (`_raw_records_that_fit_the_embedding_input`).
MEM0_EMBEDDING_MAX_INPUT_TOKENS = 8191

#: Held back from the cap when sizing a chunk. We count the string locally and
#: the endpoint counts it again server-side, and mem0 rewrites it once more on
#: the way out (`mem0/embeddings/openai.py`, `text.replace("\n", " ")`, which
#: `_embedding_input_tokens` mirrors). 256 tokens is ~3% of the cap — enough
#: that no rounding, version skew or future rewrite lands a chunk over it,
#: small enough that it does not meaningfully change what mem0 extracts from.
MEM0_EMBEDDING_CHUNK_SAFETY_MARGIN_TOKENS = 256


@lru_cache(maxsize=1)
def _encoding() -> Any:
    return tiktoken.encoding_for_model(MEM0_OPENAI_EMBEDDING_MODEL)


def mem0_embedding_chunk_policy() -> dict[str, Any]:
    """The values that decide what mem0 infers over, and how much it may emit.

    Four of them decide where a pool is cut for embedding and a fifth decides
    whether one over-budget message is cut at all: two runs that cut a pool in
    different places hand mem0 different inputs to infer over, so they are
    different runs. The last is the completion ceiling on the extraction call,
    which decides how much of what mem0 found can come back — a run that cuts
    identically but truncates its extractor is a different run too.

    The frozen mem0 contract records this dict — read from the constants,
    never copied, so the record cannot drift from the code that does the
    cutting and the calling.

    `extraction_chunk_strategy` here is what a *new* plan freezes, which is
    `message_split_v1`. It is not what every live adapter runs: a resumed run
    passes the strategy recorded in its own frozen plan to `Mem0Adapter`
    (`scripts/run_orchestrator.py`), so an old legacy-strategy run and a new
    split-strategy run can be continued in one process, each on its own.
    """
    return {
        "embedding_max_input_tokens": MEM0_EMBEDDING_MAX_INPUT_TOKENS,
        "embedding_chunk_safety_margin_tokens": (
            MEM0_EMBEDDING_CHUNK_SAFETY_MARGIN_TOKENS
        ),
        "embedding_tokenizer_encoding": _encoding().name,
        "tiktoken_version": importlib.metadata.version("tiktoken"),
        "extraction_chunk_strategy": MEM0_EXTRACTION_CHUNK_STRATEGY,
        "extraction_max_output_tokens": MEM0_EXTRACTION_MAX_OUTPUT_TOKENS,
    }


def _embedding_input_tokens(messages: list[dict[str, Any]]) -> int:
    """Tokens in the single input mem0 would embed for these messages.

    Built the way mem0 builds it: its own `parse_messages` concatenation,
    then the newline flattening its OpenAI embedder applies before the call.
    That is the exact string the endpoint tokenizes, so this is a measurement
    rather than an estimate.
    """
    return len(_encoding().encode(parse_messages(messages).replace("\n", " ")))


def _chunks_that_fit_the_embedding_input(
    messages: list[dict[str, Any]],
    *,
    turn_id: str,
) -> list[list[dict[str, Any]]]:
    """Split a pool at message boundaries into inputs mem0 can embed.

    Consecutive, order-preserving, never splitting a message: every message
    reaches mem0, in the order the PA produced it, and mem0 still infers over
    each chunk. Each chunk is measured as the concatenation mem0 will actually
    build, so the batch that is written is the batch that was checked.

    A message that alone exceeds the budget is a hard error — the pool is the
    benchmark's payload and nothing here may truncate or skip it. Under
    `MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1` such a message is cut
    into sub-messages before it arrives here, so the error is reachable only
    under the legacy strategy.
    """
    budget = MEM0_EMBEDDING_MAX_INPUT_TOKENS - MEM0_EMBEDDING_CHUNK_SAFETY_MARGIN_TOKENS
    chunks: list[list[dict[str, Any]]] = [[]]
    for index, message in enumerate(messages):
        alone = _embedding_input_tokens([message])
        if alone > budget:
            raise Mem0PersistenceError(
                "Mem0 cannot embed a pool message this large, and the pool "
                "must not be truncated: "
                f"turn_id={turn_id!r} message_index={index} tokens={alone} "
                f"budget={budget} cap={MEM0_EMBEDDING_MAX_INPUT_TOKENS}"
            )
        if chunks[-1] and _embedding_input_tokens(chunks[-1] + [message]) > budget:
            chunks.append([])
        chunks[-1].append(message)
    return chunks


def _messages_that_fit_the_embedding_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The pool, with any single over-budget message cut into sub-messages.

    `MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1`. Only a message that
    does not fit on its own is touched, and it is cut into consecutive,
    non-overlapping sub-messages of the same role whose contents concatenate
    back to the original byte for byte. The batching below is unchanged: the
    sub-messages take the original's place in the pool and
    `_chunks_that_fit_the_embedding_input` groups them with their neighbours
    the way it groups any message.

    The measure is this path's own: mem0 embeds the `parse_messages`
    concatenation, so a piece is measured wrapped in its role the way mem0
    will wrap it, not as bare content. A message whose role `parse_messages`
    drops contributes nothing to that input and is never cut.
    """
    budget = MEM0_EMBEDDING_MAX_INPUT_TOKENS - MEM0_EMBEDDING_CHUNK_SAFETY_MARGIN_TOKENS
    prepared: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or _embedding_input_tokens([message]) <= budget:
            prepared.append(message)
            continue

        def measure(piece: str, message: dict[str, Any] = message) -> int:
            return _embedding_input_tokens([{**message, "content": piece}])

        prepared.extend(
            {**message, "content": piece}
            for piece in _pieces_that_fit_the_embedding_input(
                content, budget=budget, measure=measure
            )
        )
    return prepared


def _raw_embedding_input_tokens(content: str) -> int:
    """Tokens in the single input mem0 embeds for ONE raw-dialogue record.

    The `infer=False` path never builds the `parse_messages` concatenation:
    it embeds each message's own content on its own
    (`mem0/memory/main.py:722-723`, `msg_embeddings =
    self.embedding_model.embed(msg_content, "add")`). So the string the
    endpoint tokenizes for a `simple_rag` record is that content with the
    embedder's newline flattening, and nothing else.
    """
    return len(_encoding().encode(content.replace("\n", " ")))


#: A line that closes an experience block (`memory/turn_experience.py:98`,
#: `close_line = f"[/{tag}]"`). The seam right after one is where a pool
#: message can be cut without leaving a block unterminated.
_BLOCK_CLOSE_LINE = re.compile(r"^\[/[a-z_]+\]$")


def _longest_fitting_prefix(
    content: str,
    *,
    budget: int,
    offsets: list[int],
    measure: Callable[[str], int],
) -> int:
    """The largest of these cut offsets whose prefix fits, or 0 if none does.

    The bisection is a SEARCH over the candidate offsets, not a proof about
    them: it assumes a prefix's token count grows with its length, which no
    tokenizer promises. What makes the result safe is that `best` is only ever
    an offset `measure` itself accepted, so a tokenizer that breaks the
    assumption can cost us a shorter piece — never an over-budget one. When
    the search finds nothing the caller moves to a finer granularity, and the
    piece it finally takes has still been measured.
    """
    low, high, best = 0, len(offsets) - 1, 0
    while low <= high:
        middle = (low + high) // 2
        if measure(content[: offsets[middle]]) <= budget:
            best = offsets[middle]
            low = middle + 1
        else:
            high = middle - 1
    return best


def _cut_offset(
    content: str,
    *,
    budget: int,
    measure: Callable[[str], int],
) -> int:
    """Where to cut this over-budget content, preferring the coarsest seam.

    A block seam first, so a piece is whole experience blocks; a line boundary
    next; and only when one line does not fit on its own, a cut inside that
    line. Every offset is an index into `content`, so the pieces either side
    are exact substrings — the cut removes nothing.

    Candidates are confined to the offsets that leave non-whitespace on BOTH
    sides. A pool message can open on blank lines and end on trailing
    whitespace, and the coarsest fitting seam inside such a run cuts off a
    record that is nothing but whitespace, which mem0 stores and
    `_validate_add_response` then refuses as empty memory text. Moving the
    boundary is what keeps that record out — every whitespace character still
    reaches mem0, on the piece the boundary moved it to.
    """
    lowest = len(content) - len(content.lstrip()) + 1
    highest = len(content.rstrip()) - 1
    if lowest > highest:
        raise Mem0PersistenceError(
            "Mem0 cannot embed this content and it cannot be cut without "
            "writing a whitespace-only record: "
            f"tokens={measure(content)} budget={budget} "
            f"characters={len(content)} content={content[:120]!r}"
        )
    seams: list[int] = []
    line_ends: list[int] = []
    start = 0
    for line in content.split("\n")[:-1]:
        start += len(line) + 1
        line_ends.append(start)
        if _BLOCK_CLOSE_LINE.match(line):
            seams.append(start)
    for offsets in (seams, line_ends, list(range(lowest, highest + 1))):
        cut = _longest_fitting_prefix(
            content,
            budget=budget,
            offsets=[
                offset for offset in offsets if lowest <= offset <= highest
            ],
            measure=measure,
        )
        if cut:
            return cut
    raise Mem0PersistenceError(
        "Mem0 cannot embed this content and no cut of it fits the embedding "
        "input: "
        f"tokens={measure(content)} budget={budget} "
        f"characters={len(content)} content={content[:120]!r}"
    )


def _pieces_that_fit_the_embedding_input(
    content: str,
    *,
    budget: int,
    measure: Callable[[str], int],
) -> list[str]:
    """Cut content that does not fit into consecutive pieces that do.

    Shared by both write paths, which differ only in `measure`: `infer=False`
    embeds a message's own content, `infer=True` embeds the `parse_messages`
    concatenation the message is wrapped into. Each returned piece has been
    measured within `budget` by that same function, the pieces are exact
    substrings in order, and every cut advances by at least one character, so
    the loop always terminates.
    """
    pieces: list[str] = []
    rest = content
    while measure(rest) > budget:
        cut = _cut_offset(rest, budget=budget, measure=measure)
        pieces.append(rest[:cut])
        rest = rest[cut:]
    pieces.append(rest)
    return pieces


def _raw_records_that_fit_the_embedding_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The pool's messages, with any over-budget one cut into raw records.

    `simple_rag` stores one raw record per message and embeds each message's
    own content, so message-boundary chunking never reached it and a single
    over-cap message was the one thing it could not write at all. That is what
    killed `ub_simple_rag_glm_5_3_flash_20260918` twice on 2026-09-18: one
    8291-token assistant message at `acc_014:beat_clarify_task:t1`, one 8567-
    token one at `acc_017:request_dance_spot:t0`, both a long `[reasoning]`
    block rendered into the same message as the iteration's other blocks.

    A naive RAG baseline's answer to a document that does not fit its embedder
    is to chunk it, so such a message becomes consecutive records of the same
    role, in the same order, whose contents concatenate back to the original
    byte for byte. Nothing is truncated, shortened or skipped.

    What does NOT follow is that every input already written stays one record.
    The threshold here is the 7935-token BUDGET, not the endpoint's 8191-token
    cap, and the band between the two is populated: five messages the
    2026-09-18 user_b runs already committed measure 7954 to 8138 tokens —
    `ub_simple_rag_glm_5_3_flash_20260918` at `acc_028:beat_2:t2` (7954),
    `ub_simple_rag_qwen3_8_27b_20260918` at `acc_033`/`acc_048` (7966, 8066),
    `ub_simple_rag_deepseek_v4_1_flash_20260918` at `acc_033`/`acc_044` (8033,
    8138) — each written as ONE record, because the endpoint accepted it.
    Resumed under this code, a message in that band becomes two records: the
    same bytes, but two memory entries and two retrieval units where there was
    one. Only a message at or below the budget passes through untouched, and
    that — not "everything that used to be written" — is the claim this
    function can make (owner ruling 2026-09-18, ⑥: keep the 7935 margin and
    record the consequence).
    """
    budget = MEM0_EMBEDDING_MAX_INPUT_TOKENS - MEM0_EMBEDDING_CHUNK_SAFETY_MARGIN_TOKENS
    records: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or _raw_embedding_input_tokens(content) <= budget:
            records.append(message)
            continue
        records.extend(
            {**message, "content": piece}
            for piece in _pieces_that_fit_the_embedding_input(
                content, budget=budget, measure=_raw_embedding_input_tokens
            )
        )
    return records


class Mem0Adapter:
    """Adapter for mem0 — first real memory architecture provider.

    Supports ``infer=True`` (default, LLM-extracted memories) and
    ``infer=False`` (raw chunks, used for Simple RAG ablation).
    """

    backend_name: str = "mem0"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.after_adapter_commit

    def __init__(
        self,
        *,
        client: Mem0Client,
        infer: bool = True,
        top_k: int = 20,
        threshold: float = 0.1,
        rerank: bool = False,
        agent_id: str | None = None,
        extraction_chunk_strategy: str | None = None,
        retrieval_run_id: str | None = None,
        read_only_source_run_id: str | None = None,
        shared_binding: Any | None = None,
    ) -> None:
        if retrieval_run_id is not None and (
            not isinstance(retrieval_run_id, str) or not retrieval_run_id.strip()
        ):
            raise ValueError("Mem0 retrieval_run_id must be a non-empty source run ID")
        self._client = client
        self._infer = infer
        self._extraction_chunk_strategy = (
            MEM0_EXTRACTION_CHUNK_STRATEGY
            if extraction_chunk_strategy is None else extraction_chunk_strategy
        )
        if self._extraction_chunk_strategy not in {
            MEM0_EXTRACTION_CHUNK_STRATEGY_LEGACY,
            MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1,
        }:
            raise ValueError(f"Unknown Mem0 extraction chunk strategy: {self._extraction_chunk_strategy!r}")
        self.backend_name = "mem0" if infer else "simple_rag"
        self._top_k = top_k
        self._threshold = threshold
        self._rerank = rerank
        self._agent_id = agent_id
        self._retrieval_run_id = retrieval_run_id
        if read_only_source_run_id is not None and not read_only_source_run_id.strip():
            raise ValueError("A borrowed memory source run_id must be nonempty")
        self._read_only_source_run_id = read_only_source_run_id
        self._shared_binding = shared_binding
        if shared_binding is not None:
            shared_binding.verify()
            if (read_only_source_run_id is None or
                    read_only_source_run_id != shared_binding.owner.source.name or
                    client._expected_run_id != read_only_source_run_id or
                    client._collection_name != shared_binding.namespace):
                raise ValueError('Shared memory client source/namespace binding differs')
        self._mode = AdapterMode(read_only=read_only_source_run_id is not None)
        self._scope: MemoryScope | None = None

    async def setup_scope(self, scope: MemoryScope) -> None:
        self._verify_borrowed_binding(scope)
        self._scope = scope

    def _verify_borrowed_binding(self, scope: MemoryScope | None = None) -> None:
        if self._shared_binding is not None:
            self._shared_binding.verify(scope)
            if (self._read_only_source_run_id != self._shared_binding.owner.source.name or
                    self._client._expected_run_id != self._read_only_source_run_id or
                    self._client._collection_name != self._shared_binding.namespace):
                raise ValueError('Shared memory client identity changed')

    async def set_mode(self, mode: AdapterMode) -> ModeReceipt:
        self._verify_borrowed_binding()
        if self._read_only_source_run_id is not None and not mode.read_only:
            if self._shared_binding is not None:
                self._shared_binding.owner.reject(self._shared_binding, 'set_mode(read_only=False)')
            raise ValueError("Borrowed memory must remain read-only")
        prev = self._mode
        self._mode = mode
        return ModeReceipt(mode=mode, previous_mode=prev)

    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        scope = MemoryScope(run_id=event.run_id, persona_id=event.persona_id)
        if self._mode.read_only:
            return WriteReceipt(
                status=WriteStatus.skipped_read_only,
                trace=self._trace(scope, "record_event"),
            )

        metadata = {
            "session_id": event.session_id,
            "turn_id": event.turn_id,
            "timestamp": event.timestamp.isoformat(),
        }

        t0 = time.monotonic()
        # One `add` per chunk, in order, each carrying the same scope and the
        # same metadata. Only the `infer=True` path needs more than one chunk:
        # `infer=False` embeds one message at a time and never builds the
        # concatenation, so its whole pool is a single `add` — of records that
        # each fit that per-message input.
        # Every piece and every chunk is measured before this loop starts, so
        # a pool that cannot be written is refused with nothing written. That
        # is not a transaction: if a later chunk raises, the earlier ones are
        # already in mem0 and stay there. A formal run raises
        # `AdapterWriteCommitError` from there (`memory/event_writer.py`),
        # which is a `memory_backend` failure (`harness/failure_kind.py`), and
        # the next launch rolls the whole run back to its last checkpoint —
        # collection, journal and ledger together (`harness/orchestrator.py`,
        # `resume_or_restore`). The journal's own at-least-once replay
        # (`memory/turn_journal.py`) re-executes a turn's write, so it must
        # not be what a partial write is recovered by.
        # Which pool `infer=True` batches is the frozen extraction chunk
        # strategy's decision, and a plan frozen without the field runs the
        # legacy one (`Mem0RuntimeConfig.extraction_chunk_strategy`).
        if self._infer:
            pool = (
                _messages_that_fit_the_embedding_input(event.messages)
                if self._extraction_chunk_strategy
                == MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1
                else event.messages
            )
            chunks = _chunks_that_fit_the_embedding_input(
                pool, turn_id=event.turn_id
            )
        else:
            chunks = [_raw_records_that_fit_the_embedding_input(event.messages)]
        scope_kwargs = {
            "user_id": event.persona_id,
            "agent_id": self._agent_id,
            "run_id": event.run_id,
            "metadata": metadata,
            "infer": self._infer,
        }
        native_request = {
            # What the backend was actually handed, which is also where a
            # split is visible after the fact: this list carries the records
            # `infer=False` was cut into. Identical to `event.messages` only
            # on a turn whose messages are each within the 7935-token budget
            # — NOT merely under the 8191-token cap, which is a wider band and
            # one that real turns land in (`_raw_records_that_fit_the_
            # embedding_input`).
            "messages": [message for chunk in chunks for message in chunk],
            **scope_kwargs,
            "chunk_count": len(chunks),
        }
        native_results: list[dict[str, Any]] = []
        for chunk in chunks:
            native_results.extend(
                _validate_add_response(
                    self._client.add(messages=chunk, **scope_kwargs)
                )
            )
        latency = (time.monotonic() - t0) * 1000

        result = {"results": native_results}
        native_ids = [r["id"] for r in native_results]
        native_events = [r["event"] for r in native_results]
        if not native_ids:
            if not self._infer:
                raise Mem0PersistenceError(
                    "Simple RAG received non-empty visible messages but Mem0 "
                    "persisted no raw chunks: "
                    f"run_id={event.run_id!r} session_id={event.session_id!r} "
                    f"turn_id={event.turn_id!r} messages={event.messages!r}"
                )
            return WriteReceipt(
                status=WriteStatus.accepted_no_change,
                chunk_count=len(chunks),
                trace=Trace(
                    backend_name=self.backend_name,
                    scope=scope,
                    native_operation="add",
                    native_ids=[],
                    native_events=native_events,
                    latency_ms=latency,
                    consistency_model=ConsistencyModel.immediate,
                    native_request=native_request,
                    native_response=result,
                    warnings=[
                        "Mem0 extraction completed and accepted this turn "
                        "without a memory change"
                    ],
                ),
            )

        return WriteReceipt(
            status=WriteStatus.committed,
            native_ids=native_ids,
            native_events=native_events,
            chunk_count=len(chunks),
            trace=Trace(
                backend_name=self.backend_name,
                scope=scope,
                native_operation="add",
                native_ids=native_ids,
                native_events=native_events,
                latency_ms=latency,
                consistency_model=ConsistencyModel.immediate,
                native_request=native_request,
                native_response=result,
            ),
        )

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        self._verify_borrowed_binding(request.scope)
        filters: dict[str, Any] = {
            "user_id": request.scope.persona_id,
            # Immutable checkpoint payloads keep the source run identity.
            # Probe evidence below still belongs to this independent repeat.
            # A borrowed read-only source (a shared-setting repeat) names its own run.
            "run_id": (
                self._read_only_source_run_id
                or (
                    self._retrieval_run_id
                    if self._mode.read_only and self._retrieval_run_id is not None
                    else request.scope.run_id
                )
            ),
        }
        if self._agent_id:
            filters["agent_id"] = self._agent_id
        top_k = request.budget.max_items or self._top_k

        t0 = time.monotonic()
        native_request = {
            "query": request.query,
            "top_k": top_k,
            "filters": filters,
            "threshold": self._threshold,
            "rerank": self._rerank,
        }
        # Off the event loop: `search` is a synchronous client call, and on the
        # loop it left every other coroutine's in-flight HTTP unread for its
        # whole length -- which at more than one session at a time spends
        # another session's request budget (probe parallel runner plan §9).
        result = await asyncio.to_thread(self._client.search, **native_request)
        latency = (time.monotonic() - t0) * 1000

        raw_results = _validate_search_response(result)
        if not self._infer:
            self._validate_simple_rag_search_results(raw_results)

        scores = [r.get("score") for r in raw_results if r.get("score") is not None]
        top_score = max(scores) if scores else None

        return RetrieveResult(
            raw=raw_results,
            trace=Trace(
                backend_name=self.backend_name,
                scope=request.scope,
                native_operation="search",
                native_ids=[r.get("id", "") for r in raw_results],
                latency_ms=latency,
                consistency_model=ConsistencyModel.immediate,
                retrieved_count=len(raw_results),
                top_score=top_score,
                native_request=native_request,
                native_response=result,
            ),
        )

    def format_context(self, result: RetrieveResult) -> str:
        if not result.raw:
            return ""
        lines = []
        for item in result.raw:
            text = item.get("memory", "")
            if text:
                if self._infer:
                    lines.append(text)
                    continue
                metadata = item["metadata"]
                identity = json.dumps(
                    {
                        "role": item["role"],
                        "session_id": metadata["session_id"],
                        "turn_id": metadata["turn_id"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                lines.append(f"{identity}\n{text}")
        if not lines:
            return ""
        return (
            f'<memory-context backend="{self.backend_name}">\n'
            + "\n".join(lines)
            + "\n</memory-context>"
        )

    @staticmethod
    def _validate_simple_rag_search_results(
        results: list[dict[str, Any]],
    ) -> None:
        for index, item in enumerate(results):
            role = item.get("role")
            if role not in {"user", "assistant"}:
                raise ValueError(
                    "Simple RAG SEARCH result requires a user or assistant role: "
                    f"index={index} value={role!r} result={item!r}"
                )
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    "Simple RAG SEARCH result requires metadata: "
                    f"index={index} value={metadata!r}"
                )
            for field_name in ("session_id", "turn_id"):
                value = metadata.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        "Simple RAG SEARCH result metadata requires a non-empty "
                        f"{field_name}: index={index} value={value!r}"
                    )

    async def reset_scope(self, scope: MemoryScope) -> ResetReceipt:
        if self._shared_binding is not None:
            self._verify_borrowed_binding(scope)
            self._shared_binding.owner.reject(self._shared_binding, 'reset_scope')
        if self._mode.read_only:
            raise PermissionError("Cannot reset a read-only Mem0 Probe scope")
        self._client.delete_all(
            user_id=scope.persona_id,
            agent_id=self._agent_id,
            run_id=scope.run_id,
        )
        return ResetReceipt(scope=scope, cleared=True, warnings=[])

    async def health(self) -> BackendHealth:
        latency_ms = self._client.health()
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=latency_ms,
            consistency_model=ConsistencyModel.immediate,
            warnings=[],
        )

    def _trace(
        self,
        scope: MemoryScope,
        operation: str,
    ) -> Trace:
        return Trace(
            backend_name=self.backend_name,
            scope=scope,
            native_operation=operation,
            native_ids=[],
            latency_ms=0.0,
            consistency_model=ConsistencyModel.immediate,
        )
