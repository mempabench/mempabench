"""Mem0 client construction and backend infrastructure."""

from __future__ import annotations

import asyncio
import json
import logging
import stat
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import math
import os
import threading
import time
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from pa.providers.base import LLMResponse, LLMTaskIdentity
from pa.providers.openai_compat_provider import OpenAICompatProvider
from pa.providers.registry import find_by_name
from pa.providers.request_policy import resolve_model_request_policy
from memory.backends.runtime_contracts import (
    MEM0_HISTORY_DB_RELATIVE_PATH,
    MEM0_ENTITY_MODEL,
    MEM0_SPACY_VERSION,
    MEM0_OPENAI_EMBEDDING_DIMENSIONS as OPENAI_EMBEDDING_DIMENSIONS,
    MEM0_OPENAI_EMBEDDING_MODEL as OPENAI_EMBEDDING_MODEL,
    MEM0_OPENAI_EMBEDDING_PROVIDER as OPENAI_EMBEDDING_PROVIDER,
    MEM0_QDRANT_BM25_ENABLED,
    MEM0_QDRANT_DISTANCE,
    MEM0_QDRANT_ON_DISK,
    MEM0_RETRIEVAL_RERANK,
    MEM0_RETRIEVAL_SIMILARITY_THRESHOLD,
    MEM0_SUBMODULE_COMMIT,
    MEM0_VERSION,
)
from memory.backends.qdrant_runtime import qdrant_client

if TYPE_CHECKING:
    from harness.provider_avoidance import ProviderAvoidance


_PERSISTED_EVENTS = frozenset({"ADD", "UPDATE"})
_SUPPORTED_EVENTS = frozenset({"ADD", "UPDATE", "DELETE"})
_RAW_DIALOGUE_ROLES = frozenset({"user", "assistant"})

#: The completion ceiling on one mem0 extraction call. It decides what mem0 is
#: able to extract — a call may spend its budget before yielding usable content, so this
#: belongs to the frozen mem0 contract (`mem0_embedding_chunk_policy()` folds it
#: into `Mem0RuntimeConfig`) and not to the call site.
#:
#: The 2000 this replaces killed the first live metering run: three
#: extraction attempts on one chunk, all `finish_reason=length` at
#: `completion_tokens=2000`, all with empty content. Replaying that exact chunk
#: raw showed why — the budget went to reasoning, not to output:
#:
#:     max_tokens=2000   finish_reason='length' completion_tokens=2000
#:                       reasoning_tokens=2277  content_chars=373
#:     max_tokens=16000  finish_reason='stop'   completion_tokens=1483
#:                       reasoning_tokens=1511  content_chars=771
#:
#: Both models validated for the `memory_extraction` role reason before they
#: answer, and the same payload at temperature 0.1 wanted 1511 tokens of it on
#: one sample and 2277 on the next — a 1.5x spread the old ceiling sat inside.
#: 16000 is ~7x the largest demand observed (2277) and ~10x the largest
#: completed output (1483). The ceiling is generous on purpose: it bills nothing
#: it does not use, so a call that needs 1500 tokens costs 1500 whatever the
#: ceiling is, and the worst case of a runaway reply is ~$0.003 at the measured
#: output price — against a dead run for a ceiling set too tight.
MEM0_EXTRACTION_MAX_OUTPUT_TOKENS = 16000


class Mem0ResponseError(ValueError):
    """Mem0 returned a value that does not satisfy its client contract."""


class Mem0PersistenceError(RuntimeError):
    """Mem0 reported a mutation that could not be confirmed in storage."""


def _validate_add_response(response: Any) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        raise Mem0ResponseError(
            f"Mem0 ADD response must be a dictionary: value={response!r}"
        )
    results = response.get("results")
    if not isinstance(results, list):
        raise Mem0ResponseError(
            "Mem0 ADD response requires a results list: "
            f"value={results!r} response={response!r}"
        )
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise Mem0ResponseError(
                "Mem0 ADD result must be a dictionary: "
                f"index={index} value={result!r}"
            )
        memory_id = result.get("id")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise Mem0ResponseError(
                "Mem0 ADD result requires a non-empty string id: "
                f"index={index} value={memory_id!r} result={result!r}"
            )
        event = result.get("event")
        if event not in _SUPPORTED_EVENTS:
            raise Mem0ResponseError(
                "Mem0 ADD result has an unsupported event: "
                f"index={index} value={event!r} "
                f"supported={sorted(_SUPPORTED_EVENTS)!r}"
            )
        memory = result.get("memory")
        if not isinstance(memory, str) or not memory.strip():
            raise Mem0ResponseError(
                "Mem0 ADD result requires non-empty memory text: "
                f"index={index} value={memory!r} result={result!r}"
            )
    return results


def _validate_search_response(response: Any) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        raise Mem0ResponseError(
            f"Mem0 SEARCH response must be a dictionary: value={response!r}"
        )
    results = response.get("results")
    if not isinstance(results, list):
        raise Mem0ResponseError(
            "Mem0 SEARCH response requires a results list: "
            f"value={results!r} response={response!r}"
        )
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise Mem0ResponseError(
                "Mem0 SEARCH result must be a dictionary: "
                f"index={index} value={result!r}"
            )
        memory_id = result.get("id")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise Mem0ResponseError(
                "Mem0 SEARCH result requires a non-empty string id: "
                f"index={index} value={memory_id!r} result={result!r}"
            )
        memory = result.get("memory")
        if not isinstance(memory, str) or not memory.strip():
            raise Mem0ResponseError(
                "Mem0 SEARCH result requires non-empty memory text: "
                f"index={index} value={memory!r} result={result!r}"
            )
        score = result.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
        ):
            raise Mem0ResponseError(
                "Mem0 SEARCH score must be a finite number or None: "
                f"index={index} value={score!r} result={result!r}"
            )
    return results


def _validated_raw_dialogue_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise Mem0ResponseError(
            "Simple RAG ADD messages must be a list: "
            f"value_type={type(messages).__name__} value={messages!r}"
        )
    visible_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise Mem0ResponseError(
                "Simple RAG ADD message must be a dictionary: "
                f"index={index} value={message!r}"
            )
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if role not in _RAW_DIALOGUE_ROLES:
            raise Mem0ResponseError(
                "Simple RAG ADD message has an unsupported visible role: "
                f"index={index} value={role!r} "
                f"supported={sorted(_RAW_DIALOGUE_ROLES)!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise Mem0ResponseError(
                "Simple RAG ADD message requires non-empty content: "
                f"index={index} value={content!r}"
            )
        visible_messages.append({"role": role, "content": content})
    if not visible_messages:
        raise Mem0ResponseError(
            "Simple RAG ADD requires at least one visible user or assistant "
            f"message: value={messages!r}"
        )
    return visible_messages


def _validate_raw_dialogue_results(
    results: list[dict[str, Any]],
    messages: Any,
) -> None:
    visible_messages = _validated_raw_dialogue_messages(messages)
    if len(results) != len(visible_messages):
        raise Mem0PersistenceError(
            "Simple RAG did not persist exactly one chunk per visible message: "
            f"expected_count={len(visible_messages)} actual_count={len(results)} "
            f"results={results!r}"
        )
    for index, (result, message) in enumerate(zip(results, visible_messages)):
        if result["event"] != "ADD":
            raise Mem0PersistenceError(
                "Simple RAG raw dialogue writes must report ADD: "
                f"index={index} value={result['event']!r} result={result!r}"
            )
        if result["memory"] != message["content"]:
            raise Mem0PersistenceError(
                "Simple RAG persisted different message text: "
                f"index={index} expected={message['content']!r} "
                f"actual={result['memory']!r}"
            )
        if result.get("role") != message["role"]:
            raise Mem0PersistenceError(
                "Simple RAG persisted a different message role: "
                f"index={index} expected={message['role']!r} "
                f"actual={result.get('role')!r}"
            )


def _validate_embedding_vector(
    embedding: Any,
    *,
    expected_dimensions: int,
    operation: str,
) -> None:
    if not isinstance(embedding, list):
        raise Mem0ResponseError(
            "Mem0 embedding response must be a list: "
            f"operation={operation!r} value_type={type(embedding).__name__}"
        )
    if len(embedding) != expected_dimensions:
        raise Mem0ResponseError(
            "Mem0 embedding response has the wrong dimensions: "
            f"operation={operation!r} expected={expected_dimensions!r} "
            f"actual={len(embedding)!r}"
        )
    invalid_indexes = [
        index
        for index, value in enumerate(embedding)
        if isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ]
    if invalid_indexes:
        raise Mem0ResponseError(
            "Mem0 embedding response contains non-finite or non-numeric values: "
            f"operation={operation!r} invalid_indexes={invalid_indexes[:20]!r}"
        )


def _validate_mempa_add_scope(kwargs: dict[str, Any]) -> None:
    for field_name in ("user_id", "run_id"):
        value = kwargs.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise Mem0ResponseError(
                "Mem0 ADD requires a non-empty benchmark scope field: "
                f"field={field_name!r} value={value!r}"
            )
    agent_id = kwargs.get("agent_id")
    if agent_id is not None and (
        not isinstance(agent_id, str) or not agent_id.strip()
    ):
        raise Mem0ResponseError(
            f"Mem0 ADD agent_id must be non-empty when set: value={agent_id!r}"
        )
    metadata = kwargs.get("metadata")
    if not isinstance(metadata, dict):
        raise Mem0ResponseError(
            "Mem0 ADD metadata must be a dictionary: "
            f"value_type={type(metadata).__name__} value={metadata!r}"
        )
    for field_name in ("session_id", "turn_id", "timestamp"):
        value = metadata.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise Mem0ResponseError(
                "Mem0 ADD metadata requires a non-empty operation identity: "
                f"field={field_name!r} value={value!r}"
            )


def _validate_mempa_search_request(
    query: Any,
    kwargs: dict[str, Any],
    *,
    expected_run_id: str,
) -> dict[str, str]:
    if not isinstance(query, str) or not query.strip():
        raise Mem0ResponseError(
            f"Mem0 SEARCH query must be a non-empty string: value={query!r}"
        )
    filters = kwargs.get("filters")
    if not isinstance(filters, dict):
        raise Mem0ResponseError(
            "Mem0 SEARCH filters must be a dictionary: "
            f"value_type={type(filters).__name__} value={filters!r}"
        )
    required_scope: dict[str, str] = {}
    for field_name in ("user_id", "run_id"):
        value = filters.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise Mem0ResponseError(
                "Mem0 SEARCH requires a non-empty benchmark scope filter: "
                f"field={field_name!r} value={value!r}"
            )
        required_scope[field_name] = value
    if required_scope["run_id"] != expected_run_id:
        raise Mem0ResponseError(
            "Mem0 SEARCH run filter does not match the client-owned run: "
            f"expected_run_id={expected_run_id!r} "
            f"value={required_scope['run_id']!r}"
        )
    agent_id = filters.get("agent_id")
    if agent_id is not None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise Mem0ResponseError(
                "Mem0 SEARCH agent_id filter must be non-empty when set: "
                f"value={agent_id!r}"
            )
        required_scope["agent_id"] = agent_id
    top_k = kwargs.get("top_k", 20)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise Mem0ResponseError(
            f"Mem0 SEARCH top_k must be a positive integer: value={top_k!r}"
        )
    threshold = kwargs.get("threshold", 0.1)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(float(threshold))
        or not 0 <= float(threshold) <= 1
    ):
        raise Mem0ResponseError(
            "Mem0 SEARCH threshold must be finite and between 0 and 1: "
            f"value={threshold!r}"
        )
    rerank = kwargs.get("rerank", False)
    if not isinstance(rerank, bool):
        raise Mem0ResponseError(
            f"Mem0 SEARCH rerank must be a boolean: value={rerank!r}"
        )
    return required_scope


def _validate_mempa_search_scope(
    results: list[dict[str, Any]],
    required_scope: dict[str, str],
) -> None:
    for index, result in enumerate(results):
        mismatches = {
            field_name: {
                "expected": expected_value,
                "actual": result.get(field_name),
            }
            for field_name, expected_value in required_scope.items()
            if result.get(field_name) != expected_value
        }
        if mismatches:
            raise Mem0PersistenceError(
                "Mem0 SEARCH returned memory outside the requested scope: "
                f"index={index} id={result.get('id')!r} "
                f"mismatches={mismatches!r}"
            )


class Mem0Client(Protocol):
    """Structural interface for the subset of mem0.Memory we use."""

    def add(
        self,
        messages: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> dict[str, Any]: ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        threshold: float = 0.1,
        rerank: bool = False,
    ) -> dict[str, Any]: ...

    def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]: ...

    def health(self) -> float: ...

    def native_ids(self) -> set[str]: ...

    def native_entity_ids(self) -> set[str]: ...

    def history_native_ids(self) -> set[str]: ...


class _Mem0ExtractionProvider(OpenAICompatProvider):
    """The metered provider, carrying the one request field mem0 relies on.

    mem0 asks for JSON mode — `response_format={"type": "json_object"}`
    (`mem0/memory/main.py`) — and parses the reply with `json.loads`.
    `OpenAICompatProvider.chat` has no such parameter, so the value mem0 passed
    is held for the duration of the call it belongs to and folded into the
    payload the provider builds, leaving everything else the provider's.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._response_format: dict[str, Any] | None = None

    def _build_kwargs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        request_payload = super()._build_kwargs(*args, **kwargs)
        if self._response_format is not None:
            request_payload["response_format"] = self._response_format
        return request_payload


class _MemoryExtractionLLM:
    def __init__(
        self,
        *,
        provider_name: str,
        api_key: str,
        api_base: str,
        model: str,
        extraction_max_output_tokens: int = MEM0_EXTRACTION_MAX_OUTPUT_TOKENS,
        failure_evidence_dir: Path | None = None,
        provider_avoidance: ProviderAvoidance | None = None,
    ) -> None:
        if type(extraction_max_output_tokens) is not int or extraction_max_output_tokens <= 0:
            raise ValueError("Mem0 extraction output budget must be a positive integer")
        self.extraction_max_output_tokens = extraction_max_output_tokens
        self._failure_evidence_dir = failure_evidence_dir
        if not all((provider_name, api_key, api_base, model)):
            raise ValueError(
                "Memory extraction requires a non-empty provider, API key, "
                "API base, and model"
            )
        self._model = model
        self._logical_task_key: str | None = None
        self._request_succeeded = False
        self._length_response_evidence: dict[str, Any] | None = None
        self._policy = resolve_model_request_policy(provider_name, model)
        self._policy.require_exact_role("memory_extraction")
        spec = find_by_name(provider_name)
        if spec is None:
            raise RuntimeError(
                "Memory-side provider is missing from the provider registry: "
                f"provider_name={provider_name!r}"
            )
        self._provider = _Mem0ExtractionProvider(
            api_key=api_key,
            api_base=api_base,
            spec=spec,
            default_model=model,
            # The run's upstream avoidance: these requests skip and count
            # against upstreams exactly as the PA's do.
            provider_avoidance=provider_avoidance,
        )
        # mem0's `add` is synchronous and the adapter awaits it from a running
        # loop, so the provider's coroutines cannot be driven here. One loop on
        # one thread for the life of this object keeps the provider's AsyncOpenAI
        # connection pool on a single loop.
        self._loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._loop.run_forever,
            name="mem0-extraction-llm",
            daemon=True,
        ).start()

    def _record_failure_evidence(
        self, *, response: LLMResponse, messages: list[dict[str, str]],
        response_format: dict[str, Any] | None, tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any], attempt: int, reason: str,
        cause: Exception | None = None,
    ) -> None:
        """Append private response evidence without changing failure/retry semantics.

        Only request payload and response fields are selected: provider credentials,
        headers, API URLs and arbitrary provider/error objects are never serialized.
        The run/branch-local directory is outside mutable checkpoint restore layers.
        """
        root = self._failure_evidence_dir
        if root is None:
            return
        directory_fd = None
        temporary = None
        try:
            root.mkdir(mode=0o700, exist_ok=True)
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(directory_fd)
            if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.geteuid():
                raise PermissionError("private evidence directory ownership/mode mismatch")
            evidence = {
                "schema": "mem0_extraction_failure_evidence/1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "logical_task_key": self._logical_task_key,
                "contract_attempt": attempt, "reason": reason,
                "run_status": "not_evaluated",
                "request": {"model": self._model, "messages": messages,
                    "response_format": response_format, "tools": tools,
                    "tool_choice": tool_choice if tools else None, "max_tokens": self.extraction_max_output_tokens,
                    "temperature": 0.1},
                "response": {"content": response.content, "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "reasoning_content": response.reasoning_content},
                "native_forwarding": {
                    "content": response.content or "",
                    "none_normalized_to_empty_string": response.content is None,
                } if response.finish_reason == "length" and not tools else None,
                "validation_error": None if cause is None else {
                    "type": type(cause).__name__, "message": str(cause)},
            }
            payload = (json.dumps(evidence, ensure_ascii=False, indent=2) + "\n").encode()
            name = uuid.uuid4().hex + ".json"
            candidate = "." + name + ".pending"
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory_fd)
            temporary = candidate
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # link publishes a complete file and refuses any existing destination.
            os.link(temporary, name, src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd, follow_symlinks=False)
            os.unlink(temporary, dir_fd=directory_fd)
            temporary = None
            os.fsync(directory_fd)
        except Exception as exc:
            # Never log request/response/credential material or replace the original
            # extraction error when diagnostics storage is unavailable.
            logging.getLogger(__name__).warning(
                "Mem0 private failure evidence unavailable: %s", type(exc).__name__)
        finally:
            if directory_fd is not None:
                if temporary is not None:
                    try:
                        os.unlink(temporary, dir_fd=directory_fd)
                    except OSError:
                        pass
                os.close(directory_fd)

    def set_logical_task_key(self, value: str) -> None:
        if not value.strip():
            raise ValueError(
                f"Mem0 logical task key must not be empty: value={value!r}"
            )
        self._logical_task_key = value
        self._request_succeeded = False
        self._length_response_evidence = None

    @property
    def request_succeeded(self) -> bool:
        return self._request_succeeded

    def _call_provider(
        self,
        *,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any],
        task_identity: LLMTaskIdentity,
    ) -> LLMResponse:
        self._provider._response_format = response_format
        try:
            return asyncio.run_coroutine_threadsafe(
                self._provider.chat_with_retry(
                    messages=messages,
                    tools=tools,
                    model=self._model,
                    max_tokens=self.extraction_max_output_tokens,
                    temperature=0.1,
                    tool_choice=tool_choice if tools else None,
                    task_identity=task_identity,
                ),
                self._loop,
            ).result()
        finally:
            self._provider._response_format = None

    def generate_response(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        **_: Any,
    ) -> str | dict[str, Any]:
        if self._logical_task_key is None:
            raise RuntimeError(
                "Mem0 extraction request has no logical task identity"
            )
        task_identity = LLMTaskIdentity(logical_task_key=self._logical_task_key)
        last_contract_error: ValueError | None = None
        for attempt in range(1, 4):
            response = self._call_provider(
                messages=messages,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                task_identity=task_identity,
            )
            if response.finish_reason in {"error", "length"}:
                self._record_failure_evidence(
                    response=response, messages=messages, response_format=response_format,
                    tools=tools, tool_choice=tool_choice, attempt=attempt,
                    reason="length_observed" if response.finish_reason == "length" else "error",
                )
            if response.finish_reason == "error":
                raise RuntimeError(
                    "Mem0 extraction LM request failed: "
                    f"model={self._model!r} attempt={attempt}/3 "
                    f"detail={response.error_text[:500]!r}"
                )
            if response.finish_reason == "length":
                # Native Mem0 owns parsing and the resulting no-change/write/error
                # decision. Keep the original evidence for any downstream failure.
                self._length_response_evidence = {
                    "response": response, "messages": messages,
                    "response_format": response_format, "tools": tools,
                    "tool_choice": tool_choice, "attempt": attempt,
                }
            if tools:
                self._request_succeeded = True
                return {
                    "content": response.content,
                    "tool_calls": [
                        {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.arguments, ensure_ascii=False
                            ),
                        }
                        for call in response.tool_calls
                    ],
                }
            content = response.content or ""
            if response.finish_reason == "length":
                self._request_succeeded = True
                return content
            try:
                if response_format == {"type": "json_object"}:
                    parsed = json.loads(content)
                    if (
                        not isinstance(parsed, dict)
                        or not isinstance(parsed.get("memory"), list)
                    ):
                        raise ValueError(
                            "response requires a memory list: "
                            f"value={parsed!r}"
                        )
                    for index, item in enumerate(parsed["memory"]):
                        if not isinstance(item, dict):
                            raise ValueError(
                                "memory item must be a dictionary: "
                                f"index={index} value={item!r}"
                            )
                        text = item.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(
                                "memory item requires non-empty text: "
                                f"index={index} value={text!r} item={item!r}"
                            )
            except (json.JSONDecodeError, ValueError) as exc:
                self._record_failure_evidence(
                    response=response, messages=messages, response_format=response_format,
                    tools=tools, tool_choice=tool_choice, attempt=attempt,
                    reason="json_contract", cause=exc,
                )
                last_contract_error = ValueError(
                    "Mem0 extraction LM returned an invalid response: "
                    f"model={self._model!r} attempt={attempt}/3 "
                    f"value={content[:160]!r}"
                )
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
                    continue
                raise last_contract_error from exc
            self._request_succeeded = True
            return content
        raise RuntimeError(
            "Mem0 extraction retry loop exited unexpectedly: "
            f"last_error={last_contract_error!r}"
        )


class _ValidatedMem0Client:
    def __init__(
        self,
        memory: Any,
        *,
        collection_name: str,
        qdrant_host: str,
        qdrant_port: int,
        expected_run_id: str,
        embedding_dimensions: int,
        qdrant_distance: str,
        qdrant_on_disk: bool,
        qdrant_bm25_enabled: bool,
        enable_entity_memory: bool,
    ) -> None:
        self._memory = memory
        self._collection_name = collection_name
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port
        self._expected_run_id = expected_run_id
        self._embedding_dimensions = embedding_dimensions
        self._qdrant_distance = qdrant_distance
        self._qdrant_on_disk = qdrant_on_disk
        self._qdrant_bm25_enabled = qdrant_bm25_enabled
        self._enable_entity_memory = enable_entity_memory

    @property
    def qdrant_admin(self) -> Any:
        """Reuse the normal vector-store client for checkpoint operations."""
        vector_store = getattr(self._memory, "vector_store", None)
        client = getattr(vector_store, "client", None)
        if client is not None:
            return client
        return qdrant_client(host=self._qdrant_host, port=self._qdrant_port)

    def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        _validate_mempa_add_scope(kwargs)
        if kwargs["run_id"] != self._expected_run_id:
            raise Mem0ResponseError(
                "Mem0 ADD run scope does not match the client-owned run: "
                f"expected_run_id={self._expected_run_id!r} "
                f"value={kwargs['run_id']!r}"
            )
        infer = kwargs.get("infer", True)
        if not isinstance(infer, bool):
            raise Mem0ResponseError(
                f"Mem0 ADD infer must be a boolean: value={infer!r}"
            )
        if not infer:
            _validated_raw_dialogue_messages(messages)
        result = self._memory.add(messages=messages, **kwargs)
        return self._validate_and_confirm_add(
            result,
            messages=messages,
            kwargs=kwargs,
        )

    def _validate_and_confirm_add(
        self,
        result: Any,
        *,
        messages: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        results = _validate_add_response(result)
        result_ids = [item["id"] for item in results]
        if len(result_ids) != len(set(result_ids)):
            raise Mem0ResponseError(
                "Mem0 ADD response contains duplicate memory IDs: "
                f"value={result_ids!r}"
            )
        infer = kwargs.get("infer", True)
        if not infer:
            _validate_raw_dialogue_results(results, messages)
        for item in results:
            memory_id = item["id"]
            persisted = self._memory.get(memory_id)
            if item["event"] in _PERSISTED_EVENTS:
                if not isinstance(persisted, dict) or persisted.get("id") != memory_id:
                    raise Mem0PersistenceError(
                        "Mem0 reported a persisted memory that cannot be read back: "
                        f"run_id={kwargs.get('run_id')!r} id={memory_id!r} "
                        f"event={item['event']!r} read_value={persisted!r}"
                    )
                self._validate_persisted_scope_and_content(
                    item,
                    persisted,
                    kwargs=kwargs,
                )
            elif persisted is not None:
                raise Mem0PersistenceError(
                    "Mem0 reported DELETE but the memory still exists: "
                    f"run_id={kwargs.get('run_id')!r} id={memory_id!r} "
                    f"read_value={persisted!r}"
                )
        return result

    @staticmethod
    def _validate_persisted_scope_and_content(
        item: dict[str, Any],
        persisted: dict[str, Any],
        *,
        kwargs: dict[str, Any],
    ) -> None:
        memory_id = item["id"]
        expected_fields = {
            "user_id": kwargs.get("user_id"),
            "agent_id": kwargs.get("agent_id"),
            "run_id": kwargs.get("run_id"),
            "memory": item["memory"],
        }
        item_role = item.get("role")
        if item_role is not None:
            expected_fields["role"] = item_role
        mismatches = {
            field_name: {
                "expected": expected_value,
                "actual": persisted.get(field_name),
            }
            for field_name, expected_value in expected_fields.items()
            if persisted.get(field_name) != expected_value
        }
        expected_metadata = kwargs.get("metadata") or {}
        persisted_metadata = persisted.get("metadata")
        if not isinstance(persisted_metadata, dict):
            mismatches["metadata"] = {
                "expected": expected_metadata,
                "actual": persisted_metadata,
            }
        else:
            metadata_mismatches = {
                key: {
                    "expected": value,
                    "actual": persisted_metadata.get(key),
                }
                for key, value in expected_metadata.items()
                if persisted_metadata.get(key) != value
            }
            if metadata_mismatches:
                mismatches["metadata"] = metadata_mismatches
        if mismatches:
            raise Mem0PersistenceError(
                "Mem0 read-back does not match the committed write: "
                f"id={memory_id!r} mismatches={mismatches!r}"
            )

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        required_scope = _validate_mempa_search_request(
            query,
            kwargs,
            expected_run_id=self._expected_run_id,
        )
        result = self._memory.search(query=query, **kwargs)
        results = _validate_search_response(result)
        _validate_mempa_search_scope(results, required_scope)
        return result

    def delete_all(self, **kwargs: Any) -> dict[str, Any]:
        return self._memory.delete_all(**kwargs)

    def health(self) -> float:
        started = time.monotonic()
        _probe_qdrant_collection(
            collection_name=self._collection_name,
            qdrant_host=self._qdrant_host,
            qdrant_port=self._qdrant_port,
            expected_run_id=self._expected_run_id,
            require_collection=True,
            allow_existing_collection=True,
            expected_embedding_dimensions=self._embedding_dimensions,
            expected_distance=self._qdrant_distance,
            expected_on_disk=self._qdrant_on_disk,
            expected_bm25_enabled=self._qdrant_bm25_enabled,
            expected_entity_memory_enabled=self._enable_entity_memory,
        )
        try:
            embedding = self._memory.embedding_model.embed(
                "MemPA embedding readiness probe",
                memory_action="search",
            )
        except Exception as exc:
            raise ConnectionError(
                "Mem0 embedding readiness check failed: "
                f"expected_dimensions={self._embedding_dimensions!r} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        _validate_embedding_vector(
            embedding,
            expected_dimensions=self._embedding_dimensions,
            operation="health",
        )
        return (time.monotonic() - started) * 1000

    def native_ids(self) -> set[str]:
        _, native_ids = _inspect_qdrant_collection(
            collection_name=self._collection_name,
            qdrant_host=self._qdrant_host,
            qdrant_port=self._qdrant_port,
            expected_run_id=self._expected_run_id,
            require_collection=True,
            expected_embedding_dimensions=self._embedding_dimensions,
            expected_distance=self._qdrant_distance,
            expected_on_disk=self._qdrant_on_disk,
            expected_bm25_enabled=self._qdrant_bm25_enabled,
            expected_entity_memory_enabled=self._enable_entity_memory,
        )
        return native_ids

    def native_entity_ids(self) -> set[str]:
        if not self._enable_entity_memory:
            return set()
        _, native_ids = _inspect_qdrant_collection(
            collection_name=f"{self._collection_name}_entities",
            qdrant_host=self._qdrant_host,
            qdrant_port=self._qdrant_port,
            expected_run_id=self._expected_run_id,
            require_collection=True,
            expected_embedding_dimensions=self._embedding_dimensions,
            expected_distance=self._qdrant_distance,
            expected_on_disk=self._qdrant_on_disk,
            expected_bm25_enabled=self._qdrant_bm25_enabled,
            expected_entity_memory_enabled=False,
        )
        return native_ids

    def history_native_ids(self) -> set[str]:
        db = getattr(self._memory, "db", None)
        connection = getattr(db, "connection", None)
        if connection is None:
            raise RuntimeError(
                "Mem0 history database connection is unavailable: "
                f"run_id={self._expected_run_id!r} db={db!r}"
            )
        try:
            rows = connection.execute(
                "SELECT DISTINCT memory_id FROM history"
            ).fetchall()
        except Exception as exc:
            raise RuntimeError(
                "Mem0 history database could not be inspected: "
                f"run_id={self._expected_run_id!r} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        native_ids: set[str] = set()
        for row_index, row in enumerate(rows):
            memory_id = row[0] if isinstance(row, (tuple, list)) and row else None
            if not isinstance(memory_id, str) or not memory_id.strip():
                raise Mem0ResponseError(
                    "Mem0 history database contains an invalid memory_id: "
                    f"row_index={row_index} value={row!r}"
                )
            native_ids.add(memory_id)
        return native_ids


class _ScopedMem0Client(_ValidatedMem0Client):
    def __init__(
        self,
        memory: Any,
        llm: _MemoryExtractionLLM,
        *,
        collection_name: str,
        qdrant_host: str,
        qdrant_port: int,
        expected_run_id: str,
        embedding_dimensions: int,
        qdrant_distance: str,
        qdrant_on_disk: bool,
        qdrant_bm25_enabled: bool,
        enable_entity_memory: bool,
    ) -> None:
        super().__init__(
            memory,
            collection_name=collection_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            expected_run_id=expected_run_id,
            embedding_dimensions=embedding_dimensions,
            qdrant_distance=qdrant_distance,
            qdrant_on_disk=qdrant_on_disk,
            qdrant_bm25_enabled=qdrant_bm25_enabled,
            enable_entity_memory=enable_entity_memory,
        )
        self._llm = llm

    def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        _validate_mempa_add_scope(kwargs)
        if kwargs["run_id"] != self._expected_run_id:
            raise Mem0ResponseError(
                "Mem0 ADD run scope does not match the client-owned run: "
                f"expected_run_id={self._expected_run_id!r} "
                f"value={kwargs['run_id']!r}"
            )
        infer = kwargs.get("infer", True)
        if infer is not True:
            raise Mem0ResponseError(
                "Mem0 extraction client requires infer=True: "
                f"value={infer!r}"
            )
        metadata = kwargs.get("metadata") or {}
        self._llm.set_logical_task_key(
            ":".join(
                (
                    "mem0_add",
                    str(kwargs.get("run_id") or ""),
                    str(metadata.get("session_id") or ""),
                    str(metadata.get("turn_id") or ""),
                )
            )
        )
        try:
            result = self._memory.add(messages=messages, **kwargs)
            if not self._llm.request_succeeded:
                raise RuntimeError(
                    "Mem0 extraction did not complete successfully: "
                    f"run_id={kwargs.get('run_id')!r} "
                    f"session_id={metadata.get('session_id')!r} "
                    f"turn_id={metadata.get('turn_id')!r}"
                )
            return self._validate_and_confirm_add(
                result,
                messages=messages,
                kwargs=kwargs,
            )
        except Exception as exc:
            evidence = getattr(self._llm, "_length_response_evidence", None)
            if evidence is None:
                raise
            self._llm._record_failure_evidence(
                **evidence, reason="native_memory_failure_after_length", cause=exc,
            )
            self._llm._request_succeeded = False
            # Vendor parse/persistence errors can embed the entire response.
            # Keep the real failure privately without exposing its exception chain
            # in the ordinary producer traceback; never retry or accept it here.
            raise RuntimeError(
                "Mem0 native processing failed after length response: "
                f"model={self._llm._model!r} task={self._llm._logical_task_key!r} "
                f"attempt={evidence['attempt']}/3 native_error_type={type(exc).__name__} "
                f"content_chars={len(evidence['response'].content or '')}"
            ) from None



def _require_embedding_api_key() -> str:
    embedding_api_key = os.getenv("OPENAI_EMB_API_KEY")
    if not embedding_api_key:
        raise ValueError(
            "OpenAI embedding API key is required: "
            "OPENAI_EMB_API_KEY is unset"
        )
    return embedding_api_key


def _drop_qdrant_collection_for_run(
    *,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    expected_run_id: str,
) -> None:
    """Drop this run's own collection (and its entity sibling) before creating it.

    Only ever called with `--overwrite`, which is already deleting the run
    directory: the history database, the ledger and every transcript for
    `expected_run_id` go with it, so leaving the Qdrant collection behind is
    what makes a run that died at start-up unretryable under its own run id.
    Both names are derived from that run id alone, so nothing another run owns
    is in reach.
    """
    qdrant = qdrant_client(host=qdrant_host, port=qdrant_port)
    dropped = [
        name
        for name in (collection_name, f"{collection_name}_entities")
        if qdrant.collection_exists(name)
    ]
    for name in dropped:
        qdrant.delete_collection(name)
    print(
        "overwrite: dropped Qdrant collections "
        f"{dropped!r} for run_id={expected_run_id!r}",
        flush=True,
    )


def _assert_qdrant_collection_ready(
    *,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    expected_run_id: str,
    allow_existing_collection: bool,
    drop_existing_collection: bool = False,
    expected_embedding_dimensions: int,
    expected_distance: str,
    expected_on_disk: bool,
    expected_bm25_enabled: bool,
    expected_entity_memory_enabled: bool,
) -> None:
    if drop_existing_collection:
        _drop_qdrant_collection_for_run(
            collection_name=collection_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            expected_run_id=expected_run_id,
        )
    _probe_qdrant_collection(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        expected_run_id=expected_run_id,
        require_collection=False,
        allow_existing_collection=allow_existing_collection,
        expected_embedding_dimensions=expected_embedding_dimensions,
        expected_distance=expected_distance,
        expected_on_disk=expected_on_disk,
        expected_bm25_enabled=expected_bm25_enabled,
        expected_entity_memory_enabled=expected_entity_memory_enabled,
    )


def _probe_qdrant_collection(
    *,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    expected_run_id: str,
    require_collection: bool,
    allow_existing_collection: bool,
    expected_embedding_dimensions: int,
    expected_distance: str,
    expected_on_disk: bool,
    expected_bm25_enabled: bool,
    expected_entity_memory_enabled: bool,
) -> float:
    latency_ms, _ = _inspect_qdrant_collection(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        expected_run_id=expected_run_id,
        require_collection=require_collection,
        allow_existing_collection=allow_existing_collection,
        expected_embedding_dimensions=expected_embedding_dimensions,
        expected_distance=expected_distance,
        expected_on_disk=expected_on_disk,
        expected_bm25_enabled=expected_bm25_enabled,
        expected_entity_memory_enabled=expected_entity_memory_enabled,
    )
    return latency_ms


def _inspect_qdrant_collection(
    *,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    expected_run_id: str,
    require_collection: bool,
    allow_existing_collection: bool = True,
    expected_embedding_dimensions: int,
    expected_distance: str,
    expected_on_disk: bool,
    expected_bm25_enabled: bool,
    expected_entity_memory_enabled: bool,
) -> tuple[float, set[str]]:
    started = time.monotonic()
    qdrant = qdrant_client(host=qdrant_host, port=qdrant_port)
    try:
        qdrant.get_collections()
    except Exception as exc:
        raise ConnectionError(
            "Qdrant readiness check failed: "
            f"host={qdrant_host!r} port={qdrant_port!r} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    collection_exists = qdrant.collection_exists(collection_name)
    entity_collection_name = f"{collection_name}_entities"
    entity_collection_exists = qdrant.collection_exists(entity_collection_name)
    if not expected_entity_memory_enabled and entity_collection_exists:
        raise ValueError(
            "Entity memory is disabled but its Qdrant collection exists: "
            f"collection_name={entity_collection_name!r} "
            f"expected_run_id={expected_run_id!r}"
        )
    if require_collection and not collection_exists:
        raise FileNotFoundError(
            "Expected Mem0 collection does not exist: "
            f"collection_name={collection_name!r} host={qdrant_host!r} "
            f"port={qdrant_port!r} expected_run_id={expected_run_id!r}"
        )
    if collection_exists and not allow_existing_collection:
        raise FileExistsError(
            "Refusing to reuse an existing Mem0 collection for a new run: "
            f"collection_name={collection_name!r} "
            f"expected_run_id={expected_run_id!r}"
        )
    if not collection_exists:
        return (time.monotonic() - started) * 1000, set()
    collection_info = qdrant.get_collection(collection_name)
    params = getattr(getattr(collection_info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    actual_dimensions = getattr(vectors, "size", None)
    actual_distance_value = getattr(vectors, "distance", None)
    actual_distance = getattr(actual_distance_value, "value", actual_distance_value)
    actual_on_disk = getattr(vectors, "on_disk", None)
    sparse_vectors = getattr(params, "sparse_vectors", None)
    actual_bm25_enabled = bool(sparse_vectors and "bm25" in sparse_vectors)
    schema_mismatches = {
        field_name: {"expected": expected_value, "actual": actual_value}
        for field_name, expected_value, actual_value in (
            (
                "embedding_dimensions",
                expected_embedding_dimensions,
                actual_dimensions,
            ),
            ("distance", expected_distance, actual_distance),
            ("on_disk", expected_on_disk, actual_on_disk),
            ("bm25_enabled", expected_bm25_enabled, actual_bm25_enabled),
        )
        if actual_value != expected_value
    }
    if schema_mismatches:
        raise ValueError(
            "Existing Mem0 collection schema does not match the frozen runtime "
            f"configuration: collection_name={collection_name!r} "
            f"mismatches={schema_mismatches!r}"
        )
    next_page_offset: Any | None = None
    native_ids: set[str] = set()
    while True:
        existing_points, next_page_offset = qdrant.scroll(
            collection_name=collection_name,
            limit=256,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in existing_points:
            point_id = getattr(point, "id", None)
            if not isinstance(point_id, str) or not point_id.strip():
                raise ValueError(
                    "Existing Mem0 collection contains a point with an invalid "
                    f"ID: collection_name={collection_name!r} value={point_id!r}"
                )
            if point_id in native_ids:
                raise ValueError(
                    "Existing Mem0 collection returned a duplicate point ID: "
                    f"collection_name={collection_name!r} value={point_id!r}"
                )
            native_ids.add(point_id)
            payload = getattr(point, "payload", None)
            if not isinstance(payload, dict):
                raise ValueError(
                    "Existing Mem0 collection point payload must be a dictionary: "
                    f"collection_name={collection_name!r} point_id={point_id!r} "
                    f"value={payload!r}"
                )
            stored_run_id = payload.get("run_id")
            if stored_run_id != expected_run_id:
                raise ValueError(
                    "Existing Mem0 collection contains a point owned by a "
                    "different run: "
                    f"collection_name={collection_name!r} "
                    f"expected_run_id={expected_run_id!r} "
                    f"point_id={point_id!r} "
                    f"stored_run_id={stored_run_id!r}"
                )
        if next_page_offset is None:
            break
    return (time.monotonic() - started) * 1000, native_ids


def _load_mem0_memory() -> Any:
    try:
        from mem0 import Memory  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "mem0ai is required for the mem0 adapter. "
            "Install with: pip install 'mempabench[mem0]'"
        ) from exc
    return Memory


def _prepare_history_db_path(
    history_db_path: Path,
    *,
    allow_existing_history_db: bool,
    require_existing_history_db: bool,
) -> None:
    if not history_db_path.is_absolute():
        raise ValueError(
            "Mem0 history_db_path must be absolute: "
            f"value={str(history_db_path)!r}"
        )
    if require_existing_history_db and not history_db_path.is_file():
        raise FileNotFoundError(
            "Expected Mem0 history database does not exist: "
            f"path={str(history_db_path)!r}"
        )
    if history_db_path.exists() and not history_db_path.is_file():
        raise ValueError(
            "Mem0 history_db_path exists but is not a regular file: "
            f"path={str(history_db_path)!r}"
        )
    if history_db_path.exists() and not allow_existing_history_db:
        raise FileExistsError(
            "Refusing to reuse an existing Mem0 history database for a new run: "
            f"path={str(history_db_path)!r}"
        )
    history_db_path.parent.mkdir(parents=True, exist_ok=True)


def _storage_config(
    *,
    embedding_api_key: str,
    embedding_api_base: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    qdrant_distance: str,
    qdrant_on_disk: bool,
    qdrant_bm25_enabled: bool,
    enable_entity_memory: bool,
    history_db_path: Path,
) -> dict[str, Any]:
    if embedding_provider != OPENAI_EMBEDDING_PROVIDER:
        raise ValueError(
            "Mem0 client currently requires the OpenAI embedding provider: "
            f"value={embedding_provider!r}"
        )
    normalized_embedding_api_base = embedding_api_base.strip().rstrip("/")
    parsed_embedding_api_base = urlsplit(normalized_embedding_api_base)
    if (
        parsed_embedding_api_base.scheme not in {"http", "https"}
        or not parsed_embedding_api_base.netloc
        or parsed_embedding_api_base.username is not None
        or parsed_embedding_api_base.password is not None
    ):
        raise ValueError(
            "Mem0 embedding_api_base must be an absolute HTTP(S) URL without "
            f"credentials: value={embedding_api_base!r}"
        )
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError(
            f"Mem0 embedding_model must not be empty: value={embedding_model!r}"
        )
    if (
        isinstance(embedding_dimensions, bool)
        or not isinstance(embedding_dimensions, int)
        or embedding_dimensions <= 0
    ):
        raise ValueError(
            "Mem0 embedding_dimensions must be a positive integer: "
            f"value={embedding_dimensions!r}"
        )
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError(
            f"Mem0 collection_name must not be empty: value={collection_name!r}"
        )
    if not isinstance(qdrant_host, str) or not qdrant_host.strip():
        raise ValueError(
            f"Mem0 qdrant_host must not be empty: value={qdrant_host!r}"
        )
    if (
        isinstance(qdrant_port, bool)
        or not isinstance(qdrant_port, int)
        or not 1 <= qdrant_port <= 65535
    ):
        raise ValueError(
            f"Mem0 qdrant_port must be between 1 and 65535: value={qdrant_port!r}"
        )
    if not isinstance(qdrant_on_disk, bool):
        raise ValueError(
            f"Mem0 qdrant_on_disk must be a boolean: value={qdrant_on_disk!r}"
        )
    if not isinstance(qdrant_bm25_enabled, bool):
        raise ValueError(
            "Mem0 qdrant_bm25_enabled must be a boolean: "
            f"value={qdrant_bm25_enabled!r}"
        )
    if not isinstance(enable_entity_memory, bool):
        raise ValueError(
            "Mem0 enable_entity_memory must be a boolean: "
            f"value={enable_entity_memory!r}"
        )
    if qdrant_distance != MEM0_QDRANT_DISTANCE:
        raise ValueError(
            "Mem0 client currently requires cosine Qdrant distance: "
            f"value={qdrant_distance!r}"
        )
    return {
        "history_db_path": str(history_db_path),
        "enable_entity_memory": enable_entity_memory,
        "embedder": {
            "provider": embedding_provider,
            "config": {
                "model": embedding_model,
                "embedding_dims": embedding_dimensions,
                "api_key": embedding_api_key,
                "openai_base_url": normalized_embedding_api_base,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": embedding_dimensions,
                "host": qdrant_host,
                "port": qdrant_port,
                "client": qdrant_client(host=qdrant_host, port=qdrant_port),
                "path": None,
                "on_disk": qdrant_on_disk,
                "enable_bm25": qdrant_bm25_enabled,
            },
        },
    }


def create_simple_rag_mem0_client(
    *,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    expected_run_id: str,
    allow_existing_collection: bool,
    drop_existing_collection: bool = False,
    embedding_api_base: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
    qdrant_distance: str,
    qdrant_on_disk: bool,
    qdrant_bm25_enabled: bool,
    enable_entity_memory: bool,
    history_db_path: Path,
    allow_existing_history_db: bool,
    require_existing_history_db: bool,
) -> Mem0Client:
    embedding_api_key = _require_embedding_api_key()
    _prepare_history_db_path(
        history_db_path,
        allow_existing_history_db=allow_existing_history_db,
        require_existing_history_db=require_existing_history_db,
    )
    _assert_qdrant_collection_ready(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        expected_run_id=expected_run_id,
        allow_existing_collection=allow_existing_collection,
        drop_existing_collection=drop_existing_collection,
        expected_embedding_dimensions=embedding_dimensions,
        expected_distance=qdrant_distance,
        expected_on_disk=qdrant_on_disk,
        expected_bm25_enabled=qdrant_bm25_enabled,
        expected_entity_memory_enabled=enable_entity_memory,
    )
    memory_class = _load_mem0_memory()
    memory = memory_class.from_config(
        {
            "initialize_llm": False,
            **_storage_config(
                embedding_api_key=embedding_api_key,
                embedding_api_base=embedding_api_base,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
                collection_name=collection_name,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                qdrant_distance=qdrant_distance,
                qdrant_on_disk=qdrant_on_disk,
                qdrant_bm25_enabled=qdrant_bm25_enabled,
                enable_entity_memory=enable_entity_memory,
                history_db_path=history_db_path,
            ),
        }
    )
    return _ValidatedMem0Client(
        memory,
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        expected_run_id=expected_run_id,
        embedding_dimensions=embedding_dimensions,
        qdrant_distance=qdrant_distance,
        qdrant_on_disk=qdrant_on_disk,
        qdrant_bm25_enabled=qdrant_bm25_enabled,
        enable_entity_memory=enable_entity_memory,
    )


def create_read_only_mem0_client(*, infer: bool, **kwargs: Any) -> Mem0Client:
    """Open one restored checkpoint for all Probe consumers of a setting.

    Accepts the normal client factory arguments; extraction settings are kept
    by the caller's frozen plan but no extraction provider is opened here.
    Set ``allow_missing_entity_collection`` only when checkpoint metadata
    explicitly records a null entity snapshot (an enabled but empty index).
    The owner must drain/close this client before deleting its namespace.
    """
    from memory.backends.mem0_read_only import create_read_only_client

    return create_read_only_client(infer=infer, **kwargs)


def _require_entity_nlp() -> None:
    """Reject a missing or drifting NLP runtime before creating backend state."""
    model_name, model_version = MEM0_ENTITY_MODEL.split("@")
    try:
        actual = (version("spacy"), version(model_name))
    except PackageNotFoundError as exc:
        raise RuntimeError("Mem0 entity memory requires the locked spaCy and English model; run uv sync --locked") from exc
    if actual != (MEM0_SPACY_VERSION, model_version):
        raise RuntimeError(f"Mem0 entity NLP versions differ from the frozen runtime: {actual!r}")
    from mem0.utils.spacy_models import get_nlp_full
    if get_nlp_full() is None:
        raise RuntimeError("Mem0 entity NLP model failed to load")


def create_mem0_client(
    *,
    provider_name: str,
    api_key: str,
    api_base: str,
    memory_side_model: str,
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    expected_run_id: str,
    allow_existing_collection: bool,
    drop_existing_collection: bool = False,
    embedding_api_base: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
    qdrant_distance: str,
    qdrant_on_disk: bool,
    qdrant_bm25_enabled: bool,
    enable_entity_memory: bool,
    history_db_path: Path,
    allow_existing_history_db: bool,
    require_existing_history_db: bool,
    extraction_max_output_tokens: int = MEM0_EXTRACTION_MAX_OUTPUT_TOKENS,
    provider_avoidance: ProviderAvoidance | None = None,
) -> Mem0Client:
    if type(extraction_max_output_tokens) is not int or extraction_max_output_tokens <= 0:
        raise ValueError("Mem0 extraction output budget must be a positive integer")
    if enable_entity_memory:
        _require_entity_nlp()
    embedding_api_key = _require_embedding_api_key()
    _prepare_history_db_path(
        history_db_path,
        allow_existing_history_db=allow_existing_history_db,
        require_existing_history_db=require_existing_history_db,
    )
    _assert_qdrant_collection_ready(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        expected_run_id=expected_run_id,
        allow_existing_collection=allow_existing_collection,
        drop_existing_collection=drop_existing_collection,
        expected_embedding_dimensions=embedding_dimensions,
        expected_distance=qdrant_distance,
        expected_on_disk=qdrant_on_disk,
        expected_bm25_enabled=qdrant_bm25_enabled,
        expected_entity_memory_enabled=enable_entity_memory,
    )
    memory_class = _load_mem0_memory()

    history_relative = Path(MEM0_HISTORY_DB_RELATIVE_PATH)
    failure_evidence_dir = None
    if tuple(history_db_path.parts[-len(history_relative.parts):]) == history_relative.parts:
        diagnostic_scope = history_db_path
        for _ in history_relative.parts:
            diagnostic_scope = diagnostic_scope.parent
        failure_evidence_dir = diagnostic_scope / "private_memory_diagnostics"
    else:
        # Diagnostics must not narrow the existing custom-history-path API.
        # Never guess a scope or put private response data into backend_state.
        logging.getLogger(__name__).warning(
            "Mem0 private failure evidence disabled for nonstandard history layout")
    llm = _MemoryExtractionLLM(
        provider_name=provider_name,
        api_key=api_key,
        api_base=api_base,
        model=memory_side_model,
        extraction_max_output_tokens=extraction_max_output_tokens,
        failure_evidence_dir=failure_evidence_dir,
        provider_avoidance=provider_avoidance,
    )
    config = {
        "initialize_llm": False,
        "llm": {
            "provider": "openai",
            "config": {
                "model": memory_side_model,
                "api_key": api_key,
                "openai_base_url": api_base,
            },
        },
        **_storage_config(
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            collection_name=collection_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_distance=qdrant_distance,
            qdrant_on_disk=qdrant_on_disk,
            qdrant_bm25_enabled=qdrant_bm25_enabled,
            enable_entity_memory=enable_entity_memory,
            history_db_path=history_db_path,
        ),
    }
    memory = memory_class.from_config(config)
    memory.llm = llm
    return _ScopedMem0Client(
        memory,
        llm,
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        expected_run_id=expected_run_id,
        embedding_dimensions=embedding_dimensions,
        qdrant_distance=qdrant_distance,
        qdrant_on_disk=qdrant_on_disk,
        qdrant_bm25_enabled=qdrant_bm25_enabled,
        enable_entity_memory=enable_entity_memory,
    )


# ── checkpoint support ────────────────────────────────────────────
#
# A checkpoint has to capture the Qdrant collection as well as the local
# history database, and a probe branch has to come up on a collection of its
# own so nothing it writes reaches the mainline one.

QDRANT_SNAPSHOT_ROOT = "/qdrant/snapshots"


def _qdrant_admin(client: Any) -> Any:
    """Return a Qdrant handle for the endpoint this client is pinned to.

    A client may expose `qdrant_admin` to supply its own handle; otherwise the
    handle is built from the host and port frozen into the client.
    """
    override = getattr(client, "qdrant_admin", None)
    if override is not None:
        return override
    host = getattr(client, "_qdrant_host", None)
    port = getattr(client, "_qdrant_port", None)
    if not isinstance(host, str) or not host or not isinstance(port, int):
        raise TypeError(
            "Mem0 client does not expose a Qdrant endpoint to administer: "
            f"host={host!r} port={port!r}"
        )
    return qdrant_client(host=host, port=port)


def _snapshot_location(snapshot_ref: str) -> str:
    if "://" in snapshot_ref:
        return snapshot_ref
    return f"file://{QDRANT_SNAPSHOT_ROOT}/{snapshot_ref.lstrip('/')}"


def collection_exists(client: Any, collection_name: str) -> bool:
    return bool(_qdrant_admin(client).collection_exists(collection_name))


def snapshot_collection(client: Any, collection_name: str) -> str:
    """Snapshot one Qdrant collection and return a reference to it.

    The reference is `<collection>/<snapshot>`, which is what both the
    same-name restore and the branch restore feed back to Qdrant.
    """
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError(
            f"Qdrant snapshot requires a collection name: value={collection_name!r}"
        )
    description = _qdrant_admin(client).create_snapshot(
        collection_name=collection_name
    )
    name = (
        description.get("name")
        if isinstance(description, dict)
        else getattr(description, "name", None)
    )
    if not isinstance(name, str) or not name.strip():
        raise Mem0ResponseError(
            "Qdrant snapshot response is missing a snapshot name: "
            f"collection_name={collection_name!r} value={description!r}"
        )
    return f"{collection_name}/{name}"


def delete_collection(client: Any, name: str) -> None:
    """Drop one Qdrant collection; a restore must not merge into a dirty one."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Qdrant delete requires a collection name: value={name!r}"
        )
    _qdrant_admin(client).delete_collection(collection_name=name)


def restore_collection_as(client: Any, snapshot_ref: str, target_name: str) -> None:
    """Recover a snapshot into `target_name`, leaving the snapshot in place."""
    if not isinstance(snapshot_ref, str) or not snapshot_ref.strip():
        raise ValueError(
            f"Qdrant restore requires a snapshot reference: value={snapshot_ref!r}"
        )
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError(
            f"Qdrant restore requires a target collection: value={target_name!r}"
        )
    _qdrant_admin(client).recover_snapshot(
        collection_name=target_name,
        location=_snapshot_location(snapshot_ref),
    )
