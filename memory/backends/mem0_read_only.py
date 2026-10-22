"""Shared, read-only resources for Mem0 and SimpleRAG Probe checkpoints.

Only construction differs from the pinned Mem0 runtime. Search, fusion,
entity boosts, filtering, and response formatting use the vendor methods.
There are no query, embedding, or result caches.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from urllib.parse import unquote

import httpx

from memory.backends.mem0_client import (
    _ValidatedMem0Client,
    _load_mem0_memory,
    _require_embedding_api_key,
    _require_entity_nlp,
    _storage_config,
)

if TYPE_CHECKING:
    from harness.provider_avoidance import ProviderAvoidance


class Mem0UnresolvedRequestError(RuntimeError):
    """A Qdrant request may still be running after its local worker ended."""


class ReadOnlyQdrantTransport(httpx.BaseTransport):
    """Reject writes and other namespaces at the actual HTTP boundary."""

    _READ_POST_SUFFIXES = frozenset({
        "/points", "/points/query", "/points/query/batch", "/points/search",
        "/points/search/batch", "/points/scroll", "/points/count",
    })

    def __init__(self, namespaces: set[str], transport: httpx.BaseTransport) -> None:
        self._namespaces = frozenset(namespaces)
        self._transport = transport
        self.denied_requests = 0
        self._unresolved_remote_requests = 0
        self._lock = threading.Lock()

    @property
    def unresolved_remote_requests(self) -> int:
        with self._lock:
            return self._unresolved_remote_requests

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.unresolved_remote_requests:
            raise Mem0UnresolvedRequestError("Qdrant has an unresolved request; retain the shared namespace")
        # Decode exactly once, then require a literal namespace segment. The
        # SDK encodes collection names; encoded slashes cannot widen access.
        path = unquote(request.url.raw_path.split(b"?", 1)[0].decode("ascii"))
        parts = path.split("/")
        allowed = False
        if len(parts) >= 3 and parts[:2] == ["", "collections"]:
            namespace = parts[2]
            suffix = "/" + "/".join(parts[3:]) if len(parts) > 3 else ""
            if namespace in self._namespaces:
                allowed = (
                    request.method == "GET" and suffix in {"", "/exists"}
                ) or (
                    request.method == "POST" and suffix in self._READ_POST_SUFFIXES
                )
        if allowed and request.method == "POST" and request.content:
            # Qdrant query/recommend payloads can refer to another collection
            # via lookup_from despite using an allowed URL. Native Mem0 never
            # needs a cross-collection lookup.
            payload = json.loads(request.content)

            def permitted_lookup(value: Any) -> bool:
                if isinstance(value, dict):
                    lookup = value.get("lookup_from")
                    if lookup is not None and (
                        not isinstance(lookup, dict)
                        or lookup.get("collection") not in self._namespaces
                    ):
                        return False
                    return all(permitted_lookup(item) for item in value.values())
                if isinstance(value, list):
                    return all(permitted_lookup(item) for item in value)
                return True

            allowed = permitted_lookup(payload)
        if not allowed:
            with self._lock:
                self.denied_requests += 1
            raise PermissionError(
                f"Read-only Probe Qdrant request refused: {request.method} {path}"
            )
        response = None
        try:
            response = self._transport.handle_request(request)
            # Qdrant uses fully buffered JSON responses. Read inside the guard:
            # a timeout can happen after headers, while consuming the body.
            response.read()
            return response
        except httpx.HTTPError as exc:
            if not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                with self._lock:
                    self._unresolved_remote_requests += 1
            raise
        finally:
            if response is not None:
                response.close()

    def close(self) -> None:
        self._transport.close()


def _query_qdrant(*, host: str, port: int, namespaces: set[str]) -> Any:
    from qdrant_client import QdrantClient

    transport = ReadOnlyQdrantTransport(namespaces, httpx.HTTPTransport())
    client = QdrantClient(
        host=host, port=port, timeout=120, prefer_grpc=False,
        check_compatibility=False, trust_env=False,
        transport=transport,
    )
    client._probe_read_only_transport = transport
    return client


class _SerializedEncoder:
    """Keep a shared encoder's iterator state private to one encode call."""

    def __init__(self, encoder: Any) -> None:
        self._encoder = encoder
        self._lock = threading.Lock()

    def embed(self, texts: Any) -> Any:
        with self._lock:
            return list(self._encoder.embed(texts))


class _EmptyEntityStore:
    """An enabled entity index absent from the immutable source checkpoint."""

    def search(self, **kwargs: Any) -> list[Any]:
        return []


def _existing_store(
    client: Any, config: Any, collection_name: str, *, prepare_bm25: bool = True,
) -> Any:
    from mem0.vector_stores.qdrant import Qdrant

    class ExistingQdrant(Qdrant):
        def create_col(self, vector_size: int, on_disk: bool, **kwargs: Any) -> None:
            # Qdrant.__init__ normally creates payload indexes even for an
            # existing collection. Inspect the restored schema without doing
            # so; all query methods remain the original vendor methods.
            info = self.client.get_collection(self.collection_name)
            params = info.config.params
            vectors = params.vectors
            actual = (
                getattr(vectors, "size", None),
                getattr(getattr(vectors, "distance", None), "value", None),
                getattr(vectors, "on_disk", None),
                bool(params.sparse_vectors and "bm25" in params.sparse_vectors),
            )
            expected = (vector_size, "Cosine", on_disk, self.enable_bm25)
            if actual != expected:
                raise ValueError(
                    f"Restored Probe collection schema mismatch: "
                    f"{self.collection_name!r} expected={expected!r} actual={actual!r}"
                )
            self._has_bm25_slot = actual[3]

    store = ExistingQdrant(
        collection_name=collection_name, embedding_model_dims=config.embedding_model_dims,
        client=client, on_disk=config.on_disk, enable_bm25=config.enable_bm25,
    )
    if store.enable_bm25 and prepare_bm25:
        encoder = store._get_bm25_encoder()
        if encoder is None:
            raise RuntimeError("Frozen Probe BM25 encoder could not be initialized")
        store._bm25_encoder = _SerializedEncoder(encoder)
    return store


def _native_ids(client: Any, namespace: str, expected_run_id: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=namespace, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for point in points:
            if not isinstance(point.id, str) or not point.id or point.id in ids:
                raise ValueError(f"Invalid or duplicate Probe memory ID: {point.id!r}")
            if not isinstance(point.payload, dict) or point.payload.get("run_id") != expected_run_id:
                raise ValueError(f"Restored Probe memory belongs to another run: {point.id!r}")
            ids.add(point.id)
        if offset is None:
            return ids


class ReadOnlyMem0Client(_ValidatedMem0Client):
    """Owner-held client; adapters share it, but never own its lifecycle."""

    def __init__(self, memory: Any, *, qdrant: Any, history_path: Path, **kwargs: Any) -> None:
        super().__init__(memory, **kwargs)
        self._qdrant = qdrant
        self._history_path = history_path
        self._condition = threading.Condition()
        self._active = 0
        self._draining = False
        self._closed = False
        self._close_lock = threading.Lock()

    def ensure_query_ready(self) -> None:
        """Check admission locally, without opening or querying any resource."""
        with self._condition:
            self._require_remote_resolved()
            if self._draining or self._closed:
                raise RuntimeError("Shared Probe memory is draining or closed")

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._condition:
            self.ensure_query_ready()
            self._active += 1
        try:
            yield
            # Vendor keyword/entity retrieval can swallow transport failures.
            self._require_remote_resolved()
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _require_remote_resolved(self) -> None:
        guard = getattr(self._qdrant, "_probe_read_only_transport", None)
        if guard is not None and guard.unresolved_remote_requests:
            raise Mem0UnresolvedRequestError(
                "Qdrant requests may still be running on the service; retain the shared namespace: "
                f"count={guard.unresolved_remote_requests}"
            )

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        # Counts the synchronous operation, not the lifetime of its awaiter.
        # asyncio cancellation cannot make cleanup race a live worker thread.
        with self._operation():
            guard = getattr(self._qdrant, "_probe_read_only_transport", None)
            before = guard.denied_requests if guard is not None else 0
            result = super().search(query, **kwargs)
            if guard is not None and guard.denied_requests != before:
                # Vendor keyword/entity retrieval catches backend exceptions.
                # A rejected write must not become a silently degraded answer.
                raise PermissionError("Mem0 search attempted a forbidden Qdrant request")
            return result

    def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        raise PermissionError("Shared Probe memory does not permit add")

    def delete_all(self, **kwargs: Any) -> dict[str, Any]:
        raise PermissionError("Shared Probe memory does not permit delete/reset")

    def health(self) -> float:
        with self._operation():
            started = time.monotonic()
            self._qdrant.get_collection(self._collection_name)
            return (time.monotonic() - started) * 1000

    def native_ids(self) -> set[str]:
        with self._operation():
            return _native_ids(self._qdrant, self._collection_name, self._expected_run_id)

    def history_native_ids(self) -> set[str]:
        # Search does not access history. Explicit owner audits may read the
        # immutable snapshot directly without copying, migrations, or tables.
        with self._operation():
            connection = sqlite3.connect(
                self._history_path.as_uri() + "?mode=ro&immutable=1", uri=True,
            )
            try:
                ids = {row[0] for row in connection.execute("SELECT DISTINCT memory_id FROM history")}
                if any(not isinstance(value, str) or not value.strip() for value in ids):
                    raise ValueError("Invalid memory ID in Probe history snapshot")
                return ids
            finally:
                connection.close()

    def drain(self, timeout_seconds: float | None = None) -> bool:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("Drain timeout must be nonnegative")
        with self._condition:
            self._draining = True
            if not self._condition.wait_for(lambda: self._active == 0, timeout_seconds):
                return False
            self._require_remote_resolved()
            return True

    def close(self, timeout_seconds: float | None = None) -> bool:
        try:
            if not self.drain(timeout_seconds):
                return False
        except Mem0UnresolvedRequestError:
            self._close_local_resources()
            raise
        self._close_local_resources()
        return True

    def _close_local_resources(self) -> None:
        with self._close_lock:
            if not self._closed:
                self._qdrant.close()
                embedding_client = getattr(self._memory.embedding_model, "client", None)
                if embedding_client is not None:
                    embedding_client.close()
                self._closed = True


def create_read_only_client(
    *, infer: bool, collection_name: str, qdrant_host: str, qdrant_port: int,
    expected_run_id: str, allow_existing_collection: bool,
    embedding_api_base: str, embedding_provider: str, embedding_model: str,
    embedding_dimensions: int, qdrant_distance: str, qdrant_on_disk: bool,
    qdrant_bm25_enabled: bool, enable_entity_memory: bool,
    history_db_path: Path, allow_existing_history_db: bool,
    require_existing_history_db: bool, drop_existing_collection: bool = False,
    allow_missing_entity_collection: bool = False,
    provider_name: str | None = None, api_key: str | None = None,
    api_base: str | None = None, memory_side_model: str | None = None,
    extraction_max_output_tokens: int | None = None,
    provider_avoidance: ProviderAvoidance | None = None,
) -> ReadOnlyMem0Client:
    if not isinstance(infer, bool):
        raise ValueError("infer must be a boolean")
    if drop_existing_collection or not allow_existing_collection:
        raise ValueError("Read-only Probe clients require an already restored collection")
    history_path = Path(history_db_path)
    if not history_path.is_absolute():
        raise ValueError("Probe history snapshot path must be absolute")
    if require_existing_history_db and not history_path.is_file():
        raise FileNotFoundError(history_path)
    if enable_entity_memory:
        _require_entity_nlp()
    storage = _storage_config(
        embedding_api_key=_require_embedding_api_key(),
        embedding_api_base=embedding_api_base, embedding_provider=embedding_provider,
        embedding_model=embedding_model, embedding_dimensions=embedding_dimensions,
        collection_name=collection_name, qdrant_host=qdrant_host, qdrant_port=qdrant_port,
        qdrant_distance=qdrant_distance, qdrant_on_disk=qdrant_on_disk,
        qdrant_bm25_enabled=qdrant_bm25_enabled, enable_entity_memory=enable_entity_memory,
        history_db_path=history_path,
    )
    from mem0.configs.base import MemoryConfig
    from mem0.utils.factory import EmbedderFactory

    config = MemoryConfig(**storage, initialize_llm=False)
    namespaces = {collection_name, f"{collection_name}_entities"}
    qdrant = _query_qdrant(host=qdrant_host, port=qdrant_port, namespaces=namespaces)
    memory = None
    try:
        vector_store = _existing_store(qdrant, config.vector_store.config, collection_name)
        _native_ids(qdrant, collection_name, expected_run_id)
        entity_name = f"{collection_name}_entities"
        entity_exists = qdrant.collection_exists(entity_name)
        if entity_exists and not enable_entity_memory:
            raise ValueError("Entity collection exists although frozen entity memory is disabled")
        if enable_entity_memory and not entity_exists and not allow_missing_entity_collection:
            raise FileNotFoundError(
                "Restored Probe entity collection is missing; only a checkpoint "
                "with a null entity snapshot may use an empty entity store"
            )
        entity_store: Any = _EmptyEntityStore()
        if enable_entity_memory and entity_exists:
            # Native entity boosts only call dense search, never BM25.
            entity_store = _existing_store(
                qdrant, config.vector_store.config, entity_name, prepare_bm25=False,
            )
            _native_ids(qdrant, entity_name, expected_run_id)
        # The vendor constructor creates SQLite tables, telemetry stores and
        # indexes. Bind the same search methods to explicitly prepared query
        # dependencies instead. No vendor source or global factories change.
        memory_class = _load_mem0_memory()
        memory = memory_class.__new__(memory_class)
        memory.config = config
        memory.vector_store = vector_store
        memory.embedding_model = EmbedderFactory.create(
            config.embedder.provider, config.embedder.config, config.vector_store.config,
        )
        memory.llm = None
        memory.db = None
        memory.collection_name = collection_name
        memory.api_version = config.version
        memory.custom_instructions = config.custom_instructions
        memory.reranker = None
        memory._entity_store = entity_store
        return ReadOnlyMem0Client(
            memory, qdrant=qdrant, history_path=history_path,
            collection_name=collection_name, qdrant_host=qdrant_host,
            qdrant_port=qdrant_port, expected_run_id=expected_run_id,
            embedding_dimensions=embedding_dimensions, qdrant_distance=qdrant_distance,
            qdrant_on_disk=qdrant_on_disk, qdrant_bm25_enabled=qdrant_bm25_enabled,
            enable_entity_memory=enable_entity_memory,
        )
    except BaseException:
        qdrant.close()
        embedding_client = getattr(getattr(memory, "embedding_model", None), "client", None)
        if embedding_client is not None:
            embedding_client.close()
        raise
