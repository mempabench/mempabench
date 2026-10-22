from __future__ import annotations

from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from memory.backends.mem0_client import _ValidatedMem0Client, _storage_config
from memory.backends.qdrant_runtime import (
    QDRANT_OPERATION_TIMEOUT_SECONDS,
    QDRANT_RESTORE_LOCK_PATH_ENV,
    qdrant_client,
    qdrant_restore_lease,
)


def _lease_worker(
    lock_path: str,
    start,
    active_restore,
    max_restore,
    active_execution,
    max_execution,
    counter_lock,
) -> None:
    os.environ[QDRANT_RESTORE_LOCK_PATH_ENV] = lock_path
    start.wait()
    with qdrant_restore_lease():
        with counter_lock:
            active_restore.value += 1
            max_restore.value = max(max_restore.value, active_restore.value)
        time.sleep(0.05)
        with counter_lock:
            active_restore.value -= 1
    with counter_lock:
        active_execution.value += 1
        max_execution.value = max(max_execution.value, active_execution.value)
    time.sleep(0.2)
    with counter_lock:
        active_execution.value -= 1


def test_normal_qdrant_client_has_explicit_120s_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    module = ModuleType("qdrant_client")
    module.QdrantClient = FakeQdrantClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", module)

    qdrant_client(host="127.0.0.1", port=6333)

    assert captured == [
        {
            "host": "127.0.0.1",
            "port": 6333,
            "timeout": QDRANT_OPERATION_TIMEOUT_SECONDS,
        }
    ]
    assert QDRANT_OPERATION_TIMEOUT_SECONDS == 120


def test_storage_config_supplies_one_normal_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = object()
    calls: list[tuple[str, int]] = []

    def factory(*, host: str, port: int) -> object:
        calls.append((host, port))
        return client

    monkeypatch.setattr("memory.backends.mem0_client.qdrant_client", factory)
    config = _storage_config(
        embedding_api_key="key",
        embedding_api_base="https://example.test/v1",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        collection_name="owned",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        qdrant_distance="Cosine",
        qdrant_on_disk=True,
        qdrant_bm25_enabled=False,
        enable_entity_memory=True,
        history_db_path=tmp_path / "history.sqlite3",
    )

    assert calls == [("127.0.0.1", 6333)]
    assert config["vector_store"]["config"]["client"] is client


def test_mem0_entity_store_reuses_primary_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mem0.configs.vector_stores.qdrant import QdrantConfig
    from mem0.memory.main import Memory
    from qdrant_client import QdrantClient

    client = QdrantClient.__new__(QdrantClient)
    config = QdrantConfig(
        collection_name="owned",
        embedding_model_dims=1536,
        client=client,
        host="127.0.0.1",
        port=6333,
        path=None,
    )
    memory = Memory.__new__(Memory)
    memory._entity_store = None
    memory.collection_name = "owned"
    memory.config = SimpleNamespace(
        vector_store=SimpleNamespace(provider="qdrant", config=config)
    )
    memory.vector_store = SimpleNamespace(client=client)
    captured: dict[str, Any] = {}

    def create(_provider: str, entity_config: Any) -> object:
        captured["config"] = entity_config
        return object()

    monkeypatch.setattr("mem0.memory.main.VectorStoreFactory.create", create)

    _ = memory.entity_store

    assert captured["config"].collection_name == "owned_entities"
    assert captured["config"].client is client


@pytest.mark.parametrize(
    ("initialize_llm", "enable_entity_memory"),
    [(False, False), (True, True)],
)
def test_real_mem0_config_chain_preserves_runtime_qdrant_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    initialize_llm: bool,
    enable_entity_memory: bool,
) -> None:
    from mem0.memory.main import Memory
    from mem0.utils import factory as mem0_factory
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    created: list[dict[str, Any]] = []

    class FakeVectorStore:
        def __init__(self, **config: Any) -> None:
            created.append(config)
            self.client = config["client"]

    original_load_class = mem0_factory.load_class

    def load_class(class_type: str):
        if class_type == mem0_factory.VectorStoreFactory.provider_to_class["qdrant"]:
            return FakeVectorStore
        return original_load_class(class_type)

    monkeypatch.setattr(
        "memory.backends.mem0_client.qdrant_client",
        lambda **_kwargs: client,
    )
    monkeypatch.setattr("mem0.utils.factory.load_class", load_class)
    monkeypatch.setattr(
        "mem0.memory.main.EmbedderFactory.create",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "mem0.memory.main.LlmFactory.create",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr("mem0.memory.main.MEM0_TELEMETRY", False)
    monkeypatch.setattr("mem0.memory.main.capture_event", lambda *_args: None)
    config = _storage_config(
        embedding_api_key="key",
        embedding_api_base="https://example.test/v1",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        collection_name="owned",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        qdrant_distance="Cosine",
        qdrant_on_disk=True,
        qdrant_bm25_enabled=False,
        enable_entity_memory=enable_entity_memory,
        history_db_path=tmp_path / "history.sqlite3",
    )
    config["initialize_llm"] = initialize_llm

    try:
        memory = Memory.from_config(config)
        assert memory.config.vector_store.config.client is client
        assert memory.config.vector_store.config.model_dump()["client"] is client
        assert memory.vector_store.client is client
        assert created[0]["client"] is client
        if enable_entity_memory:
            assert memory.entity_store.client is client
            assert created[1]["client"] is client
    finally:
        client.close()


def test_validated_client_reuses_normal_client_for_admin_operations() -> None:
    client = object()
    memory = SimpleNamespace(vector_store=SimpleNamespace(client=client))
    validated = _ValidatedMem0Client(
        memory,
        collection_name="owned",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        embedding_dimensions=1536,
        qdrant_distance="Cosine",
        qdrant_on_disk=True,
        qdrant_bm25_enabled=False,
        enable_entity_memory=True,
    )

    assert validated.qdrant_admin is client


def test_mem0_write_failure_is_not_retried() -> None:
    class FailingMemory:
        calls = 0

        def add(self, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            raise TimeoutError("ambiguous write")

    memory = FailingMemory()
    client = _ValidatedMem0Client(
        memory,
        collection_name="owned",
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        expected_run_id="run_001",
        embedding_dimensions=1536,
        qdrant_distance="Cosine",
        qdrant_on_disk=True,
        qdrant_bm25_enabled=False,
        enable_entity_memory=False,
    )

    with pytest.raises(TimeoutError, match="ambiguous write"):
        client.add(
            messages=[{"role": "user", "content": "hello"}],
            user_id="user_a",
            run_id="run_001",
            metadata={
                "session_id": "acc_001",
                "turn_id": "beat_01:t0",
                "timestamp": "2026-09-20T00:00:00Z",
            },
            infer=False,
        )
    assert memory.calls == 1


def test_restore_lease_requires_an_absolute_configured_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(QDRANT_RESTORE_LOCK_PATH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="is unset"):
        with qdrant_restore_lease():
            pass

    monkeypatch.setenv(QDRANT_RESTORE_LOCK_PATH_ENV, "relative.lock")
    with pytest.raises(ValueError, match="must be absolute"):
        with qdrant_restore_lease():
            pass

    monkeypatch.setenv(
        QDRANT_RESTORE_LOCK_PATH_ENV,
        str(tmp_path / "missing" / "restore.lock"),
    )
    with pytest.raises(FileNotFoundError, match="parent does not exist"):
        with qdrant_restore_lease():
            pass

    monkeypatch.setenv(QDRANT_RESTORE_LOCK_PATH_ENV, str(tmp_path))
    with pytest.raises(OSError):
        with qdrant_restore_lease():
            pass


def test_restore_lease_releases_after_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with qdrant_restore_lease():
            raise RuntimeError("boom")
    with qdrant_restore_lease() as acquired:
        assert acquired == Path(os.environ[QDRANT_RESTORE_LOCK_PATH_ENV])


def test_four_workers_serialize_restore_but_overlap_execution(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    active_restore = ctx.Value("i", 0)
    max_restore = ctx.Value("i", 0)
    active_execution = ctx.Value("i", 0)
    max_execution = ctx.Value("i", 0)
    counter_lock = ctx.Lock()
    lock_path = str(tmp_path / "qdrant-service-restore.lock")
    processes = [
        ctx.Process(
            target=_lease_worker,
            args=(
                lock_path,
                start,
                active_restore,
                max_restore,
                active_execution,
                max_execution,
                counter_lock,
            ),
        )
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert max_restore.value == 1
    assert max_execution.value > 1
