from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import yaml

from memos.api.handlers.add_handler import AddHandler
from memos.api.product_models import APIADDRequest
from memos.configs.llm import OpenAILLMConfig
from memos.graph_dbs.neo4j_community import Neo4jCommunityGraphDB
from memos.llms.openai import OpenAILLM
from memos.mem_reader.simple_struct import SimpleStructMemReader
from memos.mempa_runtime import memory_extraction_task
from memory.backends.memos_client import MEMOS_SOURCE_SHA256


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _logged_text(logger: Mock) -> str:
    calls = logger.info.call_args_list + logger.warning.call_args_list + logger.error.call_args_list
    return "\n".join(str(call) for call in calls)


def test_memos_compose_publishes_ports_on_loopback_only() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "deploy/memos/compose.yaml").read_text())

    published_ports = [
        port
        for service in compose["services"].values()
        for port in service.get("ports", [])
    ]

    assert published_ports
    assert all(str(port).startswith("127.0.0.1:") for port in published_ports)


def test_memos_compose_declares_text_memory_profile_only() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "deploy/memos/compose.yaml").read_text())
    environment = compose["services"]["memos"]["environment"]

    assert environment["ENABLE_ACTIVATION_MEMORY"] == "false"
    assert environment["ENABLE_PREFERENCE_MEMORY"] == "false"
    assert not any(name.startswith("PREFERENCE_EXTRACTOR_") for name in environment)


def test_memos_source_digest_matches_copied_source_and_compose_contract() -> None:
    memos_root = PROJECT_ROOT / "memory_raw_repos" / "MemOS"
    verifier = runpy.run_path(
        str(memos_root / "docker" / "verify_mempa_source.py")
    )
    actual = verifier["source_sha256"](memos_root)
    compose = yaml.safe_load(
        (PROJECT_ROOT / "deploy/memos/compose.yaml").read_text()
    )
    service = compose["services"]["memos"]

    assert actual == MEMOS_SOURCE_SHA256
    assert service["build"]["args"]["MEMPA_MEMOS_SOURCE_SHA256"] == actual
    assert service["environment"]["MEMPA_MEMOS_SOURCE_SHA256"] == actual


def test_memos_source_verifier_fails_on_mismatch() -> None:
    memos_root = PROJECT_ROOT / "memory_raw_repos" / "MemOS"
    result = subprocess.run(
        [
            sys.executable,
            str(memos_root / "docker" / "verify_mempa_source.py"),
            "--expected",
            "0" * 64,
            "--base-path",
            str(memos_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "source digest mismatch" in result.stderr
    assert f"actual={MEMOS_SOURCE_SHA256!r}" in result.stderr


def test_add_handler_logs_structure_without_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    private_message = "PRIVATE_ADD_MESSAGE_SENTINEL"
    handler = AddHandler.__new__(AddHandler)
    handler.logger = Mock()
    cube_view = SimpleNamespace(
        add_memories=Mock(
            return_value=[
                {
                    "memory": "PRIVATE_EXTRACTED_MEMORY_SENTINEL",
                    "memory_id": "memory_1",
                    "cube_id": "cube_1",
                }
            ]
        )
    )
    monkeypatch.setattr(handler, "_build_cube_view", Mock(return_value=cube_view))
    request = APIADDRequest(
        user_id="user_1",
        session_id="session_1",
        task_id="task_1",
        writable_cube_ids=["cube_1"],
        async_mode="sync",
        mode="fine",
        messages=[{"role": "user", "content": private_message}],
        chat_history=[],
    )

    response = handler.handle_add_memories(request)

    assert response.data
    logs = _logged_text(handler.logger)
    assert private_message not in logs
    assert "PRIVATE_EXTRACTED_MEMORY_SENTINEL" not in logs
    handler.logger.info.assert_any_call(
        "[AddHandler] add requested: message_count=%d cube_count=%d mode=%s async_mode=%s",
        1,
        1,
        "fine",
        "sync",
    )


def test_openai_wrapper_logs_metadata_without_prompt_response_or_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from test_memory.test_memos_upstream_policy import _configure_runtime

    _configure_runtime(monkeypatch, tmp_path)
    private_prompt = "PRIVATE_LLM_PROMPT_SENTINEL"
    private_response = "PRIVATE_LLM_RESPONSE_SENTINEL"
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=private_response))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )
    create = Mock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm_logger = Mock()
    timer_logger = Mock()
    monkeypatch.setattr("memos.llms.openai.openai.Client", Mock(return_value=client))
    monkeypatch.setattr("memos.llms.openai.logger", llm_logger)
    monkeypatch.setattr("memos.utils.logger", timer_logger)
    llm = OpenAILLM(
        OpenAILLMConfig(
            model_name_or_path="openai/gpt-5.5",
            api_key="PRIVATE_AUTH_SENTINEL:secret",
            api_base="https://openrouter.ai/api/v1",
            max_tokens=321,
        )
    )

    with memory_extraction_task("run_1:session_1:turn_1"):
        result = llm.generate([{"role": "user", "content": private_prompt}])

    assert result == private_response
    logs = _logged_text(llm_logger) + _logged_text(timer_logger)
    assert private_prompt not in logs
    assert private_response not in logs
    assert "PRIVATE_AUTH_SENTINEL" not in logs
    assert "message_count" in logs


class _Result:
    def __init__(self, *, ids: list[str] | None = None, deleted_count: int | None = None):
        self._ids = ids
        self._deleted_count = deleted_count

    def __iter__(self):
        return iter([{"id": memory_id} for memory_id in self._ids or []])

    def consume(self):
        counters = SimpleNamespace(nodes_deleted=self._deleted_count)
        return SimpleNamespace(counters=counters)


class _Session:
    def __init__(self, owner: "_Driver"):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def run(self, query: str, **params):
        if "RETURN n.id AS id" in query:
            self.owner.operations.append("resolve_graph_ids")
            return _Result(ids=self.owner.graph_ids)
        if "DETACH DELETE" in query:
            self.owner.operations.append("delete_graph")
            if self.owner.graph_error is not None:
                raise self.owner.graph_error
            return _Result(deleted_count=len(self.owner.graph_ids))
        raise AssertionError(f"Unexpected query: {query!r}; params={params!r}")


class _Driver:
    def __init__(self, graph_ids: list[str], graph_error: Exception | None = None):
        self.graph_ids = graph_ids
        self.graph_error = graph_error
        self.operations: list[str] = []

    def session(self, *, database: str):
        assert database == "neo4j"
        return _Session(self)


def _community_graph_db(*, graph_error: Exception | None = None):
    graph_db = Neo4jCommunityGraphDB.__new__(Neo4jCommunityGraphDB)
    graph_db.config = SimpleNamespace(use_multi_db=False, user_name=None)
    graph_db.db_name = "neo4j"
    graph_db.driver = _Driver(["memory_1", "memory_2"], graph_error=graph_error)
    graph_db.vec_db = MagicMock()
    graph_db.vec_db.delete.side_effect = lambda ids: graph_db.driver.operations.append(
        "delete_vectors"
    )
    return graph_db


def test_community_delete_removes_vectors_before_graph_and_reports_count() -> None:
    graph_db = _community_graph_db()

    deleted_count = graph_db.delete_node_by_prams(memory_ids=["memory_1", "memory_2"])

    assert deleted_count == 2
    graph_db.vec_db.delete.assert_called_once_with(["memory_1", "memory_2"])
    assert graph_db.driver.operations == [
        "resolve_graph_ids",
        "delete_vectors",
        "delete_graph",
    ]


def test_community_delete_stops_before_graph_when_vector_delete_fails() -> None:
    graph_db = _community_graph_db()
    graph_db.vec_db.delete.side_effect = RuntimeError("qdrant delete failed")

    with pytest.raises(RuntimeError, match="qdrant delete failed"):
        graph_db.delete_node_by_prams(memory_ids=["memory_1", "memory_2"])

    assert "delete_graph" not in graph_db.driver.operations


def test_community_delete_propagates_graph_failure_after_vector_delete() -> None:
    graph_db = _community_graph_db(graph_error=RuntimeError("neo4j delete failed"))

    with pytest.raises(RuntimeError, match="neo4j delete failed"):
        graph_db.delete_node_by_prams(memory_ids=["memory_1", "memory_2"])

    graph_db.vec_db.delete.assert_called_once_with(["memory_1", "memory_2"])


def test_community_record_hard_delete_uses_the_same_cross_store_order() -> None:
    graph_db = _community_graph_db()

    deleted_count = graph_db.delete_node_by_mem_cube_id(
        mem_cube_id="cube_1",
        delete_record_id="delete_1",
        hard_delete=True,
    )

    assert deleted_count == 2
    graph_db.vec_db.delete.assert_called_once_with(["memory_1", "memory_2"])
    assert graph_db.driver.operations == [
        "resolve_graph_ids",
        "delete_vectors",
        "delete_graph",
    ]


def test_tree_delete_does_not_swallow_graph_failure() -> None:
    from memos.memories.textual.tree import TreeTextMemory

    tree_memory = TreeTextMemory.__new__(TreeTextMemory)
    tree_memory.graph_store = Mock()
    tree_memory.graph_store.delete_node_by_prams.side_effect = RuntimeError("graph delete failed")

    with pytest.raises(RuntimeError, match="graph delete failed"):
        tree_memory.delete_by_memory_ids(["memory_1"])


def test_memory_extraction_failure_does_not_fallback_to_raw_conversation() -> None:
    reader = SimpleStructMemReader.__new__(SimpleStructMemReader)
    reader.llm = Mock()
    reader.llm.generate.side_effect = RuntimeError("memory LM unavailable")
    reader.config = SimpleNamespace(remove_prompt_example=False)

    with pytest.raises(RuntimeError, match="memory LM unavailable"):
        reader._get_llm_response("private conversation", custom_tags=None)


@pytest.mark.parametrize(
    ("response_text", "error_pattern"),
    [
        ('{"summary": "nothing"}', "requires 'memory list'"),
        ('{"memory list": {}}', "requires 'memory list'"),
        ('{"memory list": [{}]}', "requires non-empty value"),
        (
            '{"memory list": [{"value": "fact", "tags": "bad"}]}',
            "tags must be a list of strings",
        ),
    ],
)
def test_memory_extraction_rejects_invalid_schema(
    response_text: str,
    error_pattern: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_pattern):
        SimpleStructMemReader._parse_llm_response(response_text)


def test_memory_extraction_allows_explicit_empty_memory_list() -> None:
    assert SimpleStructMemReader._parse_llm_response(
        '{"summary": "No durable memory", "memory list": []}'
    )["memory list"] == []


def test_fine_memory_item_build_failure_is_not_treated_as_empty_extraction() -> None:
    reader = SimpleStructMemReader.__new__(SimpleStructMemReader)
    reader._iter_chat_windows = Mock(
        return_value=[{"text": "private conversation", "sources": []}]
    )
    reader._get_llm_response = Mock(
        return_value={
            "summary": "summary",
            "memory list": [
                {
                    "value": "User prefers concise updates.",
                    "memory_type": "UserMemory",
                }
            ],
        }
    )
    reader._make_memory_item = Mock(
        side_effect=RuntimeError("embedding service unavailable")
    )

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        reader._process_chat_data(
            [], {"user_id": "user_a", "session_id": "session_1"}
        )
