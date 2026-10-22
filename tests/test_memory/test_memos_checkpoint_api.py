"""Checkpoint validation must reject foreign or incomplete data before mutation."""
from copy import deepcopy

import pytest
from neo4j.time import DateTime
from memos.mempa_checkpoint import _pack, _unpack, validate_snapshot


NATIVE_ID = "3a6d8fa0-2248-4f48-9bfe-0f20d497cf0c"


def snapshot():
    return {
        "schema_version": 1,
        "storage": "neo4j-community-qdrant",
        "cube_id": "owned-cube",
        "nodes": [{"labels": ["Memory"], "properties": {
            "id": NATIVE_ID, "user_name": "owned-cube", "memory": "Keep answers concise",
            "created_at": _pack(DateTime.from_iso_format("2026-09-06T12:00:00.123456789+00:00")),
        }}],
        "edges": [],
        "points": [{"id": NATIVE_ID, "vector": [0.1, 0.2, 0.3], "payload": {"user_name": "owned-cube"}}],
    }


def test_snapshot_preserves_ids_vectors_and_nanosecond_datetime():
    data = snapshot()
    before = deepcopy(data)
    nodes, edges = validate_snapshot(data, "owned-cube", 3)
    assert nodes[0]["id"] == NATIVE_ID
    assert _pack(nodes[0]["created_at"]) == data["nodes"][0]["properties"]["created_at"]
    assert data == before
    assert edges == []


def test_snapshot_cannot_restore_into_another_cube():
    with pytest.raises(ValueError, match="original cube id"):
        validate_snapshot(snapshot(), "other-cube", 3)


@pytest.mark.parametrize("corruption", ["foreign-node", "foreign-vector", "missing-vector", "bad-vector", "foreign-edge", "duplicate-node"])
def test_snapshot_rejects_scope_and_storage_corruption(corruption):
    data = snapshot()
    if corruption == "foreign-node":
        data["nodes"][0]["properties"]["user_name"] = "other"
    elif corruption == "foreign-vector":
        data["points"][0]["payload"]["user_name"] = "other"
    elif corruption == "missing-vector":
        data["points"] = []
    elif corruption == "bad-vector":
        data["points"][0]["vector"][0] = float("nan")
    elif corruption == "foreign-edge":
        data["edges"] = [{"source": NATIVE_ID, "target": "foreign", "type": "RELATED", "properties": {}}]
    else:
        data["nodes"].append(deepcopy(data["nodes"][0]))
    with pytest.raises(ValueError):
        validate_snapshot(data, "owned-cube", 3)


def test_datetime_marker_does_not_allow_arbitrary_nested_properties():
    with pytest.raises(ValueError, match="nested"):
        _unpack({"arbitrary": "object"})


class _Session:
    """The two Cypher reads `import_cube` runs, plus its write transaction."""

    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        if "n.id IN $ids" in query:
            count = 0
        else:
            count = self.graph.node_count
        return type(
            "_Result",
            (),
            {"single": lambda self: {"count": count}, "consume": lambda self: None},
        )()

    def execute_write(self, fn):
        return fn(self)

    def consume(self):
        return None


class _VecDB:
    def __init__(self, dimensions):
        self.config = type("_Config", (), {"vector_dimension": dimensions})()
        self.batch_sizes = []
        self.stored = 0

    def get_by_ids(self, ids):
        return []

    def add(self, points):
        self.batch_sizes.append(len(points))
        self.stored += len(points)

    def count(self, filter):
        return self.stored


class Neo4jCommunityGraphDB:
    """Stands in for the real driver; only the surface `import_cube` touches."""

    def __init__(self, dimensions=3, node_count=None):
        self.config = type("_Config", (), {"use_multi_db": False})()
        self.db_name = "neo4j"
        self.vec_db = _VecDB(dimensions)
        self.node_count = node_count
        self.driver = type(
            "_Driver", (), {"session": lambda _self, database=None: _Session(self)}
        )()

    def clear(self, user_name):
        return None


def _many_points_snapshot(count):
    data = snapshot()
    data["nodes"] = []
    data["points"] = []
    for index in range(count):
        native_id = f"{index:08x}-2248-4f48-9bfe-0f20d497cf0c"
        data["nodes"].append(
            {"labels": ["Memory"], "properties": {"id": native_id, "user_name": "owned-cube"}}
        )
        data["points"].append(
            {"id": native_id, "vector": [0.1, 0.2, 0.3], "payload": {"user_name": "owned-cube"}}
        )
    return data


def test_import_upserts_points_in_bounded_batches():
    from memos.mempa_checkpoint import POINT_BATCH_SIZE, CubeCheckpointStore

    count = POINT_BATCH_SIZE * 2 + 5
    data = _many_points_snapshot(count)
    graph = Neo4jCommunityGraphDB(node_count=count)
    result = CubeCheckpointStore(graph).import_cube(data, "owned-cube")

    assert graph.vec_db.batch_sizes == [POINT_BATCH_SIZE, POINT_BATCH_SIZE, 5]
    assert result["point_count"] == count
    assert result["node_count"] == count


def test_import_raises_when_the_restored_vector_count_differs(monkeypatch):
    from memos import mempa_checkpoint
    from memos.mempa_checkpoint import CubeCheckpointStore

    monkeypatch.setattr(mempa_checkpoint, "_COUNT_SETTLE_SECONDS", 0.0)
    data = _many_points_snapshot(4)
    graph = Neo4jCommunityGraphDB(node_count=4)
    graph.vec_db.add = lambda points: setattr(
        graph.vec_db, "stored", graph.vec_db.stored + len(points) - 1
    )

    with pytest.raises(ValueError, match="restored=3 expected=4"):
        CubeCheckpointStore(graph).import_cube(data, "owned-cube")


def test_import_raises_when_the_restored_node_count_differs():
    from memos.mempa_checkpoint import CubeCheckpointStore

    data = _many_points_snapshot(4)
    graph = Neo4jCommunityGraphDB(node_count=3)

    with pytest.raises(ValueError, match="restored=3 expected=4"):
        CubeCheckpointStore(graph).import_cube(data, "owned-cube")
