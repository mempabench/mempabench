"""§9 step 12 — a probe branch runs on the namespace its checkpoint restored.

`_restore_backend_state_into_branch` imports the pinned snapshot into a
per-branch namespace for every external backend, and until now nothing was
built on it: the probe PA was handed `memory_adapter=None` for every condition
but the memory-removed baseline, so it retrieved nothing at all. These two
tests pin the two halves of the fix — the branch adapter exists and reads the
branch namespace, and it is read-only before the probe PA is built.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_orchestrator
from harness.checkpoint import CheckpointManager
from harness.memory_runtime import MemoryRuntime
from harness.orchestrator import Orchestrator, OrchestratorConfig
from harness.run_ledger import RunLedger
from harness.state_runtime import StageRuntime
from memory.adapter import PromptHistoryEvictionPolicy
from memory.event_writer import (
    AdapterWriteCommitError,
    record_adapter_write_commit,
    write_visible_memory_event,
)
from memory.events import MemoryEvent
from memory.results import (
    BackendHealth,
    ConsistencyModel,
    HealthStatus,
    ModeReceipt,
    RetrieveResult,
    WriteReceipt,
    WriteStatus,
)
from memory.scope import AdapterMode, MemoryRetrievalBudget, RetrieveRequest

from test_harness.test_orchestrator import (
    FakeExecutor,
    FakeMemoryAdapter,
    FakePACheckpoint,
    _fixture,
    _seed_run_dir,
    _two_acc_plan,
)


MAINLINE_CUBE = "mempa_run_001_memos"
RECORD_BEFORE_CHECKPOINT = "the user asks for the two-line version"


class FakeMemosClient:
    """The cube half of a memos service: export, import-as, delete.

    Records are stored as text but exported in the deployment's own snapshot
    shape, and an import runs the service's `validate_snapshot` before it takes
    effect, so the payload this orchestrator path hands `import_cube_as` is
    held to what the live service would accept.
    """

    def __init__(self) -> None:
        self.cubes: dict[str, list[str]] = {}

    def export_cube(self, cube_id: str) -> dict[str, Any]:
        records = list(self.cubes.get(cube_id, []))
        native_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{cube_id}:{index}"))
            for index in range(len(records))
        ]
        return {
            "schema_version": 1,
            "storage": "neo4j-community-qdrant",
            "cube_id": cube_id,
            "nodes": [
                {
                    "labels": ["Memory"],
                    "properties": {
                        "id": native_id,
                        "user_name": cube_id,
                        "working_binding": native_id,
                        "memory": record,
                    },
                }
                for native_id, record in zip(native_ids, records)
            ],
            "edges": [],
            "points": [
                {
                    "id": native_id,
                    "vector": [0.1, 0.2, 0.3],
                    "payload": {"user_name": cube_id, "working_binding": native_id},
                }
                for native_id in native_ids
            ],
        }

    def import_cube_as(self, payload: dict[str, Any], namespace: str) -> None:
        from memos.mempa_checkpoint import validate_snapshot

        validate_snapshot(payload, namespace, 3)
        self.cubes[namespace] = [
            node["properties"]["memory"] for node in payload["nodes"]
        ]

    def delete_cube(self, namespace: str) -> None:
        self.cubes.pop(namespace, None)


class FakeExternalAdapter:
    """An adapter whose memory lives in a remote cube, not the run directory."""

    backend_name = "memos"

    def __init__(self, client: FakeMemosClient, cube_id: str) -> None:
        self.client = client
        self.cube_id = cube_id
        self._mode = AdapterMode()

    async def setup_scope(self, scope) -> None:
        return None

    async def set_mode(self, mode: AdapterMode) -> ModeReceipt:
        previous, self._mode = self._mode, mode
        return ModeReceipt(mode=mode, previous_mode=previous)

    async def health(self) -> BackendHealth:
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=1.0,
            consistency_model=ConsistencyModel.immediate,
        )

    async def record_event(self, event: MemoryEvent) -> WriteReceipt:
        if self._mode.read_only:
            return WriteReceipt(status=WriteStatus.skipped_read_only)
        self.client.cubes.setdefault(self.cube_id, []).append(
            event.messages[0]["content"]
        )
        return WriteReceipt(status=WriteStatus.committed, native_ids=["1"])

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        return RetrieveResult(
            raw=[
                {"memory": text}
                for text in self.client.cubes.get(self.cube_id, [])
            ]
        )


class ProbeAdapterExecutor(FakeExecutor):
    """Exercises the probe's adapter while the branch is still alive.

    The branch directory and its remote namespace are both dropped the moment
    `_run_probe_in_branch` returns, so anything the probe's adapter has to
    prove has to be proved here.
    """

    def __init__(self, *, write: bool = False) -> None:
        super().__init__()
        self.probes: list[dict[str, Any]] = []
        self.write = write

    async def execute(
        self,
        step,
        *,
        steps_dir,
        stage_runtime=None,
        extra_meta=None,
        loop_context=None,
    ):
        if loop_context is not None:
            adapter = loop_context["memory_adapter"]
            view: dict[str, Any] = {
                "step_id": step.step_id,
                "adapter": adapter,
                "namespace": loop_context["memory_backend_namespace"],
            }
            if adapter is not None:
                retrieved = await adapter.retrieve(
                    RetrieveRequest(
                        query="how long should the reply be",
                        scope=None,
                        session_id=step.step_id,
                        budget=MemoryRetrievalBudget(),
                    )
                )
                view["retrieved"] = [row["memory"] for row in retrieved.raw]
                if self.write:
                    view["receipt"] = await adapter.record_event(
                        MemoryEvent(
                            event_id="e1",
                            run_id="run_001",
                            persona_id=step.persona_id,
                            agent_id="nanobot",
                            session_id=step.step_id,
                            turn_id="t1",
                            timestamp=datetime.now(timezone.utc),
                            messages=[
                                {"role": "user", "content": "probe turn"}
                            ],
                        )
                    )
                    view["cube_after_write"] = list(
                        adapter.client.cubes.get(adapter.cube_id, [])
                    )
            self.probes.append(view)
        return await super().execute(
            step,
            steps_dir=steps_dir,
            stage_runtime=stage_runtime,
            extra_meta=extra_meta,
            loop_context=loop_context,
        )


def _external_run(
    tmp_path: Path, *, executor: ProbeAdapterExecutor
) -> tuple[Orchestrator, FakeMemosClient]:
    """A memos run with one record in its cube, pinned and branched for real."""
    plan = _two_acc_plan().model_copy(update={"memory_condition": "memos"})
    run_dir = tmp_path / "runs" / plan.run_id
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    ledger = RunLedger(run_dir, plan.run_id)

    client = FakeMemosClient()
    client.cubes[MAINLINE_CUBE] = [RECORD_BEFORE_CHECKPOINT]
    manager = CheckpointManager(
        run_dir,
        ledger=ledger,
        memory_condition="memos",
        memory_backend_namespace=MAINLINE_CUBE,
        backend_client=client,
        rolling_window=10,
    )
    manager.branch_adapter_factory = (
        lambda namespace, branch_dir: FakeExternalAdapter(client, namespace)
    )

    orchestrator = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=run_dir,
            project_root=tmp_path,
            checkpoint_manager=manager,
        ),
        executor=executor,
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=_fixture(tmp_path),
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            FakeMemoryAdapter(),
            run_id=plan.run_id,
            persona_id=plan.persona_id,
            backend_namespace=MAINLINE_CUBE,
        ),
        pa_checkpoint=FakePACheckpoint([]),
        progress=lambda _line: None,
    )
    orchestrator.ledger = ledger
    return orchestrator, client


@pytest.mark.asyncio
async def test_probe_on_an_external_backend_retrieves_a_record_written_before_the_checkpoint(
    tmp_path: Path,
) -> None:
    executor = ProbeAdapterExecutor()
    orchestrator, client = _external_run(tmp_path, executor=executor)

    await orchestrator.run()

    probe = next(view for view in executor.probes if view["step_id"] == "pretest_W_A")
    adapter = probe["adapter"]
    assert adapter is not None, (
        "a probe on an external backend was built with no memory adapter, so "
        "it retrieves nothing and reads exactly like the no_memory baseline"
    )
    # Built on the namespace the branch restored, never on the mainline cube.
    assert adapter.cube_id == probe["namespace"]
    assert adapter.cube_id.startswith(f"{MAINLINE_CUBE}_br_")
    assert adapter.cube_id != MAINLINE_CUBE
    # And that namespace carries the cut point's memory.
    assert probe["retrieved"] == [RECORD_BEFORE_CHECKPOINT]
    # The mainline cube is shared with nobody and outlives the branch.
    assert client.cubes[MAINLINE_CUBE] == [RECORD_BEFORE_CHECKPOINT]


@pytest.mark.asyncio
async def test_probe_branch_adapter_is_read_only(tmp_path: Path) -> None:
    executor = ProbeAdapterExecutor(write=True)
    orchestrator, client = _external_run(tmp_path, executor=executor)

    await orchestrator.run()

    probe = next(view for view in executor.probes if view["step_id"] == "pretest_W_A")
    assert probe["adapter"] is not None
    assert probe["receipt"].status is WriteStatus.skipped_read_only
    assert probe["receipt"].native_ids == []
    # No native write: the branch cube still holds only the cut point's record.
    assert probe["cube_after_write"] == [RECORD_BEFORE_CHECKPOINT]
    assert client.cubes[MAINLINE_CUBE] == [RECORD_BEFORE_CHECKPOINT]


# ── the verifier's three refutations ──────────────────────────────


class _FakeRegistry:
    """Records what the branch factory asked the registry to build."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create(self, condition: str, **options: Any) -> Any:
        self.calls.append((condition, dict(options)))
        return SimpleNamespace(condition=condition, options=options)


@pytest.mark.asyncio
async def test_memos_branch_adapter_moves_only_the_cube_id(
    monkeypatch, tmp_path: Path
) -> None:
    registry = _FakeRegistry()
    monkeypatch.setattr(run_orchestrator, "default_registry", lambda: registry)
    client = object()

    await run_orchestrator._build_branch_memory_adapter(
        memory_condition="memos",
        namespace=f"{MAINLINE_CUBE}_br_pretest_w_a",
        branch_dir=tmp_path / "branch",
        mainline_options={"client": client, "cube_id": MAINLINE_CUBE},
    )

    condition, options = registry.calls[-1]
    assert condition == "memos"
    assert options["cube_id"] == f"{MAINLINE_CUBE}_br_pretest_w_a"
    # The memos client is the mainline one: the cube is the namespace.
    assert options["client"] is client


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["mem0", "simple_rag"])
async def test_mem0_branch_adapter_builds_a_client_on_the_branch_collection(
    monkeypatch, tmp_path: Path, condition: str
) -> None:
    registry = _FakeRegistry()
    monkeypatch.setattr(run_orchestrator, "default_registry", lambda: registry)
    built: list[dict[str, Any]] = []

    def fake_client(**kwargs: Any) -> Any:
        built.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(run_orchestrator, "create_mem0_client", fake_client)
    monkeypatch.setattr(
        run_orchestrator, "create_simple_rag_mem0_client", fake_client
    )
    branch_dir = tmp_path / "branch"
    namespace = "mempa_run_001_mem0_br_pretest_w_a"

    await run_orchestrator._build_branch_memory_adapter(
        memory_condition=condition,
        namespace=namespace,
        branch_dir=branch_dir,
        mainline_options={"client": object()},
        mem0_client_kwargs={
            "collection_name": "mempa_run_001_mem0",
            "qdrant_host": "localhost",
            "allow_existing_collection": False,
            "history_db_path": tmp_path / "mainline.db",
            "allow_existing_history_db": False,
            "require_existing_history_db": False,
        },
        mem0_llm_kwargs=(
            {"memory_side_model": "memory/model", "extraction_max_output_tokens": 32000} if condition == "mem0" else None
        ),
    )

    kwargs = built[-1]
    assert kwargs.get("extraction_max_output_tokens") == (32000 if condition == "mem0" else None)
    assert kwargs["collection_name"] == namespace
    assert kwargs["qdrant_host"] == "localhost"
    # `branch_from` restored the collection and copied the history db in.
    assert kwargs["allow_existing_collection"] is True
    # A branch never drops its restored namespace, `--overwrite` or not.
    assert kwargs["drop_existing_collection"] is False
    assert kwargs["allow_existing_history_db"] is True
    assert kwargs["require_existing_history_db"] is True
    assert Path(kwargs["history_db_path"]).is_relative_to(branch_dir)
    _condition, options = registry.calls[-1]
    assert options["client"].collection_name == namespace


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["mem0", "simple_rag"])
@pytest.mark.parametrize("foreign", [False, True])
async def test_mem0_branch_preserves_payload_owner_and_retrieves_it(
    monkeypatch, tmp_path, condition, foreign
):
    from memory.backends.mem0_client import _inspect_qdrant_collection
    from memory.scope import MemoryScope
    from test_memory.test_mem0_client import (
        _FakeQdrantClient, _install_fake_qdrant, _QDRANT_SCHEMA_KWARGS,
    )

    _install_fake_qdrant(monkeypatch)
    owner = "repeat_run"
    point = SimpleNamespace(id="memory-1", payload={"run_id": "unrelated" if foreign else owner})
    monkeypatch.setattr(_FakeQdrantClient, "collection_exists_value", True)
    monkeypatch.setattr(_FakeQdrantClient, "scroll_pages", [[point]])
    requests = []
    clients = []

    def search(**request):
        requests.append(request)
        assert request["filters"] == {"run_id": owner, "user_id": "user_b"}
        return {"results": [{"id": "memory-1", "memory": "original memory", "score": .9,
            "role": "user", "metadata": {"session_id": "acc_014", "turn_id": "t1",
            "timestamp": "2026-09-21T00:00:00+00:00"}}]}

    def client(**kwargs):
        clients.append(kwargs)
        _inspect_qdrant_collection(
            collection_name=kwargs["collection_name"], qdrant_host="unused", qdrant_port=6333,
            expected_run_id=kwargs["expected_run_id"], require_collection=True,
            **_QDRANT_SCHEMA_KWARGS,
        )
        return SimpleNamespace(search=search)

    monkeypatch.setattr(run_orchestrator, "create_mem0_client", client)
    monkeypatch.setattr(run_orchestrator, "create_simple_rag_mem0_client", client)
    kwargs = dict(memory_condition=condition, namespace="repeat_scoped_branch",
        branch_dir=tmp_path / "repeat_02/branches/fin_01/01", mainline_options={},
        mem0_client_kwargs={"expected_run_id": "repeat_run"})
    if foreign:
        with pytest.raises(ValueError, match="owned by a different run"):
            await run_orchestrator._build_branch_memory_adapter(**kwargs)
        assert requests == []
        return
    adapter = await run_orchestrator._build_branch_memory_adapter(**kwargs)
    scope = MemoryScope(run_id="repeat_run", persona_id="user_b")
    await adapter.setup_scope(scope)
    await adapter.set_mode(AdapterMode(read_only=True))
    result = await adapter.retrieve(RetrieveRequest(
        query="preference", scope=scope, session_id="fin_01", budget=MemoryRetrievalBudget()))
    assert result.raw[0]["memory"] == "original memory"
    assert result.trace.scope.run_id == "repeat_run"
    assert clients[0]["collection_name"] == "repeat_scoped_branch"
    assert clients[0]["expected_run_id"] == owner
    assert point.payload["run_id"] == owner  # Neither source nor branch payload is rewritten.
    await adapter.set_mode(AdapterMode(read_only=False))


@pytest.mark.asyncio
async def test_a_probe_on_an_external_backend_with_no_branch_adapter_raises(
    tmp_path: Path,
) -> None:
    """A silent `None` is exactly the defect §9 step 12 fixed."""
    executor = ProbeAdapterExecutor()
    orchestrator, _client = _external_run(tmp_path, executor=executor)
    # A manager that restores a namespace but builds nothing on it.
    orchestrator.config.checkpoint_manager.branch_adapter_factory = None

    with pytest.raises(RuntimeError) as caught:
        await orchestrator.run()

    message = str(caught.value)
    assert "memos" in message
    assert "branch_dir" in message


@pytest.mark.asyncio
async def test_a_read_only_probe_receipt_does_not_kill_the_probe(
    tmp_path: Path,
) -> None:
    """A probe's every turn is `skipped_read_only`; that must not raise.

    `RealSessionLoop.run_session` built every PA with
    `require_committed_adapter_writes=True`, so the first probe turn raised
    `AdapterWriteCommitError` and no probe on an external backend could
    finish. The accumulation guarantee is untouched: the same receipt still
    raises under an accumulation step's setting.
    """
    probe_requires = run_orchestrator._requires_committed_adapter_writes(
        SimpleNamespace(kind="pre_event_probe")
    )
    final_requires = run_orchestrator._requires_committed_adapter_writes(
        SimpleNamespace(kind="final_probe")
    )
    acc_requires = run_orchestrator._requires_committed_adapter_writes(
        SimpleNamespace(kind="accumulation")
    )
    assert (probe_requires, final_requires, acc_requires) == (False, False, True)

    client = FakeMemosClient()
    adapter = FakeExternalAdapter(client, "branch_cube")
    await adapter.set_mode(AdapterMode(read_only=True, reason="pretest_W_A"))

    receipt = await write_visible_memory_event(
        adapter=adapter,
        messages=[{"role": "user", "content": "probe turn"}],
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
        session_id="pretest_W_A",
        turn_id="t1",
        event_id="e1",
        timestamp=datetime.now(timezone.utc),
        require_committed=probe_requires,
        eviction_policy=PromptHistoryEvictionPolicy.after_adapter_commit,
    )
    assert receipt.status is WriteStatus.skipped_read_only
    assert client.cubes.get("branch_cube") is None

    # The probe survives its own read-only receipt.
    assert (
        record_adapter_write_commit(
            receipt=receipt,
            session_id="pretest_W_A",
            current_index=4,
            committed_index=0,
            eviction_policy=PromptHistoryEvictionPolicy.after_adapter_commit,
            require_committed=probe_requires,
        )
        == 0
    )
    # And the accu/evolve guarantee still bites on the same receipt.
    with pytest.raises(AdapterWriteCommitError):
        record_adapter_write_commit(
            receipt=receipt,
            session_id="acc_001",
            current_index=4,
            committed_index=0,
            eviction_policy=PromptHistoryEvictionPolicy.after_adapter_commit,
            require_committed=acc_requires,
        )
