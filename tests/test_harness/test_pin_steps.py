"""Owner ruling 2026-09-13 — run the accumulations now, probe them later.

A run may execute its accumulation timeline before any probe plan exists and
have one attached later, off the same checkpoints. Two things have to hold for
that. The operator must be able to pin cut points a probeless plan does not
imply — `RunPlan.pinned_step_ids` derives pins from probes alone, and
`run_plan_builder` emits no probe steps, so a frozen accumulation plan pins
nothing and the rolling window of three eats its own history. And a probe that
borrows one of those cut points under a memory condition must read the cut
point's memory: `NoMemoryAdapter` belongs to the memory-removed baseline, which
is a property of the condition, never of borrowing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from harness.checkpoint import CheckpointManager, IncompleteCheckpointError
from harness.memory_runtime import MemoryRuntime
from harness.orchestrator import Orchestrator, OrchestratorConfig
from harness.run_ledger import RunLedger
from harness.run_plan import RunPlan, RunStep
from harness.state_runtime import StageRuntime
from memory.conditions.no_memory import NoMemoryAdapter
from memory.results import BackendHealth, ConsistencyModel, HealthStatus
from scripts.run_orchestrator import _resolve_extra_pins

from test_harness.test_checkpoint import FakeMem0Client
from test_harness.test_orchestrator import (
    FakeExecutor,
    FakeMemoryAdapter,
    FakePACheckpoint,
    ProbeRecordingExecutor,
    _fixture,
    _plan,
    _seed_run_dir,
    _two_acc_plan,
)


MAINLINE_COLLECTION = "mempa_run_001_simple_rag"


# ── fixtures the two helpers below share ──────────────────────────


def _fixture_root(tmp_path: Path, acc_nums: range) -> Path:
    """`_fixture`, widened to as many accumulation sessions as a test needs."""
    root = _fixture(tmp_path)
    template = (tmp_path / "acc_001.yaml").read_text(encoding="utf-8")
    for acc_num in acc_nums:
        name = f"acc_{acc_num:03d}"
        (tmp_path / f"{name}.yaml").write_text(
            template.replace("session_id: acc_001", f"session_id: {name}"),
            encoding="utf-8",
        )
    outline = tmp_path / "Authoring/outline/drafts/user_a/outline.yaml"
    outline.write_text(
        "sessions:\n"
        + "".join(
            "  - identity:\n"
            f"      acc_num: {acc_num}\n"
            "    probe:\n"
            "      why_both_viable: compressed, so no gate\n"
            for acc_num in acc_nums
        ),
        encoding="utf-8",
    )
    return root


def _accumulations_only_plan(count: int) -> RunPlan:
    """The shape `run_plan_builder` freezes: accumulations and nothing else."""
    plan = _plan()
    base = plan.steps[0].model_dump(mode="json")
    steps = [
        RunStep(
            **{
                **base,
                "step_id": f"acc_{acc_num:03d}",
                "script_path": f"acc_{acc_num:03d}.yaml",
                "acc_num": acc_num,
                "benchmark_session_timestamp": (
                    f"2026-10-{acc_num:02d}T11:00:00+00:00"
                ),
            }
        )
        for acc_num in range(1, count + 1)
    ]
    return plan.model_copy(
        update={"steps": steps, "memory_condition": "simple_rag"}
    )


def _simple_rag_run(
    tmp_path: Path,
    plan: RunPlan,
    *,
    run_name: str,
    rolling_window: int,
    extra_pinned: frozenset[str] = frozenset(),
    pa: FakePACheckpoint | None = None,
    executor: Any | None = None,
    client: FakeMem0Client | None = None,
    branch_adapter_factory: Any = None,
    borrowed_runs_root: Path | None = None,
    memory_condition: str | None = None,
    runs_dir: Path | None = None,
) -> tuple[Orchestrator, FakeMem0Client]:
    """One run on an external backend, wired as the entrypoint wires it.

    `runs_dir` is this run's own parent, `tmp_path / "runs"` unless a test
    needs it somewhere else -- which is the only way to tell a borrow that
    honoured `borrowed_runs_root` from one that fell back to the parent.
    """
    runs_dir = runs_dir or tmp_path / "runs"
    run_dir = runs_dir / run_name
    _seed_run_dir(run_dir, plan.persona_id, plan.main_pa_session_key())
    history = run_dir / "backend_state" / "mem0_history.sqlite3"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_bytes(b"history-v1")
    ledger = RunLedger(run_dir, plan.run_id)
    client = client or FakeMem0Client()
    manager = CheckpointManager(
        run_dir,
        ledger=ledger,
        memory_condition=memory_condition or plan.memory_condition,
        memory_backend_namespace=MAINLINE_COLLECTION,
        backend_client=client,
        branch_adapter_factory=branch_adapter_factory,
        rolling_window=rolling_window,
    )
    orchestrator = Orchestrator(
        plan=plan,
        config=OrchestratorConfig(
            run_dir=run_dir,
            project_root=tmp_path,
            checkpoint_manager=manager,
            extra_pinned_step_ids=extra_pinned,
            borrowed_runs_root=borrowed_runs_root,
        ),
        executor=executor or FakeExecutor(),
        stage_runtime=StageRuntime(
            run_id=plan.run_id,
            user_id=plan.persona_id,
            fixture_root=tmp_path / "fixtures",
            runs_root=runs_dir,
            project_root=tmp_path,
        ),
        memory_runtime=MemoryRuntime(
            FakeMemoryAdapter(),
            run_id=plan.run_id,
            persona_id=plan.persona_id,
            backend_namespace=MAINLINE_COLLECTION,
        ),
        ledger=ledger,
        pa_checkpoint=pa,
        progress=lambda _line: None,
    )
    return orchestrator, client


def _branch_namespace(branch_dir: Path) -> str:
    record = json.loads(
        (branch_dir / "backend_state" / "branch_namespace.json").read_text(
            encoding="utf-8"
        )
    )
    return record["namespace"]


# ── (a) pin every accumulation ────────────────────────────────────


@pytest.mark.asyncio
async def test_pin_all_accumulations_keeps_every_cut_point_branchable(
    tmp_path: Path,
) -> None:
    plan = _accumulations_only_plan(4)
    # The premise: nothing probes, so the plan itself pins nothing at all.
    assert plan.pinned_step_ids() == set()
    _fixture_root(tmp_path, range(1, 5))
    pa = FakePACheckpoint([])
    pins = _resolve_extra_pins(
        plan, pin_steps=None, pin_all_accumulations=True
    )
    assert pins == {"acc_001", "acc_002", "acc_003", "acc_004"}

    orchestrator, _client = _simple_rag_run(
        tmp_path,
        plan,
        run_name="pin_all",
        # One, so a rolling cut point cannot survive by luck: everything that
        # is still here at the end is here because it was pinned.
        rolling_window=1,
        extra_pinned=pins,
        pa=pa,
    )
    await orchestrator.run()

    manager = orchestrator.config.checkpoint_manager
    assert sorted(manager.list_checkpoints()) == [
        "acc_001",
        "acc_002",
        "acc_003",
        "acc_004",
    ]
    # Pinned means the full pinned save path, not a second kind of pin: the
    # flush ran, the frozen plan was copied, the namespace was recorded.
    assert pa.saved == 4
    for checkpoint_id in manager.list_checkpoints():
        metadata = manager.metadata(checkpoint_id)
        assert metadata["pinned"] is True
        assert metadata["memory_backend_namespace"] == MAINLINE_COLLECTION
        cp_dir = manager.run_dir / "checkpoints" / checkpoint_id
        assert (cp_dir / "run_plan.yaml").is_file()
        # And each one is a start point a probe plan frozen later can take.
        branch = manager.branch_from(
            checkpoint_id, tmp_path / "branches" / checkpoint_id
        )
        assert _branch_namespace(branch) != MAINLINE_COLLECTION


# ── (b) pin exactly the named steps ───────────────────────────────


@pytest.mark.asyncio
async def test_pin_steps_pins_exactly_those(tmp_path: Path) -> None:
    plan = _accumulations_only_plan(4)
    _fixture_root(tmp_path, range(1, 5))
    pins = _resolve_extra_pins(
        plan, pin_steps=" acc_002 ,acc_004, ", pin_all_accumulations=False
    )
    assert pins == {"acc_002", "acc_004"}

    orchestrator, _client = _simple_rag_run(
        tmp_path,
        plan,
        run_name="pin_two",
        rolling_window=1,
        extra_pinned=pins,
        pa=FakePACheckpoint([]),
    )
    await orchestrator.run()

    manager = orchestrator.config.checkpoint_manager
    # The two pins, plus the one rolling cut point the window still holds.
    assert sorted(manager.list_checkpoints()) == [
        "acc_002",
        "acc_003",
        "acc_004",
    ]
    assert manager.metadata("acc_002")["pinned"] is True
    assert manager.metadata("acc_004")["pinned"] is True
    assert manager.metadata("acc_003")["pinned"] is False
    assert not (
        manager.run_dir / "checkpoints" / "acc_003" / "run_plan.yaml"
    ).is_file()
    with pytest.raises(ValueError, match="Only a pinned checkpoint"):
        manager.branch_from("acc_003", tmp_path / "branches" / "acc_003")


def test_pin_steps_refuses_a_step_this_plan_cannot_pin() -> None:
    plan = _accumulations_only_plan(2)
    with pytest.raises(SystemExit, match="acc_009"):
        _resolve_extra_pins(
            plan, pin_steps="acc_001,acc_009", pin_all_accumulations=False
        )


# ── (c) a borrowed start point reads memory unless the run is the baseline ──


class _BranchAdapter:
    """The adapter a probe reads its branch's restored collection through."""

    backend_name = "simple_rag"

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.modes: list[Any] = []

    async def setup_scope(self, scope) -> None:
        return None

    async def set_mode(self, mode) -> None:
        self.modes.append(mode)
        return None

    async def health(self) -> BackendHealth:
        return BackendHealth(
            status=HealthStatus.ok,
            backend_name=self.backend_name,
            latency_ms=1.0,
            consistency_model=ConsistencyModel.immediate,
        )


async def _source_run(tmp_path: Path) -> tuple[RunPlan, FakeMem0Client]:
    """A finished accumulation run on simple_rag, every cut point pinned."""
    source = _accumulations_only_plan(2)
    _fixture_root(tmp_path, range(1, 3))
    orchestrator, client = _simple_rag_run(
        tmp_path,
        source,
        run_name=source.run_id,
        rolling_window=1,
        extra_pinned=_resolve_extra_pins(
            source, pin_steps=None, pin_all_accumulations=True
        ),
        pa=FakePACheckpoint([]),
    )
    await orchestrator.run()
    return source, client


def _borrowing_plan(source: RunPlan, *, memory_condition: str) -> RunPlan:
    """A probe plan frozen later, borrowing the source run's first cut point."""
    probe = _two_acc_plan().steps[-1].model_dump(mode="json")
    return source.model_copy(
        update={
            "memory_condition": memory_condition,
            "pa_session_key": source.main_pa_session_key(),
            "steps": [
                RunStep(
                    **{
                        **probe,
                        "step_id": "pre_01",
                        "probe_overlay": {
                            **probe["probe_overlay"], "owner_probe_id": "pre_01",
                        },
                        "borrow_start_from": f"{source.run_id}/acc_001",
                    }
                )
            ],
        }
    )


@pytest.mark.asyncio
async def test_borrowed_start_point_under_a_memory_condition_reads_memory(
    tmp_path: Path,
) -> None:
    source, client = await _source_run(tmp_path)

    built: list[str] = []

    def factory(namespace: str, branch_dir: Path) -> _BranchAdapter:
        built.append(namespace)
        return _BranchAdapter(namespace)

    executor = ProbeRecordingExecutor()
    orchestrator, _client = _simple_rag_run(
        tmp_path,
        _borrowing_plan(source, memory_condition="simple_rag"),
        run_name="probe_run",
        rolling_window=3,
        executor=executor,
        client=client,
        branch_adapter_factory=factory,
        borrowed_runs_root=tmp_path / "runs",
    )
    await orchestrator.run()

    view = executor.probe_views[0]
    adapter = view["adapter"]
    assert isinstance(adapter, _BranchAdapter), (
        "a probe that borrows its start point under a memory condition was "
        "handed the memory-removed baseline's adapter, so it would retrieve "
        "nothing and read exactly like the baseline"
    )
    assert not isinstance(adapter, NoMemoryAdapter)
    # Built on the branch's own restored collection, never on the mainline one.
    assert built == [adapter.namespace]
    assert adapter.namespace.startswith(f"{MAINLINE_COLLECTION}_br_")
    # Read-only before the probe PA exists (§9 step 12).
    assert [mode.read_only for mode in adapter.modes] == [True]
    # The source run's own memory files came across with the snapshot.
    assert view["memory_files"] == ["MEMORY.md"]


@pytest.mark.asyncio
async def test_the_memory_removed_baseline_still_gets_no_memory_adapter(
    tmp_path: Path,
) -> None:
    source, client = await _source_run(tmp_path)

    built: list[str] = []
    executor = ProbeRecordingExecutor()
    orchestrator, _client = _simple_rag_run(
        tmp_path,
        _borrowing_plan(source, memory_condition="no_memory"),
        run_name="baseline_run",
        rolling_window=3,
        executor=executor,
        client=client,
        branch_adapter_factory=lambda namespace, branch_dir: built.append(
            namespace
        ),
        borrowed_runs_root=tmp_path / "runs",
        # The baseline's own condition; the snapshot is still simple_rag's.
        memory_condition="no_memory",
    )
    await orchestrator.run()

    view = executor.probe_views[0]
    assert isinstance(view["adapter"], NoMemoryAdapter)
    assert built == []
    # The PA memory directory was emptied inside the branch.
    assert view["memory_files"] == []


@pytest.mark.asyncio
async def test_the_baseline_borrows_from_an_external_backend_with_no_client(
    tmp_path: Path,
) -> None:
    """The real shape of a `--memory-condition no_memory` baseline arm.

    The test above wires a backend client in by hand, which the entrypoint
    never does for this condition: `scripts/run_orchestrator.py` fills
    `checkpoint_manager.backend_client` from the memory adapter options, and
    only mem0 / simple_rag / memos ever set that key. The baseline
    arm of a simple_rag cell therefore runs with `backend_client is None`,
    borrows a snapshot whose metadata says `simple_rag`, and would be branched
    under that condition — demanding a client it cannot have. Restoring the
    source's vector collection would be waste in any case: the probe is about
    to bind `NoMemoryAdapter` and read none of it.
    """
    source, client = await _source_run(tmp_path)
    calls_before = list(client.qdrant_admin.calls)

    built: list[str] = []
    executor = ProbeRecordingExecutor()
    orchestrator, _client = _simple_rag_run(
        tmp_path,
        _borrowing_plan(source, memory_condition="no_memory"),
        run_name="baseline_no_client",
        rolling_window=3,
        executor=executor,
        client=client,
        branch_adapter_factory=lambda namespace, branch_dir: built.append(
            namespace
        ),
        borrowed_runs_root=tmp_path / "runs",
        memory_condition="no_memory",
    )
    # What the entrypoint actually leaves on a baseline run: no client at all.
    orchestrator.config.checkpoint_manager.backend_client = None

    await orchestrator.run()

    view = executor.probe_views[0]
    assert isinstance(view["adapter"], NoMemoryAdapter)
    assert built == []
    assert view["memory_files"] == []
    # No namespace of the source's vector store was restored for this branch.
    assert client.qdrant_admin.calls == calls_before


def _oracle_plan(source: RunPlan) -> RunPlan:
    """An oracle_preference probe borrowing the source's first cut point.

    Unlike `_borrowing_plan` the probe keeps its own id, so its frozen overlay
    owner still matches its draft and the branch actually runs.
    """
    probe = _two_acc_plan().steps[-1].model_dump(mode="json")
    return source.model_copy(
        update={
            "memory_condition": "oracle_preference",
            "pa_session_key": source.main_pa_session_key(),
            "steps": [
                RunStep(
                    **{
                        **probe,
                        "target_cell": "work_verbosity",
                        "target_attribute": "verbosity",
                        "target_setting": "Terse",
                        "borrow_start_from": f"{source.run_id}/acc_001",
                    }
                )
            ],
        }
    )


@pytest.mark.asyncio
async def test_oracle_borrows_from_an_external_backend_with_no_client(
    tmp_path: Path,
) -> None:
    """Owner 2026-09-25: an oracle run may borrow any condition's cut point.

    Like the baseline, it reads none of the source's store, so its entrypoint
    has no backend client and nothing is restored for the branch.
    """
    from memory.conditions.oracle_preference import OraclePreferenceAdapter

    source, client = await _source_run(tmp_path)
    calls_before = list(client.qdrant_admin.calls)

    built: list[str] = []
    executor = ProbeRecordingExecutor()
    orchestrator, _client = _simple_rag_run(
        tmp_path,
        _oracle_plan(source),
        run_name="oracle_no_client",
        rolling_window=3,
        executor=executor,
        client=client,
        branch_adapter_factory=lambda namespace, branch_dir: built.append(
            namespace
        ),
        borrowed_runs_root=tmp_path / "runs",
        memory_condition="oracle_preference",
    )
    orchestrator.config.checkpoint_manager.backend_client = None

    await orchestrator.run()

    [view] = executor.probe_views
    assert isinstance(view["adapter"], OraclePreferenceAdapter)
    assert built == []
    assert view["memory_files"] == []
    assert client.qdrant_admin.calls == calls_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plan_condition", "branched_under"),
    [
        ("oracle_preference", "oracle_preference"),
        ("no_memory", "no_memory"),
        # Every other condition still reads the snapshot's own condition.
        ("mem0", "simple_rag"),
    ],
)
async def test_the_condition_a_borrowed_simple_rag_snapshot_is_branched_under(
    tmp_path: Path, plan_condition: str, branched_under: str
) -> None:
    source, client = await _source_run(tmp_path)
    plan = _borrowing_plan(source, memory_condition=plan_condition)
    orchestrator, _client = _simple_rag_run(
        tmp_path,
        plan,
        run_name="probe_run",
        rolling_window=3,
        client=client,
        borrowed_runs_root=tmp_path / "runs",
    )

    source_manager, checkpoint_id = orchestrator._probe_source_checkpoint(
        plan.steps[0], orchestrator.config.checkpoint_manager, set()
    )

    assert checkpoint_id == "acc_001"
    assert source_manager.memory_condition == branched_under


@pytest.mark.asyncio
async def test_a_memory_borrow_still_requires_the_backend_client(
    tmp_path: Path,
) -> None:
    """The discrimination: only the baseline may branch without a client.

    A probe under a memory condition reads the branch's restored namespace, so
    a missing client is a real defect there and must still be refused.
    """
    source, client = await _source_run(tmp_path)
    orchestrator, _client = _simple_rag_run(
        tmp_path,
        _borrowing_plan(source, memory_condition="simple_rag"),
        run_name="probe_no_client",
        rolling_window=3,
        executor=ProbeRecordingExecutor(),
        client=client,
        branch_adapter_factory=lambda namespace, branch_dir: None,
        borrowed_runs_root=tmp_path / "runs",
    )
    orchestrator.config.checkpoint_manager.backend_client = None

    with pytest.raises(
        IncompleteCheckpointError, match="without its client"
    ):
        await orchestrator.run()


# ── (d) which root a borrow reads ─────────────────────────────────


@pytest.mark.asyncio
async def test_a_given_borrowed_runs_root_is_the_one_read(
    tmp_path: Path,
) -> None:
    """The borrowing run lives somewhere else entirely, so only the configured
    root can find the source. Every other borrow test here puts both runs
    under `tmp_path / "runs"`, where honouring the value and falling back to
    `run_dir.parent` are indistinguishable."""
    source, client = await _source_run(tmp_path)

    executor = ProbeRecordingExecutor()
    orchestrator, _client = _simple_rag_run(
        tmp_path,
        _borrowing_plan(source, memory_condition="simple_rag"),
        run_name="probe_run",
        rolling_window=3,
        executor=executor,
        client=client,
        branch_adapter_factory=lambda namespace, branch_dir: _BranchAdapter(
            namespace
        ),
        runs_dir=tmp_path / "other_runs",
        borrowed_runs_root=tmp_path / "runs",
    )
    await orchestrator.run()

    assert orchestrator.config.run_dir.parent == tmp_path / "other_runs"
    assert executor.probe_views[0]["memory_files"] == ["MEMORY.md"]


@pytest.mark.asyncio
async def test_no_borrowed_runs_root_falls_back_to_the_run_s_own_parent(
    tmp_path: Path,
) -> None:
    """Unset, the borrow reads the borrowing run's own parent -- the behaviour
    every run had before `--borrowed-runs-root` existed, because nothing could
    set the field."""
    source, client = await _source_run(tmp_path)

    plan = _borrowing_plan(source, memory_condition="simple_rag")
    orchestrator, _client = _simple_rag_run(
        tmp_path,
        plan,
        run_name="probe_run",
        rolling_window=3,
        client=client,
        branch_adapter_factory=lambda namespace, branch_dir: None,
        runs_dir=tmp_path / "other_runs",
    )

    source_manager, _ = orchestrator._probe_source_checkpoint(
        plan.steps[0],
        orchestrator.config.checkpoint_manager,
        set(),
    )
    assert source_manager.run_dir == tmp_path / "other_runs" / source.run_id
