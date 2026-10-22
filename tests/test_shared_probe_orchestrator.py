"""Real File checkpoint branches across two actual Orchestrator.run calls.

Only session execution is local/fake: orchestration, the owner, CheckpointManager,
File adapters, session managers, private scenes and durable ledgers are real.
The compact existing orchestration plan retains its complete two-step timeline;
full 108-slot/33-Probe scientific-plan binding is tested separately.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harness.checkpoint import CheckpointManager
from harness.shared_probe_memory import SharedProbeMemoryOwner
from memory.scope import MemoryRetrievalBudget, MemoryScope, RetrieveRequest
from pa.session.manager import SessionNotFoundError
from test_harness import test_orchestrator as support


def file_manifest(root: Path):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class IndependentProbeExecutor(support.ProbeRecordingExecutor):
    def __init__(self):
        super().__init__()
        self.independent = []
        self.live = 0
        self.max_live = 0
        self.two_running = asyncio.Event()

    async def execute(self, step, *, loop_context=None, extra_meta=None, stage_runtime=None, **kwargs):
        assert step.kind in {"pre_event_probe", "final_probe"}, "repeat executed its borrowed timeline"
        assert loop_context is not None
        workspace = loop_context["workspace"]
        sessions = loop_context["session_manager"]
        session_key = extra_meta["pa_session_key"]
        try:
            session = sessions.get(session_key)
        except SessionNotFoundError:
            session = sessions.create(session_key)
        assert not session.messages, "a different Probe's conversation leaked into this branch"
        prompt = f"{extra_meta['run_id']}/{step.step_id} independent question"
        session.add_message("user", prompt)
        sessions.save(session)
        pa = SimpleNamespace(sessions=sessions, closed=[])
        pa.close_ix_session = lambda key, sid: pa.closed.append((key, sid)) or []
        loop_context["pa_binder"](pa)

        private_marker = stage_runtime.state_root / "private-scene.txt"
        assert not private_marker.exists(), "a different Probe's scene leaked into this branch"
        private_marker.write_text(prompt)
        memory = workspace / "memory"
        assert memory.is_symlink()
        assert (memory / ".mempa-read-only").read_text() == "shared-probe-memory\n"
        adapter = loop_context["memory_adapter"]
        scope = MemoryScope(run_id=extra_meta["run_id"], persona_id=step.persona_id)
        request = RetrieveRequest(query=prompt, scope=scope, session_id=session_key,
                                  budget=MemoryRetrievalBudget(max_items=10))
        response = await adapter.retrieve(request)
        # File memory reaches the prompt through `# Memory`, not format_context;
        # the adapter's retrieval still returns the shared MEMORY.md.
        assert "pa memory" in response.raw[0]["content"]
        assert response.trace.scope.run_id == extra_meta["run_id"]
        with pytest.raises(PermissionError):
            await adapter.reset_scope(scope)
        self.independent.append({"pa": pa, "sessions": sessions, "session": session,
            "adapter": adapter, "request": request, "response": response, "world": stage_runtime,
            "workspace": workspace, "shared_memory": memory.resolve(), "prompt": prompt,
            "run_id": extra_meta["run_id"], "probe_id": step.step_id})
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        if self.live == 2:
            self.two_running.set()
        try:
            await asyncio.wait_for(self.two_running.wait(), timeout=5)
            await asyncio.sleep(0)
            return await super().execute(step, loop_context=loop_context, extra_meta=extra_meta,
                                         stage_runtime=stage_runtime, **kwargs)
        finally:
            self.live -= 1


@pytest.mark.asyncio
async def test_actual_orchestrator_shares_file_pin_across_rounds_without_sharing_sessions(tmp_path, monkeypatch):
    original_plan = support._many_probe_plan(4)
    producer, source_ledger, source_dir = support._batch_orchestrator(
        tmp_path, original_plan, support.ProbeRecordingExecutor(), concurrency=2)
    original_plan.dump_yaml(source_dir / "run_plan.yaml")
    original_result = await producer.run()
    probes = set(support._probe_ids(original_plan))
    assert set(original_result.executed) == {"acc_001", "acc_002", *probes}
    source_manager = producer.config.checkpoint_manager
    assert isinstance(source_manager, CheckpointManager)
    assert source_manager.metadata("acc_002")["pinned"] is True
    original_bytes = file_manifest(source_dir)

    prepared, released = [], []
    real_prepare = CheckpointManager.prepare_shared_memory
    real_release = CheckpointManager.release_branch_namespace

    def track_prepare(manager, pin, directory):
        prepared.append((manager, pin, Path(directory)))
        return real_prepare(manager, pin, directory)

    def track_release(manager, directory):
        released.append(Path(directory))
        return real_release(manager, directory)

    monkeypatch.setattr(CheckpointManager, "prepare_shared_memory", track_prepare)
    monkeypatch.setattr(CheckpointManager, "release_branch_namespace", track_release)
    owner = SharedProbeMemoryOwner(tmp_path / "setting-owned-memory")
    branch_spy = AsyncMock(wraps=owner.branch)
    monkeypatch.setattr(owner, "branch", branch_spy)
    executed = []
    try:
        for repeat in (2, 3):
            steps = [step.model_copy(update={"borrow_start_from": f"{original_plan.run_id}/acc_002"})
                     if step.step_id in probes else step.model_copy() for step in original_plan.steps]
            plan = original_plan.model_copy(update={"run_id": f"{original_plan.run_id}_repeat{repeat:02d}",
                "pa_session_key": original_plan.main_pa_session_key(), "steps": steps})
            executor = IndependentProbeExecutor()
            runner, ledger, directory = support._batch_orchestrator(tmp_path, plan, executor, concurrency=2)
            plan.dump_yaml(directory / "run_plan.yaml")
            # Both runs receive the same real source checkpoint manager. The
            # borrowed-source resolver remains real (it may create cheap manager
            # wrappers), while the owner retains one prepared memory instance.
            runner.config.checkpoint_manager = source_manager
            runner.config.shared_probe_memory_owner = owner
            runner.config.probe_only = True
            runner.config.probe_source_ledger = source_ledger
            runner.config.borrowed_runs_root = source_dir.parent
            runner.config.probe_slot_root = tmp_path / "probe-slots"
            result = await asyncio.wait_for(runner.run(), timeout=15)
            assert set(result.executed) == probes and not result.failed
            assert set(result.skipped) == {"acc_001", "acc_002"}
            assert executor.max_live == 2
            assert set(ledger.steps) == probes
            assert all(record.status == "done" for record in ledger.steps.values())
            # A repeat keeps its own attempt budget (probe_attempts/), never a
            # plain run's per-launch attempt record.
            assert not (directory / "probe_launch_attempts").exists()
            durable = json.loads((directory / "ledger.json").read_text())
            assert durable["run_id"] == plan.run_id and set(durable["steps"]) == probes
            assert not list((directory / "checkpoints").iterdir())
            assert file_manifest(source_dir) == original_bytes
            assert len(prepared) == 1, "the second round restored the same pin again"
            memory_dir = prepared[0][2]
            assert memory_dir.exists() and memory_dir not in released
            record = next(iter(owner._entries.values()))["record"]
            assert record["state"] == "ready" and record["restore_count"] == 1
            assert all(state == "finished" for state in record["consumers"].values())
            assert len(record["consumers"]) == 4 * (repeat - 1)
            assert not owner._closed
            # Ordinary resume after success must not ask the owner for any new
            # branch or rerun a paid Probe, even with the setting owner alive.
            previous_calls = branch_spy.await_count
            resumed = await runner.run()
            assert not resumed.executed
            assert branch_spy.await_count == previous_calls
            executed.extend(executor.independent)

        assert branch_spy.await_count == 8
        assert all(isinstance(call.args[0], CheckpointManager) and call.args[0].run_dir == source_dir
                   and call.args[1] == "acc_002" for call in branch_spy.await_args_list)
        assert prepared[0][0].run_dir == source_dir
        assert len({row["shared_memory"] for row in executed}) == 1
        for field in ("pa", "sessions", "session", "adapter", "request", "response", "world"):
            assert len({id(row[field]) for row in executed}) == 8, f"shared mutable {field}"
        assert len({row["workspace"] for row in executed}) == 8
        assert len({row["run_id"] for row in executed}) == 2
        assert all(row["pa"].closed for row in executed)
        assert all([message["content"] for message in row["session"].messages] == [row["prompt"]]
                   for row in executed)
        assert all(not row["workspace"].exists() for row in executed)
        assert await owner.close()
        assert released.count(prepared[0][2]) == 1
        assert next(iter(owner._entries.values()))["record"]["state"] == "released"
        assert file_manifest(source_dir) == original_bytes
    finally:
        await owner.close()


class RoundBarrierExecutor(IndependentProbeExecutor):
    """Hold the actual branch open until the test releases this round."""

    def __init__(self, barrier):
        super().__init__()
        self.barrier = barrier
        self.release_round = asyncio.Event()
        # IndependentProbeExecutor's original gate is for two Probes within
        # one round. This case has one identically named Probe in each round;
        # the explicit shared barrier below coordinates the two executors.
        self.two_running.set()

    async def execute(self, step, **kwargs):
        result = await super().execute(step, **kwargs)
        run_id = kwargs["extra_meta"]["run_id"]
        self.barrier.arrivals[run_id] = self.independent[-1]
        if len(self.barrier.arrivals) == 2:
            self.barrier.both_entered.set()
        await asyncio.wait_for(self.barrier.both_entered.wait(), timeout=10)
        await asyncio.wait_for(self.release_round.wait(), timeout=10)
        return result


@pytest.mark.asyncio
async def test_two_actual_orchestrators_overlap_and_keep_file_view_until_both_rounds_finish(
    tmp_path, monkeypatch,
):
    original_plan = support._many_probe_plan(1)
    producer, source_ledger, source_dir = support._batch_orchestrator(
        tmp_path, original_plan, support.ProbeRecordingExecutor(), concurrency=2,
    )
    original_plan.dump_yaml(source_dir / "run_plan.yaml")
    original = await producer.run()
    assert set(original.executed) == {"acc_001", "acc_002", "fin_01"}
    source_manager = producer.config.checkpoint_manager
    assert isinstance(source_manager, CheckpointManager)
    source_bytes = file_manifest(source_dir)

    prepared, released = [], []
    real_prepare = CheckpointManager.prepare_shared_memory
    real_release = CheckpointManager.release_branch_namespace

    def track_prepare(manager, pin, directory):
        prepared.append((manager, pin, Path(directory)))
        return real_prepare(manager, pin, directory)

    def track_release(manager, directory):
        released.append(Path(directory))
        return real_release(manager, directory)

    monkeypatch.setattr(CheckpointManager, "prepare_shared_memory", track_prepare)
    monkeypatch.setattr(CheckpointManager, "release_branch_namespace", track_release)
    owner = SharedProbeMemoryOwner(tmp_path / "concurrent-setting-memory")
    branch_spy = AsyncMock(wraps=owner.branch)
    monkeypatch.setattr(owner, "branch", branch_spy)
    barrier = SimpleNamespace(arrivals={}, both_entered=asyncio.Event())
    rounds = []
    for repeat in (2, 3):
        plan = original_plan.model_copy(update={
            "run_id": f"{original_plan.run_id}_repeat{repeat:02d}",
            "pa_session_key": original_plan.main_pa_session_key(),
            "steps": [step.model_copy(update={
                "borrow_start_from": f"{original_plan.run_id}/acc_002",
            }) if step.step_id == "fin_01" else step.model_copy()
                      for step in original_plan.steps],
        })
        executor = RoundBarrierExecutor(barrier)
        runner, ledger, directory = support._batch_orchestrator(
            tmp_path, plan, executor, concurrency=2,
        )
        plan.dump_yaml(directory / "run_plan.yaml")
        runner.config.checkpoint_manager = source_manager
        runner.config.shared_probe_memory_owner = owner
        runner.config.probe_only = True
        runner.config.probe_source_ledger = source_ledger
        runner.config.borrowed_runs_root = source_dir.parent
        runner.config.probe_slot_root = tmp_path / "concurrent-probe-slots"
        rounds.append(SimpleNamespace(
            plan=plan, runner=runner, ledger=ledger, directory=directory,
            executor=executor, completed=asyncio.Event(), result=None,
        ))

    async def run_round(round):
        round.result = await round.runner.run()
        round.completed.set()
        return round.result

    # These are actual Orchestrator.run calls, with one real owner and source
    # CheckpointManager; only their local session executor is substituted.
    concurrent_runs = asyncio.gather(*(run_round(round) for round in rounds))
    try:
        await asyncio.wait_for(barrier.both_entered.wait(), timeout=10)
        assert all(not round.completed.is_set() for round in rounds)
        assert len(prepared) == 1 and prepared[0][1] == "acc_002"
        assert prepared[0][0].run_dir == source_dir
        assert branch_spy.await_count == 2
        assert all(call.args[0].run_dir == source_dir and call.args[1] == "acc_002"
                   for call in branch_spy.await_args_list)
        branch_dirs = [Path(call.args[2]) for call in branch_spy.await_args_list]
        assert len(set(branch_dirs)) == 2
        assert all(path.name == "01" and path.parent.name == "fin_01" for path in branch_dirs)
        rows = [barrier.arrivals[round.plan.run_id] for round in rounds]
        assert {row["probe_id"] for row in rows} == {"fin_01"}
        assert len({row["request"].session_id for row in rows}) == 1
        for field in ("pa", "sessions", "session", "adapter", "request", "response", "world"):
            assert rows[0][field] is not rows[1][field], f"overlapping rounds shared mutable {field}"
        assert rows[0]["workspace"] != rows[1]["workspace"]
        assert rows[0]["world"].state_root != rows[1]["world"].state_root
        assert rows[0]["shared_memory"] == rows[1]["shared_memory"]
        shared_directory = prepared[0][2]
        assert shared_directory.exists()
        entry = next(iter(owner._entries.values()))
        assert set(entry["record"]["consumers"].values()) == {"active"}
        assert len(entry["record"]["consumers"]) == 2
        for round, row in zip(rounds, rows):
            attempts = json.loads((round.directory / "probe_attempts/fin_01.json").read_text())
            assert attempts["attempts"] == [{"attempt": 1, "status": "started"}]
            assert [message["content"] for message in row["session"].messages] == [row["prompt"]]
            assert (row["world"].state_root / "private-scene.txt").read_text() == row["prompt"]

        # Let repeat02 finish while repeat03 still owns its live branch.
        rounds[0].executor.release_round.set()
        await asyncio.wait_for(rounds[0].completed.wait(), timeout=10)
        assert not rounds[1].completed.is_set() and not concurrent_runs.done()
        assert not owner._closed and entry["record"]["state"] == "ready"
        assert sorted(entry["record"]["consumers"].values()) == ["active", "finished"]
        assert shared_directory.exists() and shared_directory not in released
        assert not rows[0]["workspace"].exists()
        assert rows[1]["workspace"].exists() and not rows[1]["pa"].closed
        assert rows[0]["pa"].closed
        # A further read through repeat03's actual adapter proves that the
        # first round's cleanup left the shared view usable, not just present.
        still_readable = await rows[1]["adapter"].retrieve(rows[1]["request"])
        assert "pa memory" in still_readable.raw[0]["content"]
        assert still_readable.trace.scope.run_id == rounds[1].plan.run_id
        assert file_manifest(source_dir) == source_bytes

        rounds[1].executor.release_round.set()
        results = await asyncio.wait_for(concurrent_runs, timeout=10)
        for round, result in zip(rounds, results):
            assert result.executed == ["fin_01"] and not result.failed
            assert set(result.skipped) == {"acc_001", "acc_002"}
            durable = json.loads((round.directory / "ledger.json").read_text())
            assert durable["run_id"] == round.plan.run_id
            assert set(durable["steps"]) == {"fin_01"}
            assert round.ledger.steps["fin_01"].status == "done"
            attempts = json.loads((round.directory / "probe_attempts/fin_01.json").read_text())
            assert attempts["attempts"] == [{"attempt": 1, "status": "succeeded"}]
        assert all(not row["workspace"].exists() and row["pa"].closed for row in rows)
        assert len(prepared) == 1 and entry["record"]["restore_count"] == 1
        assert set(entry["record"]["consumers"].values()) == {"finished"}
        assert shared_directory.exists() and shared_directory not in released
        assert await owner.close()
        assert released.count(shared_directory) == 1
        assert entry["record"]["state"] == "released"
        assert not shared_directory.exists(), "the final owner close retained the File view"
        assert not rows[0]["shared_memory"].exists()
        assert file_manifest(source_dir) == source_bytes
    finally:
        for round in rounds:
            round.executor.release_round.set()
        if not concurrent_runs.done():
            concurrent_runs.cancel()
        await asyncio.gather(concurrent_runs, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["no_memory", "oracle_preference"])
async def test_a_reference_arm_branches_the_source_pin_without_shared_memory(tmp_path, monkeypatch, condition):
    """A no_memory / oracle_preference arm under the setting owner: each Probe gets
    the branch's own copy of the pin, emptied of memory, and the arm's adapter;
    the owner prepares and retains nothing."""
    from memory.conditions.no_memory import NoMemoryAdapter
    from memory.conditions.oracle_preference import OraclePreferenceAdapter
    original_plan = support._many_probe_plan(2)
    producer, source_ledger, source_dir = support._batch_orchestrator(
        tmp_path, original_plan, support.ProbeRecordingExecutor(), concurrency=2)
    original_plan.dump_yaml(source_dir / "run_plan.yaml")
    await producer.run()
    source_manager = producer.config.checkpoint_manager
    original_bytes = file_manifest(source_dir)
    monkeypatch.setattr(CheckpointManager, "prepare_shared_memory",
                        lambda *a, **k: pytest.fail("a reference arm prepared shared memory"))
    owner = SharedProbeMemoryOwner(tmp_path / "setting-owned-memory")
    probes = set(support._probe_ids(original_plan))
    cell = {"target_cell": "work_verbosity", "target_attribute": "verbosity", "target_setting": "Detailed"}
    steps = [step.model_copy(update={"borrow_start_from": f"{original_plan.run_id}/acc_002", **cell})
             if step.step_id in probes else step.model_copy() for step in original_plan.steps]
    plan = original_plan.model_copy(update={
        "run_id": f"{original_plan.run_id}_{condition}_test-pa_probe_repeat01", "memory_condition": condition,
        "pa_session_key": original_plan.main_pa_session_key(), "steps": steps})
    executor = support.ProbeRecordingExecutor()
    runner, ledger, directory = support._batch_orchestrator(tmp_path, plan, executor, concurrency=2)
    runner.config.checkpoint_manager = source_manager
    runner.config.shared_probe_memory_owner = owner
    runner.config.probe_only = True
    runner.config.probe_source_ledger = source_ledger
    runner.config.borrowed_runs_root = source_dir.parent
    runner.config.probe_slot_root = tmp_path / "probe-slots"
    try:
        result = await asyncio.wait_for(runner.run(), timeout=15)
        assert set(result.executed) == probes and not result.failed
        expected = NoMemoryAdapter if condition == "no_memory" else OraclePreferenceAdapter
        assert [type(view["adapter"]) for view in executor.probe_views] == [expected, expected]
        assert all(view["extra_meta"]["memory_condition"] == condition for view in executor.probe_views)
        assert not owner._entries and not list(owner.root.glob("*.json"))
        assert file_manifest(source_dir) == original_bytes
        assert await owner.close()
    finally:
        await owner.close()
