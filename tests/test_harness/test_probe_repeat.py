import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import pytest

from harness.checkpoint import CheckpointManager, branch_namespace_for
from .test_orchestrator import _many_probe_plan, _batch_orchestrator, _WidthGateExecutor


@pytest.mark.asyncio
@pytest.mark.parametrize('width', [8, 16])
async def test_real_dispatcher_overlaps_exactly_requested_width(tmp_path, width):
    plan = _many_probe_plan(33)
    executor = _WidthGateExecutor(width)
    runner, ledger, _ = _batch_orchestrator(tmp_path, plan, executor, concurrency=width)
    result = await asyncio.wait_for(runner.run(), timeout=20)
    assert executor.max_live == width
    assert not result.failed
    assert all(r.status == 'done' for r in ledger.steps.values())


def test_qdrant_lock_serializes_whole_restore(tmp_path, monkeypatch):
    lock = tmp_path / 'shared.lock'
    monkeypatch.setenv('MEMPA_QDRANT_RESTORE_LOCK_PATH', str(lock))
    calls = []
    active = peak = 0
    guard = threading.Lock()

    def body(self, cp, branch, metadata):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        calls.append(('vector', branch))
        time.sleep(.01)
        calls.append(('entities_and_history', branch))
        with guard:
            active -= 1

    monkeypatch.setattr(CheckpointManager, '_restore_backend_state_into_branch_unlocked', body)
    manager = object.__new__(CheckpointManager)
    manager.memory_condition = 'mem0'
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda n: manager._restore_backend_state_into_branch(tmp_path, str(n), {}), range(16)))
    assert peak == 1
    assert all(calls[n][1] == calls[n + 1][1] for n in range(0, 32, 2))


def test_branch_namespaces_separate_probes_and_attempts():
    names = {branch_namespace_for('source_cube', Path('branches') / probe / attempt)
             for probe in ('pre_01', 'fin_01') for attempt in ('01', '02')}
    assert len(names) == 4
    assert branch_namespace_for('source_cube', Path('branches/fin_01/01')) == 'source_cube_br_fin_01_01'
