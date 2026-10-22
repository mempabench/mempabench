import asyncio

import pytest

from harness.probe_slots import active_batch_slots, batch_capacity, configure_batch_slots, probe_batch_slot


@pytest.mark.asyncio
async def test_expanded_pool_uses_original_guards_and_rejects_busy_shrink(tmp_path):
    async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4):
        original_inode = (tmp_path / "slot-0.lock").stat().st_ino
        configure_batch_slots(tmp_path, 3)
        async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4):
            async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4):
                assert len(active_batch_slots(tmp_path)) == 3
                with pytest.raises(RuntimeError, match="occupied"):
                    configure_batch_slots(tmp_path, 2)
                assert batch_capacity(tmp_path) == 3
        configure_batch_slots(tmp_path, 2)
        assert batch_capacity(tmp_path) == 2
        assert (tmp_path / "slot-0.lock").stat().st_ino == original_inode
    assert active_batch_slots(tmp_path) == []


@pytest.mark.asyncio
async def test_capacity_eight_means_eight_real_held_round_slots(tmp_path):
    configure_batch_slots(tmp_path, 8)
    entered = 0
    release = asyncio.Event()

    async def worker():
        nonlocal entered
        async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4, poll_seconds=.01):
            entered += 1
            await release.wait()

    tasks = [asyncio.create_task(worker()) for _ in range(9)]
    try:
        for _ in range(100):
            if entered == 8:
                break
            await asyncio.sleep(.01)
        assert entered == 8 and len(active_batch_slots(tmp_path)) == 8
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert entered == 9
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
