"""Kernel-level batch exclusion plus native pending-Probe boundary behavior."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from harness.probe_slots import probe_batch_slot, workspace_root

ROOT = Path(__file__).resolve().parents[2]
CHILD = '''
import asyncio,json,os,sys,time
from pathlib import Path
from harness.probe_slots import probe_batch_slot
async def main():
    async with probe_batch_slot(Path(sys.argv[1]), stop_requested=lambda:False, workers=4, poll_seconds=.01) as yes:
        assert yes
        fd=os.open(sys.argv[2],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        os.write(fd,(json.dumps([time.monotonic_ns(),os.getpid(),1])+"\\n").encode())
        print("acquired",flush=True)
        await asyncio.sleep(.3)
        os.write(fd,(json.dumps([time.monotonic_ns(),os.getpid(),-1])+"\\n").encode())
        os.close(fd)
asyncio.run(main())
'''


def test_real_processes_never_hold_more_than_two_batches(tmp_path):
    children = [subprocess.Popen([sys.executable, "-c", CHILD, str(tmp_path / "slots"), str(tmp_path / "events")],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for _ in range(7)]
    try:
        for child in children:
            out, err = child.communicate(timeout=20)
            assert child.returncode == 0, err
            assert out.strip() == "acquired"
        live = maximum = 0
        events = sorted(json.loads(line) for line in (tmp_path / "events").read_text().splitlines())
        for _, _, change in events:
            live += change
            maximum = max(maximum, live)
            assert 0 <= live <= 2
        assert maximum == 2 and live == 0
        assert len(list((tmp_path / "slots").glob("slot-*.lock"))) == 2
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()


@pytest.mark.asyncio
async def test_stop_waiting_and_exception_leave_stable_slots_reusable(tmp_path):
    stop = False
    async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4):
        async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4):
            async def wait():
                async with probe_batch_slot(tmp_path, stop_requested=lambda: stop, workers=4, poll_seconds=.01) as yes:
                    return yes
            waiter = asyncio.create_task(wait())
            await asyncio.sleep(.03)
            assert not waiter.done()
            stop = True
            assert await asyncio.wait_for(waiter, 1) is False
    inode = (tmp_path / "slot-0.lock").stat().st_ino
    with pytest.raises(RuntimeError, match="batch failure"):
        async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4):
            raise RuntimeError("batch failure")
    async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4) as yes:
        assert yes
    assert (tmp_path / "slot-0.lock").stat().st_ino == inode


@pytest.mark.asyncio
async def test_killed_holder_releases_kernel_slot(tmp_path):
    code = CHILD.replace("await asyncio.sleep(.3)", "await asyncio.sleep(60)")
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(tmp_path / "events")],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 5) == "acquired\n"
        async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4):
            child.kill()
            await asyncio.to_thread(child.wait)
            async with probe_batch_slot(tmp_path, stop_requested=lambda: False, workers=4) as yes:
                assert yes
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        child.stdout.close()
        child.stderr.close()


def test_workspace_is_explicit_absolute_and_shared_across_checkouts(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPA_WORKSPACE_ROOT", str(tmp_path / "data"))
    assert workspace_root(tmp_path / "checkout-a") == workspace_root(tmp_path / "checkout-b")
    monkeypatch.setenv("MEMPA_WORKSPACE_ROOT", "relative")
    with pytest.raises(ValueError, match="absolute"):
        workspace_root(tmp_path)
