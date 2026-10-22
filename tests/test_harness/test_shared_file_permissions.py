"""Actual POSIX writes against shared file views; no mocked filesystem."""
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile

import pytest

from harness.run_ledger import RunLedger
from harness.shared_probe_memory import SharedProbeMemoryOwner
from memory.conditions.nanobot_file_memory import NanobotFileMemoryAdapter
from memory.scope import MemoryRetrievalBudget, MemoryScope, RetrieveRequest
from test_harness.test_checkpoint import PERSONA, RUN_ID, _build_run_dir, _manager, _write_session


@pytest.fixture
def file_case():
    # A standalone /tmp tree can be handed to an unprivileged UID if pytest
    # itself is root. Its control write proves denial is not a parent-dir issue.
    with tempfile.TemporaryDirectory(prefix="mempa-shared-file-") as name:
        root = Path(name)
        run = _build_run_dir(root)
        _write_session(run, cursors=(3, 3, 3))
        ledger = RunLedger(run, RUN_ID)
        ledger.mark_running("acc_108")
        manager = _manager(run, ledger)
        manager.save("acc_108", pinned=True)
        try:
            yield root, run, manager
        finally:
            # Retained failed views deliberately remain read-only in production.
            for directory, _, _ in os.walk(root, followlinks=False):
                Path(directory).chmod(0o700)


def _view_bytes(view: Path) -> dict[str, bytes]:
    """Every memory file of a shared view, by path."""
    return {str(path.relative_to(view)): path.read_bytes()
            for path in sorted((view / "pa_workspace").glob("*/memory/**/*")) if path.is_file()}


def _request():
    return RetrieveRequest(query="fixed response preference", scope=MemoryScope(
        run_id=RUN_ID, persona_id=PERSONA), session_id="original_main",
        budget=MemoryRetrievalBudget(max_items=2, max_tokens=4000))


def _child_credentials(root):
    if os.geteuid() != 0:
        return {}
    account = pwd.getpwnam("nobody")
    for directory, dirs, files in os.walk(root, followlinks=False):
        for path in [Path(directory), *(Path(directory) / n for n in dirs + files)]:
            os.chown(path, account.pw_uid, account.pw_gid, follow_symlinks=False)
    return {"user": account.pw_uid, "group": account.pw_gid, "extra_groups": []}


@pytest.mark.asyncio
async def test_direct_file_mutations_are_denied_and_owner_alone_releases(file_case):
    root, run, manager = file_case
    owner = SharedProbeMemoryOwner(root / "shared")
    branches = [root / repeat for repeat in ("repeat02", "repeat03")]
    original = NanobotFileMemoryAdapter(run / "pa_workspace" / PERSONA / "memory")
    adapters = [await owner.branch(manager, "acc_108", branch) for branch in branches]
    memory = branches[0] / "pa_workspace" / PERSONA / "memory"
    view = next(iter(owner._entries.values()))["directory"]
    before = _view_bytes(view)
    for adapter in adapters:
        result = await adapter.retrieve(_request())
        baseline = await original.retrieve(_request())
        assert result.raw == baseline.raw
        assert adapter.format_context(result) == original.format_context(baseline)
    control = root / "control"
    control.mkdir()
    credentials = _child_credentials(root)
    code = """
import os, sys
from pathlib import Path
memory, control, operation = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
replacement = control / 'replacement'
replacement.write_text('control write succeeds')
assert memory.stat().st_uid == os.geteuid(), 'Test must use the view owner UID'
actions = {
    'write': lambda: (memory / 'MEMORY.md').write_text('pollution'),
    'unlink': lambda: (memory / 'MEMORY.md').unlink(),
    'replace': lambda: os.replace(replacement, memory / 'MEMORY.md'),
    'new_file': lambda: (memory / 'new.md').write_text('pollution'),
    'remove_marker': lambda: (memory / '.mempa-read-only').unlink(),
    'move_memory_directory': lambda: os.rename(memory.resolve(), control / 'moved-memory'),
    'write_history': lambda: (memory / 'HISTORY.md').open('a').write('pollution'),
}
try:
    actions[operation]()
except PermissionError:
    print('denied')
else:
    raise AssertionError('Direct mutation unexpectedly succeeded: ' + operation)
"""
    try:
        for operation in ("write", "unlink", "replace", "new_file", "remove_marker",
                          "move_memory_directory", "write_history"):
            result = subprocess.run([sys.executable, "-B", "-c", code, str(memory), str(control), operation],
                                    capture_output=True, text=True, **credentials)
            assert result.returncode == 0, result.stdout + result.stderr
            assert result.stdout.strip() == "denied"
        assert _view_bytes(view) == before
        assert (run / "pa_workspace" / PERSONA / "memory/MEMORY.md").read_text() == "pa memory\n"
        owner.release_consumer(branches[0])
        assert view.exists() and (memory / "MEMORY.md").is_file()
        owner.release_consumer(branches[1])
        assert await owner.close()
        assert not view.exists()
        assert next(iter(owner._entries.values()))["record"]["state"] == "released"
        assert await owner.close()
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_active_consumer_retains_sealed_view_without_owner_unlock(file_case):
    root, _, manager = file_case
    owner = SharedProbeMemoryOwner(root / "shared")
    await owner.branch(manager, "acc_108", root / "repeat02")
    entry = next(iter(owner._entries.values()))
    view = entry["directory"]
    assert not await owner.close()
    assert entry["record"]["state"] == "cleanup_pending"
    assert view.exists()
    assert all(path.stat().st_mode & 0o222 == 0 for path in [view, *view.rglob("*")])


@pytest.mark.asyncio
async def test_a_changed_view_is_still_released_and_source_bytes_stay(file_case):
    """Plan R5: the view's contents are not compared at close; the source run's
    memory is never touched."""
    root, run, manager = file_case
    owner = SharedProbeMemoryOwner(root / "shared")
    branch = root / "repeat02"
    await owner.branch(manager, "acc_108", branch)
    entry = next(iter(owner._entries.values()))
    memory = entry["directory"] / "pa_workspace" / PERSONA / "memory/MEMORY.md"
    # Deliberate test corruption; the permission boundary is not an adversarial
    # sandbox against the same owner calling chmod or running with root powers.
    memory.chmod(0o644)
    memory.write_text("injected corruption")
    memory.chmod(0o444)
    owner.release_consumer(branch)
    assert await owner.close()
    assert entry["record"]["state"] == "released"
    assert not entry["directory"].exists()
    assert (run / "pa_workspace" / PERSONA / "memory/MEMORY.md").read_text() == "pa memory\n"
