"""Launch or resume one explicitly named run with a process-lifetime lock.

Use the same arguments as scripts.run_orchestrator, including --run-id or
--plan. Run this entrypoint inside tmux to survive an SSH disconnect.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path

from harness.probe_slots import workspace_root as configured_workspace_root
from harness.run_layout import PRODUCER_LOCKS_SEGMENT


@contextmanager
def workspace_lock(*, exclusive=False, workspace_root: Path | None = None):
    """Managed producers/scorers share this lock; cold maintenance excludes them."""
    folder = (Path(workspace_root).resolve() if workspace_root is not None
              else configured_workspace_root(Path(__file__).resolve().parents[1]))
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".maintenance.lock").open("a") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Workspace is busy with a producer, scorer, or maintenance") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def producer_lock(run_dir: Path, *, workspace_root: Path | None = None):
    with workspace_lock(workspace_root=workspace_root), _run_lock(run_dir):
        yield


@contextmanager
def _run_lock(run_dir: Path):
    run_dir = run_dir.resolve()
    folder = run_dir.parent / PRODUCER_LOCKS_SEGMENT
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(str(run_dir).encode()).hexdigest() + ".lock"
    # Keep the inode across releases and --overwrite; unlinking a lock can
    # let two processes lock different inodes for the same run.
    with (folder / name).open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"A producer already owns this run: {run_dir}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + "\n")
        stream.flush()
        try:
            yield
        finally:
            stream.seek(0)
            stream.truncate()
            fcntl.flock(stream, fcntl.LOCK_UN)


def main() -> int:
    from scripts import run_orchestrator as runner
    from harness.runtime_activity import observe_model_calls

    parser = runner._build_arg_parser()
    args = parser.parse_args()
    if args.plan:
        run_dir = Path(args.plan).resolve().parent
    elif args.run_id:
        run_dir = runner._run_dir(
            pa_model=args.pa_model, memory_condition=args.memory_condition,
            persona_id=args.persona_id, run_id=args.run_id,
        )
    else:
        parser.error("managed launches require an explicit --run-id or --plan")
    shared = configured_workspace_root(Path(__file__).resolve().parents[1])
    if not run_dir.resolve().is_relative_to(shared):
        parser.error("managed run must be inside MEMPA_WORKSPACE_ROOT")
    with producer_lock(run_dir, workspace_root=shared):
        with observe_model_calls(run_dir):
            return asyncio.run(runner._execute_benchmark_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
