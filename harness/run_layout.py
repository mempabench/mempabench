"""The shape of a run directory: where it lives, and what it keeps inside.

One owner for both halves, because two of them exist: the orchestrator writes
the layout and Experiment Admin reads it back, and a slug rule or a segment
name that drifts between them turns a healthy run into an anomaly report.

"Where it lives" is `runs_root()`. It used to be spelled out eight times over
five modules, so the tree the benchmark writes into, the tree Experiment Admin
lists, the tree the review page reads and the tree a diagnostic refuses to
write into were four independent literals that only happened to agree.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: A run lives at ``<runs_root>/<model>/<condition>/<persona>/<run_id>``, so a
#: run directory always sits exactly this many levels below the runs root.
RUN_LAYOUT_DEPTH = 4

#: Where formal runs live when nothing is configured, read against
#: `PROJECT_ROOT`. This is the pre-configuration spelling, kept as the default
#: so that setting nothing keeps every path exactly where it already is.
DEFAULT_RUNS_ROOT = "workspace/runs"

#: The environment half of the configuration. `scripts/run_orchestrator.py`
#: loads the project `.env` (`load_dotenv(PROJECT_ROOT / ".env")`), so a value
#: written there reaches a real launch; nothing else does, which is why the
#: file half below exists too.
RUNS_ROOT_ENV = "MEMPA_RUNS_ROOT"

#: The file half, for the entrypoints that never load `.env` -- `python -m
#: experiment_admin`, the diagnostics, the review page. Read the way
#: `pa/providers/request_policy.py` reads its catalog, except that absent,
#: empty, or holding no `runs_root:` is not an error here: it means nothing is
#: configured and the default stands.
RUNS_LAYOUT_CONFIG = PROJECT_ROOT / "configs/runs_layout.yaml"

#: The archive sits beside the runs root under this name rather than under a
#: setting of its own, so moving the root moves both halves together and
#: `experiment_admin archive` can never file a run into the tree it left.
ARCHIVE_ROOT_NAME = "runs_archive"


def _configured_runs_root() -> str | None:
    """`runs_root:` as `RUNS_LAYOUT_CONFIG` spells it, or `None`."""
    try:
        raw = yaml.safe_load(RUNS_LAYOUT_CONFIG.read_text(encoding="utf-8"))
    except OSError:
        return None  # absent or unreadable: nothing is configured
    if not isinstance(raw, dict):
        return None  # comments only, or empty: same thing
    value = raw.get("runs_root")
    return str(value) if value else None


def runs_root() -> Path:
    """The resolved root every formal run lives under.

    `MEMPA_RUNS_ROOT` wins, then `configs/runs_layout.yaml`'s `runs_root:`,
    then `DEFAULT_RUNS_ROOT`. The environment comes first because it is the
    per-launch override, where the file is the machine's standing answer.

    A configured value goes through `expanduser()`, and a relative one is read
    **against `PROJECT_ROOT`** -- never against the process's working
    directory, which is whatever the operator happened to be standing in when
    they typed the command. The returned path is resolved.

    Called, never frozen: a module constant or a function default would bind
    the root at import time, before a launcher or a test can set either source.
    """
    configured = os.environ.get(RUNS_ROOT_ENV) or _configured_runs_root()
    root = Path(configured or DEFAULT_RUNS_ROOT).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def archive_root() -> Path:
    """Where `experiment_admin archive` files a run: beside `runs_root()`."""
    return runs_root().parent / ARCHIVE_ROOT_NAME

CHECKPOINTS_SEGMENT = "checkpoints"
BRANCHES_SEGMENT = "branches"
CRASH_SCENE_SEGMENT = "crash_scene"
STAGE_WORK_SEGMENT = "stage_work"
STEPS_SEGMENT = "steps"
SUPERSEDED_SEGMENT = "steps_superseded"

#: The lock folder `scripts.run_locked` keeps beside the runs it launches, one
#: `<sha256 of the run path>.lock` per run. It is not a run.
PRODUCER_LOCKS_SEGMENT = ".producer-locks"

#: The stamp every directory one `supersede_step_dirs` call sets aside carries.
#: It is read back as the cohort id, so the format is part of that contract and
#: not a detail of the caller.
SUPERSEDED_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Directories a run keeps for its own machinery. They carry their own copy of
#: ``run_plan.yaml``/``ledger.json`` or of the world, so none of them is ever a
#: run in its own right. ``stage_work`` belongs here too: it is a step's
#: working copy of the world, and the wrapper's `ix_contract_failure` path
#: leaves one behind.
INTERNAL_SEGMENTS = frozenset(
    {
        CHECKPOINTS_SEGMENT,
        BRANCHES_SEGMENT,
        CRASH_SCENE_SEGMENT,
        STAGE_WORK_SEGMENT,
    }
)


def slug(value: str) -> str:
    """The path-segment form of a plan field (model, condition, persona)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def run_dir_for(
    runs_root: Path | str,
    *,
    pa_model: str,
    memory_condition: str,
    persona_id: str,
    run_id: str,
) -> Path:
    """The directory this run belongs in, spelled out in exactly one place."""
    return (
        Path(runs_root)
        / slug(pa_model)
        / slug(memory_condition or "no_memory")
        / slug(persona_id)
        / run_id
    )


def internal_leftovers(run_dir: Path | str) -> list[str]:
    """What this run has left inside itself, as ``segment/child`` names.

    A `stage_work/<step>` says a step died between `begin_step` and
    `finish_step` — the wrapper's `ix_contract_failure` is raised from inside
    the executor, so it leaves one behind — and `CheckpointManager.save`
    refuses to publish while one is there. A `crash_scene/<timestamp>` says a
    restore has already rolled this run back once. Neither is an error by
    itself; both are facts `experiment_admin inspect` has to be able to show.
    """
    run_dir = Path(run_dir)
    found: list[str] = []
    for segment in (STAGE_WORK_SEGMENT, CRASH_SCENE_SEGMENT):
        try:
            children = sorted(
                child.name
                for child in (run_dir / segment).iterdir()
                if child.is_dir()
            )
        except OSError:
            continue  # absent or unreadable: nothing to report either way
        found.extend(f"{segment}/{name}" for name in children)
    return found


def orphan_branch_dirs(run_dir: Path | str) -> list[str]:
    """Probe branches still on disk, as ``branches/<step_id>/<NN>`` names.

    A probe branch is a throwaway copy of a pinned cut point and is removed
    the moment its evidence is out, so any that survives is one a hard death
    left behind. The orchestrator reaps them at run start; Experiment Admin
    reports them. Only the two-level ``<step_id>/<NN>`` directories count.
    """
    run_dir = Path(run_dir)
    found: list[str] = []
    try:
        step_dirs = sorted(
            child
            for child in (run_dir / BRANCHES_SEGMENT).iterdir()
            if child.is_dir()
        )
    except OSError:
        return found  # absent or unreadable: nothing to report either way
    for step_dir in step_dirs:
        for attempt in sorted(step_dir.iterdir()):
            if attempt.is_dir():
                found.append(f"{BRANCHES_SEGMENT}/{step_dir.name}/{attempt.name}")
    return found


def _free_cohort_stamp(superseded_root: Path) -> str:
    """The earliest second from now on that no cohort has taken yet.

    One stamp is one call, so a second that already holds a cohort cannot be
    reused: the three attempts `harness.orchestrator.PROBE_ATTEMPTS` allows set
    the same probe aside twice, often inside the same second, and both
    set-asides would otherwise land on one name — where `shutil.move` quietly
    nests the second cohort *inside* the first instead of failing. Reusing a
    second another call already stamped would be just as wrong the other way:
    `--cohort <stamp>` would list two calls' probes as one cohort.

    The format is unchanged, so the stamps still sort chronologically and read
    back as before; the only cost is that a cohort set aside in a second that
    is taken carries the next free second instead.
    """
    taken: set[str] = set()
    if superseded_root.is_dir():
        taken = {
            path.name.rsplit("__", 1)[1]
            for path in superseded_root.glob("*__*")
            if path.is_dir()
        }
    moment = datetime.now(timezone.utc).replace(microsecond=0)
    while (stamp := moment.strftime(SUPERSEDED_STAMP_FORMAT)) in taken:
        moment += timedelta(seconds=1)
    return stamp


def supersede_step_dirs(
    run_dir: Path | str, step_ids: list[str]
) -> list[Path]:
    """Set each step's evidence aside as ``steps_superseded/<id>__<stamp>/``.

    One stamp for the whole call, so everything one supersede sets aside reads
    back as one cohort, and no two calls share a stamp (`_free_cohort_stamp`).
    A step with nothing on disk moves nothing and is simply absent from the
    result.

    The layout has two writers — `--rerun-probe-steps` before a resume
    (`scripts/run_orchestrator.py`) and the in-batch void retry
    (`Orchestrator._run_one_probe`) — and the manifest script reads whatever
    either of them wrote, so the move lives here rather than in either caller.
    Only the directory moves: the ledger belongs to the caller, and the
    checkpoints, the mainline world and the memory pool are untouched, which is
    what makes setting a probe aside safe where an accumulation rerun is not.
    """
    run_dir = Path(run_dir)
    stamp = _free_cohort_stamp(run_dir / SUPERSEDED_SEGMENT)
    moved: list[Path] = []
    for step_id in step_ids:
        source = run_dir / STEPS_SEGMENT / step_id
        if not source.is_dir():
            continue
        target = run_dir / SUPERSEDED_SEGMENT / f"{step_id}__{stamp}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append(target)
    return moved


def clear_stage_work(run_dir: Path | str, crash_scene: Path | str) -> list[str]:
    """Move every leftover step working stage into the crash scene.

    Restore is where this becomes explicit rather than incidental. Until now
    the leftovers were cleaned only by the next `begin_step`, which happens to
    run first on a resume — an invariant nobody had written down, and one that
    stops holding the moment the restore target is not the step that died.
    Nothing is deleted: the leftovers land beside the layers being rolled back,
    so the next `save()` has a cut point and the evidence is still on disk.
    """
    working_root = Path(run_dir) / STAGE_WORK_SEGMENT
    if not working_root.is_dir():
        return []
    moved: list[str] = []
    for child in sorted(working_root.iterdir()):
        if not child.is_dir():
            continue
        target = Path(crash_scene) / STAGE_WORK_SEGMENT
        target.mkdir(parents=True, exist_ok=True)
        shutil.move(str(child), str(target / child.name))
        moved.append(child.name)
    return moved
