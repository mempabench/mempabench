"""Run-directory checkpoints: every layer that a resume has to roll back.

A checkpoint is published only when every layer is present, so a snapshot
either restores the whole cut point or does not exist at all.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from loguru import logger

from harness.memory_runtime import capabilities_for
from harness.run_layout import (
    CRASH_SCENE_SEGMENT,
    STAGE_WORK_SEGMENT,
    clear_stage_work,
)
from harness.run_ledger import RunLedger
from pa.agent.ix.render_types import (
    IX_CONTRACT_VERSION,
    IX_RENDER_SCHEMA_VERSION,
)

PA_WORKSPACE_LAYER = "pa_workspace"
CANONICAL_STAGE_LAYER = "canonical_stage"
SIMULATOR_WORKSPACE_LAYER = "simulator_workspace"
LEDGER_LAYER = "ledger.json"
MEMORY_IO_LAYER = "memory_io.jsonl"
RUN_PLAN_LAYER = "run_plan.yaml"
STAGING_PREFIX = ".staging-"
# Expired checkpoints stay outside the candidate set until remote deletion
# succeeds, so interrupted cleanup can be retried without advertising a
# checkpoint whose Qdrant snapshots have already been partly removed.
PRUNING_PREFIX = f"{STAGING_PREFIX}prune-"
#: Where a rollback puts the snapshots it revoked. Beside `checkpoints/` and
#: never inside it, so the shared candidate enumeration cannot see them again.
CHECKPOINT_ARCHIVE_SEGMENT = "checkpoint_archive"
#: The run-local record of a rollback that has begun and not finished.
RECOVERY_SEGMENT = "recovery"
RECOVERY_PENDING_FILENAME = "pending.json"
BACKEND_STATE_LAYER = "backend_state"
MEM0_HISTORY_FILENAME = "mem0_history.sqlite3"
MEMOS_CUBE_EXPORT_FILENAME = "memos_cube.json"
BRANCH_NAMESPACE_FILENAME = "branch_namespace.json"
# Conditions whose whole memory state lives inside the run directory, so a
# directory copy is already a complete, independent clone.
LOCAL_MEMORY_CONDITIONS = frozenset(
    {"nanobot_file_memory", "no_memory", "oracle_preference"}
)
MEM0_CONDITIONS = frozenset({"mem0", "simple_rag"})
MEMOS_CONDITION = "memos"


def branch_namespace_for(base_namespace: str, branch_dir: Path) -> str:
    """Derive this branch's own remote namespace from its directory name.

    A branch never shares the mainline database, so the name has to be unique
    per branch directory and reproducible from it: `branches/<step>/<attempt>`
    is what distinguishes one probe attempt from the next.
    """
    branch_dir = Path(branch_dir)
    suffix = f"{branch_dir.parent.name}_{branch_dir.name}"
    sanitized = "".join(
        character if character.isalnum() else "_" for character in suffix
    ).strip("_")
    return f"{base_namespace}_br_{sanitized}"


#: Fixed namespace for the branch-local ids below. Frozen: changing it changes
#: every branch node id and a re-branch would stop reproducing the last one.
MEMOS_BRANCH_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "mempa:memos:branch-node-id"
)


def rebind_memos_cube_to_branch(
    payload: dict[str, Any], branch_cube_id: str, mainline_cube_id: str
) -> dict[str, Any]:
    """Rewrite one cube export so it restores as a true copy of another cube.

    MemOS Community keeps every cube in one Neo4j database and one Qdrant
    collection, so native ids are global to the service and the only thing
    separating cubes is the `user_name` property. Its validator ties four
    things to the restore target — `cube_id`, every node's `user_name`, every
    point payload's `user_name`, and the point-id set against the node-id set
    (`memory_raw_repos/MemOS/src/memos/mempa_checkpoint.py:50-51,58-59,73,81-82`)
    — and refuses a payload that names another cube. Reusing the mainline's
    native ids under a branch cube would collide with the cube that already
    owns them, which `import_cube` also refuses (`:139-149`), so the branch
    gets ids of its own: `uuid5` of this branch cube and the mainline id, which
    makes a re-branch of the same directory reproduce the same ids exactly.

    Node `properties.id`, node `properties.working_binding`, point `id` and
    point `payload.working_binding` are the whole id-bearing surface of a real
    export, plus both endpoints of an edge; properties are deep-walked rather
    than listed so a field that starts carrying an id later is remapped too.
    """
    if branch_cube_id in (mainline_cube_id, payload.get("cube_id")):
        raise ValueError(
            "A branch restore must target its own cube and never the "
            "mainline's, because import_cube clears the target cube before it "
            "writes (memory_raw_repos/MemOS/src/memos/mempa_checkpoint.py:152): "
            f"branch_cube_id={branch_cube_id!r} "
            f"mainline_cube_id={mainline_cube_id!r} "
            f"payload_cube_id={payload.get('cube_id')!r}"
        )
    mapped = {
        node["properties"]["id"]: str(
            uuid.uuid5(
                MEMOS_BRANCH_ID_NAMESPACE,
                f"{branch_cube_id}:{node['properties']['id']}",
            )
        )
        for node in payload["nodes"]
    }

    def remap(value: Any) -> Any:
        if isinstance(value, str):
            return mapped.get(value, value)
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        return value

    return {
        **payload,
        "cube_id": branch_cube_id,
        "nodes": [
            {
                **node,
                "properties": {
                    **remap(node["properties"]),
                    "user_name": branch_cube_id,
                },
            }
            for node in payload["nodes"]
        ],
        "edges": [
            {
                **edge,
                "source": mapped[edge["source"]],
                "target": mapped[edge["target"]],
            }
            for edge in payload["edges"]
        ],
        "points": [
            {
                **point,
                "id": mapped[point["id"]],
                "payload": {
                    **remap(point["payload"]),
                    "user_name": branch_cube_id,
                },
            }
            for point in payload["points"]
        ],
    }


#: Builds a memory adapter on one branch's restored namespace.
BranchAdapterFactory = Callable[[str, Path], Any]


def read_branch_namespace(branch_dir: Path) -> dict[str, Any] | None:
    """Return the remote namespace a branch restored, if it restored one."""
    path = Path(branch_dir) / BACKEND_STATE_LAYER / BRANCH_NAMESPACE_FILENAME
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _set_prompt_history_start_to_end(session_path: Path) -> None:
    """Point one session's prompt window past its last message.

    Only the metadata line is rewritten: the messages and the other two
    cursors stay byte-for-byte as the snapshot recorded them.
    """
    lines = session_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    try:
        metadata = json.loads(lines[0])
    except json.JSONDecodeError:
        return
    if not isinstance(metadata, dict) or metadata.get("_type") != "metadata":
        return
    message_count = sum(1 for line in lines[1:] if line.strip())
    if metadata.get("prompt_history_start_index") == message_count:
        return
    metadata["prompt_history_start_index"] = message_count
    lines[0] = json.dumps(metadata, ensure_ascii=False)
    session_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class IncompleteCheckpointError(RuntimeError):
    """A checkpoint could not be published because a layer was missing."""


class RecoveryInProgressError(RuntimeError):
    """An unfinished rollback recovery forbids what was asked for.

    Raised at every entry point that would move the run on from a half-rolled
    back state: saving, pruning or branching while a recovery is open, and
    restoring to any target but the one that recovery already committed to.
    """


def _sync_directory(directory: Path) -> None:
    """Make a rename in ``directory`` durable, where the platform allows it.

    A marker is only worth publishing if it survives the power cut it exists
    for, and on POSIX the rename itself is not durable until the directory is
    synced. Platforms that refuse to open or sync a directory are left alone:
    the marker is written and replaced atomically regardless, and this is the
    extra guarantee, not the mechanism.
    """
    try:
        handle = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one small JSON file as a single, complete, durable step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _sync_directory(path.parent)


def pending_recovery_path(run_dir: Path) -> Path:
    return Path(run_dir) / RECOVERY_SEGMENT / RECOVERY_PENDING_FILENAME


def read_pending_recovery(run_dir: Path) -> dict[str, Any] | None:
    """The rollback this run began and has not finished, or None.

    A marker that is present but unreadable is *not* reported as "no recovery
    in progress": that is the one reading of it that would let a run walk past
    a half-rolled back state, which is the whole thing this file prevents.
    """
    path = pending_recovery_path(run_dir)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryInProgressError(
            "A rollback recovery marker is present but cannot be read, so this "
            "run may be half rolled back and must not be continued: "
            f"path={path} error={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(record, dict) or not record.get("target_checkpoint_id"):
        raise RecoveryInProgressError(
            "A rollback recovery marker is present but does not name its "
            f"target, so the run must not be continued: path={path} "
            f"value={record!r}"
        )
    return record


def verify_layers(
    cp_dir: Path,
    *,
    memory_condition: str,
    pinned: bool,
) -> list[str]:
    """Return the names of the layers this checkpoint directory is missing.

    Pure: reads the directory and writes nothing.
    """
    missing: list[str] = []
    for layer in (
        PA_WORKSPACE_LAYER,
        CANONICAL_STAGE_LAYER,
        SIMULATOR_WORKSPACE_LAYER,
    ):
        if not (cp_dir / layer).is_dir():
            missing.append(layer)
    ledger_path = cp_dir / LEDGER_LAYER
    if not ledger_path.is_file():
        missing.append(LEDGER_LAYER)
    if not (cp_dir / MEMORY_IO_LAYER).is_file():
        # The audit file may legitimately not exist before the first step, but
        # once the ledger records a step the audit must roll back with it.
        if _snapshot_ledger_has_steps(ledger_path):
            missing.append(MEMORY_IO_LAYER)
    if pinned and not (cp_dir / RUN_PLAN_LAYER).is_file():
        missing.append(RUN_PLAN_LAYER)
    return missing


def checkpoint_restorable(
    cp_dir: Path,
    *,
    memory_condition: str,
) -> tuple[bool, list[str]]:
    """Whether this one snapshot directory is complete enough to restore from.

    Three things have to hold together here: every layer is present, this
    backend declares restore support, and the external snapshots that
    `metadata.json` names are actually reachable. That is the whole of the
    per-checkpoint judgement and no more — whether the *run* can be resumed
    also depends on its frozen `run_plan.yaml`, which lives one level up.
    `run_restorable` is the run-level owner that folds both together, and it
    is what a caller asking "can this run be continued" should ask.

    Pure: reads the checkpoint directory and writes nothing.
    """
    cp_dir = Path(cp_dir)
    metadata_path = cp_dir / "metadata.json"
    if not metadata_path.is_file():
        return False, [f"missing metadata.json: {cp_dir}"]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"unreadable metadata.json: {type(exc).__name__}: {exc}"]
    if not isinstance(metadata, dict):
        return False, ["metadata.json is not a mapping"]

    reasons = [
        f"missing layer: {layer}"
        for layer in verify_layers(
            cp_dir,
            memory_condition=memory_condition,
            pinned=bool(metadata.get("pinned") or metadata.get("retained")),
        )
    ]
    if not capabilities_for(memory_condition).supports_restore:
        reasons.append(
            "backend declares no restore support: "
            f"memory_condition={memory_condition!r}"
        )
    reasons.extend(_unreachable_external_refs(cp_dir, metadata, memory_condition))
    return (not reasons), reasons


def run_plan_contract_reason(run_dir: Path) -> str | None:
    """Why a resume of this run would be refused at `load_run_plan`, or None.

    The half of `run_restorable` that has nothing to do with rolling back: a run
    whose `run_plan.yaml` cannot be read cannot be continued at all, snapshot
    or no snapshot. A caller judging a run that stopped *cleanly* — there is
    nothing to restore, so the checkpoint gate does not apply — asks this half
    on its own. A plan frozen under another IX wrapper or pool version loads
    and resumes; `run_info.yaml` records what each launch ran under.

    Pure: reads the run directory and writes nothing.
    """
    plan_path = Path(run_dir) / RUN_PLAN_LAYER
    try:
        plan_data = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return f"unreadable {RUN_PLAN_LAYER}: {type(exc).__name__}: {exc}"
    if not isinstance(plan_data, dict):
        return f"{RUN_PLAN_LAYER} is not a mapping: {plan_path}"
    return None


def run_restorable(
    run_dir: Path,
    *,
    memory_condition: str,
) -> tuple[bool, list[str]]:
    """The one judgement of whether a dead run can be rolled back and resumed.

    Two independent gates decide it, and every caller — the harness and
    Experiment Admin alike — asks this function rather than only the half it
    happens to know about: the `run_plan.yaml` is readable, because otherwise
    `load_run_plan` refuses the resume before a snapshot is ever opened; and at
    least one published checkpoint passes `checkpoint_restorable`. Asking only
    the second is what let Admin report a run as resumable that the harness
    would reject on sight.

    Pure: reads the run directory and writes nothing.
    """
    run_dir = Path(run_dir)
    reasons: list[str] = []

    contract_reason = run_plan_contract_reason(run_dir)
    if contract_reason is not None:
        reasons.append(contract_reason)

    # An unfinished rollback is stated in its own words rather than left to be
    # inferred from the snapshots, because the snapshots are exactly what does
    # not say it: one of the older ones is complete, and reading only that is
    # how a half-rolled back run gets reported as ready to continue.
    #
    # Reporting is not acting, and this function only reports. A marker that
    # cannot be parsed is a reason the run is not ready, not an exception out
    # of a function whose whole contract is to return a verdict — Admin calls
    # it to print reasons. The entry points that would move the run on refuse
    # outright instead; that is their job, and this is this one's.
    try:
        pending = read_pending_recovery(run_dir)
    except RecoveryInProgressError as exc:
        reasons.append(str(exc))
        candidates = _published_checkpoint_dirs(run_dir)
    else:
        if pending is not None:
            reasons.append(
                "rollback recovery unfinished, target "
                f"{str(pending['target_checkpoint_id'])!r}: finish it through "
                "CheckpointManager.restore on that target before this run "
                "continues"
            )
        candidates = _published_checkpoints(run_dir)

    failures: list[str] = []
    for cp_dir in candidates:
        restorable, cp_reasons = checkpoint_restorable(
            cp_dir, memory_condition=memory_condition
        )
        if restorable:
            break
        failures.append(f"{cp_dir.name}: {'; '.join(cp_reasons)}")
    else:
        if failures:
            reasons.append("no restorable checkpoint: " + " | ".join(failures))
        else:
            reasons.append("no checkpoint to roll back to")

    return (not reasons), reasons


def _published_checkpoint_dirs(run_dir: Path) -> list[Path]:
    """Every published snapshot directory under ``run_dir``, revoked or not.

    A `.staging-` directory left by a half-written save is excluded by name as
    well as by its missing `metadata.json`, so one that died after writing its
    metadata still never reads as a snapshot.

    This is the raw contents of `checkpoints/`. Only the rollback itself wants
    that: everyone asking which cut points may be *used* asks
    `_published_checkpoints` below.
    """
    checkpoints_dir = Path(run_dir) / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    return [
        child
        for child in sorted(checkpoints_dir.iterdir())
        if child.is_dir()
        and not child.name.startswith(STAGING_PREFIX)
        and (child / "metadata.json").is_file()
    ]


def _published_checkpoints(run_dir: Path) -> list[Path]:
    """The snapshots a caller may restore from or branch right now.

    The one candidate rule, shared by the resume path and by Experiment Admin
    (`run_restorable`) so the two cannot disagree about what this run has to go
    back to. While a rollback is unfinished that is the target and the history
    already confirmed to sit behind it, and nothing else: the snapshots the
    recovery is on its way to archiving must not be offered to anybody as a cut
    point, because they are exactly what the rollback revoked.
    """
    run_dir = Path(run_dir)
    pending = read_pending_recovery(run_dir)
    if pending is None:
        return _published_checkpoint_dirs(run_dir)
    usable = {str(pending["target_checkpoint_id"])}
    usable.update(str(name) for name in (pending.get("history") or []))
    return [
        child
        for child in _published_checkpoint_dirs(run_dir)
        if child.name in usable
    ]


def _checkpoint_history(cp_dir: Path) -> tuple[str, set[str]] | None:
    """This snapshot's own cut point and the done set its ledger carries.

    None when either cannot be read, which is a verdict and not an absence:
    a candidate whose own history is unknown cannot be shown to belong to the
    target's, so it is archived rather than silently kept.
    """
    try:
        metadata = json.loads((cp_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    step_id = metadata.get("step_id") or metadata.get("checkpoint_id")
    if not isinstance(step_id, str) or not step_id:
        return None
    ledger_path = cp_dir / LEDGER_LAYER
    try:
        ledger_data = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(ledger_data, dict):
        return None
    return step_id, _snapshot_done_step_ids(ledger_path)


def partition_checkpoints_by_target(
    run_dir: Path, target_checkpoint_id: str
) -> tuple[list[str], list[dict[str, str]]]:
    """Split the published snapshots into the target's history and the revoked.

    The done set of the *target's own* ledger is what says which steps this run
    still has behind it once the rollback lands. A candidate stays only when
    its own cut point is one of those steps and its own history is contained in
    the target's. That one rule covers a superseded rerun, a forced event, a
    skipped session number and a snapshot with the same done *count* but an
    incompatible history alike — without guessing from names, from ordering or
    from timestamps, none of which can tell a revoked snapshot from a kept one.

    Pure: reads the run directory and writes nothing.
    """
    run_dir = Path(run_dir)
    target_dir = run_dir / "checkpoints" / target_checkpoint_id
    target = _checkpoint_history(target_dir)
    if target is None:
        raise IncompleteCheckpointError(
            "Refusing to roll back to a snapshot whose own metadata or ledger "
            "cannot be read, before any live state is touched: "
            f"checkpoint_id={target_checkpoint_id!r} path={target_dir}"
        )
    _, target_done = target

    kept = [target_checkpoint_id]
    revoked: list[dict[str, str]] = []
    for cp_dir in _published_checkpoint_dirs(run_dir):
        if cp_dir.name == target_checkpoint_id:
            continue
        history = _checkpoint_history(cp_dir)
        if history is None:
            revoked.append(
                {
                    "checkpoint_id": cp_dir.name,
                    "reason": "its own metadata or ledger cannot be read",
                }
            )
            continue
        step_id, done = history
        if step_id not in target_done:
            revoked.append(
                {
                    "checkpoint_id": cp_dir.name,
                    "reason": (
                        f"its cut point {step_id!r} is not a done step of the "
                        "target's ledger"
                    ),
                }
            )
            continue
        if not done <= target_done:
            revoked.append(
                {
                    "checkpoint_id": cp_dir.name,
                    "reason": (
                        "its history is not contained in the target's: "
                        f"extra={sorted(done - target_done)!r}"
                    ),
                }
            )
            continue
        kept.append(cp_dir.name)
    return sorted(kept), revoked


def _unreachable_external_refs(
    cp_dir: Path,
    metadata: dict[str, Any],
    memory_condition: str,
) -> list[str]:
    """Report external snapshot references this checkpoint cannot reach.

    Whatever the snapshot left inside the checkpoint directory is checked as a
    file. Mem0's collection snapshot lives on the remote server, so only its
    declaration and the history database beside it can be judged without a
    live client; a read-only scan never opens a connection.
    """
    if memory_condition in LOCAL_MEMORY_CONDITIONS:
        return []
    reasons: list[str] = []

    if memory_condition in MEM0_CONDITIONS:
        required = ("mem0_collection_name", "mem0_snapshot_ref")
        if memory_condition == "mem0":
            required += ("mem0_entity_collection_name",)
            if "mem0_entity_snapshot_ref" not in metadata:
                reasons.append("metadata omits enabled Mem0 entity index snapshot")
            entity_name = metadata.get("mem0_entity_collection_name")
            if entity_name and entity_name != f"{metadata.get('mem0_collection_name')}_entities":
                reasons.append("Mem0 entity snapshot names a different scope")
        path_key = None
        local_file = f"{BACKEND_STATE_LAYER}/{MEM0_HISTORY_FILENAME}"
    elif memory_condition == MEMOS_CONDITION:
        required = ("memos_cube_id", "memos_cube_export_path")
        path_key = "memos_cube_export_path"
        local_file = None
    else:
        return [
            "backend has no checkpoint export yet: "
            f"memory_condition={memory_condition!r}"
        ]

    for key in required:
        if not metadata.get(key):
            reasons.append(f"metadata does not name {key}")
    if path_key is not None:
        reference = metadata.get(path_key)
        if reference and not (cp_dir / str(reference)).is_file():
            reasons.append(f"external snapshot is unreachable: {reference}")
    if local_file is not None and not (cp_dir / local_file).is_file():
        reasons.append(f"external snapshot is unreachable: {local_file}")
    return reasons


def _snapshot_ledger_has_steps(ledger_path: Path) -> bool:
    if not ledger_path.is_file():
        return False
    try:
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(isinstance(data, dict) and (data.get("steps") or {}))


def _snapshot_done_step_ids(ledger_path: Path) -> set[str]:
    try:
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, dict):
        return set()
    return {
        step_id
        for step_id, record in steps.items()
        if isinstance(record, dict) and record.get("status") in {"done", "skipped"}
    }


#: Set while `_restore_backend_state_into_branch` holds the shared Qdrant
#: restore lock for this thread, so the inner restore lease does not re-take it.
_BRANCH_QDRANT_LEASE = threading.local()


class CheckpointManager:
    """Publish and restore whole-run-directory checkpoints."""

    def __init__(
        self,
        run_dir: Path,
        *,
        ledger: RunLedger,
        memory_condition: str,
        memory_backend_namespace: str | None = None,
        backend_client: Any | None = None,
        branch_adapter_factory: BranchAdapterFactory | None = None,
        rolling_window: int = 3,
        retained_step_ids: frozenset[str] = frozenset(),
        retention_interval: int | None = None,
        retention_mainline_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.run_dir = Path(run_dir)
        self.ledger = ledger
        self.memory_condition = memory_condition
        self.memory_backend_namespace = memory_backend_namespace
        # The live client for an external backend. Its state lives outside the
        # run directory, so a snapshot has to be taken through it.
        self.backend_client = backend_client
        # Builds one adapter on a branch's own restored namespace. Injected by
        # the run entrypoint, which owns the adapter options the mainline was
        # built from, so a branch is built through the same registry call.
        self.branch_adapter_factory = branch_adapter_factory
        # Opt-in factories for a setting-owned, read-only Probe memory resource.
        self.shared_query_factory: Any | None = None
        self.shared_query_identity: str | None = None
        self.rolling_window = rolling_window
        self.retained_step_ids = retained_step_ids
        if retention_interval is not None and (
            retention_interval < 1 or not retention_mainline_ids
        ):
            raise ValueError("retention requires a positive interval and frozen mainline IDs")
        self.retention_interval = retention_interval
        self.retention_mainline_ids = retention_mainline_ids
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        return self.checkpoints_dir / checkpoint_id

    # ── save ──────────────────────────────────────────────────────

    def save(
        self,
        checkpoint_id: str,
        *,
        pinned: bool,
        summary: dict[str, Any] | None = None,
        summary_schema: int | None = None,
    ) -> Path:
        return self._save(
            checkpoint_id,
            pinned=pinned,
            terminal_status="done",
            summary=summary,
            summary_schema=summary_schema,
        )

    def save_skipped(
        self,
        checkpoint_id: str,
        *,
        pinned: bool,
        summary: dict[str, Any],
        summary_schema: int,
        artifacts: dict[str, str] | None = None,
        error: str | None = None,
        failure_kind: str | None = None,
    ) -> Path:
        """Publish and permanently retain one real partial skipped state."""
        return self._save(
            checkpoint_id,
            pinned=pinned,
            terminal_status="skipped",
            summary=summary,
            summary_schema=summary_schema,
            retain=True,
            artifacts=artifacts,
            error=error,
            failure_kind=failure_kind,
        )

    def _save(
        self,
        checkpoint_id: str,
        *,
        pinned: bool,
        terminal_status: str,
        summary: dict[str, Any] | None,
        summary_schema: int | None,
        retain: bool = False,
        artifacts: dict[str, str] | None = None,
        error: str | None = None,
        failure_kind: str | None = None,
    ) -> Path:
        self._require_no_pending_recovery(f"publish checkpoint {checkpoint_id!r}")
        retained = retain or checkpoint_id in self.retained_step_ids
        if terminal_status == "done" and self.retention_interval is not None and checkpoint_id in self.retention_mainline_ids:
            # save publishes a ledger with this step overridden to done. Count
            # that same set, not planned ordinals or snapshots left after prune.
            # The live ledger is authoritative after resume/rollback; skipped
            # rows and probes must never advance the successful-mainline count.
            completed = {
                step_id for step_id, record in self.ledger.steps.items()
                if step_id in self.retention_mainline_ids and record.status == "done"
            }
            completed.add(checkpoint_id)
            retained = retained or len(completed) % self.retention_interval == 0
        staging = self.checkpoints_dir / f"{STAGING_PREFIX}{checkpoint_id}"
        if staging.exists():
            if terminal_status == "skipped":
                raise IncompleteCheckpointError(
                    "A prior skipped-checkpoint publication left staging state; "
                    "its external snapshot outcome is uncertain and must be "
                    "reconciled before another snapshot attempt"
                )
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            self._copy_pa_workspace(staging)
            self._copy_canonical_stage(staging)
            self._copy_simulator_workspace(staging)
            self._copy_ledger(
                staging,
                checkpoint_id,
                terminal_status=terminal_status,
                summary=summary,
                summary_schema=summary_schema,
                artifacts=artifacts,
                error=error,
                failure_kind=failure_kind,
            )
            self._copy_memory_io(staging)
            self._copy_pinned_config(staging, pinned=pinned or retained)
            backend_metadata = self._copy_backend_state(staging)
            missing = verify_layers(
                staging,
                memory_condition=self.memory_condition,
                pinned=pinned or retained,
            )
            if missing:
                raise IncompleteCheckpointError(
                    "Refusing to publish a checkpoint that is missing layers: "
                    f"checkpoint_id={checkpoint_id!r} missing={missing!r}"
                )
        except BaseException:
            if terminal_status != "skipped":
                shutil.rmtree(staging, ignore_errors=True)
            raise

        cp_dir = self._checkpoint_path(checkpoint_id)
        if cp_dir.exists():
            shutil.rmtree(cp_dir)
        os.replace(staging, cp_dir)

        metadata: dict[str, Any] = {
            "checkpoint_id": checkpoint_id,
            "step_id": checkpoint_id,
            "terminal_status": terminal_status,
            "pinned": pinned,
            "memory_condition": self.memory_condition,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # A record of the wrapper that produced the PA session inside
            # this snapshot; nothing compares it.
            "ix_contract_version": IX_CONTRACT_VERSION,
            "ix_render_schema_version": IX_RENDER_SCHEMA_VERSION,
        }
        if retained:
            metadata["retained"] = True
        if (pinned or retained) and self.memory_backend_namespace:
            metadata["memory_backend_namespace"] = self.memory_backend_namespace
        metadata.update(backend_metadata)
        (cp_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Checkpoint {} published (pinned={})", checkpoint_id, pinned)
        # Retention runs only after a successful publication, so a failed save
        # can never take the older snapshot it was meant to replace with it.
        self.prune(self.rolling_window)
        return cp_dir

    def _copy_pa_workspace(self, staging: Path) -> None:
        source = self.run_dir / PA_WORKSPACE_LAYER
        if source.is_dir():
            shutil.copytree(source, staging / PA_WORKSPACE_LAYER)

    def _copy_canonical_stage(self, staging: Path) -> None:
        working_root = self.run_dir / STAGE_WORK_SEGMENT
        if working_root.is_dir() and any(working_root.iterdir()):
            raise IncompleteCheckpointError(
                "Refusing to checkpoint while a step working stage is still "
                f"present, so this is not a cut point: {working_root}"
            )
        source = self.run_dir / CANONICAL_STAGE_LAYER
        if source.is_dir():
            shutil.copytree(source, staging / CANONICAL_STAGE_LAYER)

    def _copy_simulator_workspace(self, staging: Path) -> None:
        source = self.run_dir / SIMULATOR_WORKSPACE_LAYER
        if source.is_dir():
            shutil.copytree(source, staging / SIMULATOR_WORKSPACE_LAYER)

    def _copy_ledger(
        self,
        staging: Path,
        checkpoint_id: str,
        *,
        terminal_status: str = "done",
        summary: dict[str, Any] | None = None,
        summary_schema: int | None = None,
        artifacts: dict[str, str] | None = None,
        error: str | None = None,
        failure_kind: str | None = None,
    ) -> None:
        # The snapshot carries a ledger in which this step is already done, so
        # it agrees with the world and memory captured beside it. The mainline
        # ledger is untouched: it stays running until publication succeeds.
        # `summary` is the one the caller's own `mark_done` is about to write,
        # so a rollback onto this cut point keeps what the step produced —
        # for a forced event, its `preference_overlay` flip.
        self.ledger.dump_to(
            staging / LEDGER_LAYER,
            override_status=(checkpoint_id, terminal_status),
            override_summary=summary,
            override_summary_schema=summary_schema,
            override_artifacts=artifacts,
            override_error=error,
            override_failure_kind=failure_kind,
        )

    def _copy_memory_io(self, staging: Path) -> None:
        source = self.run_dir / MEMORY_IO_LAYER
        if source.is_file():
            shutil.copy2(source, staging / MEMORY_IO_LAYER)

    def _copy_pinned_config(self, staging: Path, *, pinned: bool) -> None:
        if not pinned:
            return
        source = self.run_dir / RUN_PLAN_LAYER
        if source.is_file():
            shutil.copy2(source, staging / RUN_PLAN_LAYER)

    # ── external backend state ────────────────────────────────────

    def _require_backend_client(self) -> Any:
        if self.backend_client is None:
            raise IncompleteCheckpointError(
                "An external memory backend cannot be checkpointed without its "
                "client, because its state does not live in the run directory: "
                f"memory_condition={self.memory_condition!r}"
            )
        return self.backend_client

    def _copy_backend_state(self, staging: Path) -> dict[str, Any]:
        """Capture the part of memory that lives outside the run directory.

        Returns the snapshot references to record in `metadata.json`. Anything
        that fails here fails the whole publication: a checkpoint that names a
        backend it did not capture is worse than no checkpoint.
        """
        if self.memory_condition in LOCAL_MEMORY_CONDITIONS:
            return {}
        client = self._require_backend_client()
        target = staging / BACKEND_STATE_LAYER
        target.mkdir(parents=True, exist_ok=True)

        if self.memory_condition in MEM0_CONDITIONS:
            from memory.backends.mem0_client import collection_exists, snapshot_collection

            history = (
                self.run_dir / BACKEND_STATE_LAYER / MEM0_HISTORY_FILENAME
            )
            if not history.is_file():
                raise IncompleteCheckpointError(
                    "Mem0 checkpoint is missing its history database, so the "
                    f"snapshot would be partial: {history}"
                )
            shutil.copy2(history, target / MEM0_HISTORY_FILENAME)
            collection = self._require_namespace()
            metadata = {
                "mem0_collection_name": collection,
                "mem0_snapshot_ref": snapshot_collection(client, collection),
            }
            if self.memory_condition == "mem0":
                entity_collection = f"{collection}_entities"
                metadata["mem0_entity_collection_name"] = entity_collection
                metadata["mem0_entity_snapshot_ref"] = (
                    snapshot_collection(client, entity_collection)
                    if collection_exists(client, entity_collection) else None
                )
            return metadata

        if self.memory_condition == MEMOS_CONDITION:
            cube_id = self._require_namespace()
            payload = client.export_cube(cube_id)
            (target / MEMOS_CUBE_EXPORT_FILENAME).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return {
                "memos_cube_id": cube_id,
                "memos_cube_export_path": (
                    f"{BACKEND_STATE_LAYER}/{MEMOS_CUBE_EXPORT_FILENAME}"
                ),
            }

        raise IncompleteCheckpointError(
            "This memory backend has no checkpoint export yet, so a snapshot "
            f"would silently omit its memory: memory_condition={self.memory_condition!r}"
        )

    def _require_namespace(self) -> str:
        if not self.memory_backend_namespace:
            raise IncompleteCheckpointError(
                "An external memory backend needs its namespace to be known "
                "before it can be captured: "
                f"memory_condition={self.memory_condition!r}"
            )
        return self.memory_backend_namespace

    @staticmethod
    def _required_ref(metadata: dict[str, Any], key: str) -> Any:
        value = metadata.get(key)
        if not value:
            raise IncompleteCheckpointError(
                "This checkpoint does not name the external snapshot it would "
                "be restored from, so nothing may be dropped: "
                f"checkpoint_id={metadata.get('checkpoint_id')!r} missing={key!r}"
            )
        return value

    def _restore_backend_state(
        self, cp_dir: Path, metadata: dict[str, Any]
    ) -> None:
        """Put the external backend back under its own name.

        The reference is checked before anything is dropped, so a checkpoint
        that cannot be reloaded never destroys the state it was meant to
        replace. No dirty copy is kept: the name is reused exactly.
        """
        if self.memory_condition in LOCAL_MEMORY_CONDITIONS:
            return
        client = self._require_backend_client()

        if self.memory_condition in MEM0_CONDITIONS:
            from memory.backends.mem0_client import (
                collection_exists,
                delete_collection,
                restore_collection_as,
            )
            from memory.backends.qdrant_runtime import qdrant_restore_lease

            snapshot_ref = self._required_ref(metadata, "mem0_snapshot_ref")
            collection = self._required_ref(metadata, "mem0_collection_name")
            snapshot_history = (
                cp_dir / BACKEND_STATE_LAYER / MEM0_HISTORY_FILENAME
            )
            if not snapshot_history.is_file():
                raise IncompleteCheckpointError(
                    "This checkpoint has no Mem0 history database to restore: "
                    f"{snapshot_history}"
                )
            entity = self._mem0_entity_snapshot(metadata, collection)
            with qdrant_restore_lease():
                delete_collection(client, collection)
                restore_collection_as(client, snapshot_ref, collection)
                if entity is not None:
                    name, reference = entity
                    if collection_exists(client, name):
                        delete_collection(client, name)
                    if reference is not None:
                        restore_collection_as(client, reference, name)
            live_history = (
                self.run_dir / BACKEND_STATE_LAYER / MEM0_HISTORY_FILENAME
            )
            live_history.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot_history, live_history)
            return

        if self.memory_condition == MEMOS_CONDITION:
            cube_id = self._required_ref(metadata, "memos_cube_id")
            export_path = cp_dir / self._required_ref(
                metadata, "memos_cube_export_path"
            )
            if not export_path.is_file():
                raise IncompleteCheckpointError(
                    f"This checkpoint has no MemOS cube export to restore: {export_path}"
                )
            payload = json.loads(export_path.read_text(encoding="utf-8"))
            # The service validates the full snapshot before replacing this cube.
            result = client.import_cube_as(payload, cube_id)
            restored = result.get("point_count")
            if restored != len(payload.get("points") or []):
                raise IncompleteCheckpointError(
                    "MemOS restored a different number of vectors than this "
                    f"checkpoint holds: restored={restored!r} "
                    f"expected={len(payload.get('points') or [])!r}"
                )
            return

        raise IncompleteCheckpointError(
            "This memory backend has no checkpoint restore yet: "
            f"memory_condition={self.memory_condition!r}"
        )

    def _mem0_entity_snapshot(
        self, metadata: dict[str, Any], collection: str
    ) -> tuple[str, str | None] | None:
        if self.memory_condition != "mem0":
            return None
        name = f"{collection}_entities"
        if (
            metadata.get("mem0_entity_collection_name") != name
            or "mem0_entity_snapshot_ref" not in metadata
        ):
            raise IncompleteCheckpointError("Mem0 checkpoint omits its enabled entity index")
        reference = metadata["mem0_entity_snapshot_ref"]
        if reference is not None and (
            not isinstance(reference, str) or not reference.strip()
        ):
            raise IncompleteCheckpointError("Mem0 entity snapshot reference is invalid")
        return name, reference

    def build_branch_adapter(self, branch_dir: Path) -> Any | None:
        """The adapter a probe in this branch reads its memory through.

        `_restore_backend_state_into_branch` has already imported the pinned
        snapshot into a namespace of this branch's own; this builds the
        adapter that reads it, so the probe sees the cut point's memory
        instead of nothing. Returns None when there is no namespace to read —
        a file-backed condition, whose whole memory is the directory copy —
        or when no factory was injected. A factory may return an awaitable
        instead of an adapter; the caller awaits what is awaitable.
        """
        record = read_branch_namespace(branch_dir)
        if record is None or self.branch_adapter_factory is None:
            return None
        namespace = record.get("namespace")
        if not namespace:
            return None
        return self.branch_adapter_factory(namespace, Path(branch_dir))

    def release_branch_namespace(self, branch_dir: Path) -> str | None:
        """Drop the remote namespace one branch restored for itself.

        Called when the branch is discarded, so 33 probes per run do not leave
        a collection or cube behind each.
        """
        record = read_branch_namespace(branch_dir)
        if record is None:
            return None
        if record.get("shared_owner"):
            # Consumer directories never own the referenced remote namespace.
            return None
        namespace = record.get("namespace")
        condition = record.get("memory_condition")
        if not namespace:
            return None
        client = self.backend_client
        if client is None:
            return None
        if condition in MEM0_CONDITIONS:
            from memory.backends.mem0_client import collection_exists, delete_collection

            if condition == "mem0":
                entity_name = f"{namespace}_entities"
                if collection_exists(client, entity_name):
                    delete_collection(client, entity_name)
            delete_collection(client, namespace)
            return namespace
        if condition == MEMOS_CONDITION:
            client.delete_cube(namespace)
            return namespace
        return None

    # ── retention ─────────────────────────────────────────────────

    def progress_key(self, checkpoint_id: str) -> tuple[int, str]:
        """How far into the run this cut point is: done steps, then timestamp.

        Not the wall clock on its own:
        `timestamp` comes from `datetime.now`, so a host clock that steps
        backwards can make an earlier cut point read as the newest one — and
        both the resume that picks a checkpoint and the retention that drops
        one would then act on a completed, paid session. Every checkpoint
        carries the ledger it was published with, and how many steps that
        ledger has behind it only ever grows.
        """
        done = _snapshot_done_step_ids(
            self._checkpoint_path(checkpoint_id) / LEDGER_LAYER
        )
        timestamp = str(self.metadata(checkpoint_id).get("timestamp") or "")
        return len(done), timestamp

    def prune(self, window: int) -> list[str]:
        """Drop rolling checkpoints; preserve probe pins and retained cut points."""
        self._require_no_pending_recovery("prune checkpoints")
        rolling: list[tuple[tuple[int, str], str]] = []
        for checkpoint_id in self.list_checkpoints():
            metadata = self.metadata(checkpoint_id)
            if metadata.get("pinned") or metadata.get("retained"):
                continue
            rolling.append((self.progress_key(checkpoint_id), checkpoint_id))
        rolling.sort()
        removed: list[str] = []
        for _position, checkpoint_id in rolling[: max(len(rolling) - window, 0)]:
            source = self._checkpoint_path(checkpoint_id)
            if self.memory_condition in MEM0_CONDITIONS:
                pending = self.checkpoints_dir / f"{PRUNING_PREFIX}{checkpoint_id}-{uuid.uuid4().hex}"
                os.replace(source, pending)
            else:
                shutil.rmtree(source)
            removed.append(checkpoint_id)
        if self.memory_condition in MEM0_CONDITIONS:
            self._finish_mem0_pruning()
        return removed

    def _finish_mem0_pruning(self) -> None:
        """Delete only expired refs owned by this run; preserve shared refs.

        Qdrant snapshots live outside the local checkpoint tree. Removing only
        that tree leaks a full remote snapshot after every completed step.
        Pending directories retain the exact refs across process failures;
        missing snapshots are harmless when retrying a partially finished job.
        Rollback archives keep their refs, as do all remaining checkpoints.
        """
        pending = sorted(self.checkpoints_dir.glob(f"{PRUNING_PREFIX}*"))
        if not pending:
            return
        from memory.backends.mem0_client import _qdrant_admin

        collection = self._require_namespace()
        keys = ("mem0_snapshot_ref", "mem0_entity_snapshot_ref")
        metadata_paths = [
            path / "metadata.json" for path in _published_checkpoint_dirs(self.run_dir)
        ]
        metadata_paths.extend(
            (self.run_dir / CHECKPOINT_ARCHIVE_SEGMENT).rglob("metadata.json")
        )
        protected: set[str] = set()
        for path in metadata_paths:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            protected.update(metadata[key] for key in keys if metadata.get(key))

        jobs: list[tuple[Path, list[tuple[str, str, str]]]] = []
        for path in pending:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if (metadata.get("memory_condition") != self.memory_condition
                    or metadata.get("mem0_collection_name") != collection):
                raise IncompleteCheckpointError("Refusing to prune a foreign snapshot reference")
            refs = [(collection, self._required_ref(metadata, keys[0]))]
            entity = self._mem0_entity_snapshot(metadata, collection)
            if entity is not None and entity[1] is not None:
                refs.append(entity)
            validated: list[tuple[str, str, str]] = []
            for source, reference in refs:
                parts = reference.split("/") if isinstance(reference, str) else []
                if (len(parts) != 2 or parts[0] != source
                        or parts[1] in {"", ".", ".."} or "://" in reference):
                    raise IncompleteCheckpointError(
                        f"Refusing to prune an invalid snapshot reference: {reference!r}"
                    )
                validated.append((source, parts[1], reference))
            jobs.append((path, validated))

        admin = _qdrant_admin(self._require_backend_client())
        for path, refs in jobs:
            for source, name, reference in refs:
                if reference in protected:
                    continue
                held = admin.list_snapshots(collection_name=source)
                names = {
                    item.get("name") if isinstance(item, dict) else item.name
                    for item in held
                }
                if name in names:
                    admin.delete_snapshot(collection_name=source, snapshot_name=name, wait=True)
            shutil.rmtree(path)

    # ── restore ───────────────────────────────────────────────────

    # ── recovery marker ───────────────────────────────────────────

    def pending_recovery(self) -> dict[str, Any] | None:
        """The rollback this run began and has not finished, or None."""
        return read_pending_recovery(self.run_dir)

    def _require_no_pending_recovery(self, operation: str) -> None:
        """Refuse to move the run on while it is half rolled back.

        `save`, `prune` and `branch_from` all read or rewrite `checkpoints/`,
        which is precisely what an open recovery is in the middle of
        rearranging. A new snapshot published now would record a live state
        that is part target and part crash, and a prune could drop the target
        itself.
        """
        pending = self.pending_recovery()
        if pending is None:
            return
        raise RecoveryInProgressError(
            f"Refusing to {operation} while a rollback recovery is unfinished, "
            "because this run's state is part snapshot and part crash: "
            f"target_checkpoint_id={str(pending['target_checkpoint_id'])!r} "
            f"recovery_id={str(pending.get('recovery_id'))!r}"
        )

    def _publish_pending_recovery(self, record: dict[str, Any]) -> None:
        _atomic_write_json(pending_recovery_path(self.run_dir), record)
        logger.warning(
            "Rollback recovery {} opened on target {}; {} snapshot(s) to "
            "archive",
            record["recovery_id"],
            record["target_checkpoint_id"],
            len(record.get("revoked") or []),
        )

    def _clear_pending_recovery(self, run_dir: Path) -> None:
        """Close the recovery. Last, so a crash before it only repeats work."""
        path = pending_recovery_path(run_dir)
        if path.is_file():
            path.unlink()
            _sync_directory(path.parent)

    def _archive_revoked_checkpoints(self, pending: dict[str, Any]) -> list[str]:
        """Move the revoked snapshots out of `checkpoints/`, repeatably.

        Each move is its own resumable step, so a recovery interrupted part way
        through archiving picks up where it stopped: a source that is still
        there is moved, one already at its destination is passed over, and a
        snapshot present in *both* places stops the recovery rather than having
        one of the two copies overwritten. Nothing is deleted and no remote
        snapshot is dropped — the archive keeps each directory's metadata,
        ledger and backend references exactly as they were, so the evidence for
        why a rollback revoked them stays readable.
        """
        archive_root = (
            self.run_dir
            / CHECKPOINT_ARCHIVE_SEGMENT
            / str(pending["recovery_id"])
        )
        archived: list[str] = []
        for entry in pending.get("revoked") or []:
            checkpoint_id = str(entry["checkpoint_id"])
            source = self._checkpoint_path(checkpoint_id)
            target = archive_root / checkpoint_id
            if source.exists() and target.exists():
                raise IncompleteCheckpointError(
                    "A revoked snapshot is present both in `checkpoints/` and "
                    "in this recovery's archive; refusing to overwrite either "
                    "copy, because one of them is evidence: "
                    f"checkpoint_id={checkpoint_id!r} source={source} "
                    f"archived={target}"
                )
            if target.exists():
                continue
            if not source.exists():
                raise IncompleteCheckpointError(
                    "A revoked snapshot has vanished from `checkpoints/` "
                    "without reaching this recovery's archive, so its evidence "
                    "is gone and the rollback cannot be completed honestly: "
                    f"checkpoint_id={checkpoint_id!r} expected={target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            archived.append(checkpoint_id)
        if archived:
            logger.warning(
                "Rollback recovery {} archived the revoked snapshot(s) {} "
                "under {}",
                pending["recovery_id"],
                ", ".join(archived),
                archive_root,
            )
        return archived

    def _open_attempt_dir(self, pending: dict[str, Any]) -> Path:
        """This attempt's own evidence directory under the fixed crash scene.

        The original crash scene is attempt 1 and every retry gets its own
        directory beside it, so a second attempt cannot overwrite the state the
        run actually died in — which is the only copy of it.
        """
        evidence = self.run_dir / str(pending["evidence_dir"])
        evidence.mkdir(parents=True, exist_ok=True)
        existing = sum(
            1 for child in evidence.iterdir() if child.name.startswith("attempt_")
        )
        attempt = evidence / f"attempt_{existing + 1:02d}"
        attempt.mkdir(parents=True)
        return attempt

    # ── restore ───────────────────────────────────────────────────

    def restore(self, checkpoint_id: str, *, ledger: RunLedger) -> Path:
        """Roll the whole run directory back to one cut point, resumably.

        Two properties hold whatever happens in the middle. Any interruption is
        continued on the *same* target, because the marker published before the
        first live change names it and `resume_or_restore` reads that marker
        before it reads the ledger — a ledger already replaced from the
        snapshot reads as perfectly resumable, so it cannot be what decides.
        And once this returns, no later automatic recovery can re-select a
        snapshot this rollback revoked, because those are no longer in
        `checkpoints/` at all.

        Every retry restores every layer from the fixed target again. What was
        reached on the previous attempt is not trusted and not resumed from:
        the cost is one repeated copy, and what it buys is that no combination
        of half-restored layers has to be reasoned about.
        """
        cp_dir = self._checkpoint_path(checkpoint_id)
        pending = self.pending_recovery()
        if pending is not None and (
            str(pending["target_checkpoint_id"]) != checkpoint_id
        ):
            raise RecoveryInProgressError(
                "Refusing to restore a second target while a rollback recovery "
                "is unfinished: the run is already part way back to the first, "
                "and starting another would overwrite that work with no record "
                "of either. Finish it, then roll back again if that is still "
                f"wanted: unfinished_target="
                f"{str(pending['target_checkpoint_id'])!r} "
                f"requested_target={checkpoint_id!r}"
            )

        if not (cp_dir / "metadata.json").is_file():
            raise FileNotFoundError(f"Checkpoint not found: {cp_dir}")
        # Same place, same reason: `save`
        # refuses to publish a checkpoint that is missing a layer, so restoring
        # one is a contradiction — and by the time the missing layer is
        # noticed, the live layer it was meant to replace is already in the
        # crash scene with nothing put back in its place.
        restorable, reasons = checkpoint_restorable(
            cp_dir, memory_condition=self.memory_condition
        )
        if not restorable:
            raise IncompleteCheckpointError(
                "Refusing to restore a checkpoint that is not complete: "
                f"checkpoint_id={checkpoint_id!r} reasons={reasons!r}"
            )

        # Everything above reads, and this asks the backend without changing
        # it: up to here the run can still be left exactly as it was found, so
        # a recovery that cannot succeed is refused having done nothing at all.
        self._require_reachable_backend_snapshot(self.metadata(checkpoint_id))

        if pending is None:
            # The last such moment. Where this run is going is recorded now,
            # before the first live change, rather than inferred afterwards
            # from a state that is half one thing and half another.
            history, revoked = partition_checkpoints_by_target(
                self.run_dir, checkpoint_id
            )
            recovery_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            pending = {
                "recovery_id": recovery_id,
                "target_checkpoint_id": checkpoint_id,
                "history": history,
                "revoked": revoked,
                "evidence_dir": f"{CRASH_SCENE_SEGMENT}/{recovery_id}",
                "memory_condition": self.memory_condition,
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }
            self._publish_pending_recovery(pending)
        else:
            logger.warning(
                "Continuing the unfinished rollback recovery {} on target {}",
                pending.get("recovery_id"),
                checkpoint_id,
            )

        # Archived first: while a revoked snapshot is still in `checkpoints/`
        # it is a candidate, and a crash between here and the end of the
        # restore is exactly when something would come looking for one.
        self._archive_revoked_checkpoints(pending)

        attempt = self._open_attempt_dir(pending)
        for layer in (
            PA_WORKSPACE_LAYER,
            CANONICAL_STAGE_LAYER,
            SIMULATOR_WORKSPACE_LAYER,
        ):
            live = self.run_dir / layer
            # On a retry this may be a half-finished copy from the last
            # attempt. It is moved aside whole rather than copied over, so the
            # layer that gets rebuilt is only ever the target's.
            if live.exists():
                shutil.move(str(live), str(attempt / layer))
            snapshot = cp_dir / layer
            if snapshot.is_dir():
                shutil.copytree(snapshot, live)

        live_audit = self.run_dir / MEMORY_IO_LAYER
        if live_audit.is_file():
            shutil.copy2(live_audit, attempt / MEMORY_IO_LAYER)
        snapshot_audit = cp_dir / MEMORY_IO_LAYER
        if snapshot_audit.is_file():
            tmp = live_audit.with_name(live_audit.name + ".tmp")
            shutil.copy2(snapshot_audit, tmp)
            os.replace(tmp, live_audit)

        self._restore_backend_state(cp_dir, self.metadata(checkpoint_id))
        self._move_post_checkpoint_steps(cp_dir, attempt)
        # A step that died inside the wrapper leaves its working stage behind,
        # and `save()` refuses to publish while one is there. Clearing it here
        # is what makes the next checkpoint after a resume reachable.
        cleared = clear_stage_work(self.run_dir, attempt)
        if cleared:
            logger.info(
                "Leftover step working stages moved into the crash scene: {}",
                ", ".join(cleared),
            )

        ledger.replace_from(cp_dir / LEDGER_LAYER)
        # Only now: the marker is what makes every step above repeatable, so it
        # is dropped after the last of them and not before. A crash in the line
        # above costs one more identical recovery; a marker cleared early costs
        # the guarantee.
        self._clear_pending_recovery(self.run_dir)
        logger.info(
            "Checkpoint {} restored; crash scene at {}", checkpoint_id, attempt
        )
        return attempt

    def _require_reachable_backend_snapshot(
        self, metadata: dict[str, Any]
    ) -> None:
        """Confirm the remote snapshot is readable before anything is dropped.

        `checkpoint_restorable` deliberately opens no connection
        (`_unreachable_external_refs` says so), but `_restore_backend_state`
        deletes the live Mem0 collection and only then recovers the snapshot
        into its name. On that path the single honest moment to discover that
        the snapshot is gone from the server is *before* the delete, so that is
        where it is discovered: an unreachable server or a missing snapshot is
        refused here, with the live collection still standing.

        This is a check and not a guarantee. The network can still fail on the
        call after it; what covers that is the kept marker and the snapshot
        still being on the server, so the recovery is retried — not any claim
        that a remote replacement is atomic.

        MemOS restores from a file inside the checkpoint directory, which
        `checkpoint_restorable` has already confirmed is present, and validates
        the whole payload server-side before it touches the cube. It does not
        drop live state on the strength of a reference alone, so it needs no
        second look here.
        """
        if self.memory_condition not in MEM0_CONDITIONS:
            return
        from memory.backends import mem0_client

        client = self._require_backend_client()
        admin = mem0_client._qdrant_admin(client)
        collection = self._required_ref(metadata, "mem0_collection_name")
        references = [
            ("mem0_snapshot_ref", self._required_ref(metadata, "mem0_snapshot_ref"))
        ]
        entity = self._mem0_entity_snapshot(metadata, collection)
        if entity is not None and entity[1] is not None:
            references.append(("mem0_entity_snapshot_ref", entity[1]))

        for key, reference in references:
            source, _, name = str(reference).rpartition("/")
            if not source or not name or "://" in str(reference):
                raise IncompleteCheckpointError(
                    "This checkpoint's snapshot reference is not the "
                    "'<collection>/<snapshot>' pair `snapshot_collection` "
                    "writes, so it cannot be confirmed readable before the "
                    f"live collection is dropped: {key}={reference!r}"
                )
            try:
                held = admin.list_snapshots(collection_name=source)
            except Exception as exc:
                raise IncompleteCheckpointError(
                    "Refusing to drop a live collection while the snapshot "
                    "server cannot be reached, because the snapshot is the "
                    "only copy of what would replace it: "
                    f"{key}={reference!r} error={type(exc).__name__}: {exc}"
                ) from exc
            available = {
                (
                    description.get("name")
                    if isinstance(description, dict)
                    else getattr(description, "name", None)
                )
                for description in (held or [])
            }
            if name not in available:
                raise IncompleteCheckpointError(
                    "Refusing to drop a live collection for a snapshot the "
                    f"server no longer holds: {key}={reference!r} "
                    f"available={sorted(n for n in available if n)!r}"
                )

    def _move_post_checkpoint_steps(self, cp_dir: Path, crash_scene: Path) -> None:
        """Move step artifacts produced after the cut point out of the way.

        Transcripts are append-only, so a rerun must face an empty directory
        rather than a partially written one. Nothing is deleted.
        """
        steps_dir = self.run_dir / "steps"
        if not steps_dir.is_dir():
            return
        done = _snapshot_done_step_ids(cp_dir / LEDGER_LAYER)
        for child in sorted(steps_dir.iterdir()):
            if not child.is_dir() or child.name in done:
                continue
            target = crash_scene / "steps"
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(child), str(target / child.name))

    # ── branch ────────────────────────────────────────────────────

    def prepare_shared_memory(self, checkpoint_id: str, target: Path) -> dict[str, Any]:
        """Restore only memory once; private scenes/sessions are still per Probe."""
        self._require_no_pending_recovery(f"prepare shared checkpoint {checkpoint_id!r}")
        cp_dir = self._checkpoint_path(checkpoint_id)
        metadata = self.metadata(checkpoint_id)
        if not metadata.get("pinned"):
            raise ValueError("Only a pinned checkpoint can supply shared Probe memory")
        target = Path(target)
        target.mkdir(parents=True, exist_ok=False)
        (target / "checkpoint_metadata.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")
        if self.memory_condition == "nanobot_file_memory":
            for source in (cp_dir / PA_WORKSPACE_LAYER).glob("*/memory"):
                destination = target / PA_WORKSPACE_LAYER / source.parent.name / "memory"
                shutil.copytree(source, destination)
                (destination / ".mempa-read-only").write_text("shared-probe-memory\n")
            # Protect directory entries as well as file contents. A writable
            # parent would still permit unlink/replace of a 0444 memory file,
            # or replacement of the entire memory directory behind a symlink.
            for entry in target.rglob("*"):
                entry.chmod(0o555 if entry.is_dir() else 0o444)
            target.chmod(0o555)
        self._restore_backend_state_into_branch(cp_dir, target, metadata)
        return metadata

    def branch_from(
        self, checkpoint_id: str, branch_dir: Path, *, shared_memory_dir: Path | None = None
    ) -> Path:
        """Copy private execution state and obtain the pinned memory state.

        A probe's start point is this copy itself: the cut point's memory, the
        cut point's world, and the mainline PA session continued from where it
        stood. The only thing that moves is `prompt_history_start_index`, so
        the probe's prompt starts empty while memory and the other cursors stay
        exactly as the snapshot left them.
        With `shared_memory_dir`, the probe-variance owner has restored the
        memory once; this branch links to it instead of restoring its own.
        """
        # A probe branches a cut point and is scored as the state of the run at
        # that point, so branching one while the mainline is half rolled back
        # would measure neither state. A pin the rollback revoked is not a cut
        # point of this run at all any more, and by here it is not in
        # `checkpoints/` either — this refuses the whole operation one step
        # earlier, before the target is even looked at.
        self._require_no_pending_recovery(f"branch checkpoint {checkpoint_id!r}")
        cp_dir = self._checkpoint_path(checkpoint_id)
        if not (cp_dir / "metadata.json").is_file():
            raise FileNotFoundError(f"Checkpoint not found: {cp_dir}")
        metadata = self.metadata(checkpoint_id)
        if not metadata.get("pinned"):
            raise ValueError(
                "Only a pinned checkpoint may be branched from, because only "
                "pinned snapshots were flushed to a complete memory state: "
                f"checkpoint_id={checkpoint_id!r}"
            )

        branch_dir = Path(branch_dir)
        if branch_dir.exists():
            shutil.rmtree(branch_dir)
        branch_dir.mkdir(parents=True)
        for layer in (
            PA_WORKSPACE_LAYER,
            CANONICAL_STAGE_LAYER,
            SIMULATOR_WORKSPACE_LAYER,
        ):
            source = cp_dir / layer
            if source.is_dir():
                def ignore_shared_memory(directory: str, names: list[str]) -> list[str]:
                    relative = Path(directory).relative_to(source)
                    return ["memory"] if layer == PA_WORKSPACE_LAYER and len(relative.parts) == 1 and "memory" in names else []
                shutil.copytree(
                    source, branch_dir / layer,
                    ignore=ignore_shared_memory if shared_memory_dir is not None else None,
                )
        if shared_memory_dir is not None:
            for memory in (Path(shared_memory_dir) / PA_WORKSPACE_LAYER).glob("*/memory"):
                destination = branch_dir / PA_WORKSPACE_LAYER / memory.parent.name / "memory"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(memory.resolve(), target_is_directory=True)
        for layer in (LEDGER_LAYER, RUN_PLAN_LAYER):
            source = cp_dir / layer
            if source.is_file():
                shutil.copy2(source, branch_dir / layer)

        if shared_memory_dir is None:
            self._restore_backend_state_into_branch(cp_dir, branch_dir, metadata)
        else:
            record = read_branch_namespace(Path(shared_memory_dir)) or {
                "memory_condition": self.memory_condition,
                "namespace": None,
            }
            record = {**record, "shared_owner": str(Path(shared_memory_dir).resolve())}
            (branch_dir / BACKEND_STATE_LAYER).mkdir(parents=True, exist_ok=True)
            (branch_dir / BACKEND_STATE_LAYER / BRANCH_NAMESPACE_FILENAME).write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
            )
        self._move_branch_prompt_start_to_end(branch_dir)
        logger.info(
            "Branched checkpoint {} into {}", checkpoint_id, branch_dir
        )
        return branch_dir

    def _restore_backend_state_into_branch(
        self,
        cp_dir: Path,
        branch_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        if self.memory_condition in MEM0_CONDITIONS and os.environ.get(
            "MEMPA_QDRANT_RESTORE_LOCK_PATH"
        ):
            from memory.backends import qdrant_runtime

            # The whole branch restore holds the shared Qdrant lock. The inner
            # per-restore lease (user_d line) is that same lock, and a second
            # flock on it from this process would wait on itself, so it stands
            # down while held.
            with qdrant_runtime.qdrant_restore_lease():
                _BRANCH_QDRANT_LEASE.held = True
                try:
                    return self._restore_backend_state_into_branch_unlocked(
                        cp_dir, branch_dir, metadata
                    )
                finally:
                    _BRANCH_QDRANT_LEASE.held = False
        return self._restore_backend_state_into_branch_unlocked(cp_dir, branch_dir, metadata)

    def _restore_backend_state_into_branch_unlocked(
        self, cp_dir: Path, branch_dir: Path, metadata: dict[str, Any]
    ) -> None:
        """Extension point for backends that live outside the run directory.

        File-backed conditions (`nanobot_file_memory`, `no_memory`) keep every
        byte inside `pa_workspace/`, so the directory copy above is already the
        whole clone. The external backends restore a namespace of their own
        from the snapshot references in `metadata.json`, named after this
        branch directory, so the mainline database is shared with nobody and a
        rerun reads the cut point's memory rather than what accumulated after
        it. The snapshot itself is never consumed: it is read as many times as
        there are branches.
        """
        if self.memory_condition in LOCAL_MEMORY_CONDITIONS:
            return
        client = self._require_backend_client()
        backend_state = branch_dir / BACKEND_STATE_LAYER
        backend_state.mkdir(parents=True, exist_ok=True)

        if self.memory_condition in MEM0_CONDITIONS:
            from memory.backends.mem0_client import restore_collection_as
            from memory.backends.qdrant_runtime import qdrant_restore_lease

            snapshot_ref = self._required_ref(metadata, "mem0_snapshot_ref")
            source = self._required_ref(metadata, "mem0_collection_name")
            namespace = branch_namespace_for(source, branch_dir)
            entity = self._mem0_entity_snapshot(metadata, source)
            with (
                nullcontext()
                if getattr(_BRANCH_QDRANT_LEASE, "held", False)
                else qdrant_restore_lease()
            ):
                restore_collection_as(client, snapshot_ref, namespace)
                if entity is not None and entity[1] is not None:
                    restore_collection_as(
                        client,
                        entity[1],
                        f"{namespace}_entities",
                    )
            shutil.copy2(
                cp_dir / BACKEND_STATE_LAYER / MEM0_HISTORY_FILENAME,
                backend_state / MEM0_HISTORY_FILENAME,
            )
        elif self.memory_condition == MEMOS_CONDITION:
            source = self._required_ref(metadata, "memos_cube_id")
            export_path = cp_dir / self._required_ref(
                metadata, "memos_cube_export_path"
            )
            namespace = branch_namespace_for(source, branch_dir)
            payload = json.loads(export_path.read_text(encoding="utf-8"))
            client.import_cube_as(
                rebind_memos_cube_to_branch(payload, namespace, source),
                namespace,
            )
        else:
            raise NotImplementedError(
                "Branching an external memory backend needs its namespace "
                "restored from the checkpoint's snapshot reference, which is "
                "not implemented yet: "
                f"memory_condition={self.memory_condition!r} "
                f"checkpoint_id={metadata.get('checkpoint_id')!r}"
            )

        (backend_state / BRANCH_NAMESPACE_FILENAME).write_text(
            json.dumps(
                {
                    "memory_condition": self.memory_condition,
                    "namespace": namespace,
                    "source_namespace": source,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _move_branch_prompt_start_to_end(self, branch_dir: Path) -> None:
        """Start this probe's prompt empty without touching anything else.

        A pinned checkpoint was flushed before it was saved, so this cursor is
        usually already at the end and nothing is written.
        """
        for session_path in sorted(
            (branch_dir / PA_WORKSPACE_LAYER).glob("*/sessions/*.jsonl")
        ):
            _set_prompt_history_start_to_end(session_path)

    # ── inspection ────────────────────────────────────────────────

    def metadata(self, checkpoint_id: str) -> dict[str, Any]:
        path = self._checkpoint_path(checkpoint_id) / "metadata.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_checkpoints(self) -> list[str]:
        """The cut points this run may go back to, newest-agnostic and sorted.

        The same enumeration `run_restorable` asks, so the resume path and
        Experiment Admin cannot disagree about what is available — and so a
        snapshot an unfinished rollback revoked is offered by neither.
        """
        return sorted(
            cp_dir.name for cp_dir in _published_checkpoints(self.run_dir)
        )

    def exists(self, checkpoint_id: str) -> bool:
        return (self._checkpoint_path(checkpoint_id) / "metadata.json").is_file()
