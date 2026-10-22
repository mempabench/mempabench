"""State fixture initialization and stage policy helpers for benchmark runs."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from harness.run_layout import STAGE_WORK_SEGMENT
from harness.run_layout import runs_root as configured_runs_root
from pa.config.schema import MCPServerConfig
from state_server.store import FIXTURE_RECORDS_NAME, apply_fixture_records


@dataclass(frozen=True)
class StateRuntime:
    """Run-local state namespace plus MCP config for the PA."""

    run_id: str
    user_id: str
    session_id: str
    state_root: Path
    project_root: Path
    enabled_tools: list[str]
    session_date: str | None = None
    # The session's full start time (`RunStep.benchmark_session_timestamp`),
    # for hour-granular `visible_from` stamps. `session_date` stays the store's
    # own clock; this one is visibility-only (`state_server.store
    # .stamp_visible`).
    session_timestamp: str | None = None

    def mcp_server_config(self) -> dict[str, MCPServerConfig]:
        """Return the MCP server config that exposes this state namespace."""
        env: dict[str, str] = {
            "MEMPA_STATE_ROOT": str(self.state_root),
            "MEMPA_RUN_ID": self.run_id,
            "MEMPA_USER_ID": self.user_id,
            "MEMPA_SESSION_ID": self.session_id,
            "PYTHONPATH": str(self.project_root),
        }
        if self.session_date:
            env["MEMPA_SESSION_DATE"] = self.session_date
        if self.session_timestamp:
            env["MEMPA_SESSION_TIMESTAMP"] = self.session_timestamp
        return {
            "state": MCPServerConfig(
                type="stdio",
                command=sys.executable,
                args=["-m", "state_server.server"],
                env=env,
                enabled_tools=self.enabled_tools,
            )
        }


def initialize_state_runtime(
    *,
    user_id: str,
    session_id: str,
    run_id: str,
    fixture_root: Path | str = Path("data/state_fixtures"),
    runs_root: Path | str | None = None,
    project_root: Path | str = Path("."),
    enabled_tools: list[str] | None = None,
    overwrite: bool = False,
) -> StateRuntime:
    """Copy the complete persona world into a run-local mutable directory.

    `runs_root` is *this run's parent directory*, not the global runs root:
    only `/<run_id>` is joined to it, and a formal launch passes
    `run_dir.parent` (`scripts/run_orchestrator.py`) because a run lives four
    levels below the root. An explicit value therefore wins outright.

    It defaults to the configured runs root
    (`harness/run_layout.py::runs_root`), read at call time rather than frozen
    into the signature. The old default was the *relative* `workspace/runs`,
    which put a run's state wherever the process happened to be standing.
    """
    parent = Path(runs_root) if runs_root is not None else configured_runs_root()
    run_dir = parent / run_id
    state_root = run_dir / "state"
    if state_root.exists():
        if not overwrite:
            raise FileExistsError(f"state already exists: {state_root}")
        shutil.rmtree(state_root)
    state_root.parent.mkdir(parents=True, exist_ok=True)

    fixture = Path(fixture_root) / user_id
    if not fixture.exists():
        raise FileNotFoundError(f"state fixture not found: {fixture}")
    shutil.copytree(fixture, state_root)

    return StateRuntime(
        run_id=run_id,
        user_id=user_id,
        session_id=session_id,
        state_root=state_root.resolve(),
        project_root=Path(project_root).resolve(),
        enabled_tools=enabled_tools or _default_enabled_tools(),
    )


_DEFAULT_TOOLS = ["document_read", "email_draft_create"]


def _default_enabled_tools() -> list[str]:
    return list(_DEFAULT_TOOLS)


def world_toolbox_tools(project_root: Path) -> list[str]:
    """Every implemented catalog tool — the PA's full standing toolbox."""
    import yaml

    catalog_path = Path(project_root) / "data/state_tools/catalog.yaml"
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    return sorted(
        name
        for name, spec in (raw.get("tools") or {}).items()
        if isinstance(spec, dict) and spec.get("status") == "implemented"
    )


def _artifact_uses_world_toolbox(artifact_path: Path) -> bool:
    import yaml

    try:
        raw = yaml.safe_load(artifact_path.read_text(encoding="utf-8"))
    except OSError:
        return False
    if not isinstance(raw, dict):
        return False
    meta = raw.get("meta")
    if isinstance(meta, dict) and meta.get("tool_status") == "world_toolbox":
        return True
    event = raw.get("event")
    return isinstance(event, dict) and event.get("tool_status") == "world_toolbox"


FIXTURE_OVERLAY_DIRNAME = "fixture"


def _resolve_fixture_overlay(
    explicit: Path | str | None,
    artifact_path: Path | None,
) -> Path | None:
    """Where this step's small scene fixture lives, if it has one.

    A paired final probe is handed its partner's overlay directory explicitly,
    so both halves of an evolving pair face the same task conditions; every
    other step derives it from its own artifact and simply has no such
    directory.
    """
    if explicit is not None:
        return Path(explicit)
    if artifact_path is None:
        return None
    return artifact_path.parent / FIXTURE_OVERLAY_DIRNAME


def _apply_fixture_overlay(work_dir: Path, overlay: Path | None) -> list[str]:
    """Overwrite same-named files only; the rest of the world is inherited."""
    if overlay is None or not overlay.is_dir():
        return []
    applied: list[str] = []
    for source in sorted(overlay.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(overlay)
        if relative == Path(FIXTURE_RECORDS_NAME):
            # Rows to append into the stores below, not a file of the world.
            continue
        target = work_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        applied.append(str(relative))
    return applied


def _apply_fixture_records(
    work_dir: Path, overlay: Path | None, *, merge_records: bool
) -> list[str]:
    """Append the overlay's `records.jsonl` rows into this step's world.

    Probe steps only. A probe branches a pinned cut point and throws the world
    away afterwards, so a record appended for its scene reaches nothing else;
    an accumulation step's world is committed back onto the mainline, where the
    same append would silently become part of every later session.
    """
    if overlay is None or not (overlay / FIXTURE_RECORDS_NAME).is_file():
        return []
    if not merge_records:
        raise ValueError(
            f"{FIXTURE_RECORDS_NAME} is a probe-only overlay and this step is "
            f"not a probe: overlay={str(overlay)!r} step_world={str(work_dir)!r}"
        )
    return apply_fixture_records(work_dir, overlay)


StagePolicy = Literal["commit", "discard"]
_STEP_AUDIT_FILES = ("tool_log.jsonl", "state_diff.jsonl")
#: Cleared from a step's working stage on the way in and on the way out. The
#: audit files are the step's own record; `gate_open` is the marker
#: `harness.gate.gate_marker_hook` touches when a gated session reaches its
#: reveal beat and `gate_satisfied` the one the store writes when a gated write
#: first carries the user's information, and a committed stage carrying either
#: would leave every later session's gate open, or already met, before its
#: first turn. `session_closing` is the marker
#: `harness.gate.session_closing_hook` touches on a compressed session's last
#: user turn; a committed stage carrying it would refuse every write of every
#: later session from its first turn on.
_STEP_RESET_FILES = (
    *_STEP_AUDIT_FILES,
    "gate_open",
    "gate_satisfied",
    "session_closing",
)


@dataclass(frozen=True)
class StageStepRuntime:
    """Per-step mutable state namespace plus MCP config for the PA."""

    run_id: str
    user_id: str
    session_id: str
    step_id: str
    state_root: Path
    project_root: Path
    enabled_tools: list[str]
    session_date: str | None = None
    # The session's full start time (`RunStep.benchmark_session_timestamp`),
    # for hour-granular `visible_from` stamps. `session_date` stays the store's
    # own clock; this one is visibility-only (`state_server.store
    # .stamp_visible`).
    session_timestamp: str | None = None
    # This session's dictated-text gate (`harness.gate.resolve_session_gate`),
    # handed to the state server so it can refuse a gated write the PA composed
    # itself. None for every step without a gate.
    gate: dict[str, Any] | None = None
    # Which entered beat is this compressed session's reaction beat
    # (`harness.gate.resolve_compressed_reaction_index`). It never reaches the
    # state server: `session_closing_hook` reads it to know which
    # `simulator_turn` closes the session to writes, and the store learns that
    # from the marker file alone. None for every step that is not compressed.
    compressed_reaction_index: int | None = None

    def mcp_server_config(self) -> dict[str, MCPServerConfig]:
        env = {
            "MEMPA_STATE_ROOT": str(self.state_root),
            "MEMPA_RUN_ID": self.run_id,
            "MEMPA_USER_ID": self.user_id,
            "MEMPA_SESSION_ID": self.session_id,
            "PYTHONPATH": str(self.project_root),
        }
        if self.session_date:
            env["MEMPA_SESSION_DATE"] = self.session_date
        if self.session_timestamp:
            env["MEMPA_SESSION_TIMESTAMP"] = self.session_timestamp
        if self.gate:
            env["MEMPA_GATE"] = json.dumps(self.gate, ensure_ascii=False)
        return {
            "state": MCPServerConfig(
                type="stdio",
                command=sys.executable,
                args=["-m", "state_server.server"],
                env=env,
                enabled_tools=self.enabled_tools,
            )
        }


class StageRuntime:
    """Canonical stage plus per-step working copies."""

    def __init__(
        self,
        *,
        run_id: str,
        user_id: str,
        fixture_root: Path | str = Path("data/state_fixtures"),
        runs_root: Path | str | None = None,
        project_root: Path | str = Path("."),
        overwrite: bool = False,
        run_dir: Path | str | None = None,
    ) -> None:
        """`runs_root` is this run's parent directory, as in
        `initialize_state_runtime`: explicit wins, `run_dir` overrides it
        outright, and `None` falls back to the configured runs root read at
        call time rather than to a relative path frozen in the signature."""
        self.run_id = run_id
        self.user_id = user_id
        self.fixture_root = Path(fixture_root)
        # A probe branch is a whole run directory of its own, so it says where
        # it lives instead of being derived from runs_root/run_id.
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        elif runs_root is not None:
            self.run_dir = Path(runs_root) / run_id
        else:
            self.run_dir = configured_runs_root() / run_id
        self.project_root = Path(project_root).resolve()
        self.canonical_stage = self.run_dir / "canonical_stage"
        self.working_root = self.run_dir / STAGE_WORK_SEGMENT
        fixture = self.fixture_root / user_id
        if not fixture.exists():
            raise FileNotFoundError(f"state fixture not found: {fixture}")
        if self.canonical_stage.exists():
            if overwrite:
                shutil.rmtree(self.canonical_stage)
            else:
                return
        self.canonical_stage.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fixture, self.canonical_stage)

    def begin_step(
        self,
        *,
        step_id: str,
        session_id: str,
        enabled_tools: list[str] | None = None,
        runtime_artifact_path: Path | str | None = None,
        session_date: str | None = None,
        session_timestamp: str | None = None,
        fixture_overlay: Path | str | None = None,
        merge_records: bool = False,
        gate: dict[str, Any] | None = None,
        compressed_reaction_index: int | None = None,
    ) -> StageStepRuntime:
        artifact_path = (
            Path(runtime_artifact_path)
            if runtime_artifact_path is not None
            else None
        )
        if artifact_path is not None and not artifact_path.is_file():
            # Four executor tests spent months reading "must declare
            # tool_status=world_toolbox" about a transcript that had never been
            # written. A file that is not there declares nothing.
            raise FileNotFoundError(
                f"runtime artifact does not exist: {artifact_path}"
            )
        if artifact_path is not None and not _artifact_uses_world_toolbox(
            artifact_path
        ):
            raise ValueError(
                "runtime artifact must declare tool_status=world_toolbox: "
                f"{artifact_path}"
            )
        work_dir = self.working_root / step_id
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.canonical_stage, work_dir)
        overlay = _resolve_fixture_overlay(fixture_overlay, artifact_path)
        _apply_fixture_overlay(work_dir, overlay)
        _apply_fixture_records(work_dir, overlay, merge_records=merge_records)
        for filename in _STEP_RESET_FILES:
            (work_dir / filename).unlink(missing_ok=True)
        return StageStepRuntime(
            run_id=self.run_id,
            user_id=self.user_id,
            session_id=session_id,
            step_id=step_id,
            state_root=work_dir.resolve(),
            project_root=self.project_root,
            enabled_tools=enabled_tools
            or (
                world_toolbox_tools(self.project_root)
                if artifact_path is not None
                else _default_enabled_tools()
            ),
            session_date=session_date,
            session_timestamp=session_timestamp,
            gate=gate,
            compressed_reaction_index=compressed_reaction_index,
        )

    def finish_step(self, step_runtime: StageStepRuntime, policy: StagePolicy) -> None:
        work_dir = step_runtime.state_root
        for filename in _STEP_RESET_FILES:
            (work_dir / filename).unlink(missing_ok=True)
        if policy == "commit":
            if self.canonical_stage.exists():
                shutil.rmtree(self.canonical_stage)
            shutil.copytree(work_dir, self.canonical_stage)
        if work_dir.exists():
            shutil.rmtree(work_dir)


def preserve_step_state_artifacts(
    step_runtime: StageStepRuntime,
    step_dir: Path | str,
) -> dict[str, str]:
    """Copy one step's state audit before its working stage is removed."""
    state_dir = Path(step_dir) / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for filename in _STEP_AUDIT_FILES:
        source = step_runtime.state_root / filename
        target = state_dir / filename
        if source.is_file():
            shutil.copy2(source, target)
        else:
            target.write_text("", encoding="utf-8")
        artifacts[filename.removesuffix(".jsonl")] = str(target)
    return artifacts
