"""The overlay a probe reads is the overlay it is frozen with -- for a final
that a pre_event probe pairs with, the pre's `fixture/` -- resolved one shared
way by the plan freeze and the orchestrator, and
refused outright when the frozen GT and the drafts' own pairing disagree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from harness import run_plan_builder
from harness.orchestrator import Orchestrator
from scripts.probe_overlay import _declared_pairing, effective_overlay
from scripts.resolved_gt import PreEvent

PERSONA = "user_x"
#: `pre_01` pairs with `fin_01`; `fin_02` pairs with nothing.
GT_PAIRS = {"pre_01": PreEvent("pre_01", "work_attr01", 15, "fin_01")}


# ── the probe tree these cases run over ────────────────────────────


def _draft(probe_id: str, *, kind: str, pairs_with: str | None) -> str:
    meta: dict[str, Any] = {
        "artifact_kind": "probe_session",
        "persona_id": PERSONA,
        "probe_id": probe_id,
        "session_id": probe_id,
        "probe_kind": kind,
    }
    if pairs_with is not None:
        meta["pairs_with"] = pairs_with
    return yaml.safe_dump({"meta": meta}, sort_keys=False)


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`pre_01` (paired with `fin_01`), `fin_01` and the unpaired `fin_02`, each with
    an overlay of its own."""
    drafts = tmp_path / "Authoring/probe/drafts" / PERSONA
    for probe_id, kind, partner in (
        ("pre_01", "pre_event", "fin_01"),
        ("fin_01", "final", "pre_01"),
        ("fin_02", "final", None),
    ):
        directory = drafts / probe_id
        (directory / "fixture").mkdir(parents=True)
        (directory / "fixture/records.jsonl").write_text(
            json.dumps({"id": f"rec_{probe_id}"}) + "\n", encoding="utf-8"
        )
        (directory / f"{probe_id}.yaml").write_text(
            _draft(probe_id, kind=kind, pairs_with=partner), encoding="utf-8"
        )
    monkeypatch.setattr(run_plan_builder, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _drafts_root(root: Path) -> Path:
    return root / "Authoring/probe/drafts"


def _draft_path(root: Path, probe_id: str) -> Path:
    return _drafts_root(root) / PERSONA / probe_id / f"{probe_id}.yaml"


def _gate(root: Path, probe_id: str) -> Any:
    return run_plan_builder._validate_probe_draft(
        probe_id, _draft_path(root, probe_id), gt_pairs=GT_PAIRS
    )


def test_the_freeze_records_the_overlay_the_probe_reads(tree: Path) -> None:
    """A final that `pre_01` pairs with reads the PRE's `fixture/` at runtime, so
    that is the overlay the freeze records for it."""
    frozen = _gate(tree, "fin_01")
    assert frozen.owner_probe_id == "pre_01"


# ── the two pairing records disagree ───────────────────────────────


def test_gt_and_draft_pairing_disagreeing_is_an_error(tree: Path) -> None:
    draft = _draft_path(tree, "fin_01")
    draft.write_text(_draft(("fin_01"), kind="final", pairs_with="pre_09"), encoding="utf-8")

    with pytest.raises(ValueError, match="differs between resolved_gt.yaml and the"):
        effective_overlay(
            PERSONA, "fin_01", drafts_root=_drafts_root(tree), gt_pairs=GT_PAIRS
        )
    with pytest.raises(ValueError, match="differs between resolved_gt.yaml and the"):
        _gate(tree, "fin_01")


def test_a_pairing_only_the_gt_knows_is_refused_at_the_freeze(tree: Path) -> None:
    """The run reads the drafts, not `resolved_gt.yaml`: a final whose draft drops
    `meta.pairs_with` would be frozen over the pre's `fixture/` and then read its own.
    The draft says nothing, so the resolution follows the GT and the freeze refuses."""
    draft = _draft_path(tree, "fin_01")
    draft.write_text(_draft("fin_01", kind="final", pairs_with=None), encoding="utf-8")

    assert effective_overlay(
        PERSONA, "fin_01", drafts_root=_drafts_root(tree), gt_pairs=GT_PAIRS
    ).owner_probe_id == "pre_01"
    with pytest.raises(ValueError) as excinfo:
        _gate(tree, "fin_01")
    message = str(excinfo.value)
    assert "would overlay a different directory than the resolved GT" in message
    assert "resolved_gt_owner='pre_01' runtime_owner='fin_01'" in message


def test_two_pres_pairing_with_one_probe_is_an_error(tree: Path) -> None:
    pairs = dict(GT_PAIRS)
    pairs["pre_02"] = PreEvent("pre_02", "work_attr02", 30, "fin_01")
    with pytest.raises(ValueError, match="pairs more than one pre_event probe"):
        effective_overlay(PERSONA, "fin_01", drafts_root=_drafts_root(tree), gt_pairs=pairs)


# ── the call sites resolve one directory ───────────────────────────


@pytest.mark.parametrize("probe_id", ["pre_01", "fin_01", "fin_02"])
def test_the_call_sites_resolve_the_same_directory(
    tree: Path, probe_id: str
) -> None:
    drafts_root = _drafts_root(tree)
    shared = effective_overlay(
        PERSONA, probe_id, drafts_root=drafts_root, gt_pairs=GT_PAIRS
    ).root
    step = SimpleNamespace(
        step_id=probe_id,
        persona_id=PERSONA,
        kind="pre_event_probe" if probe_id.startswith("pre_") else "final_probe",
        script_path=Path(f"Authoring/probe/drafts/{PERSONA}/{probe_id}/{probe_id}.yaml"),
    )
    orchestrator = SimpleNamespace(config=SimpleNamespace(project_root=tree))

    plan = SimpleNamespace(
        steps=[
            SimpleNamespace(
                step_id=other,
                persona_id=PERSONA,
                kind="pre_event_probe" if other.startswith("pre_") else "final_probe",
                script_path=Path(
                    f"Authoring/probe/drafts/{PERSONA}/{other}/{other}.yaml"
                ),
            )
            for other in ("pre_01", "fin_01", "fin_02")
        ]
    )
    orchestrator.plan = plan

    assert Orchestrator._probe_fixture_overlay(orchestrator, step) == shared
    # And it is the pre's directory for both halves of the pair.
    owner = "pre_01" if probe_id in {"pre_01", "fin_01"} else "fin_02"
    assert shared == drafts_root / PERSONA / owner / "fixture"


# ── a draft that cannot be read is an error, not "no pairing" ──────
#
# `_declared_pairing` inherited `except (OSError, yaml.YAMLError): return None`
# from `harness/orchestrator.py::_probe_pairs_with`, which
# an exemption audit judged a **hatch** -- "the
# paired fixture overlay silently dropped, so the two halves of an evolving pair
# face different conditions" -- with `raise` as the fix. Sharing the function put
# that one swallow behind three call sites, so it has to go: an unreadable draft
# reads as "declares no partner", and a paired final then resolves to its OWN
# `fixture/` instead of the pre's, which is the very split the pairing prevents.


def test_a_corrupt_draft_is_an_error_not_an_absent_pairing(tree: Path) -> None:
    draft = _draft_path(tree, "fin_01")
    draft.write_text("meta: {probe_kind: final, pairs_with: [unclosed\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        _declared_pairing(draft)


def test_a_corrupt_draft_stops_the_call_sites_that_share_the_rule(tree: Path) -> None:
    """Not swallowed at any of them: the resolution and the freeze."""
    draft = _draft_path(tree, "fin_01")
    draft.write_text("meta: {probe_kind: final, pairs_with: [unclosed\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        effective_overlay(
            PERSONA, "fin_01", drafts_root=_drafts_root(tree), gt_pairs=GT_PAIRS
        )

    with pytest.raises(yaml.YAMLError):
        _gate(tree, "fin_01")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-0 file anyway")
def test_an_unreadable_draft_is_an_error_too(tree: Path) -> None:
    draft = _draft_path(tree, "fin_01")
    draft.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            _declared_pairing(draft)
    finally:
        draft.chmod(0o644)


def test_a_draft_not_generated_yet_still_declares_nothing(tree: Path) -> None:
    """The boundary the fix must keep: probe drafts are written AFTER the outline,
    and generating the probe outline resolves overlays while the directory
    holds no draft at all. Absent is a state, not a failure being hidden -- so it
    stays `(None, None)` and `resolved_gt.yaml` decides."""
    assert _declared_pairing(_drafts_root(tree) / PERSONA / "fin_09/fin_09.yaml") == (
        None,
        None,
    )
    _draft_path(tree, "fin_01").unlink()
    assert effective_overlay(
        PERSONA, "fin_01", drafts_root=_drafts_root(tree), gt_pairs=GT_PAIRS
    ).owner_probe_id == "pre_01"
