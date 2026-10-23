"""Whose ``fixture/`` overlay a probe reads.

``overlay_owner`` is the one ownership rule; the plan freeze
(``harness/run_plan_builder.py``) and the orchestrator read it through these
helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.resolved_gt import PreEvent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_DRAFTS_ROOT = PROJECT_ROOT / "Authoring/probe/drafts"


@dataclass(frozen=True)
class EffectiveOverlay:
    """Which `fixture/` a probe actually reads, and whose it is."""

    persona: str
    probe_id: str
    #: The probe whose directory holds the overlay: itself, or the pre it pairs with.
    owner_probe_id: str
    root: Path


def _declared_pairing(draft_path: Path) -> tuple[str | None, str | None]:
    """`(probe_kind, pairs_with)` off a probe draft's `meta`.

    A draft that is not on disk has declared nothing, and that is a real state:
    probe drafts are written after the outline, and generating the probe
    outline resolves overlays over a directory that holds none yet. So
    `(None, None)`, and `resolved_gt.yaml` decides.

    A draft that IS there and cannot be read or parsed raises. Swallowing the
    error would drop the paired fixture overlay silently, so the two halves of
    an evolving pair would face different conditions. Silence there is not "we
    do not know", it is the answer "no partner", under which a paired final
    resolves to its OWN `fixture/` instead of the pre's. Three call sites share this
    function, so one unreadable byte would split the pair's conditions in all
    three at once.
    """
    try:
        text = draft_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    raw = yaml.safe_load(text)
    meta = raw.get("meta") if isinstance(raw, dict) else None
    if not isinstance(meta, dict):
        return None, None
    kind = meta.get("probe_kind")
    partner = meta.get("pairs_with")
    return (
        kind if isinstance(kind, str) and kind.strip() else None,
        partner if isinstance(partner, str) and partner.strip() else None,
    )


def overlay_owner(
    probe_id: str,
    *,
    persona: str,
    declared_kind: str | None,
    declared_pairs_with: str | None,
    gt_pairs: Mapping[str, PreEvent] | None = None,
) -> str:
    """Whose `fixture/` a probe reads: the one ownership rule, shared by the call
    sites that used to each carry their own (this validator and `harness/
    orchestrator.py::Orchestrator._probe_fixture_overlay`). Each site maps the id
    it gets back onto its own layout.

    Both halves of an evolving pair overlay ONE directory, the pre_event probe's: a
    final a pre probe pairs with resolves back to that pre's `fixture/`, whose
    directory is the canonical one even when the final has one of its own. A final no
    pre pairs with keeps its own.

    Two independent records say who the owner is and BOTH are read where both are at
    hand: the persona's frozen `resolved_gt.yaml` (`gt_pairs`, pre -> its final) and
    the pair's own declaration (`meta.probe_kind` / `meta.pairs_with`, which the
    schema writes into both halves; the runtime knows the kind from its frozen plan
    step instead). They disagreeing means the pairing itself is unsettled, so it
    raises rather than picking one side -- picking one is how a stamp taken over one
    directory came to cover a probe that reads another. A record that is simply not
    there yet (no `gt_pairs` handed in, no draft on disk) leaves the other deciding.
    """
    from_gt: str | None = None
    if gt_pairs is not None:
        owners = sorted(
            pre.probe_id for pre in gt_pairs.values() if pre.pairs_with == probe_id
        )
        if len(owners) > 1:
            raise ValueError(
                "resolved_gt.yaml pairs more than one pre_event probe with this "
                f"probe: persona={persona!r} probe_id={probe_id!r} pres={owners}"
            )
        from_gt = owners[0] if owners else probe_id

    # Silence is not a claim: a final whose draft names no partner has said nothing
    # about the pairing, so `resolved_gt.yaml` decides where it has an answer. That
    # a pairing exists which the drafts alone do not show is what the plan freeze
    # refuses (`harness/run_plan_builder.py::_require_probe_validation`), because the
    # run resolves the overlay from the drafts.
    from_declaration: str | None = None
    if declared_kind == "pre_event":
        from_declaration = probe_id
    elif declared_kind == "final" and declared_pairs_with:
        from_declaration = declared_pairs_with

    if (
        from_gt is not None
        and from_declaration is not None
        and from_gt != from_declaration
    ):
        raise ValueError(
            "the probe's overlay owner differs between resolved_gt.yaml and the "
            f"drafts' own pairing: persona={persona!r} probe_id={probe_id!r} "
            f"resolved_gt={from_gt!r} draft={from_declaration!r}"
        )
    return from_gt if from_gt is not None else (from_declaration or probe_id)


def effective_overlay(
    persona: str,
    probe_id: str,
    *,
    drafts_root: Path | None = None,
    gt_pairs: Mapping[str, PreEvent] | None = None,
    probe_kind: str | None = None,
) -> EffectiveOverlay:
    """`overlay_owner` over the authoring layout: `<drafts_root>/<persona>/<owner>/
    fixture`, the probe's second world root, same layout as S0. The kind comes off
    the draft's own `meta.probe_kind` unless the caller knows it from elsewhere.
    """
    root = drafts_root or PROBE_DRAFTS_ROOT
    kind, partner = _declared_pairing(root / persona / probe_id / f"{probe_id}.yaml")
    owner = overlay_owner(
        probe_id,
        persona=persona,
        declared_kind=probe_kind or kind,
        declared_pairs_with=partner,
        gt_pairs=gt_pairs,
    )
    return EffectiveOverlay(
        persona, probe_id, owner, root / persona / owner / "fixture"
    )


def _ids_of(record: dict[str, Any]) -> set[str]:
    return {str(value) for key, value in record.items() if key == "id" or key.endswith("_id")}
