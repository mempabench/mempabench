"""Aggregate scores E1–E5 over one run's judge outputs (design §5; §7 row 23).

Pure data, no LLM. Inputs, all under one run directory:

- `judge_output/<judge_id>/<probe_id>.json` — one record per probe (§4.6). Both faces
  are read from the record: `ix_face.exact_match` / `ix_face.pa_setting` (computed by
  code) and `text_face.score` (1–5). A record missing either face raises (B9).
- `judge_input/<probe_id>/ground_truth.json` — `context` / `attribute` / `setting`
  of the cell the probe reads; `setting` is already `setting_for(probe_kind, record)`
  (pre-event → old GT, final → new GT), so E4 needs no second resolution.
- `run_plan.yaml` — only for a probe that exhausted its attempts (the judge audit's
  `not_applicable`, see `harness.probe_na`; the CLI reads it from
  `judge_audit/manifest.json`). It is never judged; it gets the floor
  score `FLOOR_IX` / `FLOOR_TEXT` (rule `FLOOR_RULE`) with the GT of its plan step and
  stays in every denominator, so a pair with a floored endpoint counts and fails.
  The output lists those probes under `rule_assigned_scores`.

The 33 probe ids are fixed (`fin_01..28` + `pre_01..05`); a run missing any raises.
A pre-event probe pairs with the final probe that reads the same cell.

Run:
    PYTHONPATH=. python -m harness.metrics --run-dir <run> --judge-id <judge_id> \
        [--baseline-run-dir <run>] [--out-json <p>] [--out-md <p>]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

#: The fixed probe set (§2.1): 28 finals (14 attributes × work / personal) + 5 pre-event.
FINAL_PROBE_IDS = tuple(f"fin_{i:02d}" for i in range(1, 29))
PRE_EVENT_PROBE_IDS = tuple(f"pre_{i:02d}" for i in range(1, 6))
PROBE_IDS = FINAL_PROBE_IDS + PRE_EVENT_PROBE_IDS

SCORE_DOMAIN = (1, 2, 3, 4, 5)
#: E2: a text-face score at or below this is "low".
LOW_SCORE_MAX = 2
CONTEXTS = ("work", "personal")

#: The score of a probe that exhausted its attempts (owner 2026-10-16): IX wrong,
#: Text 1, denominator unchanged. The paper export reads these same names.
FLOOR_RULE = "retain_scores_model_failure_floor_v1"
FLOOR_IX = 0
FLOOR_TEXT = 1


@dataclass(frozen=True)
class ProbeScore:
    probe_id: str
    context: str
    attribute: str
    gt_setting: str
    pa_setting: str | None
    ix_match: bool
    text_score: int

    @property
    def cell(self) -> str:
        return f"{self.context}_{self.attribute}"

    @property
    def is_final(self) -> bool:
        return self.probe_id.startswith("fin_")


def floor_score(probe_id: str, context: str, attribute: str, gt_setting: str) -> ProbeScore:
    """The score `FLOOR_RULE` gives a probe that exhausted its attempts."""
    return ProbeScore(probe_id=probe_id, context=context, attribute=attribute, gt_setting=gt_setting,
                      pa_setting=None, ix_match=bool(FLOOR_IX), text_score=FLOOR_TEXT)


def floor_assignments(probe_ids) -> dict[str, dict]:
    """`rule_assigned_scores`: the floored probes, each with its rule and both face values."""
    return {pid: {"source": FLOOR_RULE, "ix": FLOOR_IX, "text": FLOOR_TEXT} for pid in sorted(probe_ids)}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing metrics input: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _plan_steps(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "run_plan.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing metrics input: {path}")
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {step["step_id"]: step for step in plan["steps"]}


def load_probe_scores(run_dir: Path, judge_id: str,
                      not_applicable: frozenset[str] = frozenset()) -> dict[str, ProbeScore]:
    """Load both faces for all 33 probes; raise on any missing probe, face or field.

    ``not_applicable`` names Probes that exhausted every attempt (see
    ``harness.probe_na``); each gets ``floor_score`` with its plan step's GT.
    """
    if not set(not_applicable) <= set(PROBE_IDS):
        raise ValueError("unknown N/A probe ids")
    steps = _plan_steps(run_dir) if not_applicable else {}
    scores: dict[str, ProbeScore] = {}
    for probe_id in PROBE_IDS:
        if probe_id in not_applicable:
            step = steps[probe_id]
            scores[probe_id] = floor_score(probe_id, step["context"], step["target_attribute"],
                                           step["target_setting"])
            continue
        record = _read_json(run_dir / "judge_output" / judge_id / f"{probe_id}.json")
        gt = _read_json(run_dir / "judge_input" / probe_id / "ground_truth.json")
        ix_face = record.get("ix_face")
        text_face = record.get("text_face")
        if not isinstance(ix_face, dict) or "exact_match" not in ix_face:
            raise ValueError(f"{probe_id}: judge output has no ix_face.exact_match (both faces required)")
        if not isinstance(text_face, dict) or "score" not in text_face:
            raise ValueError(f"{probe_id}: judge output has no text_face.score (both faces required)")
        score = text_face["score"]
        if score not in SCORE_DOMAIN:
            raise ValueError(f"{probe_id}: text_face.score {score!r} not in {list(SCORE_DOMAIN)}")
        if gt["context"] not in CONTEXTS:
            raise ValueError(f"{probe_id}: ground_truth.context {gt['context']!r} not in {list(CONTEXTS)}")
        scores[probe_id] = ProbeScore(
            probe_id=probe_id,
            context=gt["context"],
            attribute=gt["attribute"],
            gt_setting=gt["setting"],
            pa_setting=ix_face.get("pa_setting"),
            ix_match=bool(ix_face["exact_match"]),
            text_score=int(score),
        )
    return scores


def _consistency(probes: list[ProbeScore]) -> dict:
    """E1 shape: IX exact-match mean and text-face sum normalised to a percent."""
    n = len(probes)
    if n == 0:
        return {"n": 0, "ix_mean": None, "text_percent": None}
    return {
        "n": n,
        "ix_mean": sum(p.ix_match for p in probes) / n,
        "text_percent": 100.0 * sum(p.text_score for p in probes) / (max(SCORE_DOMAIN) * n),
    }


def _finals(scores: dict[str, ProbeScore]) -> list[ProbeScore]:
    return [scores[pid] for pid in FINAL_PROBE_IDS]


def e1_consistency(scores: dict[str, ProbeScore]) -> dict:
    return _consistency(_finals(scores))


def e2_cross_tab(scores: dict[str, ProbeScore]) -> dict:
    finals = _finals(scores)
    n = len(finals)
    said_not_done = [p.probe_id for p in finals if p.ix_match and p.text_score <= LOW_SCORE_MAX]
    done_not_said = [p.probe_id for p in finals if not p.ix_match and p.text_score > LOW_SCORE_MAX]
    return {
        "n": n,
        "low_score_max": LOW_SCORE_MAX,
        "ix_right_text_low": {"count": len(said_not_done), "share": len(said_not_done) / n, "probes": said_not_done},
        "ix_wrong_text_high": {"count": len(done_not_said), "share": len(done_not_said) / n, "probes": done_not_said},
    }


def e3_cross_scenario(scores: dict[str, ProbeScore]) -> dict:
    finals = _finals(scores)
    gt_by_cell = {p.cell: p.gt_setting for p in finals}
    attributes = sorted({p.attribute for p in finals})
    differing = [a for a in attributes if gt_by_cell[f"work_{a}"] != gt_by_cell[f"personal_{a}"]]
    cells = [p for p in finals if p.attribute in differing]
    other = {"work": "personal", "personal": "work"}
    crossed = [p for p in cells if p.pa_setting == gt_by_cell[f"{other[p.context]}_{p.attribute}"]]
    return {
        "differing_attributes": differing,
        "consistency": _consistency(cells),
        "cross_scenario_error": {
            "count": len(crossed),
            "share": len(crossed) / len(cells) if cells else None,
            "probes": [p.probe_id for p in crossed],
        },
    }


def e4_pet(scores: dict[str, ProbeScore]) -> dict:
    final_by_cell = {p.cell: p for p in _finals(scores)}
    pairs = []
    for pid in PRE_EVENT_PROBE_IDS:
        pre = scores[pid]
        final = final_by_cell.get(pre.cell)
        if final is None:
            raise ValueError(f"{pid}: no final probe reads cell {pre.cell}")
        pairs.append((pre, final))
    probes = [p for pair in pairs for p in pair]
    return {
        "pairs": [{"pre": pre.probe_id, "final": final.probe_id, "cell": pre.cell} for pre, final in pairs],
        **_consistency(probes),
    }


def e5_iqi(consistency: dict, baseline_consistency: dict | None) -> dict:
    if baseline_consistency is None:
        return {"baseline": None, "ix_diff": None, "text_diff": None}
    return {
        "baseline": baseline_consistency,
        "ix_diff": consistency["ix_mean"] - baseline_consistency["ix_mean"],
        "text_diff": consistency["text_percent"] - baseline_consistency["text_percent"],
    }


def compute_metrics(run_dir: Path, judge_id: str, baseline_run_dir: Path | None = None,
                    not_applicable: frozenset[str] = frozenset()) -> dict:
    not_applicable = frozenset(not_applicable)
    scores = load_probe_scores(run_dir, judge_id, not_applicable)
    e1 = e1_consistency(scores)
    baseline_e1 = None
    if baseline_run_dir is not None:
        baseline_e1 = e1_consistency(load_probe_scores(baseline_run_dir, judge_id))
    result = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "baseline_run_dir": None if baseline_run_dir is None else str(baseline_run_dir),
        "judge_id": judge_id,
        "E1_consistency": e1,
        "E2_cross_tab": e2_cross_tab(scores),
        "E3_cross_scenario": e3_cross_scenario(scores),
        "E4_pet": e4_pet(scores),
        "E5_iqi": e5_iqi(e1, baseline_e1),
    }
    if not_applicable:
        # Only runs with floored probes carry this key, so complete-run metrics stay byte-identical.
        result["rule_assigned_scores"] = floor_assignments(not_applicable)
    return result


def _fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%" if percent else f"{value:.3f}"


def render_markdown(metrics: dict) -> str:
    e1, e2, e3, e4, e5 = (metrics[k] for k in ("E1_consistency", "E2_cross_tab", "E3_cross_scenario", "E4_pet", "E5_iqi"))
    e3c = e3["consistency"]
    rows = [
        (f"E1 consistency ({e1['n']} finals)", _fmt(e1["ix_mean"]), _fmt(e1["text_percent"], percent=True)),
        ("E2 IX right & text low (<=2)", f"{e2['ix_right_text_low']['count']}/{e2['n']}", _fmt(100.0 * e2["ix_right_text_low"]["share"], percent=True)),
        ("E2 IX wrong & text high (>2)", f"{e2['ix_wrong_text_high']['count']}/{e2['n']}", _fmt(100.0 * e2["ix_wrong_text_high"]["share"], percent=True)),
        (f"E3 consistency on {e3c['n']} differing cells", _fmt(e3c["ix_mean"]), _fmt(e3c["text_percent"], percent=True)),
        ("E3 cross-scenario error share", f"{e3['cross_scenario_error']['count']}/{e3c['n']}", _fmt(None if e3["cross_scenario_error"]["share"] is None else 100.0 * e3["cross_scenario_error"]["share"], percent=True)),
        (f"E4 PET ({len(e4['pairs'])} pre + {len(e4['pairs'])} final)", _fmt(e4["ix_mean"]), _fmt(e4["text_percent"], percent=True)),
        ("E5 IQI (run - baseline)", _fmt(e5["ix_diff"]), _fmt(e5["text_diff"], percent=True)),
    ]
    lines = [
        f"# Metrics — judge `{metrics['judge_id']}`",
        "",
        f"run: `{metrics['run_dir']}`",
        f"baseline: `{metrics['baseline_run_dir'] or 'none'}`",
        f"E3 differing attributes: {', '.join(e3['differing_attributes']) or 'none'}",
        *([f"Floored (exhausted attempts, {FLOOR_RULE}: IX {FLOOR_IX}, text {FLOOR_TEXT}): "
           f"{', '.join(metrics['rule_assigned_scores'])}"] if metrics.get("rule_assigned_scores") else []),
        "",
        "| metric | IX face | text face |",
        "|---|---|---|",
    ]
    lines.extend(f"| {name} | {ix} | {text} |" for name, ix, text in rows)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate E1–E5 over one run's judge outputs (design §5).")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="run directory holding judge_input/<probe_id>/ and judge_output/<judge_id>/")
    parser.add_argument("--judge-id", required=True,
                        help="judge id as in judge_output/<judge_id>/, e.g. llm__anthropic_claude-x or human__jl")
    parser.add_argument("--baseline-run-dir", type=Path, default=None,
                        help="memory-removed baseline run directory, same layout and judge id; enables E5")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="output JSON path; default <run-dir>/metrics/<judge-id>.json")
    parser.add_argument("--out-md", type=Path, default=None,
                        help="output markdown table path; default <run-dir>/metrics/<judge-id>.md")
    args = parser.parse_args(argv)

    # The exhausted Probes the scorer recorded, floored as score_completed_run does
    # (imported here: harness.judge_audit imports this module).
    from harness.judge_audit import not_applicable_ids
    audit = args.run_dir / "judge_audit" / "manifest.json"
    not_applicable = not_applicable_ids(json.loads(audit.read_text(encoding="utf-8"))) if audit.is_file() else frozenset()
    metrics = compute_metrics(args.run_dir, args.judge_id, args.baseline_run_dir, not_applicable)
    out_json = args.out_json or args.run_dir / "metrics" / f"{args.judge_id}.json"
    out_md = args.out_md or args.run_dir / "metrics" / f"{args.judge_id}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(metrics)
    out_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
