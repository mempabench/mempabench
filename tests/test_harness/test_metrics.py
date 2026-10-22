"""Aggregation E1–E5 (design §5) over synthetic 33-probe judge outputs.

Pure data, no LLM. Run:
    PYTHONPATH=. python -m pytest tests/test_harness/test_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.metrics import (
    FINAL_PROBE_IDS,
    PRE_EVENT_PROBE_IDS,
    PROBE_IDS,
    compute_metrics,
    main,
    render_markdown,
)

JUDGE_ID = "llm__test_judge"
ATTRIBUTES = [f"attr_{i:02d}" for i in range(1, 15)]
# 28 finals: fin_01..14 = work × attributes, fin_15..28 = personal × the same order.
FINAL_CELLS = [("work", a) for a in ATTRIBUTES] + [("personal", a) for a in ATTRIBUTES]
# attr_01 / attr_02 differ across contexts; every other attribute has the same GT in both.
DIFFERING = {"attr_01", "attr_02"}
# The 5 pre-event probes read work_attr_01..05 (pre_0k → fin_0k).
PRE_CELLS = [("work", a) for a in ATTRIBUTES[:5]]


def _gt_setting(context: str, attribute: str) -> str:
    return f"{context}_{attribute}_gt" if attribute in DIFFERING else f"{attribute}_gt"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_run(
    root: Path,
    *,
    ix_match: dict[str, bool] | None = None,
    scores: dict[str, int] | None = None,
    pa_setting: dict[str, str] | None = None,
    drop_face: tuple[str, str] | None = None,
    skip_probe: str | None = None,
) -> Path:
    """Write judge_input/ground_truth.json + judge_output records for all 33 probes.

    Defaults: every IX selection matches, every text score is 5.
    """
    ix_match = ix_match or {}
    scores = scores or {}
    pa_setting = pa_setting or {}
    cells = dict(zip(FINAL_PROBE_IDS, FINAL_CELLS)) | dict(zip(PRE_EVENT_PROBE_IDS, PRE_CELLS))
    for probe_id in PROBE_IDS:
        if probe_id == skip_probe:
            continue
        context, attribute = cells[probe_id]
        gt = _gt_setting(context, attribute)
        matched = ix_match.get(probe_id, True)
        _write(root / "judge_input" / probe_id / "ground_truth.json",
               {"persona_id": "user_a", "context": context, "attribute": attribute, "setting": gt})
        record = {
            "schema_version": 1, "probe_id": probe_id, "judge_id": JUDGE_ID,
            "ix_face": {"gt_setting": gt,
                        "pa_setting": pa_setting.get(probe_id, gt if matched else "other"),
                        "exact_match": matched, "no_selection": False, "computed_by": "code"},
            "text_face": {"score": scores.get(probe_id, 5), "score_domain": [1, 2, 3, 4, 5],
                          "rationale": "", "evidence": [], "asked_user_for_preference": False},
        }
        if drop_face and drop_face[0] == probe_id:
            del record[drop_face[1]]
        _write(root / "judge_output" / JUDGE_ID / f"{probe_id}.json", record)
    return root


def test_e1_consistency_two_faces(tmp_path):
    # 7 of 28 finals IX-wrong; text scores: all finals 3 except fin_01 = 5; pre probes ignored by E1.
    wrong = {pid: False for pid in FINAL_PROBE_IDS[:7]}
    scores = {pid: 3 for pid in FINAL_PROBE_IDS} | {"fin_01": 5} | {pid: 1 for pid in PRE_EVENT_PROBE_IDS}
    run = build_run(tmp_path / "run", ix_match=wrong, scores=scores)
    e1 = compute_metrics(run, JUDGE_ID)["E1_consistency"]
    assert e1["n"] == 28
    assert e1["ix_mean"] == pytest.approx(21 / 28)
    assert e1["text_percent"] == pytest.approx(100.0 * (27 * 3 + 5) / 140)


def test_e2_cross_tab_low_is_score_at_most_two(tmp_path):
    # fin_01: IX right, score 2 (low) → said-not-done. fin_02: IX right, score 3 → neither.
    # fin_03: IX wrong, score 3 (high) → done-not-said. fin_04: IX wrong, score 2 → neither.
    run = build_run(tmp_path / "run",
                    ix_match={"fin_03": False, "fin_04": False},
                    scores={"fin_01": 2, "fin_02": 3, "fin_03": 3, "fin_04": 2})
    e2 = compute_metrics(run, JUDGE_ID)["E2_cross_tab"]
    assert e2["ix_right_text_low"] == {"count": 1, "share": pytest.approx(1 / 28), "probes": ["fin_01"]}
    assert e2["ix_wrong_text_high"] == {"count": 1, "share": pytest.approx(1 / 28), "probes": ["fin_03"]}


def test_e3_only_differing_attributes_and_crossed_selection(tmp_path):
    # Differing cells: work/personal × attr_01, attr_02 → fin_01, fin_02, fin_15, fin_16.
    # fin_01 selects the personal GT of attr_01 (crossed); fin_16 is wrong but not crossed.
    # fin_03 (same-GT attribute) is wrong and low, and must not enter E3 at all.
    run = build_run(tmp_path / "run",
                    ix_match={"fin_01": False, "fin_16": False, "fin_03": False},
                    pa_setting={"fin_01": _gt_setting("personal", "attr_01")},
                    scores={"fin_01": 1, "fin_02": 5, "fin_15": 5, "fin_16": 3, "fin_03": 1})
    e3 = compute_metrics(run, JUDGE_ID)["E3_cross_scenario"]
    assert e3["differing_attributes"] == ["attr_01", "attr_02"]
    assert e3["consistency"]["n"] == 4
    assert e3["consistency"]["ix_mean"] == pytest.approx(2 / 4)
    assert e3["consistency"]["text_percent"] == pytest.approx(100.0 * (1 + 5 + 5 + 3) / 20)
    assert e3["cross_scenario_error"] == {"count": 1, "share": pytest.approx(1 / 4), "probes": ["fin_01"]}


def test_e4_pet_averages_pre_and_paired_finals(tmp_path):
    # pre_01..05 pair with fin_01..05 (same cell). pre all score 1 & IX wrong; finals score 5 & IX right.
    run = build_run(tmp_path / "run",
                    ix_match={pid: False for pid in PRE_EVENT_PROBE_IDS},
                    scores={pid: 1 for pid in PRE_EVENT_PROBE_IDS})
    e4 = compute_metrics(run, JUDGE_ID)["E4_pet"]
    assert [(p["pre"], p["final"]) for p in e4["pairs"]] == [(f"pre_{i:02d}", f"fin_{i:02d}") for i in range(1, 6)]
    assert e4["n"] == 10
    assert e4["ix_mean"] == pytest.approx(5 / 10)
    assert e4["text_percent"] == pytest.approx(100.0 * (5 * 1 + 5 * 5) / 50)


def test_e5_iqi_is_difference_against_baseline(tmp_path):
    run = build_run(tmp_path / "run", scores={pid: 4 for pid in FINAL_PROBE_IDS})
    baseline = build_run(tmp_path / "baseline",
                         ix_match={pid: False for pid in FINAL_PROBE_IDS[:14]},
                         scores={pid: 2 for pid in FINAL_PROBE_IDS})
    metrics = compute_metrics(run, JUDGE_ID, baseline)
    e5 = metrics["E5_iqi"]
    assert e5["baseline"]["ix_mean"] == pytest.approx(0.5)
    assert e5["ix_diff"] == pytest.approx(0.5)
    assert e5["text_diff"] == pytest.approx(80.0 - 40.0)
    assert compute_metrics(run, JUDGE_ID)["E5_iqi"] == {"baseline": None, "ix_diff": None, "text_diff": None}


@pytest.mark.parametrize("face", ["ix_face", "text_face"])
def test_missing_face_raises(tmp_path, face):
    run = build_run(tmp_path / "run", drop_face=("fin_07", face))
    with pytest.raises(ValueError, match=f"fin_07: judge output has no {face}"):
        compute_metrics(run, JUDGE_ID)


def test_missing_probe_raises(tmp_path):
    run = build_run(tmp_path / "run", skip_probe="pre_03")
    with pytest.raises(FileNotFoundError, match="pre_03"):
        compute_metrics(run, JUDGE_ID)


def test_cli_writes_json_and_markdown(tmp_path, capsys):
    run = build_run(tmp_path / "run")
    assert main(["--run-dir", str(run), "--judge-id", JUDGE_ID]) == 0
    written = json.loads((run / "metrics" / f"{JUDGE_ID}.json").read_text(encoding="utf-8"))
    assert written["E1_consistency"]["text_percent"] == pytest.approx(100.0)
    md = (run / "metrics" / f"{JUDGE_ID}.md").read_text(encoding="utf-8")
    assert md == render_markdown(written)
    assert "| E1 consistency (28 finals) | 1.000 | 100.0% |" in md
    assert "| E5 IQI (run - baseline) | n/a | n/a |" in md
    assert "E1 consistency" in capsys.readouterr().out
