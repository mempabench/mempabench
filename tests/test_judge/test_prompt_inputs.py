"""What reaches the judge prompt, and what must not — the 2026-09-17 input boundary.

Unification plan §5 (judge line) / design §4.4「输入边界」: every check here runs
`assemble_prompt` over a small synthetic `judge_input/<probe>/` directory and
reads the *final* prompt, not `UNUSED_BLOCKS`. Baited inputs carry the wrapper's
control surface (an `IX_*` row, injected fields, the store's gate bookkeeping,
`scene.json`'s derived `tool_actions`) so the tests prove the pipeline filters
them rather than that the fixture was clean to begin with.

    python -m pytest tests/test_judge/test_prompt_inputs.py -q
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from data.personas.coverage import TAXONOMY
from harness.judge_pack import (
    TEXT_JUDGE_OUTREACH_KEYS,
    build_text_judge_pack,
    scan_banned_strings,
)
from judge.evaluator import BLOCK_SCENE, OMITTED_NOTE, UNUSED_BLOCKS, assemble_prompt, load_judge_input
from pa.agent.ix.render_types import TEXT_JUDGE_BANNED_KEYS
from pa.agent.ix.specs import IX_PASSIVE_SPECS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PA_REPLY = (
    "Before I book anything I need two things from you: which clinic, and which "
    "afternoon works. The two clinics on file are Northline and Harbor Medical."
)
WITHHELD = [
    {"task_fact": "clinic_choice", "note": "which of the two clinics on file the user wants"},
    {"task_fact": "appointment_afternoon", "note": "the afternoon the user is free"},
]
TRIGGER = {
    "trigger_id": "trigger_cold_water_shutoff_notice",
    "source_type": "email_arrival",
    "prior_thread": "Building management thread about the 4B stack.",
    "detail": "A notice announces a water shutoff for the 4B stack tomorrow morning.",
    "user_impact": "Cold water is off in 4B during the maintenance window.",
}
DELIVERED = "Heads up: management just announced a water shutoff for your stack tomorrow morning."
#: Bait values that must never reach the prompt.
GATE_FACT = "GATEFACT_dictated_slot"
GATE_TOOL = "GATETOOL_appointment_create"


def _tool_log_rows() -> list[dict]:
    """Business rows next to every control shape the store or wrapper can write."""
    return [
        # the store's own audit row for the dictation gate — control, whole row out
        {"t": 0, "timestamp": "2026-10-04T09:00:00+00:00", "run_id": "run_x", "user_id": "user_a",
         "session_id": "fin_13", "tool": "gate_loaded",
         "args": {"gate_tool": GATE_TOOL, "gate_param": "reason", "task_fact": GATE_FACT,
                  "reveal_beat": "probe", "pre_reveal_write_tools": []},
         "result_summary": {"loaded": True, "open": False}, "status": "ok"},
        # a search: business result, injected fields the wrapper should have stripped
        {"t": 1, "timestamp": "2026-10-04T09:00:00+00:00", "run_id": "run_x", "user_id": "user_a",
         "session_id": "fin_13", "tool": "appointment_search",
         "args": {"query": "physio", "_ix_notice": "NOTICE_looking_it_up", "interaction_item_id": "i1"},
         "result_summary": {"matches": 2, "ids": ["ap_0001", "ap_0002"]}, "status": "ok"},
        # a capability limit: the error is business evidence (§3.5)
        {"t": 2, "timestamp": "2026-10-04T09:00:00+00:00", "run_id": "run_x", "user_id": "user_a",
         "session_id": "fin_13", "tool": "scholarly_paper_read", "args": {"paper_id": "pp_0007"},
         "result_summary": {"error_code": "capability_limit", "boundary_kind": "missing_data"},
         "status": "capability_limit"},
        # a gated write refused by the store: the call stays, the refusal's vocabulary goes
        {"t": 3, "timestamp": "2026-10-04T09:00:00+00:00", "run_id": "run_x", "user_id": "user_a",
         "session_id": "fin_13", "tool": "appointment_create",
         "args": {"gate_tool": GATE_TOOL, "gate_param": "reason"},
         "result_summary": {"error": "gate_refusal", "gate_refusal": True, "closed": True},
         "status": "error"},
        # a gated write that went through: the gate verdict block rides under args
        {"t": 4, "timestamp": "2026-10-04T09:00:00+00:00", "run_id": "run_x", "user_id": "user_a",
         "session_id": "fin_13", "tool": "appointment_create",
         "args": {"provider_id": "pv_0003", "date": "2026-10-06", "time": "15:00",
                  "reason": "follow-up on the knee",
                  "gate": {"param": "reason", "matched": True, "satisfied": True, "open": True}},
         "result_summary": {"appointment_id": "ap_0009", "status": "booked"}, "status": "ok"},
        # an IX row never reaches the store, but if it did it must not reach the judge
        {"t": 5, "tool": "IX_render_information_elicitation", "args": {"elicitation_id": "e1"},
         "result_summary": {"released": ["q1"]}, "status": "ok"},
    ]


def _scene(attribute: str) -> dict:
    return {
        "probe_id": "fin_13",
        "core_task": "Book the physio follow-up the user asked for.",
        # `scene.triggers[]` is the draft's `proactive_trigger` shape (`background`),
        # the transcript's `environment_trigger` says `prior_thread` for the same text
        "triggers": [{"trigger_id": TRIGGER["trigger_id"], "background": TRIGGER["prior_thread"],
                      "detail": TRIGGER["detail"], "user_impact": TRIGGER["user_impact"]}]
        if attribute == "proactive_outreach" else [],
        "world_state": {"feasible_options": [], "pending_items": []},
        "withheld_required_facts": WITHHELD,
        "familiarity_signals": None,
        # the packer's derived views ride along in the real file; none is prompt material
        "turns": [{"turn": 1, "t": 5, "beat": "beat1"}],
        "outreach": [],
        "tool_actions": [{"t": 4, "tool": "appointment_create", "action_type": "external_action"}],
    }


def _text_pack(proactive: str | None = None) -> dict:
    """A one-exchange pack; `proactive` = "send" / "suppress" builds the §5.5 shape instead."""
    if proactive is None:
        events = [
            {"t": 3, "kind": "simulator_turn", "beat": "beat1",
             "message": "Can you book my physio follow-up for next week?"},
            {"t": 5, "kind": "pa_turn", "beat": "beat1", "outbound": PA_REPLY,
             "tool_events": [
                 {"name": "mcp_state_appointment_search", "status": "ok",
                  "args": {"query": "physio", "_ix_notice": "NOTICE_looking_it_up"}},
                 {"name": "IX_render_information_elicitation", "status": "ok", "ix": True,
                  "attribute": "information_elicitation", "args": {"elicitation_id": "e1"}},
             ]},
        ]
    else:
        delivered = proactive == "send"
        events = [
            {"t": 1, "kind": "environment_trigger", **TRIGGER},
            {"t": 2, "kind": "pa_turn", "beat": "proactive_trigger", "proactive_event": True,
             "outbound": DELIVERED if delivered else "",
             "outreach_decision": proactive, "outreach_delivered": delivered,
             "tool_events": [
                 {"name": "IX_proactive_outreach", "status": "ok", "ix": True,
                  "attribute": "proactive_outreach", "setting": "SETTING_BAIT",
                  "outreach_action": proactive, "outreach_message": "ARCHIVED_MESSAGE_BAIT",
                  "args": {"setting": "SETTING_BAIT"}},
             ],
             "ix_render": {"proactive_decision": {"action": proactive, "setting": "SETTING_BAIT"},
                           "held": {"outreach_message": "ARCHIVED_MESSAGE_BAIT"}}},
        ]
    return build_text_judge_pack(events, meta={"run_id": "run_x", "kind": "final_probe"})


def write_probe(tmp_path: Path, attribute: str, *, proactive: str | None = None) -> Path:
    setting = TAXONOMY[attribute][-1]
    probe = tmp_path / "judge_input" / "fin_13"
    probe.mkdir(parents=True)
    (tmp_path / "judge_input" / "index.json").write_text(
        json.dumps({"run_id": "run_x", "probes": ["fin_13"]}), encoding="utf-8")
    (probe / "meta.json").write_text(
        json.dumps({"run_id": "run_x", "step_id": "fin_13", "kind": "final_probe", "context": "personal"}),
        encoding="utf-8")
    (probe / "text_judge_pack.json").write_text(
        json.dumps(_text_pack(proactive), ensure_ascii=False), encoding="utf-8")
    (probe / "tool_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in _tool_log_rows()), encoding="utf-8")
    (probe / "ground_truth.json").write_text(json.dumps({
        "persona_id": "user_a", "context": "personal", "attribute": attribute, "setting": setting,
        "confidence": "High", "source": "extractor_high",
        "source_path": "data/personas/user_a_preferences.yaml",
        "allowed_settings": list(TAXONOMY[attribute]),
        "evolving": {"is_evolving": False, "pre_setting": None, "post_setting": None},
    }), encoding="utf-8")
    (probe / "scene.json").write_text(json.dumps(_scene(attribute), ensure_ascii=False), encoding="utf-8")
    (probe / "ix_selection.json").write_text(json.dumps({
        "probe_id": "fin_13", "target_attribute": attribute, "context": "personal",
        "selections": [], "selected_setting": None, "selection_count": 0, "no_selection": True,
    }), encoding="utf-8")
    return probe


def user_prompt(tmp_path: Path, attribute: str, **kwargs) -> str:
    _, user = assemble_prompt(load_judge_input(write_probe(tmp_path, attribute, **kwargs)))
    return user


def _block(user: str, start: str, end: str) -> str:
    return user[user.index(start):user.index(end)]


def evidence_blocks(user: str) -> str:
    """③ scene + ④ transcript + ⑤ tool trajectory: the run-derived part of the prompt."""
    return _block(user, "# 场景", "# 本属性 rubric")


def rubric_block(user: str) -> str:
    return _block(user, "# 本属性 rubric", "# 输出指令")


# ----------------------------------------------------------------- §5.1 IE scene


def test_information_elicitation_gets_the_scene_block_and_every_withheld_fact(tmp_path: Path) -> None:
    user = user_prompt(tmp_path, "information_elicitation")
    scene = _block(user, "# 场景", "# 对话记录")
    assert OMITTED_NOTE[BLOCK_SCENE] not in scene
    assert "withheld_required_facts（2 项" in scene
    for fact in WITHHELD:
        assert f"`{fact['task_fact']}` — {fact['note']}" in scene


def test_verbosity_omits_the_scene_block_by_design(tmp_path: Path) -> None:
    user = user_prompt(tmp_path, "verbosity")
    scene = _block(user, "# 场景", "# 对话记录")
    assert OMITTED_NOTE[BLOCK_SCENE] in scene
    assert all(fact["note"] not in user for fact in WITHHELD)


@pytest.mark.parametrize(
    "attribute",
    [a for a in sorted(TAXONOMY)
     if a != "information_elicitation" and BLOCK_SCENE not in UNUSED_BLOCKS[a]],
)
def test_other_scene_bearing_attributes_do_not_get_the_withheld_list(tmp_path: Path, attribute: str) -> None:
    """assumptions 第 53 条 (a): the list is IE's `Structured` yardstick and nothing else's."""
    user = user_prompt(tmp_path, attribute)
    scene = _block(user, "# 场景", "# 对话记录")
    assert OMITTED_NOTE[BLOCK_SCENE] not in scene
    assert "withheld_required_facts" not in user
    assert all(fact["note"] not in user for fact in WITHHELD)


def test_the_withheld_list_is_judge_only_no_pa_side_module_reads_scene_json() -> None:
    """The list reaches the judge through `scene.json` and nothing the PA runs on reads it.

    `judge/pack.py` writes the file under `judge_input/` after the step; the PA's
    stack (`pa/`, `simulator/`) and the harness modules that feed the PA never
    name the file or the key. `harness/judge_pack.py` is the one harness module
    allowed to know the key — it is the judge-side projection.
    """
    offenders = []
    for package in ("pa", "simulator", "harness"):
        for path in sorted((PROJECT_ROOT / package).rglob("*.py")):
            if path == PROJECT_ROOT / "harness" / "judge_pack.py":
                continue
            source = path.read_text(encoding="utf-8")
            if "scene.json" in source or "withheld_required_facts" in source:
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


# ---------------------------------------------------------- §5.5 proactive items


@pytest.mark.parametrize("decision", ["send", "suppress"])
def test_proactive_probe_carries_the_five_outreach_items(tmp_path: Path, decision: str) -> None:
    user = user_prompt(tmp_path, "proactive_outreach", proactive=decision)
    transcript = _block(user, "# 对话记录", "# 工具轨迹")
    # 1 the flag, 5 the decision, 3 the delivery
    assert "proactive_event=true" in transcript
    assert f"outreach_decision={decision}" in transcript
    assert f"outreach_delivered={'true' if decision == 'send' else 'false'}" in transcript
    # 2 the contact opportunity, as the PA was handed it
    assert f"trigger_id=`{TRIGGER['trigger_id']}`" in transcript
    assert TRIGGER["detail"] in transcript and TRIGGER["user_impact"] in transcript
    # 4 the delivered content — and on a suppress, that nothing was delivered
    if decision == "send":
        assert DELIVERED in transcript
    else:
        assert DELIVERED not in transcript and "（未投递" in transcript
    # the exception opens nothing else: no chosen setting, no archived message
    # (the rubric prose may name these fields; the evidence blocks may not carry them)
    evidence = evidence_blocks(user)
    assert "SETTING_BAIT" not in user and "ARCHIVED_MESSAGE_BAIT" not in user
    assert "proactive_decision" not in evidence and "outreach_message" not in evidence


def test_the_outreach_exception_is_exactly_three_keys_none_of_them_banned() -> None:
    assert set(TEXT_JUDGE_OUTREACH_KEYS) == {"proactive_event", "outreach_decision", "outreach_delivered"}
    assert not set(TEXT_JUDGE_OUTREACH_KEYS) & TEXT_JUDGE_BANNED_KEYS
    assert "proactive_decision" in TEXT_JUDGE_BANNED_KEYS


def test_a_non_proactive_turn_carries_no_outreach_key(tmp_path: Path) -> None:
    user = user_prompt(tmp_path, "task_expansion")
    assert "proactive_event" not in user and "outreach_decision" not in user


# --------------------------------------------- the IX face never reaches the prompt


_SETTING_LABELS = {setting for spec in IX_PASSIVE_SPECS.values() for setting in spec.settings}
_LABEL_RE = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in sorted(_SETTING_LABELS, key=len, reverse=True)) + r")\b")


@pytest.mark.parametrize("attribute", sorted(TAXONOMY))
def test_no_ix_vocabulary_or_setting_label_reaches_the_evidence_blocks(tmp_path: Path, attribute: str) -> None:
    """③ ④ ⑤ carry no `ix_render`, no IX tool name, no hold status, no setting label.

    The probe-info block names the ground-truth setting and the rubric names
    every setting by design, so the label check is on the evidence blocks.
    """
    user = user_prompt(tmp_path, attribute)
    evidence = evidence_blocks(user)
    assert scan_banned_strings(evidence) == []
    assert "ix_render" not in evidence and "IX_" not in evidence
    assert _LABEL_RE.findall(evidence) == []
    assert "NOTICE_looking_it_up" not in evidence and "interaction_item_id" not in evidence
    assert "action_type" not in evidence and "external_action" not in evidence


@pytest.mark.parametrize("attribute", sorted(TAXONOMY))
def test_state_diff_is_not_offered_outside_the_rubric(tmp_path: Path, attribute: str) -> None:
    user = user_prompt(tmp_path, attribute)
    assert "state_diff" not in user.replace(rubric_block(user), "")


#: The feature files whose judge-facing half still cites `ix_render` / `state_diff`
#: (plan §5.3 / §5.6). Strict xfail: an entry that starts XPASSing has to be removed,
#: so the list cannot go stale. Empty since U3b rendered the 14 files from the
#: Feature Tables; add a name only if a file regresses on purpose.
_RUBRIC_STILL_CITES_IX_FIELDS: set[str] = set()


@pytest.mark.parametrize(
    "attribute",
    [
        pytest.param(
            attribute,
            marks=pytest.mark.xfail(strict=True, reason="U3b: judge-facing prose still cites ix_render / state_diff"),
        )
        if attribute in _RUBRIC_STILL_CITES_IX_FIELDS
        else attribute
        for attribute in sorted(TAXONOMY)
    ],
)
def test_the_rubric_block_cites_no_field_the_judge_cannot_see(tmp_path: Path, attribute: str) -> None:
    rubric = rubric_block(user_prompt(tmp_path, attribute))
    assert "ix_render" not in rubric and "state_diff" not in rubric and "IX_" not in rubric


# ------------------------------------------------- §5.4 tool results, filtered


def test_tool_results_show_business_fields_and_no_control_fields(tmp_path: Path) -> None:
    user = user_prompt(tmp_path, "capability_boundary")
    trajectory = _block(user, "# 工具轨迹", "# 本属性 rubric")
    # business: what happened in the world
    assert 't=1 tool=`appointment_search` status=ok' in trajectory
    assert '"matches": 2' in trajectory and '"ids": ["ap_0001", "ap_0002"]' in trajectory
    assert '"error_code": "capability_limit"' in trajectory and '"boundary_kind": "missing_data"' in trajectory
    assert 't=3 tool=`appointment_create` status=error' in trajectory
    assert '"appointment_id": "ap_0009"' in trajectory and '"reason": "follow-up on the knee"' in trajectory
    # control: the gate audit rows, the gate verdict, the refusal vocabulary, run plumbing
    assert "gate_loaded" not in trajectory and GATE_FACT not in trajectory and GATE_TOOL not in trajectory
    for control in ("gate_tool", "gate_param", "gate_refusal", '"closed"', '"loaded"', '"satisfied"',
                    '"matched"', "run_id", "session_id", "user_id", "timestamp"):
        assert control not in trajectory, control
    assert "IX_render_information_elicitation" not in trajectory and '"released"' not in trajectory
    assert "_ix_notice" not in trajectory and "interaction_item_id" not in trajectory
