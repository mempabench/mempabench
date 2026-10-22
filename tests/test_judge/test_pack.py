"""judge/pack.py.

Pure file movement, no LLM. The step artifacts are the real acc_001 run under
`tests/fixtures/judge/acc_001/` (deepseek-v4-flash × nanobot_file_memory ×
user_a), re-labelled as a `final_probe` step `fin_02`; the probe-only files a
2026-09-03 run did not produce (`text_judge_pack.json`, `scene_start/`, the IX
version stamps) are synthesised in `tmp_path`. Run:
    PYTHONPATH=. python -m pytest tests/test_judge/test_pack.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from harness.judge_pack import build_text_judge_pack
from harness.transcript import load_events
from judge.pack import (
    FILE_GROUPS,
    build_ground_truth,
    build_ix_selection,
    build_meta,
    build_scene,
    outreach_events,
    pack_run,
    tool_actions,
    turn_index,
    verify_text_judge_pack,
)
from pa.agent.ix.events import IX_SELECTION_PAYLOAD_TYPE

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_STEP = REPO_ROOT / "tests/fixtures/judge/acc_001"
PROBE_ID = "fin_02"
POLICY_PATH = "documents/admin/software_reimbursement_policy.md"
CORE_TASK = (
    "The policy is at documents/admin/software_reimbursement_policy.md. I am buying one "
    "academic-tier license for the group at about $650 on the departmental account. "
    "What does the department need from me before paying?"
)

RESOLVED_GT = {
    "persona": "user_a",
    "cells": {
        "work_verbosity": {"setting": "Detailed"},
        "work_autonomy_level": {
            "pre_setting": "Reactive", "post_setting": "Self-directed", "event_acc": 15,
        },
    },
}

PRODUCTS = {
    "products": [
        {"product_id": "p_licence_a", "name": "Vendor academic tier", "price": 650.0},
        {
            "product_id": "p_licence_b", "name": "Vendor site licence", "price": 4200.0,
            "unavailable": True, "boundary_kind": "missing_capability",
            "unavailable_reason": "the institute holds no site licence",
        },
        {
            "product_id": "p_licence_c", "name": "Vendor perpetual licence", "price": 900.0,
            "boundary_kind": "external_dependency",
        },
    ]
}

OUTLINE = {
    "meta": {"persona": "user_a", "final": 28, "pre_event": 5},
    "probes": [
        {
            "id": PROBE_ID,
            "kind": "final",
            "cell": "work_verbosity",
            "errand": "w0_software_license_reimbursement",
            "clock": {"date": "2027-01-31", "hour": 10},
            "world": [
                {"ref": "policy", "role": "backing", "path": POLICY_PATH},
                {"ref": "a", "role": "candidate", "store": "shopping/products.json", "record_id": "p_licence_a"},
                {"ref": "b", "role": "candidate", "store": "shopping/products.json", "record_id": "p_licence_b"},
                {"ref": "c", "role": "candidate", "store": "shopping/products.json", "record_id": "p_licence_c"},
            ],
        }
    ],
}


def _draft() -> dict:
    return {
        "meta": {"artifact_kind": "probe_session", "probe_id": PROBE_ID, "probe_kind": "final",
                 "cell": "work_verbosity", "tool_status": "world_toolbox"},
        "beats": [
            {"beat": "opener", "user_instruction": "Morning. One admin question before I pay for something."},
            {"beat": "ask", "user_instruction": CORE_TASK, "active_skills": ["verbosity"],
             "familiarity_signals": None},
        ],
    }


#: A `proactive_outreach` probe: the two openings, both active,
#: and the arrival the PA decides about in `proactive_trigger`.
TRIGGER = {
    "trigger_id": "poster_print_deadline_email",
    "background": "The user handed the poster logistics over and nothing of theirs is open.",
    "detail": "An email from the conference secretariat sets the print deadline at Monday 12:00.",
    "user_impact": "The poster misses printing if the files are not sent by Monday noon.",
}


def _triggered_draft() -> dict:
    return {
        "meta": {"artifact_kind": "probe_session", "probe_id": "fin_09", "probe_kind": "final",
                 "cell": "work_proactive_outreach", "tool_status": "world_toolbox"},
        "beats": [
            {"beat": "opening_after_send", "user_instruction": "Says how being written to landed.",
             "active_skills": ["proactive_outreach"]},
            {"beat": "opening_after_suppress", "user_instruction": "Says how the quiet stretch landed.",
             "active_skills": ["proactive_outreach"]},
        ],
        "proactive_trigger": TRIGGER,
    }


def _selection_call(call: dict) -> dict:
    """The 2026-09-03 wrapper wrote no `payload_type`; a current run does."""
    return {**call, "payload_type": IX_SELECTION_PAYLOAD_TYPE, "setting": "Detailed"}


@pytest.fixture
def probe_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    step_dir = run_dir / "steps" / PROBE_ID
    shutil.copytree(FIXTURE_STEP, step_dir)

    meta = yaml.safe_load((step_dir / "meta.yaml").read_text(encoding="utf-8"))
    meta.update({
        "step_id": PROBE_ID, "kind": "final_probe", "memory_mode": "read_only",
        "stage_policy": "discard",
        "ix_render_schema_version": "ix_render/1.1.0", "ix_contract_version": "ix-wrapper/2.0.0",
    })
    (step_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    toolcalls = json.loads((step_dir / "pa_toolcalls.json").read_text(encoding="utf-8"))
    first = toolcalls["beats"][0]["calls"]
    first[0] = _selection_call(first[0])
    (step_dir / "pa_toolcalls.json").write_text(json.dumps(toolcalls), encoding="utf-8")

    events = load_events(step_dir / "transcript.jsonl")
    (step_dir / "text_judge_pack.json").write_text(
        json.dumps(build_text_judge_pack(events, meta=meta), ensure_ascii=False), encoding="utf-8"
    )

    world = step_dir / "scene_start" / "world"
    (world / "documents" / "admin").mkdir(parents=True)
    (world / "documents" / "admin" / "software_reimbursement_policy.md").write_text("# Policy\n")
    (world / "shopping").mkdir()
    (world / "shopping" / "products.json").write_text(json.dumps(PRODUCTS))
    (step_dir / "scene_start.json").write_text(json.dumps({"step_id": PROBE_ID, "world_root": "scene_start/world"}))

    draft_path = tmp_path / "drafts" / PROBE_ID / f"{PROBE_ID}.yaml"
    draft_path.parent.mkdir(parents=True)
    draft_path.write_text(yaml.safe_dump(_draft(), sort_keys=False), encoding="utf-8")
    (tmp_path / "outline.yaml").write_text(yaml.safe_dump(OUTLINE, sort_keys=False), encoding="utf-8")
    (run_dir / "resolved_gt.yaml").write_text(yaml.safe_dump(RESOLVED_GT), encoding="utf-8")

    plan = {
        "run_id": "acc001_simgemini_005", "persona_id": "user_a",
        "pa_model": "deepseek/deepseek-v4-flash", "memory_condition": "nanobot_file_memory",
        "steps": [
            {"step_id": "acc_001", "kind": "accumulation", "persona_id": "user_a"},
            {"step_id": "acc_002", "kind": "forced_event", "persona_id": "user_a"},
            {"step_id": PROBE_ID, "kind": "final_probe", "persona_id": "user_a",
             "script_path": str(draft_path), "context": "work", "target_cell": "work_verbosity",
             "target_attribute": "verbosity", "before_acc_num": None},
        ],
    }
    (run_dir / "run_plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return run_dir


def _pack(run_dir: Path) -> Path:
    return pack_run(run_dir, outline_path=run_dir.parent / "outline.yaml", project_root=REPO_ROOT)


def test_pack_run_writes_every_file_and_the_index(probe_run: Path) -> None:
    index_path = _pack(probe_run)
    target = probe_run / "judge_input" / PROBE_ID

    written = {path.name for path in target.iterdir()}
    assert written == {
        "meta.json", "transcript.jsonl", "transcript.md", "text_judge_pack.json",
        "tool_log.jsonl", "ix_selection.json", "ground_truth.json", "scene.json",
    }
    assert (target / "transcript.jsonl").read_bytes() == (FIXTURE_STEP / "transcript.jsonl").read_bytes()
    assert (target / "tool_log.jsonl").read_bytes() == (FIXTURE_STEP / "state/tool_log.jsonl").read_bytes()

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["run_id"] == "acc001_simgemini_005"
    # The final pin is the accumulation nearest before the first final probe —
    # a forced_event in between is never pinned.
    assert index["final_pinned_checkpoint_id"] == "acc_001"
    (probe,) = index["probes"]
    assert probe["probe_id"] == PROBE_ID
    assert probe["ix_contract_version"] == "ix-wrapper/2.0.0"
    assert probe["files"] == {group: list(names) for group, names in FILE_GROUPS.items()}
    # Two-layer split: the text face never gets the raw transcript or
    # the selection file; the IX face never gets the text pack.
    assert "transcript.jsonl" not in FILE_GROUPS["text_face"]
    assert "ix_selection.json" not in FILE_GROUPS["text_face"] + FILE_GROUPS["ix_face"]
    assert "text_judge_pack.json" not in FILE_GROUPS["ix_face"]


def test_pack_run_is_idempotent(probe_run: Path) -> None:
    _pack(probe_run)
    first = {p.name: p.read_bytes() for p in (probe_run / "judge_input" / PROBE_ID).iterdir()}
    _pack(probe_run)
    second = {p.name: p.read_bytes() for p in (probe_run / "judge_input" / PROBE_ID).iterdir()}
    assert first == second


def test_pack_run_skips_probes_that_have_not_run(probe_run: Path) -> None:
    plan = yaml.safe_load((probe_run / "run_plan.yaml").read_text(encoding="utf-8"))
    plan["steps"].append({**plan["steps"][-1], "step_id": "fin_03", "target_cell": "work_tone_formality"})
    (probe_run / "run_plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    index = json.loads(_pack(probe_run).read_text(encoding="utf-8"))
    assert [probe["probe_id"] for probe in index["probes"]] == [PROBE_ID]


def test_ix_selection_takes_only_selection_payloads_for_the_target(probe_run: Path) -> None:
    _pack(probe_run)
    selection = json.loads(
        (probe_run / "judge_input" / PROBE_ID / "ix_selection.json").read_text(encoding="utf-8")
    )
    assert selection["probe_id"] == PROBE_ID
    assert selection["target_attribute"] == "verbosity"
    assert selection["context"] == "work"
    assert selection["selected_setting"] == "Detailed"
    assert selection["selection_count"] == 1
    assert selection["no_selection"] is False
    (entry,) = selection["selections"]
    assert entry["beat"] == "beat1" and entry["call_index"] == 0
    assert set(entry) == {"beat", "call_index", "attribute", "setting", "evidence", "application"}


def test_ix_selection_filters_by_payload_type_before_taking() -> None:
    """A per-turn payload entry has an empty `setting`; without the
    `payload_type` filter it would be read as a selection of ""."""
    toolcalls = {
        "beats": [
            {
                "beat": "beat1",
                "calls": [
                    {"call_index": 0, "name": "IX_render_verbosity", "type": "ix",
                     "payload_type": "ix_render_payload", "attribute": "verbosity", "setting": ""},
                    {"call_index": 1, "name": "beat_gate_turn", "type": "ix", "ix_event": True},
                    {"call_index": 2, "name": "IX_tone_formality", "type": "ix",
                     "payload_type": IX_SELECTION_PAYLOAD_TYPE, "attribute": "tone_formality",
                     "setting": "Formal"},
                ],
            }
        ]
    }
    selection = build_ix_selection(toolcalls, probe_id="fin_02", target_attribute="verbosity", context="work")
    assert selection["selections"] == []
    assert selection["selected_setting"] is None
    assert selection["no_selection"] is True


def test_two_different_selections_for_one_attribute_is_an_error() -> None:
    call = {"call_index": 0, "name": "IX_verbosity", "type": "ix",
            "payload_type": IX_SELECTION_PAYLOAD_TYPE, "attribute": "verbosity"}
    toolcalls = {"beats": [{"beat": "b", "calls": [
        {**call, "setting": "Terse"}, {**call, "call_index": 1, "setting": "Detailed"},
    ]}]}
    with pytest.raises(ValueError, match="more than once"):
        build_ix_selection(toolcalls, probe_id="fin_02", target_attribute="verbosity", context="work")


def test_ground_truth_reads_the_frozen_record_through_setting_for(probe_run: Path) -> None:
    _pack(probe_run)
    gt = json.loads((probe_run / "judge_input" / PROBE_ID / "ground_truth.json").read_text(encoding="utf-8"))
    assert gt == {
        "persona_id": "user_a", "context": "work", "attribute": "verbosity",
        "setting": "Detailed", "confidence": "High", "source": "extractor_high",
        "source_path": "data/personas/user_a_preferences.yaml",
        "allowed_settings": ["Terse", "Detailed"],
        "evolving": {"is_evolving": False, "pre_setting": None, "post_setting": None},
    }


@pytest.mark.parametrize(
    ("step_kind", "expected"),
    [("pre_event_probe", "Reactive"), ("final_probe", "Self-directed")],
)
def test_evolving_cell_scores_pre_against_pre_and_final_against_post(step_kind: str, expected: str) -> None:
    matrix = yaml.safe_load((REPO_ROOT / "data/personas/user_a_preferences.yaml").read_text(encoding="utf-8"))
    gt = build_ground_truth(
        RESOLVED_GT, matrix, persona_id="user_a", cell="work_autonomy_level",
        step_kind=step_kind, source_path="data/personas/user_a_preferences.yaml",
    )
    assert gt["setting"] == expected
    assert gt["evolving"] == {"is_evolving": True, "pre_setting": "Reactive", "post_setting": "Self-directed"}
    assert gt["allowed_settings"] == ["Reactive", "Self-directed", "Autonomous"]


def test_ground_truth_for_an_unfrozen_cell_is_an_error() -> None:
    with pytest.raises(ValueError, match="no record for cell"):
        build_ground_truth(
            RESOLVED_GT, {}, persona_id="user_a", cell="personal_verbosity",
            step_kind="final_probe", source_path="x",
        )


def test_scene_feasible_options_strip_markers_and_drop_unavailable(probe_run: Path) -> None:
    _pack(probe_run)
    scene = json.loads((probe_run / "judge_input" / PROBE_ID / "scene.json").read_text(encoding="utf-8"))

    options = scene["world_state"]["feasible_options"]
    assert [option["record_id"] for option in options] == ["p_licence_a", "p_licence_c"]
    for option in options:
        assert not {"unavailable", "boundary_kind", "unavailable_reason"} & set(option["record"])
    assert options[1]["record"] == {"product_id": "p_licence_c", "name": "Vendor perpetual licence", "price": 900.0}
    assert "site licence" not in json.dumps(scene)
    assert scene["world_state"]["pending_items"] == []


def test_scene_withheld_required_facts_come_from_the_outline_missing_fields(probe_run: Path) -> None:
    """The Structured coverage reference is the outline's authored list, not the runtime's."""
    entry = {**OUTLINE["probes"][0], "world": [
        *OUTLINE["probes"][0]["world"],
        {"role": "missing_field", "task_fact": "selected_role",
         "note": "Which role the user has decided to prepare."},
        {"role": "missing_field", "task_fact": "taping_day", "note": "Which day she records."},
    ]}
    scene = build_scene(
        probe_id=PROBE_ID, draft=_draft(), outline_entry=entry,
        world_root=probe_run / "steps" / PROBE_ID / "scene_start" / "world", events=[], tool_log=[],
    )
    assert scene["withheld_required_facts"] == [
        {"task_fact": "selected_role", "note": "Which role the user has decided to prepare."},
        {"task_fact": "taping_day", "note": "Which day she records."},
    ]


def test_scene_withholds_nothing_when_the_outline_has_no_missing_field(probe_run: Path) -> None:
    _pack(probe_run)
    scene = json.loads((probe_run / "judge_input" / PROBE_ID / "scene.json").read_text(encoding="utf-8"))
    assert scene["withheld_required_facts"] == []


def test_scene_core_task_triggers_and_familiarity_come_from_the_draft(probe_run: Path) -> None:
    _pack(probe_run)
    scene = json.loads((probe_run / "judge_input" / PROBE_ID / "scene.json").read_text(encoding="utf-8"))

    assert scene["probe_id"] == PROBE_ID
    assert scene["core_task"] == CORE_TASK
    # A question-shaped probe carries no trigger, so the trigger view is empty.
    assert scene["triggers"] == []
    assert scene["familiarity_signals"] is None


def test_scene_numbers_pa_turns_by_t(probe_run: Path) -> None:
    events = load_events(FIXTURE_STEP / "transcript.jsonl")
    turns = turn_index(reversed(events))
    assert [turn["turn"] for turn in turns] == list(range(1, 11))
    assert [turn["t"] for turn in turns] == sorted(turn["t"] for turn in turns)
    assert turns[0] == {"turn": 1, "t": 3, "beat": "beat1"}


def test_scene_tool_actions_classify_every_tool_log_row(probe_run: Path) -> None:
    _pack(probe_run)
    scene = json.loads((probe_run / "judge_input" / PROBE_ID / "scene.json").read_text(encoding="utf-8"))
    rows = load_events(FIXTURE_STEP / "state/tool_log.jsonl")
    assert len(scene["tool_actions"]) == len(rows) == 18
    assert scene["tool_actions"][0] == {"t": 1, "tool": "scholarly_paper_search", "action_type": "read"}
    assert tool_actions([{"t": 9, "tool": "email_send"}]) == [
        {"t": 9, "tool": "email_send", "action_type": "external_action"}
    ]


def test_outreach_fields_come_from_the_transcript_tool_events() -> None:
    events = [
        {"t": 4, "kind": "pa_turn", "beat": "beat1", "outbound": "…",
         "proactive_event": "deadline", "outreach_decision": "send", "outreach_delivered": True,
         "tool_events": [
             {"name": "mcp_state_email_read", "status": "ok"},
             {"name": "IX_proactive_outreach", "ix": True, "attribute": "proactive_outreach",
              "payload_type": IX_SELECTION_PAYLOAD_TYPE, "setting": "High",
              "outreach_action": "send", "outreach_message": "Heads up: the grant closes Friday."},
         ]},
        {"t": 7, "kind": "pa_turn", "beat": "beat2", "outbound": "…",
         "tool_events": [{"name": "IX_verbosity", "ix": True, "attribute": "verbosity", "setting": "Terse"}]},
    ]
    assert outreach_events(events) == [{
        "t": 4, "beat": "beat1", "outreach_decision": "send", "outreach_delivered": True,
        "outreach_action": "send", "outreach_message": "Heads up: the grant closes Friday.",
    }]


def test_scene_requires_exactly_one_active_beat(probe_run: Path) -> None:
    draft = _draft()
    draft["beats"][0]["active_skills"] = ["verbosity"]
    with pytest.raises(ValueError, match="exactly one active beat"):
        build_scene(
            probe_id=PROBE_ID, draft=draft, outline_entry=OUTLINE["probes"][0],
            world_root=probe_run / "steps" / PROBE_ID / "scene_start" / "world", events=[], tool_log=[],
        )


def test_a_triggered_probe_has_two_active_beats_and_the_arrival_as_core_task(probe_run: Path) -> None:
    """The two openings are both active and the PA acted on the arrival,
    so `core_task` is the trigger's detail, not any user question."""
    world = probe_run / "steps" / PROBE_ID / "scene_start" / "world"
    entry = {**OUTLINE["probes"][0], "cell": "work_proactive_outreach", "world": []}
    scene = build_scene(
        probe_id="fin_09", draft=_triggered_draft(), outline_entry=entry,
        world_root=world, events=[], tool_log=[],
    )
    assert scene["core_task"] == TRIGGER["detail"]
    assert scene["triggers"] == [TRIGGER]
    assert scene["familiarity_signals"] is None

    one_opening = _triggered_draft()
    one_opening["beats"][1]["active_skills"] = []
    with pytest.raises(ValueError, match="both are active beats; found 1"):
        build_scene(
            probe_id="fin_09", draft=one_opening, outline_entry=entry,
            world_root=world, events=[], tool_log=[],
        )


def test_an_ingredient_missing_from_the_scene_start_world_is_an_error(probe_run: Path) -> None:
    entry = {**OUTLINE["probes"][0], "world": [
        {"ref": "ghost", "role": "candidate", "store": "shopping/products.json", "record_id": "p_none"},
    ]}
    with pytest.raises(ValueError, match="no record for ingredient 'ghost'"):
        build_scene(
            probe_id=PROBE_ID, draft=_draft(), outline_entry=entry,
            world_root=probe_run / "steps" / PROBE_ID / "scene_start" / "world", events=[], tool_log=[],
        )


def test_meta_json_carries_the_two_ix_version_stamps(probe_run: Path) -> None:
    _pack(probe_run)
    meta = json.loads((probe_run / "judge_input" / PROBE_ID / "meta.json").read_text(encoding="utf-8"))
    assert meta["ix_render_schema_version"] == "ix_render/1.1.0"
    assert meta["ix_contract_version"] == "ix-wrapper/2.0.0"
    assert meta["kind"] == "final_probe"
    with pytest.raises(ValueError, match="ix_contract_version"):
        build_meta({"step_id": PROBE_ID, "ix_render_schema_version": "ix_render/1.1.0"})


def test_a_leaking_text_judge_pack_is_refused(probe_run: Path) -> None:
    pack_path = probe_run / "steps" / PROBE_ID / "text_judge_pack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["turns"][0]["ix_render"] = {"trace": "verbosity=Detailed"}
    pack_path.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(ValueError, match="carries the IX face"):
        _pack(probe_run)
    with pytest.raises(ValueError):
        verify_text_judge_pack(pack)


def test_a_missing_step_artifact_is_an_error(probe_run: Path) -> None:
    (probe_run / "steps" / PROBE_ID / "text_judge_pack.json").unlink()
    with pytest.raises(FileNotFoundError, match="text_judge_pack.json"):
        _pack(probe_run)
