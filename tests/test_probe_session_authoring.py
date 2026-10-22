"""The `probe_session` branch of `scripts/session_authoring_schema.py`
and the runtime loader that dispatches on the same discriminator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from scripts.session_authoring_schema import (
    NormalSessionDraft,
    ProbeSessionDraft,
    SessionDraft,
)
from simulator.schemas import SessionScript, authored_session_to_session_script

ROOT = Path(__file__).resolve().parents[1]
NORMAL_FIXTURE_PATH = ROOT / "tests/fixtures/authoring/normal_session_compressed.yaml"


def _step(step_id: str, **overrides: object) -> dict:
    step = {
        "step_id": step_id,
        "source_outline_phase_id": "probe",
        "purpose": f"{step_id} purpose",
        "actor": "user",
        "action_kind": "signal",
        "required_observation": f"{step_id} observed",
    }
    step.update(overrides)
    return step


def _beat(beat: str, active_skills: list[str] | None = None) -> dict:
    return {
        "beat": beat,
        "goal": f"{beat} goal",
        "user_instruction": f"{beat} instruction",
        "entry_condition": f"{beat} entry",
        "active_skills": active_skills or [],
    }


def _probe(kind: str = "final") -> dict:
    """The minimal fin_02 / pre_02 draft: one setup beat, one active beat."""
    meta = {
        "artifact_kind": "probe_session",
        "tool_status": "world_toolbox",
        "persona_id": "user_a",
        "probe_id": "fin_02",
        "session_id": "fin_02",
        "probe_kind": "final",
        "cell": "work_verbosity",
    }
    if kind == "pre_event":
        meta.update(
            probe_id="pre_02", session_id="pre_02", probe_kind="pre_event",
            pairs_with="fin_02", before_acc=30,
        )
    return {
        "meta": meta,
        "task_execution": {
            "atomic_steps": [
                _step("open_ask"),
                _step(
                    "read_policy",
                    actor="tool",
                    action_kind="read",
                    depends_on=["open_ask"],
                    tool="document_read",
                    tool_arguments={"path": "documents/admin/policy.md"},
                ),
            ]
        },
        "scenario": "The user is about to buy a software license.",
        "task": "Find out what the department needs before paying.",
        "life_context": {
            "what_happened": "x", "mood": "x", "time_pressure": "x",
            "energy": "x", "continuity_note": "x",
        },
        "beats": [_beat("setup"), _beat("ask", ["verbosity"])],
        "world": [{"role": "backing", "path": "documents/admin/policy.md"}],
    }


TRIGGER = {
    "trigger_id": "poster_print_deadline_email",
    "background": "The poster logistics were handed over and no request of the user's is open.",
    "detail": "An email from the conference secretariat sets the print deadline at Monday noon.",
    "user_impact": "The poster misses printing if the files are not sent in time.",
}


def _triggered_probe() -> dict:
    """The `proactive_outreach` shape: the record arrives, the PA decides,
    and the two openings are the beats -- both declaring the attribute."""
    data = _probe()
    data["meta"].update(probe_id="fin_09", session_id="fin_09", cell="work_proactive_outreach")
    data["task_execution"]["atomic_steps"] = [
        _step("env_trigger_arrives", actor="environment"),
        _step(
            "read_em_0105",
            actor="tool",
            action_kind="read",
            depends_on=["env_trigger_arrives"],
            tool="email_message_read",
            tool_arguments={"email_id": "$task_facts.records.em_0105"},
        ),
    ]
    data["beats"] = [
        _beat("opening_after_send", ["proactive_outreach"]),
        _beat("opening_after_suppress", ["proactive_outreach"]),
    ]
    data["world"] = [{"role": "backing", "store": "email/inbox.jsonl", "record_id": "em_0105"}]
    data["task_facts"] = {"records": {"em_0105": "em_0105"}}
    data["proactive_trigger"] = dict(TRIGGER)
    return data


def test_a_minimal_probe_draft_parses() -> None:
    draft = SessionDraft.model_validate(_probe()).root
    assert isinstance(draft, ProbeSessionDraft)
    assert draft.meta.context == "work"
    assert draft.meta.target_attribute == "verbosity"
    assert [(obj.role, obj.path) for obj in draft.world] == [("backing", "documents/admin/policy.md")]


def test_world_is_required_and_is_the_outline_world_object() -> None:
    data = _probe()
    del data["world"]
    with pytest.raises(ValidationError, match="world\n  Field required"):
        ProbeSessionDraft.model_validate(data)
    data = _probe()
    data["world"] = []
    with pytest.raises(ValidationError, match="at least 1 item"):
        ProbeSessionDraft.model_validate(data)
    data = _probe()
    data["world"] = [{"role": "gate", "task_fact": "note_text"}]
    with pytest.raises(ValidationError, match=r"missing \['gate_param', 'gate_tool'\]"):
        ProbeSessionDraft.model_validate(data)
    data["world"] = [{"role": "backing", "ref": "x", "path": "documents/admin/policy.md"}]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProbeSessionDraft.model_validate(data)


def test_a_pre_event_probe_carries_both_pairing_keys() -> None:
    draft = ProbeSessionDraft.model_validate(_probe("pre_event"))
    assert (draft.meta.pairs_with, draft.meta.before_acc) == ("fin_02", 30)
    for key in ("pairs_with", "before_acc"):
        data = _probe("pre_event")
        del data["meta"][key]
        with pytest.raises(ValidationError, match="pre_event probe carries both"):
            ProbeSessionDraft.model_validate(data)


def test_a_final_never_carries_before_acc() -> None:
    data = _probe()
    data["meta"]["before_acc"] = 30
    with pytest.raises(ValidationError, match="only on a pre_event probe"):
        ProbeSessionDraft.model_validate(data)


def test_a_final_pairs_with_only_its_pre_probe() -> None:
    data = _probe()
    data["meta"]["pairs_with"] = "pre_02"
    assert ProbeSessionDraft.model_validate(data).meta.pairs_with == "pre_02"
    data["meta"]["pairs_with"] = "fin_03"
    with pytest.raises(ValidationError, match="pairs with a pre_XX probe"):
        ProbeSessionDraft.model_validate(data)


def test_probe_id_prefix_follows_probe_kind() -> None:
    data = _probe()
    data["meta"]["probe_id"] = data["meta"]["session_id"] = "pre_02"
    with pytest.raises(ValidationError, match="takes a fin_XX probe_id"):
        ProbeSessionDraft.model_validate(data)
    data = _probe()
    data["meta"]["session_id"] = "fin_03"
    with pytest.raises(ValidationError, match="must equal probe_id"):
        ProbeSessionDraft.model_validate(data)


def test_assistance_emergence_is_not_a_probe_field() -> None:
    data = _probe()
    data["assistance_emergence"] = {
        "pattern": "explicit_request", "rationale": "x", "evidence_beat_ids": ["ask"],
    }
    with pytest.raises(ValidationError, match="assistance_emergence"):
        ProbeSessionDraft.model_validate(data)


def test_a_probe_is_one_to_three_beats() -> None:
    data = _probe()
    data["beats"] = [_beat("ask", ["verbosity"])]
    ProbeSessionDraft.model_validate(data)
    data["beats"] = [_beat("s1"), _beat("s2"), _beat("ask", ["verbosity"])]
    ProbeSessionDraft.model_validate(data)
    data["beats"] = [_beat("s1"), _beat("s2"), _beat("s3"), _beat("ask", ["verbosity"])]
    with pytest.raises(ValidationError, match="at most 3 items"):
        ProbeSessionDraft.model_validate(data)
    data["beats"] = []
    with pytest.raises(ValidationError, match="at least 1 item"):
        ProbeSessionDraft.model_validate(data)


def test_setup_beats_declare_no_active_skills() -> None:
    data = _probe()
    data["beats"] = [_beat("setup", ["verbosity"]), _beat("ask", ["verbosity"])]
    with pytest.raises(ValidationError, match=r"setup beats declare no active_skills; \['setup'\]"):
        ProbeSessionDraft.model_validate(data)


def test_the_active_beat_is_last_and_declares_the_cell_attribute() -> None:
    data = _probe()
    data["beats"] = [_beat("ask", ["verbosity"]), _beat("after")]
    with pytest.raises(ValidationError, match=r"setup beats declare no active_skills; \['ask'\]"):
        ProbeSessionDraft.model_validate(data)
    data["beats"] = [_beat("setup"), _beat("after")]
    with pytest.raises(ValidationError, match="the last beat 'after' is the probe's one active beat"):
        ProbeSessionDraft.model_validate(data)
    data["beats"] = [_beat("ask", ["tone_formality"])]
    with pytest.raises(ValidationError, match=r"declares exactly \['verbosity'\]"):
        ProbeSessionDraft.model_validate(data)


def test_shared_skeleton_checks_apply_to_a_probe() -> None:
    data = _probe()
    data["task_execution"]["atomic_steps"][1]["depends_on"] = ["missing"]
    with pytest.raises(ValidationError, match="unknown dependencies"):
        ProbeSessionDraft.model_validate(data)
    data = _probe()
    data["reveal_at"] = {"amount": "ask"}
    with pytest.raises(ValidationError, match="names no key in task_facts"):
        ProbeSessionDraft.model_validate(data)


def test_a_triggered_probe_is_the_two_openings() -> None:
    draft = ProbeSessionDraft.model_validate(_triggered_probe())
    assert [beat.beat for beat in draft.beats] == ["opening_after_send", "opening_after_suppress"]
    assert draft.proactive_trigger is not None
    assert draft.proactive_trigger.trigger_id == TRIGGER["trigger_id"]

    data = _triggered_probe()
    data["beats"] = [_beat("setup"), *data["beats"]]
    with pytest.raises(ValidationError, match="exactly the beats 'opening_after_send'"):
        ProbeSessionDraft.model_validate(data)

    data = _triggered_probe()
    data["beats"] = list(reversed(data["beats"]))
    with pytest.raises(ValidationError, match="exactly the beats 'opening_after_send'"):
        ProbeSessionDraft.model_validate(data)


def test_both_openings_of_a_triggered_probe_declare_the_attribute() -> None:
    data = _triggered_probe()
    data["beats"][1]["active_skills"] = []
    with pytest.raises(
        ValidationError, match=r"declare exactly \['proactive_outreach'\].*opening_after_suppress"
    ):
        ProbeSessionDraft.model_validate(data)


def test_opening_beats_still_belong_to_a_triggered_probe_alone() -> None:
    data = _triggered_probe()
    del data["proactive_trigger"]
    with pytest.raises(ValidationError, match="belong to sessions with proactive_trigger only"):
        ProbeSessionDraft.model_validate(data)


def test_the_runtime_loader_reads_a_triggered_probe(tmp_path: Path) -> None:
    """The probe is the PA's own trigger turn, so the script runs as the
    accumulation side's triggered session does -- one exchange, then it stops."""
    path = tmp_path / "fin_09.yaml"
    path.write_text(yaml.safe_dump(_triggered_probe()), encoding="utf-8")
    script = SessionScript.from_authored_session_yaml(path)
    assert script.proactive_trigger is not None
    assert script.proactive_trigger.trigger_id == TRIGGER["trigger_id"]
    assert script.proactive_trigger.background == TRIGGER["background"]
    # No trigger type is authored anywhere any more; the runtime default stands.
    assert script.proactive_trigger.activation_type == "external_event"
    assert script.runtime_contract_class == "triggered_interaction"
    assert script.compressed is True
    assert [beat.active_skills for beat in script.beats] == [
        ["proactive_outreach"], ["proactive_outreach"]
    ]
    assert authored_session_to_session_script(_triggered_probe()) == script


def test_the_discriminator_routes_a_normal_draft_unchanged() -> None:
    data = yaml.safe_load(NORMAL_FIXTURE_PATH.read_text(encoding="utf-8"))
    draft = SessionDraft.model_validate(data).root
    assert isinstance(draft, NormalSessionDraft)
    assert draft == NormalSessionDraft.model_validate(deepcopy(data))
    data["meta"]["artifact_kind"] = "forced_event"
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        SessionDraft.model_validate(data)


def test_the_runtime_loader_reads_a_probe_draft(tmp_path: Path) -> None:
    path = tmp_path / "fin_02.yaml"
    path.write_text(yaml.safe_dump(_probe()), encoding="utf-8")
    script = SessionScript.from_authored_session_yaml(path)
    assert script.session_id == "fin_02"
    assert script.context == "work"
    assert script.session_type == "final"
    assert script.compressed is False
    assert script.runtime_contract_class is None
    assert script.proactive_trigger is None
    assert [beat.active_skills for beat in script.beats] == [[], ["verbosity"]]
    # No setting is authored on a probe, so none is pushed to the simulator.
    assert script.beats[1].active_skill_settings == {}
    assert authored_session_to_session_script(_probe()) == script
