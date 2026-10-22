"""The probe session's simulator prompt template and its selection by kind.

Design §2.4.4 / §0-A3: probe sessions use `generate_message_probe.md`, a
sibling of `generate_message.md` that keeps the `<consent_decisions>` rules
verbatim and differs only in the probe rule (correct content, never manner,
never an earlier session).

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_probe_prompt_template.py
"""

from __future__ import annotations

# Read-only use of the judge pack's vocabulary (same pattern as
# `test_turn_budget.py` reading `pa.agent.ix.render_types`); no simulator
# runtime code imports it.
from harness.judge_pack import IX_SETTING_LABELS, scan_banned_strings
from simulator.agent.context import SimulatorContextBuilder
from simulator.profiles import load_simulator_profile
from simulator.prompting import (
    PROBE_PROMPT_TEMPLATE_PATH,
    PROMPT_TEMPLATE_PATH,
    assemble_prompt,
    prompt_template_path,
)
from simulator.schemas import (
    Beat,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

BASE = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
PROBE = PROBE_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _session(artifact_kind: str) -> SessionScript:
    return SessionScript(
        session_id="fin_02",
        context="work",
        task="Ask what the department needs before paying for the license.",
        artifact_kind=artifact_kind,
        task_facts={"policy": "documents/admin/software_reimbursement_policy.md"},
        beats=[
            Beat(
                beat="ask",
                active_skills=["verbosity"],
                actor_instruction="Ask what the department needs before paying.",
            )
        ],
    )


def _section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def test_template_is_chosen_by_artifact_kind() -> None:
    assert prompt_template_path(_session("normal_session")) == PROMPT_TEMPLATE_PATH
    assert prompt_template_path(_session("probe_session")) == PROBE_PROMPT_TEMPLATE_PATH
    assert prompt_template_path(SessionScript(session_id="acc_001", context="work")) == (
        PROMPT_TEMPLATE_PATH
    )


def test_assembled_system_prompt_follows_the_kind() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    probe_session = _session("probe_session")
    probe = assemble_prompt(identity, matrix, probe_session, probe_session.beats[0])
    assert "No earlier sessions exist." in probe[0]["content"]
    normal_session = _session("normal_session")
    normal = assemble_prompt(identity, matrix, normal_session, normal_session.beats[0])
    assert "No earlier sessions exist." not in normal[0]["content"]
    assert "React to the PA's actual move" in normal[0]["content"]


def test_agent_context_builder_follows_the_kind() -> None:
    profile = load_simulator_profile("simulator/profiles/user_a")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    builder = SimulatorContextBuilder(profile, matrix)
    probe_session = _session("probe_session")
    assert "No earlier sessions exist." in builder.build_system_prompt(
        probe_session, probe_session.beats[0]
    )
    normal_session = _session("normal_session")
    assert "No earlier sessions exist." not in builder.build_system_prompt(
        normal_session, normal_session.beats[0]
    )


def test_both_templates_carry_the_same_consent_and_output_blocks() -> None:
    """§2.4.4 hard requirement 1: the consent block must survive verbatim, or
    every IX gate release fails closed on probes."""
    consent_rules = ("**`<consent_decisions>`**", "## Response skeleton")
    assert _section(BASE, *consent_rules) == _section(PROBE, *consent_rules)
    skeleton_start = "## Response skeleton"
    assert BASE[BASE.index(skeleton_start):] == PROBE[PROBE.index(skeleton_start):]
    for text in (BASE, PROBE):
        assert "<consent_decisions>" in text
        assert "interaction_item_id" in text
        assert "decision: allow | deny | unclear" in text
        for placeholder in (
            "<<CHARACTER_CARD>>", "<<ACTIVE_SKILLS>>", "<<LIFE_CONTEXT>>",
            "<<SESSION_OUTLINE>>", "<<BEAT_BLOCK>>", "<<TASK_FACTS>>",
            "<<DIALOGUE_NATURALNESS>>",
        ):
            assert placeholder in text


def test_probe_template_states_the_probe_rule() -> None:
    assert "Correct facts, never manner." in PROBE
    assert "No earlier sessions exist." in PROBE
    assert "React to the PA's actual move" not in PROBE
    assert "say your piece on the assistant's manner" not in PROBE


def test_probe_template_names_no_ix_setting_label() -> None:
    """The judge pack's whole-token label scan, run over the template as an
    IX-shaped leaf. Only the consent block's own field name (control
    vocabulary the design mandates) may remain."""
    hits = scan_banned_strings({"setting": PROBE})
    labels = [hit for hit in hits if hit.split(": ", 1)[1] in IX_SETTING_LABELS]
    assert labels == []
    assert set(hits) <= {"setting: interaction_item_id"}
