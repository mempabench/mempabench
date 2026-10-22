"""identity schema v2: the six new PersonaIdentity blocks.

`PersonaIdentity` is `extra="ignore"`, so a field the schema forgot to declare is
dropped silently. Loading is therefore not evidence — every new field is asserted
non-empty one by one.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from simulator.prompting import PERSONALITY_MODEL_PRIOR, render_character_card
from simulator.schemas import PersonaIdentity, Reaction


PERSONA_DIR = Path("data/personas")

# Every block of the v2 yaml, filled in so each one can be asserted non-empty.
SAMPLE_IDENTITY = """
name: "User Z"
source: ~
background: "Rendered from life."

life:
  occupation:
    role: "theoretical physicist"
    field: "string theory"
    workplace: "Westlake Institute of Technology"
    workplace_type: "private research university"
  living:
    place: "Glenmont, CA"
    arrangement: "shares a two-bedroom apartment with a colleague"
  relationships:
    - name: "Marcus Chen"
      role: "roommate and co-worker"
      closeness: "closest, and the one he overrules most"
  routines:
    - "Thai food on Thursday"
  hobbies:
    - "trains"

personality_traits:
  - "Intellectually arrogant"

triggers:
  irritated_by:
    - "someone sitting in his spot"
  pleased_by:
    - "a correction delivered with the derivation attached"

personality_model:
  big_five:
    conscientiousness:
      level: "very_high"
      evidence: ["S01E01 Scene 1"]
    agreeableness:
      level: "low"
      evidence: ["S01E03 Scene 4"]
  jungian_stack:
    - fn: "Ti"
      level: "very_high"
      shows_as: "builds the rule first, then checks the case against it"
    - fn: "Si"
      level: "high"
      shows_as: "cites the precedent by date"

speech_style:
  tone: "condescending to matter-of-fact"
  vocabulary_level: "academic"
  sentence_patterns:
    - "premise, derivation, conclusion"
  catchphrases:
    - "a catchphrase"
  quirks:
    - "corrects others' grammar unprompted"

register_by_audience:
  close: "blunt, no preamble"
  colleague: "formal, cites the literature"
  institution: "stiffly correct, uses full titles"
  stranger_or_service: "transactional and exacting"

reaction:
  to_others:
    gets_it_wrong: "names the one defect, does not restate the attempt"
    gets_it_right: "drops the objection, moves to the next instruction"
    questions_them: "gets sharper, not longer"
    acts_without_asking: "demands to be consulted next time"
  to_events:
    plan_disrupted: "recites what the schedule was"
    bad_outcome: "reconstructs the causal chain out loud"
    demand_from_others: "converts it into a constraint problem"
    conflict_with_third_party: "escalates to the governing rule"
    own_error_exposed: "concedes narrowly, with a boundary condition"

act_repertoire:
  - "Launch unsolicited lectures"

knowledge_bounds:
  expert_in:
    - "theoretical physics"
  weak_in:
    - "reading emotional cues"

sim_rules: []
"""

# The line each new block opens with in the character card.
NEW_CARD_HEADERS = (
    "Life:",
    "Triggers:",
    "Personality model:",
    "Register by audience:",
    "How you react:",
    "What you know:",
)


@pytest.fixture
def sample(tmp_path: Path) -> PersonaIdentity:
    path = tmp_path / "user_z_identity.yaml"
    path.write_text(SAMPLE_IDENTITY, encoding="utf-8")
    return PersonaIdentity.from_yaml(path)


@pytest.mark.parametrize(
    "path", sorted(PERSONA_DIR.glob("*_identity.yaml")), ids=lambda p: p.stem
)
def test_existing_identity_files_still_load(path: Path) -> None:
    identity = PersonaIdentity.from_yaml(path)
    assert identity.name


def test_every_new_field_survives_loading(sample: PersonaIdentity) -> None:
    life = sample.life
    assert life.occupation.role == "theoretical physicist"
    assert life.occupation.field == "string theory"
    assert life.occupation.workplace == "Westlake Institute of Technology"
    assert life.occupation.workplace_type == "private research university"
    assert life.living.place == "Glenmont, CA"
    assert life.living.arrangement
    assert [r.name for r in life.relationships] == ["Marcus Chen"]
    assert life.relationships[0].role and life.relationships[0].closeness
    assert life.routines == ["Thai food on Thursday"]
    assert life.hobbies == ["trains"]

    assert sample.triggers.irritated_by and sample.triggers.pleased_by

    model = sample.personality_model
    assert set(model.big_five) == {"conscientiousness", "agreeableness"}
    assert model.big_five["conscientiousness"].level == "very_high"
    assert model.big_five["conscientiousness"].evidence == ["S01E01 Scene 1"]
    assert [f.fn for f in model.jungian_stack] == ["Ti", "Si"]
    assert model.jungian_stack[0].level and model.jungian_stack[0].shows_as

    register = sample.register_by_audience
    assert register.close and register.colleague
    assert register.institution and register.stranger_or_service

    to_others = sample.reaction.to_others
    assert to_others.gets_it_wrong and to_others.gets_it_right
    assert to_others.questions_them and to_others.acts_without_asking
    to_events = sample.reaction.to_events
    assert to_events.plan_disrupted and to_events.bad_outcome
    assert to_events.demand_from_others and to_events.conflict_with_third_party
    assert to_events.own_error_exposed

    assert sample.knowledge_bounds.expert_in == ["theoretical physics"]
    assert sample.knowledge_bounds.weak_in == ["reading emotional cues"]


def test_card_renders_every_new_block(sample: PersonaIdentity) -> None:
    card = render_character_card(sample)
    lines = card.splitlines()

    for header in NEW_CARD_HEADERS:
        assert header in lines, f"{header!r} missing from the card"
    assert PERSONALITY_MODEL_PRIOR in lines
    assert lines.index(PERSONALITY_MODEL_PRIOR) < lines.index("Personality model:")

    for excerpt in (
        "Westlake Institute of Technology",
        "Marcus Chen",
        "Thai food on Thursday",
        "someone sitting in his spot",
        "conscientiousness: very_high",
        "builds the rule first, then checks the case against it",
        "blunt, no preamble",
        "names the one defect, does not restate the attempt",
        "concedes narrowly, with a boundary condition",
        "reading emotional cues",
    ):
        assert excerpt in card


def test_current_personas_render_no_new_blocks() -> None:
    identity = PersonaIdentity.from_yaml(PERSONA_DIR / "user_a_identity.yaml")
    lines = render_character_card(identity).splitlines()

    for header in NEW_CARD_HEADERS:
        assert header not in lines
    assert PERSONALITY_MODEL_PRIOR not in lines


def test_reaction_rejects_an_unknown_key() -> None:
    with pytest.raises(ValidationError):
        Reaction.model_validate({"to_others": {"gets_it_wrongg": "typo"}})
