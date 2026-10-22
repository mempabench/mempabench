"""Pydantic data models for the simulator.

Schema reflects the 2026-04-22 design (see MemPABench_Simulator_Design.md):
- Persona is split into PersonaIdentity (file: {name}_identity.yaml) and
  PreferenceMatrix (file: {name}_{seasons}_preferences.yaml).
- PreferenceMatrix is a 2 x 14 grid (context x active IPaS attribute). Every cell
  names a setting and its evidence source; a cell is never empty.
- Sessions declare a `context` selecting which matrix column the simulator slices.
- Beats declare `active_skills` — only those attributes are injected as full skill
  specs at prompt-assembly time.
"""

from __future__ import annotations

import re
from datetime import datetime

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

if TYPE_CHECKING:
    from scripts.session_authoring_schema import NormalSessionDraft, ProbeSessionDraft


class YamlModelLoadError(ValueError):
    """Raised when a YAML-backed model cannot be loaded or validated."""


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_yaml_model(model_type: type[ModelT], path: str | Path) -> ModelT:
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise YamlModelLoadError(
            f"Failed to read YAML model: path={source_path} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise YamlModelLoadError(
            f"Invalid YAML syntax: path={source_path} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        preview = repr(data)
        if len(preview) > 160:
            preview = preview[:157] + "..."
        raise YamlModelLoadError(
            f"YAML root must be a mapping: path={source_path} "
            f"root_type={type(data).__name__} value={preview}"
        )
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise YamlModelLoadError(
            f"Invalid {model_type.__name__}: path={source_path} error={exc}"
        ) from exc


# Attributes whose sessions are COMPRESSED: exactly two beats, the probe and
# one reaction, and the session ends after the reaction's single exchange.
#
# A triggered session (`proactive_trigger`) compresses the same way with the
# probe somewhere else: the probe is the PA's own trigger turn, and the two
# beats are the two entry-conditioned openings, of which `_beats_after_outreach`
# plays exactly one. The loop stops after that one, on either path.
#
# The source of truth is the COMPRESSED_ATTRIBUTES table in
# `scripts/session_authoring_schema.py`, which is what the authoring schema
# enforces (min = max = 2 beats). This copy exists because the simulator never
# imports `scripts/` at runtime; `tests/test_simulator/test_compressed_stop.py`
# asserts the two sets are equal.
COMPRESSED_ATTRIBUTES = frozenset(
    {
        "verbosity",
        "tone_formality",
        "emotional_engagement",
        "guidance_level",
        "solution_breadth",
        "uncertainty_expression",
        "task_expansion",
        "capability_boundary",
        "proactive_outreach",
    }
)


# ---------------------------------------------------------------------------
# Persona Identity
# ---------------------------------------------------------------------------

class SpeechStyle(BaseModel):
    """Linguistic style block — keeps the LLM sounding like this character."""
    tone: str = ""
    vocabulary_level: str = ""
    sentence_patterns: list[str] = Field(default_factory=list)
    catchphrases: list[str] = Field(default_factory=list)
    quirks: list[str] = Field(default_factory=list)


class Occupation(BaseModel):
    role: str | None = None
    field: str | None = None
    workplace: str | None = None
    workplace_type: str | None = None


class Living(BaseModel):
    place: str | None = None
    arrangement: str | None = None


class Relationship(BaseModel):
    name: str | None = None
    role: str | None = None
    closeness: str | None = None


class Life(BaseModel):
    """Who this person is in the world — the block anonymization rewrites."""
    occupation: Occupation = Field(default_factory=Occupation)
    living: Living = Field(default_factory=Living)
    relationships: list[Relationship] = Field(default_factory=list)
    routines: list[str] = Field(default_factory=list)
    hobbies: list[str] = Field(default_factory=list)


class Triggers(BaseModel):
    irritated_by: list[str] = Field(default_factory=list)
    pleased_by: list[str] = Field(default_factory=list)


class BigFiveDim(BaseModel):
    level: str | None = None
    evidence: list[str] = Field(default_factory=list)


class JungFn(BaseModel):
    fn: str
    level: str | None = None
    shows_as: str | None = None


class PersonalityModel(BaseModel):
    """Compressed prior — specific traits/reactions/speech items outrank it."""
    big_five: dict[str, BigFiveDim] = Field(default_factory=dict)
    jungian_stack: list[JungFn] = Field(default_factory=list)


class RegisterByAudience(BaseModel):
    """How the register shifts by who is listening — used when dictating."""
    close: str | None = None
    colleague: str | None = None
    institution: str | None = None
    stranger_or_service: str | None = None


class ReactionToOthers(BaseModel):
    """Reactions to what the PA just did. Unknown keys are typos, not data."""
    model_config = ConfigDict(extra="forbid")

    gets_it_wrong: str | None = None
    gets_it_right: str | None = None
    questions_them: str | None = None
    acts_without_asking: str | None = None


class ReactionToEvents(BaseModel):
    """Reactions to outside events the PA relays."""
    model_config = ConfigDict(extra="forbid")

    plan_disrupted: str | None = None
    bad_outcome: str | None = None
    demand_from_others: str | None = None
    conflict_with_third_party: str | None = None
    own_error_exposed: str | None = None


class Reaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to_others: ReactionToOthers = Field(default_factory=ReactionToOthers)
    to_events: ReactionToEvents = Field(default_factory=ReactionToEvents)


class KnowledgeBounds(BaseModel):
    expert_in: list[str] = Field(default_factory=list)
    weak_in: list[str] = Field(default_factory=list)


class PersonaIdentity(BaseModel):
    """Loaded from data/personas/{name}_identity.yaml.

    Identity-only: personality, speech style, act repertoire. NOT preferences —
    those live in PreferenceMatrix and are sliced per-session by context.
    """
    model_config = ConfigDict(extra="ignore")

    name: str
    source: str | None = ""  # may be null in anonymized personas (e.g. user_a)
    background: str = ""
    life: Life = Field(default_factory=Life)
    personality_traits: list[str] = Field(default_factory=list)
    triggers: Triggers = Field(default_factory=Triggers)
    personality_model: PersonalityModel = Field(default_factory=PersonalityModel)
    speech_style: SpeechStyle = Field(default_factory=SpeechStyle)
    register_by_audience: RegisterByAudience = Field(default_factory=RegisterByAudience)
    reaction: Reaction = Field(default_factory=Reaction)
    act_repertoire: list[str] = Field(default_factory=list)
    knowledge_bounds: KnowledgeBounds = Field(default_factory=KnowledgeBounds)
    sim_rules: list[str] = Field(default_factory=list)  # patched by /patch-simulator-prompt

    @classmethod
    def from_yaml(cls, path: str | Path) -> PersonaIdentity:
        return _load_yaml_model(cls, path)


# ---------------------------------------------------------------------------
# Preference Matrix (2 contexts x 14 attributes)
# ---------------------------------------------------------------------------

VALID_CONTEXTS = (
    "work",
    "personal",
)
ContextLiteral = Literal["work", "personal"]


#: Setting labels removed from the IPaS taxonomy, mapped to what replaced them.
#: A matrix cell that still names one is stale, not merely unusual, so loading
#: fails fast the way the `status` guard below does. Kept in step by hand with
#: `RETIRED_SETTINGS` in `scripts/outline_schema.py`: `simulator/` imports no
#: authoring module at load time.
#:
#: Keyed by `(attribute, setting)`, with `"*"` for a label retired everywhere.
#: `Suggest` belonged to `autonomy_level` alone, so a bare-label lookup was
#: enough; the other labels went one attribute at a time, and a label retired on
#: one attribute may still be live on another, so the attribute is part of the key.
RETIRED_SETTINGS: dict[tuple[str, str], str] = {
    ('*', 'Suggest'): (
        'autonomy_level merged it into `Self-directed` on 2026-09-05; '
        '`Self-directed` now carries the same gate'
    ),
    ('task_expansion', 'Medium'): (
        'task_expansion was cut to `Low` / `High` on 2026-09-06; the flag-only '
        'behaviour Medium named now belongs to `Low`, which permits naming an '
        'adjacent concern as long as no adjacent step is carried out'
    ),
    ('verbosity', 'Moderate'): (
        'verbosity was cut to `Terse` / `Detailed` on 2026-09-07; the one '
        'supporting move Moderate named now belongs to `Terse`, which allows at '
        'most one and no closing summary'
    ),
    ('solution_breadth', 'Medium'): (
        'solution_breadth was cut to `Low` / `High` on 2026-09-07; the '
        'shortlist with a named recommendation Medium described now belongs to '
        '`High`, which compares at least two options and may or may not '
        'recommend one'
    ),
    ('guidance_level', 'Calibrated'): (
        'guidance_level was cut to `Assumed` / `Guided` on 2026-09-07; the '
        'gloss of the term the user did not know now belongs to `Guided`, which '
        'glosses every term new to the reply and orders the steps'
    ),
}


def retired_setting_reason(attribute: str, setting: str) -> str | None:
    """Why `setting` is no longer a taxonomy setting for `attribute`, else None."""
    label = setting.strip()
    return RETIRED_SETTINGS.get((attribute, label)) or RETIRED_SETTINGS.get(("*", label))

# Provenance values a persona YAML may carry. Runtime patches may use other
# sources (e.g. `scripted_dynamic`); only file-loaded matrices are checked
# against this vocabulary.
PERSONA_CELL_SOURCES = frozenset({
    "extractor_high",
    "extractor_accepted",
    "mbti_accepted",
    "hitl_override",
    "seed",
})


class PreferenceCell(BaseModel):
    """One cell in the 2 x 14 runtime matrix.

    The only rule: the cell names a setting.
    """
    model_config = ConfigDict(extra="forbid")

    setting: str
    source: str
    confidence: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _named_setting(self) -> PreferenceCell:
        self.setting = self.setting.strip()
        if not self.setting:
            raise ValueError("setting must be a non-empty string")
        return self


class PreferencePatch(BaseModel):
    """Runtime override for one preference cell.

    A patch always names a concrete setting; nothing at runtime can empty a
    cell.
    """
    model_config = ConfigDict(extra="forbid")

    setting: str | None = None
    source: str = "dynamic_override"
    confidence: str | None = None
    notes: str | None = None


class PreferenceMatrix(BaseModel):
    """Loaded from data/personas/{name}_{tag}_preferences.yaml.

    `matrix` keys are the 2 IPaS contexts; inner keys are 14 attribute names.
    Counts by source are never stored here; `data/personas/coverage.py`
    computes them from the cells.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    character: str
    sources: dict[str, str | None] = Field(default_factory=dict)
    matrix: dict[str, dict[str, PreferenceCell]] = Field(default_factory=dict)
    merged_from: str | None = Field(default=None, alias="_merged_from")
    status: str | None = None

    @model_validator(mode="after")
    def _shape_and_sources(self) -> PreferenceMatrix:
        if self.status:
            raise ValueError(
                f"matrix is marked status={self.status!r} and must not be loaded: "
                "it holds settings removed from the IPaS taxonomy on 2026-09-05. "
                "Re-extract the affected cells under the current settings and drop "
                "the status key before using this persona."
            )
        contexts = set(self.matrix)
        if contexts != set(VALID_CONTEXTS):
            raise ValueError(
                f"matrix contexts must be exactly {sorted(VALID_CONTEXTS)}; got {sorted(contexts)}"
            )
        attribute_sets = {ctx: set(cells) for ctx, cells in self.matrix.items()}
        if len({frozenset(v) for v in attribute_sets.values()}) != 1:
            raise ValueError(
                "both contexts must define the same attributes; "
                f"work={sorted(attribute_sets['work'])} personal={sorted(attribute_sets['personal'])}"
            )
        for ctx, cells in self.matrix.items():
            for attribute, cell in cells.items():
                if cell.source not in PERSONA_CELL_SOURCES:
                    raise ValueError(
                        f"matrix[{ctx}][{attribute}].source={cell.source!r} is not a persona "
                        f"provenance value; allowed: {sorted(PERSONA_CELL_SOURCES)}"
                    )
                replacement = retired_setting_reason(attribute, cell.setting)
                if replacement is not None:
                    raise ValueError(
                        f"matrix[{ctx}][{attribute}].setting={cell.setting!r} is not a "
                        f"taxonomy setting any more: {replacement}. Re-extract the cell "
                        "before using this persona."
                    )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreferenceMatrix:
        return _load_yaml_model(cls, path)

    def get(self, context: str, attribute: str) -> PreferenceCell:
        return self.matrix[context][attribute]


# ---------------------------------------------------------------------------
# Beat + Session Script
# ---------------------------------------------------------------------------

class LifeContext(BaseModel):
    """Backdrop coloring the whole session."""
    what_happened: str = ""
    mood: str = ""
    time_pressure: str = ""
    energy: str = ""
    continuity_note: str = ""


class BeatBranch(BaseModel):
    """Variant of a beat triggered by a specific PA behavior pattern."""
    condition: str = ""
    goal: str = ""
    constraint: str = ""
    actor_instruction: str = ""
    language_style: str = ""
    avoid: str = ""


_NON_BRANCH_IF_FIELDS = frozenset({"if_not_met"})


class Beat(BaseModel):
    """One beat in a session script.

    YAML keys starting with `if_` are collected into `reactions` automatically.
    """
    model_config = ConfigDict(extra="ignore")

    beat: str  # the beat id (YAML uses `beat:` as the id key)
    dramatic_function: str = ""
    actor_instruction: str = ""
    language_style: str = ""
    goal: str = ""
    constraint: str = ""
    life_context_hint: str = ""
    entry_condition: str = ""
    # What the user does when what they were waiting for never happened.
    if_not_met: str = ""
    avoid: str = ""
    line: str = ""  # forced line — skip LLM if non-empty
    active_skills: list[str] = Field(default_factory=list)
    # `task_facts` keys this beat consumes without naming them in the
    # instruction — the draft's `uses`. They are required values all the same.
    uses: list[str] = Field(default_factory=list)
    active_skill_settings: dict[str, str] = Field(default_factory=dict)
    preference_updates: dict[str, dict[str, PreferencePatch]] = Field(default_factory=dict)
    elicitation: dict[str, Any] = Field(default_factory=dict)
    reactions: dict[str, BeatBranch] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _extract_if_branches(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        reactions: dict[str, Any] = dict(data.get("reactions") or {})
        for key in list(data.keys()):
            # `if_not_met` is a plain text field on the beat, not a branch.
            if key.startswith("if_") and key not in _NON_BRANCH_IF_FIELDS:
                reactions[key] = data.pop(key)
        if reactions:
            data["reactions"] = reactions
        return data


class ProactiveTrigger(BaseModel):
    """Environment event that may start a PA-initiated contact."""

    model_config = ConfigDict(extra="forbid")

    trigger_id: str
    activation_type: Literal[
        "scheduled_time", "deadline", "state_change", "external_event"
    ]
    background: str
    detail: str
    user_impact: str

    @model_validator(mode="after")
    def _validate_required_text(self) -> ProactiveTrigger:
        for field_name in ("trigger_id", "background", "detail", "user_impact"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"proactive_trigger.{field_name} must not be empty")
            setattr(self, field_name, value)
        return self


# A session with a proactive_trigger opens one of two ways, decided live by the
# PA: the contact arrived, or no contact came and the user returns on their own.
# Both openings are authored; the runtime plays exactly one, then the shared beats.
OPENING_AFTER_SEND = "opening_after_send"
OPENING_AFTER_SUPPRESS = "opening_after_suppress"
OPENING_BEATS = (OPENING_AFTER_SEND, OPENING_AFTER_SUPPRESS)


class SessionScript(BaseModel):
    """A full session script: meta + life_context + scenario + beats + task_facts."""
    model_config = ConfigDict(extra="ignore")

    session_id: str
    scenario: str = ""
    context: ContextLiteral
    task: str = ""
    # In-world date and time of this session. Set by the harness at runtime
    # (never authored in YAML) so the simulated user knows what day it is.
    session_clock: datetime | None = None
    life_context: LifeContext = Field(default_factory=LifeContext)
    beats: list[Beat] = Field(default_factory=list)
    proactive_trigger: ProactiveTrigger | None = None
    # Runtime-only routing marker. It is not rendered into the simulator prompt.
    runtime_contract_class: str | None = None
    # Runtime-only, set by SimulatorLoop, never authored: beat ids the loop
    # jumped over, and the one-shot note the landing beat's first trigger
    # carries. Both feed the "what you are here to do" prompt section.
    skipped_beats: list[str] = Field(default_factory=list)
    pending_jump_note: str = ""
    # Runtime-only, set by SimulatorLoop, never authored: the attributes this
    # session has actually asked under, in the order the loop entered them.
    # The record of what has already joined, which is what keeps a triggered
    # attribute from joining twice.
    session_reception_skills: list[str] = Field(default_factory=list)
    # Runtime-only, set by SimulatorLoop at every beat entry: the attributes
    # whose reception rules are loaded right now — the ones that joined during
    # the previous beat. The actor reacts once, on that one beat, and works
    # under the manner it got from there on. `None` on the prompt-only path,
    # which has no runtime history and falls back to the outline.
    session_reception_carry: list[str] | None = None
    # Runtime-only, set by SimulatorLoop at every beat entry, never authored:
    # the id of this compressed session's reaction beat when the loop is
    # standing on it, `None` otherwise. The loop is the only place that knows
    # which beat it is — an untriggered session's reaction is beat 2, a
    # triggered one's is the played opening at beat 1 — and the prompt and the
    # agent-side checks read the answer off here instead of re-deriving it.
    compressed_reaction_beat: str | None = None
    preference_overrides: dict[str, dict[str, PreferencePatch]] = Field(default_factory=dict)
    session_type: str = "free"
    # The draft's `meta.artifact_kind`: `normal_session` (the default, also
    # for forced events and hand-built scripts) or `probe_session`. Selects the
    # simulator prompt template (`simulator/prompting.py`).
    artifact_kind: str = "normal_session"
    # Set at load time from the draft's `meta.target_attribute` (see
    # `COMPRESSED_ATTRIBUTES`), never authored. A compressed session is exactly
    # two beats and the loop stops it after the second beat's one exchange —
    # see `simulator.loop.SimulatorLoop.run_session`.
    compressed: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)

    # Ground-truth facts the simulator (as character) knows. PA's outputs are checked
    # against these; violations get logged in <factual_check> per turn. PA never
    # sees this block — see Simulator Design § Access Boundary.
    task_facts: dict[str, Any] = Field(default_factory=dict)
    # Information gating: `task_facts` key -> the beat id at which the user
    # first reveals it. Before that beat the fact is withheld from the
    # simulator prompt entirely (see `simulator.prompting.gated_fact_keys`).
    reveal_at: dict[str, str] = Field(default_factory=dict)
    # Runtime-only, set by the harness from `harness.gate.resolve_session_gate`,
    # never authored in the draft YAML: this session's dictated_text gate as the
    # simulator needs to see it — the literal substrings the store keys the gate
    # on, the sentence the user dictates, and the beat that unlocks it. The store
    # passes a gated write only when its value carries every term, so every turn
    # that delivers the sentence has to carry them: live regress15 `acc_039` t55
    # restated the reason as "chest tightness ... three weeks" and a PA copying
    # the user's own words would have been refused on "chest tightening".
    gate_key_terms: list[str] = Field(default_factory=list)
    gate_dictated_text: str = ""
    gate_reveal_beat: str | None = None
    # Runtime-only in the same way, and part of the same gate: the outline's
    # `gate_cue`, the phrase the session's own prose gives the PA that makes
    # asking for the gated fact natural, and the beat whose goal or instruction
    # carries it. The cue is the actor's to say — live regress18 lost it on
    # three of the five gated sessions when a style retry shortened the turn
    # (`acc_009` t3, `acc_011` t11, `acc_039` t6), and a PA with no cue never
    # asks, so the gate is never reached as designed. `gate_cue_beat` is None
    # when the cue sits only in the scenario (live `acc_023`); then there is no
    # beat to check it on.
    gate_cue: str | None = None
    gate_cue_beat: str | None = None
    # Runtime-only, set by the harness, never authored: the catalog tool named
    # by the step this session hangs `authorization: required_before_step` on.
    # It is what the consent sideband's pre-authorisation entry names, so the
    # PA's external_action_floor does not ask again for an act the user handed
    # over on the turn before (live regress19 `acc_039` t40/t41). Independent
    # of the gate fields above and under no together-rule with them: a session
    # can gate a dictation without authorising anything, and the other way
    # round.
    authorised_tool: str | None = None
    # Beats that inspect the end state of a declared state change: the first
    # beat entered after the write the draft hangs a `readback_step_id` on.
    # Authoring-derived, never authored on the beat itself — see
    # `readback_beat_ids`. The loop holds them back from the
    # `goal_already_met` collapse (`simulator.loop._held_from_collapse`).
    readback_beats: list[str] = Field(default_factory=list)
    # Per readback beat, the (write tool, read tool) of the ONE state change it
    # inspects — the catalog tools its `atomic_step_id` and `readback_step_id`
    # steps name, `None` where that step names none. Authoring-derived like
    # `readback_beats`; see `readback_beat_tools`. The loop reads it to decide
    # whether the PA already ran THIS beat's readback
    # (`simulator.loop._readback_already_delivered`).
    readback_beat_tools: dict[str, tuple[str | None, str | None]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _validate_proactive_shape(self) -> SessionScript:
        if (
            self.runtime_contract_class == "triggered_interaction"
            and self.proactive_trigger is None
        ):
            raise ValueError(
                "triggered_interaction session requires proactive_trigger"
            )
        if (
            self.proactive_trigger is not None
            and self.runtime_contract_class != "triggered_interaction"
        ):
            raise ValueError(
                "proactive_trigger requires a triggered_interaction contract"
            )
        beat_ids = [beat.beat for beat in self.beats]
        if self.proactive_trigger is not None:
            if tuple(beat_ids[:2]) != OPENING_BEATS:
                raise ValueError(
                    "a session with proactive_trigger must open with the beats "
                    f"{OPENING_AFTER_SEND!r} then {OPENING_AFTER_SUPPRESS!r}"
                )
        elif set(beat_ids) & set(OPENING_BEATS):
            raise ValueError(
                f"{OPENING_AFTER_SEND!r} / {OPENING_AFTER_SUPPRESS!r} beats belong "
                "to sessions with proactive_trigger only"
            )
        return self

    @model_validator(mode="after")
    def _validate_gate_key_terms(self) -> SessionScript:
        """The gate fields arrive together, and name a beat of this session.

        Half a gate is worse than none: terms with no sentence to watch for arm
        the dictation check nowhere, and a sentence with no terms arms it
        against nothing. A reveal beat the session does not have is the same
        hole one step later — the check would never fire and the run would
        still look armed.
        """
        armed = (
            bool(self.gate_key_terms),
            bool(self.gate_dictated_text),
            bool(self.gate_reveal_beat),
            bool(self.gate_cue),
        )
        if any(armed) and not all(armed):
            raise ValueError(
                "gate_key_terms, gate_dictated_text, gate_reveal_beat and "
                "gate_cue are set together or not at all; a partial gate arms "
                "nothing"
            )
        beat_ids = {beat.beat for beat in self.beats}
        if self.gate_reveal_beat and self.gate_reveal_beat not in beat_ids:
            raise ValueError(
                f"gate_reveal_beat {self.gate_reveal_beat!r} is not a beat of "
                f"session {self.session_id!r}"
            )
        # `gate_cue_beat` is the one optional half: a cue the session states
        # only in its scenario has no beat to be owed on.
        if self.gate_cue_beat:
            if not self.gate_cue:
                raise ValueError(
                    "gate_cue_beat requires gate_cue; a beat to watch with no "
                    "phrase to watch for arms nothing"
                )
            if self.gate_cue_beat not in beat_ids:
                raise ValueError(
                    f"gate_cue_beat {self.gate_cue_beat!r} is not a beat of "
                    f"session {self.session_id!r}"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> SessionScript:
        """Load a legacy raw SessionScript YAML.

        New canonical normal sessions must use ``from_authored_session_yaml`` so
        evaluation and authoring fields are removed by an allowlist conversion.
        """
        return _load_yaml_model(cls, path)

    @classmethod
    def from_legacy_yaml(cls, path: str | Path) -> SessionScript:
        return _load_yaml_model(cls, path)

    @classmethod
    def from_authored_session_yaml(cls, path: str | Path) -> SessionScript:
        """Load a draft of either `meta.artifact_kind` -- `normal_session` or
        `probe_session` -- through the one discriminated `SessionDraft`."""
        from scripts.session_authoring_schema import SessionDraft

        session = _load_yaml_model(SessionDraft, path).root
        return authored_session_to_session_script(session)


# A beat's `entry_condition` is the actor's view of what the PA has already
# done, so the beat right after a write opens with the write reported as
# finished. These are the past-tense forms that report it; `confirms` /
# `confirmed` are listed rather than `confirm\w*` on purpose — "asks for
# confirmation to place the order" is the *authorization* beat, one beat too
# early (live `acc_004` b5).
_STATE_CHANGE_DONE_RE = re.compile(
    r"\b(?:updat(?:ed|es)|complet(?:ed|ion)|executed|saved|created"
    r"|booked|submitted|applied|entered|confirms|confirmed|written"
    r"|merged|sent|added|placed|done)\b"
)
# Path noise that names no artefact: extensions and the generic container words
# every fixture path repeats.
_TARGET_PATH_NOISE = frozenset({"md", "json", "jsonl", "txt", "store", "file"})


def _target_words(target: str) -> set[str]:
    """The artefact words of a state change target, singular and plural."""
    words: set[str] = set()
    for part in re.split(r"[/_.\-\s]+", target.lower()):
        if not part or part in _TARGET_PATH_NOISE:
            continue
        words.add(part)
        words.add(part[:-1] if part.endswith("s") else f"{part}s")
    return words


def readback_beat_ids(session: NormalSessionDraft | ProbeSessionDraft) -> list[str]:
    """Beat ids that inspect the end state of a declared state change.

    Every `state_changes` row carries a `readback_step_id` — the read that
    proves the write landed — but nothing links a step to a beat, so the beat
    is found from the beats themselves: the first beat after the write whose
    `entry_condition` reports that change's target as done. That is the beat
    the user spends on the artefact, and collapsing it hands the check to the
    PA's own claim about its own write: live `acc_001` beat_06 ("see how the
    converted entry now sits in the bibliography notes") collapsed on a PA
    placement assertion that was false, and `acc_019` beat_6 went the same way.

    One beat per state change, in step order — a later change may not reuse an
    earlier change's beat, so a session's writes hold the beats that follow
    them and nothing before.

    A COMPRESSED session has no such beat at all. Owner ruling E (2026-09-07):
    the write and its readback both happen inside beat 1 and the PA shows the
    result in its beat-1 reply, so beat 2 is pure reaction and never asks for
    the product, to see it, or for a readback. Beat 2 is nonetheless this
    scan's first candidate, and its `entry_condition` says what the PA did —
    "the PA has updated the note and shown what it now says" carries both a
    done-verb and the target's words — so without this guard the loop would
    mark it `is_readback`, force `require_ask`, and push the actor to ask to be
    shown an artefact it was already shown.
    """
    return [beat_id for beat_id, _ in _readback_beat_changes(session)]


def _readback_beat_changes(
    session: NormalSessionDraft | ProbeSessionDraft,
) -> list[tuple[str, Any]]:
    """Each readback beat with the `state_changes` row it inspects.

    The pairing `readback_beat_ids` describes, kept in one place because the
    loop needs the change as well as the beat: which record the beat asks to be
    shown decides which write and which read count as having shown it.
    """
    if session.meta.target_attribute in COMPRESSED_ATTRIBUTES:
        return []
    held: list[tuple[str, Any]] = []
    first_candidate = 1  # never the opening beat: nothing has happened yet
    for change in session.task_execution.state_changes:
        words = _target_words(change.target)
        for index in range(first_candidate, len(session.beats)):
            beat = session.beats[index]
            condition = beat.entry_condition.casefold()
            if not _STATE_CHANGE_DONE_RE.search(condition):
                continue
            if not set(re.findall(r"[a-z0-9]+", condition)) & words:
                continue
            held.append((beat.beat, change))
            first_candidate = index + 1
            break
    return held


def readback_beat_tools(
    session: NormalSessionDraft | ProbeSessionDraft,
) -> dict[str, tuple[str | None, str | None]]:
    """Per readback beat, the (write tool, read tool) of its state change.

    The change names both steps outright — `atomic_step_id` is the write,
    `readback_step_id` is the read that proves it landed — and an
    `actor: tool` step names its catalog tool, so the beat's own pair is a
    lookup. A step with no tool (`actor: pa`) yields `None` on that side.

    Without this the loop's "the PA already read it back" switch keyed on any
    write followed by any read anywhere in the session, so live regress21r
    `acc_039` t=20 took the t=15 calendar pair as proof that the clinic
    appointment `beat_6` asks for had been shown — while its
    `appointment_create` had errored and no such record existed.
    """
    tools = {
        step.step_id: step.tool for step in session.task_execution.atomic_steps
    }
    return {
        beat_id: (
            tools.get(change.atomic_step_id),
            tools.get(change.readback_step_id),
        )
        for beat_id, change in _readback_beat_changes(session)
    }


def authored_session_to_session_script(
    session: NormalSessionDraft | ProbeSessionDraft,
) -> SessionScript:
    """Render only the simulator-visible fields of an authored session.

    Both `meta.artifact_kind` branches go through here. A `probe_session`
    draft has no `session_type`, `contract_class` or `target_setting`: the
    probe's active beat declares the attribute, and the setting is not in the
    draft (the GT lives in `resolved_gt.yaml`), so the simulator plays the
    persona's own setting for it. A `proactive_outreach` probe does carry a
    `proactive_trigger` (design §2.1.2b): its beats are the two openings, so it
    runs as `triggered_interaction` and `compressed`, exactly as the
    accumulation side's triggered session does.

    `active_skills` marking: when ANY authored beat declares a non-empty
    `active_skills`, the per-beat values are used verbatim (the other beats get
    an empty list). Older drafts declare none; those keep the previous
    behaviour, where the target attribute is active on every beat.
    `active_skill_settings` follows the same rule.
    """

    from scripts.session_authoring_schema import (
        NormalSessionDraft,
        ProbeSessionDraft,
        SessionDraft,
    )

    if not isinstance(session, (NormalSessionDraft, ProbeSessionDraft)):
        session = SessionDraft.model_validate(session).root
    is_probe = isinstance(session, ProbeSessionDraft)

    proactive = None
    if session.proactive_trigger is not None:
        trigger_data = session.proactive_trigger.model_dump()
        # Normal sessions no longer author a trigger type; the runtime field
        # (which the PA receives as source_type) defaults to external_event.
        if trigger_data.get("activation_type") is None:
            trigger_data["activation_type"] = "external_event"
        proactive = ProactiveTrigger.model_validate(trigger_data)

    per_beat_marking = any(item.active_skills for item in session.beats)
    target_settings = (
        {session.meta.target_attribute: session.meta.target_setting}
        if not is_probe and session.meta.target_setting is not None
        else {}
    )

    beats: list[Beat] = []
    for item in session.beats:
        reactions = {
            reaction_id: BeatBranch(
                condition=reaction.condition,
                actor_instruction=reaction.user_instruction,
                language_style=reaction.language_style,
                avoid=reaction.avoid,
            )
            for reaction_id, reaction in item.reactions.items()
        }
        if per_beat_marking:
            active_skills = list(item.active_skills)
        else:
            # The target attribute is always active: it is the PA-side IX
            # cue, and a control session must look like any other. With no
            # setting (control) there is no override; the simulator falls
            # back to the persona for that attribute.
            active_skills = [session.meta.target_attribute]
        beats.append(
            Beat(
                beat=item.beat,
                goal=item.goal,
                actor_instruction=item.user_instruction,
                language_style=item.language_style,
                entry_condition=item.entry_condition,
                if_not_met=item.if_not_met,
                avoid=item.avoid,
                active_skills=active_skills,
                uses=list(item.uses),
                active_skill_settings={
                    attribute: setting
                    for attribute, setting in target_settings.items()
                    if attribute in active_skills
                },
                reactions=reactions,
            )
        )

    return SessionScript(
        session_id=session.meta.session_id,
        scenario=session.scenario,
        context=session.meta.context,
        task=session.task,
        life_context=LifeContext.model_validate(session.life_context.model_dump()),
        beats=beats,
        proactive_trigger=proactive,
        # A probe declares no contract class; the triggered one is implied by its
        # trigger block, and the loop needs the class to play the trigger turn.
        runtime_contract_class=(
            ("triggered_interaction" if proactive is not None else None)
            if is_probe
            else session.meta.contract_class
        ),
        preference_overrides={},
        session_type=session.meta.probe_kind if is_probe else session.meta.session_type,
        artifact_kind=session.meta.artifact_kind,
        # A probe is 1-3 beats ending on its active beat, not the two-beat
        # compressed shape whose stop rule the loop keys on -- unless it is the
        # triggered one, which plays the trigger turn and then the single
        # opening that matched, and so stops like any compressed session.
        compressed=(
            proactive is not None
            if is_probe
            else session.meta.target_attribute in COMPRESSED_ATTRIBUTES
        ),
        meta={},
        task_facts=dict(session.task_facts),
        reveal_at=dict(session.reveal_at),
        readback_beats=readback_beat_ids(session),
        readback_beat_tools=readback_beat_tools(session),
    )
