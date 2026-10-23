"""Strict schema for compact Stage 2 MemPABench outlines."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from state_server.store import ID_PREFIXES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


#: Setting labels removed from the IPaS taxonomy, mapped to what replaced them.
#: An outline that still names one was authored against a taxonomy that no
#: longer exists, so it is refused on load rather than silently resolved against
#: the current settings table. `simulator/schemas.py` keeps the same table for
#: persona matrices (the two packages do not share modules).
#:
#: Keyed by `(attribute, setting)`, with `"*"` for a label retired everywhere.
#: `Suggest` belonged to `autonomy_level` alone, so a bare-label lookup was
#: enough; `Medium` and `Moderate` are retired per attribute, and other
#: attributes still use those labels, so the attribute has to be part of the key.
RETIRED_SETTINGS: dict[tuple[str, str], str] = {
    ("*", "Suggest"): (
        "autonomy_level merged it into `Self-directed`; "
        "`Self-directed` now carries the same gate"
    ),
    ("task_expansion", "Medium"): (
        "task_expansion was cut to `Low` / `High`; the flag-only "
        "behaviour Medium named now belongs to `Low`, which permits naming an "
        "adjacent concern as long as no adjacent step is carried out"
    ),
    ("verbosity", "Moderate"): (
        "verbosity was cut to `Terse` / `Detailed`; the one "
        "supporting move Moderate named now belongs to `Terse`, which allows at "
        "most one and no closing summary"
    ),
    ("solution_breadth", "Medium"): (
        "solution_breadth was cut to `Low` / `High`; the shortlist "
        "with a named recommendation Medium described now belongs to `High`, "
        "which compares at least two options and may or may not recommend one"
    ),
    ("guidance_level", "Calibrated"): (
        "guidance_level was cut to `Assumed` / `Guided`; the gloss "
        "of the term the user did not know now belongs to `Guided`, which "
        "glosses every term new to the reply and orders the steps"
    ),
}


#: The nine attributes whose session is authored compressed: the task is
#: completable in one PA reply after one user message, so the outline entry may
#: not describe a write-then-tell-someone chain and may not hold a fact back
#: past the probe phase. The other five attributes are authored uncompressed and
#: declare a layer-1 gate instead (`validate_outline._gate_errors`).
#:
#: `proactive_outreach` is the triggered member. Its
#: probe is the PA's own trigger turn, which the entry names as
#: `trigger_phase_id`, so the entry collapses to that one coarse phase and
#: `preference_phase_ids` is exactly it. The "hold a fact back" clause above
#: still reads on coarse phases, of which it has one; what the trigger message
#: says enters the user's knowledge at the draft layer, on the suppress branch.
#:
#: `scripts/session_authoring_schema.py` mirrors this table for the draft layer
#: (the outline and authoring packages do not share modules);
#: `tests/test_generate_outline.py` checks the two sets are equal.
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

#: The required tool parameters that carry a WRITE PAYLOAD — the prose the PA
#: composes rather than a value the world could supply. These are the ONLY
#: parameters a gate may sit in: the user
#: dictates the wording at the probe beat, so the task cannot finish until they
#: speak. `validate_normal_event._WRITE_PAYLOAD_KEYS` is this same object, so
#: the two layers cannot drift.
GATE_PAYLOAD_PARAMS = frozenset({"body", "content", "note", "notes", "purpose", "reason", "relation", "replacement", "subject", "title"})

#: Words a key term may not begin with. A term that opens on an article or a
#: determiner is bound to the user's sentence rather than to his information:
#: acc_039 refused "chest tightening when taking stairs" against a term
#: of "the stairs" — the PA had asked and said the same thing.
GATE_TERM_ARTICLES = frozenset({"the", "a", "an", "this", "that", "these", "those"})

#: One word of a key term: letters and digits, an apostrophe inside a word
#: allowed ("driver's"). No hyphens — welding two words into one token was the
#: cheapest way past the leak rule and produced terms no user says aloud
#: ("zip-ties", "stage-left", "white-tablecloth").
#: A digit-grouping comma or a decimal point INSIDE a number is the second
#: shape allowed: "1,200 dollars" and "2.5 hours" are how people say and write
#: amounts, so they are natural, not coined. Digits on both sides — "1," and
#: "a,b" stay refused.
_GATE_TERM_WORD_RE = re.compile(
    r"(?:[0-9A-Za-z]+(?:'[0-9A-Za-z]+)*|[0-9]+(?:[.,][0-9]+)+)\Z"
)


def gate_term_shape_error(term: str) -> str | None:
    """Why `term` is not a term shape the gate can hold, or None if it is.

    The gate is keyed on information, so a term is
    the information's content word as the user says it. It is
    also a word the user would really say — so ONE word, or TWO words that
    belong together in ordinary speech ("zip ties", "three weeks", "October
    23"), and never two words welded into one token. Hyphenating was the
    cheapest way past the leak rule and produced terms nobody says
    ("zip-ties", "stage-left", "white-tablecloth", "braiding-statistics"), so a
    hyphen or any other punctuation is `gate_key_term_coined`. Nothing is lost
    by writing the words apart: the store normalises punctuation to spaces
    anyway and matches a multi-word term as its words in order.

    Anything longer is a fragment of the sentence rather than the information in
    it, and the store matches a multi-word term as an ordered phrase, so it dies
    on any faithful rewording: acc_039 refused two payloads that both
    carried what the user said. Articles, verb and adjective inflections and
    filler cannot be detected as such, so the shape is the mechanical proxy —
    they do not survive being written as one content word.
    """
    words = term.split()
    if not all(_GATE_TERM_WORD_RE.match(word) for word in words):
        return (
            f"gate_key_term_coined: key term {term!r} carries a hyphen or other "
            "punctuation; a term is a word or short phrase the user would say "
            "aloud, letters, digits and word-internal apostrophes only. Write "
            "the words apart ('zip ties', not 'zip-ties') — the store matches a "
            "multi-word term as its words in order"
        )
    if words and words[0].casefold() in GATE_TERM_ARTICLES:
        return (
            f"key term {term!r} begins with an article or determiner; a term is "
            "the information, not the user's phrasing around it"
        )
    if len(words) <= 2:
        return None
    return (
        f"key term {term!r} is {len(words)} words; a term is one word, or two "
        "words that belong together in ordinary speech ('zip ties', 'three "
        "weeks', 'October 23'). A longer term is matched as an ordered phrase "
        "and dies on any faithful rewording"
    )


def retired_setting_reason(attribute: str, setting: str) -> str | None:
    """Why `setting` is no longer a taxonomy setting for `attribute`, else None."""
    label = setting.strip()
    return RETIRED_SETTINGS.get((attribute, label)) or RETIRED_SETTINGS.get(("*", label))


def _reject_retired(label: str, attribute: str, setting: str) -> None:
    replacement = retired_setting_reason(attribute, setting)
    if replacement is not None:
        raise ValueError(
            f"{label}={setting!r} is not a taxonomy setting any more: {replacement}. "
            "Re-author this entry against the current settings."
        )


class OutlineMeta(StrictModel):
    persona: str = Field(min_length=1)
    range: str = Field(pattern=r"^acc_\d{3}-\d{3}$")
    model: str = Field(min_length=1)
    temperature: float
    seed: int | None
    active_session_count: int = Field(ge=1)
    status: Literal["in_progress", "complete"] = "complete"
    pending_acc_nums: list[int] = Field(default_factory=list)


class EvolvingArc(StrictModel):
    target_cell: str = Field(pattern=r"^(work|personal)_[a-z0-9_]+$")
    pre_setting: str = Field(min_length=1)
    post_setting: str = Field(min_length=1)
    # Event causality belongs to the generated evolving_event session, not to a
    # second model-authored map.  These legacy notes remain readable but are not
    # required by the current Outline contract.
    event_logic: str | None = None
    preference_change_pressure: str | None = None

    @model_validator(mode="after")
    def _settings_are_current(self) -> "EvolvingArc":
        # `target_cell` is `<context>_<attribute>`; the guard is keyed by attribute.
        attribute = self.target_cell.split("_", 1)[1]
        _reject_retired(
            f"evolving_map.{self.target_cell}.pre_setting", attribute, self.pre_setting
        )
        _reject_retired(
            f"evolving_map.{self.target_cell}.post_setting", attribute, self.post_setting
        )
        return self


class OperationalSource(StrictModel):
    attribute: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    # 1 (normal) or 2 (forced pre/post) settings.
    settings: list[str] = Field(min_length=1, max_length=2)
    contract_class: Literal[
        "output_realization",
        "interaction_sequence",
        "action_scope",
        "triggered_interaction",
    ]
    operational_heading: str = Field(min_length=1)

    @model_validator(mode="after")
    def _settings_are_current(self) -> "OperationalSource":
        for setting in self.settings:
            _reject_retired(
                f"operational_source[{self.attribute}].settings", self.attribute, setting
            )
        return self


# Normal openings are allocated from the full Timeline before OpenRouter writes
# stories; the model may omit the candidate field and the generator stamps the
# fixed value. Forced openings are selected by the OpenRouter Outline generator
# from evolving-story causality.
OpeningPattern = Literal[
    "explicit_request",
    "contextual_signal",
    "partial_intent",
    "ongoing_task_bottleneck",
    "result_driven_next_step",
    "prior_state_change_or_repair",
]


class Activation(StrictModel):
    """External trigger for an evolving_event (forced) entry."""

    type: Literal[
        "scheduled_time",
        "deadline",
        "state_change",
        "external_event",
    ]
    detail: str = Field(min_length=1)


class CoarseTaskPhase(StrictModel):
    phase_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    purpose: str = Field(min_length=1)


class SessionIdentity(StrictModel):
    acc_num: int = Field(ge=1)
    session_type: Literal["stable", "evolving_pre", "evolving_event", "evolving_post"]
    timeline_slot: str = Field(min_length=1)
    target_cell: str = Field(pattern=r"^(work|personal)_[a-z0-9_]+$")


class SessionStory(StrictModel):
    domain: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    story_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    # The World Design errand this session was dealt (assign_errands.py); the model
    # may omit it and the generator stamps it, the validator requires it.
    errand_id: str = Field(pattern=r"^[wp][1-6]_[a-z0-9_]+$")
    time_position: str = Field(min_length=1)
    # The frozen in-world clock for this session: the day it happens and the
    # hour the conversation starts. Must fall inside the time_position window
    # and increase with acc_num (see CompactOutline).
    date: str = Field(pattern=r"^20\d{2}-\d{2}-\d{2}$")
    hour: int = Field(ge=0, le=23)
    # Fixture files this session's task actually touches, fixture-relative; empty when
    # the errand needs no document. The session generator loads exactly these bodies.
    documents: list[str] = Field(default_factory=list)
    # Terms and titles the task text quotes, each copied verbatim from a bound document
    # or a scholarly record.
    key_terms: list[str] = Field(default_factory=list, max_length=5)


# The frozen world-toolbox domains. Kept in sync with
# data/state_tools/catalog.yaml by a test. Fine-grained by real-world context:
# where you book/submit/pay differs, the domain (and tool names) differ.
ToolboxDomain = Literal[
    "email",
    "message",
    "phone",
    "calendar",
    "documents",
    "contacts",
    "maps_places",
    "weather",
    "shopping",
    "shipping",
    "finance",
    "hotel",
    "appointment",
    "restaurant",
    "room",
    "flight",
    "scholarly",
    "recipe",
    "reimbursement",
    "conference_registration",
    "grade_submission",
    "comic_store",
    "cinema",
    "household",
    "exhibition",
    "media_transcode",
]


class SessionTask(StrictModel):
    user_goal: str = Field(min_length=1)
    completion_condition: str = Field(min_length=1)
    background: str = Field(min_length=1)
    # Candidates may omit this until the generator binds the fixed preallocated
    # mapping: normal openings from opening_assignment.yaml, evolving_event
    # (forced) ones from forced_opening_assignment.yaml.
    opening: OpeningPattern | None = None
    # Only world-triggered stories carry this: what concretely fires (the PA
    # proactive-outreach flow reads it). Must be non-empty when present.
    opening_detail: str | None = None
    # evolving_event (forced) entries keep the legacy block; normal sessions
    # must not have it.
    activation: Activation | None = None
    # Two or more, except for the triggered compressed entry, which is exactly
    # one. The floor itself lives on `OutlineSession._validate_phase_floor`,
    # which is the level that can see the attribute and the contract class.
    coarse_task_phases: list[CoarseTaskPhase] = Field(min_length=1)
    expected_domains: list[ToolboxDomain] = Field(min_length=1)
    # Set when the story needs an assistant capability not implemented by the
    # catalog, fixture set, or StateServer, even when its domain already exists.
    toolbox_gap: str | None = None

    @model_validator(mode="after")
    def _validate_opening_detail(self) -> "SessionTask":
        if self.opening_detail is not None and not self.opening_detail.strip():
            raise ValueError("opening_detail must be non-empty when present")
        return self


class ProbeBranch(StrictModel):
    behavior: str = Field(min_length=1)


class ProbeBranches(StrictModel):
    correct: ProbeBranch
    wrong: ProbeBranch


#: Record-id field names the world's stores mint themselves, read from the one
#: table `StateStore._mint_id` and `validate_l0_world.check_formats_and_ids` both
#: use. A `fixture_additions` record may not supply any of them: the patcher
#: mints the id so collisions are impossible by construction.
_MINTED_ID_FIELDS = frozenset({"id"} | {field for field, _ in ID_PREFIXES.values()})


#: Retained compatibility vocabulary for structural viability checks and old
#: reports. Exposure of one of these objects does not void a probe by role
#: alone: semantic review compares exact earlier bytes with the withheld
#: gate/trigger/time fact, and runtime uses its content-bound exposure audit.
PROBE_VOIDING_ROLES = frozenset({"candidate", "unresolved", "missing_field"})


def voids_probe(role: str) -> bool:
    """Legacy structural-role classification; never sufficient exposure proof."""
    return role in PROBE_VOIDING_ROLES


class IngredientObject(StrictModel):
    """One world object a probe beat depends on ("probe ingredient").

    A probe can be schema-valid, gated, and on-definition and still be dead: if
    the world holds nothing for the setting to act on, every setting of the
    target attribute produces the same reply and nothing is measured. This names
    the objects that keep it alive, so both the fixture sweep and the draft
    validator can check them.
    """

    # Author-chosen, unique inside the entry; a `fixture_additions` row links here.
    ref: str = Field(min_length=1)
    role: Literal[
        "backing", "unresolved", "capability_limit", "candidate", "missing_field", "gate"
    ]
    # Fixture store path relative to the persona fixture root, e.g. `calendar/events.json`.
    # `None` on a `missing_field` only, and required there: an elicited value
    # lives with the user, so naming a store makes the probe unsatisfiable --
    # the draft layer then wants the fact both held by the world and ungated.
    # `validate_outline._missing_field_store_errors` enforces both directions.
    store: str | None = Field(default=None, min_length=1)
    # Exactly one locator. `record_id` for a minted record, `path` for a document,
    # `task_fact` only for a fact no store holds: the key the opening omits (role
    # `missing_field`) or the value only the user holds (role `gate`). Every other
    # role, `unresolved` and `candidate` included, locates a real record with
    # `record_id` or `path` -- for `unresolved` the record whose contents fall short
    # of the boundary, for `candidate` the candidate record itself. A prose
    # `task_fact` there can never appear in a probe's tool arguments, so the draft
    # validator could never reach the ingredient.
    record_id: str | None = None
    path: str | None = None
    task_fact: str | None = None
    # Every optional is `| None` on purpose: `generate_outline._drop_nulls` is
    # shallow, so nested explicit nulls from the strict response schema survive
    # into the model and have to be accepted here.
    field: str | None = None
    reveal_at: str | None = None
    note: str | None = None
    # Role `candidate` on an `information_elicitation` cell only:
    # the `ref` of the `missing_field` this record is one of the world's answers for.
    # `Infer` is a viable setting only when the world really holds more than one value
    # the PA could pick, so each missing field names at least two of these; with one,
    # the PA reads the single value and `Infer` and `Structured` render the same turn.
    # `validate_outline._missing_field_candidate_errors` holds the count and the roles.
    missing_field_ref: str | None = None
    # Role `gate` only: the layer-1 gate of an uncompressed session. `gate_tool`
    # is the catalog tool the task cannot skip, `gate_param` one of that tool's
    # REQUIRED parameters, and `task_fact` the key whose value only the user
    # holds. Together they are the gate definition: a required parameter
    # of a tool call whose value is nowhere in the world.
    gate_tool: str | None = None
    gate_param: str | None = None
    # The only gate shape there is. Derived from `gate_param` by
    # `validate_outline._gate_errors` and bound after generation; the field is
    # kept because drafts read it.
    #
    # `required_value` is RETIRED. It let any required
    # parameter gate as long as its value was absent from fixture TEXT, which
    # missed that a value a read or listing tool can return is still in the
    # world and therefore guessable. acc_039 gated on `appointment_create.time`
    # and was defeated three live rounds running: one undated `appointment_search`
    # listed all nine slots, so the PA picked the gated one and booked without
    # ever asking. acc_004's gate was an address id the PA brute-forced. A gate
    # is now only ever text the user COMPOSES and says at the probe beat, which
    # no tool can hand over.
    gate_kind: Literal["dictated_text"] | None = None
    # The literal terms the store checks the PA's payload for.
    # The gate is keyed on INFORMATION, not on words: the store passes a payload
    # iff every one of these is present, so the PA may paraphrase everything
    # around them and still be refused until the user has spoken. This retired
    # the 0.7 content-token ratio, which measured the filler as well as the
    # substance -- acc_009's own wording scored 0.571 against itself once a
    # sentence was rearranged, while a PA that never asked scored 0.714 off the
    # framing prose.
    gate_key_terms: list[str] | None = None
    # The phrase the task's OPENING context gives that makes asking for the
    # gated information natural.
    # A gate nothing in the task motivates is not a
    # gate: acc_009 gated the review focus, but the opening only asked
    # for "a reply proposing review dates", so no PA — Structured or Infer —
    # would ever ask what the review should concentrate on, the actor had to
    # volunteer it unasked, and the two IX branches never separated. The cue is
    # the mechanical proxy for that motivation: `validate_normal_event` requires
    # it in the draft's text BEFORE the reveal beat.
    gate_cue: str | None = None
    # The slot the dictated value fills, named in the author's own words -- the
    # noun the user would use for it ("note", "reason for the visit", "the
    # heading"). Until this field the judge's `withheld_fact_before_ask` had to
    # INFER the slot from the dictated sentence and got it wrong: `acc_003`
    # authored a note and the judge read it as a reason, so a PA that asked for
    # exactly the right thing was scored as if it had not asked at all. Presence
    # is `validate_outline._gate_errors`' `gate_slot_required`.
    gate_slot: str | None = None

    @model_validator(mode="after")
    def _gate_fields_belong_to_a_gate(self) -> "IngredientObject":
        if self.role == "gate":
            missing = sorted(
                name
                for name in ("gate_tool", "gate_param", "task_fact")
                if not (getattr(self, name) or "").strip()
            )
            if missing:
                raise ValueError(
                    f"ingredient {self.ref!r}: a gate names gate_tool, gate_param and "
                    f"the task_fact key the user supplies; missing {missing}"
                )
        else:
            supplied = sorted(
                name
                for name in (
                    "gate_tool",
                    "gate_param",
                    "gate_kind",
                    "gate_key_terms",
                    "gate_cue",
                    "gate_slot",
                )
                if getattr(self, name) is not None
            )
            if supplied:
                raise ValueError(
                    f"ingredient {self.ref!r}: {supplied} belong to role 'gate', "
                    f"not to role {self.role!r}"
                )
        return self

    @model_validator(mode="after")
    def _dictated_text_names_its_key_terms(self) -> "IngredientObject":
        """A `dictated_text` gate names the terms the store checks for.

        Required exactly where `gate_kind` is `dictated_text` and forbidden
        everywhere else, the same shape `_gate_fields_belong_to_a_gate` gives
        the other gate fields. There is no compat path: a
        dictated_text gate without key terms is the retired word-ratio gate, so
        it is refused on load rather than resolved against a default.

        Two terms is the floor because one term is a single word the framing
        prose routinely already carries; three characters is the floor per term
        for the same reason `_search_tokens` drops shorter ones. Duplicates are
        refused case-insensitively: the store's check is case-insensitive, so a
        repeat is a term that adds nothing while reading as if it did. And each
        term has to be a shape the information survives in
        (`gate_term_shape_error`).
        """
        if self.gate_kind != "dictated_text":
            if self.gate_key_terms is not None:
                raise ValueError(
                    f"ingredient {self.ref!r}: gate_key_terms belongs to a "
                    f"dictated_text gate, not to gate_kind {self.gate_kind!r}"
                )
            return self
        if self.gate_key_terms is None:
            raise ValueError(
                f"ingredient {self.ref!r}: gate_key_terms_missing — a dictated_text "
                "gate names the literal terms the store checks the payload for; "
                "without them nothing distinguishes the user's information from a "
                "paraphrase the assistant invented"
            )
        terms = self.gate_key_terms
        stripped = [term.strip() for term in terms]
        if len(terms) < 2:
            raise ValueError(
                f"ingredient {self.ref!r}: gate_key_terms carries {len(terms)} term(s); "
                "a gate names at least two, because one term is routinely already in "
                "the prose the assistant is handed"
            )
        blank = [index for index, term in enumerate(stripped) if not term]
        if blank:
            raise ValueError(
                f"ingredient {self.ref!r}: gate_key_terms entries {blank} are empty "
                "once stripped; every term is text the store looks for"
            )
        short = sorted({term for term in stripped if 0 < len(term) < 3})
        if short:
            raise ValueError(
                f"ingredient {self.ref!r}: gate_key_terms carries {short}, shorter "
                "than the 3 characters a term needs to name anything"
            )
        folded = [term.casefold() for term in stripped]
        duplicates = sorted({term for term in folded if folded.count(term) > 1})
        if duplicates:
            raise ValueError(
                f"ingredient {self.ref!r}: gate_key_terms repeats {duplicates}; the "
                "store's check is case-insensitive, so a repeated term adds nothing"
            )
        shape = [
            reason
            for reason in (gate_term_shape_error(term) for term in stripped)
            if reason
        ]
        if shape:
            raise ValueError(f"ingredient {self.ref!r}: " + "; ".join(shape))
        return self

    @model_validator(mode="after")
    def _dictated_text_names_its_cue(self) -> "IngredientObject":
        """A `dictated_text` gate names the cue that makes the ask natural.

        Required exactly where `gate_kind` is `dictated_text` and forbidden
        everywhere else, the same shape the other gate fields have. Three words
        is the floor: a gate is motivated by a phrase the opening gives, and one
        or two words is a topic ("the reply", "the visit") rather than the thing
        the PA has to come back to the user for.

        Whether the draft's opening really carries the cue, and whether the cue
        leaks a key term, are facts about the draft — `validate_normal_event`'s
        `gate_cue_absent_before_reveal` and `gate_cue_contains_key_term`.
        """
        if self.gate_kind != "dictated_text":
            if self.gate_cue is not None:
                raise ValueError(
                    f"ingredient {self.ref!r}: gate_cue belongs to a dictated_text "
                    f"gate, not to gate_kind {self.gate_kind!r}"
                )
            return self
        cue = (self.gate_cue or "").strip()
        if not cue:
            raise ValueError(
                f"ingredient {self.ref!r}: gate_cue_missing — a dictated_text gate "
                "names the phrase the task's opening gives that makes asking for "
                "the gated information natural; a gate nothing in the task "
                "motivates is a gate no PA would ever open"
            )
        if len(cue.split()) < 3:
            raise ValueError(
                f"ingredient {self.ref!r}: gate_cue {cue!r} is "
                f"{len(cue.split())} word(s); a cue is at least 3, because one or "
                "two words name the topic rather than what the user still holds"
            )
        return self

    @model_validator(mode="after")
    def _exactly_one_locator(self) -> "IngredientObject":
        locators = {
            "record_id": self.record_id,
            "path": self.path,
            "task_fact": self.task_fact,
        }
        present = sorted(name for name, value in locators.items() if value is not None)
        if len(present) != 1:
            raise ValueError(
                f"ingredient {self.ref!r} needs exactly one of record_id / path / "
                f"task_fact, got {present or 'none'}"
            )
        return self


class FixtureAddition(StrictModel):
    """An add-only record the world does not hold yet, written by the patcher."""

    # Must match an `ingredient_objects` entry in the same probe contract.
    ref: str = Field(min_length=1)
    store: str = Field(min_length=1)
    record: dict[str, Any]
    # Arrival stamp; must be on or before the probe session's date.
    visible_from: str | None = None

    @model_validator(mode="after")
    def _no_author_supplied_id(self) -> "FixtureAddition":
        supplied = sorted(set(self.record) & _MINTED_ID_FIELDS)
        if supplied:
            raise ValueError(
                f"fixture_addition {self.ref!r} supplies minted id field(s) "
                f"{supplied}; the patcher mints the id"
            )
        return self


class ProbeRecovery(StrictModel):
    """`capability_boundary` only: the route out of the block, and what it costs.

    The probe starts from a real `capability_limit`
    return, and the two settings separate on what the PA does in that same turn:
    "Suggest alternatives" states the limit and then makes at least one further
    call along this route; "Find and hand off" states the limit, names the route
    and hands control back. Neither arm can be written, and neither can be told
    apart, unless the entry says WHICH route is available and what taking it
    costs the user -- a route with no cost is one no assistant would pause over,
    and the two settings collapse onto the same reply.
    """

    #: A catalog tool, and never the one the world blocks: the recovery is a
    #: DIFFERENT call, not a retry of the call that just raised the limit. It MAY
    #: be an `external_write` tool -- one that reaches another person or spends
    #: money. The cost has to land in the recovery
    #: step itself, and a step that only reads a record costs her nothing, so the
    #: route is allowed to be the message that does. `validate_outline`
    #: (`_compressed_errors`) then lets the one phase named in
    #: `ActionScopeContract.recovery_phase_ids` carry that one outward call and
    #: holds the rest of the compressed shape exactly as it was.
    tool: str = Field(min_length=1)
    #: One sentence: the money, the wait, or the giving-up the route costs her.
    #: It belongs to the step that incurs it -- reading is not contacting,
    #: pricing is not paying -- so where the route runs through an outward tool,
    #: this sentence says what the call itself costs her.
    cost: str = Field(min_length=1)


class ProbeContractBase(StrictModel):
    why_both_viable: str = Field(min_length=1)
    preference_phase_ids: list[str] = Field(min_length=1)
    branches: ProbeBranches
    # World objects this probe depends on, and the records to add when the world
    # does not hold them. REQUIRED: every
    # attribute's probe is measured over something the world holds, so an entry
    # that declares nothing is an entry whose settings render the same reply.
    # There is no compat path — an entry authored before the field existed does
    # not load, it is re-authored. `validate_outline._INGREDIENT_REQUIREMENT`
    # then says which roles, and how many, each of the 14 attributes needs.
    ingredient_objects: list[IngredientObject] = Field(min_length=1)
    # `autonomy_level` only, and required there (the same key and
    # the same rule as `scripts.probe_outline_schema.ProbeEntry.outward_tools`): the
    # one catalog `external_write` tool the task's OUTWARD step calls, after the
    # internal write the gate sits on. Reactive stops before the internal write,
    # Self-directed runs it and stops before this call, Autonomous makes it and
    # reports -- so all three settings separate on one ask. Exactly one entry: a
    # session releases one outward act. Empty on every other attribute.
    outward_tools: list[str] = Field(default_factory=list)
    # `capability_boundary` only, and required there: the
    # recovery route out of the block and its cost. `validate_outline`
    # (`_capability_recovery_errors`) requires it on capability cells and refuses
    # it on every other attribute.
    recovery: ProbeRecovery | None = None
    fixture_additions: list[FixtureAddition] | None = None
    # `no_gate_reason` is ABOLISHED with no compat path.
    # It let an uncompressed entry name a tool-call guard instead of a gate, but
    # a guard the PA can satisfy on its own leaves the task finishable before
    # the probe beat, so the PA places its IX selection wherever it likes and
    # the beat measures nothing. Every uncompressed entry now declares exactly
    # one `role: gate` ingredient. `StrictModel` forbids extra keys, so an
    # outline still carrying the field fails to parse.


class OutputRealizationContract(ProbeContractBase):
    contract_class: Literal["output_realization"]
    viable_option_count: int | None = Field(default=None, ge=2)


class EventAnchor(StrictModel):
    event: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    phase_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")


class EventOrders(StrictModel):
    correct: list[EventAnchor] = Field(min_length=1)
    wrong: list[EventAnchor] = Field(min_length=1)


class TopicPhaseGroup(StrictModel):
    topic_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    phase_ids: list[str] = Field(min_length=1)


class InteractionSequenceContract(ProbeContractBase):
    contract_class: Literal["interaction_sequence"]
    event_order: EventOrders
    topic_phase_groups: list[TopicPhaseGroup] | None = None


class IncludedPhases(StrictModel):
    correct: list[str] = Field(min_length=1)
    wrong: list[str] = Field(min_length=1)


class ActionScopeContract(ProbeContractBase):
    contract_class: Literal["action_scope"]
    branch_point_phase_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    branch_condition: str = Field(min_length=1)
    included_phases: IncludedPhases
    optional_phase_ids: list[str] | None = None
    # `capability_boundary` only, and required there: the phase that takes the
    # route `ProbeRecovery` declares. Where that route is an `external_write`
    # tool, this is the one phase of a compressed entry allowed to reach another
    # person, and it names exactly one phase -- a session releases one outward
    # act (`validate_outline._compressed_errors`).
    recovery_phase_ids: list[str] | None = None


class ContactDecisions(StrictModel):
    correct: Literal["send", "suppress"]
    wrong: Literal["send", "suppress"]


class TriggeredInteractionContract(ProbeContractBase):
    contract_class: Literal["triggered_interaction"]
    trigger_phase_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    trigger_condition: str = Field(min_length=1)
    proactive_contact_decision: ContactDecisions


ProbeContract = Annotated[
    Union[
        OutputRealizationContract,
        InteractionSequenceContract,
        ActionScopeContract,
        TriggeredInteractionContract,
    ],
    Field(discriminator="contract_class"),
]


class OutlineSession(StrictModel):
    identity: SessionIdentity
    story: SessionStory
    task: SessionTask
    probe: ProbeContract
    operational_source: OperationalSource

    @model_validator(mode="after")
    def _validate_opening_shape(self) -> "OutlineSession":
        if self.identity.session_type == "evolving_event":
            if self.task.activation is None:
                raise ValueError(
                    f"acc_{self.identity.acc_num:03d}: evolving_event entries "
                    "keep the legacy activation block"
                )
            if self.task.opening is None:
                raise ValueError(
                    f"acc_{self.identity.acc_num:03d}: evolving_event entries "
                    "must carry `opening` (the OpenRouter Outline generator "
                    "sets it from story facts)"
                )
            if self.task.opening_detail is not None:
                raise ValueError(
                    f"acc_{self.identity.acc_num:03d}: evolving_event entries "
                    "do not take opening_detail (activation.detail carries it)"
                )
        else:
            if self.task.activation is not None:
                raise ValueError(
                    f"acc_{self.identity.acc_num:03d}: normal sessions use the flat "
                    "opening field, not an activation block"
                )
        return self

    @model_validator(mode="after")
    def _validate_phase_floor(self) -> "OutlineSession":
        """A coarse task plan is two phases or more — one, when it is triggered
        and compressed.

        `proactive_outreach` is both: the probe is
        the PA's own trigger turn, and exactly one user beat follows it, so the
        entry has the trigger phase and nothing to put in a second. Every other
        entry still describes a task with a shape.
        """
        compressed_trigger = (
            self.operational_source.attribute in COMPRESSED_ATTRIBUTES
            and isinstance(self.probe, TriggeredInteractionContract)
        )
        if not compressed_trigger and len(self.task.coarse_task_phases) < 2:
            raise ValueError(
                f"acc_{self.identity.acc_num:03d}: coarse_task_phases needs at "
                "least two phases; only a triggered compressed entry collapses "
                "to one"
            )
        return self

    @property
    def acc_num(self) -> int:
        return self.identity.acc_num

    @property
    def context(self) -> str:
        return "work" if self.identity.target_cell.startswith("work_") else "personal"


class CompactOutline(StrictModel):
    meta: OutlineMeta
    evolving_map: dict[str, EvolvingArc]
    sessions: list[OutlineSession] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_clock(self) -> "CompactOutline":
        from datetime import date as _date

        from harness.benchmark_clock import TIME_POSITION_WINDOWS

        previous: tuple[int, str, int] | None = None
        for session in sorted(self.sessions, key=lambda item: item.acc_num):
            story = session.story
            window = TIME_POSITION_WINDOWS.get(story.time_position)
            if window is None:
                raise ValueError(
                    f"acc_{session.acc_num:03d}: unsupported time_position "
                    f"{story.time_position!r}"
                )
            day = _date.fromisoformat(story.date)
            if not (window[0] <= day <= window[1]):
                raise ValueError(
                    f"acc_{session.acc_num:03d}: date {story.date} is outside the "
                    f"{story.time_position} window {window[0]}..{window[1]}"
                )
            current = (session.acc_num, story.date, story.hour)
            # duplicate acc_nums are reported by the uniqueness check, not here
            if (
                previous is not None
                and session.acc_num != previous[0]
                and (story.date, story.hour) <= previous[1:]
            ):
                raise ValueError(
                    f"acc_{session.acc_num:03d}: clock {story.date} {story.hour:02d}:00 "
                    f"must come after acc_{previous[0]:03d} "
                    f"({previous[1]} {previous[2]:02d}:00)"
                )
            previous = current
        return self
