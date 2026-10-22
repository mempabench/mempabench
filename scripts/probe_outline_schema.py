"""Strict schema for probe outlines (`Authoring/probe/outline/<persona>/outline.yaml`).

A probe is one question asked at a fixed world point to read one preference
cell. An entry is exactly five blocks plus one provenance field: identity
(`id` / `kind` /
`cell`), the world point (`clock`), the world material (`world`), the ask
(`ask`) and the criteria (`answer`), plus `errand` — and, on a
`proactive_outreach` cell alone, the `trigger` block that attribute's probe is
fired by instead of a question. Nothing else — no story,
no phases, no opening, no identity section — because probes need neither
diversity nor a causal chain.

The outline layer is what the owner reviews; `scripts/validate_probe_outline.py`
runs the A1–A12 rows over a parsed `ProbeOutline`.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data.personas.coverage import TAXONOMY
from scripts.outline_schema import IngredientObject, ProbeRecovery


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


#: `fin_01..28` for finals (work × TAXONOMY order, then personal × the same
#: order), `pre_01..05` for the five pre_event probes.
PROBE_ID_RE = re.compile(r"^(fin|pre)_\d{2}$")

CHOICE_TOKEN_RE = re.compile(r"\$choice\.([a-z][a-z0-9_]*)\.([A-Za-z][A-Za-z0-9_]*)")


def resolve_choice_refs(value: Any, selected: dict[str, dict[str, Any]]) -> Any:
    """Resolve only declared finite-choice tokens; leave other refs intact."""
    if isinstance(value, dict):
        return {key: resolve_choice_refs(child, selected) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_choice_refs(child, selected) for child in value]
    if not isinstance(value, str):
        return value
    matches = list(CHOICE_TOKEN_RE.finditer(value))
    for match in matches:
        key, field = match.groups()
        if key not in selected or field not in selected[key]:
            raise ValueError(f"undeclared choice projection {key}.{field}")
    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        key, field = matches[0].groups()
        return selected[key][field]
    return CHOICE_TOKEN_RE.sub(
        lambda match: str(selected[match.group(1)][match.group(2)]), value
    )

#: The probe errand deck. Same shape as the acc deck (`^[wp][1-6]_…` in
#: `scripts/outline_schema.py`) but cycle digit 0 is reserved for probes, so the
#: two decks' id spaces never intersect. Widened to `[0-6]` here ONLY;
#: the acc schema keeps `[1-6]`.
PROBE_ERRAND_ID_RE = re.compile(r"^[wp][0-6]_[a-z0-9_]+$")

#: A `task_fact` is the KEY of a `$task_facts` entry, not the value the user
#: dictates: a short snake_case name (`calendar_hold_title`), as on the acc side.
TASK_FACT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

CELL_CONTEXTS = ("work", "personal")

#: The same role vocabulary as the acc outline's ingredient objects, read off
#: `IngredientObject` so the two layers cannot drift.
WorldRole = IngredientObject.model_fields["role"].annotation


def split_cell(cell: str) -> tuple[str, str]:
    """`work_verbosity` -> `("work", "verbosity")`; raises when either half is unknown."""
    context, _, attribute = cell.partition("_")
    if context not in CELL_CONTEXTS or attribute not in TAXONOMY:
        raise ValueError(
            f"cell {cell!r} must be `{{work|personal}}_{{attribute}}` with attribute "
            f"in TAXONOMY {sorted(TAXONOMY)}"
        )
    return context, attribute


def final_probe_id(cell: str) -> str:
    """`fin_01..14` = work × TAXONOMY order, `fin_15..28` = personal × the same order."""
    context, attribute = split_cell(cell)
    offset = 0 if context == "work" else len(TAXONOMY)
    return f"fin_{offset + list(TAXONOMY).index(attribute) + 1:02d}"


class ProbeClock(StrictModel):
    """The world point a probe runs at: ISO `date` plus a 24h `hour` (0–23)."""

    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    hour: int = Field(ge=0, le=23)


class ProbeMeta(StrictModel):
    persona: str = Field(min_length=1)
    status: Literal["in_progress", "complete"] = "complete"
    # Expected counts: 28 finals (2 contexts × 14 attributes), 5 pre_event.
    final: int = Field(ge=0)
    pre_event: int = Field(ge=0)
    # Every `kind: final` entry's clock equals this verbatim (A3 / A5).
    final_clock: ProbeClock


class ProbePayloadSpan(StrictModel):
    """One exact excerpt written into a gate payload from an already-bound document."""

    path: str = Field(min_length=1)
    field: str | None = Field(default=None, min_length=1)
    exact: str = Field(min_length=1)


class ProbePayloadCore(StrictModel):
    """Deterministic source excerpts plus where the user's dictated text belongs."""

    dictated_position: Literal["opening", "closing"] | None = None
    spans: list[ProbePayloadSpan] = Field(min_length=1)
    fact_refs: list[str] = Field(default_factory=list)


class ProbeRequiredInternalWrite(StrictModel):
    """One additional internal deliverable fixed by the outline."""

    key: str = Field(pattern=TASK_FACT_KEY_RE.pattern)
    tool: str = Field(min_length=1)
    position: Literal["before_gate", "after_gate"]
    ask_evidence: str = Field(min_length=1)
    tool_arguments: dict[str, str] = Field(default_factory=dict)
    allowed_model_arguments: list[str] = Field(default_factory=list)
    model_argument_evidence: dict[str, str] = Field(default_factory=dict)
    payload_param: str | None = Field(default=None, min_length=1)
    payload_core: ProbePayloadCore | None = None

    @model_validator(mode="after")
    def _payload_pair(self) -> "ProbeRequiredInternalWrite":
        if (self.payload_param is None) != (self.payload_core is None):
            raise ValueError("payload_param and payload_core appear together")
        if len(self.allowed_model_arguments) != len(set(self.allowed_model_arguments)):
            raise ValueError("allowed_model_arguments contains duplicates")
        return self


class ProbeChoiceCandidate(StrictModel):
    store: str = Field(min_length=1)
    record_id: str = Field(min_length=1)


class ProbeChoiceResolution(StrictModel):
    key: str = Field(pattern=TASK_FACT_KEY_RE.pattern)
    candidates: list[ProbeChoiceCandidate] = Field(min_length=2)
    projected_fields: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_values(self) -> "ProbeChoiceResolution":
        pairs = [(item.store, item.record_id) for item in self.candidates]
        if len(pairs) != len(set(pairs)):
            raise ValueError("choice candidates contain duplicates")
        record_ids = [item.record_id for item in self.candidates]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("choice candidate record_id values must be unique")
        if len(self.projected_fields) != len(set(self.projected_fields)):
            raise ValueError("projected_fields contains duplicates")
        return self


class ProbeWorldObject(StrictModel):
    """One world object the probe runs on ("probe ingredient").

    Roles and fields are those of `scripts.outline_schema.IngredientObject`
    (`role`, exactly one locator of `record_id` / `path` / `task_fact`, plus
    `store` / `field` / `note`, and on a `gate` the `gate_tool` / `gate_param` /
    `gate_kind` triple), minus the acc-only `ref`, `reveal_at` and the
    key-term / cue / slot gate keys. A field whose value would be null is
    omitted rather than written as null.
    """

    role: WorldRole
    # Fixture store path relative to the persona fixture root, e.g.
    # `email/inbox.jsonl`. Required with `record_id`; forbidden on `missing_field`.
    store: str | None = Field(default=None, min_length=1)
    # Exactly one locator. `record_id` for a minted record, `path` for a
    # document (fixture-relative, e.g. `documents/admin/policy.md`), `task_fact`
    # only for a fact no store holds (roles `missing_field` and `gate`).
    record_id: str | None = Field(default=None, min_length=1)
    path: str | None = Field(default=None, min_length=1)
    # A `$task_facts` KEY, as the acc side spells one (`calendar_hold_title`,
    # `student_reply_line_the_user_dictates`) -- the name of the fact, never the
    # wording the user dictates. That wording belongs in `ask.say` (A4).
    task_fact: str | None = Field(default=None, min_length=1, pattern=TASK_FACT_KEY_RE.pattern)
    # The section or record field the probe reads, e.g. `4. Limits and pre-approval`.
    field: str | None = Field(default=None, min_length=1)
    note: str | None = Field(default=None, min_length=1)
    # Role `candidate` on an `information_elicitation` probe only (A4, owner
    # 2026-09-16): the `missing_field` this record is one possible answer to. The acc
    # line spells the same key `IngredientObject.missing_field_ref` and points it at
    # that ingredient's `ref`; a probe world object carries no `ref`, so here it names
    # the field's `task_fact`, which is unique within the entry and already validated
    # as a fact key. Every missing field needs at least two of these: `Infer` settles
    # an omitted field from what it can read, and with one candidate it reads the one
    # value and renders the same turn as `Structured`.
    missing_field_ref: str | None = Field(
        default=None, min_length=1, pattern=TASK_FACT_KEY_RE.pattern
    )
    # Role `gate` only, the owner's gate definition as on `IngredientObject`:
    # `gate_tool` is the catalog tool the task cannot skip, `gate_param` one of
    # its REQUIRED parameters, `task_fact` the key whose value only the user
    # holds. `gate_kind` has the one shape there is; A4 checks the tool and the
    # parameter against the catalog.
    gate_tool: str | None = Field(default=None, min_length=1)
    gate_param: str | None = Field(default=None, min_length=1)
    gate_kind: Literal["dictated_text"] | None = None
    payload_core: ProbePayloadCore | None = None
    payload_param: str | None = Field(default=None, min_length=1)
    body_from_internal_write: str | None = Field(
        default=None, pattern=TASK_FACT_KEY_RE.pattern
    )
    tool_arguments: dict[str, str] | None = None
    payload_template: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _no_explicit_nulls(self) -> "ProbeWorldObject":
        nulls = sorted(
            name for name in self.model_fields_set if getattr(self, name) is None
        )
        if nulls:
            raise ValueError(f"world object: omit null fields instead of writing them {nulls}")
        return self

    @model_validator(mode="after")
    def _exactly_one_locator(self) -> "ProbeWorldObject":
        locators = [
            name for name in ("record_id", "path", "task_fact") if getattr(self, name)
        ]
        if len(locators) != 1:
            raise ValueError(
                f"world object with role {self.role!r} needs exactly one locator of "
                f"record_id / path / task_fact; got {locators}"
            )
        locator = locators[0]
        if locator == "task_fact" and self.role not in ("missing_field", "gate"):
            raise ValueError(
                f"role {self.role!r} locates a real record with record_id or path; "
                "task_fact is only for missing_field and gate"
            )
        if self.role == "missing_field" and self.store is not None:
            raise ValueError("a missing_field lives with the user; it names no store")
        if locator == "record_id" and self.store is None:
            raise ValueError("record_id needs the store that holds the record")
        return self

    @model_validator(mode="after")
    def _gate_fields_belong_to_a_gate(self) -> "ProbeWorldObject":
        if self.role == "gate":
            missing = sorted(
                name
                for name in ("gate_tool", "gate_param", "task_fact")
                if not (getattr(self, name) or "").strip()
            )
            if missing:
                raise ValueError(
                    "a gate names gate_tool, gate_param and the task_fact key the "
                    f"user supplies; missing {missing}"
                )
            if self.payload_template is not None:
                if self.payload_param != self.gate_param:
                    raise ValueError("payload_template requires payload_param equal to gate_param")
                if self.payload_core is not None or self.body_from_internal_write is not None:
                    raise ValueError("payload_template cannot combine with payload_core or body reuse")
        else:
            supplied = sorted(
                name
                for name in (
                    "gate_tool", "gate_param", "gate_kind", "payload_core",
                    "payload_param", "body_from_internal_write",
                    "tool_arguments", "payload_template",
                )
                if getattr(self, name) not in (None, {}, [])
            )
            if supplied:
                raise ValueError(f"{supplied} belong to role 'gate', not to role {self.role!r}")
        return self


class ProbeAsk(StrictModel):
    # The situation the question happens in (simulator-side knowledge).
    background: str = Field(min_length=1)
    # The one user turn: what to say. Copied verbatim into the active beat.
    say: str = Field(min_length=1)
    # Wording forbidden in that turn; preference-neutrality lives here.
    avoid: str = Field(min_length=1)
    # Literal strings the turn must carry. A6: required (at least one) when the
    # entry binds a document, may be empty when it binds only records; each term
    # occurs verbatim in a bound document or in a bound record's field values.
    key_terms: list[str] = Field(min_length=0)


class ProbeTrigger(StrictModel):
    """The arrival a `proactive_outreach` probe is read on.

    That attribute is the one an external event fires instead of a user turn, so
    its probe reuses the accumulation side's triggered shape: `trigger_id` /
    `detail` / `user_impact` become the draft's `proactive_trigger` (whose
    `background` is `ask.background`), and `say_after_send` is what the user says
    in the opening that plays when the PA did make contact, as `ask.say` is what
    they say in the opening that plays when it stayed silent.
    """

    trigger_id: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    user_impact: str = Field(min_length=1)
    say_after_send: str = Field(min_length=1)


class ProbeAnswer(StrictModel):
    # Why every canonical setting of the target attribute is a complete answer.
    all_viable: str = Field(min_length=1)
    # Setting name -> what a reply under that setting looks like. The key set
    # equals `TAXONOMY[attribute]` exactly (A8); no GT key lives here.
    by_setting: dict[str, str] = Field(min_length=1)


class ProbeEntry(StrictModel):
    id: str = Field(pattern=PROBE_ID_RE.pattern)
    kind: Literal["final", "pre_event"]
    # `{work|personal}_{attribute}`; attribute is one of the 14 in TAXONOMY.
    cell: str = Field(min_length=1)
    # Provenance only: the probe-deck errand id (`w0_…` / `p0_…`).
    errand: str = Field(pattern=PROBE_ERRAND_ID_RE.pattern)
    clock: ProbeClock
    world: list[ProbeWorldObject] = Field(min_length=1)
    ask: ProbeAsk
    answer: ProbeAnswer
    required_internal_writes: list[ProbeRequiredInternalWrite] = Field(default_factory=list)
    choice_resolutions: list[ProbeChoiceResolution] = Field(default_factory=list)
    # `proactive_outreach` only, and required there (A12): the record that
    # arrives and the line the user says once the PA has made contact. Every
    # other attribute is read by a question and carries no trigger block.
    trigger: ProbeTrigger | None = None
    # `autonomy_level` only, and required there (A17, owner 2026-09-16): the one
    # catalog `external_write` tool the ask's OUTWARD step calls, after the
    # internal write the gate sits on. Reactive stops before the internal write,
    # Self-directed runs it and stops before this call, Autonomous makes it and
    # reports -- so all three settings separate on one ask. Exactly one entry:
    # a session releases one outward act. Empty on every other attribute.
    outward_tools: list[str] = Field(default_factory=list)
    # `capability_boundary` only, and required there (A19, owner 2026-09-16): the
    # route out of the block and what taking it costs her. The SAME model the acc
    # line uses (`scripts.outline_schema.ProbeRecovery`), because the ruling is one
    # ruling: "Suggest alternatives" states the limit and takes the route in the
    # blocked turn, "Find and hand off" states it, names the route and hands control
    # back, so both arms are written from one declared route. Null on every other
    # attribute.
    recovery: ProbeRecovery | None = None
    # pre_event only, always together (A9): the final probe this one pairs
    # with (`fin_NN`) and the acc number of the evolving event it runs before.
    pairs_with: str | None = Field(default=None, pattern=r"^fin_\d{2}$")
    before_acc: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _cell_is_known(self) -> "ProbeEntry":
        split_cell(self.cell)
        return self

    @model_validator(mode="after")
    def _pairing_keys_iff_pre_event(self) -> "ProbeEntry":
        present = sorted(
            name for name in ("pairs_with", "before_acc") if getattr(self, name) is not None
        )
        if self.kind == "pre_event" and len(present) != 2:
            raise ValueError(
                f"A9: {self.id} is pre_event, so pairs_with and before_acc are both "
                f"present; got {present}"
            )
        if self.kind == "final" and present:
            raise ValueError(
                f"A9: {self.id} is final, so neither pairs_with nor before_acc may "
                f"appear; got {present}"
            )
        return self


class ProbeOutline(StrictModel):
    meta: ProbeMeta
    probes: list[ProbeEntry]
