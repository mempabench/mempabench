"""Strict authoring schemas for MemPABench normal sessions and forced events."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    RootModel,
    Tag,
    field_validator,
    model_validator,
)

from scripts.probe_outline_schema import ProbeWorldObject

EVENT_ACC_NUMS = frozenset({15, 30, 45, 60, 75, 90})

#: The evolving-event ladder is the same six fixed slots for every persona, of
#: which each persona retires exactly one (User A retires acc 90, User B acc
#: 60), so the event id is the acc's ordinal position in the ladder rather than
#: a per-persona table: 15/30/45/60/75/90 -> evolv_01..evolv_06.
FORCED_EVENT_ID_BY_ACC = {
    acc_num: f"evolv_{position:02d}"
    for position, acc_num in enumerate(sorted(EVENT_ACC_NUMS), start=1)
}

#: Attributes authored as a *compressed* session: exactly two beats. Beat 1 is
#: the probe -- the user states the whole query up front and the PA's single
#: reply carries both the selection and the surface being measured -- and beat
#: 2 is one user reaction plus one PA reply, after which the session ends
#: whatever the goal state is. There is no close beat. Owner rulings A-D,
#: 2026-09-07.
#:
#: The table is attribute names, not `contract_class`: the set crosses classes
#: in both directions. `task_expansion` and `capability_boundary` are not
#: `output_realization` yet compress, and `reasoning_visibility` is
#: `output_realization` yet stays uncompressed. Everything absent here keeps
#: today's floors.
#:
#: `proactive_outreach` compresses with the probe somewhere else (owner ruling,
#: 2026-09-08): its session is triggered, so the probe is the PA's own trigger
#: turn -- the external message arrives, and sending is High while staying
#: silent is Low. The two authored beats are then the two entry-conditioned
#: openings, of which the runtime plays exactly one, and that one is the
#: reaction. Both openings declare the target, and neither is empty. See the
#: triggered fork of the beat rule on `NormalSessionDraft`.
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


def is_compressed_attribute(target_cell: str) -> bool:
    """Whether `target_cell`'s attribute is authored as a compressed session.

    Takes the cell (`work_verbosity`), not the bare attribute, so callers that
    only hold a cell do not have to strip the context prefix themselves.
    """
    _, _, attribute = target_cell.partition("_")
    return attribute in COMPRESSED_ATTRIBUTES


#: Read tools whose store search takes one free-text argument plus the session
#: date. An empty query matches every record (``_search_tokens("")`` is empty,
#: so the token filter is skipped), so the call enumerates exactly what that
#: tool resolves on that date — recurrence expansion included, because the
#: store's own matcher runs.
DATE_FILTERED_PROBES = {
    "calendar_event_search": "query",
    "appointment_search": "service",
    "restaurant_reservation_search": "query",
    "room_booking_search": "query",
}
#: Catalog parameter names that filter a read on a calendar date.
DATE_ARGUMENTS = frozenset({"date", "check_in", "check_out"})


from simulator.schemas import OPENING_AFTER_SEND, OPENING_AFTER_SUPPRESS, OPENING_BEATS

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalSessionMeta(StrictModel):
    artifact_kind: Literal["normal_session"]
    # world_toolbox: steps reference catalog tools directly; the runtime stages
    # the full world and exposes the whole toolbox (no binding/bundle step).
    tool_status: Literal["world_toolbox"]
    persona_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    session_id: str = Field(pattern=r"^acc_\d{3}$")
    session_type: Literal["stable", "evolving_pre", "evolving_post"]
    target_cell: str = Field(pattern=r"^(work|personal)_[a-z0-9_]+$")
    target_setting: str
    contract_class: Literal[
        "output_realization",
        "interaction_sequence",
        "action_scope",
        "triggered_interaction",
    ]
    # sha256 of the tool catalog file the generator planned these steps against.
    # Optional on the MODEL so an unstamped draft still parses and the validator
    # can say which draft is unstamped and against which catalog; it is the
    # validator, not the schema, that refuses a draft without it. A stamp that
    # does not match the catalog the validator loads means the world's tools
    # moved under a frozen plan: the step arguments were checked against tools
    # that no longer exist in that shape.
    catalog_sha256: str | None = None

    @model_validator(mode="after")
    def _setting_is_named(self) -> NormalSessionMeta:
        if not self.target_setting.strip():
            raise ValueError("target_setting must not be blank")
        return self

    @property
    def context(self) -> Literal["work", "personal"]:
        return "work" if self.target_cell.startswith("work_") else "personal"

    @property
    def target_attribute(self) -> str:
        return self.target_cell.split("_", 1)[1]


class AssistanceEmergence(StrictModel):
    pattern: Literal[
        "explicit_request",
        "contextual_signal",
        "partial_intent",
        "ongoing_task_bottleneck",
        "result_driven_next_step",
        "prior_state_change_or_repair",
    ]
    rationale: str = Field(min_length=1)
    evidence_beat_ids: list[str] = Field(default_factory=list)
    evidence_atomic_step_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_evidence(self) -> AssistanceEmergence:
        if not self.evidence_beat_ids and not self.evidence_atomic_step_ids:
            raise ValueError("assistance_emergence requires concrete evidence")
        return self


class NormalAssistanceEmergence(StrictModel):
    """Normal-session variant: evidence must land in visible exchanges."""

    pattern: Literal[
        "explicit_request",
        "contextual_signal",
        "partial_intent",
        "ongoing_task_bottleneck",
        "result_driven_next_step",
        "prior_state_change_or_repair",
    ]
    rationale: str = Field(min_length=1)
    evidence_beat_ids: list[str] = Field(min_length=1)


class AtomicStep(StrictModel):
    step_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source_outline_phase_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    purpose: str = Field(min_length=1)
    actor: Literal["user", "pa", "tool", "environment"]
    action_kind: Literal[
        "signal",
        "clarify",
        "choose",
        "authorize",
        "read",
        "compare",
        "decide",
        "execute",
        "wait",
        "repair",
        "report",
    ]
    depends_on: list[str] = Field(default_factory=list)
    # Direct world-toolbox reference: an actor=tool step names one catalog tool
    # and the planned call arguments (real values the story establishes).
    tool: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    required_observation: str = Field(min_length=1)
    # A read or attempted write the world refuses with a capability_limit.
    # The dry run requires that outcome; a refused write also must leave task
    # state unchanged before it can omit successful-write readback/state changes.
    expects_capability_limit: bool = False
    authorization: Literal["not_required", "required_before_step"] = "not_required"
    readback_step_id: str | None = None

    @model_validator(mode="after")
    def _validate_tool_actor(self) -> AtomicStep:
        if (self.actor == "tool") != (self.tool is not None):
            raise ValueError(
                f"step {self.step_id}: actor=tool exactly when a catalog tool is named"
            )
        if self.tool_arguments and self.tool is None:
            raise ValueError(f"step {self.step_id}: tool_arguments require a tool")
        return self


class StateChange(StrictModel):
    """One write this session's PA makes, and what the author expects of it.

    The world accumulates across sessions in acc order: every write a session's
    PA makes stands for every session after it. Nothing is rolled back between
    sessions, so a later session must not anchor on text or records an earlier
    session's PA is expected to change. `scripts/check_cross_acc_state.py` is
    the mechanical check for that.

    `expected_before` and `expected_after` are AUTHORING-ONLY: they record what
    the author believes the target looks like before and after this write, they
    are read by the static checks (and by a human reading the draft), and no
    runtime module ever compares them against the live world.
    """

    # Retired field: `persistence: session | run` said whether a write survived
    # past its own session. Under accumulation every write does, so the field
    # had nothing left to distinguish. `StrictModel`'s `extra="forbid"` makes a
    # draft that still carries it an error; `scripts/migrate_drop_persistence.py`
    # removes it. Docstrings reach the generator's structured-output schema, so
    # the retired name stays in this comment rather than in the description.
    change_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    atomic_step_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    # Defaults keep null values round-trip safe (dumps use exclude_none).
    expected_before: Any = None
    expected_after: Any = None
    readback_step_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")


class SideInteraction(StrictModel):
    interaction_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    purpose: str = Field(min_length=1)
    needed_after_step_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    user_input: str = Field(min_length=1)


class TaskExecution(StrictModel):
    atomic_steps: list[AtomicStep] = Field(min_length=2)
    state_changes: list[StateChange] = Field(default_factory=list)


class SessionLifeContext(StrictModel):
    what_happened: str = Field(min_length=1)
    mood: str = Field(min_length=1)
    time_pressure: str = Field(min_length=1)
    energy: str = Field(min_length=1)
    continuity_note: str = Field(min_length=1)
    # The calendar record this scene leans on, when it leans on one: the shift
    # the user just came off, the class she is between, the appointment she is
    # waiting at. Written by the author so the claim can be checked; without it
    # the only way to tell "she finished a lunch shift" from "she has a lunch
    # shift tomorrow" is to read the verb, and a verb scan is not a check.
    # `validate_normal_event`'s `life_context_outside_calendar` then asks the
    # arithmetic question: does that event's recurrence reach this weekday, and
    # do its hours reach this hour. Null when the scene rests on the user's own
    # ordinary state and no appointment — which is most sessions.
    calendar_event_id: str | None = None


class NormalReactionBranch(StrictModel):
    condition: str = Field(min_length=1)
    user_instruction: str = Field(min_length=1)
    language_style: str = ""
    avoid: str = Field(min_length=1)


class BeatFamiliaritySignals(StrictModel):
    """The `guidance_level` probe's term pair, said in the user's own message.

    One domain term the user handles confidently and one they visibly do not,
    both of which have to appear in the probe beat's `user_instruction` when the
    pair is declared at all. It never reaches the simulator —
    `authored_session_to_session_script` builds each `Beat` field by field — and
    is the record the judge reads off the probe beat.
    """

    known_term: str = Field(min_length=1)
    unknown_term: str = Field(min_length=1)


class AuthoringBeat(StrictModel):
    beat: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    # One short line: what the user wants out of this beat. No how, no facts.
    goal: str = Field(min_length=1)
    user_instruction: str = Field(min_length=1)
    language_style: str = ""
    # Actor view: what the user is waiting for from the PA before this beat.
    entry_condition: str = Field(min_length=1)
    avoid: str = ""
    # Attribute ids whose target preference surfaces in this beat.
    active_skills: list[str] = Field(default_factory=list)
    # `task_facts` keys this beat consumes without them reaching any tool
    # argument: a fact that drives the user's own decision or the wording of
    # their instruction (acc_027's `daily_walk_time` decides which slot they
    # take). A fact consumed through a tool is linked by a
    # `$task_facts.<key>` reference in `tool_arguments` instead; this field is
    # the only way to link one that never gets there.
    uses: list[str] = Field(default_factory=list)
    # Actor-view fallback: what the user does if the thing they were waiting
    # for never happened.
    if_not_met: str = ""
    reactions: dict[str, NormalReactionBranch] = Field(default_factory=dict)
    # Optional on a `guidance_level` probe beat, absent everywhere else.
    familiarity_signals: BeatFamiliaritySignals | None = None


class AuthoringProactiveTrigger(StrictModel):
    trigger_id: str = Field(min_length=1)
    # Retired for normal sessions (the outline no longer carries a type; the
    # generator strips it). Forced events still set and check it.
    activation_type: Literal[
        "scheduled_time", "deadline", "state_change", "external_event"
    ] | None = None
    background: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    user_impact: str = Field(min_length=1)


class NormalSessionDraft(StrictModel):
    meta: NormalSessionMeta
    assistance_emergence: NormalAssistanceEmergence
    task_execution: TaskExecution
    scenario: str = Field(min_length=1)
    task: str = Field(min_length=1)
    life_context: SessionLifeContext
    beats: list[AuthoringBeat] = Field(default_factory=list)
    proactive_trigger: AuthoringProactiveTrigger | None = None
    task_facts: dict[str, Any] = Field(default_factory=dict)
    # Information gating: a `task_facts` key the user reveals only when the
    # named beat is played. Before that beat the fact is not rendered to the
    # simulated user at all, so neither an eager PA nor an eager actor can do
    # the preference beat's work early.
    reveal_at: dict[str, str] = Field(default_factory=dict)
    # The beat on which the user first lets the PA know there is something still
    # to be said -- the gate's cue, pinned to a beat by the author instead of
    # recovered by substring search at run time. The runtime resolver
    # (`run_orchestrator._gate_cue_beat`) fell to None on every gated user_a
    # draft because the cue sits in the scenario prose while the beats
    # paraphrase it, and a cue no beat owes is a cue the style retry may cut.
    # Whether the field is required is a question about the outline's gate, so
    # `validate_normal_event` owns that arm (`gate_cue_beat_missing` /
    # `gate_cue_beat_without_gate` / `gate_cue_beat_unresolved`); the schema
    # owns only the shape -- a named beat this draft has.
    gate_cue_beat: str | None = None

    @model_validator(mode="after")
    def _validate_cross_surface_contract(self) -> NormalSessionDraft:
        meta = self.meta
        _validate_step_graph(self.task_execution)

        beat_ids = [beat.beat for beat in self.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("beats must have unique beat ids")
        # A session with a proactive trigger (triggered_interaction) is
        # authored with both openings (the PA's contact arrived / no contact
        # came and the user returns on their own); the runtime plays exactly
        # one of them, then the shared beats -- or, when the attribute is
        # compressed, nothing after it: the played opening is the last beat.
        triggered = self.proactive_trigger is not None
        if triggered:
            if tuple(beat_ids[:2]) != OPENING_BEATS:
                raise ValueError(
                    "a session with proactive_trigger must start with the beats "
                    f"{OPENING_AFTER_SEND!r} then {OPENING_AFTER_SUPPRESS!r}"
                )
        elif set(beat_ids) & set(OPENING_BEATS):
            raise ValueError(
                f"{OPENING_AFTER_SEND!r} / {OPENING_AFTER_SUPPRESS!r} beats belong "
                "to sessions with proactive_trigger only"
            )
        # A compressed attribute is exactly two beats -- beat 1 the probe, beat
        # 2 one user reaction and one PA reply, after which the session ends
        # whatever the goal state is (owner ruling A, 2026-09-07): there is no
        # close beat, so this is a fixed length and not a floor. Every other
        # session needs its warm-up beats, so five is the floor (owner ruling
        # B: the explicit_request exception of four is withdrawn; a session
        # that cannot reach five naturally is raised case by case). A triggered
        # session plays only one of its two openings, so it needs one more
        # authored beat to keep five exchanges on either path.
        compressed = is_compressed_attribute(meta.target_cell)
        if compressed:
            if len(self.beats) != 2:
                raise ValueError(
                    f"compressed attribute {meta.target_attribute!r} is authored as "
                    "exactly two exchanges (the probe beat and one reaction), got "
                    f"{len(self.beats)}"
                )
            if triggered:
                # A triggered compressed session has no probe beat: the probe
                # is the PA's own trigger turn, where sending is High and
                # staying silent is Low. Its two beats are the two openings
                # (ids already enforced above), the runtime plays exactly one
                # by entry condition, and whichever plays is the single
                # reaction. So both declare the target -- the measurement is on
                # either path -- and there is no beat to keep empty (owner
                # ruling, 2026-09-08). `validate_normal_event` accordingly holds
                # BOTH openings to the reception-beat standard
                # (`reception_beat_has_no_reaction_move`), not "the beat after
                # the first declaring beat".
                undeclared = [
                    beat.beat
                    for beat in self.beats
                    if meta.target_attribute not in beat.active_skills
                ]
                if undeclared:
                    raise ValueError(
                        f"triggered compressed attribute {meta.target_attribute!r} is "
                        "measured on whichever opening plays, so both openings declare "
                        f"it in active_skills; {undeclared} do not"
                    )
            else:
                # The probe is beat 1 and it declares the attribute by name:
                # "some beat carries some skill" would let a draft measure one
                # attribute while the outline binds another.
                probe, reaction = self.beats
                if meta.target_attribute not in probe.active_skills:
                    raise ValueError(
                        f"compressed attribute {meta.target_attribute!r} must be declared "
                        f"in the active_skills of the first beat {probe.beat!r}, which "
                        f"declares {sorted(probe.active_skills)}"
                    )
                # Beat 2 reacts to what beat 1 produced; declaring the attribute
                # again would make the reaction a second probe. No draft in the
                # corpus declares any attribute other than its own target, so the
                # reaction beat carries nothing at all.
                if reaction.active_skills:
                    raise ValueError(
                        f"the reaction beat {reaction.beat!r} of compressed attribute "
                        f"{meta.target_attribute!r} declares "
                        f"{sorted(reaction.active_skills)}; it reacts to the probe and "
                        "declares no active_skills of its own"
                    )
        else:
            minimum_beats = 6 if triggered else 5
            if len(self.beats) < minimum_beats:
                raise ValueError(
                    "normal session requires at least "
                    f"{ {5: 'five', 6: 'six'}[minimum_beats] } exchanges "
                    f"for {self.assistance_emergence.pattern} openings"
                    + (" (both openings plus four shared beats)" if triggered else "")
                )

        evidence_beats = set(self.assistance_emergence.evidence_beat_ids)
        if not evidence_beats <= set(beat_ids):
            raise ValueError("assistance emergence references unknown exchanges")

        _validate_beat_fact_links(self)
        return self


def _validate_step_graph(runtime: TaskExecution) -> None:
    """Step ids unique, dependencies known and earlier, readbacks and state
    changes pointing at real steps. Shared by both `meta.artifact_kind`
    branches of the session draft."""
    steps = runtime.atomic_steps
    step_ids = [step.step_id for step in steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("task_execution.atomic_steps must have unique step_id values")
    step_id_set = set(step_ids)
    step_position = {step_id: index for index, step_id in enumerate(step_ids)}
    for step in steps:
        if len(step.depends_on) != len(set(step.depends_on)):
            raise ValueError(f"atomic step {step.step_id} repeats dependencies")
        unknown_dependencies = set(step.depends_on) - step_id_set
        if unknown_dependencies:
            raise ValueError(
                f"atomic step {step.step_id} has unknown dependencies: "
                f"{sorted(unknown_dependencies)}"
            )
        if any(
            step_position[item] >= step_position[step.step_id]
            for item in step.depends_on
        ):
            raise ValueError(
                f"atomic step {step.step_id} dependencies must appear earlier"
            )
        if step.readback_step_id and step.readback_step_id not in step_id_set:
            raise ValueError(
                f"atomic step {step.step_id} has unknown readback_step_id"
            )

    for change in runtime.state_changes:
        if change.atomic_step_id not in step_id_set:
            raise ValueError(f"state change {change.change_id} references unknown step")
        if change.readback_step_id not in step_id_set:
            raise ValueError(f"state change {change.change_id} has unknown readback step")


def _validate_beat_fact_links(draft: NormalSessionDraft | ProbeSessionDraft) -> None:
    """`uses`, `reveal_at` and `gate_cue_beat` name real task_facts keys and
    real beats. Shared by both `meta.artifact_kind` branches."""
    beat_ids = {beat.beat for beat in draft.beats}
    for beat in draft.beats:
        if len(beat.uses) != len(set(beat.uses)):
            raise ValueError(f"beat {beat.beat} repeats a uses key")
        unknown_uses = [key for key in beat.uses if key not in draft.task_facts]
        if unknown_uses:
            raise ValueError(
                f"beat {beat.beat}.uses names no key in task_facts: "
                f"{sorted(unknown_uses)}"
            )

    for fact_key, beat_id in draft.reveal_at.items():
        if fact_key not in draft.task_facts:
            raise ValueError(
                f"reveal_at.{fact_key} names no key in task_facts"
            )
        if beat_id not in beat_ids:
            raise ValueError(
                f"reveal_at.{fact_key} names an unknown beat {beat_id!r}"
            )
    if draft.gate_cue_beat is not None and draft.gate_cue_beat not in beat_ids:
        raise ValueError(
            f"gate_cue_beat names an unknown beat {draft.gate_cue_beat!r}"
        )


#: A probe draft's id: `fin_XX` reads one cell at the frozen final cut point,
#: `pre_XX` reads an evolving cell just before its event acc.
PROBE_ID_PATTERN = r"^(fin|pre)_\d{2}$"


class ProbeSessionMeta(StrictModel):
    """`meta` of the `probe_session` branch (design §2.1.3 table).

    Identity is `probe_id` + `probe_kind` + `cell`; `context` and
    `target_attribute` are derived from `cell` and never authored. No
    `target_setting` (the GT lives in `resolved_gt.yaml`) and no
    `session_type` / `contract_class`.
    """

    artifact_kind: Literal["probe_session"]
    tool_status: Literal["world_toolbox"]
    persona_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    probe_id: str = Field(pattern=PROBE_ID_PATTERN)
    # The runtime keys checkpoints, ledgers and transcripts on `session_id`;
    # a probe has one identity, so it is the probe id repeated.
    session_id: str = Field(pattern=PROBE_ID_PATTERN)
    probe_kind: Literal["final", "pre_event"]
    cell: str = Field(pattern=r"^(work|personal)_[a-z_]+$")
    # Pairing of an evolving cell's two reads. `before_acc` (the event acc's
    # number) exists exactly on `pre_event`; `pairs_with` is written into BOTH
    # drafts of a pair -- pre -> its final, final -> its pre -- because the
    # runtime overlay lookup (`harness/orchestrator._probe_pairs_with`) reads
    # it off the final side. So `pairs_with` + `before_acc` both present <=>
    # `pre_event`, and a final's `pairs_with`, when present, names a `pre_XX`.
    pairs_with: str | None = Field(default=None, pattern=PROBE_ID_PATTERN)
    before_acc: int | None = Field(default=None, ge=1)
    # Same meaning as on `NormalSessionMeta`: the validator, not the schema,
    # refuses an unstamped draft.
    catalog_sha256: str | None = None

    @model_validator(mode="after")
    def _identity_and_pairing(self) -> ProbeSessionMeta:
        if self.session_id != self.probe_id:
            raise ValueError(
                f"session_id {self.session_id!r} must equal probe_id {self.probe_id!r}"
            )
        expected_prefix = "pre_" if self.probe_kind == "pre_event" else "fin_"
        if not self.probe_id.startswith(expected_prefix):
            raise ValueError(
                f"probe_kind {self.probe_kind!r} takes a {expected_prefix}XX probe_id, "
                f"got {self.probe_id!r}"
            )
        if self.probe_kind == "pre_event":
            if self.pairs_with is None or self.before_acc is None:
                raise ValueError(
                    "a pre_event probe carries both pairs_with (its final) and "
                    "before_acc (the event acc number)"
                )
            if not self.pairs_with.startswith("fin_"):
                raise ValueError(
                    f"a pre_event probe pairs with a fin_XX probe, got {self.pairs_with!r}"
                )
        else:
            if self.before_acc is not None:
                raise ValueError("before_acc exists only on a pre_event probe")
            if self.pairs_with is not None and not self.pairs_with.startswith("pre_"):
                raise ValueError(
                    f"a final probe pairs with a pre_XX probe, got {self.pairs_with!r}"
                )
        return self

    @property
    def context(self) -> Literal["work", "personal"]:
        return "work" if self.cell.startswith("work_") else "personal"

    @property
    def target_attribute(self) -> str:
        return self.cell.split("_", 1)[1]


class ProbeSessionDraft(StrictModel):
    """The `probe_session` branch: the `NormalSessionDraft` skeleton without
    `assistance_emergence`, and a fixed beat shape -- 0-2 setup beats that
    declare no `active_skills`, then exactly one active beat that declares the
    cell's attribute (design §2.1.3, C5). `task_execution` / `task_facts` /
    `reveal_at` / `gate_cue_beat` mean what they mean on the normal branch.
    `world` is the outline entry's `world[]` carried over verbatim, so
    `validate_probe_suite --drafts` (S3) can compare a pair's material.

    A `proactive_outreach` probe is the one exception to that beat shape: it is
    triggered rather than asked, so it carries `proactive_trigger` and its beats
    are exactly the two openings, both declaring the attribute (§2.1.2b).
    """

    meta: ProbeSessionMeta
    task_execution: TaskExecution
    scenario: str = Field(min_length=1)
    task: str = Field(min_length=1)
    life_context: SessionLifeContext
    beats: list[AuthoringBeat] = Field(min_length=1, max_length=3)
    world: list[ProbeWorldObject] = Field(min_length=1)
    # `proactive_outreach` alone (design §2.1.2b): that attribute is read by an
    # arriving record instead of a question, so its probe reuses the
    # accumulation side's triggered shape -- this block and the two openings
    # instead of an ask beat.
    proactive_trigger: AuthoringProactiveTrigger | None = None
    task_facts: dict[str, Any] = Field(default_factory=dict)
    reveal_at: dict[str, str] = Field(default_factory=dict)
    gate_cue_beat: str | None = None

    @model_validator(mode="after")
    def _validate_probe_shape(self) -> ProbeSessionDraft:
        _validate_step_graph(self.task_execution)
        beat_ids = [beat.beat for beat in self.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("beats must have unique beat ids")
        attribute = self.meta.target_attribute
        if self.proactive_trigger is not None:
            # The triggered probe has no ask beat and no setup beat: the probe
            # is the PA's own trigger turn, where sending is High and staying
            # silent is Low. Its beats are the two openings, the runtime plays
            # exactly one, and whichever plays is the measurement -- so both
            # declare the attribute, as on the accumulation side.
            if tuple(beat_ids) != OPENING_BEATS:
                raise ValueError(
                    "a probe with proactive_trigger is exactly the beats "
                    f"{OPENING_AFTER_SEND!r} then {OPENING_AFTER_SUPPRESS!r}; got {beat_ids}"
                )
            undeclared = [
                beat.beat for beat in self.beats if beat.active_skills != [attribute]
            ]
            if undeclared:
                raise ValueError(
                    f"both openings of a triggered probe declare exactly [{attribute!r}] "
                    f"in active_skills; {undeclared} do not"
                )
            _validate_beat_fact_links(self)
            return self
        if set(beat_ids) & set(OPENING_BEATS):
            raise ValueError(
                f"{OPENING_AFTER_SEND!r} / {OPENING_AFTER_SUPPRESS!r} beats belong "
                "to sessions with proactive_trigger only"
            )
        *setup, active = self.beats
        declaring = [beat.beat for beat in setup if beat.active_skills]
        if declaring:
            raise ValueError(
                f"setup beats declare no active_skills; {declaring} do"
            )
        if active.active_skills != [attribute]:
            raise ValueError(
                f"the last beat {active.beat!r} is the probe's one active beat and "
                f"declares exactly [{attribute!r}] in active_skills, got "
                f"{active.active_skills}"
            )
        _validate_beat_fact_links(self)
        return self


def _artifact_kind(value: Any) -> str | None:
    if isinstance(value, dict):
        meta = value.get("meta")
        if isinstance(meta, dict):
            kind = meta.get("artifact_kind")
            return kind if isinstance(kind, str) else None
        return None
    meta = getattr(value, "meta", None)
    return getattr(meta, "artifact_kind", None)


class SessionDraft(
    RootModel[
        Annotated[
            Annotated[NormalSessionDraft, Tag("normal_session")]
            | Annotated[ProbeSessionDraft, Tag("probe_session")],
            Discriminator(_artifact_kind),
        ]
    ]
):
    """A session draft of either kind, told apart by `meta.artifact_kind`.

    `SessionDraft.model_validate(data).root` is the `NormalSessionDraft` or
    `ProbeSessionDraft`; an unknown or missing kind is a validation error.
    """


PreferenceChangeMechanism = Literal[
    "material_task_consequence",
    "corrective_labor",
    "loss_of_control",
    "attention_cost",
    "workflow_disruption",
    "output_usability_cost",
    "social_or_professional_cost",
    "delay_or_missed_window",
    "unpredictability_cost",
    "repair_contrast",
    "changed_stakes",
    "accumulated_friction_tipping_point",
    "positive_discovery",
]


class ForcedEventMeta(StrictModel):
    artifact_kind: Literal["forced_event"]
    tool_status: Literal["world_toolbox"]
    persona_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    event_id: str = Field(pattern=r"^evolv_\d{2}$")
    source_acc_num: int = Field(ge=1)
    context: Literal["work", "personal"]
    target_cell: str = Field(pattern=r"^(work|personal)_[a-z0-9_]+$")
    pre_setting: str = Field(min_length=1)
    post_setting: str = Field(min_length=1)
    contract_class: Literal[
        "output_realization",
        "interaction_sequence",
        "action_scope",
        "triggered_interaction",
    ]

    @property
    def target_attribute(self) -> str:
        return self.target_cell.split("_", 1)[1]

    @model_validator(mode="after")
    def _validate_forced_identity(self) -> ForcedEventMeta:
        prefix = f"{self.context}_"
        if not self.target_cell.startswith(prefix):
            raise ValueError("target_cell must match context")
        if self.pre_setting == self.post_setting:
            raise ValueError("forced event must change the target setting")
        return self


class ForcedAtomicStep(StrictModel):
    step_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source_outline_phase_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    purpose: str = Field(min_length=1)
    actor: Literal["user", "pa", "tool", "environment"]
    action_kind: Literal[
        "signal",
        "clarify",
        "choose",
        "authorize",
        "read",
        "compare",
        "decide",
        "execute",
        "wait",
        "repair",
        "report",
    ]
    depends_on: list[str] = Field(default_factory=list)
    tool: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    effect: Literal["none", "read", "write", "external_write"] = "none"
    required_observation: str = Field(min_length=1)
    authorization: Literal["not_required", "required_before_step"] = "not_required"
    readback_step_id: str | None = None

    @model_validator(mode="after")
    def _validate_tool_actor(self) -> ForcedAtomicStep:
        if (self.actor == "tool") != (self.tool is not None):
            raise ValueError(
                f"step {self.step_id}: actor=tool exactly when a catalog tool is named"
            )
        if self.tool_arguments and self.tool is None:
            raise ValueError(f"step {self.step_id}: tool_arguments require a tool")
        return self


class ForcedStateChange(StateChange):
    """A forced event's state change, whose `target` is a LIST of files.

    One write is one state change and one `log_state_diff` record, but a handler
    may persist more than one file for it: `bill_payment_create` writes both
    `finance/bills.json` and `finance/payments.jsonl`. A single-path `target`
    could not say so, and a second row was not available -- that would raise
    `state_diff_count`, which the executor compares against the store's mutation
    count and not against the file count. So the paths go on the one row as a
    list.

    The list is the only shape. A one-file write writes a one-item list, never a
    bare string: a field that accepts two shapes is two shapes every reader has
    to handle, and the string form is an error with a message that names the
    list.
    """

    target: list[str] = Field(min_length=1)

    @field_validator("target", mode="before")
    @classmethod
    def _require_list_form(cls, value: Any) -> Any:
        if isinstance(value, str):
            raise ValueError(
                f"target must be a list of file paths, not a bare string: write "
                f"[{value!r}] instead of {value!r}"
            )
        return value


class ForcedTaskExecution(StrictModel):
    atomic_steps: list[ForcedAtomicStep] = Field(min_length=2)
    state_changes: list[ForcedStateChange] = Field(default_factory=list)
    side_interactions: list[SideInteraction] = Field(default_factory=list)


class CausalityEvidence(StrictModel):
    description: str = Field(min_length=1)
    atomic_step_ids: list[str] = Field(default_factory=list)
    beat_ids: list[str] = Field(default_factory=list)
    intervention_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_causality_evidence(self) -> CausalityEvidence:
        if not self.atomic_step_ids and not self.beat_ids and not self.intervention_ids:
            raise ValueError("causality stage requires concrete evidence ids")
        return self


class UserExperiencedConsequence(CausalityEvidence):
    practical_cost: str = Field(min_length=1)


class CausalityCatalyst(CausalityEvidence):
    source: Literal["third_party", "environment", "tool_result"]
    source_id: str = Field(min_length=1)


class UserRealization(CausalityEvidence):
    pass


class UserAuthoredBoundary(CausalityEvidence):
    post_setting_behavior: str = Field(min_length=1)


class PreferenceChangeCausality(StrictModel):
    primary_mechanism: PreferenceChangeMechanism
    supporting_mechanisms: list[PreferenceChangeMechanism] = Field(default_factory=list)
    target_caused_behavior: CausalityEvidence
    user_experienced_consequence: UserExperiencedConsequence
    catalyst: CausalityCatalyst | None = None
    user_realization: UserRealization
    user_authored_boundary: UserAuthoredBoundary

    @model_validator(mode="after")
    def _validate_mechanisms(self) -> PreferenceChangeCausality:
        if len(self.supporting_mechanisms) != len(set(self.supporting_mechanisms)):
            raise ValueError("supporting mechanisms must be unique")
        if self.primary_mechanism in self.supporting_mechanisms:
            raise ValueError("primary mechanism cannot repeat as supporting")
        return self


class PostContinuity(StrictModel):
    operational_boundary: str = Field(min_length=1)


class PostSettingCounterfactual(StrictModel):
    behavior: str = Field(min_length=1)
    ordered_actions: list[str] = Field(min_length=2)
    expected_avoidance: str = Field(min_length=1)


class PreferenceChange(StrictModel):
    actual_pre_setting_behavior: str = Field(min_length=1)
    post_setting_counterfactual: PostSettingCounterfactual
    causality: PreferenceChangeCausality
    post_continuity: PostContinuity


class InterventionVisibility(StrictModel):
    visible_to: list[Literal["pa", "user", "simulator", "god"]] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def _validate_visibility(self) -> InterventionVisibility:
        if len(self.visible_to) != len(set(self.visible_to)):
            raise ValueError("intervention visibility audiences must be unique")
        if "god" in self.visible_to and len(self.visible_to) != 1:
            raise ValueError("god_only intervention cannot be visible elsewhere")
        return self

    def includes(self, audience: Literal["pa", "user", "simulator", "god"]) -> bool:
        return audience in self.visible_to


class InterventionItem(StrictModel):
    intervention_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: Literal[
        "forced_pa_behavior",
        "tool_result",
        "environment_action",
        "third_party_message",
        "simulator_reaction",
        "god_setup",
    ]
    source_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    atomic_step_ids: list[str] = Field(default_factory=list)
    visibility: InterventionVisibility
    transcript_requirement: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_source_visibility(self) -> InterventionItem:
        source = {
            "forced_pa_behavior": "pa",
            "tool_result": "tool",
            "environment_action": "environment",
            "third_party_message": "third_party",
            "simulator_reaction": "simulator",
            "god_setup": "god",
        }[self.kind]
        if source == "god" and not self.visibility.includes("god"):
            raise ValueError("god source must remain god_only")
        return self


class ForcedReactionBranch(StrictModel):
    condition: str = Field(min_length=1)
    user_state: str = Field(min_length=1)
    user_instruction: str = Field(min_length=1)
    language_style: str = ""
    avoid: str = Field(min_length=1)


class ForcedBeat(AuthoringBeat):
    """A forced-event beat: the normal beat plus the fields the forced line adds.

    It SUBCLASSES `AuthoringBeat` rather than repeating its fields, so every
    normal-session beat check — `validate_actor_text_policy`,
    `_forbidden_actor_language`, the whole `outline_gate_binding` /
    `probe_gate_findings` stack — reads the same attributes here that it reads
    on a normal draft, and a field added to the normal beat reaches the forced
    beat by construction. `goal`, `active_skills`, `uses` and `if_not_met` come
    from the base: `active_skills` marks the beat where the pre-setting
    behaviour is performed, and `if_not_met` describes that behaviour.

    What the forced line adds: `user_state` (the actor's footing at the start of
    the beat, which the transcript writer plays), `atomic_step_ids` (forced
    beats cover the step list, which a live normal session works out at
    runtime), and `voice_examples` (semantic anchors for the authored
    transcript). `avoid` is narrowed from optional to required, and `reactions`
    from `NormalReactionBranch` to the forced branch that carries `user_state`.
    """

    user_state: str = Field(min_length=1)
    avoid: str = Field(min_length=1)
    atomic_step_ids: list[str] = Field(default_factory=list)
    reactions: dict[str, ForcedReactionBranch] = Field(default_factory=dict)
    voice_examples: list[str] = Field(default_factory=list, max_length=2)


class ForcedSimulation(StrictModel):
    scenario: str = Field(min_length=1)
    task: str = Field(min_length=1)
    life_context: SessionLifeContext
    task_facts: dict[str, Any] = Field(default_factory=dict)
    proactive_trigger: AuthoringProactiveTrigger | None = None
    proactive_contact_behavior: Literal[
        "not_applicable", "send", "suppress_then_later_discovery"
    ]
    assistance_emergence: AssistanceEmergence
    # Information gating, same field and same meaning as `NormalSessionDraft`:
    # a `task_facts` key the user reveals only when the named beat is played.
    reveal_at: dict[str, str] = Field(default_factory=dict)
    # No fixed floor. The old `min_length=7` predated the compressed shape and
    # forced acc_030/045/075 — whose normal siblings are exactly two beats — into
    # seven. The real requirement is that the story is complete, which
    # `validate_forced_event` checks against the beats the causality stages cite.
    beats: list[ForcedBeat]


class ForcedEventDraft(StrictModel):
    event_meta: ForcedEventMeta
    task_execution: ForcedTaskExecution
    simulation: ForcedSimulation
    intervention_plan: list[InterventionItem] = Field(min_length=1)
    preference_change: PreferenceChange

    @model_validator(mode="after")
    def _validate_forced_surfaces(self) -> ForcedEventDraft:
        runtime = self.task_execution
        simulator = self.simulation
        preference_change = self.preference_change
        step_ids = [step.step_id for step in runtime.atomic_steps]
        step_set = set(step_ids)
        if len(step_ids) != len(step_set):
            raise ValueError("atomic step ids must be unique")
        step_position = {step_id: index for index, step_id in enumerate(step_ids)}
        for step in runtime.atomic_steps:
            if len(step.depends_on) != len(set(step.depends_on)):
                raise ValueError(f"atomic step {step.step_id} repeats dependencies")
            if not set(step.depends_on) <= step_set:
                raise ValueError(f"atomic step {step.step_id} has unknown dependencies")
            if any(
                step_position[item] >= step_position[step.step_id]
                for item in step.depends_on
            ):
                raise ValueError(
                    f"atomic step {step.step_id} dependencies must appear earlier"
                )
            if step.effect in {"write", "external_write"} and not step.readback_step_id:
                raise ValueError(f"write step {step.step_id} requires readback")
            if step.readback_step_id and step.readback_step_id not in step_set:
                raise ValueError(f"atomic step {step.step_id} has unknown readback")

        for change in runtime.state_changes:
            if change.atomic_step_id not in step_set:
                raise ValueError(f"state change {change.change_id} has unknown step")
            if change.readback_step_id not in step_set:
                raise ValueError(f"state change {change.change_id} has unknown readback")
        for side in runtime.side_interactions:
            if side.needed_after_step_id not in step_set:
                raise ValueError(f"side interaction {side.interaction_id} has unknown step")

        beat_ids = [beat.beat for beat in simulator.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("forced beat ids must be unique")
        # No forced event has a fixed beat count, compressed or not. A
        # compressed attribute's normal session is exactly two beats (the probe
        # and one reaction) and its forced event starts from that base, but the
        # forced story then runs until the user has stated the new rule, so the
        # causality stages decide how many beats it needs — which
        # `validate_forced_event` checks against the beats they cite.
        for beat in simulator.beats:
            if len(beat.uses) != len(set(beat.uses)):
                raise ValueError(f"beat {beat.beat} repeats a uses key")
            unknown_uses = [
                key for key in beat.uses if key not in simulator.task_facts
            ]
            if unknown_uses:
                raise ValueError(
                    f"beat {beat.beat}.uses names no key in task_facts: "
                    f"{sorted(unknown_uses)}"
                )
        for fact_key, beat_id in simulator.reveal_at.items():
            if fact_key not in simulator.task_facts:
                raise ValueError(f"reveal_at.{fact_key} names no key in task_facts")
            if beat_id not in set(beat_ids):
                raise ValueError(
                    f"reveal_at.{fact_key} names an unknown beat {beat_id!r}"
                )
        covered_steps = {
            step_id for beat in simulator.beats for step_id in beat.atomic_step_ids
        }
        if covered_steps != step_set:
            raise ValueError("forced beats must cover every actual atomic step")
        for beat in simulator.beats:
            if not set(beat.atomic_step_ids) <= step_set:
                raise ValueError(f"beat {beat.beat} references unknown steps")

        intervention_ids = [item.intervention_id for item in self.intervention_plan]
        if len(intervention_ids) != len(set(intervention_ids)):
            raise ValueError("intervention ids must be unique")
        for item in self.intervention_plan:
            if not set(item.atomic_step_ids) <= step_set:
                raise ValueError(
                    f"intervention {item.intervention_id} references unknown steps"
                )

        causality = preference_change.causality
        evidence_groups = (
            causality.target_caused_behavior,
            causality.user_experienced_consequence,
            causality.user_realization,
            causality.user_authored_boundary,
            *([causality.catalyst] if causality.catalyst else []),
        )
        for evidence in evidence_groups:
            if not set(evidence.atomic_step_ids) <= step_set:
                raise ValueError("causality evidence references unknown atomic steps")
            if not set(evidence.beat_ids) <= set(beat_ids):
                raise ValueError("causality evidence references unknown exchanges")
            if not set(evidence.intervention_ids) <= set(intervention_ids):
                raise ValueError("causality evidence references unknown interventions")
        emergence = simulator.assistance_emergence
        if not set(emergence.evidence_beat_ids) <= set(beat_ids):
            raise ValueError("assistance emergence references unknown exchanges")
        if not set(emergence.evidence_atomic_step_ids) <= step_set:
            raise ValueError("assistance emergence references unknown steps")
        return self
