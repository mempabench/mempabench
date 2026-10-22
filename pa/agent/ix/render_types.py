"""IX wrapper interface freeze (P0) -- types, constants and serialised shapes only, no runtime logic.

This module is the single interface source between P1-P5 and P6a: the
implementations of the renderer (`render.py`), the collector and the gates are
not here. The one function, `turn_budget()`, is the §3.3 (part 5) formula
itself, not runtime wiring.

The twelve contracts (§7.1 P0; the numbers are the `contract N` referenced
throughout below):

- contract 1  lifecycle of the per-session setting and the per-turn payload (§3.0) --
  `IX_SELECTION_TOOL_NAMES` / `IX_RENDER_TOOL_NAMES` / `IX_CONTINUE_TOOL_NAMES` /
  `RENDER_PAYLOAD_FIELDS` / `IX_TOOL_NAME_PREFIX` / `TurnPayloads` / `RequiredContext`
- contract 2  the five projections and the full turn envelope (§2.4) --
  `PublicTurn` / `RuntimeHistory` / `MemoryProjection` / `TextJudgeProjection` / `AuditTrace`
- contract 3  injection, stripping and fail-closed validation of `interaction_item_id` (§3.3 (part 2)·A, §5.3 carrier C) --
  `PER_CALL_INJECTED_FIELDS` / `ToolCallRecord` / `PlanItem.bound_item_ids` / `Batch.call_binding`
- contract 4  the two-phase capability protocol (§3.5 (parts 1/4)) --
  `CapabilityStage` / `BoundaryState` / `STOP_REASON_IX_CONTRACT_FAILURE`
  / `BOUNDARY_FAILURE_REASONS`
- contract 5  the `capability_limit` error-code contract (§3.5 (part 3)) --
  `CAPABILITY_LIMIT_ERROR_CODE` / `CAPABILITY_LIMIT_TOOL_STATUS` / `ToolResultStatus`
  / `CAPABILITY_LIMIT_RESULT_KEYS` / `CapabilityLimitResult`
- contract 6  retired 2026-09-06. The number is left unused rather than renumbered
  so existing `contract 6` cross-references still resolve. The one rule it carried
  that survives belongs to no contract because it is unconditional: the three
  internal position markers `BATCH_MARKER_RE` / `QUESTION_MARKER_RE` /
  `CERTAINTY_MARKER_RE` are **always** stripped deterministically, whatever the
  setting
- contract 7  episode-scoped holds and no open hold at a step boundary (§4.1, §4.2) --
  `HoldEpisode` / `IXHold` / `EPISODE_ID_FIELDS` / `HoldEpisodeKind`
- contract 8  consent-aware continuation and the trusted sideband hard gate (§3.7) --
  `ReleaseDecision` / `ConsentDecision` / `ConsentRecord` / `CONSENT_DECISIONS_BLOCK`
  / `SIDEBAND_CONSENT_VALUES` / `HOLD_ROUNDS_HARD_CAP`
- contract 9  the beat budget constants (§3.3 (part 5)) --
  `BASE_TURNS` / `CLOSE_ROUNDS` / `GLOBAL_TURN_CAP` / `UNIT_CAP` / `turn_budget()`
  / `TURN_BUDGET_PARAMS` (a pure-data copy for the harness to inject)
- contract 10 the action-type total-coverage contract (§3.8) --
  `ActionType` / `UNKNOWN_TOOL_ACTION_TYPE` (an unknown tool fails closed to `external_action`)
- contract 11 the proactive render fields (§3.1 that row, §4.4) --
  `ProactiveDecision` / `HeldJSON.outreach_message` / `IXRenderJSON.proactive_decision`
- contract 12 the version boundary of `ix_contract_version` + `schema_version` --
  `IX_CONTRACT_VERSION` / `IX_RENDER_SCHEMA_VERSION` / `from_ix_render_metadata()`
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, NotRequired, TypedDict

# ---------------------------------------------------------------------------
# contract 12 · the version boundary
# ---------------------------------------------------------------------------

#: Identifier for the setting / spec / renderer contract that produced a render
#: record. The one recorded in the frozen RunPlan must equal it (§7.1 contract 12).
#: 2.0.0: the "No known preference" neutral setting left the taxonomy, so every
#: attribute's setting domain shrank. Every run frozen under 1.x is read-only —
#: it cannot be resumed or branched, because its selections were made against a
#: domain this code no longer offers.
#: 2.1.0: `IX_SELECTION_TOOL_NAMES` and `IX_PASSIVE_SPECS` took the document order
#: (`solution_breadth` after `task_expansion`). No setting and no description
#: changed, so `setting_universe_sha256` and `spec_sha256` stand; the renderer
#: contract hash covers this tuple's order, so it moves, and with it every run
#: frozen under 2.0.0 becomes read-only (owner ruling 2026-09-17).
IX_CONTRACT_VERSION = "ix-wrapper/2.1.0"

#: The schema version of the `metadata["ix_render"]` blob itself (§4.4).
#: 1.1.0: new optional top-level key `raw_blocks`, and new optional columns
#: `claim` / `strategy` on `certainties[]`.
#: **The reader matches on major**: inside one major only optional fields may be
#: added, so a 1.0.0 record is still valid under a 1.1.0 reader; a different
#: major always fails fast (contract 12).
IX_RENDER_SCHEMA_VERSION = "ix_render/1.1.0"


def _schema_major(version: str) -> str:
    """`"ix_render/1.1.0"` → `"ix_render/1"`. Minor and patch add optional fields only."""
    name, _, number = str(version).rpartition("/")
    return f"{name}/{number.split('.')[0]}"


class IXContractVersionError(ValueError):
    """Version mismatch. Contract 12: load / resume / branch / judge-pack all fail fast."""


# ---------------------------------------------------------------------------
# contract 9 · the beat budget constants (§3.3 (part 5))
# ---------------------------------------------------------------------------

#: The existing `MAX_TURNS_PER_BEAT` (`simulator/loop.py:72`).
BASE_TURNS = 3

#: The one forced-close round of §3.7 (part 4).
CLOSE_ROUNDS = 1

#: The global hard cap. Without it a beat with all four hold kinds active would
#: compute 14 turns, which the in-world envelope cannot hold (§3.3 (part 6)).
GLOBAL_TURN_CAP = 8

#: How many units each attribute may release this beat at most. **Per attribute,
#: independent of which setting was chosen** -- `Organize` (all at once) and
#: `One-at-a-time` (one batch per round) get exactly the same budget, so "how
#: many turns this beat got" cannot be worked back into the setting (§3.3 (part 5)).
UNIT_CAP: dict[str, int] = {
    "autonomy_level": 3,          # unit: plan item (one item, one gate = one round)
    "topic_management": 3,        # unit: batch
    "information_elicitation": 3, # unit: question batch
    "capability_boundary": 1,     # unit: boundary resolution
}

#: The four attributes that can produce a cross-turn hold (§3.3 (part 5) `HOLD_BEARING`).
HOLD_BEARING_ATTRIBUTES: frozenset[str] = frozenset(UNIT_CAP)

#: The two attributes whose payload declares call ids (§3.3 (part 2)·A): an
#: `IX_render_autonomy_level` plan step (`bound_item_ids`) and an
#: `IX_render_topic_management` batch (`call_binding`). The other two
#: hold-bearing attributes have no carrier for one, so a turn governed by them
#: alone can only fill `interaction_item_id` by inventing an id.
ID_DECLARING_ATTRIBUTES: frozenset[str] = frozenset(
    {"autonomy_level", "topic_management"}
)


def turn_budget(active_skills: Iterable[str]) -> int:
    """The §3.3 (part 5) per-beat budget formula. Pure, reading only this beat's `active_skills`.

    The formula (§3.3 (part 5), verbatim)::

        turn_budget(beat) = min(
            GLOBAL_TURN_CAP,                                  # global hard cap, proposed 8
            BASE_TURNS                                        # 3 (the existing MAX_TURNS_PER_BEAT)
              + Σ over attr ∈ (HOLD_BEARING ∩ beat.active_skills) : UNIT_CAP[attr]
              + CLOSE_ROUNDS                                  # 1 (the forced-close round, §3.7 (part 4))
        )

    **Who calls it** (owner's decision, how contract 9 lands): the formula and
    the constants have **exactly one definition**, here in `pa/`. **The harness
    injects it** -- `harness/` imports `turn_budget` (or reads
    `TURN_BUDGET_PARAMS` and computes it itself) and passes it (a callable, or
    the already-computed per-beat turn count) in when it builds the simulator;
    **`simulator/` never imports `pa/`**. The formula reads only
    `beat.active_skills` and no setting, so the budget leaks no GT (§3.3 (part 5)).

    One point lands per the owner's P0 acceptance example (see the deviation
    record in §9): a beat with no hold-bearing attribute returns `BASE_TURNS`
    (3) without `CLOSE_ROUNDS` -- `CLOSE_ROUNDS` is the §3.7 (part 4)
    forced-close round, and with no episode there is nothing to close, so there
    is no reason to widen.
    """
    units = sum(
        UNIT_CAP[attr]
        for attr in dict.fromkeys(active_skills)  # dedup: a repeated skill name is not billed twice
        if attr in UNIT_CAP
    )
    if units == 0:
        return BASE_TURNS
    return min(GLOBAL_TURN_CAP, BASE_TURNS + units + CLOSE_ROUNDS)


#: Contract 9's **pure-data export**: a serialisable copy of the four constants above.
#: [owner decision]: the **only definition** of the formula and the constants is
#: this module's `turn_budget()` and those four constants; `harness/` imports
#: `turn_budget` (or reads these params and computes it itself) and injects it --
#: a callable, or the already-computed per-beat turn count -- when it builds the
#: simulator; **`simulator/` never imports `pa/`**. This exists so the injection
#: point can pass pure data rather than a function (writable into the RunPlan,
#: across processes, into logs).
TURN_BUDGET_PARAMS: dict[str, Any] = {
    "base": BASE_TURNS,
    "close_rounds": CLOSE_ROUNDS,
    "global_cap": GLOBAL_TURN_CAP,
    "unit_cap": dict(UNIT_CAP),
}


# ---------------------------------------------------------------------------
# attribute groupings (§3.1, §3.1.1, §3.6)
# ---------------------------------------------------------------------------

#: The attributes the renderer applies a setting-specific transform to (§3.6,
#: the "what is not covered" row). `autonomy_level`'s gate part is here, while
#: its `Autonomous` instruction part goes through the §3.6 per-turn restatement.
RENDERED_ATTRIBUTES: frozenset[str] = frozenset({
    "reasoning_visibility",
    "uncertainty_expression",
    "process_visibility",
    "autonomy_level",
    "topic_management",
    "information_elicitation",
    "capability_boundary",
    "proactive_outreach",
})

#: Rendered attributes that **produce no cross-turn hold** -- they have no
#: `UNIT_CAP` entry and take no part in the §3.3 (part 5) budget (contract 9).
NON_HOLD_RENDERED_ATTRIBUTES: frozenset[str] = RENDERED_ATTRIBUTES - HOLD_BEARING_ATTRIBUTES

#: The six attributes the renderer does not transform, whose runtime effect the
#: §3.6 per-turn restatement carries instead.
PASS_THROUGH_ATTRIBUTES: frozenset[str] = frozenset({
    "tone_formality",
    "verbosity",
    "emotional_engagement",
    "guidance_level",
    "solution_breadth",
    "task_expansion",
})


# ---------------------------------------------------------------------------
# contract 1 · the per-attribute per-turn tool names (§3.0, §5.3 carrier B / D / E)
# ---------------------------------------------------------------------------

#: The prefix every internal IX tool shares. `§3.8`'s start-up coverage check
#: uses it to exclude the IX tools from `ToolRegistry.tool_names`
#: (`set(base) − {internal IX tools} ⊆ action map`), and the §2.4 projection 4
#: packing uses it to drop IX tools out of `task_tool_events`.
IX_TOOL_NAME_PREFIX = "IX_"

#: Carrier A: the per-session setting selection tools, one per attribute for all
#: 14 (§3.0, first row of that table).
#: **Literals, not derived from `pa.agent.ix.specs`**: this module is P0's
#: interface freeze point and depends on the standard library only; importing
#: `pa.agent.ix.specs` would also execute `pa/agent/ix/__init__.py` (which
#: imports `tools` / `registry` / `policy`), while P1's `tools.py` has to read
#: this module's payload shapes in turn -- that is a cycle. Consistency is
#: guaranteed by `tests/test_pa/test_ix_render_types.py` asserting against
#: `IX_PASSIVE_SPECS`, not by a runtime import. The order is the document order
#: of `configs/setting_definitions.yaml`.
IX_SELECTION_TOOL_NAMES: tuple[str, ...] = (
    "IX_tone_formality",
    "IX_verbosity",
    "IX_emotional_engagement",
    "IX_guidance_level",
    "IX_reasoning_visibility",
    "IX_uncertainty_expression",
    "IX_process_visibility",
    "IX_autonomy_level",
    "IX_proactive_outreach",
    "IX_task_expansion",
    "IX_solution_breadth",
    "IX_capability_boundary",
    "IX_information_elicitation",
    "IX_topic_management",
)

#: Carrier B: the per-turn payload tools, **one per attribute, no shared
#: envelope**. Six of them (§3.0).
IX_RENDER_TOOL_NAMES: dict[str, str] = {
    "reasoning_visibility": "IX_render_reasoning_visibility",
    "uncertainty_expression": "IX_render_uncertainty_expression",
    "process_visibility": "IX_render_process_visibility",
    "autonomy_level": "IX_render_autonomy_level",
    "topic_management": "IX_render_topic_management",
    "information_elicitation": "IX_render_information_elicitation",
}

#: Carrier D: the cross-turn release tools, which only consume an existing hold.
#: One per hold-bearing attribute, four in all (§3.7 (part 2)).
IX_CONTINUE_TOOL_NAMES: dict[str, str] = {
    attribute: f"IX_continue_{attribute}" for attribute in sorted(HOLD_BEARING_ATTRIBUTES)
}

#: Carrier E: the forced boundary-declaration tool (§3.5 (parts 1/4)).
IX_CAPABILITY_RESOLVE_TOOL = "IX_capability_resolve"

#: The one attribute a *runtime trigger* activates rather than a beat: the first
#: `capability_limit` tool result in a session puts it in the session's active
#: set for good (§3.5; pre-run-test audit §2, acc_026). Named here so the runner
#: and the loop agree on the string.
CAPABILITY_BOUNDARY_ATTRIBUTE = "capability_boundary"

#: Which `TurnPayloads` fields each `IX_render_<attribute>` carries (§3.0, that
#: table, and the per-field carrier notes inside `TurnPayloads`). P1 aligns its
#: schema and P2 its payload collection against this, so the field list is not
#: written out twice.
RENDER_PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {
    "IX_render_reasoning_visibility": ("user_facing_rationale",),
    "IX_render_uncertainty_expression": ("certainties",),
    "IX_render_process_visibility": ("completion_message",),
    "IX_render_autonomy_level": ("plan_id", "plan", "plan_done"),
    "IX_render_topic_management": ("batch_set_id", "item_count", "batches"),
    "IX_render_information_elicitation": ("elicitation_id", "questions", "assumptions"),
}


# ---------------------------------------------------------------------------
# contract 3 · the three per-call fields injected into task tool schemas (§5.3 carrier C)
# ---------------------------------------------------------------------------

#: process_visibility's per-call notice (§3.1 that row). Stripped before the real tool is forwarded.
IX_NOTICE_FIELD = "_ix_notice"

#: The one binding id, written by the model itself into the function arguments
#: (§3.3 (part 2)·A). Stripped before forwarding.
INTERACTION_ITEM_ID_FIELD = "interaction_item_id"

#: The ask, verbatim, written by the PA at the moment it issues this real call
#: (§3.3 (part 2)·C). Stripped before forwarding.
IX_CONFIRMATION_MESSAGE_FIELD = "_ix_confirmation_message"

#: All three are injected onto every task tool's schema keyed on
#: **`session_active_ix`** (not `turn_active_ix`), and stripped together before
#: `execute()` forwards (§3.0, §5.3 carrier C).
PER_CALL_INJECTED_FIELDS: tuple[str, str, str] = (
    IX_NOTICE_FIELD,
    INTERACTION_ITEM_ID_FIELD,
    IX_CONFIRMATION_MESSAGE_FIELD,
)


# ---------------------------------------------------------------------------
# contract 8 · the simulator-side PA-blind private block name (§3.7 (part 3))
# ---------------------------------------------------------------------------

#: The fourth private block in the simulator's per-turn output, a sibling of
#: `<factual_check>` / `<turn_assessment>`.
#: **Never into `<message>`, never into the simulator's `_history`, never
#: anywhere in the PA prompt.**
#: It reaches the wrapper over the existing metadata channel under a typed key
#: of the same name (the harness is the only wiring point).
CONSENT_DECISIONS_BLOCK = "consent_decisions"


# ---------------------------------------------------------------------------
# enums / Literals
# ---------------------------------------------------------------------------

#: Contract 10 · action types (`pa/agent/ix/action_types.py`).
ActionType = Literal["read", "draft", "internal_write", "external_action"]

#: Contract 10 · the runtime floor for an unknown tool: changed from `"read"` to
#: the most tightly gated class, because **"unknown means exempt" is not
#: allowed** (§3.8).
UNKNOWN_TOOL_ACTION_TYPE: ActionType = "external_action"

#: Contract 5 · the stable status codes for a tool result. `capability_limit`
#: coexists with, but stays distinguishable from, the existing error verdict of
#: `_tool_result_error` (§3.5 (parts 2/3)).
ToolResultStatus = Literal["ok", "error", "capability_limit"]

#: Contract 5 · only this can send the wrapper into `IX_capability_resolve` (§3.5 (part 3)).
CAPABILITY_LIMIT_ERROR_CODE = "capability_limit"

#: Contract 5 · the status value on the `log_tool` side (§3.5 (part 3)).
CAPABILITY_LIMIT_TOOL_STATUS: ToolResultStatus = "capability_limit"

#: Contract 4 · the boundary-kind enum (§3.5 (part 1)). Not in the enum =
#: illegal payload -> contract failure.
BoundaryKind = Literal[
    "missing_data",
    "missing_capability",
    "action_not_permitted",
    "external_dependency",
]

#: Contract 4 · capability_boundary's two-phase protocol (the three stages of §3.5 (part 1)).
CapabilityStage = Literal["select_setting", "declare_boundary", "apply_setting"]

#: Contract 5 · the **three required keys** of a `capability_limit` tool result (§3.5 (part 3)).
#: The plan's shape is `{"ok": false, "error_code": "capability_limit", "boundary_kind": …}`;
#: the `log_tool` side also records `status="capability_limit"`
#: (`CAPABILITY_LIMIT_TOOL_STATUS`), so that it **coexists with, yet stays
#: distinguishable from**, the existing error verdict of `_tool_result_error`
#: (`pa/agent/runner.py:37-55`) -- only this shape can send the wrapper into
#: `IX_capability_resolve`.
CAPABILITY_LIMIT_RESULT_KEYS: tuple[str, str, str] = ("ok", "error_code", "boundary_kind")


class CapabilityLimitResult(TypedDict):
    """Contract 5 · the capability-boundary result shape a state tool returns (§3.5 (part 3); P0 fixes the shape, P6a lands it).

    The production rule (§3.5 (part 3)'s source table, not optional): only a
    read/retrieve **whose argument is an exact, already-resolved resource id**
    may produce it, and only when the fixture **explicitly marks that entry
    unobtainable**; a free-text query search **always** gives an ordinary empty
    result (`status="ok"`), and an id the model made up still goes to
    `not_found`. The verdict looks only at `error_code`, and `boundary_kind`
    must fall inside the `BoundaryKind` enum -- outside it = illegal payload ->
    contract failure (§3.5 (part 6)).
    """

    ok: Literal[False]
    error_code: Literal["capability_limit"]
    boundary_kind: BoundaryKind
    detail: NotRequired[str]
    """An optional human-readable note. **It takes no part in any verdict**, and goes to `audit_trace` only."""

#: Contract 3 / 8 · the gate stage recorded when a gate holds a call (§3.3
#: (part 2)·D; the four gates share one piece of code and only swap the status).
HoldStatus = Literal[
    "held_for_confirmation",  # autonomy_level
    "held_for_handoff",       # capability_boundary
    "held_for_batch",         # topic_management
    "held_for_questions",     # information_elicitation
    "held_for_binding",       # a contract violation, belonging to no gate
]

#: Attribute -> the `status` that attribute's gate records (§3.3 (part 2)·D).
HELD_STATUS_BY_ATTRIBUTE: dict[str, HoldStatus] = {
    "autonomy_level": "held_for_confirmation",
    "capability_boundary": "held_for_handoff",
    "topic_management": "held_for_batch",
    "information_elicitation": "held_for_questions",
}

#: **Reason** -> `status`, which wins over the attribute (added after the round 7 live run).
#: `held_for_confirmation` means "the user has been asked and we are waiting for
#: an answer", so it **can only** come from the path that really opened a gate.
#: In live run `v7r06` t18, all six `binding_invalid` rows were written as
#: `held_for_confirmation` + `gate: null` + `asked: false` -- the user was
#: neither asked nor told, and three reads simply vanished. These reasons open
#: no gate (see `_NO_GATE_HOLD_REASONS`), so they get their own status instead
#: of borrowing a gate's name.
HELD_STATUS_BY_REASON: dict[str, HoldStatus] = {
    "binding_invalid": "held_for_binding",
    "ix_selection_invalid": "held_for_binding",
    "release_spent": "held_for_binding",
    "batch_not_released": "held_for_batch",
    "batch_undivided": "held_for_batch",
    "questions_gate": "held_for_questions",
}

#: Hold reasons that open no gate and ask the user nothing. They may **never**
#: be recorded as `held_for_confirmation` -- that status means "already asked,
#: waiting for an answer". `HELD_STATUS_BY_REASON` guarantees the invariant
#: structurally (it does not depend on the call site passing the right
#: attribute), and the tests check the full `reason × attribute` cross product.
NO_ASK_HOLD_REASONS: frozenset[str] = frozenset({
    "binding_invalid", "ix_selection_invalid", "release_spent",
    "batch_not_released", "batch_undivided", "questions_gate",
})


def held_status(attribute: str, reason: str = "") -> HoldStatus:
    """The `status` of one held record: **reason first, attribute second** (§3.3 (part 2)·D, restated in round 7).

    One entry point, so the collector's gate path and the runner's back-fill
    path cannot give two answers for the same `(attribute, reason)` -- which is
    exactly where round 5 finding 4 came from.
    """
    by_reason = HELD_STATUS_BY_REASON.get(reason)
    if by_reason is not None:
        return by_reason
    return HELD_STATUS_BY_ATTRIBUTE.get(attribute, "held_for_binding")

#: Contract 7 · the four kinds of hold episode (§4.1).
HoldEpisodeKind = Literal["plan", "batch_set", "elicitation", "boundary"]

#: Contract 8 · the hard cap on one episode's `rounds_open` (§3.7 (part 4)). On
#: the round that reaches it: every item still not `release`d is marked
#: `cancelled` (**never** released just because "time is up"),
#: `ix_hold_forced_close` fires, and `IX_continue_<attribute>.closing_message`
#: **becomes required** -- one closing sentence must render into `public_turn`
#: telling the user these things were not done.
#: It is also the round the §3.3 (part 5) budget formula has to reach: any beat
#: with one 3-unit attribute budgets `min(8, 3+3+1) = 7`, enough to hit round 4.
HOLD_ROUNDS_HARD_CAP = 4

#: Contract 7 · which payload field each attribute's episode id is written in (§4.1).
EPISODE_ID_FIELDS: dict[str, str] = {
    "autonomy_level": "plan_id",
    "topic_management": "batch_set_id",
    "information_elicitation": "elicitation_id",
    "capability_boundary": "boundary_id",
}

#: Contract 8 · the PA's verdict on each held id inside `IX_continue_<attribute>` (§3.7 (part 2)).
#: `release` is the PA's **intent**, not permission to execute (§5.3 carrier D).
ReleaseDecisionValue = Literal["release", "keep", "skip"]

#: Contract 8 · the PA-blind sideband's verdict on each held id shown to the user (§3.7 (part 3)).
#: `missing` appears in the audit record only, and means the id was absent from
#: the sideband -- fail-closed and treated as `keep`, the same as `deny` /
#: `unclear`.
ConsentDecisionValue = Literal["allow", "deny", "unclear", "missing"]

#: Contract 8 · the three values the simulator's `<consent_decisions>` block may
#: carry **on the input path** (§3.7 (part 3)).
#: `"missing"` is a member of `ConsentDecisionValue`, but it is **only an audit
#: record value** -- meaning "this held id did not appear in this turn's
#: sideband at all", a state the wrapper infers for itself, not a verdict the
#: simulator may write. **The sideband input path must reject `"missing"`**
#: (treating it as an illegal payload), or the model side would gain a channel
#: for claiming "I gave no verdict".
#: All three, like absence, are fail-closed and treated as `keep`; only `allow`
#: releases an `external_action`.
SIDEBAND_CONSENT_VALUES: tuple[str, str, str] = ("allow", "deny", "unclear")

#: Contract 11 · the action on a proactive event turn (§3.1 that row).
ProactiveAction = Literal["send", "suppress"]

#: §3.1 uncertainty row: the confidence level on a `[[c:…]]` marker.
CertaintyLevel = Literal["high", "medium", "low"]

# --- the three internal position markers in the prose (§3.4 (part 1), §3.1 uncertainty / elicitation rows) ---
#: They are **position information**, written by the model into its prose to
#: divide it: `[[batch:b1]] …prose… [[batch:b2]] …prose…`. On release the marker
#: is stripped before delivery; **stripping is unconditional and happens under
#: every setting**, and not one may reach the user verbatim. The capture group
#: is the id in the matching payload
#: (`Batch.batch_id` / `Question.id` / `Certainty.ref`).
_MARKER_ID = r"[A-Za-z0-9_.-]+"


def _marker_re(short: str, long: str) -> re.Pattern[str]:
    """One marker kind, in every shape the model actually writes.

    The canonical form is `[[q:q1]]`, and it is what the tool descriptions ask
    for. Live, two acc_009 runs wrote `[[q1]]` on every question instead, so the
    strict `\\[\\[q:(id)\\]\\]` pattern matched nothing: no question was ever
    recorded as released and the raw bracket shipped to the user verbatim.
    Reading the lenient shapes costs nothing and loses no information — the
    capture group is still the payload id.

    Accepted: `[[q:q1]]`, `[[q1]]`, `[[q:1]]`, `[[question:q1]]`. The leading
    lookahead keeps the kinds apart: a bare `[[b1]]` is a batch and can never be
    read as a question, because the token has to start with the kind's letter
    when the prefix is left off.
    """
    return re.compile(
        rf"\[\[(?={short})(?:{short}(?:{long[len(short):]})?:)?({_MARKER_ID})\]\]"
    )


#: `[[batch:b1]]` / `[[b1]]` -- `IX_render_topic_management.batches[].batch_id` (§3.4 (part 1)).
#: `IX_render_topic_management`'s required predicate reads it: a `[[batch:…]]`
#: in the prose makes the call mandatory (§3.0).
BATCH_MARKER_RE = _marker_re("b", "batch")

#: `[[q:q1]]` / `[[q1]]` -- `IX_render_information_elicitation.questions[].id` (§3.1 that row).
QUESTION_MARKER_RE = _marker_re("q", "question")

#: `[[c:c1]]` / `[[c1]]` -- `IX_render_uncertainty_expression.certainties[].ref` (§3.1 that row).
CERTAINTY_MARKER_RE = _marker_re("c", "certainty")

#: Any `[[…]]` token the three above did not recognise. Letting one reach the
#: user verbatim is absolutely barred (§3.1.1, the "internal control markers"
#: row), so after the three kinds are stripped this sweeps up once more, and
#: what it strips goes to `trace`.
UNRECOGNISED_MARKER_RE = re.compile(r"\[\[([^\[\]]*)\]\]")

#: Why this turn ended. `ix_gate` = a gate fired (§3.3 (part 2)·D);
#: `ix_hold` = a batch / question hold (§3.4 (part 3), which closes on the spot
#: the same way a gate does); `ix_contract_failure` = `IX_capability_resolve`
#: missing or illegal a second time (§3.5 (part 6)).
StopReason = Literal["ix_gate", "ix_hold", "ix_contract_failure"]
STOP_REASON_IX_GATE: StopReason = "ix_gate"
STOP_REASON_IX_HOLD: StopReason = "ix_hold"
STOP_REASON_IX_CONTRACT_FAILURE: StopReason = "ix_contract_failure"

#: The diagnostic event names the wrapper emits (§3.3 (part 2)·A, §3.3 (part 2)·C,
#: §3.5 (part 6), §3.7 (parts 2/3/4), §4.3).
IX_EVENT_NAMES: tuple[str, ...] = (
    "ix_binding_invalid",
    "ix_gate_no_message",
    "ix_render_payload_missing",
    #: Carries `call_index` as well as `tool`/`tools`: the collector's events
    #: are appended to `tool_events` *after* every executed call
    #: (`runner.py`'s `tool_events.extend(collector.events)`), so the position
    #: of a ladder event in that list says nothing about when it fired. Three
    #: regress21 reads took the tail position for chronology and concluded the
    #: wrapper had pinned a carrier the turn already had (`acc_007` t2,
    #: `acc_026` t4). `call_index` is the number of calls recorded when the pin
    #: went out, on the same scale as `ToolCallRecord.order`, so the record says
    #: which calls preceded it.
    "ix_render_payload_forced",
    "ix_release_missing",
    "ix_release_folded",
    "ix_release_spent",
    "ix_release_inherited",
    #: Owner ruling 2026-09-08: the floor let an external act through on a
    #: standing authorisation the user spoke in the immediately preceding turn,
    #: so no ask was needed and none was made. Carries `tool` and the item id
    #: the act ran under.
    "ix_release_preauthorised",
    #: Its counterpart: the standing authorisation was still unspent when the
    #: turn ended, so it lapsed rather than carrying into the next one. Carries
    #: `tool` only -- there is no item to name.
    "ix_preauthorisation_expired",
    "ix_consent_denied",
    "ix_consent_unresolved",
    "ix_consent_unasked",
    "ix_consent_tool_changed",
    "ix_selection_invalid",
    "ix_selection_duplicate",
    "ix_batch_undivided",
    "ix_batch_rebound",
    "ix_questions_undivided",
    "ix_questions_unmarked",
    "ix_hold_forced_close",
    "ix_hold_closing_tool_call_ignored",
    "ix_close_no_message",
    "ix_completion_without_content",
    "ix_markup_stripped",
    "ix_dsml_calls_recovered",
    "ix_certainty_claim_not_found",
    "ix_certainties_all_dropped",
    "ix_leadin_dropped",
    "ix_rationale_work_log",
    "ix_claims_unexecuted_action",
    "ix_boundary_unresolved",
    "ix_prose_ask_unbound",
    "ix_empty_reply",
    "beat_gate_turn",
    #: Not an IX diagnostic: the provider cut the reply off at `max_tokens` with
    #: nothing in it, and the runner answered it instead of counting it as a
    #: ladder miss (regress12 acc_026). Registered here because it is emitted
    #: with the same `ix_event` stamp and has to be read and banned with the rest.
    "provider_truncated",
    #: Same family, one step later: the turn's *last* response was the cut-off
    #: one, so the prose it would have delivered was never finished. The tail
    #: after the last sentence end is dropped before the turn ships, and this
    #: carries `completion_tokens` and `dropped_chars` (regress16 acc_006, whose
    #: two PA turns both reached the user as a mid-tool-sequence aside).
    "turn_truncated",
    #: Same family: the provider marked the response an error, so its `content`
    #: is the gateway's error body and not a reply. One more request with the
    #: same messages is spent on it before the turn ends on the error path
    #: (regress12 acc_003, whose error body shipped as the PA's whole turn).
    "provider_error_retried",
    #: Also not an IX diagnostic: one tool failed identically three times in a
    #: turn on a value the model was guessing, so it was stopped for the rest of
    #: the turn and the model told to ask the user (regress12 acc_004).
    "tool_failure_strike",
    #: Its other half (owner ruling 2026-09-08): the same tool answered with an
    #: error on three separate iterations of one turn, so the runner stopped
    #: re-prompting and shipped the turn. Carries `tool`, `count`, `turn_index`
    #: and `beat`; the turn ends `completed`, not on a contract failure.
    "tool_retry_cap_reached",
    #: Also not an IX diagnostic: a per-session selection tool called on a beat
    #: that no longer offers it, after the session already recorded the setting
    #: (regress20 acc_027). The call is answered and dropped -- it spends no
    #: repair budget and changes no selection. Carries `attribute`, `setting`,
    #: `turn_index`, `beat` and `call_index`.
    "ix_reselection_ignored",
)

#: Contract 4 · the value domain of `reason` in an `ix_boundary_unresolved` event
#: (§3.5 (part 4), step 6).
#: `IX_capability_resolve` still not called in the second response after the
#: reminder = `"missing"`; called but with a field missing / the wrong type /
#: a `boundary_kind` outside the `BoundaryKind` enum = `"invalid_payload"`.
#: The consequence is the same either way: the turn ends with
#: `stop_reason = "ix_contract_failure"`, no further task call runs, no scorable
#: reply is produced, the step fails and may be retried, and **no state change
#: is left behind** (changed in round 6, replacing round 3's fail-open).
#: The event also carries `attempts` (always 2).
BOUNDARY_FAILURE_REASONS: tuple[str, str] = ("missing", "invalid_payload")

# The setting value domains of the rendered attributes (§3.1's master table +
# §6's three reductions). Each one is exactly `IX_PASSIVE_SPECS[attribute].settings`
# — every selection names a taxonomy setting, and there is no extra option
# outside the taxonomy.
# The three reductions (reasoning_visibility drops `Summarize`,
# proactive_outreach drops `Medium`, topic_management drops `Follow user's
# flow`) land in `specs.py` via P1 / P6b; what is frozen here is the domain
# after the reduction.
ReasoningVisibilitySetting = Literal["Show", "Hide"]
UncertaintyExpressionSetting = Literal["Express", "Hide"]
ProcessVisibilitySetting = Literal["Silent", "Bookend", "Full narration"]
AutonomyLevelSetting = Literal["Reactive", "Self-directed", "Autonomous"]
CapabilityBoundarySetting = Literal["Suggest alternatives", "Find and hand off"]
InformationElicitationSetting = Literal["Infer", "Structured", "Iterative"]
TopicManagementSetting = Literal["Organize", "One-at-a-time"]
ProactiveOutreachSetting = Literal["Low", "High"]


# ---------------------------------------------------------------------------
# contract 7 · the data shapes of the cross-turn cache (§4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanItem:
    """One item of `IX_render_autonomy_level.plan` (§3.3 (part 1))."""

    id: str
    intent: str
    """**At the level of intent**, with no exact arguments -- later steps depend
    on earlier results, and the arguments come only at the real call."""
    is_key: bool = False
    """Whether this is a **decision point**: making a choice, causing a real
    state change, or having an outward effect. Not "is this step important".
    Read by the wrapper under `Self-directed` only, and **only in effect for
    `internal_write`** (§3.3 (part 2)·B)."""
    bound_item_ids: list[str] = field(default_factory=list)
    """The `interaction_item_id`s this item claims, written out explicitly by the model (§3.3 (part 2)·A)."""


@dataclass(frozen=True)
class Batch:
    """One batch of `IX_render_topic_management.batches` (§3.4)."""

    batch_id: str
    label: str
    body: str
    """The passage the `[[batch:b1]]` marker delimits in the prose. While
    unreleased it is **kept verbatim** in `audit_trace`."""
    call_binding: list[str] = field(default_factory=list)
    """**`interaction_item_id` only** (§3.4 (part 3)) -- not the provider `tool_call_id`, and not `plan_item_id`."""


@dataclass(frozen=True)
class Question:
    """One entry of `IX_render_information_elicitation.questions` (§3.1 that row), marked `[[q:q1]]` in the prose."""

    id: str
    text: str
    about: str


@dataclass(frozen=True)
class Assumption:
    """One entry of `IX_render_information_elicitation.assumptions`. Must be non-empty under `Infer` (§3.1 that row)."""

    about: str
    assumed: str


@dataclass(frozen=True)
class Certainty:
    """One entry of `IX_render_uncertainty_expression.certainties` (§3.1 that row).

    `claim` is the **production channel**: in the tool call the model writes the
    sentence it is about to put verbatim in its reply, and the renderer finds it
    in the prose and inserts the parenthesis after it. Tool calls are produced
    **before** the prose, so the old route of "place a `[[c:ref]]` marker in
    prose not yet written" had a constant hit rate of 0 in live runs;
    `[[c:*]]` markers are still honoured (used when the model happens to write
    one), but they are no longer the only channel.
    """

    ref: str
    level: CertaintyLevel
    claim: str = ""
    """The passage in this turn's reply this confidence refers to, verbatim.
    "" = the `[[c:ref]]` marker channel only."""


@dataclass(frozen=True)
class GateRecord:
    """The record of a gate firing. §4.1's four fields + the two more the §4.4 `gate` blob carries."""

    attribute: str
    item_id: str
    reason: str
    """E.g. `"external_action_floor"`. **`audit_trace` only** -- the text-plane
    judge never sees it (§2.4)."""
    asked_at_turn: int = 0
    action_type: ActionType = UNKNOWN_TOOL_ACTION_TYPE
    """§4.4 `gate.action_type`. The default is contract 10's fail-closed floor --
    an action that was never explicitly classified is recorded as the most
    tightly gated class, because **"unfilled means exempt" is not allowed** (§3.8)."""
    message_source: str = IX_CONFIRMATION_MESSAGE_FIELD
    """§4.4 `gate.message_source`: the field the ask came from, since the
    renderer **never writes for the PA** (§3.3 (part 2)·C)."""


@dataclass(frozen=True)
class BoundaryState:
    """The product of capability_boundary's two-phase protocol (§3.5), submitted by `IX_capability_resolve`."""

    blocked_goal: str
    boundary_kind: BoundaryKind
    recovery_remaining: list[PlanItem] = field(default_factory=list)
    """The alternative path; shaped like `PlanItem` but independent of it (§3.5 (part 1))."""
    handoff_message: str = ""
    """The sentence delivered to the user verbatim when `Find and hand off`
    opens its gate, written by the PA."""


@dataclass(frozen=True)
class ProactiveDecision:
    """Contract 11 · proactive_outreach's render decision (§3.1 that row, §4.4).

    Rendering happens before persistence: this is the **only source** for
    `outreach_decision` / `outreach_delivered` / history / memory, which no
    longer fish it straight out of the tool event.
    """

    action: ProactiveAction
    setting: str
    """The setting chosen for this event turn: always `Low` or `High`."""
    message: str
    """The reminder the model always writes -- non-empty under both `send` and
    `suppress` (§3.1 that row). Under `suppress` it goes **only** to
    `held.outreach_message`, and `public_turn` is the empty string (§4.4)."""


@dataclass(frozen=True)
class HoldEpisode:
    """The hold state of one task / question set. **It belongs to the episode, not the session** (§4.1, contract 7)."""

    episode_id: str
    """`plan_id` / `batch_set_id` / `elicitation_id` / `boundary_id`, written by the PA itself."""
    attribute: str
    opened_at_turn: int = 0
    rounds_open: int = 0
    """Consecutive unconverged rounds **for this episode**, hard-capped at
    `HOLD_ROUNDS_HARD_CAP` (§3.7 (part 4)). The round that reaches it triggers
    the forced close: items not `release`d go to `cancelled`, and
    `closing_message` becomes required."""
    closed: bool = False
    """**Closes this episode only**, not the attribute -- a new task opens a new
    id and can go round again (§4.1)."""
    last_gate: GateRecord | None = None
    plan_remaining: list[PlanItem] = field(default_factory=list)      # autonomy_level
    batches_held: list[Batch] = field(default_factory=list)           # topic_management
    questions_held: list[Question] = field(default_factory=list)      # information_elicitation
    boundary: BoundaryState | None = None                             # capability_boundary
    calls_held: list[dict[str, Any]] = field(default_factory=list)
    """Calls held before execution, in `HeldItem`'s serialised shape. **They flow
    to `audit_trace` only**, generate nothing into `runtime_history`, and leave
    no synthesised tool result to fill in (§3.3 (part 2)·D)."""
    cancelled: list[dict[str, Any]] = field(default_factory=list)
    """Items cancelled on hitting the 4-round hard cap (§3.7 (part 4)); **never released**."""
    executed_item_ids: list[str] = field(default_factory=list)
    """`interaction_item_id`s that have spent their release allowance: one
    release buys one world change.

    The same id carrying another `internal_write` / `external_action` in a later
    response is held again (`reason="release_spent"`) -- otherwise one `allow`
    could release arbitrarily many outward actions."""


@dataclass(frozen=True)
class IXHold:
    """The "attribute -> currently open episode" index + the audit list of finished episodes (§4.1).

    Lifecycle (§4.2, contract 7): every episode must be `closed` / `cancelled`
    before the session ends, or the step fails and **publishes no checkpoint**;
    success / cancel / exception / same-step retry all clear `_ix_hold_cache` +
    `_ix_selection_cache` + the rest of the IX active state atomically in **one
    `finally`**. IX state in a mid-session checkpoint is out of scope.
    """

    open_episodes: dict[str, HoldEpisode] = field(default_factory=dict)
    finished_episodes: list[HoldEpisode] = field(default_factory=list)
    batch_bindings: dict[tuple[str, str], tuple[str, ...]] = field(
        default_factory=dict
    )
    """acc_023 — every `(batch_set_id, batch_id)` this session has declared, with
    the calls it bound the first time.

    A batch set that releases everything leaves no episode behind (§4.1), so
    `open_episodes` and `finished_episodes` between them remember nothing about
    it: live t18 declared `bs_001` with `b1` = "Calendar deadline" (binding
    `cal_1`), released both batches, and t20 re-declared `bs_001` with the same
    `b1` relabelled "Send confirmation email" (binding `snd_1`). One batch id,
    two different calls, and the wrapper released it. First binding wins — the
    record is what a redeclaration is checked against.

    The set is half the key. Keying by `batch_id` alone made a *new* set re-using
    `b1` a rebinding too, and that is the move the `batch_set_id` description
    asks for once a set has closed."""

    act_releases: dict[str, str] = field(default_factory=dict)
    """Owner ruling 2026-09-08 — `tool -> the item id whose release is still
    unspent on it`.

    §3.3 (part 3) says the user authorised *an action*, not an id, and the tool
    name is what identifies the action. The id, though, is minted fresh every
    time the PA re-plans: live acc_039 asked authorisation for one clinic
    booking three times, because a rewritten reason (`book_appt_1` ->
    `book_appt_2`) and a store refusal (`book_appt_2` -> `book_final`) each
    produced a new plan item on the same `appointment_create`, and each new id
    walked back into the floor gate.

    So a release survives the re-mint: a new external-act item on a tool listed
    here inherits it instead of asking again. What ends it is the act
    succeeding -- the entry is dropped by the first `ok` call on that tool, so a
    genuine *second* act (which by definition comes after a success) still gets
    its own ask -- or the user withdrawing it (`deny` on the sideband, `skip`
    from the PA) while the released item is still held."""


# ---------------------------------------------------------------------------
# contract 3 / 8 · this turn's calls, held records and authorisation verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallRecord:
    """One task tool call, recorded in order by the collector (§2.2 component 1).

    `notice` / `interaction_item_id` / `ask_text` are the values of the three
    per-call injected fields, already stripped before the real tool is forwarded
    (§5.3 carrier C).
    """

    order: int
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""
    """The protocol id assigned by the provider (`pa/providers/base.py:14-20`).
    It is what §2.4's `executed_call_ids` / `held_call_ids` refer to -- the model
    **cannot refer to it**, which is why binding goes through
    `interaction_item_id` instead (§3.3 (part 2)·A)."""
    notice: str = ""                    # `_ix_notice`
    interaction_item_id: str = ""       # `interaction_item_id`
    ask_text: str = ""                  # `_ix_confirmation_message`
    action_type: ActionType = UNKNOWN_TOOL_ACTION_TYPE
    """Contract 10's fail-closed floor (§3.8): the default is not `"read"` -- a
    call `tool_action_type()` has not classified yet is recorded as the most
    tightly gated `external_action`."""
    status: ToolResultStatus = "ok"
    """One of contract 5's three stable status codes (§3.5 (parts 2/3)). **No
    longer an open string**: `capability_limit` has to stay reliably
    distinguishable from an ordinary error, so the domain cannot be `str`."""
    error: str = ""
    executed: bool = False


@dataclass(frozen=True)
class HeldItem:
    """One call a gate held **before execution** (§3.3 (part 2)·D; §7.1 freezes the shape along with it).

    Its only destination is `audit_trace`: the judge has to be able to tell "the
    PA did not do it" from "the PA did it but it was held".
    """

    status: HoldStatus
    interaction_item_id: str
    attribute: str
    tool: str
    plan_item_id: str = ""
    episode_id: str = ""
    batch_id: str = ""
    reason: str = ""
    """§4.4 `held.calls[].reason`, e.g. `"external_action_floor"`."""
    repaired: bool = False
    """The refusal was answered inside the same turn and the resent call ran.

    Only a fail-closed refusal can be repaired in-turn (`binding_invalid`): the
    model is asked once to resend the batch, and a resend that executes leaves
    the original rows describing work that did happen. Live `acc_009` read them
    as steps still outstanding and told the user they could not be carried out.
    The rows are kept — the refusal is real history — and marked instead."""
    repaired_by: str = ""
    """`tool_call_id` of the retry call that answered the refusal."""
    asked: bool = False
    """The ask **rendered out this turn really named** this id (the gate's
    `item_id`, plus any id appearing verbatim in the ask's prose that the
    collector can map back). §3.7 (part 3)'s criterion is "every held id shown to
    the user", not "every held id still on hold": in a live run one ask asked
    only about cancelling a booking, yet the sideband came back with `allow` for
    all six ids. An `allow` for an id that was not asked about is ignored
    (`ix_consent_unasked`), and that id stays held waiting for its own ask."""


@dataclass(frozen=True)
class HeldNarration:
    """The prose written in the **same assistant message** as a call the gate held (§3.3 (part 2)·D addendum).

    It narrates something that did not happen -- live, `v3g03` t12's `📧 Email
    confirmation sent to Dr. Okafor` while `email_01` was being held. So it goes
    where held prose goes: into `audit_trace`'s `held.narration` alone, with the
    `public_turn` / `runtime_history` / memory projections seeing not one byte
    of it.
    """

    item_id: str
    text: str


@dataclass(frozen=True)
class ReleaseDecision:
    """The PA's verdict on one held id inside `IX_continue_<attribute>.decisions` (§5.3 carrier D).

    **Every** held id has to appear; missing any one voids this turn's response
    (§3.7 (part 2)). `release` is the PA's **intent**, not permission to execute
    -- whether an `external_action` item actually goes through is up to the
    sideband.
    """

    id: str
    decision: ReleaseDecisionValue
    user_evidence: str = ""
    """An **audit field** (downgraded in round 6): the whitespace-normalised
    substring check still runs and still goes to `audit_trace`, but **its result
    no longer decides the release** (§3.7 (part 3))."""


@dataclass(frozen=True)
class ConsentDecision:
    """Contract 8 · one entry of the PA-blind sideband `<consent_decisions>` (§3.7 (part 3)).

    Produced by the **simulator** (which knows whether it consented or refused
    this turn), one entry for **every** held id that this turn's rendered ask
    showed the user. With no held id shown, the block is an empty list.

    **PA-blind**: never into `<message>`, never into the simulator's `_history`,
    never anywhere in the PA prompt; the wrapper reads it as **gate state** only.

    Owner ruling 2026-09-08 ("这个肯定要改") adds a second shape: an entry with an
    **empty** `interaction_item_id` and a `tool` is a **standing authorisation**
    -- the actor spoke the go-ahead for an outward act on its authorising beat
    and no held id existed yet to answer. Live acc_039 t40: "You may proceed
    with booking that slot at 08:30", t41 the floor held `appointment_create`
    and asked anyway, t43 the user authorised the same act a second time.
    A standing entry is **never matched against a held id** (`is_standing`
    keeps it out of `sideband_decision`); it seeds the act-release map for its
    tool for the one PA turn that follows.
    """

    interaction_item_id: str
    decision: ConsentDecisionValue
    tool: str = ""
    """Catalog tool name. Set **only** on a standing authorisation; an entry that
    answers a held ask names the id, and the tool is already on the held call."""

    @property
    def is_standing(self) -> bool:
        """A pre-authorisation for a tool rather than an answer about a held id."""
        return not self.interaction_item_id and bool(self.tool)


@dataclass(frozen=True)
class ConsentRecord:
    """One row of §4.4's `consent[]`: the PA's intent, the sideband's ground truth and the gate's final result, **recorded separately**."""

    attribute: str
    item_id: str
    pa_decision: ReleaseDecisionValue
    sideband_decision: ConsentDecisionValue
    released: bool
    user_evidence: str = ""
    evidence_verified: bool = False
    tool_changed: bool = False
    """This turn submitted **a different tool** under the same id, so last turn's
    authorisation does not apply (§3.3 (part 3)). The user consented to the
    action that was asked about, not to the id: a different tool is a different
    action, and it goes through the gate to be asked again."""


# ---------------------------------------------------------------------------
# contract 1 · the per-turn payload (§3.0, §5.3 carrier B / D / E)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnPayloads:
    """This turn's values for the six `IX_render_<attribute>` + `IX_capability_resolve` + `IX_continue_*`.

    Each field is annotated with its carrier tool. The required / optional
    conditions are in §3.0's layered table -- the verdict is reached **after**
    the model's response (`RequiredContext`).
    """

    # IX_render_reasoning_visibility -- required under both Show and Hide (whether it is required must not leak the setting)
    user_facing_rationale: str = ""
    # IX_render_uncertainty_expression -- required every turn; an empty list when the prose has no marker
    certainties: list[Certainty] = field(default_factory=list)
    # IX_render_process_visibility -- required under all three settings
    completion_message: str = ""
    # IX_render_autonomy_level
    plan_id: str = ""
    plan: list[PlanItem] = field(default_factory=list)
    plan_done: bool = False
    # IX_render_topic_management
    batch_set_id: str = ""
    item_count: int = 0
    batches: list[Batch] = field(default_factory=list)
    # IX_render_information_elicitation
    elicitation_id: str = ""
    questions: list[Question] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    # IX_capability_resolve (carrier E)
    boundary_id: str = ""
    boundary: BoundaryState | None = None
    # IX_continue_<attribute> (carrier D)
    release_decisions: list[ReleaseDecision] = field(default_factory=list)
    closing_message: str = ""
    """Normally may be empty; required on the `rounds_open == 4` round (§3.7
    (part 4)). When it is missing the renderer **does not write for the PA**."""


#: The three values of `IXToolSpec.required` (§7.1 P0's slot decision:
#: "response-evaluated conditionally-required").
RequiredMode = Literal["always", "never", "predicate"]


@dataclass(frozen=True)
class RequiredContext:
    """The input a named required predicate may read (§7.1 P0's slot).

    The evaluation point is the existing check at `pa/agent/runner.py:161-174` --
    which sits exactly **after the response**. Judged required but not called by
    the model -> insert a reminder `internal_user_message`, void the turn and go
    again, **retrying only once**. Judged not required -> **neither offered nor
    reminded about**.
    """

    session_active_ix: tuple[str, ...] = ()
    hold: IXHold = field(default_factory=IXHold)
    response_tool_names: tuple[str, ...] = ()
    response_interaction_item_ids: tuple[str, ...] = ()
    produces_user_visible_message: bool = True
    capability_limit_seen: bool = False
    response_text: str = ""
    """This turn's response prose (the iterations joined together, the same
    material as `IXRenderInput.segments`). The predicates use it **only to find
    internal position markers**, never to read meaning."""
    response_markers: tuple[str, ...] = ()
    """The marker ids scanned out of `response_text`, shaped like
    `("batch:b1", "q:q1", "c:c1")`. The caller need only sweep once with
    `BATCH_MARKER_RE` / `QUESTION_MARKER_RE` / `CERTAINTY_MARKER_RE`; keeping a
    separate copy is what makes the predicates pure and unit-testable, with no
    need to rescan the prose every time.

    `IX_render_topic_management`'s required predicate rests on it (§3.0, that
    table): a task call this turn **or** a `[[batch:…]]` marker in the prose ->
    required; a pure-conversation turn with neither -> not offered."""


#: The signature of a named predicate. P1 offers, P2 consumes, and P0 freezes the shape.
RequiredPredicate = Callable[[RequiredContext], bool]


# ---------------------------------------------------------------------------
# the renderer's input and output (§2.2 component 3, §2.3)
# ---------------------------------------------------------------------------

#: §2.2 component 3's `trace: list[RenderStep]`; §4.4 serialises it into a list
#: of strings, e.g. `"topic_management=One-at-a-time -> release 1/3"`.
#: **`audit_trace` only, never the text plane.**
RenderStep = str


@dataclass(frozen=True)
class IXRenderInput:
    """The renderer's input (§2.3): this beat's raw material × the per-attribute payload × the cross-turn hold.

    The renderer is pure: `render(render_input, directive, hold) -> RenderedTurn`,
    with no IO and no LLM calls.
    """

    turn_index: int = 0
    segments: list[str] = field(default_factory=list)
    """The model text of each iteration (`pa/agent/runner.py:179-187` currently
    only pushes it into messages)."""
    provider_reasoning: list[str] = field(default_factory=list)
    """`reasoning_content` and the `<think>` blocks `_strip_think` discarded.
    **`audit_trace` only; the renderer does not read it** (§3.2)."""
    calls: list[ToolCallRecord] = field(default_factory=list)
    gate: GateRecord | None = None
    gate_remaining: list[PlanItem] = field(default_factory=list)
    """The plan items left unexecuted when the gate fired (§2.2 component 1's `gate.remaining`)."""
    payloads: TurnPayloads = field(default_factory=TurnPayloads)
    proactive_decision: ProactiveDecision | None = None
    """Non-empty on a proactive event turn only (§3.1 that row, contract 11)."""
    consent_decisions: list[ConsentDecision] = field(default_factory=list)
    """The PA-blind sideband (§3.7 (part 3)). **Read as gate state only, never
    given to the model as text.**"""
    hold: IXHold = field(default_factory=IXHold)
    act_releases: dict[str, str] = field(default_factory=dict)
    """`IXHold.act_releases` as it stands after this turn, computed on the
    collector side (the sideband lives there) and carried through so the hold
    write-back can persist it without reading consent itself."""
    held_narration: list[HeldNarration] = field(default_factory=list)
    """The prose in the same assistant message as a held call (§3.3 (part 2)·D
    addendum). The renderer **skips** these segments when it delivers prose;
    they appear only in `audit_trace`'s `held.narration`."""
    empty_reply: bool = False
    """Round 9 item 1: this beat can deliver nothing at all, and the extra prose
    the runner asked for is empty too. The renderer delivers the fixed safe
    sentence on that basis (`close_line_source: template`). The verdict lives in
    the runner and the sentence in the renderer -- so an "empty reply" has one
    rule and one exit."""
    binding_refused: bool = False
    """Round 9 item 2: the fail-closed binding exit really fired (bound wrongly
    again on the second attempt, so the whole batch does not execute). A binding
    failure that was repaired the first time **does not set this** -- its held
    row stays on the audit plane, and judging by held rows would tell the user
    "could not be executed" on the very beat where the retry went through."""


@dataclass(frozen=True)
class RenderedTurn:
    """The renderer's output (§2.2 component 3)."""

    message: str
    """The one message the user sees. It equals `public_turn` verbatim (§2.4 hard constraint 1)."""
    held: list[HeldItem] = field(default_factory=list)
    """What this beat held (into `audit_trace` / the IX plane only)."""
    hold_next: IXHold = field(default_factory=IXHold)
    """Written back to the cache."""
    proactive: ProactiveDecision | None = None
    executed_call_ids: list[str] = field(default_factory=list)
    """The provider `tool_call_id`s that really executed this turn, **in call
    order** (§2.4, mixed tool calls)."""
    held_call_ids: list[str] = field(default_factory=list)
    """The provider `tool_call_id`s a gate held and that never executed. Their
    declarations are removed from the assistant protocol message in
    `runtime_history`, so **no** synthesised tool result is needed."""
    trace: list[RenderStep] = field(default_factory=list)
    """Whether each rule fired. **Audit only: `audit_trace`, never the text plane.**"""
    cancelled_now: list[HoldEpisode] = field(default_factory=list)
    """The episodes cancelled **this beat** on hitting `HOLD_ROUNDS_HARD_CAP`
    (§3.7 (part 4)). `hold_next.finished_episodes` accumulates **over the whole
    session**, so emitting events from it would repeat one cancellation on every
    later beat (live run `v6g01`: 2 cancellations -> 9 `ix_hold_forced_close`)."""
    release_folded: list[dict] = field(default_factory=list)
    """§4.3 — the carried questions this beat released by decision alone, with no
    `[[q:…]]` marker in the prose: they are folded into the reply and settle.
    One entry per attribute (`attribute`, `episode_id`, `question_ids`), which
    `loop.py` turns into one `ix_release_folded` so the batch review can find
    every beat where a held question went out unmarked."""
    close_line_source: str = ""
    """Where the forced-close sentence came from: `payload` (the
    `closing_message` the PA wrote itself) / `none` (the PA wrote none and the
    renderer writes nothing in its place -- owner decision 2026-09-06, §10 item
    d; the miss lives on the audit plane as `ix_close_no_message` and
    `cancelled_items`) / "" (nothing was cancelled this beat). `template` is
    still written by the two fail-closed exits that own a safety line of their
    own: `binding_refused` and `empty_reply`."""
    completion_delivered: bool = False
    """Whether the §3.1 closing line reached `message` this beat — appended as
    the field, or already standing as the prose tail (delivered once either
    way). `completion.released` on the audit plane is this flag: the setting
    plus a non-empty `completion_message` is **not** the same statement, and
    reading `released` off those two claimed a closing line on gate turns that
    carried none."""
    completion_skipped: str = ""
    """Why a line the setting would have released did not go out: `gate` (the
    ask ends the turn) / `stripped` (the `WRAPPER_VOCABULARY` filter cut the
    line after assembly) / "" (nothing suppressed it)."""
    withheld_narration: list[HeldNarration] = field(default_factory=list)
    """Round 9 item 3: whole sentences the `WRAPPER_VOCABULARY` filter held,
    verbatim, `item_id: "wrapper_vocabulary"`; the unparsed tool-call blocks the
    markup filter held (acc_001 t32/t34), `item_id: "dsml_markup"`; and the
    sentences claiming an action the gate is still asking about (acc_039),
    `item_id: "claim_of_action"`. It goes where `held.narration` goes:
    `audit_trace` only, with `public_turn` seeing not one byte."""
    claims_not_found: list[str] = field(default_factory=list)
    """acc_035 — certainty refs whose declared claim the locator could not place
    in the prose, so no hedge was inserted and the turn shipped flat.

    Trace-only for three rounds, which is why a regression in `claim_end` went
    unnoticed until the first `low` claim in four rounds was dropped on the one
    cell that exists to produce it. `ix_certainty_claim_not_found` is what makes
    it countable."""
    leadin_dropped: str = ""
    """regress13 acc_009 — the framing sentence the excision left pointing at
    nothing, verbatim, or "". Under `Infer` the release is 0/N, so three turns of
    that session reached the user as a lead-in and a colon with the questions
    gone. It goes where the other withheld text goes: `audit_trace` only."""
    prose_held: bool = False
    """Whether the renderer excised a `[[batch:…]]` / `[[q:…]]` section out of
    *this* reply.

    The fallbacks that ship `final_content` when the renderer produced no
    message (`loop.py`'s `public_turn`, `runner._delivers_something`) read the
    model's raw text, which on a hold turn still contains every held section --
    so on a fully-held turn with no framing they hand the user exactly the
    questions the setting withheld. This is what tells them the empty message
    was a decision and not an absence."""


# ---------------------------------------------------------------------------
# contract 2 · the five projections and the full turn envelope (§2.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicTurn:
    """Projection 1 · the message the user actually receives -> `OutboundMessage.content` -> the simulator.

    Explicitly excludes: any held prose.
    """

    message: str


@dataclass(frozen=True)
class RuntimeHistory:
    """Projection 2 · the history the next provider call receives, **three sections in order** (§2.4).

    Explicitly excludes: held prose; any declaration or result of a
    `held_call_ids` call; prose from intermediate iterations that was not shown;
    §3.6's restatement and §4.3's hold notice (both carry `_mempa_internal`).

    The two of the four hard constraints that concern this projection: the
    current user message appears **exactly once** with content equal, verbatim,
    to what the user really sent; the content of the last assistant message
    **equals `public_turn` verbatim**; and after persistence the declared call
    ids and the tool result ids are **exactly equal in order**.
    """

    user_message: dict[str, Any]
    """(1) A canonical copy of the current user message, exactly once. Fields
    that may be kept, like timestamp / source, are kept; internal markers like
    `_mempa_internal` are stripped."""
    tool_protocol: list[dict[str, Any]] = field(default_factory=list)
    """(2) The filtered executed tool protocol: **copy** the original assistant
    protocol message but filter `tool_calls` down to the `executed_call_ids`
    subset, followed only by the tool results for those ids. With
    `executed_call_ids` empty this section does not appear. Reusing the original
    mixed message directly is **forbidden**, and so is dropping it wholesale."""
    assistant_message: dict[str, Any] = field(default_factory=dict)
    """(3) One assistant message whose content **equals `public_turn` verbatim**."""

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The three sections expanded in order, i.e. the input to `_save_turn`."""
        return [self.user_message, *self.tool_protocol, self.assistant_message]


@dataclass(frozen=True)
class MemoryProjection:
    """Projection 3 · the PA's **whole first-person experience** of the turn, as the memory adapter receives it (§2.4; owner ruling 2026-09-13).

    The old rule chose the wrong subject. It kept the two messages *the person*
    saw, and the subject of memory is the assistant. What the PA produced, or
    perceived in its role as the assistant, is experience and goes in: tool
    calls with their arguments and their results, `reasoning_content` and
    `thinking_blocks`, intermediate prose, a sub-agent's return, an environment
    trigger, and the PA's own `IX_<attribute>` selection with the outcome the
    registry gave it — that selection is the half of the (choice, reaction)
    pair preference learning needs, and it is the one the old projection
    dropped. What exists only because a benchmark is running — the system
    prompt, IX contract repairs and reminders, probe restatements, the
    runtime-context prefix — is not experience and stays out.

    The rendering is `build_turn_pool()` (`memory/event_writer.py`): untagged
    content is dialogue, every non-dialogue item is a tagged block on an
    `assistant`-role message, and nothing that is not the person's own speech
    is ever `role: "user"`. One rendering, produced once, handed to every
    backend identically — the harness performs no selection, because selecting
    what is worth remembering is the backend's job and is what the benchmark
    measures. `ForcedEventExecutor` renders through the same grammar, so normal
    and forced still share one memory event schema.

    Still excluded, and these are hard: held prose at hold time (O4 — the turn
    carries the `[ix_result]` hold status, and the released text enters as
    dialogue in the turn that delivers it); the provider output of an iteration
    the wrapper voided (Q1); `consent_decisions`, `target_cell` /
    `target_attribute` / `target_setting`, probe ids and `acc_num`, none of
    which is a message at all — the builder reads messages and never `metadata`.

    This dataclass is the two-message *shape* the wrapper's own envelope test
    still names; the live projection is `TurnPool`, which carries the same
    content in `messages` plus its pre-parsed `items` sibling.
    """

    user_message: dict[str, Any]
    assistant_message: dict[str, Any]
    """The content **equals `public_turn` verbatim** (§2.4 hard constraint 2)."""

    @property
    def messages(self) -> list[dict[str, Any]]:
        return [self.user_message, self.assistant_message]


@dataclass(frozen=True)
class TextJudgeProjection:
    """Projection 4 · the **setting-blind** text-plane pack (§2.4, added in round 6; §4.4 "two judge planes").

    Explicitly excludes: attribute names, setting names, any string a setting
    could be worked back from; `evidence` / `application`; held prose (held
    batch / question / plan item / outreach message); `consent` (including the
    sideband verdicts); provider reasoning; the gate reason (`gate.reason`);
    `trace`; **the whole `ix_render` block**.
    """

    public_turn: str
    task_tool_events: list[dict[str, Any]] = field(default_factory=list)
    """The **executed** task tool events: `name` / `args` / `status` / ordering.
    **No IX tool and no internal tool.**"""

    # The ask the user actually saw when a gate fired is already inside
    # `public_turn`; there is no second field for it here, in case that becomes
    # a separate signal a gate could be worked back from.


@dataclass(frozen=True)
class AuditTrace:
    """Projection 5 · all of this turn's raw material (§2.4). **The only destination for held content.**

    Where it goes: `metadata["ix_render"]` -> transcript -> **the IX-plane judge
    and manual review** (§4.4).
    It goes into **none** of the four projections above, and it **may not**
    become the source for restoring the current user message (§2.4 hard
    constraint 4).
    """

    segments: list[str] = field(default_factory=list)
    provider_reasoning: list[str] = field(default_factory=list)
    held: list[HeldItem] = field(default_factory=list)
    consent: list[ConsentRecord] = field(default_factory=list)
    trace: list[RenderStep] = field(default_factory=list)
    ix_render: dict[str, Any] = field(default_factory=dict)
    """§4.4's blob; the whole thing is `metadata["ix_render"]`."""


#: The names of the five projections (§2.4). P4 / P7's banned-field tests check
#: against this list.
PROJECTION_NAMES: tuple[str, str, str, str, str] = (
    "public_turn",
    "runtime_history",
    "memory_projection",
    "text_judge_projection",
    "audit_trace",
)


# ---------------------------------------------------------------------------
# §4.4 · the JSON shape of `OutboundMessage.metadata["ix_render"]`
# ---------------------------------------------------------------------------


class EpisodeStatusJSON(TypedDict):
    """`episodes[<attribute>]` (§4.4, added in round 4)."""

    episode_id: str
    rounds_open: int
    closed: bool


class ReleasedJSON(TypedDict):
    """`released`: the ids this turn really released (§4.4)."""

    batches: list[str]
    questions: list[str]
    plan_executed: list[str]


class HeldNarrationJSON(TypedDict):
    """`held.narration[]`: the prose in the same message as a held call, verbatim (§4.4)."""

    item_id: str
    text: str


class HeldJSON(TypedDict):
    """`held`: the held content, prose **kept verbatim** (§4.4). The text-plane judge sees not one byte."""

    batches: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    plan_remaining: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    cancelled: list[dict[str, Any]]
    outreach_message: str
    narration: NotRequired[list[HeldNarrationJSON]]
    """The **narration** held this turn: the prose declaring some held action
    already done. With none this turn the key is omitted entirely (which is why
    a 1.1.0 record without it stays valid)."""


class GateJSON(TypedDict):
    """`gate` (§4.4)."""

    attribute: str
    item_id: str
    reason: str
    action_type: str
    message_source: str


class ConsentJSON(TypedDict):
    """`consent[]` (§4.4, reshaped in round 6)."""

    attribute: str
    item_id: str
    pa_decision: str
    sideband_decision: str
    released: bool
    user_evidence: str
    evidence_verified: bool
    tool_changed: bool


class NoticeJSON(TypedDict):
    """`notices[]`: process_visibility's per-call notices (§4.4)."""

    order: int
    tool: str
    text: str


class CompletionJSON(TypedDict):
    """`completion`: the `completion_message` and whether it was released this turn (§4.4, added in round 4).

    `text` is the line the PA wrote, held or not — the audit plane keeps it
    either way. `released` is a statement about the **delivered reply**: it is
    true only when that sentence actually reached the user, so it is read off
    `RenderedTurn.completion_delivered` and never re-derived from the setting.
    """

    released: bool
    text: str
    skipped: NotRequired[str]
    """Why a line the setting would have released did not go out: `gate` — the
    turn ended on the gate's ask, and §4.4 gives the closing line no place after
    it; `stripped` — the wrapper vocabulary filter removed the line after
    assembly. Omitted when nothing suppressed the line (including when the setting
    itself holds it, which `released: false` already says)."""


class ReasoningJSON(TypedDict):
    """`reasoning`: after this turn `source` has only the one value `user_facing_rationale` (§3.2)."""

    shown: bool
    source: str
    provider_reasoning_present: bool


class CertaintyJSON(TypedDict):
    """`certainties[]`: `rendered` is the parenthesis rendered out, "" for none appended (§4.4).

    `claim` / `strategy` are two audit-side columns (added in round 7):
    `rendered` alone cannot tell "located but withheld under this setting" from
    "never located at all" -- under `Hide` both are "".
    `strategy ∈ {exact, normalized, token_overlap, not_found}` is the pass
    `claim_end()` hit on; a ref that came through the `[[c:…]]` marker channel
    has no such column.
    """

    ref: str
    level: str
    rendered: str
    claim: NotRequired[str]
    strategy: NotRequired[str]


class ProactiveDecisionJSON(TypedDict):
    """`proactive_decision` (§4.4, contract 11). `null` on an ordinary turn."""

    action: str
    setting: str
    message: str


class RawBlockJSON(TypedDict):
    """`raw_blocks[]`: the carrier blocks the model handwrote, verbatim (§4.4, added in round 7)."""

    kind: str
    text: str


class CancelledItemJSON(TypedDict):
    """`cancelled_items[]`: one item cancelled on the hard cap and **never executed** (§4.4, added in round 7)."""

    attribute: str
    id: str
    kind: str
    intent: str


class IXRenderJSON(TypedDict):
    """The full shape of `OutboundMessage.metadata["ix_render"]` (§4.4)."""

    schema_version: str
    ix_contract_version: str
    episodes: dict[str, EpisodeStatusJSON]
    released: ReleasedJSON
    held: HeldJSON
    gate: GateJSON | None
    consent: list[ConsentJSON]
    notices: list[NoticeJSON]
    completion: CompletionJSON
    reasoning: ReasoningJSON
    certainties: list[CertaintyJSON]
    proactive_decision: ProactiveDecisionJSON | None
    trace: list[str]
    raw_blocks: NotRequired[list[RawBlockJSON]]
    """The carrier blocks stripped out of this turn's reply (today only
    `<certainties>`). **Audit plane only**: it *is* held prose, and not one byte
    of it may appear in the text-plane pack (`TEXT_JUDGE_BANNED_KEYS`).
    With no block this turn the key is omitted entirely (which is why a 1.0.0
    record without it stays valid)."""
    stop_reason: NotRequired[str]
    """How this beat ended (`ix_hold` / `ix_gate` / `ix_contract_failure` / …).
    In live runs not one `pa_turn` line carried it: round 5 could only infer
    `ix_hold` from the held status and the prompt tokens in the API log, and the
    stop reason `f19702b` added could not be read out of the artefacts at all."""
    closing_iteration: NotRequired[bool]
    """Whether §3.4 (part 5)'s closing request got prose back (`True` = yes,
    `False` = empty). The key being absent altogether means this beat made
    **no** closing request."""
    cancelled_items: NotRequired[list[CancelledItemJSON]]
    """§3.7 (part 4) · every item cancelled **so far this session** (id +
    intent), **written on every beat from the cancellation onward**.
    The judge uses it to check "did a later beat call something done that was
    never executed" -- live, `v6g01` t35 listed an email that was never sent
    under a "done" heading, on a beat that held nothing, so structurally no
    detector could reach it. With nothing cancelled this session the key is
    omitted entirely."""
    close_line_source: NotRequired[str]
    """Where the forced-close sentence came from: `payload` / `template`. With
    nothing cancelled this beat the key is omitted entirely."""


#: The top-level keys of the §4.4 blob, in the order they appear in §4.4.
IX_RENDER_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version",
    "ix_contract_version",
    "episodes",
    "released",
    "held",
    "gate",
    "consent",
    "notices",
    "completion",
    "reasoning",
    "certainties",
    "proactive_decision",
    "trace",
    "raw_blocks",
    "stop_reason",
    "closing_iteration",
    "cancelled_items",
    "close_line_source",
)

#: The top-level keys that **may be absent** (from schema 1.1.0 on). Every key
#: added inside one major goes here, or a 1.0.0 record would be judged to have a
#: "missing key" by the reader.
IX_RENDER_OPTIONAL_KEYS: frozenset[str] = frozenset({
    "raw_blocks", "stop_reason", "closing_iteration",
    "cancelled_items", "close_line_source",
})

#: The keys that may **never appear** in the text-plane pack (§2.4 projection 4,
#: §4.4 "two judge planes").
#: The rule is "**remove the whole `ix_render` block**" -- not delete a few of
#: its fields, since one `trace` string is enough to leak a setting.
#: P4's banned-field test (T11) uses it.
TEXT_JUDGE_BANNED_KEYS: frozenset[str] = frozenset({
    # the whole ix_render block and each of its top-level keys
    "ix_render",
    *IX_RENDER_TOP_LEVEL_KEYS,
    # the old IX event groupings (dropped along with P4)
    "ix_directives",
    "process_events",
    "elicitation_events",
    # the payload fields of the IX selection tools
    "setting",
    "evidence",
    "application",
    # the three per-call injected fields (they should already be stripped before
    # forwarding, and leaking into the pack is a violation all the same)
    *PER_CALL_INJECTED_FIELDS,
    # the three paths provider reasoning is read from (§3.2)
    "reasoning_content",
    "reasoning_details",
    "provider_reasoning",
    # the fourth: the carrier for Anthropic extended thinking
    # (`pa/providers/base.py:51` `ProviderResponse.thinking_blocks`, carried all
    # the way to metadata with the assistant message). Same in kind as the three
    # above, only under another key name; miss it and the scan will not catch
    # provider thinking.
    "thinking_blocks",
    # the simulator's PA-blind consent sideband (`pa/agent/ix/collector.py`
    # `consent_decisions_from_metadata`, which also honours the alias `id`). It
    # is the gate's ground truth: one row tells the text-plane judge a gate
    # opened and consent was refused, exactly what projection 4 must hold back.
    "consent_decisions",
})


def to_ix_render_metadata(
    rendered: RenderedTurn,
    *,
    episodes: Mapping[str, EpisodeStatusJSON] | None = None,
    released: ReleasedJSON | None = None,
    held: HeldJSON | None = None,
    gate: GateRecord | None = None,
    consent: Sequence[ConsentRecord] = (),
    notices: Sequence[NoticeJSON] = (),
    completion: CompletionJSON | None = None,
    reasoning: ReasoningJSON | None = None,
    certainties: Sequence[CertaintyJSON] = (),
    raw_blocks: Sequence[RawBlockJSON] = (),
    stop_reason: str = "",
    closing_iteration: bool | None = None,
    cancelled_items: Sequence[CancelledItemJSON] = (),
    schema_version: str = IX_RENDER_SCHEMA_VERSION,
    ix_contract_version: str = IX_CONTRACT_VERSION,
) -> IXRenderJSON:
    """Assemble one `RenderedTurn` and its audit-side material into §4.4's `ix_render` blob.

    `proactive_decision` and `trace` come straight from `rendered`; P2 supplies
    the other sections at render time.
    Contract 12: every record carries its own `schema_version` and restates the
    run's `ix_contract_version`.

    When `closing_iteration` is not `None` it is written as a key and also
    appended as a `trace` line -- §3.4 (part 5)'s request previously left no
    trace in the artefacts at all, only one extra call in the API log.
    """
    trace = list(rendered.trace)
    if closing_iteration is not None:
        trace.append(
            "closing iteration ran" if closing_iteration else "closing iteration empty"
        )
    return {
        "schema_version": schema_version,
        "ix_contract_version": ix_contract_version,
        "episodes": dict(episodes or {}),
        "released": released
        or {"batches": [], "questions": [], "plan_executed": []},
        "held": held
        or {
            "batches": [],
            "questions": [],
            "plan_remaining": [],
            "calls": [],
            "cancelled": [],
            "outreach_message": "",
        },
        "gate": None
        if gate is None
        else {
            "attribute": gate.attribute,
            "item_id": gate.item_id,
            "reason": gate.reason,
            "action_type": gate.action_type,
            "message_source": gate.message_source,
        },
        "consent": [asdict(record) for record in consent],
        "notices": [dict(notice) for notice in notices],
        "completion": completion or {"released": False, "text": ""},
        "reasoning": reasoning
        or {
            "shown": False,
            "source": "user_facing_rationale",
            "provider_reasoning_present": False,
        },
        "certainties": [dict(item) for item in certainties],
        **(
            {"raw_blocks": [dict(block) for block in raw_blocks]}
            if raw_blocks else {}
        ),
        "proactive_decision": None
        if rendered.proactive is None
        else {
            "action": rendered.proactive.action,
            "setting": rendered.proactive.setting,
            "message": rendered.proactive.message,
        },
        "trace": trace,
        **({"stop_reason": stop_reason} if stop_reason else {}),
        **(
            {"closing_iteration": closing_iteration}
            if closing_iteration is not None else {}
        ),
        # §3.7 (part 4): written on **every** turn from the cancellation onward, so a
        # later summary that calls a cancelled item done can be checked against
        # a record on the same turn. There is no structural detector for that
        # sentence -- the turn withholds nothing -- so the judge needs the list.
        **({"cancelled_items": [dict(item) for item in cancelled_items]}
           if cancelled_items else {}),
        **({"close_line_source": rendered.close_line_source}
           if rendered.close_line_source else {}),
    }


def from_ix_render_metadata(
    blob: Mapping[str, Any],
    *,
    expected_contract_version: str = IX_CONTRACT_VERSION,
    expected_schema_version: str = IX_RENDER_SCHEMA_VERSION,
) -> IXRenderJSON:
    """Read an `ix_render` blob back, failing fast on both shape and version (contract 12).

    A version mismatch raises; it is **never silently reinterpreted under the
    new semantics**. Old artefacts go through an explicit legacy reader, not
    this entry point (§7.1 contract 12, §7.2 P6b item 6).
    """
    missing = [
        key for key in IX_RENDER_TOP_LEVEL_KEYS
        if key not in blob and key not in IX_RENDER_OPTIONAL_KEYS
    ]
    if missing:
        raise ValueError(f"ix_render blob missing keys: {', '.join(missing)}")
    unknown = [key for key in blob if key not in IX_RENDER_TOP_LEVEL_KEYS]
    if unknown:
        raise ValueError(f"ix_render blob has unknown keys: {', '.join(sorted(unknown))}")
    schema_version = blob["schema_version"]
    # Major match, not equality: inside one major the only permitted change is a
    # new optional field, so a 1.0.0 record is still read correctly by a 1.1.0
    # reader. A different major means the same key can mean something else, and
    # that is the case contract 12 exists to stop.
    if _schema_major(schema_version) != _schema_major(expected_schema_version):
        raise IXContractVersionError(
            f"ix_render schema_version {schema_version!r} != {expected_schema_version!r}"
        )
    contract_version = blob["ix_contract_version"]
    if contract_version != expected_contract_version:
        raise IXContractVersionError(
            f"ix_contract_version {contract_version!r} != {expected_contract_version!r}"
        )
    return {  # type: ignore[return-value]
        key: blob[key] for key in IX_RENDER_TOP_LEVEL_KEYS if key in blob
    }


def text_judge_projection(
    rendered: RenderedTurn,
    render_input: IXRenderInput,
) -> TextJudgeProjection:
    """Build projection 4 (§2.4). **P0 freezes only the field set; the packing lands in P4.**

    Two hard requirements for the implementation: keep only `public_turn` and
    the `name` / `args` / `status` / ordering of the **executed** task tool
    events; and let no key from `TEXT_JUDGE_BANNED_KEYS` appear, nor any
    attribute name, setting name or fragment of held prose anywhere in the text
    (T11's banned strings).
    """
    raise NotImplementedError(
        "text_judge_projection packing lands in P4 (§4.4 two judge planes); "
        "P0 freezes only the field set and TEXT_JUDGE_BANNED_KEYS."
    )
