"""Collector, gates, hold bookkeeping and turn projections.

This module is the Runner/Loop-side half of the IX wrapper.
It owns:

- :class:`TurnCollector` — per-iteration model text plus per-call metadata,
  assembled into the frozen :class:`~pa.agent.ix.render_types.IXRenderInput`.
- the gate predicates and the fail-closed
  binding validation;
- the step-boundary hold check and the controlled reminder;
- the runtime-history projection built from a rendered turn.

The pure renderer itself lives in ``pa/agent/ix/render.py``, and the hold
write-back with it; :func:`render_turn` is this side's single call site.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from pa.agent.ix.action_types import DEFAULT_TOOL_ACTION_TYPES
from pa.agent.ix.events import IXSelection
from pa.agent.ix.rationale import (
    CERTAINTY_BLOCK_INSTRUCTION,
    parse_certainty_block,
    strip_certainty_block,
)
from pa.agent.ix.render import (
    ASK_BEARING_ATTRIBUTES,
    WORLD_CHANGING_ACTION_TYPES,
    delivered_segments,
    released_item_ids,
    render,
    strip_meta_acknowledgement,
)
from pa.agent.ix.render_types import (
    BATCH_MARKER_RE,
    CERTAINTY_MARKER_RE,
    CONSENT_DECISIONS_BLOCK,
    held_status,
    INTERACTION_ITEM_ID_FIELD,
    IX_CAPABILITY_RESOLVE_TOOL,
    IX_CONFIRMATION_MESSAGE_FIELD,
    IX_CONTINUE_TOOL_NAMES,
    IX_NOTICE_FIELD,
    IX_RENDER_TOOL_NAMES,
    PER_CALL_INJECTED_FIELDS,
    QUESTION_MARKER_RE,
    SIDEBAND_CONSENT_VALUES,
    UNRECOGNISED_MARKER_RE,
    Assumption,
    Batch,
    BoundaryState,
    Certainty,
    ConsentDecision,
    ConsentRecord,
    GateRecord,
    HeldItem,
    HeldNarration,
    IXHold,
    IXRenderInput,
    PlanItem,
    ProactiveDecision,
    Question,
    ReleaseDecision,
    RenderedTurn,
    RuntimeHistory,
    ToolCallRecord,
    TurnPayloads,
)
from pa.agent.ix.specs import IX_PASSIVE_SPECS, NO_WRAPPER_TALK

__all__ = [
    "ASK_BEARING_ATTRIBUTES",
    "IX_PAYLOAD_RESULT_TOOLS",
    "WORLD_CHANGING_ACTION_TYPES",
    "RecurrenceRepair",
    "TurnCollector",
    "assert_action_type_coverage",
    "binding_attribute",
    "build_runtime_history",
    "collect_payloads",
    "catalog_tool_name",
    "consent_decisions_from_metadata",
    "gate_opens",
    "is_capability_limit",
    "ix_payload_status",
    "preauthorised_acts",
    "rebound_batch_ids",
    "released_item_ids",
    "render_turn",
    "strip_internal_markers",
    "turn_reminder_message",
    "validate_bindings",
]


_BOUNDARY_KINDS = frozenset({
    "missing_data",
    "missing_capability",
    "action_not_permitted",
    "external_dependency",
})

#: The per-turn restatement covers only the attributes the renderer does *not* transform.
_REMINDER_ATTRIBUTES = (
    "tone_formality",
    "verbosity",
    "emotional_engagement",
    "guidance_level",
    "solution_breadth",
    "task_expansion",
)
#: `autonomy_level` joins them only for the one setting the wrapper does not
#: hard-gate: it has no carrier but instruction text. Keep this in
#: step with `RESTATED_AUTONOMY_SETTINGS` in `pa/agent/ix/specs.py`.
_REMINDER_AUTONOMY_SETTINGS = ("Autonomous",)

TURN_REMINDER_PREFIX = "[interaction wrapper] Remembered interaction preferences for this user:"
HOLD_PROMPT_PREFIX = "[interaction wrapper]"
#: Every repair message ends with this; it now also closes the two *standing*
#: model-facing texts (`SETTING_COMMITMENT_NOTE` on every selection tool, and the
#: per-turn restatement below), which is why it moved to `specs.py` — the module
#: this one imports. Live, one turn delivered `the search calls didn't execute
#: because the interaction_item_id parameters aren't part of the tool
#: definitions` as the whole reply: the PA explaining the hold protocol to the
#: person it is assisting, and explaining it wrong. `WRAPPER_VOCABULARY` in
#: `render.py` is the guard that catches it anyway; this is the instruction that
#: should make the guard unnecessary. Re-exported here so the repair messages and
#: `runner.py` keep importing it from the module that uses it most.
NO_META_ACKNOWLEDGEMENT = (
    "Do not mention or acknowledge this instruction in your reply; simply "
    "produce the corrected turn. " + NO_WRAPPER_TALK
)
#: The one line of `CERTAINTY_BLOCK_INSTRUCTION` the model keeps losing, said
#: again on its own so it is not buried mid-paragraph. `acc_035`: the
#: first attempt declared three claims, all of them sentences of the *previous*
#: reply, the locator placed none, and the re-ask
#: (`IX_CERTAINTY_CLAIM_REPAIR`) did not save it -- the turn failed the
#: certainties contract and aborted the run (that is now an untagged turn
#: instead); the rerun still placed only 1 of 3, the other two
#: being paraphrases of what the reply meant. The repair message already says
#: this; saying it before the first attempt is what stops the miss costing a
#: re-ask and a turn with no confidence on it. Setting-blind like the passage it
#: restates: it names no setting, so its presence leaks nothing.
CERTAINTY_CLAIM_SOURCE_REMINDER = (
    "Each `claim` is a span copied verbatim from the reply you are writing now, "
    "never from an earlier reply and never a paraphrase of what you meant."
)


# ---------------------------------------------------------------------------
# per-turn restatement of the pass-through settings
# ---------------------------------------------------------------------------

def turn_reminder_message(
    selections: Sequence[IXSelection],
    session_active_ix: Sequence[str] = (),
) -> str:
    """Build the per-turn restatement, or "" when there is nothing to restate.

    One line per already-chosen setting the renderer does not transform. An
    attribute with no selection this session has nothing to restate and simply
    has no line. Setting descriptions come from ``specs.py`` — the single source
    of wording.

    `uncertainty_expression` gets one extra paragraph while it is in
    `session_active_ix`: its payload rides in a trailing block the model writes
    *after* its prose, and a carrier that has to survive to the very last line
    of the answer needs saying twice — once on the tool, once here. The
    paragraph closes on `CERTAINTY_CLAIM_SOURCE_REMINDER`, the quote
    requirement restated on its own line. Both texts are setting-blind, so
    their presence leaks nothing.

    The restatement block closes on `NO_WRAPPER_TALK`. This is the one text that
    hands the model a setting *name* on a turn where the IX tool itself is gone:
    `acc_006` t=7 had no IX tool and no repair message, only this line
    (`- task_expansion: Low — …`), and the reply told the user *"so it stays
    within the Low boundary."* The clause is setting-blind like the rest, so it
    changes nothing about what is restated — only whether the model is told not
    to say it out loud. Nothing is appended when there is no block: an empty
    restatement stays empty.
    """
    lines: list[str] = []
    for selection in selections:
        attribute = selection.attribute
        if attribute == "autonomy_level":
            if selection.setting not in _REMINDER_AUTONOMY_SETTINGS:
                continue
        elif attribute not in _REMINDER_ATTRIBUTES:
            continue
        spec = IX_PASSIVE_SPECS.get(attribute)
        description = (spec.setting_descriptions.get(selection.setting, "") if spec else "")
        lines.append(
            f"- {attribute}: {selection.setting}"
            + (f" — {description}" if description else "")
        )
    blocks: list[str] = []
    if lines:
        blocks.append("\n".join([TURN_REMINDER_PREFIX, *lines, NO_WRAPPER_TALK]))
    if "uncertainty_expression" in tuple(session_active_ix):
        blocks.append(
            f"{HOLD_PROMPT_PREFIX} {CERTAINTY_BLOCK_INSTRUCTION}\n"
            + CERTAINTY_CLAIM_SOURCE_REMINDER
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# action type coverage
# ---------------------------------------------------------------------------

def assert_action_type_coverage(
    tool_names: Iterable[str],
    overrides: Mapping[str, Any] | None = None,
) -> None:
    """Fail before the world is touched when a task tool has no action type.

    ``set(base tool names) - {IX tools}`` must be a subset of the action map
    plus the per-run overrides (contract 10).
    """
    known = set(DEFAULT_TOOL_ACTION_TYPES) | set(overrides or {})
    missing = sorted(
        name for name in tool_names
        if not name.startswith("IX_") and name not in known
    )
    if missing:
        raise ValueError(
            "task tools without a declared IX action type: " + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# payload parsing (carriers B/D/E) — read off the model's own arguments
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _plan_items(raw: Any) -> list[PlanItem]:
    items: list[PlanItem] = []
    for entry in _as_list(raw):
        if not isinstance(entry, dict):
            continue
        items.append(PlanItem(
            id=str(entry.get("id") or ""),
            intent=str(entry.get("intent") or ""),
            is_key=bool(entry.get("is_key")),
            bound_item_ids=[str(item) for item in _as_list(entry.get("bound_item_ids"))],
        ))
    return items


def _batches(raw: Any) -> list[Batch]:
    items: list[Batch] = []
    for entry in _as_list(raw):
        if not isinstance(entry, dict):
            continue
        items.append(Batch(
            batch_id=str(entry.get("batch_id") or ""),
            label=str(entry.get("label") or ""),
            body=str(entry.get("body") or ""),
            call_binding=[str(item) for item in _as_list(entry.get("call_binding"))],
        ))
    return items


def collect_payloads(
    tool_calls: Sequence[Any],
    payloads: TurnPayloads | None = None,
) -> TurnPayloads:
    """Read this turn's ``IX_render_*`` / ``IX_continue_*`` / resolve payloads.

    The payload is what the model wrote in the tool's *arguments*, so this
    works regardless of what the tool's ``execute()`` echoes back.
    """
    values: dict[str, Any] = {}
    render_tools = {name: attr for attr, name in IX_RENDER_TOOL_NAMES.items()}
    continue_tools = set(IX_CONTINUE_TOOL_NAMES.values())
    for tool_call in tool_calls:
        name = getattr(tool_call, "name", "")
        args = getattr(tool_call, "arguments", None) or {}
        if not isinstance(args, dict):
            continue
        attribute = render_tools.get(name)
        if attribute == "reasoning_visibility":
            values["user_facing_rationale"] = str(args.get("user_facing_rationale") or "")
        elif attribute == "uncertainty_expression":
            values["certainties"] = [
                Certainty(
                    ref=str(item.get("ref") or ""),
                    level=str(item.get("level") or "high"),
                    claim=str(item.get("claim") or ""),
                )
                for item in _as_list(args.get("certainties"))
                if isinstance(item, dict)
            ]
        elif attribute == "process_visibility":
            values["completion_message"] = str(args.get("completion_message") or "")
        elif attribute == "autonomy_level":
            values["plan_id"] = str(args.get("plan_id") or "")
            values["plan"] = _plan_items(args.get("plan"))
            values["plan_done"] = bool(args.get("done") or args.get("plan_done"))
        elif attribute == "topic_management":
            values["batch_set_id"] = str(args.get("batch_set_id") or "")
            values["item_count"] = int(args.get("item_count") or 0)
            values["batches"] = _batches(args.get("batches"))
        elif attribute == "information_elicitation":
            values["elicitation_id"] = str(args.get("elicitation_id") or "")
            values["questions"] = [
                Question(
                    id=str(item.get("id") or ""),
                    text=str(item.get("text") or ""),
                    about=str(item.get("about") or ""),
                )
                for item in _as_list(args.get("questions"))
                if isinstance(item, dict)
            ]
            values["assumptions"] = [
                Assumption(
                    about=str(item.get("about") or ""),
                    assumed=str(item.get("assumed") or ""),
                )
                for item in _as_list(args.get("assumptions"))
                if isinstance(item, dict)
            ]
        elif name in continue_tools:
            values.setdefault("release_decisions", []).extend(
                ReleaseDecision(
                    id=str(item.get("id") or ""),
                    decision=str(item.get("decision") or "keep"),
                    user_evidence=str(item.get("user_evidence") or ""),
                )
                for item in _as_list(args.get("decisions"))
                if isinstance(item, dict)
            )
            if args.get("closing_message"):
                values["closing_message"] = str(args["closing_message"])
        elif name == IX_CAPABILITY_RESOLVE_TOOL:
            values["boundary_id"] = str(args.get("boundary_id") or "")
            values["boundary"] = BoundaryState(
                blocked_goal=str(args.get("blocked_goal") or ""),
                boundary_kind=str(args.get("boundary_kind") or ""),
                recovery_remaining=_plan_items(args.get("recovery_options")),
                handoff_message=str(args.get("handoff_message") or ""),
            )
    if not values:
        return payloads or TurnPayloads()
    base = payloads or TurnPayloads()
    return replace(base, **values)


def boundary_payload_invalid(payloads: TurnPayloads) -> bool:
    """True when ``IX_capability_resolve`` was called with an illegal payload."""
    boundary = payloads.boundary
    if boundary is None:
        return True
    return not (
        boundary.blocked_goal
        and boundary.handoff_message
        and boundary.boundary_kind in _BOUNDARY_KINDS
    )


# ---------------------------------------------------------------------------
# PA-blind consent sideband
# ---------------------------------------------------------------------------

def consent_decisions_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> list[ConsentDecision]:
    """Read the simulator's ``consent_decisions`` block off the PA call metadata.

    The block is evaluation-side state, never text: it is read here as a gate
    input and never reaches a prompt. ``missing`` is rejected on the input path:
    it is a status the wrapper infers, not one the model may claim.

    Two shapes are accepted. An entry naming an ``interaction_item_id`` answers
    a held ask. An entry with **no** id but a ``tool`` is a standing
    authorisation: the actor spoke the go-ahead for an
    outward act before any hold existed, and ``preauthorised_acts`` turns it
    into a one-turn release on that tool. An entry with neither is dropped.
    """
    raw = (metadata or {}).get(CONSENT_DECISIONS_BLOCK)
    decisions: list[ConsentDecision] = []
    for entry in _as_list(raw):
        if not isinstance(entry, dict):
            continue
        decision = str(entry.get("decision") or "")
        item_id = str(entry.get(INTERACTION_ITEM_ID_FIELD) or entry.get("id") or "")
        tool = str(entry.get("tool") or "")
        if (not item_id and not tool) or decision not in SIDEBAND_CONSENT_VALUES:
            continue
        decisions.append(ConsentDecision(
            interaction_item_id=item_id, decision=decision, tool=tool,
        ))
    return decisions


#: How a catalog tool is named at run time. A standing authorisation names the
#: bare catalog tool (`appointment_create`) because that is what the session
#: draft's atomic steps name and what `harness/gate.py` derives; the PA calls
#: the same tool through the state MCP server, which prefixes it. The same rule
#: as `harness/transcript.py:STATE_TOOL_NAME_PREFIX` and the dual registration
#: in `pa.agent.ix.action_types.DEFAULT_TOOL_ACTION_TYPES`.
STATE_TOOL_NAME_PREFIX = "mcp_state_"


def catalog_tool_name(tool_name: str) -> str:
    """The bare catalog name behind a runtime tool call name.

    The pre-authorisation path is the one place the two spellings meet: the
    sideband is in catalog space and `ToolCallRequest.name` is in runtime
    space, so both ends are put through here before they are compared. Live
    `acc_039` t31 is what happens without it -- the standing allow
    for `appointment_create` never matched the PA's `mcp_state_appointment_create`,
    the floor held the call and asked for a permission the user had already
    given, and the grant lapsed unspent the same turn.

    `live_act_releases` needs no such thing: both of its ends (the hold's
    stored `tool`, and `ToolCallRecord.tool_name`) are runtime names already.
    """
    if tool_name.startswith(STATE_TOOL_NAME_PREFIX):
        return tool_name[len(STATE_TOOL_NAME_PREFIX):]
    return tool_name


def preauthorised_acts(
    consent_decisions: Sequence[ConsentDecision],
    calls: Sequence[ToolCallRecord],
) -> set[str]:
    """The tools this turn's standing authorisations still cover.

    live acc_039 t40 -- "You may proceed with booking that slot at 08:30" -- is
    the authorising beat, and t41's `appointment_create` was the act it
    authorised. Nothing was held when the user spoke, so there was no id for the
    sideband to answer and the floor asked for the same permission a second
    time. The standing entry names the tool instead.

    Scope is deliberately one PA turn: it is read from `consent_decisions`,
    which arrives with the user message and is never written into `IXHold`, so
    an unspent authorisation lapses when the turn ends rather than sitting
    around for a later act the user never spoke about. Within the turn it
    behaves like `live_act_releases`: an `ok` world-changing call spends it, an
    errored one leaves it standing for the retry.

    Keyed on the tool, not the arguments -- a wrong slot booked under a standing
    go-ahead is caught by the read-back, not by the floor (risk accepted).

    The set is in **catalog space**: the sideband's spelling, `harness/gate.py`'s
    spelling, and what `ix_preauthorisation_expired` already reported. Runtime
    call names go through `catalog_tool_name` on the way in, so the act that
    spends a grant is recognised whichever way the tool was named.
    """
    tools = {
        catalog_tool_name(decision.tool)
        for decision in consent_decisions
        if decision.is_standing and decision.decision == "allow"
    }
    for call in calls:
        if (
            call.executed
            and call.status == "ok"
            and call.action_type in WORLD_CHANGING_ACTION_TYPES
        ):
            tools.discard(catalog_tool_name(call.tool_name))
    return tools


def sideband_decision(
    decisions: Sequence[ConsentDecision], item_id: str
) -> str:
    """The verdict for one held id. Duplicated ids resolve fail-closed.

    The sideband list is not de-duplicated upstream, so an id can appear more
    than once. Only a unanimous `allow` releases: one `deny` or `unclear`
    anywhere in the list keeps the item held, which is the safe direction when
    the two entries disagree about an outward-facing action.

    Standing authorisations are skipped: they name a tool and no id, and a held
    call that reached the episode without an id would otherwise collect their
    verdict on the strength of two empty strings matching.
    """
    verdicts = [
        decision.decision
        for decision in decisions
        if not decision.is_standing and decision.interaction_item_id == item_id
    ]
    if not verdicts:
        return "missing"
    if "deny" in verdicts:
        return "deny"
    if "unclear" in verdicts:
        return "unclear"
    return "allow"


def live_act_releases(
    hold: IXHold,
    payloads: TurnPayloads,
    consent_decisions: Sequence[ConsentDecision],
    calls: Sequence[ToolCallRecord],
) -> dict[str, str]:
    """`tool -> the item id whose release on it is still unspent`.

    An authorisation is bound to the *action*, and the tool name is
    what names the action; the `interaction_item_id` is minted fresh whenever
    the PA re-plans. Live acc_039 asked for one booking three times because of
    that: a rewritten reason and then a store refusal each produced a new plan
    item on `appointment_create`, and each new id walked back into the floor
    gate one turn after the user had said "Proceed".

    So the release outlives the id. What ends it:

    - the act **succeeding** -- an `ok` call on the tool drops the entry, so the
      next act on it is asked for afresh. A genuine second act comes after a
      success, which is why binding to the tool alone was
      wrong and this is not: an errored or never-issued call spends nothing.
    - the user **withdrawing** it -- a `deny` on the sideband, or the PA's own
      `skip`, while the released item is still held.

    Only `autonomy_level` holds are read: they are the ones the floor gate asks
    about, and `asked` is required because an id the ask never named authorises
    nothing.
    """
    grants = dict(hold.act_releases)
    decisions = {
        decision.id: decision.decision for decision in payloads.release_decisions
    }
    for episode in hold.open_episodes.values():
        if episode.attribute != "autonomy_level":
            continue
        for call in episode.calls_held:
            item_id = str(call.get(INTERACTION_ITEM_ID_FIELD) or "")
            tool = str(call.get("tool") or "")
            if not (item_id and tool):
                continue
            verdict = sideband_decision(consent_decisions, item_id)
            if verdict == "deny" or decisions.get(item_id) == "skip":
                if grants.get(tool) == item_id:
                    del grants[tool]
            elif (
                call.get("asked")
                and decisions.get(item_id) == "release"
                and verdict == "allow"
            ):
                grants[tool] = item_id
    for call in calls:
        if (
            call.executed
            and call.status == "ok"
            and call.action_type in WORLD_CHANGING_ACTION_TYPES
        ):
            grants.pop(call.tool_name, None)
    return grants


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def gate_opens(policy: str, action_type: str, is_key: bool) -> bool:
    """Whether the autonomy gate opens for one call.

    ``read`` never opens a gate. ``Reactive`` gates every substantive action.
    ``Self-directed`` never gates reversible ``draft``, always gates
    ``external_action`` (the deterministic floor the model cannot argue away
    with ``is_key=False``), and gates ``internal_write`` only when the model
    marked the item as the key decision. ``Autonomous`` gates nothing — it is
    instruction text, not a hard gate.
    """
    if action_type == "read":
        return False
    if policy == "confirm_substantive_actions":
        return action_type in {"draft", "internal_write", "external_action"}
    if policy == "confirm_key_actions":
        if action_type == "draft":
            return False
        if action_type == "external_action":
            return True
        return bool(is_key)
    return False


def declared_item_ids(payloads: TurnPayloads, hold: IXHold) -> set[str]:
    """Every ``interaction_item_id`` this turn's payloads (or open holds) claim."""
    ids: set[str] = set()
    for item in payloads.plan:
        ids.update(item.bound_item_ids)
    for batch in payloads.batches:
        ids.update(batch.call_binding)
    for episode in hold.open_episodes.values():
        for item in episode.plan_remaining:
            ids.update(item.bound_item_ids)
        for batch in episode.batches_held:
            ids.update(batch.call_binding)
        for call in episode.calls_held:
            item_id = call.get(INTERACTION_ITEM_ID_FIELD)
            if item_id:
                ids.add(str(item_id))
    return ids


#: Which attribute's payload a binding declaration belongs to. Only these two
#: carry `interaction_item_id`s: a plan through `bound_item_ids`, a
#: batch through `call_binding`. Questions and boundaries declare no ids, so a
#: binding refusal can never be theirs.
_ID_DECLARING_ATTRIBUTES = ("autonomy_level", "topic_management")


def binding_attribute(payloads: TurnPayloads, hold: IXHold, fallback: str) -> str:
    """Whose declaration the failed binding belongs to.

    Live, one turn stamped all six `binding_invalid` rows
    `attribute: "autonomy_level"` while the response had declared
    `topic_management` batches and the rows carried `batch_id: "b1"` — so
    anything that keys held rows by attribute (the episode's `calls_held`, the
    release list) looked in the wrong place. The response says which
    payload declared ids; when exactly one did, that is the answer. When two did
    the wrapper cannot tell which id was meant, so it falls back rather than
    picking — a guess here is the same defect with better odds.
    """
    declared = [
        attribute for attribute, present in (
            ("autonomy_level", bool(payloads.plan)),
            ("topic_management", bool(payloads.batches)),
        ) if present
    ]
    if len(declared) == 1:
        return declared[0]
    open_here = [
        attribute for attribute in _ID_DECLARING_ATTRIBUTES
        if attribute in hold.open_episodes
    ]
    if len(open_here) == 1:
        return open_here[0]
    return fallback


def rebound_batch_ids(payloads: TurnPayloads, hold: IXHold) -> list[str]:
    """Batch ids this session already bound to a different set of calls (acc_023).

    Live t18 declared `bs_001` with `b1` = "Calendar deadline" binding `cal_1`
    and released both its batches; t20 re-declared `bs_001` with the same `b1`
    relabelled "Send confirmation email", now binding `snd_1`, and the wrapper
    released it. A batch id is the handle the user's consent and the audit both
    hang on, so within its set it has to mean one thing — otherwise an
    authorisation given for one topic is spent on another.

    Within its set: the id alone is not the handle. Keying this by `batch_id`
    made a *new* `batch_set_id` re-using `b1` a rebinding, which is what the
    `batch_set_id` description asks the PA to do once a set has closed. Live
    it cost acc_023 two turns, held `email_draft_create` twice, and put
    the wrapper's own stall text in front of the user; the PA escaped only by
    minting a fresh `c1` prefix.

    An id declared again with the *same* binding is a re-send, not a rebinding,
    and passes.
    """
    return sorted(
        batch.batch_id
        for batch in payloads.batches
        if (payloads.batch_set_id, batch.batch_id) in hold.batch_bindings
        and tuple(batch.call_binding)
        != hold.batch_bindings[(payloads.batch_set_id, batch.batch_id)]
    )


#: The refusal a rebound batch id gets: the same fail-closed exit as an invalid
#: binding, because it is one — the id no longer names the calls it was released
#: for.
IX_BATCH_REBIND_REPAIR = (
    f"{HOLD_PROMPT_PREFIX} No task tool ran this turn: batch {{batch_ids}} was "
    "already declared in this session binding different calls. A batch id names "
    "one topic for the whole session; reusing it for another is how an "
    "authorisation given for one topic gets spent on a different one. Resend "
    "this turn with a new batch id for the new topic, or with the binding the "
    "id already has. " + NO_META_ACKNOWLEDGEMENT
)


def validate_bindings(
    task_calls: Sequence[Any],
    declared: set[str],
    question_ids: set[str] | frozenset[str] = frozenset(),
    *,
    ids_declarable: bool = True,
) -> str | None:
    """Fail-closed binding check. Returns repair text, or None.

    Unknown or duplicate ``interaction_item_id`` means the whole batch of task
    calls is not executed and the model is asked once to resend it. So does a
    *missing* one, but only while some payload of this session could have
    declared an id: ``ids_declarable`` is
    ``session_active_ix & ID_DECLARING_ATTRIBUTES``. Only an
    ``IX_render_autonomy_level`` plan step and an ``IX_render_topic_management``
    batch declare call ids, so a turn governed by ``information_elicitation`` or
    ``capability_boundary`` alone can carry the field only by inventing an id
    that binds to nothing — live `acc_009` spent its first attempt on exactly
    that, and was then refused for the invented id. There, an absent id is
    correct and the call runs. Where a plan or a batch *is* the governing
    carrier, an absent id is still a refusal even on a response that declared
    none: that is the shape that once let seven world-changing calls out with
    ``interaction_item_id = None``.

    A question id is an *unknown* binding like any other — questions declare no
    call ids — but it is the one confusion the model makes on its
    own: live, the PA bound three task calls to `q1`/`q1`/`q2` on the turn it
    declared those questions, and read the generic "unknown id" back as "the
    field is not part of the tool definitions". `question_ids` only changes what
    the repair *says*; the refusal is the same.
    """
    seen: set[str] = set()
    problems: list[str] = []
    for tool_call in task_calls:
        args = getattr(tool_call, "arguments", None) or {}
        item_id = str(args.get(INTERACTION_ITEM_ID_FIELD) or "") if isinstance(args, dict) else ""
        name = getattr(tool_call, "name", "?")
        if not item_id:
            if ids_declarable:
                problems.append(f"{name}: missing {INTERACTION_ITEM_ID_FIELD}")
        elif item_id in question_ids and item_id not in declared:
            problems.append(
                f"{name}: question id used as call id ({item_id!r}) — a question "
                "declares no call binding"
            )
        elif declared and item_id not in declared:
            problems.append(f"{name}: unknown {INTERACTION_ITEM_ID_FIELD} {item_id!r}")
        elif item_id in seen:
            problems.append(f"{name}: duplicate {INTERACTION_ITEM_ID_FIELD} {item_id!r}")
        else:
            seen.add(item_id)
    if not problems:
        return None
    return (
        f"{HOLD_PROMPT_PREFIX} No task tool ran this turn: the interaction item "
        "bindings are not valid — " + "; ".join(problems)
        + f". Resend this batch of task calls with a correct {INTERACTION_ITEM_ID_FIELD} "
        "on every call, each one declared by this turn's IX_render payload. "
        + NO_META_ACKNOWLEDGEMENT
    )


#: The one repair `One-at-a-time` gets when the response arrives as a
#: single undivided batch. Asked once; a second undivided response is accepted
#: and recorded (`ix_batch_undivided`), because the wrapper cannot split topics
#: the model refuses to name.
IX_BATCH_SPLIT_REPAIR = (
    f"{HOLD_PROMPT_PREFIX} No task tool ran this turn. You told "
    "IX_render_topic_management that the user raised {item_count} topics, but "
    "one batch binds every call of this response, so nothing can be held back. "
    "Resend this turn with one batch per topic, each binding only the calls "
    "that belong to its own topic. " + NO_META_ACKNOWLEDGEMENT
)


def batch_decomposition_undivided(payloads: TurnPayloads, setting: str) -> bool:
    """True when a `One-at-a-time` turn declared batches that hold nothing back.

    `released_item_ids` releases the first batch, so a payload whose only batch
    binds everything releases the whole turn: live, a user asking for a calendar
    entry *and* an email got both, executed and reported together, under the
    setting that exists to prevent exactly that. `item_count` is the model's own
    count of the topics the user raised, which is what makes the mismatch
    checkable without the wrapper reading the user's message.

    The question is asked of the payload, never of this response's calls: a
    later turn legitimately sends only the calls of the batch it is releasing,
    and that is the setting working, not a turn that failed to divide.
    """
    if (setting or "").casefold() != "one-at-a-time" or payloads.item_count <= 1:
        return False
    if len(payloads.batches) < 2:
        return True
    bound = {item for batch in payloads.batches for item in batch.call_binding}
    return len(bound) > 1 and any(
        bound <= set(batch.call_binding) for batch in payloads.batches
    )


def plan_item_for(payloads: TurnPayloads, item_id: str) -> PlanItem | None:
    for item in payloads.plan:
        if item_id in item.bound_item_ids:
            return item
    return None


def is_capability_limit(result: Any) -> bool:
    """True for the stable capability-boundary code, not for ordinary errors."""
    payload = result
    if isinstance(result, str):
        import json

        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return False
    return (
        isinstance(payload, dict)
        and payload.get("ok") is False
        and payload.get("error_code") == "capability_limit"
    )


@dataclass
class RecurrenceRepair:
    """gpt acc_082 — one authorised calendar item owed a `recurrence` repair.

    `business_args` is the call's arguments with `recurrence` and the wrapper's
    own per-call fields taken out, so a repair is recognised only when the
    arrangement itself is unchanged: a different date, title, location or
    participant list is a different act and goes back through the gate.
    """

    tool: str
    business_args: str
    spent: bool = False


# ---------------------------------------------------------------------------
# the collector
# ---------------------------------------------------------------------------

@dataclass
class TurnCollector:
    """Per-turn raw material for the renderer."""

    turn_index: int = 0
    segments: list[str] = field(default_factory=list)
    provider_reasoning: list[str] = field(default_factory=list)
    calls: list[ToolCallRecord] = field(default_factory=list)
    gate: GateRecord | None = None
    gate_remaining: list[PlanItem] = field(default_factory=list)
    payloads: TurnPayloads = field(default_factory=TurnPayloads)
    held: list[HeldItem] = field(default_factory=list)
    consent: list[ConsentRecord] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    #: Repair prompt for a fail-closed binding batch; "" when the batch is fine.
    repair: str = ""
    #: Repair prompt for an undivided `One-at-a-time` turn, and how many
    #: times it has been sent this turn (it is sent at most once).
    batch_repair: str = ""
    batch_repair_attempts: int = 0
    #: Provider tool_call ids that never ran, so no tool result is appended.
    skipped_call_ids: set[str] = field(default_factory=set)
    capability_limit_seen: bool = False
    #: `interaction_item_id`s that have already spent their release
    #: on a world-changing call. Seeded from the open episodes at the first
    #: gate of the turn, so the limit holds across the turn's iterations and
    #: across turns while the episode is open.
    executed_item_ids: set[str] = field(default_factory=set)
    #: `{attribute: setting}` this turn was rendered with, kept so the `ix_render` blob
    #: can re-run the pure per-attribute renderers for its audit rows.
    settings: dict[str, str] = field(default_factory=dict)
    #: The prose of the response currently being handled, "" when it wrote none.
    #: A hold raised while handling it owns exactly this text -- reading
    #: `segments[-1]` instead would steal an earlier iteration's prose whenever
    #: the gated response was tool calls only.
    current_segment: str = ""
    #: Prose written in the same assistant message as a call the
    #: gate held. It narrates work that did not happen, so it is audit-only.
    held_narration: list[HeldNarration] = field(default_factory=list)
    #: Whether the one closing request after a batch / question hold
    #: came back with prose. `None` means no closing request was made this turn.
    #: The audit blob has no other record of it: live it was visible only as an
    #: extra provider call whose prompt was ~14k tokens shorter.
    closing_iteration: bool | None = None
    #: The turn would deliver nothing and the one extra
    #: provider request the runner sends came back empty too. It is the runner
    #: that decides this; the renderer is the one allowed to write the line.
    empty_reply: bool = False
    #: acc_006 — some response this turn carried the model's own
    #: tool-call syntax as text, recovered at the provider seam or left
    #: unparsable. Either way the PA had work in hand, so the empty-reply
    #: template is a false statement about this turn.
    dsml_markup_seen: bool = False
    #: The fail-closed binding exit fired: a second response
    #: bound its task calls wrongly and the whole batch is gone for this turn.
    #: A *repaired* first refusal never sets it, so the close line cannot land
    #: on a turn whose retry actually ran the calls.
    binding_refused: bool = False
    #: acc_004 — `{(tool, error class): failures}` for this turn, where
    #: the class is the error text with the call's own argument values taken out.
    #: Live, `shopping_order_create` was called 46 times with guessed
    #: `delivery_address_id`s (addr_home, addr_001, addr_4b, ...) and answered
    #: "unknown delivery address" every time; the runner reads this to stop the
    #: tool after the third identical failure and send the PA to the user.
    tool_failure_strikes: dict[tuple[str, str], int] = field(default_factory=dict)
    #: gpt acc_082 — the one exception to the strike, by the authorised item id
    #: it belongs to. A `calendar_event_create` refused on `recurrence` alone is
    #: a parameter fault on a call the user has already authorised, and nothing
    #: was written, so the third refusal must not read as "ask the user for the
    #: value it needs": there is no missing fact. Keyed on the item id and not
    #: on the tool, so two nearby calendar to-dos carry their own budgets, and
    #: spent once -- a repair that fails again is over.
    recurrence_repairs: dict[str, RecurrenceRepair] = field(default_factory=dict)
    #: acc_006 — `completion_tokens` of the response currently in
    #: hand when the provider ended it at the token limit, `None` when the last
    #: response came back whole. Only the *last* response matters: a truncation
    #: the retry answered is over, and the turn that follows it is a normal one.
    truncated_completion_tokens: int | None = None

    @property
    def boundary_declared(self) -> bool:
        return self.payloads.boundary is not None

    @property
    def last_segment(self) -> str:
        """The loop's fallback text when the renderer produced no message.

        It is a delivery surface, so everything the renderer strips is stripped
        here too: the certainty block, the position markers, a leading sentence
        that answers an internal repair message -- and the free prose of a
        segment that shares its assistant message with a call the gate held.
        """
        segments = delivered_segments(self.segments, self.held_narration)
        last = next(
            (segment for segment in reversed(segments) if segment and segment.strip()),
            "",
        )
        if not last:
            return ""
        text, _dropped = strip_meta_acknowledgement(strip_certainty_block(last))
        return strip_internal_markers(text)

    def record_for(self, tool_call_id: str) -> ToolCallRecord | None:
        for record in reversed(self.calls):
            if record.tool_call_id == tool_call_id:
                return record
        return None

    def update_call(
        self,
        tool_call_id: str,
        *,
        status: str,
        error: str = "",
        executed: bool | None = None,
    ) -> None:
        for index, record in enumerate(self.calls):
            if record.tool_call_id == tool_call_id:
                self.calls[index] = replace(
                    record,
                    status=status,
                    error=error,
                    executed=record.executed if executed is None else executed,
                )
                return

    def add_segment(self, text: str | None) -> None:
        # Reset first: this is the one call per provider response, so a response
        # that wrote no prose has to leave `current_segment` empty rather than
        # inherit the previous iteration's.
        self.current_segment = ""
        if not (text and text.strip()):
            return
        self.current_segment = text
        # `segments` stays raw: the block is part of what the model wrote, and
        # `audit_trace` is the one place it is allowed to survive.
        self.segments.append(text)
        # Carrier F — the trailing block is the primary certainty carrier,
        # because it quotes prose the model has already written; the tool call
        # is emitted before that prose exists and can only promise a span. A
        # block therefore replaces the tool payload; no block leaves it alone.
        block = parse_certainty_block(text)
        if block.found:
            self.payloads = replace(self.payloads, certainties=block.rows)

    @property
    def _delivered_index(self) -> int | None:
        """Which segment `render()` delivers: the renderer's own `_final_segment` pick."""
        segments = delivered_segments(self.segments, self.held_narration)
        for index in reversed(range(len(segments))):
            if segments[index] and segments[index].strip():
                return index
        return None

    @property
    def delivered_segment(self) -> str:
        """That segment's raw text — `last_segment` is the same pick, cleaned.

        A rule that *rewrites* the delivered prose needs the uncleaned form, or
        the rewrite would strip the `[[batch:…]]` / `[[q:…]]` markers the
        renderer selects sections by and release what the setting held.
        """
        index = self._delivered_index
        if index is None:
            return ""
        return delivered_segments(self.segments, self.held_narration)[index]

    def replace_delivered_segment(self, text: str) -> None:
        """Rewrite that segment. Empty `text` blanks every segment instead.

        The turn's prose was cut off, so an earlier iteration's interstitial
        narration must not stand in for it — blanking only the one segment
        would just move the renderer's pick an iteration further back.
        """
        if not text.strip():
            self.segments = ["" for _ in self.segments]
            return
        index = self._delivered_index
        if index is not None:
            self.segments[index] = text

    def add_reasoning(self, *texts: str | None) -> None:
        for text in texts:
            if text and str(text).strip():
                self.provider_reasoning.append(str(text))

    def add_payloads(self, tool_calls: Sequence[Any]) -> None:
        self.payloads = collect_payloads(tool_calls, self.payloads)

    def record_call(
        self,
        tool_call: Any,
        *,
        action_type: str,
        executed: bool,
        status: str = "ok",
        error: str = "",
    ) -> ToolCallRecord:
        args = getattr(tool_call, "arguments", None) or {}
        args = args if isinstance(args, dict) else {}
        record = ToolCallRecord(
            order=len(self.calls),
            tool_name=getattr(tool_call, "name", ""),
            args={k: v for k, v in args.items() if k not in PER_CALL_INJECTED_FIELDS},
            tool_call_id=str(getattr(tool_call, "id", "") or ""),
            notice=str(args.get(IX_NOTICE_FIELD) or ""),
            interaction_item_id=str(args.get(INTERACTION_ITEM_ID_FIELD) or ""),
            ask_text=str(args.get(IX_CONFIRMATION_MESSAGE_FIELD) or ""),
            action_type=action_type,
            status=status,
            error=error,
            executed=executed,
        )
        self.calls.append(record)
        return record

    def _mint_hold_id(self, record: ToolCallRecord) -> ToolCallRecord:
        """Give a held call an id when the PA bound it to none (acc_009).

        A `questions_gate` held `email_draft_create` with
        `interaction_item_id: ""`, so `held.calls` exposed no id, the harness
        reported `held ids: []`, and both of the user's consent grants — "The
        draft is acceptable. Send it." — were dropped as unknown ids. A hold
        nobody can name is unreleasable by construction, which is the cost side
        of requiring the field only where autonomy / topic are active.

        Derived from the provider's own call id, so a replay of the same turn
        mints the same id, and written back into `calls` because the audit rows
        and the release path both read the id off the call record, not off the
        hold.
        """
        minted = replace(
            record,
            interaction_item_id=(
                f"held_{record.tool_call_id or f'{record.order}_{record.tool_name}'}"
            ),
        )
        for index, existing in enumerate(self.calls):
            if existing is record:
                self.calls[index] = minted
                break
        return minted

    def hold_call(
        self,
        record: ToolCallRecord,
        *,
        attribute: str,
        reason: str,
        plan_item_id: str = "",
        episode_id: str = "",
        batch_id: str = "",
    ) -> None:
        if not record.interaction_item_id:
            record = self._mint_hold_id(record)
        self.held.append(HeldItem(
            status=held_status(attribute, reason),
            interaction_item_id=record.interaction_item_id,
            attribute=attribute,
            tool=record.tool_name,
            plan_item_id=plan_item_id,
            episode_id=episode_id,
            batch_id=batch_id,
            reason=reason,
        ))
        # The prose of this same assistant message describes the
        # call that is being stopped -- live, `📧 Email confirmation sent to
        # Dr. Okafor` while the send was held. It goes to the audit plane with
        # the call, never to the user.
        if self.current_segment:
            narration = HeldNarration(
                item_id=record.interaction_item_id, text=self.current_segment,
            )
            # A repaired retry can write the same prose again, and the same
            # message can hold several calls: one row per (id, text), not one
            # per hold.
            if narration not in self.held_narration:
                self.held_narration.append(narration)

    def clear_holds(self) -> None:
        """Drop this turn's hold bookkeeping so a repaired retry starts clean.

        Used when the wrapper sends the model back for a payload it owed (a
        gate with no ask). The records of the refused attempt would
        otherwise be written into an episode and shipped as `held.calls`,
        reporting a hold that the retry has already answered.

        `consent[]` follows `calls`, not `held`: a row for an id whose call
        already **executed** records an authorisation the world has acted on,
        and the retry will not re-issue that call, so wiping it loses the row
        for good. Live, one turn released `item_2`, ran `appointment_create`,
        then opened a gate for `item_3` with no ask -- the repair cleared the
        turn and `consent[]` shipped empty, so two authorisations in the run
        were recorded as one.
        """
        kept = [call for call in self.calls if call.executed]
        executed_ids = {
            call.interaction_item_id for call in kept if call.interaction_item_id
        }
        self.calls = [replace(call, order=index) for index, call in enumerate(kept)]
        self.held.clear()
        self.consent[:] = [row for row in self.consent if row.item_id in executed_ids]
        self.skipped_call_ids.clear()
        self.gate = None
        self.gate_remaining = []

    def mark_binding_repaired(self, repaired_by: str) -> None:
        """Stamp this turn's fail-closed refusals as answered by `repaired_by`.

        A `binding_invalid` refusal opens no episode, so its rows only ever
        reach `audit_trace` — and there live `acc_009` read them as steps still
        outstanding, on the very turn whose retry carried them out. Kept rather
        than cleared, because the refusal is real history; the flag is what
        separates "not done" from "refused once, then done".
        """
        self.held[:] = [
            replace(item, repaired=True, repaired_by=repaired_by)
            if item.reason == "binding_invalid" and not item.repaired
            else item
            for item in self.held
        ]

    def emit(self, name: str, **fields: Any) -> None:
        self.events.append({"ix_event": True, "name": name, **fields})

    @property
    def executed_call_ids(self) -> list[str]:
        return [call.tool_call_id for call in self.calls if call.executed and call.tool_call_id]

    @property
    def held_call_ids(self) -> list[str]:
        return [
            call.tool_call_id
            for call in self.calls
            if not call.executed and call.tool_call_id
        ]

    def to_render_input(
        self,
        *,
        hold: IXHold,
        consent_decisions: Sequence[ConsentDecision] = (),
        proactive: ProactiveDecision | None = None,
    ) -> IXRenderInput:
        return IXRenderInput(
            turn_index=self.turn_index,
            segments=list(self.segments),
            provider_reasoning=list(self.provider_reasoning),
            calls=list(self.calls),
            gate=self.gate,
            gate_remaining=list(self.gate_remaining),
            payloads=self.payloads,
            proactive_decision=proactive,
            consent_decisions=list(consent_decisions),
            hold=hold,
            act_releases=live_act_releases(
                hold, self.payloads, consent_decisions, self.calls
            ),
            held_narration=list(self.held_narration),
            empty_reply=self.empty_reply,
            binding_refused=self.binding_refused,
        )


# ---------------------------------------------------------------------------
# the renderer call site
# ---------------------------------------------------------------------------

_MARKER_RES = (BATCH_MARKER_RE, QUESTION_MARKER_RE, CERTAINTY_MARKER_RE)


def strip_internal_markers(text: str) -> str:
    """Remove ``[[batch:…]]`` / ``[[q:…]]`` / ``[[c:…]]``.

    Plus anything else still wearing double brackets: an id shape the wrapper
    does not recognise is still an internal control marker, and shipping it raw
    is the one outcome ruled out.
    """
    for pattern in (*_MARKER_RES, UNRECOGNISED_MARKER_RE):
        text = pattern.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def render_turn(
    render_input: IXRenderInput,
    settings: Mapping[str, str],
    *,
    held: Sequence[HeldItem] = (),
    executed_call_ids: Sequence[str] = (),
    held_call_ids: Sequence[str] = (),
) -> RenderedTurn:
    """The single renderer call site (no bypass, no second entry point).

    The gate records which calls actually ran, so those two lists are passed
    back in rather than re-derived: the renderer is pure and must not have to
    reason about execution.
    """
    rendered = render(render_input, dict(settings), held_items=held)
    if not rendered.executed_call_ids and executed_call_ids:
        rendered = replace(
            rendered,
            executed_call_ids=list(executed_call_ids),
            held_call_ids=list(held_call_ids),
        )
    if not rendered.held and held:
        rendered = replace(rendered, held=list(held))
    return rendered


# ---------------------------------------------------------------------------
# hold episodes
#
# The write-back itself belongs to the renderer (`render.py` returns
# `hold_next`): one owner, so a turn cannot advance an episode twice. What
# lives here is what the loop needs around it -- the step-boundary check and
# the controlled reminder the PA sees next turn.
# ---------------------------------------------------------------------------

def open_episode_ids(hold: IXHold) -> list[tuple[str, str]]:
    """``(attribute, episode_id)`` still open — a step boundary must have none."""
    return [
        (attribute, episode.episode_id)
        for attribute, episode in hold.open_episodes.items()
        if not episode.closed
    ]


def hold_prompt(hold: IXHold, unreleased: Sequence[str] = ()) -> str:
    """The controlled reminder: hold ids only, never the user's own words.

    An episode holding only questions or batches gets the reminder too. It used
    to be skipped for having no held *call*, so an
    `Iterative` turn that withheld `q2` and made no tool call left the next turn
    with nothing to resume from, while `IX_continue_information_elicitation` was
    still required. Held questions and batches are listed by id and their short
    label -- never `Question.text` / `Batch.body`, which stay in `audit_trace`:
    the PA rewrites in its own words.

    The two marker kinds settle differently, so each gets its own sentence next
    to its own ids. A carried *question* with a bare `release` folds into
    the reply -- "a reply carrying none of them is released whole". A carried
    *batch* has no such fold: it stays held until its `[[batch:id]]` marker goes
    out, a bound call runs, the PA reports `skip`, or the round cap cancels it.
    One sentence for both said the question rule twice.
    """
    lines: list[str] = []
    for attribute, episode in hold.open_episodes.items():
        if episode.closed:
            continue
        held = [
            f"{call.get(INTERACTION_ITEM_ID_FIELD)} ({call.get('tool')})"
            for call in episode.calls_held
        ]
        marker_sentences: list[str] = []
        if episode.batches_held:
            marker_sentences.append(
                "[[batch:id]] markers in your prose are what deliver a held "
                "batch; one you leave unmarked is still held when the episode "
                f"closes: {', '.join(batch.batch_id for batch in episode.batches_held)}."
            )
        if episode.questions_held:
            marker_sentences.append(
                "[[q:id]] markers in your prose are what divide this reply; a "
                "reply carrying none of them is released whole: "
                f"{', '.join(question.id for question in episode.questions_held)}."
            )
        held += [
            f"{item_id} ({label.strip()})" if label.strip() else item_id
            for item_id, label in (
                [(batch.batch_id, batch.label) for batch in episode.batches_held]
                + [(question.id, question.about) for question in episode.questions_held]
            )
        ]
        if not held:
            continue
        line = (
            f"{HOLD_PROMPT_PREFIX} You still hold: {', '.join(held)}. Call "
            f"IX_continue_{attribute} and give every id a release / keep / skip. "
            "When you release an outward-facing item, fill user_evidence with the "
            "user's own words from their latest message (audit field)."
        )
        if episode.calls_held:
            # acc_009 t=12/t=14: a `questions_gate` stopped `email_draft_create`
            # before it ran and told the PA nothing, so the PA showed the user a
            # draft that did not exist and then announced a send that never
            # happened. A held call produces no tool result by design, so this
            # sentence is the only thing that says the call did not run.
            line += (
                " Nothing ran for "
                + ", ".join(sorted({
                    str(call.get("tool") or "")
                    for call in episode.calls_held if call.get("tool")
                }))
                + "; those calls were stopped before they executed, changed "
                "nothing, and must not be reported to the user as done or sent."
            )
        if marker_sentences:
            line += (
                " " + " ".join(marker_sentences)
                + " Say it in your own words — nothing you held is reused."
            )
        lines.append(line)
    if unreleased:
        lines.append(
            f"{HOLD_PROMPT_PREFIX} Not cleared for execution last turn: "
            + ", ".join(unreleased)
            + "."
        )
    # A cancelled item is a terminal fact, and the PA is told it on
    # **every** later turn of the session -- not only on the turn it happened.
    # Live, the round cap dropped `email1` in silence and three turns
    # later the PA's own summary listed the never-sent email under "Completed".
    cancelled: list[str] = []
    for episode in hold.finished_episodes:
        for entry in episode.cancelled:
            intent = str(entry.get("intent") or "").strip()
            item_id = str(entry.get("id") or "")
            cancelled.append(f"{item_id} ({intent})" if intent else item_id)
    if cancelled:
        lines.append(
            f"{HOLD_PROMPT_PREFIX} "
            + "; ".join(dict.fromkeys(cancelled))
            + ": cancelled, never executed — do not describe it as done."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the persistence projections
# ---------------------------------------------------------------------------

#: The payload carriers whose persisted **result** is replaced. Their
#: `execute()` echoes the arguments back verbatim, so a held `Question.text` or
#: `Batch.body` landed in `runtime_history` twice: once as the PA's own
#: declaration, once as a tool result that read like a delivered answer. The
#: declaration is PA-internal state and stays; the echo is a duplicate that also
#: hid the one fact the record owed — that the item was **not** released.
IX_PAYLOAD_RESULT_TOOLS: frozenset[str] = frozenset(
    (*IX_RENDER_TOOL_NAMES.values(), *IX_CONTINUE_TOOL_NAMES.values())
)


def ix_payload_status(
    *,
    released: Sequence[str] = (),
    held_not_released: Sequence[str] = (),
    cancelled: Sequence[str] = (),
) -> str:
    """The persisted result body for an `IX_render_*` / `IX_continue_*` call.

    Ids only, in call order, never a word of the payload's own text.

    The three lists are **per turn**, and together they say what became of the
    turn's payload sections: `released` went out, `held_not_released` is still
    held, and `cancelled` hit `HOLD_ROUNDS_HARD_CAP` and will never go out.
    `cancelled` exists because an empty `held_not_released`
    does not mean everything was delivered — a forced close empties the hold by
    cancelling it — and the memory pool decides on both lists: a payload call's
    arguments are the delivered text and are remembered, unless this turn held
    or cancelled something, in which case they are text nobody received.
    """
    return json.dumps(
        {
            "status": "rendered",
            "released": [item for item in dict.fromkeys(released) if item],
            "held_not_released": [
                item for item in dict.fromkeys(held_not_released) if item
            ],
            "cancelled": [item for item in dict.fromkeys(cancelled) if item],
        },
        ensure_ascii=False,
    )


def build_runtime_history(
    *,
    user_message: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    executed_call_ids: Sequence[str],
    public_turn: str,
    released: Sequence[str] = (),
    held_not_released: Sequence[str] = (),
    cancelled: Sequence[str] = (),
) -> RuntimeHistory:
    """Projection 2: user message once → executed subset → public turn.

    The assistant protocol message is *copied* and its ``tool_calls`` filtered
    to ``executed_call_ids``; only those ids' results follow. Held call
    declarations are dropped entirely, so declared ids and result ids stay
    equal in order and no synthetic tool result is ever needed.

    The PA's own ``IX_render_*`` / ``IX_continue_*`` *arguments* stay as it
    wrote them — the payload is the PA's internal state and it may read its own
    back next turn. Their **results** do not: ``IXPerTurnTool.execute()`` echoes
    the whole payload, which filed every held ``Question.text`` and
    ``Batch.body`` in the record a second time and said nothing about their
    having been held. :data:`IX_PAYLOAD_RESULT_TOOLS` results are replaced by
    :func:`ix_payload_status`, which drops the echo and states the hold.
    """
    executed = list(dict.fromkeys(executed_call_ids))
    executed_set = set(executed)
    status = ix_payload_status(
        released=released, held_not_released=held_not_released, cancelled=cancelled,
    )
    payload_call_ids = {
        str(call.get("id") or "")
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
        if str((call.get("function") or {}).get("name") or "") in IX_PAYLOAD_RESULT_TOOLS
    }
    protocol: list[dict[str, Any]] = []
    for message in messages:
        if message.get("_mempa_internal") is True:
            continue
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            kept = [
                call for call in message["tool_calls"]
                if str(call.get("id") or "") in executed_set
            ]
            if kept:
                # The prose of an intermediate iteration was never shown, so it
                # must not survive as assistant text in the runtime history the
                # PA is re-shown: only the call declarations carry over. This
                # says nothing about memory — the pool is built from the raw
                # loop messages and keeps that prose as its own block.
                protocol.append({**dict(message), "tool_calls": kept, "content": ""})
        elif role == "tool" and str(message.get("tool_call_id") or "") in executed_set:
            record = dict(message)
            if (
                str(record.get("tool_call_id") or "") in payload_call_ids
                or str(record.get("name") or "") in IX_PAYLOAD_RESULT_TOOLS
            ):
                record["content"] = status
            protocol.append(record)

    declared = [
        str(call.get("id") or "")
        for message in protocol
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]
    results = [
        str(message.get("tool_call_id") or "")
        for message in protocol
        if message.get("role") == "tool"
    ]
    if declared != results:
        raise ValueError(
            "runtime_history tool protocol is not well formed: "
            f"declared={declared} results={results}"
        )

    return RuntimeHistory(
        user_message=dict(user_message),
        tool_protocol=protocol,
        assistant_message={"role": "assistant", "content": public_turn},
    )

