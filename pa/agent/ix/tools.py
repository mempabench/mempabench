"""Model-facing IX tools: per-session selection, per-turn payload, continuation.

Five carriers (IX wrapper plan §5.3):

- A `IX_<attribute>`            — the setting for this session, shown once
- B `IX_render_<attribute>`     — this turn's payload, six attributes (§3.0)
- C three per-call fields injected onto task tool schemas (`registry.py`)
- D `IX_continue_<attribute>`   — consumes an existing hold, four attributes (§3.7)
- E `IX_capability_resolve`     — forced boundary declaration after `capability_limit` (§3.5)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pa.agent.ix.render_types import (
    IX_CAPABILITY_RESOLVE_TOOL,
    IX_CONTINUE_TOOL_NAMES,
    IX_RENDER_TOOL_NAMES,
    IX_TOOL_NAME_PREFIX,
    RequiredContext,
    RequiredMode,
    RequiredPredicate,
)
from pa.agent.ix.rationale import (
    CERTAINTY_BLOCK_INSTRUCTION,
    has_certainty_block,
)
from pa.agent.ix.specs import (
    SETTING_COMMITMENT_NOTE,
    UNMARKED_PROSE_NOTE,
    IXToolSpec,
)
from pa.agent.tools.base import Tool


class IXPreferenceTool(Tool):
    """A per-turn tool for choosing one active IPaS preference setting."""

    def __init__(self, spec: IXToolSpec, *, proactive_event: bool = False):
        self._spec = spec
        self._proactive_event = proactive_event and spec.attribute == "proactive_outreach"

    @property
    def spec(self) -> IXToolSpec:
        return self._spec

    @property
    def attribute(self) -> str:
        return self._spec.attribute

    def is_required(self, context: RequiredContext) -> bool:
        """Selection tools are `always` required; the slot is shared with the
        per-turn tools so the registry can evaluate one rule for every IX tool."""
        if self._spec.required == "always":
            return True
        if self._spec.required == "never":
            return False
        assert self._spec.required_predicate is not None  # guarded in IXToolSpec
        return self._spec.required_predicate(context)

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        settings = "; ".join(
            f"{setting}: {self._spec.setting_descriptions[setting]}"
            for setting in self._spec.settings
        )
        description = (
            f"Choose the {self._spec.label} interaction preference setting for this turn. "
            f"Call this tool before the final answer when it is available. "
            f"Allowed settings: {settings}. "
            f"{SETTING_COMMITMENT_NOTE}"
        )
        if self._proactive_event:
            description += (
                " This turn was started by an environment event while no user request "
                "is active. Also decide whether to send a new user contact or suppress "
                "it. Either way, provide the exact concise user-visible message you "
                "would send; it is delivered only when you choose send. Low requires "
                "suppress and High requires send."
            )
        return description

    @property
    def parameters(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "setting": {"type": "string", "enum": list(self._spec.settings)},
            "evidence": {
                "type": "string",
                "description": (
                    "Brief evidence from the user message, memory/profile, or context. "
                    "Leave it empty when you have none: the first time you meet this "
                    "attribute for this user there is nothing to cite, and an empty "
                    "string is the correct answer there. Never invent evidence to fill it."
                ),
            },
            "application": {
                "type": "string",
                "description": "How this setting will shape the next PA response.",
            },
        }
        required = ["setting"]
        if self._proactive_event:
            properties.update({
                "outreach_action": {
                    "type": "string",
                    "enum": ["send", "suppress"],
                },
                "outreach_message": {
                    "type": "string",
                    "description": (
                        "Exact user-visible contact you would send. Always write it, "
                        "including on suppress, where it is kept for audit and never "
                        "delivered."
                    ),
                },
            })
            required.extend(["outreach_action", "outreach_message"])
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    async def execute(
        self,
        setting: str,
        evidence: str | None = None,
        application: str | None = None,
        outreach_action: str | None = None,
        outreach_message: str | None = None,
    ) -> str:
        payload = {
            "type": "ix_preference_selection",
            "attribute": self._spec.attribute,
            "setting": setting,
            "evidence": evidence or "",
            "application": application or "",
        }
        if self._proactive_event:
            if outreach_action not in {"send", "suppress"}:
                return "Error: outreach_action must be 'send' or 'suppress'."
            if setting == "Low" and outreach_action != "suppress":
                return "Error: Proactive Outreach Low requires outreach_action='suppress'."
            if setting == "High" and outreach_action != "send":
                return "Error: Proactive Outreach High requires outreach_action='send'."
            message = (outreach_message or "").strip()
            if not message:
                # §3.1 proactive row: the model always writes the contact, so a
                # suppressed candidate can be audited against a sent one.
                return "Error: outreach_message is required for send and for suppress."
            payload["outreach_action"] = outreach_action
            payload["outreach_message"] = message
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Required predicates — §7.1 P0 "response-evaluated conditionally-required"
# ---------------------------------------------------------------------------
# Each predicate is pure over `RequiredContext` (frozen in P0) and is evaluated
# after this turn's provider response is in hand, at the existing check in
# `pa/agent/runner.py:161-174`. Required-but-not-called inserts one nudge and
# retries the turn once; not required means not shown and not nudged.


def _open_episode(context: RequiredContext, attribute: str):
    episode = context.hold.open_episodes.get(attribute)
    if episode is None or episode.closed:
        return None
    return episode


def episode_has_held_items(episode) -> bool:
    return bool(
        episode.plan_remaining
        or episode.batches_held
        or episode.questions_held
        or episode.calls_held
    )


def _has_task_call(context: RequiredContext) -> bool:
    """A task call is any call that is not one of the IX internal tools."""
    return any(
        not name.startswith(IX_TOOL_NAME_PREFIX) for name in context.response_tool_names
    )


def requires_visible_message(context: RequiredContext) -> bool:
    """§3.0 — required on every turn that produces a user-visible message.

    Used by the three payloads whose absence would make the message unrenderable
    under at least one setting: rationale, certainties, completion message.
    Required under every setting of the attribute, so requiredness leaks nothing.
    """
    return context.produces_user_visible_message


def requires_certainty_payload(context: RequiredContext) -> bool:
    """§3.0 — required on a visible-message turn that carries no trailing block.

    The `<certainties>` block (§5.3 carrier F) is the primary carrier: it quotes
    prose the model has already written, which a tool call emitted *before* that
    prose cannot do. With a block in the response the tool payload is optional;
    with neither, the turn owes a payload and falls to the generic
    one-reminder-then-contract-failure rule, exactly as it did before.

    Requiredness still leaks no setting: the block is asked for identically in
    all three, and an empty `<certainties></certainties>` is a valid answer.
    """
    return context.produces_user_visible_message and not has_certainty_block(
        context.response_text
    )


def requires_new_plan_binding(context: RequiredContext) -> bool:
    """§3.0 — required when the response carries an item id the open plan has not claimed."""
    episode = _open_episode(context, "autonomy_level")
    claimed = {
        item_id
        for plan_item in (episode.plan_remaining if episode is not None else [])
        for item_id in plan_item.bound_item_ids
    }
    return any(item_id not in claimed for item_id in context.response_interaction_item_ids)


def requires_batch_payload(context: RequiredContext) -> bool:
    """§3.0 — required when the response has a task call or a `[[batch:…]]` marker."""
    return _has_task_call(context) or any(
        marker.startswith("batch:") for marker in context.response_markers
    )


def requires_open_elicitation(context: RequiredContext) -> bool:
    """§3.0 — required on every turn while the attribute is active.

    It used to be required only while an elicitation episode was open, and only
    this tool can open one: nothing bootstrapped it, so across two live acc_009
    runs it was called on 0 of 20 PA turns and `Infer` / `Structured` /
    `Iterative` were behaviourally identical. Its two siblings both have a
    bootstrap trigger (`requires_batch_payload` fires on any task call,
    `requires_new_plan_binding` on an unclaimed id); this one now fires on the
    attribute being active, which is the same "you owe this payload every turn"
    rule the three per-message carriers use. An empty `questions` list is a
    valid payload, so requiredness still leaks no setting.
    """
    return (
        "information_elicitation" in context.session_active_ix
        or _open_episode(context, "information_elicitation") is not None
    )


def requires_capability_declaration(context: RequiredContext) -> bool:
    """§3.5 (part 4) — required on the iteration after a tool returned `capability_limit`."""
    return context.capability_limit_seen


def hold_release_predicate(attribute: str) -> RequiredPredicate:
    """§3.7 (part 2) — `IX_continue_<attribute>` is required exactly when its hold is non-empty."""

    def _predicate(context: RequiredContext) -> bool:
        episode = _open_episode(context, attribute)
        return episode is not None and episode_has_held_items(episode)

    return _predicate


# ---------------------------------------------------------------------------
# Carriers B / D / E — one generic tool driven by a spec table
# ---------------------------------------------------------------------------

RENDER_PAYLOAD_TYPE = "ix_render_payload"
CONTINUE_PAYLOAD_TYPE = "ix_continue"
CAPABILITY_RESOLVE_PAYLOAD_TYPE = "ix_capability_resolve"


@dataclass(frozen=True)
class IXPerTurnToolSpec:
    """One per-turn IX tool: payload carrier, continuation, or boundary declaration."""

    name: str
    attribute: str
    payload_type: str
    description: str
    parameters: dict[str, Any]
    required: RequiredMode = "predicate"
    required_predicate: RequiredPredicate | None = None

    def __post_init__(self) -> None:
        if (self.required == "predicate") != (self.required_predicate is not None):
            raise ValueError(
                f"{self.name} required='predicate' needs a required_predicate, "
                "and a required_predicate needs required='predicate'"
            )


class IXPerTurnTool(Tool):
    """Model-facing tool for a per-turn payload, a hold release, or a boundary."""

    def __init__(self, spec: IXPerTurnToolSpec):
        self._spec = spec

    @property
    def spec(self) -> IXPerTurnToolSpec:
        return self._spec

    @property
    def attribute(self) -> str:
        return self._spec.attribute

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._spec.parameters

    def is_required(self, context: RequiredContext) -> bool:
        if self._spec.required == "always":
            return True
        if self._spec.required == "never":
            return False
        assert self._spec.required_predicate is not None  # guarded in __post_init__
        return self._spec.required_predicate(context)

    async def execute(self, **params: Any) -> str:
        payload = {
            "type": self._spec.payload_type,
            "attribute": self._spec.attribute,
            **params,
        }
        return json.dumps(payload, ensure_ascii=False)


_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}


def _render_spec(
    attribute: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    predicate: RequiredPredicate,
) -> IXPerTurnToolSpec:
    return IXPerTurnToolSpec(
        name=IX_RENDER_TOOL_NAMES[attribute],
        attribute=attribute,
        payload_type=RENDER_PAYLOAD_TYPE,
        description=description,
        parameters={"type": "object", "properties": properties, "required": required},
        required_predicate=predicate,
    )


IX_RENDER_TOOL_SPECS: dict[str, IXPerTurnToolSpec] = {
    "reasoning_visibility": _render_spec(
        "reasoning_visibility",
        (
            "Submit this turn's user-facing rationale. Call it on every turn that "
            "produces a message to the user, under both settings — whether it is "
            "required never reveals which setting is active. Show prints it ahead of "
            "your answer; Hide keeps it out of the user's view."
        ),
        {
            "user_facing_rationale": {
                "type": "string",
                "description": (
                    "The evidence, decision factors and causal links a reader needs to "
                    "audit your conclusion — deliverable prose, not private "
                    "chain-of-thought and not a list of the tools you called. Write it "
                    "to the user, in the second person: it gives the reasons behind the "
                    "answer, and it is never a recap of what they asked for or of what "
                    "you did, written about them in the third person. Bad, a work log: "
                    "'The user asked me to prepare the checklist. I allocated the 30 "
                    "minutes into three segments. I added the pharmacy note.' Good, two "
                    "sentences of evidence and the deciding factor: 'The clinic's "
                    "Tuesday slot is the only one before the workshop; I chose it over "
                    "Thursday because the lab handover runs through the 19th.' A turn "
                    "that makes no claim — an acknowledgement, a closing line, anything "
                    "with nothing decided in it — has no reasons to give: send the "
                    "field empty rather than filling it."
                ),
            }
        },
        ["user_facing_rationale"],
        requires_visible_message,
    ),
    "uncertainty_expression": _render_spec(
        "uncertainty_expression",
        (
            "Submit this turn's confidence levels. The place to do it is the block at "
            "the end of your reply, not this tool, because that block quotes prose you "
            "have already written while this call happens before it exists.\n"
            + CERTAINTY_BLOCK_INSTRUCTION
            + "\nCall this tool only on a turn where you cannot end with that block — "
            "then list the same rows here, each entry's `claim` being a span you commit "
            "to writing verbatim in the reply, and pass an empty list when no claim's "
            "confidence matters. Either way the confidence is attached by the renderer "
            "at the claim it locates — do not hedge in the prose."
        ),
        {
            "certainties": {
                "type": "array",
                "description": (
                    "One entry per claim in this turn's reply whose confidence matters."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "description": "A short id for this claim, unique within this call.",
                        },
                        "level": {"type": "string", "enum": ["high", "medium", "low"]},
                        "claim": {
                            "type": "string",
                            "description": (
                                "The exact span of your reply this confidence is about — "
                                "one sentence or clause, written here word for word as it "
                                "will appear in the reply, so it can be found in the text. "
                                "Not a paraphrase and not a summary of the whole turn."
                            ),
                        },
                    },
                    "required": ["ref", "level", "claim"],
                },
            }
        },
        ["certainties"],
        requires_certainty_payload,
    ),
    "process_visibility": _render_spec(
        "process_visibility",
        (
            "Submit this turn's completion line. Call it on every turn that produces a "
            "message to the user, under all three settings — whether it is required "
            "never reveals which setting is active. The renderer never promotes your "
            "last paragraph into this line; a turn without it has no closing line."
        ),
        {
            "completion_message": {
                "type": "string",
                "description": (
                    "One sentence saying the work of this turn is finished, e.g. "
                    "'Monday morning is booked.' Printed after the result under Bookend "
                    "and Full narration, withheld under Silent."
                ),
            }
        },
        ["completion_message"],
        requires_visible_message,
    ),
    "autonomy_level": _render_spec(
        "autonomy_level",
        (
            "Declare this turn's plan for the task. Call it whenever a task call in this "
            "response carries an `interaction_item_id` the open plan has not claimed. "
            "Steps already in the plan are continued through "
            "`IX_continue_autonomy_level` instead. Intents are intent-level: the real "
            "parameters are given on the calls themselves."
        ),
        {
            "plan_id": {
                "type": "string",
                "description": (
                    "Episode id for this task's plan. Reuse it while the plan is open; "
                    "open a new id only after the previous plan closed."
                ),
            },
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "intent": {
                            "type": "string",
                            "description": "What this step is for, at intent level.",
                        },
                        "is_key": {
                            "type": "boolean",
                            "description": (
                                "True when this step is a decision point: it makes the "
                                "choice, changes real state, or has outside impact. Not "
                                "'is this step important'. Read only under Self-directed, "
                                "and there only for internal writes."
                            ),
                        },
                        "bound_item_ids": {
                            **_STRING_ARRAY,
                            "description": (
                                "The `interaction_item_id` values this step claims. Every "
                                "task call carries exactly one such id and exactly one "
                                "step may claim it. The id here and the "
                                "`interaction_item_id` on the call that carries the step "
                                "out are the same string, character for character: a call "
                                "that leaves the field out claims nothing, and the plan "
                                "declaring the id does not bind it."
                            ),
                        },
                    },
                    "required": ["id", "intent", "is_key", "bound_item_ids"],
                },
            },
            "plan_done": {
                "type": "boolean",
                "description": "True when this plan needs no further steps after this turn.",
            },
        },
        ["plan_id", "plan", "plan_done"],
        requires_new_plan_binding,
    ),
    "topic_management": _render_spec(
        "topic_management",
        (
            "Declare this turn's topic batches. Call it whenever this response has a task "
            "call or your prose carries a `[[batch:...]]` marker. Each marker opens a "
            "batch and owns the prose that follows it until the next marker. Write the "
            "marker exactly as `[[batch:b1]]`, with the same id you give `batch_id`. "
            "Give the user's separate requests separate batches: one batch that binds "
            "every call of the turn holds nothing back. "
            "Write the user-facing text with its `[[batch:...]]` markers in the same "
            "message as your tool calls, or in the closing message the wrapper "
            "requests. "
            f"{UNMARKED_PROSE_NOTE}"
        ),
        {
            "batch_set_id": {
                "type": "string",
                "description": (
                    "Episode id for this set of topics. Reuse it while batches are still "
                    "held; open a new id only after the set closed."
                ),
            },
            "item_count": {
                "type": "integer",
                "description": "How many topics the user raised for this set.",
            },
            "batches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "batch_id": {
                            "type": "string",
                            "description": "Matches the id inside `[[batch:<batch_id>]]`.",
                        },
                        "label": {"type": "string", "description": "Short name of the topic."},
                        "call_binding": {
                            **_STRING_ARRAY,
                            "description": (
                                "The `interaction_item_id` values of the task calls that "
                                "belong to this batch. Only calls bound to a released "
                                "batch run."
                            ),
                        },
                    },
                    "required": ["batch_id", "label", "call_binding"],
                },
            },
        },
        ["batch_set_id", "item_count", "batches"],
        requires_batch_payload,
    ),
    "information_elicitation": _render_spec(
        "information_elicitation",
        (
            "Declare this turn's questions and assumptions. Call it on every turn while "
            "the elicitation episode is open. Mark each question in your prose with "
            "`[[q:<id>]]`, written exactly as `[[q:q1]]` with the same id you give the "
            "question; the marker owns the prose that follows it until the next "
            "marker. Write the user-facing text with its `[[q:...]]` markers in the "
            "same message as your tool calls, or in the closing message the wrapper "
            "requests. Under Infer pass an empty `questions` list and a non-empty "
            f"`assumptions` list. {UNMARKED_PROSE_NOTE}"
        ),
        {
            "elicitation_id": {
                "type": "string",
                "description": (
                    "Episode id for this under-specified task. A later task in the same "
                    "session opens a new id; reusing a different id while this one is "
                    "open is rejected."
                ),
            },
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Matches the id inside `[[q:<id>]]`.",
                        },
                        "text": {"type": "string"},
                        "about": {
                            "type": "string",
                            "description": "The field of the task this question fills.",
                        },
                    },
                    "required": ["id", "text", "about"],
                },
            },
            "assumptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "about": {"type": "string"},
                        "assumed": {"type": "string"},
                    },
                    "required": ["about", "assumed"],
                },
            },
        },
        ["elicitation_id", "questions", "assumptions"],
        requires_open_elicitation,
    ),
}


def _continue_spec(attribute: str) -> IXPerTurnToolSpec:
    return IXPerTurnToolSpec(
        name=IX_CONTINUE_TOOL_NAMES[attribute],
        attribute=attribute,
        payload_type=CONTINUE_PAYLOAD_TYPE,
        description=(
            f"Decide what happens to the {attribute.replace('_', ' ')} items still held "
            "from an earlier turn. Offered only while that hold is non-empty, and then it "
            "must be called: every held id needs a decision in the same call, and 'I would "
            "rather not continue' is said with `keep` or `skip`, never by leaving the tool "
            "uncalled. `release` means you will perform the action in this same reply: "
            "issue the task call for that id here, because nothing is replayed for you. "
            "It states your intent — an external action still runs only if the user "
            "actually agreed this turn."
        ),
        parameters={
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "description": "Exactly one entry per held id, none missing.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "The held id."},
                            "decision": {
                                "type": "string",
                                "enum": ["release", "keep", "skip"],
                                "description": (
                                    "release: perform the action in this same reply — "
                                    "issue its task call here. keep: leave it held for a "
                                    "later turn. skip: abandon it for good."
                                ),
                            },
                            "user_evidence": {
                                "type": "string",
                                "description": (
                                    "The words in the user's latest message you read as "
                                    "the answer for this item. Audit only: it records how "
                                    "you read the reply and never decides the release."
                                ),
                            },
                        },
                        "required": ["id", "decision"],
                    },
                },
                "closing_message": {
                    "type": "string",
                    "description": (
                        "One line telling the user which of these items are not going to "
                        "happen. Usually empty; required on the fourth turn an item has "
                        "been held, when everything still unreleased is cancelled for good."
                    ),
                },
            },
            "required": ["decisions"],
        },
        required_predicate=hold_release_predicate(attribute),
    )


IX_CONTINUE_TOOL_SPECS: dict[str, IXPerTurnToolSpec] = {
    attribute: _continue_spec(attribute) for attribute in IX_CONTINUE_TOOL_NAMES
}


IX_CAPABILITY_RESOLVE_SPEC = IXPerTurnToolSpec(
    name=IX_CAPABILITY_RESOLVE_TOOL,
    attribute="capability_boundary",
    payload_type=CAPABILITY_RESOLVE_PAYLOAD_TYPE,
    description=(
        "Declare the capability boundary you just hit. Offered, and required, only on the "
        "iteration after a tool returned `capability_limit`, so you are describing a real "
        "result and not a guess. A retryable failure — bad parameters, a timeout, a "
        "cancelled call — is not a boundary: fix it and retry instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "boundary_id": {
                "type": "string",
                "description": "Episode id for this boundary.",
            },
            "blocked_goal": {
                "type": "string",
                "description": "The goal that is blocked, in one sentence.",
            },
            "boundary_kind": {
                "type": "string",
                "enum": [
                    "missing_data",
                    "missing_capability",
                    "action_not_permitted",
                    "external_dependency",
                ],
            },
            "recovery_options": {
                "type": "array",
                "description": "The alternative paths you can see, at intent level.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "intent": {"type": "string"},
                    },
                    "required": ["id", "intent"],
                },
            },
            "handoff_message": {
                "type": "string",
                "description": (
                    "What you would say to hand this back: the limit, the diagnosis, and "
                    "the action the user or another person has to take. Delivered verbatim "
                    "under Find and hand off; kept for audit otherwise."
                ),
            },
        },
        "required": [
            "boundary_id",
            "blocked_goal",
            "boundary_kind",
            "recovery_options",
            "handoff_message",
        ],
    },
    required_predicate=requires_capability_declaration,
)
