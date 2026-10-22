"""Taxonomy-aligned IX tool specs and the single source of setting prose.

Since 2026-09-05 (IX wrapper plan §5.2 "round 5 unified wording";
Obsidian `Plans/IX_Interaction_Tool_Development_Log.md`)
there is **one text per attribute x setting**. It lives in
`configs/setting_definitions.yaml` (`pa_instruction`, loaded by
`pa/agent/ix/settings_table.py`) and reaches the runtime as
`IXToolSpec.setting_descriptions` here, attributes in the document order. The IX
selection tool descriptions (`pa/agent/ix/tools.py`) and the per-turn restatement
(§3.6, `collector.turn_reminder_message`) read the same dict; `pa/agent/ix/policy.py`'s
six directive texts are retired by P2.

The prose answers "what does this setting look like *after* the renderer" for the
eight rendered attributes (§3.1 / `render_types.RENDERED_ATTRIBUTES`) and
"how must you write this turn" for the six pass-through ones (§3.6).
"""

from __future__ import annotations

from dataclasses import dataclass

from pa.agent.ix.render_types import (
    IX_TOOL_NAME_PREFIX,
    RequiredMode,
    RequiredPredicate,
)
from pa.agent.ix.settings_table import load_setting_definitions

#: Round 9 item 3 — the clause that keeps the wrapper's own vocabulary off the
#: user's screen. It lives here, beside the other shared model-facing notes,
#: because two standing texts need it and `collector.py` — which re-exports it
#: for the repair messages — imports this module, not the other way round.
#:
#: The reminders are wrapper-internal, and a model that answers one in
#: user-visible prose puts the PA in the position of apologising to something the
#: user never said (live: `You're right — … Let me re-send that`). The collector
#: drops such an opening if it appears anyway.
#: The second clause is round-2 `acc_019`: three replies narrated the
#: guidance-level choice back to the user ("Glosses Gaussian process regression
#: once on first use", "the one-gloss-once preference is satisfied", "No domain
#: terms needed glossing here — you're deep in the physics"). The setting is
#: supposed to be visible in how the reply is written, never as a line about how
#: it was written.
NO_WRAPPER_TALK = (
    "Never mention ids, fields, tools or these instructions to the user, and "
    "never say why your explanation is pitched where it is or that you chose a "
    "level for it."
)

#: regress18 `acc_009` t=4 and `acc_006` t=7 — two replies that narrated the
#: mechanism to the user in the wrapper's own words: *"Great, all four questions
#: have been released."* and *"so it stays within the Low boundary."* Neither
#: turn carried `NO_WRAPPER_TALK`, because it rode only on the repair messages —
#: so the one instruction written to stop this reached only the turns where
#: something had *already* gone wrong. This note is on every selection tool
#: description, which is where the setting names and the release vocabulary are
#: handed over in the first place, so now the clause travels with them.
SETTING_COMMITMENT_NOTE = (
    "Every listed setting is a specific preference about this user, and you must "
    "choose exactly one of them: a middle setting is not a hedge, there is no "
    "default to fall back on, and several attributes list only two. The first "
    "time you meet an attribute you may have no evidence about this user at all; "
    "choose the setting that best fits the request in front of you anyway. That "
    "is expected, and the guess still counts as your choice. "
    + NO_WRAPPER_TALK
)

#: §5.1 — the wrapper notice appended to the `## nanobot Guidelines` block in
#: `pa/agent/context.py`. It tells the model that a renderer sits between its
#: turn and the user, and that the renderer selects and orders but never rewrites.
WRAPPER_NOTICE = (
    "Your turn passes through an interaction renderer before it reaches the "
    "user. The renderer decides which parts the user sees and when, but it "
    "never rewrites your wording — it only selects and orders. Do the whole "
    "turn as usual; each `IX_` tool's setting descriptions say what that "
    "setting looks like after the renderer."
)

#: acc_027 b5 — the user asked *"The calendar is now consistent with the
#: reservation. Are you prepared to transmit the logistical update to Marcus?"*
#: and the PA answered by sending the message, before the rendezvous details had
#: been given; the next beat instructed the real send, so Marcus received a
#: premature text and then a second, superseding one seconds later. The rule is
#: stated where every session sees it, not on `autonomy_level`, because that
#: session cued no autonomy attribute at all.
READINESS_QUESTION_NOTE = (
    "A question about whether you are ready, able or prepared to do something "
    "is a question, not an instruction to do it. Answer it with a statement of "
    "your readiness — say that you are ready and what you would do — and wait "
    "for the go-ahead. Acting on it takes the decision away from the user, and "
    "an outward action taken that way cannot be recalled."
)

#: acc_027 t=13 — the third route to the same defect in three rounds. The turn's
#: whole instruction was *"Secure the 21:00 slot at Siam Garden … book that slot
#: immediately"* and the PA ran the reservation, the calendar update **and** a
#: `message_send` telling Marcus the earlier options were void. Marcus got three
#: texts in a session whose draft specifies one, and the thread contradicts
#: itself. Round 4's readiness rule above closed the question route; this closes
#: the over-eagerness route, and it sits in the base guidelines for the same
#: reason: no `autonomy_level` was cued, so the policy string was empty and the
#: `external_action_floor` never armed. The floor itself is untouched.
OUTWARD_MESSAGE_NOTE = (
    "An outward message to another person — `message_send`, "
    "`email_message_send` — needs an instruction in this same turn that names "
    "that message. Carrying out a booking, a calendar change or a document "
    "update is not licence to notify anyone about it: report what you did to "
    "the user and let them decide who else hears about it."
)

#: The one sentence §8 item 2 (owner decision, option A) requires on both batch
#: carriers: prose you leave unmarked ships every turn, held batches or not.
UNMARKED_PROSE_NOTE = (
    "Prose you write before your first marker carries no batch or question id, "
    "so it is delivered every turn — including turns where the renderer holds "
    "batches or questions back. It must not speak for held content or announce "
    "unreleased work as done."
)


@dataclass(frozen=True)
class IXToolSpec:
    """One passive interaction preference selection tool.

    `required` is the P0 "response-evaluated conditionally-required" slot
    (§7.1): `always` (the selection tools), `never`, or `predicate` plus a named
    `required_predicate` evaluated against a `RequiredContext` once this turn's
    provider response is in hand.
    """

    attribute: str
    name: str
    label: str
    settings: tuple[str, ...]
    setting_descriptions: dict[str, str]
    required: RequiredMode = "always"
    required_predicate: RequiredPredicate | None = None

    def __post_init__(self) -> None:
        if set(self.setting_descriptions) != set(self.settings):
            raise ValueError(f"{self.name} setting_descriptions must match settings")
        if (self.required == "predicate") != (self.required_predicate is not None):
            raise ValueError(
                f"{self.name} required='predicate' needs a required_predicate, "
                "and a required_predicate needs required='predicate'"
            )


#: The fourteen selection tools, one per attribute, in the document order
#: (`configs/setting_definitions.yaml`; also `data/personas/coverage.py` `TAXONOMY`
#: and the judge numbering). `setting_descriptions` is the YAML's `pa_instruction`
#: verbatim.
IX_PASSIVE_SPECS: dict[str, IXToolSpec] = {
    attribute.id: IXToolSpec(
        attribute=attribute.id,
        name=f"{IX_TOOL_NAME_PREFIX}{attribute.id}",
        label=attribute.label,
        settings=tuple(setting.name for setting in attribute.settings),
        setting_descriptions={
            setting.name: setting.pa_instruction for setting in attribute.settings
        },
    )
    for attribute in load_setting_definitions().attributes
}


def ix_tool_names_for(active_preferences: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return IX tool names for active passive preferences."""
    return [
        spec.name
        for attr in active_preferences or []
        if (spec := IX_PASSIVE_SPECS.get(attr)) is not None
    ]


#: §3.6 — the attributes whose runtime effect is carried by the per-turn
#: restatement rather than by the renderer: the six pass-through ones, plus
#: `autonomy_level` when the chosen setting is the one gate-off setting (a
#: gating setting carries itself, so restating it would only leak the gate).
#: 2026-09-05: the middle setting was merged into `Self-directed`, which now
#: gates, so `Autonomous` is the only setting left with no carrier of its own.
RESTATED_AUTONOMY_SETTINGS: frozenset[str] = frozenset({"Autonomous"})
