"""The actor's private record of what the assistant's latest reply actually did.

Owner ruling 2026-09-17 (`project_rules_setting_semantics`): a reaction beat's
`if_…` branch is chosen by the simulated user from the PA's visible reply plus
exactly four facts — which state-changing tools ran (catalog name and the
catalog's own line, never their arguments), how many task calls the turn made,
how many progress lines and whether a closing line were delivered, and whether
bracketed confidence tags appeared. The actor is **never** told whether a batch
or question was held, what was held back, a gate's parameters, or a setting
name — a hold has to be inferred from the reply, the way a real user would.

Two sources, one record. The writes and the call count come off the turn's
`tool_events`, which the simulator already reads for the judge; the three
wrapper-side facts come off the harness key `reply_facts`, which
`harness/pa_factory.py::reply_facts_from_ix_render` writes so that this
package never reads `ix_render` itself (`simulator/` does not import `pa/`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from simulator.judge.catalogue import OUTREACH_ACT, executed_act_label

#: The harness key on a PA turn's metadata, and its exact shape.
REPLY_FACTS_KEY = "reply_facts"
_REPLY_FACTS_FIELDS: dict[str, type] = {
    "notices_released": int,
    "completion_released": bool,
    "confidence_brackets": bool,
}


@dataclass(frozen=True)
class ReplyFacts:
    """What one delivered PA reply did, as far as the actor may know it."""

    #: State changes carried out in that reply, in call order, one line per
    #: call — a tool that ran twice is listed twice, because "a second change
    #: beyond the one asked for" is exactly what a task-expansion reaction
    #: turns on. Refused and errored calls changed nothing and are not here.
    executed_acts: tuple[str, ...]
    #: State changes carried out on earlier replies this session, same shape.
    earlier_acts: tuple[str, ...]
    #: Task calls the reply made — every call on the world, blocked and failed
    #: ones included, IX selections and wrapper control records excluded. A
    #: call that came back blocked is still a call the assistant made, and on
    #: a capability-limit turn it is the one the reaction pivots on.
    task_calls: int
    #: The wrapper-side three; `None` when the harness wrote no `reply_facts`
    #: key for this turn (a turn the wrapper did not render).
    notices_released: int | None = None
    completion_released: bool | None = None
    confidence_brackets: bool | None = None


def normalize_reply_facts(raw: Any) -> dict[str, Any] | None:
    """Read the harness's `reply_facts` key off a PA turn.

    Absent is the normal case for a turn the wrapper did not render, and for
    the prompt-only path. Anything present has to be exactly the three typed
    fields: a malformed value is a wiring bug between harness and simulator,
    and reading it as "nothing delivered" would look exactly like a Silent
    turn — the same reason `_normalize_held_items` fails instead of defaulting.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != set(_REPLY_FACTS_FIELDS):
        raise ValueError(
            f"{REPLY_FACTS_KEY} must carry exactly {sorted(_REPLY_FACTS_FIELDS)}: "
            f"value={raw!r}"
        )
    for name, kind in _REPLY_FACTS_FIELDS.items():
        value = raw[name]
        # `bool` is an `int`; an int where a bool belongs is still wrong.
        if type(value) is not kind:
            raise ValueError(
                f"{REPLY_FACTS_KEY}.{name} must be {kind.__name__}: value={value!r}"
            )
    return {name: raw[name] for name in _REPLY_FACTS_FIELDS}


def _act_lines(names: Iterable[str]) -> tuple[str, ...]:
    lines: list[str] = []
    for name in names:
        label = executed_act_label(str(name or ""))
        if label:
            lines.append(label)
    return tuple(lines)


def reply_facts_from_pa_turn(
    tool_events: Iterable[Mapping[str, Any]],
    raw_reply_facts: Any,
    *,
    earlier_tools: Iterable[str],
    earlier_outreach_sent: bool,
    outreach_delivered: bool = False,
) -> ReplyFacts:
    """Build the record for one PA turn.

    `earlier_tools` is the session's executed-tool list as it stood BEFORE
    this turn (`SimulatorLoop._pa_executed_tools`, already minus errored
    calls); `earlier_outreach_sent` whether a triggered outreach had already
    been delivered before this turn, and `outreach_delivered` whether this
    turn is the one that delivered it.
    """
    events = list(tool_events)
    task = [e for e in events if not e.get("ix") and not e.get("ix_event")]
    executed = _act_lines(
        str(e.get("name") or "") for e in task if not e.get("error")
    )
    if outreach_delivered:
        executed = (OUTREACH_ACT, *executed)
    earlier = _act_lines(earlier_tools)
    if earlier_outreach_sent:
        earlier = (OUTREACH_ACT, *earlier)
    wrapper = normalize_reply_facts(raw_reply_facts) or {}
    return ReplyFacts(
        executed_acts=executed,
        earlier_acts=earlier,
        task_calls=len(task),
        notices_released=wrapper.get("notices_released"),
        completion_released=wrapper.get("completion_released"),
        confidence_brackets=wrapper.get("confidence_brackets"),
    )


def render_reply_facts_block(facts: ReplyFacts | None) -> str:
    """Stage direction naming what the latest reply did. Never spoken.

    Rendered on the turn after a PA reply, next to the held-items block. The
    reply's own wording is still what the user answers; this record is what
    decides a conditional branch keyed on the assistant's actions, and it is
    the reason a reaction can say "you changed nothing" without inventing it.
    """
    if facts is None:
        return ""
    acts = "; ".join(facts.executed_acts) or "none — it changed nothing"
    earlier = "; ".join(facts.earlier_acts) or "none"
    lines = [
        "Private record of what the assistant's latest reply actually did (from "
        "the run log, not from its wording — stage direction, never quoted, "
        "never spoken; the reply itself is still what you answer):",
        f"- Changes it carried out in that reply: {acts}",
        f"- Changes it had already carried out earlier in this conversation: {earlier}",
        "- Tool calls it made for the task in that reply (counting a call that "
        f"came back blocked or failed): {facts.task_calls}",
    ]
    if facts.notices_released is not None:
        lines.append(
            f"- Progress lines it delivered about its own steps: {facts.notices_released}"
        )
    if facts.completion_released is not None:
        lines.append(
            "- A closing line saying the work was done: "
            + ("delivered" if facts.completion_released else "not delivered")
        )
    if facts.confidence_brackets is not None:
        lines.append(
            "- Bracketed confidence tags beside its claims: "
            + ("present" if facts.confidence_brackets else "none")
        )
    lines.append(
        "A change the reply describes but this record does not list did not "
        "happen. When a conditional branch on this beat turns on what the "
        "assistant did, choose it from this record together with the reply's "
        "visible text."
    )
    return "\n".join(lines)
