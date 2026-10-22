"""Structured IX policy objects and event helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


#: The `type` an IX **selection** tool result declares. The per-turn payload
#: tools declare their own types, and only this one carries a chosen setting.
IX_SELECTION_PAYLOAD_TYPE = "ix_preference_selection"


@dataclass(frozen=True)
class IXSelection:
    attribute: str
    tool_name: str
    setting: str
    evidence: str = ""
    application: str = ""
    forced: bool = False
    """Seeded by `--force-ix`, not chosen by the model.

    A forced entry is never replaced by a selection the model made later in the
    same benchmark session. On an ordinary turn that costs nothing — the
    attribute's `IX_` tool is already withdrawn once it is in the cache. It
    matters on the environment-event turn, which deliberately inherits no
    settings (`AgentLoop._run_agent_loop`) and therefore still offers
    `IX_proactive_outreach`: that tool is the only carrier of the outreach
    message, so it has to stay on the turn, and without this flag the model's
    own setting would overwrite the seed before the renderer reads it.
    """


@dataclass(frozen=True)
class IXDirective:
    """Policy codes for the current turn.

    The per-setting instruction prose that used to live here was the only
    runtime effect of the pass-through attributes, and it was injected only on
    turns that already had tool calls. Both are gone: the wording now has a
    single home in ``specs.py`` and reaches the model through the per-turn
    restatement (plan §3.6). What stays here are the **policy codes** the gates
    read.
    """

    process_visibility: str = "silent"
    emit_start: bool = False
    emit_progress: bool = False
    emit_completion: bool = False
    autonomy_level: str = ""
    autonomy_level_policy: str = ""
    information_elicitation: str = "infer"
    elicitation_policy: str = ""
    topic_management: str = ""
    topic_management_policy: str = ""
    solution_breadth: str = ""
    solution_breadth_policy: str = ""
    proactive_outreach: str = ""
    proactive_outreach_policy: str = ""
    task_expansion: str = ""
    task_expansion_policy: str = ""
    capability_boundary: str = ""
    capability_boundary_policy: str = ""

    @property
    def has_control_directive(self) -> bool:
        return any([
            self.emit_start,
            self.emit_progress,
            self.emit_completion,
            self.autonomy_level_policy,
            self.information_elicitation not in ("", "infer"),
            self.topic_management,
            self.solution_breadth,
            self.proactive_outreach,
            self.task_expansion,
            self.capability_boundary,
        ])


def extract_ix_payload(tool_name: str, result: Any) -> dict[str, str] | None:
    """Parse an IX tool result into its structured payload fields.

    Returns None for non-IX tools or unparseable results.
    """
    if not tool_name.startswith("IX_"):
        return None
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return None
    if isinstance(payload, dict):
        extracted = {
            # Every IX tool result declares what kind of payload it is. Without
            # it a per-turn payload result is indistinguishable from a setting
            # choice, and reads back as a selection with an empty setting.
            "payload_type": str(payload.get("type", "")),
            "attribute": str(payload.get("attribute", "")),
            "setting": str(payload.get("setting", "")),
            "evidence": str(payload.get("evidence", "")),
            "application": str(payload.get("application", "")),
        }
        for field in ("outreach_action", "outreach_message"):
            if field in payload:
                extracted[field] = str(payload[field])
        return extracted
    return None
