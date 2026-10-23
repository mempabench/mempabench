"""Policy mapping from IX selections to runner directives."""

from __future__ import annotations

from pa.agent.ix.events import (
    IX_SELECTION_PAYLOAD_TYPE,
    IXDirective,
    IXSelection,
)


def selections_from_tool_events(tool_events: list[dict]) -> list[IXSelection]:
    """Extract IX selections from runner tool events.

    Only the per-session selection tools carry a setting. The per-turn payload
    tools share the `IX_` prefix and echo the same attribute, so without this
    filter a later `IX_render_<attribute>` result would overwrite the chosen
    setting with an empty one and silently switch every gate off.
    """
    selections: list[IXSelection] = []
    for event in tool_events:
        if not event.get("ix"):
            continue
        if str(event.get("payload_type") or "") != IX_SELECTION_PAYLOAD_TYPE:
            continue
        selections.append(IXSelection(
            attribute=str(event.get("attribute") or ""),
            tool_name=str(event.get("name") or ""),
            setting=str(event.get("setting") or ""),
            evidence=str(event.get("evidence") or ""),
            application=str(event.get("application") or ""),
        ))
    return selections


def directive_for(selections: list[IXSelection]) -> IXDirective:
    """Return the current turn directive implied by IX selections."""
    values: dict = {}
    _apply_process_visibility(values, selections)
    _apply_topic_management(values, selections)
    _apply_information_elicitation(values, selections)
    _apply_autonomy_level(values, selections)
    _apply_solution_breadth(values, selections)
    _apply_proactive_outreach(values, selections)
    _apply_task_expansion(values, selections)
    _apply_capability_boundary(values, selections)
    return IXDirective(**values)


# ---------------------------------------------------------------------------
# Per-attribute apply functions
# ---------------------------------------------------------------------------

def _latest_selection(
    selections: list[IXSelection], attribute: str
) -> IXSelection | None:
    """Latest selection for the attribute, or `None` when the session has none.

    `None` is the unset fallback: every `_apply_*` below returns without writing
    a directive, so the attribute has no policy at all this turn. It is only
    reachable for an attribute this session never exposed — an exposed one's
    `IX_<attribute>` tool is `required == "always"`.
    """
    return next((s for s in reversed(selections) if s.attribute == attribute), None)


def _apply_process_visibility(values: dict, selections: list[IXSelection]) -> None:
    sel = _latest_selection(selections, "process_visibility")
    if sel is None:
        return
    setting = sel.setting.casefold()
    if setting == "bookend":
        values["process_visibility"] = "bookend"
        values["emit_start"] = True
        values["emit_completion"] = True
    elif setting == "full narration":
        values["process_visibility"] = "full_narration"
        values["emit_start"] = True
        values["emit_progress"] = True
        values["emit_completion"] = True


def _apply_topic_management(values: dict, selections: list[IXSelection]) -> None:
    sel = _latest_selection(selections, "topic_management")
    if sel is None:
        return
    values["topic_management_policy"] = _topic_management_policy(sel.setting)
    values["topic_management"] = sel.setting


def _apply_information_elicitation(values: dict, selections: list[IXSelection]) -> None:
    sel = _latest_selection(selections, "information_elicitation")
    if sel is None:
        return
    mode, policy = _information_elicitation_policy(sel.setting)
    values["information_elicitation"] = mode
    values["elicitation_policy"] = policy


def _apply_autonomy_level(values: dict, selections: list[IXSelection]) -> None:
    sel = _latest_selection(selections, "autonomy_level")
    if sel is None:
        return
    values["autonomy_level_policy"] = _autonomy_level_policy(sel.setting)
    values["autonomy_level"] = sel.setting


def _apply_solution_breadth(values: dict, selections: list[IXSelection]) -> None:
    sel = _latest_selection(selections, "solution_breadth")
    if sel is None:
        return
    values["solution_breadth_policy"] = _solution_breadth_policy(sel.setting)
    values["solution_breadth"] = sel.setting


def _apply_proactive_outreach(values: dict, selections: list[IXSelection]) -> None:
    sel = _latest_selection(selections, "proactive_outreach")
    if sel is None:
        return
    values["proactive_outreach_policy"] = _proactive_outreach_policy(sel.setting)
    values["proactive_outreach"] = sel.setting


def _apply_task_expansion(values: dict, selections: list[IXSelection]) -> None:
    sel = _latest_selection(selections, "task_expansion")
    if sel is None:
        return
    values["task_expansion_policy"] = _task_expansion_policy(sel.setting)
    values["task_expansion"] = sel.setting


def _apply_capability_boundary(values: dict, selections: list[IXSelection]) -> None:
    sel = _latest_selection(selections, "capability_boundary")
    if sel is None:
        return
    values["capability_boundary_policy"] = _capability_boundary_policy(sel.setting)
    values["capability_boundary"] = sel.setting


# ---------------------------------------------------------------------------
# Per-attribute policy helpers
# ---------------------------------------------------------------------------

def _topic_management_policy(setting: str) -> str:
    normalized = setting.casefold()
    if normalized == "organize":
        return "structured_multi_topic"
    if normalized == "one-at-a-time":
        return "single_topic_per_turn"
    return ""


def _information_elicitation_policy(setting: str) -> tuple[str, str]:
    normalized = setting.casefold()
    if normalized == "structured":
        return "structured", "structured_upfront"
    if normalized == "iterative":
        return "iterative", "iterative_incremental"
    return "infer", "infer_proceed"


def _autonomy_level_policy(setting: str) -> str:
    normalized = setting.casefold()
    if normalized == "reactive":
        return "confirm_substantive_actions"
    if normalized == "self-directed":
        return "confirm_key_actions"
    if normalized == "autonomous":
        return "execute_delegated_task"
    return ""


def _solution_breadth_policy(setting: str) -> str:
    # Two settings only: `Medium` merged into `High`, so the old
    # `medium -> shortlist` branch was unreachable and its code was referenced
    # nowhere else in the tree.
    normalized = setting.casefold()
    if normalized == "low":
        return "one_best_answer"
    if normalized == "high":
        return "broad_alternatives"
    return ""


def _proactive_outreach_policy(setting: str) -> str:
    # Two settings only: `Medium` was removed, and `ProactiveOutreachSetting`
    # freezes the domain to `Low` / `High`, so the old
    # `medium -> material_trigger_only` branch was unreachable and its code was
    # referenced nowhere else in the tree.
    normalized = setting.casefold()
    if normalized == "low":
        return "wait_for_user"
    if normalized == "high":
        return "relevant_trigger_contact"
    return ""


def _task_expansion_policy(setting: str) -> str:
    # Two settings only: `Medium` was cut, so the
    # split is executed vs not executed — zero adjacent steps carried out
    # against at least one. The old `medium -> necessary_adjacent_steps` branch
    # was referenced nowhere else in the tree.
    normalized = setting.casefold()
    if normalized == "low":
        return "explicit_scope_only"
    if normalized == "high":
        return "adjacent_support_tasks"
    return ""


def _capability_boundary_policy(setting: str) -> str:
    normalized = setting.casefold()
    if normalized == "suggest alternatives":
        return "suggest_in_pa_workarounds"
    if normalized == "find and hand off":
        return "find_handoff_path"
    return ""
