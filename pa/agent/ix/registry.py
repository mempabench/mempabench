"""Per-turn IX tool registry wrapper.

Three jobs (IX wrapper plan §3.0, §5.3 carrier C, §7.1;
Obsidian `Plans/IX_Interaction_Tool_Development_Log.md`):

1. show the right IX tools for this turn — per-session selection tools for the
   attributes still unchosen, per-turn payload tools for every rendered attribute
   in `session_active_ix`, a continuation tool only while its hold is non-empty,
   and `IX_capability_resolve` only after a tool returned `capability_limit`;
2. inject the three per-call fields onto every task tool schema, keyed on
   **`session_active_ix`** (not this turn's `active_preferences`, which drops an
   attribute as soon as it has been chosen), and strip them again before the real
   tool is called — one change covering all task tools;
3. answer "which IX tools must have been called this turn" both unconditionally
   (`required_ix_tool_names`) and by evaluating the P0 required predicates against
   this turn's response (`required_ix_tool_names_for`).
"""

from __future__ import annotations

import inspect
from typing import Any

from pa.agent.ix.action_types import tool_action_type
from pa.agent.ix.render import ASK_BEARING_ATTRIBUTES
from pa.agent.ix.render_types import (
    HOLD_BEARING_ATTRIBUTES,
    INTERACTION_ITEM_ID_FIELD,
    IX_CONFIRMATION_MESSAGE_FIELD,
    IX_NOTICE_FIELD,
    PER_CALL_INJECTED_FIELDS,
    IXHold,
    RequiredContext,
)
from pa.agent.ix.specs import IX_PASSIVE_SPECS
from pa.agent.ix.tools import (
    IX_CAPABILITY_RESOLVE_SPEC,
    IX_CONTINUE_TOOL_SPECS,
    IX_RENDER_TOOL_SPECS,
    IXPerTurnTool,
    IXPreferenceTool,
    episode_has_held_items,
)
from pa.agent.tools.registry import ToolRegistry

_PER_CALL_FIELD_SCHEMAS: dict[str, dict[str, Any]] = {
    IX_NOTICE_FIELD: {
        "type": "string",
        "description": (
            "One short sentence, written to the user in the user's own language "
            "and register, saying what you are about to do with this call and "
            "why. It must be a sentence with a verb, not a noun fragment, not a "
            "heading or search phrase, and it must not name a tool, a function "
            "or a file path. Good: 'Let me check your calendar for a free slot "
            "on Monday morning.' Bad: 'Monday morning calendar availability.' "
            "The renderer decides whether it is shown."
        ),
    },
    INTERACTION_ITEM_ID_FIELD: {
        "type": "string",
        "description": (
            "The id that binds this call to one interaction item this turn declared. "
            "Only two payloads declare call ids: an `IX_render_autonomy_level` plan "
            "step (`bound_item_ids`) and an `IX_render_topic_management` batch "
            "(`call_binding`). Pass this id only when a plan step or a batch of this "
            "same turn declared it for this call: it must appear in exactly one such "
            "list and be unique within this turn, and a missing, unknown or duplicated "
            "id means none of this turn's task calls run. When neither payload declared "
            "an id this turn, this call has nothing to bind to, so omit the field -- an "
            "id you invent to fill it binds to nothing, and a question id is not a call "
            "binding either."
        ),
    },
    IX_CONFIRMATION_MESSAGE_FIELD: {
        "type": "string",
        "description": (
            "What you would say to ask the user before this specific call, using the "
            "real recipient, draft or amount you now have. Delivered verbatim if this "
            "call stops the turn; nothing is written for you if you leave it out."
        ),
    },
}


class TurnToolRegistry:
    """Per-turn view over benchmark task tools plus active IX tools."""

    def __init__(
        self,
        base: ToolRegistry,
        active_preferences: list[str] | tuple[str, ...] | None = None,
        ix_context: dict[str, Any] | None = None,
    ) -> None:
        self._base = base
        context = ix_context or {}
        proactive_event = bool(context.get("proactive_event"))
        session_active_ix: tuple[str, ...] = tuple(context.get("session_active_ix") or ())
        hold = context.get("hold")
        self._hold: IXHold = hold if isinstance(hold, IXHold) else IXHold()
        self._session_active_ix = session_active_ix
        self._injected_fields = self._fields_to_inject(session_active_ix)

        self._ix_tools: dict[str, IXPreferenceTool | IXPerTurnTool] = {}
        for attr in active_preferences or []:
            spec = IX_PASSIVE_SPECS.get(attr)
            if spec is not None:
                self._ix_tools[spec.name] = IXPreferenceTool(
                    spec,
                    proactive_event=proactive_event,
                )
        for attr in session_active_ix:
            render_spec = IX_RENDER_TOOL_SPECS.get(attr)
            if render_spec is not None:
                self._ix_tools[render_spec.name] = IXPerTurnTool(render_spec)
            continue_spec = IX_CONTINUE_TOOL_SPECS.get(attr)
            if continue_spec is not None and self._has_hold(attr):
                self._ix_tools[continue_spec.name] = IXPerTurnTool(continue_spec)
        if context.get("capability_limit_seen"):
            self._ix_tools[IX_CAPABILITY_RESOLVE_SPEC.name] = IXPerTurnTool(
                IX_CAPABILITY_RESOLVE_SPEC
            )

    # -- offering ----------------------------------------------------------

    def _has_hold(self, attribute: str) -> bool:
        episode = self._hold.open_episodes.get(attribute)
        return episode is not None and not episode.closed and episode_has_held_items(episode)

    @staticmethod
    def _fields_to_inject(session_active_ix: tuple[str, ...]) -> tuple[str, ...]:
        """§5.3 carrier C — which per-call fields this session's attributes need."""
        active = set(session_active_ix)
        fields: list[str] = []
        if "process_visibility" in active:
            fields.append(IX_NOTICE_FIELD)
        if active & HOLD_BEARING_ATTRIBUTES:
            fields.append(INTERACTION_ITEM_ID_FIELD)
        if active & ASK_BEARING_ATTRIBUTES:
            fields.append(IX_CONFIRMATION_MESSAGE_FIELD)
        return tuple(fields)

    @property
    def injected_per_call_fields(self) -> tuple[str, ...]:
        return self._injected_fields

    @property
    def tool_names(self) -> list[str]:
        return [*self._base.tool_names, *self._ix_tools]

    @staticmethod
    def _ix_parameters(tool: IXPreferenceTool | IXPerTurnTool) -> dict[str, Any]:
        """Return the existing IX dispatch schema without mutating its shared spec."""
        return dict(tool.parameters or {})

    @classmethod
    def _rejects_unknown_ix_parameters(
        cls, tool: IXPreferenceTool | IXPerTurnTool
    ) -> bool:
        """Whether this IX implementation has a closed top-level argument set."""
        parameters = cls._ix_parameters(tool)
        additional = parameters.get("additionalProperties")
        if additional is False:
            return True
        if additional is not None:
            return False
        return not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(tool.execute).parameters.values()
        )

    def get_definitions(self) -> list[dict[str, Any]]:
        return [
            *(self._inject(definition) for definition in self._base.get_definitions()),
            *(tool.to_schema() for tool in self._ix_tools.values()),
        ]

    def _inject(self, definition: dict[str, Any]) -> dict[str, Any]:
        """Add the per-call fields to one task tool schema (§5.3 carrier C)."""
        if not self._injected_fields:
            return definition
        function = definition.get("function") or {}
        parameters = dict(function.get("parameters") or {})
        properties = dict(parameters.get("properties") or {})
        required = list(parameters.get("required") or [])
        # A read never has to carry an ask: §3.3 (part 2)·C requires it only for
        # draft / internal_write / external_action calls.
        is_read = tool_action_type(str(function.get("name") or "")) == "read"
        for field in self._injected_fields:
            properties[field] = _PER_CALL_FIELD_SCHEMAS[field]
            if field == IX_CONFIRMATION_MESSAGE_FIELD and is_read:
                continue
            # Offered, never demanded: only an `IX_render_autonomy_level` plan
            # step and an `IX_render_topic_management` batch declare call ids,
            # so on a turn governed by `information_elicitation` or
            # `capability_boundary` there is nothing to bind to and a required
            # field can only be filled by inventing one — live `acc_009`, where
            # the invented id was then refused as unknown. The description says
            # to omit it; `required` said the opposite.
            if field == INTERACTION_ITEM_ID_FIELD:
                continue
            if field not in required:
                required.append(field)
        parameters["properties"] = properties
        parameters["required"] = required
        return {
            **definition,
            "function": {**function, "parameters": parameters},
        }

    # -- required judgement ------------------------------------------------

    @property
    def required_ix_tool_names(self) -> list[str]:
        """The IX tools required no matter what this turn's response contains.

        That is the `required == "always"` tier — the per-session selection tools.
        Conditionally-required tools are answered by `required_ix_tool_names_for`
        once the response is in hand (§7.1: "response-evaluated
        conditionally-required"); asking for them here would force a call the
        model has nothing to say with.
        """
        return [name for name, tool in self._ix_tools.items() if tool.spec.required == "always"]

    def required_ix_tool_names_for(self, context: RequiredContext) -> list[str]:
        """Evaluate every offered IX tool's required rule against this turn's response."""
        return [name for name, tool in self._ix_tools.items() if tool.is_required(context)]

    # -- execution ---------------------------------------------------------

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        if name in self._ix_tools:
            tool = self._ix_tools[name]
            params = tool.cast_params(params)
            errors = tool.validate_params(params)
            parameters = self._ix_parameters(tool)
            if isinstance(params, dict) and self._rejects_unknown_ix_parameters(tool):
                properties = parameters.get("properties") or {}
                errors.extend(
                    f"unknown parameter: {key}"
                    for key in sorted(set(params) - set(properties))
                )
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            return await tool.execute(**params)
        # Carrier C: the injected fields are wrapper state, never tool arguments.
        # Stripped unconditionally, so a model that writes one on a turn where it
        # was not injected still cannot reach the real tool with it.
        forwarded = {key: value for key, value in params.items() if key not in PER_CALL_INJECTED_FIELDS}
        return await self._base.execute(name, forwarded)
