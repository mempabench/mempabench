"""P1 · which IX tools a turn shows, which it forces, and what it injects.

IX wrapper plan §3.0 (the required/optional tiers and
"per-call injection is keyed on `session_active_ix`"), §5.3 carrier C (inject then
strip), and §7.1's response-evaluated conditionally-required slot.
"""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from pa.agent.ix.registry import TurnToolRegistry
from pa.agent.ix.render_types import (
    INTERACTION_ITEM_ID_FIELD,
    IX_CONFIRMATION_MESSAGE_FIELD,
    IX_NOTICE_FIELD,
    IX_RENDER_TOOL_NAMES,
    PER_CALL_INJECTED_FIELDS,
    Batch,
    HoldEpisode,
    IXHold,
    PlanItem,
    Question,
    RequiredContext,
)
from pa.agent.ix.collector import NO_WRAPPER_TALK, turn_reminder_message
from pa.agent.ix.events import IXSelection
from pa.agent.ix.rationale import CERTAINTY_BLOCK_INSTRUCTION
from pa.agent.ix.specs import IX_PASSIVE_SPECS
from pa.agent.ix.tools import (
    IX_CAPABILITY_RESOLVE_SPEC,
    IX_CONTINUE_TOOL_SPECS,
    IX_RENDER_TOOL_SPECS,
    IXPerTurnTool,
)
from pa.agent.tools.base import Tool
from pa.agent.tools.registry import ToolRegistry


class _EmailSend(Tool):
    name = "email_message_send"
    description = "Send an email."
    parameters = {
        "type": "object",
        "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
        "required": ["to"],
    }

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return "sent"


class _DocumentRead(Tool):
    name = "document_read"
    description = "Read a document."
    parameters = {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": []}

    async def execute(self, **kwargs):
        return "read"


def _base() -> tuple[ToolRegistry, _EmailSend]:
    registry = ToolRegistry()
    send = _EmailSend()
    registry.register(send)
    registry.register(_DocumentRead())
    return registry, send


def _definition(registry: TurnToolRegistry, name: str) -> dict:
    return next(d for d in registry.get_definitions() if d["function"]["name"] == name)


def _hold(attribute: str, **held) -> IXHold:
    return IXHold(
        open_episodes={
            attribute: HoldEpisode(episode_id="e1", attribute=attribute, rounds_open=1, **held)
        }
    )


# --- offering ---------------------------------------------------------------


def test_render_tools_are_offered_for_every_session_active_attribute() -> None:
    """§3.0 — keyed on `session_active_ix`, so they survive the turn the setting was chosen."""
    base, _ = _base()
    turn = TurnToolRegistry(
        base,
        active_preferences=[],  # already selected on an earlier turn
        ix_context={"session_active_ix": ["topic_management", "reasoning_visibility"]},
    )
    names = turn.tool_names
    assert IX_RENDER_TOOL_NAMES["topic_management"] in names
    assert IX_RENDER_TOOL_NAMES["reasoning_visibility"] in names
    assert "IX_topic_management" not in names  # the selection tool is done for this session


def test_continue_tool_appears_only_while_the_hold_is_non_empty() -> None:
    base, _ = _base()
    context = {"session_active_ix": ["topic_management"]}

    empty = TurnToolRegistry(base, [], ix_context=context)
    assert "IX_continue_topic_management" not in empty.tool_names

    holding = TurnToolRegistry(
        base,
        [],
        ix_context={**context, "hold": _hold("topic_management", batches_held=[Batch("b2", "Hotel", "body")])},
    )
    assert "IX_continue_topic_management" in holding.tool_names


def test_capability_resolve_appears_only_after_a_capability_limit() -> None:
    base, _ = _base()
    context = {"session_active_ix": ["capability_boundary"]}
    assert IX_CAPABILITY_RESOLVE_SPEC.name not in TurnToolRegistry(base, [], ix_context=context).tool_names
    seen = TurnToolRegistry(base, [], ix_context={**context, "capability_limit_seen": True})
    assert IX_CAPABILITY_RESOLVE_SPEC.name in seen.tool_names


# --- required tiers ---------------------------------------------------------


def test_selection_tools_are_the_always_required_tier() -> None:
    base, _ = _base()
    turn = TurnToolRegistry(
        base,
        ["topic_management"],
        ix_context={"session_active_ix": ["topic_management", "reasoning_visibility"]},
    )
    # The unconditional list holds the selection tool only: forcing a per-turn tool
    # here would make the model call it on turns it has nothing to say with.
    assert turn.required_ix_tool_names == ["IX_topic_management"]


def test_continue_tool_is_offered_but_only_required_while_holding() -> None:
    """§7.1 — "offered but conditionally required": shown with a hold, forced with a hold."""
    base, _ = _base()
    hold = _hold("topic_management", batches_held=[Batch("b2", "Hotel", "body")])
    turn = TurnToolRegistry(base, [], ix_context={"session_active_ix": ["topic_management"], "hold": hold})

    assert "IX_continue_topic_management" not in turn.required_ix_tool_names
    required = turn.required_ix_tool_names_for(RequiredContext(hold=hold))
    assert "IX_continue_topic_management" in required

    # once the hold is empty the tool is neither shown nor forced
    empty = TurnToolRegistry(base, [], ix_context={"session_active_ix": ["topic_management"]})
    assert "IX_continue_topic_management" not in empty.tool_names
    assert "IX_continue_topic_management" not in empty.required_ix_tool_names_for(RequiredContext())


@pytest.mark.parametrize(
    "attribute,context,expected",
    [
        ("reasoning_visibility", RequiredContext(produces_user_visible_message=True), True),
        ("reasoning_visibility", RequiredContext(produces_user_visible_message=False), False),
        ("process_visibility", RequiredContext(produces_user_visible_message=True), True),
        ("uncertainty_expression", RequiredContext(produces_user_visible_message=True), True),
        ("topic_management", RequiredContext(response_tool_names=("email_message_send",)), True),
        ("topic_management", RequiredContext(response_markers=("batch:b1",)), True),
        ("topic_management", RequiredContext(response_tool_names=("IX_topic_management",)), False),
        (
            "autonomy_level",
            RequiredContext(response_interaction_item_ids=("i9",)),
            True,
        ),
        (
            "autonomy_level",
            RequiredContext(
                response_interaction_item_ids=("i3",),
                hold=IXHold(
                    open_episodes={
                        "autonomy_level": HoldEpisode(
                            episode_id="p1",
                            attribute="autonomy_level",
                            plan_remaining=[PlanItem("a3", "send", True, ["i3"])],
                        )
                    }
                ),
            ),
            False,
        ),
        ("information_elicitation", RequiredContext(), False),
        (
            "information_elicitation",
            RequiredContext(
                hold=IXHold(
                    open_episodes={
                        "information_elicitation": HoldEpisode(
                            episode_id="el1",
                            attribute="information_elicitation",
                            questions_held=[Question("q2", "seat?", "seat")],
                        )
                    }
                )
            ),
            True,
        ),
    ],
)
def test_render_tool_required_predicates(attribute: str, context: RequiredContext, expected: bool) -> None:
    """§3.0's required/optional table, one case per row."""
    tool = IXPerTurnTool(IX_RENDER_TOOL_SPECS[attribute])
    assert tool.is_required(context) is expected


def test_capability_resolve_is_required_only_on_the_iteration_after_the_code() -> None:
    tool = IXPerTurnTool(IX_CAPABILITY_RESOLVE_SPEC)
    assert tool.is_required(RequiredContext(capability_limit_seen=True)) is True
    assert tool.is_required(RequiredContext()) is False


# --- per-call injection (carrier C) ----------------------------------------


def test_no_injection_without_a_session_active_attribute_that_needs_it() -> None:
    base, _ = _base()
    turn = TurnToolRegistry(base, [], ix_context={"session_active_ix": ["verbosity"]})
    assert turn.injected_per_call_fields == ()
    parameters = _definition(turn, "email_message_send")["function"]["parameters"]
    assert set(parameters["properties"]) == {"to", "body"}


def test_notice_is_injected_when_process_visibility_is_session_active() -> None:
    base, _ = _base()
    turn = TurnToolRegistry(base, [], ix_context={"session_active_ix": ["process_visibility"]})
    parameters = _definition(turn, "email_message_send")["function"]["parameters"]
    assert IX_NOTICE_FIELD in parameters["properties"]
    assert IX_NOTICE_FIELD in parameters["required"]
    assert INTERACTION_ITEM_ID_FIELD not in parameters["properties"]


@pytest.mark.parametrize(
    "attribute",
    ["autonomy_level", "capability_boundary", "topic_management", "information_elicitation"],
)
def test_interaction_item_id_is_injected_for_every_gate_bearing_attribute(attribute: str) -> None:
    """Offered on all four — and demanded on none of them.

    acc_009: `information_elicitation` and `capability_boundary` have no payload
    that declares a call id, so a `required` field left the PA no way to answer
    but to invent one, which the same check then refused as unknown.
    """
    base, _ = _base()
    turn = TurnToolRegistry(base, [], ix_context={"session_active_ix": [attribute]})
    parameters = _definition(turn, "email_message_send")["function"]["parameters"]
    assert INTERACTION_ITEM_ID_FIELD in parameters["properties"]
    assert INTERACTION_ITEM_ID_FIELD not in parameters["required"]


def test_interaction_item_id_says_which_payloads_may_declare_an_id() -> None:
    """acc_009: the PA invented `draft1` on a turn that declared no id at all.

    Only the autonomy plan and the topic batch declare call ids, so the
    description has to name them and say to omit the field otherwise.
    """
    base, _ = _base()
    turn = TurnToolRegistry(
        base, [], ix_context={"session_active_ix": ["information_elicitation"]},
    )
    parameters = _definition(turn, "email_message_send")["function"]["parameters"]
    description = parameters["properties"][INTERACTION_ITEM_ID_FIELD]["description"]

    assert "IX_render_autonomy_level" in description
    assert "IX_render_topic_management" in description
    assert "only when a plan step or a batch of this same turn declared it" in description
    assert "omit the field" in description


def test_confirmation_message_is_required_for_actions_and_optional_for_reads() -> None:
    """§3.3 (part 2)·C — a pure read may leave the ask empty."""
    base, _ = _base()
    turn = TurnToolRegistry(base, [], ix_context={"session_active_ix": ["autonomy_level"]})

    send = _definition(turn, "email_message_send")["function"]["parameters"]
    assert IX_CONFIRMATION_MESSAGE_FIELD in send["required"]

    read = _definition(turn, "document_read")["function"]["parameters"]
    assert IX_CONFIRMATION_MESSAGE_FIELD in read["properties"]
    assert IX_CONFIRMATION_MESSAGE_FIELD not in read["required"]


def test_injection_keys_on_session_active_ix_not_on_this_turn_s_selection() -> None:
    """§3.0 — `turn_active_ix` drops an attribute once chosen; injection must not."""
    base, _ = _base()
    turn = TurnToolRegistry(
        base,
        active_preferences=[],  # nothing left to select this turn
        ix_context={"session_active_ix": ["autonomy_level", "process_visibility"]},
    )
    parameters = _definition(turn, "email_message_send")["function"]["parameters"]
    for field in PER_CALL_INJECTED_FIELDS:
        assert field in parameters["properties"]


def test_injected_fields_never_reach_the_real_tool() -> None:
    base, send = _base()
    turn = TurnToolRegistry(
        base, [], ix_context={"session_active_ix": ["autonomy_level", "process_visibility"]}
    )

    async def run() -> None:
        await turn.execute(
            "email_message_send",
            {
                "to": "dale@example.com",
                IX_NOTICE_FIELD: "Sending it now.",
                INTERACTION_ITEM_ID_FIELD: "i3",
                IX_CONFIRMATION_MESSAGE_FIELD: "Shall I send it?",
            },
        )

    asyncio.run(run())
    assert send.calls == [{"to": "dale@example.com"}]


def test_injected_fields_are_stripped_even_when_they_were_not_injected() -> None:
    """Fail-safe: a field the model invents must not reach the tool either."""
    base, send = _base()
    turn = TurnToolRegistry(base, [], ix_context={"session_active_ix": ["verbosity"]})

    async def run() -> None:
        await turn.execute(
            "email_message_send", {"to": "x@example.com", INTERACTION_ITEM_ID_FIELD: "i1"}
        )

    asyncio.run(run())
    assert send.calls == [{"to": "x@example.com"}]


# --- per-turn tool payloads -------------------------------------------------


def test_render_tool_payload_carries_its_attribute() -> None:
    tool = IXPerTurnTool(IX_RENDER_TOOL_SPECS["topic_management"])

    async def run() -> str:
        return await tool.execute(
            batch_set_id="bs1",
            item_count=3,
            batches=[{"batch_id": "b1", "label": "Flights", "call_binding": ["i1"]}],
        )

    payload = json.loads(asyncio.run(run()))
    assert payload["attribute"] == "topic_management"
    assert payload["batch_set_id"] == "bs1"


def test_continue_tool_rejects_a_decision_outside_the_three_values() -> None:
    tool = IXPerTurnTool(IX_CONTINUE_TOOL_SPECS["autonomy_level"])
    errors = tool.validate_params({"decisions": [{"id": "a3", "decision": "maybe"}]})
    assert errors and "must be one of" in errors[0]
    assert tool.validate_params({"decisions": [{"id": "a3", "decision": "keep"}]}) == []


def test_capability_resolve_rejects_a_boundary_kind_outside_the_enum() -> None:
    tool = IXPerTurnTool(IX_CAPABILITY_RESOLVE_SPEC)
    payload = {
        "boundary_id": "bd1",
        "blocked_goal": "get the full text",
        "boundary_kind": "vibes",
        "recovery_options": [{"id": "r1", "intent": "ask the author"}],
        "handoff_message": "Shall I write to Imran?",
    }
    assert tool.validate_params(payload)
    payload["boundary_kind"] = "missing_data"
    assert tool.validate_params(payload) == []


def test_the_certainty_carrier_asks_for_a_claim_it_will_write() -> None:
    """`measurements_render.md` round 2, Finding 1; round 3, Findings 1–2.

    Asking for `[[c:<ref>]]` markers cannot work from this carrier: the tool
    call is produced in the iteration *before* the prose, so the model would
    have to mark text it has not written — two live rounds produced a filled
    `certainties` list and zero markers. Committing to a span instead did not
    fix the ordering either: round 3's three `Express` claims were paraphrases
    of markdown the model had not written yet, `0/3` located. So the carrier
    now points at the trailing block (§5.3 carrier F) — written *after* the
    prose, quoting text that already exists — and keeps the arguments as the
    fallback. It still reads the same under every setting.
    """
    spec = IX_RENDER_TOOL_SPECS["uncertainty_expression"]

    assert CERTAINTY_BLOCK_INSTRUCTION in spec.description
    assert "verbatim" in spec.description
    assert "empty list" in spec.description
    # No marker syntax anywhere in the carrier — it is not the production path.
    assert "[[c:" not in spec.description
    # Nothing in the carrier text distinguishes Express from Hide.
    for setting in IX_PASSIVE_SPECS["uncertainty_expression"].settings:
        assert setting not in spec.description

    item = spec.parameters["properties"]["certainties"]["items"]
    assert item["required"] == ["ref", "level", "claim"]
    claim = item["properties"]["claim"]["description"]
    assert "word for word" in claim and "paraphrase" in claim

    # The production instruction is gone from the setting descriptions; they
    # only say what the reader ends up seeing.
    for text in IX_PASSIVE_SPECS["uncertainty_expression"].setting_descriptions.values():
        assert "[[c:" not in text


def test_the_block_makes_the_certainty_tool_optional_but_never_the_payload() -> None:
    """§3.0 — one of the two carriers is owed on every visible-message turn.

    Requiredness is still setting-blind: it turns on whether the response
    carries a block, which every setting is asked for identically.
    """
    tool = IXPerTurnTool(IX_RENDER_TOOL_SPECS["uncertainty_expression"])

    with_block = RequiredContext(
        produces_user_visible_message=True,
        response_text="Booked.\n<certainties></certainties>",
    )
    without = RequiredContext(produces_user_visible_message=True, response_text="Booked.")

    assert tool.is_required(without) is True
    assert tool.is_required(with_block) is False
    # A turn that produces no user-visible message owes nothing either way.
    assert tool.is_required(RequiredContext(produces_user_visible_message=False)) is False


def test_the_narrating_settings_demand_a_notice_on_a_turn_with_task_calls() -> None:
    """Round-2 `acc_003`: Bookend rendered `notices 0/0` behind 21 task calls.

    The PA selected Bookend, made twenty-one calls without an `_ix_notice` on
    any of them, and the user read a turn indistinguishable from Silent. The
    setting text described what the user would see and never said the notice
    was owed, so the rule is stated where the setting is chosen.
    """
    descriptions = IX_PASSIVE_SPECS["process_visibility"].setting_descriptions

    for setting in ("Bookend", "Full narration"):
        text = descriptions[setting]
        assert "at least one task call" in text
        assert f"must carry at least one `{IX_NOTICE_FIELD}`" in text
    # Silent keeps its own rule: fill the field, nothing is printed.
    assert "at least one task call" not in descriptions["Silent"]


def test_the_bookend_opening_line_frames_the_turn_not_the_first_step() -> None:
    """acc_003 t=11: eight calls, and the opening line was notice #1 verbatim.

    "Inserting the Reyes citation key into the introduction draft." described
    one eighth of the turn, and the user read it as "a coherent account of the
    steps taken" — accepting a Bookend as if it were the Full narration the
    persona wanted. Under Bookend the first notice is the only one shown, so it
    has to be written as the frame it will be read as.
    """
    text = IX_PASSIVE_SPECS["process_visibility"].setting_descriptions["Bookend"]

    assert "a frame for the whole turn" in text
    assert "not as a description of the first step alone" in text
    # Full narration prints every notice in call order, so its notices stay
    # per-step and must not inherit the framing rule.
    assert "frame for the whole turn" not in (
        IX_PASSIVE_SPECS["process_visibility"].setting_descriptions["Full narration"]
    )


def test_is_key_names_what_is_not_key_and_not_only_what_is() -> None:
    """acc_039 — the model marked all four plan items key, twice over.

    Under Self-directed `is_key` is the only thing between the user and a
    permission ask on a reversible internal write, and the definitional wording
    (259d9db) held in round 3 and lost in round 4: "Adjust calendar event end
    time from 10:15 to 09:50" — a correction the user had just ordered — came
    through as key. The negatives are concrete now.
    """
    text = IX_PASSIVE_SPECS["autonomy_level"].setting_descriptions["Self-directed"]

    assert "irreversible" in text
    assert "calendar event, note or document draft is not key" in text
    assert "re-timing an entry you created earlier in this session is not key" in text
    assert "named in the imperative" in text


def test_the_plan_says_a_bound_id_and_the_call_id_are_the_same_string() -> None:
    """Round-2 `acc_039`: two writes held `binding_invalid` against a full plan.

    `IX_render_autonomy_level` declared `bound_item_ids: ["create_placeholder"]`
    and the `calendar_event_create` that carried the step out shipped no
    `interaction_item_id` at all, so the collector read `""` and the binding
    failed. Both were repaired in the same turn, but the user lost the two
    narration paragraphs the held calls were carrying.
    """
    description = IX_RENDER_TOOL_SPECS["autonomy_level"].parameters["properties"][
        "plan"
    ]["items"]["properties"]["bound_item_ids"]["description"]

    assert "the same string, character for character" in description
    assert INTERACTION_ITEM_ID_FIELD in description
    assert "leaves the field out" in description


def test_the_rationale_field_contrasts_a_work_log_with_evidence() -> None:
    """`measurements_render.md` Finding 4 — all four released paragraphs were logs."""
    description = IX_RENDER_TOOL_SPECS["reasoning_visibility"].parameters[
        "properties"
    ]["user_facing_rationale"]["description"]

    assert "Bad" in description and "Good" in description
    assert "deciding factor" in description


def test_the_rationale_field_is_second_person_reasons_and_not_owed_every_turn() -> None:
    """Round-2 `acc_011`: 4 of the 5 released rationales were third-person recaps.

    "The user confirmed the 15/12/3 time split…" opened the turn, prepended
    verbatim, on turns that decided nothing — two of them pure closing
    acknowledgements. The field description said what a rationale contains and
    never said who it is addressed to, or that a claimless turn has none.
    """
    description = IX_RENDER_TOOL_SPECS["reasoning_visibility"].parameters[
        "properties"
    ]["user_facing_rationale"]["description"]

    assert "in the second person" in description
    assert "never a recap" in description and "third person" in description
    assert "makes no claim" in description and "send the field empty" in description


def test_batch_carriers_tell_the_pa_its_unmarked_closing_line_still_ships() -> None:
    """§8 item 2 option A — the promise has to be in the tool text the PA reads."""
    for attribute in ("topic_management", "information_elicitation"):
        description = IX_RENDER_TOOL_SPECS[attribute].description
        assert "delivered every turn" in description
        assert "held" in description
        assert "announce" in description


def test_autonomy_level_offers_exactly_the_three_settings() -> None:
    """2026-09-05: `Suggest` was merged into `Self-directed`. 2026-09-06: the
    neutral option was removed, so the enum is the taxonomy and nothing else —
    the model-facing schema is the one place a stale extra setting would still
    be selectable."""
    spec = IX_PASSIVE_SPECS["autonomy_level"]
    assert spec.settings == ("Reactive", "Self-directed", "Autonomous")

    schema = next(
        definition
        for definition in TurnToolRegistry(ToolRegistry(), ["autonomy_level"]).get_definitions()
        if definition["function"]["name"] == "IX_autonomy_level"
    )
    assert schema["function"]["parameters"]["properties"]["setting"]["enum"] == [
        "Reactive", "Self-directed", "Autonomous",
    ]


def test_every_selection_enum_is_exactly_its_taxonomy_settings() -> None:
    """No IX tool offers an option outside its own taxonomy (2026-09-06)."""
    for attribute, spec in IX_PASSIVE_SPECS.items():
        schema = next(
            definition
            for definition in TurnToolRegistry(ToolRegistry(), [attribute]).get_definitions()
            if definition["function"]["name"] == spec.name
        )
        assert schema["function"]["parameters"]["properties"]["setting"]["enum"] == list(
            spec.settings
        ), attribute


# --- §3.6 restatement -------------------------------------------------------


def _selections(**settings: str) -> list[IXSelection]:
    return [
        IXSelection(attribute=attribute, tool_name=f"IX_{attribute}", setting=setting)
        for attribute, setting in settings.items()
    ]


def test_restatement_covers_pass_through_and_the_gate_off_autonomy_setting() -> None:
    text = turn_reminder_message(
        _selections(
            verbosity="Terse",
            solution_breadth="Low",
            autonomy_level="Autonomous",
            topic_management="One-at-a-time",  # rendered, so not restated
        )
    )
    assert "verbosity: Terse" in text
    assert "solution_breadth: Low" in text
    assert "autonomy_level: Autonomous" in text
    assert "topic_management" not in text
    # the restatement quotes the one description in specs.py, not a second copy
    assert IX_PASSIVE_SPECS["verbosity"].setting_descriptions["Terse"] in text


def test_restatement_is_empty_when_nothing_needs_restating() -> None:
    """Only a selected pass-through attribute has a line: with none selected, or
    only rendered ones, the restatement is empty."""
    assert turn_reminder_message([]) == ""
    assert turn_reminder_message(_selections(reasoning_visibility="Show")) == ""
    assert turn_reminder_message(_selections(topic_management="Organize")) == ""


def test_the_restatement_closes_on_the_wrapper_talk_rule() -> None:
    """regress18 `acc_006` t=7 — the only text on that turn that named a setting.

    The beat had no IX tool left (`active_ix_tools == []`, the selection was made
    at t=4) and no repair message, so the restatement line
    `- task_expansion: Low — …` was the whole of what handed the model the word
    `Low`; the reply came back saying *"so it stays within the Low boundary."*
    The clause is setting-blind, so restating it leaks nothing, and an empty
    restatement stays empty rather than becoming a bare instruction.
    """
    text = turn_reminder_message(_selections(task_expansion="Low"))

    assert "task_expansion: Low" in text
    assert text.endswith(NO_WRAPPER_TALK)
    # No block, no clause: the rule rides on the restatement, it is not a turn
    # reminder of its own.
    assert turn_reminder_message([]) == ""
    assert turn_reminder_message(_selections(topic_management="Organize")) == ""


def test_autonomy_gate_settings_are_not_restated() -> None:
    """§3.6 — Reactive / Self-directed are enforced by the gate, so restating them
    would give the model two voices on the same rule."""
    assert turn_reminder_message(_selections(autonomy_level="Reactive")) == ""
    assert turn_reminder_message(_selections(autonomy_level="Self-directed")) == ""


# --- IX argument closure ---------------------------------------------------


class _ExtrasIXTool(Tool):
    name = "IX_test_extras"
    description = "Test an IX tool that explicitly permits extra properties."
    parameters = {
        "type": "object",
        "properties": {"known": {"type": "string"}},
        "required": ["known"],
        "additionalProperties": True,
    }

    async def execute(self, **kwargs):
        return kwargs


class _InternalTypeErrorIXTool(Tool):
    name = "IX_test_internal_type_error"
    description = "Raise from inside a schema-valid IX implementation."
    parameters = {
        "type": "object",
        "properties": {"known": {"type": "string"}},
        "required": ["known"],
    }

    async def execute(self, known: str):
        raise TypeError(f"internal IX failure: {known}")


class _FixedSignatureIXTool(Tool):
    name = "IX_test_fixed_signature"
    description = "Record whether a fixed-signature IX body ran."
    parameters = {
        "type": "object",
        "properties": {"known": {"type": "string"}},
        "required": ["known"],
    }

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, known: str):
        self.calls += 1
        return known


def test_ix_dispatch_closure_does_not_mutate_model_facing_shared_schemas() -> None:
    base, _ = _base()
    turn = TurnToolRegistry(
        base,
        ["verbosity"],
        ix_context={"session_active_ix": ["process_visibility"]},
    )
    selection_parameters = copy.deepcopy(turn._ix_tools["IX_verbosity"].parameters)
    render_parameters = copy.deepcopy(
        turn._ix_tools["IX_render_process_visibility"].parameters
    )

    assert "additionalProperties" not in _definition(turn, "IX_verbosity")["function"][
        "parameters"
    ]
    assert "additionalProperties" not in _definition(
        turn, "IX_render_process_visibility"
    )["function"]["parameters"]
    # Dispatch closure does not rewrite the frozen model-facing contract or shared specs.
    assert turn._ix_tools["IX_verbosity"].parameters == selection_parameters
    assert turn._ix_tools["IX_render_process_visibility"].parameters == render_parameters


def test_fixed_signature_selection_unknown_key_is_structured_invalid_argument() -> None:
    async def run() -> None:
        base, _ = _base()
        turn = TurnToolRegistry(
            base,
            ["verbosity"],
            ix_context={"session_active_ix": ["process_visibility"]},
        )

        selection = await turn.execute("IX_verbosity", {"setting": "Terse", "13": "extra"})
        assert selection == (
            "Error: Invalid parameters for tool 'IX_verbosity': unknown parameter: 13"
        )

    asyncio.run(run())


def test_kwargs_render_tool_keeps_its_existing_extra_payload_behavior() -> None:
    async def run() -> None:
        base, _ = _base()
        turn = TurnToolRegistry(
            base,
            [],
            ix_context={"session_active_ix": ["process_visibility"]},
        )

        result = json.loads(await turn.execute(
            "IX_render_process_visibility",
            {"completion_message": "Done.", "unexpected": "preserved"},
        ))

        assert result["unexpected"] == "preserved"

    asyncio.run(run())


def test_ix_tool_explicitly_allowing_additional_properties_keeps_them() -> None:
    async def run() -> None:
        turn = TurnToolRegistry(ToolRegistry())
        tool = _ExtrasIXTool()
        turn._ix_tools[tool.name] = tool

        parameters = _definition(turn, tool.name)["function"]["parameters"]
        assert parameters["additionalProperties"] is True
        assert await turn.execute(tool.name, {"known": "yes", "extra": 7}) == {
            "known": "yes",
            "extra": 7,
        }

    asyncio.run(run())


def test_schema_valid_internal_type_error_is_not_reclassified_as_bad_arguments() -> None:
    async def run() -> None:
        turn = TurnToolRegistry(ToolRegistry())
        tool = _InternalTypeErrorIXTool()
        turn._ix_tools[tool.name] = tool

        with pytest.raises(TypeError, match="internal IX failure: valid"):
            await turn.execute(tool.name, {"known": "valid"})

    asyncio.run(run())


def test_unknown_argument_never_invokes_the_fixed_signature_ix_body() -> None:
    async def run() -> None:
        turn = TurnToolRegistry(ToolRegistry())
        tool = _FixedSignatureIXTool()
        turn._ix_tools[tool.name] = tool

        result = await turn.execute(tool.name, {"known": "valid", "unknown": "no"})

        assert "unknown parameter: unknown" in result
        assert tool.calls == 0

    asyncio.run(run())
