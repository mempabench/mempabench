"""Direct tests for IX helper functions."""

import pytest

from pa.agent.ix.action_types import tool_action_type
from pa.agent.ix.collector import assert_action_type_coverage, gate_opens
from pa.agent.ix.events import IXDirective, IXSelection, extract_ix_payload
from pa.agent.ix.policy import directive_for


def test_extract_ix_payload_parses_dict_and_json_string() -> None:
    payload = {
        "type": "ix_preference_selection",
        "attribute": "solution_breadth",
        "setting": "High",
        "evidence": "The user asked for a shortlist.",
        "application": "Compare two options and recommend one.",
    }
    # `payload_type` distinguishes a setting choice from a per-turn payload;
    # both kinds of result carry an attribute, only one carries a setting.
    expected = {
        "payload_type": "ix_preference_selection",
        **{key: value for key, value in payload.items() if key != "type"},
    }

    assert extract_ix_payload("IX_solution_breadth", payload) == expected
    assert extract_ix_payload(
        "IX_solution_breadth",
        (
            '{"type": "ix_preference_selection", '
            '"attribute": "solution_breadth", "setting": "High", '
            '"evidence": "The user asked for a shortlist.", '
            '"application": "Compare two options and recommend one."}'
        ),
    ) == expected


def test_extract_ix_payload_ignores_non_ix_or_unparseable_results() -> None:
    assert extract_ix_payload("email_message_search", {"setting": "Medium"}) is None
    assert extract_ix_payload("IX_solution_breadth", "not json") is None
    assert extract_ix_payload("IX_solution_breadth", ["Medium"]) is None


def test_tool_action_type_uses_overrides_defaults_and_fails_closed() -> None:
    assert tool_action_type("email_message_send") == "external_action"
    assert tool_action_type("document_update") == "internal_write"
    assert tool_action_type("cinema_ticket_purchase") == "external_action"
    assert tool_action_type("mcp_state_email_draft_create") == "draft"
    assert tool_action_type("mcp_state_weather_forecast_read") == "read"
    # An unregistered tool used to be treated as a read, which let any newly
    # added send / pay / book tool slip past both the Reactive gate and the
    # Self-directed floor. Unknown now means the most tightly gated class.
    assert tool_action_type("unknown_tool") == "external_action"
    assert tool_action_type(
        "unknown_tool",
        {"unknown_tool": "read"},
    ) == "read"


def test_action_type_coverage_rejects_an_unregistered_task_tool() -> None:
    assert_action_type_coverage(["email_message_send", "IX_autonomy_level"]) is None
    assert_action_type_coverage(
        ["brand_new_tool"], {"brand_new_tool": "internal_write"},
    ) is None
    with pytest.raises(ValueError, match="brand_new_tool"):
        assert_action_type_coverage(["email_message_send", "brand_new_tool"])


def test_ix_directive_has_control_directive() -> None:
    assert IXDirective().has_control_directive is False
    assert IXDirective(information_elicitation="infer").has_control_directive is False
    assert IXDirective(information_elicitation="structured").has_control_directive is True
    assert IXDirective(emit_start=True).has_control_directive is True
    assert IXDirective(autonomy_level_policy="confirm_key_actions").has_control_directive is True
    assert IXDirective(solution_breadth="High").has_control_directive is True


# --- repair messages say not to answer them out loud (round-2 finding 5) -----


def test_every_repair_message_forbids_a_meta_acknowledgement() -> None:
    """The three internal repairs a turn can send all carry the same closing.

    Live, the gate repair produced `You're right — … Let me re-send that` in the
    user-visible turn: the PA answering the wrapper where the user can read it.
    """
    from pa.agent.ix.collector import NO_META_ACKNOWLEDGEMENT, validate_bindings
    from pa.agent.runner import IX_GATE_MESSAGE_REPAIR

    binding_repair = validate_bindings(
        [_FakeCall("task_send", {})], {"i1"},
    )
    assert binding_repair is not None
    assert binding_repair.endswith(NO_META_ACKNOWLEDGEMENT)
    assert IX_GATE_MESSAGE_REPAIR.endswith(NO_META_ACKNOWLEDGEMENT)


def test_the_wrapper_talk_rule_also_bans_narrating_the_chosen_level() -> None:
    """Round-2 `acc_019`: the PA explained its own guidance-level choice.

    Three replies said the explanation was pitched for someone unfamiliar with
    the method, or that one gloss was all that was needed. The setting has to
    show in how the reply is written, not in a line about how it was written,
    so the rule the repairs already carry names that case too.
    """
    from pa.agent.ix.collector import NO_META_ACKNOWLEDGEMENT, NO_WRAPPER_TALK

    assert (
        "never say why your explanation is pitched where it is or that you "
        "chose a level for it"
    ) in NO_WRAPPER_TALK
    # Every repair closes with it, so the new clause travels with them.
    assert NO_META_ACKNOWLEDGEMENT.endswith(NO_WRAPPER_TALK)


def test_the_standing_texts_carry_the_wrapper_talk_rule_too() -> None:
    """regress18 `acc_009` t=4 / `acc_006` t=7 — the rule reached neither turn.

    `NO_WRAPPER_TALK` rode only on the repair messages, so the one instruction
    written to stop the PA narrating the mechanism was delivered only on turns
    where something had already gone wrong. Both leaks came off turns with no
    repair at all: t=4 answered the wrapper's own release vocabulary back to the
    user (*"Great, all four questions have been released."*) and t=7 named the
    setting it had chosen (*"so it stays within the Low boundary."*). The two
    standing texts that hand the model that vocabulary now close on the clause.
    """
    from pa.agent.ix.collector import NO_WRAPPER_TALK
    from pa.agent.ix.specs import SETTING_COMMITMENT_NOTE

    assert SETTING_COMMITMENT_NOTE.endswith(NO_WRAPPER_TALK)


def test_the_fallback_text_drops_a_leading_meta_acknowledgement() -> None:
    """`last_segment` is a delivery surface, so it strips what the renderer strips."""
    from pa.agent.ix.collector import TurnCollector

    collector = TurnCollector()
    collector.add_segment(
        "You're right, let me re-send that.\n\nThe 10:00 slot is free."
    )

    assert collector.last_segment == "The 10:00 slot is free."
    # `segments` keeps the raw text: the audit plane is the one place it lives.
    assert collector.segments[-1].startswith("You're right")


class _FakeCall:
    def __init__(self, name: str, arguments: dict) -> None:
        self.name = name
        self.arguments = arguments


def test_self_directed_gates_key_writes_and_every_external_action() -> None:
    """§3.3 (part 2)·B — since 2026-09-05 the merged middle setting carries the gate that
    used to belong to the removed fourth setting: reads and drafts run, a key
    internal write stops the turn, and an external action always stops it."""
    directive = directive_for([
        IXSelection(
            attribute="autonomy_level",
            tool_name="IX_autonomy_level",
            setting="Self-directed",
            evidence="The user delegated the routine sends.",
            application="Prepare, then ask before it goes out.",
        )
    ])
    policy = directive.autonomy_level_policy
    assert policy == "confirm_key_actions"

    assert gate_opens(policy, "internal_write", True) is True
    assert gate_opens(policy, "internal_write", False) is False
    # The external floor holds whatever the model says about `is_key`.
    assert gate_opens(policy, "external_action", True) is True
    assert gate_opens(policy, "external_action", False) is True
    # Reversible preparation never stops the turn, `is_key` or not.
    assert gate_opens(policy, "draft", True) is False
    assert gate_opens(policy, "read", True) is False
