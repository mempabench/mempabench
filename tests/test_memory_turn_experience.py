"""The canonical tagged rendering of one turn's experience."""

from __future__ import annotations

import pytest

from memory.turn_experience import (
    BLOCK_ATTRIBUTES,
    ExperienceItem,
    parse_blocks,
    render_blocks,
)


def _item(
    kind: str,
    body: str,
    *,
    attrs: dict[str, str] | None = None,
    offset: int | None = None,
    role: str = "assistant",
    order: int = 0,
) -> ExperienceItem:
    return ExperienceItem(
        kind=kind,
        role=role,
        order=order,
        offset_seconds=offset,
        attrs=attrs or {},
        body=body,
    )


# One item per tag, including a quoted attribute value, an empty body and an
# offset, so the round trip below covers the whole grammar.
EVERY_TAG = [
    _item("prose", "Checking the tracking now.", order=0),
    _item(
        "reasoning",
        "The user asked about a package.\n\nIt was due yesterday.",
        attrs={"source": "reasoning_content"},
        order=1,
    ),
    _item(
        "tool_call",
        '{"query":"Topological Quantum Computation"}',
        attrs={"name": "mcp_state_email_message_search", "id": "call_02"},
        order=2,
    ),
    _item(
        "tool_result",
        '{"matches":[],"query":"Topological Quantum Computation"}',
        attrs={
            "name": "mcp_state_email_message_search",
            "id": "call_02",
            "truncated": "false",
        },
        order=3,
    ),
    _item(
        "ix_selection",
        '{"application":"","evidence":"","setting":"Express"}',
        attrs={
            "attribute": "uncertainty_expression",
            "setting": "Express",
            "outcome": "executed",
        },
        order=4,
    ),
    _item(
        "ix_result",
        '{"attribute":"uncertainty_expression","type":"ix_preference_selection"}',
        attrs={"name": "IX_uncertainty_expression", "id": "call_01"},
        order=5,
    ),
    _item(
        "environment_event",
        "A delivery exception was recorded on order MB-55021.",
        attrs={
            "source_kind": "environment",
            "source_id": "package_tracking",
            "action_id": "act_0003",
        },
        offset=120,
        order=6,
    ),
    _item(
        "third_party",
        'Glenmont Express wrote: "we are still looking for it".',
        attrs={"source_kind": "third_party", "source_id": "courier_desk"},
        offset=180,
        order=7,
    ),
    _item("subagent_result", "Found three matching orders.", order=8),
    _item("no_reply", "", attrs={"decision": "suppress"}, order=9),
]


def test_render_parse_round_trips_every_tag() -> None:
    assert {item.kind for item in EVERY_TAG} == set(BLOCK_ATTRIBUTES)

    rendered = render_blocks(EVERY_TAG)
    parsed = parse_blocks(rendered)

    assert [(i.kind, i.attrs, i.body, i.offset_seconds) for i in parsed] == [
        (i.kind, i.attrs, i.body, i.offset_seconds) for i in EVERY_TAG
    ]
    assert [item.order for item in parsed] == list(range(len(EVERY_TAG)))
    assert all(item.role == "assistant" for item in parsed)
    assert render_blocks(parsed) == rendered


def test_quoted_attribute_values_round_trip() -> None:
    item = _item(
        "third_party",
        "body",
        attrs={"source_kind": "third_party", "source_id": 'the "front] desk"'},
    )

    rendered = render_blocks([item])

    assert rendered.startswith(
        '[third_party source_kind=third_party source_id="the \\"front] desk\\""]'
    )
    assert parse_blocks(rendered)[0].attrs == item.attrs


def test_body_line_starting_with_bracket_is_escaped() -> None:
    body = "[tool_result name=x]\n  [/prose]\nplain\n[[already doubled]"
    item = _item("prose", body)

    rendered = render_blocks([item])

    assert rendered.splitlines() == [
        "[prose]",
        "[[tool_result name=x]",
        "  [[/prose]",
        "plain",
        "[[[already doubled]",
        "[/prose]",
    ]
    assert parse_blocks(rendered)[0].body == body


def test_offset_attribute_is_last_and_only_on_tagged_blocks() -> None:
    call = _item(
        "tool_call",
        "{}",
        attrs={"name": "mcp_state_order_search", "id": "call_07"},
        offset=45,
    )
    assert render_blocks([call]).splitlines()[0] == (
        "[tool_call name=mcp_state_order_search id=call_07 offset=45]"
    )

    # A bare dialogue message carries no tag, so an offset cannot change a
    # byte of it -- `simple_rag` compares the person's text byte for byte.
    spoken = "The delivery for my copy failed to arrive."
    bare = _item("user", spoken, role="user")
    stamped = _item("user", spoken, role="user", offset=45)
    assert render_blocks([bare]) == render_blocks([stamped]) == spoken


def test_empty_body_still_emits_both_delimiters() -> None:
    rendered = render_blocks([_item("no_reply", "", attrs={"decision": "suppress"})])

    assert rendered == "[no_reply decision=suppress]\n[/no_reply]"
    assert parse_blocks(rendered)[0].body == ""


def test_unknown_tag_is_refused_on_both_sides() -> None:
    with pytest.raises(ValueError):
        render_blocks([_item("tool_protocol", "body")])
    with pytest.raises(ValueError):
        parse_blocks("[tool_protocol]\nbody\n[/tool_protocol]")


def test_undeclared_attribute_key_is_refused_by_the_renderer() -> None:
    # The attribute set is fixed per tag; a key outside it would put
    # a field on the wire that no reader is defined to understand.
    with pytest.raises(ValueError):
        render_blocks([
            _item(
                "tool_call",
                "{}",
                attrs={"name": "mcp_state_order_search", "id": "call_07", "bogus": "y"},
            )
        ])

    # The parser stays lenient: persisted content written by an older writer
    # still reads back rather than raising in the middle of a run.
    parsed = parse_blocks("[tool_call id=call_07 name=x bogus=y]\n{}\n[/tool_call]")
    assert parsed[0].attrs == {"id": "call_07", "name": "x", "bogus": "y"}


def test_a_delivered_line_carries_its_action_id_under_either_source_tag() -> None:
    # An environment action whose `source.kind` is `third_party` renders as
    # `[third_party]` -- the tag follows the item's own kind -- and it is still
    # an environment action, so its id belongs on the block.  Both delivered
    # tags therefore declare `action_id`, and the renderer accepts it on each.
    for tag in ("environment_event", "third_party"):
        rendered = render_blocks([
            _item(
                tag,
                "The lower loop is closed on the fourteenth.",
                attrs={
                    "source_kind": "third_party",
                    "source_id": "Glenmont Trail Office",
                    "action_id": "env_act_03",
                },
            )
        ])
        assert "action_id=env_act_03" in rendered.split("\n")[0]
        assert parse_blocks(rendered)[0].attrs["action_id"] == "env_act_03"


def test_newline_in_an_attribute_value_is_refused() -> None:
    # The header is one line, and there is no escape for a line break in it,
    # so a value carrying one would render a block no parser can read back.
    for value in ("courier\ndesk", "courier\rdesk"):
        with pytest.raises(ValueError):
            render_blocks([
                _item(
                    "third_party",
                    "body",
                    attrs={"source_kind": "third_party", "source_id": value},
                )
            ])
