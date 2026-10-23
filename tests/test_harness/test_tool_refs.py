"""A `$producer.path` argument that cannot be parsed is an error.

`REF_RE` only accepts identifier-shaped path segments, so a key with a space in
it -- `nightly_rates.ocean view`, which is how the user_c hotel fixtures label a
rate -- did not match, and `refs_in` skipped it. The argument was then sent to
the world as the literal string `"$s.matches.0.nightly_rates.ocean view"`: the
author's reference silently became text, and nothing said so.

The fix is an error, not a new grammar. There is no quoted-key syntax to learn;
the message tells the author that key cannot be referenced.
"""

import pytest

from harness.tool_refs import ToolRefError, refs_in, resolve_refs


def test_a_ref_whose_key_has_a_space_is_an_error():
    with pytest.raises(ToolRefError, match="ocean view"):
        refs_in({"rate": "$s.matches.0.nightly_rates.ocean view"})


def test_the_error_names_the_argument_and_says_the_key_cannot_be_referenced():
    with pytest.raises(ToolRefError) as excinfo:
        refs_in({"rate": "$s.matches.0.nightly_rates.ocean view"})

    message = str(excinfo.value)
    assert "'rate'" in message
    assert "cannot be referenced" in message


def test_resolve_refs_raises_on_the_same_value():
    with pytest.raises(ToolRefError):
        resolve_refs(
            {"rate": "$s.matches.0.nightly_rates.ocean view"},
            {"s": {"matches": [{"nightly_rates": {"ocean view": 310}}]}},
        )


def test_a_valid_ref_still_resolves():
    arguments = {"email_id": "$search_inbox.matches.0.email_id"}

    assert refs_in(arguments) == {
        "email_id": ("search_inbox", ["matches", "0", "email_id"])
    }
    assert resolve_refs(
        arguments, {"search_inbox": {"matches": [{"email_id": "em_0007"}]}}
    ) == {"email_id": "em_0007"}


# Only a value that is *trying* to be a reference -- the sigil, a producer name
# and a dotted path -- is held to the grammar. A price is not a failed
# reference, and the drafts are full of them.
@pytest.mark.parametrize(
    "value",
    ["$45.00", "$2,040.00", "$$$", "$", "$0.67", "$d^{*}$", "plain text", "$notadot"],
)
def test_a_string_that_is_not_a_reference_attempt_is_left_alone(value: str):
    assert refs_in({"note": value}) == {}
    assert resolve_refs({"note": value}, {}) == {"note": value}


# A dictated gate value may be EMBEDDED in a larger argument
# -- the body a draft writes around `$task_facts.<key>`. The reference stays
# verbatim in the draft and resolves in place at run time.
def test_an_embedded_task_fact_reference_is_read_and_substituted_in_place():
    from harness.tool_refs import (
        embedded_task_fact_paths,
        resolve_task_fact_refs,
        task_fact_refs,
    )

    arguments = {
        "replacement": "Notes: $task_facts.visit_note\nFiled by the assistant.",
        "title": "$task_facts.visit_note",
    }
    # The template is not a reference-VALUED argument, so `refs_in` leaves it
    # alone -- and, critically, no longer rejects it as a broken reference.
    assert refs_in(arguments) == {"title": ("task_facts", ["visit_note"])}
    assert embedded_task_fact_paths(arguments["replacement"]) == [["visit_note"]]
    assert task_fact_refs(arguments) == {
        "replacement": ["visit_note"],
        "title": ["visit_note"],
    }
    assert resolve_task_fact_refs(arguments, {"visit_note": "bring the blood panel"}) == {
        "replacement": "Notes: bring the blood panel\nFiled by the assistant.",
        "title": "bring the blood panel",
    }


def test_an_embedded_reference_at_the_head_of_a_template_is_not_a_broken_ref():
    from harness.tool_refs import resolve_task_fact_refs

    # `REF_ATTEMPT_RE` matches this string, so it used to raise.
    arguments = {"content": "$task_facts.visit_note -- please file it."}

    assert refs_in(arguments) == {}
    assert resolve_task_fact_refs(arguments, {"visit_note": "blood panel"}) == {
        "content": "blood panel -- please file it."
    }


def test_an_embedded_reference_to_a_fact_nobody_holds_is_left_standing():
    from harness.tool_refs import resolve_task_fact_refs

    arguments = {"content": "Note: $task_facts.absent_key."}

    assert resolve_task_fact_refs(arguments, {}) == arguments


def test_a_nested_embedded_reference_resolves():
    from harness.tool_refs import resolve_task_fact_refs, task_fact_refs

    arguments = {"content": "See $task_facts.file_locations.draft_intro now."}

    assert task_fact_refs(arguments) == {"content": ["file_locations", "draft_intro"]}
    assert resolve_task_fact_refs(
        arguments, {"file_locations": {"draft_intro": "~/Drafts/intro.md"}}
    ) == {"content": "See ~/Drafts/intro.md now."}


# A reference may also sit in a LIST element.
# `email_message_send.attachments` is an array of state-relative paths, and the
# path an earlier read returns is the only non-brittle way to name the file --
# writing the path out is what the draft path rules forbid. Before this, the
# four functions below only walked top-level argument values, so
# `attachments: [$step.path]` went to the world verbatim and the store raised
# FileNotFoundError on the literal string.
def test_a_ref_in_a_list_element_is_read_and_resolved():
    arguments = {
        "to": "$read_contact.contact.email",
        "attachments": ["$read_packet.path"],
    }

    assert refs_in(arguments) == {
        "to": ("read_contact", ["contact", "email"]),
        "attachments.0": ("read_packet", ["path"]),
    }
    assert resolve_refs(
        arguments,
        {
            "read_contact": {"contact": {"email": "h.brenner@example"}},
            "read_packet": {"path": "documents/acting/day_player_packet.md"},
        },
    ) == {
        "to": "h.brenner@example",
        "attachments": ["documents/acting/day_player_packet.md"],
    }


def test_non_reference_list_elements_are_left_alone():
    arguments = {"attachments": ["documents/acting/packet.md", "$45.00"]}

    assert refs_in(arguments) == {}
    assert resolve_refs(arguments, {}) == arguments


def test_a_broken_ref_in_a_list_element_is_an_error_naming_its_position():
    with pytest.raises(ToolRefError) as excinfo:
        refs_in({"attachments": ["$s.matches.0.nightly_rates.ocean view"]})

    assert "'attachments.0'" in str(excinfo.value)


def test_a_task_fact_ref_in_a_list_element_is_read_and_substituted():
    from harness.tool_refs import resolve_task_fact_refs, task_fact_refs

    arguments = {"attachments": ["$task_facts.packet_path", "Note: $task_facts.extra"]}

    assert task_fact_refs(arguments) == {
        "attachments.0": ["packet_path"],
        "attachments.1": ["extra"],
    }
    assert resolve_task_fact_refs(
        arguments, {"packet_path": "documents/acting/packet.md", "extra": "keep a copy"}
    ) == {"attachments": ["documents/acting/packet.md", "Note: keep a copy"]}


def test_ref_producers_are_rewritten_inside_a_list():
    from harness.tool_refs import rewrite_ref_producers

    arguments = {"attachments": ["$read_packet.path"], "to": "someone@example"}

    assert rewrite_ref_producers(arguments, {"read_packet": "call_03"}) == {
        "attachments": ["$call_03.path"],
        "to": "someone@example",
    }


def test_resolving_a_list_ref_does_not_mutate_the_caller_s_list():
    original = ["$read_packet.path"]
    arguments = {"attachments": original}

    resolve_refs(arguments, {"read_packet": {"path": "documents/a.md"}})

    assert original == ["$read_packet.path"]


# `$<step>.files[<state-relative path>]`. A folder listing is
# the one place a fixed index is a STRUCTURAL error rather than an off-by-one.
# The three worlds below are the same folder, `documents/bistro`, at three real
# moments of user_b's arc, and `documents/bistro/tipout_and_paystubs.md` sits at
# a different index in each:
#
#   * the L0 fixture (8 files, index 7) -- `data/state_fixtures/user_b`
#   * L0 plus what acc_014 and acc_033 write (10 files, index 9)
#   * the live run's acc_074 checkpoint (11 files, index 10), where three of the
#     names carry a runtime date the PA itself chose
#
# So no fixed index survives the arc, and the selector is the only way the draft
# can name the file it means.
_TARGET = "documents/bistro/tipout_and_paystubs.md"
_SELECTOR_REF = f"$list_bistro_folder.files[{_TARGET}]"
_INDEX_REF = "$list_bistro_folder.files.7"

_WORLD_L0 = [
    "documents/bistro/allergen_sheet.md",
    "documents/bistro/bar_count_sheet.md",
    "documents/bistro/floor_plan_and_bookings.md",
    "documents/bistro/health_code_binder.md",
    "documents/bistro/incident_log.md",
    "documents/bistro/schedule_posting.md",
    "documents/bistro/staff_contacts_and_notes.md",
    "documents/bistro/tipout_and_paystubs.md",
]

# acc_014 creates `bar_order_sheet.md`; acc_033 creates
# `carla_pay_and_hours_note.md`. `read_document` sorts, so they land ahead of
# the target and push it to 9.
_WORLD_PLUS_TWO = sorted(
    _WORLD_L0
    + [
        "documents/bistro/bar_order_sheet.md",
        "documents/bistro/carla_pay_and_hours_note.md",
    ]
)

# Read off the live run's acc_074 checkpoint (`canonical_stage`).
_WORLD_LIVE = [
    "documents/bistro/allergen_sheet.md",
    "documents/bistro/bar_count_sheet.md",
    "documents/bistro/bar_order_sheet_2026-10-05.md",
    "documents/bistro/floor_plan_and_bookings.md",
    "documents/bistro/health_code_binder.md",
    "documents/bistro/incident_log.md",
    "documents/bistro/note_to_carla_missing_hours_2026-09-19.md",
    "documents/bistro/pay_claim_missing_hours_2026-09-19.md",
    "documents/bistro/schedule_posting.md",
    "documents/bistro/staff_contacts_and_notes.md",
    "documents/bistro/tipout_and_paystubs.md",
]


def _listing(files: list[str]) -> dict[str, object]:
    """What `StateStore.read_document` hands back for a directory."""
    return {"path": "documents/bistro", "type": "directory", "files": files}


@pytest.mark.parametrize(
    "files, seventh",
    [
        (_WORLD_L0, "documents/bistro/tipout_and_paystubs.md"),
        (_WORLD_PLUS_TWO, "documents/bistro/schedule_posting.md"),
        (_WORLD_LIVE, "documents/bistro/pay_claim_missing_hours_2026-09-19.md"),
    ],
    ids=["l0_8_files", "plus_two_10_files", "live_checkpoint_11_files"],
)
def test_the_selector_picks_the_named_file_whatever_the_folder_holds(
    files: list[str], seventh: str
):
    results = {"list_bistro_folder": _listing(files)}

    assert resolve_refs({"path": _SELECTOR_REF}, results) == {"path": _TARGET}
    # And the fixed index resolves to whatever happens to be at 7 in this world,
    # which is the target in the L0 world alone.
    assert resolve_refs({"path": _INDEX_REF}, results) == {"path": seventh}
    assert files[7] == seventh


def test_no_single_fixed_index_names_the_target_in_all_three_worlds():
    """The matrix cannot be passed by a fixed index -- the point of the selector."""
    worlds = [_WORLD_L0, _WORLD_PLUS_TWO, _WORLD_LIVE]
    assert [world.index(_TARGET) for world in worlds] == [7, 9, 10]
    assert not any(
        all(index < len(world) and world[index] == _TARGET for world in worlds)
        for index in range(max(len(world) for world in worlds))
    )
    # The selector names it in every one.
    for world in worlds:
        assert resolve_refs(
            {"path": _SELECTOR_REF}, {"list_bistro_folder": _listing(world)}
        ) == {"path": _TARGET}


def test_a_selector_that_names_no_returned_path_is_an_error_naming_the_list():
    """No fallback. The message prints the list it actually got, so the author
    can see whether the file is absent or is there under another name."""
    missing = [path for path in _WORLD_L0 if path != _TARGET]

    with pytest.raises(ToolRefError) as excinfo:
        resolve_refs({"path": _SELECTOR_REF}, {"list_bistro_folder": _listing(missing)})

    message = str(excinfo.value)
    assert "'path'" in message
    assert "matches none of the 7 paths" in message
    # The list it really saw, not a summary.
    assert "documents/bistro/schedule_posting.md" in message
    assert _TARGET in message


def test_a_selector_matching_more_than_one_returned_path_is_ambiguous():
    """The selector's third limit: a tie fails rather than picking one."""
    doubled = _WORLD_L0 + [_TARGET]

    with pytest.raises(ToolRefError) as excinfo:
        resolve_refs({"path": _SELECTOR_REF}, {"list_bistro_folder": _listing(doubled)})

    message = str(excinfo.value)
    assert "matches 2 of the paths" in message
    assert "ambiguous" in message


def test_a_selector_needs_a_list_of_strings_to_choose_from():
    with pytest.raises(ToolRefError, match="needs a list of strings"):
        resolve_refs(
            {"path": "$read_doc.content[documents/bistro/x.md]"},
            {"read_doc": {"content": "a document body"}},
        )


def test_the_selector_is_one_path_segment_and_is_not_split_on_dots():
    """Split on `.`, the `.md` suffix would become a segment of its own."""
    from harness.tool_refs import parse_ref

    assert parse_ref(_SELECTOR_REF) == (
        "list_bistro_folder",
        ["files", f"[{_TARGET}]"],
    )
    assert refs_in({"path": _SELECTOR_REF}) == {
        "path": ("list_bistro_folder", ["files", f"[{_TARGET}]"])
    }


def test_a_selector_ref_survives_the_producer_rewrite_and_parses_again():
    """The forced-transcript rewrite rebuilds the string; `files.[...]` would
    not parse a second time."""
    from harness.tool_refs import parse_ref, rewrite_ref_producers

    rewritten = rewrite_ref_producers(
        {"path": _SELECTOR_REF}, {"list_bistro_folder": "call_02"}
    )

    assert rewritten == {"path": f"$call_02.files[{_TARGET}]"}
    assert parse_ref(rewritten["path"]) == ("call_02", ["files", f"[{_TARGET}]"])


# The grammar stayed narrow on purpose: one selector, at the end, no spaces, no
# wildcard, no `..`. Each of these reads as a reference (`REF_ATTEMPT_RE`) and
# does not parse, which is the module's hard error -- it never becomes a
# literal string the world then fails to find.
@pytest.mark.parametrize(
    "value",
    [
        "$s.files[documents/bistro/tipout and paystubs.md]",  # a space
        "$s.files[*.md]",  # a wildcard
        "$s.files[../bistro/x.md]",  # a traversal
        "$s.files[documents/a.md].name",  # not at the end
        "$s.files[a.md][b.md]",  # two selectors
        "$s.files[]",  # empty
    ],
)
def test_a_selector_outside_the_grammar_is_a_hard_error_not_a_literal(value: str):
    with pytest.raises(ToolRefError) as excinfo:
        refs_in({"path": value})

    assert "reads as a reference but does not parse" in str(excinfo.value)


# acc_089 adopts the same selector, on a second folder. `my_desktop`
# holds five files in the L0 fixture with `expense_log.md` alphabetically first,
# and acc_074 -- earlier in the arc -- creates `my_desktop/brian_gift_shortlist.md`,
# which sorts ahead of it. So `files.0` names the expense log before acc_074 and
# the gift shortlist after it.
_DESKTOP_TARGET = "my_desktop/expense_log.md"
_DESKTOP_L0 = [
    "my_desktop/expense_log.md",
    "my_desktop/monthly_bills.md",
    "my_desktop/patch_notes.txt",
    "my_desktop/screenplay_draft.md",
    "my_desktop/title_page_and_header.md",
]
_DESKTOP_AFTER_ACC_074 = sorted(_DESKTOP_L0 + ["my_desktop/brian_gift_shortlist.md"])


@pytest.mark.parametrize(
    "files, first",
    [
        (_DESKTOP_L0, _DESKTOP_TARGET),
        (_DESKTOP_AFTER_ACC_074, "my_desktop/brian_gift_shortlist.md"),
    ],
    ids=["l0_5_files", "after_acc_074_6_files"],
)
def test_the_selector_names_the_expense_log_before_and_after_acc_074(
    files: list[str], first: str
):
    results = {
        "list_desktop_folder": {
            "path": "my_desktop",
            "type": "directory",
            "files": files,
        }
    }

    assert resolve_refs(
        {"path": f"$list_desktop_folder.files[{_DESKTOP_TARGET}]"}, results
    ) == {"path": _DESKTOP_TARGET}
    # The index acc_089 used to carry, for contrast.
    assert resolve_refs({"path": "$list_desktop_folder.files.0"}, results) == {
        "path": first
    }
    assert files[0] == first


def test_no_fixed_index_names_the_expense_log_both_before_and_after_acc_074():
    worlds = [_DESKTOP_L0, _DESKTOP_AFTER_ACC_074]
    assert [world.index(_DESKTOP_TARGET) for world in worlds] == [0, 1]
    assert not any(
        all(index < len(world) and world[index] == _DESKTOP_TARGET for world in worlds)
        for index in range(max(len(world) for world in worlds))
    )
