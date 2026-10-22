import json
from pathlib import Path

import pytest

from state_server.store import GATE_OPEN_MARKER, StateStore


def test_document_read_and_email_draft_create(tmp_path: Path):
    state = tmp_path / "state"
    (state / "documents").mkdir(parents=True)
    (state / "email").mkdir()
    (state / "documents" / "intro.md").write_text("draft text", encoding="utf-8")

    store = StateStore(
        state, run_id="run_001", user_id="user_a", session_id="session_01"
    )

    doc = store.read_document("documents/intro.md")
    assert doc["content"] == "draft text"
    assert doc["bytes"] == len("draft text")

    result = store.save_email_draft(
        to="management@example.test",
        subject="Elevator",
        body="Please repair the elevator.",
        attachments=["documents/intro.md"],
    )
    assert result == {"draft_id": "draft_0001", "status": "saved"}

    drafts = (state / "email" / "drafts.jsonl").read_text(encoding="utf-8")
    assert "Please repair the elevator." in drafts
    assert "documents/intro.md" in drafts

    tool_log = (state / "tool_log.jsonl").read_text(encoding="utf-8")
    assert "document_read" in tool_log
    assert "email_draft_create" in tool_log

    state_diff = (state / "state_diff.jsonl").read_text(encoding="utf-8")
    assert "email.drafts" in state_diff


def test_email_draft_preserves_comma_separated_recipients(tmp_path: Path) -> None:
    state = tmp_path / "state"
    (state / "email").mkdir(parents=True)
    store = StateStore(state, run_id="run_001", user_id="user_a", session_id="session_01")
    recipients = "first@example.test, second@example.test"

    result = store.save_email_draft(
        to=recipients,
        subject="Class notice",
        body="The reading is posted.",
    )

    assert result == {"draft_id": "draft_0001", "status": "saved"}
    record = json.loads((state / "email" / "drafts.jsonl").read_text(encoding="utf-8"))
    assert record["to"] == recipients


def _document_store(tmp_path: Path, text: str = "draft text") -> StateStore:
    state = tmp_path / "state"
    (state / "documents").mkdir(parents=True)
    (state / "documents" / "intro.md").write_text(text, encoding="utf-8")
    return StateStore(state, run_id="run_001", user_id="user_a", session_id="session_03")


def test_document_update_replaces_one_anchor_and_records_the_write(tmp_path: Path):
    store = _document_store(tmp_path, "# Intro\n\nFLAG — resolve this.\n\n# End\n")
    state = tmp_path / "state"

    result = store.update_document(
        "documents/intro.md", "FLAG — resolve this.", "RESOLVED — same mechanism."
    )

    assert result["path"] == "documents/intro.md"
    assert result["status"] == "updated"
    assert result["match_tier"] == "exact"
    assert (state / "documents" / "intro.md").read_text(encoding="utf-8") == (
        "# Intro\n\nRESOLVED — same mechanism.\n\n# End\n"
    )

    tool_log = (state / "tool_log.jsonl").read_text(encoding="utf-8")
    assert "document_update" in tool_log

    state_diff = (state / "state_diff.jsonl").read_text(encoding="utf-8")
    assert "documents" in state_diff
    assert "documents/intro.md" in state_diff


def test_document_update_rejects_a_missing_file(tmp_path: Path):
    store = _document_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.update_document("documents/absent.md", "draft", "final")


def test_document_update_rejects_a_missing_anchor(tmp_path: Path):
    store = _document_store(tmp_path)
    with pytest.raises(ValueError, match="anchor not found"):
        store.update_document("documents/intro.md", "no such span", "final")


def test_document_update_rejects_a_duplicate_anchor(tmp_path: Path):
    store = _document_store(tmp_path, "note\nnote\n")
    with pytest.raises(ValueError, match="occurs 2 times"):
        store.update_document("documents/intro.md", "note", "final")


# A bare heading line as the anchor. Live (regress13 acc_001 t=18) the PA sent
# `## Raw — needs converting to a proper entry` and a replacement that opened
# with its own heading: the section heading was consumed and the raw note it
# introduced was left under the replacement's heading.


@pytest.mark.parametrize(
    "heading",
    ["# Intro", "## Raw — needs converting", "###### Deep note"],
)
def test_document_update_refuses_a_bare_heading_line_anchor(
    tmp_path: Path, heading: str
):
    text = f"{heading}\n\nThe first content line.\n"
    store = _document_store(tmp_path, text)
    state = tmp_path / "state"

    result = store.update_document("documents/intro.md", heading, "## Converted\n\nnew")

    assert result["ok"] is False
    assert result["error_code"] == "anchor_is_heading"
    assert "content line under the heading" in result["detail"]
    # The document is untouched, and the refusal is on the record as an error.
    assert (state / "documents" / "intro.md").read_text(encoding="utf-8") == text
    entry = json.loads(
        (state / "tool_log.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert entry["tool"] == "document_update"
    assert entry["status"] == "error"
    assert entry["result_summary"] == {"error_code": "anchor_is_heading"}


# An anchor that stops at the end of a wrapped list item's first line. Live
# (regress19 acc_035 t=8) the PA anchored on the first line of a two-line
# "Notes to self" bullet and inserted a new bullet after it; the old bullet's
# clause was cut in half around the insertion.

_WRAPPED = (
    "Notes to self:\n"
    "\n"
    "- (*) Rahimi Q3: regrade requested — decision pending, not\n"
    "  applied. The table shows the original mark (7).\n"
    "- Second note.\n"
)


def test_document_update_refuses_an_anchor_ending_inside_a_wrapped_item(
    tmp_path: Path,
):
    store = _document_store(tmp_path, _WRAPPED)
    state = tmp_path / "state"
    anchor = "- (*) Rahimi Q3: regrade requested — decision pending, not"

    result = store.update_document(
        "documents/intro.md", anchor, anchor + "\n- Rahimi Q1: challenge upheld."
    )

    assert result["ok"] is False
    assert result["error_code"] == "anchor_ends_mid_item"
    assert "applied. The table shows" in result["detail"]
    assert (state / "documents" / "intro.md").read_text(encoding="utf-8") == _WRAPPED
    entry = json.loads(
        (state / "tool_log.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert entry["status"] == "error"
    assert entry["result_summary"] == {"error_code": "anchor_ends_mid_item"}


@pytest.mark.parametrize(
    "anchor",
    [
        # the whole wrapped item
        "- (*) Rahimi Q3: regrade requested — decision pending, not\n"
        "  applied. The table shows the original mark (7).",
        # its last wrapped line
        "  applied. The table shows the original mark (7).",
        # a mid-line phrase inside the first line
        "decision pending",
        # a single-line item followed by nothing indented
        "- Second note.",
    ],
)
def test_document_update_accepts_anchors_that_do_not_split_a_wrapped_item(
    tmp_path: Path, anchor: str
):
    store = _document_store(tmp_path, _WRAPPED)
    result = store.update_document("documents/intro.md", anchor, "X")
    assert result["status"] == "updated"


def test_document_update_does_not_treat_bibtex_fields_as_a_wrapped_item(tmp_path: Path):
    text = "@article{varga2026review,\n  author  = {Varga, Elena},\n  title   = {Review},\n}\n"
    store = _document_store(tmp_path, text)
    result = store.update_document(
        "documents/intro.md", "  author  = {Varga, Elena},", "  author  = {Varga, E.},"
    )
    assert result["status"] == "updated"


def test_document_update_accepts_a_heading_plus_its_first_content_line(tmp_path: Path):
    store = _document_store(tmp_path, "## Raw entry\n\nBraithwaite, unformatted.\n")
    state = tmp_path / "state"

    result = store.update_document(
        "documents/intro.md",
        "## Raw entry\n\nBraithwaite, unformatted.",
        "## Formatted entry\n\nBraithwaite, converted.",
    )

    assert result["status"] == "updated"
    assert (state / "documents" / "intro.md").read_text(encoding="utf-8") == (
        "## Formatted entry\n\nBraithwaite, converted.\n"
    )


def test_a_trailing_hash_or_hashes_without_a_heading_are_ordinary_anchors(
    tmp_path: Path,
):
    """Only a real heading line is refused; `#` inside prose is not one."""
    store = _document_store(tmp_path, "Issue #4 is open.\n")

    result = store.update_document("documents/intro.md", "#4", "#5")

    assert result["status"] == "updated"


def test_a_fallback_tier_match_tells_the_pa_which_span_it_edited(tmp_path: Path):
    """`match_tier` alone was ignored live (regress13 acc_003, two `unescaped`
    rescues), so a non-exact rung now carries a note the PA can act on."""
    store = _document_store(tmp_path, '# Notes\n\nHe said "yes" to the venue.\n')

    unescaped = store.update_document(
        "documents/intro.md",
        'He said \\"yes\\" to the venue.',
        "He said yes to the venue.",
    )

    assert unescaped["match_tier"] == "unescaped"
    assert unescaped["note"].startswith("anchor matched after undoing")

    whitespace = store.update_document(
        "documents/intro.md", "He said  yes  to the venue.", "He confirmed."
    )

    assert whitespace["match_tier"] == "whitespace"
    assert whitespace["note"].startswith("anchor matched only after loosening")

    exact = store.update_document("documents/intro.md", "He confirmed.", "Confirmed.")

    assert exact["match_tier"] == "exact"
    assert "note" not in exact


# The anchor match ladder. Live, 27 `document_update` calls failed with "anchor
# not found" and every one was a transcription artifact of the JSON-encoded
# `document_read` result the PA reads: JSON escapes copied literally, a run of
# newlines miscounted, a hard-wrapped line collapsed to one space.


def test_a_json_escaped_quote_in_the_anchor_matches_the_plain_quote(tmp_path: Path):
    store = _document_store(tmp_path, '# Notes\n\nHe said "yes" to the venue.\n')
    state = tmp_path / "state"

    result = store.update_document(
        "documents/intro.md",
        'He said \\"yes\\" to the venue.',
        "He confirmed the venue.",
    )

    assert result["match_tier"] == "unescaped"
    assert (state / "documents" / "intro.md").read_text(encoding="utf-8") == (
        "# Notes\n\nHe confirmed the venue.\n"
    )


def test_a_miscounted_run_of_newlines_still_matches_the_document_span(tmp_path: Path):
    store = _document_store(tmp_path, "Line A" + "\n" * 8 + "Line B\n")
    state = tmp_path / "state"

    result = store.update_document(
        "documents/intro.md", "Line A" + "\n" * 9 + "Line B", "Line C"
    )

    assert result["match_tier"] == "whitespace"
    # The document's own eight newlines were the span replaced, not the nine the
    # anchor claimed.
    assert (state / "documents" / "intro.md").read_text(encoding="utf-8") == "Line C\n"


def test_a_wrapped_line_collapsed_to_a_space_still_matches(tmp_path: Path):
    store = _document_store(
        tmp_path, "The quarterly report covers\nrevenue and headcount.\n"
    )
    state = tmp_path / "state"

    result = store.update_document(
        "documents/intro.md",
        "The quarterly report covers revenue and headcount.",
        "The report covers revenue only.",
    )

    assert result["match_tier"] == "whitespace"
    assert (state / "documents" / "intro.md").read_text(encoding="utf-8") == (
        "The report covers revenue only.\n"
    )


def test_a_lenient_anchor_matching_two_spans_is_still_ambiguous(tmp_path: Path):
    store = _document_store(tmp_path, "Item one\nItem one\n")
    with pytest.raises(ValueError, match="occurs 2 times"):
        store.update_document("documents/intro.md", "Item  one", "Item two")


def test_the_not_found_error_never_shows_the_anchor_as_an_escaped_repr(tmp_path: Path):
    """The old error echoed `{anchor!r}`, so a multi-line anchor came back with
    literal `\\n` in it and taught the PA to send escapes on the retry."""
    store = _document_store(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        store.update_document("documents/intro.md", "first line\nsecond line", "final")

    message = str(excinfo.value)
    assert "\\" not in message
    assert "22 chars" in message
    assert "starting: first line" in message


def test_the_tool_log_records_which_match_tier_fired(tmp_path: Path):
    store = _document_store(tmp_path, "The quarterly report covers\nrevenue.\n")
    state = tmp_path / "state"

    store.update_document(
        "documents/intro.md", "The quarterly report covers revenue.", "Rewritten."
    )

    entries = [
        json.loads(line)
        for line in (state / "tool_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    updates = [entry for entry in entries if entry["tool"] == "document_update"]
    assert [entry["result_summary"]["match_tier"] for entry in updates] == ["whitespace"]


def test_document_create_writes_a_new_document(tmp_path: Path):
    store = _document_store(tmp_path)
    state = tmp_path / "state"

    result = store.create_document("documents/new_note.md", "fresh text")

    assert result == {
        "path": "documents/new_note.md",
        "status": "created",
        "bytes": len("fresh text"),
    }
    assert (state / "documents" / "new_note.md").read_text(encoding="utf-8") == "fresh text"
    assert "Created document" in (state / "state_diff.jsonl").read_text(encoding="utf-8")


def test_document_create_on_an_existing_path_reports_exists_and_writes_nothing(
    tmp_path: Path,
):
    store = _document_store(tmp_path)
    state = tmp_path / "state"

    result = store.create_document("documents/intro.md", "clobbered")

    assert result["status"] == "exists"
    assert (state / "documents" / "intro.md").read_text(encoding="utf-8") == "draft text"
    assert not (state / "state_diff.jsonl").exists()


def test_document_delete_removes_an_existing_document(tmp_path: Path):
    store = _document_store(tmp_path)
    state = tmp_path / "state"

    result = store.delete_document("documents/intro.md")

    assert result == {"path": "documents/intro.md", "status": "deleted"}
    assert not (state / "documents" / "intro.md").exists()
    assert "Deleted document" in (state / "state_diff.jsonl").read_text(encoding="utf-8")


def test_document_delete_rejects_a_missing_document(tmp_path: Path):
    store = _document_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.delete_document("documents/absent.md")


def test_email_message_send_records_external_action(tmp_path: Path):
    state = tmp_path / "state"
    (state / "email").mkdir(parents=True)
    store = StateStore(state, run_id="run_001", user_id="user_a", session_id="session_09")

    result = store.send_email(
        to="marcus@example.test",
        subject="Train exhibition",
        body="Proposed logistics.",
    )
    assert result == {"sent_id": "sent_0001", "status": "sent"}

    sent = (state / "email" / "sent.jsonl").read_text(encoding="utf-8")
    assert "Proposed logistics." in sent

    tool_log = (state / "tool_log.jsonl").read_text(encoding="utf-8")
    assert "email_message_send" in tool_log

    state_diff = (state / "state_diff.jsonl").read_text(encoding="utf-8")
    assert "email.sent" in state_diff


def test_document_read_rejects_path_escape(tmp_path: Path):
    store = StateStore(tmp_path / "state", run_id="run", user_id="user", session_id="session")

    try:
        store.read_document("../outside.md")
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("path escape was allowed")


def test_scripted_email_search_and_route_options(tmp_path: Path):
    state = tmp_path / "state"
    (state / "email").mkdir(parents=True)
    (state / "maps").mkdir()
    (state / "email" / "inbox.jsonl").write_text(
        '{"id":"email_0001","from":"events@example.test","date":"2026-05-05",'
        '"subject":"Train Exhibition","body":"The exhibition is at Glenmont Civic Exhibition Hall."}\n',
        encoding="utf-8",
    )
    (state / "maps" / "route_options.json").write_text(
        '{"origin":"Home","stop":"Comic Store","destination":"Exhibition Hall",'
        '"options":[{"mode":"transit","duration_minutes":52,"estimated_price_usd":4.5}]}',
        encoding="utf-8",
    )

    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_04")

    email = store.search_email("train")
    assert email["query"] == "train"
    assert email["matches"][0]["subject"] == "Train Exhibition"

    full_email = store.read_email("email_0001")
    assert full_email["body"] == "The exhibition is at Glenmont Civic Exhibition Hall."

    routes = store.route_options("Home", "Exhibition Hall", stop="Comic Store")
    assert routes["options"][0]["mode"] == "transit"
    assert routes["options"][0]["estimated_price_usd"] == 4.5

    tool_log = (state / "tool_log.jsonl").read_text(encoding="utf-8")
    assert "email_message_search" in tool_log
    assert "email_message_read" in tool_log
    assert "map_route_search" in tool_log


def test_route_options_selects_matching_route_from_multi_route_fixture(tmp_path: Path):
    state = tmp_path / "state"
    (state / "maps").mkdir(parents=True)
    (state / "maps" / "route_options.json").write_text(
        '{"routes":['
        '{"origin":"Home","stop":"Comic Store","destination":"Exhibition Hall",'
        '"options":[{"duration_minutes":34}]},'
        '{"origin":"Home","stop":"Marcus pickup","destination":"Cinema",'
        '"options":[{"duration_minutes":15},{"duration_minutes":20}]}'
        ']}',
        encoding="utf-8",
    )
    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_05")

    routes = store.route_options("Home", "Cinema", stop="Marcus pickup")

    assert [option["duration_minutes"] for option in routes["options"]] == [15, 20]


def _home_and_cinema_state(tmp_path: Path) -> Path:
    """Fixture shaped like user_a: a home origin and a place sharing its town name."""
    state = tmp_path / "state"
    (state / "maps").mkdir(parents=True)
    (state / "shipping").mkdir()
    (state / "shipping" / "addresses.json").write_text(
        '{"addresses":[{"address_id":"addr_home","label":"Apartment",'
        '"street":"Glenmont Heights, 12 Corvid Ave, Apt 4B","city":"Glenmont",'
        '"state":"CA","zip":"93710","default":true}]}',
        encoding="utf-8",
    )
    (state / "maps" / "route_options.json").write_text(
        '{"routes":['
        '{"origin":"Glenmont Heights apartment","stop":"","destination":"Casa Marisol",'
        '"options":[{"mode":"walk","duration_minutes":14}]},'
        '{"origin":"Glenmont Cinema","stop":"","destination":"Casa Marisol",'
        '"options":[{"mode":"walk","duration_minutes":8}]}'
        ']}',
        encoding="utf-8",
    )
    return state


@pytest.mark.parametrize(
    "origin",
    ["home", "my apartment", "Glenmont Heights, apartment 4B", "12 Corvid Ave, Glenmont"],
)
def test_route_options_resolves_home_endpoints_to_the_home_record(tmp_path: Path, origin: str):
    # regress14b acc_002: the address text shared one token with the home record
    # and one with "Glenmont Cinema", and the cinema won on the shorter name.
    store = StateStore(
        _home_and_cinema_state(tmp_path),
        run_id="run",
        user_id="user_a",
        session_id="session_06",
    )

    routes = store.route_options(origin, "Casa Marisol")

    assert [option["duration_minutes"] for option in routes["options"]] == [14]


def test_route_options_keeps_a_place_name_off_the_home_record(tmp_path: Path):
    store = StateStore(
        _home_and_cinema_state(tmp_path),
        run_id="run",
        user_id="user_a",
        session_id="session_07",
    )

    routes = store.route_options("Glenmont Cinema", "Casa Marisol")

    assert [option["duration_minutes"] for option in routes["options"]] == [8]


def test_endpoint_matches_rejects_a_single_shared_town_token():
    assert not StateStore._endpoint_matches("12 Corvid Ave, Glenmont", "Glenmont Cinema")
    assert not StateStore._endpoint_matches("Glenmont Cinema", "Glenmont Post Office")
    assert StateStore._endpoint_matches("the WIT campus gate", "WIT campus")


def test_weather_forecast_selects_matching_window(tmp_path: Path):
    state = tmp_path / "state"
    (state / "weather").mkdir(parents=True)
    (state / "weather" / "forecasts.json").write_text(
        '{"forecasts":[{"location":"Glenmont, CA","date":"2026-12-08",'
        '"time_window":"18:00-22:00","forecast":{"recommendation":"Bring an umbrella.",'
        '"regional_confidence":"high"}}]}',
        encoding="utf-8",
    )
    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_05")

    result = store.weather_forecast_read("Glenmont, CA", "2026-12-08", "18:00-22:00")

    assert result["forecast"]["recommendation"] == "Bring an umbrella."
    assert result["forecast"]["regional_confidence"] == "high"
    assert "weather_forecast_read" in (state / "tool_log.jsonl").read_text(encoding="utf-8")


def test_scripted_store_item_and_cinema_showtimes(tmp_path: Path):
    state = tmp_path / "state"
    (state / "comic_store").mkdir(parents=True)
    (state / "cinema").mkdir()
    (state / "places").mkdir()
    (state / "comic_store" / "items.json").write_text(
        '[{"store":"Dale\'s Comic Center","item":"The Flash Archive Volume 3",'
        '"status":"available_for_pickup"}]',
        encoding="utf-8",
    )
    (state / "cinema" / "showtimes.json").write_text(
        '[{"cinema":"Glenmont Cinema","movie":"The Time Machine",'
        '"date":"2026-05-07","time":"19:10"}]',
        encoding="utf-8",
    )
    (state / "places" / "nearby.json").write_text(
        '[{"name":"Civic Plaza Diner","near":"Glenmont Cinema",'
        '"category":"dinner","tags":["casual"]}]',
        encoding="utf-8",
    )

    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_06")

    item = store.comic_store_inventory_read("The Flash Archive Volume 3")
    assert item["matches"][0]["status"] == "available_for_pickup"

    showtimes = store.cinema_showtime_search(
        "Glenmont Cinema",
        "The Time Machine",
        "2026-05-07",
    )
    assert showtimes["matches"][0]["time"] == "19:10"

    places = store.place_search("Glenmont Cinema", "dinner")
    assert places["matches"][0]["name"] == "Civic Plaza Diner"

    tool_log = (state / "tool_log.jsonl").read_text(encoding="utf-8")
    assert "comic_store_inventory_read" in tool_log
    assert "cinema_showtime_search" in tool_log
    assert "place_search" in tool_log


def test_cinema_ticket_purchase_records_external_action(tmp_path: Path):
    state = tmp_path / "state"
    (state / "finance").mkdir(parents=True)
    (state / "finance" / "payment_methods.json").write_text(
        '{"payment_methods":[{"method_id":"pm_visa_4821","kind":"credit_card"}]}',
        encoding="utf-8",
    )
    (state / "cinema").mkdir()
    (state / "cinema" / "showtimes.json").write_text(
        '[{"cinema":"Glenmont Cinema","movie":"Temporal Echo",'
        '"date":"2026-07-21","time":"20:00"}]',
        encoding="utf-8",
    )
    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_039")

    result = store.buy_cinema_tickets(
        cinema="Glenmont Cinema",
        movie="Temporal Echo",
        date="2026-07-21",
        time="20:00",
        quantity=2,
        payment_method_id="pm_visa_4821",
    )

    assert result == {"ticket_id": "ticket_0001", "status": "purchased"}
    tickets = (state / "cinema" / "tickets.jsonl").read_text(encoding="utf-8")
    assert "Temporal Echo" in tickets

    ticket = store.cinema_ticket_read("ticket_0001")
    assert ticket["ticket"]["quantity"] == 2

    tool_log = (state / "tool_log.jsonl").read_text(encoding="utf-8")
    assert "cinema_ticket_purchase" in tool_log

    state_diff = (state / "state_diff.jsonl").read_text(encoding="utf-8")
    assert "cinema.tickets" in state_diff

    with pytest.raises(ValueError, match="unknown payment method"):
        store.buy_cinema_tickets(
            cinema="Glenmont Cinema",
            movie="Temporal Echo",
            date="2026-07-21",
            time="20:00",
            quantity=1,
            payment_method_id="pm_missing",
        )


def test_contact_search_returns_handoff_channel(tmp_path: Path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "contacts.json").write_text(
        '{"accessibility":{"name":"Glenmont Civic Exhibition Hall Accessibility Desk",'
        '"email":"accessibility@glenmontcivic.example",'
        '"tags":["quiet-entry","accommodation"]}}',
        encoding="utf-8",
    )

    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_22")
    result = store.lookup_contacts("quiet-entry")

    assert result["matches"][0]["email"] == "accessibility@glenmontcivic.example"
    tool_log = (state / "tool_log.jsonl").read_text(encoding="utf-8")
    assert "contact_search" in tool_log


def test_contact_search_matches_long_queries_by_tokens(tmp_path: Path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "contacts.json").write_text(
        '{"accessibility":{"name":"Glenmont Civic Exhibition Hall Accessibility Desk",'
        '"email":"accessibility@glenmontcivic.example",'
        '"tags":["quiet-entry","accommodation","low-crowd","train exhibition"]}}',
        encoding="utf-8",
    )

    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_22")
    result = store.lookup_contacts(
        "Glenmont Civic Exhibition Hall accessibility quiet entry low crowd accommodations"
    )

    assert result["matches"][0]["name"] == "Glenmont Civic Exhibition Hall Accessibility Desk"


def test_document_read_named_directory_returns_file_listing(tmp_path: Path):
    state = tmp_path / "state"
    (state / "my_desktop").mkdir(parents=True)
    (state / "my_desktop" / "note.md").write_text("hello", encoding="utf-8")

    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_22")
    result = store.read_document("my_desktop")

    assert result["type"] == "directory"
    assert "my_desktop/note.md" in result["files"]


def test_document_read_rejects_internal_authoring_files(
    tmp_path: Path,
):
    state = tmp_path / "state"
    internal = state / "authoring" / "acc_001" / "fixture.yaml"
    internal.parent.mkdir(parents=True)
    internal.write_text("private authoring facts", encoding="utf-8")

    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_22")

    with pytest.raises(PermissionError, match="not user-visible"):
        store.read_document("authoring/acc_001/fixture.yaml")


def test_document_read_top_level_lists_only_user_visible_folders(tmp_path: Path):
    state = tmp_path / "state"
    (state / "documents" / "research").mkdir(parents=True)
    (state / "documents" / "research" / "citation_notes.md").write_text("notes", encoding="utf-8")
    (state / "my_desktop").mkdir(parents=True)
    (state / "authoring").mkdir(parents=True)
    (state / "memory").mkdir(parents=True)

    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_22")

    for raw in ("", ".", "/", "./"):
        result = store.read_document(raw)
        assert result["type"] == "directory"
        assert result["path"] == "."
        assert result["files"] == ["documents/", "my_desktop/"]

    # The listed folders are readable, and everything else stays rejected.
    assert "documents/research/citation_notes.md" in store.read_document("documents")["files"]
    with pytest.raises(PermissionError, match="not user-visible"):
        store.read_document("authoring")
    with pytest.raises(FileNotFoundError):
        store.read_document("memory/MEMORY.md")
    with pytest.raises(ValueError, match="absolute"):
        store.read_document("/etc/passwd")
    with pytest.raises(ValueError, match="escapes"):
        store.read_document("../outside.md")


def test_reads_reflect_real_writes_without_scripted_layer(tmp_path: Path):
    state = tmp_path / "state"
    store = StateStore(state, run_id="run", user_id="user_a", session_id="session_01")

    store.create_document("documents/draft.md", "before")
    assert store.read_document("documents/draft.md")["content"] == "before"

    store.update_document("documents/draft.md", "before", "after")
    assert store.read_document("documents/draft.md")["content"] == "after"


def test_a_tool_result_never_carries_the_run_bookkeeping(monkeypatch, tmp_path: Path):
    """The stamp belongs on disk and in the tool log, never back in the prompt.

    Live (`v2g03`), the PA created an appointment and read it back, and the tool
    message in its own session file ended `"run_id": "v2g03", "session_id":
    "acc_039"`. A descriptive run id would have named the setting under test.
    """
    state = tmp_path / "state"
    availability = state / "appointment"
    availability.mkdir(parents=True)
    (availability / "availability.json").write_text(
        json.dumps({"providers": [{
            "provider_id": "clinic_1",
            "name": "Glenmont Family Clinic",
            "location": "Glenmont",
            "slots": [{"date": "2026-11-20", "time": "10:00"}],
        }]}),
        encoding="utf-8",
    )
    for key, value in (
        ("MEMPA_STATE_ROOT", str(state)),
        ("MEMPA_RUN_ID", "v2g03"),
        ("MEMPA_USER_ID", "user_a"),
        ("MEMPA_SESSION_ID", "acc_039"),
    ):
        monkeypatch.setenv(key, value)
    import state_server.server as server

    created = server.appointment_create(
        provider_id="clinic_1", date="2026-11-20", time="10:00", patient="the user",
        reason="routine checkup",
    )
    result = server.appointment_read(appointment_id=created["appointment_id"])

    assert result["appointment"]["date"] == "2026-11-20"
    for banned in ("run_id", "user_id", "session_id"):
        assert banned not in result["appointment"]
        assert f'"{banned}"' not in json.dumps(result)
    # …and the record on disk still carries it, because that is what attributes
    # a written row to a run.
    on_disk = json.loads(
        (state / "appointment" / "appointments.jsonl").read_text(encoding="utf-8")
        .strip().splitlines()[-1]
    )
    assert on_disk["run_id"] == "v2g03"
    assert "v2g03" in (state / "tool_log.jsonl").read_text(encoding="utf-8")


def test_a_tool_result_never_carries_the_world_visibility_bookkeeping(
    monkeypatch, tmp_path: Path
):
    """`visible_from` is how the fixture hides a record before its story date.

    It is the store's own bookkeeping (`_date_visible`), it says out loud that
    the world is staged, and it reached the model: `restaurant_reservation_read`
    handed back `reservation.visible_from` on every read.
    """
    state = tmp_path / "state"
    (state / "restaurant").mkdir(parents=True)
    (state / "restaurant" / "reservations.jsonl").write_text(
        json.dumps({
            "reservation_id": "res_0001",
            "restaurant_name": "Siam Garden",
            "date": "2026-10-13",
            "time": "17:30",
            "status": "confirmed",
            "visible_from": "2026-10-03",
        }) + "\n",
        encoding="utf-8",
    )
    for key, value in (
        ("MEMPA_STATE_ROOT", str(state)),
        ("MEMPA_RUN_ID", "v3g07"),
        ("MEMPA_USER_ID", "user_a"),
        ("MEMPA_SESSION_ID", "acc_026"),
    ):
        monkeypatch.setenv(key, value)
    import state_server.server as server

    result = server.restaurant_reservation_read(reservation_id="res_0001")

    assert result["reservation"]["restaurant_name"] == "Siam Garden"
    assert "visible_from" not in result["reservation"]
    assert "visible_from" not in json.dumps(result)
    # The filter still reads the full record off disk.
    on_disk = json.loads(
        (state / "restaurant" / "reservations.jsonl")
        .read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert on_disk["visible_from"] == "2026-10-03"


def _places_world(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    (state / "places").mkdir(parents=True)
    (state / "places" / "nearby.json").write_text(
        json.dumps({
            "places": [
                {
                    "name": "Glenmont Pharmacy",
                    "near": "Alder St",
                    "category": "pharmacy",
                    "tags": ["late-hours"],
                },
                {
                    "name": "Civic Plaza Diner",
                    "near": "Civic Plaza",
                    "category": "restaurant",
                    "tags": ["dinner"],
                },
            ]
        }),
        encoding="utf-8",
    )
    return state


def test_place_search_finds_a_listing_by_its_own_name(tmp_path: Path):
    """The store is name-keyed, so the PA has to be able to pass the name.

    Every outline ingredient over places/nearby.json cites its record by name;
    before `query` the only way in was guessing the street and the category.
    """
    store = StateStore(
        _places_world(tmp_path), run_id="run", user_id="user_a", session_id="s"
    )

    hit = store.place_search(query="Glenmont Pharmacy")
    assert [record["name"] for record in hit["matches"]] == ["Glenmont Pharmacy"]

    # A name the world does not list is an empty result, not an error: the PA
    # asked about a place, not about a closed vocabulary.
    assert store.place_search(query="Marisol Bakery")["matches"] == []

    # query combines with the other two selectors rather than replacing them.
    assert store.place_search(
        location="Alder St", category="pharmacy", query="Glenmont"
    )["matches"][0]["name"] == "Glenmont Pharmacy"
    assert store.place_search(location="Civic Plaza", query="Glenmont")["matches"] == []


def test_place_search_rejects_an_unknown_category_instead_of_returning_nothing(
    tmp_path: Path,
):
    """`category` was the silent-empty path left on `location`'s sibling."""
    store = StateStore(
        _places_world(tmp_path), run_id="run", user_id="user_a", session_id="s"
    )

    with pytest.raises(ValueError) as excinfo:
        store.place_search(location="Alder St", category="bowling")
    message = str(excinfo.value)
    assert "category='bowling'" in message
    assert "pharmacy" in message

    with pytest.raises(ValueError, match="at least one of location, category or query"):
        store.place_search()


def test_place_search_still_selects_by_street_and_category(tmp_path: Path):
    store = StateStore(
        _places_world(tmp_path), run_id="run", user_id="user_a", session_id="s"
    )

    result = store.place_search("Civic Plaza", "dinner")
    assert [record["name"] for record in result["matches"]] == ["Civic Plaza Diner"]
    assert result["query"] is None


def test_utility_bill_read_finds_a_bill_by_the_provider_on_it(tmp_path: Path):
    """The description named `provider`; the handler only matched `account`.

    Four bills across user_b/c/d have a provider that is not a substring of the
    account string, so the name printed on the bill raised not_found. These are
    those four.
    """
    state = tmp_path / "state"
    (state / "finance").mkdir(parents=True)
    (state / "finance" / "bills.json").write_text(
        json.dumps({
            "bills": [
                {
                    "bill_id": "bill_0001",
                    "provider": "Fairhaven Renters Insurance",
                    "account": "Fairhaven Renters — policy RN-4471902",
                    "period": "2026-10",
                },
                {
                    "bill_id": "bill_0002",
                    "provider": "Corvid Court Homeowners Association",
                    "account": "Corvid Court HOA — account 12-CC",
                    "period": "2026-10",
                },
                {
                    "bill_id": "bill_0003",
                    "provider": "Corvid Mutual",
                    "account": "Renters insurance — policy RN-4471-09",
                    "period": "2026-10",
                },
                {
                    "bill_id": "bill_0004",
                    "provider": "Glenmont Credit Union",
                    "account": "Visa ****3390",
                    "period": "2026-10",
                },
                {
                    "bill_id": "bill_0005",
                    "provider": "Glenmont Credit Union",
                    "account": "Visa ****3390",
                    "period": "2026-09",
                },
            ]
        }),
        encoding="utf-8",
    )
    store = StateStore(state, run_id="run", user_id="user_a", session_id="s")

    for provider, bill_id in (
        ("Fairhaven Renters Insurance", "bill_0001"),
        ("Corvid Court Homeowners Association", "bill_0002"),
        ("Corvid Mutual", "bill_0003"),
        ("Glenmont Credit Union", "bill_0004"),
    ):
        assert store.utility_bill_read(provider)["bills"][0]["bill_id"] == bill_id

    # The account string still resolves, and the latest period is still what a
    # call without one returns.
    assert [
        bill["bill_id"] for bill in store.utility_bill_read("Visa ****3390")["bills"]
    ] == ["bill_0004", "bill_0005"]
    assert [
        bill["bill_id"]
        for bill in store.utility_bill_read("Glenmont Credit Union", "2026-09")["bills"]
    ] == ["bill_0005"]

    with pytest.raises(FileNotFoundError):
        store.utility_bill_read("Glenmont Power")


def test_exhibition_search_reads_the_store_the_purchase_validates_against(
    tmp_path: Path,
):
    """exhibition/exhibitions.json had no read tool at all.

    The purchase checks venue, exhibit and date against this file, and the PA
    was pointed at the place listing, which carries hours but not the open days.
    """
    state = tmp_path / "state"
    (state / "finance").mkdir(parents=True)
    (state / "finance" / "payment_methods.json").write_text(
        '{"payment_methods":[{"method_id":"pm_visa_4821","kind":"credit_card"}]}',
        encoding="utf-8",
    )
    (state / "exhibition").mkdir(parents=True)
    (state / "exhibition" / "exhibitions.json").write_text(
        json.dumps({
            "exhibitions": [
                {
                    "name": "Autumn Model Train Exhibition",
                    "venue": "Glenmont Civic Exhibition Hall",
                    "note": "closed Mondays",
                    "slots": [{"date": "2026-10-03"}, {"date": "2026-10-04"}],
                },
                {
                    "name": "Coastal Watercolours",
                    "venue": "Rowan Park Gallery",
                    "slots": [{"date": "2026-10-05"}],
                },
            ]
        }),
        encoding="utf-8",
    )
    store = StateStore(state, run_id="run", user_id="user_a", session_id="s")

    by_name = store.exhibition_search(query="model train")
    assert [record["name"] for record in by_name["matches"]] == [
        "Autumn Model Train Exhibition"
    ]
    match = by_name["matches"][0]
    # The fields the purchase validates on come back, so a result feeds it.
    assert match["venue"] == "Glenmont Civic Exhibition Hall"
    assert match["open_dates"] == ["2026-10-03", "2026-10-04"]
    assert match["note"] == "closed Mondays"
    store.exhibition_ticket_purchase(
        venue=match["venue"],
        exhibit=match["name"],
        date=match["open_dates"][0],
        quantity=1,
        payment_method_id="pm_visa_4821",
    )

    by_date = store.exhibition_search(date="2026-10-05")
    assert [record["name"] for record in by_date["matches"]] == ["Coastal Watercolours"]

    assert store.exhibition_search(date="2026-10-06")["matches"] == []
    assert store.exhibition_search(query="sculpture")["matches"] == []
    assert len(store.exhibition_search()["matches"]) == 2

    with pytest.raises(ValueError, match=r"date='Oct 5' is not YYYY-MM-DD"):
        store.exhibition_search(date="Oct 5")


def test_a_weather_forecasts_visibility_field_is_data_and_survives_the_seam(
    monkeypatch, tmp_path: Path
):
    """`visibility` is a real forecast field, not world bookkeeping.

    The strip used to delete any key called `visibility` at any depth, so
    user_d's `forecast.visibility` -- "poor before 08:00 on the Route 41 spur",
    the one thing a fog-drive forecast is read for -- never reached the PA. The
    store's own filter is `visible_from` (`_date_visible`); nothing reads or
    writes `visibility`. The real markers still come off in the same result.
    """
    state = tmp_path / "state"
    (state / "weather").mkdir(parents=True)
    (state / "weather" / "forecasts.json").write_text(
        json.dumps({"forecasts": [{
            "location": "Bitterfield, CA",
            "date": "2026-11-06",
            "time_window": "06:00-10:00",
            "visible_from": "2026-10-01",
            "forecast": {
                "summary": "Fog on the valley floor until mid-morning",
                "visibility": "poor before 08:00 on the Route 41 spur",
                "run_id": "not a real field, but deeper than a record level",
            },
        }]}),
        encoding="utf-8",
    )
    for key, value in (
        ("MEMPA_STATE_ROOT", str(state)),
        ("MEMPA_RUN_ID", "v3g07"),
        ("MEMPA_USER_ID", "user_d"),
        ("MEMPA_SESSION_ID", "acc_001"),
    ):
        monkeypatch.setenv(key, value)
    import state_server.server as server

    result = server.weather_forecast_read(
        location="Bitterfield, CA", date="2026-11-06", time_window="06:00-10:00",
    )

    assert result["forecast"]["visibility"] == (
        "poor before 08:00 on the Route 41 spur"
    )
    # The record's own `visible_from` is bookkeeping and never leaves.
    assert "visible_from" not in json.dumps(result)
    # Record level is where a marker is stripped, and only there.
    from state_server.store import strip_run_identity

    assert strip_run_identity(
        {"booking": {"run_id": "v3g07", "notes": {"run_id": "a guest's own note"}}}
    ) == {"booking": {"notes": {"run_id": "a guest's own note"}}}


def _bills_world(tmp_path: Path, bills: list[dict]) -> Path:
    state = tmp_path / "state"
    (state / "finance").mkdir(parents=True)
    (state / "finance" / "bills.json").write_text(
        json.dumps({"bills": bills}), encoding="utf-8"
    )
    return state


def test_utility_bill_read_returns_every_matching_bill(tmp_path: Path):
    """A card statement and its minimum share account, provider and period.

    The handler sorted by period and took `bills[0]`, so the minimum-due record
    was unreachable: user_b's acc_080 wants the $42.00 minimum and the PA got
    the $1418.60 statement balance, with nothing in the result to say a second
    bill existed. Every match is returned now.
    """
    state = _bills_world(
        tmp_path,
        [
            {
                "bill_id": "CC-2026-12",
                "account": "Glenmont Credit Union Visa — ****9064",
                "provider": "Glenmont Credit Union",
                "period": "2026-12",
                "total": 1418.60,
                "notes": "Full statement balance.",
            },
            {
                "bill_id": "CC-2026-12-MIN",
                "account": "Glenmont Credit Union Visa — ****9064",
                "provider": "Glenmont Credit Union",
                "period": "2026-12",
                "total": 42.00,
                "notes": "The minimum on statement CC-2026-12.",
            },
            {
                "bill_id": "GP-B-2026-11",
                "account": "Glenmont Power — acct 204-71-8836",
                "provider": "Glenmont Power",
                "period": "2026-11",
                "total": 58.25,
            },
        ],
    )
    store = StateStore(state, run_id="run", user_id="user_b", session_id="s")

    result = store.utility_bill_read("Glenmont Credit Union", "2026-12")
    assert [bill["bill_id"] for bill in result["bills"]] == [
        "CC-2026-12",
        "CC-2026-12-MIN",
    ]
    # The fields that tell the two apart come back with them.
    assert [bill["total"] for bill in result["bills"]] == [1418.60, 42.00]
    assert all("notes" in bill for bill in result["bills"])

    # A single match is still a one-element list, not a bare record.
    single = store.utility_bill_read("Glenmont Power")
    assert [bill["bill_id"] for bill in single["bills"]] == ["GP-B-2026-11"]


def test_utility_bill_read_returns_three_accounts_with_one_provider(tmp_path: Path):
    """user_c holds three Trade Press Group subscriptions billed the same month.

    Picking the first left two invisible, and the three differ only by the
    subscriber id on the account string and the amount.
    """
    state = _bills_world(
        tmp_path,
        [
            {
                "bill_id": "TPG-2027-A",
                "account": "Trade Press Group — subscriber 4471-A",
                "provider": "Trade Press Group",
                "period": "2027-01",
                "total": 69.00,
            },
            {
                "bill_id": "TPG-2027-B",
                "account": "Trade Press Group — subscriber 4471-B",
                "provider": "Trade Press Group",
                "period": "2027-01",
                "total": 54.00,
            },
            {
                "bill_id": "TPG-2027-C",
                "account": "Trade Press Group — subscriber 4471-C",
                "provider": "Trade Press Group",
                "period": "2027-01",
                "total": 148.00,
            },
        ],
    )
    store = StateStore(state, run_id="run", user_id="user_c", session_id="s")

    result = store.utility_bill_read("Trade Press Group", "2027-01")
    assert sorted(bill["bill_id"] for bill in result["bills"]) == [
        "TPG-2027-A",
        "TPG-2027-B",
        "TPG-2027-C",
    ]
    assert sorted(bill["account"] for bill in result["bills"]) == [
        "Trade Press Group — subscriber 4471-A",
        "Trade Press Group — subscriber 4471-B",
        "Trade Press Group — subscriber 4471-C",
    ]


def test_shopping_order_read_does_not_echo_the_delivery_address_id(tmp_path: Path):
    """Saved address ids are user-supplied by design (state_server/server.py).

    No tool lists or derives them; only the account holder knows one, which is
    what makes an address id usable as a gate. Echoing it back on every order
    read handed the PA a value it was supposed to have to ask for -- one order
    read and the gate is defeated for every later order. It is stripped the way
    `placed_at` already was.
    """
    state = tmp_path / "state"
    (state / "shopping").mkdir(parents=True)
    (state / "shopping" / "order_store.json").write_text(
        json.dumps({
            "orders": [
                {
                    "order_id": "GM-0001",
                    "product_id": "shp_umb_compact",
                    "product_name": "Compact Umbrella",
                    "quantity": 1,
                    "delivery_method": "home_delivery",
                    "delivery_address_id": "addr_k4w8pv",
                    "payment_method_id": "pm_visa_4821",
                    "status": "placed",
                    "placed_at": "2026-11-14T09:12:00",
                }
            ]
        }),
        encoding="utf-8",
    )
    store = StateStore(state, run_id="run", user_id="user_b", session_id="s")

    order = store.shopping_order_read("GM-0001")["order"]

    assert "delivery_address_id" not in order
    assert "placed_at" not in order
    # Everything the PA legitimately needs is still there.
    assert order["order_id"] == "GM-0001"
    assert order["status"] == "placed"
    assert order["delivery_method"] == "home_delivery"


def _return_case_store(tmp_path: Path) -> tuple[StateStore, Path]:
    state = tmp_path / "state"
    (state / "shopping").mkdir(parents=True)
    (state / "email").mkdir()
    case_path = state / "shopping" / "return_cases.jsonl"
    case_path.write_text(
        json.dumps({
            "case_id": "rcase_0001",
            "retailer": "GlenMart",
            "order_id": "GM-126104",
            "status": "open_on_hold",
            "route": None,
            "filing": None,
            "note_draft_id": None,
            "updated_at": None,
        }) + "\n",
        encoding="utf-8",
    )
    (state / "email" / "drafts.jsonl").write_text(
        json.dumps({
            "draft_id": "draft_0001",
            "to": "GlenMart returns",
            "subject": "Return Case Details - Order GM-126104",
            "body": "Prepared note",
        }) + "\n",
        encoding="utf-8",
    )
    return StateStore(state, run_id="run", user_id="user_c", session_id="s"), case_path


def test_shopping_return_case_update_persists_and_reads_back(tmp_path: Path):
    store, case_path = _return_case_store(tmp_path)
    untouched = {
        "case_id": "rcase_0002",
        "retailer": "Northstar",
        "order_id": "NS-2",
        "status": "open_on_hold",
        "route": None,
        "filing": None,
        "note_draft_id": None,
        "updated_at": None,
    }
    with case_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(untouched) + "\n")

    updated = store.shopping_return_case_update(
        "rcase_0001",
        "the returns desk, not the shop floor on the plaza",
        "the case already open on the order, not a new one raised from scratch",
        "draft_0001",
    )["case"]
    readback = store.shopping_return_case_read("rcase_0001")["case"]

    assert updated == readback
    assert readback["retailer"] == "GlenMart"
    assert readback["order_id"] == "GM-126104"
    assert readback["status"] == "open_on_hold"
    assert readback["note_draft_id"] == "draft_0001"
    assert readback["updated_at"]
    records = [json.loads(line) for line in case_path.read_text().splitlines()]
    assert records[1] == untouched


def test_shopping_return_case_update_replace_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store, case_path = _return_case_store(tmp_path)
    before = case_path.read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("state_server.store.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        store.shopping_return_case_update(
            "rcase_0001", "returns desk", "existing case", "draft_0001"
        )

    assert case_path.read_bytes() == before
    assert list(case_path.parent.glob(f".{case_path.name}.*.tmp")) == []


@pytest.mark.parametrize(
    "case_id,route,filing,draft_id",
    [
        ("rcase_9999", "returns desk", "existing case", "draft_0001"),
        ("rcase_0001", "", "existing case", "draft_0001"),
        ("rcase_0001", "returns desk", " ", "draft_0001"),
        ("rcase_0001", "returns desk", "existing case", "draft_9999"),
    ],
)
def test_shopping_return_case_update_rejections_leave_store_byte_identical(
    tmp_path: Path,
    case_id: str,
    route: str,
    filing: str,
    draft_id: str,
):
    store, case_path = _return_case_store(tmp_path)
    before = case_path.read_bytes()

    with pytest.raises(ValueError):
        store.shopping_return_case_update(case_id, route, filing, draft_id)

    assert case_path.read_bytes() == before


def test_shopping_return_case_update_invalid_frozen_gate_fails_before_write(
    tmp_path: Path,
):
    store, case_path = _return_case_store(tmp_path)
    store.gate = {
        "tool": "shopping_return_case_update",
        "param": "route",
        "key_terms": ["returns desk"],
    }
    before = case_path.read_bytes()

    with pytest.raises(ValueError, match="must be the user's own words"):
        store.shopping_return_case_update(
            "rcase_0001", "returns desk", "existing case", "draft_0001"
        )

    assert case_path.read_bytes() == before

    (store.state_root / GATE_OPEN_MARKER).touch()
    with pytest.raises(ValueError, match="gate misconfiguration"):
        store.shopping_return_case_update(
            "rcase_0001", "returns desk", "existing case", "draft_0001"
        )

    assert case_path.read_bytes() == before


def test_shopping_return_case_update_rejects_closed_case_without_mutation(tmp_path: Path):
    store, case_path = _return_case_store(tmp_path)
    record = json.loads(case_path.read_text(encoding="utf-8"))
    record["status"] = "closed"
    case_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    before = case_path.read_bytes()

    with pytest.raises(ValueError, match="cannot be changed"):
        store.shopping_return_case_update(
            "rcase_0001", "returns desk", "existing case", "draft_0001"
        )

    assert case_path.read_bytes() == before


# A search row names the records it handed back
#
# Owner ruling 2026-09-16: `{"matches": 1}` cannot be attributed to anything.
# Live acc_096's row was `contact_search {"query": "Sarah Bennett"} ->
# {"matches": 1}`, so nothing in the finished run said the record was
# `ct_0012`, and the run-level exposure scan could not report that fin_28's
# probe object had already been handed to the timeline PA.

CONTACT_ROWS = [
    {"contact_id": "ct_0012", "name": "Sarah Bennett", "relation": "stage manager"},
    {"contact_id": "ct_0023", "name": "Fifth Street Studio office"},
]


def _search_store(tmp_path: Path) -> StateStore:
    state = tmp_path / "state"
    for relative, payload in (
        ("contacts.json", {"contacts": CONTACT_ROWS}),
        (
            "calendar/events.json",
            {"events": [{"event_id": "ev_0007", "title": "Studio B hold", "date": "2026-11-04"}]},
        ),
        (
            "shopping/products.json",
            {
                "currency": "USD",
                "products": [
                    {
                        "product_id": "prod_0004",
                        "item_type": "lamp",
                        "name": "Desk lamp",
                        "price": 32.0,
                    }
                ],
            },
        ),
    ):
        path = state / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    inbox = state / "email" / "inbox.jsonl"
    inbox.parent.mkdir(parents=True, exist_ok=True)
    inbox.write_text(
        json.dumps(
            {
                "id": "email_0003",
                "from": "office@example.test",
                "date": "2026-11-01",
                "subject": "Studio hold",
                "body": "Weekly slot confirmed.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return StateStore(state, run_id="run", user_id="user_b", session_id="session_31")


def _last_row(store: StateStore) -> dict:
    return json.loads(
        (store.state_root / "tool_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )


@pytest.mark.parametrize(
    "search,ids",
    [
        (lambda store: store.lookup_contacts("Bennett"), ["ct_0012"]),
        (lambda store: store.search_email("Studio"), ["email_0003"]),
        (lambda store: store.calendar_event_search("Studio"), ["ev_0007"]),
        (lambda store: store.shopping_product_search("lamp"), ["prod_0004"]),
    ],
)
def test_a_search_row_names_the_records_it_returned(tmp_path: Path, search, ids):
    store = _search_store(tmp_path)

    search(store)

    summary = _last_row(store)["result_summary"]
    assert summary["matches"] == len(ids)
    assert summary["ids"] == ids


def test_a_search_row_caps_the_ids_it_names(tmp_path: Path):
    """A wide search must not turn one log row into the whole address book."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "contacts.json").write_text(
        json.dumps(
            {
                "contacts": [
                    {"contact_id": f"ct_{index:04d}", "name": f"Rehearsal room {index}"}
                    for index in range(1, 26)
                ]
            }
        ),
        encoding="utf-8",
    )
    store = StateStore(state, run_id="run", user_id="user_b", session_id="session_32")

    found = store.lookup_contacts("rehearsal")

    assert len(found["matches"]) == 25
    summary = _last_row(store)["result_summary"]
    assert summary["matches"] == 25
    assert summary["ids"] == [f"ct_{index:04d}" for index in range(1, 21)]
    assert summary["truncated"] is True


def test_a_search_that_returned_no_record_with_an_id_names_none(tmp_path: Path):
    """Nothing matched, or a store whose records carry no id at all: there is
    no id to name, and the row stays the count it always was."""
    store = _search_store(tmp_path)

    store.lookup_contacts("zzzz nothing here zzzz")

    assert _last_row(store)["result_summary"] == {"matches": 0}
