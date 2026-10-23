"""Tests for the world-toolbox handlers: new domains, CRUD pairs, date filter."""

import json
from pathlib import Path

import pytest

from state_server.store import RECURRENCE_FORMAT, StateStore


def _store(tmp_path: Path, session_date: str | None = None) -> StateStore:
    return StateStore(
        tmp_path / "state",
        run_id="run",
        user_id="user_a",
        session_id="session_01",
        session_date=session_date,
    )


def test_full_session_timestamp_is_the_write_clock_and_keeps_future_rows_hidden(
    tmp_path: Path,
) -> None:
    store = StateStore(
        tmp_path / "state",
        run_id="run",
        user_id="user_a",
        session_id="fin_12",
        session_date="2027-01-31",
        session_timestamp="2027-01-31T10:00:00+00:00",
    )
    sent = store.send_email("alerts@example.test", "Change", "Please change it.")
    assert store.read_sent_email(sent["sent_id"])["sent_at"] == (
        "2027-01-31T10:00:00+00:00"
    )

    _write(
        tmp_path / "state",
        "email/inbox.jsonl",
        '{"id":"em_future","from":"alerts@example.test",'
        '"date":"2027-01-31T10:01:00+00:00","subject":"Future","body":"Later"}\n',
    )
    assert store.search_email("Future")["matches"] == []
    with pytest.raises(FileNotFoundError):
        store.read_email("em_future")


def test_date_only_write_clock_retains_the_noon_fallback(tmp_path: Path) -> None:
    store = _store(tmp_path, session_date="2027-01-31")
    sent = store.send_email("alerts@example.test", "Change", "Please change it.")
    assert store.read_sent_email(sent["sent_id"])["sent_at"] == (
        "2027-01-31T12:00:00+00:00"
    )


@pytest.mark.parametrize(
    ("session_date", "session_timestamp"),
    [
        ("2027-01-31", "not-a-timestamp"),
        ("2027-01-31", "2027-01-31"),
        ("2027-01-31", "2027-02-01T10:00:00+00:00"),
    ],
)
def test_invalid_full_write_clock_refuses_before_persisting(
    tmp_path: Path, session_date: str, session_timestamp: str
) -> None:
    store = StateStore(
        tmp_path / "state",
        run_id="run",
        user_id="user_a",
        session_id="fin_12",
        session_date=session_date,
        session_timestamp=session_timestamp,
    )
    with pytest.raises(ValueError, match="session_timestamp"):
        store.send_email("alerts@example.test", "Change", "Please change it.")
    assert not (tmp_path / "state" / "email" / "sent.jsonl").exists()


def test_timestamp_only_derives_date_and_preserves_wall_clock_offset(
    tmp_path: Path,
) -> None:
    stamp = "2027-01-31T10:00:00-05:00"
    store = StateStore(
        tmp_path / "state",
        run_id="run",
        user_id="user_a",
        session_id="fin_12",
        session_timestamp=stamp,
    )
    draft = store.save_email_draft("alerts@example.test", "Change", "First")
    store.save_email_draft(
        "alerts@example.test", "Change", "Second", draft_id=draft["draft_id"]
    )
    row = json.loads(
        (tmp_path / "state" / "email" / "drafts.jsonl").read_text().strip()
    )
    assert store.session_date == "2027-01-31"
    assert row["created_at"] == stamp
    assert row["updated_at"] == stamp


def _write(state: Path, relative: str, payload) -> None:
    path = state / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


PAYMENT_METHODS = {
    "payment_methods": [
        {"method_id": "pm_visa_4821", "kind": "credit_card", "masked": "Visa ****4821"},
    ]
}


# Message domain

def test_message_send_search_read_roundtrip(tmp_path: Path):
    store = _store(tmp_path)

    sent = store.message_send("Marcus", "Running ten minutes late for dinner.")
    assert sent == {"message_id": "msg_0001", "status": "sent"}

    found = store.message_search("dinner")
    assert found["matches"][0]["message_id"] == "msg_0001"
    assert found["matches"][0]["direction"] == "outgoing"

    read = store.message_read("msg_0001")
    assert read["body"] == "Running ten minutes late for dinner."

    with pytest.raises(FileNotFoundError):
        store.message_read("msg_9999")


def test_message_search_hides_future_records(tmp_path: Path):
    store = _store(tmp_path, session_date="2026-10-20")
    _write(
        tmp_path / "state",
        "message/threads.jsonl",
        '{"message_id":"msg_0001","with":"Marcus","direction":"incoming",'
        '"body":"Movie tonight?","date":"2026-10-18"}\n'
        '{"message_id":"msg_0002","with":"Marcus","direction":"incoming",'
        '"body":"Thanksgiving plans?","date":"2026-11-20"}\n',
    )

    found = store.message_search("Marcus")
    assert [m["message_id"] for m in found["matches"]] == ["msg_0001"]
    with pytest.raises(FileNotFoundError):
        store.message_read("msg_0002")


# Email draft revision

def test_email_draft_revise_with_draft_id_overwrites_in_place(tmp_path: Path):
    store = _store(tmp_path)

    first = store.save_email_draft("reyes@example.test", "Draft v1", "First body.")
    assert first == {"draft_id": "draft_0001", "status": "saved"}

    revised = store.save_email_draft(
        "reyes@example.test", "Draft v2", "Second body.", draft_id="draft_0001"
    )
    assert revised == {"draft_id": "draft_0001", "status": "updated"}

    read = store.read_email_draft("draft_0001")
    assert read["subject"] == "Draft v2"
    assert read["body"] == "Second body."

    drafts = (tmp_path / "state" / "email" / "drafts.jsonl").read_text(encoding="utf-8")
    assert drafts.count("draft_0001") == 1

    with pytest.raises(FileNotFoundError):
        store.save_email_draft("x@example.test", "S", "B", draft_id="draft_0999")


# Email date filter

def test_email_search_and_read_hide_mail_that_has_not_arrived(tmp_path: Path):
    _write(
        tmp_path / "state",
        "email/inbox.jsonl",
        '{"id":"em_0001","from":"okafor@wit.example","date":"2026-10-08",'
        '"subject":"Room booking confirmed","body":"Confirmed."}\n'
        '{"id":"em_0002","from":"reyes@example.test","date":"2026-11-12",'
        '"subject":"Section 2 comments","body":"Comments on the draft."}\n',
    )

    october = _store(tmp_path, session_date="2026-10-20")
    assert [m["id"] for m in october.search_email("reyes")["matches"]] == []
    assert [m["id"] for m in october.search_email("okafor")["matches"]] == ["em_0001"]
    with pytest.raises(FileNotFoundError):
        october.read_email("em_0002")

    november = _store(tmp_path, session_date="2026-11-15")
    assert [m["id"] for m in november.search_email("reyes")["matches"]] == ["em_0002"]

    unfiltered = _store(tmp_path)
    assert len(unfiltered.search_email("2026")["matches"]) == 2


# Calendar

def test_calendar_create_read_update_and_world_guard(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "calendar/events.json",
        {
            "events": [
                {
                    "event_id": "ev_world_xmas",
                    "title": "Christmas Day",
                    "date": "2026-12-25",
                    "scope": "world",
                },
            ]
        },
    )
    store = _store(tmp_path)

    created = store.calendar_event_create(
        "Dinner with Marcus",
        "2026-10-21",
        start_time="19:00",
        location="Thai place",
        participants=["Marcus Chen"],
    )
    event_id = created["event_id"]
    assert created["status"] == "created"

    read = store.calendar_event_read(event_id)
    assert read["event"]["scope"] == "personal"
    assert read["event"]["location"] == "Thai place"

    updated = store.calendar_event_update(event_id, date="2026-10-22", status="confirmed")
    assert updated == {"event_id": event_id, "status": "confirmed"}
    assert store.calendar_event_read(event_id)["event"]["date"] == "2026-10-22"

    cancelled = store.calendar_event_update(event_id, status="cancelled")
    assert cancelled["status"] == "cancelled"

    with pytest.raises(ValueError, match="world events"):
        store.calendar_event_update("ev_world_xmas", date="2026-12-26")

    world = store.calendar_event_search("Christmas")
    assert world["matches"][0]["event_id"] == "ev_world_xmas"


def test_calendar_event_update_writes_notes(tmp_path: Path):
    """acc_027: five updates failed because `notes` was not an accepted field.

    The PA then reported the update as done. `notes` alone is a change.
    """
    _write(
        tmp_path / "state",
        "calendar/events.json",
        {
            "events": [
                {
                    "event_id": "ev_0015",
                    "title": "Dinner with Marcus",
                    "date": "2026-10-13",
                    "scope": "personal",
                    "status": "confirmed",
                    "created_date": "2026-09-01",
                    "notes": "Weekly Tuesday dinner.",
                },
            ]
        },
    )
    store = _store(tmp_path)

    updated = store.calendar_event_update("ev_0015", notes="Resumes October 20.")

    assert updated == {"event_id": "ev_0015", "status": "confirmed"}
    assert store.calendar_event_read("ev_0015")["event"]["notes"] == "Resumes October 20."


def test_calendar_visibility_uses_created_date_not_event_date(tmp_path: Path):
    _write(
        tmp_path / "state",
        "calendar/events.json",
        {
            "events": [
                {
                    "event_id": "ev_jan_workshop",
                    "title": "January workshop",
                    "date": "2027-01-14",
                    "created_date": "2026-10-01",
                },
                {
                    "event_id": "ev_surprise",
                    "title": "Surprise seminar",
                    "date": "2026-10-25",
                    "created_date": "2026-11-01",
                },
            ]
        },
    )
    store = _store(tmp_path, session_date="2026-10-20")

    # A far-future event is visible because it was created in the past;
    # an event about a near date is hidden because it does not exist yet.
    assert store.calendar_event_search("workshop")["matches"]
    assert not store.calendar_event_search("surprise")["matches"]


RECURRING_EVENTS = {
    "events": [
        {
            "event_id": "ev_weekly",
            "title": "Dinner at Siam Garden with Marcus",
            "date": "2026-10-13",  # a Tuesday
            "created_date": "2026-10-01",
            "recurrence": "weekly on Tuesdays",
        },
        {
            "event_id": "ev_daily",
            "title": "Evening walk",
            "date": "2026-10-01",
            "created_date": "2026-10-01",
            "recurrence": "daily",
        },
        {
            "event_id": "ev_bounded",
            "title": "Office hours",
            "date": "2026-10-28",  # a Wednesday
            "created_date": "2026-10-01",
            "recurrence": "weekly on Mondays and Wednesdays through 2026-12-09",
        },
        {
            "event_id": "ev_once",
            "title": "Dentist",
            "date": "2026-10-20",
            "created_date": "2026-10-01",
        },
    ]
}


def test_calendar_search_expands_recurrence_from_the_anchor(tmp_path: Path):
    _write(tmp_path / "state", "calendar/events.json", RECURRING_EVENTS)
    store = _store(tmp_path)

    # 2026-11-10 is a Tuesday after the anchor.
    later = store.calendar_event_search("Siam Garden", date="2026-11-10")["matches"]
    assert [record["event_id"] for record in later] == ["ev_weekly"]
    # The occurrence the caller asked about, not the series anchor.
    assert later[0]["date"] == "2026-11-10"

    # The anchor itself still matches, unchanged.
    anchor = store.calendar_event_search("Siam Garden", date="2026-10-13")["matches"]
    assert anchor[0]["date"] == "2026-10-13"

    # A Wednesday is not a Tuesday, and dates before the anchor are not covered.
    assert not store.calendar_event_search("Siam Garden", date="2026-11-11")["matches"]
    assert not store.calendar_event_search("Siam Garden", date="2026-10-06")["matches"]


def test_calendar_search_daily_and_bounded_recurrence(tmp_path: Path):
    _write(tmp_path / "state", "calendar/events.json", RECURRING_EVENTS)
    store = _store(tmp_path)

    for day in ("2026-10-01", "2026-11-10", "2026-12-08"):
        assert store.calendar_event_search("Evening walk", date=day)["matches"]
    assert not store.calendar_event_search("Evening walk", date="2026-09-30")["matches"]

    # "through 2026-12-09" ends the series.
    assert store.calendar_event_search("Office hours", date="2026-12-09")["matches"]
    assert not store.calendar_event_search("Office hours", date="2026-12-14")["matches"]


def test_calendar_search_without_recurrence_stays_exact(tmp_path: Path):
    _write(tmp_path / "state", "calendar/events.json", RECURRING_EVENTS)
    store = _store(tmp_path)

    assert store.calendar_event_search("Dentist", date="2026-10-20")["matches"]
    assert not store.calendar_event_search("Dentist", date="2026-10-27")["matches"]


def test_calendar_search_covers_an_end_date_span_without_recurrence(tmp_path: Path):
    _write(
        tmp_path / "state",
        "calendar/events.json",
        {
            "events": [
                {
                    "event_id": "ev_span",
                    "title": "Autumn Model Train Exhibition",
                    "date": "2026-10-03",
                    "created_date": "2026-10-01",
                    "end_date": "2026-11-29",
                }
            ]
        },
    )
    store = _store(tmp_path)

    interior = store.calendar_event_search("Exhibition", date="2026-10-17")["matches"]
    assert [record["event_id"] for record in interior] == ["ev_span"]
    # The span keeps its own start date; only recurrences report an occurrence.
    assert interior[0]["date"] == "2026-10-03"

    assert store.calendar_event_search("Exhibition", date="2026-10-03")["matches"]
    assert store.calendar_event_search("Exhibition", date="2026-11-29")["matches"]
    assert not store.calendar_event_search("Exhibition", date="2026-11-30")["matches"]
    assert not store.calendar_event_search("Exhibition", date="2026-10-02")["matches"]


def test_unknown_recurrence_raises_with_the_record_id(tmp_path: Path):
    _write(
        tmp_path / "state",
        "calendar/events.json",
        {
            "events": [
                {
                    "event_id": "ev_freetext",
                    "title": "Book club",
                    "date": "2026-10-13",
                    "recurrence": "every other Tuesday",
                }
            ]
        },
    )
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="ev_freetext"):
        store.calendar_event_search("Book club")


def test_calendar_event_create_rejects_free_text_recurrence(tmp_path: Path):
    store = _store(tmp_path)

    # gpt acc_082: refused before the append, and classified, so the wrapper
    # can tell a wrong parameter from a missing fact without reading the text.
    refused = store.calendar_event_create(
        "Book club", "2026-10-13", recurrence="sometimes"
    )
    assert refused == {
        "ok": False,
        "error_code": "calendar_recurrence_invalid",
        "detail": (
            f"ev_0001: recurrence='sometimes' is not {RECURRENCE_FORMAT}"
        ),
    }
    assert not store.calendar_event_search("Book club")["matches"]

    store.calendar_event_create(
        "Book club", "2026-10-13", recurrence="weekly on Tuesdays"
    )
    assert store.calendar_event_search("Book club", date="2026-11-10")["matches"]


# Contacts

def test_contact_create_read_update_search(tmp_path: Path):
    store = _store(tmp_path)

    created = store.contact_create(
        "Dale",
        email="dale@comiccenter.example",
        organization="Dale's Comic Center",
        relation="store owner",
    )
    contact_id = created["contact_id"]

    read = store.contact_read(contact_id)
    assert read["contact"]["organization"] == "Dale's Comic Center"

    store.contact_update(contact_id, phone="555-0142")
    assert store.contact_read(contact_id)["contact"]["phone"] == "555-0142"

    found = store.lookup_contacts("comic")
    assert found["matches"][0]["contact_id"] == contact_id

    with pytest.raises(FileNotFoundError):
        store.contact_read("ct_9999")
    with pytest.raises(ValueError, match="at least one field"):
        store.contact_update(contact_id)


def test_contact_create_preserves_legacy_dict_contacts(tmp_path: Path):
    _write(
        tmp_path / "state",
        "contacts.json",
        {"management": {"name": "Building Management", "email": "mgmt@example.test"}},
    )
    store = _store(tmp_path)

    store.contact_create("Dr. Okafor", organization="WIT")

    found = store.lookup_contacts("Building Management")
    assert found["matches"][0]["email"] == "mgmt@example.test"
    assert store.lookup_contacts("Okafor")["matches"]


# Shopping

def _seed_shopping(state: Path) -> None:
    _write(
        state,
        "shopping/products.json",
        {
            "currency": "USD",
            "products": [
                {
                    "product_id": "prod_filter_std",
                    "item_type": "water filter",
                    "name": "AquaPure Standard Filter",
                    "price": 19.5,
                    "in_stock": True,
                    "delivery_methods": ["home_delivery"],
                },
                {
                    "product_id": "prod_filter_pro",
                    "item_type": "water filter",
                    "name": "AquaPure Pro Filter",
                    "price": 44.0,
                    "in_stock": True,
                    "delivery_methods": ["home_delivery", "pickup"],
                },
            ],
        },
    )
    _write(
        state,
        "shipping/addresses.json",
        {"addresses": [{"address_id": "addr_home", "label": "Apartment"}]},
    )
    _write(state, "shopping/order_store.json", {"orders": []})
    _write(state, "finance/payment_methods.json", PAYMENT_METHODS)


def test_shopping_order_lifecycle(tmp_path: Path):
    state = tmp_path / "state"
    _seed_shopping(state)
    store = _store(tmp_path)

    found = store.shopping_product_search("water filter")
    assert len(found["matches"]) == 2
    cheap = store.shopping_product_search("water filter", max_price=25.0)
    assert [p["product_id"] for p in cheap["matches"]] == ["prod_filter_std"]

    order = store.shopping_order_create(
        "prod_filter_std", 2, "addr_home", "pm_visa_4821"
    )
    order_id = order["order_id"]
    assert order["status"] == "placed"

    read = store.shopping_order_read(order_id)
    assert read["order"]["quantity"] == 2
    assert read["order"]["payment_method_id"] == "pm_visa_4821"

    store.shopping_order_update(order_id, quantity=1)
    assert store.shopping_order_read(order_id)["order"]["quantity"] == 1

    store.shopping_order_update(order_id, status="cancelled")
    with pytest.raises(ValueError, match="no longer be changed"):
        store.shopping_order_update(order_id, quantity=3)

    with pytest.raises(ValueError, match="unknown payment method"):
        store.shopping_order_create("prod_filter_std", 1, "addr_home", "pm_missing")


def test_shopping_order_create_continues_the_merchant_numbering(tmp_path: Path):
    """Seeded orders carry GlenMart's own numbers, quoted verbatim in the emails
    that announce them, so a new order is the next number in that run."""
    state = tmp_path / "state"
    _seed_shopping(state)
    _write(
        state,
        "shopping/order_store.json",
        {
            "orders": [
                {"order_id": "GM-118427", "status": "delivered"},
                {"order_id": "GM-124309", "status": "shipped"},
            ]
        },
    )
    store = _store(tmp_path)

    placed = store.shopping_order_create(
        "prod_filter_std", 1, "addr_home", "pm_visa_4821"
    )

    assert placed["order_id"] == "GM-124310"
    assert store.shopping_order_read("GM-124310")["order"]["quantity"] == 1


def test_an_unknown_merchant_gets_a_code_without_a_table_edit():
    """The fallback is the merchant's first two capitals -- the same rule that
    would have produced GlenMart's own GM."""
    from state_server.store import StateStore, order_merchant_code

    assert order_merchant_code("PineRidge Supply") == "PR"
    code = order_merchant_code("PineRidge Supply")
    assert StateStore._mint_order_id([], code) == "PR-000001"
    assert (
        StateStore._mint_order_id(
            [{"order_id": "GM-124309"}, {"order_id": "PR-000007"}],
            code,
        )
        == "PR-000008"
    )


def test_a_fixture_that_names_no_merchant_keeps_the_glenmart_default():
    """The key is optional, so the first world's fixture -- which has none --
    goes on minting GM- exactly as before."""
    from state_server.store import order_merchant

    assert order_merchant({}) == ("GlenMart", "GM")
    assert order_merchant({"orders": []}) == ("GlenMart", "GM")


def test_the_order_store_names_its_own_merchant(tmp_path: Path):
    """A world with no GlenMart in it mints its own shop's numbers, not
    GlenMart's: the merchant is declared at the top of the persona's own
    `shopping/order_store.json`."""
    from state_server.store import order_merchant

    state = tmp_path / "state"
    _seed_shopping(state)
    _write(
        state,
        "shopping/order_store.json",
        {
            "merchant": {"name": "Harbor Market", "code": "HM"},
            "orders": [{"order_id": "HM-004120", "status": "delivered"}],
        },
    )
    store = _store(tmp_path)

    placed = store.shopping_order_create(
        "prod_filter_std", 1, "addr_home", "pm_visa_4821"
    )

    assert placed["order_id"] == "HM-004121"
    assert store.shopping_order_read("HM-004121")["order"]["quantity"] == 1
    # The name survives the store's own write, and the declared merchant is a
    # dict beside `orders`, so the loaders that take the document's single list
    # value still find the records.
    document = json.loads(
        (state / "shopping" / "order_store.json").read_text(encoding="utf-8")
    )
    assert order_merchant(document) == ("Harbor Market", "HM")
    assert [value for value in document.values() if isinstance(value, list)] == [
        document["orders"]
    ]


def test_a_world_without_the_persona_layer_stores_reads_empty(tmp_path: Path):
    """comic_store and household are persona-layer stores: a world that has
    neither answers the way the cinema does when it has no showtimes -- the
    read's own shape, no matches, no exception."""
    store = _store(tmp_path)

    cinema = store.cinema_showtime_search("Any Cinema", "Any Film", "2026-10-01")
    assert cinema == {
        "cinema": "Any Cinema",
        "movie": "Any Film",
        "date": "2026-10-01",
        "matches": [],
    }
    assert store.comic_store_inventory_read("Any Title #1") == {
        "item": "Any Title #1",
        "matches": [],
    }
    assert store.household_inventory_search("kitchen", "filter") == {
        "location": "kitchen",
        "item": "filter",
        "usable_item_available": False,
        "matches": [],
    }


# Finance

def test_bill_read_payment_create_and_settlement(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "finance/bills.json",
        {
            "bills": [
                {
                    "bill_id": "bill_water_oct",
                    "account": "Glenmont Water",
                    "period": "2026-10",
                    "total": 43.75,
                    "status": "due",
                },
                {
                    "bill_id": "bill_water_sep",
                    "account": "Glenmont Water",
                    "period": "2026-09",
                    "total": 41.10,
                    "status": "paid",
                },
            ]
        },
    )
    _write(state, "finance/payment_methods.json", PAYMENT_METHODS)
    store = _store(tmp_path)

    both = store.utility_bill_read("water")
    assert [bill["bill_id"] for bill in both["bills"]] == [
        "bill_water_oct",
        "bill_water_sep",
    ]
    september = store.utility_bill_read("water", period="2026-09")
    assert [bill["bill_id"] for bill in september["bills"]] == ["bill_water_sep"]

    methods = store.payment_method_read()
    assert methods["payment_methods"][0]["method_id"] == "pm_visa_4821"

    with pytest.raises(ValueError, match="does not match the bill total"):
        store.bill_payment_create("bill_water_oct", "pm_visa_4821", amount=1.00)

    payment = store.bill_payment_create("bill_water_oct", "pm_visa_4821", amount=43.75)
    payment_id = payment["payment_id"]
    assert payment["status"] == "completed"

    read = store.bill_payment_read(payment_id)
    assert read["payment"]["bill_id"] == "bill_water_oct"

    settled = store.utility_bill_read("water", period="2026-10")["bills"][0]
    assert settled["status"] == "paid"
    assert settled["paid_by"] == payment_id

    with pytest.raises(ValueError, match="already paid"):
        store.bill_payment_create("bill_water_oct", "pm_visa_4821")


# Restaurant reservations

def test_restaurant_reservation_lifecycle(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "restaurant/slots.json",
        {
            "restaurants": [
                {
                    "restaurant_id": "rest_thai_house",
                    "name": "Siam Garden Thai House",
                    "cuisine": "thai",
                    "slots": [
                        {"date": "2026-10-21", "time": "19:00", "max_party": 4},
                        {"date": "2026-10-21", "time": "19:30", "max_party": 2},
                        {"date": "2026-10-22", "time": "19:00", "max_party": 6},
                    ],
                },
            ]
        },
    )
    store = _store(tmp_path)

    found = store.restaurant_reservation_search("Thai", date="2026-10-21", party_size=3)
    assert len(found["matches"]) == 1
    assert len(found["matches"][0]["slots"]) == 1  # 19:30 seats only 2

    reservation = store.restaurant_reservation_create(
        "rest_thai_house", "2026-10-21", "19:00", party_size=2,
        contact_phone="(555) 014-3388",
    )
    reservation_id = reservation["reservation_id"]
    assert reservation["status"] == "confirmed"

    read = store.restaurant_reservation_read(reservation_id)
    assert read["reservation"]["restaurant_name"] == "Siam Garden Thai House"

    store.restaurant_reservation_update(reservation_id, date="2026-10-22", time="19:00")
    assert store.restaurant_reservation_read(reservation_id)["reservation"]["date"] == "2026-10-22"

    with pytest.raises(ValueError, match="no open slot"):
        store.restaurant_reservation_update(reservation_id, time="23:45")

    store.restaurant_reservation_update(reservation_id, status="cancelled")
    with pytest.raises(ValueError, match="already cancelled"):
        store.restaurant_reservation_update(reservation_id, party_size=3)

    with pytest.raises(ValueError, match="unknown restaurant"):
        store.restaurant_reservation_create(
            "rest_missing", "2026-10-21", "19:00", 2, "(555) 014-3388"
        )


def test_a_party_larger_than_the_slot_cap_is_rejected(tmp_path: Path):
    """Search hides slots too small for the party; create has to refuse them too."""
    state = tmp_path / "state"
    _write(
        state,
        "restaurant/slots.json",
        {
            "restaurants": [
                {
                    "restaurant_id": "rest_thai_house",
                    "name": "Siam Garden Thai House",
                    "slots": [{"date": "2026-10-21", "time": "19:30", "max_party": 2}],
                },
            ]
        },
    )
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="party_size=3 exceeds max_party=2"):
        store.restaurant_reservation_create(
            "rest_thai_house", "2026-10-21", "19:30", 3, "(555) 014-3388"
        )

    booked = store.restaurant_reservation_create(
        "rest_thai_house", "2026-10-21", "19:30", 2, "(555) 014-3388"
    )
    assert booked["status"] == "confirmed"


def test_a_resource_with_no_slots_list_refuses_every_booking(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "restaurant/slots.json",
        {"restaurants": [{"restaurant_id": "rest_open", "name": "Open House"}]},
    )
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="rest_open has no slots list"):
        store.restaurant_reservation_create(
            "rest_open", "2026-10-21", "19:30", 2, "(555) 014-3388"
        )


def test_a_slot_with_no_max_party_is_a_fixture_error_not_an_open_table(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "restaurant/slots.json",
        {
            "restaurants": [
                {
                    "restaurant_id": "rest_uncapped",
                    "name": "Uncapped",
                    "slots": [{"date": "2026-10-21", "time": "19:30"}],
                },
            ]
        },
    )
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="19:30 at rest_uncapped has no max_party"):
        store.restaurant_reservation_create(
            "rest_uncapped", "2026-10-21", "19:30", 9, "(555) 014-3388"
        )
    with pytest.raises(ValueError, match="has no max_party"):
        store.restaurant_reservation_search("Uncapped", date="2026-10-21", party_size=2)


# Appointments

def test_appointment_lifecycle(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "appointment/availability.json",
        {
            "providers": [
                {
                    "provider_id": "prov_patel",
                    "name": "Dr. Patel",
                    "specialty": "family medicine",
                    "location": "Glenmont Family Clinic",
                    "slots": [
                        {"date": "2026-11-04", "time": "09:30"},
                        {"date": "2026-11-06", "time": "14:00"},
                    ],
                },
            ]
        },
    )
    store = _store(tmp_path)

    found = store.appointment_search("family medicine", date="2026-11-04")
    assert found["matches"][0]["provider_id"] == "prov_patel"

    created = store.appointment_create(
        "prov_patel", "2026-11-04", "09:30", patient="User A", reason="annual physical"
    )
    assert created == {"appointment_id": "apt_0001", "status": "confirmed"}
    appointment_id = created["appointment_id"]

    read = store.appointment_read(appointment_id)
    assert read["appointment"]["provider_name"] == "Dr. Patel"

    store.appointment_update(appointment_id, date="2026-11-06", time="14:00")
    assert store.appointment_read(appointment_id)["appointment"]["time"] == "14:00"

    store.appointment_update(appointment_id, status="cancelled")
    with pytest.raises(ValueError, match="already cancelled"):
        store.appointment_update(appointment_id, time="09:30")


def test_appointment_create_validates_and_persists_booking_route(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "appointment/availability.json",
        {
            "providers": [
                {
                    "provider_id": "prov_repair",
                    "name": "Repair Service",
                    "specialty": "appliance repair",
                    "location": "Home",
                    "booking_routes": [
                        {"contact_id": "ct_direct", "role": "direct", "channel": "email"},
                        {
                            "contact_id": "ct_agent",
                            "role": "intermediary",
                            "channel": "online",
                            "availability": "24/7",
                            "confirmation": "instant",
                        },
                    ],
                    "slots": [
                        {"date": "2027-01-22", "time": "09:00"},
                        {"date": "2027-01-29", "time": "13:00"},
                    ],
                }
            ]
        },
    )
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="requires a booking_contact_id"):
        store.appointment_create(
            "prov_repair", "2027-01-22", "09:00", "User C", "freezer fault"
        )
    with pytest.raises(ValueError, match="is not a route"):
        store.appointment_create(
            "prov_repair",
            "2027-01-22",
            "09:00",
            "User C",
            "freezer fault",
            booking_contact_id="ct_unknown",
        )

    direct = store.appointment_create(
        "prov_repair",
        "2027-01-22",
        "09:00",
        "User C",
        "freezer fault",
        booking_contact_id="ct_direct",
    )
    assert direct["booking_contact_id"] == "ct_direct"
    assert direct["booking_role"] == "direct"
    direct_record = store.appointment_read(direct["appointment_id"])["appointment"]
    assert direct_record["date"] == "2027-01-22"

    created = store.appointment_create(
        "prov_repair",
        "2027-01-29",
        "13:00",
        "User C",
        "freezer fault",
        booking_contact_id="ct_agent",
    )
    assert created["booking_contact_id"] == "ct_agent"
    assert created["booking_role"] == "intermediary"
    appointment = store.appointment_read(created["appointment_id"])["appointment"]
    assert appointment["date"] == "2027-01-29"
    assert appointment["booking_contact_id"] == "ct_agent"
    assert appointment["booking_role"] == "intermediary"
    assert appointment["reason"] == "freezer fault"
    assert len((state / "appointment/appointments.jsonl").read_text().splitlines()) == 2


# Workplace room bookings

def test_room_booking_lifecycle(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "room/rooms.json",
        {
            "rooms": [
                {
                    "room_id": "room_a101",
                    "name": "A101",
                    "building": "Physics",
                    "capacity": 12,
                    "slots": [
                        {"date": "2026-10-29", "time": "10:00"},
                        {"date": "2026-11-05", "time": "10:00"},
                    ],
                },
            ]
        },
    )
    store = _store(tmp_path)

    found = store.room_booking_search("A101", date="2026-10-29")
    assert len(found["matches"][0]["slots"]) == 1

    booked = store.room_booking_create(
        "room_a101", "2026-10-29", "10:00", end_time="11:30", purpose="manuscript discussion"
    )
    booking_id = booked["booking_id"]

    assert store.room_booking_read(booking_id)["booking"]["room_name"] == "A101"

    store.room_booking_update(booking_id, date="2026-11-05", time="10:00")
    store.room_booking_update(booking_id, status="cancelled")
    with pytest.raises(ValueError, match="already cancelled"):
        store.room_booking_update(booking_id, time="10:00")


# Hotel stays

def test_hotel_search_book_update_with_computed_totals(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "hotel/hotels.json",
        {
            "hotels": [
                {
                    "hotel_id": "htl_beacon",
                    "name": "Beacon Harbor Hotel",
                    "city": "Boston",
                    "nightly_rates": {"standard": 189.0, "deluxe": 249.0},
                },
            ]
        },
    )
    _write(state, "finance/payment_methods.json", PAYMENT_METHODS)
    store = _store(tmp_path)

    found = store.hotel_search("Boston", "2026-12-15", "2026-12-18")
    assert found["matches"][0]["hotel_id"] == "htl_beacon"

    booked = store.hotel_book(
        "htl_beacon", "2026-12-15", "2026-12-18", "standard", "User A", "pm_visa_4821"
    )
    booking_id = booked["booking_id"]

    record = store.hotel_booking_read(booking_id)["booking"]
    assert record["nights"] == 3
    assert record["total"] == 567.0  # 189 x 3, self-consistent

    store.hotel_booking_update(booking_id, check_out="2026-12-17")
    record = store.hotel_booking_read(booking_id)["booking"]
    assert record["nights"] == 2
    assert record["total"] == 378.0

    with pytest.raises(ValueError, match="room type"):
        store.hotel_book(
            "htl_beacon", "2026-12-15", "2026-12-16", "penthouse", "User A", "pm_visa_4821"
        )
    with pytest.raises(ValueError, match="must be after check_in"):
        store.hotel_search("Boston", "2026-12-18", "2026-12-15")


# Named submissions (reimbursement / conference registration / grades)

def test_named_submissions_lifecycle(tmp_path: Path):
    state = tmp_path / "state"
    _write(state, "finance/payment_methods.json", PAYMENT_METHODS)
    store = _store(tmp_path)

    filed = store.reimbursement_submission_create(
        "Winter Quantum Workshop travel", 412.80, notes="registration + flights"
    )
    read = store.reimbursement_submission_read(filed["submission_id"])
    assert read["submission"]["amount"] == 412.80

    registered = store.conference_registration_create(
        "Winter Quantum Workshop 2026", 180.0, "pm_visa_4821", attendee="User A"
    )
    reg = store.conference_registration_read(registered["registration_id"])
    assert reg["registration"]["fee"] == 180.0
    assert reg["registration"]["payment_method_id"] == "pm_visa_4821"

    graded = store.grade_submission_create(
        "PHYS 731 seminar", {"student_a": "A", "student_b": "B+"}, term="Fall 2026"
    )
    grades = store.grade_submission_read(graded["submission_id"])
    assert grades["submission"]["roster"]["student_b"] == "B+"

    with pytest.raises(ValueError, match="must not be empty"):
        store.grade_submission_create("PHYS 731 seminar", {})
    with pytest.raises(ValueError, match="unknown payment method"):
        store.conference_registration_create("WQW", 180.0, "pm_missing")


# Exhibition admissions

def test_exhibition_ticket_purchase_and_read(tmp_path: Path):
    state = tmp_path / "state"
    _write(state, "finance/payment_methods.json", PAYMENT_METHODS)
    store = _store(tmp_path)

    bought = store.exhibition_ticket_purchase(
        "Glenmont Civic Exhibition Hall",
        "Winter Science Exhibit",
        "2027-01-10",
        2,
        "pm_visa_4821",
    )
    ticket = store.exhibition_ticket_read(bought["ticket_id"])
    assert ticket["ticket"]["quantity"] == 2
    assert ticket["ticket"]["venue"] == "Glenmont Civic Exhibition Hall"


# Travel

def _seed_travel(state: Path) -> None:
    _write(
        state,
        "flight/flights.json",
        {
            "flights": [
                {
                    "flight_id": "WA1408",
                    "carrier": "Westwind Air",
                    "origin": "LAX",
                    "origin_city": "Los Angeles",
                    "destination": "BOS",
                    "destination_city": "Boston",
                    "date": "2026-12-14",
                    "depart_time": "08:05",
                    "arrive_time": "16:40",
                    "fares": {"Y": 428.0, "W": 610.0},
                },
                {
                    "flight_id": "WA1512",
                    "carrier": "Westwind Air",
                    "origin": "LAX",
                    "destination": "BOS",
                    "date": "2026-12-15",
                    "depart_time": "11:20",
                    "arrive_time": "19:55",
                    "fares": {"Y": 391.0},
                },
            ]
        },
    )
    _write(state, "finance/payment_methods.json", PAYMENT_METHODS)


def test_flight_search_read_book_and_readback(tmp_path: Path):
    state = tmp_path / "state"
    _seed_travel(state)
    store = _store(tmp_path)

    found = store.flight_search("LAX", "Boston", "2026-12-14")
    assert [f["flight_id"] for f in found["matches"]] == ["WA1408"]

    read = store.flight_read("WA1408")
    assert read["flight"]["fares"]["Y"] == 428.0

    booked = store.flight_book("WA1408", "User A", "pm_visa_4821", fare_class="Y")
    booking_id = booked["booking_id"]
    assert booked["status"] == "booked"

    readback = store.flight_booking_read(booking_id)
    assert readback["booking"]["flight_id"] == "WA1408"
    assert readback["booking"]["fare"] == 428.0

    with pytest.raises(ValueError, match="fare class"):
        store.flight_book("WA1512", "User A", "pm_visa_4821", fare_class="J")
    with pytest.raises(ValueError, match="unknown payment method"):
        store.flight_book("WA1408", "User A", "pm_missing")


# Shipping

def test_package_tracking_hides_future_scan_events(tmp_path: Path):
    _write(
        tmp_path / "state",
        "shipping/tracking.json",
        {
            "packages": [
                {
                    "tracking_id": "GLX240871",
                    "carrier": "Glenmont Express",
                    "events": [
                        {"date": "2026-10-18", "status": "label_created", "location": "Fresno, CA"},
                        {"date": "2026-10-19", "status": "in_transit", "location": "Bakersfield, CA"},
                        {"date": "2026-10-21", "status": "delivered", "location": "Glenmont, CA"},
                    ],
                }
            ]
        },
    )
    store = _store(tmp_path, session_date="2026-10-20")

    result = store.package_tracking_read("GLX240871")
    assert result["status"] == "in_transit"
    assert len(result["events"]) == 2

    later = _store(tmp_path, session_date="2026-10-22")
    assert later.package_tracking_read("GLX240871")["status"] == "delivered"


def test_delivery_address_read_lists_and_selects(tmp_path: Path):
    """The saved delivery addresses are readable, so no one has to recite an id.

    Before this the store held `shipping/addresses.json` with no way in, and
    `shopping_order_create` told the PA to ask the user for the raw `addr_*`.
    """
    state = tmp_path / "state"
    _write(
        state,
        "shipping/addresses.json",
        {
            "addresses": [
                {
                    "address_id": "addr_home01",
                    "label": "Apartment",
                    "recipient": "User A",
                    "street": "12 Corvid Ave, Apt 4B",
                    "city": "Glenmont",
                    "state": "CA",
                    "zip": "93710",
                    "default": True,
                },
                {
                    "address_id": "addr_dock02",
                    "label": "Branch loading dock",
                    "recipient": "User A",
                    "street": "480 Foothill Rd",
                    "city": "Glenmont",
                    "state": "CA",
                    "zip": "93710",
                    "default": False,
                },
            ]
        },
    )
    store = _store(tmp_path)

    every = store.delivery_address_read()["addresses"]
    assert [record["address_id"] for record in every] == ["addr_home01", "addr_dock02"]
    # The label and the default flag are what let the PA pick from a plain
    # description like "the apartment" without being handed the id.
    assert [record["label"] for record in every] == ["Apartment", "Branch loading dock"]
    assert [record["default"] for record in every] == [True, False]

    one = store.delivery_address_read("addr_dock02")["addresses"]
    assert [record["address_id"] for record in one] == ["addr_dock02"]

    with pytest.raises(FileNotFoundError):
        store.delivery_address_read("addr_missing")


def test_delivery_address_read_hides_an_address_saved_later(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "shipping/addresses.json",
        {
            "addresses": [
                {"address_id": "addr_home01", "label": "Apartment", "default": True},
                {
                    "address_id": "addr_new03",
                    "label": "Sublet",
                    "default": False,
                    "visible_from": "2026-10-20",
                },
            ]
        },
    )
    before = _store(tmp_path, session_date="2026-10-01").delivery_address_read()
    assert [record["address_id"] for record in before["addresses"]] == ["addr_home01"]

    after = _store(tmp_path, session_date="2026-10-21").delivery_address_read()
    assert [record["address_id"] for record in after["addresses"]] == [
        "addr_home01",
        "addr_new03",
    ]


# Reference

def test_scholarly_paper_and_recipe_search(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "scholarly/papers.json",
        {
            "papers": [
                {
                    "paper_id": "arxiv_2610_04412",
                    "title": "Decoherence pathways in non-Abelian anyonic systems",
                    "authors": ["R. Vasquez", "L. Chen"],
                    "abstract": "We analyze decoherence pathways for braided states.",
                    "fulltext": "Full readable text of the decoherence pathways paper.",
                },
                {
                    "paper_id": "arxiv_2609_11020",
                    "title": "Thermal noise in superconducting qubits",
                    "authors": ["M. Ito"],
                    "abstract": "Noise characterization for transmon devices.",
                },
            ]
        },
    )
    _write(
        state,
        "recipe/recipes.json",
        {
            "recipes": [
                {
                    "name": "Weeknight pad thai",
                    "ingredients": ["rice noodles", "egg", "peanuts"],
                    "total_minutes": 35,
                    "tags": ["thai"],
                },
                {
                    "name": "Slow braised short ribs",
                    "ingredients": ["short ribs"],
                    "total_minutes": 240,
                    "tags": ["dinner"],
                },
            ]
        },
    )
    store = _store(tmp_path)

    papers = store.scholarly_paper_search("anyonic decoherence")
    assert papers["matches"][0]["paper_id"] == "arxiv_2610_04412"
    # search returns metadata + abstract only; the heavy text needs paper_read
    assert "fulltext" not in papers["matches"][0]
    read = store.scholarly_paper_read("arxiv_2610_04412")
    assert "decoherence pathways" in read["paper"]["fulltext"]

    quick = store.recipe_search("noodles", max_minutes=60)
    assert [r["name"] for r in quick["matches"]] == ["Weeknight pad thai"]


# Id allocation and order visibility

def test_new_ids_skip_gaps_left_by_deleted_rows(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "message/threads.jsonl",
        '{"message_id": "msg_0001", "with": "Sam", "direction": "incoming", "body": "hi", "date": "2026-10-01"}\n'
        '{"message_id": "msg_0003", "with": "Sam", "direction": "incoming", "body": "later", "date": "2026-10-02"}\n',
    )
    store = _store(tmp_path)

    sent = store.message_send("Sam", "new one")

    assert sent["message_id"] == "msg_0004"
    assert store.message_read("msg_0004")["body"] == "new one"


def test_shopping_order_read_hides_orders_placed_after_session_date(tmp_path: Path):
    state = tmp_path / "state"
    _seed_shopping(state)
    _write(
        state,
        "shopping/order_store.json",
        {
            "orders": [
                {"order_id": "GM-1", "status": "delivered", "placed_at": "2026-10-12"},
                {"order_id": "GM-2", "status": "placed", "placed_at": "2027-01-19"},
            ]
        },
    )
    store = _store(tmp_path, session_date="2026-12-01")

    assert store.shopping_order_read("GM-1")["order"]["status"] == "delivered"
    with pytest.raises(ValueError, match="unknown order"):
        store.shopping_order_read("GM-2")
    with pytest.raises(ValueError, match="unknown order"):
        store.shopping_order_update("GM-2", quantity=2)


# Free-text query matching: every query word has to appear, in any order

def test_product_search_matches_query_words_out_of_order(tmp_path: Path):
    _write(
        tmp_path / "state",
        "shopping/products.json",
        {
            "currency": "USD",
            "products": [
                {
                    "product_id": "shp_ta15_aerator",
                    "item_type": "tap aerator",
                    "name": "AquaPure TA-15 tap aerator, M22×1 male",
                    "description": "Replacement aerator for AquaPure TA-series taps.",
                    "price": 6.4,
                    "in_stock": True,
                    "delivery_methods": ["home_delivery"],
                },
            ],
        },
    )
    store = _store(tmp_path)

    # "tap" sits between the words the PA typed, and the part numbers stay whole
    for query in ("AquaPure TA-15 aerator", "M22×1 aerator", "aerator AquaPure"):
        found = store.shopping_product_search(query)
        assert [p["product_id"] for p in found["matches"]] == ["shp_ta15_aerator"], query

    assert store.shopping_product_search("AquaPure TA-15 washer")["matches"] == []


def test_recipe_search_matches_query_words_out_of_order(tmp_path: Path):
    _write(
        tmp_path / "state",
        "recipe/recipes.json",
        {
            "recipes": [
                {
                    "name": "Weeknight pad thai",
                    "ingredients": ["rice noodles", "egg", "peanuts"],
                    "total_minutes": 35,
                    "tags": ["thai"],
                },
            ]
        },
    )
    store = _store(tmp_path)

    assert [
        r["name"] for r in store.recipe_search("thai noodles")["matches"]
    ] == ["Weeknight pad thai"]
    assert store.recipe_search("thai noodles beef")["matches"] == []


def test_email_search_matches_query_words_out_of_order(tmp_path: Path):
    _write(
        tmp_path / "state",
        "email/inbox.jsonl",
        '{"id":"em_0001","from":"okafor@wit.example","date":"2026-10-08",'
        '"subject":"Room booking confirmed","body":"Seminar room 3 is held for you."}\n',
    )
    store = _store(tmp_path)

    assert [
        m["id"] for m in store.search_email("seminar room booking")["matches"]
    ] == ["em_0001"]
    assert store.search_email("seminar room cancelled")["matches"] == []


def test_message_search_matches_query_words_out_of_order(tmp_path: Path):
    _write(
        tmp_path / "state",
        "message/threads.jsonl",
        '{"message_id":"msg_0001","with":"Marcus","direction":"incoming",'
        '"body":"Running ten minutes late for dinner.","date":"2026-10-18"}\n',
    )
    store = _store(tmp_path)

    assert [
        m["message_id"] for m in store.message_search("Marcus dinner late")["matches"]
    ] == ["msg_0001"]
    assert store.message_search("Marcus dinner cancelled")["matches"] == []


def test_contact_search_matches_short_and_reordered_query_words(tmp_path: Path):
    _write(
        tmp_path / "state",
        "contacts.json",
        {
            "contacts": [
                {
                    "contact_id": "ct_0001",
                    "name": "Amara Okafor",
                    "organization": "Westbrook Institute of Technology",
                    "relation": "department chair",
                },
            ]
        },
    )
    store = _store(tmp_path)

    assert [
        c["contact_id"] for c in store.lookup_contacts("chair Okafor")["matches"]
    ] == ["ct_0001"]
    # Both words are under the ranker's three-character token floor, so the
    # per-word match is the only thing that can find this record at all.
    assert [
        c["contact_id"] for c in store.lookup_contacts("Am Ok")["matches"]
    ] == ["ct_0001"]
    assert store.lookup_contacts("Am Xu")["matches"] == []


def test_paper_search_matches_short_and_reordered_query_words(tmp_path: Path):
    _write(
        tmp_path / "state",
        "scholarly/papers.json",
        {
            "papers": [
                {
                    "paper_id": "arxiv_2610_04412",
                    "title": "Decoherence pathways in non-Abelian anyonic systems",
                    "authors": ["R. Vasquez", "L. Chen"],
                    "abstract": "We analyze decoherence pathways for braided states.",
                },
            ]
        },
    )
    store = _store(tmp_path)

    assert [
        p["paper_id"] for p in store.scholarly_paper_search("anyonic braided Chen")["matches"]
    ] == ["arxiv_2610_04412"]
    # A citation-style query whose words are all under the ranker's
    # three-character token floor: only the per-word match can find it.
    assert [
        p["paper_id"] for p in store.scholarly_paper_search("R. Va 26")["matches"]
    ] == ["arxiv_2610_04412"]
    assert store.scholarly_paper_search("R. Xu 26")["matches"] == []


# Location-kind vocabularies

def _seed_world_with_locations(state: Path) -> None:
    """A world whose location fields and whose named records both exist.

    `near` is the vocabulary `place_search.location` matches on; the place
    *names* beside it are what the PA passed instead in acc_027.
    """
    _write(
        state,
        "places/nearby.json",
        {
            "places": [
                {
                    "name": "Casa Marisol",
                    "near": "Glenmont, CA — Civic Plaza",
                    "category": "restaurant",
                    "tags": ["dinner"],
                },
                {
                    "name": "Golden Peak Szechuan",
                    "near": "Glenmont, CA — Market St",
                    "category": "restaurant",
                    "tags": ["dinner"],
                },
                {
                    "name": "Glenmont Cinema",
                    "near": "Glenmont, CA — Civic Plaza",
                    "category": "cinema",
                },
            ]
        },
    )


def test_a_restaurant_name_where_place_search_wants_a_street_is_refused(tmp_path: Path):
    """acc_027: three searches, three empty results, no error.

    The PA read "0 matches" as "nothing there" rather than "wrong kind of
    value", so the world now says which kind of thing it was handed.
    """
    _seed_world_with_locations(tmp_path / "state")
    store = _store(tmp_path)

    with pytest.raises(ValueError) as raised:
        store.place_search(location="Casa Marisol", category="restaurant")

    message = str(raised.value)
    assert "'Casa Marisol' is a restaurant name" in message
    assert "restaurant_reservation_search" in message
    assert "Glenmont, CA — Civic Plaza" in message
    assert "Glenmont, CA — Market St" in message


def test_a_location_the_world_never_uses_is_refused_with_the_known_ones(tmp_path: Path):
    _seed_world_with_locations(tmp_path / "state")
    store = _store(tmp_path)

    with pytest.raises(ValueError) as raised:
        store.place_search(location="downtown", category="restaurant")

    message = str(raised.value)
    assert "location='downtown' is not a location this world knows" in message
    assert "Glenmont, CA — Civic Plaza" in message
    # Not any of the world's named records, so no kind is guessed at.
    assert " name; use " not in message


def test_a_street_the_world_does_use_still_searches(tmp_path: Path):
    _seed_world_with_locations(tmp_path / "state")
    store = _store(tmp_path)

    found = store.place_search(location="Civic Plaza", category="restaurant")

    assert [match["name"] for match in found["matches"]] == ["Casa Marisol"]


def test_weather_refuses_a_location_no_forecast_covers(tmp_path: Path):
    _write(
        tmp_path / "state",
        "weather/forecasts.json",
        {
            "forecasts": [
                {
                    "location": "Glenmont, CA",
                    "date": "2026-10-13",
                    "time_window": "18:00-22:00",
                    "forecast": {"recommendation": "Bring an umbrella."},
                }
            ]
        },
    )
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="Known location values: Glenmont, CA"):
        store.weather_forecast_read("Chicago", "2026-10-13", "18:00-22:00")

    assert store.weather_forecast_read("Glenmont, CA", "2026-10-13", "18:00-22:00")[
        "forecast"
    ]["recommendation"] == "Bring an umbrella."


def test_hotel_search_refuses_a_city_it_has_no_hotels_in(tmp_path: Path):
    _write(
        tmp_path / "state",
        "hotel/hotels.json",
        {"hotels": [{"hotel_id": "htl_1", "name": "Lakeshore Inn", "city": "Chicago"}]},
    )
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="Known city values: Chicago"):
        store.hotel_search("Glenmont", "2026-12-15", "2026-12-18")

    assert store.hotel_search("Chicago", "2026-12-15", "2026-12-18")["matches"]


def test_flight_search_refuses_an_endpoint_off_the_route_map(tmp_path: Path):
    _seed_travel(tmp_path / "state")
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="origin='Casa Marisol'"):
        store.flight_search("Casa Marisol", "Boston", "2026-12-14")
    with pytest.raises(ValueError, match="destination='Glenmont'"):
        store.flight_search("LAX", "Glenmont", "2026-12-14")

    assert store.flight_search("LAX", "Boston", "2026-12-14")["matches"]


def test_household_search_refuses_a_room_the_inventory_has_no_record_of(tmp_path: Path):
    _write(
        tmp_path / "state",
        "household/inventory.json",
        {
            "items": [
                {"location": "hall closet", "item": "umbrella", "quantity": 1,
                 "usable": True},
            ]
        },
    )
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="Known location values: hall closet"):
        store.household_inventory_search("garage", "umbrella")

    assert store.household_inventory_search("hall closet", "umbrella")[
        "usable_item_available"
    ]


# Write-side date visibility

VISIBILITY_PAYMENT = {
    "payment_methods": [
        {"method_id": "pm_visa_4821", "kind": "card", "masked": "****4821"}
    ]
}


def test_a_flight_published_after_the_session_cannot_be_booked(tmp_path: Path):
    """user_b's WA4412 is published 2026-11-15, the day after a 2026-11-14 session.

    `flight_read` hid it and `flight_book` resolved it anyway, so a PA that
    guessed the id -- or carried one over from a later session -- could book a
    flight the world had not published yet. The write path now applies the same
    visibility check as the read path.
    """
    state = tmp_path / "state"
    _write(
        state,
        "flight/flights.json",
        {
            "flights": [
                {
                    "flight_id": "WA4412",
                    "carrier": "Westwind Air",
                    "origin": "FAT",
                    "destination": "KFV",
                    "date": "2026-12-22",
                    "depart_time": "23:40",
                    "fares": {"Y": 148.00},
                    "published_date": "2026-11-15",
                    "seats_available": 9,
                }
            ]
        },
    )
    _write(state, "finance/payment_methods.json", VISIBILITY_PAYMENT)
    store = StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="s",
        session_date="2026-11-14",
    )

    with pytest.raises(FileNotFoundError):
        store.flight_read("WA4412")
    with pytest.raises(ValueError, match="unknown flight"):
        store.flight_book("WA4412", "Dana Whitfield", "pm_visa_4821", fare_class="Y")

    # The day it is published, both paths find it.
    later = StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="s",
        session_date="2026-11-15",
    )
    assert later.flight_read("WA4412")["flight"]["flight_id"] == "WA4412"
    assert later.flight_book(
        "WA4412", "Dana Whitfield", "pm_visa_4821", fare_class="Y"
    )["status"] == "booked"


def test_a_hotel_published_after_the_session_cannot_be_booked(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "hotel/hotels.json",
        {
            "hotels": [
                {
                    "hotel_id": "htl_late",
                    "name": "The Larkspur",
                    "city": "Kessler Falls",
                    "nightly_rates": {"standard": 112.00},
                    "published_date": "2026-11-15",
                }
            ]
        },
    )
    _write(state, "finance/payment_methods.json", VISIBILITY_PAYMENT)
    store = StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="s",
        session_date="2026-11-14",
    )

    assert store.hotel_search("Kessler Falls", "2026-12-22", "2026-12-23")["matches"] == []
    with pytest.raises(ValueError, match="unknown hotel"):
        store.hotel_book(
            "htl_late", "2026-12-22", "2026-12-23", "standard", "Dana", "pm_visa_4821"
        )


def test_a_product_published_after_the_session_cannot_be_ordered(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "shopping/products.json",
        {
            "currency": "USD",
            "products": [
                {
                    "product_id": "shp_late",
                    "item_type": "lamp",
                    "name": "Desk lamp",
                    "price": 30.00,
                    "in_stock": True,
                    "delivery_methods": ["home_delivery"],
                    "visible_from": "2026-11-15",
                }
            ],
        },
    )
    _write(state, "shipping/addresses.json", {"addresses": [{"address_id": "addr_1"}]})
    _write(state, "finance/payment_methods.json", VISIBILITY_PAYMENT)
    store = StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="s",
        session_date="2026-11-14",
    )

    assert store.shopping_product_search("lamp")["matches"] == []
    with pytest.raises(ValueError, match="unknown product"):
        store.shopping_order_create("shp_late", 1, "addr_1", "pm_visa_4821")


def test_a_slot_resource_published_after_the_session_cannot_be_booked(tmp_path: Path):
    """`_slotted_search` filtered on published_date; `_require_slot` did not.

    One helper covers appointment, restaurant and room in both directions.
    """
    state = tmp_path / "state"
    _write(
        state,
        "room/rooms.json",
        {
            "rooms": [
                {
                    "room_id": "room_late",
                    "name": "Annex B",
                    "published_date": "2026-11-15",
                    "slots": [{"date": "2026-11-20", "time": "10:00"}],
                }
            ]
        },
    )
    store = StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="s",
        session_date="2026-11-14",
    )

    assert store.room_booking_search("Annex")["matches"] == []
    with pytest.raises(ValueError, match="unknown room"):
        store.room_booking_create("room_late", "2026-11-20", "10:00")


def test_a_bill_issued_after_the_session_cannot_be_paid(tmp_path: Path):
    state = tmp_path / "state"
    _write(
        state,
        "finance/bills.json",
        {
            "bills": [
                {
                    "bill_id": "GP-2026-12",
                    "account": "Glenmont Power — acct 204-71-8836",
                    "provider": "Glenmont Power",
                    "period": "2026-12",
                    "issued_date": "2026-12-08",
                    "total": 78.65,
                }
            ]
        },
    )
    _write(state, "finance/payment_methods.json", VISIBILITY_PAYMENT)
    store = StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="s",
        session_date="2026-11-14",
    )

    with pytest.raises(FileNotFoundError):
        store.utility_bill_read("Glenmont Power")
    with pytest.raises(ValueError, match="unknown bill"):
        store.bill_payment_create("GP-2026-12", "pm_visa_4821")
