"""Hour-granular `visible_from` (owner, 2026-09-14).

A pre-event probe sits 60 minutes before its forced event, on a day that
already carries earlier accumulation sessions: pre_01 runs 17:00 on 2026-10-05,
after acc_013 at 10:00 and acc_014 at 14:00. A day-granular stamp opens the
probe's material at midnight, so that morning's sessions read it first. A stamp
written `YYYY-MM-DDTHH:MM` is compared against the session's own timestamp
instead; a date-only stamp keeps its day-granular meaning exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from state_server.store import StateStore, filter_visible_blocks, stamp_visible


# --- stamp_visible: the four combinations, at both edges -------------------


@pytest.mark.parametrize(
    "stamp, session_date, session_timestamp, visible",
    [
        # A date-only stamp is unchanged by the session's hour, in both
        # directions and on the day itself.
        ("2026-10-05", "2026-10-05", None, True),
        ("2026-10-05", "2026-10-05", "2026-10-05T00:00:00+00:00", True),
        ("2026-10-06", "2026-10-05", "2026-10-05T23:00:00+00:00", False),
        ("2026-10-04", "2026-10-05", "2026-10-05T00:00:00+00:00", True),
        # A timed stamp against a timed session: the new behaviour. 17:00 is
        # hidden from that morning's session and open to the evening's, and the
        # edge (exactly the session's own minute) is visible, the same `<=` a
        # date-only stamp gets on its own day.
        ("2026-10-05T17:00", "2026-10-05", "2026-10-05T14:00:00+00:00", False),
        ("2026-10-05T17:00", "2026-10-05", "2026-10-05T16:59:00+00:00", False),
        ("2026-10-05T17:00", "2026-10-05", "2026-10-05T17:00:00+00:00", True),
        ("2026-10-05T17:00", "2026-10-05", "2026-10-05T17:01:00+00:00", True),
        # Across days the hour never overrides the day.
        ("2026-10-05T17:00", "2026-10-06", "2026-10-06T09:00:00+00:00", True),
        ("2026-10-06T09:00", "2026-10-05", "2026-10-05T23:59:00+00:00", False),
        # A timed stamp against a date-only session keeps today's verdict: the
        # date part decides, so the authoring-time validators are unchanged.
        ("2026-10-05T17:00", "2026-10-05", None, True),
        ("2026-10-05T23:59", "2026-10-05", None, True),
        ("2026-10-06T00:01", "2026-10-05", None, False),
        # No session clock at all hides nothing.
        ("2026-10-06T09:00", None, None, True),
    ],
)
def test_stamp_visible(
    stamp: str, session_date: str | None, session_timestamp: str | None, visible: bool
) -> None:
    assert stamp_visible(stamp, session_date, session_timestamp) is visible


def test_a_timestamp_alone_is_enough_of_a_clock() -> None:
    """A caller that hands down only the timestamp still gets visibility, and
    the store fills its own `session_date` in from it."""
    assert stamp_visible("2026-10-05T17:00", None, "2026-10-05T14:00:00+00:00") is False
    assert stamp_visible("2026-10-06", None, "2026-10-05T14:00:00+00:00") is False
    assert stamp_visible("2026-10-05", None, "2026-10-05T14:00:00+00:00") is True


def test_seconds_and_an_offset_on_the_stamp_are_accepted() -> None:
    assert stamp_visible("2026-10-05T17:00:00", "2026-10-05", "2026-10-05T14:00:00+00:00") is False
    assert stamp_visible("2026-10-05T17:00:00+00:00", "2026-10-05", "2026-10-05T18:00:00+00:00") is True


# --- filter_visible_blocks with a timed marker -----------------------------


NOTES = """\
# Coaching notes

The standing note.

<!-- visible_from: 2026-10-05T17:00 -->

The line staged for the 17:00 probe.

<!-- visible_from: end -->

Filed under: acting.
"""


def test_a_timed_marker_is_dropped_from_that_mornings_session() -> None:
    visible = filter_visible_blocks(NOTES, "2026-10-05", "2026-10-05T14:00:00+00:00")

    assert "The standing note." in visible
    assert "staged for the 17:00 probe" not in visible
    assert "Filed under: acting." in visible


def test_a_timed_marker_is_open_at_its_own_hour() -> None:
    visible = filter_visible_blocks(NOTES, "2026-10-05", "2026-10-05T17:00:00+00:00")

    assert "staged for the 17:00 probe" in visible


def test_a_timed_marker_against_a_date_only_session_keeps_the_day_verdict() -> None:
    assert "staged for the 17:00 probe" in filter_visible_blocks(NOTES, "2026-10-05")
    assert "staged for the 17:00 probe" not in filter_visible_blocks(NOTES, "2026-10-04")


# --- the store's own reads -------------------------------------------------


CONTACT = {
    "contact_id": "ct_0023",
    "name": "Fifth Street Studio office",
    "organization": "Fifth Street Studio",
    "visible_from": "2026-10-05T17:00",
}


def _store(tmp_path: Path, session_timestamp: str | None) -> StateStore:
    state = tmp_path / "state"
    (state / "documents/acting").mkdir(parents=True)
    (state / "documents/acting/coaching_notes.md").write_text(NOTES, encoding="utf-8")
    (state / "contacts.json").write_text(
        json.dumps({"contacts": [CONTACT]}), encoding="utf-8"
    )
    return StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="acc_014",
        session_date="2026-10-05",
        session_timestamp=session_timestamp,
    )


def test_a_record_stamped_for_the_probe_is_hidden_from_that_mornings_session(
    tmp_path: Path,
) -> None:
    morning = _store(tmp_path / "morning", "2026-10-05T14:00:00+00:00")
    evening = _store(tmp_path / "evening", "2026-10-05T17:00:00+00:00")

    assert morning.lookup_contacts("Fifth Street")["matches"] == []
    assert [
        match["contact_id"] for match in evening.lookup_contacts("Fifth Street")["matches"]
    ] == ["ct_0023"]


def test_without_a_timestamp_the_store_keeps_its_day_granular_verdict(
    tmp_path: Path,
) -> None:
    """Every caller that knows only the day -- the authoring validators -- sees
    exactly what it saw before the hour existed."""
    store = _store(tmp_path, None)

    assert [
        match["contact_id"] for match in store.lookup_contacts("Fifth Street")["matches"]
    ] == ["ct_0023"]


def test_document_read_serves_the_session_hour(tmp_path: Path) -> None:
    morning = _store(tmp_path / "morning", "2026-10-05T14:00:00+00:00").read_document(
        "documents/acting/coaching_notes.md"
    )
    evening = _store(tmp_path / "evening", "2026-10-05T17:00:00+00:00").read_document(
        "documents/acting/coaching_notes.md"
    )

    assert "staged for the 17:00 probe" not in morning["content"]
    assert "staged for the 17:00 probe" in evening["content"]
    assert morning["bytes"] == len(morning["content"].encode("utf-8"))


def test_the_timestamp_alone_fills_in_the_session_date(tmp_path: Path) -> None:
    """`session_date` stays the store's own clock (`_now`, every authored
    `date` default), so it is derived rather than left unset."""
    state = tmp_path / "state"
    state.mkdir()
    store = StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="acc_014",
        session_timestamp="2026-10-05T14:00:00+00:00",
    )

    assert store.session_date == "2026-10-05"
