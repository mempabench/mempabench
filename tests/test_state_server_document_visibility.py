"""Section-level date scoping inside a document (owner, 2026-09-12).

A `<!-- visible_from: YYYY-MM-DD -->` marker on a line of its own opens a block
the store hides from a session that runs before that date, so one working
document can serve readers months apart: what its author knew in October and
what they added in December are the same byte stream, and the October session
never reads the December lines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from state_server.store import StateStore, filter_visible_blocks

ROSTER = """\
# Roster

Q1 marks are in.

<!-- visible_from: 2026-11-03 -->

Q2 (Nov 3) marks are in.

<!-- visible_from: 2026-12-01 -->

Q3 (Dec 1) marks and the pencilled letters are in.

<!-- visible_from: end -->

Filed under: teaching.
"""


def _store(tmp_path: Path, session_date: str | None) -> StateStore:
    state = tmp_path / "state"
    (state / "documents/admin").mkdir(parents=True)
    (state / "documents/admin/roster.md").write_text(ROSTER, encoding="utf-8")
    return StateStore(
        state,
        run_id="run",
        user_id="user_a",
        session_id="acc_001",
        session_date=session_date,
    )


def test_content_before_the_first_marker_is_always_visible() -> None:
    assert "Q1 marks are in." in filter_visible_blocks(ROSTER, "2026-10-19")


def test_a_block_dated_after_the_session_is_dropped() -> None:
    visible = filter_visible_blocks(ROSTER, "2026-10-19")

    assert "Q2 (Nov 3)" not in visible
    assert "Q3 (Dec 1)" not in visible


def test_a_block_dated_on_the_session_date_is_visible() -> None:
    """The same `<=` a record's own `visible_from` gets in `_date_visible`."""
    visible = filter_visible_blocks(ROSTER, "2026-11-03")

    assert "Q2 (Nov 3)" in visible
    assert "Q3 (Dec 1)" not in visible


def test_a_block_runs_to_the_next_marker_and_end_reopens_the_document() -> None:
    visible = filter_visible_blocks(ROSTER, "2026-10-19")

    # `end` closes the 12-01 block, so the trailing line is unscoped again.
    assert "Filed under: teaching." in visible


def test_the_markers_themselves_never_reach_the_reader() -> None:
    assert "visible_from" not in filter_visible_blocks(ROSTER, "2026-12-17")


def test_without_a_session_date_everything_is_visible() -> None:
    visible = filter_visible_blocks(ROSTER, None)

    assert "Q2 (Nov 3)" in visible
    assert "Q3 (Dec 1)" in visible
    assert "visible_from" not in visible


def test_a_document_with_no_marker_comes_back_byte_for_byte() -> None:
    plain = "# Notes\n\nnothing scoped here.\n"

    assert filter_visible_blocks(plain, "2026-10-19") == plain


@pytest.mark.parametrize(
    "marker",
    [
        "<!-- visible_from: soon -->",
        "<!-- visible_from: 2026/11/03 -->",
    ],
)
def test_a_malformed_date_keeps_its_block_visible(marker: str) -> None:
    """Hiding on a marker nobody can read would be the silent failure.

    The fixture validator rejects the marker before a run starts; the store's
    job is to not quietly swallow text on a shape it does not understand.
    """
    text = f"head\n\n{marker}\n\nbody\n"

    assert "body" in filter_visible_blocks(text, "2026-10-19")


def test_a_marker_not_alone_on_its_line_scopes_nothing() -> None:
    text = "head <!-- visible_from: 2026-12-01 --> body\n"

    assert filter_visible_blocks(text, "2026-10-19") == text


def test_document_read_serves_the_session_date(tmp_path: Path) -> None:
    early = _store(tmp_path / "early", "2026-10-19").read_document(
        "documents/admin/roster.md"
    )
    late = _store(tmp_path / "late", "2026-12-17").read_document(
        "documents/admin/roster.md"
    )

    assert "Q2 (Nov 3)" not in early["content"]
    assert "Q3 (Dec 1)" in late["content"]
    assert early["bytes"] == len(early["content"].encode("utf-8"))
    assert early["bytes"] < late["bytes"]


def test_document_read_with_no_session_date_hides_nothing(tmp_path: Path) -> None:
    result = _store(tmp_path, None).read_document("documents/admin/roster.md")

    assert "Q3 (Dec 1)" in result["content"]
