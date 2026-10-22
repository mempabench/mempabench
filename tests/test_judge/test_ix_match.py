"""judge/ix_match.py — the `ix_face` block. Pure code, no LLM."""

from __future__ import annotations

import pytest

from judge.ix_match import ix_face, ix_match


@pytest.mark.parametrize(
    ("selected", "expected", "matched"),
    [
        ("Detailed", "Detailed", True),
        ("Terse", "Detailed", False),
        ("detailed", "Detailed", False),  # exact: case is part of the label
        (None, "Detailed", False),
    ],
)
def test_ix_match_is_an_exact_string_comparison(selected, expected, matched) -> None:
    assert ix_match(selected, expected) == {"matched": matched, "selected": selected, "expected": expected}


def test_ix_face_has_the_section_4_6_shape_and_is_computed_by_code() -> None:
    face = ix_face(
        {"selected_setting": "Detailed", "no_selection": False},
        {"setting": "Detailed", "allowed_settings": ["Terse", "Detailed"]},
    )
    assert face == {
        "gt_setting": "Detailed", "pa_setting": "Detailed", "exact_match": True,
        "no_selection": False, "computed_by": "code",
    }


def test_ix_face_with_no_selection_is_a_mismatch() -> None:
    face = ix_face({"selected_setting": None, "no_selection": True}, {"setting": "Terse"})
    assert face["exact_match"] is False
    assert face["pa_setting"] is None
    assert face["no_selection"] is True
