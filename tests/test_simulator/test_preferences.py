from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from simulator.preferences import apply_preference_patches
from simulator.schemas import PreferenceMatrix

_FIXTURE_YAML = (
    Path(__file__).resolve().parents[1] / "fixtures/preference_patch_matrix.yaml"
)


def _fixture_matrix() -> PreferenceMatrix:
    """A synthetic matrix in the persona shape, used only to exercise patch behaviour."""
    return PreferenceMatrix.from_yaml(_FIXTURE_YAML)


def test_apply_preference_patches_returns_updated_copy() -> None:
    matrix = _fixture_matrix()

    updated = apply_preference_patches(
        matrix,
        {
            "personal": {
                "verbosity": {
                    "setting": "Detailed",
                    "source": "scripted_dynamic",
                    "confidence": "High",
                    "notes": "User shifted after a long-form task.",
                }
            }
        },
    )

    assert matrix.get("personal", "verbosity").setting == "Terse"
    cell = updated.get("personal", "verbosity")
    assert cell.setting == "Detailed"
    assert cell.source == "scripted_dynamic"
    assert cell.confidence == "High"
    assert cell.notes == "User shifted after a long-form task."


def test_apply_preference_patches_cannot_empty_a_cell() -> None:
    matrix = _fixture_matrix()

    # The retired flag is rejected outright.
    with pytest.raises(ValidationError):
        apply_preference_patches(
            matrix,
            {"personal": {"proactive_outreach": {"no_preference": True}}},
        )

    # A patch that names no setting keeps the cell's own setting; it cannot
    # leave the cell empty.
    kept = apply_preference_patches(
        matrix,
        {"work": {"proactive_outreach": {"source": "scripted_dynamic"}}},
    )
    assert kept.get("work", "proactive_outreach").setting == "Low"

    updated = apply_preference_patches(
        matrix,
        {"work": {"proactive_outreach": {"setting": "High", "source": "scripted_dynamic"}}},
    )
    assert updated.get("work", "proactive_outreach").setting == "High"


def test_apply_preference_patches_rejects_unknown_attribute() -> None:
    matrix = _fixture_matrix()

    with pytest.raises(KeyError):
        apply_preference_patches(
            matrix,
            {"personal": {"not_an_attribute": {"setting": "Terse"}}},
        )
