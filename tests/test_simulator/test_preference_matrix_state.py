"""A persona-matrix cell always names a setting; the runtime schema enforces it."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from simulator.schemas import PreferenceCell, PreferenceMatrix, PreferencePatch

ROOT = Path(__file__).resolve().parents[2]
PERSONA_FILES = sorted((ROOT / "data/personas").glob("*_preferences.yaml"))


def _cell(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {"setting": "Terse", "source": "extractor_high", "confidence": "High", "notes": None}
    base.update(kwargs)
    return base


def _matrix(work: dict[str, dict[str, object]], personal: dict[str, dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "character": "T",
        "sources": {"pass2": "x"},
        "matrix": {"work": work, "personal": personal if personal is not None else work},
    }


def test_a_cell_names_its_setting() -> None:
    cell = PreferenceCell.model_validate(_cell(setting=" Terse "))
    assert cell.setting == "Terse" and cell.source == "extractor_high"


@pytest.mark.parametrize(
    "cell",
    [
        _cell(setting=None),  # no setting at all
        _cell(setting="  "),  # blank setting names nothing
        _cell(no_preference=True),  # retired flag
    ],
)
def test_cells_without_a_named_setting_are_rejected(cell: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PreferenceCell.model_validate(cell)


def test_patch_rejects_the_retired_no_preference_flag() -> None:
    with pytest.raises(ValidationError):
        PreferencePatch.model_validate({"no_preference": True})


def test_matrix_rejects_stored_counts_and_bad_shapes() -> None:
    with pytest.raises(ValidationError, match="stats"):
        PreferenceMatrix.model_validate({**_matrix({"verbosity": _cell()}), "stats": {"seed": 1}})
    with pytest.raises(ValidationError, match="contexts must be exactly"):
        PreferenceMatrix.model_validate({"character": "T", "matrix": {"work": {"verbosity": _cell()}}})
    with pytest.raises(ValidationError, match="same attributes"):
        PreferenceMatrix.model_validate(_matrix({"verbosity": _cell()}, {"tone_formality": _cell()}))
    with pytest.raises(ValidationError, match="provenance"):
        PreferenceMatrix.model_validate(_matrix({"verbosity": _cell(source="scripted_dynamic")}))
    # `no_preference` left the persona provenance vocabulary with the empty cell.
    with pytest.raises(ValidationError, match="provenance"):
        PreferenceMatrix.model_validate(_matrix({"verbosity": _cell(source="no_preference")}))


def test_a_setting_retired_from_the_taxonomy_is_refused_at_load() -> None:
    """`Suggest` was merged into `Self-directed` on 2026-09-05 (autonomy_level went
    from four settings to three). A cell still naming it is stale, not merely
    unusual, so the matrix must not load at all."""
    data = _matrix({"autonomy_level": _cell(setting="Suggest")})
    with pytest.raises(ValidationError, match="not a taxonomy setting any more"):
        PreferenceMatrix.model_validate(data)

    data = _matrix({"autonomy_level": _cell(setting="Self-directed")})
    assert PreferenceMatrix.model_validate(data).get("work", "autonomy_level").setting == (
        "Self-directed"
    )


INVALID_MARKER = "invalid_settings_removed_20260905"


def _is_marked_invalid(path: Path) -> bool:
    return f"status: {INVALID_MARKER}" in path.read_text(encoding="utf-8")


def test_matrix_marked_invalid_is_refused_at_load() -> None:
    """Matrices holding settings removed on 2026-09-05 must not load at all."""
    with pytest.raises(ValidationError, match=INVALID_MARKER):
        PreferenceMatrix.model_validate(
            {**_matrix({"verbosity": _cell()}), "status": INVALID_MARKER}
        )
    # Every shipped matrix has been re-extracted under the current settings.
    # user_b was in this marked set while its cells were being re-extracted
    # (7d6a3b1); the promoted matrix carries no marker, so the set is empty and
    # user_b has to load like any other persona.
    marked = [path for path in PERSONA_FILES if _is_marked_invalid(path)]
    assert marked == []
    user_b = ROOT / "data/personas/user_b_preferences.yaml"
    assert user_b in PERSONA_FILES and not _is_marked_invalid(user_b)
    assert set(PreferenceMatrix.from_yaml(user_b).matrix) == {"work", "personal"}
    for path in marked:
        with pytest.raises(Exception):
            PreferenceMatrix.from_yaml(path)


def test_every_persona_file_loads_with_every_cell_named() -> None:
    assert PERSONA_FILES, "no persona matrices found"
    for path in PERSONA_FILES:
        if _is_marked_invalid(path):
            continue
        matrix = PreferenceMatrix.from_yaml(path)
        for context, cells in matrix.matrix.items():
            assert len(cells) == 14, (path.name, context, len(cells))
            for attr, cell in cells.items():
                assert cell.setting, (path.name, context, attr)
