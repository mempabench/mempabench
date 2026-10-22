"""IX exact match — test doc §0-B2 (a), §4.6 `ix_face` (row 19 of §7).

The explicit-selection face never goes through an LLM: the setting the PA
chose with its IX tool is compared verbatim with the run's frozen GT.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def ix_match(selected: str | None, expected: str) -> dict[str, Any]:
    """Exact string match of the PA's selection against the GT setting.

    `selected` is `ix_selection.json`'s `selected_setting` (None when no
    selection was made); `expected` is `ground_truth.json`'s `setting`.
    """
    return {"matched": selected == expected, "selected": selected, "expected": expected}


def ix_face(ix_selection: Mapping[str, Any], ground_truth: Mapping[str, Any]) -> dict[str, Any]:
    """The §4.6 `ix_face` block for one judge output, from the two packed files."""
    result = ix_match(ix_selection.get("selected_setting"), ground_truth["setting"])
    return {
        "gt_setting": result["expected"],
        "pa_setting": result["selected"],
        "exact_match": result["matched"],
        "no_selection": bool(ix_selection.get("no_selection")),
        "computed_by": "code",
    }
