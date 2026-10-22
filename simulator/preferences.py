"""Runtime preference matrix overlays for dynamic simulator preferences."""

from __future__ import annotations

from typing import Mapping

from simulator.schemas import PreferenceCell, PreferenceMatrix, PreferencePatch


PreferencePatchMap = Mapping[str, Mapping[str, PreferencePatch | dict]]


def apply_preference_patches(
    matrix: PreferenceMatrix,
    patches: PreferencePatchMap | None,
) -> PreferenceMatrix:
    """Return a copy of `matrix` with runtime preference patches applied.

    Patch shape mirrors `PreferenceMatrix.matrix`:

    ```yaml
    personal:
      verbosity:
        setting: Detailed
        source: scripted_dynamic
        confidence: High
        notes: Changed after repeated terse PA answers.
    ```
    """
    if not patches:
        return matrix

    updated = matrix.model_copy(deep=True)
    for context, attr_patches in patches.items():
        if context not in updated.matrix:
            raise KeyError(
                f"Preference patch context {context!r} not in matrix. "
                f"Available: {sorted(updated.matrix.keys())}"
            )
        for attr, raw_patch in attr_patches.items():
            if attr not in updated.matrix[context]:
                raise KeyError(
                    f"Preference patch attribute {attr!r} not in matrix[{context!r}]. "
                    f"Available: {sorted(updated.matrix[context].keys())}"
                )
            patch = raw_patch if isinstance(raw_patch, PreferencePatch) else PreferencePatch(**raw_patch)
            current = updated.matrix[context][attr]
            # A patch that names no setting keeps the cell's own setting;
            # PreferenceCell rejects an empty one either way.
            updated.matrix[context][attr] = PreferenceCell(
                setting=patch.setting if patch.setting is not None else current.setting,
                source=patch.source or current.source,
                confidence=patch.confidence if patch.confidence is not None else current.confidence,
                notes=patch.notes if patch.notes is not None else current.notes,
            )
    return updated
