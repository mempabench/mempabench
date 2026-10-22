"""Actor-skill-spec loader for simulator prompt assembly.

Actor skill specs live at `data/actor_skill_specs/{attribute}/{setting}.md`
(45 files, character-agnostic). At simulator runtime we slice the persona's
preference matrix by `session.context` and load the spec for each attribute
named in the beat's `active_skills`.
"""

from __future__ import annotations

import re
from pathlib import Path

from simulator.schemas import PreferenceCell, PreferenceMatrix


DEFAULT_ACTOR_SKILL_SPECS_ROOT = Path("data/actor_skill_specs")


def normalize_setting(setting: str) -> str:
    """Map a matrix setting string ('Find and hand off') to its filename stem
    ('find-and-hand-off').
    """
    normalized = re.sub(r"[^a-z0-9]+", "-", setting.lower().replace("'", ""))
    return normalized.strip("-")


def _strip_frontmatter(body: str) -> str:
    """Drop a leading YAML frontmatter block (--- ... ---) if present."""
    if not body.startswith("---"):
        return body
    end = body.find("\n---", 3)
    if end == -1:
        return body
    after = body[end + 4 :]
    return after.lstrip("\n")


# The spec sections that describe how the actor REACTS to what the PA said.
# The dropped ones (`## Base Def`, `## Production skeleton`, `## Anti-behaviors`)
# tell the actor what to produce, which is not what a carried-over spec governs.
RECEPTION_SECTIONS: tuple[str, ...] = (
    "Core behavior",
    "Reception skeleton",
    "Trigger signals",
)


def reception_only(body: str) -> str:
    """Cut a spec body down to its title and the reception sections."""
    kept: list[str] = []
    keeping = True
    for line in body.splitlines():
        if line.startswith("## "):
            keeping = line[3:].strip() in RECEPTION_SECTIONS
        if keeping:
            kept.append(line)
    return "\n".join(kept).strip()


def load_actor_skill_spec(
    attribute: str,
    setting: str,
    root: Path | str = DEFAULT_ACTOR_SKILL_SPECS_ROOT,
) -> str:
    """Read a single actor skill spec and return its body (frontmatter dropped)."""
    root_path = Path(root)
    filename = f"{normalize_setting(setting)}.md"
    path = root_path / attribute / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Actor skill spec not found: {path} "
            f"(attribute={attribute!r}, setting={setting!r})"
        )
    return _strip_frontmatter(path.read_text(encoding="utf-8"))


def resolve_active_skills(
    matrix: PreferenceMatrix,
    context: str,
    active_skills: list[str],
    setting_overrides: dict[str, str] | None = None,
    root: Path | str = DEFAULT_ACTOR_SKILL_SPECS_ROOT,
) -> list[tuple[str, str, str]]:
    """Resolve a beat's active_skills against the matrix.

    Returns list of (attribute, setting, spec_body). Every cell names a
    setting, so every listed attribute resolves to a skill.
    """
    if context not in matrix.matrix:
        raise KeyError(
            f"Context {context!r} not in matrix. "
            f"Available: {sorted(matrix.matrix.keys())}"
        )

    cells = matrix.matrix[context]
    overrides = setting_overrides or {}
    resolved: list[tuple[str, str, str]] = []
    for attr in active_skills:
        if attr not in cells:
            raise KeyError(
                f"Attribute {attr!r} not in matrix[{context!r}]. "
                f"Available: {sorted(cells.keys())}"
            )
        cell: PreferenceCell = cells[attr]
        setting = overrides.get(attr) or cell.setting
        body = load_actor_skill_spec(attr, setting, root=root)
        resolved.append((attr, setting, body))
    return resolved
