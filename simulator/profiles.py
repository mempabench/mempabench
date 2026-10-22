"""Nanobot-style profile workspaces for simulator personas."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from simulator.memory import (
    MAX_CONSOLIDATION_MAX_TOKENS,
    MemoryConsolidationContractError,
    MemoryConsolidationError,
    UserMemoryStore,
)
from simulator.prompting import render_character_card
from simulator.schemas import PersonaIdentity, PreferenceMatrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_FILES = ("TOOLS.md",)
DEFAULT_ACTOR_SKILL_SPECS_ROOT = PROJECT_ROOT / "data/actor_skill_specs"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ROLE_FRAME_PATH = PROMPTS_DIR / "profile_role_frame.md"
GROUND_RULES_PATH = PROMPTS_DIR / "profile_ground_rules.md"


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content.strip()
    end = content.find("\n---", 3)
    if end == -1:
        return content.strip()
    return content[end + 4 :].strip()


def _display_name_from_id(profile_id: str) -> str:
    parts = [part for part in profile_id.replace("-", "_").split("_") if part]
    if not parts:
        return profile_id
    if len(parts) == 2 and parts[0].lower() == "user" and len(parts[1]) == 1:
        return f"User {parts[1].upper()}"
    return " ".join(part.capitalize() for part in parts)


def _infer_preference_matrix(profile_id: str) -> Path | None:
    personas_dir = PROJECT_ROOT / "data/personas"
    exact = personas_dir / f"{profile_id}_preferences.yaml"
    if exact.exists():
        return exact
    candidates = sorted(personas_dir.glob(f"{profile_id}*preferences.yaml"))
    return candidates[-1] if candidates else None


def infer_identity_path(profile_id: str) -> Path | None:
    personas_dir = PROJECT_ROOT / "data/personas"
    exact = personas_dir / f"{profile_id}_identity.yaml"
    if exact.exists():
        return exact
    candidates = sorted(personas_dir.glob(f"{profile_id}*identity.yaml"))
    return candidates[-1] if candidates else None


@dataclass
class SimulatorProfile:
    """A simulator user profile stored as a nanobot-style workspace."""

    root: Path
    id: str
    display_name: str
    identity_path: Path | None = None
    preference_matrix_path: Path | None = None
    actor_skill_specs_root: Path | None = None
    default_context: str = "personal"
    memory_mode: str = "nanobot"
    history_context_chars: int = 12000
    raw_config: dict[str, Any] = field(default_factory=dict)

    def load_identity(self) -> PersonaIdentity:
        if self.identity_path is None or not Path(self.identity_path).exists():
            raise FileNotFoundError(
                f"profile {self.id!r} has no identity yaml: {self.identity_path}"
            )
        return PersonaIdentity.from_yaml(self.identity_path)

    def load_matrix(self) -> PreferenceMatrix:
        if self.preference_matrix_path is None:
            raise FileNotFoundError(f"profile {self.id!r} does not declare preference_matrix")
        return PreferenceMatrix.from_yaml(self.preference_matrix_path)

    def build_context(self) -> str:
        identity = self.load_identity()
        role_frame = ROLE_FRAME_PATH.read_text(encoding="utf-8").strip().replace(
            "<<DISPLAY_NAME>>", self.display_name
        )
        parts: list[str] = [
            f"## Identity\n\n{role_frame}\n\n{render_character_card(identity)}",
            "## Ground rules\n\n"
            + GROUND_RULES_PATH.read_text(encoding="utf-8").strip(),
        ]
        for filename in BOOTSTRAP_FILES:
            path = self.root / filename
            if path.exists():
                parts.append(f"## {filename}\n\n{path.read_text(encoding='utf-8').strip()}")

        memory = self._read_memory_context()
        if memory:
            parts.append(f"## Profile Memory\n\n{memory}")

        skills = self._read_profile_skills()
        if skills:
            parts.append(f"## Profile Skills\n\n{skills}")

        return "\n\n---\n\n".join(parts)

    async def record_session(
        self,
        exchanges: list[dict],
        *,
        provider: Any,
        model: str,
        session_id: str,
        timestamp: str | None,
    ) -> bool:
        """Write one first-person memory entry for a finished session.

        `timestamp` is in-world benchmark time, supplied by the caller. HISTORY
        is read back by the next session, so wall-clock time is never used here.
        """
        if self.memory_mode in {"off", "read_only", "readonly"}:
            return True
        if not exchanges:
            return True
        if not timestamp:
            raise MemoryConsolidationError(
                "MISSING_BENCHMARK_TIMESTAMP: "
                f"profile={self.id!r} session_id={session_id!r}"
            )
        store = UserMemoryStore(self.root, user_name=self.display_name)
        last_error: MemoryConsolidationContractError | None = None
        # Attempt 1 spends the provider's configured budget. A reply cut off at
        # the cap is the one failure an identical re-ask cannot fix, so that
        # failure -- and only that one -- doubles the budget for the next
        # attempt and asks for a shorter note. The count stays at 3: the
        # existing attempts grow, no attempt is added.
        budget: int = provider.generation.max_tokens
        budgets: list[int] = []
        finish_reasons: list[str] = []
        shorter_note = False
        for _ in range(3):
            budgets.append(budget)
            try:
                await store.consolidate(
                    exchanges,
                    provider,
                    model,
                    timestamp=timestamp,
                    max_tokens=budget,
                    shorter_note=shorter_note,
                )
                return True
            except MemoryConsolidationContractError as exc:
                last_error = exc
                finish_reasons.append(str(exc.finish_reason))
                if exc.finish_reason == "length":
                    budget = min(budget * 2, MAX_CONSOLIDATION_MAX_TOKENS)
                    shorter_note = True
        counts = Counter(finish_reasons)
        raise MemoryConsolidationError(
            "INVALID_CONSOLIDATION: "
            f"profile={self.id!r} session_id={session_id!r} "
            f"turns={len(exchanges)} failed after {len(budgets)} attempts "
            f"(finish_reason={','.join(f'{r} x{n}' for r, n in counts.items())}"
            f", max_tokens={'→'.join(str(b) for b in budgets)}) "
            f"last_error={last_error}"
        ) from last_error

    def _read_memory_context(self) -> str:
        memory_dir = self.root / "memory"
        parts: list[str] = []

        memory_file = memory_dir / "MEMORY.md"
        if memory_file.exists():
            body = memory_file.read_text(encoding="utf-8").strip()
            if body:
                parts.append(f"### Long-term Memory\n\n{body}")

        history_file = memory_dir / "HISTORY.md"
        if history_file.exists():
            body = history_file.read_text(encoding="utf-8").strip()
            if body:
                if self.history_context_chars > 0 and len(body) > self.history_context_chars:
                    body = body[-self.history_context_chars :].lstrip()
                parts.append(f"### Recent History\n\n{body}")

        return "\n\n".join(parts)

    def _read_profile_skills(self) -> str:
        skills_dir = self.root / "skills"
        if not skills_dir.exists():
            return ""
        blocks: list[str] = []
        for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
            content = _strip_frontmatter(skill_file.read_text(encoding="utf-8"))
            if content:
                blocks.append(f"### Skill: {skill_file.parent.name}\n\n{content}")
        return "\n\n---\n\n".join(blocks)


def load_simulator_profile(path: str | Path) -> SimulatorProfile:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"simulator profile not found: {root}")

    config_path = root / "profile.yaml"
    profile_config: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        profile_config = loaded or {}
    if not isinstance(profile_config, dict):
        raise ValueError(f"profile config must be a mapping: {config_path}")

    def resolve_profile_path(value: str | None) -> Path | None:
        if not value:
            return None
        candidate_path = Path(value).expanduser()
        if candidate_path.is_absolute():
            return candidate_path
        profile_relative = root / candidate_path
        if profile_relative.exists():
            return profile_relative
        return PROJECT_ROOT / candidate_path

    memory_cfg = (
        profile_config.get("memory")
        if isinstance(profile_config.get("memory"), dict)
        else {}
    )
    memory_mode = str(
        memory_cfg.get("mode")
        or profile_config.get("memory_mode")
        or "nanobot"
    )
    history_chars = int(
        memory_cfg.get("history_context_chars")
        or profile_config.get("history_context_chars")
        or 12000
    )
    profile_id = str(profile_config.get("id") or root.name)

    return SimulatorProfile(
        root=root,
        id=profile_id,
        display_name=str(
            profile_config.get("display_name")
            or profile_config.get("name")
            or _display_name_from_id(profile_id)
        ),
        identity_path=(
            resolve_profile_path(profile_config.get("identity"))
            or infer_identity_path(profile_id)
        ),
        preference_matrix_path=(
            resolve_profile_path(profile_config.get("preference_matrix"))
            or _infer_preference_matrix(profile_id)
        ),
        actor_skill_specs_root=(
            resolve_profile_path(profile_config.get("actor_skill_specs_root"))
            or DEFAULT_ACTOR_SKILL_SPECS_ROOT
        ),
        default_context=str(profile_config.get("default_context") or "personal"),
        memory_mode=memory_mode,
        history_context_chars=history_chars,
        raw_config=profile_config,
    )
