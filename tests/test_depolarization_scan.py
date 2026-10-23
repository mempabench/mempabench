"""Guard: pre-assigned-outcome mechanism fields must not reappear.

The fields that used to pre-decide which side a session would play are
deleted. They may survive only as rejection guards (leak-key bans,
do-not-output prompt lists) and in tests that prove they are rejected.
Anything else is a regression.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BANNED_FIELDS = (
    "probe_polarity",
    "realized_branch",
    "applies_to",
    "contact_outcome",
    "concrete_event_order",
)

SCAN_ROOTS = (
    "scripts",
    "simulator",
    "harness",
    "state_server",
    "pa",
    "data/scripts/prompts",
    ".agents/skills",
)

# (relative path, banned field) pairs that are rejection guards, not usage.
EXEMPT = {
    ("scripts/validate_normal_event.py", "probe_polarity"),
    ("data/scripts/prompts/session_outline_system.md", "probe_polarity"),
}


def test_banned_mechanism_fields_do_not_reappear() -> None:
    hits: list[str] = []
    for root in SCAN_ROOTS:
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in {".py", ".md", ".yaml"} or not path.is_file():
                continue
            relative = str(path.relative_to(PROJECT_ROOT))
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for field in BANNED_FIELDS:
                if field in text and (relative, field) not in EXEMPT:
                    hits.append(f"{relative}: {field}")
    assert not hits, (
        "banned pre-assigned-outcome fields found outside their rejection "
        f"guards: {hits}"
    )
