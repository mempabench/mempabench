"""simulator/ must never import from pa/. harness/ is the only integration point."""

from __future__ import annotations

import re
from pathlib import Path

SIMULATOR_ROOT = Path(__file__).resolve().parents[2] / "simulator"

_PA_IMPORT = re.compile(r"^\s*(?:from\s+pa(?:\.|\s)|import\s+pa(?:\.|\s|$))")


def test_simulator_has_no_pa_imports() -> None:
    offenders: list[str] = []
    for path in sorted(SIMULATOR_ROOT.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _PA_IMPORT.match(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, "simulator/ imports from pa/:\n" + "\n".join(offenders)
