"""Manual harness: simulator generates one User A turn (3 blocks) for acc_001.beat[0].

Loads the authored-session test fixture (not the live draft) so this file stays
importable regardless of the state of Authoring/sessions/normal/drafts/.

Skips if OPENROUTER_API_KEY is unset. Prints all three blocks so we can eyeball:
  - <message> for in-character dialog quality
  - <factual_check> should be empty (no PA reply yet on the open beat)
  - <turn_assessment> should be next_beat: stay (opening turn, goal not yet met)

Run with:
    PYTHONPATH=. OPENROUTER_API_KEY=sk-... python tests/test_simulator/test_single_beat.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from simulator.config import SimulatorConfig
from simulator.loop import generate_beat_turn
from simulator.schemas import PersonaIdentity, PreferenceMatrix, SessionScript

ROOT = Path(__file__).resolve().parents[2]
SESSION_FIXTURE_PATH = ROOT / "tests/fixtures/authoring/normal_session_acc_001.yaml"


async def _run() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY unset — skipping live LLM test")
        return 0

    identity = PersonaIdentity.from_yaml(ROOT / "data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml(
        ROOT / "data/personas/user_a_preferences.yaml"
    )
    session = SessionScript.from_authored_session_yaml(SESSION_FIXTURE_PATH)
    open_beat = session.beats[0]

    config = SimulatorConfig()
    provider = config.create_provider()
    print(f"Provider: {config.provider_name}, model: {config.model}")
    print(f"Temperature: {config.generation.temperature}")
    print(f"Beat: {open_beat.beat}, active_skills: {open_beat.active_skills}")
    print(f"Context: {session.context}")
    print(f"task_facts keys: {sorted(session.task_facts.keys())}")
    print()

    output = await generate_beat_turn(
        identity, matrix, session, open_beat, provider, model=config.model
    )

    print("=" * 70)
    print("SIMULATOR OUTPUT - acc_001.open")
    print("=" * 70)
    print("--- <message> (PA-visible) ---")
    print(output.message)
    print()
    print("--- <factual_check> (eval-only, PA-blind) ---")
    if output.factual_check:
        for v in output.factual_check:
            print(f"  • {v.fact}: expected={v.expected!r}, pa_said={v.pa_said!r}")
    else:
        print("  (empty — expected for opening turn)")
    print()
    print("--- <turn_assessment> (eval-only, PA-blind) ---")
    print(f"  next_beat:  {output.turn_assessment.next_beat}")
    print()
    if output.parse_warnings:
        print("--- parse warnings ---")
        for w in output.parse_warnings:
            print(f"  ! {w}")
        print()
    print("=" * 70)

    assert output.message, "LLM returned empty message"
    assert len(output.message) > 20, f"message suspiciously short: {output.message!r}"
    print(f"\n  -> message: {len(output.message)} chars OK")
    print(f"  -> factual_check: {len(output.factual_check)} violations (open beat → expect 0)")
    print(f"  -> turn_assessment.next_beat: {output.turn_assessment.next_beat}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
