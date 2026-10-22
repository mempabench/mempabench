"""Backend protocol for simulator turn generation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from simulator.output import SimulatorOutput
from simulator.reply_facts import ReplyFacts
from simulator.schemas import Beat, PersonaIdentity, PreferenceMatrix, SessionScript


class SimulatorBackend(Protocol):
    """Produces one parsed simulator turn for a script beat."""

    name: str

    def get_default_model(self) -> str: ...

    async def drain_memory_tasks(self, *, timestamp: str | None = None) -> None: ...

    async def generate_turn(
        self,
        identity: PersonaIdentity,
        matrix: PreferenceMatrix,
        session: SessionScript,
        beat: Beat,
        history: list[dict] | None = None,
        actor_skill_specs_root: Path | str | None = None,
        model: str | None = None,
        first_turn_in_beat: bool = True,
        benchmark_timestamp: str | None = None,
        held_items: Iterable[dict[str, str]] = (),
        pa_executed_tools: Iterable[str] = (),
        pa_tool_log: Iterable[str] = (),
        pa_outreach_sent: bool = False,
        judge: Any | None = None,
        must_ask: bool = False,
        reply_facts: ReplyFacts | None = None,
    ) -> SimulatorOutput: ...
