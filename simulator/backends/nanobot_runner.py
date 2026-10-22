"""Nanobot-style simulator backend using the shared AgentRunner loop."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from simulator.agent import SimulatorAgentLoop
from simulator.output import SimulatorOutput
from simulator.profiles import SimulatorProfile, infer_identity_path
from simulator.reply_facts import ReplyFacts
from simulator.schemas import Beat, PersonaIdentity, PreferenceMatrix, SessionScript


class NanobotRunnerSimulatorBackend:
    """Generate simulator turns through the shared agent runner.

    The simulator is still a user actor, not a PA. It gets no task tools by
    default; this backend only reuses the lower-level model execution loop.
    """

    name = "nanobot"

    def __init__(
        self,
        provider: Any,
        *,
        profile=None,
        max_iterations: int = 1,
        output_contract_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.profile = profile
        self.max_iterations = max_iterations
        self.output_contract_retries = output_contract_retries
        self._agents: list[SimulatorAgentLoop] = []
        self._session_id: str | None = None

    def get_default_model(self) -> str:
        return self.provider.get_default_model()

    async def drain_memory_tasks(self, *, timestamp: str | None = None) -> None:
        """Session close: write one consolidated memory entry for the session.

        `timestamp` is in-world benchmark time. Without one, the last buffered
        turn's benchmark timestamp is used; wall-clock time is never used.
        """
        exchanges: list[dict] = []
        model: str | None = None
        buffered_timestamp: str | None = None
        for agent in list(self._agents):
            exchanges.extend(agent.take_pending_memory())
            model = agent.memory_model or model
            buffered_timestamp = agent.memory_timestamp or buffered_timestamp
        if not exchanges or self.profile is None:
            return
        await self.profile.record_session(
            exchanges,
            provider=self.provider,
            model=model or self.get_default_model(),
            session_id=self._session_id or "unknown_session",
            timestamp=timestamp or buffered_timestamp,
        )

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
    ) -> SimulatorOutput:
        fallback_id = identity.name.lower().replace(" ", "_")
        profile = self.profile or SimulatorProfile(
            root=Path("simulator/profiles") / fallback_id,
            id=fallback_id,
            display_name=identity.name,
            identity_path=infer_identity_path(fallback_id),
        )
        agent = SimulatorAgentLoop(
            self.provider,
            profile,
            matrix,
            actor_skill_specs_root=actor_skill_specs_root,
            max_iterations=self.max_iterations,
            record_memory=self.profile is not None,
            output_contract_retries=self.output_contract_retries,
        )
        self._agents.append(agent)
        self._session_id = session.session_id
        return await agent.generate_turn(
            session,
            beat,
            history=history,
            model=model,
            first_turn_in_beat=first_turn_in_beat,
            benchmark_timestamp=benchmark_timestamp,
            held_items=held_items,
            pa_executed_tools=pa_executed_tools,
            pa_tool_log=pa_tool_log,
            pa_outreach_sent=pa_outreach_sent,
            judge=judge,
            must_ask=must_ask,
            reply_facts=reply_facts,
        )
