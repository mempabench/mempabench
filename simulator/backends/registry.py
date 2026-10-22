"""Factory helper for the simulator's nanobot backend."""

from __future__ import annotations

from typing import Any

from simulator.backends.nanobot_runner import NanobotRunnerSimulatorBackend


def create_simulator_backend(
    provider: Any,
    *,
    profile=None,
    output_contract_retries: int = 2,
):
    return NanobotRunnerSimulatorBackend(
        provider,
        profile=profile,
        output_contract_retries=output_contract_retries,
    )
