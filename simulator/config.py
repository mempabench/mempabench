"""Simulator configuration — provider + model + generation settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from simulator.providers.base import GenerationSettings
from simulator.providers.registry import ProviderSpec, find_by_name, PROVIDERS


@dataclass
class SimulatorConfig:
    """Configuration for the simulator's LLM calls.

    Defaults to OpenRouter with temperature=0 for reproducibility.
    """

    model: str = "deepseek/deepseek-v4-flash"
    api_key: str = ""
    api_base: str = ""
    provider_name: str = "openrouter"
    generation: GenerationSettings = field(
        # 4096: the three-block output plus a long-winded persona message must fit;
        # 2048 truncated mid-message and dropped the closing blocks.
        default_factory=lambda: GenerationSettings(temperature=0.0, max_tokens=4096)
    )

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")

    @property
    def provider_spec(self) -> ProviderSpec | None:
        return find_by_name(self.provider_name)

    def create_provider(self):
        """Create an LLM provider instance from this config."""
        spec = self.provider_spec
        if not spec:
            raise ValueError(f"Unknown provider: {self.provider_name}")

        if spec.backend == "anthropic":
            from simulator.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(
                api_key=self.api_key,
                api_base=self.api_base or None,
            )
        else:
            from simulator.providers.openai_compat_provider import OpenAICompatProvider
            provider = OpenAICompatProvider(
                api_key=self.api_key,
                api_base=self.api_base or spec.default_api_base or None,
                spec=spec,
            )

        provider.generation = self.generation
        return provider

    def create_backend(
        self,
        provider=None,
        *,
        profile=None,
        output_contract_retries: int = 3,
    ):
        """Create the selected simulator backend from this config."""
        from simulator.backends import create_simulator_backend

        return create_simulator_backend(
            provider or self.create_provider(),
            profile=profile,
            output_contract_retries=output_contract_retries,
        )
