"""LLM provider abstraction module for simulator."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from simulator.providers.base import LLMProvider, LLMResponse

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "AnthropicProvider",
    "OpenAICompatProvider",
]

_LAZY_IMPORTS = {
    "AnthropicProvider": ".anthropic_provider",
    "OpenAICompatProvider": ".openai_compat_provider",
}

if TYPE_CHECKING:
    from simulator.providers.anthropic_provider import AnthropicProvider
    from simulator.providers.openai_compat_provider import OpenAICompatProvider


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
