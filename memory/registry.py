"""Plugin-style registry for memory adapters and controlled conditions."""

from __future__ import annotations

from typing import Any, Callable

from memory.adapter import MemoryAdapter


AdapterFactory = Callable[..., MemoryAdapter]


class _AdapterEntry:
    __slots__ = ("factory", "kind")

    def __init__(self, factory: AdapterFactory, kind: str) -> None:
        self.factory = factory
        self.kind = kind  # "backend" or "condition"


class MemoryAdapterRegistry:
    """Maps memory-condition names to adapter factories.

    Usage::

        reg = MemoryAdapterRegistry()
        reg.register("no_memory", no_memory_factory, kind="condition")
        adapter = reg.create("no_memory")
    """

    def __init__(self) -> None:
        self._entries: dict[str, _AdapterEntry] = {}

    def register(
        self,
        name: str,
        factory: AdapterFactory,
        *,
        kind: str = "backend",
        allow_override: bool = False,
    ) -> None:
        if name in self._entries and not allow_override:
            raise ValueError(f"Memory adapter '{name}' is already registered")
        self._entries[name] = _AdapterEntry(factory=factory, kind=kind)

    def create(self, name: str, **kwargs: Any) -> MemoryAdapter:
        entry = self._entries.get(name)
        if entry is None:
            available = ", ".join(sorted(self._entries)) or "(none)"
            raise KeyError(f"Unknown memory adapter '{name}'. Available: {available}")
        return entry.factory(**kwargs)

    def adapter_kind(self, name: str) -> str:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"Unknown memory adapter '{name}'")
        return entry.kind

    def available(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


def _simple_rag_factory(**kwargs: Any) -> MemoryAdapter:
    from memory.backends.mem0_adapter import Mem0Adapter

    return Mem0Adapter(infer=False, **kwargs)


def default_registry() -> MemoryAdapterRegistry:
    """Return a registry pre-loaded with built-in memory adapters."""
    from memory.backends.mem0_adapter import Mem0Adapter
    from memory.backends.memos_adapter import MemOSAdapter
    from memory.conditions.nanobot_file_memory import NanobotFileMemoryAdapter
    from memory.conditions.no_memory import NoMemoryAdapter
    from memory.conditions.oracle_preference import OraclePreferenceAdapter

    reg = MemoryAdapterRegistry()
    reg.register("no_memory", NoMemoryAdapter, kind="condition")
    reg.register("oracle_preference", OraclePreferenceAdapter, kind="condition")
    reg.register("nanobot_file_memory", NanobotFileMemoryAdapter, kind="condition")
    reg.register("mem0", Mem0Adapter, kind="backend")
    reg.register("simple_rag", _simple_rag_factory, kind="condition")
    reg.register("memos", MemOSAdapter, kind="backend")
    return reg
