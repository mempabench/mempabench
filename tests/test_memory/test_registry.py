"""Tests for the memory adapter plugin registry."""

from __future__ import annotations

import pytest

from memory.adapter import MemoryAdapter
from memory.conditions.no_memory import NoMemoryAdapter
from memory.registry import MemoryAdapterRegistry, default_registry


class TestMemoryAdapterRegistry:
    def test_register_and_create(self) -> None:
        reg = MemoryAdapterRegistry()
        reg.register("fake", NoMemoryAdapter, kind="condition")
        adapter = reg.create("fake")
        assert isinstance(adapter, NoMemoryAdapter)

    def test_unknown_adapter_raises(self) -> None:
        reg = MemoryAdapterRegistry()
        with pytest.raises(KeyError, match="Unknown memory adapter 'missing'"):
            reg.create("missing")

    def test_duplicate_registration_rejected(self) -> None:
        reg = MemoryAdapterRegistry()
        reg.register("x", NoMemoryAdapter)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("x", NoMemoryAdapter)

    def test_duplicate_with_override(self) -> None:
        reg = MemoryAdapterRegistry()
        reg.register("x", NoMemoryAdapter)
        reg.register("x", NoMemoryAdapter, allow_override=True)
        assert "x" in reg

    def test_available_lists_names(self) -> None:
        reg = MemoryAdapterRegistry()
        reg.register("b_prov", NoMemoryAdapter)
        reg.register("a_prov", NoMemoryAdapter)
        assert reg.available() == ["a_prov", "b_prov"]

    def test_adapter_kind(self) -> None:
        reg = MemoryAdapterRegistry()
        reg.register("cond", NoMemoryAdapter, kind="condition")
        reg.register("back", NoMemoryAdapter, kind="backend")
        assert reg.adapter_kind("cond") == "condition"
        assert reg.adapter_kind("back") == "backend"

    def test_adapter_kind_unknown_raises(self) -> None:
        reg = MemoryAdapterRegistry()
        with pytest.raises(KeyError):
            reg.adapter_kind("nope")

    def test_contains(self) -> None:
        reg = MemoryAdapterRegistry()
        assert "x" not in reg
        reg.register("x", NoMemoryAdapter)
        assert "x" in reg


class TestDefaultRegistry:
    def test_no_memory_available(self) -> None:
        reg = default_registry()
        assert "no_memory" in reg

    def test_no_memory_is_condition(self) -> None:
        reg = default_registry()
        assert reg.adapter_kind("no_memory") == "condition"

    def test_create_no_memory(self) -> None:
        reg = default_registry()
        adapter = reg.create("no_memory")
        assert isinstance(adapter, MemoryAdapter)
        assert adapter.backend_name == "no_memory"

    def test_oracle_preference_is_a_local_condition(self) -> None:
        from harness.checkpoint import LOCAL_MEMORY_CONDITIONS
        from memory.conditions.oracle_preference import OraclePreferenceAdapter

        reg = default_registry()
        assert reg.adapter_kind("oracle_preference") == "condition"
        # The entrypoint builds every condition's run-level adapter this way.
        adapter = reg.create("oracle_preference")
        assert isinstance(adapter, OraclePreferenceAdapter)
        assert adapter.backend_name == "oracle_preference"
        assert "oracle_preference" in LOCAL_MEMORY_CONDITIONS
