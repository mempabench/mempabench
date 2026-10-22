"""Small runtime wrapper for benchmark memory modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from memory.adapter import PromptHistoryEvictionPolicy, SupportsReadiness
from memory.results import BackendHealth, BackendReadiness, ConsistencyModel
from memory.scope import AdapterMode, MemoryScope


@dataclass(frozen=True)
class MemoryRuntimeCapabilities:
    supports_checkpoint: bool = False
    supports_restore: bool = False
    strategy: Literal["native", "scope_clone", "none"] = "none"


# What a checkpoint can actually do for each backend. A backend appears here
# only once its own export/restore/branch path exists; anything absent keeps
# the conservative default of declaring no support.
_BACKEND_CAPABILITIES: dict[str, MemoryRuntimeCapabilities] = {
    "nanobot_file_memory": MemoryRuntimeCapabilities(True, True, "scope_clone"),
    "no_memory": MemoryRuntimeCapabilities(True, True, "none"),
    "mem0": MemoryRuntimeCapabilities(True, True, "scope_clone"),
    "simple_rag": MemoryRuntimeCapabilities(True, True, "scope_clone"),
    "memos": MemoryRuntimeCapabilities(True, True, "scope_clone"),
}


def capabilities_for(memory_condition: str | None) -> MemoryRuntimeCapabilities:
    """What a checkpoint can do for one memory condition.

    Backend names and memory-condition names are the same strings, so a
    checkpoint on disk can be judged without holding a live adapter.
    """
    return _BACKEND_CAPABILITIES.get(
        memory_condition or "", MemoryRuntimeCapabilities()
    )


class MemoryRuntime:
    """Apply run/step memory policy to a `MemoryAdapter`."""

    def __init__(
        self,
        adapter: Any | None,
        *,
        run_id: str,
        persona_id: str,
        agent_id: str = "nanobot",
        backend_namespace: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.scope = MemoryScope(
            run_id=run_id,
            persona_id=persona_id,
            backend_namespace=backend_namespace,
            agent_id=agent_id,
        )
        self.capabilities = capabilities_for(
            getattr(adapter, "backend_name", "") or ""
        )

    async def setup(self) -> None:
        if self.adapter is not None:
            await self.adapter.setup_scope(self.scope)

    async def apply_mode(self, memory_mode: str, *, reason: str = "") -> None:
        if self.adapter is None:
            return
        await self.adapter.setup_scope(self.scope)
        await self.adapter.set_mode(
            AdapterMode(read_only=(memory_mode == "read_only"), reason=reason)
        )

    async def await_checkpoint_ready(
        self,
        timeout_s: float,
        *,
        session: Any | None = None,
    ) -> None:
        """Block until this backend's in-flight writes have landed.

        Backends that declare `SupportsReadiness` are asked directly. Backends
        that do not must acknowledge their writes immediately, because a local
        cursor only proves the write reached the adapter, never that a remote
        derivation finished.
        """
        if self.adapter is None:
            return
        if isinstance(self.adapter, SupportsReadiness):
            readiness: BackendReadiness = await self.adapter.wait_until_ready(
                self.scope, timeout_s
            )
            if not readiness.ready:
                raise RuntimeError(
                    "Memory backend is not ready at the checkpoint: "
                    f"backend={self.adapter.backend_name!r} "
                    f"pending_count={readiness.pending_count!r} "
                    f"warnings={readiness.warnings!r}"
                )
            return

        health = await self.adapter.health()
        consistency = getattr(health, "consistency_model", None)
        if consistency != ConsistencyModel.immediate:
            raise RuntimeError(
                "Memory backend acknowledges writes asynchronously but does "
                "not implement SupportsReadiness, so a checkpoint cannot prove "
                "its writes landed: "
                f"backend={self.adapter.backend_name!r} "
                f"consistency_model={consistency!r}"
            )
        policy = getattr(self.adapter, "prompt_history_eviction_policy", None)
        if (
            policy == PromptHistoryEvictionPolicy.after_adapter_commit
            and session is not None
            and session.adapter_committed_end_index != len(session.messages)
        ):
            raise RuntimeError(
                "Memory adapter has uncommitted turns at the checkpoint: "
                f"backend={self.adapter.backend_name!r} "
                f"adapter_committed_end_index="
                f"{session.adapter_committed_end_index!r} "
                f"messages={len(session.messages)!r}"
            )

    async def health(self) -> BackendHealth | None:
        if self.adapter is None:
            return None
        return await self.adapter.health()
