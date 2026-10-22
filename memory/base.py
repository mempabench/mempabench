"""Checkpointable native memory-state interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryBackend(ABC):
    """Native memory state that supports checkpoint and restore.

    The NanoBot file-memory condition uses this interface for Markdown state.
    External experiment conditions such as Mem0 and MemOS implement
    ``MemoryAdapter`` instead.
    """

    @abstractmethod
    async def add(self, messages: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
        """Ingest conversation messages into memory.

        Args:
            messages: List of message dicts (role, content, timestamp, etc.)
            metadata: Optional metadata (session_id, context, etc.)
        """

    @abstractmethod
    async def search(self, query: str, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve relevant memories for a query.

        Args:
            query: Search query string.
            user_id: User identifier for scoping.
            limit: Maximum number of results.

        Returns:
            List of memory entries (format depends on backend).
        """

    @abstractmethod
    def get_context_string(self, user_id: str) -> str:
        """Build a string for injection into the system prompt.

        Args:
            user_id: User identifier for scoping.

        Returns:
            Formatted memory context string (may be empty).
        """

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Capture full memory state for checkpointing.

        Returns:
            Serializable dict representing all memory state.
        """

    @abstractmethod
    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore memory state from a checkpoint snapshot.

        Args:
            snapshot: Dict previously returned by snapshot().
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory state."""
