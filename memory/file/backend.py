"""File state for NanoBot MEMORY.md and HISTORY.md.

This is the baseline backend that uses MEMORY.md + HISTORY.md files
with LLM-driven consolidation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memory.base import MemoryBackend


class FileMemoryBackend(MemoryBackend):
    """File-based memory using MEMORY.md (long-term) + HISTORY.md (log).

    Delegates actual consolidation to the PA's MarkdownMemoryStore, which
    uses LLM calls to summarize conversations into structured memory.
    This backend provides the checkpoint/restore interface on top.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        if not (self.memory_dir / ".mempa-read-only").is_file():
            self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    async def add(self, messages: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
        """For file backend, consolidation is handled by MemoryConsolidator in the PA.

        This method is a no-op because MemoryConsolidator invokes
        MarkdownMemoryStore as part of the PA processing pipeline.
        """
        pass

    async def search(self, query: str, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """File backend doesn't support semantic search — return full memory."""
        content = self.get_context_string(user_id)
        return [{"content": content}] if content else []

    def get_context_string(self, user_id: str) -> str:
        """Read MEMORY.md content for prompt injection."""
        if self.memory_file.exists():
            content = self.memory_file.read_text(encoding="utf-8").strip()
            if content:
                return f"## Long-term Memory\n{content}"
        return ""

    def snapshot(self) -> dict[str, Any]:
        """Capture MEMORY.md + HISTORY.md content."""
        memory = ""
        history = ""
        if self.memory_file.exists():
            memory = self.memory_file.read_text(encoding="utf-8")
        if self.history_file.exists():
            history = self.history_file.read_text(encoding="utf-8")
        return {
            "backend": "file",
            "memory": memory,
            "history": history,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore MEMORY.md + HISTORY.md from snapshot."""
        self._require_writable()
        self.memory_file.write_text(snapshot.get("memory", ""), encoding="utf-8")
        self.history_file.write_text(snapshot.get("history", ""), encoding="utf-8")

    def reset(self) -> None:
        """Clear both memory files."""
        self._require_writable()
        self.memory_file.write_text("", encoding="utf-8")
        self.history_file.write_text("", encoding="utf-8")

    def _require_writable(self) -> None:
        if (self.memory_dir / ".mempa-read-only").is_file():
            raise PermissionError("Shared Probe file memory is immutable")
