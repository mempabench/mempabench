"""Scope, mode, budget, and retrieve-request types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryScope:
    """Identifies an isolated memory partition for a benchmark run."""

    run_id: str
    persona_id: str
    backend_namespace: str | None = None
    agent_id: str | None = None


@dataclass(frozen=True)
class AdapterMode:
    """Controls read-only vs read-write behaviour.

    Set by the harness before/after test sessions so that test-session
    turns are never written into retrievable backend memory.
    """

    read_only: bool = False
    reason: str = ""


@dataclass(frozen=True)
class MemoryRetrievalBudget:
    """Token and item limits for one memory retrieval operation."""

    max_tokens: int | None = None
    max_items: int | None = None


@dataclass
class RetrieveRequest:
    """What the agent loop sends to the adapter when it needs context."""

    query: str
    scope: MemoryScope
    session_id: str
    budget: MemoryRetrievalBudget
