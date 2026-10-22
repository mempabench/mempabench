"""Memory event — the shared input unit for all adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from memory.turn_experience import ExperienceItem


@dataclass
class MemoryEvent:
    """A Nanobot runtime event to be ingested by a memory backend.

    This is *not* a universal MemoryItem.  Adapters map these fields
    into whatever native write unit the backend requires.

    `messages` is the canonical, lossless form and is what every adapter
    writes by default.  `items` is the same content pre-parsed, for a backend
    whose native API can carry structure.  An adapter may read `items` to
    decide *how* to present the turn; it may never present fewer items than it
    received — dropping is the backend's own job, and that is what the
    benchmark measures.
    """

    event_id: str
    run_id: str
    persona_id: str
    agent_id: str
    session_id: str
    turn_id: str
    timestamp: datetime
    messages: list[dict[str, Any]]
    items: list[ExperienceItem] = field(default_factory=list)
