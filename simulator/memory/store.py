"""Simulator-owned memory: the user's own recollection of a session.

Kept separate from the PA's memory store on purpose. The PA writes an
assistant's working memory; this writes a person's memory of *using* an
assistant, and the two prompts must be able to drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts/user_memory.md"

SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save what I will remember about this session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "One first-person entry for this whole session, "
                        "2-4 sentences, starting with [YYYY-MM-DD HH:MM]. What I asked "
                        "for, how working with the assistant went, what I decided, what "
                        "is still open. No subject matter, no file paths, no numbers.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "My full updated long-term memory: durable "
                        "impressions of the assistant and standing facts about my own "
                        "work, at the level a person keeps in their head. No research "
                        "content. Return unchanged if nothing durable changed.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


#: Ceiling for the growing consolidation budget. google/gemini-3.8-flash spends
#: its thinking tokens inside `max_tokens`, so a forced save_memory call can be
#: cut off before it emits a single tool-call token -- the same failure that
#: took the session-end reflection to 8192 (`SELF_REPORT_MAX_TOKENS` in
#: simulator/self_report.py). 16384 is twice that: the widest output this repo
#: asks of the simulator model, and past it the truncation is not the budget.
MAX_CONSOLIDATION_MAX_TOKENS = 16384

#: Appended to the prompt on a re-ask that follows a cut-off reply. The budget
#: grows and the note asks for less text, so the two pull the same way.
SHORTER_NOTE_INSTRUCTION = (
    "\n\nYour last attempt ran out of output space before the save_memory call "
    "was finished. Write a shorter memory note this time: hold history_entry to "
    "2 sentences, and keep memory_update to the impressions that still matter.\n"
)


class MemoryConsolidationError(RuntimeError):
    """Raised when a simulator memory write cannot produce a valid entry."""


class MemoryConsolidationContractError(MemoryConsolidationError):
    """Raised when the consolidation LM violates the save_memory contract.

    `finish_reason` carries the provider's own reason for stopping, so the
    caller can tell a reply cut off at the token cap -- the one failure an
    identical re-ask cannot fix -- from a contract the model simply broke.
    """

    def __init__(self, *args: Any, finish_reason: str | None = None) -> None:
        super().__init__(*args)
        self.finish_reason = finish_reason


def format_session_transcript(exchanges: list[dict]) -> str:
    """Render buffered turns from the user's own point of view."""
    labels = {"pa": "Assistant", "simulated_user": "Me"}
    lines = []
    for message in exchanges:
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        label = labels.get(str(message.get("role")), str(message.get("role")).upper())
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def render_user_memory_prompt(
    *,
    user_name: str,
    timestamp: str,
    current_memory: str,
    exchanges: list[dict],
) -> str:
    """Render the user-recollection consolidation prompt."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("<<USER_NAME>>", user_name)
        .replace("<<TIMESTAMP>>", timestamp)
        .replace("<<CURRENT_MEMORY>>", current_memory.strip() or "(empty)")
        .replace("<<SESSION_TRANSCRIPT>>", format_session_transcript(exchanges))
    )


def _ensure_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_args(args: Any) -> dict[str, Any] | None:
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


class UserMemoryStore:
    """MEMORY.md (durable impressions) + HISTORY.md (one entry per session)."""

    def __init__(self, workspace: Path, *, user_name: str):
        self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.user_name = user_name

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as handle:
            handle.write(entry.rstrip() + "\n\n")

    async def consolidate(
        self,
        exchanges: list[dict],
        provider: Any,
        model: str,
        *,
        timestamp: str,
        max_tokens: int | None = None,
        shorter_note: bool = False,
    ) -> bool:
        """Write one first-person session entry through the simulator's model.

        `max_tokens` overrides the provider's configured output budget; `None`
        inherits it. `shorter_note` appends the re-ask instruction, and both are
        set by the caller's attempt loop after a reply cut off at the cap.
        """
        if not exchanges:
            return True
        current_memory = self.read_long_term()
        prompt = render_user_memory_prompt(
            user_name=self.user_name,
            timestamp=timestamp,
            current_memory=current_memory,
            exchanges=exchanges,
        )
        if shorter_note:
            prompt += SHORTER_NOTE_INSTRUCTION
        budget = {} if max_tokens is None else {"max_tokens": max_tokens}
        response = await provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            tools=SAVE_MEMORY_TOOL,
            model=model,
            tool_choice={"type": "function", "function": {"name": "save_memory"}},
            **budget,
        )
        if not response.has_tool_calls:
            raise MemoryConsolidationContractError(
                "No save_memory call: "
                f"finish_reason={response.finish_reason!r} "
                f"content_preview={(response.content or '')[:1_000]!r}",
                finish_reason=response.finish_reason,
            )
        raw_args = response.tool_calls[0].arguments
        try:
            args = _normalize_args(raw_args)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise MemoryConsolidationContractError(
                f"Invalid save_memory arguments: value={str(raw_args)[:200]!r} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        if args is None:
            raise MemoryConsolidationContractError(
                f"Unexpected save_memory arguments: value={str(raw_args)[:200]!r}"
            )
        missing = [f for f in ("history_entry", "memory_update") if args.get(f) is None]
        if missing:
            raise MemoryConsolidationContractError(
                f"Missing save_memory fields: missing_fields={missing!r} "
                f"value={str(args)[:200]!r}"
            )
        entry = _ensure_text(args["history_entry"]).strip()
        if not entry:
            raise MemoryConsolidationContractError(
                f"Empty save_memory history_entry: value={str(args)[:200]!r}"
            )
        self.append_history(entry)
        update = _ensure_text(args["memory_update"])
        if update != current_memory:
            self.memory_file.write_text(update, encoding="utf-8")
        logger.info(
            "Simulator user memory written: turns={} history_appended=true",
            len(exchanges),
        )
        return True
