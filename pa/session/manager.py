"""Session management for conversation history."""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from memory.turn_journal import journal_path
from pa.utils.helpers import ensure_dir, safe_filename

#: The turn journal lives beside the session file it belongs to
#: (`<safe_key>.journal.jsonl`, and `<safe_key>.journal.<n>.jsonl` once `/new`
#: rotates one aside). It is not a session, so the sessions directory is
#: enumerated past it.
_JOURNAL_FILE_PATTERNS = ("*.journal.jsonl", "*.journal.*.jsonl")

#: The journal's name inside a checkpoint directory, beside `session.jsonl`.
#: Fixed rather than key-derived, because a checkpoint may be restored under a
#: different session key than the one it was taken from.
_CHECKPOINT_JOURNAL_NAME = "session.journal.jsonl"


class UnsupportedSessionSchemaError(RuntimeError):
    """Raised when a session artifact does not use the current schema."""


class InvalidSessionIndexError(RuntimeError):
    """Raised when a persisted session index is outside its message list."""


class SessionAlreadyExistsError(RuntimeError):
    """Raised when explicit session creation would overwrite existing state."""


class SessionNotFoundError(RuntimeError):
    """Raised when required persisted session state does not exist."""


class SessionLoadError(RuntimeError):
    """Raised when an existing session artifact cannot be decoded."""


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Messages are append-only for auditability. Independent exclusive indices
    track file consolidation, committed adapter writes, and prompt history.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    file_consolidated_end_index: int = 0
    adapter_committed_end_index: int = 0
    prompt_history_start_index: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        """Find first index where every tool result has a matching assistant tool_call."""
        declared: set[str] = set()
        legal_start_index = 0
        for message_index, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant":
                for tc in message.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = message.get("tool_call_id")
                if tid and str(tid) not in declared:
                    legal_start_index = message_index + 1
                    declared.clear()
                    for prev in messages[
                        legal_start_index:message_index + 1
                    ]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return legal_start_index

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return prompt-visible messages aligned to a legal tool-call boundary."""
        prompt_history = self.messages[self.prompt_history_start_index:]
        sliced = prompt_history[-max_messages:]

        # Drop leading non-user messages to avoid starting mid-turn when possible.
        for message_index, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[message_index:]
                break

        # Some providers reject orphan tool results if the matching assistant
        # tool_calls message fell outside the fixed-size history window.
        legal_start_index = self._find_legal_start(sliced)
        if legal_start_index:
            sliced = sliced[legal_start_index:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.file_consolidated_end_index = 0
        self.adapter_committed_end_index = 0
        self.prompt_history_start_index = 0
        self.updated_at = datetime.now()

    def retain_recent_legal_suffix(self, max_messages: int) -> None:
        """Keep a legal recent suffix, mirroring get_history boundary rules."""
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return

        retained_start_index = max(
            0,
            len(self.messages) - max_messages,
        )

        # If the cutoff lands mid-turn, extend backward to the nearest user turn.
        while (
            retained_start_index > 0
            and self.messages[retained_start_index].get("role") != "user"
        ):
            retained_start_index -= 1

        retained = self.messages[retained_start_index:]

        # Mirror get_history(): avoid persisting orphan tool results at the front.
        legal_start_index = self._find_legal_start(retained)
        if legal_start_index:
            retained = retained[legal_start_index:]

        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.file_consolidated_end_index = max(
            0,
            self.file_consolidated_end_index - dropped,
        )
        self.adapter_committed_end_index = max(
            0,
            self.adapter_committed_end_index - dropped,
        )
        self.prompt_history_start_index = max(
            0,
            self.prompt_history_start_index - dropped,
        )
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    _SESSION_SCHEMA_VERSION = 3

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def create(self, key: str) -> Session:
        """Create and persist a new empty session."""
        path = self._get_session_path(key)
        if key in self._cache or path.exists():
            raise SessionAlreadyExistsError(
                f"Session already exists: key={key!r} path={path}"
            )
        session = Session(key=key)
        self.save(session)
        return session

    def get(self, key: str) -> Session:
        """Return an existing session or fail if its persisted state is absent."""
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            raise SessionNotFoundError(
                f"Session not found: key={key!r} path={self._get_session_path(key)}"
            )

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        return self._load_from_path(path, key)

    def _load_from_path(self, path: Path, key: str | None = None) -> Session | None:
        """Load a session from an arbitrary JSONL file path."""
        if not path.exists():
            return None

        messages = []
        metadata = {}
        created_at = None
        index_values = {
            "file_consolidated_end_index": 0,
            "adapter_committed_end_index": 0,
            "prompt_history_start_index": 0,
        }

        try:
            with open(path, encoding="utf-8") as f:
                for line_number, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        preview = line[:160]
                        raise SessionLoadError(
                            "Invalid session JSONL: "
                            f"key={key!r} path={path} line={line_number} "
                            f"value={preview!r}"
                        ) from exc
                    if not isinstance(data, dict):
                        raise SessionLoadError(
                            "Invalid session JSONL record: "
                            f"key={key!r} path={path} line={line_number} "
                            f"value={data!r}"
                        )

                    if data.get("_type") == "metadata":
                        schema_version = data.get("session_schema_version")
                        if schema_version != self._SESSION_SCHEMA_VERSION:
                            raise UnsupportedSessionSchemaError(
                                "Unsupported session schema: "
                                f"expected {self._SESSION_SCHEMA_VERSION}, "
                                f"found {schema_version!r} in {path}"
                            )
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        index_values = {
                            field_name: int(data.get(field_name, 0))
                            for field_name in index_values
                        }
                    else:
                        messages.append(data)
        except (SessionLoadError, UnsupportedSessionSchemaError):
            raise
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise SessionLoadError(
                f"Failed to load session: key={key!r} path={path} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc

        for field_name, index_value in index_values.items():
            if not 0 <= index_value <= len(messages):
                raise InvalidSessionIndexError(
                    f"{field_name}={index_value} is outside "
                    f"0..{len(messages)} in {path}"
                )

        return Session(
            key=key,
            messages=messages,
            created_at=created_at or datetime.now(),
            metadata=metadata,
            **index_values,
        )

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "session_schema_version": self._SESSION_SCHEMA_VERSION,
                "file_consolidated_end_index": session.file_consolidated_end_index,
                "adapter_committed_end_index": session.adapter_committed_end_index,
                "prompt_history_start_index": session.prompt_history_start_index,
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            if any(path.match(pattern) for pattern in _JOURNAL_FILE_PATTERNS):
                continue
            # Read just the metadata line. Existing corrupt state must be visible.
            with open(path, encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                raise SessionLoadError(
                    f"Session metadata is empty: path={path} value={first_line!r}"
                )
            try:
                data = json.loads(first_line)
            except json.JSONDecodeError as exc:
                raise SessionLoadError(
                    f"Invalid session metadata JSON: path={path} value={first_line[:160]!r}"
                ) from exc
            if not isinstance(data, dict):
                raise SessionLoadError(
                    f"Invalid session metadata record: path={path} value={data!r}"
                )
            if data.get("_type") != "metadata":
                raise SessionLoadError(
                    f"Invalid session metadata type: path={path} "
                    f"value={data.get('_type')!r}"
                )
            key = data.get("key") or path.stem.replace("_", ":", 1)
            sessions.append({
                "key": key,
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "path": str(path)
            })

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    def checkpoint(self, session: Session, checkpoint_dir: Path) -> Path:
        """Save a copy of the session to a checkpoint directory.

        Args:
            session: The session to checkpoint.
            checkpoint_dir: Directory to save the checkpoint file.

        Returns:
            Path to the checkpoint file.
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "session.jsonl"

        # Save current state first
        self.save(session)

        # Copy session file to checkpoint
        src = self._get_session_path(session.key)
        shutil.copy2(str(src), str(checkpoint_path))

        # The journal is saved and restored with the checkpoint. A
        # session restored without it loses the pool writes its records still
        # owe. A session that has written none has nothing to copy.
        src_journal = journal_path(self.sessions_dir, session.key)
        if src_journal.exists():
            shutil.copy2(
                str(src_journal), str(checkpoint_dir / _CHECKPOINT_JOURNAL_NAME)
            )
        return checkpoint_path

    def restore_checkpoint(self, key: str, checkpoint_dir: Path) -> Session:
        """Restore a session from a checkpoint.

        Args:
            key: Session key to restore as.
            checkpoint_dir: Directory containing the checkpoint file.

        Returns:
            Restored session.
        """
        checkpoint_path = checkpoint_dir / "session.jsonl"
        session = self._load_from_path(checkpoint_path, key)
        if session is None:
            raise SessionNotFoundError(
                f"Checkpoint session not found: key={key!r} path={checkpoint_path}"
            )
        session.key = key

        # The journal the checkpoint carries, restored beside the session it
        # belongs to. A checkpoint holding none leaves none behind: a journal
        # written after the checkpoint records turns the restored session never
        # took.
        src_journal = checkpoint_dir / _CHECKPOINT_JOURNAL_NAME
        dest_journal = journal_path(self.sessions_dir, key)
        if src_journal.exists():
            shutil.copy2(str(src_journal), str(dest_journal))
        elif dest_journal.exists():
            dest_journal.unlink()

        self._cache[key] = session
        return session
