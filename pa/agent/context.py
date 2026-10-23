"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from pa.utils.helpers import current_time_str

from pa.agent.ix.specs import (
    OUTWARD_MESSAGE_NOTE,
    READINESS_QUESTION_NOTE,
    WRAPPER_NOTICE,
)
from pa.agent.memory import MarkdownMemoryStore
from pa.agent.skills import SkillsLoader
from pa.utils.helpers import build_assistant_message, detect_image_mime


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(
        self,
        workspace: Path,
        timezone: str | None = None,
        *,
        native_file_memory_enabled: bool = True,
    ):
        self.workspace = workspace
        self.timezone = timezone
        self.native_file_memory_enabled = native_file_memory_enabled
        self.file_memory_store = (
            MarkdownMemoryStore(workspace, log_scope="pa_file_memory")
            if native_file_memory_enabled
            else None
        )
        self.skills = SkillsLoader(workspace)
        self._benchmark_identity: str | None = None  # Override for benchmark experiments

    def set_benchmark_identity(self, identity_prompt: str) -> None:
        """Override the default PA identity with a benchmark-specific prompt.

        Args:
            identity_prompt: Full identity section text (replaces _get_identity output).
        """
        self._benchmark_identity = identity_prompt

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        adapter_memory_context: str | None = None,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._benchmark_identity or self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = (
            self.file_memory_store.get_memory_context()
            if self.file_memory_store is not None
            else ""
        )
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        if adapter_memory_context:
            parts.append(f"# Retrieved Memory\n\n{adapter_memory_context}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. Any skill that applies to every turn is already included above under "# Active Skills"; the SKILL.md locations below are reference only and cannot be opened with your tools.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = self._prompt_workspace_path()
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        platform_policy = ""
        if system == "Windows":
            platform_policy = """## Platform Policy (Windows)
- You are running on Windows. Do not assume GNU tools like `grep`, `sed`, or `awk` exist.
- If terminal output is garbled, retry with UTF-8 output enabled.
"""
        else:
            platform_policy = """## Platform Policy (POSIX)
- You are running on a POSIX system. Prefer UTF-8.
"""

        memory_paths = ""
        if self.native_file_memory_enabled:
            # No file name at all: a printed path was copied into document_read
            # (acc_004), and a printed basename was guessed into one
            # (acc_009, `my_desktop/MEMORY.md`). The memory is described
            # by what it is, and the PA is told there is no file for it to open.
            memory_paths = """- Long-term memory — maintained for you automatically outside the conversation; its current content is already included in the "# Memory" section of this prompt.
- History log — maintained for you automatically outside the conversation; each entry starts with [YYYY-MM-DD HH:MM].
- Neither is a document: there is no memory file for you to read, search or write with any tool, under any name or path. What you know from memory is in this prompt; what should be remembered you simply say.
"""

        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
{memory_paths}- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

{platform_policy}

## nanobot Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Use the available benchmark tools to read or mutate task state. Do not claim that an external action happened unless a tool result confirms it.
- {READINESS_QUESTION_NOTE}
- {OUTWARD_MESSAGE_NOTE}
- `document_update` (anchored replace) is the only way to change an existing document; there is no whole-file write tool, so a rewritten file is sent as anchored replacements, never as one block of new content.
- When creating artifacts such as email drafts, write the content yourself; tools persist or send artifacts, they do not author them for you.
- Interaction preference tools whose names start with `IX_` are benchmark answer cards. When an `IX_` tool is available, call it before the final answer for that turn, then make the response match the selected setting.
- Every `IX_` tool requires one of its listed settings. When you have no evidence about this user for that attribute, pick the setting that best fits the request in front of you and leave `evidence` empty; a first-exposure guess is expected and counts as your selection like any other.
- Only tools present in the current turn's tool list may be called; a selection tool that is no longer listed must not be called again.
- {WRAPPER_NOTICE}
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. If you save or modify task state through a tool, summarize the user-visible result and mention any identifier the tool returned."""

    def _prompt_workspace_path(self) -> str:
        override = os.environ.get("NANOBOT_PROMPT_WORKSPACE")
        if override:
            return override

        workspace = self.workspace.expanduser().resolve()
        parts = workspace.parts
        managed_runtime_path = any(
            part == "workspace"
            and index + 1 < len(parts)
            and parts[index + 1] in {"runs", "diagnostics"}
            for index, part in enumerate(parts)
        )
        if (
            len(parts) >= 4
            and parts[-2] == "pa_workspace"
            and managed_runtime_path
        ):
            return f"/workspace/{parts[-1]}"
        return str(workspace)

    @staticmethod
    def _build_runtime_context(
        channel: str | None, chat_id: str | None, timezone: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        if now is not None:
            lines = [f"Current Time: {now.strftime('%Y-%m-%d %H:%M (%A)')}"]
        else:
            lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        adapter_memory_context: str | None = None,
        timestamp: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(
            channel, chat_id, self.timezone, now=timestamp,
        )
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        return [
            {"role": "system", "content": self.build_system_prompt(
                skill_names, adapter_memory_context=adapter_memory_context,
            )},
            *history,
            {"role": current_role, "content": merged},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            # Detect real MIME type from magic bytes; fallback to filename guess
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: Any,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
