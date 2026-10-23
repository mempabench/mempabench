"""The canonical tagged rendering of the PA's first-person turn experience.

A memory backend accepts two roles and non-empty string content
(`memory/backends/mem0_client.py:38`, `:130-164`), so anything richer than
dialogue has to live inside the content string.  Untagged content is dialogue;
every non-dialogue item is a tagged block on an `assistant`-role message, and
nothing that is not the person's own speech is ever `role: "user"`.

The grammar is one block per item::

    [<tag> <k>=<v> …]
    <body lines>
    [/<tag>]

`offset=` is the one attribute every tag may carry.  It is emitted last, only
when the item has an offset, and only on a tagged block: a bare dialogue
message keeps its exact bytes, which is what `simple_rag`'s byte-equality
contract (`mem0_client.py:183-188`) compares.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperienceItem:
    """One item of a turn: a bare dialogue message, or one tagged block."""

    kind: str          # "user" | "assistant" | one of the block tags
    role: str          # "user" | "assistant" — what the rendering used
    order: int         # position within the turn, 0-based
    offset_seconds: int | None
    attrs: dict[str, str]
    body: str


DIALOGUE_KINDS = frozenset({"user", "assistant"})

#: The version of this grammar, frozen into every RunPlan
#: (`harness/run_plan.py`).  A run frozen before the pool change wrote the PA's
#: turn to memory as two messages, so its memory is not the same artefact and
#: its results are not comparable with a pool run's.  Bumping this string is
#: what invalidates those runs: `load_run_plan` refuses a plan that carries a
#: different version or none at all, and Experiment Admin reports them as
#: non-resumable through the same check.  Bump it whenever a grammar change
#: makes a stored pool mean something else.
MEMORY_POOL_VERSION = "memory-pool/1.0.0"

#: Each tag's own attributes, in the order they are emitted.  `offset=` is
#: appended after these; an attribute the item does not carry is omitted.
BLOCK_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "prose": (),
    "reasoning": ("source",),
    "tool_call": ("name", "id"),
    "tool_result": ("name", "id", "truncated"),
    "ix_selection": ("attribute", "setting", "outcome"),
    "ix_result": ("name", "id"),
    "environment_event": ("source_kind", "source_id", "action_id"),
    # `action_id` is declared on both delivered-line tags: an environment
    # action whose `source.kind` is `third_party` renders as `[third_party]`
    # (the tag follows the item's own kind) and still has the action's id.
    "third_party": ("source_kind", "source_id", "action_id"),
    "subagent_result": (),
    "no_reply": ("decision",),
}

_QUOTE_TRIGGERS = ' ]"='


def render_blocks(items: Iterable[ExperienceItem]) -> str:
    """Render one message's items into its content string."""
    rendered = []
    for item in items:
        if item.kind in DIALOGUE_KINDS:
            rendered.append(item.body)
        else:
            rendered.append(_render_block(item))
    return "\n".join(rendered)


def parse_blocks(content: str, *, start_order: int = 0) -> list[ExperienceItem]:
    """Read tagged content back into items.

    Bare dialogue is never parsed — its kind comes from its own
    `ExperienceItem`, so text that looks like a block inside the person's own
    words stays the person's own words.
    """
    items: list[ExperienceItem] = []
    lines = content.split("\n")
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        tag, attrs, offset_seconds = _parse_open_line(lines[index])
        close_line = f"[/{tag}]"
        body_lines: list[str] = []
        index += 1
        while index < len(lines) and lines[index] != close_line:
            body_lines.append(lines[index])
            index += 1
        if index == len(lines):
            raise ValueError(f"Unterminated experience block: {tag!r}")
        index += 1
        items.append(
            ExperienceItem(
                kind=tag,
                role="assistant",
                order=start_order + len(items),
                offset_seconds=offset_seconds,
                attrs=attrs,
                body=_unescape_body("\n".join(body_lines)),
            )
        )
    return items


def _render_block(item: ExperienceItem) -> str:
    if item.kind not in BLOCK_ATTRIBUTES:
        raise ValueError(f"Unknown experience block tag: {item.kind!r}")
    declared = BLOCK_ATTRIBUTES[item.kind]
    undeclared = [key for key in item.attrs if key not in declared]
    if undeclared:
        raise ValueError(
            f"Undeclared attribute(s) on [{item.kind}]: {sorted(undeclared)!r}"
        )
    keys = [key for key in declared if key in item.attrs]
    header = [item.kind] + [_render_attribute(key, item.attrs[key]) for key in keys]
    if item.offset_seconds is not None:
        header.append(f"offset={item.offset_seconds}")
    open_line = "[" + " ".join(header) + "]"
    close_line = f"[/{item.kind}]"
    if not item.body:
        # The delimiters are always emitted, so an empty body is still a
        # non-empty content string: `mem0_client.py:152-156` and
        # `pa/agent/memory.py:123-124` both drop empty content, and
        # `[no_reply]` is an interaction choice that is scored.
        return f"{open_line}\n{close_line}"
    return f"{open_line}\n{_escape_body(item.body)}\n{close_line}"


def _render_attribute(key: str, value: str) -> str:
    if "\n" in value or "\r" in value:
        # A header is one line and the grammar defines no escape for a break
        # inside it, so such a value would render a block nothing can read
        # back. No attribute source carries one; this says so out loud.
        raise ValueError(f"Line break in experience attribute {key!r}: {value!r}")
    if any(char in value for char in _QUOTE_TRIGGERS):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{key}="{escaped}"'
    return f"{key}={value}"


def _escape_body(body: str) -> str:
    """Double the leading bracket of any body line that opens with one."""
    lines = []
    for line in body.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("["):
            line = line[: len(line) - len(stripped)] + "[" + stripped
        lines.append(line)
    return "\n".join(lines)


def _unescape_body(body: str) -> str:
    lines = []
    for line in body.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("[["):
            line = line[: len(line) - len(stripped)] + stripped[1:]
        lines.append(line)
    return "\n".join(lines)


def _parse_open_line(line: str) -> tuple[str, dict[str, str], int | None]:
    if not (line.startswith("[") and line.endswith("]")):
        raise ValueError(f"Not an experience block delimiter: {line!r}")
    tokens = _split_header(line[1:-1])
    tag = tokens[0]
    if tag not in BLOCK_ATTRIBUTES:
        raise ValueError(f"Unknown experience block tag: {tag!r}")
    attrs: dict[str, str] = {}
    offset_seconds: int | None = None
    for token in tokens[1:]:
        key, _, raw = token.partition("=")
        if key == "offset":
            offset_seconds = int(_unquote(raw))
        else:
            attrs[key] = _unquote(raw)
    return tag, attrs, offset_seconds


def _split_header(header: str) -> list[str]:
    """Split a block header on spaces that are outside a quoted value."""
    tokens: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for char in header:
        if escaped:
            current.append(char)
            escaped = False
        elif quoted and char == "\\":
            current.append(char)
            escaped = True
        elif char == '"':
            quoted = not quoted
            current.append(char)
        elif char == " " and not quoted:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _unquote(raw: str) -> str:
    if len(raw) < 2 or not (raw.startswith('"') and raw.endswith('"')):
        return raw
    value: list[str] = []
    escaped = False
    for char in raw[1:-1]:
        if escaped:
            value.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            value.append(char)
    return "".join(value)
