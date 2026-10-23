"""Per-call token counts for the LLM MemOS runs inside its own container.

MemOS's memory-side LLM is not a Python object in the orchestrator process, so
it cannot reach the run log through ``pa/providers``' ``_log_openrouter_call``
the way mem0's extraction LLM does.  It does, however, log its own
usage: the patched wrapper writes one INFO record per call
(``memory_raw_repos/MemOS/src/memos/llms/openai.py:135-139``, fields built at
``:23-31``), and the fork sends INFO to
``MEMOS_BASE_PATH/.memos/logs/memos.log`` (``src/memos/log.py:198-207`` +
``log.py:37`` + ``settings.py:6`` + ``deploy/memos/compose.yaml:29``).
``docker logs`` shows none of it, because the console handler sits at WARNING
(``log.py:29``, ``settings.DEBUG`` being the hardcoded ``False`` at
``settings.py:7``).

So the harness reads that file out of the container and prints one line per
call into the run's api log, beside the ``openrouter_call`` lines the in-process
providers emit.

**There is no cost field.** The container logs
tokens and seconds and no price, and the repo has no price table, so any dollar
figure here could only be an estimate.  Reading the fork is also the only option
that leaves the pinned MemOS source alone: any fork edit would move
``MEMPA_MEMOS_SOURCE_SHA256`` (build arg and environment in
``deploy/memos/compose.yaml``).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

#: ``deploy/memos/compose.yaml:13``.
MEMOS_CONTAINER_NAME = "mempa-memos-api"
#: ``deploy/memos/compose.yaml:29`` (``MEMOS_BASE_PATH``) resolved through the
#: fork's ``settings.py:6`` and ``log.py:37``.
MEMOS_LOG_PATH = "/var/lib/memos/.memos/logs/memos.log"

#: The fork's record layout (``log.py:177``, the ``standard`` formatter the file
#: handler uses): seven ``" | "`` fields, the last of which is
#: ``name - LEVEL - file:lineno - func - message``.
_FIELD_SEPARATOR = " | "
_FIELD_COUNT = 7
_MESSAGE_SEPARATOR = " - "
_MESSAGE_PART_COUNT = 5
#: The message itself (``llms/openai.py:135-139``): the ``repr`` of the dict
#: ``_response_log_fields`` returns, then the elapsed seconds.  A Python literal,
#: not JSON, so it is read with ``ast.literal_eval`` rather than ``json.loads``.
_USAGE_PREFIX = "OpenAI LLM response: fields="
_SECONDS_KEY = " cost_seconds="

CommandRunner = Callable[[list[str]], bytes]


@dataclass(frozen=True)
class MemosUsageRecord:
    """One memory-side LLM call, as the container logged it."""

    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    seconds: float


def parse_usage_line(line: str) -> MemosUsageRecord | None:
    """Read one container log line; ``None`` when it is not a usage line."""
    fields = line.split(_FIELD_SEPARATOR, _FIELD_COUNT - 1)
    if len(fields) < _FIELD_COUNT:
        return None
    parts = fields[-1].split(_MESSAGE_SEPARATOR, _MESSAGE_PART_COUNT - 1)
    if len(parts) < _MESSAGE_PART_COUNT:
        return None
    message = parts[-1]
    if not message.startswith(_USAGE_PREFIX):
        return None

    rendered, separator, seconds = message[len(_USAGE_PREFIX) :].rpartition(
        _SECONDS_KEY
    )
    if not separator:
        raise ValueError(
            f"MemOS usage line carries no {_SECONDS_KEY.strip()}: line={line!r}"
        )
    parsed = ast.literal_eval(rendered)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"MemOS usage line fields are not a mapping: line={line!r}"
        )
    return MemosUsageRecord(
        model=parsed.get("response_model"),
        prompt_tokens=parsed.get("prompt_tokens"),
        completion_tokens=parsed.get("completion_tokens"),
        total_tokens=parsed.get("total_tokens"),
        seconds=float(seconds),
    )


def _docker_stdout(argv: list[str]) -> bytes:
    return subprocess.run(argv, capture_output=True, check=True).stdout


class MemosUsageTail:
    """A byte cursor into the container's log; a drain reads only what is new.

    The cursor starts at end-of-file, so a run never reports calls that were
    made before it began, and it only ever advances past a complete line.
    """

    def __init__(
        self,
        *,
        container: str = MEMOS_CONTAINER_NAME,
        log_path: str = MEMOS_LOG_PATH,
        run_command: CommandRunner = _docker_stdout,
    ) -> None:
        self._container = container
        self._log_path = log_path
        self._run_command = run_command
        self._cursor = self._size()

    def _size(self) -> int:
        stdout = self._run_command(
            ["docker", "exec", self._container, "stat", "-c", "%s", self._log_path]
        )
        return int(stdout.decode().strip())

    def drain(self) -> list[MemosUsageRecord]:
        size = self._size()
        if size < self._cursor:
            # The file handler rotates at midnight (``log.py:200-203``).
            self._cursor = 0
        if size == self._cursor:
            return []
        chunk = self._run_command(
            [
                "docker",
                "exec",
                self._container,
                "tail",
                "-c",
                f"+{self._cursor + 1}",
                self._log_path,
            ]
        )
        end = chunk.rfind(b"\n")
        if end < 0:
            # A half-written line: leave the cursor, read it whole next time.
            return []
        self._cursor += end + 1
        text = chunk[: end + 1].decode("utf-8", errors="replace")
        records = [parse_usage_line(line) for line in text.splitlines()]
        return [record for record in records if record is not None]


@dataclass
class _SessionTotals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class MemosUsageReporter:
    """One api-log line per memory-side call, plus one summary per session.

    `session` is the benchmark session — the step id the run log's own
    `[001/003] acc_001 …` lines carry — which only the orchestrator knows: a
    `MemoryEvent` names the run-scoped PA session key instead, so the
    orchestrator opens each session here and the adapter only supplies turns.
    """

    def __init__(
        self,
        *,
        tail: MemosUsageTail,
        run_id: str,
        stream: TextIO | None = None,
    ) -> None:
        self._tail = tail
        self._run_id = run_id
        self._stream = stream
        self._session_id = "-"
        self._turn_id = "-"
        self._totals = _SessionTotals()

    def open_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._turn_id = "-"
        self._totals = _SessionTotals()

    def emit_calls(self, *, turn_id: str) -> None:
        self._turn_id = turn_id
        self._drain()

    def emit_session_summary(self) -> None:
        self._drain()
        self._print(
            f"memos_extraction_session run={self._run_id} "
            f"session={self._session_id} calls={self._totals.calls} "
            f"prompt_tokens={self._totals.prompt_tokens} "
            f"completion_tokens={self._totals.completion_tokens} "
            f"total_tokens={self._totals.total_tokens}"
        )

    def _drain(self) -> None:
        for record in self._tail.drain():
            self._print(
                f"memos_extraction_call run={self._run_id} "
                f"session={self._session_id} turn={self._turn_id} "
                f"model={record.model} "
                f"prompt_tokens={record.prompt_tokens} "
                f"completion_tokens={record.completion_tokens} "
                f"total_tokens={record.total_tokens} "
                f"seconds={record.seconds:.3f}"
            )
            self._totals.calls += 1
            self._totals.prompt_tokens += record.prompt_tokens or 0
            self._totals.completion_tokens += record.completion_tokens or 0
            self._totals.total_tokens += record.total_tokens or 0

    def _print(self, line: str) -> None:
        print(line, file=self._stream or sys.stderr, flush=True)
