"""The memos container's memory-side LLM calls, read back as token counts.

The fixture lines below were captured verbatim from the running container
(`docker exec mempa-memos-api ... /var/lib/memos/.memos/logs/memos.log`) during
a memos backend smoke run.
"""

from __future__ import annotations

import io
import subprocess

import pytest

from memory.backends.memos_usage_log import (
    MemosUsageReporter,
    MemosUsageTail,
    parse_usage_line,
)
from scripts import run_orchestrator


_PREFIX = (
    "2026-09-13 16:32:06,804 | 1f85002c51985308603f4adf4488fe32 | "
    "path=/product/add | env=None | user_type=None | user_name=None | "
)
_USAGE_LINE_1 = _PREFIX + (
    "memos.llms.openai - INFO - openai.py:135 - generate - OpenAI LLM "
    "response: fields={'response_model': 'openai/gpt-5.5', 'choice_count': 1, "
    "'prompt_tokens': 1988, 'completion_tokens': 659, 'total_tokens': 2647} "
    "cost_seconds=6.464"
)
_USAGE_LINE_2 = _PREFIX.replace("16:32:06,804", "16:32:08,069") + (
    "memos.llms.openai - INFO - openai.py:135 - generate - OpenAI LLM "
    "response: fields={'response_model': 'openai/gpt-5.5', 'choice_count': 1, "
    "'prompt_tokens': 2280, 'completion_tokens': 723, 'total_tokens': 3003} "
    "cost_seconds=7.724"
)
_NOISE_LINES = [
    _PREFIX + (
        "memos.llms.openai - INFO - openai.py:126 - generate - OpenAI LLM "
        "request: model=openai/gpt-5.5 message_count=1 tool_count=0 "
        "stream=false"
    ),
    _PREFIX + (
        "httpx - INFO - _client.py:1025 - _send_single_request - HTTP "
        'Request: POST https://openrouter.ai/api/v1/chat/completions '
        '"HTTP/1.1 200 OK"'
    ),
    (
        "2026-09-13 16:32:01,594 | 17202fb472f982f3831d6afddd7f768a | "
        "path=/health | env=None | user_type=None | user_name=None | "
        "memos.api.middleware.request_context - INFO - request_context.py:70 "
        "- dispatch - Request started, source: server_api, method: GET, "
        "path: /health"
    ),
]

_CAPTURED_LOG = "\n".join(
    [_NOISE_LINES[0], _NOISE_LINES[1], _NOISE_LINES[2], _USAGE_LINE_1, _USAGE_LINE_2]
) + "\n"


class _FakeContainerLog:
    """Stands in for `docker exec` reading the container's log file."""

    def __init__(self, text: str = "") -> None:
        self.content = text.encode()
        self.reads = 0

    def append(self, text: str) -> None:
        self.content += text.encode()

    def __call__(self, argv: list[str]) -> bytes:
        if argv[3] == "stat":
            return f"{len(self.content)}\n".encode()
        self.reads += 1
        return self.content[int(argv[5].lstrip("+")) - 1 :]


def _tail(log: _FakeContainerLog) -> MemosUsageTail:
    return MemosUsageTail(
        container="mempa-memos-api",
        log_path="/var/lib/memos/.memos/logs/memos.log",
        run_command=log,
    )


def test_captured_container_lines_parse_to_usage_records() -> None:
    assert [parse_usage_line(line) for line in _NOISE_LINES] == [None, None, None]

    record = parse_usage_line(_USAGE_LINE_1)
    assert record is not None
    assert record.model == "openai/gpt-5.5"
    assert record.prompt_tokens == 1988
    assert record.completion_tokens == 659
    assert record.total_tokens == 2647
    assert record.seconds == pytest.approx(6.464)


def test_emitted_call_line_carries_no_cost_field() -> None:
    log = _FakeContainerLog()
    stream = io.StringIO()
    reporter = MemosUsageReporter(tail=_tail(log), run_id="run_x", stream=stream)
    reporter.open_session("acc_001")
    log.append(_CAPTURED_LOG)
    reporter.emit_calls(turn_id="acc_001:b1:t0")

    line = stream.getvalue().splitlines()[0]
    assert "cost" not in line
    keys = [pair.split("=", 1)[0] for pair in line.split(" ")]
    assert keys == [
        "memos_extraction_call",
        "run",
        "session",
        "turn",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "seconds",
    ]
    assert line == (
        "memos_extraction_call run=run_x session=acc_001 "
        "turn=acc_001:b1:t0 model=openai/gpt-5.5 prompt_tokens=1988 "
        "completion_tokens=659 total_tokens=2647 seconds=6.464"
    )


def test_tail_cursor_never_re_emits_a_drained_line() -> None:
    log = _FakeContainerLog()
    tail = _tail(log)
    log.append(_CAPTURED_LOG)

    first = tail.drain()
    assert [record.total_tokens for record in first] == [2647, 3003]
    assert tail.drain() == []

    log.append(_USAGE_LINE_1 + "\n")
    assert [record.total_tokens for record in tail.drain()] == [2647]
    assert tail.drain() == []


def test_tail_holds_a_half_written_line_until_it_is_complete() -> None:
    log = _FakeContainerLog()
    tail = _tail(log)
    log.append(_USAGE_LINE_1[:80])
    assert tail.drain() == []

    log.append(_USAGE_LINE_1[80:] + "\n")
    assert [record.total_tokens for record in tail.drain()] == [2647]


def test_rotated_log_rewinds_the_cursor() -> None:
    log = _FakeContainerLog(_CAPTURED_LOG)
    tail = _tail(log)
    assert tail.drain() == []

    # The file handler rotates at midnight: same path, fresh and shorter file.
    log.content = (_USAGE_LINE_2 + "\n").encode()
    assert [record.total_tokens for record in tail.drain()] == [3003]


def test_session_summary_counts_calls_and_tokens() -> None:
    log = _FakeContainerLog()
    stream = io.StringIO()
    reporter = MemosUsageReporter(tail=_tail(log), run_id="run_x", stream=stream)

    reporter.open_session("acc_001")
    log.append(_USAGE_LINE_1 + "\n")
    reporter.emit_calls(turn_id="acc_001:b1:t0")
    log.append(_USAGE_LINE_2 + "\n")
    reporter.emit_calls(turn_id="acc_001:b2:t1")
    reporter.emit_session_summary()

    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert lines[-1] == (
        "memos_extraction_session run=run_x session=acc_001 calls=2 "
        "prompt_tokens=4268 completion_tokens=1382 total_tokens=5650"
    )
    assert "cost" not in lines[-1]


def test_a_calls_line_and_its_summary_name_the_same_session() -> None:
    """The whole point of `open_session`: a `MemoryEvent` names the PA session
    key (the run), so a session read off the event would label the per-call
    lines one way and the summary another, and the summary would count zero."""
    log = _FakeContainerLog()
    stream = io.StringIO()
    reporter = MemosUsageReporter(tail=_tail(log), run_id="run_x", stream=stream)

    reporter.open_session("acc_001")
    log.append(_USAGE_LINE_1 + "\n")
    reporter.emit_calls(turn_id="acc_001:b1:t0")
    reporter.emit_session_summary()
    reporter.open_session("acc_002")
    log.append(_USAGE_LINE_2 + "\n")
    reporter.emit_calls(turn_id="acc_002:b1:t0")
    reporter.emit_session_summary()

    lines = stream.getvalue().splitlines()
    sessions = [
        pair.split("=", 1)[1]
        for line in lines
        for pair in line.split(" ")
        if pair.startswith("session=")
    ]
    assert sessions == ["acc_001", "acc_001", "acc_002", "acc_002"]
    assert lines[1].endswith("calls=1 prompt_tokens=1988 completion_tokens=659 total_tokens=2647")
    assert lines[3].endswith("calls=1 prompt_tokens=2280 completion_tokens=723 total_tokens=3003")


def _recorded_docker_commands(
    monkeypatch: pytest.MonkeyPatch, log: _FakeContainerLog
) -> list[list[str]]:
    """Answer every `docker exec` from `log` and record the command; no docker runs."""
    commands: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=log(argv), stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    return commands


def test_harness_reader_uses_the_compose_container_when_the_name_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEMOS_CONTAINER_NAME", raising=False)
    commands = _recorded_docker_commands(monkeypatch, _FakeContainerLog())

    run_orchestrator._memos_usage_reporter("run_x")

    assert commands == [
        [
            "docker",
            "exec",
            "mempa-memos-api",
            "stat",
            "-c",
            "%s",
            "/var/lib/memos/.memos/logs/memos.log",
        ]
    ]


def test_harness_reader_execs_into_the_named_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second MemOS deployment beside the paper's `mempa-memos-api` runs under
    its own container name; its run must count that container's calls."""
    monkeypatch.setenv("MEMOS_CONTAINER_NAME", "mempabench-reltest-memos-api")
    log = _FakeContainerLog()
    commands = _recorded_docker_commands(monkeypatch, log)

    reporter = run_orchestrator._memos_usage_reporter("run_x")
    reporter.open_session("acc_001")
    log.append(_USAGE_LINE_1 + "\n")
    reporter.emit_calls(turn_id="acc_001:b1:t0")

    assert [argv[3] for argv in commands] == ["stat", "stat", "tail"]
    assert [argv[:3] for argv in commands] == [
        ["docker", "exec", "mempabench-reltest-memos-api"]
    ] * 3


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_container_name_is_an_error_not_the_default_container(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Falling back to `mempa-memos-api` would read another deployment's log."""
    monkeypatch.setenv("MEMOS_CONTAINER_NAME", value)
    commands = _recorded_docker_commands(monkeypatch, _FakeContainerLog())

    with pytest.raises(ValueError, match="MEMOS_CONTAINER_NAME"):
        run_orchestrator._memos_usage_reporter("run_x")
    assert commands == []
