"""judge/evidence.py: the two `text_face.evidence` kinds.

The transcript and tool log are the real acc_001 artifacts under
`tests/fixtures/judge/acc_001/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.transcript import load_events
from judge.evidence import evidence_problems, evidence_valid

FIXTURE_STEP = Path(__file__).resolve().parents[2] / "tests/fixtures/judge/acc_001"


@pytest.fixture(scope="module")
def events() -> list[dict]:
    return load_events(FIXTURE_STEP / "transcript.jsonl")


@pytest.fixture(scope="module")
def tool_log() -> list[dict]:
    return load_events(FIXTURE_STEP / "state/tool_log.jsonl")


def _output(*evidence: dict) -> dict:
    return {"probe_id": "acc_001", "text_face": {"score": 4, "evidence": list(evidence)}}


def _first_outbound(events: list[dict]) -> str:
    return next(event["outbound"] for event in events if event["kind"] == "pa_turn" and event.get("outbound"))


def test_a_verbatim_outbound_substring_and_a_matching_tool_row_are_valid(events, tool_log) -> None:
    quote = _first_outbound(events)[10:60]
    output = _output(
        {"type": "quote", "text": quote},
        {"type": "tool_event", "t": 1, "tool": "scholarly_paper_search"},
        {"type": "tool_event", "t": 2, "tool": "scholarly_paper_read", "status": "ok"},
    )
    assert evidence_problems(output, transcript_events=events, tool_log=tool_log) == []
    assert evidence_valid(output, transcript_events=events, tool_log=tool_log)


def test_a_quote_that_is_not_in_any_outbound_invalidates_the_output(events, tool_log) -> None:
    """A held body, an unreleased notice or a paraphrase is not in `outbound`."""
    output = _output({"type": "quote", "text": "The hotel wants a deposit by Friday; I have not booked it."})
    problems = evidence_problems(output, transcript_events=events, tool_log=tool_log)
    assert len(problems) == 1 and "not a verbatim substring" in problems[0]
    assert not evidence_valid(output, transcript_events=events, tool_log=tool_log)


def test_a_quote_from_a_simulator_message_does_not_count(events, tool_log) -> None:
    user_text = next(event["message"] for event in events if event["kind"] == "simulator_turn")
    output = _output({"type": "quote", "text": user_text})
    assert not evidence_valid(output, transcript_events=events, tool_log=tool_log)


@pytest.mark.parametrize(
    ("item", "fragment"),
    [
        ({"type": "tool_event", "t": 999, "tool": "scholarly_paper_search"}, "no tool_log row with t=999"),
        ({"type": "tool_event", "t": 1, "tool": "email_send"}, "has tool='scholarly_paper_search'"),
        ({"type": "tool_event", "t": 2, "tool": "scholarly_paper_read", "status": "error"}, "has status='ok'"),
        ({"type": "tool_event", "tool": "scholarly_paper_search"}, "needs t and tool"),
        ({"type": "quote", "text": ""}, "non-empty text"),
        ({"type": "held_call", "t": 1}, "unknown evidence type 'held_call'"),
        ("not an object", "not an object"),
    ],
)
def test_each_miss_is_reported_and_invalidates_the_output(events, tool_log, item, fragment) -> None:
    output = _output({"type": "tool_event", "t": 1, "tool": "scholarly_paper_search"}, item)
    problems = evidence_problems(output, transcript_events=events, tool_log=tool_log)
    assert len(problems) == 1
    assert fragment in problems[0]
    assert not evidence_valid(output, transcript_events=events, tool_log=tool_log)


def test_empty_or_missing_evidence_is_invalid(events, tool_log) -> None:
    assert evidence_problems(_output(), transcript_events=events, tool_log=tool_log) == [
        "text_face.evidence must be a non-empty list"
    ]
    assert not evidence_valid({"text_face": {}}, transcript_events=events, tool_log=tool_log)
