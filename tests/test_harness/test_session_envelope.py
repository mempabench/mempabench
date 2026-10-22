"""The harness owns the in-world clock: envelope, resume delay, beat budget."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness.benchmark_clock import (
    SESSION_ENVELOPE_MINUTES,
    SUPPRESS_RESUME_DELAY_MINUTES,
    benchmark_session_timestamps,
    max_triggered_session_beats,
)
from simulator.loop import MAX_TURNS_PER_BEAT

ROOT = Path(__file__).resolve().parents[2]


def test_envelope_holds_the_longest_suppress_path() -> None:
    played = max_triggered_session_beats(MAX_TURNS_PER_BEAT)
    # Seven exchanges is the authored floor for a triggered session's path.
    assert played >= 7
    # Last exchange of the suppress path: resume at +delay, one minute per
    # exchange, so the final one lands at delay + turns - 1.
    last_turn = SUPPRESS_RESUME_DELAY_MINUTES + MAX_TURNS_PER_BEAT * played - 1
    assert last_turn < SESSION_ENVELOPE_MINUTES
    assert SUPPRESS_RESUME_DELAY_MINUTES + MAX_TURNS_PER_BEAT * (played + 1) - 1 >= (
        SESSION_ENVELOPE_MINUTES
    )


def test_clock_rejects_sessions_closer_than_the_envelope() -> None:
    def row(hour: int, minute_offset_day: int = 1):
        return {"time_position": "early_oct", "date": f"2026-10-0{minute_offset_day}", "hour": hour}

    assert list(benchmark_session_timestamps({1: row(9), 2: row(10)})) == [1, 2]
    with pytest.raises(ValueError, match="session envelope"):
        benchmark_session_timestamps({1: row(9), 2: row(9)})


def test_live_user_a_outline_fits_the_envelope() -> None:
    outline = yaml.safe_load(
        (ROOT / "Authoring/outline/drafts/user_a/outline.yaml").read_text(encoding="utf-8")
    )
    by_acc = {
        item["identity"]["acc_num"]: {
            key: item["story"][key] for key in ("time_position", "date", "hour")
        }
        for item in outline["sessions"]
    }
    timestamps = benchmark_session_timestamps(by_acc)
    assert len(timestamps) == len(by_acc)
