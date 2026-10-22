"""Frozen in-world clock shared by the run planner and the authoring validators.

Persona Timelines fix each session's ``date`` and ``hour``; a generated Outline
carries that frozen clock for runtime use. This module only checks the embedded
values against their time-position window and turns them into UTC timestamps.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

# The harness owns the in-world clock inside a session. A session starts at its
# Timeline slot; every simulator turn adds one minute; after a suppressed
# proactive contact the user comes back on their own SUPPRESS_RESUME_DELAY_MINUTES
# after the trigger. Everything must finish before the next session's slot, so
# consecutive sessions must start at least SESSION_ENVELOPE_MINUTES apart, and a
# session that would run past its envelope fails loudly instead of drifting.
SESSION_ENVELOPE_MINUTES = 60
SUPPRESS_RESUME_DELAY_MINUTES = 20


def max_triggered_session_turns() -> int:
    """Most simulator↔PA exchanges a triggered session's played beats may spend.

    First user turn at +delay; each exchange adds a minute; the last exchange
    must land strictly before the deadline.
    """
    return SESSION_ENVELOPE_MINUTES - SUPPRESS_RESUME_DELAY_MINUTES


def max_triggered_session_beats(max_turns_per_beat: int) -> int:
    """Most beats a triggered session may play (one opening plus the shared
    beats) so that even the suppress path, held to the turn cap on every beat,
    ends inside the envelope.

    Assumes one flat cap for every beat. Under the IX wrapper the per-beat
    budget varies with the beat's hold-bearing attributes, so a caller that
    knows the beats should sum their budgets against
    `max_triggered_session_turns()` instead.
    """
    return max_triggered_session_turns() // max_turns_per_beat

TIME_POSITION_WINDOWS: dict[str, tuple[date, date]] = {
    "early_oct": (date(2026, 10, 1), date(2026, 10, 10)),
    "mid_oct": (date(2026, 10, 11), date(2026, 10, 20)),
    "late_oct": (date(2026, 10, 21), date(2026, 10, 31)),
    "nov": (date(2026, 11, 1), date(2026, 11, 30)),
    "dec": (date(2026, 12, 1), date(2026, 12, 31)),
    "early_jan": (date(2027, 1, 1), date(2027, 1, 15)),
    "late_jan": (date(2027, 1, 16), date(2027, 1, 31)),
}


def benchmark_session_timestamps(
    outline_by_acc: dict[int, dict[str, Any]],
) -> dict[int, datetime]:
    """Map each session's authored clock onto a UTC timestamp, checking order."""
    timestamps: dict[int, datetime] = {}
    previous: datetime | None = None
    for acc_num, session in sorted(outline_by_acc.items()):
        time_position = session.get("time_position")
        window = TIME_POSITION_WINDOWS.get(time_position)
        if window is None:
            raise ValueError(
                "Session outline has an unsupported time_position: "
                f"acc_num={acc_num!r} value={time_position!r} "
                f"supported={sorted(TIME_POSITION_WINDOWS)!r}"
            )
        try:
            day = date.fromisoformat(str(session.get("date")))
            hour = int(session.get("hour"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Session outline must carry date (YYYY-MM-DD) and hour (0-23): "
                f"acc_num={acc_num!r} date={session.get('date')!r} "
                f"hour={session.get('hour')!r}"
            ) from exc
        if not (window[0] <= day <= window[1]):
            raise ValueError(
                f"Session date is outside its time_position window: acc_num={acc_num!r} "
                f"date={day.isoformat()} window={window[0]}..{window[1]}"
            )
        timestamp = datetime(day.year, day.month, day.day, hour, 0, tzinfo=timezone.utc)
        if previous is not None and timestamp < previous + timedelta(
            minutes=SESSION_ENVELOPE_MINUTES
        ):
            raise ValueError(
                "Synthetic benchmark timestamps must increase with acc_num by at "
                f"least the session envelope ({SESSION_ENVELOPE_MINUTES} minutes): "
                f"acc_num={acc_num!r} timestamp={timestamp.isoformat()!r} "
                f"previous={previous.isoformat()!r}"
            )
        timestamps[acc_num] = timestamp
        previous = timestamp
    return timestamps
