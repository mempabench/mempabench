"""Per-beat turn budget and the in-world envelope pre-check.

The budget formula and its four constants live in `pa/agent/ix/render_types.py`
and are injected by the harness at construction; `simulator/` never imports
`pa/`, so with nothing injected every beat gets the flat fallback of 3.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_turn_budget.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from simulator.loop import (
    MAX_TURNS_PER_BEAT,
    MAX_UNCHARGED_EMPTY_REPLIES_PER_BEAT,
    SimulatorLoop,
)
from simulator.output import SimulatorOutput, TurnAssessment
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

# The harness's own import: it is what P4 passes in as `turn_budget_fn`.
from pa.agent.ix.render import IX_EMPTY_REPLY_LINE
from pa.agent.ix.render_types import turn_budget
from pa.bus.events import OutboundMessage

START = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


class _ScriptedProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "scripted-model"

    async def chat(self, *args, **kwargs) -> LLMResponse:
        raise AssertionError("the scripted backend never calls the provider")


class _StayingBackend:
    """Never advances, so a beat always spends its whole budget."""

    name = "staying"

    def __init__(self) -> None:
        self.turns = 0

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        self.turns += 1
        return SimulatorOutput(
            message=f"turn {self.turns}, can you check?",
            turn_assessment=TurnAssessment(next_beat="stay"),
            raw_response="scripted",
        )

    async def drain_memory_tasks(self) -> None:
        return None


class _StubPA:
    async def process_direct_message(self, user_message, session_key, **kwargs):
        return OutboundMessage(channel="cli", chat_id="direct", content="PA reply")


def _identity() -> PersonaIdentity:
    return PersonaIdentity(name="User A", speech_style={})


def _matrix() -> PreferenceMatrix:
    return PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")


def _script(beats: list[Beat]) -> SessionScript:
    return SessionScript(
        session_id="budget_test",
        scenario="turn budget",
        context="personal",
        task="Get the thing resolved.",
        life_context=LifeContext(),
        task_facts={},
        beats=beats,
    )


def _loop(beats: list[Beat], *, pa: object | None = None, **kwargs) -> SimulatorLoop:
    return SimulatorLoop(
        _ScriptedProvider(),
        _identity(),
        _matrix(),
        _script(beats),
        pa or _StubPA(),
        **kwargs,
    )


# -- the budget itself ------------------------------------------------------

def test_a_beat_with_no_hold_attribute_gets_the_flat_fallback() -> None:
    loop = _loop([Beat(beat="open", active_skills=["verbosity"])])
    assert MAX_TURNS_PER_BEAT == 3
    assert loop._beat_turn_budget(loop.script.beats[0]) == 3
    # And with the real formula injected, a beat with no hold attribute is
    # still 3: CLOSE_ROUNDS is conditional on there being something to close.
    injected = _loop(
        [Beat(beat="open", active_skills=["verbosity"])], turn_budget_fn=turn_budget
    )
    assert injected._beat_turn_budget(injected.script.beats[0]) == 3
    assert _loop(
        [Beat(beat="open", active_skills=[])], turn_budget_fn=turn_budget
    )._beat_turn_budget(Beat(beat="open", active_skills=[])) == 3


def test_autonomy_alone_widens_the_beat_to_seven() -> None:
    loop = _loop(
        [Beat(beat="open", active_skills=["autonomy_level"])],
        turn_budget_fn=turn_budget,
    )
    # BASE 3 + UNIT_CAP[autonomy_level] 3 + CLOSE_ROUNDS 1
    assert loop._beat_turn_budget(loop.script.beats[0]) == 7


def test_all_four_hold_attributes_are_held_to_the_global_cap() -> None:
    beat = Beat(
        beat="open",
        active_skills=[
            "autonomy_level",
            "topic_management",
            "information_elicitation",
            "capability_boundary",
        ],
    )
    loop = _loop([beat], turn_budget_fn=turn_budget)
    # 3 + (3+3+3+1) + 1 = 14, held to GLOBAL_TURN_CAP.
    assert loop._beat_turn_budget(loop.script.beats[0]) == 8


def test_the_budget_only_ever_sees_the_beats_active_skills() -> None:
    """The budget must not be able to read the selected setting: seeing one
    would let the number of turns a beat was given leak the ground truth."""
    seen: list[object] = []

    def recording(active_skills):
        seen.append(active_skills)
        return 3

    loop = _loop(
        [Beat(beat="open", active_skills=["autonomy_level"])],
        turn_budget_fn=recording,
    )
    loop._beat_turn_budget(loop.script.beats[0])
    assert seen == [["autonomy_level"]]


def test_a_budget_that_is_not_a_positive_int_fails_loudly() -> None:
    zero = _loop([Beat(beat="open")], turn_budget_fn=lambda _skills: 0)
    with pytest.raises(ValueError, match="turn_budget_fn must return a positive int"):
        zero._beat_turn_budget(zero.script.beats[0])
    text = _loop([Beat(beat="open")], turn_budget_fn=lambda _skills: "3")
    with pytest.raises(ValueError, match="turn_budget_fn must return a positive int"):
        text._beat_turn_budget(text.script.beats[0])
    # The envelope pre-check runs the same validation, so a session on the
    # clock never gets as far as its first turn with a broken budget.
    with pytest.raises(ValueError, match="turn_budget_fn must return a positive int"):
        _loop(
            [Beat(beat="open")],
            benchmark_session_timestamp=START,
            benchmark_session_deadline=START + timedelta(minutes=60),
            turn_budget_fn=lambda _skills: 0,
        )


# -- beat_turn_limit --------------------------------------------------------

def test_beat_turn_limit_reports_the_budget_this_beat_ran_on() -> None:
    async def run():
        backend = _StayingBackend()
        loop = _loop(
            [Beat(beat="open", active_skills=["autonomy_level"])],
            backend=backend,
            turn_budget_fn=turn_budget,
        )
        events = await loop.run_session()

        limits = [e for e in events if e["kind"] == "beat_turn_limit"]
        assert [e["turns"] for e in limits] == [7]
        # The reported number is the denominator the judge reads, so it has to
        # equal the turns actually spent.
        assert sum(1 for e in events if e["kind"] == "simulator_turn") == 7

    asyncio.run(run())


def test_a_plain_beat_still_stops_at_three() -> None:
    async def run():
        backend = _StayingBackend()
        loop = _loop(
            [Beat(beat="open", active_skills=["verbosity"])],
            backend=backend,
            turn_budget_fn=turn_budget,
        )
        events = await loop.run_session()

        assert [e["turns"] for e in events if e["kind"] == "beat_turn_limit"] == [3]
        assert sum(1 for e in events if e["kind"] == "simulator_turn") == 3

    asyncio.run(run())


# -- the in-world envelope --------------------------------------------------

def test_a_session_whose_worst_case_fits_is_built_and_nothing_is_widened() -> None:
    beats = [Beat(beat=f"b{i}", active_skills=["autonomy_level"]) for i in range(8)]
    deadline = START + timedelta(minutes=60)
    loop = _loop(
        beats,
        benchmark_session_timestamp=START,
        benchmark_session_deadline=deadline,
        turn_budget_fn=turn_budget,
    )
    # 8 beats * 7 = 56 exchanges; the last one lands at +55, inside the hour.
    assert loop._worst_case_turn_offset() == 55
    assert loop.benchmark_session_deadline == deadline


def test_a_session_whose_worst_case_cannot_fit_is_refused_before_the_first_turn() -> None:
    beats = [Beat(beat=f"b{i}", active_skills=["autonomy_level"]) for i in range(9)]
    with pytest.raises(RuntimeError, match="cannot fit its in-world envelope"):
        _loop(
            beats,
            benchmark_session_timestamp=START,
            benchmark_session_deadline=START + timedelta(minutes=60),
            turn_budget_fn=turn_budget,
        )


def test_the_envelope_is_only_checked_when_the_harness_set_a_deadline() -> None:
    beats = [Beat(beat=f"b{i}", active_skills=["autonomy_level"]) for i in range(40)]
    loop = _loop(beats, turn_budget_fn=turn_budget)
    assert loop.benchmark_session_deadline is None


# -- a beat is not over while the PA is waiting on the user -------------------

class _AdvancingBackend:
    """Always advances, so only a held PA turn can keep a beat open."""

    name = "advancing"

    def __init__(self) -> None:
        self.turns = 0

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        self.turns += 1
        return SimulatorOutput(
            message=f"turn {self.turns}, can you check?",
            turn_assessment=TurnAssessment(next_beat="advance"),
            raw_response="scripted",
        )

    async def drain_memory_tasks(self) -> None:
        return None


class _HoldingPA:
    """Reports held items on its first `hold_turns` turns, then nothing."""

    def __init__(self, hold_turns: int) -> None:
        self.hold_turns = hold_turns
        self.turns = 0

    async def process_direct_message(self, user_message, session_key, **kwargs):
        self.turns += 1
        held = (
            [{"interaction_item_id": "i1", "intent": "send the extension request"}]
            if self.turns <= self.hold_turns else []
        )
        return OutboundMessage(
            channel="cli", chat_id="direct",
            content="Shall I send it?" if held else "Done.",
            metadata={"tool_events": [], "held_items": held},
        )


def test_a_held_pa_turn_keeps_the_last_beat_open_for_one_more_answer() -> None:
    """Live (`v5g02`), the gate opened on the story's last beat, the simulator
    advanced out of it, and the step failed on the open hold episode with no
    checkpoint published. The user answers the ask before the session ends."""
    async def run():
        backend = _AdvancingBackend()
        pa = _HoldingPA(hold_turns=1)
        loop = _loop(
            [Beat(beat="close", active_skills=["autonomy_level"])],
            backend=backend, pa=pa, turn_budget_fn=turn_budget,
        )
        events = await loop.run_session()

        # Turn 1 asked; turn 2 is the answer, inside the same beat.
        assert backend.turns == 2
        assert pa.turns == 2
        assert sum(1 for e in events if e["kind"] == "simulator_turn") == 2
        # It closed on its own, well inside the budget.
        assert [e for e in events if e["kind"] == "beat_turn_limit"] == []

    asyncio.run(run())


def test_a_pa_that_asks_for_nothing_closes_the_beat_as_before() -> None:
    async def run():
        backend = _AdvancingBackend()
        pa = _HoldingPA(hold_turns=0)
        loop = _loop(
            [Beat(beat="close", active_skills=["autonomy_level"])],
            backend=backend, pa=pa, turn_budget_fn=turn_budget,
        )
        events = await loop.run_session()

        assert backend.turns == 1
        assert sum(1 for e in events if e["kind"] == "simulator_turn") == 1
        assert [e for e in events if e["kind"] == "beat_turn_limit"] == []

    asyncio.run(run())


# -- a hold the script cannot release yet -----------------------------------
#
# Live regress15 `acc_039`: the PA's `external_action_floor` hold opened on the
# probe beat, three beats before the beat that gives the go-ahead. The pin
# demanded a release the script could not give before the authorising beat;
# the actor wrote one 12 times, had each bounced, and
# invented offstage deferral reasons for 7 exchanges on the probe and 3 on the
# reaction beat, which ate the reaction. Owner ruling: a beat that routed
# `advance` and is pinned only by a hold it cannot release advances and carries
# the hold on.


def _carry_beats() -> list[Beat]:
    """`acc_039`'s shape in miniature: probe, reaction, go-ahead, close."""
    return [
        Beat(
            beat="b1_probe",
            goal="Pick the checkup slot and have it put on the calendar.",
            active_skills=["autonomy_level"],
        ),
        Beat(beat="b2_react", goal="React to how the calendar entry was handled."),
        Beat(beat="b3_authorise", goal="Give the go-ahead for the clinic booking."),
        Beat(beat="b4_close", goal="Close the loop on the confirmation."),
    ]


class _RecordingAdvancingBackend(_AdvancingBackend):
    """Keeps what each turn was handed, so the carried hold can be read back."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[str, tuple]] = []

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        self.seen.append((beat.beat, tuple(kwargs.get("held_items") or ())))
        return await super().generate_turn(identity, matrix, session, beat, **kwargs)


def test_a_beat_before_the_authorising_beat_advances_and_carries_the_hold() -> None:
    """The probe routed `advance`, the PA held `i1`, and the go-ahead belongs to
    `b3_authorise`. The beat leaves, and `i1` is the next beat's first stage
    direction — carried, not cleared, so the user can still answer it."""
    async def run():
        backend = _RecordingAdvancingBackend()
        pa = _HoldingPA(hold_turns=1)
        loop = _loop(_carry_beats(), backend=backend, pa=pa, turn_budget_fn=turn_budget)
        events = await loop.run_session()

        carried = [e for e in events if e["kind"] == "beat_advanced_with_hold"]
        assert len(carried) == 1
        assert carried[0]["beat"] == "b1_probe"
        assert carried[0]["next_beat"] == "b2_react"
        assert carried[0]["held_items"] == ["i1"]
        assert carried[0]["pending_interaction"] is False
        # The probe spent one exchange, not its seven-turn autonomy budget.
        assert [e["beat"] for e in events if e["kind"] == "beat_turn_limit"] == []
        # And the hold reached the next beat's first turn as stage direction.
        assert backend.seen[0] == ("b1_probe", ())
        assert backend.seen[1][0] == "b2_react"
        assert [item["interaction_item_id"] for item in backend.seen[1][1]] == ["i1"]
        # One exchange per beat, the carried hold notwithstanding.
        assert [beat for beat, _ in backend.seen] == [
            "b1_probe", "b2_react", "b3_authorise", "b4_close"
        ]

    asyncio.run(run())


def test_the_authorising_beat_is_still_pinned_by_a_hold_and_says_so() -> None:
    """From the authorising beat on, the pin is exactly what it was: the user is
    expected to answer the hold there, so `advance` does not move and every kept
    turn is on record as `beat_kept_for_hold`."""
    async def run():
        backend = _RecordingAdvancingBackend()
        pa = _HoldingPA(hold_turns=99)  # never lets go
        loop = _loop(_carry_beats(), backend=backend, pa=pa, turn_budget_fn=turn_budget)
        events = await loop.run_session()

        # The two beats before it carried the hold on; the third one holds.
        assert [
            e["beat"] for e in events if e["kind"] == "beat_advanced_with_hold"
        ] == ["b1_probe", "b2_react"]
        kept = [e for e in events if e["kind"] == "beat_kept_for_hold"]
        # `b4_close` is pinned too — it is the last beat, where there is no next
        # beat to carry a hold into and leaving would end the session with the
        # episode open (`v5g02`).
        assert [e["beat"] for e in kept] == ["b3_authorise"] * 3 + ["b4_close"] * 3
        assert [e["turn"] for e in kept] == [1, 2, 3, 1, 2, 3]
        assert kept[0]["held_items"] == ["i1"]
        assert kept[0]["pending_interaction"] is False
        # Pinned to the budget, which is what stops it.
        assert [e["beat"] for e in events if e["kind"] == "beat_turn_limit"] == [
            "b3_authorise",
            "b4_close",
        ]

    asyncio.run(run())


def test_an_advance_with_nothing_held_is_untouched_by_the_carry_rule() -> None:
    async def run():
        backend = _RecordingAdvancingBackend()
        pa = _HoldingPA(hold_turns=0)
        loop = _loop(_carry_beats(), backend=backend, pa=pa, turn_budget_fn=turn_budget)
        events = await loop.run_session()

        assert [beat for beat, _ in backend.seen] == [
            "b1_probe", "b2_react", "b3_authorise", "b4_close"
        ]
        assert [e for e in events if e["kind"] == "beat_advanced_with_hold"] == []
        assert [e for e in events if e["kind"] == "beat_kept_for_hold"] == []
        assert [e for e in events if e["kind"] == "beat_turn_limit"] == []

    asyncio.run(run())


class _PendingOnlyPA:
    """Holds something that is consent-able by nothing (R7).

    An Iterative question hold and a One-at-a-time batch hold both leave
    `held_items` empty — there is no `interaction_item_id` to answer about — so
    only `has_pending_interaction` can tell the simulator the PA is still
    waiting. It lets go once the user has answered.
    """

    def __init__(self, hold_turns: int, content: str) -> None:
        self.hold_turns = hold_turns
        self.content = content
        self.turns = 0

    async def process_direct_message(self, user_message, session_key, **kwargs):
        self.turns += 1
        holding = self.turns <= self.hold_turns
        return OutboundMessage(
            channel="cli", chat_id="direct",
            content=self.content if holding else "Done.",
            metadata={
                "tool_events": [],
                # The consent list stays empty throughout: this is the whole
                # point of the case.
                "held_items": [],
                "has_pending_interaction": holding,
            },
        )


@pytest.mark.parametrize(
    "content",
    [
        "And what is the budget?",          # information_elicitation, Iterative
        "That is the first of three. Ready for the next?",  # topic_management
    ],
)
def test_a_hold_with_nothing_to_consent_to_still_keeps_the_last_beat_open(
    content: str,
) -> None:
    """R7: the advance rule used to read the consent list, which a question-only
    or batch-only hold leaves empty. The simulator walked out of the question
    and, on the last beat, ended the session with the episode still open."""
    async def run():
        backend = _AdvancingBackend()
        pa = _PendingOnlyPA(hold_turns=1, content=content)
        loop = _loop(
            [Beat(beat="close", active_skills=["information_elicitation"])],
            backend=backend, pa=pa, turn_budget_fn=turn_budget,
        )
        events = await loop.run_session()

        # Turn 1 asked; turn 2 is the answer, inside the same beat.
        assert backend.turns == 2
        assert pa.turns == 2
        assert sum(1 for e in events if e["kind"] == "simulator_turn") == 2
        assert [e for e in events if e["kind"] == "beat_turn_limit"] == []

    asyncio.run(run())


def test_a_pa_reporting_no_pending_interaction_closes_the_beat_as_before() -> None:
    async def run():
        backend = _AdvancingBackend()
        pa = _PendingOnlyPA(hold_turns=0, content="unused")
        loop = _loop(
            [Beat(beat="close", active_skills=["information_elicitation"])],
            backend=backend, pa=pa, turn_budget_fn=turn_budget,
        )
        events = await loop.run_session()

        assert backend.turns == 1
        assert sum(1 for e in events if e["kind"] == "simulator_turn") == 1
        assert [e for e in events if e["kind"] == "beat_turn_limit"] == []

    asyncio.run(run())


def test_the_pending_flag_never_reaches_the_simulator_turn() -> None:
    """It is control flow, not stage direction: no prompt, no history, no event.

    The flag says something is waiting; what is waiting stays evaluation-side.
    """
    seen: list[dict] = []

    class _RecordingBackend(_AdvancingBackend):
        async def generate_turn(self, identity, matrix, session, beat, **kwargs):
            seen.append(dict(kwargs))
            return await super().generate_turn(identity, matrix, session, beat, **kwargs)

    async def run():
        backend = _RecordingBackend()
        pa = _PendingOnlyPA(hold_turns=1, content="And what is the budget?")
        loop = _loop(
            [Beat(beat="close", active_skills=["information_elicitation"])],
            backend=backend, pa=pa, turn_budget_fn=turn_budget,
        )
        events = await loop.run_session()

        assert backend.turns == 2  # it did keep the beat open
        for kwargs in seen:
            assert "has_pending_interaction" not in kwargs
            assert "has_pending_interaction" not in json.dumps(kwargs, default=str)
            assert not kwargs.get("held_items")
        for message in loop._history:
            assert "has_pending_interaction" not in message["content"]
        assert "has_pending_interaction" not in json.dumps(events, default=str)

    asyncio.run(run())


def test_the_budget_still_bounds_a_pa_that_never_lets_go() -> None:
    """The stay is not open-ended: the beat's own budget ends it, and the
    wrapper's round cap closes the episode from its side."""
    async def run():
        backend = _AdvancingBackend()
        pa = _HoldingPA(hold_turns=99)
        loop = _loop(
            [Beat(beat="close", active_skills=["autonomy_level"])],
            backend=backend, pa=pa, turn_budget_fn=turn_budget,
        )
        events = await loop.run_session()

        assert backend.turns == 7
        assert [e["turns"] for e in events if e["kind"] == "beat_turn_limit"] == [7]

    asyncio.run(run())


# -- an empty PA turn is not one of the beat's exchanges ---------------------

class _EmptyReplyPA:
    """Delivers the wrapper's empty-reply line on the given turns.

    `ix_empty_reply` is a wrapper control record: the PA model returned no
    content, the one retry came back empty too, and `pa/agent/runner.py` closed
    the turn with a template line. It rides `tool_events` as an `ix_event`,
    which is the only place the simulator can see it.
    """

    def __init__(
        self, empty_turns: set[int] | None = None, always: bool = False
    ) -> None:
        self.empty_turns = empty_turns or set()
        self.always = always
        self.turns = 0

    async def process_direct_message(self, user_message, session_key, **kwargs):
        self.turns += 1
        empty = self.always or self.turns in self.empty_turns
        return OutboundMessage(
            channel="cli", chat_id="direct",
            content=IX_EMPTY_REPLY_LINE if empty else "Here is the answer.",
            metadata={
                "tool_events": (
                    [{"ix_event": True, "name": "ix_empty_reply",
                      "stop_reason": "completed"}]
                    if empty else []
                ),
                "held_items": [],
            },
        )


def test_an_empty_pa_reply_does_not_spend_one_of_the_beats_exchanges() -> None:
    """Live `acc_026`: a stalled PA turn ate one of the opening beat's three
    exchanges, the budget ran out, and two scripted beats were skipped. The
    empty turn is not charged, so the third real exchange still happens."""
    async def run():
        backend = _StayingBackend()
        pa = _EmptyReplyPA(empty_turns={2})
        loop = _loop(
            [Beat(beat="open", active_skills=["verbosity"])],
            backend=backend, pa=pa,
        )
        events = await loop.run_session()

        assert loop._beat_turn_budget(loop.script.beats[0]) == 3
        # Four exchanges: the three the budget pays for, plus the empty one.
        assert pa.turns == 4
        assert sum(1 for e in events if e["kind"] == "simulator_turn") == 4
        empty_turns = [
            e for e in events
            if e["kind"] == "pa_turn"
            and any(t.get("name") == "ix_empty_reply" for t in e["tool_events"])
        ]
        assert len(empty_turns) == 1
        # The denominator the judge reads is still the budget, not the exchanges.
        assert [e["turns"] for e in events if e["kind"] == "beat_turn_limit"] == [3]

    asyncio.run(run())


def test_an_endlessly_empty_pa_still_runs_the_beat_out() -> None:
    """The forgiveness is capped, so a provider that returns nothing forever
    reaches `beat_turn_limit` instead of looping inside one beat."""
    async def run():
        backend = _StayingBackend()
        pa = _EmptyReplyPA(always=True)
        loop = _loop(
            [Beat(beat="open", active_skills=["verbosity"])],
            backend=backend, pa=pa,
        )
        events = await loop.run_session()

        assert MAX_UNCHARGED_EMPTY_REPLIES_PER_BEAT == 2
        # 3 charged + 2 forgiven, and no more.
        assert pa.turns == MAX_TURNS_PER_BEAT + MAX_UNCHARGED_EMPTY_REPLIES_PER_BEAT
        assert [e["turns"] for e in events if e["kind"] == "beat_turn_limit"] == [3]

    asyncio.run(run())


def test_the_forgiveness_is_per_beat_not_per_session() -> None:
    async def run():
        backend = _StayingBackend()
        pa = _EmptyReplyPA(always=True)
        loop = _loop(
            [Beat(beat="open", active_skills=["verbosity"]),
             Beat(beat="close", active_skills=["verbosity"])],
            backend=backend, pa=pa,
        )
        await loop.run_session()

        assert pa.turns == 2 * (
            MAX_TURNS_PER_BEAT + MAX_UNCHARGED_EMPTY_REPLIES_PER_BEAT
        )

    asyncio.run(run())
