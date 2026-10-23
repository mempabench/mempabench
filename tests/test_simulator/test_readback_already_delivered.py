"""A readback beat whose readback the PA already ran does not force an ask.

A readback beat exists to make the assistant show the end state of its own
write instead of asserting it, so `simulator.loop` arms the ask gate on its
first turn (record-only: a missing ask is recorded, never re-asked). When the
read already happened — on the user's own request a beat earlier, or of the
PA's own accord — there is nothing left to ask for, and the gate reads a miss on
a beat that owed nothing. Live `acc_039`: the PA
printed the clinic confirmation on `beat_5` after `appointment_read`, and
`beat_6_confirm_booking_record` then spent three attempts on "The confirmation
record is sufficient..." before the session aborted `beat_ask_unfulfilled`.

The beat stays held from the `goal_already_met` collapse either way; only the
forced ask goes, so acknowledge-and-close satisfies the beat.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_readback_already_delivered.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from pa.bus.events import OutboundMessage

from test_simulator.judge_fake import ScriptedJudgeClient

from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.judge.session import JudgeSession
from simulator.loop import SimulatorLoop
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

ACKNOWLEDGE_AND_CLOSE = (
    "The confirmation record is sufficient. The schedule is now settled."
)

# The two pairs `readback_beat_tools` derives from live `acc_039`: the calendar
# hold its `beat_4` inspects, and the clinic appointment its `beat_6` does.
APPOINTMENT_PAIR = ("appointment_create", "appointment_read")
CALENDAR_PAIR = ("calendar_event_create", "calendar_event_read")


def _turn(message: str, next_beat: str) -> str:
    return (
        f"<message>\n{message}\n</message>\n\n"
        "<factual_check>\n</factual_check>\n\n"
        f"<turn_assessment>\nnext_beat: {next_beat}\n</turn_assessment>"
    )


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def get_default_model(self) -> str:
        return "scripted-model"

    async def chat(self, messages, *args, **kwargs) -> LLMResponse:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("no scripted response left")
        return LLMResponse(content=self.responses.pop(0))


class _ToolScriptedPA:
    """A PA whose every reply carries the tool names that turn executed.

    A name on its own is an `ok` call; a `(name, status)` pair is how the
    harness records one the store refused.
    """

    def __init__(self, tool_names_per_turn: list[list[Any]]) -> None:
        self.tool_names_per_turn = list(tool_names_per_turn)

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
    ) -> OutboundMessage:
        names = (
            self.tool_names_per_turn.pop(0) if self.tool_names_per_turn else []
        )
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="PA reply",
            metadata={
                "tool_events": [
                    {"name": name, "status": "ok"}
                    if isinstance(name, str)
                    else {
                        "name": name[0],
                        "status": name[1],
                        "error": f"Error executing tool {name[0]}",
                    }
                    for name in names
                ]
            },
        )


def _script(
    *,
    compressed: bool = False,
    pair: tuple[str | None, str | None] | None = None,
) -> SessionScript:
    """The live `acc_039` shape: a write beat, a beat the user asks to see it
    on, and the readback beat that follows. `pair` is the (write, read) the
    readback beat's own state change declares, as `readback_beat_tools` derives
    it."""
    script = SessionScript(
        session_id="readback_delivered",
        scenario="the assistant books and then shows the booking",
        context="personal",
        task="Get the appointment booked and on the record.",
        life_context=LifeContext(),
        compressed=compressed,
        beats=[
            Beat(beat="b1_book", goal="Have the assistant place the booking."),
            Beat(beat="b2_review", goal="Have the assistant show the record."),
            Beat(
                beat="b3_confirm_record",
                goal="Confirm the booking record is on file.",
            ),
        ],
    )
    if not compressed:
        script.readback_beats = ["b3_confirm_record"]
        if pair is not None:
            script.readback_beat_tools = {"b3_confirm_record": pair}
    return script


#: The ask gate is the judge's alone, so every session here runs
#: one. These are the candidate turns that put nothing to the assistant; every
#: other turn reads as an ask, quoted out of the turn itself.
_NO_ASK = (
    "The 10:00 slot is the one I want.",
    "That settles it.",
    ACKNOWLEDGE_AND_CLOSE,
)


def _judge() -> JudgeSession:
    return JudgeSession(mode="live", client=ScriptedJudgeClient(no_ask=_NO_ASK))


def _run(
    responses: list[str],
    tool_names_per_turn: list[list[str]],
    *,
    compressed=False,
    pair: tuple[str | None, str | None] | None = None,
) -> list[dict]:
    async def run():
        provider = _ScriptedProvider(responses)
        loop = SimulatorLoop(
            provider,
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            _script(compressed=compressed, pair=pair),
            _ToolScriptedPA(tool_names_per_turn),
            backend=NanobotRunnerSimulatorBackend(provider),
            judge=_judge(),
        )
        return await loop.run_session()

    return asyncio.run(run())


def _of_kind(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["kind"] == kind]


def _gate_calls(events: list[dict]) -> list[dict]:
    """Every `carries_an_ask` the session put to the judge. The forced-ask
    check is commented out, so this is empty however
    the switch below falls: the beat that keeps its ask costs no judge call
    either."""
    return [e for e in _of_kind(events, "judge_call") if e["rule"] == "carries_an_ask"]


def test_a_read_after_the_write_switches_the_forced_ask_off() -> None:
    """The live `acc_039` shape itself: the beat's appointment record was
    written and then read back, so the beat has nothing left to ask for."""
    events = _run(
        [
            _turn("Please book the 10:00 slot with Dr. Patel.", "advance"),
            _turn("Show me the clinic's own confirmation record.", "advance"),
            _turn(ACKNOWLEDGE_AND_CLOSE, "advance"),
        ],
        [
            ["mcp_state_appointment_create"],
            ["mcp_state_appointment_read"],
            [],
        ],
        pair=APPOINTMENT_PAIR,
    )

    delivered = _of_kind(events, "readback_already_delivered")
    assert len(delivered) == 1
    assert delivered[0]["beat"] == "b3_confirm_record"
    assert delivered[0]["read_tool"] == "mcp_state_appointment_read"
    assert delivered[0]["write_tool"] == "mcp_state_appointment_create"

    # No forced ask, so the acknowledge-and-close turn is the one played.
    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert [
        event["message"] for event in _of_kind(events, "simulator_turn")
    ][-1] == ACKNOWLEDGE_AND_CLOSE


def test_another_records_write_and_read_do_not_answer_this_beat() -> None:
    """Live `acc_039` t=20. The PA created the calendar hold and
    read it back at t=15, and its `appointment_create` then errored, so nothing
    was booked. `beat_6_confirm_booking_record` is the readback beat of the
    appointment, not of the calendar entry, and the calendar pair is no answer
    to it — the beat keeps its ask."""
    events = _run(
        [
            _turn("Please book the 10:00 slot with Dr. Patel.", "advance"),
            _turn("Put the hold on my calendar first.", "advance"),
            _turn(ACKNOWLEDGE_AND_CLOSE, "advance"),
            _turn("Show me the clinic's booking record as it stands.", "advance"),
        ],
        [
            ["mcp_state_calendar_event_create"],
            ["mcp_state_calendar_event_read"],
            [],
        ],
        pair=APPOINTMENT_PAIR,
    )

    assert _of_kind(events, "readback_already_delivered") == []
    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []


def test_the_same_calls_answer_the_beat_that_inspects_the_calendar() -> None:
    """The same two calls, and the beat that does inspect the calendar entry —
    `acc_039`'s `beat_4` — is answered by them. The switch is on the beat's own
    artefact, not on the calls."""
    events = _run(
        [
            _turn("Please book the 10:00 slot with Dr. Patel.", "advance"),
            _turn("Put the hold on my calendar first.", "advance"),
            _turn(ACKNOWLEDGE_AND_CLOSE, "advance"),
        ],
        [
            ["mcp_state_calendar_event_create"],
            ["mcp_state_calendar_event_read"],
            [],
        ],
        pair=CALENDAR_PAIR,
    )

    delivered = _of_kind(events, "readback_already_delivered")
    assert len(delivered) == 1
    assert delivered[0]["write_tool"] == "mcp_state_calendar_event_create"
    assert delivered[0]["read_tool"] == "mcp_state_calendar_event_read"


def test_an_unprefixed_call_is_the_same_tool() -> None:
    """A draft names the catalog tool (`appointment_read`); the PA executes it
    under the state server's MCP prefix. Either form is the declared tool."""
    events = _run(
        [
            _turn("Please book the 10:00 slot with Dr. Patel.", "advance"),
            _turn("Show me the clinic's own confirmation record.", "advance"),
            _turn(ACKNOWLEDGE_AND_CLOSE, "advance"),
        ],
        [
            ["appointment_create"],
            ["appointment_read"],
            [],
        ],
        pair=APPOINTMENT_PAIR,
    )

    delivered = _of_kind(events, "readback_already_delivered")
    assert len(delivered) == 1
    assert delivered[0]["write_tool"] == "appointment_create"
    assert delivered[0]["read_tool"] == "appointment_read"


def test_a_beat_whose_change_names_no_tool_falls_back_to_the_effect() -> None:
    """A state change can hang on an `actor: pa` step, which names no catalog
    tool. Nothing is declared then, and the switch reads the effect suffix as
    it did before — any write, any read after it."""
    events = _run(
        [
            _turn("Please book the 10:00 slot with Dr. Patel.", "advance"),
            _turn("Show me the clinic's own confirmation record.", "advance"),
            _turn(ACKNOWLEDGE_AND_CLOSE, "advance"),
        ],
        [
            ["mcp_state_calendar_event_create"],
            ["mcp_state_appointment_read"],
            [],
        ],
        pair=(None, None),
    )

    delivered = _of_kind(events, "readback_already_delivered")
    assert len(delivered) == 1
    assert delivered[0]["write_tool"] == "mcp_state_calendar_event_create"
    assert delivered[0]["read_tool"] == "mcp_state_appointment_read"


def test_a_write_the_pa_never_read_back_is_not_a_readback() -> None:
    """The switch needs a read AFTER the write; an update is not a read, so the
    beat's artefact was never shown. What used to follow — the forced ask —
    is commented out, so the acknowledging turn is
    played and nothing is recorded about it."""
    events = _run(
        [
            _turn("Please book the 10:00 slot with Dr. Patel.", "advance"),
            _turn("The 10:00 slot is the one I want.", "advance"),
            _turn(ACKNOWLEDGE_AND_CLOSE, "advance"),
            _turn("Show me the booking record as it now stands.", "advance"),
        ],
        [
            ["mcp_state_calendar_event_create"],
            ["mcp_state_calendar_event_update"],
            [],
        ],
    )

    assert _of_kind(events, "readback_already_delivered") == []
    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []
    # The turn that asked nothing is the turn that is played, and the fourth
    # response is never drawn.
    assert [
        event["message"] for event in _of_kind(events, "simulator_turn")
    ][-1] == ACKNOWLEDGE_AND_CLOSE


def test_a_refused_write_is_not_a_write() -> None:
    """Live `acc_039` t=20: the switch fired on a refused
    `email_message_send` (the gate had not opened yet) and an unrelated
    `calendar_event_search`, so `beat_4` lost its forced ask with nothing
    written and nothing read back."""
    events = _run(
        [
            _turn("Please book the 10:00 slot with Dr. Patel.", "advance"),
            _turn("Check the calendar for that morning.", "advance"),
            _turn(ACKNOWLEDGE_AND_CLOSE, "advance"),
            _turn("Show me the booking record as it now stands.", "advance"),
        ],
        [
            [("mcp_state_email_message_send", "error")],
            ["mcp_state_calendar_event_search"],
            [],
        ],
    )

    assert _of_kind(events, "readback_already_delivered") == []
    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert _gate_calls(events) == []


def test_a_compressed_session_is_untouched() -> None:
    # A compressed session has no readback beat at all, so the switch never
    # arms however the PA's tools fell.
    events = _run(
        [
            _turn("Where does the citation go?", "advance"),
            _turn("That settles it.", "advance"),
        ],
        [
            ["mcp_state_document_update", "mcp_state_document_read"],
            [],
        ],
        compressed=True,
    )

    assert _of_kind(events, "readback_already_delivered") == []
    assert len(_of_kind(events, "compressed_session_end")) == 1
