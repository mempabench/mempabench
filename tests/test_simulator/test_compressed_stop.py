"""A compressed session is exactly two beats and stops after the second.

Owner ruling (2026-09-07): for a target attribute in `COMPRESSED_ATTRIBUTES`
the session is the probe and one reaction — one user turn and one PA reply —
and then it ends. Not "ends unless the goal was already met", not "ends unless
a hold is still open", not "ends unless the reaction asked for something": the
stop is unconditional, and the things that would otherwise extend the session
are recorded on a `compressed_session_end` event instead of acted on.

Owner ruling A (2026-09-08): beat 1 does not keep the ordinary loop either. It
is the probe — the user states the whole query and the PA replies once — so it
advances to the reaction after that reply whatever `turn_assessment` said, an
open hold included. What it asked for is recorded on the same event as
`probe_ignored_routing`. The one carve-out is a PA turn that delivered nothing:
the probe is the one measurement, so an empty reply takes the ordinary
empty-reply retry first. The reaction beat has no such carve-out.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_compressed_stop.py
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import pytest

from pa.agent.ix.render import IX_EMPTY_REPLY_LINE
from pa.bus.events import OutboundMessage

from test_simulator.judge_fake import ScriptedJudgeClient

from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.judge.catalogue import NO_EXECUTED_ACTS, OUTREACH_ACT
from simulator.judge.client import JudgeCall
from simulator.judge.session import JudgeSession
from simulator.loop import SimulatorLoop
from simulator.output import SimulatorOutputContractError
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    COMPRESSED_ATTRIBUTES,
    OPENING_AFTER_SEND,
    OPENING_AFTER_SUPPRESS,
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    ProactiveTrigger,
    SessionScript,
)
from simulator.self_report import SessionEndSelfReporter


def _turn(message: str, next_beat: str, goal_already_met: str | None = None) -> str:
    met = "" if goal_already_met is None else f"goal_already_met: {goal_already_met}\n"
    return (
        f"<message>\n{message}\n</message>\n\n"
        "<factual_check>\n</factual_check>\n\n"
        f"<turn_assessment>\nnext_beat: {next_beat}\n{met}</turn_assessment>"
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


class _RecordingPA:
    """A PA that holds nothing."""

    def __init__(self, held_items: list[dict[str, str]] | None = None) -> None:
        self.held_items = held_items or []

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
    ) -> OutboundMessage:
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="PA reply",
            metadata=(
                {"held_items": list(self.held_items), "has_pending_interaction": True}
                if self.held_items
                else {}
            ),
        )


def _compressed_script() -> SessionScript:
    """The two-beat shape: the probe, then one pure reaction beat."""
    return SessionScript(
        session_id="compressed",
        scenario="the assistant answers a placement question",
        context="work",
        task="Settle where the citation goes.",
        life_context=LifeContext(),
        compressed=True,
        beats=[
            Beat(
                beat="b1_probe",
                goal="Have the assistant lay out its placement call.",
                active_skills=["verbosity"],
            ),
            Beat(beat="b2_react", goal="React to how it came back."),
        ],
    )


def _five_beat_script() -> SessionScript:
    return SessionScript(
        session_id="uncompressed",
        scenario="the assistant sorts out dinner",
        context="personal",
        task="Get tonight's dinner sorted.",
        life_context=LifeContext(),
        beats=[
            Beat(beat="b1_open", goal="Have the assistant read Marcus's message."),
            Beat(beat="b2_check", goal="Have the assistant check tonight."),
            Beat(
                beat="b3_probe",
                goal="Ask for a replacement venue.",
                active_skills=["solution_breadth"],
            ),
            Beat(beat="b4_react", goal="Take the recommended venue."),
            Beat(beat="b5_close", goal="Have the assistant message Marcus."),
        ],
    )


#: What the judge says about `_INVENTED` below: the turn credits the assistant
#: with an act that `<executed_acts>` does not carry.
_INVENTION_ANSWERS = {
    "reaction_asserts_unexecuted_pa_action__q1": (
        "yes",
        "You have already updated the whole bibliography",
    )
}


#: The one turn in this file that puts nothing to the assistant. The ask gate
#: is the judge's alone, so a test about the gate has to say so.
_ACKNOWLEDGEMENT = ("Fine. That is more than I needed.",)


def _run(
    responses: list[str],
    script: SessionScript,
    pa: _RecordingPA | None = None,
    judge: Any | None = None,
    judge_answers: dict[str, tuple[str, str]] | None = None,
    no_ask: tuple[str, ...] = (),
) -> tuple[list[dict], _ScriptedProvider]:
    async def run():
        provider = _ScriptedProvider(responses)
        loop = SimulatorLoop(
            provider,
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            script,
            pa or _RecordingPA(),
            backend=NanobotRunnerSimulatorBackend(provider),
            judge=judge
            or (
                JudgeSession(
                    mode="live",
                    client=ScriptedJudgeClient(judge_answers or {}, no_ask=no_ask),
                )
                if judge_answers is not None or no_ask
                else None
            ),
        )
        return await loop.run_session(), provider

    return asyncio.run(run())


def _of_kind(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["kind"] == kind]


# -- the attribute table ----------------------------------------------------

def test_the_simulator_copy_of_the_table_matches_the_authoring_schema() -> None:
    """`scripts/session_authoring_schema.py` owns the table; the simulator keeps
    a copy because it never imports `scripts/` at runtime. The two must agree.
    """
    schema = pytest.importorskip(
        "scripts.session_authoring_schema",
        reason="scripts/session_authoring_schema.py is not importable",
    )
    table = getattr(schema, "COMPRESSED_ATTRIBUTES", None)
    if table is None:
        pytest.skip(
            "scripts.session_authoring_schema.COMPRESSED_ATTRIBUTES does not "
            "exist yet (the authoring-schema half of the compressed-session "
            "change has not landed); the simulator copy stands alone until it "
            "does"
        )
    assert set(table) == set(COMPRESSED_ATTRIBUTES)


def test_the_table_is_the_nine_attributes_the_owner_named() -> None:
    assert COMPRESSED_ATTRIBUTES == frozenset(
        {
            "verbosity",
            "tone_formality",
            "emotional_engagement",
            "guidance_level",
            "solution_breadth",
            "uncertainty_expression",
            "task_expansion",
            "capability_boundary",
            # The ninth, added 2026-09-08: triggered, so its probe is the PA's
            # own trigger turn and its two beats are the two openings.
            "proactive_outreach",
        }
    )


def test_the_flag_is_derived_from_the_drafts_target_attribute() -> None:
    """The runtime script never reads the table itself: the flag is stamped
    once, at the authored-draft -> `SessionScript` conversion.

    The two sides of the flag are two fixtures rather than one fixture with its
    `target_cell` rewritten: a compressed draft is exactly two beats and an
    uncompressed one at least five, so a draft re-targeted across the boundary
    no longer validates.
    """
    from pathlib import Path

    import yaml
    from scripts.session_authoring_schema import NormalSessionDraft

    from simulator.schemas import authored_session_to_session_script

    fixtures = Path(__file__).resolve().parents[1] / "fixtures/authoring"

    def _draft(name: str) -> NormalSessionDraft:
        data = yaml.safe_load((fixtures / name).read_text(encoding="utf-8"))
        return NormalSessionDraft.model_validate(data)

    compressed = _draft("normal_session_compressed.yaml")
    assert compressed.meta.target_attribute in COMPRESSED_ATTRIBUTES
    assert authored_session_to_session_script(compressed).compressed is True

    uncompressed = _draft("normal_session_acc_001.yaml")
    assert uncompressed.meta.target_attribute not in COMPRESSED_ATTRIBUTES
    assert authored_session_to_session_script(uncompressed).compressed is False


def test_a_triggered_probe_converts_to_a_triggered_compressed_script() -> None:
    """A `proactive_outreach` probe is the one probe with a
    trigger: the probe IS the PA's trigger turn, so the script it converts to
    carries the trigger, runs under the triggered contract and stops after the
    single opening that played -- the same shape the accumulation side runs.

    A probe draft authors no contract class and no target setting, so both come
    from the conversion; the other probes stay uncompressed and classless.
    """
    from simulator.schemas import authored_session_to_session_script

    from test_probe_session_authoring import _probe, _triggered_probe

    script = authored_session_to_session_script(_triggered_probe())
    assert script.proactive_trigger is not None
    assert script.runtime_contract_class == "triggered_interaction"
    assert script.compressed is True
    assert [beat.beat for beat in script.beats] == [OPENING_AFTER_SEND, OPENING_AFTER_SUPPRESS]

    question_shaped = authored_session_to_session_script(_probe())
    assert question_shaped.proactive_trigger is None
    assert question_shaped.runtime_contract_class is None
    assert question_shaped.compressed is False


# -- uncompressed sessions are untouched ------------------------------------

def test_a_five_beat_uncompressed_script_runs_all_five_beats() -> None:
    events, _ = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("And tonight?", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _turn("That one will do.", "advance", "false"),
            _turn("Tell Marcus, then.", "advance", "false"),
        ],
        _five_beat_script(),
    )

    assert _of_kind(events, "compressed_session_end") == []
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == [
        "b1_open",
        "b2_check",
        "b3_probe",
        "b4_react",
        "b5_close",
    ]
    assert _of_kind(events, "session_end")[0]["beat_count"] == 5


# -- the probe advance ------------------------------------------------------

def test_the_probe_advances_after_one_pa_reply_despite_a_stay() -> None:
    """Owner ruling A: beat 1 is the probe — the user states the whole query,
    the PA replies once, and the loop moves on. `next_beat: stay` with
    `goal_already_met: false` is the whole case for staying, and it buys
    nothing. Live regress14 (`acc_004`, `acc_002`) stayed in beat 1 for two and
    three exchanges on exactly this assessment and ran the task on inside the
    probe.
    """
    events, provider = _run(
        [
            _turn("Where does the citation go?", "stay", "false"),
            _turn("That is more than I asked for.", "advance", "false"),
            # The loop must never reach this one.
            _turn("And the rest of it?", "advance", "false"),
        ],
        _compressed_script(),
    )

    assert [e["beat"] for e in _of_kind(events, "beat_enter")] == ["b1_probe", "b2_react"]
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == [
        "b1_probe",
        "b2_react",
    ]
    assert [e["beat"] for e in _of_kind(events, "pa_turn")] == ["b1_probe", "b2_react"]
    assert len(provider.calls) == 2
    # Leaving the probe is an advance, not a spent budget.
    assert _of_kind(events, "beat_turn_limit") == []
    end = _of_kind(events, "compressed_session_end")[0]
    assert end["exchanges_in_probe"] == 1
    # Recorded and dropped, the same way the reaction's routing is.
    assert end["probe_ignored_routing"] == "stay"


class _EmptyFirstReplyPA:
    """Delivers the wrapper's empty-reply line once, then answers normally.

    `ix_empty_reply` is the wrapper's own control record: the PA model returned
    no content, the one retry came back empty too, and `pa/agent/runner.py`
    closed the turn with a template line. It rides `tool_events`, which is the
    only place the simulator can see it.
    """

    def __init__(self) -> None:
        self.turns = 0

    async def process_direct_message(self, *args: Any, **kwargs: Any) -> OutboundMessage:
        self.turns += 1
        empty = self.turns == 1
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content=IX_EMPTY_REPLY_LINE if empty else "PA reply",
            metadata={
                "tool_events": (
                    [{"ix_event": True, "name": "ix_empty_reply",
                      "stop_reason": "completed"}]
                    if empty
                    else []
                ),
                "held_items": [],
            },
        )


def test_an_empty_pa_reply_does_not_spend_the_probe() -> None:
    """The probe is the one measurement, so a turn that delivered nothing is
    not it. The empty reply gets the ordinary empty-reply retry — the same cap
    as any other beat — and only the reply that actually landed advances to the
    reaction. The reaction beat's stop is untouched by this and stays
    unconditional.
    """
    pa = _EmptyFirstReplyPA()
    events, provider = _run(
        [
            _turn("Where does the citation go?", "stay", "false"),
            _turn("Anything at all?", "stay", "false"),
            _turn("That is more than I asked for.", "advance", "false"),
            # The loop must never reach this one.
            _turn("And the rest of it?", "advance", "false"),
        ],
        _compressed_script(),
        pa,
    )

    # Three PA turns: the empty one, the retry that landed, and the reaction.
    assert pa.turns == 3
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == [
        "b1_probe",
        "b1_probe",
        "b2_react",
    ]
    assert len(provider.calls) == 3
    assert [e["beat"] for e in _of_kind(events, "beat_enter")] == ["b1_probe", "b2_react"]
    assert _of_kind(events, "beat_turn_limit") == []
    end = _of_kind(events, "compressed_session_end")[0]
    # The empty turn is counted in what the probe cost, and it is the landed
    # reply's routing that was ignored.
    assert end["exchanges_in_probe"] == 2
    assert end["probe_ignored_routing"] == "stay"


# -- the hard stop ----------------------------------------------------------

def test_the_session_ends_after_the_reaction_exchange_despite_a_stay() -> None:
    """`goal_already_met: false` and `next_beat: stay` between them are the
    whole case for another turn, and neither buys one."""
    events, provider = _run(
        [
            _turn("Where does the citation go?", "advance", "false"),
            _turn("That is more than I asked for.", "stay", "false"),
        ],
        _compressed_script(),
    )

    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == [
        "b1_probe",
        "b2_react",
    ]
    assert [e["beat"] for e in _of_kind(events, "pa_turn")] == ["b1_probe", "b2_react"]
    # The reaction's routing never reached a third turn of any kind.
    assert len(provider.calls) == 2
    assert _of_kind(events, "beat_turn_limit") == []
    assert _of_kind(events, "beat_collapsed") == []
    # Nothing is said after the reaction's PA reply.
    last_pa_turn = max(i for i, e in enumerate(events) if e["kind"] == "pa_turn")
    assert [e["kind"] for e in events[last_pa_turn:]] == [
        "pa_turn",
        "compressed_session_end",
        "session_end",
    ]


def test_the_reaction_beat_is_not_collapsed_when_the_goal_is_already_met() -> None:
    """The reaction beat is played even with nothing left to ask for: the
    reaction to the probe's surface is the measurement."""
    script = _compressed_script()
    # The worst case for the collapse guard: the probe declares no skills, so
    # `previous_active_skills` no longer protects the beat after it.
    script.beats[0].active_skills = []
    events, _ = _run(
        [
            _turn("Where does the citation go?", "advance", "false"),
            _turn("You have settled it already, and at length.", "stay", "true"),
        ],
        script,
    )

    assert _of_kind(events, "beat_collapsed") == []
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == [
        "b1_probe",
        "b2_react",
    ]
    assert len(_of_kind(events, "compressed_session_end")) == 1


def test_the_event_carries_the_probe_cost_the_open_hold_and_the_routing() -> None:
    """The probe routed to `stay` with the PA still holding an item — the whole
    case for a second exchange inside beat 1 — and the reaction routed to `stay`
    too. All of it is on the event; none of it changes what the loop does.
    """
    events, _ = _run(
        [
            _turn("Can you send the extension request?", "stay", "false"),
            _turn("What is stopping you?", "stay", "false"),
            # The loop must never reach these.
            _turn("Then say what you can do.", "stay", "false"),
            _turn("Fine, that is a straight answer.", "stay", "false"),
        ],
        _compressed_script(),
        _RecordingPA([{"interaction_item_id": "i3", "intent": "send_email"}]),
    )
    ends = _of_kind(events, "compressed_session_end")

    assert len(ends) == 1
    assert ends[0]["beat"] == "b2_react"
    assert ends[0]["exchanges_in_probe"] == 1
    assert ends[0]["open_hold"] is True
    assert ends[0]["probe_ignored_routing"] == "stay"
    assert ends[0]["ignored_routing"] == "stay"
    # One exchange each, the open hold notwithstanding.
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == [
        "b1_probe",
        "b2_react",
    ]
    # Recorded, not resolved: the hold is left exactly as the PA left it.
    assert _of_kind(events, "session_end")[0]["beat_count"] == 2


def test_a_one_exchange_probe_with_nothing_held_reports_that() -> None:
    events, _ = _run(
        [
            _turn("Where does the citation go?", "advance", "false"),
            _turn("Shorter next time.", "advance", "false"),
        ],
        _compressed_script(),
    )
    end = _of_kind(events, "compressed_session_end")[0]

    assert end["exchanges_in_probe"] == 1
    assert end["open_hold"] is False
    assert end["ignored_routing"] == "advance"


def test_an_uncompressed_script_with_two_beats_still_runs_normally() -> None:
    """The stop keys off the flag, not the beat count."""
    script = _compressed_script()
    script.compressed = False
    events, _ = _run(
        [
            _turn("Where does the citation go?", "advance", "false"),
            _turn("That is more than I asked for.", "stay", "false"),
            _turn("Still too much.", "stay", "false"),
            _turn("Leave it there.", "advance", "false"),
        ],
        script,
    )

    assert _of_kind(events, "compressed_session_end") == []
    assert len([e for e in _of_kind(events, "simulator_turn")]) == 4


# -- the triggered compressed session ---------------------------------------
#
# `proactive_outreach` is compressed and always triggered, so its probe is not
# a beat: it is the PA's own trigger turn, where sending is High and staying
# silent is Low. The two authored beats are the two entry-conditioned openings
# and `_beats_after_outreach` drops the one the decision did not pick, so the
# opening that plays is the reaction and the session stops after it — first
# beat entered, not second. Owner ruling, 2026-09-08.


class _ProactivePA:
    """Takes the trigger turn, then answers direct messages like `_RecordingPA`."""

    def __init__(self, decision: str) -> None:
        self.decision = decision
        self.direct_calls = 0

    async def process_environment_event(self, **kwargs: Any) -> OutboundMessage:
        message = "The water is off Monday morning." if self.decision == "send" else ""
        return OutboundMessage(
            channel="benchmark",
            chat_id="direct",
            content=message,
            metadata={
                "outreach_decision": self.decision,
                "outreach_delivered": self.decision == "send",
            },
        )

    async def process_direct_message(self, *args: Any, **kwargs: Any) -> OutboundMessage:
        self.direct_calls += 1
        return OutboundMessage(channel="benchmark", chat_id="direct", content="PA reply")


def _triggered_compressed_script() -> SessionScript:
    """Two beats, and they are the two openings. Nothing follows the one played."""
    return SessionScript(
        session_id="triggered_compressed",
        scenario="a building notice arrives while the user works through the afternoon",
        context="personal",
        task="Have the assistant judge whether the notice was worth a ping.",
        life_context=LifeContext(),
        compressed=True,
        runtime_contract_class="triggered_interaction",
        proactive_trigger=ProactiveTrigger(
            trigger_id="water_shutoff_email_delivery",
            activation_type="external_event",
            background="Building management writes to the residents.",
            detail="The stack serving the user's line is down for a morning.",
            user_impact="The user loses cold water for part of a working morning.",
        ),
        beats=[
            Beat(
                beat=OPENING_AFTER_SEND,
                goal="React to having been contacted about it.",
                active_skills=["proactive_outreach"],
                entry_condition="A message from the assistant has arrived.",
            ),
            Beat(
                beat=OPENING_AFTER_SUPPRESS,
                goal="React to not having been contacted about it.",
                active_skills=["proactive_outreach"],
                entry_condition="No contact came and the user found the notice himself.",
            ),
        ],
    )


@pytest.mark.parametrize(
    ("decision", "played", "skipped"),
    [
        ("send", OPENING_AFTER_SEND, OPENING_AFTER_SUPPRESS),
        ("suppress", OPENING_AFTER_SUPPRESS, OPENING_AFTER_SEND),
    ],
)
def test_a_triggered_compressed_session_stops_after_the_one_opening(
    decision: str, played: str, skipped: str
) -> None:
    pa = _ProactivePA(decision)
    events, _ = _run(
        [
            _turn("So that is what you thought was worth my afternoon.", "stay", "false"),
            # The loop must never reach this one.
            _turn("And the rest of it?", "advance", "false"),
        ],
        _triggered_compressed_script(),
        pa=pa,
    )

    assert [e["beat"] for e in _of_kind(events, "beat_enter")] == [played]
    assert [e["beat"] for e in _of_kind(events, "simulator_turn")] == [played]
    assert skipped == _of_kind(events, "outreach_path")[0]["skipped_beat"]
    end = _of_kind(events, "compressed_session_end")
    assert [e["beat"] for e in end] == [played]
    # The trigger turn is the probe and holds no simulator exchange of its own.
    assert end[0]["exchanges_in_probe"] == 0
    # One user message and one PA reply, then the stop — the scripted "stay"
    # did not buy a second exchange.
    assert pa.direct_calls == 1
    assert _of_kind(events, "session_end")[0]["beat_count"] == 1


# -- the played opening is never pushed to ask ------------------------------

PURE_REACTION = (
    "I did not ask to be notified about building maintenance, and I would "
    "prefer you refrain from interrupting my work with unsolicited updates."
)


@pytest.mark.parametrize("decision", ["send", "suppress"])
def test_the_played_opening_is_not_forced_to_carry_an_ask(decision: str) -> None:
    """Live regress14c `acc_007` t4-t6. Both openings declare the attribute by
    contract, so `require_ask` derived from `bool(beat.active_skills)` threw the
    clean reaction away and the retry came back with "What are the specific
    hours for this water shutoff?" — a question the beat's own `avoid` forbids.
    The played opening is the reaction beat, and a reaction beat never asks.
    """
    events, provider = _run(
        [
            _turn(PURE_REACTION, "stay", "false"),
            _turn("And what were the hours?", "advance", "false"),
        ],
        _triggered_compressed_script(),
        pa=_ProactivePA(decision),
    )

    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    # One generation, not two: the reaction that asked for nothing was played.
    assert len(provider.calls) == 1
    assert [e["message"] for e in _of_kind(events, "simulator_turn")] == [PURE_REACTION]


SENT_REACTION = (
    "An unrequested message arrived while no request was active, and having "
    "it sent was unwelcome."
)


def test_a_delivered_outreach_is_listed_for_the_judge_on_the_played_opening() -> None:
    """Live regress24m `acc_029`. The PA decided `send`, the harness delivered
    the message, and no task tool ran — so the write check read the session
    write-free and the actor's true "having it sent was unwelcome" died as an
    act credited to an assistant that had carried none out. The delivery is an
    act, and since the owner's ruling of 2026-09-12 it is a line of the list
    the judge is given rather than a boolean that clears the turn.
    """
    client = ScriptedJudgeClient(
        {"reaction_asserts_unexecuted_pa_action__q1": ("no", "")},
        no_ask=(SENT_REACTION,),
    )
    events, _ = _run(
        [_turn(SENT_REACTION, "stay", "false")],
        _triggered_compressed_script(),
        pa=_ProactivePA("send"),
        judge=JudgeSession(mode="live", client=client),
    )
    asked = [
        prompt
        for prompt in client.prompts
        if "reaction_asserts_unexecuted_pa_action__q1" in prompt
    ]

    assert asked and OUTREACH_ACT in asked[0]
    assert [e["message"] for e in _of_kind(events, "simulator_turn")] == [SENT_REACTION]
    assert _of_kind(events, "session_end")[0]["beat_count"] == 1


def test_a_suppressed_outreach_leaves_the_list_empty() -> None:
    """The negative: nothing was delivered, so the assistant has carried out
    nothing and a turn crediting it with a send is still the rule's own case.
    """
    client = ScriptedJudgeClient(
        {"reaction_asserts_unexecuted_pa_action__q1": ("yes", "candidate_turn#1")},
        no_ask=(SENT_REACTION,),
    )
    events, _ = _run(
        [_turn(SENT_REACTION, "stay", "false")] * 6,
        _triggered_compressed_script(),
        pa=_ProactivePA("suppress"),
        judge=JudgeSession(mode="live", client=client),
    )
    asked = [
        prompt
        for prompt in client.prompts
        if "reaction_asserts_unexecuted_pa_action__q1" in prompt
    ]

    assert asked and NO_EXECUTED_ACTS in asked[0]
    # Non-blocking since the owner's ruling of 2026-09-13: the empty list still
    # makes the claim an invention, and the turn is played with it recorded.
    assert _of_kind(events, "simulator_validation_failure") == []
    assert [e["diagnostics"] for e in _of_kind(events, "simulator_turn_diagnostics")] == [
        ["reaction_asserts_unexecuted_pa_action"]
    ]


@pytest.mark.skip(reason="forced-ask check off, owner ruling 2026-09-14")
def test_an_untriggered_compressed_probe_still_owes_an_ask() -> None:
    """The other half of the same flag: beat 1 of an untriggered compressed
    session is the probe, it declares the attribute, and an acknowledgement
    there is a miss. Record-only since owner ruling 2026-09-14 ("只记"): the
    gate still arms, and it records instead of re-asking."""
    events, provider = _run(
        [
            _turn("That is more than I needed.", "advance", "false"),
            _turn("Where does the citation go, and show me how it reads?", "advance", "false"),
            _turn("Fine.", "advance", "false"),
        ],
        _compressed_script(),
        no_ask=("That is more than I needed.",),
    )

    assert _of_kind(events, "beat_ask_retry") == []
    unfulfilled = _of_kind(events, "beat_ask_unfulfilled")
    assert [e["beat"] for e in unfulfilled] == ["b1_probe"]
    assert unfulfilled[0]["attempts"] == 1
    assert unfulfilled[0]["aborted"] is False
    assert [e["message"] for e in _of_kind(events, "simulator_turn")][0] == (
        "That is more than I needed."
    )
    assert len(provider.calls) == 2


ASK_LINE = "This is the step where you ask for what this conversation is about."
REACTION_LINE = (
    "This is the one turn where you react to whether the assistant got in touch"
)


def test_the_played_opening_is_told_to_react_rather_than_to_ask() -> None:
    """The other half of the same defect: the beat block put the ask line a few
    lines above the beat's own "Avoid: do not ask what the message said"."""
    _, provider = _run(
        [_turn(PURE_REACTION, "stay", "false")],
        _triggered_compressed_script(),
        pa=_ProactivePA("send"),
    )

    system = str(provider.calls[0][0]["content"])
    assert REACTION_LINE in system
    assert ASK_LINE not in system


OPEN_WITH_THE_ASK = "open with the ask itself"


def _trigger_of(messages: list[dict[str, Any]]) -> str:
    """The per-turn trigger: the last user message the backend sent."""
    return next(
        str(m.get("content") or "")
        for m in reversed(messages)
        if m.get("role") == "user"
    )


def test_the_suppress_branch_is_not_told_to_open_with_an_ask() -> None:
    """Nothing is delivered when the assistant stays silent, so the turn's
    history is empty and the trigger fell to the opening-beat wording — which
    tells the actor to open with the ask, against everything the beat block
    says on this beat."""
    _, provider = _run(
        [_turn(PURE_REACTION, "stay", "false")],
        _triggered_compressed_script(),
        pa=_ProactivePA("suppress"),
    )

    trigger = _trigger_of(provider.calls[0])
    assert OPEN_WITH_THE_ASK not in trigger
    assert "stayed silent" in trigger
    assert "Ask the assistant for nothing." in trigger


def test_the_send_branch_trigger_is_the_ordinary_one() -> None:
    """A delivered message is in the history, so the send branch never took the
    opening-beat wording and is left exactly as it was."""
    _, provider = _run(
        [_turn(PURE_REACTION, "stay", "false")],
        _triggered_compressed_script(),
        pa=_ProactivePA("send"),
    )

    trigger = _trigger_of(provider.calls[0])
    assert OPEN_WITH_THE_ASK not in trigger
    assert "stayed silent" not in trigger
    assert trigger.startswith(f"Produce only the current `{OPENING_AFTER_SEND}` beat.")


def test_an_untriggered_compressed_probe_keeps_the_ask_line() -> None:
    _, provider = _run(
        [
            _turn("Where does the citation go, and show me how it reads?", "advance", "false"),
            _turn("Fine. That is more than I needed.", "advance", "false"),
        ],
        _compressed_script(),
    )

    system = str(provider.calls[0][0]["content"])
    assert ASK_LINE in system
    assert REACTION_LINE not in system


# -- the played opening is the reaction beat --------------------------------

@pytest.mark.parametrize("decision", ["send", "suppress"])
def test_the_played_opening_arms_the_reaction_beat_checks(decision: str) -> None:
    """`_is_reaction_beat` could not derive this from `session.beats`: the
    played opening is beat index 0 and declares the attribute by contract, and
    the old derivation rejected it on both counts, so every reaction-beat check
    was dead on either branch of every triggered session."""
    from simulator.agent.loop import _is_reaction_beat

    script = _triggered_compressed_script()
    _run(
        [_turn(PURE_REACTION, "stay", "false")],
        script,
        pa=_ProactivePA(decision),
    )
    played = OPENING_AFTER_SEND if decision == "send" else OPENING_AFTER_SUPPRESS
    assert script.compressed_reaction_beat == played

    # Either opening, whichever the decision played.
    for beat in script.beats:
        script.compressed_reaction_beat = beat.beat
        assert _is_reaction_beat(script, beat) is True


class _SilentJudgeClient:
    """Records that it was never spoken to. `mode="off"` opens no socket."""

    model = "scripted-judge"
    base_url = "http://10.0.0.2:30001/v1"

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def ask(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        self.calls.append((args, kwargs))
        raise AssertionError("the judge was called with MEMPA_JUDGE=off")


@pytest.mark.parametrize("decision", ["send", "suppress"])
def test_the_played_opening_arms_the_reception_beat_checks(decision: str) -> None:
    """The other half of the same hole (2026-09-09).

    `_is_reaction_beat` had already been taught to read the stamp; the
    `reception_beat` line still asked for a carried attribute the played
    opening cannot have — it is the first beat entered and it declares the
    attribute itself, so `reception_carry_attributes` subtracts it away.
    Every reception check was therefore unarmed on the one beat a triggered
    compressed session measures, regress21 `acc_007` among them. Armed here
    means armed: the invention is thrown back, not merely tolerated.

    Both decisions again, as before 197fbd8e. That commit dropped `send`
    because a delivered outreach set the session boolean and the boolean
    cleared every claim the turn made, the invented bibliography update
    included. The owner's ruling of 2026-09-12 took the boolean out: the
    outreach is one line of a list, an invention that is not on the list is
    still an invention, and the decision changes neither the arming nor the
    verdict here.
    """
    events, provider = _run(
        [_INVENTED, _turn(PURE_REACTION, "advance", "false")],
        _triggered_compressed_script(),
        pa=_ProactivePA(decision),
        judge_answers=_INVENTION_ANSWERS,
    )

    # The arming is what this test is about, and the `judge_call` row is what
    # proves it: the rule is asked on the played opening. Non-blocking since
    # the owner's ruling of 2026-09-13, so the answer is recorded and the
    # invention is played on the first attempt instead of being rewritten.
    assert [
        e["verdicts"]
        for e in _of_kind(events, "judge_call")
        if e["rule"] == "reaction_asserts_unexecuted_pa_action"
    ] == [{"reaction_asserts_unexecuted_pa_action__q1": "yes"}]
    assert _of_kind(events, "simulator_validation_failure") == []
    assert [e["diagnostics"] for e in _of_kind(events, "simulator_turn_diagnostics")] == [
        ["reaction_asserts_unexecuted_pa_action"]
    ]
    assert len(provider.calls) == 1


@pytest.mark.parametrize("decision", ["send", "suppress"])
def test_an_invention_on_the_played_opening_no_longer_voids(
    decision: str,
) -> None:
    """The void path, gone: non-blocking, owner ruling 2026-09-13.

    The played opening IS `compressed_reaction_beat`, so the
    `PLAYED_AFTER_BUDGET` carve-out used to end the step on a contract error
    here rather than deliver the invention. The rule never reaches `findings`
    now, so the carve-out is never read and the fabricated act plays through
    on the first attempt, recorded on the turn's diagnostics.

    Both decisions again, for the reason above: the delivered outreach no
    longer clears an invention nothing on the list accounts for.
    """
    events, provider = _run(
        [_INVENTED, _INVENTED, _INVENTED, _INVENTED],
        _triggered_compressed_script(),
        pa=_ProactivePA(decision),
        judge_answers=_INVENTION_ANSWERS,
    )

    assert _of_kind(events, "simulator_validation_failure") == []
    assert [e["diagnostics"] for e in _of_kind(events, "simulator_turn_diagnostics")] == [
        ["reaction_asserts_unexecuted_pa_action"]
    ]
    assert len(_of_kind(events, "simulator_turn")) == 1
    assert len(provider.calls) == 1


def test_the_triggered_arming_calls_no_judge_with_the_judge_off(monkeypatch) -> None:
    """This is the one commit of the Phase 3 ladder that changes behaviour
    with the judge off, so the mode-off case asserts only that the judge
    was not called — acc_007's findings are meant to change."""
    monkeypatch.delenv("MEMPA_JUDGE", raising=False)
    client = _SilentJudgeClient()
    _run(
        [_INVENTED, _turn(PURE_REACTION, "advance", "false")],
        _triggered_compressed_script(),
        pa=_ProactivePA("send"),
        judge=JudgeSession(mode="off", client=client),
    )

    assert client.calls == []


def test_the_stamp_is_cleared_while_the_probe_beat_generates() -> None:
    """The stamp is per-beat state on a script that outlives the beat, so the
    thing to check is that the probe's own turn sees `None`. If it leaked, the
    probe would get the reaction line, lose its ask, and the compressed
    session's one measurement would go with it."""
    seen: list[str | None] = []
    script = _compressed_script()

    class _StampWatchingProvider(_ScriptedProvider):
        async def chat(self, messages, *args: Any, **kwargs: Any) -> LLMResponse:
            seen.append(script.compressed_reaction_beat)
            return await super().chat(messages, *args, **kwargs)

    async def run():
        provider = _StampWatchingProvider(
            [
                _turn("Where does the citation go, and show me how it reads?", "advance", "false"),
                _turn("Fine. That is more than I needed.", "advance", "false"),
            ]
        )
        loop = SimulatorLoop(
            provider,
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            script,
            _RecordingPA(),
            backend=NanobotRunnerSimulatorBackend(provider),
        )
        await loop.run_session()

    asyncio.run(run())

    assert seen == [None, "b2_react"]


def test_the_untriggered_reaction_beat_is_still_the_second_one() -> None:
    from simulator.agent.loop import _is_reaction_beat

    script = _compressed_script()
    _run(
        [
            _turn("Where does the citation go, and show me how it reads?", "advance", "false"),
            _turn("Fine. That is more than I needed.", "advance", "false"),
        ],
        script,
    )
    assert script.compressed_reaction_beat == "b2_react"

    # Unstamped, the derivation is the one it always was: the probe is not the
    # reaction beat, the beat after it is.
    script.compressed_reaction_beat = None
    assert [_is_reaction_beat(script, beat) for beat in script.beats] == [False, True]


# -- the reaction beat is never a readback beat -----------------------------

def test_the_reaction_beat_is_never_pushed_to_ask_to_be_shown_the_artefact() -> None:
    """Owner ruling E: the write and its readback both happen inside beat 1 and
    the PA shows the result there, so beat 2 asks for nothing. `readback_beats`
    is what would break that — a marked beat sets `is_readback`, which forces
    `require_ask` and the readback retry note, and the actor would be told to
    ask to be shown an artefact it was already shown.
    `simulator.schemas.readback_beat_ids` returns `[]` for a compressed draft,
    so the reaction turn below acknowledges and is left alone.
    """
    script = _compressed_script()
    assert script.readback_beats == []
    events, provider = _run(
        [
            _turn("Where does the citation go, and show me how it reads?", "advance", "false"),
            # An acknowledgement: it asks for nothing at all.
            _turn("Fine. That is more than I needed.", "advance", "false"),
        ],
        script,
        no_ask=_ACKNOWLEDGEMENT,
    )

    assert _of_kind(events, "beat_ask_retry") == []
    assert _of_kind(events, "beat_ask_unfulfilled") == []
    assert len(provider.calls) == 2
    assert [e["message"] for e in _of_kind(events, "simulator_turn")][1] == (
        "Fine. That is more than I needed."
    )


@pytest.mark.skip(reason="forced-ask check off, owner ruling 2026-09-14")
def test_a_marked_reaction_beat_is_what_the_guard_prevents() -> None:
    """The same script and the same acknowledging turn, with the reaction beat
    marked by hand: the gate arms and the reaction is recorded as a miss. This
    is the behaviour `readback_beat_ids` no longer produces for a compressed
    draft.

    Record-only since owner ruling 2026-09-14 ("只记"), so the miss is all that
    happens: the turn is played as it came and no note goes back.
    """
    script = _compressed_script()
    script.readback_beats = ["b2_react"]
    events, provider = _run(
        [
            _turn("Where does the citation go, and show me how it reads?", "advance", "false"),
            _turn("Fine. That is more than I needed.", "advance", "false"),
            _turn("Show me how the entry reads now.", "advance", "false"),
        ],
        script,
        no_ask=_ACKNOWLEDGEMENT,
    )

    assert _of_kind(events, "beat_ask_retry") == []
    assert [e["beat"] for e in _of_kind(events, "beat_ask_unfulfilled")] == ["b2_react"]
    assert len(provider.calls) == 2
    assert not any(
        "Keep what you said" in str(m.get("content") or "")
        for call in provider.calls
        for m in call
    )


class _QuotablePA(_RecordingPA):
    """A PA whose reply is long enough for the self-report to quote it."""

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
    ) -> OutboundMessage:
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="I stopped after the answer.",
        )


def _reflection_response() -> str:
    return (
        "<session_end_reflection>It answered, then it stopped, which is what I "
        "wanted.</session_end_reflection>\n"
        "<final_emotion>\n"
        "trigger: it stopped after the answer\n"
        "reaction: satisfied\n"
        "</final_emotion>\n"
        "<self_report>\n"
        "felt_understood_in_this_context: 4\n"
        "shift_perception: pa_finally_got_it\n"
        "felt_effort: 2\n"
        "noticed_contradictions: no\n"
        "overall_experience: 4\n"
        "</self_report>\n"
        "<reflection_evidence>\n"
        "- sentence: It answered, then it stopped, which is what I wanted.\n"
        '  user_quote: ""\n'
        '  pa_quote: ""\n'
        "  turn: null\n"
        "- sentence: it stopped after the answer\n"
        '  user_quote: ""\n'
        "  pa_quote: I stopped after the answer.\n"
        "  turn: 1\n"
        "</reflection_evidence>"
    )


def test_the_compressed_stop_still_runs_the_session_end_reflection() -> None:
    """Owner (2026-09-07 night): the hard stop after beat 2 exits through the
    same end-of-session path as every other session, so the self-report
    questionnaire is still asked and its record still lands in the transcript."""

    async def run():
        provider = _ScriptedProvider(
            [
                _turn("Where does the citation go?", "advance", "false"),
                _turn("That is more than I asked for.", "stay", "false"),
            ]
        )
        reflection_provider = _ScriptedProvider([_reflection_response()])
        loop = SimulatorLoop(
            provider,
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            _compressed_script(),
            _QuotablePA(),
            backend=NanobotRunnerSimulatorBackend(provider),
            self_reporter=SessionEndSelfReporter(
                reflection_provider,
                profile_context="User A private actor context",
                model="self-report-model",
            ),
        )
        return await loop.run_session(), reflection_provider

    events, reflection_provider = asyncio.run(run())

    assert len(reflection_provider.calls) == 1
    reflections = _of_kind(events, "session_end_reflection")
    assert len(reflections) == 1
    assert reflections[0]["reflection"].startswith("It answered, then it stopped")
    assert reflections[0]["final_emotion"] == {
        "trigger": "it stopped after the answer",
        "reaction": "satisfied",
    }
    assert _of_kind(events, "session_end_reflection_missing") == []
    kinds = [e["kind"] for e in events]
    assert kinds.index("compressed_session_end") < kinds.index("session_end_reflection")
    assert kinds.index("session_end_reflection") < kinds.index("session_end")


# -- an invented PA action on the reaction beat -----------------------------
# Owner ruling (2026-09-08): on a compressed session the reaction beat's one
# exchange IS the measurement, so a reaction to work the assistant never did
# scores the attribute against a fiction — an instrument error, not a style
# note. Live regress16 `acc_006` t7 credited the PA with reformatting the whole
# bibliography into APA style; its only prior turn had read two documents and
# written nothing, and every one of the 22 updates it went on to make was a
# deletion. The check existed and fired, but only in the session loop, after
# the retry budget was spent, so the turn was delivered anyway.


class _WritingPA(_RecordingPA):
    """A PA whose reply reports the write tools it ran."""

    def __init__(self, tool_names: list[str]) -> None:
        super().__init__()
        self.tool_names = tool_names

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str,
        metadata: dict | None = None,
    ) -> OutboundMessage:
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="PA reply",
            metadata={
                "tool_events": [{"name": name} for name in self.tool_names]
            },
        )


def _contract_retry_note(provider: _ScriptedProvider) -> str:
    return next(
        message["content"]
        for call in reversed(provider.calls)
        for message in reversed(call)
        if "violated the private output contract" in str(message.get("content"))
    )


_INVENTED_MESSAGE = (
    "You have already updated the whole bibliography into a consistent style, "
    "which I did not ask for."
)

_INVENTED = _turn(_INVENTED_MESSAGE, "advance", "false")


def test_a_reaction_crediting_a_write_that_never_ran_is_recorded_not_rewritten() -> None:
    """Non-blocking, owner ruling 2026-09-13: judged and recorded for
    observation this round, never a finding. The invention reaches the PA on
    the first attempt, with the rule and the judge's evidence on the turn's
    diagnostics."""
    events, _ = _run(
        [
            _turn("Where does the citation go, and show me how it reads?", "advance", "false"),
            _INVENTED,
            _turn("Fine. That is more than I needed.", "advance", "false"),
        ],
        _compressed_script(),
        judge_answers=_INVENTION_ANSWERS,
    )

    assert [e["message"] for e in _of_kind(events, "simulator_turn")] == [
        "Where does the citation go, and show me how it reads?",
        _INVENTED_MESSAGE,
    ]
    assert _of_kind(events, "simulator_validation_failure") == []
    diagnostics = _of_kind(events, "simulator_turn_diagnostics")
    assert [e["diagnostics"] for e in diagnostics] == [
        ["reaction_asserts_unexecuted_pa_action"]
    ]
    # The judge's own evidence rides along, so a reader has the offending
    # sentence without the transcript. The scripted words resolve to the whole
    # numbered sentence, the way every quoted answer does.
    assert diagnostics[0]["diagnostics_detail"][
        "reaction_asserts_unexecuted_pa_action"
    ] == [_INVENTED_MESSAGE]


def test_a_reaction_to_a_write_the_assistant_really_made_is_left_alone() -> None:
    """The judge is shown the tool log and reports no invention, so the same
    turn stands.

    The log side is scripted through the composed answer rather than through
    the tool-log question: a `yes` there has to quote four words of the block
    it reads, and a tool log is a list of one-word names.
    """
    events, _ = _run(
        [
            _turn("Where does the citation go, and show me how it reads?", "advance", "false"),
            _INVENTED,
        ],
        _compressed_script(),
        _WritingPA(["mcp_state_document_read", "mcp_state_document_update"]),
        judge_answers={},
    )

    assert _of_kind(events, "simulator_validation_failure") == []
    assert len(_of_kind(events, "simulator_turn")) == 2


def test_an_invention_on_a_compressed_reaction_beat_no_longer_voids() -> None:
    """The void path, gone: non-blocking, owner ruling 2026-09-13.

    On the compressed reaction beat the one exchange IS the measurement, so
    the rule used to be kept out of `PLAYED_AFTER_BUDGET` there and the step
    ended on a contract error. It never reaches `findings` now, so there is no
    retry to exhaust: the fabricated act is played on the first attempt and the
    session runs to its end with the rule on the turn's diagnostics.
    """
    events, provider = _run(
        [
            _turn("Where does the citation go, and show me how it reads?", "advance", "false"),
            _INVENTED,
            _INVENTED,
            _INVENTED,
            _INVENTED,
        ],
        _compressed_script(),
        judge_answers=_INVENTION_ANSWERS,
    )

    assert _of_kind(events, "simulator_validation_failure") == []
    assert [e["diagnostics"] for e in _of_kind(events, "simulator_turn_diagnostics")] == [
        ["reaction_asserts_unexecuted_pa_action"]
    ]
    assert [e["message"] for e in _of_kind(events, "simulator_turn")] == [
        "Where does the citation go, and show me how it reads?",
        _INVENTED_MESSAGE,
    ]
    # One generation per turn: nothing was asked for again.
    assert len(provider.calls) == 2


def test_an_uncompressed_reception_beat_is_recorded_too() -> None:
    """The arming is unchanged — every reaction beat, not the compressed one
    alone (owner ruling 2026-09-08, the three uncompressed hits over
    regress12-19: regress14c `acc_039` t17, regress18 `acc_009` t13 and
    regress18 `acc_039` t23). Non-blocking since 2026-09-13, so the turn is
    played on the first attempt and the verdict lands on its diagnostics."""
    events, provider = _run(
        [
            _turn("Anything from Marcus?", "advance", "false"),
            _turn("Tonight is clear then.", "advance", "false"),
            _turn("So where do we eat instead?", "advance", "false"),
            _INVENTED,
            _turn("That one will do.", "advance", "false"),
            _turn("Message Marcus, then.", "advance", "false"),
        ],
        _five_beat_script(),
        judge_answers=_INVENTION_ANSWERS,
    )

    assert _of_kind(events, "simulator_validation_failure") == []
    assert [e["diagnostics"] for e in _of_kind(events, "simulator_turn_diagnostics")] == [
        ["reaction_asserts_unexecuted_pa_action"]
    ]
    # The invention reached the PA; nothing was asked for again.
    assert [e["message"] for e in _of_kind(events, "simulator_turn")] == [
        "Anything from Marcus?",
        "Tonight is clear then.",
        "So where do we eat instead?",
        _INVENTED_MESSAGE,
        "That one will do.",
    ]
