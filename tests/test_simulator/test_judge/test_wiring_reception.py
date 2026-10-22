"""The reaction beat's four rules, judged and no longer matched.

The reception two — `reaction_asserts_unexecuted_pa_action` and
`reaction_authorises_pa_action` — and the reaction two —
`reaction_misstates_contractions`, `reaction_states_future_policy`.
None of them has a regex half left: the judge is the only decider, so with
`MEMPA_JUDGE=off` the beat is not judged at all and none of them can fire.

All four are record-only since the owner's ruling of 2026-09-14 ("只记"): each
is still asked and each still writes its `judge_call` row, and the verdict
rides out on the played turn's diagnostics instead of buying a rewrite. Three
of them were hard-and-played-after-the-budget before that;
`reaction_authorises_pa_action` was the one that blocked, a consequence it took
over from the deleted `authorised_before_its_beat` regex.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_judge/test_wiring_reception.py
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pa.bus.events import OutboundMessage

from test_simulator.judge_fake import ScriptedJudgeClient

from simulator.agent.loop import PLAYED_AFTER_BUDGET
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


def _turn(message: str, next_beat: str = "advance") -> str:
    return (
        f"<message>\n{message}\n</message>\n\n"
        "<factual_check>\n</factual_check>\n\n"
        f"<turn_assessment>\nnext_beat: {next_beat}\n"
        "goal_already_met: false\n</turn_assessment>"
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
    #: What each turn reports having run, if anything.
    tool_events: list[dict[str, Any]] = []

    async def process_direct_message(
        self, user_message: str, session_key: str, metadata: dict | None = None
    ) -> OutboundMessage:
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="Want me to book it? I have not booked anything yet.",
            metadata={"tool_events": [dict(event) for event in self.tool_events]},
        )


#: The path the writing PA below updates, and the value that has to reach the
#: judge: the name `mcp_state_document_update` alone does not say which record
#: the assistant touched (live regress22 `acc_006`).
WROTE_PATH = "documents/research/citation_notes.md"


class _WritingPA(_RecordingPA):
    """A PA whose turns actually wrote something, plus one IX selection.

    The write stays inside the run's own stores, and it is one of the acts
    `TurnContext.executed_acts` lists: the IX selection is not.
    """

    tool_events = [
        {"name": "IX_task_expansion", "ix": True, "status": "ok",
         "args": {"setting": "Low"}},
        {"name": "mcp_state_document_update", "status": "ok",
         "args": {"path": WROTE_PATH, "anchor": "@article{okonkwo2024arxiv",
                  "replacement": ""}},
    ]


class _SendingPA(_RecordingPA):
    """A PA whose turn left the sandbox: the other act shape the list carries."""

    tool_events = [
        {"name": "mcp_state_email_message_send", "status": "ok",
         "args": {"to": "marcus@example.com", "subject": "Tonight"}},
    ]


#: The beat's own ban, the block `reaction_authorises_pa_action__q1` reads.
RECEPTION_AVOID = (
    "Do not endorse, accept or answer any offer or question the assistant "
    "closed with, and do not tell it what to do next."
)


def _script(declares: str = "solution_breadth") -> SessionScript:
    """Four beats, the last of which carries the reception rules and declares
    none of its own — the window all three checks are armed on."""
    return SessionScript(
        session_id="judge_reception",
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
                active_skills=[declares],
            ),
            Beat(
                beat="b4_react",
                goal="React to how it came back.",
                avoid=RECEPTION_AVOID,
            ),
        ],
    )


def _run(
    reaction: str,
    answers: dict[str, tuple[str, str]] | None = None,
    mode: str = "live",
    declares: str = "solution_breadth",
    pa: Any | None = None,
) -> tuple[list[dict], ScriptedJudgeClient]:
    client = ScriptedJudgeClient(answers, no_ask=())
    provider = _ScriptedProvider(
        [
            _turn("Anything from Marcus?"),
            _turn("Tonight is clear then."),
            _turn("So where do we eat instead?"),
            _turn(reaction),
            # What the retry gets, on every rule that rewrites the turn, and
            # then whatever the beat asks for after it.
            *[_turn("That one will do.")] * 4,
        ]
    )
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script(declares),
        pa or _RecordingPA(),
        backend=NanobotRunnerSimulatorBackend(provider),
        judge=JudgeSession(mode=mode, client=client),
    )
    return asyncio.run(loop.run_session()), client


def _failures(events: list[dict]) -> list[list[str]]:
    return [
        event["error_codes"]
        for event in events
        if event["kind"] == "simulator_validation_failure"
    ]


def _diagnostics(events: list[dict]) -> list[list[str]]:
    """What the played turns carry that never became a finding."""
    return [
        event["diagnostics"]
        for event in events
        if event["kind"] == "simulator_turn_diagnostics"
    ]


def _judge_verdicts(events: list[dict], rule: str) -> list[dict[str, str]]:
    """The `judge_call` rows for one rule: the rule is still asked and recorded
    whatever the consequence of its answer."""
    return [
        event["verdicts"]
        for event in events
        if event["kind"] == "judge_call" and event["rule"] == rule
    ]


# ------------------------------------------------------------------ mode off

#: One turn that breaks both rules at once: it credits the assistant with a
#: write it never made and hands it a go-ahead its own `avoid` forbids.
BREAKS_BOTH = (
    "You have already booked the table, which I never authorised, and I asked "
    "for a single quiet venue with no music. Go ahead with the booking now."
)


def test_mode_off_leaves_the_reception_beat_unjudged() -> None:
    """Off, the two rules are not decided by anything: the pattern that used
    to decide them is gone, and the judge is never called."""
    events, client = _run(BREAKS_BOTH, None, mode="off")

    assert client.calls == []
    assert _failures(events) == []
    assert BREAKS_BOTH in [
        event["message"] for event in events if event["kind"] == "simulator_turn"
    ]


# ----------------------------------------------------------------- live, hard


def test_an_invented_write_is_recorded_and_never_a_finding() -> None:
    """Non-blocking, owner ruling 2026-09-13: judged and recorded for
    observation this round, never a finding.

    The call is made on every reaction turn exactly as before and its
    `judge_call` row carries the verdict; what changed is the consequence —
    no rewrite, no retry note, no void. The invention is played on the first
    attempt with the rule on the turn's diagnostics.
    """
    events, _ = _run(
        BREAKS_BOTH,
        {
            "reaction_asserts_unexecuted_pa_action__q1": (
                "yes",
                "You have already booked the table",
            )
        },
    )

    assert _judge_verdicts(events, "reaction_asserts_unexecuted_pa_action") == [
        {"reaction_asserts_unexecuted_pa_action__q1": "yes"}
    ]
    assert _diagnostics(events) == [["reaction_asserts_unexecuted_pa_action"]]
    assert _failures(events) == []
    assert BREAKS_BOTH in [
        event["message"] for event in events if event["kind"] == "simulator_turn"
    ]


def test_the_acts_the_session_ran_reach_the_judge() -> None:
    """The code-side half of the rule, and the live path feeds it.

    Live regress22 `acc_006` stopped four times on this rule after seven
    `mcp_state_document_update` calls that all came back ok: q2 asked the model
    to find the change in `<tool_log>` and it hunted for a tool named after the
    user's phrasing instead. Owner ruling B of 2026-09-10 deleted q2; the
    ruling of 2026-09-12 made the code side a LIST of the acts carried out, and
    the session loop hands the executed names over for it.
    """
    events, client = _run(
        BREAKS_BOTH,
        {"reaction_asserts_unexecuted_pa_action__q1": ("no", "")},
        pa=_SendingPA(),
    )
    asked = [
        prompt
        for prompt in client.prompts
        if "reaction_asserts_unexecuted_pa_action__q1" in prompt
    ]

    assert asked, "the reception beat was never judged"
    assert _failures(events) == []
    assert "email_message_send — send an email" in asked[0]
    # the block the deleted question read is not printed to it any more
    assert not [prompt for prompt in client.prompts if "<tool_log>" in prompt]


def test_a_write_inside_the_sandbox_is_one_of_the_acts() -> None:
    """An in-sandbox write is an act too, and a claim off the list still fails.

    Live user_b `acc_001` on the mem0 line (2026-09-13): the PA really ran
    `mcp_state_document_update` on the bistro staff notes, the list was built
    from external effects alone and read "none yet", and the user thanking it
    for that edit was judged a fiction four rewrites running. The list carries
    the edit now — and the point of a list rather than the boolean it replaced
    survives, because the invented booking below is still not on it.
    """
    events, client = _run(
        BREAKS_BOTH,
        {
            "reaction_asserts_unexecuted_pa_action__q1": (
                "yes",
                "You have already booked the table",
            )
        },
        pa=_WritingPA(),
    )
    asked = [
        prompt
        for prompt in client.prompts
        if "reaction_asserts_unexecuted_pa_action__q1" in prompt
    ]

    assert asked, "the reception beat was never judged"
    assert "document_update — edit an existing document" in asked[0]
    # Non-blocking since the owner's ruling of 2026-09-13: the claim off the
    # list is still judged and recorded, and it no longer fails the turn.
    assert _diagnostics(events) == [["reaction_asserts_unexecuted_pa_action"]]
    assert _failures(events) == []


def test_a_go_ahead_the_beats_avoid_forbids_is_recorded() -> None:
    """q1 reads the ban out of `avoid`, q2 the release out of the turn; the
    carve-out questions come from `avoid` too and answer `no` here.

    It was the one blocking rule of the four (it took the consequence over from
    the deleted `authorised_before_its_beat` regex). Owner ruling 2026-09-14
    ("只记"): the questions are unchanged and the verdict is recorded, but
    nothing is rewritten and nothing is voided.
    """
    events, _ = _run(
        BREAKS_BOTH,
        {
            "reaction_authorises_pa_action__q1": (
                "yes",
                "do not tell it what to do next",
            ),
            "reaction_authorises_pa_action__q2": (
                "yes",
                "Go ahead with the booking now",
            ),
        },
    )

    verdicts = _judge_verdicts(events, "reaction_authorises_pa_action")
    assert len(verdicts) == 1
    assert verdicts[0]["reaction_authorises_pa_action__q1"] == "yes"
    assert verdicts[0]["reaction_authorises_pa_action__q2"] == "yes"
    assert _failures(events) == []
    assert _diagnostics(events) == [["reaction_authorises_pa_action"]]


def test_a_repeated_go_ahead_is_played_rather_than_ending_the_step() -> None:
    """The same turn said over and over used to spend the budget and raise.
    Record-only: the first attempt plays, and the release reaches the PA with
    the rule on the turn's diagnostics."""
    answers = {
        "reaction_authorises_pa_action__q1": (
            "yes",
            "do not tell it what to do next",
        ),
        "reaction_authorises_pa_action__q2": ("yes", "the booking"),
    }
    client = ScriptedJudgeClient(answers, no_ask=())
    provider = _ScriptedProvider(
        [
            _turn("Anything from Marcus?"),
            _turn("Tonight is clear then."),
            _turn("So where do we eat instead?"),
            *[_turn("Go ahead with the booking now.")] * 8,
        ]
    )
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script(),
        _RecordingPA(),
        backend=NanobotRunnerSimulatorBackend(provider),
        judge=JudgeSession(mode="live", client=client),
    )

    events = asyncio.run(loop.run_session())

    assert _failures(events) == []
    assert _diagnostics(events) == [["reaction_authorises_pa_action"]]
    assert "Go ahead with the booking now." in [
        event.get("message")
        for event in loop._events
        if event["kind"] == "simulator_turn"
    ]


def test_mode_off_leaves_a_repeated_go_ahead_unjudged() -> None:
    """Off, nothing decides the rule, so the same turn plays."""
    events, client = _run("Go ahead with the booking now.", answers=None, mode="off")

    assert client.calls == []
    assert _failures(events) == []


# -------------------------------------------------- the reaction beat's two

#: One turn for both: it gets the reply's contractions backwards and
#: legislates the sessions to come.
THE_REACTION_TWO = (
    "Your reply used no contractions at all. In future, give me one option "
    "and nothing else."
)


def test_mode_off_leaves_the_reaction_two_unjudged() -> None:
    events, client = _run(THE_REACTION_TWO, None, mode="off")

    assert client.calls == []
    assert _failures(events) == []
    assert THE_REACTION_TWO in [
        event["message"] for event in events if event["kind"] == "simulator_turn"
    ]


@pytest.mark.skip(reason="future_policy commented out, owner ruling 2026-09-13")
def test_a_future_policy_is_a_hard_finding() -> None:
    """A diagnostic while a frame of patterns decided it, hard now (裁定 vi)."""
    events, _ = _run(
        THE_REACTION_TWO,
        {
            "reaction_states_future_policy": (
                "violated",
                "In future, give me one option and nothing else",
            )
        },
    )

    assert _failures(events) == [["reaction_states_future_policy"]]
    assert "reaction_states_future_policy" in PLAYED_AFTER_BUDGET


def test_the_contraction_question_waits_for_a_tone_formality_beat() -> None:
    """The arming stays the harness's: only that attribute reads the register
    off the reply, and what goes into the prompt is the contraction list."""
    _, off_attribute = _run(THE_REACTION_TWO, {})
    assert "reaction_misstates_contractions__q1" not in [
        question for call in off_attribute.calls for question in call
    ]

    events, on_attribute = _run(
        THE_REACTION_TWO,
        {
            "reaction_misstates_contractions__q1": (
                "yes",
                "Your reply used no contractions at all",
            )
        },
        declares="tone_formality",
    )

    assert "reaction_misstates_contractions__q1" in [
        question for call in on_attribute.calls for question in call
    ]
    # `reaction_misstates_contractions.md` says `status: unmeasured`: asked,
    # recorded as advisory, and deciding nothing until a replay promotes it.
    assert _failures(events) == []
    assert [e["rule"] for e in events if e["kind"] == "judge_advisory"] == [
        "reaction_misstates_contractions"
    ]
    assert "reaction_misstates_contractions" in PLAYED_AFTER_BUDGET
