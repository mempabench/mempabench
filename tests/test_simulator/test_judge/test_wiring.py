"""The judge inside the ladder: mode, the void door, and the two other sites.

These tests cover the wiring, so nothing here calls Ollama. The judge is a scripted
client that answers whatever it is asked, which is enough to pin the two
things the wiring decides: `off` leaves the judge's rules unchecked, and `live`
hands each of them to the judge.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_judge/test_wiring.py
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest

from pa.bus.events import OutboundMessage

from test_simulator.judge_fake import ScriptedJudgeClient

from simulator.agent.loop import _judge_probe_fact_missing
from simulator.backends import NanobotRunnerSimulatorBackend
from simulator.judge.catalogue import TurnContext
from simulator.judge.client import JudgeCall, JudgeUnavailable
from simulator.judge.guards import JudgeTimeout
from simulator.judge.session import JudgeSession, current_mode, preflight
from simulator.loop import SimulatorLoop
from simulator.output import (
    SessionInvalidError,
    SimulatorOutput,
    TurnAssessment,
)
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)


def _asked(client: ScriptedJudgeClient) -> list[str]:
    """Every question id the client was asked, in order."""
    return [question for call in client.calls for question in call]


# ---------------------------------------------------------------- the session


def _turn(message: str, next_beat: str = "advance", goal_already_met: str = "false") -> str:
    return (
        f"<message>\n{message}\n</message>\n\n"
        "<factual_check>\n</factual_check>\n\n"
        f"<turn_assessment>\nnext_beat: {next_beat}\n"
        f"goal_already_met: {goal_already_met}\n</turn_assessment>"
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
    async def process_direct_message(
        self, user_message: str, session_key: str, metadata: dict | None = None
    ) -> OutboundMessage:
        return OutboundMessage(channel="cli", chat_id="direct", content="PA reply")


class _RecordingWriter:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.attempts: list[dict] = []

    def log_event(self, event: dict) -> None:
        self.events.append(event)

    def log_validation_attempt(self, attempt: dict) -> None:
        self.attempts.append(attempt)


def _script() -> SessionScript:
    return SessionScript(
        session_id="judge_wiring",
        scenario="the assistant runs ahead",
        context="personal",
        task="Get tonight's dinner sorted.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(beat="b1_open", goal="Have the assistant read Marcus's message."),
            Beat(beat="b2_check", goal="Have the assistant check tonight."),
            Beat(
                beat="b3_probe",
                goal="Ask for a replacement venue.",
                active_skills=["solution_breadth"],
            ),
        ],
    )


def _script_with_a_fact() -> SessionScript:
    """The same session, with a fact the opening beat has to say.

    The forced-ask check used to be the loop's only judge call and it is
    commented out, so a session that has to reach the
    judge through the loop needs a per-turn rule instead: `probe_fact_missing`
    is asked on any beat with a required fact value.
    """
    session = _script()
    session.task_facts = {"venue": "the Cortez on Hill Street"}
    session.beats[0].uses = ["venue"]
    return session


def _run(
    responses: list[str], judge: JudgeSession, writer: _RecordingWriter | None = None
) -> tuple[list[dict], _ScriptedProvider]:
    provider = _ScriptedProvider(responses)
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script(),
        _RecordingPA(),
        backend=NanobotRunnerSimulatorBackend(provider),
        transcript_writer=writer,
        judge=judge,
    )
    return asyncio.run(loop.run_session()), provider


def _of_kind(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["kind"] == kind]


#: A declaring beat's first turn that puts nothing to the assistant. The regex
#: reads no ask in it; the judge, in these tests, reads one.
ACKNOWLEDGEMENT = "Right, you have already settled it."
ACK_QUOTE = "you have already settled it"
AN_ASK = "So where do we eat instead?"

BASE_TURNS = [
    _turn("Anything from Marcus?"),
    _turn("Tonight is clear then."),
]


# ------------------------------------------------------------------ the mode


def test_mode_off_is_the_default_and_never_calls_the_judge(monkeypatch) -> None:
    monkeypatch.delenv("MEMPA_JUDGE", raising=False)
    assert current_mode() == "off"
    client = ScriptedJudgeClient()
    judge = JudgeSession(mode="off", client=client)
    events, _ = _run([*BASE_TURNS, _turn(AN_ASK)], judge)

    assert client.calls == []
    # Off, the ask gate is not checked at all, so no retry can fire on it.
    assert _of_kind(events, "beat_ask_retry") == []


def test_an_unknown_mode_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("MEMPA_JUDGE", "maybe")
    with pytest.raises(ValueError, match="MEMPA_JUDGE"):
        current_mode()


# -------------------------------------------------------------- the ask gate


def test_the_judge_answers_the_gate_and_an_ask_it_reads_is_played() -> None:
    """A turn no string check would call an ask, played because the judge does."""
    client = ScriptedJudgeClient({"carries_an_ask": ("ask", ACK_QUOTE)})
    judge = JudgeSession(mode="live", client=client)
    events, provider = _run([*BASE_TURNS, _turn(ACKNOWLEDGEMENT)], judge)

    assert _of_kind(events, "beat_ask_retry") == []
    assert [event["message"] for event in _of_kind(events, "simulator_turn")][-1] == (
        ACKNOWLEDGEMENT
    )
    assert len(provider.calls) == 3


@pytest.mark.skip(reason="forced-ask check off")
def test_the_judge_can_read_no_ask_and_the_row_carries_its_evidence() -> None:
    client = ScriptedJudgeClient(
        {
            "carries_an_ask": [
                # `not_applicable` is not a quoting answer, so whatever the
                # judge wrote in `evidence` goes on the row as it stands.
                ("not_applicable", "he only says where to eat"),
                ("ask", "where do we eat instead"),
            ]
        }
    )
    judge = JudgeSession(mode="live", client=client)
    writer = _RecordingWriter()
    events, _ = _run([*BASE_TURNS, _turn(AN_ASK), _turn(AN_ASK)], judge, writer)

    # Record-only: the gate reads the turn
    # once and the turn is played whatever it reads.
    assert _of_kind(events, "beat_ask_retry") == []
    assert [
        event["beat"] for event in _of_kind(events, "beat_ask_unfulfilled")
    ] == ["b3_probe"]
    # The words the judge read the gate out of, on the row the record keeps —
    # the event itself carries 200 characters and no reasoning.
    rejected = [
        row
        for row in writer.attempts
        if row.get("error_codes") == ["beat_ask_unfulfilled"]
    ]
    assert [row["judge_evidence"] for row in rejected] == ["he only says where to eat"]


def test_the_gate_is_never_asked_through_the_loop() -> None:
    """The forced-ask check is commented out, so the
    armed beat costs no judge call at all — the ~13 s the gate spent per
    readback beat is the whole point — and the turn is played as it came."""
    client = ScriptedJudgeClient({"carries_an_ask": ("ask", "where do we eat instead")})
    judge = JudgeSession(mode="live", client=client)
    events, provider = _run([*BASE_TURNS, _turn(AN_ASK)], judge)

    assert [q for q in _asked(client) if q == "carries_an_ask"] == []
    assert [event["message"] for event in _of_kind(events, "simulator_turn")][-1] == (
        AN_ASK
    )
    assert len(provider.calls) == 3


def test_the_judge_call_event_names_the_box_that_answered() -> None:
    """With the pool, one run's verdicts come from whichever machine each
    session drew; the record has to say which, or a disagreement between the
    two boxes cannot be read back out of a finished run."""
    client = ScriptedJudgeClient({"carries_an_ask": ("ask", "where do we eat instead")})
    client.last_call = JudgeCall(
        raw="", seconds=1.2, prompt_tokens=1, eval_tokens=1, endpoint="box_a"
    )
    judge, seen = _void_judge(client)
    judge.judge("carries_an_ask", _ask_gate_context())

    assert _of_kind(seen, "judge_call")[0]["endpoint"] == "box_a"


def test_the_judge_call_event_carries_the_evidence_and_the_reason() -> None:
    """`verdicts` says what was answered; `answers` says what it pointed at.

    acc_007 voided on four `violated` answers whose sentence nobody
    could name afterwards: the event kept the verdict alone. Both fields are on
    the record now, the quote untruncated.
    """
    judge, seen = _void_judge(
        ScriptedJudgeClient({"carries_an_ask": ("ask", "where do we eat instead")})
    )
    judge.judge("carries_an_ask", _ask_gate_context())

    call = _of_kind(seen, "judge_call")[0]
    assert call["verdicts"] == {"carries_an_ask": "ask"}
    assert call["answers"] == [
        {
            "question_id": "carries_an_ask",
            "verdict": "ask",
            # the scripted words resolve to the whole numbered sentence: since
            # evidence became a reference, the event carries the sentence text
            "evidence": "So where do we eat instead?",
            "reason": "scripted",
        }
    ]


# ------------------------------------------------------------- the void door


def _void_judge(client: ScriptedJudgeClient) -> tuple[JudgeSession, list[dict]]:
    seen: list[dict] = []
    judge = JudgeSession(
        mode="live",
        client=client,
        on_event=lambda kind, **payload: seen.append({"kind": kind, **payload}),
    )
    return judge, seen


def _ask_gate_context() -> TurnContext:
    return TurnContext(
        turn_uid="b3_probe:t0",
        pa_previous_turn="Nothing from Marcus yet.",
        candidate_turn=AN_ASK,
        must_ask=True,
    )


def test_an_unparseable_reply_voids_the_session() -> None:
    judge, seen = _void_judge(ScriptedJudgeClient(raw=["not json", "still not", "no"]))
    with pytest.raises(SessionInvalidError) as caught:
        judge.judge("carries_an_ask", _ask_gate_context())

    assert caught.value.rule == "judge_unparseable"
    assert [event["kind"] for event in seen] == ["judge_unparseable"]
    assert seen[0]["attempts"] == 3
    assert seen[0]["question_id"] == "carries_an_ask"


def test_evidence_that_points_nowhere_releases_the_turn() -> None:
    """Twice unable to give a reference that resolves, the answer is not
    substantiated: the event is recorded and the turn goes through.
    The session is NOT void — that is for a judge that
    cannot be read at all."""
    judge, seen = _void_judge(
        ScriptedJudgeClient({"carries_an_ask": ("ask", "words this turn never said")})
    )
    # the gate's non-fired value is the permissive one: the turn is played
    assert judge.judge("carries_an_ask", _ask_gate_context()) is True

    assert [event["kind"] for event in seen] == ["judge_evidence_invalid"]
    assert seen[0]["released"] is True
    assert seen[0]["evidence"] == "words this turn never said"
    # The transcript event is self-diagnosing: which block the question pointed
    # at, that block as the judge was given it, and its last raw reply. The
    # candidate turn goes away with the attempt, so nothing else keeps them.
    assert seen[0]["block_id"] == "candidate_turn"
    assert seen[0]["block_text"] == f"<candidate_turn>\n[1] {AN_ASK}\n</candidate_turn>"
    assert json.loads(seen[0]["raw_response"])["verdicts"][0]["evidence"] == (
        "words this turn never said"
    )
    assert judge.last_verdicts == {}


def test_a_released_group_answers_the_rules_non_fired_value() -> None:
    """Every other rule's release is silence: no finding, no diagnostic, no
    rewrite. Only the ask gate's release is a `True`, because there the value
    that fires nothing is the one that lets the turn through."""
    ctx = TurnContext(
        turn_uid="b3_probe:t0",
        pa_previous_turn="Nothing from Marcus yet.",
        candidate_turn=AN_ASK,
        fact_values={"venue": "the place on Mill Street"},
    )
    for rule, released in (
        # Commented out: the rule is not asked; gold, prompt and builder
        # stay. It is out of the
        # catalogue, so `judge.judge` no longer has a spec for it.
        # ("reaction_states_future_policy", False),
        ("probe_fact_missing", []),
    ):
        judge, _ = _void_judge(
            ScriptedJudgeClient(
                {
                    "reaction_states_future_policy": ("violated", "never said this"),
                    "probe_fact_missing:venue": ("yes", "never said this"),
                }
            )
        )
        assert judge.judge(rule, ctx) == released


def test_a_timeout_voids_the_session() -> None:
    judge, seen = _void_judge(
        ScriptedJudgeClient(raises=JudgeTimeout(question_ids=["carries_an_ask"], seconds=15.4))
    )
    with pytest.raises(SessionInvalidError) as caught:
        judge.judge("carries_an_ask", _ask_gate_context())

    assert caught.value.rule == "judge_timeout"
    assert [event["kind"] for event in seen] == ["judge_timeout"]
    assert seen[0]["seconds"] == pytest.approx(15.4)


def test_an_exhausted_budget_voids_the_session(monkeypatch) -> None:
    monkeypatch.setenv("MEMPA_JUDGE_TURN_BUDGET", "0")
    judge, seen = _void_judge(
        ScriptedJudgeClient({"carries_an_ask": ("ask", "where do we eat instead")})
    )
    judge.begin_generation("b3_probe:a1", beat="b3_probe")
    with pytest.raises(SessionInvalidError) as caught:
        judge.judge("carries_an_ask", _ask_gate_context())

    assert caught.value.rule == "judge_budget_exhausted"
    assert caught.value.beat == "b3_probe"
    assert [event["kind"] for event in seen] == ["judge_budget_exhausted"]
    assert seen[0]["turn_uid"] == "b3_probe:a1"


def test_a_judge_failure_ends_the_whole_session_through_the_loop() -> None:
    client = ScriptedJudgeClient(raw=["not json", "still not", "no"])
    judge = JudgeSession(mode="live", client=client)
    provider = _ScriptedProvider([*BASE_TURNS, _turn(AN_ASK)])
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script_with_a_fact(),
        _RecordingPA(),
        backend=NanobotRunnerSimulatorBackend(provider),
        judge=judge,
    )

    with pytest.raises(SessionInvalidError) as caught:
        asyncio.run(loop.run_session())

    assert caught.value.rule == "judge_unparseable"
    invalid = _of_kind(loop._events, "session_invalid")
    assert [event["rule"] for event in invalid] == ["judge_unparseable"]
    assert len(_of_kind(loop._events, "judge_unparseable")) == 1


def test_a_dead_judge_voids_the_session_and_stops_the_round() -> None:
    """Ollama gone after the preflight: recorded like any void session, and the
    round stops anyway.

    A plain `SessionInvalidError` is what the harness reads as "this one
    measured nothing, run the next one" — and the next one would run with no
    judge at all, recorded against another standard without anyone noticing.
    So the record is written and the original error is what leaves the loop.
    """
    client = ScriptedJudgeClient(
        raises=JudgeUnavailable("ollama is not reachable at http://127.0.0.1:11434")
    )
    judge = JudgeSession(mode="live", client=client)
    provider = _ScriptedProvider([*BASE_TURNS, _turn(AN_ASK)])
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script_with_a_fact(),
        _RecordingPA(),
        backend=NanobotRunnerSimulatorBackend(provider),
        judge=judge,
    )

    with pytest.raises(JudgeUnavailable, match="not reachable"):
        asyncio.run(loop.run_session())

    unavailable = _of_kind(loop._events, "judge_unavailable")
    assert [(event["model"], "not reachable" in event["reason"]) for event in unavailable] == [
        ("scripted-judge", True)
    ]
    assert [event["rule"] for event in _of_kind(loop._events, "session_invalid")] == [
        "judge_unavailable"
    ]


def test_a_dead_judge_does_not_leave_as_an_ordinary_invalid_session() -> None:
    """The distinction the round depends on: `SessionInvalidError` is caught by
    a harness that then runs the next session, and this must not be."""
    client = ScriptedJudgeClient(raises=JudgeUnavailable("model was unloaded"))
    judge = JudgeSession(mode="live", client=client)
    provider = _ScriptedProvider([*BASE_TURNS, _turn(AN_ASK)])
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _script_with_a_fact(),
        _RecordingPA(),
        backend=NanobotRunnerSimulatorBackend(provider),
        judge=judge,
    )

    with pytest.raises(JudgeUnavailable):
        asyncio.run(loop.run_session())
    assert not isinstance(JudgeUnavailable("x"), SessionInvalidError)


def test_the_budget_starts_over_on_each_candidate_generation(monkeypatch) -> None:
    monkeypatch.setenv("MEMPA_JUDGE_TURN_BUDGET", "1")
    client = ScriptedJudgeClient({"carries_an_ask": ("ask", "where do we eat instead")})
    judge = JudgeSession(mode="live", client=client)

    judge.begin_generation("b3_probe:a1")
    judge.judge("carries_an_ask", _ask_gate_context())
    judge.begin_generation("b3_probe:a2")
    judge.judge("carries_an_ask", _ask_gate_context())

    assert len(client.calls) == 2
    # Without the reset the second generation's first call is already over.
    with pytest.raises(SessionInvalidError):
        judge.judge("carries_an_ask", _ask_gate_context())


# -------------------------------------------------------- probe_fact_missing


def _output(message: str) -> SimulatorOutput:
    return SimulatorOutput(
        message=message,
        turn_assessment=TurnAssessment(next_beat="stay"),
        raw_response="scripted",
    )


def _fact_beat() -> tuple[SessionScript, Beat]:
    session = _script()
    return session, session.beats[2]


def test_a_fact_the_judge_reads_in_the_turn_leaves_no_diagnostic() -> None:
    """The judge finds the fact a string match would have missed."""
    message = "I need the Nakamura preprint located in the catalogue tonight."
    output = _output(message)
    client = ScriptedJudgeClient(
        {"probe_fact_missing:paper_query": ("yes", "the Nakamura preprint located in the catalogue")}
    )
    judge = JudgeSession(mode="live", client=client)
    session, beat = _fact_beat()

    _judge_probe_fact_missing(
        judge,
        output,
        session=session,
        beat=beat,
        required_values={"paper_query": "Non-Markovian corrections"},
        on_request_keys=[],
        user_history=[],
        last_pa_message="What should I look for?",
    )

    assert output.diagnostics == []
    assert "probe_fact_missing" not in output.diagnostics_detail


def test_the_judges_list_is_the_one_the_row_carries() -> None:
    """A fact the judge does not read in the turn, with its key on the row."""
    output = _output("I need something from the catalogue tonight.")
    client = ScriptedJudgeClient()  # every fact answers `no`
    judge = JudgeSession(mode="live", client=client)
    session, beat = _fact_beat()

    _judge_probe_fact_missing(
        judge,
        output,
        session=session,
        beat=beat,
        required_values={"paper_query": "Non-Markovian corrections"},
        on_request_keys=[],
        user_history=[],
        last_pa_message="What should I look for?",
    )

    assert output.diagnostics == ["probe_fact_missing"]
    assert output.diagnostics_detail["probe_fact_missing"] == ["paper_query"]


def test_a_fact_held_on_request_is_not_owed_until_the_assistant_asks() -> None:
    """The arming is the harness's: the judge only says whether the words are
    there, and this key is not owed on a turn the assistant asked nothing on."""
    output = _output("Tonight is clear then.")
    client = ScriptedJudgeClient()  # every fact answers `no`
    judge = JudgeSession(mode="live", client=client)
    session, beat = _fact_beat()

    _judge_probe_fact_missing(
        judge,
        output,
        session=session,
        beat=beat,
        required_values={"held_key": "the walking limit"},
        on_request_keys=["held_key"],
        user_history=[],
        last_pa_message="Here is the plan.",
    )

    assert output.diagnostics == []


# ------------------------------------------------------------------ preflight


def test_preflight_is_a_no_op_when_the_judge_is_off() -> None:
    preflight("off", client=ScriptedJudgeClient())


def test_preflight_refuses_when_the_pinned_model_is_not_installed(monkeypatch) -> None:
    monkeypatch.setattr(
        "simulator.judge.session._installed_models", lambda base_url: {"something:else"}
    )
    with pytest.raises(JudgeUnavailable, match="is not served at"):
        preflight("live", client=ScriptedJudgeClient())


def test_preflight_refuses_when_the_probe_call_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "simulator.judge.session._installed_models", lambda base_url: {"scripted-judge"}
    )
    client = ScriptedJudgeClient(raises=JudgeTimeout(question_ids=["carries_an_ask"], seconds=15.0))
    client.probe = lambda: client.generate(
        system="", prompt="", allowed={"carries_an_ask": ("ask", "not_applicable")}
    )
    with pytest.raises(JudgeTimeout):
        preflight("live", client=client)


def test_preflight_passes_on_a_probe_that_parses(monkeypatch) -> None:
    monkeypatch.setattr(
        "simulator.judge.session._installed_models", lambda base_url: {"scripted-judge"}
    )
    client = ScriptedJudgeClient()
    client.probe = lambda: JudgeCall(raw="", seconds=0.2, prompt_tokens=1, eval_tokens=1)
    preflight("live", client=client)


# -------------------------------------------------------------- the gate rule
#
# One rule, with no semantic regex, so the judge is
# the only thing that can produce the finding: `off` produces it on neither arm
# of the union, and on both it is the judge's answer composed against the
# harness's own arming.
#
# The judge half is commented out at the call site
# (`simulator/agent/loop.py`), so neither arm is asked any more. What is left is
# the code half — `missing_key_terms()` before the reveal beat — and the three
# tests below are what the simulator side does now. The catalogue builder, the
# composer and the replay path are untouched and keep their own tests.

GATE_TERMS = ["chest tightening", "the stairs", "three weeks"]
GATE_TEXT = (
    "the chest tightening when I take the stairs in the mornings, which began "
    "about three weeks ago"
)
# The pre-reveal invention: live `acc_039` t12, on the beat before
# the reveal, supplying a reason the actor does not hold yet.
INVENTED_REASON = "The reason for the visit is a routine preventive checkup."
# The dictation itself, delivered on the reveal beat with every gate term in it.
DICTATED_TURN = (
    "The reason is the chest tightening when I take the stairs in the "
    "mornings, which began about three weeks ago."
)
# Every gate term before the reveal beat, in the actor's own words rather than
# as the dictated sentence, so `pre_reveal_arm` stays armed: the code half's own
# case.
PRE_REVEAL_KEY_TERMS = "Chest tightening on the stairs, three weeks now."


def _withheld_questions(client: ScriptedJudgeClient) -> list[str]:
    """The gate rule's question ids the client was asked — none, since the
    judge half is commented out."""
    return [
        question
        for question in _asked(client)
        if question.startswith("withheld_fact_before_ask")
    ]


def _gated_script() -> SessionScript:
    """One gate, and a reveal beat whose own prose holds the sentence back.

    The instruction never names the key: `_reveal_at_fact_keys` drops a key the
    beat's own words carry, and `_on_request_fact_keys` is what arms
    `withheld_fact_before_ask`.
    """
    return SessionScript(
        session_id="judge_gate",
        scenario="the checkup",
        context="personal",
        task="Book the checkup.",
        life_context=LifeContext(),
        task_facts={"checkup_reason_the_user_dictates": GATE_TEXT},
        reveal_at={"checkup_reason_the_user_dictates": "b2_probe"},
        beats=[
            Beat(beat="b1_open", goal="Report the scheduling bottleneck."),
            Beat(
                beat="b2_probe",
                goal="Say what the visit is for.",
                actor_instruction=(
                    "Says in his own words what the visit is for, and hands it "
                    "over only when the PA asks for it."
                ),
            ),
        ],
        gate_key_terms=GATE_TERMS,
        gate_dictated_text=GATE_TEXT,
        gate_reveal_beat="b2_probe",
        gate_cue="the reason for the visit",
    )


def _gate_turn(
    message: str,
    beat_id: str,
    judge: JudgeSession,
    history: list[dict],
) -> Any:
    """One candidate on one beat, played once the retry budget is spent.

    Both rules are in `PLAYED_AFTER_BUDGET`, so a turn the judge condemns
    leaves on its diagnostics rather than raising — which is where the
    assertion can read it.
    """
    script = _gated_script()
    backend = NanobotRunnerSimulatorBackend(
        _ScriptedProvider([_turn(message)]), output_contract_retries=0
    )
    return asyncio.run(
        backend.generate_turn(
            PersonaIdentity(name="User A", speech_style={}),
            PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
            script,
            next(beat for beat in script.beats if beat.beat == beat_id),
            history=history,
            judge=judge,
        )
    )


def test_invented_pre_reveal_wording_is_no_longer_judged() -> None:
    """The first arm of the union, with the judge half off: a beat before the
    reveal, on a turn that does not deliver the real sentence, so the wording it
    hands over is invented.

    The code has no delivery to count here — the gate terms are missing — and
    `q_handed` was the only thing that could say the turn handed something over,
    so nothing is asked and nothing is recorded. 29/29 of that question's live
    verdicts were false, which is why it is off.
    """
    client = ScriptedJudgeClient(
        {
            "withheld_fact_before_ask__q_handed": (
                "yes",
                "The reason for the visit is a routine preventive checkup",
            )
        }
    )
    output = _gate_turn(
        INVENTED_REASON,
        "b1_open",
        JudgeSession(mode="live", client=client),
        [{"role": "user", "content": "What is the visit for?"}],
    )

    assert "withheld_fact_before_ask" not in output.diagnostics
    assert _withheld_questions(client) == []


def test_the_on_request_window_is_no_longer_armed() -> None:
    """The second arm went with the judge half: the reveal beat holds the
    sentence on request and every gate term is in the turn, and the two
    exemptions that used to decide it were questions of the judge's. Nothing is
    asked and nothing is recorded — the code half runs before the reveal beat
    alone."""
    # The fact rule, a hard one, would otherwise end this turn on a
    # rule the test is not about — and this turn does say the reason, so `yes`
    # is the honest answer.
    client = ScriptedJudgeClient(
        {
            "probe_fact_missing:checkup_reason_the_user_dictates": (
                "yes",
                "candidate_turn#1",
            )
        }
    )
    output = _gate_turn(
        DICTATED_TURN,
        "b2_probe",
        JudgeSession(mode="live", client=client),
        [{"role": "user", "content": "How long should the session be?"}],
    )

    assert "withheld_fact_before_ask" not in output.diagnostics
    assert _withheld_questions(client) == []


def test_the_code_half_records_a_pre_reveal_key_term_delivery() -> None:
    """What is left of the rule: before the reveal beat the gated sentence is
    not in the prompt, so a turn carrying every key term of it has handed the
    thing over. `missing_key_terms()` says so for free, the name goes on the
    turn's diagnostics with `key_terms_delivered`, and no question is asked."""
    client = ScriptedJudgeClient()
    output = _gate_turn(
        PRE_REVEAL_KEY_TERMS,
        "b1_open",
        JudgeSession(mode="live", client=client),
        [{"role": "user", "content": "What is the visit for?"}],
    )

    assert "withheld_fact_before_ask" in output.diagnostics
    assert output.diagnostics_detail["withheld_fact_before_ask"] == [
        "key_terms_delivered"
    ]
    assert _withheld_questions(client) == []


def test_off_judges_neither_arm_of_the_gate_rule_and_calls_nothing() -> None:
    client = ScriptedJudgeClient()
    judge = JudgeSession(mode="off", client=client)

    substitute = _gate_turn(
        INVENTED_REASON,
        "b1_open",
        judge,
        [{"role": "user", "content": "What is the visit for?"}],
    )
    withheld = _gate_turn(
        DICTATED_TURN,
        "b2_probe",
        judge,
        [{"role": "user", "content": "How long should the session be?"}],
    )

    assert client.calls == []
    assert "withheld_fact_before_ask" not in substitute.diagnostics
    assert "withheld_fact_before_ask" not in withheld.diagnostics


# ------------------------------------------------------------ the general form


def _general_script() -> SessionScript:
    """A beat whose own direction says a must and two must-nots.

    Neither beat declares an attribute, so the ask gate never arms and every
    judge call in these tests is the general one.
    """
    return SessionScript(
        session_id="judge_general",
        scenario="the assistant runs ahead",
        context="personal",
        task="Get tonight's dinner sorted.",
        life_context=LifeContext(),
        task_facts={},
        beats=[
            Beat(
                beat="b1_open",
                goal="Have the assistant read Marcus's message.",
                actor_instruction="Say that Marcus has written and ask what it says.",
                avoid="Do not name a restaurant. Do not give a go-ahead.",
            ),
            Beat(beat="b2_check", goal="Have the assistant check tonight."),
        ],
    )


def _run_general(
    responses: list[str], judge: JudgeSession, writer: _RecordingWriter | None = None
) -> tuple[list[dict], _ScriptedProvider]:
    provider = _ScriptedProvider(responses)
    loop = SimulatorLoop(
        provider,
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        _general_script(),
        _RecordingPA(),
        backend=NanobotRunnerSimulatorBackend(provider),
        transcript_writer=writer,
        judge=judge,
    )
    return asyncio.run(loop.run_session()), provider


#: A turn that breaks the beat's second `avoid` clause. The evidence is a
#: verbatim span of it, which is what the guards check.
A_GO_AHEAD = "Marcus has written; go ahead and take whatever he suggests."
GO_AHEAD_QUOTE = "go ahead and take whatever"
CLEAN_TURN = "Marcus has written. What does his message say?"


def test_the_general_form_is_not_called_with_the_judge_off(monkeypatch) -> None:
    """Mode off is still the default, and the beat's own direction is judged by
    nobody: the turn that breaks `avoid` is played, exactly as without the judge."""
    monkeypatch.delenv("MEMPA_JUDGE", raising=False)
    client = ScriptedJudgeClient()
    events, _ = _run_general(
        [_turn(A_GO_AHEAD), _turn("Nothing else tonight.")],
        JudgeSession(mode="off", client=client),
    )

    assert client.calls == []
    assert _of_kind(events, "simulator_validation_failure") == []
    assert [event["message"] for event in _of_kind(events, "simulator_turn")] == [
        A_GO_AHEAD,
        "Nothing else tonight.",
    ]


def test_every_avoid_clause_is_asked_on_its_own() -> None:
    client = ScriptedJudgeClient()
    _run_general(
        [_turn(CLEAN_TURN), _turn("Nothing else tonight.")],
        JudgeSession(mode="live", client=client),
    )

    # The two per-turn rules are asked on every turn and come first, so the
    # general form is not the session's opening call. It goes out chunked: the
    # instruction on the head call, then the `avoid` clauses.
    assert ["user_instruction"] in client.calls
    assert ["avoid_1", "avoid_2"] in client.calls


def test_an_avoid_clause_the_turn_breaks_is_recorded_and_never_a_finding() -> None:
    """Non-blocking (32/33 false positives in live runs): judged and recorded,
    never a finding, until `_AVOID_LEAD` is fixed and replayed.

    The clause is still asked on its own and its `judge_call` row still carries
    the `violated`; what changed is the consequence — no rewrite, no retry note.
    The turn is played on the first attempt with the rule on the turn's
    diagnostics and the clause and the judge's evidence on the detail.
    """
    client = ScriptedJudgeClient({"avoid_2": [("violated", GO_AHEAD_QUOTE)]})
    events, provider = _run_general(
        [_turn(A_GO_AHEAD), _turn(CLEAN_TURN)],
        JudgeSession(mode="live", client=client),
    )

    assert [
        call["verdicts"]["avoid_2"]
        for call in _of_kind(events, "judge_call")
        if "avoid_2" in call["verdicts"]
    ] == ["violated"]
    assert _of_kind(events, "simulator_validation_failure") == []
    assert [
        message["content"]
        for call in provider.calls
        for message in call
        if "violated the private output contract" in str(message.get("content"))
    ] == []
    recorded = [
        (event["beat"], event["diagnostics"], event["diagnostics_detail"])
        for event in _of_kind(events, "simulator_turn_diagnostics")
        if "beat_avoid_violated" in event["diagnostics"]
    ]
    assert [beat for beat, _, _ in recorded] == ["b1_open"]
    # The clause the judge answered on, which is the detail the finding already
    # carried for its retry note. The words the judge quoted are not repeated
    # here: they are on the `judge_call` row's `answers`, where every other
    # rule's evidence lives, and that row is still written.
    assert recorded[0][2]["beat_avoid_violated"] == ["Do not give a go-ahead."]
    assert [event["message"] for event in _of_kind(events, "simulator_turn")] == [
        A_GO_AHEAD,
        CLEAN_TURN,
    ]


def test_a_repeated_avoid_violation_never_spends_an_attempt() -> None:
    """What `PLAYED_AFTER_BUDGET` used to cover: the turn is played anyway.

    Non-blocking, so it is played without spending the budget
    at all — one provider call per beat, no `session_invalid`, and the rule
    recorded on the beat that broke it.
    """
    client = ScriptedJudgeClient(
        {"avoid_2": [("violated", GO_AHEAD_QUOTE)] * 5}
    )
    events, provider = _run_general(
        [_turn(A_GO_AHEAD), _turn("Nothing else tonight.")],
        JudgeSession(mode="live", client=client),
    )

    assert [event["message"] for event in _of_kind(events, "simulator_turn")] == [
        A_GO_AHEAD,
        "Nothing else tonight.",
    ]
    assert len(provider.calls) == 2
    assert [event["kind"] for event in events].count("session_invalid") == 0
    recorded = [
        event
        for event in _of_kind(events, "simulator_turn_diagnostics")
        if "beat_avoid_violated" in event["diagnostics"]
    ]
    assert [event["beat"] for event in recorded] == ["b1_open"]


def test_the_instruction_is_asked_only_on_the_turn_it_binds() -> None:
    """An instruction covers a beat and a beat runs over several turns. Eight of
    the 28 false alarms a measurement counted were the judge reading a
    beat-wide instruction as owed by every turn of the beat."""
    client = ScriptedJudgeClient()
    _run_general(
        [
            _turn(CLEAN_TURN, next_beat="stay"),
            _turn("And what does he want to do about it?"),
            _turn("Nothing else tonight."),
        ],
        JudgeSession(mode="live", client=client),
    )

    assert _asked(client).count("user_instruction") == 1
    # The `avoid` clauses bind every turn of the beat, and are asked on both.
    assert _asked(client).count("avoid_1") == 2


# ------------------------------------------------------- the unmeasured guard


def test_an_unmeasured_rule_is_asked_and_decides_nothing(monkeypatch):
    """`status: unmeasured` in the prompt header: the call happens, the answer
    is recorded as advisory, and the rule returns its non-firing value."""
    from simulator.judge import session as judge_session

    events: list[tuple[str, dict]] = []
    judge = judge_session.JudgeSession(
        mode="live",
        client=object(),
        on_event=lambda kind, **payload: events.append((kind, payload)),
    )
    monkeypatch.setattr(
        judge_session,
        "guarded_ask",
        lambda *a, **k: [
            judge_session.Verdict(
                "reaction_names_setting", "violated", "four whole words here", ""
            )
        ],
    )
    ctx = judge_session.TurnContext(candidate_turn="Be more Proactive from now on.")
    assert judge.judge("reaction_names_setting", ctx) is False
    kinds = [kind for kind, _ in events]
    assert "judge_call" in kinds and "judge_advisory" in kinds
    advisory = next(payload for kind, payload in events if kind == "judge_advisory")
    assert advisory == {"rule": "reaction_names_setting", "composed": True}


@pytest.mark.skip(reason="future_policy commented out")
def test_a_measured_rule_still_decides(monkeypatch):
    from simulator.judge import session as judge_session

    judge = judge_session.JudgeSession(mode="live", client=object())
    monkeypatch.setattr(
        judge_session,
        "guarded_ask",
        lambda *a, **k: [
            judge_session.Verdict(
                "reaction_states_future_policy", "violated", "four whole words here", ""
            )
        ],
    )
    ctx = judge_session.TurnContext(candidate_turn="From now on, keep it short.")
    assert judge.judge("reaction_states_future_policy", ctx) is True


# ------------------------------------------------- the judge is off the loop

#: How long a gated call waits for the loop to release it. Reached only when
#: the wrap is gone, where it is the test's red instead of a hang.
_GATE_S = 0.5


class _GatedJudge:
    """A judge whose call at one site blocks until the event loop lets it go.

    Not a latency probe: the assertion is on the order the two halves record
    themselves in, so a slow box makes this test slower, never redder.
    """

    enabled = True

    def __init__(self, site: str) -> None:
        self._site = site
        self.entered = threading.Event()
        self.released = threading.Event()
        self.order: list[str] = []

    def _gate(self, site: str) -> None:
        if site != self._site or self.entered.is_set():
            return
        self.entered.set()
        self.released.wait(timeout=_GATE_S)
        self.order.append("the judge call returned")

    def begin_generation(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def judge(self, rule: str, _ctx: Any) -> Any:
        self._gate("reception" if rule.startswith("reaction_") else rule)
        return [] if rule == "probe_fact_missing" else False

    def ask(self, _group: Any) -> dict:
        self._gate("general")
        return {}

    def last_evidence(self, _question_id: str) -> str:
        return ""


@pytest.mark.parametrize("site", ["reception", "probe_fact_missing", "general"])
def test_a_blocking_judge_call_leaves_the_event_loop_running(site: str) -> None:
    """The judge client is synchronous `urllib` under a process-wide lock, so
    on the loop one call left every other coroutine's in-flight HTTP unread for
    its whole length -- and with more than one session at a time that is
    another session's request budget.

    Every armed site goes out through `asyncio.to_thread`, so a sibling
    coroutine gets to run while the call is out -- and it is the sibling that
    releases it, which is why the order is the assertion and the clock is not.
    The reaction site takes the same wrap and has no armed rule
    (`reaction_states_future_policy` is commented out), so nothing here drives
    it.
    """
    judge = _GatedJudge(site)
    script = _gated_script().model_copy(
        update={"compressed_reaction_beat": "b2_probe"}
    )
    backend = NanobotRunnerSimulatorBackend(
        _ScriptedProvider([_turn("Can you book the checkup?")]),
        output_contract_retries=0,
    )

    async def _sibling() -> None:
        while not judge.entered.is_set():
            await asyncio.sleep(0.005)
        judge.order.append("the loop kept running")
        judge.released.set()

    async def _both() -> None:
        await asyncio.gather(
            backend.generate_turn(
                PersonaIdentity(name="User A", speech_style={}),
                PreferenceMatrix.from_yaml(
                    "data/personas/user_a_preferences.yaml"
                ),
                script,
                next(beat for beat in script.beats if beat.beat == "b2_probe"),
                history=[
                    {"role": "assistant", "content": "What is the visit for?"}
                ],
                judge=judge,
            ),
            _sibling(),
        )

    asyncio.run(_both())

    assert judge.order == ["the loop kept running", "the judge call returned"]
