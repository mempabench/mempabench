"""T13–T16 and the T18 cases the loop-level pair does not reach.

T18's four sideband verdicts, the missing release and the blind
assertion all live in `tests/test_pa/test_ix_wrapper_loop.py`; what is added
here is the two-and-three-held-id shape, the four-round cancellation, and the
three contracts that need either every attribute (T13), every setting (T14), a
real `SimulatorLoop` (T15) or two episodes in one session (T16).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from pa.agent.ix.collector import TURN_REMINDER_PREFIX, turn_reminder_message
from pa.agent.ix.events import IXSelection
from pa.agent.ix.render_types import (
    HOLD_ROUNDS_HARD_CAP,
    IX_CONFIRMATION_MESSAGE_FIELD,
    IX_EVENT_NAMES,
    turn_budget,
)
from pa.agent.loop import (
    IX_SELECTION_ARGS_CONTRACT_FAILURE_DETAIL,
    IX_SELECTION_CONTRACT_FAILURE_DETAIL,
    AgentProviderError,
)
from pa.agent.ix.render import IX_EMPTY_REPLY_LINE
from pa.agent.runner import (
    IX_CAPABILITY_RESOLVE_REMINDER,
    IX_PER_TURN_REMINDER_PREFIX,
    TOOL_RETRY_CAP,
)
from pa.providers.base import LLMProvider, LLMResponse

from harness.ix_seed import seed_ix_selection

from simulator.loop import SimulatorLoop
from simulator.output import SimulatorOutput, TurnAssessment
from simulator.schemas import (
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

from harness.pa_factory import RecordingPA, held_items_from_ix_render

from .conftest import (
    SCRIPT_SESSION,
    SESSION_KEY,
    CapabilityLimitTool,
    RefusingTool,
    ScriptProvider,
    build_loop,
    executed_tools,
    ix,
    plan_call,
    run,
    run_turn,
    selection,
    task_call,
)

ACTION_TYPES = {"task_read": "read", "task_send": "external_action"}


def _blob(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# T13 · the per-turn restatement, one case per pass-through setting
# ---------------------------------------------------------------------------

#: The per-turn restatement covers the six attributes the renderer does not transform, plus the one
#: `autonomy_level` setting whose whole effect is instruction text.
RESTATED = [
    ("tone_formality", "Formal"),
    ("verbosity", "Detailed"),
    ("emotional_engagement", "Relationship-focused"),
    ("guidance_level", "Guided"),
    ("solution_breadth", "High"),
    ("task_expansion", "Low"),
    ("autonomy_level", "Autonomous"),
]


@pytest.mark.parametrize("attribute,setting", RESTATED)
def test_a_chosen_setting_is_restated_on_a_later_toolless_turn(
    loop_factory, attribute: str, setting: str
) -> None:
    """T13: the setting must not lapse on a turn that calls no tool.

    The restatement is an internal message: it reaches the model and nothing
    else — not the transcript, not memory, not the user.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[selection(attribute, setting)]),
            LLMResponse(content="Turn one answer."),
            LLMResponse(content="Turn two answer."),
        ])
        loop, adapter = loop_factory(provider)

        await run_turn(loop, "Explain the plan.", active_ix=[attribute])
        second_start = len(provider.requests)
        outbound = await run_turn(loop, "And the budget?", active_ix=[attribute])

        reminders = [
            message for message in provider.requests[second_start]
            if str(message.get("content") or "").startswith(TURN_REMINDER_PREFIX)
        ]
        assert len(reminders) == 1
        assert f"{attribute}: {setting}" in reminders[0]["content"]

        assert TURN_REMINDER_PREFIX not in outbound.content
        assert TURN_REMINDER_PREFIX not in _blob(loop.sessions.get(SESSION_KEY).messages)
        assert TURN_REMINDER_PREFIX not in _blob(
            [event.messages for event in adapter.events]
        )

    run(scenario())


def test_the_restatement_of_an_in_turn_selection_is_the_next_turn_text(
    loop_factory,
) -> None:
    """No second wording for the same setting.

    The in-turn channel says what the next turn would have said, byte for
    byte: one builder, `collector.turn_reminder_message`, whichever turn it is
    read on.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[selection("solution_breadth", "Low")]),
            LLMResponse(content="Il Fornaio, 7pm."),
        ])
        loop, _ = loop_factory(provider)

        await run_turn(loop, "Where should we eat?", active_ix=["solution_breadth"])

        composing = provider.requests[1]
        reminders = [
            message for message in composing
            if str(message.get("content") or "").startswith(TURN_REMINDER_PREFIX)
        ]
        assert len(reminders) == 1
        assert reminders[0]["content"] == turn_reminder_message([IXSelection(
            attribute="solution_breadth",
            tool_name="IX_solution_breadth",
            setting="Low",
            evidence="the user said so last time",
            application="apply it to this turn",
        )])

    run(scenario())


def test_a_setting_chosen_at_call_zero_shapes_the_same_turn(loop_factory) -> None:
    """`acc_002` t5.

    `IX_solution_breadth(setting="Low")` at call 0, then a five-row table in
    the same reply: the restatement was built from `inherited_ix_selections`,
    which the loop writes only after the run, so the Low surface arrived a turn
    late. The composing request must carry the setting on the turn it was
    chosen — and the restatement stays internal, exactly as on a later turn.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[selection("solution_breadth", "Low")]),
            LLMResponse(content="Il Fornaio, 7pm."),
        ])
        loop, adapter = loop_factory(provider)

        outbound = await run_turn(
            loop, "Where should we eat?", active_ix=["solution_breadth"],
        )

        # The selection is call 0 of the first request, so the first request
        # cannot carry it; the second one composes the reply and must.
        assert TURN_REMINDER_PREFIX not in _blob(provider.requests[0])
        reminders = [
            message for message in provider.requests[1]
            if str(message.get("content") or "").startswith(TURN_REMINDER_PREFIX)
        ]
        assert len(reminders) == 1
        assert "solution_breadth: Low" in reminders[0]["content"]

        assert TURN_REMINDER_PREFIX not in outbound.content
        assert TURN_REMINDER_PREFIX not in _blob(loop.sessions.get(SESSION_KEY).messages)
        assert TURN_REMINDER_PREFIX not in _blob(
            [event.messages for event in adapter.events]
        )

    run(scenario())


def test_a_gating_autonomy_setting_is_not_restated_as_instruction_text(
    loop_factory,
) -> None:
    """`Reactive` / `Self-directed` are enforced by the gate, not by a reminder.

    Restating them would give the model a second, softer copy of a rule the
    wrapper already enforces, and the two could disagree.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[selection("autonomy_level", "Reactive")]),
            LLMResponse(content="Turn one answer."),
            LLMResponse(content="Turn two answer."),
        ])
        loop, _ = loop_factory(provider)

        await run_turn(loop, "Explain the plan.", active_ix=["autonomy_level"])
        second_start = len(provider.requests)
        await run_turn(loop, "And the budget?", active_ix=["autonomy_level"])

        assert not [
            message for message in provider.requests[second_start]
            if str(message.get("content") or "").startswith(TURN_REMINDER_PREFIX)
        ]

    run(scenario())


# ---------------------------------------------------------------------------
# T14 · requiredness must not leak the setting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("setting", ["Silent", "Bookend", "Full narration"])
def test_the_process_payload_is_forced_under_every_process_setting(
    loop_factory, setting: str
) -> None:
    """T14: `Silent` demands the completion line too.

    If the payload were only required where it is rendered, the model could
    read the requirement and infer the setting.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[selection("process_visibility", setting)]),
            LLMResponse(content="Monday 10:00 is booked."),
            LLMResponse(content=None, tool_calls=[
                ix("IX_render_process_visibility", completion_message="All set."),
                # A task call, because the closing line is only released over
                # one; the requiredness under test
                # is unchanged by it.
                task_call("c1", "task_create", "", notice="Putting the meeting in now."),
            ]),
            LLMResponse(content="Monday 10:00 is booked."),
        ])
        loop, _ = loop_factory(provider, "task_create")

        outbound = await run_turn(
            loop, "Book Monday morning.", active_ix=["process_visibility"],
            action_types={"task_create": "internal_write"},
        )

        nudges = {
            str(message.get("content"))
            for request in provider.requests
            for message in request
            if str(message.get("content") or "").startswith(IX_PER_TURN_REMINDER_PREFIX)
        }
        assert len(nudges) == 1
        assert "IX_render_process_visibility" in nudges.pop()
        # The line itself is only released by the two settings that narrate.
        released = setting in {"Bookend", "Full narration"}
        assert ("All set." in outbound.content) is released
        assert outbound.metadata["ix_render"]["completion"] == {
            "released": released, "text": "All set.",
        }

    run(scenario())


# ---------------------------------------------------------------------------
# T15 · three batches through a real SimulatorLoop
# ---------------------------------------------------------------------------

class _SimProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "scripted-simulator"

    async def chat(self, *args, **kwargs):
        raise AssertionError("the scripted backend never calls the simulator provider")


class _ScriptedBackend:
    """A simulated user that replies until the PA has delivered every item."""

    name = "scripted"

    def __init__(self, replies: list[str], *, advance_on_last: bool = True) -> None:
        self.replies = list(replies)
        self.advance_on_last = advance_on_last
        self.turns = 0
        self.held_seen: list[list[dict[str, str]]] = []

    def get_default_model(self) -> str:
        return "backend-model"

    async def generate_turn(self, identity, matrix, session, beat, **kwargs):
        self.held_seen.append(list(kwargs.get("held_items") or []))
        message = (
            self.replies[self.turns] if self.turns < len(self.replies)
            else "Anything else?"
        )
        self.turns += 1
        last = self.turns >= len(self.replies)
        return SimulatorOutput(
            message=message,
            turn_assessment=TurnAssessment(
                next_beat="advance" if (last and self.advance_on_last) else "stay"
            ),
            raw_response="scripted",
        )

    async def drain_memory_tasks(self) -> None:
        return None


def _sim_loop(pa_loop, beat_skills: list[str], backend: _ScriptedBackend) -> SimulatorLoop:
    script = SessionScript(
        session_id="acc_ix",
        scenario="three things to settle",
        context="personal",
        task="Get the three of them resolved.",
        life_context=LifeContext(),
        task_facts={},
        beats=[Beat(beat="open", active_skills=beat_skills)],
    )
    return SimulatorLoop(
        _SimProvider(),
        PersonaIdentity(name="User A", speech_style={}),
        PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml"),
        script,
        pa_loop,
        session_key=SESSION_KEY,
        backend=backend,
        turn_budget_fn=turn_budget,
    )


def _batch_payload(*batch_ids: str) -> Any:
    return ix(
        "IX_render_topic_management",
        batch_set_id="bs1",
        item_count=len(batch_ids),
        batches=[
            {"batch_id": bid, "label": bid, "body": "", "call_binding": []}
            for bid in batch_ids
        ],
    )


def _batch_prose(*batch_ids: str) -> str:
    body = "\n\n".join(f"[[batch:{bid}]] Item {bid} is settled." for bid in batch_ids)
    return f"Here is where things stand.\n\n{body}"


def test_three_batches_one_at_a_time_finish_inside_the_beats_budget() -> None:
    """T15: three One-at-a-time rounds fit in the widened beat."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "One-at-a-time"), _batch_payload("b1", "b2", "b3"),
            ]),
            LLMResponse(content=_batch_prose("b1", "b2", "b3")),
            LLMResponse(content=None, tool_calls=[
                _batch_payload("b2", "b3"),
                ix("IX_continue_topic_management", decisions=[
                    {"id": "b2", "decision": "release", "user_evidence": "go on"},
                    {"id": "b3", "decision": "keep", "user_evidence": ""},
                ]),
            ]),
            LLMResponse(content=_batch_prose("b2", "b3")),
            LLMResponse(content=None, tool_calls=[
                _batch_payload("b3"),
                ix("IX_continue_topic_management", decisions=[
                    {"id": "b3", "decision": "release", "user_evidence": "and the last"},
                ]),
            ]),
            LLMResponse(content=_batch_prose("b3")),
        ])
        with tempfile.TemporaryDirectory() as workspace:
            pa_loop, _ = build_loop(Path(workspace), provider)
            backend = _ScriptedBackend(["Where are we?", "And next?", "And the last one?"])
            sim = _sim_loop(RecordingPA(pa_loop), ["topic_management"], backend)

            events = await sim.run_session()

        pa_turns = [event for event in events if event["kind"] == "pa_turn"]
        assert len(pa_turns) == 3
        # One batch per round, in order, and never two at once.
        assert [
            [bid for bid in ("b1", "b2", "b3") if f"Item {bid}" in turn["outbound"]]
            for turn in pa_turns
        ] == [["b1"], ["b2"], ["b3"]]
        # The beat had 3 + 3 + 1 turns to spend and did not need them all.
        assert turn_budget(["topic_management"]) == 7
        assert not [event for event in events if event["kind"] == "beat_turn_limit"]
        assert pa_loop.close_ix_session(SESSION_KEY, "acc_ix") == []

    run(scenario())


def test_a_beat_that_never_converges_reports_the_budget_it_ran_on() -> None:
    """T15's other half: `beat_turn_limit.turns` is this beat's own budget."""
    async def scenario() -> None:
        payload = ix(
            "IX_render_information_elicitation",
            elicitation_id="el1",
            questions=[{"id": "q1", "text": "Which airport?", "about": "departure"}],
            assumptions=[],
        )
        # The payload carrier is owed on every turn the attribute is active,
        # not only while an episode is open, so each PA turn is a pair: the
        # payload, then the prose that carries the marker.
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Iterative"), payload,
            ]),
            *[
                response
                for _ in range(20)
                for response in (
                    LLMResponse(content="[[q:q1]] Which airport do you want?"),
                    LLMResponse(content=None, tool_calls=[payload]),
                )
            ],
        ])
        with tempfile.TemporaryDirectory() as workspace:
            pa_loop, _ = build_loop(Path(workspace), provider)
            backend = _ScriptedBackend(
                ["I need a trip booked."] * 12, advance_on_last=False,
            )
            sim = _sim_loop(RecordingPA(pa_loop), ["information_elicitation"], backend)

            events = await sim.run_session()

        limits = [event for event in events if event["kind"] == "beat_turn_limit"]
        assert [event["turns"] for event in limits] == [
            turn_budget(["information_elicitation"])
        ]
        assert limits[0]["turns"] == 7
        assert sum(1 for event in events if event["kind"] == "simulator_turn") == 7

    run(scenario())


# ---------------------------------------------------------------------------
# T16 · a second episode in one session
# ---------------------------------------------------------------------------

def _questions(*ids: str) -> Any:
    return [
        {"id": qid, "text": f"Question {qid}?", "about": qid} for qid in ids
    ]


def test_a_closed_elicitation_episode_lets_a_later_task_open_a_new_one(
    loop_factory,
) -> None:
    """T16: Structured closes its episode, so a new task may ask again."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Structured"),
                ix(
                    "IX_render_information_elicitation",
                    elicitation_id="el1",
                    questions=_questions("q1", "q2"),
                    assumptions=[],
                ),
            ]),
            LLMResponse(content="Two things first.\n\n[[q:q1]] Which airport?\n\n[[q:q2]] Aisle or window?"),
            LLMResponse(content=None, tool_calls=[
                ix(
                    "IX_render_information_elicitation",
                    elicitation_id="el2",
                    questions=_questions("q3"),
                    assumptions=[],
                ),
            ]),
            LLMResponse(content="One more thing.\n\n[[q:q3]] Which card should I use?"),
        ])
        loop, _ = loop_factory(provider)

        first = await run_turn(
            loop, "Book me a flight.", active_ix=["information_elicitation"],
        )
        assert "Which airport?" in first.content and "Aisle or window?" in first.content
        assert first.metadata["ix_render"]["episodes"] == {}

        second = await run_turn(
            loop, "SFO, aisle. Also sort the hotel.",
            active_ix=["information_elicitation"],
        )
        assert "Which card should I use?" in second.content
        # A brand new elicitation id, asked and released in one turn again.
        assert second.metadata["ix_render"]["episodes"] == {}
        payload_ids = [
            event["args"]["elicitation_id"]
            for event in second.metadata["tool_events"]
            if event.get("name") == "IX_render_information_elicitation"
        ]
        assert payload_ids == ["el2"]

    run(scenario())


def test_a_new_episode_id_while_the_old_one_is_open_is_refused(loop_factory) -> None:
    """T16's fail-closed half: an open hold has to be answered, not renamed.

    `IX_continue_<attribute>` is required for as long as the episode holds
    anything. Declaring a fresh `elicitation_id` instead does not clear it: the
    turn is nudged, and a second miss is recorded rather than accepted.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Iterative"),
                ix(
                    "IX_render_information_elicitation",
                    elicitation_id="el1",
                    questions=_questions("q1", "q2", "q3"),
                    assumptions=[],
                ),
            ]),
            LLMResponse(
                content="I need three things.\n\n[[q:q1]] Which airport?\n\n"
                        "[[q:q2]] Aisle or window?\n\n[[q:q3]] Which card?",
            ),
            # Turn 2 tries to start over under a new id, twice.
            *[
                LLMResponse(content=None, tool_calls=[
                    ix(
                        "IX_render_information_elicitation",
                        elicitation_id="el2",
                        questions=_questions("q9"),
                        assumptions=[],
                    ),
                ])
                for _ in range(2)
            ],
            LLMResponse(content="Never mind, [[q:q9]] which card?"),
        ])
        loop, _ = loop_factory(provider)

        await run_turn(loop, "Book me a flight.", active_ix=["information_elicitation"])
        second_start = len(provider.requests)
        second = await run_turn(
            loop, "SFO.", active_ix=["information_elicitation"],
        )

        nudges = {
            str(message.get("content"))
            for request in provider.requests[second_start:]
            for message in request
            if str(message.get("content") or "").startswith(IX_PER_TURN_REMINDER_PREFIX)
        }
        assert any("IX_continue_information_elicitation" in nudge for nudge in nudges)
        # T16: nudge once, record the second miss **once**, then deliver.
        # `IX_continue_<attribute>` is neither forcible nor fatal, so the check
        # re-fired on every remaining iteration -- live (acc_009 t8) one
        # owed continuation was recorded eight times.
        assert [
            event.get("name") for event in second.metadata["tool_events"]
            if event.get("name") == "ix_render_payload_missing"
        ] == ["ix_render_payload_missing"]

    run(scenario())


# ---------------------------------------------------------------------------
# T18 (c) / (f) · several held ids, and the round the hold is given up on
# ---------------------------------------------------------------------------

def test_one_allow_among_three_verdicts_releases_only_that_item(loop_factory) -> None:
    """T18 (c): the sideband is per id; two silent ids are not carried along."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Lining them up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                    {"id": "a3", "bound_item_ids": ["i3"]},
                ),
                task_call("send_1", "task_send", "i1", ask="Mail Dale?"),
                task_call("send_2", "task_send", "i2", ask="Mail Theo?"),
                task_call("send_3", "task_send", "i3", ask="Mail Nadia?"),
            ]),
            LLMResponse(content="Sending the first one.", tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "yes to Dale"},
                    {"id": "i2", "decision": "release", "user_evidence": "yes to Dale"},
                    {"id": "i3", "decision": "release", "user_evidence": "yes to Dale"},
                ]),
                task_call("send_1b", "task_send", "i1", ask="Mail Dale?"),
                task_call("send_2b", "task_send", "i2", ask="Mail Theo?"),
                task_call("send_3b", "task_send", "i3", ask="Mail Nadia?"),
            ]),
            LLMResponse(content="Sent the first."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Mail all three of them.", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Only Dale, please.", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
            consent=[
                {"interaction_item_id": "i1", "decision": "allow"},
                {"interaction_item_id": "i2", "decision": "deny"},
                {"interaction_item_id": "i3", "decision": "unclear"},
            ],
        )

        assert executed_tools(second) == ["task_send"]
        consent = {row["item_id"]: row for row in second.metadata["ix_render"]["consent"]}
        assert consent["i1"]["sideband_decision"] == "allow"
        assert consent["i1"]["released"] is True
        for item_id, verdict in (("i2", "deny"), ("i3", "unclear")):
            assert consent[item_id]["pa_decision"] == "release"
            assert consent[item_id]["sideband_decision"] == verdict
            assert consent[item_id]["released"] is False

    run(scenario())


def test_a_release_reused_for_another_tool_is_asked_again(loop_factory) -> None:
    """Consent is for the action, not for the id.

    `i1` was held as an email and the user allowed the email. Reissuing `i1` as
    a payment is a different action: it inherits nothing, and the gate asks
    again. Not an error — the PA may have changed how it does the step — so the
    turn ends on a fresh ask, and the audit row says the release did not apply.
    """
    action_types = {"task_send": "external_action", "task_pay": "external_action"}
    new_ask = "Shall I pay the invoice instead?"

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Lining it up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Mail Dale?"),
            ]),
            LLMResponse(content="Paying it instead.", tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "go ahead"},
                ]),
                task_call("pay_1", "task_pay", "i1", ask=new_ask),
            ]),
            # Never reached: the second turn ends on the fresh ask. It is here
            # so that a regression -- the payment inheriting the email's
            # consent -- fails on the side-effect assertion below rather than
            # on an exhausted script.
            LLMResponse(content="The invoice is paid."),
        ])
        loop, _ = loop_factory(provider, "task_send", "task_pay")

        await run_turn(
            loop, "Mail Dale about the invoice.", active_ix=["autonomy_level"],
            action_types=action_types,
        )
        second = await run_turn(
            loop, "Go ahead.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        # The allow bought an email, so the payment did not happen.
        assert loop.tools.get("task_pay").calls == []
        assert executed_tools(second) == []
        assert second.content.rstrip().endswith(new_ask)

        held = second.metadata["ix_render"]["held"]["calls"]
        assert [(item["tool"], item["reason"]) for item in held] == [
            ("task_pay", "external_action_floor")
        ]
        consent = {row["item_id"]: row for row in second.metadata["ix_render"]["consent"]}
        assert consent["i1"]["sideband_decision"] == "allow"
        assert consent["i1"]["released"] is False
        assert consent["i1"]["tool_changed"] is True
        assert [
            event.get("name") for event in second.metadata["tool_events"]
        ].count("ix_consent_tool_changed") == 1

    run(scenario())


def test_a_new_act_on_the_same_tool_is_asked_again_once_the_first_one_succeeded(
    loop_factory,
) -> None:
    """The release ends where the act succeeds.

    A re-minted item inherits an unspent release on the same tool, which is
    what stopped acc_039 asking three times for one booking. The boundary of
    that rule is the success: the send here went through, so the id
    minted afterwards is a *second* act, and the floor asks about it on its own
    account. That is the same shape as
    `test_a_stale_allow_does_not_release_anything_after_the_episode_closed`
    from the wrapper's side — a new plan item, a new id, the same tool — and it
    is why binding the floor to the tool name alone was
    wrong: it silenced the genuine second act along with the re-mint.
    """
    action_types = {"task_send": "external_action"}
    second_ask = "Shall I submit the booking?"

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Lining it up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Book it?"),
            ]),
            LLMResponse(content=None, tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "go ahead"},
                ]),
                task_call("send_2", "task_send", "i1"),
            ]),
            LLMResponse(content="Booked."),
            # The same act, re-minted under a fresh plan and a fresh id.
            LLMResponse(content="Submitting the booking.", tool_calls=[
                plan_call({"id": "b1", "bound_item_ids": ["i2"]}, plan_id="p2"),
                task_call("send_3", "task_send", "i2", ask=second_ask),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Book the checkup.", active_ix=["autonomy_level"],
            action_types=action_types,
        )
        second = await run_turn(
            loop, "Proceed with the booking.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )
        assert executed_tools(second) == ["task_send"]

        third = await run_turn(
            loop, "Is it done?", active_ix=["autonomy_level"],
            action_types=action_types,
        )

        assert executed_tools(third) == []
        assert third.metadata["ix_render"]["gate"]["reason"] == "external_action_floor"
        assert third.content.rstrip().endswith(second_ask)
        assert _inherit_events(third) == []

    run(scenario())


def _inherit_events(outbound: Any) -> list[dict[str, Any]]:
    return [
        event for event in outbound.metadata["tool_events"]
        if event.get("name") == "ix_release_inherited"
    ]


def _floor_ask_script(second_ask: str, plan_id: str = "p2") -> list[LLMResponse]:
    """Ask for one act, release it, then re-mint the same act under a new id."""
    return [
        LLMResponse(content="Lining it up.", tool_calls=[
            selection("autonomy_level", "Reactive"),
            plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
            task_call("send_1", "task_send", "i1", ask="Book it?"),
        ]),
        LLMResponse(content=None, tool_calls=[
            ix("IX_continue_autonomy_level", decisions=[
                {"id": "i1", "decision": "release", "user_evidence": "go ahead"},
            ]),
            task_call("send_2", "task_send", "i1"),
        ]),
        LLMResponse(content="The clinic would not take that reason."),
        LLMResponse(content="Submitting it again.", tool_calls=[
            plan_call({"id": "b1", "bound_item_ids": ["i2"]}, plan_id=plan_id),
            task_call("send_3", "task_send", "i2", ask=second_ask),
        ]),
        LLMResponse(content="Sent."),
    ]


def test_a_re_mint_after_the_tool_refused_inherits_the_release(loop_factory) -> None:
    """A refused call spends nothing.

    acc_039 t31: the user said "Proceed", the release went through,
    and the store rejected `appointment_create`'s reason. The act had not
    happened, so t34's `book_final` was the same booking under a third id — and
    it walked back into the floor gate and asked the user a third time.
    """
    action_types = {"task_send": "external_action"}
    second_ask = "Shall I submit the booking?"

    async def scenario() -> None:
        provider = ScriptProvider(_floor_ask_script(second_ask))
        loop, _ = loop_factory(provider, "task_send")
        loop.tools.register(RefusingTool("task_send"))

        await run_turn(
            loop, "Book the checkup.", active_ix=["autonomy_level"],
            action_types=action_types,
        )
        second = await run_turn(
            loop, "Proceed with the booking.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )
        # The call ran and the world refused it: the booking has not happened.
        assert executed_tools(second) == ["task_send"]
        assert [
            event.get("status") for event in second.metadata["tool_events"]
            if event.get("name") == "task_send"
        ] == ["error"]

        third = await run_turn(
            loop, "Is it done?", active_ix=["autonomy_level"],
            action_types=action_types,
        )

        assert executed_tools(third) == ["task_send"]
        assert third.metadata["ix_render"]["gate"] is None
        assert second_ask not in third.content
        assert [event["release_inherited_from"] for event in _inherit_events(third)] == [
            "i1"
        ]

    run(scenario())


def test_a_retry_after_the_tool_errored_runs_in_the_same_turn(loop_factory) -> None:
    """An errored call spends nothing inside one turn either — acc_039 t19.

    The released `appointment_create` came back `unknown provider`; the booking
    had not happened, and the retry on the next iteration of the same turn was
    refused `release_spent` -- as were the 24 after it. Only an `ok`
    world-changing call spends the release, whichever iteration the retry lands
    on.
    """
    action_types = {"task_send": "external_action"}

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Lining it up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Book it?"),
            ]),
            LLMResponse(content=None, tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "go ahead"},
                ]),
                task_call("send_2", "task_send", "i1"),
            ]),
            # The store refused the first attempt; the same act, same turn.
            LLMResponse(content=None, tool_calls=[
                task_call("send_3", "task_send", "i1"),
            ]),
            LLMResponse(content="Booked."),
        ])
        loop, _ = loop_factory(provider, "task_send")
        loop.tools.register(RefusingTool("task_send"))

        await run_turn(
            loop, "Book the checkup.", active_ix=["autonomy_level"],
            action_types=action_types,
        )
        second = await run_turn(
            loop, "Proceed with the booking.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        assert executed_tools(second) == ["task_send", "task_send"]
        assert _named_events(second, "ix_release_spent") == []

    run(scenario())


def test_the_same_call_held_over_and_over_ends_the_turn(loop_factory) -> None:
    """The maximum-steps cap — the wrapper's half of it.

    acc_039 t19 re-issued one booking 25 times after the release was
    spent. Every one of them was held before the tool ran, so the failure cap
    never saw an error and the turn died on the 40-iteration limit with the
    runner's own "maximum number of tool call iterations" string as the reply.
    An identical call held again for a reason that asks the user nothing counts
    like a failing tool.
    """
    action_types = {"task_send": "external_action"}

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Lining it up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Book it?"),
            ]),
            LLMResponse(content=None, tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "go ahead"},
                ]),
                task_call("send_2", "task_send", "i1"),
            ]),
            # The act went through; every one of these is the refused duplicate.
            *[
                LLMResponse(content="Trying again.", tool_calls=[
                    task_call(f"send_{index}", "task_send", "i1"),
                ])
                for index in range(3, 9)
            ],
            LLMResponse(content="Never reached."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Book the checkup.", active_ix=["autonomy_level"],
            action_types=action_types,
        )
        second = await run_turn(
            loop, "Proceed with the booking.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        capped = _named_events(second, "tool_retry_cap_reached")
        assert [(event["tool"], event["count"]) for event in capped] == [
            ("task_send", TOOL_RETRY_CAP)
        ]
        # The turn ends on the capped exit, before the scripted responses run
        # out, and what ships is a reply -- the prose of a held call never is,
        # so it is the empty-reply line rather than the runner's own
        # "maximum number of tool call iterations" string.
        assert provider.responses
        assert second.content == IX_EMPTY_REPLY_LINE
        # The act itself ran exactly once.
        assert executed_tools(second) == ["task_send"]

    run(scenario())


def test_a_re_mint_after_a_content_objection_inherits_the_release(loop_factory) -> None:
    """The never-issued call spends nothing either.

    acc_039 t28: the user allowed the booking but objected to what the
    PA had put in the reason, so the PA released `book_appt_1`, issued no call
    for it, and re-minted `book_appt_2` with the reason rewritten. Same act,
    same tool, one turn later — and the second ask fired inside the very turn
    that answered the first.
    """
    action_types = {"task_send": "external_action"}
    second_ask = "Shall I submit the booking?"

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Lining it up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Book it?"),
            ]),
            # The release and the re-mint arrive in the same response.
            LLMResponse(content="Rewriting the reason.", tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "proceed"},
                ]),
                plan_call({"id": "b1", "bound_item_ids": ["i2"]}, plan_id="p2"),
                task_call("send_2", "task_send", "i2", ask=second_ask),
            ]),
            LLMResponse(content="Booked."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Book the checkup.", active_ix=["autonomy_level"],
            action_types=action_types,
        )
        second = await run_turn(
            loop,
            "Proceed, but say only that the chest tightening started three weeks ago.",
            active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        assert executed_tools(second) == ["task_send"]
        assert second.metadata["ix_render"]["gate"] is None
        assert second_ask not in second.content

    run(scenario())


def test_an_inherited_release_still_buys_only_one_call(loop_factory) -> None:
    """The one-release-one-execution rule survives the inheritance: one authorisation, one act.

    The release moves to the re-minted item; it does not multiply. A response
    that mints two items on the tool gets the first one through and is asked
    about the second.
    """
    action_types = {"task_send": "external_action"}
    second_ask = "And shall I book the follow-up as well?"

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Lining it up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Book it?"),
            ]),
            LLMResponse(content="Rewriting the reason.", tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "proceed"},
                ]),
                plan_call(
                    {"id": "b1", "bound_item_ids": ["i2"]},
                    {"id": "b2", "bound_item_ids": ["i3"]},
                    plan_id="p2",
                ),
                task_call("send_2", "task_send", "i2"),
                task_call("send_3", "task_send", "i3", ask=second_ask),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Book the checkup.", active_ix=["autonomy_level"],
            action_types=action_types,
        )
        second = await run_turn(
            loop, "Proceed with the booking.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        assert executed_tools(second) == ["task_send"]
        assert [event["item_id"] for event in _inherit_events(second)] == ["i2"]
        gate = second.metadata["ix_render"]["gate"]
        assert (gate["item_id"], gate["reason"]) == ("i3", "external_action_floor")

    run(scenario())


def test_the_inherited_release_names_the_item_it_came_from(loop_factory) -> None:
    """The inheritance is on the record.

    Nothing else in the transcript would show that the second act ran without
    an ask of its own, so the row carries the id whose authorisation it is
    spending. `IX_EVENT_NAMES` registers the name for the readers that key on
    the tuple.
    """
    action_types = {"task_send": "external_action"}

    async def scenario() -> None:
        provider = ScriptProvider(_floor_ask_script("Shall I submit the booking?"))
        loop, _ = loop_factory(provider, "task_send")
        loop.tools.register(RefusingTool("task_send"))

        await run_turn(
            loop, "Book the checkup.", active_ix=["autonomy_level"],
            action_types=action_types,
        )
        await run_turn(
            loop, "Proceed with the booking.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )
        third = await run_turn(
            loop, "Is it done?", active_ix=["autonomy_level"],
            action_types=action_types,
        )

        assert _inherit_events(third) == [{
            "ix_event": True,
            "name": "ix_release_inherited",
            "item_id": "i2",
            "tool": "task_send",
            "release_inherited_from": "i1",
        }]
        assert "ix_release_inherited" in IX_EVENT_NAMES

    run(scenario())


def test_a_denied_release_is_not_inherited_by_the_re_minted_item(loop_factory) -> None:
    """The user can take the authorisation back.

    The release persists across the re-mint, not across a change of mind: a
    `deny` on the sideband while the released item is still held ends it, and
    the next act on the tool goes back through the floor.
    """
    action_types = {"task_send": "external_action"}
    second_ask = "Shall I submit the booking?"

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Lining it up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Book it?"),
            ]),
            # Released, but the call never went out: `i1` stays held.
            LLMResponse(content=None, tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "go ahead"},
                ]),
            ]),
            LLMResponse(content="I will send it in a moment."),
            LLMResponse(content="Submitting the booking.", tool_calls=[
                # `i1` is still held, so the continuation protocol runs too.
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "keep", "user_evidence": ""},
                ]),
                plan_call({"id": "b1", "bound_item_ids": ["i2"]}, plan_id="p2"),
                task_call("send_2", "task_send", "i2", ask=second_ask),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Book the checkup.", active_ix=["autonomy_level"],
            action_types=action_types,
        )
        await run_turn(
            loop, "Proceed with the booking.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )
        third = await run_turn(
            loop, "Actually, do not book it.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "deny"}],
        )

        assert executed_tools(third) == []
        assert third.metadata["ix_render"]["gate"]["reason"] == "external_action_floor"
        assert third.content.rstrip().endswith(second_ask)
        assert _inherit_events(third) == []

    run(scenario())


# ---------------------------------------------------------------------------
# The floor pre-authorisation
# ---------------------------------------------------------------------------

#: The sideband shape the wrapper accepts for a standing authorisation: no id,
#: because nothing was held when the actor spoke, and the catalog tool name in
#: its place.
STANDING_ALLOW = {"interaction_item_id": "", "decision": "allow", "tool": "task_send"}

#: The same shape with a real catalog tool in it. The sideband's name is the one
#: `harness/gate.py` derives off the draft's atomic steps, and the PA reaches the
#: same tool through the state MCP server as `mcp_state_appointment_create`.
STANDING_ALLOW_CATALOG = {
    "interaction_item_id": "", "decision": "allow", "tool": "appointment_create",
}


def _named_events(outbound: Any, name: str) -> list[dict[str, Any]]:
    return [
        event for event in outbound.metadata["tool_events"]
        if event.get("name") == name
    ]


def _one_act_script(ask: str, tool: str = "task_send") -> list[LLMResponse]:
    """One `external_action`, carrying the ask the floor would deliver."""
    return [
        LLMResponse(content="Booking the slot.", tool_calls=[
            selection("autonomy_level", "Self-directed"),
            plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
            task_call("send_1", tool, "i1", ask=ask),
        ]),
        LLMResponse(content="The booking is in."),
    ]


def test_a_go_ahead_in_the_turn_just_gone_pre_authorises_the_act(loop_factory) -> None:
    """acc_039 t40 → t41.

    t40 the user says "You may proceed with booking that slot at 08:30"; t41 the
    PA calls `appointment_create` and the floor holds it and asks for the same
    permission a second time; t43 the user gives it again; t44 the act runs.
    Nothing was held when the user spoke, so the sideband had no id to answer —
    it names the tool instead, and the floor lets that one act through.
    """
    ask = "Shall I book the slot?"

    async def scenario() -> None:
        provider = ScriptProvider(_one_act_script(ask))
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "You may proceed with booking that slot at 08:30.",
            active_ix=["autonomy_level"],
            action_types={"task_send": "external_action"},
            consent=[STANDING_ALLOW],
        )

        # Still PA-blind. The entry is the one verdict the anchor in
        # `loop.py` does not drop, and it reaches gate state only.
        assert "consent_decisions" not in _blob(provider.requests)
        assert executed_tools(outbound) == ["task_send"]
        assert outbound.metadata["ix_render"]["gate"] is None
        assert ask not in outbound.content
        assert _named_events(outbound, "ix_release_preauthorised") == [{
            "ix_event": True,
            "name": "ix_release_preauthorised",
            "item_id": "i1",
            "tool": "task_send",
        }]
        assert "ix_release_preauthorised" in IX_EVENT_NAMES
        # Spent, so there is nothing left to lapse at the end of the turn.
        assert _named_events(outbound, "ix_preauthorisation_expired") == []
        # The judge counts authorisations off `consent[]`, and this one
        # never passed through a hold. The tool name stands in for the item id.
        assert outbound.metadata["ix_render"]["consent"] == [{
            "attribute": "autonomy_level",
            "item_id": "task_send",
            "pa_decision": "keep",
            "sideband_decision": "allow",
            "released": True,
            "user_evidence": "",
            "evidence_verified": False,
            "tool_changed": False,
        }]

    run(scenario())


def test_the_go_ahead_reaches_the_act_under_its_runtime_name(loop_factory) -> None:
    """`acc_039` t30 → t31 — the two spellings of one act.

    The sideband names the catalog tool (`appointment_create`), which is what
    the draft's atomic steps name and what `harness/gate.py` derives from them;
    the PA reaches that tool through the state MCP server, which prefixes the
    name. Live, the two were compared raw, so the standing allow never matched:
    the floor held `mcp_state_appointment_create` and asked for a permission the
    user had given on the beat before, and the grant lapsed unspent on the same
    turn. No `tool_action_types` override here — the runtime name is classified
    by the default map, the way it is in a real run.
    """
    ask = "Shall I book the slot?"

    async def scenario() -> None:
        provider = ScriptProvider(
            _one_act_script(ask, "mcp_state_appointment_create")
        )
        loop, _ = loop_factory(provider, "mcp_state_appointment_create")

        outbound = await run_turn(
            loop, "You may proceed with booking that slot at 08:30.",
            active_ix=["autonomy_level"],
            consent=[STANDING_ALLOW_CATALOG],
        )

        assert executed_tools(outbound) == ["mcp_state_appointment_create"]
        assert outbound.metadata["ix_render"]["gate"] is None
        assert ask not in outbound.content
        # The grant is named in catalog space wherever it is reported, so a
        # release and a lapse of the same grant read as the same tool.
        assert _named_events(outbound, "ix_release_preauthorised") == [{
            "ix_event": True,
            "name": "ix_release_preauthorised",
            "item_id": "i1",
            "tool": "appointment_create",
        }]
        assert _named_events(outbound, "ix_preauthorisation_expired") == []
        assert outbound.metadata["ix_render"]["consent"] == [{
            "attribute": "autonomy_level",
            "item_id": "appointment_create",
            "pa_decision": "keep",
            "sideband_decision": "allow",
            "released": True,
            "user_evidence": "",
            "evidence_verified": False,
            "tool_changed": False,
        }]

    run(scenario())


def test_the_runtime_name_is_normalised_without_blurring_the_act(loop_factory) -> None:
    """Dropping the `mcp_state_` prefix must not merge two neighbouring acts.

    `appointment_update` is not the act the user spoke about, so the floor asks
    about it in the ordinary way and the unused grant lapses at the end of the
    turn — the same outcome as a go-ahead for a wholly unrelated tool.
    """
    ask = "Shall I move the appointment?"

    async def scenario() -> None:
        provider = ScriptProvider(
            _one_act_script(ask, "mcp_state_appointment_update")
        )
        loop, _ = loop_factory(provider, "mcp_state_appointment_update")

        outbound = await run_turn(
            loop, "You may proceed with booking that slot at 08:30.",
            active_ix=["autonomy_level"],
            consent=[STANDING_ALLOW_CATALOG],
        )

        assert executed_tools(outbound) == []
        assert _named_events(outbound, "ix_release_preauthorised") == []
        gate = outbound.metadata["ix_render"]["gate"]
        assert (gate["item_id"], gate["reason"]) == ("i1", "external_action_floor")
        assert outbound.content.rstrip().endswith(ask)
        assert [
            event["tool"]
            for event in _named_events(outbound, "ix_preauthorisation_expired")
        ] == ["appointment_create"]

    run(scenario())


def test_an_unspent_pre_authorisation_does_not_survive_the_turn(loop_factory) -> None:
    """Scope is the one PA turn that follows the go-ahead, and no further.

    The user authorised the act on the beat they spoke it. A tool the PA did
    not reach this turn is asked about in the ordinary way next turn — the
    permission is not banked against some later act the user never heard of.
    """
    ask = "Shall I book the slot?"

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Checking the diary first.", tool_calls=[
                selection("autonomy_level", "Self-directed"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("read_1", "task_read", "i1"),
            ]),
            LLMResponse(content="The slot is still free."),
            LLMResponse(content="Booking it.", tool_calls=[
                plan_call({"id": "b1", "bound_item_ids": ["i2"]}, plan_id="p2"),
                task_call("send_1", "task_send", "i2", ask=ask),
            ]),
            LLMResponse(content="Waiting on you."),
        ])
        loop, _ = loop_factory(provider, "task_read", "task_send")

        first = await run_turn(
            loop, "You may proceed with booking that slot at 08:30.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
            consent=[STANDING_ALLOW],
        )
        assert executed_tools(first) == ["task_read"]
        assert _named_events(first, "ix_preauthorisation_expired") == [{
            "ix_event": True,
            "name": "ix_preauthorisation_expired",
            "tool": "task_send",
        }]
        assert "ix_preauthorisation_expired" in IX_EVENT_NAMES

        second = await run_turn(
            loop, "And then?", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
        )

        assert executed_tools(second) == []
        assert _named_events(second, "ix_release_preauthorised") == []
        gate = second.metadata["ix_render"]["gate"]
        assert (gate["item_id"], gate["reason"]) == ("i2", "external_action_floor")
        assert second.content.rstrip().endswith(ask)

    run(scenario())


def test_one_standing_entry_buys_one_call(loop_factory) -> None:
    """One go-ahead is one act, not a licence on the tool for the turn.

    The user authorised *the* booking. A second outward call on the same tool in
    the same response is a second act, and it goes through the floor — the same
    rule `ix_release_inherited` follows for a re-minted item.
    """
    ask = "Shall I book the second one too?"

    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Booking them.", tool_calls=[
                selection("autonomy_level", "Self-directed"),
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                ),
                task_call("send_1", "task_send", "i1"),
                task_call("send_2", "task_send", "i2", ask=ask),
            ]),
            LLMResponse(content="The first one is in."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "You may proceed with booking that slot at 08:30.",
            active_ix=["autonomy_level"],
            action_types={"task_send": "external_action"},
            consent=[STANDING_ALLOW],
        )

        assert executed_tools(outbound) == ["task_send"]
        assert [
            event["item_id"]
            for event in _named_events(outbound, "ix_release_preauthorised")
        ] == ["i1"]
        gate = outbound.metadata["ix_render"]["gate"]
        assert (gate["item_id"], gate["reason"]) == ("i2", "external_action_floor")
        assert outbound.content.rstrip().endswith(ask)

    run(scenario())


def test_a_standing_allow_for_another_tool_does_not_release_this_one(
    loop_factory,
) -> None:
    """The authorisation is keyed on the act the user named, not on the turn.

    A go-ahead for one outward act says nothing about a different one, so the
    floor asks about `task_send` as usual and the unused entry lapses.
    """
    ask = "Shall I book the slot?"

    async def scenario() -> None:
        loop, _ = loop_factory(ScriptProvider(_one_act_script(ask)), "task_send")

        outbound = await run_turn(
            loop, "You may go ahead and file the expense claim.",
            active_ix=["autonomy_level"],
            action_types={"task_send": "external_action"},
            consent=[{
                "interaction_item_id": "", "decision": "allow", "tool": "task_other",
            }],
        )

        assert executed_tools(outbound) == []
        assert _named_events(outbound, "ix_release_preauthorised") == []
        gate = outbound.metadata["ix_render"]["gate"]
        assert (gate["item_id"], gate["reason"]) == ("i1", "external_action_floor")
        assert outbound.content.rstrip().endswith(ask)
        assert [
            event["tool"]
            for event in _named_events(outbound, "ix_preauthorisation_expired")
        ] == ["task_other"]

    run(scenario())


def test_a_hold_that_never_converges_is_cancelled_with_the_pa_s_own_line(
    loop_factory,
) -> None:
    """T18 (f): round four cancels; it never releases by default."""
    async def scenario() -> None:
        questions = ix(
            "IX_render_information_elicitation",
            elicitation_id="el1",
            questions=_questions("q1", "q2", "q3"),
            assumptions=[],
        )
        prose = (
            "I still need three things.\n\n[[q:q1]] Which airport?\n\n"
            "[[q:q2]] Aisle or window?\n\n[[q:q3]] Which card?"
        )
        responses = [
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Infer"), questions,
            ]),
            LLMResponse(content=prose),
        ]
        for round_index in range(2, HOLD_ROUNDS_HARD_CAP + 1):
            keep = [
                {"id": qid, "decision": "keep", "user_evidence": ""}
                for qid in ("q1", "q2", "q3")
            ]
            continue_args: dict[str, Any] = {"decisions": keep}
            if round_index == HOLD_ROUNDS_HARD_CAP:
                continue_args["closing_message"] = (
                    "I will stop asking and go with the usual choices."
                )
            responses.append(LLMResponse(content=None, tool_calls=[
                questions, ix("IX_continue_information_elicitation", **continue_args),
            ]))
            responses.append(LLMResponse(content=prose))
        loop, _ = loop_factory(ScriptProvider(responses))

        outbound = None
        for round_index in range(HOLD_ROUNDS_HARD_CAP):
            outbound = await run_turn(
                loop, f"Turn {round_index}.", active_ix=["information_elicitation"],
            )
            # `Infer` never asks: no question body is delivered on any round.
            assert "Which airport?" not in outbound.content

        assert "I will stop asking and go with the usual choices." in outbound.content
        assert outbound.metadata["ix_render"]["episodes"] == {}
        assert [
            entry["id"]
            for entry in outbound.metadata["ix_render"]["held"]["cancelled"]
        ] == ["q1", "q2", "q3"]
        assert any(
            event.get("name") == "ix_hold_forced_close"
            for event in outbound.metadata["tool_events"]
        )

    run(scenario())


# ---------------------------------------------------------------------------
# The wrapper path is not conditional on the response looking interesting
# ---------------------------------------------------------------------------

def test_a_response_with_no_ix_call_is_still_binding_checked(loop_factory) -> None:
    """A turn with no IX call is still the IX layer being engaged.

    A response that happened to carry no IX call used to fall through to the
    plain concurrent path — which never validates bindings, never gates and
    never reads a `capability_limit`. Live, seven task calls ran with
    `interaction_item_id = None`. The check is keyed on `session_active_ix`,
    not on what this particular response contains: turn two below selects
    nothing and still gets its bindings validated.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Autonomous"),
            ]),
            LLMResponse(content="Nothing to do yet."),
            # Turn two: a world-changing call, no IX call, no control directive.
            LLMResponse(content="Sending it.", tool_calls=[
                task_call("send_1", "task_send"),
            ]),
            LLMResponse(content="Sending it.", tool_calls=[
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_2", "task_send", "i1", ask="Send it to Dale?"),
            ]),
            LLMResponse(content="Sent."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(loop, "Stand by.", active_ix=["autonomy_level"],
                       action_types=ACTION_TYPES)
        second_start = len(provider.requests)
        outbound = await run_turn(
            loop, "Mail Dale.", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
        )

        # The unbound call never reached the world; the repaired one did.
        assert executed_tools(outbound) == ["task_send"]
        repairs = [
            str(message.get("content"))
            for request in provider.requests[second_start:]
            for message in request
            if "bindings are not valid" in str(message.get("content") or "")
        ]
        assert repairs and "missing interaction_item_id" in repairs[0]
        held = outbound.metadata["ix_render"]["held"]["calls"]
        # acc_009: a held call the PA bound to nothing is given an id off its own
        # provider call id, so the row names something rather than nothing.
        assert [item["interaction_item_id"] for item in held] == ["held_send_1"]
        # A refused binding batch is not a cross-turn hold: it was answered
        # inside the turn, so it opens no episode and the sideband is not
        # asked to consent to it.
        assert held[0]["episode_id"] == ""
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def test_a_refusal_repaired_in_the_same_turn_leaves_no_outstanding_hold(
    loop_factory,
) -> None:
    """acc_009: the retry ran, and the PA still reported the steps as not done.

    A `binding_invalid` refusal opens no episode, so its rows only ever reach
    `audit_trace` — where they read as work that did not happen. The refusal is
    real history and stays, marked with the call that answered it; nothing the
    PA is shown carries it forward.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Sending it.", tool_calls=[
                selection("autonomy_level", "Autonomous"),
                task_call("send_1", "task_send"),
            ]),
            LLMResponse(content="Sending it.", tool_calls=[
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_2", "task_send", "i1", ask="Send it to Dale?"),
            ]),
            LLMResponse(content="Sent."),
            LLMResponse(content="Nothing else."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Mail Dale.", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
        )

        assert executed_tools(outbound) == ["task_send"]
        held = outbound.metadata["ix_render"]["held"]["calls"]
        # The row survives as history, stamped with the retry that answered it…
        assert [item["reason"] for item in held] == ["binding_invalid"]
        assert held[0]["repaired"] is True
        assert held[0]["repaired_by"] == "send_2"
        # … and nothing is left outstanding.
        assert [item for item in held if not item["repaired"]] == []

        # The next turn's reminder says nothing about it: the refusal opened no
        # episode, so there is no held id to resume.
        next_start = len(provider.requests)
        await run_turn(
            loop, "Anything else?", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
        )
        reminders = [
            str(message.get("content") or "")
            for request in provider.requests[next_start:]
            for message in request
            if "You still hold" in str(message.get("content") or "")
            or "Not cleared for execution" in str(message.get("content") or "")
        ]
        assert reminders == []

    run(scenario())


def test_a_refusal_repaired_in_the_same_turn_claims_no_unexecuted_action(
    loop_factory,
) -> None:
    """The stamped row is not an outstanding action either.

    `mark_binding_repaired` marks `held.calls`, but the flag above scans
    `collector.calls`, where the refused attempt still sits `executed=False`.
    So the turn whose retry ran `task_send` told the judge that `task_send` was
    claimed and never executed -- the same false verdict `acc_001` t6
    drew one path over, on `ix_selection_invalid`.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Sending it.", tool_calls=[
                selection("autonomy_level", "Autonomous"),
                task_call("send_1", "task_send"),
            ]),
            LLMResponse(content="Sending it.", tool_calls=[
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_2", "task_send", "i1", ask="Send it to Dale?"),
            ]),
            LLMResponse(content="Sent."),
            LLMResponse(content="Nothing else."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Mail Dale.", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
        )

        # The retry ran the send, and the refused row is stamped as answered…
        assert executed_tools(outbound) == ["task_send"]
        held = outbound.metadata["ix_render"]["held"]["calls"]
        assert [item["repaired"] for item in held] == [True]
        # … so nothing on this turn claims an action went unexecuted.
        assert "ix_claims_unexecuted_action" not in [
            event.get("name") for event in outbound.metadata["tool_events"]
        ]

    run(scenario())


def test_a_capability_limit_is_seen_on_a_turn_with_no_ix_call(loop_factory) -> None:
    """The two-phase protocol does not need an IX call to start.

    Live, a `--force-ix` seed put `capability_boundary` in the session before
    the beat that lists it in `session_active_ix`. `scholarly_paper_read`
    returned `capability_limit` inside a response with no IX tool call, the
    wrapper never learned of it, `IX_capability_resolve` was never forced, and
    the fail-closed contract never engaged.
    """
    async def scenario() -> None:
        resolve = ix(
            "IX_capability_resolve",
            boundary_id="bd1",
            blocked_goal="get the full text",
            boundary_kind="missing_data",
            recovery_options=[{"id": "r1", "intent": "email the author"}],
            handoff_message="The catalogue cannot reach that preprint.",
        )
        provider = ScriptProvider([
            # No IX call, no attribute in this beat's `session_active_ix`.
            LLMResponse(content="Fetching it.", tool_calls=[
                task_call("read_1", "task_paper_read"),
            ]),
            # The wrapper's reminder lands here; the model ignores it once.
            LLMResponse(content="Here are the results anyway."),
            LLMResponse(content="Declaring the boundary.", tool_calls=[resolve]),
            LLMResponse(content="I cannot reach the full text; here is the abstract."),
        ])
        loop, _ = loop_factory(provider)
        loop.tools.register(CapabilityLimitTool("task_paper_read"))
        seed_ix_selection(
            loop, session_key=SESSION_KEY, script_session_id=SCRIPT_SESSION,
            attribute="capability_boundary", setting="Suggest alternatives",
        )

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=[], action_types={"task_paper_read": "read"},
        )

        reminders = [
            message
            for request in provider.requests
            for message in request
            if str(message.get("content") or "") == IX_CAPABILITY_RESOLVE_REMINDER
        ]
        assert reminders, "the capability limit was never detected"
        assert "IX_capability_resolve" in [
            event.get("name") for event in outbound.metadata["tool_events"]
        ]
        assert not [
            event for event in outbound.metadata["tool_events"]
            if event.get("name") == "ix_boundary_unresolved"
        ]

    run(scenario())


# ---------------------------------------------------------------------------
# a refused selection is attempted, never completed
# ---------------------------------------------------------------------------

#: `Suggest` was retired when `autonomy_level` went to three settings, so the
#: registry rejects it — while the same response carries a real external action.
def _retired_setting_responses(*, repaired: bool) -> list[LLMResponse]:
    refused = LLMResponse(content="Sending it now.", tool_calls=[
        selection("autonomy_level", "Suggest"),
        plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
        task_call("send_1", "task_send", "i1", ask="Shall I mail Dale?"),
    ])
    if not repaired:
        return [refused, refused, refused]
    return [
        refused,
        LLMResponse(content="Sending it now.", tool_calls=[
            selection("autonomy_level", "Reactive"),
            plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
            task_call("send_2", "task_send", "i1", ask="Shall I mail Dale?"),
        ]),
        # Never reached: the repaired turn ends at the gate. It is here so that
        # a regression -- the refused response executing after all -- fails on
        # the side-effect assertion rather than on an exhausted script.
        LLMResponse(content="The extension request is on its way."),
    ]


def test_a_refused_selection_withholds_the_same_turns_task_calls(loop_factory) -> None:
    """The action waits for a valid policy instead of running under an empty one.

    A rejected selection contributes no payload, so the directive is rebuilt
    empty — and an empty policy gates nothing. The refused response's
    `external_action` used to go straight out, unasked and unheld; under the
    `Reactive` the repair then chose, the very same call is gated instead.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_retired_setting_responses(repaired=True))
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert loop.tools.get("task_send").calls == []
        assert executed_tools(outbound) == []
        assert outbound.content.rstrip().endswith("Shall I mail Dale?")
        # …and the repair is on the record, asked through the payload reminder.
        assert [
            event.get("name") for event in outbound.metadata["tool_events"]
        ].count("ix_selection_invalid") == 1
        assert [
            str(message.get("content"))
            for request in provider.requests
            for message in request
            if str(message.get("content") or "").startswith(
                IX_PER_TURN_REMINDER_PREFIX
            )
        ] != []

    run(scenario())


def test_a_selection_refused_to_the_limit_fails_the_turn_closed(loop_factory) -> None:
    """It never degrades to an empty policy: the turn ends on the contract.

    And it ends on the *arguments* sentence, not the not-offered one. Live
    `acc_001` is this shape: `active_ix_tools: ["IX_verbosity"]`, the
    PA called `IX_verbosity` with `args: {}`, the registry answered "Invalid
    parameters for tool 'IX_verbosity': missing required setting" three times,
    and the failure was still reported as "the attribute's tool is not offered
    on this beat" -- which pointed the reader at a tool list that was correct.
    Here `autonomy_level` is likewise offered and `Suggest` is a retired
    setting, so the refusal is the same one.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_retired_setting_responses(repaired=False))
        loop, _ = loop_factory(provider, "task_send")

        with pytest.raises(AgentProviderError) as caught:
            await run_turn(
                loop, "Mail Dale the extension request.",
                active_ix=["autonomy_level"], action_types=ACTION_TYPES,
            )

        assert str(caught.value) == (
            "PA turn failed the IX selection contract: "
            + IX_SELECTION_ARGS_CONTRACT_FAILURE_DETAIL
        )
        assert [
            event["cause"]
            for event in caught.value.failed_turn_events["tool_events"]
            if event.get("name") == "ix_selection_invalid"
        ] == ["selection_args"] * 3
        assert loop.tools.get("task_send").calls == []
        assert loop.sessions.get(SESSION_KEY).messages == []

    run(scenario())


#: `acc_027`, the reaction beat: `active_ix_tools: []`, so
#: `IX_emotional_engagement` is not offered and every call comes back "not
#: found" -- three refusals, the fail-closed break, no reply.
def _never_offered_selection_responses() -> list[LLMResponse]:
    refused = LLMResponse(content="That is a lot to land at once.", tool_calls=[
        selection("emotional_engagement", "Relationship-focused"),
    ])
    return [refused, refused, refused]


def test_a_never_offered_selection_fails_on_the_selection_contract(
    loop_factory,
) -> None:
    """The refused-selection break says what it broke, not what it did not.

    Live `acc_027` ended with `PA turn failed the IX capability contract:
    IX_capability_resolve was missing or invalid twice after a tool reported a
    capability limit`. No tool on that turn reported one -- all ten store calls
    returned ok and `capability_required` was false on every iteration. The
    loop read event names and fell through to the capability text by default,
    so a path with no capability in it wore the capability label. The
    fail-closed policy is unchanged; only the sentence is.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_never_offered_selection_responses())
        loop, _ = loop_factory(provider, "task_send")

        with pytest.raises(AgentProviderError) as caught:
            await run_turn(
                loop, "They moved the review up again.",
                active_ix=[],
                action_types=ACTION_TYPES,
                extra_ix_context={"session_active_ix": ["emotional_engagement"]},
            )

        assert str(caught.value) == (
            "PA turn failed the IX selection contract: "
            + IX_SELECTION_CONTRACT_FAILURE_DETAIL
        )
        # And the record now names the beat and the attribute, which the live
        # `failed_turn_events.json` could not: three `tools`-only rows.
        refusals = [
            event
            for event in caught.value.failed_turn_events["tool_events"]
            if event.get("name") == "ix_selection_invalid"
        ]
        assert [event["attempt"] for event in refusals] == [1, 2, 3]
        assert [event["tools"] for event in refusals] == (
            [["IX_emotional_engagement"]] * 3
        )
        assert refusals[-1]["session_active_ix"] == ["emotional_engagement"]
        # The beat listing the attribute is not the same thing as the beat
        # offering its tool, which is why the cause reads this turn's tool view.
        assert [event["cause"] for event in refusals] == ["selection"] * 3

    run(scenario())


#: `acc_001` t6. The reception beat cues no IX tool, so the model's
#: `IX_verbosity` call comes back "not found" -- a refused selection -- and the
#: same response's two `document_update` calls are withheld with it. The retry
#: drops the selection and reissues the writes, which run, exactly as intended
#: ("reissued with the corrected selection or not at all").
def _reissued_write_responses() -> list[LLMResponse]:
    refused = LLMResponse(content="Both updates are saved.", tool_calls=[
        ix(
            "IX_verbosity", setting="Detailed",
            evidence="", application="confirm the recorded changes",
        ),
        task_call("w1", "task_write"),
        task_call("w2", "task_write"),
    ])
    return [
        refused,
        LLMResponse(content="Both updates are saved.", tool_calls=[
            task_call("w3", "task_write"),
            task_call("w4", "task_write"),
        ]),
        LLMResponse(content="Both updates are saved."),
    ]


def test_a_refused_selections_holds_do_not_outlive_the_reissued_calls(
    loop_factory,
) -> None:
    """The refused attempt's hold rows go with it when the retry runs the calls.

    Live `acc_001` t6 delivered both `document_update`s -- `state_diff.jsonl`
    and `canonical_stage` carry them -- while `held.calls` still listed the
    withheld first attempt as `held_for_binding` and
    `ix_claims_unexecuted_action` fired on the very turn that carried them out.
    Same shape as the `batch_undivided` retry (`clear_holds`) and the
    `binding_invalid` retry (`mark_binding_repaired`); this path had neither.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_reissued_write_responses())
        loop, _ = loop_factory(provider, "task_write")

        outbound = await run_turn(
            loop, "The citation for that sentence is resolved.",
            active_ix=[],
            action_types={"task_write": "internal_write"},
            extra_ix_context={"session_active_ix": ["verbosity"]},
        )

        # The refusal happened, and the reissued writes really ran.
        names = [event.get("name") for event in outbound.metadata["tool_events"]]
        assert names.count("ix_selection_invalid") == 1
        assert len(loop.tools.get("task_write").calls) == 2

        # So nothing is outstanding: no unrepaired hold row, and no claim that
        # the turn left an action unexecuted.
        held = outbound.metadata["ix_render"]["held"]["calls"]
        assert [row for row in held if not row["repaired"]] == []
        assert "ix_claims_unexecuted_action" not in names

    run(scenario())


# ---------------------------------------------------------------------------
# a gate with no ask asks the user nothing
# ---------------------------------------------------------------------------

def _no_ask_gate_responses(*, repaired: bool) -> list[LLMResponse]:
    first = LLMResponse(content="Let me set up the plan.", tool_calls=[
        selection("autonomy_level", "Reactive"),
        plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
        task_call("send_1", "task_send", "i1"),
    ])
    second_ask = "Shall I mail Dale the extension request?" if repaired else ""
    second = LLMResponse(content="Let me set up the plan.", tool_calls=[
        task_call("send_2", "task_send", "i1", ask=second_ask),
    ])
    if repaired:
        return [first, second]
    third = LLMResponse(content="Let me set up the plan.", tool_calls=[
        task_call("send_3", "task_send", "i1"),
    ])
    return [first, second, third]


def test_a_gate_without_the_ask_is_repaired_once(loop_factory) -> None:
    """The repair asks for the field, and the retry's ask ships verbatim.

    The renderer refuses to write the ask for the PA, so an unrepaired gate
    turn delivers the PA's own prose and asks nothing — live, two gate turns
    shipped as `Let me set up the plan.` while the simulator still answered
    `allow`.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_no_ask_gate_responses(repaired=True))
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        repairs = [
            str(message.get("content"))
            for request in provider.requests
            for message in request
            if IX_CONFIRMATION_MESSAGE_FIELD in str(message.get("content") or "")
            and str(message.get("content") or "").startswith("[interaction wrapper]")
        ]
        assert len(repairs) == 1
        assert outbound.content.rstrip().endswith(
            "Shall I mail Dale the extension request?"
        )
        assert executed_tools(outbound) == []
        # The refused attempt is not reported as a second held call.
        held = outbound.metadata["ix_render"]["held"]["calls"]
        assert [item["tool"] for item in held] == ["task_send"]
        assert [
            event.get("name") for event in outbound.metadata["tool_events"]
        ].count("ix_gate_no_message") == 1

    run(scenario())


def test_a_gate_without_the_ask_three_times_fails_and_leaves_the_gate_open(
    loop_factory,
) -> None:
    """The third miss fails the turn; the sole held gate is still open at close."""
    async def scenario() -> None:
        responses = _no_ask_gate_responses(repaired=False)
        responses.append(responses[-1])
        provider = ScriptProvider(responses)
        loop, _ = loop_factory(provider, "task_send")

        with pytest.raises(AgentProviderError) as caught:
            await run_turn(
                loop, "Mail Dale the extension request.",
                active_ix=["autonomy_level"], action_types=ACTION_TYPES,
            )

        error = caught.value
        assert len(provider.requests) == 3
        assert error.ix_contract_exhaustion == {
            "attempts": 3,
            "contract": "gate",
            "operation": "gate_confirmation_message",
        }
        assert [
            attribute for attribute, _ in loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION)
        ] == ["autonomy_level"]
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []
        # Nothing was persisted for a turn that produced no scorable reply.
        assert loop.sessions.get(SESSION_KEY).messages == []

    run(scenario())


@pytest.mark.parametrize("ask", ["", "I'll mail Dale the extension request."])
def test_failed_gate_keeps_the_final_unexecuted_confirmation_for_diagnosis(
    loop_factory, ask: str,
) -> None:
    async def scenario() -> None:
        responses = _no_ask_gate_responses(repaired=False)
        responses[-1].tool_calls = [task_call("send_2", "task_send", "i1", ask=ask)]
        provider = ScriptProvider(responses)
        loop, _ = loop_factory(provider, "task_send")
        with pytest.raises(AgentProviderError) as caught:
            await run_turn(
                loop, "Mail Dale the extension request.",
                active_ix=["autonomy_level"], action_types=ACTION_TYPES,
            )
        assert caught.value.failed_turn_events["unexecuted_calls"] == [{
            "tool": "task_send", "call_id": "send_2", "interaction_item_id": "i1",
            "confirmation_message": ask, "status": "held",
        }]
        # The server line's gate budget: three actual responses before the
        # turn fails (test_ix_adjacent_operation_budget).
        assert len(provider.requests) == 3
        assert loop.tools.get("task_send").calls == []
        assert loop.sessions.get(SESSION_KEY).messages == []

    run(scenario())


def test_an_empty_ask_on_an_earlier_held_call_does_not_shadow_the_real_ask(
    loop_factory,
) -> None:
    """The gate's ask is the first held call that *wrote* one.

    Live t24: one item held two calls, a refused `document_read` with no
    `_ix_confirmation_message` ahead of the `document_update` that carried the
    real ask. The runner's lookup skipped the empty one and concluded the ask
    existed; the renderer's took the first id match and concluded it did not.
    Neither the repair nor the ask fired, and every held call shipped
    `asked: false` — so the user's next `allow` would authorise nothing.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            # Refused on its bindings: `task_read` binds `i1` and writes no ask,
            # and the held record of that attempt outlives the repair.
            LLMResponse(content="Lining it up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("read_1", "task_read", "i1"),
                task_call("send_bad", "task_send", "i2", ask="Mail Dale?"),
            ]),
            # The corrected batch: the gate holds this one, and it carries the ask.
            LLMResponse(content="Lining it up.", tool_calls=[
                task_call(
                    "send_2", "task_send", "i1",
                    ask="Shall I mail Dale the extension request?",
                ),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send", "task_read")

        outbound = await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        # The two lookups agree: an ask exists, so no repair is owed...
        assert [
            event.get("name") for event in outbound.metadata["tool_events"]
        ].count("ix_gate_no_message") == 0
        # ...and the renderer delivers it rather than reporting it missing.
        assert outbound.content.rstrip().endswith(
            "Shall I mail Dale the extension request?"
        )
        held = outbound.metadata["ix_render"]["held"]["calls"]
        gated = [item for item in held if item["reason"] == "external_action_floor"]
        assert [item["tool"] for item in gated] == ["task_send"]
        assert [item["asked"] for item in gated] == [True]

    run(scenario())


def test_an_allow_for_an_id_the_ask_never_named_releases_nothing(loop_factory) -> None:
    """The consent rule as written: "every held id **shown** to the user", not "every held id".

    Live (t19/t21), an ask that named only a cancellation came back with
    an `allow` for six ids, five of them searches. The sideband is the user's
    voice, so an `allow` for an id no ask put to them authorises nothing: the
    item keeps waiting for an ask of its own.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Lining them up.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                ),
                task_call("send_1", "task_send", "i1", ask="Mail Dale?"),
                task_call("send_2", "task_send", "i2", ask="Mail Nadia?"),
            ]),
            LLMResponse(content="Sending both.", tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "yes"},
                    {"id": "i2", "decision": "release", "user_evidence": "yes"},
                ]),
                task_call("send_1b", "task_send", "i1", ask="Mail Dale?"),
                task_call("send_2b", "task_send", "i2", ask="Mail Nadia?"),
            ]),
            LLMResponse(content="Sent."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        first = await run_turn(
            loop, "Mail both of them.", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
        )
        # One ask went out, and it named one id.
        assert first.content.rstrip().endswith("Mail Dale?")
        assert {
            call["interaction_item_id"]: call["asked"]
            for call in first.metadata["ix_render"]["held"]["calls"]
        } == {"i1": True, "i2": False}
        # The simulator is offered exactly that id, and nothing else.
        assert [
            item["interaction_item_id"] for item in held_items_from_ix_render(
                first.metadata["ix_render"]
            )
        ] == ["i1"]

        second = await run_turn(
            loop, "Go ahead.", active_ix=["autonomy_level"], action_types=ACTION_TYPES,
            consent=[
                {"interaction_item_id": "i1", "decision": "allow"},
                {"interaction_item_id": "i2", "decision": "allow"},
            ],
        )

        assert executed_tools(second) == ["task_send"]
        unasked = next(
            event for event in second.metadata["tool_events"]
            if event.get("name") == "ix_consent_unasked"
        )
        assert unasked["item_id"] == "i2"
        consent = {row["item_id"]: row for row in second.metadata["ix_render"]["consent"]}
        assert consent["i1"]["released"] is True
        assert consent["i2"]["sideband_decision"] == "allow"
        assert consent["i2"]["released"] is False

    run(scenario())


# ---------------------------------------------------------------------------
# the narration of a held call is not delivered
# ---------------------------------------------------------------------------

def test_a_gate_turn_does_not_ship_the_narration_of_the_held_call(loop_factory) -> None:
    """The prose beside a held call claims work that never happened.

    Live (t12), the PA wrote `📧 Email confirmation sent to Dr. Okafor`
    while the send was held; the id never executed in the whole run. Prose in the
    same assistant message as a held call is audit-only, and the ask is what the
    user gets.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Email sent! Dr. Okafor has the confirmation.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Shall I mail Dr. Okafor?"),
            ]),
        ])
        loop, adapter = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Confirm receipt with Dr. Okafor.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert executed_tools(outbound) == []
        assert "Email sent!" not in outbound.content
        assert outbound.content.rstrip().endswith("Shall I mail Dr. Okafor?")
        # … and it is not smuggled back in through history or memory either.
        assert "Email sent!" not in adapter.events[0].messages[-1]["content"]
        assert "Email sent!" not in _blob(loop.sessions.get(SESSION_KEY).messages)

        blob = outbound.metadata["ix_render"]
        assert blob["held"]["narration"] == [
            {"item_id": "i1", "text": "Email sent! Dr. Okafor has the confirmation."}
        ]
        assert "held narration withheld (i1)" in blob["trace"]

    run(scenario())


def test_the_executed_iterations_prose_survives_a_later_held_narration(loop_factory) -> None:
    """Only the held call's own message is withheld, not the whole turn.

    The first iteration's read really ran, so what it said about it is true and
    still goes out; the second iteration's `Email sent!` does not.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="I pulled up his announcement.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                ),
                task_call("read_1", "task_read", "i1"),
            ]),
            LLMResponse(content="Email sent!", tool_calls=[
                task_call("send_1", "task_send", "i2", ask="Shall I mail Dr. Okafor?"),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_read", "task_send")

        outbound = await run_turn(
            loop, "Read his note, then confirm receipt.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        assert executed_tools(outbound) == ["task_read"]
        assert "I pulled up his announcement." in outbound.content
        assert "Email sent!" not in outbound.content
        assert outbound.content.rstrip().endswith("Shall I mail Dr. Okafor?")
        assert outbound.metadata["ix_render"]["held"]["narration"] == [
            {"item_id": "i2", "text": "Email sent!"}
        ]

    run(scenario())


def test_a_questions_gate_hold_carries_an_id_the_sideband_can_name(
    loop_factory,
) -> None:
    """acc_009 t=12/t=14 — a hold nobody could name, and nobody could release.

    `information_elicitation` alone does not make `interaction_item_id`
    declarable, so the PA is right not to send one. The questions gate then held
    `email_draft_create` with `interaction_item_id: ""`, `held.calls` exposed no
    id, the harness reported `held ids: []`, and both of the user's consent
    grants — "The draft is acceptable. Send it." — were dropped as unknown ids.
    The wrapper now mints the id off the provider's own call id.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(
                content="I can draft it once I know one thing.\n\n"
                        "[[q:q1]] Which airport?",
                tool_calls=[
                    selection("information_elicitation", "Structured"),
                    ix(
                        "IX_render_information_elicitation",
                        elicitation_id="el1",
                        questions=_questions("q1"),
                        assumptions=[],
                    ),
                    task_call("draft_1", "task_draft"),
                ],
            ),
        ])
        loop, _ = loop_factory(provider, "task_draft")

        outbound = await run_turn(
            loop, "Draft the mail.", active_ix=["information_elicitation"],
            action_types={"task_draft": "draft"},
        )

        assert executed_tools(outbound) == []
        held = outbound.metadata["ix_render"]["held"]["calls"]
        assert [item["tool"] for item in held] == ["task_draft"]
        assert [item["interaction_item_id"] for item in held] == ["held_draft_1"]
        assert held[0]["reason"] == "questions_gate"

    run(scenario())
