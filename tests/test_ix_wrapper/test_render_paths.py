"""T5–T8, T10 and one end-to-end render path per rendered attribute.

`tests/test_pa/test_ix_render.py` proves the renderer is a
correct pure function; `tests/test_pa/test_runner_ix.py` proves the gate stops
the right calls. What is missing between them is the whole path: a scripted
provider response going through `AgentLoop.process_direct_message` and coming
out as the one message the simulated user sees, with the tool log and the
`ix_render` blob agreeing about what ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from harness.judge_pack import build_text_judge_pack
from pa.agent.loop import AgentProviderError
from pa.agent.runner import (
    IX_MISSING_NOTICE_PROMPT,
    IX_PER_TURN_REMINDER_PREFIX,
    IX_RATIONALE_WORK_LOG_REPAIR,
    IX_UNFINISHED_TURN_PROMPT,
)
from pa.providers.base import LLMResponse

from .conftest import (
    SESSION_KEY,
    ScriptProvider,
    executed_tools,
    ix,
    plan_call,
    run,
    run_turn,
    selection,
    task_call,
)


def _ix_events(outbound: Any) -> list[dict]:
    return [
        event for event in outbound.metadata["tool_events"] or []
        if event.get("ix_event")
    ]


def _provider_requests(loop: Any) -> list[dict]:
    """Every message the scripted provider was handed this session."""
    return [message for request in loop.pa_llm_provider.requests for message in request]


# ---------------------------------------------------------------------------
# T5 · one plan item, three calls
# ---------------------------------------------------------------------------

def test_one_plan_item_that_expands_into_three_calls_is_gated_as_one(
    loop_factory,
) -> None:
    """T5: the gate is on the item, so all three of its calls stop."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Filing all three.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1", "i2", "i3"]}),
                task_call("w1", "task_write", "i1", ask="Shall I file all three?"),
                task_call("w2", "task_write", "i2"),
                task_call("w3", "task_write", "i3"),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_write")

        outbound = await run_turn(
            loop, "File the three notes.", active_ix=["autonomy_level"],
            action_types={"task_write": "internal_write"},
        )

        assert executed_tools(outbound) == []
        held = outbound.metadata["ix_render"]["held"]["calls"]
        assert [item["interaction_item_id"] for item in held] == ["i1", "i2", "i3"]
        assert {item["plan_item_id"] for item in held} == {"a1"}
        assert outbound.content.rstrip().endswith("Shall I file all three?")

    run(scenario())


# ---------------------------------------------------------------------------
# T6 · one call, one claiming plan item
# ---------------------------------------------------------------------------

def test_one_call_is_claimed_by_exactly_one_plan_item(loop_factory) -> None:
    """T6: a call carries one id, so it settles one plan item only.

    The two plan items nobody called are neither executed nor held — they are
    still outstanding, and the next turn has to come back to them.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Starting with the first.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call(
                    {"id": "a1", "intent": "file the note", "bound_item_ids": ["i1"]},
                    {"id": "a2", "intent": "mail Dale", "bound_item_ids": ["i2"]},
                    {"id": "a3", "intent": "update the sheet", "bound_item_ids": ["i3"]},
                ),
                task_call("w1", "task_write", "i1", ask="File the note?"),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_write")

        outbound = await run_turn(
            loop, "Three things, please.", active_ix=["autonomy_level"],
            action_types={"task_write": "internal_write"},
        )

        held = outbound.metadata["ix_render"]["held"]
        assert [item["plan_item_id"] for item in held["calls"]] == ["a1"]
        # Only the claimed item is carried into the hold; a2 and a3 were never
        # bound to a call this turn, so the wrapper does not invent one for them.
        assert [item["id"] for item in held["plan_remaining"]] == ["a1"]
        assert executed_tools(outbound) == []

    run(scenario())


# ---------------------------------------------------------------------------
# T7 · a bad binding stops the batch, twice over
# ---------------------------------------------------------------------------

def test_a_second_invalid_binding_delivers_text_and_runs_nothing(loop_factory) -> None:
    """T7 at loop level, with only `topic_management` on.

    The runner-level case in `tests/test_pa/test_runner_ix.py` shows the repair
    prompt. This one shows what the *user* gets when the repair does not take:
    a tool log with no task call in it, and **not** the prose — `Doing the
    hotel.` was written in the same response as the calls that never ran, so it
    is held narration and lives only on the audit plane.
    """
    async def scenario() -> None:
        unbound = task_call("w1", "task_echo", "")
        payload = ix(
            "IX_render_topic_management",
            batch_set_id="bs1",
            item_count=1,
            batches=[{"batch_id": "b1", "label": "hotel", "body": "",
                      "call_binding": ["i1"]}],
        )
        provider = ScriptProvider([
            LLMResponse(content="Doing the hotel.", tool_calls=[
                selection("topic_management", "Organize"), payload, unbound,
            ]),
            LLMResponse(content="Doing the hotel.", tool_calls=[payload, unbound]),
            # The closing iteration now runs on this exit too, and this model writes nothing
            # when it is asked -- which is what left live t18 with no message.
            LLMResponse(content=None),
        ])
        loop, adapter = loop_factory(provider)

        outbound = await run_turn(
            loop, "Sort the hotel.", active_ix=["topic_management"],
            action_types={"task_echo": "internal_write"},
        )

        assert executed_tools(outbound) == []
        assert "Doing the hotel." not in outbound.content
        assert any(
            event.get("name") == "ix_binding_invalid" for event in _ix_events(outbound)
        )
        assert "Doing the hotel." not in adapter.events[0].messages[-1]["content"]
        assert [
            row["text"] for row in outbound.metadata["ix_render"]["held"]["narration"]
        ] == ["Doing the hotel."]
        # …and the user is told *something*, not the loop's placeholder.
        assert outbound.content == "I could not carry out those steps this turn; I will retry."
        assert outbound.metadata["ix_render"]["close_line_source"] == "template"

    run(scenario())


def _batch_payload(label: str, item_id: str, batch_id: str = "b1") -> Any:
    return ix(
        "IX_render_topic_management",
        batch_set_id="bs_001",
        item_count=1,
        batches=[{"batch_id": batch_id, "label": label, "body": "",
                  "call_binding": [item_id]}],
    )


def test_a_batch_id_may_not_be_rebound_to_different_calls(loop_factory) -> None:
    """acc_023 t18 / t20 — one batch id, two different calls, one session.

    t18 declared `bs_001` with `b1` = "Calendar deadline" binding the calendar
    write and released it; t20 re-declared `bs_001` with the same `b1`
    relabelled "Send confirmation email", now binding the send, and the wrapper
    released it and the mail went out. A batch id is the handle consent and the
    audit both hang on, so the rebinding takes the fail-closed exit.
    """
    async def scenario() -> None:
        rebound = [
            _batch_payload("Send confirmation email", "snd_1"),
            task_call("snd1", "task_echo", "snd_1"),
        ]
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "Organize"),
                _batch_payload("Calendar deadline", "cal_1"),
                task_call("cal1", "task_echo", "cal_1"),
            ]),
            LLMResponse(content="[[batch:b1]] Deadline logged."),
            LLMResponse(content=None, tool_calls=list(rebound)),
            LLMResponse(content=None, tool_calls=list(rebound)),
            LLMResponse(content="I will come back to that."),
        ])
        loop, _ = loop_factory(provider)

        first = await run_turn(
            loop, "Log the deadline.", active_ix=["topic_management"],
            action_types={"task_echo": "internal_write"},
        )
        assert executed_tools(first) == ["task_echo"]

        second = await run_turn(
            loop, "Now send the confirmation.", active_ix=["topic_management"],
            action_types={"task_echo": "internal_write"},
        )

        assert executed_tools(second) == []
        event = next(
            item for item in _ix_events(second)
            if item["name"] == "ix_batch_rebound"
        )
        assert event["batch_ids"] == ["b1"]
        repairs = [
            str(message.get("content") or "")
            for request in provider.requests
            for message in request
            if "binding different calls" in str(message.get("content") or "")
        ]
        assert repairs

    run(scenario())


def test_the_same_batch_id_with_the_same_binding_is_a_resend(loop_factory) -> None:
    """The guard: re-declaring an id unchanged is not a rebinding.

    A carry turn legitimately re-sends the payload it already declared, and that
    must not take the fail-closed exit.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "Organize"),
                _batch_payload("Calendar deadline", "cal_1"),
            ]),
            LLMResponse(content="[[batch:b1]] Deadline noted."),
            LLMResponse(content=None, tool_calls=[
                _batch_payload("Calendar deadline", "cal_1"),
                task_call("cal1", "task_echo", "cal_1"),
            ]),
            LLMResponse(content="[[batch:b1]] Deadline logged."),
        ])
        loop, _ = loop_factory(provider)

        await run_turn(
            loop, "Note the deadline.", active_ix=["topic_management"],
            action_types={"task_echo": "internal_write"},
        )
        second = await run_turn(
            loop, "Log it now.", active_ix=["topic_management"],
            action_types={"task_echo": "internal_write"},
        )

        assert executed_tools(second) == ["task_echo"]
        assert "ix_batch_rebound" not in [
            item["name"] for item in _ix_events(second)
        ]

    run(scenario())


def test_a_binding_refusal_gets_its_closing_iteration(loop_factory) -> None:
    """The closing iteration extended to the binding refusal's fail-closed exit.

    Live t18: a `binding_invalid` hold took all six calls, the closing
    iteration was never run on this path, `last_segment` was empty, and the
    entire reply the user received was the loop's placeholder while three reads
    were silently dropped. The closing request now runs here too, with task
    tools gone and held results that name the real problem.

    The template line is **appended** to the model's own closing text, never
    substituted for it. The two
    sentences are two different facts — what the PA has to say, and that the
    steps did not run — and substituting one for the other is how one live turn
    delivered a refusal instead of the four questions it had just declared.
    """
    async def scenario() -> None:
        unbound = task_call("w1", "task_echo", "")
        payload = ix(
            "IX_render_topic_management",
            batch_set_id="bs1",
            item_count=1,
            batches=[{"batch_id": "b1", "label": "hotel", "body": "",
                      "call_binding": ["i1"]}],
        )
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "Organize"), payload, unbound,
            ]),
            LLMResponse(content=None, tool_calls=[payload, unbound]),
            LLMResponse(content="I could not book the hotel yet; I will try again."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Sort the hotel.", active_ix=["topic_management"],
            action_types={"task_echo": "internal_write"},
        )

        assert executed_tools(outbound) == []
        # The model's own closing text survives, with the template line after it.
        assert outbound.content == (
            "I could not book the hotel yet; I will try again.\n\n"
            "I could not carry out those steps this turn; I will retry."
        )
        assert outbound.metadata["ix_render"]["close_line_source"] == "template"
        assert outbound.metadata["ix_render"]["closing_iteration"] is True
        # The closing request said why the calls did not run, in as many words,
        # and offered no task tool to run them with.
        closing_request = provider.requests[-1]
        assert any(
            message.get("role") == "tool"
            and "missing/duplicate interaction_item_id" in str(message.get("content"))
            for message in closing_request
        ), closing_request
        assert "task_echo" not in provider.offered(-1)

    run(scenario())


def test_the_sole_outstanding_item_is_not_written_in_for_an_unbound_call(
    loop_factory,
) -> None:
    """The binding check fails closed, with no hatch.

    An earlier version had the wrapper fill in the `interaction_item_id` the model
    omitted whenever the plan left exactly one item outstanding, so a
    world-changing call executed under an id the wrapper picked and a check
    written to refuse it passed instead. One outstanding item is the easiest
    case to guess and still a guess: the refusal is what a missing id gets.
    """
    async def scenario() -> None:
        unbound = task_call("w1", "task_write", "")
        payload = plan_call({"id": "step1", "bound_item_ids": ["i1"]})
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Self-directed"), payload, unbound,
            ]),
            LLMResponse(content=None, tool_calls=[payload, unbound]),
            LLMResponse(content="I will come back to it."),
        ])
        loop, _ = loop_factory(provider, "task_write")

        outbound = await run_turn(
            loop, "Put the checkup in the calendar.", active_ix=["autonomy_level"],
            action_types={"task_write": "internal_write"},
        )

        assert executed_tools(outbound) == []
        assert "ix_binding_invalid" in [
            event["name"] for event in _ix_events(outbound)
        ]

    run(scenario())


def test_the_close_line_is_dropped_when_the_prose_already_admits_the_step(
    loop_factory,
) -> None:
    """acc_039 t=22 told the user the same thing twice.

    The PA's own closing paragraph said the notes update *"didn't complete"* and
    the wrapper appended *"I could not carry out those steps this turn; I will
    retry."* directly underneath it. The line exists to stop silence about a
    refused batch, not to repeat an admission the PA has already made.
    """
    async def scenario() -> None:
        unbound = task_call("w1", "task_write", "")
        payload = plan_call(
            {"id": "step1", "bound_item_ids": ["i1"]},
            {"id": "step2", "bound_item_ids": ["i2"]},
        )
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("autonomy_level", "Self-directed"), payload, unbound,
            ]),
            LLMResponse(content=None, tool_calls=[payload, unbound]),
            LLMResponse(
                content="One thing that didn't complete: updating your clinic "
                        "notes. I'll take care of that at the start of the next "
                        "turn.",
            ),
        ])
        loop, _ = loop_factory(provider, "task_write")

        outbound = await run_turn(
            loop, "Put the checkup in the calendar.", active_ix=["autonomy_level"],
            action_types={"task_write": "internal_write"},
        )

        assert executed_tools(outbound) == []
        assert "I could not carry out those steps this turn" not in outbound.content
        assert outbound.content.startswith("One thing that didn't complete")
        assert outbound.metadata["ix_render"].get("close_line_source", "") == ""
        assert any(
            "template close line suppressed" in step
            for step in outbound.metadata["ix_render"]["trace"]
        )

    run(scenario())


# ---------------------------------------------------------------------------
# an empty reply is a miss, whatever emptied it
# ---------------------------------------------------------------------------

def test_the_loop_placeholder_string_is_gone_from_the_pa_package() -> None:
    """The structural half: the fallback itself was the defect.

    Three distinct routes to `I've completed processing but have no response to
    give.` turned up in live runs, among them the `binding_invalid` fail-closed
    exit and a provider completion with no text and no tool calls at all. Guarding them one at a
    time was not converging, so the string is not allowed to exist in `pa/`.
    """
    package = Path(__file__).resolve().parents[2] / "pa"
    # Whitespace-normalised whole-file text, not line by line: the sentence
    # spent one revision alive as a comment wrapped across two lines, which a
    # per-line grep reads as absent.
    hits = [
        str(path.relative_to(package.parent))
        for path in sorted(package.rglob("*.py"))
        if "have no response to give"
        in " ".join(path.read_text(encoding="utf-8").split())
    ]
    assert hits == []


def test_an_empty_provider_completion_buys_one_more_request_then_the_template(
    loop_factory,
) -> None:
    """The runtime half — the empty provider completion seen live.

    The PA returned an empty completion with no tool calls at all, answering a
    two-part instruction the user then heard nothing about: nothing was held, so
    the reason-specific guard could not fire and the loop's placeholder was
    the whole reply. The rule now lives at the runner's single exit and is keyed
    on the emptiness, not on the reason: one more request with the task tools
    withdrawn, and the template line only if that comes back empty too.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None),
            LLMResponse(content=None),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(loop, "Book the room and move the event.",
                                  active_ix=[])

        assert outbound.content == "I have nothing further to report on that just now."
        assert outbound.metadata["ix_render"]["close_line_source"] == "template"
        assert [
            event["name"] for event in _ix_events(outbound)
            if event["name"] == "ix_empty_reply"
        ] == ["ix_empty_reply"]
        # Exactly one extra request, and it asked for the reply in as many words
        # while offering nothing to run.
        assert len(provider.requests) == 2
        assert any(
            "no user-facing text" in str(message.get("content") or "")
            for message in provider.requests[-1]
        ), provider.requests[-1]
        assert provider.offered(-1) == []

    run(scenario())


def test_a_reply_the_model_writes_on_the_second_ask_is_delivered_as_is(
    loop_factory,
) -> None:
    """The extra request is a chance at a real sentence, not a formality.

    When it comes back with prose, that prose is the turn — the template line is
    the floor underneath it, never a replacement for it.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None),
            LLMResponse(content="The room is booked and the event now starts at 09:00."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(loop, "Book the room and move the event.",
                                  active_ix=[])

        assert outbound.content == (
            "The room is booked and the event now starts at 09:00."
        )
        assert "close_line_source" not in outbound.metadata["ix_render"]
        assert not [
            event for event in _ix_events(outbound)
            if event["name"] == "ix_empty_reply"
        ]

    run(scenario())


# ---------------------------------------------------------------------------
# questions survive a binding refusal
# ---------------------------------------------------------------------------

def test_released_questions_survive_a_binding_refusal_on_the_same_turn(
    loop_factory,
) -> None:
    """A live turn's shape, which cost the run its attribute.

    Under `Structured` the PA declared its questions *and* bound its task calls
    to the question ids. The binding check refused the batch, and the close line
    it delivered replaced the four questions the setting exists to test:
    `released.questions` was empty for the whole run and the elicitation episode
    closed having released nothing. The refusal and the questions are two facts,
    so the user gets both — the questions, and the line saying the steps did not
    run — and the repair now says which mistake was actually made.
    """
    async def scenario() -> None:
        questions = [
            {"id": "q1", "text": "How long?", "about": "duration"},
            {"id": "q2", "text": "Which format?", "about": "format"},
            {"id": "q3", "text": "Which week?", "about": "timing"},
            {"id": "q4", "text": "Who else joins?", "about": "attendees"},
        ]
        payload = ix(
            "IX_render_information_elicitation",
            elicitation_id="el1",
            questions=questions,
            assumptions=[],
        )
        misbound = [
            task_call("c1", "task_echo", "q1"),
            task_call("c2", "task_echo", "q2"),
        ]
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Structured"),
                payload, *misbound,
            ]),
            LLMResponse(content=None, tool_calls=[payload, *misbound]),
            LLMResponse(
                content="Before I book anything:\n\n"
                        "[[q:q1]] How long should the meeting be?\n\n"
                        "[[q:q2]] Written comments first, or talk it through cold?\n\n"
                        "[[q:q3]] Which week suits you?\n\n"
                        "[[q:q4]] Should anyone else be in the room?"
            ),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Set up the review meeting with Reyes.",
            active_ix=["information_elicitation"],
            action_types={"task_echo": "internal_write"},
        )

        assert executed_tools(outbound) == []
        for question in (
            "How long should the meeting be?",
            "Written comments first, or talk it through cold?",
            "Which week suits you?",
            "Should anyone else be in the room?",
        ):
            assert question in outbound.content
        assert outbound.content.endswith(
            "I could not carry out those steps this turn; I will retry."
        )
        assert outbound.metadata["ix_render"]["released"]["questions"] == [
            "q1", "q2", "q3", "q4",
        ]
        assert outbound.metadata["ix_render"]["close_line_source"] == "template"
        # The repair names the mistake the model actually made.
        repairs = [
            str(message.get("content"))
            for request in provider.requests
            for message in request
            if "question id used as call id" in str(message.get("content") or "")
        ]
        assert repairs, "the binding repair never named the question ids"

    run(scenario())


# ---------------------------------------------------------------------------
# wrapper vocabulary never reaches the user
# ---------------------------------------------------------------------------

def test_a_sentence_that_names_the_wrapper_is_withheld_from_the_user(
    loop_factory,
) -> None:
    """The first meta leak seen live.

    The whole reply was the PA explaining the hold protocol to the person it is
    assisting, and explaining it wrong (`interaction_item_id` *is* accepted; the
    ids were invalid). A judge reading that transcript sees the benchmark's own
    scaffolding. The sentence is withheld, the rest of the paragraph is not, and
    the full text still reaches the audit plane.
    """
    async def scenario() -> None:
        leak = (
            "I see - the search calls didn't execute because the "
            "interaction_item_id parameters aren't part of the tool definitions."
        )
        provider = ScriptProvider([
            LLMResponse(content=f"{leak} I have the three slots you asked for."),
        ])
        loop, adapter = loop_factory(provider)

        outbound = await run_turn(loop, "Find me three slots.", active_ix=[])

        assert outbound.content == "I have the three slots you asked for."
        assert "interaction_item_id" not in outbound.content
        assert "tool definitions" not in outbound.content
        # Withheld, not lost: the audit plane keeps the sentence verbatim.
        narration = outbound.metadata["ix_render"]["held"]["narration"]
        assert [row["item_id"] for row in narration] == ["wrapper_vocabulary"]
        assert narration[0]["text"] == leak
        assert any(
            "wrapper vocabulary withheld" in step
            for step in outbound.metadata["ix_render"]["trace"]
        )
        # And memory never sees it either.
        assert leak not in adapter.events[0].messages[-1]["content"]

    run(scenario())


def test_a_reply_that_is_only_wrapper_talk_is_asked_again(loop_factory) -> None:
    """The two rules meet: filtering a turn empty is an empty turn.

    The live turn had nothing else in it. Withholding the sentence leaves the turn
    with no user-facing text, which is exactly the condition the empty-reply rule answers —
    one more request, and the template line only if that fails too.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(
                content="The tool definitions rejected the interaction_item_id field."
            ),
            LLMResponse(content="Here are the three slots: Tue 10:00, Wed 14:00, Fri 09:00."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(loop, "Find me three slots.", active_ix=[])

        assert outbound.content == (
            "Here are the three slots: Tue 10:00, Wed 14:00, Fri 09:00."
        )
        assert len(provider.requests) == 2

    run(scenario())


# ---------------------------------------------------------------------------
# T8 · the gate stops before the first open item, and stays stopped
# ---------------------------------------------------------------------------

def test_the_reads_after_an_open_gate_are_not_run_either(loop_factory) -> None:
    """T8 on the tool log: the turn ends at the gate."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Working.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                    {"id": "a3", "bound_item_ids": ["i3"]},
                ),
                task_call("r1", "task_read", "i1"),
                task_call("s1", "task_send", "i2", ask="Send it?"),
                task_call("r2", "task_read", "i3"),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_read", "task_send")

        outbound = await run_turn(
            loop, "Read, send, then read again.", active_ix=["autonomy_level"],
            action_types={"task_read": "read", "task_send": "external_action"},
        )

        assert executed_tools(outbound) == ["task_read"]
        held = outbound.metadata["ix_render"]["held"]["calls"]
        # The send *and* the read behind it: both are recorded under the one
        # gate that stopped the turn, so the audit says why in one place.
        assert [item["tool"] for item in held] == ["task_send", "task_read"]
        assert [item["attribute"] for item in held] == ["autonomy_level"] * 2
        # …but each row keeps the gate's own reason for it, so the audit can
        # still tell the call that opened the gate from the one behind it.
        assert [item["reason"] for item in held] == [
            "external_action_floor", "after_open_gate",
        ]
        assert outbound.metadata["ix_render"]["gate"]["item_id"] == "i2"

    run(scenario())


# ---------------------------------------------------------------------------
# T10 · questions before work
# ---------------------------------------------------------------------------

def test_questions_and_a_task_call_in_one_response_run_no_task(loop_factory) -> None:
    """T10: under `Structured` the asking turn does no work."""
    async def scenario() -> None:
        # The gate ends the turn, so the question prose has to ride along with
        # the response that carried the task call.
        provider = ScriptProvider([
            LLMResponse(
                content="Before I book: [[q:q1]] which date works?",
                tool_calls=[
                    selection("information_elicitation", "Structured"),
                    ix(
                        "IX_render_information_elicitation",
                        elicitation_id="el1",
                        questions=[{"id": "q1", "text": "Which date?", "about": "date"}],
                        assumptions=[],
                    ),
                    task_call("book_1", "task_book", "i1"),
                ],
            ),
        ])
        loop, _ = loop_factory(provider, "task_book")

        outbound = await run_turn(
            loop, "Book something.", active_ix=["information_elicitation"],
            action_types={"task_book": "external_action"},
        )

        assert executed_tools(outbound) == []
        assert "which date works?" in outbound.content
        assert [
            item["status"] for item in outbound.metadata["ix_render"]["held"]["calls"]
        ] == ["held_for_questions"]

    run(scenario())


# ---------------------------------------------------------------------------
# per-attribute render paths, end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "setting,notices,completion",
    [
        ("Silent", 0, False),
        ("Bookend", 1, True),
        ("Full narration", 2, True),
    ],
)
def test_process_visibility_notices_and_completion_end_to_end(
    loop_factory, setting: str, notices: int, completion: bool
) -> None:
    """process_visibility through the loop: the notices are the PA's own fields."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("process_visibility", setting),
                ix("IX_render_process_visibility",
                   completion_message="Monday morning is in the calendar."),
                task_call("c1", "task_search", "", notice="Let me look for a free slot."),
                task_call("c2", "task_create", "", notice="Putting the meeting in now."),
            ]),
            LLMResponse(content="Monday 10:00 is booked."),
        ])
        loop, _ = loop_factory(provider, "task_search", "task_create")

        outbound = await run_turn(
            loop, "Book Monday morning.", active_ix=["process_visibility"],
            action_types={"task_search": "read", "task_create": "internal_write"},
        )

        assert outbound.content.count("Let me look for a free slot.") == (
            1 if notices >= 1 else 0
        )
        assert outbound.content.count("Putting the meeting in now.") == (
            1 if notices >= 2 else 0
        )
        assert ("Monday morning is in the calendar." in outbound.content) is completion
        assert "Monday 10:00 is booked." in outbound.content
        # The audit plane always has both notices, whatever was released.
        assert len(outbound.metadata["ix_render"]["notices"]) == 2

    run(scenario())


OPENING_LINE = "Checking the calendar for a free Monday slot first."


def _narrating_turn_without_notice(
    setting: str, *replies: str | None, task: bool = True,
) -> ScriptProvider:
    """A `setting` turn whose task call carries no `_ix_notice`, then `replies`."""
    calls = [
        selection("process_visibility", setting),
        ix("IX_render_process_visibility", completion_message="All set."),
    ]
    if task:
        calls.append(task_call("c1", "task_search", ""))
    return ScriptProvider([
        LLMResponse(content=None, tool_calls=calls),
        LLMResponse(content="Monday 10:00 is free."),
        *[LLMResponse(content=reply) for reply in replies],
    ])


def _internal_prompts(provider: ScriptProvider) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in provider.requests[-1]
        if message.get("_mempa_internal") is True
    ]


@pytest.mark.parametrize("setting", ["Bookend", "Full narration"])
def test_a_narrating_turn_without_a_notice_is_asked_for_its_opening_line(
    loop_factory, setting: str
) -> None:
    """acc_003 t=3.

    The PA selected Bookend, made its lookups, wrote no `_ix_notice`, and the
    user read a reply with no opening line. The wrapper now asks for the line
    and puts the reply where the first task call's notice would have been, so
    the setting the PA chose is the setting the user sees.
    """
    async def scenario() -> None:
        provider = _narrating_turn_without_notice(setting, OPENING_LINE)
        loop, _ = loop_factory(provider, "task_search")

        outbound = await run_turn(
            loop, "Book Monday morning.", active_ix=["process_visibility"],
            action_types={"task_search": "read"},
        )

        assert IX_MISSING_NOTICE_PROMPT in _internal_prompts(provider)
        assert outbound.content.startswith(OPENING_LINE)
        assert "Monday 10:00 is free." in outbound.content
        assert outbound.content.rstrip().endswith("All set.")
        assert [
            row["text"] for row in outbound.metadata["ix_render"]["notices"]
        ] == [OPENING_LINE]
        event = next(
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_notice_repaired"
        )
        assert event["attempts"] == 1
        assert len(provider.requests) == 3

    run(scenario())


def test_the_opening_line_is_asked_for_twice_before_the_turn_fails(
    loop_factory,
) -> None:
    async def scenario() -> None:
        provider = _narrating_turn_without_notice("Bookend", "", OPENING_LINE)
        loop, _ = loop_factory(provider, "task_search")

        outbound = await run_turn(
            loop, "Book Monday morning.", active_ix=["process_visibility"],
            action_types={"task_search": "read"},
        )

        assert outbound.content.startswith(OPENING_LINE)
        event = next(
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_notice_repaired"
        )
        assert event["attempts"] == 2
        assert len(provider.requests) == 4

    run(scenario())


def test_a_narrating_turn_still_without_a_notice_fails_the_step(
    loop_factory,
) -> None:
    """No third ask: the turn fails the IX contract like a missing payload.

    Nothing is persisted, so the accumulation step is retried from a clean
    state rather than carrying a narrating turn that did not narrate.
    """
    async def scenario() -> None:
        provider = _narrating_turn_without_notice("Bookend", "", "")
        loop, adapter = loop_factory(provider, "task_search")
        before_messages = list(loop.sessions.get(SESSION_KEY).messages)

        with pytest.raises(AgentProviderError, match="IX notice contract"):
            await run_turn(
                loop, "Book Monday morning.", active_ix=["process_visibility"],
                action_types={"task_search": "read"},
            )

        assert len(provider.requests) == 4
        assert loop.sessions.get(SESSION_KEY).messages == before_messages
        assert adapter.events == []

    run(scenario())


@pytest.mark.parametrize(
    "setting,task",
    [("Silent", True), ("Bookend", False)],
    ids=["silent-turn", "bookend-without-task-call"],
)
def test_no_opening_line_is_asked_for_where_none_would_show(
    loop_factory, setting: str, task: bool
) -> None:
    """Silent shows no notice, and a zero-call turn has nothing to open."""
    async def scenario() -> None:
        provider = _narrating_turn_without_notice(setting, task=task)
        loop, _ = loop_factory(provider, "task_search")

        outbound = await run_turn(
            loop, "Book Monday morning.", active_ix=["process_visibility"],
            action_types={"task_search": "read"},
        )

        assert IX_MISSING_NOTICE_PROMPT not in _internal_prompts(provider)
        assert "Monday 10:00 is free." in outbound.content
        assert "ix_notice_repaired" not in [
            item.get("name") for item in outbound.metadata["tool_events"]
        ]

    run(scenario())


def test_a_zero_call_turn_holds_the_closing_line(loop_factory) -> None:
    """acc_003: no task call, no closing line.

    Three text-only turns each shipped the PA's own closing prose and then the
    `completion_message` again; a fourth (t=20) shipped the closing
    line over no work at all. The field is still recorded for the audit plane.
    """
    async def scenario() -> None:
        provider = _narrating_turn_without_notice("Bookend", task=False)
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Keep the rest of the notes to this standard.",
            active_ix=["process_visibility"],
        )

        assert outbound.content == "Monday 10:00 is free."
        completion = outbound.metadata["ix_render"]["completion"]
        assert completion["text"] == "All set."
        assert completion["released"] is False

    run(scenario())


COMPLETION_LINE = "Monday morning is in the calendar."
COMPLETION_GATE_ASK = "Shall I send Dale the invitation?"


@pytest.mark.parametrize(
    "setting,gated",
    [
        ("Bookend", True),
        ("Bookend", False),
        ("Silent", False),
        ("Full narration", True),
    ],
    ids=["bookend-gate", "bookend-no-gate", "silent-no-gate", "full-narration-gate"],
)
def test_completion_released_reports_what_the_reply_carried(
    loop_factory, setting: str, gated: bool
) -> None:
    """`completion.released` is a statement about the delivered reply.

    `loop.py` used to re-derive it from the setting plus a non-empty
    `completion_message`, while the renderer writes the closing line only in the
    branch where no gate fired. A `Bookend` gate turn therefore recorded
    `released: true` for a reply whose last sentence was the gate's ask, and a
    judge reading the blob saw a closing line that never went out.
    """
    async def scenario() -> None:
        calls = [
            selection("process_visibility", setting),
            ix("IX_render_process_visibility", completion_message=COMPLETION_LINE),
        ]
        active_ix = ["process_visibility"]
        action_types = {"task_search": "read"}
        if gated:
            calls = [
                selection("autonomy_level", "Self-directed"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"], "is_key": True}),
                task_call("s1", "task_send", "i1", ask=COMPLETION_GATE_ASK),
                *calls,
            ]
            active_ix = ["autonomy_level", *active_ix]
            action_types["task_send"] = "external_action"
        else:
            # The closing line is only released over a task call; the
            # ungated turn does one read.
            calls = [
                task_call("r1", "task_search", "", notice="Checking Monday first."),
                *calls,
            ]

        provider = ScriptProvider([
            LLMResponse(content="Monday 10:00 is free.", tool_calls=calls),
            LLMResponse(content="Monday 10:00 is free."),
        ])
        loop, _ = loop_factory(provider, "task_search", "task_send")

        outbound = await run_turn(
            loop, "Book Monday morning.", active_ix=active_ix,
            action_types=action_types,
        )

        completion = outbound.metadata["ix_render"]["completion"]
        # The line the PA wrote is kept either way; only `released` moves.
        assert completion["text"] == COMPLETION_LINE
        if gated:
            assert outbound.content.rstrip().endswith(COMPLETION_GATE_ASK)
            assert COMPLETION_LINE not in outbound.content
            assert completion["released"] is False
            assert completion["skipped"] == "gate"
        else:
            released = setting != "Silent"
            assert (COMPLETION_LINE in outbound.content) is released
            assert completion["released"] is released
            # Nothing suppressed the line: `released` alone tells the story.
            assert "skipped" not in completion

    run(scenario())


def test_a_render_payload_missing_after_a_forced_call_fails_the_step(loop_factory) -> None:
    """Fail-closed: an unrenderable turn must not be delivered anyway.

    Before this rule the wrapper reminded once and then shipped the turn with
    `completion.released == false`, so under `Bookend` the closing sentence
    simply vanished and the judge read that as the PA's own choice. Now the
    miss ends the turn the way `IX_capability_resolve` does: no scorable reply,
    nothing persisted, the step retryable from a clean state — but only after
    the third attempt pins the tool with `tool_choice`, because a model that
    ignores a reminder twice may still answer a forced call.
    """
    async def scenario() -> None:
        no_payload = LLMResponse(content="All done.")
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("process_visibility", "Bookend"),
                task_call("c1", "task_search", "", notice="Let me look for a free slot."),
            ]),
            no_payload,
            no_payload,
            no_payload,
        ])
        loop, adapter = loop_factory(provider, "task_search")
        before_messages = list(loop.sessions.get(SESSION_KEY).messages)

        with pytest.raises(AgentProviderError, match="IX render contract"):
            await run_turn(
                loop, "Book Monday morning.", active_ix=["process_visibility"],
                action_types={"task_search": "read"},
            )

        assert loop.sessions.get(SESSION_KEY).messages == before_messages
        assert adapter.events == []
        assert provider.tool_choices == [
            None,
            None,
            None,
            {"type": "function", "function": {"name": "IX_render_process_visibility"}},
        ]

    run(scenario())


def test_a_forced_render_payload_saves_the_turn(loop_factory) -> None:
    """The third attempt is the point: a model that ignores both reminders but
    honours `tool_choice` finishes the turn instead of failing the step.
    """
    async def scenario() -> None:
        class _IgnoresReminders(ScriptProvider):
            """Answers with prose until the payload tool is forced on it."""

            async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                           temperature=0.7, reasoning_effort=None, tool_choice=None):
                scripted = await super().chat(
                    messages, tools=tools, model=model, max_tokens=max_tokens,
                    temperature=temperature, reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice,
                )
                if not isinstance(tool_choice, dict):
                    return scripted
                return LLMResponse(content=None, tool_calls=[
                    ix(tool_choice["function"]["name"],
                       completion_message="Monday morning is in the calendar."),
                ])

        provider = _IgnoresReminders([
            LLMResponse(content=None, tool_calls=[
                selection("process_visibility", "Bookend"),
                task_call("c1", "task_search", "", notice="Let me look for a free slot."),
            ]),
        ] + [LLMResponse(content="All done.")] * 6)
        loop, _ = loop_factory(provider, "task_search")

        outbound = await run_turn(
            loop, "Book Monday morning.", active_ix=["process_visibility"],
            action_types={"task_search": "read"},
        )

        assert "Monday morning is in the calendar." in outbound.content
        assert outbound.metadata["ix_render"]["completion"]["released"] is True
        forced = [
            event for event in outbound.metadata["tool_events"]
            if event.get("name") == "ix_render_payload_forced"
        ]
        assert [event["tool"] for event in forced] == [
            "IX_render_process_visibility"
        ]
        assert not [
            event for event in outbound.metadata["tool_events"]
            if event.get("name") == "ix_render_payload_missing"
        ]

    run(scenario())


def test_a_render_payload_missing_once_then_supplied_releases_the_completion(
    loop_factory,
) -> None:
    """The single reminder still works: one miss is a nudge, not a failure."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("process_visibility", "Bookend"),
                task_call("c1", "task_search", "", notice="Let me look for a free slot."),
            ]),
            LLMResponse(content="All done."),
            LLMResponse(content=None, tool_calls=[
                ix("IX_render_process_visibility",
                   completion_message="Monday morning is in the calendar."),
            ]),
            LLMResponse(content="Monday 10:00 is booked."),
        ])
        loop, _ = loop_factory(provider, "task_search")

        outbound = await run_turn(
            loop, "Book Monday morning.", active_ix=["process_visibility"],
            action_types={"task_search": "read"},
        )

        assert outbound.metadata["ix_render"]["completion"]["released"] is True
        assert "Monday morning is in the calendar." in outbound.content
        assert "Let me look for a free slot." in outbound.content
        assert "Monday 10:00 is booked." in outbound.content

    run(scenario())


# ---------------------------------------------------------------------------
# the selection tools ride the same ladder as the render payloads
#
# `tone_formality` is a pass-through attribute, so the turn owes exactly one IX
# tool: the selection. Before this the ladder skipped `IX_<attribute>` at both
# ends -- not forcible, not fatal -- and a turn that owed only a selection spent
# all 40 iterations being nudged and then shipped the unset fallback.
# ---------------------------------------------------------------------------

def test_a_selection_never_made_after_a_forced_call_fails_the_step(loop_factory) -> None:
    """Nudge, pin, fail: an exposed attribute has no unset surface to render
    under, so an unmade selection ends the turn the way a missing render
    payload does -- never on `max_iterations`."""
    async def scenario() -> None:
        no_selection = LLMResponse(content="Roughly eleven.")
        provider = ScriptProvider([no_selection] * 3)
        loop, adapter = loop_factory(provider)
        before_messages = list(loop.sessions.get(SESSION_KEY).messages)

        with pytest.raises(AgentProviderError, match="IX render contract"):
            await run_turn(loop, "What time is it?", active_ix=["tone_formality"])

        assert loop.sessions.get(SESSION_KEY).messages == before_messages
        assert adapter.events == []
        # Exactly three requests: the original, the nudged retry, the pinned one.
        assert provider.tool_choices == [
            None,
            None,
            {"type": "function", "function": {"name": "IX_tone_formality"}},
        ]

    run(scenario())


def test_a_forced_selection_saves_the_turn(loop_factory) -> None:
    """A model that ignores the nudge but honours `tool_choice` finishes the
    turn, and the pin is on the audit plane under `ix_render_payload_forced`."""
    async def scenario() -> None:
        class _IgnoresReminders(ScriptProvider):
            """Answers with prose until the selection tool is forced on it."""

            async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                           temperature=0.7, reasoning_effort=None, tool_choice=None):
                scripted = await super().chat(
                    messages, tools=tools, model=model, max_tokens=max_tokens,
                    temperature=temperature, reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice,
                )
                if not isinstance(tool_choice, dict):
                    return scripted
                return LLMResponse(content=None, tool_calls=[
                    selection("tone_formality", "Casual"),
                ])

        # Three prose refusals (the third is what the pinned request replaces),
        # then the answer the turn ends on.
        provider = _IgnoresReminders(
            [LLMResponse(content="Roughly eleven.")] * 3
            + [LLMResponse(content="It is about eleven.")]
        )
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "What time is it?", active_ix=["tone_formality"],
        )

        assert "It is about eleven." in outbound.content
        forced = [
            event for event in outbound.metadata["tool_events"]
            if event.get("name") == "ix_render_payload_forced"
        ]
        assert [event["tool"] for event in forced] == ["IX_tone_formality"]
        assert not [
            event for event in outbound.metadata["tool_events"]
            if event.get("name") == "ix_render_payload_missing"
        ]

    run(scenario())


def test_a_selection_is_not_pinned_when_the_policy_forbids_forcing(loop_factory) -> None:
    """No pin to spend: the turn is reminded again, unpinned, and fails on the
    third miss -- an unavailable pin is not an early exhaustion."""
    async def scenario() -> None:
        class _NoForcedChoice(ScriptProvider):
            def supports_forced_tool_choice(self, model: str | None = None) -> bool:
                return False

        provider = _NoForcedChoice([LLMResponse(content="Roughly eleven.")] * 3)
        loop, adapter = loop_factory(provider)

        with pytest.raises(AgentProviderError, match="IX render contract"):
            await run_turn(loop, "What time is it?", active_ix=["tone_formality"])

        assert adapter.events == []
        assert provider.tool_choices == [None, None, None]

    run(scenario())


#: acc_011 t=27, shortened. Under `Show` this printed ahead of the answer, so it
#: was the first thing the reader saw — being talked about in the third person,
#: two lines above "Done! ... has been saved to your desktop".
WORK_LOG_RATIONALE = (
    "The user approved the prioritization logic and asked me to save the "
    "document. I created clinic_prep_plan.md on the desktop."
)
#: What the field is for: the reasons, written to the reader.
REASONS_RATIONALE = (
    "The refill route is the only errand with a hard closing time, so it "
    "anchors the order and the pharmacy note sits with it."
)


def _rationale_turn(*rationales: str) -> ScriptProvider:
    """One payload per rationale, then the answer the turn ends on."""
    return ScriptProvider([
        *(
            LLMResponse(content=None, tool_calls=[
                selection("reasoning_visibility", "Show"),
                ix("IX_render_reasoning_visibility",
                   user_facing_rationale=rationale),
            ])
            for rationale in rationales
        ),
        LLMResponse(content="Done! The file is on your desktop."),
    ])


def test_a_rationale_written_about_the_user_is_asked_for_again(
    loop_factory,
) -> None:
    """acc_011 t=27 — the reply opened by talking about the reader.

    Wording fixed this for a while and then lost it, and no counter
    could see it: the call was well-formed, so `ix_contract_failure` stayed 0.
    The wrapper asks; it never writes the reasons.
    """
    async def scenario() -> None:
        provider = _rationale_turn(WORK_LOG_RATIONALE, REASONS_RATIONALE)
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Save it to my desktop.", active_ix=["reasoning_visibility"],
        )

        assert IX_RATIONALE_WORK_LOG_REPAIR in _internal(provider.requests[-1])
        assert "The user approved" not in outbound.content
        assert REASONS_RATIONALE in outbound.content
        assert "ix_rationale_work_log" in [
            event.get("name") for event in outbound.metadata["tool_events"]
        ]

    run(scenario())


def test_a_rationale_written_to_the_user_is_left_alone(loop_factory) -> None:
    """The guard: reasons in the second person are the field working."""
    async def scenario() -> None:
        provider = _rationale_turn(REASONS_RATIONALE)
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Save it to my desktop.", active_ix=["reasoning_visibility"],
        )

        assert REASONS_RATIONALE in outbound.content
        assert IX_RATIONALE_WORK_LOG_REPAIR not in _internal(provider.requests[-1])
        assert "ix_rationale_work_log" not in [
            event.get("name") for event in outbound.metadata["tool_events"]
        ]

    run(scenario())


@pytest.mark.parametrize(
    "setting,shown", [("Show", True), ("Hide", False)],
)
def test_reasoning_visibility_end_to_end(loop_factory, setting: str, shown: bool) -> None:
    """Only the PA's own `user_facing_rationale` can ever be released."""
    async def scenario() -> None:
        rationale = "Only em_0100 mentions 4B: its stack id matches the batch you gave me."
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("reasoning_visibility", setting),
                ix("IX_render_reasoning_visibility", user_facing_rationale=rationale),
            ]),
            LLMResponse(
                content="It is the first of the two emails.",
                reasoning_content="<think>maybe the other one</think>",
            ),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Which email did you mean?", active_ix=["reasoning_visibility"],
        )

        assert (rationale in outbound.content) is shown
        assert "It is the first of the two emails." in outbound.content
        assert "maybe the other one" not in outbound.content
        reasoning = outbound.metadata["ix_render"]["reasoning"]
        assert reasoning["shown"] is shown
        assert reasoning["source"] == "user_facing_rationale"
        assert reasoning["provider_reasoning_present"] is True

    run(scenario())


@pytest.mark.parametrize(
    "setting,expected",
    [
        ("Express", ["(confidence: low)", "(confidence: high)"]),
        ("Hide", []),
    ],
)
def test_uncertainty_expression_end_to_end(
    loop_factory, setting: str, expected: list[str]
) -> None:
    """uncertainty_expression: the parenthesis is the renderer's, not the prose's."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", setting),
                ix("IX_render_uncertainty_expression", certainties=[
                    # `claim` is required by the frozen schema even on the
                    # marker path; the marker still wins where both exist.
                    {"ref": "c1", "level": "low", "claim": "The invoice is unpaid"},
                    {"ref": "c2", "level": "high", "claim": "the flight is confirmed"},
                ]),
            ]),
            LLMResponse(
                content="The invoice is unpaid[[c:c1]] and the flight is confirmed[[c:c2]].",
            ),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        for text in expected:
            assert text in outbound.content
        for text in ("(confidence: low)", "(confidence: high)", "(confidence: medium)"):
            if text not in expected:
                assert text not in outbound.content
        assert "[[c:" not in outbound.content

    run(scenario())


@pytest.mark.parametrize(
    "setting,expected",
    [
        ("Express", ["The invoice is unpaid(confidence: low)", "confirmed(confidence: high)"]),
        ("Hide", []),
    ],
)
def test_uncertainty_expression_annotates_by_claim(
    loop_factory, setting: str, expected: list[str]
) -> None:
    """The production channel is `claim`, not a marker the model cannot place.

    The render tool call is emitted in the iteration *before* the prose exists,
    so a `[[c:ref]]` marker cannot be cross-referenced against text that has
    not been written; two live rounds produced zero markers. The model instead
    commits to a span it will write, and the renderer finds it.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", setting),
                ix("IX_render_uncertainty_expression", certainties=[
                    {"ref": "c1", "level": "low", "claim": "The invoice is unpaid"},
                    {"ref": "c2", "level": "high", "claim": "the flight is confirmed"},
                ]),
            ]),
            LLMResponse(
                content="The invoice is unpaid and the flight is confirmed.",
            ),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        for text in expected:
            assert text in outbound.content
        if not expected:
            assert outbound.content.strip() == (
                "The invoice is unpaid and the flight is confirmed."
            )
        assert outbound.content.count("(confidence: ") == len(expected)
        rows = outbound.metadata["ix_render"]["certainties"]
        assert [row["ref"] for row in rows] == ["c1", "c2"]

    run(scenario())


def test_a_claim_the_reply_never_makes_is_traced_not_rendered(loop_factory) -> None:
    """`Express` with an orphaned claim: no parenthetical, no payload miss.

    Its companion `c2` is the reply's own words, so the turn still renders the
    attribute and the all-dropped rule below does not take it -- the subject
    here is one orphan beside a placed claim, which is the shape that ships.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Express"),
                ix("IX_render_uncertainty_expression", certainties=[
                    {"ref": "c1", "level": "high",
                     "claim": "Farid's answer qualifies for full marks"},
                    {"ref": "c2", "level": "low", "claim": "I can update it"},
                ]),
            ]),
            LLMResponse(content="Once you have checked the paper copies, I can update it."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "What now?", active_ix=["uncertainty_expression"],
        )

        assert outbound.content.count("(confidence: ") == 1
        assert "I can update it(confidence: low)" in outbound.content
        blob = outbound.metadata["ix_render"]
        assert blob["certainties"][0] == {
            "ref": "c1", "level": "high", "rendered": "",
            "claim": "Farid's answer qualifies for full marks",
            "strategy": "not_found",
        }
        assert "certainty ref c1: claim not found" in blob["trace"]

    run(scenario())


# ---------------------------------------------------------------------------
# carrier F · the trailing `<certainties>` block
# ---------------------------------------------------------------------------

BLOCK_REPLY = (
    "The invoice is unpaid and the flight is confirmed.\n"
    "<certainties>\n"
    "- ref: c1 | level: low | claim: The invoice is unpaid\n"
    "- ref: c2 | level: high | claim: the flight is confirmed\n"
    "</certainties>"
)
BLOCK_PROSE = "The invoice is unpaid and the flight is confirmed."


def _block_provider(setting: str, reply: str = BLOCK_REPLY) -> ScriptProvider:
    """One turn: choose the setting, then answer with a block-terminated reply.

    No `IX_render_uncertainty_expression` call anywhere — the block is the
    carrier, and the per-turn tool is the fallback it replaces.
    """
    return ScriptProvider([
        LLMResponse(content=None, tool_calls=[selection("uncertainty_expression", setting)]),
        LLMResponse(content=reply),
    ])


@pytest.mark.parametrize(
    "setting,expected",
    [
        ("Express", ["The invoice is unpaid(confidence: low)", "confirmed(confidence: high)"]),
        ("Hide", []),
    ],
)
def test_the_trailing_block_carries_the_certainties(
    loop_factory, setting: str, expected: list[str]
) -> None:
    """The tool call cannot quote prose that does not exist yet.

    The block is written *after* the answer, so the span it names is text the
    model has already produced — which is what makes the claim locatable at all.
    """
    async def scenario() -> None:
        loop, _ = loop_factory(_block_provider(setting))

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        for text in expected:
            assert text in outbound.content
        assert outbound.content.count("(confidence: ") == len(expected)
        if not expected:
            assert outbound.content.strip() == BLOCK_PROSE
        rows = outbound.metadata["ix_render"]["certainties"]
        assert [row["ref"] for row in rows] == ["c1", "c2"]

    run(scenario())


def test_the_block_reaches_no_projection_but_the_audit_one(loop_factory) -> None:
    """The block is wrapper bookkeeping: `public_turn`, the history the next

    provider call sees, the memory event and the setting-blind judge pack must
    all be free of it. Only `audit_trace` / the provider trace keep it.
    """
    async def scenario() -> None:
        loop, adapter = loop_factory(_block_provider("Express"))

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        pack = build_text_judge_pack([
            {"t": 0, "kind": "simulator_turn", "beat": "b1", "message": "Where?"},
            {
                "t": 1,
                "kind": "pa_turn",
                "beat": "b1",
                "outbound": outbound.content,
                "tool_events": outbound.metadata["tool_events"],
                "ix_render": outbound.metadata["ix_render"],
            },
        ])
        history = loop.sessions.get(SESSION_KEY).messages
        memory = [event.messages for event in adapter.events]

        for needle in ("<certainties>", "</certainties>", "ref: c1", "level: high"):
            assert needle not in outbound.content
            assert needle not in json.dumps(history, ensure_ascii=False)
            assert needle not in json.dumps(memory, default=str, ensure_ascii=False)
            assert needle not in json.dumps(pack, default=str, ensure_ascii=False)
        # The last assistant message the next provider call replays is the
        # rendered turn, verbatim and block-free.
        assert history[-1]["content"] == outbound.content

    run(scenario())


def test_hide_withholds_a_located_block_claim_and_says_which(loop_factory) -> None:
    """`Hide` has to be falsifiable from `ix_render` alone.

    "Located every claim and suppressed it" and "found none" used to read the
    same — empty `rendered` on every row and a `0/N` summary line.
    """
    async def scenario() -> None:
        loop, _ = loop_factory(_block_provider("Hide"))

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        blob = outbound.metadata["ix_render"]
        assert "(confidence: " not in outbound.content
        assert [row["rendered"] for row in blob["certainties"]] == ["", ""]
        for ref in ("c1", "c2"):
            assert f"certainty ref {ref}: located (exact), withheld (Hide)" in blob["trace"]
        assert not [line for line in blob["trace"] if "claim not found" in line]
        # …and the row itself says so, not only the trace: `strategy` is how the
        # claim was located, `claim` is what was committed to (certainties[]).
        assert [(row["claim"], row["strategy"]) for row in blob["certainties"]] == [
            ("The invoice is unpaid", "exact"),
            ("the flight is confirmed", "exact"),
        ]

    run(scenario())


def test_the_block_itself_is_kept_verbatim_on_the_audit_plane(loop_factory) -> None:
    """`raw_blocks`: the carrier the renderer strips has to survive somewhere.

    Every delivered surface drops the block, so without this the only record of
    what the model committed to would be gone — and `AuditTrace.segments`, the
    field the plan names for it, is never constructed by anything.
    """
    async def scenario() -> None:
        loop, _ = loop_factory(_block_provider("Hide"))

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        blocks = outbound.metadata["ix_render"]["raw_blocks"]
        assert [block["kind"] for block in blocks] == ["certainties"]
        assert blocks[0]["text"].startswith("<certainties>")
        assert "ref: c1 | level: low | claim: The invoice is unpaid" in blocks[0]["text"]
        # It is held content: the text face must not see a byte of it.
        pack = build_text_judge_pack([
            {"t": 0, "kind": "simulator_turn", "beat": "b1", "message": "Where?"},
            {
                "t": 1,
                "kind": "pa_turn",
                "beat": "b1",
                "outbound": outbound.content,
                "tool_events": outbound.metadata["tool_events"],
                "ix_render": outbound.metadata["ix_render"],
            },
        ])
        assert "raw_blocks" not in json.dumps(pack, default=str)
        assert "<certainties>" not in json.dumps(pack, default=str)

    run(scenario())


def test_a_turn_without_a_block_writes_no_raw_blocks_key(loop_factory) -> None:
    """`raw_blocks` is optional, so a blockless turn's record is unchanged."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Express"),
                ix("IX_render_uncertainty_expression", certainties=[]),
            ]),
            LLMResponse(content="The invoice is unpaid."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        assert "raw_blocks" not in outbound.metadata["ix_render"]

    run(scenario())


def test_an_empty_block_is_a_valid_payload_for_the_turn(loop_factory) -> None:
    """No claim's confidence matters this turn — that is an answer, not a miss."""
    async def scenario() -> None:
        loop, _ = loop_factory(
            _block_provider("Express", "Both are settled.\n<certainties></certainties>")
        )

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        assert outbound.content.strip() == "Both are settled."
        assert outbound.metadata["ix_render"]["certainties"] == []
        assert not [event for event in _ix_events(outbound) if "missing" in event["name"]]

    run(scenario())


def test_a_block_replaces_the_tool_payload_it_contradicts(loop_factory) -> None:
    """Both carriers on one turn: the block wins, because it quotes real prose.

    The tool call is emitted an iteration earlier and can only promise a span;
    live, three of three such promises came back as paraphrases of markdown the
    model had not written yet.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Express"),
                ix("IX_render_uncertainty_expression", certainties=[
                    {"ref": "promised", "level": "high",
                     "claim": "Both items are fully settled"},
                ]),
            ]),
            LLMResponse(content=BLOCK_REPLY),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        rows = outbound.metadata["ix_render"]["certainties"]
        assert [row["ref"] for row in rows] == ["c1", "c2"]
        assert "The invoice is unpaid(confidence: low)" in outbound.content

    run(scenario())


def test_the_tool_payload_still_carries_a_turn_that_writes_no_block(loop_factory) -> None:
    """The fallback is a fallback, not dead code: no block, tool payload applies."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Express"),
                ix("IX_render_uncertainty_expression", certainties=[
                    {"ref": "c1", "level": "low", "claim": "The invoice is unpaid"},
                ]),
            ]),
            LLMResponse(content=BLOCK_PROSE),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        assert "The invoice is unpaid(confidence: low)" in outbound.content
        assert [row["ref"] for row in outbound.metadata["ix_render"]["certainties"]] == ["c1"]

    run(scenario())


def test_neither_carrier_on_an_active_turn_still_fails_the_contract(loop_factory) -> None:
    """One reminder, one forced call, then no scorable answer at all."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Express"),
            ]),
            LLMResponse(content=BLOCK_PROSE),
            LLMResponse(content=BLOCK_PROSE),
            LLMResponse(content=BLOCK_PROSE),
        ])
        loop, _ = loop_factory(provider)

        with pytest.raises(AgentProviderError):
            await run_turn(
                loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
            )
        # One reminder in the whole turn: the last request carries every
        # internal message the turn produced, so counting there counts sends.
        reminder = [
            message for message in provider.requests[-1]
            if IX_PER_TURN_REMINDER_PREFIX in str(message.get("content") or "")
        ]
        assert len(reminder) == 1
        assert provider.tool_choices[-1] == {
            "type": "function",
            "function": {"name": "IX_render_uncertainty_expression"},
        }

    run(scenario())


def test_the_per_turn_reminder_asks_for_the_block_setting_blind(loop_factory) -> None:
    """The carrier has to survive to the last line of the answer, so it

    is stated twice: once on the tool, once in the per-turn reminder. The text
    is the same one under every setting.
    """
    async def scenario() -> None:
        loop, _ = loop_factory(_block_provider("Hide"))

        await run_turn(
            loop, "Where do the two stand?", active_ix=["uncertainty_expression"],
        )

        injected = [
            str(message.get("content") or "")
            for message in _provider_requests(loop)
            if "<certainties>" in str(message.get("content") or "")
        ]
        assert injected, "the block instruction never reached the model"
        for setting in ("Express", "Hide"):
            assert setting not in injected[0]

    run(scenario())


@pytest.mark.parametrize(
    "setting,first_turn,second_turn",
    [
        ("Organize", ["Flights", "Hotel", "Invoice"], []),
        ("One-at-a-time", ["Flights"], ["Hotel"]),
    ],
)
def test_topic_management_across_two_turns(
    loop_factory, setting: str, first_turn: list[str], second_turn: list[str]
) -> None:
    """`Organize` finishes in one turn, `One-at-a-time` takes the next one."""
    async def scenario() -> None:
        def payload(*batch_ids: str):
            return ix(
                "IX_render_topic_management",
                batch_set_id="bs1",
                item_count=len(batch_ids),
                batches=[
                    {"batch_id": bid, "label": bid, "body": "", "call_binding": []}
                    for bid in batch_ids
                ],
            )

        def prose(*labels: tuple[str, str]) -> str:
            body = "\n\n".join(
                f"[[batch:{bid}]] {label}: settled." for bid, label in labels
            )
            return f"Here is where they stand.\n\n{body}"

        all_three = (("b1", "Flights"), ("b2", "Hotel"), ("b3", "Invoice"))
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", setting), payload("b1", "b2", "b3"),
            ]),
            LLMResponse(content=prose(*all_three)),
            LLMResponse(content=None, tool_calls=[
                payload("b2", "b3"),
                ix("IX_continue_topic_management", decisions=[
                    {"id": "b2", "decision": "release", "user_evidence": "go on"},
                    {"id": "b3", "decision": "keep", "user_evidence": ""},
                ]),
            ]),
            LLMResponse(content=prose(*all_three[1:])),
            LLMResponse(content="Nothing else pending."),
        ])
        loop, _ = loop_factory(provider)

        first = await run_turn(
            loop, "Where are we on the trip?", active_ix=["topic_management"],
        )
        for label in ("Flights", "Hotel", "Invoice"):
            assert (label in first.content) is (label in first_turn)
        assert "Here is where they stand." in first.content

        if not second_turn:
            return
        second = await run_turn(loop, "Go on.", active_ix=["topic_management"])
        for label in ("Hotel", "Invoice"):
            assert (label in second.content) is (label in second_turn)

    run(scenario())


@pytest.mark.parametrize(
    "setting,released,gated",
    [
        # `Infer` asks nothing this turn, so there is no question to work
        # behind: it opens no gate and the call runs.
        ("Infer", [], False),
        ("Structured", ["airport", "Aisle", "card"], True),
        ("Iterative", ["airport"], True),
    ],
)
def test_information_elicitation_release_and_questions_gate(
    loop_factory, setting: str, released: list[str], gated: bool
) -> None:
    """Which questions ship, and whether the turn may also do work.

    The gate is on `Structured` / `Iterative` — the two settings that put a
    question in front of the user this turn. `Infer` asks nothing, and the
    unset fallback opens no gate at all.
    """
    async def scenario() -> None:
        prose = (
            "I can book it once I know three things.\n\n"
            "[[q:q1]] Which airport do you want?\n\n"
            "[[q:q2]] Aisle or window?\n\n"
            "[[q:q3]] Should I use the team card?"
        )
        provider = ScriptProvider([
            LLMResponse(content=prose, tool_calls=[
                selection("information_elicitation", setting),
                ix(
                    "IX_render_information_elicitation",
                    elicitation_id="el1",
                    questions=[
                        {"id": "q1", "text": "Which airport?", "about": "departure"},
                        {"id": "q2", "text": "Aisle or window?", "about": "seat"},
                        {"id": "q3", "text": "Which card?", "about": "payment"},
                    ],
                    assumptions=[],
                ),
                task_call("book_1", "task_book", "i1"),
            ]),
            # Only reached where no gate opens: the unset fallback lets the
            # call run, so the turn goes round once more for its answer.
            LLMResponse(content=prose),
        ])
        loop, _ = loop_factory(provider, "task_book")

        outbound = await run_turn(
            loop, "Book me the trip.", active_ix=["information_elicitation"],
            action_types={"task_book": "read"},
        )

        for needle in ("airport", "Aisle", "card"):
            assert (needle in outbound.content) is (needle in released)
        assert "[[q:" not in outbound.content
        assert executed_tools(outbound) == ([] if gated else ["task_book"])

    run(scenario())


def test_an_exposed_attribute_that_is_never_selected_never_renders(loop_factory) -> None:
    """The unset fallback is unreachable for an attribute exposed this session.

    With the neutral option gone the PA must commit to a
    named setting, so "no selection" is not an answer the model may give — it is
    only the state of an attribute this session never exposed. `IX_<attribute>`
    is `required == "always"`, and a turn that keeps refusing it fails the render
    contract instead of being rendered with the fallback.
    """
    async def scenario() -> None:
        # The model writes a batch marker and never calls `IX_topic_management`
        # or its render payload, however often it is asked.
        provider = ScriptProvider([
            LLMResponse(content="Morning.\n\n[[batch:b1]] Flights are open."),
        ] * 12)
        loop, _ = loop_factory(provider)

        with pytest.raises(AgentProviderError) as excinfo:
            await run_turn(
                loop, "How is the trip looking?", active_ix=["topic_management"],
            )

        assert "IX render contract" in str(excinfo.value)

    run(scenario())


@pytest.mark.parametrize(
    "setting,action,delivered", [("High", "send", True), ("Low", "suppress", False)],
)
def test_proactive_outreach_end_to_end(
    loop_factory, setting: str, action: str, delivered: bool
) -> None:
    """The rendered decision is what leaves the PA."""
    from datetime import datetime, timezone

    async def scenario() -> None:
        message = "The shop closes at six and your pull list is waiting."
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                ix(
                    "IX_proactive_outreach",
                    setting=setting,
                    evidence="the matrix says so",
                    application=f"{action} it",
                    outreach_action=action,
                    outreach_message=message,
                ),
            ]),
            LLMResponse(content="Internal completion text."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await loop.process_environment_event(
            trigger_id="pickup_deadline",
            source_type="deadline",
            prior_thread="The comic order is ready.",
            detail="The store closes in two hours.",
            user_impact="User A may want to collect it today.",
            session_key="benchmark:main",
            metadata={"run_id": "run_001", "persona_id": "user_a"},
            timestamp=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
        )

        assert outbound.content == (message if delivered else "")
        proactive = outbound.metadata["ix_render"]["proactive_decision"]
        assert proactive == {"action": action, "setting": setting, "message": message}

    run(scenario())


# ---------------------------------------------------------------------------
# the elicitation carrier can actually be reached
# ---------------------------------------------------------------------------

def _questions_payload(*questions: dict[str, str], elicitation_id: str = "el1") -> Any:
    return ix(
        "IX_render_information_elicitation",
        elicitation_id=elicitation_id,
        questions=list(questions),
        assumptions=[],
    )


QUESTIONS = (
    {"id": "q1", "text": "Which date?", "about": "date"},
    {"id": "q2", "text": "Which room?", "about": "room"},
)


@pytest.mark.parametrize(
    "setting,delivered,held_questions",
    [("Structured", ["Which date", "Which room"], []),
     ("Iterative", ["Which date"], ["q2"])],
)
def test_asking_turn_does_no_work_and_the_next_turn_proceeds(
    loop_factory, setting: str, delivered: list[str], held_questions: list[str]
) -> None:
    """The questions gate has to be reachable before it can be a gate.

    `IX_render_information_elicitation` used to be required only while an
    elicitation episode was open, and only that tool can open one. Nothing
    bootstrapped it: across two live acc_009 runs it was called on 0 of 20 PA
    turns, no question was ever released, and `Infer` / `Structured` /
    `Iterative` were behaviourally identical. Now the payload is owed on every
    turn the attribute is active, so the asking turn really does stop before
    the work and the turn after really does proceed.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(
                content=(
                    "Before I book:\n\n[[q:q1]] Which date works?\n\n"
                    "[[q:q2]] Which room do you want?"
                ),
                tool_calls=[
                    selection("information_elicitation", setting),
                    _questions_payload(*QUESTIONS),
                    task_call("book_1", "task_book", "i1"),
                ],
            ),
            LLMResponse(content=None, tool_calls=[
                _questions_payload(),
                ix("IX_continue_information_elicitation", decisions=[
                    {"id": "q2", "decision": "skip", "user_evidence": ""},
                    {"id": "i1", "decision": "release", "user_evidence": "the 14th"},
                ]),
                task_call("book_2", "task_book", "i1"),
            ]),
            LLMResponse(content="Booked the 14th."),
        ])
        loop, _ = loop_factory(provider, "task_book")

        first = await run_turn(
            loop, "Book me a room.", active_ix=["information_elicitation"],
            action_types={"task_book": "internal_write"},
        )

        # The asking turn asks and does nothing else.
        assert executed_tools(first) == []
        for phrase in delivered:
            assert phrase in first.content
        blob = first.metadata["ix_render"]
        assert blob["released"]["questions"] == [
            question["id"] for question in QUESTIONS
            if question["id"] not in held_questions
        ]
        assert [
            question["id"] for question in blob["held"]["questions"]
        ] == held_questions
        assert [call["status"] for call in blob["held"]["calls"]] == ["held_for_questions"]
        assert blob["episodes"]["information_elicitation"]["episode_id"] == "el1"

        second = await run_turn(
            loop, "The 14th, the small room.",
            active_ix=["information_elicitation"],
            action_types={"task_book": "internal_write"},
        )

        # The turn after answers proceeds, and the episode closes behind it.
        assert executed_tools(second) == ["task_book"]
        assert second.metadata["ix_render"]["episodes"] == {}
        assert second.metadata["ix_render"]["held"]["calls"] == []
        assert loop.close_ix_session(SESSION_KEY, "acc_001") == []

    run(scenario())


def test_the_elicitation_payload_is_owed_on_a_turn_with_no_questions(
    loop_factory,
) -> None:
    """An empty `questions` list is a valid payload, and still required.

    Requiredness that tracked the episode leaked nothing but reached nothing
    either. Requiredness that tracks the attribute is setting-blind: `Infer`
    owes the same call, with an empty list and its assumptions instead.
    """
    async def scenario() -> None:
        payload = ix(
            "IX_render_information_elicitation",
            elicitation_id="el1",
            questions=[],
            assumptions=[{"about": "room", "assumed": "the usual one"}],
        )
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Infer"), payload,
            ]),
            LLMResponse(content="Booked the usual room for Tuesday."),
            # Turn two: no episode is open and no question is being asked, and
            # the payload is still owed — the wrapper nudges once.
            LLMResponse(content="Tuesday at nine, then."),
            LLMResponse(content=None, tool_calls=[payload]),
            LLMResponse(content="Tuesday at nine, then."),
        ])
        loop, _ = loop_factory(provider)

        await run_turn(loop, "Book me a room.", active_ix=["information_elicitation"])
        second_start = len(provider.requests)
        outbound = await run_turn(
            loop, "Nine works.", active_ix=["information_elicitation"],
        )

        nudges = [
            str(message.get("content"))
            for request in provider.requests[second_start:]
            for message in request
            if str(message.get("content") or "").startswith(IX_PER_TURN_REMINDER_PREFIX)
        ]
        assert nudges and "IX_render_information_elicitation" in nudges[0]
        assert "IX_render_information_elicitation" in [
            event.get("name") for event in outbound.metadata["tool_events"]
        ]
        assert outbound.content == "Tuesday at nine, then."
        assert outbound.metadata["ix_render"]["episodes"] == {}

    run(scenario())


# ---------------------------------------------------------------------------
# acc_003 · a closing line delivered in place of the answer
# ---------------------------------------------------------------------------

#: The whole of acc_003 t=20, asked for the exact updated metadata fields.
EMPTY_CLAIM = "Summary of the two updated entries is complete."


def _unfinished_turn(*replies: str) -> ScriptProvider:
    return ScriptProvider([
        LLMResponse(content=None, tool_calls=[
            selection("process_visibility", "Bookend"),
            ix("IX_render_process_visibility", completion_message=EMPTY_CLAIM),
        ]),
        *[LLMResponse(content=reply) for reply in replies],
    ])


def _internal(request: list[dict]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in request
        if message.get("_mempa_internal") is True
    ]


def test_a_completion_standing_in_for_the_answer_buys_one_re_prompt(
    loop_factory,
) -> None:
    """acc_003 t=20: 47 characters, no summary, no tool call, released as done.

    The wrapper writes none of the answer — it says what is missing and asks
    once. Here the second reply carries the content, so the turn is delivered on
    its own merits and nothing is recorded against it.
    """
    async def scenario() -> None:
        answer = (
            "Reyes: year 2024, venue Physical Review X. "
            "Brandao: year 2020, venue Reviews of Modern Physics."
        )
        provider = _unfinished_turn(EMPTY_CLAIM, answer)
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "State the exact updated metadata fields.",
            active_ix=["process_visibility"],
        )

        assert IX_UNFINISHED_TURN_PROMPT in _internal(provider.requests[-1])
        assert answer in outbound.content
        assert "ix_completion_without_content" not in [
            event.get("name") for event in outbound.metadata["tool_events"]
        ]

    run(scenario())


def test_a_second_empty_claim_is_delivered_and_recorded(loop_factory) -> None:
    """Asked once, then whatever comes back is the turn.

    The wrapper cannot write the summary the PA claimed to have written, so a
    second reply that is still nothing but the closing line ships as it stands
    and the audit is what says so.
    """
    async def scenario() -> None:
        provider = _unfinished_turn(EMPTY_CLAIM, EMPTY_CLAIM)
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "State the exact updated metadata fields.",
            active_ix=["process_visibility"],
        )

        assert outbound.content == EMPTY_CLAIM
        event = next(
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_completion_without_content"
        )
        assert event["chars"] == len(EMPTY_CLAIM)
        # One extra request, not two.
        assert len(provider.requests) == 3

    run(scenario())


def test_a_short_turn_that_actually_answers_is_asked_but_not_recorded(
    loop_factory,
) -> None:
    """The widened trigger cannot tell this turn from acc_003 t=17, and pays for it.

    A short answer under a work-asserting closing line looks exactly like a
    claim standing in for one, so it buys the same one request. What it must not
    buy is a defect: the second reply carries the content and the audit stays
    empty.
    """
    async def scenario() -> None:
        answer = "Reyes is 2024, Brandao 2020. " + EMPTY_CLAIM
        provider = _unfinished_turn(answer, answer + " Both are now consistent.")
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "State the exact updated metadata fields.",
            active_ix=["process_visibility"],
        )

        assert "Reyes is 2024" in outbound.content
        assert IX_UNFINISHED_TURN_PROMPT in _internal(provider.requests[-1])
        assert "ix_completion_without_content" not in [
            event.get("name") for event in outbound.metadata["tool_events"]
        ]

    run(scenario())


def test_a_claim_worded_differently_from_the_closing_line_is_re_prompted(
    loop_factory,
) -> None:
    """acc_003 t=17 — the same turn, with the prose reworded.

    "That covers all five remaining entries" is not the closing line repeated,
    so containment lets it through; what it is, is a reply too short to be the
    list it says it has laid out, under a completion that asserts the work.
    """
    async def scenario() -> None:
        answer = (
            "Reyes: year 2024, venue Physical Review X. "
            "Brandao: year 2020, venue Reviews of Modern Physics."
        )
        provider = _unfinished_turn(
            "That covers both remaining entries. Shall I start the first lookup?",
            answer,
        )
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "State the exact updated metadata fields.",
            active_ix=["process_visibility"],
        )

        assert IX_UNFINISHED_TURN_PROMPT in _internal(provider.requests[-1])
        assert answer in outbound.content

    run(scenario())


def test_a_short_claim_with_the_work_behind_it_is_not_re_prompted(
    loop_factory,
) -> None:
    """The guard the second trigger needs: a turn that did the work.

    Same closing line, same short prose — but a task tool ran, so the completion
    is reporting something real and there is nothing missing to ask for.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("process_visibility", "Bookend"),
                ix("IX_render_process_visibility", completion_message=EMPTY_CLAIM),
                task_call("w1", "task_write", notice="Updating the two entries"),
            ]),
            LLMResponse(content="That covers both remaining entries."),
        ])
        loop, _ = loop_factory(provider, "task_write")

        outbound = await run_turn(
            loop, "Update the two entries.", active_ix=["process_visibility"],
            action_types={"task_write": "internal_write"},
        )

        assert executed_tools(outbound) == ["task_write"]
        assert IX_UNFINISHED_TURN_PROMPT not in _internal(provider.requests[-1])

    run(scenario())


def test_a_long_reply_under_a_work_claim_is_not_re_prompted(loop_factory) -> None:
    """The other guard: prose past the floor is a body, not a stand-in.

    The reply says nothing the closing line says, and no tool ran, but it is
    300+ characters of actual answer — exactly the turn the length floor is
    there to leave alone.
    """
    async def scenario() -> None:
        answer = " ".join(
            f"Entry {index}: the catalog gives the venue and the year, and the "
            f"note in the file disagrees with both."
            for index in range(1, 5)
        )
        assert len(answer) >= 300
        provider = _unfinished_turn(answer)
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "State the exact updated metadata fields.",
            active_ix=["process_visibility"],
        )

        assert "Entry 4" in outbound.content
        assert len(provider.requests) == 2
        assert IX_UNFINISHED_TURN_PROMPT not in _internal(provider.requests[-1])

    run(scenario())


# ---------------------------------------------------------------------------
# acc_001 · the model's tool-call syntax arriving as prose
# ---------------------------------------------------------------------------

#: acc_001 t34, shortened. The full-width bar is what the model really emits.
DSML_BLOCK = (
    '<｜DSML｜tool_calls>\n'
    '<｜DSML｜invoke name="mcp_state_document_update">\n'
    '<｜DSML｜parameter name="path" string="true">'
    'documents/research/citation_notes.md</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="anchor" string="true">## Formatted entries'
)


def _gate_turn(prose: str) -> ScriptProvider:
    """acc_039 idx 24's shape: the plan and its prose, then the call the gate stops.

    The prose has to arrive in an *earlier* iteration than the held call —
    the wrapper withholds anything written in the same assistant message as
    a held call, which is why the leak needed two iterations to happen at all.
    """
    return ScriptProvider([
        LLMResponse(content=prose, tool_calls=[
            selection("autonomy_level", "Self-directed"),
            plan_call({"id": "a1", "bound_item_ids": ["i1"], "is_key": True}),
        ]),
        LLMResponse(content=None, tool_calls=[
            task_call("s1", "task_send", "i1", ask="Shall I go ahead?"),
        ]),
    ])


async def _run_gate_turn(loop) -> Any:
    return await run_turn(
        loop, "Book the checkup.", active_ix=["autonomy_level"],
        action_types={"task_send": "external_action"},
    )


def test_a_gate_turn_never_ships_a_claim_that_the_action_happened(
    loop_factory,
) -> None:
    """acc_039 idx 24/26 — "I'll proceed with booking now", four lines above "Shall I go ahead?".

    Nothing executed on that turn, and two of the three gate turns opened in the
    present progressive over calls that were held. Withheld, never rewritten:
    what the user is owed here is the ask the PA wrote.
    """
    async def scenario() -> None:
        provider = _gate_turn(
            "Here are the details I've prepared. I'll proceed with booking now.\n"
            "Adding the checkup appointment to your calendar."
        )
        loop, _ = loop_factory(provider, "task_send")

        outbound = await _run_gate_turn(loop)

        assert executed_tools(outbound) == []
        assert "I'll proceed" not in outbound.content
        assert "Adding the checkup appointment" not in outbound.content
        # The sentence that only sets the ask up is not a claim, and stays.
        assert "Here are the details I've prepared." in outbound.content
        assert outbound.content.rstrip().endswith("Shall I go ahead?")
        narration = outbound.metadata["ix_render"]["held"]["narration"]
        assert [row["item_id"] for row in narration] == ["claim_of_action"] * 2
        assert {row["text"] for row in narration} == {
            "I'll proceed with booking now.",
            "Adding the checkup appointment to your calendar.",
        }

    run(scenario())


def test_a_gate_turn_keeps_a_proposal_that_only_reads_like_a_claim(
    loop_factory,
) -> None:
    """The guard: a modal makes the same opening a proposal, which is the point of the ask."""
    async def scenario() -> None:
        provider = _gate_turn("Adding a second slot would give you a fallback.")
        loop, _ = loop_factory(provider, "task_send")

        outbound = await _run_gate_turn(loop)

        assert "Adding a second slot would give you a fallback." in outbound.content
        assert "narration" not in outbound.metadata["ix_render"]["held"]

    run(scenario())


def test_a_claim_the_locator_cannot_place_is_recorded(loop_factory) -> None:
    """acc_035 — a dropped claim was trace-only for three rounds.

    No hedge is inserted, so that claim ships flat; nothing counted it, which is
    how a `claim_end` gap ate the first `low` claim in four rounds without
    anyone noticing until the read. The trace is one row per claim with the
    `claim` head on it: the four drops of one live session sat in two `refs` lists,
    and the head is what shows the read that three of them were sentences out of
    an *earlier* turn's reply rather than locator misses.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Express"),
                ix("IX_render_uncertainty_expression", certainties=[
                    {"ref": "c1", "level": "low",
                     "claim": "The roster has no per-question breakdown"},
                    {"ref": "c2", "level": "high",
                     "claim": "I have not touched the roster yet"},
                ]),
            ]),
            LLMResponse(content="I have not touched the roster yet."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Who else lost the point?", active_ix=["uncertainty_expression"],
        )

        dropped = [
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_certainty_claim_not_found"
        ]
        assert len(dropped) == 1
        assert dropped[0]["ref"] == "c1"
        assert dropped[0]["claim"] == "The roster has no per-question breakdown"
        assert dropped[0]["attribute"] == "uncertainty_expression"
        assert "turn" in dropped[0]

    run(scenario())


def test_every_declared_certainty_dropped_is_re_asked_once(loop_factory) -> None:
    """acc_035 t=7 — three declared, `annotated 0/3`, a plain reply.

    The Express surface vanished from the very turn the reception beat reads,
    and the actor approved a reply with no hedges in it. The read says the
    claims were sentences of the *previous* turn, so the repair asks for
    fragments of this one; nothing is appended to the prose, because a claim the
    model did not write into its reply is not the wrapper's sentence to add.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Express"),
                ix("IX_render_uncertainty_expression", certainties=[
                    {"ref": "c1", "level": "high",
                     "claim": "The regrade request is about Quiz 3, question 2"},
                ]),
            ]),
            LLMResponse(content="All updates are applied."),
            LLMResponse(content=(
                "All updates are applied.\n"
                "<certainties>\n"
                "- ref: c1 | level: high | claim: All updates are applied\n"
                "</certainties>"
            )),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Apply the roster updates.", active_ix=["uncertainty_expression"],
        )

        assert "All updates are applied(confidence: high)" in outbound.content
        event = next(
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_certainties_all_dropped"
        )
        assert event["declared"] == 1
        assert event["attempt"] == 1
        assert event["attribute"] == "uncertainty_expression"
        # The repair names the reply, never a sentence for the model to copy.
        assert any(
            "failed to match a sentence of the reply" in str(message.get("content"))
            for message in provider.requests[-1]
        )

    run(scenario())


def test_a_second_all_dropped_turn_is_recorded_and_the_step_continues(
    loop_factory,
) -> None:
    """The other half of the rule, softened.

    The second all-drop used to raise out of the loop: no turn persisted, the
    step failed and was retried, and one run spent a whole rerun of acc_035 on
    one turn's slip. The reply now ships and the failure is an event, because
    the wrapper's own answer to an unplaced claim is to append nothing -- so
    what the abort bought was a lost run, not a different surface.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Express"),
                ix("IX_render_uncertainty_expression", certainties=[
                    {"ref": "c1", "level": "high",
                     "claim": "The regrade request is about Quiz 3, question 2"},
                ]),
            ]),
            LLMResponse(content="All updates are applied."),
            LLMResponse(content="All updates are applied."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Apply the roster updates.", active_ix=["uncertainty_expression"],
        )

        assert outbound.content == "All updates are applied."
        final = [
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_certainties_all_dropped"
        ][-1]
        assert final["attempt"] == 2
        assert final["detail"] == "certainties contract"
        assert final["continued"] is True

    run(scenario())


def test_the_continued_turn_carries_no_certainty_label_at_all(loop_factory) -> None:
    """What "untagged" means: the delivered reply has no parenthesis on it.

    The claims are the only thing `Express` renders, so a turn that locates none
    of them reaches the user exactly as a `Hide` turn would. That is the loss
    the event records; the wrapper does not paper over it by writing a hedge the
    PA never put in its prose.
    """
    async def scenario() -> None:
        declared = ix("IX_render_uncertainty_expression", certainties=[
            {"ref": "c1", "level": "high", "claim": "Quiz 3 has been regraded"},
            {"ref": "c2", "level": "low", "claim": "Farid submitted on time"},
        ])
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Express"), declared,
            ]),
            LLMResponse(content="The roster is up to date."),
            LLMResponse(content="The roster is up to date."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Apply the roster updates.", active_ix=["uncertainty_expression"],
        )

        assert outbound.content == "The roster is up to date."
        assert "confidence:" not in outbound.content
        assert [
            item["attempt"] for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_certainties_all_dropped"
        ] == [1, 2]

    run(scenario())


def test_a_dropped_claim_under_hide_is_never_re_asked(loop_factory) -> None:
    """`Hide` renders no parenthesis, so an unplaced claim changes not one byte.

    Spending a request there would buy a difference the user cannot see -- and
    would put the setting into the request pattern, which the carrier is written
    to be blind to (carrier F).
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("uncertainty_expression", "Hide"),
                ix("IX_render_uncertainty_expression", certainties=[
                    {"ref": "c1", "level": "high",
                     "claim": "a sentence from some other turn"},
                ]),
            ]),
            LLMResponse(content="All updates are applied."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Apply the roster updates.", active_ix=["uncertainty_expression"],
        )

        assert outbound.content == "All updates are applied."
        assert not [
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_certainties_all_dropped"
        ]
        assert not provider.responses

    run(scenario())


def test_a_dangling_lead_in_is_dropped_and_recorded(loop_factory) -> None:
    """acc_009 t=10 — the whole delivered turn was a colon.

    Under `Infer` the release is 0/2, so the questions the PA wrote came out and
    the sentence introducing them shipped alone; the user read *"…I need two
    details from you:"* and nothing followed it, on three turns of one beat.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Infer"),
                ix("IX_render_information_elicitation",
                   elicitation_id="section_review_draft",
                   questions=[
                       {"id": "q1", "text": "How long?", "about": "window"},
                       {"id": "q2", "text": "What workflow?", "about": "workflow"},
                   ],
                   assumptions=[]),
            ]),
            LLMResponse(content=(
                "Happy to hear the draft structure is acceptable! "
                "To finalize it, I need two details from you:\n\n"
                "[[q:q1]] How long should the review window be?\n\n"
                "[[q:q2]] How would you like the comments workflow to work?"
            )),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "The structure looks fine.",
            active_ix=["information_elicitation"],
        )

        assert outbound.content == "Happy to hear the draft structure is acceptable!"
        event = next(
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_leadin_dropped"
        )
        assert event["sentence"] == "To finalize it, I need two details from you:"
        assert "turn" in event

    run(scenario())


def test_a_turn_that_is_only_a_lead_in_is_re_asked_not_leaked(loop_factory) -> None:
    """The other half: with the lead-in gone there is no turn left to deliver.

    The fallbacks that ship `final_content` when the renderer produced no
    message read the model's *raw* text, which still carries every excised
    section — so before `prose_held` a fully-held `Infer` turn with no framing
    handed the user both withheld questions verbatim. Now the turn takes the
    empty-reply re-ask and the PA writes what it can say without them.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Infer"),
                ix("IX_render_information_elicitation",
                   elicitation_id="section_review_draft",
                   questions=[
                       {"id": "q1", "text": "How long?", "about": "window"},
                       {"id": "q2", "text": "What workflow?", "about": "workflow"},
                   ],
                   assumptions=[]),
            ]),
            LLMResponse(content=(
                "I need two details from you:\n\n"
                "[[q:q1]] How long should the review window be?\n\n"
                "[[q:q2]] How would you like the comments workflow to work?"
            )),
            LLMResponse(content="I will work from the defaults for now."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "The structure looks fine.",
            active_ix=["information_elicitation"],
        )

        assert outbound.content == "I will work from the defaults for now."
        assert "review window" not in outbound.content
        assert "comments workflow" not in outbound.content

    run(scenario())


def test_the_repair_prompt_s_own_words_never_reach_the_user(loop_factory) -> None:
    """acc_009 t=17 — `Here's the corrected turn:` as the delivered message.

    Nothing in the wrapper writes that sentence: it is the model reading
    `NO_META_ACKNOWLEDGEMENT` ("simply produce the corrected turn"), sent with
    that turn's forced render payload, and answering the instruction in the
    user's face. `ix_close_no_message` had already suppressed the only other
    candidate sentence, so the echo was the whole head of the reply.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Structured"),
                ix("IX_render_information_elicitation",
                   elicitation_id="e1", questions=[], assumptions=[]),
            ]),
            LLMResponse(content=(
                "Here's the corrected turn:\n\n"
                "Thank you for your answers. I've now updated the draft."
            )),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "60 minutes, comments a day ahead.",
            active_ix=["information_elicitation"],
        )

        assert outbound.content == (
            "Thank you for your answers. I've now updated the draft."
        )
        narration = outbound.metadata["ix_render"]["held"]["narration"]
        assert {row["item_id"] for row in narration} == {"wrapper_vocabulary"}
        assert "Here's the corrected turn:" in {row["text"] for row in narration}

    run(scenario())


def test_source_brackets_in_a_quoted_fixture_are_not_wrapper_vocabulary(
    loop_factory,
) -> None:
    """acc_035 — the filter deleted the answer key the turn existed to retrieve.

    `documents/admin/seminar_quiz_raw.md` writes the key as `[[keyed: …]]`, and
    `"[["` / `"]]"` were vocabulary entries in their own right, so the sentence
    was withheld and the user read an `### Official Answer Key` heading with
    nothing under it. Bare brackets are ordinary source markup.
    """
    async def scenario() -> None:
        quoted = (
            "### Official Answer Key\n\n"
            "> **[[keyed:** ONE degenerate fusion space (the two outcomes 1 and "
            "psi span a single two-dimensional space), so exchange acts as a "
            "matrix on that space, not as a phase."
        )
        provider = ScriptProvider([LLMResponse(content=quoted)])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(loop, "Retrieve the answer key.", active_ix=[])

        assert "ONE degenerate fusion space" in outbound.content
        assert "keyed" in outbound.content
        # Nothing was withheld: the key is the deliverable, not wrapper talk.
        assert "narration" not in outbound.metadata["ix_render"]["held"]

    run(scenario())


def test_a_real_marker_left_in_the_prose_is_still_wrapper_vocabulary() -> None:
    """The guard, at the filter itself.

    The marker machinery resolves every well-formed `[[q:…]]` long before this
    filter runs, so the bracket rule only ever fired on residue — which is what
    made the bare tokens so costly. Residue that really is a wrapper marker
    still goes.
    """
    from pa.agent.ix.render import names_the_wrapper

    for marker in ("[[q:q1]]", "[[q1]]", "[[batch:b1]]", "[[b1]]", "[[c:c1]]"):
        assert names_the_wrapper(f"Left over: {marker} which airport?"), marker
    for source in ("[[keyed: ONE degenerate fusion space", "see [[19]] and [[58]]"):
        assert not names_the_wrapper(source), source


def _markup_turn(*replies: str) -> ScriptProvider:
    """The acc_001 shape: the markup is the finished turn, not an intermediate step."""
    return ScriptProvider([
        LLMResponse(content=None, tool_calls=[
            selection("process_visibility", "Bookend"),
            ix("IX_render_process_visibility", completion_message=""),
        ]),
        *[LLMResponse(content=reply) for reply in replies],
    ])


def test_tool_call_markup_is_stripped_and_the_prose_survives(loop_factory) -> None:
    """acc_001 t34: real prose, then an unparsed `document_update` invoke.

    The update never ran and t38 told the user it had. The markup comes off and
    the sentences around it are delivered untouched.
    """
    async def scenario() -> None:
        prose = "The paper covers the P1 mechanism. Updating the annotation now."
        provider = _markup_turn(f"{prose}\n\n{DSML_BLOCK}")
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Resolve the Brandao flag.", active_ix=["process_visibility"],
        )

        assert "DSML" not in outbound.content
        assert "document_update" not in outbound.content
        assert prose in outbound.content
        event = next(
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_markup_stripped"
        )
        assert event["blocks"] == 1
        assert event["chars"] == len(DSML_BLOCK)

    run(scenario())


def test_a_reply_that_is_only_tool_call_markup_is_re_prompted(loop_factory) -> None:
    """acc_001 t32: 7056 characters, all of it unparsed invoke, none of it a reply.

    Nothing survives the strip, so the turn has said nothing — it goes back on
    the unfinished-turn path and the second reply is what the user reads.
    """
    async def scenario() -> None:
        answer = "The Brandao flag is unresolved: their pointer basis is not our P1 label."
        provider = _markup_turn(DSML_BLOCK, answer)
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Resolve the Brandao flag.", active_ix=["process_visibility"],
        )

        assert IX_UNFINISHED_TURN_PROMPT in _internal(provider.requests[-1])
        assert outbound.content == answer
        # One extra request, not two: the empty-reply rule does not ask again.
        assert len(provider.requests) == 3

    run(scenario())


#: acc_006 t13, verbatim — and it stops mid-JSON, mid-parameter, with no
#: closing tag of any kind: the model ran out of completion tokens inside its own
#: argument list. Nothing here can be turned into a call.
ACC_006_T13_TRUNCATED = (
    '<｜DSML｜tool_calls>\n'
    '<｜DSML｜invoke name="mcp_state_document_update">\n'
    '<｜DSML｜parameter name="path" string="true">'
    'documents/research/citation_notes.md</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="anchored_replacements" string="false">'
    '[{"anchor": "  pages   = {115312},", "old": "\\n  year    = {2020},", '
    '"new": "\\n  year    = {2021},"}, '
    '{"anchor": "check the fulltext before hanging the 2.2 sentence on it.", '
    '"old": "\\n\\n```bibtex\\n@article{okonkwo2024arxiv,\\n'
)


def test_a_turn_that_typed_out_its_calls_never_ships_the_empty_reply_template(
    loop_factory,
) -> None:
    """acc_006 t13/t16/t20/t26 — the defect this test is named after.

    Four turns whose whole reply was tool-call syntax were stripped to nothing
    and then shipped `I have nothing further to report on that just now.` The PA
    had four document edits in hand and typed every one of them, so the line was
    false, and false in the direction that hides the miss: the turn reads like a
    polite no-op and scores as one instead of as the writes that never happened.
    The one re-prompt is still spent; what changes is that its failure ships
    nothing rather than a sentence nobody can act on.
    """
    async def scenario() -> None:
        provider = _markup_turn(ACC_006_T13_TRUNCATED, "")
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Fix the Sarpong and Novak years.",
            active_ix=["process_visibility"],
        )

        assert outbound.content == ""
        assert "nothing further to report" not in (outbound.content or "")
        assert outbound.metadata["ix_render"].get("close_line_source", "") != "template"
        # No markup either: the block is on the audit plane, not in the reply.
        assert "DSML" not in (outbound.content or "")
        assert [row["item_id"] for row in
                outbound.metadata["ix_render"]["held"]["narration"]] == ["dsml_markup"]
        # The miss is still recorded, and now says which kind it was.
        event = next(
            item for item in _ix_events(outbound) if item["name"] == "ix_empty_reply"
        )
        assert event["dsml_markup"] is True
        # Still exactly one extra request.
        assert len(provider.requests) == 3

    run(scenario())


def test_a_turn_whose_calls_were_recovered_ships_no_template_either(
    loop_factory,
) -> None:
    """The other half: the seam parsed the block, the call ran, and the model
    then wrote no prose. The turn did real work, so `nothing further to report`
    is as wrong here as it is over an unparsable block."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(
                content=None,
                tool_calls=[task_call("dsml_1", "task_send")],
                dsml_converted=1,
            ),
            LLMResponse(content=None),
            LLMResponse(content=None),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(loop, "Send the summary.", active_ix=[])

        assert executed_tools(outbound) == ["task_send"]
        assert outbound.content == ""
        assert outbound.metadata["ix_render"].get("close_line_source", "") != "template"
        assert [
            item["name"] for item in _ix_events(outbound)
            if item["name"] == "ix_dsml_calls_recovered"
        ] == ["ix_dsml_calls_recovered"]
        assert len(provider.requests) == 3

    run(scenario())


def test_a_turn_with_no_markup_still_gets_the_template(loop_factory) -> None:
    """The guard on the rule above: an ordinary silent turn is unchanged.

    The live empty completion has no markup anywhere in it, and the template line is
    the right answer there — the PA really did have nothing to say.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None),
            LLMResponse(content=None),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(loop, "Book the room.", active_ix=[])

        assert outbound.content == "I have nothing further to report on that just now."
        assert outbound.metadata["ix_render"]["close_line_source"] == "template"

    run(scenario())
