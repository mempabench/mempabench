"""T1–T4, T9 — the turn envelope across turns and across packages.

`tests/test_pa/test_ix_wrapper_loop.py` already owns the
single-turn half of T3/T4/T9 (plain turn, one executed + one held call, the
history the next provider call accepts). What is here is the part that only
shows up when more than one turn, or more than one package, is involved: a
setting chosen on turn 1 still driving the payload contract on turn 3, a held
batch that must not reach the record on the *next* turn, and the two turn kinds
whose envelope is built somewhere other than `_process_message`'s main path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from harness.judge_pack import build_text_judge_pack
from memory.event_writer import build_turn_pool
from pa.agent.ix.collector import HOLD_PROMPT_PREFIX, TURN_REMINDER_PREFIX
from pa.agent.runner import (
    IX_PER_TURN_REMINDER_PREFIX,
    IX_TASK_CALLS_DEFERRED_FOR_CARRIER,
)
from pa.providers.base import LLMResponse

from .conftest import (
    SCRIPT_SESSION,
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

THREE_BATCHES = (
    "Here is where the three of them stand.\n\n"
    "[[batch:b1]] Flights: the 09:40 is still open.\n\n"
    "[[batch:b2]] Hotel: two rooms left at the Kimpton.\n\n"
    "[[batch:b3]] Invoice: finance has not signed it off."
)

BATCH_PAYLOAD = ix(
    "IX_render_topic_management",
    batch_set_id="bs1",
    item_count=3,
    batches=[
        {"batch_id": "b1", "label": "Flights", "body": "", "call_binding": ["i1"]},
        {"batch_id": "b2", "label": "Hotel", "body": "", "call_binding": ["i2"]},
        {"batch_id": "b3", "label": "Invoice", "body": "", "call_binding": ["i3"]},
    ],
)


def _blob(messages: Any) -> str:
    return json.dumps(messages, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# T1 · a setting chosen on turn 1 still binds the payload contract on turn 3
# ---------------------------------------------------------------------------

def _payload_turn(notice: str, completion: str, item_id: str, text: str) -> list[LLMResponse]:
    """One turn: payloads + one read, then the answer prose."""
    return [
        LLMResponse(content=None, tool_calls=[
            ix("IX_render_process_visibility", completion_message=completion),
            plan_call({"id": f"a_{item_id}", "bound_item_ids": [item_id]}),
            task_call(f"read_{item_id}", "task_read", item_id, notice=notice),
        ]),
        LLMResponse(content=text),
    ]


def test_a_setting_chosen_on_turn_one_still_forces_the_payload_on_turn_three(
    loop_factory,
) -> None:
    """T1: selection is per session, the payload contract is per turn.

    Turn 3 brings a task the plan has never claimed. The selection tools are
    gone — that attribute is settled — but `IX_render_<attribute>` is still
    offered, still judged required, the per-call fields are still injected, and
    the new payload is still rendered under turn 1's setting.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            # turn 1 — choose both settings and render the first task
            LLMResponse(content=None, tool_calls=[
                selection("process_visibility", "Full narration"),
                selection("autonomy_level", "Self-directed"),
                ix("IX_render_process_visibility", completion_message="That one is done."),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("read_1", "task_read", "i1", notice="Checking the calendar."),
            ]),
            LLMResponse(content="Monday is free."),
            # turn 2 — an ordinary answer in between
            *_payload_turn("Looking at the invoice.", "Invoice checked.", "i2", "Unpaid."),
            # turn 3 — a brand new task, and the model forgets the plan payload
            LLMResponse(content=None, tool_calls=[
                ix("IX_render_process_visibility", completion_message="Booked."),
                task_call("book_3", "task_read", "i9", notice="Booking it now."),
            ]),
            # The nudged response is discarded whole, so the resend carries
            # both payloads again, exactly as a real retry would.
            LLMResponse(content=None, tool_calls=[
                ix("IX_render_process_visibility", completion_message="Booked."),
                plan_call({"id": "a9", "bound_item_ids": ["i9"]}, plan_id="p9"),
                task_call("book_3b", "task_read", "i9", notice="Booking it now."),
            ]),
            LLMResponse(content="It is booked."),
        ])
        loop, _ = loop_factory(provider, "task_read")
        active = ["process_visibility", "autonomy_level"]
        # `task_read` is not in the real action-type table, and an unregistered
        # tool fails closed to `external_action`, which `Self-directed`
        # gates. This scenario is about payload requiredness, not the gate.
        reads = {"task_read": "read"}

        await run_turn(loop, "Is Monday free?", active_ix=active, action_types=reads)
        await run_turn(loop, "And the invoice?", active_ix=active, action_types=reads)
        turn_three_start = len(provider.tool_definitions)
        outbound = await run_turn(
            loop, "Book the 09:40 for me.", active_ix=active, action_types=reads,
        )

        offered = provider.offered(turn_three_start)
        # The per-session selection is settled, so those tools are gone …
        assert "IX_process_visibility" not in offered
        assert "IX_autonomy_level" not in offered
        # … while the per-turn payload tools are still there.
        assert {"IX_render_process_visibility", "IX_render_autonomy_level"} <= set(offered)

        # Carrier C is keyed on `session_active_ix`, not on "already selected".
        task_schema = provider.definition("task_read", turn_three_start)["function"]
        assert {"interaction_item_id", "_ix_notice"} <= set(
            task_schema["parameters"]["properties"]
        )
        # Offered, not demanded: only a plan or a batch declares a call id, so a
        # required field could only ever be filled by inventing one (acc_009).
        assert "interaction_item_id" not in task_schema["parameters"]["required"]
        assert "_ix_notice" in task_schema["parameters"]["required"]

        # Still judged required: the missing payload is nudged for, once. (Each
        # later request replays the whole message list, so count the distinct
        # nudges rather than their appearances.)
        nudges = {
            str(message.get("content"))
            for request in provider.requests[turn_three_start:]
            for message in request
            if str(message.get("content") or "").startswith((
                IX_PER_TURN_REMINDER_PREFIX,
                IX_TASK_CALLS_DEFERRED_FOR_CARRIER.split("{tools}")[0],
            ))
        }
        assert len(nudges) == 1
        assert "IX_render_autonomy_level" in nudges.pop()

        # And turn 1's setting still renders turn 3's new payload.
        assert "Booking it now." in outbound.content
        assert outbound.content.rstrip().endswith("Booked.")
        assert outbound.metadata["ix_render"]["completion"]["released"] is True

    run(scenario())


# ---------------------------------------------------------------------------
# T2 · a held batch is invisible to the next turn's record
# ---------------------------------------------------------------------------

def test_a_batch_held_on_turn_one_is_absent_from_turn_twos_history_and_memory(
    loop_factory,
) -> None:
    """T2: the PA may know b2 exists, but only through the hold channel."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "One-at-a-time"),
                BATCH_PAYLOAD,
            ]),
            LLMResponse(content=THREE_BATCHES),
            LLMResponse(content=None, tool_calls=[
                ix("IX_continue_topic_management", decisions=[
                    {"id": "b2", "decision": "keep", "user_evidence": ""},
                    {"id": "b3", "decision": "keep", "user_evidence": ""},
                ]),
            ]),
            LLMResponse(content="Still just the flights, then."),
        ])
        loop, adapter = loop_factory(provider)

        first = await run_turn(
            loop, "Where are we on the trip?", active_ix=["topic_management"],
        )
        assert "Flights" in first.content
        assert "Kimpton" not in first.content

        second_turn_start = len(provider.tool_definitions)
        await run_turn(loop, "Fine, and the rest?", active_ix=["topic_management"])

        record = _blob(loop.sessions.get(SESSION_KEY).messages)
        memory = _blob([event.messages for event in adapter.events])
        for held_body in ("Kimpton", "finance has not signed it off"):
            assert held_body not in record
            assert held_body not in memory
        # Nor does the held body reach the model on the next turn …
        assert "Kimpton" not in _blob(provider.requests[second_turn_start])
        # … but the controlled continuation channel is open: the release tool is
        # offered exactly because this attribute still holds something.
        assert "IX_continue_topic_management" in provider.offered(second_turn_start)

    run(scenario())


# ---------------------------------------------------------------------------
# T2 (R1) · the same contract when the *payload field* carries the held text
#
# The T2 case above puts the held wording in the model's prose and leaves the
# payload's `body` empty, so it never exercised the carrier the payload tools
# actually persist. `IXPerTurnTool.execute()` echoes the whole payload back as
# a tool result, and the executed-subset filter kept it: a held `Question.text`
# / `Batch.body` was filed twice in `runtime_history`, the second time looking
# like a delivered answer. The declaration stays (it is the PA's own state);
# the echo is replaced by a status object that names the hold.
# ---------------------------------------------------------------------------

HELD_SECRET = "HELD_SECRET_Q2"

BODY_BATCHES = ix(
    "IX_render_topic_management",
    batch_set_id="bs1",
    item_count=2,
    batches=[
        {"batch_id": "b1", "label": "Flights",
         "body": "The 09:40 is still open.", "call_binding": []},
        {"batch_id": "b2", "label": "Hotel",
         "body": f"Two rooms left at the {HELD_SECRET}.", "call_binding": []},
    ],
)

TEXT_QUESTIONS = ix(
    "IX_render_information_elicitation",
    elicitation_id="el1",
    questions=[
        {"id": "q1", "text": "Which date works?", "about": "date"},
        {"id": "q2", "text": f"Which room, the {HELD_SECRET} one?", "about": "room"},
    ],
    assumptions=[],
)


def _public_faces(loop, adapter, outbound) -> dict[str, str]:
    """(a)–(d) of the isolation contract, each flattened to one string.

    The simulator is handed exactly two things by `harness/pa_factory.py`: the
    public turn and the narrow `held_items` id list, so those are its face.
    """
    session = loop.sessions.get(SESSION_KEY)
    pack = build_text_judge_pack([{
        "t": 0,
        "kind": "pa_turn",
        "beat": "open",
        "outbound": outbound.content,
        "tool_events": outbound.metadata.get("tool_events") or [],
        "ix_render": outbound.metadata.get("ix_render") or {},
    }])
    return {
        "public outbound": outbound.content,
        "memory_projection": _blob(build_turn_pool(session.messages).messages),
        "memory adapter events": _blob([event.messages for event in adapter.events]),
        "text judge pack": _blob(pack),
        "simulator held_items": _blob(outbound.metadata.get("held_items")),
    }


def _payload_results(loop) -> list[dict[str, Any]]:
    """The persisted tool results for this session's IX payload calls."""
    return [
        json.loads(message["content"])
        for message in loop.sessions.get(SESSION_KEY).messages
        if message.get("role") == "tool"
        and str(message.get("name") or "").startswith(("IX_render_", "IX_continue_"))
    ]


def test_a_held_question_text_reaches_no_public_face_and_is_annotated_in_history(
    loop_factory,
) -> None:
    """R1: `Question.text` is a payload field, and the echo persisted it.

    `Iterative` releases `q1` and holds `q2`. `q2`'s text never went out, so it
    may not appear on any of the four public faces; it stays in the record only
    as the PA's own declaration, and the result that used to echo it now says
    which ids were released and which are still held.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Iterative"),
                TEXT_QUESTIONS,
            ]),
            LLMResponse(
                content="[[q:q1]] Which date works?\n\n[[q:q2]] Which room do you want?",
            ),
        ])
        loop, adapter = loop_factory(provider)

        outbound = await run_turn(
            loop, "Book me a room.", active_ix=["information_elicitation"],
        )
        assert "Which date works?" in outbound.content

        for face, text in _public_faces(loop, adapter, outbound).items():
            assert HELD_SECRET not in text, face

        # Present, with the annotation: the declaration the PA wrote is still
        # its own state, and the result beside it states the hold.
        record = loop.sessions.get(SESSION_KEY).messages
        assert HELD_SECRET in _blob([
            message for message in record if message.get("role") == "assistant"
            and message.get("tool_calls")
        ])
        assert _payload_results(loop) == [
            {
                "status": "rendered", "released": ["q1"],
                "held_not_released": ["q2"], "cancelled": [],
            },
        ]

    run(scenario())


def test_a_held_batch_body_reaches_no_public_face_and_is_annotated_in_history(
    loop_factory,
) -> None:
    """R1, the `topic_management` half: `Batch.body` is the same carrier."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "One-at-a-time"),
                BODY_BATCHES,
            ]),
            LLMResponse(content=(
                "Here is where the two of them stand.\n\n"
                "[[batch:b1]] Flights: the 09:40 is still open.\n\n"
                "[[batch:b2]] Hotel: I still have to look."
            )),
        ])
        loop, adapter = loop_factory(provider)

        outbound = await run_turn(
            loop, "Where are we on the trip?", active_ix=["topic_management"],
        )
        assert "the 09:40 is still open" in outbound.content

        for face, text in _public_faces(loop, adapter, outbound).items():
            assert HELD_SECRET not in text, face

        assert _payload_results(loop) == [
            {
                "status": "rendered", "released": ["b1"],
                "held_not_released": ["b2"], "cancelled": [],
            },
        ]

    run(scenario())


# ---------------------------------------------------------------------------
# T3 · the envelope of the two turn kinds built off the main path
# ---------------------------------------------------------------------------

def test_a_gate_turn_keeps_the_user_message_verbatim_and_no_internal_prompt(
    loop_factory,
) -> None:
    """T3, gate round: both spoken lines verbatim, nothing internal."""
    async def scenario() -> None:
        user_text = "Check the stock and then email Dale about the extension."
        provider = ScriptProvider([
            LLMResponse(content="Checking first.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                selection("verbosity", "Detailed"),
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                ),
                task_call("read_1", "task_read", "i1"),
                task_call("send_1", "task_send", "i2", ask="Shall I send it to Dale?"),
            ]),
        ])
        loop, adapter = loop_factory(provider, "task_read", "task_send")

        outbound = await run_turn(
            loop, user_text, active_ix=["autonomy_level", "verbosity"],
            action_types={"task_read": "read", "task_send": "external_action"},
        )

        persisted = loop.sessions.get(SESSION_KEY).messages
        visible = [m for m in persisted if m["role"] in {"user", "assistant"} and m["content"]]
        assert [(m["role"], m["content"]) for m in visible] == [
            ("user", user_text),
            ("assistant", outbound.content),
        ]
        pool = adapter.events[0].messages
        # The two spoken lines keep their exact bytes and their own ends of the
        # turn; the PA's own acts travel between them as tagged blocks.
        assert pool[0] == {"role": "user", "content": user_text}
        assert pool[-1] == {"role": "assistant", "content": outbound.content}
        assert [m for m in pool if m["role"] == "user"] == [pool[0]]
        assert (
            "[ix_selection attribute=autonomy_level setting=Reactive outcome=executed]"
        ) in _blob(pool)

        # What stays out is what exists only because a benchmark is running,
        # and what the wrapper withheld. The gated call's *declaration* is in —
        # the PA did declare it (item #5) — but carrier C, which is wrapper
        # state rather than an argument the PA chose, is not.
        memory = _blob([event.messages for event in adapter.events])
        for internal in (
            TURN_REMINDER_PREFIX, HOLD_PROMPT_PREFIX,
            "_ix_confirmation_message", "interaction_item_id",
        ):
            assert internal not in memory
        assert '[tool_call name=task_send id=send_1]' in memory
        assert "[tool_result name=task_send" not in memory

    run(scenario())


def test_a_proactive_round_persists_only_the_delivered_message(loop_factory) -> None:
    """T3, proactive round: no user turn, so the envelope is assistant-only."""
    async def scenario() -> None:
        message = "Your pull list is waiting and the shop shuts at six."
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                ix(
                    "IX_proactive_outreach",
                    setting="High",
                    evidence="one relevant nudge is welcome",
                    application="send it",
                    outreach_action="send",
                    outreach_message=message,
                ),
            ]),
            LLMResponse(content="Internal completion text."),
        ])
        loop, adapter = loop_factory(provider)

        outbound = await loop.process_environment_event(
            trigger_id="pickup_deadline",
            source_type="deadline",
            prior_thread="The comic order is ready.",
            detail="The store closes in two hours.",
            user_impact="User A may want to collect it today.",
            session_key=SESSION_KEY,
            metadata={"run_id": "run_001", "persona_id": "user_a"},
            timestamp=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
        )

        assert outbound.content == message
        persisted = loop.sessions.get(SESSION_KEY).messages
        assert [(m["role"], m["content"]) for m in persisted] == [("assistant", message)]
        # Session history stores only what was delivered. The pool is the whole
        # turn: the trigger the PA perceived, its own selection and that
        # selection's result, then the contact it sent.
        assert [item.kind for item in adapter.events[0].items] == [
            "environment_event", "ix_selection", "ix_result", "assistant",
        ]
        assert adapter.events[0].messages[-1] == {
            "role": "assistant", "content": message,
        }
        # The environment event itself is internal and never becomes a user turn.
        assert "The store closes in two hours." not in _blob(persisted)
        assert "Internal completion text." not in _blob(persisted)
        assert "Internal completion text." not in _blob(adapter.events[0].messages)
        assert not [
            m for m in adapter.events[0].messages if m["role"] == "user"
        ]

    run(scenario())


def test_a_forced_event_exchange_yields_the_same_pool_shape() -> None:
    """T3, forced-event round: the same writer, so the same envelope shape.

    `harness/forced_event_executor.py` builds `session_messages` by hand and
    hands them to `build_turn_pool()` — the builder the live loop uses. A
    forced exchange therefore renders the way a live turn does: the spoken
    lines bare, the tool protocol as tagged blocks on assistant messages.
    """
    session_messages = [
        {
            "role": "user",
            "content": "Did the deposit clear?",
            "timestamp": "2026-10-01T09:00:00+00:00",
            "source": {"kind": "user"},
            "visibility": "visible",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "fe_1", "type": "function",
                "function": {"name": "mcp_state_document_read", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "fe_1", "name": "mcp_state_document_read",
         "content": "{\"ok\": true}"},
        {
            "role": "assistant",
            "content": "It cleared on Friday.",
            "timestamp": "2026-10-01T09:01:00+00:00",
            "source": {"kind": "pa"},
            "visibility": "visible",
        },
    ]

    pool = build_turn_pool(session_messages)

    assert pool.messages == [
        {"role": "user", "content": "Did the deposit clear?"},
        {
            "role": "assistant",
            "content": (
                "[tool_call name=mcp_state_document_read id=fe_1]\n{}\n[/tool_call]"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "[tool_result name=mcp_state_document_read id=fe_1 truncated=false]\n"
                '{"ok": true}\n'
                "[/tool_result]"
            ),
        },
        {"role": "assistant", "content": "It cleared on Friday."},
    ]
    assert [item.kind for item in pool.items] == [
        "user", "tool_call", "tool_result", "assistant",
    ]


# ---------------------------------------------------------------------------
# T4 / T9 · a held call in the middle of an executed run
# ---------------------------------------------------------------------------

def test_a_call_held_between_two_reads_leaves_an_ordered_protocol(loop_factory) -> None:
    """T4/T9 with the gate in the middle, not at the end.

    The loop-level pair in `tests/test_pa/test_ix_wrapper_loop.py` gates the
    last call. Here the gated call sits between two reads, so the filtered
    protocol has to keep *order* as well as membership: the read before the gate
    runs, the read after it does not, and the declared ids still line up with
    the result ids one for one.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Working through it.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                    {"id": "a3", "bound_item_ids": ["i3"]},
                ),
                task_call("read_1", "task_read", "i1"),
                task_call("send_1", "task_send", "i2", ask="Send it to Dale?"),
                task_call("read_2", "task_read", "i3"),
            ]),
            LLMResponse(content="Understood."),
            LLMResponse(content="Understood."),
        ])
        loop, adapter = loop_factory(provider, "task_read", "task_send")
        action_types = {"task_read": "read", "task_send": "external_action"}

        first = await run_turn(
            loop, "Check the stock, mail Dale, then check the invoice.",
            active_ix=["autonomy_level"], action_types=action_types,
        )

        assert executed_tools(first) == ["task_read"]
        persisted = loop.sessions.get(SESSION_KEY).messages
        declared = [
            call["id"] for message in persisted for call in message.get("tool_calls") or []
        ]
        results = [m["tool_call_id"] for m in persisted if m["role"] == "tool"]
        assert declared == results
        assert declared[-1] == "read_1"
        for dropped in ("send_1", "read_2"):
            assert dropped not in declared
        assert first.content.rstrip().endswith("Send it to Dale?")
        assert adapter.events[0].messages[-1]["content"] == first.content

        # T9: the next real request carries that protocol and is accepted.
        second_start = len(provider.requests)
        await run_turn(
            loop, "Not yet.", active_ix=["autonomy_level"], action_types=action_types,
        )
        sent = provider.requests[second_start]
        sent_declared = [
            call["id"] for message in sent for call in message.get("tool_calls") or []
        ]
        sent_results = [
            m["tool_call_id"] for m in sent if m.get("role") == "tool"
        ]
        assert sent_declared == sent_results
        assert "send_1" not in _blob(sent)

    run(scenario())


def test_close_ix_session_reports_the_open_episode_of_a_gate_turn(loop_factory) -> None:
    """T9's step-boundary companion: the gate left one episode open."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Checking.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Send it?"),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Mail Dale.", active_ix=["autonomy_level"],
            action_types={"task_send": "external_action"},
        )

        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == [
            ("autonomy_level", "p1")
        ]

    run(scenario())
