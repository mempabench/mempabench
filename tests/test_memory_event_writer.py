from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory.adapter import PromptHistoryEvictionPolicy
from memory.event_writer import (
    TOOL_RESULT_MAX_CHARS,
    build_turn_pool,
    record_adapter_write_commit,
    write_visible_memory_event,
)
from memory.events import MemoryEvent
from memory.results import WriteReceipt, WriteStatus
from memory.turn_experience import ExperienceItem
from pa.providers.base import ToolCallRequest


class RecordingAdapter:
    backend_name = "recording"
    prompt_history_eviction_policy = PromptHistoryEvictionPolicy.after_adapter_commit

    def __init__(self) -> None:
        self.scopes = []
        self.events = []

    async def setup_scope(self, scope) -> None:
        self.scopes.append(scope)

    async def record_event(self, event) -> WriteReceipt:
        self.events.append(event)
        return WriteReceipt(status=WriteStatus.committed)


# One turn as the loop leaves it: the person's text, an iteration that
# selected a preference, a second that called a world tool, and the reply.
TURN = [
    {"role": "system", "content": "you are the assistant"},
    {"role": "user", "content": "Any news on my order?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_01",
                "function": {
                    "name": "IX_uncertainty_expression",
                    "arguments": '{"setting":"Express"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_01",
        "name": "IX_uncertainty_expression",
        "content": '{"type":"ix_preference_selection","setting":"Express"}',
    },
    {
        "role": "assistant",
        "content": "Let me look it up.",
        "reasoning_content": "The order id is MB-55021; search the mailbox.",
        "tool_calls": [
            {
                "id": "call_02",
                "function": {
                    "name": "mcp_state_email_message_search",
                    "arguments": '{"query":"MB-55021"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_02",
        "name": "mcp_state_email_message_search",
        "content": '{"matches":["GLX722480"]}',
    },
    {"role": "assistant", "content": "It is out for delivery."},
    {"role": "user", "content": "reminder", "_mempa_internal": True},
]


def test_pool_carries_tool_calls_results_reasoning_and_ix() -> None:
    pool = build_turn_pool(TURN)

    assert [(item.kind, item.role) for item in pool.items] == [
        ("user", "user"),
        ("ix_selection", "assistant"),
        ("ix_result", "assistant"),
        ("reasoning", "assistant"),
        ("prose", "assistant"),
        ("tool_call", "assistant"),
        ("tool_result", "assistant"),
        ("assistant", "assistant"),
    ]
    assert [item.order for item in pool.items] == list(range(8))

    selection = pool.items[1]
    assert selection.attrs == {
        "attribute": "uncertainty_expression",
        "setting": "Express",
    }
    assert selection.body == '{"setting":"Express"}'
    assert pool.items[5].attrs == {
        "name": "mcp_state_email_message_search",
        "id": "call_02",
    }
    assert pool.items[6].attrs["truncated"] == "false"

    # The reasoning, the prose and the call of one iteration travel together;
    # every result is its own message; the delivered reply stays bare.
    assert [message["role"] for message in pool.messages] == [
        "user", "assistant", "assistant", "assistant", "assistant", "assistant",
    ]
    assert pool.messages[0]["content"] == "Any news on my order?"
    assert pool.messages[3]["content"].splitlines()[0] == (
        "[reasoning source=reasoning_content]"
    )
    assert "[prose]" in pool.messages[3]["content"]
    assert pool.messages[-1]["content"] == "It is out for delivery."
    assert all(message["content"] for message in pool.messages)


def test_pool_never_uses_user_role_for_non_user_text() -> None:
    pool = build_turn_pool(TURN)

    spoken = [item for item in pool.items if item.role == "user"]
    assert [item.body for item in spoken] == ["Any news on my order?"]
    assert [m for m in pool.messages if m["role"] == "user"] == [
        {"role": "user", "content": "Any news on my order?"}
    ]
    # The tool result the loop returned is a `role: "tool"` message the old
    # projection dropped outright; it now travels as a tagged assistant block,
    # which is the half of this rule the pool has to earn.
    assert {
        "role": "assistant",
        "content": (
            "[tool_result name=mcp_state_email_message_search id=call_02 truncated=false]\n"
            '{"matches":["GLX722480"]}\n'
            "[/tool_result]"
        ),
    } in pool.messages
    # The system prompt is identity and runtime rules, not experience, and an
    # `_mempa_internal` injection exists only because a benchmark is running.
    blob = "".join(message["content"] for message in pool.messages)
    assert "you are the assistant" not in blob
    assert "reminder" not in blob


def test_pool_carries_no_metadata_key() -> None:
    # Ground truth reaches the PA through `metadata`, never through a message.
    # The builder reads messages only, and projects just role and content —
    # on the non-dialogue messages the pool added as much as on the person's.
    labelled = [
        {
            "role": "user",
            "content": "Any news on my order?",
            "script_session_id": "acc_020",
            "acc_num": 20,
            "target_setting": "Express",
            "consent_decisions": {"gate": "granted"},
        },
        {
            "role": "tool",
            "tool_call_id": "call_01",
            "name": "mcp_state_order_search",
            "content": '{"matches":[]}',
            "script_session_id": "acc_020",
            "target_setting": "Express",
        },
    ]

    pool = build_turn_pool(labelled)

    assert pool.messages == [
        {"role": "user", "content": "Any news on my order?"},
        {
            "role": "assistant",
            "content": (
                "[tool_result name=mcp_state_order_search id=call_01 truncated=false]\n"
                '{"matches":[]}\n'
                "[/tool_result]"
            ),
        },
    ]
    blob = "".join(message["content"] for message in pool.messages)
    for banned in ("script_session_id", "acc_020", "target_setting", "consent_decisions"):
        assert banned not in blob
    assert [item.attrs for item in pool.items] == [
        {},
        {"name": "mcp_state_order_search", "id": "call_01", "truncated": "false"},
    ]


def test_json_bodied_blocks_carry_canonical_compact_json() -> None:
    # The provider serializes call arguments with `json.dumps` defaults, which
    # inserts spaces and keeps the model's key order (`pa/providers/base.py`);
    # the three JSON-bodied tags re-emit them in the one canonical form.
    selection = ToolCallRequest(
        id="call_01",
        name="IX_uncertainty_expression",
        arguments={"setting": "Express", "evidence": "asked for a firm date"},
    ).to_openai_tool_call()
    call = ToolCallRequest(
        id="call_02",
        name="mcp_state_email_message_search",
        arguments={"query": "MB-55021", "folder": "收件箱"},
    ).to_openai_tool_call()
    assert call["function"]["arguments"] == '{"query": "MB-55021", "folder": "收件箱"}'

    pool = build_turn_pool([
        {"role": "assistant", "content": "", "tool_calls": [selection, call]},
        {
            "role": "tool",
            "tool_call_id": "call_01",
            "name": "IX_uncertainty_expression",
            "content": '{"type": "ix_preference_selection", "setting": "Express"}',
        },
    ])

    assert pool.messages[0]["content"] == (
        "[ix_selection attribute=uncertainty_expression setting=Express]\n"
        '{"evidence":"asked for a firm date","setting":"Express"}\n'
        "[/ix_selection]\n"
        "[tool_call name=mcp_state_email_message_search id=call_02]\n"
        '{"folder":"收件箱","query":"MB-55021"}\n'
        "[/tool_call]"
    )
    assert pool.messages[1]["content"] == (
        "[ix_result name=IX_uncertainty_expression id=call_01]\n"
        '{"setting":"Express","type":"ix_preference_selection"}\n'
        "[/ix_result]"
    )


def test_unparsable_json_body_stays_the_verbatim_text() -> None:
    # A DSML-recovered call carries the model's raw text, not JSON. The act was
    # real, so the arguments go in as they stand rather than being dropped.
    raw = "query: MB-55021"

    pool = build_turn_pool([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_02",
                "function": {
                    "name": "mcp_state_email_message_search",
                    "arguments": raw,
                },
            }],
        },
    ])

    assert pool.messages[0]["content"] == (
        "[tool_call name=mcp_state_email_message_search id=call_02]\n"
        f"{raw}\n"
        "[/tool_call]"
    )


def test_truncated_is_true_only_when_the_loop_actually_cut_the_result() -> None:
    # `_save_turn` appends the marker only after cutting at
    # `TOOL_RESULT_MAX_CHARS`, so the marker on its own does not mean a cut.
    natural = "The courier's note ends\n... (truncated)"
    cut = "x" * (TOOL_RESULT_MAX_CHARS + 500)
    cut = cut[:TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"

    pool = build_turn_pool([
        {
            "role": "tool",
            "tool_call_id": "call_01",
            "name": "mcp_state_order_search",
            "content": natural,
        },
        {
            "role": "tool",
            "tool_call_id": "call_02",
            "name": "mcp_state_order_search",
            "content": cut,
        },
    ])

    assert [item.attrs["truncated"] for item in pool.items] == ["false", "true"]


def test_the_loop_cuts_at_the_length_the_pool_reads_back() -> None:
    # One constant, not two: the writer decides `truncated=` from the same
    # number the loop cuts at.
    from pa.agent.loop import AgentLoop

    assert AgentLoop._TOOL_RESULT_MAX_CHARS == TOOL_RESULT_MAX_CHARS


def test_the_acc_slice_reaches_the_pool_already_cut() -> None:
    # The cut belongs to the loop, not to this module: cutting here as well
    # would chop a further 16 characters off a result `_save_turn` had already
    # cut. `AgentLoop._cut_tool_result` is the one site, and both writers of a
    # tool result -- the persisted history and the pool input -- go through it.
    #
    # Found live: run `live2_pool_acc_001`, user_a acc_001, `simple_rag` --
    # `mcp_state_scholarly_paper_search` stored 25 348 characters with
    # `truncated=false`, because the pool was built from the raw provider slice.
    # The wiring is pinned end to end by
    # `tests/test_pa/test_agent_loop.py::test_the_acc_slice_reaches_the_pool_already_cut`;
    # this checks the helper the loop applies.
    from pa.agent.loop import AgentLoop

    oversized = "x" * (TOOL_RESULT_MAX_CHARS + 9_348)
    cut = AgentLoop.__new__(AgentLoop)._cut_tool_result(oversized)

    pool = build_turn_pool([
        {
            "role": "tool",
            "tool_call_id": "call_01",
            "name": "mcp_state_scholarly_paper_search",
            "content": cut,
        },
    ])

    item = pool.items[0]
    assert len(item.body) == TOOL_RESULT_MAX_CHARS + len("\n... (truncated)")
    assert item.attrs["truncated"] == "true"


@pytest.mark.asyncio
async def test_write_visible_memory_event_builds_shared_contract() -> None:
    adapter = RecordingAdapter()
    timestamp = datetime(2026, 12, 8, 18, 30, tzinfo=timezone.utc)

    receipt = await write_visible_memory_event(
        adapter=adapter,
        messages=[
            {"role": "user", "content": "Should I bring it?"},
            {
                "role": "tool",
                "tool_call_id": "call_01",
                "name": "mcp_state_weather_read",
                "content": "weather result",
            },
            {"role": "assistant", "content": "Yes."},
        ],
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
        session_id="run_001_main",
        turn_id="evolv_05:exchange_01",
        event_id="run_001:evolv_05:exchange_01",
        timestamp=timestamp,
        require_committed=True,
    )

    assert receipt is not None and receipt.status == WriteStatus.committed
    event = adapter.events[0]
    assert event.timestamp == timestamp
    assert event.messages == [
        {"role": "user", "content": "Should I bring it?"},
        {
            "role": "assistant",
            "content": (
                "[tool_result name=mcp_state_weather_read id=call_01 truncated=false]\n"
                "weather result\n"
                "[/tool_result]"
            ),
        },
        {"role": "assistant", "content": "Yes."},
    ]
    assert [item.kind for item in event.items] == ["user", "tool_result", "assistant"]


def test_event_carries_items_parallel_to_messages() -> None:
    items = [
        ExperienceItem(
            kind="user",
            role="user",
            order=0,
            offset_seconds=None,
            attrs={},
            body="Any news on the order?",
        ),
        ExperienceItem(
            kind="tool_result",
            role="assistant",
            order=1,
            offset_seconds=None,
            attrs={"name": "mcp_state_order_search", "id": "call_01", "truncated": "false"},
            body='{"matches":[]}',
        ),
    ]

    event = MemoryEvent(
        event_id="run_001:acc_020:turn_01",
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
        session_id="run_001_main",
        turn_id="acc_020:turn_01",
        timestamp=datetime(2026, 12, 8, 18, 30, tzinfo=timezone.utc),
        messages=[
            {"role": "user", "content": "Any news on the order?"},
            {
                "role": "assistant",
                "content": (
                    "[tool_result name=mcp_state_order_search id=call_01"
                    " truncated=false]\n"
                    '{"matches":[]}\n'
                    "[/tool_result]"
                ),
            },
        ],
        items=items,
    )

    # `messages` stays the canonical, lossless form; `items` is the same
    # content pre-parsed, for a backend whose native API carries structure.
    assert [item.order for item in event.items] == [0, 1]
    assert event.messages[0]["content"] == event.items[0].body
    assert event.messages[1]["content"].splitlines()[1] == event.items[1].body
    assert MemoryEvent(
        event_id="e",
        run_id="r",
        persona_id="p",
        agent_id="a",
        session_id="s",
        turn_id="t",
        timestamp=datetime(2026, 12, 8, 18, 30, tzinfo=timezone.utc),
        messages=[],
    ).items == []


def test_record_adapter_write_commit_advances_only_on_commit() -> None:
    index = record_adapter_write_commit(
        receipt=WriteReceipt(status=WriteStatus.committed),
        session_id="run_001_main",
        current_index=12,
        committed_index=4,
        eviction_policy=PromptHistoryEvictionPolicy.after_adapter_commit,
        require_committed=True,
    )

    assert index == 12


# ---------------------------------------------------------------------------
# `outcome=` on `[ix_selection]`, and F1's payload-argument rule
# ---------------------------------------------------------------------------

def test_ix_selection_outcome_comes_from_the_call_record() -> None:
    # The block is built from the executed request and its tool event,
    # not from the assistant message's serialized copy — which is written
    # before dedupe and before the voided fields are restored, so it can carry
    # an ignored duplicate or a blank `evidence` the registry never saw.
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_01",
                    "function": {
                        "name": "IX_uncertainty_expression",
                        "arguments": '{"setting":"Express","evidence":""}',
                    },
                },
                {
                    "id": "call_02",
                    "function": {
                        "name": "IX_uncertainty_expression",
                        "arguments": '{"setting":"Suppress"}',
                    },
                },
                {
                    "id": "call_03",
                    "function": {
                        "name": "IX_autonomy_level",
                        "arguments": '{"setting":"Nonsense"}',
                    },
                },
            ],
        },
    ]

    pool = build_turn_pool(
        messages,
        ix_selections=[
            {
                "id": "call_01",
                "name": "IX_uncertainty_expression",
                "args": {"setting": "Express", "evidence": "asked for a firm date"},
                "outcome": "executed",
            },
            {
                "id": "call_02",
                "name": "IX_uncertainty_expression",
                "args": {"setting": "Suppress"},
                "outcome": "ignored_duplicate",
            },
            {
                "id": "call_03",
                "name": "IX_autonomy_level",
                "args": {"setting": "Nonsense"},
                "outcome": "rejected",
            },
        ],
    )

    assert pool.messages[0]["content"] == (
        "[ix_selection attribute=uncertainty_expression setting=Express"
        " outcome=executed]\n"
        '{"evidence":"asked for a firm date","setting":"Express"}\n'
        "[/ix_selection]\n"
        "[ix_selection attribute=uncertainty_expression setting=Suppress"
        " outcome=ignored_duplicate]\n"
        '{"setting":"Suppress"}\n'
        "[/ix_selection]\n"
        "[ix_selection attribute=autonomy_level setting=Nonsense outcome=rejected]\n"
        '{"setting":"Nonsense"}\n'
        "[/ix_selection]"
    )


def test_no_selection_outcome_outside_the_three_the_grammar_names() -> None:
    with pytest.raises(ValueError):
        build_turn_pool(
            [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_01",
                    "function": {
                        "name": "IX_autonomy_level",
                        "arguments": '{"setting":"Reactive"}',
                    },
                }],
            }],
            ix_selections=[{
                "id": "call_01",
                "name": "IX_autonomy_level",
                "args": {"setting": "Reactive"},
                "outcome": "held",
            }],
        )


_RENDER_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{
        "id": "call_09",
        "function": {
            "name": "IX_render_information_elicitation",
            "arguments": '{"questions":[{"id":"q2","text":"HELD_SECRET_Q2"}]}',
        },
    }],
}


def test_an_executed_payload_call_keeps_the_text_it_delivered() -> None:
    # F1 (owner ruling 2026-09-13). The arguments of a released `IX_render_*`
    # call are the delivered text, and item #5 rules the act and its arguments
    # IN. Blanking them by tool name dropped what the PA actually said.
    pool = build_turn_pool(
        [_RENDER_CALL],
        ix_payload_status=(
            '{"status": "rendered", "released": ["q2"], "held_not_released": []}'
        ),
    )

    assert pool.messages[0]["content"] == (
        "[tool_call name=IX_render_information_elicitation id=call_09]\n"
        '{"questions":[{"id":"q2","text":"HELD_SECRET_Q2"}]}\n'
        "[/tool_call]"
    )


def test_a_payload_call_is_blanked_while_the_turn_still_holds_something() -> None:
    # O4: held prose stays out at hold time. The status is per turn, so one
    # held item blanks the turn's payload arguments; the act is still recorded.
    pool = build_turn_pool(
        [_RENDER_CALL],
        ix_payload_status=(
            '{"status": "rendered", "released": [], "held_not_released": ["q2"]}'
        ),
    )

    assert pool.messages[0]["content"] == (
        "[tool_call name=IX_render_information_elicitation id=call_09]\n"
        "[/tool_call]"
    )


def test_a_cancelled_payload_is_blanked_even_though_nothing_is_still_held() -> None:
    """F1(c): an empty `held_not_released` does not mean "delivered".

    A forced close *empties* the hold by cancelling what was in it,
    so on the cancelling turn the status reads `held_not_released:
    []` while the cancelled section reached nobody. Reading that list alone
    would write the wording of a question the PA never asked into memory as if
    it had been asked.
    """
    pool = build_turn_pool(
        [_RENDER_CALL],
        ix_payload_status=(
            '{"status": "rendered", "released": ["q1"], '
            '"held_not_released": [], "cancelled": ["q2"]}'
        ),
    )

    assert pool.messages[0]["content"] == (
        "[tool_call name=IX_render_information_elicitation id=call_09]\n"
        "[/tool_call]"
    )


def test_the_hold_status_is_read_off_the_turn_when_none_is_passed() -> None:
    # The forced-event line hands in `session.messages`, where the payload
    # result content was already replaced by `ix_payload_status`.
    held = '{"status": "rendered", "released": [], "held_not_released": ["q2"]}'
    pool = build_turn_pool([
        _RENDER_CALL,
        {
            "role": "tool",
            "tool_call_id": "call_09",
            "name": "IX_render_information_elicitation",
            "content": held,
        },
    ])

    assert "HELD_SECRET_Q2" not in pool.messages[0]["content"]


def test_a_raw_payload_echo_is_replaced_by_the_turn_status() -> None:
    # Item #9: what the PA perceived of a payload call is the hold status, not
    # the registry's echo of the whole payload back at it.
    status = '{"status": "rendered", "released": ["q2"], "held_not_released": []}'
    pool = build_turn_pool(
        [
            _RENDER_CALL,
            {
                "role": "tool",
                "tool_call_id": "call_09",
                "name": "IX_render_information_elicitation",
                "content": '{"questions":[{"id":"q2","text":"HELD_SECRET_Q2"}]}',
            },
        ],
        ix_payload_status=status,
    )

    assert pool.messages[1]["content"] == (
        "[ix_result name=IX_render_information_elicitation id=call_09]\n"
        '{"held_not_released":[],"released":["q2"],"status":"rendered"}\n'
        "[/ix_result]"
    )


def test_the_wrapper_carrier_fields_never_reach_a_tool_call_block() -> None:
    # Carrier C is wrapper state, stripped before the real tool runs
    # (`pa/agent/runner.py:3739-3743`). It exists only because a benchmark
    # enforces the IX contract, so it is not experience.
    from pa.agent.ix.render_types import PER_CALL_INJECTED_FIELDS

    from memory.event_writer import PER_CALL_INJECTED_FIELDS as POOL_FIELDS

    assert set(POOL_FIELDS) == set(PER_CALL_INJECTED_FIELDS)

    pool = build_turn_pool([{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "send_1",
            "function": {
                "name": "task_send",
                "arguments": (
                    '{"value":"i2","interaction_item_id":"i2",'
                    '"_ix_confirmation_message":"Shall I send it to Dale?"}'
                ),
            },
        }],
    }])

    assert pool.messages[0]["content"] == (
        "[tool_call name=task_send id=send_1]\n"
        '{"value":"i2"}\n'
        "[/tool_call]"
    )


# ---------------------------------------------------------------------------
# the environment branch, the delivered reply, `[no_reply]`
# ---------------------------------------------------------------------------

_ENVIRONMENT_BODY = (
    "[Environment event; no user request is currently active]\n"
    "Trigger ID: pickup_deadline\n"
    "Source type: deadline\n"
    "Prior thread: The comic order is ready for pickup.\n"
    "Event: The store closes in two hours.\n"
    "Potential user impact: User A may want time to collect the order today."
)


def test_the_environment_event_leads_the_pool_and_is_never_a_user_message() -> None:
    pool = build_turn_pool(
        [{"role": "system", "content": _ENVIRONMENT_BODY, "_mempa_internal": True}],
        environment_event={
            "source_id": "deadline",
            "action_id": "pickup_deadline",
            "body": _ENVIRONMENT_BODY,
        },
        delivered="The comic store closes at 6 PM today.",
    )

    assert [message["role"] for message in pool.messages] == ["assistant", "assistant"]
    assert pool.items[0].kind == "environment_event"
    assert pool.items[0].attrs == {
        "source_kind": "environment",
        "source_id": "deadline",
        "action_id": "pickup_deadline",
    }
    # The body is doubled-bracket escaped by the grammar, so it reads back.
    assert "Use IX_proactive_outreach" not in pool.messages[0]["content"]
    assert pool.messages[-1] == {
        "role": "assistant", "content": "The comic store closes at 6 PM today.",
    }


def test_nothing_delivered_writes_a_no_reply_block_with_a_non_empty_content() -> None:
    pool = build_turn_pool(
        [{"role": "assistant", "content": "Internal completion text."}],
        environment_event={
            "source_id": "deadline",
            "action_id": "pickup_deadline",
            "body": _ENVIRONMENT_BODY,
        },
        delivered="",
    )

    assert pool.items[-1].kind == "no_reply"
    assert pool.messages[-1] == {
        "role": "assistant",
        "content": "[no_reply decision=suppress]\n[/no_reply]",
    }
    assert all(message["content"] for message in pool.messages)


def test_the_delivered_reply_is_public_turn_and_never_the_raw_prose() -> None:
    # Q3: the pre-render text is the wrapper's input, not a delivered act, so
    # the final iteration contributes its reasoning and nothing else.
    pool = build_turn_pool(
        [
            {"role": "user", "content": "Any news?"},
            {
                "role": "assistant",
                "content": "raw pre-render prose",
                "reasoning_content": "the order is late",
            },
        ],
        delivered="It is out for delivery.",
    )

    assert [item.kind for item in pool.items] == ["user", "reasoning", "assistant"]
    blob = "".join(message["content"] for message in pool.messages)
    assert "raw pre-render prose" not in blob
    assert pool.messages[-1]["content"] == "It is out for delivery."


def test_a_subagent_result_is_tagged_from_data_not_from_its_role() -> None:
    pool = build_turn_pool(
        [{
            "role": "assistant",
            "content": "The sub-agent found three candidates.",
            "source_kind": "subagent",
        }],
        delivered="Here are the three candidates.",
    )

    assert [item.kind for item in pool.items] == ["subagent_result", "assistant"]
    assert pool.messages[0]["content"] == (
        "[subagent_result]\n"
        "The sub-agent found three candidates.\n"
        "[/subagent_result]"
    )


@pytest.mark.asyncio
async def test_a_given_pool_is_written_verbatim_and_never_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pool=` is the rendering, not a hint to be re-derived.

    The forced-event line renders its own exchange because two things it needs
    are not readable off a message dict -- a delivered line's `source.kind`,
    which decides between `[environment_event]` and `[third_party]`, and each
    item's `offset_seconds`, which reaches the wire as `offset=`. If the writer
    rebuilt the pool from `messages` it would silently drop both: the tag would
    become bare `user` dialogue -- a third party's words written down as the
    person's -- and every offset would vanish.

    So this pins two things at once. The builder is made to raise, so any call
    to it fails the test rather than quietly producing something plausible; and
    the `messages` handed in are ones whose rebuilt pool would visibly differ,
    so the assertion still means something if the monkeypatch ever stops
    biting.
    """
    import memory.event_writer as event_writer

    def _must_not_run(messages):
        raise AssertionError(
            "build_turn_pool was called even though a pool was supplied: "
            f"{messages!r}"
        )

    monkeypatch.setattr(event_writer, "build_turn_pool", _must_not_run)

    delivered = ExperienceItem(
        kind="third_party",
        role="assistant",
        order=1,
        offset_seconds=90,
        attrs={"source_kind": "third_party", "source_id": "Glenmont Trail Office"},
        body="The lower loop is closed on the fourteenth.",
    )
    spoken = ExperienceItem(
        kind="user",
        role="user",
        order=0,
        offset_seconds=None,
        attrs={},
        body="Which entrance stays open?",
    )
    pool = event_writer.TurnPool(
        messages=[
            {"role": "user", "content": spoken.body},
            {
                "role": "assistant",
                # Spelled out, not built with the renderer under test (F5).
                "content": (
                    '[third_party source_kind=third_party'
                    ' source_id="Glenmont Trail Office" offset=90]\n'
                    "The lower loop is closed on the fourteenth.\n"
                    "[/third_party]"
                ),
            },
        ],
        items=[spoken, delivered],
    )
    # The same content as the session saw it: rebuilt, the delivered line would
    # come back as bare `role: "user"` text with no tag and no offset.
    session_messages = [
        {"role": "user", "content": spoken.body},
        {
            "role": "user",
            "content": delivered.body,
            "source": {"kind": "third_party", "id": "Glenmont Trail Office"},
            "offset_seconds": 90,
        },
    ]
    adapter = RecordingAdapter()

    receipt = await write_visible_memory_event(
        adapter=adapter,
        messages=session_messages,
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
        session_id="run_001_main",
        turn_id="evolv_05:exchange_04",
        event_id="run_001:evolv_05:exchange_04",
        timestamp=datetime(2026, 12, 8, 18, 30, tzinfo=timezone.utc),
        require_committed=True,
        pool=pool,
    )

    assert receipt is not None and receipt.status == WriteStatus.committed
    event = adapter.events[0]
    assert event.messages == pool.messages
    assert event.items == pool.items
    # The two things a rebuild would have lost.
    assert event.messages[1]["content"].startswith("[third_party ")
    assert event.messages[1]["content"].split("\n")[0].endswith("offset=90]")
    assert [item.kind for item in event.items] == ["user", "third_party"]
    assert delivered.body not in [
        message["content"]
        for message in event.messages
        if message["role"] == "user"
    ]
