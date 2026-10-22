"""T11, T12 — the two projections that must not carry the wrapper across.

`tests/test_harness/test_judge_pack.py` covers one worked session
for T11; here the same scan runs once per rendered attribute, so no attribute's
own vocabulary can slip through on a session that happened not to be the
example. T12 puts a normal exchange and a forced-event exchange side by side and
asks whether the memory they produce is the same shape.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from harness.judge_pack import (
    IX_SETTING_LABELS,
    build_text_judge_pack,
    scan_banned_keys,
    scan_banned_strings,
)
from memory.event_writer import write_visible_memory_event
from pa.agent.ix.render_types import (
    IX_CONTRACT_VERSION,
    IX_RENDER_SCHEMA_VERSION,
    RENDERED_ATTRIBUTES,
)
from pa.agent.ix.specs import IX_PASSIVE_SPECS
from pa.providers.base import LLMResponse

from .conftest import (
    SESSION_KEY,
    RecordingMemoryIO,
    ScriptProvider,
    ix,
    plan_call,
    run,
    run_turn,
    selection,
    task_call,
)

HELD_BODY = "The hotel wants a deposit by Friday; I have not booked it."
GATE_REASON = "external_action_floor"
PROVIDER_REASONING = "The user sounds annoyed, so I will keep this short."


def _blob(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# T11 · one banned-field / banned-string scan per rendered attribute
# ---------------------------------------------------------------------------

def _ix_render_blob(attribute: str, setting: str) -> dict:
    """An `ix_render` record whose every slot names this attribute or setting."""
    trace_line = f"{attribute}={setting} -> release 1/2"
    return {
        "schema_version": IX_RENDER_SCHEMA_VERSION,
        "ix_contract_version": IX_CONTRACT_VERSION,
        "episodes": {attribute: {"episode_id": "e1", "rounds_open": 1, "closed": False}},
        "released": {"batches": ["b1"], "questions": [], "plan_executed": ["a1"]},
        "held": {
            "batches": [{"batch_id": "b2", "label": "hotel", "body": HELD_BODY}],
            "questions": [],
            "plan_remaining": [{"id": "a3", "intent": "send the email"}],
            "calls": [{
                "interaction_item_id": "i3",
                "plan_item_id": "a3",
                "tool": "mcp_state_email_message_send",
                "reason": GATE_REASON,
                "status": "held_for_confirmation",
            }],
            "cancelled": [],
            "outreach_message": "",
        },
        "gate": {
            "attribute": attribute,
            "item_id": "a3",
            "reason": GATE_REASON,
            "action_type": "external_action",
            "message_source": "_ix_confirmation_message",
        },
        "consent": [{
            "attribute": attribute,
            "item_id": "a3",
            "pa_decision": "release",
            "sideband_decision": "allow",
            "released": True,
            "user_evidence": "go ahead",
            "evidence_verified": True,
        }],
        "notices": [{"order": 0, "tool": "mcp_state_document_read", "text": "Checking."}],
        "completion": {"released": False, "text": "All set."},
        "reasoning": {
            "shown": True,
            "source": "user_facing_rationale",
            "provider_reasoning_present": True,
        },
        "certainties": [{"ref": "c1", "level": "low", "rendered": "(confidence: low)"}],
        "proactive_decision": None,
        "trace": [trace_line],
    }


def _session_events(attribute: str, setting: str) -> list[dict]:
    return [
        {"t": 0, "kind": "session_start", "session_id": "s1", "persona": "User A"},
        {"t": 1, "kind": "beat_enter", "beat": "open", "active_skills": [attribute]},
        {
            "t": 2,
            "kind": "simulator_turn",
            "beat": "open",
            "message": "Book the hotel and mail the team.",
            "turn_assessment": {"next_beat": "stay"},
        },
        {
            "t": 3,
            "kind": "pa_turn",
            "beat": "open",
            "outbound": "I read your trip note. Shall I send the email to the team?",
            "tool_events": [
                {
                    "name": f"IX_{attribute}",
                    "status": "ok",
                    "ix": True,
                    "attribute": attribute,
                    "setting": setting,
                    "evidence": "The user asked for it last time.",
                    "application": "Apply it to this turn.",
                    "args": {"setting": setting},
                },
                {
                    "name": f"IX_render_{attribute}",
                    "status": "ok",
                    "args": {"plan_id": "p1"},
                },
                {
                    "name": "mcp_state_document_read",
                    "status": "ok",
                    "args": {
                        "path": "trip.md",
                        "_ix_notice": "I am checking your trip note.",
                        "interaction_item_id": "i1",
                    },
                    "detail": "trip contents",
                },
                {"process": True, "stage": "start", "message": "legacy control event"},
            ],
            "ix_render": _ix_render_blob(attribute, setting),
            "provider_reasoning": [PROVIDER_REASONING],
        },
        {"t": 4, "kind": "session_end", "simulator_turns": 1, "beat_count": 1},
    ]


def _settings_for(attribute: str) -> list[str]:
    return list(IX_PASSIVE_SPECS[attribute].settings)


ATTRIBUTE_SETTINGS = [
    (attribute, setting)
    for attribute in sorted(RENDERED_ATTRIBUTES)
    for setting in _settings_for(attribute)
]


@pytest.mark.parametrize("attribute,setting", ATTRIBUTE_SETTINGS)
def test_the_text_pack_leaks_no_attribute_and_no_setting(
    attribute: str, setting: str
) -> None:
    """T11 (projection 4) once per rendered attribute and setting."""
    events = _session_events(attribute, setting)

    pack = build_text_judge_pack(events)
    text = _blob(pack)

    assert scan_banned_keys(pack) == []
    assert scan_banned_strings(
        pack, needles=(HELD_BODY, GATE_REASON, PROVIDER_REASONING),
    ) == []
    for banned in (
        attribute, setting, "ix_render", "IX_", HELD_BODY, GATE_REASON,
        PROVIDER_REASONING, "interaction_item_id", "_ix_notice",
    ):
        assert banned not in text
    # The one thing that does survive is the public turn and the executed task.
    pa_turn = pack["turns"][1]
    assert pa_turn["public_turn"].startswith("I read your trip note.")
    assert [event["name"] for event in pa_turn["task_tool_events"]] == [
        "mcp_state_document_read"
    ]
    assert pa_turn["task_tool_events"][0]["args"] == {"path": "trip.md"}


@pytest.mark.parametrize("attribute,setting", ATTRIBUTE_SETTINGS)
def test_the_ix_face_of_the_same_session_keeps_all_of_it(
    attribute: str, setting: str
) -> None:
    """The removals are a packing rule, not a loss of evidence."""
    ix_face = _blob(_session_events(attribute, setting))

    for needle in (attribute, setting, HELD_BODY, GATE_REASON, "ix_render"):
        assert needle in ix_face


def test_the_vocabulary_the_scan_uses_covers_every_rendered_attribute() -> None:
    """A scan is only as good as its word list, so pin the list to the taxonomy."""
    for attribute, setting in ATTRIBUTE_SETTINGS:
        assert attribute in IX_SETTING_LABELS
        assert setting in IX_SETTING_LABELS


def test_a_real_gated_turn_packs_clean(loop_factory) -> None:
    """The same scan against a pack built from a turn the wrapper really ran."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Checking the stock first.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                ),
                task_call("read_1", "task_read", "i1"),
                task_call("send_1", "task_send", "i2", ask="Shall I send it to Dale?"),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_read", "task_send")

        outbound = await run_turn(
            loop, "Check the stock and mail Dale.", active_ix=["autonomy_level"],
            action_types={"task_read": "read", "task_send": "external_action"},
        )

        events = [
            {
                "t": 0,
                "kind": "simulator_turn",
                "beat": "open",
                "message": "Check the stock and mail Dale.",
            },
            {
                "t": 1,
                "kind": "pa_turn",
                "beat": "open",
                "outbound": outbound.content,
                "tool_events": outbound.metadata["tool_events"],
                "ix_render": outbound.metadata["ix_render"],
            },
        ]
        pack = build_text_judge_pack(events)

        assert scan_banned_keys(pack) == []
        assert scan_banned_strings(pack, needles=("external_action_floor",)) == []
        text = _blob(pack)
        assert "Reactive" not in text
        assert "autonomy_level" not in text
        # The ask is part of what the user read, so it stays.
        assert "Shall I send it to Dale?" in pack["turns"][1]["public_turn"]
        # The held call did not run, so it is not in the executed list.
        names = [event["name"] for event in pack["turns"][1]["task_tool_events"]]
        assert "task_read" in names
        assert "task_send" not in names

    run(scenario())


def test_the_text_pack_carries_no_wrapper_control_event(loop_factory) -> None:
    """T11's banned-field half, stated over the events the wrapper really emits.

    A row named `beat_gate_turn` or `ix_consent_denied` in the task tool list
    tells the setting-blind judge that a gate fired and that consent was
    refused — the two facts projection 4 exists to withhold. The names
    carry no attribute string, so the vocabulary scan cannot see them; the
    filter has to.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Checking the stock first.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Mail Dale.", active_ix=["autonomy_level"],
            action_types={"task_send": "external_action"},
        )

        pack = build_text_judge_pack([{
            "t": 0,
            "kind": "pa_turn",
            "beat": "open",
            "outbound": outbound.content,
            "tool_events": outbound.metadata["tool_events"],
            "ix_render": outbound.metadata["ix_render"],
        }])

        packed = [event["name"] for event in pack["turns"][0]["task_tool_events"]]
        emitted = {
            event["name"] for event in outbound.metadata["tool_events"]
            if event.get("ix_event")
        }
        assert emitted  # the turn really did emit wrapper control events
        assert not (emitted & set(packed))

    run(scenario())


# ---------------------------------------------------------------------------
# T12 · normal and forced-event memory are the same schema
# ---------------------------------------------------------------------------

FORCED_EXCHANGE = [
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
    {
        "role": "tool", "tool_call_id": "fe_1",
        "name": "mcp_state_document_read", "content": "{\"ok\": true}",
    },
    {
        "role": "assistant",
        "content": "It cleared on Friday.",
        "timestamp": "2026-10-01T09:01:00+00:00",
        "source": {"kind": "pa"},
        "visibility": "visible",
    },
]


def test_normal_and_forced_event_exchanges_write_the_same_memory_schema(
    loop_factory,
) -> None:
    """T12 (projection 3): one memory schema on both paths.

    Both paths go through `build_turn_pool()`, so a persona's memory must not
    show two different kinds of turn. The pool carries the PA's own experience
    now, the IX selection included, so what the scan below asks is narrower:
    nothing the wrapper *withheld* may reach the backend.
    """
    async def scenario() -> None:
        memory_io = RecordingMemoryIO()
        provider = ScriptProvider([
            LLMResponse(content="Checking.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Shall I send it?"),
            ]),
        ])
        loop, adapter = loop_factory(provider, "task_send", memory_io=memory_io)

        outbound = await run_turn(
            loop, "Mail Dale the extension request.", active_ix=["autonomy_level"],
            action_types={"task_send": "external_action"},
        )

        # The forced-event half: the executor's own call, same adapter.
        await write_visible_memory_event(
            adapter=adapter,
            messages=FORCED_EXCHANGE,
            run_id="run_001",
            persona_id="user_a",
            agent_id="nanobot",
            session_id=SESSION_KEY,
            turn_id="evolv_01:exchange_00",
            event_id="run_001:evolv_01:exchange_00",
            timestamp=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
            step_id="evolv_01",
            audit_sink=memory_io,
            require_committed=True,
        )

        normal, forced = adapter.events[0], adapter.events[1]
        assert type(normal) is type(forced)
        assert vars(normal).keys() == vars(forced).keys()
        for event in (normal, forced):
            assert {message["role"] for message in event.messages} <= {
                "user", "assistant",
            }
            assert all(set(m) == {"role", "content"} for m in event.messages)
        # The person's text and the delivered reply keep their exact bytes on
        # both paths; everything between them is a tagged block.
        assert normal.messages[0] == {
            "role": "user", "content": "Mail Dale the extension request.",
        }
        assert normal.messages[-1] == {
            "role": "assistant", "content": outbound.content,
        }
        # The forced exchange's tool call is the PA's own act and now travels
        # with it, tagged, never as the person's speech. Spelled out on the
        # wire rather than re-derived from the builder under test.
        assert forced.messages == [
            {"role": "user", "content": "Did the deposit clear?"},
            {
                "role": "assistant",
                "content": (
                    "[tool_call name=mcp_state_document_read id=fe_1]\n"
                    "{}\n"
                    "[/tool_call]"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "[tool_result name=mcp_state_document_read id=fe_1"
                    " truncated=false]\n"
                    '{"ok": true}\n'
                    "[/tool_result]"
                ),
            },
            {"role": "assistant", "content": "It cleared on Friday."},
        ]

        # What the wrapper withheld reaches no backend. The gated call's
        # declaration is the PA's own act and now travels (item #5), but
        # carrier C is wrapper state and does not — and no result is invented
        # for a call that never ran.
        assert memory_io.records
        for record in memory_io.records:
            payload = _blob({
                key: value for key, value in record.items()
                # `audit_only_labels` is the audit plane's own row and carries
                # the beat's active attribute list by design.
                if key != "audit_only_labels"
            })
            for banned in (
                "_ix_notice", "_ix_confirmation_message", "interaction_item_id",
                "[tool_result name=task_send",
            ):
                assert banned not in payload, record["operation"]

    run(scenario())


# ---------------------------------------------------------------------------
# `held` and `released` describe *this* turn
# ---------------------------------------------------------------------------

ACTION_TYPES = {"task_send": "external_action"}


def test_held_reports_open_episodes_and_released_gets_its_writers(
    loop_factory,
) -> None:
    """The two halves of the `ix_render` blob have to agree about the same turn.

    `_held_json` read `open_episodes` **and** `finished_episodes`, and a
    closing episode is filed with its `plan_remaining` intact — so live, a
    booking that had already happened was still reported held three turns
    later, on the same blobs whose `episodes` said `{}`. `released` had no
    writer at all: `batches` / `questions` / `plan_executed` were `[]` on all
    81 PA turns of nine runs, including turns where a plan item plainly ran.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Drafting it.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "intent": "mail Dale", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            LLMResponse(content="Sending it.", tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "yes, send it"},
                ]),
                task_call("send_2", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            LLMResponse(content="Sent."),
            LLMResponse(content="Nothing else outstanding."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        gated = await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )
        released = await run_turn(
            loop, "Yes, send it.", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )
        later = await run_turn(
            loop, "Anything else?", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
        )

        gate_blob = gated.metadata["ix_render"]
        assert [item["id"] for item in gate_blob["held"]["plan_remaining"]] == ["a1"]
        assert gate_blob["released"] == {
            "batches": [], "questions": [], "plan_executed": [],
        }

        done_blob = released.metadata["ix_render"]
        assert done_blob["released"]["plan_executed"] == ["a1"]
        # `episodes` and `held` are two views of one turn; they used to disagree.
        assert done_blob["episodes"] == {}
        assert done_blob["held"]["plan_remaining"] == []
        assert done_blob["held"]["calls"] == []

        assert later.metadata["ix_render"]["held"]["plan_remaining"] == []
        assert later.metadata["ix_render"]["episodes"] == {}

    run(scenario())


def test_the_ix_render_blob_still_reaches_only_the_ix_face(loop_factory) -> None:
    """`released` is an audit row like the rest: the text face never sees it."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "One-at-a-time"),
                ix(
                    "IX_render_topic_management",
                    batch_set_id="bs1",
                    item_count=2,
                    batches=[
                        {"batch_id": "b1", "label": "hotel", "call_binding": []},
                        {"batch_id": "b2", "label": "invoice", "call_binding": []},
                    ],
                ),
            ]),
            LLMResponse(
                content=f"[[batch:b1]] The hotel is booked.\n\n[[batch:b2]] {HELD_BODY}",
            ),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Hotel and invoice, please.", active_ix=["topic_management"],
        )

        blob = outbound.metadata["ix_render"]
        assert blob["released"]["batches"] == ["b1"]
        assert [batch["batch_id"] for batch in blob["held"]["batches"]] == ["b2"]
        assert HELD_BODY not in outbound.content

        pack = build_text_judge_pack([
            {"t": 1, "kind": "pa_turn", "beat": "b", "outbound": outbound.content,
             "tool_events": outbound.metadata["tool_events"],
             "ix_render": blob},
        ])
        assert scan_banned_keys(pack) == []
        assert scan_banned_strings(pack, needles=[HELD_BODY, "bs1", "b2"]) == []

    run(scenario())
