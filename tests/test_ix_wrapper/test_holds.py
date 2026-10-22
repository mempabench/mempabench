"""T17, T19, T20 — the step boundary, the proactive four-way, the action map.

Plan §7.2 P7. Each of these three crosses a package line: the hold cache is
`pa/`'s but the step boundary is `harness/`'s; the proactive decision is
rendered in `pa/` and has to look the same in four different places; the action
map is `pa/`'s but the tool surface it must cover is the world toolbox's.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from harness.pa_factory import (
    OpenHoldEpisodeError,
    RecordingPA,
    assert_benchmark_action_type_coverage,
    held_items_from_ix_render,
)
from harness.judge_pack import build_text_judge_pack
from harness.transcript import TranscriptWriter, load_events
from memory.event_writer import build_turn_pool
from pa.agent.ix.action_types import (
    DEFAULT_TOOL_ACTION_TYPES,
    UNKNOWN_TOOL_ACTION_TYPE,
    tool_action_type,
)
from pa.agent.ix.collector import assert_action_type_coverage, hold_prompt
from pa.agent.ix.render_types import (
    HOLD_ROUNDS_HARD_CAP,
    Batch,
    HoldEpisode,
    IXHold,
    Question,
)
from pa.agent.ix.registry import TurnToolRegistry
from pa.agent.tools.registry import ToolRegistry
from pa.providers.base import LLMResponse

from .conftest import (
    SCRIPT_SESSION,
    SESSION_KEY,
    EchoTool,
    ScriptProvider,
    executed_tools,
    ix,
    plan_call,
    run,
    run_turn,
    selection,
    task_call,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTION_TYPES = {"task_send": "external_action"}


def _blob(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _gate_responses() -> list[LLMResponse]:
    return [
        LLMResponse(content="Drafting it.", tool_calls=[
            selection("autonomy_level", "Reactive"),
            plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
            task_call("send_1", "task_send", "i1", ask="Shall I send it to Dale?"),
        ]),
    ]


# ---------------------------------------------------------------------------
# T17 · the step boundary and the retry after it
# ---------------------------------------------------------------------------

def test_the_harness_step_boundary_sees_the_open_episode_through_the_recording_pa(
    loop_factory,
) -> None:
    """T17 (a): the seam the orchestrator closes a step with (§4.2, contract 7).

    `harness/orchestrator.py` calls `close_ix_session(step_id)` on its PA
    handle inside the step's `finally` and raises `OpenHoldEpisodeError` on a
    non-empty answer, before any checkpoint is published. That handle is
    `RecordingPA`, so this is the whole path from the wrapper's hold cache to
    the error that fails the step.
    """
    async def scenario() -> None:
        loop, _ = loop_factory(ScriptProvider(_gate_responses()), "task_send")
        recording = RecordingPA(loop)

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        still_open = recording.close_ix_session(SESSION_KEY, SCRIPT_SESSION)
        assert still_open == [("autonomy_level", "p1")]

        error = OpenHoldEpisodeError(SCRIPT_SESSION, still_open)
        assert error.episodes == still_open
        assert "no checkpoint was published" in str(error)
        assert "autonomy_level=p1" in str(error)

    run(scenario())


def test_a_retry_after_a_failed_step_starts_with_no_hold_and_no_setting(
    loop_factory,
) -> None:
    """T17 (b) + (c): the retry runs a second attempt, not a resumed one.

    Keeping the chosen setting without its hold — or the reverse — would leave
    two state planes describing different turns.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            *_gate_responses(),
            # The retry: the model has to choose the setting again, because the
            # selection cache went out with the hold.
            LLMResponse(content="Drafting it.", tool_calls=[
                selection("autonomy_level", "Autonomous"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_2", "task_send", "i1"),
            ]),
            LLMResponse(content="Sent."),
        ])
        loop, _ = loop_factory(provider, "task_send")
        key = (SESSION_KEY, SCRIPT_SESSION)

        await run_turn(
            loop, "Mail Dale.", active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )
        assert key in loop._ix_hold_cache
        assert key in loop._ix_selection_cache

        loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION)
        assert key not in loop._ix_hold_cache
        assert key not in loop._ix_selection_cache

        retry_start = len(provider.tool_definitions)
        second = await run_turn(
            loop, "Mail Dale.", active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        # The selection tool is offered again — nothing was inherited …
        assert "IX_autonomy_level" in provider.offered(retry_start)
        # … and the hold prompt from the first attempt is nowhere in the retry.
        assert "You still hold" not in _blob(provider.requests[retry_start])
        assert second.metadata["ix_render"]["held"]["calls"] == []

    run(scenario())


# ---------------------------------------------------------------------------
# T19 · the proactive decision has to look the same in four places
# ---------------------------------------------------------------------------

async def _proactive(loop_factory, *, action: str, setting: str, message: str):
    provider = ScriptProvider([
        LLMResponse(content=None, tool_calls=[
            ix(
                "IX_proactive_outreach",
                setting=setting,
                evidence="the matrix says so",
                application=f"{action} this contact",
                outreach_action=action,
                outreach_message=message,
            ),
        ]),
        LLMResponse(content="Internal completion text."),
    ])
    loop, adapter = loop_factory(provider)
    recording = RecordingPA(loop)
    outbound = await recording.process_environment_event(
        trigger_id="pickup_deadline",
        source_type="deadline",
        prior_thread="The comic order is ready.",
        detail="The store closes in two hours.",
        user_impact="User A may want to collect it today.",
        session_key=SESSION_KEY,
        metadata={"run_id": "run_001", "persona_id": "user_a"},
        timestamp=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
    )
    return loop, adapter, outbound


def _transcript_events(tmp_path: Path, outbound: Any) -> list[dict]:
    run_dir = tmp_path / "transcript_run"
    with TranscriptWriter(run_dir) as writer:
        writer.log_event({
            "t": 0,
            "kind": "pa_turn",
            "beat": "proactive_trigger",
            "proactive_event": True,
            "outbound": outbound.content,
            "outreach_decision": outbound.metadata["outreach_decision"],
            "outreach_delivered": outbound.metadata["outreach_delivered"],
            "tool_events": outbound.metadata["tool_events"],
            "ix_render": outbound.metadata["ix_render"],
        })
    return load_events(run_dir / "transcript.jsonl")


def test_a_suppressed_reminder_exists_only_on_the_audit_plane(
    loop_factory, tmp_path: Path
) -> None:
    """T19, `Low` / suppress: four planes, and the candidate is in exactly one."""
    async def scenario() -> None:
        message = "The comic store closes at 6 tonight."
        loop, adapter, outbound = await _proactive(
            loop_factory, action="suppress", setting="Low", message=message,
        )

        # 1 · what the simulator is handed
        assert outbound.content == ""
        assert outbound.metadata["outreach_decision"] == "suppress"
        assert outbound.metadata["outreach_delivered"] is False
        assert "held_items" not in outbound.metadata
        # 2 · PA session history — nothing was delivered, so nothing is stored
        assert loop.sessions.get(SESSION_KEY).messages == []
        # 3 · the memory adapter: the decision to stay silent is remembered,
        #     because `proactive_outreach` is scored on exactly that decision;
        #     the candidate wording is not, so the plane count still holds.
        assert len(adapter.events) == 1
        assert [item.kind for item in adapter.events[0].items] == [
            "environment_event", "ix_selection", "ix_result", "no_reply",
        ]
        assert message not in _blob(adapter.events[0].messages)
        # 4 · the transcript: the candidate is an IX-face fact, and the text
        #     face of the same transcript has no word of it.
        events = _transcript_events(tmp_path, outbound)
        assert message in _blob(events)
        assert events[0]["outbound"] == ""
        assert message not in _blob(build_text_judge_pack(events))
        ix_render = outbound.metadata["ix_render"]
        assert ix_render["held"]["outreach_message"] == message
        assert ix_render["proactive_decision"] == {
            "action": "suppress", "setting": "Low", "message": message,
        }

    run(scenario())


def test_a_sent_reminder_is_the_same_text_on_every_plane(
    loop_factory, tmp_path: Path
) -> None:
    """T19, `High` / send: the rendered decision is the only source for all four."""
    async def scenario() -> None:
        message = "Your pull list is ready and the shop shuts at six."
        loop, adapter, outbound = await _proactive(
            loop_factory, action="send", setting="High", message=message,
        )

        assert outbound.content == message
        assert outbound.metadata["outreach_decision"] == "send"
        assert outbound.metadata["outreach_delivered"] is True
        assert [
            (m["role"], m["content"]) for m in loop.sessions.get(SESSION_KEY).messages
        ] == [("assistant", message)]
        # The pool carries the whole turn; the contact is its delivered reply.
        assert [item.kind for item in adapter.events[0].items] == [
            "environment_event", "ix_selection", "ix_result", "assistant",
        ]
        assert adapter.events[0].messages[-1] == {
            "role": "assistant", "content": message,
        }
        assert _transcript_events(tmp_path, outbound)[0]["outbound"] == message
        # The PA's internal completion text is not the contact and never ships.
        assert "Internal completion text." not in _blob(
            loop.sessions.get(SESSION_KEY).messages
        )
        assert "Internal completion text." not in _blob(adapter.events[0].messages)

    run(scenario())


# ---------------------------------------------------------------------------
# T20 · the standing tool surface is fully classified
# ---------------------------------------------------------------------------

def test_every_standing_world_tool_has_a_declared_action_type() -> None:
    """T20 (§3.8, contract 10) against the real toolbox, not a stub.

    `assert_benchmark_action_type_coverage` is the check the harness runs once
    at benchmark start; it reads the world-toolbox catalog, so an unclassified
    tool fails the run before any step touches the world.
    """
    tools = assert_benchmark_action_type_coverage(PROJECT_ROOT)

    assert len(tools) > 50
    assert set(tools) <= set(DEFAULT_TOOL_ACTION_TYPES)
    assert {tool_action_type(name) for name in tools} <= {
        "read", "draft", "internal_write", "external_action",
    }


def test_a_turn_registrys_task_tools_are_all_in_the_action_map() -> None:
    """The same subset relation stated the way §3.8 states it: registry ⊆ map."""
    base = ToolRegistry()
    for name in assert_benchmark_action_type_coverage(PROJECT_ROOT):
        base.register(EchoTool(name))
    registry = TurnToolRegistry(
        base,
        ["autonomy_level"],
        ix_context={"session_active_ix": ["autonomy_level", "process_visibility"]},
    )

    task_tools = {name for name in registry.tool_names if not name.startswith("IX_")}

    assert task_tools
    assert task_tools <= set(DEFAULT_TOOL_ACTION_TYPES)
    assert assert_action_type_coverage(registry.tool_names) is None


def test_an_unclassified_tool_is_gated_as_the_tightest_class_and_refused_up_front() -> None:
    """T20's fail-closed half: unknown is `external_action`, and it still fails.

    The fail-closed floor is the runtime backstop; the start-up check is what
    makes sure it is never the thing actually deciding.
    """
    assert UNKNOWN_TOOL_ACTION_TYPE == "external_action"
    assert tool_action_type("brand_new_tool") == "external_action"

    with pytest.raises(ValueError, match="brand_new_tool"):
        assert_action_type_coverage(["email_message_send", "brand_new_tool"])


# ---------------------------------------------------------------------------
# §3.4 (part 3) · a withheld batch is a hold, not a one-turn pause
# ---------------------------------------------------------------------------

BATCH_ACTION_TYPES = {"task_write": "internal_write", "task_send": "external_action"}


def _batches(*bound: tuple[str, str]) -> Any:
    """`IX_render_topic_management` for `(batch_id, interaction_item_id)` pairs."""
    return ix(
        "IX_render_topic_management",
        batch_set_id="bs1",
        item_count=len(bound),
        batches=[
            {"batch_id": batch_id, "label": batch_id, "call_binding": [item_id]}
            for batch_id, item_id in bound
        ],
    )


def _batch_prose(*bound: tuple[str, str]) -> str:
    return "Here is where things stand.\n\n" + "\n\n".join(
        f"[[batch:{batch_id}]] Item {item_id} is settled." for batch_id, item_id in bound
    )


def test_one_at_a_time_holds_the_rest_across_three_turns(loop_factory) -> None:
    """The batch gate holds calls *and* remembers them (§3.4 (part 3), §3.7 (part 2)).

    Live, `One-at-a-time` held `email_message_send` for exactly one turn and
    then forgot it: no episode was opened, `IX_continue_topic_management` was
    never required, `held.batches` and `released.batches` were `[]` on every
    turn, and the simulator's explicit verdict had nothing to land on — while
    the PA told the user the held email had been sent.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            # §3.4 (part 5): the calls arrive in a message with no prose of its own,
            # and the marked prose is written in the closing iteration the
            # wrapper asks for once the hold is recorded.
            LLMResponse(
                content=None,
                tool_calls=[
                    selection("topic_management", "One-at-a-time"),
                    _batches(("b1", "c1"), ("b2", "c2"), ("b3", "c3")),
                    task_call("w1", "task_write", "c1"),
                    task_call("s2", "task_send", "c2", ask="Send Dr. Okafor the note?"),
                    task_call("w3", "task_write", "c3"),
                ],
            ),
            LLMResponse(content=_batch_prose(("b1", "c1"), ("b2", "c2"), ("b3", "c3"))),
            LLMResponse(content=None, tool_calls=[
                _batches(("b2", "c2"), ("b3", "c3")),
                ix("IX_continue_topic_management", decisions=[
                    {"id": "c2", "decision": "release", "user_evidence": "yes, send it"},
                    {"id": "c3", "decision": "keep", "user_evidence": ""},
                ]),
                task_call("s2b", "task_send", "c2", ask="Send Dr. Okafor the note?"),
            ]),
            LLMResponse(content=_batch_prose(("b2", "c2"), ("b3", "c3"))),
            LLMResponse(content=None, tool_calls=[
                _batches(("b3", "c3")),
                ix("IX_continue_topic_management", decisions=[
                    {"id": "c3", "decision": "release", "user_evidence": "and the invoice"},
                ]),
                task_call("w3b", "task_write", "c3"),
            ]),
            LLMResponse(content=_batch_prose(("b3", "c3"))),
        ])
        loop, _ = loop_factory(provider, "task_write", "task_send")

        first = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor and file the invoice.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )
        second_start = len(provider.tool_definitions)
        second = await run_turn(
            loop, "Go ahead with the email.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
            consent=[{"interaction_item_id": "c2", "decision": "allow"}],
        )
        third = await run_turn(
            loop, "And the invoice.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        turns = [first, second, third]
        # One batch per turn, and the world action happened on exactly one.
        assert [executed_tools(turn) for turn in turns] == [
            ["task_write"], ["task_send"], ["task_write"]
        ]
        assert [
            turn.metadata["ix_render"]["released"]["batches"] for turn in turns
        ] == [["b1"], ["b2"], ["b3"]]
        assert [
            [batch["batch_id"] for batch in turn.metadata["ix_render"]["held"]["batches"]]
            for turn in turns
        ] == [["b2", "b3"], ["b3"], []]

        # Turn one: a real cross-turn hold, on the topic episode.
        blob = first.metadata["ix_render"]
        assert blob["episodes"]["topic_management"]["episode_id"] == "bs1"
        assert [call["interaction_item_id"] for call in blob["held"]["calls"]] == ["c2", "c3"]
        assert {call["status"] for call in blob["held"]["calls"]} == {"held_for_batch"}
        assert {call["episode_id"] for call in blob["held"]["calls"]} == {"bs1"}
        assert held_items_from_ix_render(blob) == [
            {"interaction_item_id": "c2", "intent": ""},
            {"interaction_item_id": "c3", "intent": ""},
        ]
        # Turn two: `c2` was released and ran, `c3` was answered `keep`. No call
        # was stopped on this turn, so `rendered.held` is empty — but the episode
        # still carries `c3`, and `held.calls` has to say so the way
        # `held.batches` already does. Reading this turn's holds alone reported
        # `calls: []` next to `batches: [b3]` on the same blob (live `v11s02` t7).
        second_blob = second.metadata["ix_render"]
        assert [
            call["interaction_item_id"] for call in second_blob["held"]["calls"]
        ] == ["c3"]
        assert {call["episode_id"] for call in second_blob["held"]["calls"]} == {"bs1"}
        assert {call["status"] for call in second_blob["held"]["calls"]} == {"held_for_batch"}
        # … so the continuation tool is on the table for the next turn …
        assert "IX_continue_topic_management" in provider.offered(second_start)
        # … and the held external action ran only after the sideband allowed it.
        consent = second.metadata["ix_render"]["consent"]
        assert [(row["item_id"], row["sideband_decision"], row["released"]) for row in consent] == [
            ("c2", "allow", True)
        ]
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def test_a_denied_batch_item_stays_held(loop_factory) -> None:
    """§3.7 (part 3): the PA's `release` is intent; the sideband is the gate."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "One-at-a-time"),
                _batches(("b1", "c1"), ("b2", "c2")),
                task_call("w1", "task_write", "c1"),
                task_call("s2", "task_send", "c2", ask="Send the note?"),
            ]),
            LLMResponse(content=_batch_prose(("b1", "c1"), ("b2", "c2"))),
            LLMResponse(content=None, tool_calls=[
                _batches(("b2", "c2")),
                ix("IX_continue_topic_management", decisions=[
                    {"id": "c2", "decision": "release", "user_evidence": "send the note"},
                ]),
                task_call("s2b", "task_send", "c2", ask="Send the note?"),
            ]),
            LLMResponse(content=_batch_prose(("b2", "c2"))),
        ])
        loop, _ = loop_factory(provider, "task_write", "task_send")

        await run_turn(
            loop, "File the note and mail it.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Do not send that email; I will retrieve it myself.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
            consent=[{"interaction_item_id": "c2", "decision": "deny"}],
        )

        assert executed_tools(second) == []
        assert [
            (row["pa_decision"], row["sideband_decision"], row["released"])
            for row in second.metadata["ix_render"]["consent"]
        ] == [("release", "deny", False)]
        assert any(
            event.get("name") == "ix_consent_denied"
            for event in second.metadata["tool_events"]
        )

    run(scenario())


# ---------------------------------------------------------------------------
# §3.3 (part 2)·A · one release, one execution
# ---------------------------------------------------------------------------

def test_a_released_id_carries_one_world_changing_call_only(loop_factory) -> None:
    """A second send under the same released id is refused (finding 4).

    `validate_bindings` resets its duplicate check per provider response, and a
    turn is many responses. Live, one `allow` on `ix_01` bought both an
    `appointment_create` and a `calendar_event_create`, and the rendered ask
    had named only the booking.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            *_gate_responses(),
            LLMResponse(content="Sending it.", tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "yes, send it"},
                ]),
                task_call("send_2", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            # Same id, same turn, a second world-changing call.
            LLMResponse(content="And one more.", tool_calls=[
                task_call("send_3", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            LLMResponse(content="Sent."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Yes, send it.", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        assert executed_tools(second) == ["task_send"]
        spent = [
            event for event in second.metadata["tool_events"]
            if event.get("name") == "ix_release_spent"
        ]
        assert [event["item_id"] for event in spent] == ["i1"]
        # A refused duplicate is not a new hold: answering it would mean
        # releasing the same id again, which is exactly what was refused.
        assert second.metadata["ix_render"]["episodes"] == {}
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def test_one_authorisation_writes_one_consent_row(loop_factory) -> None:
    """Round-3 finding 5: `consent[]` counted calls, not authorisations.

    `v3g02` t24 carried four rows for one `allow` on `call_3` — the read, the
    write it bought, and the two duplicates `ix_release_spent` refused. The
    flags were individually right, but a judge counting authorisations off
    `consent[]` over-counted four to one. One row per (attribute, item); what
    each call did stays in `held.calls` and in the notices.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            *_gate_responses(),
            # Three calls, one id, three responses — `validate_bindings`
            # refuses a duplicate id inside a single response, so this is the
            # shape the live turn had: a read that costs nothing, the write the
            # release bought, and a duplicate of that write.
            LLMResponse(content="Reading it back.", tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "yes, send it"},
                ]),
                task_call("read_2", "task_read", "i1"),
            ]),
            LLMResponse(content="Sending it.", tool_calls=[
                task_call("send_2", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            LLMResponse(content="And one more.", tool_calls=[
                task_call("send_3", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            LLMResponse(content="Sent."),
        ])
        loop, _ = loop_factory(provider, "task_send", "task_read")
        action_types = {**ACTION_TYPES, "task_read": "read"}

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=action_types,
        )
        second = await run_turn(
            loop, "Yes, send it.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        consent = second.metadata["ix_render"]["consent"]
        assert [
            (row["attribute"], row["item_id"], row["sideband_decision"], row["released"])
            for row in consent
        ] == [("autonomy_level", "i1", "allow", True)]
        # The refused duplicate is still on the record, just not as consent.
        assert [
            event["item_id"] for event in second.metadata["tool_events"]
            if event.get("name") == "ix_release_spent"
        ] == ["i1"]
        assert executed_tools(second) == ["task_read", "task_send"]
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def _one_batch(*item_ids: str, item_count: int) -> Any:
    """`IX_render_topic_management` with a single batch binding every call."""
    return ix(
        "IX_render_topic_management",
        batch_set_id="bs1",
        item_count=item_count,
        batches=[{
            "batch_id": "batch_001",
            "label": "everything",
            "call_binding": list(item_ids),
        }],
    )


def test_one_undivided_batch_is_repaired_once(loop_factory) -> None:
    """Round-2 finding 3: `One-at-a-time` bypassed by declaring one batch.

    Live, a user asking for a calendar entry *and* an email got one batch
    (`batch_001`) binding all three calls, so the first batch was the whole turn
    and nothing was held. The repair asks for the split before anything runs.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "One-at-a-time"),
                _one_batch("c1", "c2", item_count=2),
                task_call("w1", "task_write", "c1"),
                task_call("s2", "task_send", "c2", ask="Send Dr. Okafor the note?"),
            ]),
            LLMResponse(content=None, tool_calls=[
                _batches(("b1", "c1"), ("b2", "c2")),
                task_call("w1b", "task_write", "c1"),
                task_call("s2b", "task_send", "c2", ask="Send Dr. Okafor the note?"),
            ]),
            LLMResponse(content=_batch_prose(("b1", "c1"), ("b2", "c2"))),
        ])
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        # The repair went in, and the undivided response executed nothing.
        repairs = [
            message for message in provider.requests[-1]
            if "one batch binds every call" in str(message.get("content") or "")
        ]
        assert len(repairs) == 1
        assert executed_tools(outbound) == ["task_write"]
        assert outbound.metadata["ix_render"]["released"]["batches"] == ["b1"]
        assert [
            call["interaction_item_id"]
            for call in outbound.metadata["ix_render"]["held"]["calls"]
        ] == ["c2"]
        assert not [
            event for event in outbound.metadata["tool_events"]
            if event.get("name") == "ix_batch_undivided"
        ]

    run(scenario())


def test_a_second_undivided_batch_is_accepted_and_recorded(loop_factory) -> None:
    """One repair, not a loop: the wrapper cannot name the topics for the model."""
    async def scenario() -> None:
        def undivided(suffix: str) -> list:
            return [
                _one_batch("c1", "c2", item_count=2),
                task_call(f"w1{suffix}", "task_write", "c1"),
                task_call(f"s2{suffix}", "task_send", "c2", ask="Send the note?"),
            ]
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "One-at-a-time"), *undivided(""),
            ]),
            LLMResponse(content=None, tool_calls=undivided("b")),
            LLMResponse(content="Both are done."),
        ])
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        event = next(
            event for event in outbound.metadata["tool_events"]
            if event.get("name") == "ix_batch_undivided"
        )
        assert (event["item_count"], event["batches"]) == (2, 1)
        # Accepted, not failed: the batch releases and the turn is delivered.
        assert executed_tools(outbound) == ["task_write", "task_send"]
        assert outbound.content
        assert outbound.metadata["ix_render"]["released"]["batches"] == ["batch_001"]

    run(scenario())


# ---------------------------------------------------------------------------
# §3.3 (part 2)·D addendum · a batch hold's narration is withheld, its released batch is not
# ---------------------------------------------------------------------------

def test_a_batch_hold_withholds_the_narration_and_keeps_the_released_batch(
    loop_factory,
) -> None:
    """`One-at-a-time` held the email and the same message said it had gone out.

    The framing prose beside the held call goes to `held.narration`; the
    `[[batch:b1]]` section, whose call really ran, is delivered as before.
    """
    async def scenario() -> None:
        narration = (
            "Done — the email to Dr. Okafor has gone out.\n\n"
            "[[batch:b1]] The deadline is on your calendar."
        )
        provider = ScriptProvider([
            LLMResponse(content=narration, tool_calls=[
                selection("topic_management", "One-at-a-time"),
                _batches(("b1", "c1"), ("b2", "c2")),
                task_call("w1", "task_write", "c1"),
                task_call("s2", "task_send", "c2", ask="Send the note?"),
            ]),
            LLMResponse(content=None),
        ])
        loop, adapter = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        assert executed_tools(outbound) == ["task_write"]
        assert "has gone out" not in outbound.content
        assert "The deadline is on your calendar." in outbound.content
        assert "has gone out" not in adapter.events[0].messages[-1]["content"]
        blob = outbound.metadata["ix_render"]
        assert [row["item_id"] for row in blob["held"]["narration"]] == ["c2"]
        assert blob["released"]["batches"] == ["b1"]

    run(scenario())


# ---------------------------------------------------------------------------
# §3.4 (part 3) · a batch or question hold ends the turn, the way a gate does
# ---------------------------------------------------------------------------

_CLOSING_TEXT = (
    "Here is where things stand.\n\n"
    "[[batch:b1]] Item c1 is settled.\n\n"
    "[[batch:b2]] Shall I mail Dr. Okafor the note?"
)


def _held_result_seen(provider: ScriptProvider, index: int = -1) -> list[str]:
    """The synthetic `held` results the request at `index` carried."""
    return [
        str(message.get("tool_call_id"))
        for message in provider.requests[index]
        if message.get("role") == "tool" and '"status": "held"' in str(message.get("content"))
    ]


def test_a_batch_hold_asks_one_closing_message_and_narrates_only_what_ran(
    loop_factory,
) -> None:
    """The narration that comes *after* the held call is asked for once, closed.

    Live (`measurements_gates.md` round 3, `v3g03`), deepseek wrote the held
    `email_message_send` in a tool-call message with **empty content** and its
    narration in the next, text-only message: `📧 Email confirmation sent to
    Dr. Okafor`. The same-message rule (§3.3 (part 3)·a) cannot see that shape. So
    the turn stops the work at the hold and asks for exactly that message under
    conditions where it can do no harm (§3.4 (part 5)): no task tool is on the table,
    the held call's result says it did not run, and only the released batch's
    section is delivered.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            # The batch payload, before the calls.
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "One-at-a-time"),
                _batches(("b1", "c1"), ("b2", "c2")),
            ]),
            # (a) the held call, in a message with no content of its own.
            LLMResponse(content=None, tool_calls=[
                task_call("w1", "task_write", "c1"),
                task_call("s2", "task_send", "c2", ask="Send Dr. Okafor the note?"),
            ]),
            # (b) the closing message, requested with task tools withdrawn.
            LLMResponse(content=_CLOSING_TEXT),
            LLMResponse(content=None, tool_calls=[
                _batches(("b2", "c2")),
                ix("IX_continue_topic_management", decisions=[
                    {"id": "c2", "decision": "release", "user_evidence": "yes, send it"},
                ]),
                task_call("s2b", "task_send", "c2", ask="Send Dr. Okafor the note?"),
            ]),
            LLMResponse(content=_batch_prose(("b2", "c2"))),
        ])
        loop, adapter = loop_factory(provider, "task_write", "task_send")

        first = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        # The turn ended after (b): two scripted responses are still unused.
        assert len(provider.responses) == 2
        assert executed_tools(first) == ["task_write"]
        # The closing request: no task tool, and the held call said so itself.
        assert [
            name for name in provider.offered(-1) if not name.startswith("IX_")
        ] == []
        assert _held_result_seen(provider) == ["s2"]
        # Only the released batch's section ships; b2 stays held with its ask.
        assert "Item c1 is settled." in first.content
        assert "Dr. Okafor" not in first.content
        assert "Dr. Okafor" not in adapter.events[0].messages[-1]["content"]
        blob = first.metadata["ix_render"]
        assert blob["released"]["batches"] == ["b1"]
        assert [call["interaction_item_id"] for call in blob["held"]["calls"]] == ["c2"]
        assert {call["status"] for call in blob["held"]["calls"]} == {"held_for_batch"}
        # The episode is open, so the next turn resumes it through `IX_continue`.
        assert blob["episodes"]["topic_management"]["closed"] is False

        second = await run_turn(
            loop, "Go ahead with the email.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
            consent=[{"interaction_item_id": "c2", "decision": "allow"}],
        )
        assert executed_tools(second) == ["task_send"]
        assert "Item c2 is settled." in second.content
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def _hold_turn_responses(closing: LLMResponse) -> list[LLMResponse]:
    """The `One-at-a-time` hold shape, with `closing` as the closing iteration."""
    return [
        LLMResponse(content=None, tool_calls=[
            selection("topic_management", "One-at-a-time"),
            _batches(("b1", "c1"), ("b2", "c2")),
        ]),
        LLMResponse(content=_batch_prose(("b1", "c1"), ("b2", "c2")), tool_calls=[
            task_call("w1", "task_write", "c1"),
            task_call("s2", "task_send", "c2", ask="Send Dr. Okafor the note?"),
        ]),
        closing,
    ]


def test_a_closing_iteration_that_calls_tools_anyway_is_used_for_its_text_only(
    loop_factory,
) -> None:
    """Nothing runs after the hold: the closing round's calls are dropped."""
    async def scenario() -> None:
        provider = ScriptProvider(_hold_turn_responses(
            LLMResponse(content=_CLOSING_TEXT, tool_calls=[
                task_call("s3", "task_send", "c2", ask="Send Dr. Okafor the note?"),
            ]),
        ))
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        assert provider.responses == []
        assert executed_tools(outbound) == ["task_write"]
        assert "Item c1 is settled." in outbound.content
        assert [
            event.get("tools") for event in outbound.metadata["tool_events"]
            if event.get("name") == "ix_hold_closing_tool_call_ignored"
        ] == [["task_send"]]

    run(scenario())


def test_a_closing_iteration_with_no_text_leaves_the_turn_where_the_hold_did(
    loop_factory,
) -> None:
    """The fallback is unchanged: the prose the turn already had, markers only."""
    async def scenario() -> None:
        provider = ScriptProvider(_hold_turn_responses(LLMResponse(content=None)))
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        assert provider.responses == []
        assert executed_tools(outbound) == ["task_write"]
        # §3.3 (part 3)·a still cuts the framing prose it shared with the held call.
        assert "Item c1 is settled." in outbound.content
        assert "Here is where things stand." not in outbound.content

    run(scenario())


def test_a_gate_turn_asks_for_no_closing_message(loop_factory) -> None:
    """§3.4 (part 5) is the hold exit only: a gate's ask is already the delivered text."""
    async def scenario() -> None:
        provider = ScriptProvider(_gate_responses())
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )

        # One request, so no closing iteration was made.
        assert len(provider.requests) == 1
        assert executed_tools(outbound) == []
        assert "Shall I send it to Dale?" in outbound.content
        assert _held_result_seen(provider) == []

    run(scenario())


def test_organize_holds_nothing_so_the_turn_runs_to_its_own_end(loop_factory) -> None:
    """The stop belongs to the hold, not to the attribute: `Organize` keeps going."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "Organize"),
                _batches(("b1", "c1"), ("b2", "c2")),
                task_call("w1", "task_write", "c1"),
                task_call("s2", "task_send", "c2"),
            ]),
            LLMResponse(content=_batch_prose(("b1", "c1"), ("b2", "c2"))),
        ])
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        assert provider.responses == []
        assert executed_tools(outbound) == ["task_write", "task_send"]
        assert outbound.metadata["ix_render"]["released"]["batches"] == ["b1", "b2"]
        assert outbound.metadata["ix_render"]["held"]["calls"] == []

    run(scenario())


def test_a_question_hold_still_ends_the_turn_through_the_gate(loop_factory) -> None:
    """§3.4 (part 2) is unchanged: the asking turn already stopped, and still does."""
    async def scenario() -> None:
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
                    task_call("s1", "task_send", "i1"),
                ],
            ),
            LLMResponse(content="Booked!"),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Book something.", active_ix=["information_elicitation"],
            action_types=ACTION_TYPES,
        )

        assert len(provider.responses) == 1
        assert executed_tools(outbound) == []
        assert "which date works?" in outbound.content
        assert "Booked!" not in outbound.content
        assert [
            call["status"] for call in outbound.metadata["ix_render"]["held"]["calls"]
        ] == ["held_for_questions"]

    run(scenario())


# ---------------------------------------------------------------------------
# §3.7 (part 3) · a release has to issue its call in the same reply
# ---------------------------------------------------------------------------

def _release_turn_two(*, with_call: bool, tail: str) -> list[LLMResponse]:
    decisions = ix("IX_continue_autonomy_level", decisions=[
        {"id": "i1", "decision": "release", "user_evidence": "yes, send it"},
    ])
    calls = [decisions]
    if with_call:
        calls.append(task_call("send_2", "task_send", "i1", ask="Shall I send it to Dale?"))
    return [LLMResponse(content=None, tool_calls=calls), LLMResponse(content=tail)]


def _release_repairs(provider: ScriptProvider) -> list[str]:
    """Every release repair of the turn, read off the last request's history."""
    last = provider.requests[-1] if provider.requests else []
    return [
        str(message.get("content"))
        for message in last
        if str(message.get("content") or "").startswith("[interaction wrapper] You reported")
    ]


def test_a_release_that_carries_its_call_executes(loop_factory) -> None:
    """The clean shape: released, called, executed, episode closed."""
    async def scenario() -> None:
        provider = ScriptProvider([
            *_gate_responses(),
            *_release_turn_two(with_call=True, tail="Sent."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Go ahead.", active_ix=["autonomy_level"], action_types=ACTION_TYPES,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        assert executed_tools(second) == ["task_send"]
        assert _release_repairs(provider) == []
        assert second.metadata["ix_render"]["episodes"] == {}
        assert not any(
            event.get("name") == "ix_release_missing"
            for event in second.metadata["tool_events"]
        )

    run(scenario())


def test_a_release_without_a_call_is_repaired_once_and_then_executes(loop_factory) -> None:
    """Live (`v3g03` t15), `email_01` was released and no call followed.

    Releases are never replayed (§3.3 (part 3)), so the episode closed and the email
    the user had just authorised was never sent. One repair asks for the call.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            *_gate_responses(),
            LLMResponse(content=None, tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "yes, send it"},
                ]),
            ]),
            LLMResponse(content=None, tool_calls=[
                task_call("send_2", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            LLMResponse(content="Sent."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Go ahead.", active_ix=["autonomy_level"], action_types=ACTION_TYPES,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        repairs = _release_repairs(provider)
        assert len(repairs) == 1 and "i1" in repairs[0]
        assert executed_tools(second) == ["task_send"]
        assert not any(
            event.get("name") == "ix_release_missing"
            for event in second.metadata["tool_events"]
        )
        assert second.metadata["ix_render"]["episodes"] == {}

    run(scenario())


def test_a_release_without_a_call_twice_keeps_the_item_held(loop_factory) -> None:
    """The repair is asked once. A second miss holds the id instead of losing it."""
    async def scenario() -> None:
        provider = ScriptProvider([
            *_gate_responses(),
            *_release_turn_two(with_call=False, tail="I will take care of it."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Go ahead.", active_ix=["autonomy_level"], action_types=ACTION_TYPES,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        assert len(_release_repairs(provider)) == 1
        assert executed_tools(second) == []
        missing = [
            event for event in second.metadata["tool_events"]
            if event.get("name") == "ix_release_missing"
        ]
        assert [event["item_id"] for event in missing] == ["i1"]
        blob = second.metadata["ix_render"]
        assert [
            (call["interaction_item_id"], call["reason"]) for call in blob["held"]["calls"]
        ] == [("i1", "release_missing")]
        # The episode is still open, so the id can still be acted on next turn.
        assert blob["episodes"]["autonomy_level"]["closed"] is False

    run(scenario())


def test_the_repaired_release_call_runs_before_the_payload_reminder(
    loop_factory,
) -> None:
    """The repair is answered with the call and no render payload — it runs.

    The conditional-required check is evaluated before the response's tool
    calls execute, and a task call makes `IX_render_topic_management` required.
    So the response the release repair asked for was discarded for a missing
    payload, and since the repair is sent only once, the send the user had
    authorised was never made. The payload is still owed on the next iteration.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "One-at-a-time"),
                _batches(("b1", "c1"), ("b2", "c2")),
                task_call("w1", "task_write", "c1"),
                task_call("s2", "task_send", "c2", ask="Send Dr. Okafor the note?"),
            ]),
            LLMResponse(content=_batch_prose(("b1", "c1"), ("b2", "c2"))),
            # The release, with no call of its own: one repair follows.
            LLMResponse(content=None, tool_calls=[
                ix("IX_continue_topic_management", decisions=[
                    {"id": "c2", "decision": "release", "user_evidence": "yes, send it"},
                ]),
            ]),
            # The repair answered: the call, and no `IX_render_topic_management`.
            LLMResponse(content=None, tool_calls=[
                task_call("s2b", "task_send", "c2", ask="Send Dr. Okafor the note?"),
            ]),
            LLMResponse(content="Sent."),
        ])
        loop, _ = loop_factory(provider, "task_write", "task_send")

        await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Go ahead with the email.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
            consent=[{"interaction_item_id": "c2", "decision": "allow"}],
        )

        assert len(_release_repairs(provider)) == 1
        assert executed_tools(second) == ["task_send"]
        assert not any(
            event.get("name") == "ix_release_missing"
            for event in second.metadata["tool_events"]
        )
        assert second.metadata["ix_render"]["held"]["calls"] == []
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


# ---------------------------------------------------------------------------
# §3.3 (part 2)·D · a held row says which gate held it, whether or not one opened
# ---------------------------------------------------------------------------

def test_a_batch_hold_records_its_own_attribute_reason_and_status(loop_factory) -> None:
    """Live (`v5g01` t12), a batch-held row read `attribute: ""`, `reason: ""`.

    A batch hold opens no gate, and the row was derived from the gate, so the
    audit could not say which episode was holding the call — while the episode
    itself was holding it under `topic_management` all along.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_hold_turn_responses(LLMResponse(content=_CLOSING_TEXT)))
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        blob = outbound.metadata["ix_render"]
        assert blob["gate"] is None
        assert [
            (call["interaction_item_id"], call["attribute"], call["reason"], call["status"])
            for call in blob["held"]["calls"]
        ] == [("c2", "topic_management", "batch_not_released", "held_for_batch")]
        # The same fact on the episode the next turn resumes.
        episode = loop._ix_hold_cache[
            (SESSION_KEY, SCRIPT_SESSION)
        ].open_episodes["topic_management"]
        assert [call["interaction_item_id"] for call in episode.calls_held] == ["c2"]
        assert [call["reason"] for call in episode.calls_held] == ["batch_not_released"]

    run(scenario())


def test_the_held_status_is_the_one_the_runner_ended_the_turn_on(loop_factory) -> None:
    """Live (`v5g01` t19/t22/t25): `held_for_confirmation` on turns with no gate.

    The renderer read the status off the delivered sections, so a hold turn
    whose prose carried no `[[batch:…]]` marker was filed as an autonomy
    confirmation — the one status that means "a gate asked the user" — while
    the runner had ended the turn on `held_for_batch`.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_hold_turn_responses(
            LLMResponse(content="Both items are in hand."),
        ))
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        blob = outbound.metadata["ix_render"]
        assert blob["held"]["batches"] == []
        assert blob["gate"] is None
        assert [call["status"] for call in blob["held"]["calls"]] == ["held_for_batch"]
        # …which is the status the runner's own hold record carried, and the
        # one that took the `ix_hold` exit.
        assert blob["stop_reason"] == "ix_hold"

    run(scenario())


def test_a_batch_release_without_its_call_is_repaired_once_and_then_executes(
    loop_factory,
) -> None:
    """§3.7 (part 3) on the batch path. Live, `v5g01` released `c2` three times and
    never sent it: the repair is the only thing that asks for the call back."""
    async def scenario() -> None:
        provider = ScriptProvider([
            *_hold_turn_responses(LLMResponse(content=_CLOSING_TEXT)),
            # The release arrives with no task call behind it.
            LLMResponse(content=None, tool_calls=[
                _batches(("b2", "c2")),
                ix("IX_continue_topic_management", decisions=[
                    {"id": "c2", "decision": "release", "user_evidence": "yes, send it"},
                ]),
            ]),
            # …and the repair gets it.
            LLMResponse(content=None, tool_calls=[
                task_call("s2b", "task_send", "c2", ask="Send Dr. Okafor the note?"),
            ]),
            LLMResponse(content=_batch_prose(("b2", "c2"))),
        ])
        loop, _ = loop_factory(provider, "task_write", "task_send")

        await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Go ahead with the email.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
            consent=[{"interaction_item_id": "c2", "decision": "allow"}],
        )

        assert len(_release_repairs(provider)) == 1
        assert executed_tools(second) == ["task_send"]
        assert not any(
            event.get("name") == "ix_release_missing"
            for event in second.metadata["tool_events"]
        )
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


# ---------------------------------------------------------------------------
# §3.4 (part 5) · the closing iteration leaves a record, and what it delivers counts
# ---------------------------------------------------------------------------

def test_the_closing_iteration_and_the_stop_reason_are_on_the_record(
    loop_factory,
) -> None:
    """Neither had any trace before: `ix_hold` had to be derived from held
    statuses and from how much shorter the closing request's prompt was."""
    async def scenario() -> None:
        provider = ScriptProvider(_hold_turn_responses(LLMResponse(content=_CLOSING_TEXT)))
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        blob = outbound.metadata["ix_render"]
        assert blob["stop_reason"] == "ix_hold"
        assert blob["closing_iteration"] is True
        assert "closing iteration ran" in blob["trace"]

    run(scenario())


def test_an_empty_closing_iteration_says_so(loop_factory) -> None:
    async def scenario() -> None:
        provider = ScriptProvider(_hold_turn_responses(LLMResponse(content=None)))
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        blob = outbound.metadata["ix_render"]
        assert blob["closing_iteration"] is False
        assert "closing iteration empty" in blob["trace"]

    run(scenario())


def test_a_turn_that_holds_nothing_records_neither_key(loop_factory) -> None:
    """Both keys are optional: absent means no hold exit and no closing round."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection("topic_management", "Organize"),
                _batches(("b1", "c1")),
                task_call("w1", "task_write", "c1"),
            ]),
            LLMResponse(content=_batch_prose(("b1", "c1"))),
        ])
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        blob = outbound.metadata["ix_render"]
        assert "closing_iteration" not in blob
        assert not [line for line in blob["trace"] if line.startswith("closing iteration")]

    run(scenario())


# ---------------------------------------------------------------------------
# `ix_claims_unexecuted_action` · structural, but not on a turn that delivered
# ---------------------------------------------------------------------------

def _claims_events(outbound: Any) -> list[dict]:
    return [
        event for event in outbound.metadata["tool_events"]
        if event.get("name") == "ix_claims_unexecuted_action"
    ]


def test_a_hold_turn_that_narrates_the_released_batch_makes_no_claim(
    loop_factory,
) -> None:
    """§3.4 (part 5) puts prose back on exactly these turns, so a structural test that
    only asks "was the message non-empty" now fires by construction: live,
    `v5g01` scored 3, and two of the three were honest text about the batch
    that did run. A released batch is what the prose is about."""
    async def scenario() -> None:
        provider = ScriptProvider(_hold_turn_responses(LLMResponse(content=_CLOSING_TEXT)))
        loop, _ = loop_factory(provider, "task_write", "task_send")

        outbound = await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )

        assert outbound.metadata["ix_render"]["released"]["batches"] == ["b1"]
        assert executed_tools(outbound) == ["task_write"]
        assert _claims_events(outbound) == []

    run(scenario())


def test_a_turn_that_released_nothing_still_claims(loop_factory) -> None:
    """The shape the flag is for: a world-changing call withheld, no gate, and
    prose that delivered no batch, no question and no ask — so nothing in the
    message says anything is pending. Here the refused duplicate of a spent
    release ships under "Sent."."""
    async def scenario() -> None:
        provider = ScriptProvider([
            *_gate_responses(),
            LLMResponse(content="Sending it.", tool_calls=[
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "yes, send it"},
                ]),
                task_call("send_2", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            LLMResponse(content="And one more.", tool_calls=[
                task_call("send_3", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            LLMResponse(content="Sent."),
        ])
        loop, _ = loop_factory(provider, "task_send")

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Yes, send it.", active_ix=["autonomy_level"],
            action_types=ACTION_TYPES,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        blob = second.metadata["ix_render"]
        assert blob["gate"] is None
        assert blob["released"] == {"batches": [], "questions": [], "plan_executed": ["a1"]}
        assert second.content.strip()
        assert [event["item_ids"] for event in _claims_events(second)] == [["i1"]]

    run(scenario())


# ---------------------------------------------------------------------------
# §3.4 (part 3) · the release *is* the release (round-6 finding 1 / 3, owner decision)
# ---------------------------------------------------------------------------

def _re_declared_release_responses(closing: str) -> list[LLMResponse]:
    """Turn 1 holds `c2` in `b2`; turn 2 re-declares the identical pair and
    releases `c2` through the continuation tool, carrying its call again."""
    return [
        LLMResponse(content=None, tool_calls=[
            selection("topic_management", "One-at-a-time"),
            _batches(("b1", "c1"), ("b2", "c2")),
            task_call("w1", "task_write", "c1"),
            task_call("s2", "task_send", "c2", ask="Send Dr. Okafor the note?"),
        ]),
        LLMResponse(content=_batch_prose(("b1", "c1"), ("b2", "c2"))),
        LLMResponse(content=None, tool_calls=[
            # Verbatim the same batch payload as turn 1 -- this is what `v6g01`
            # sent on all nine of its topic turns.
            _batches(("b1", "c1"), ("b2", "c2")),
            ix("IX_continue_topic_management", decisions=[
                {"id": "c2", "decision": "release", "user_evidence": "send the email now"},
            ]),
            task_call("s2b", "task_send", "c2", ask="Send Dr. Okafor the note?"),
        ]),
        LLMResponse(content=closing),
    ]


def test_a_release_overrides_the_batch_hold_for_that_id(loop_factory) -> None:
    """Round-6 finding 1/3: the batch protocol re-held what the release let out.

    `v6g01` released `email1` on eight turns, issued `email_message_send` on all
    eight, and the PA's re-declared `b2` — identical every turn — re-held it
    every time with `batch_not_released`. Eight authorisations, zero sends, and
    the run's final summary listed the email under "Completed". The release the
    user's own words bought is the release; a re-declared batch does not undo it.
    """
    async def scenario() -> None:
        provider = ScriptProvider(
            _re_declared_release_responses(_batch_prose(("b1", "c1")))
        )
        loop, _ = loop_factory(provider, "task_write", "task_send")

        await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )
        second = await run_turn(
            loop, "Send the email now.", active_ix=["topic_management"],
            action_types=BATCH_ACTION_TYPES,
            consent=[{"interaction_item_id": "c2", "decision": "allow"}],
        )

        assert executed_tools(second) == ["task_send"]
        blob = second.metadata["ix_render"]
        assert blob["held"]["calls"] == []
        assert [(row["item_id"], row["released"]) for row in blob["consent"]] == [
            ("c2", True)
        ]
        assert not [
            event for event in second.metadata["tool_events"]
            if event.get("name") == "ix_release_missing"
        ]
        # The episode has nothing left to hold, so it closes on its own.
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def test_the_batch_override_does_not_lift_the_consent_gate(loop_factory) -> None:
    """It lifts `batch_not_released` and nothing else (§3.7 (part 3) still governs).

    Same shape as above — the id sits in a re-declared, unreleased second batch
    — but the sideband says `deny`. The PA's `release` is intent; the sideband
    is the gate, and it is still the gate after the batch restriction is lifted.
    """
    async def scenario() -> None:
        provider = ScriptProvider(
            _re_declared_release_responses(_batch_prose(("b1", "c1"), ("b2", "c2")))
        )
        loop, _ = loop_factory(provider, "task_write", "task_send")

        await run_turn(
            loop, "Log the deadline and mail Dr. Okafor.",
            active_ix=["topic_management"], action_types=BATCH_ACTION_TYPES,
        )
        second = await run_turn(
            loop, "No, do not send that email.", active_ix=["topic_management"],
            action_types=BATCH_ACTION_TYPES,
            consent=[{"interaction_item_id": "c2", "decision": "deny"}],
        )

        assert executed_tools(second) == []
        assert [
            (call["interaction_item_id"], call["reason"])
            for call in second.metadata["ix_render"]["held"]["calls"]
        ] == [("c2", "consent_deny")]
        assert any(
            event.get("name") == "ix_consent_denied"
            for event in second.metadata["tool_events"]
        )

    run(scenario())


def test_a_held_call_does_not_satisfy_a_release() -> None:
    """§3.7 (part 3): only an **executed** call answers a release (owner decision 2).

    `_releases_without_calls` read `collector.calls`, which carries the withheld
    ones too, so a call the wrapper itself re-held counted as the call the
    release promised — that is why `v6g01` recorded 0 `ix_release_missing` while
    `email1` never went out. A call the wrapper refused *this turn* is still not
    reported as missing: the refusal is already a held row, and a repair would
    only get it held again.
    """
    from pa.agent.ix.collector import TurnCollector
    from pa.agent.ix.render_types import (
        HeldItem,
        HoldEpisode,
        IXHold,
        ReleaseDecision,
        ToolCallRecord,
        TurnPayloads,
    )
    from pa.agent.runner import AgentRunSpec, AgentRunner

    def spec_with_held_c2() -> AgentRunSpec:
        return AgentRunSpec(
            initial_messages=[], tools=None, model="m", max_iterations=1,
            hold=IXHold(open_episodes={"topic_management": HoldEpisode(
                episode_id="bs1", attribute="topic_management",
                calls_held=[{"interaction_item_id": "c2", "tool": "task_send"}],
            )}),
        )

    def collector_with(*calls: ToolCallRecord, held: bool = False) -> TurnCollector:
        collector = TurnCollector()
        collector.payloads = TurnPayloads(release_decisions=[
            ReleaseDecision(id="c2", decision="release", user_evidence="send it"),
        ])
        collector.calls = list(calls)
        if held:
            collector.held = [HeldItem(
                status="held_for_batch", interaction_item_id="c2",
                attribute="topic_management", tool="task_send", reason="consent_deny",
            )]
        return collector

    executed = ToolCallRecord(
        order=0, tool_name="task_send", interaction_item_id="c2", executed=True,
    )
    withheld = replace(executed, executed=False, status="held")

    # The release was answered: the call ran.
    assert AgentRunner._releases_without_calls(
        spec_with_held_c2(), collector_with(executed)
    ) == []
    # No call at all: the action the user authorised never arrived.
    assert AgentRunner._releases_without_calls(
        spec_with_held_c2(), collector_with()
    ) == [("c2", "topic_management", "task_send")]
    # A held call is *not* the call — this is the row the old `called` set
    # silently accepted.
    assert AgentRunner._releases_without_calls(
        spec_with_held_c2(), collector_with(withheld)
    ) == [("c2", "topic_management", "task_send")]
    # …but once it is on the record as a hold, the repair is not owed: the
    # refusal is the answer, and re-issuing would only be refused again.
    assert AgentRunner._releases_without_calls(
        spec_with_held_c2(), collector_with(withheld, held=True)
    ) == []


# ---------------------------------------------------------------------------
# §3.7 (part 4) · a cancelled item is told to the user, and stays told
# ---------------------------------------------------------------------------

_EMAIL_LABEL = "the confirmation email to Dr. Okafor"


def _labelled_batches() -> Any:
    """The `v6g01` pair, with a real label on the batch that gets cancelled."""
    return ix(
        "IX_render_topic_management",
        batch_set_id="bs1",
        item_count=2,
        batches=[
            {"batch_id": "b1", "label": "the calendar deadline", "call_binding": ["c1"]},
            {"batch_id": "b2", "label": _EMAIL_LABEL, "call_binding": ["c2"]},
        ],
    )


def _cap_responses(*, closing_message: str = "") -> list[LLMResponse]:
    """Four topic turns that keep `b2` held until the round cap cancels it."""
    prose = _batch_prose(("b1", "c1"), ("b2", "c2"))
    responses = [
        LLMResponse(content=None, tool_calls=[
            selection("topic_management", "One-at-a-time"),
            _labelled_batches(),
            task_call("w1", "task_write", "c1"),
            task_call("s2", "task_send", "c2", ask="Send Dr. Okafor the note?"),
        ]),
        LLMResponse(content=prose),
    ]
    for round_index in range(2, HOLD_ROUNDS_HARD_CAP + 1):
        continue_args: dict[str, Any] = {
            "decisions": [{"id": "c2", "decision": "keep", "user_evidence": ""}],
        }
        if closing_message and round_index == HOLD_ROUNDS_HARD_CAP:
            continue_args["closing_message"] = closing_message
        responses.append(LLMResponse(content=None, tool_calls=[
            _labelled_batches(),
            ix("IX_continue_topic_management", **continue_args),
        ]))
        responses.append(LLMResponse(content=prose))
    # Two turns after the cancellation: the PA summarises, holding nothing.
    responses.append(LLMResponse(content=None, tool_calls=[_labelled_batches()]))
    responses.append(LLMResponse(content="[[batch:b1]] That is everything."))
    return responses


async def _run_to_the_cap(loop) -> list[Any]:
    turns = []
    for round_index in range(HOLD_ROUNDS_HARD_CAP + 1):
        turns.append(await run_turn(
            loop, f"Turn {round_index}.", active_ix=["topic_management"],
            action_types=BATCH_ACTION_TYPES,
        ))
    return turns


def test_a_cancelled_item_is_recorded_and_the_user_reads_only_the_pa_prose(
    loop_factory,
) -> None:
    """§3.7 (part 4) + owner decision 2026-09-06 (§10 item d): no template close line.

    Live `v6g01`: the round cap cancelled `email1` twice, `ix_close_no_message`
    fired both times, and the PA wrote no closing line — the answer to that was
    a renderer-composed sentence, and `v11s02` t11 showed what it costs: it was
    appended under an answer the PA had just given, contradicting it, with a
    snake_case intent id in the user's face. The miss stays fully recorded; the
    sentence goes.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_cap_responses())
        loop, _ = loop_factory(provider, "task_write", "task_send")

        turns = await _run_to_the_cap(loop)
        cancelling = turns[HOLD_ROUNDS_HARD_CAP - 1]

        assert "I have not done" not in cancelling.content
        assert "on hold" not in cancelling.content
        assert _EMAIL_LABEL not in cancelling.content
        blob = cancelling.metadata["ix_render"]
        assert blob["close_line_source"] == "none"
        # Nothing is papered over: the failure and the ids are on the audit plane.
        assert [
            event["name"] for event in cancelling.metadata["tool_events"]
            if event.get("name") in {"ix_hold_forced_close", "ix_close_no_message"}
        ] == ["ix_hold_forced_close", "ix_close_no_message"]
        assert [item["id"] for item in blob["cancelled_items"]] == ["b2"]

    run(scenario())


def test_the_pa_s_own_closing_line_is_still_delivered(loop_factory) -> None:
    """`closing_message` is PA text, so it is delivered (§3.7 (part 4) unchanged)."""
    async def scenario() -> None:
        line = "I am dropping the email for now; tell me when to send it."
        provider = ScriptProvider(_cap_responses(closing_message=line))
        loop, _ = loop_factory(provider, "task_write", "task_send")

        turns = await _run_to_the_cap(loop)
        cancelling = turns[HOLD_ROUNDS_HARD_CAP - 1]

        assert line in cancelling.content
        assert "I have not done" not in cancelling.content
        assert cancelling.metadata["ix_render"]["close_line_source"] == "payload"
        assert not [
            event for event in cancelling.metadata["tool_events"]
            if event.get("name") == "ix_close_no_message"
        ]

    run(scenario())


def test_a_cancellation_fires_its_forced_close_once(loop_factory) -> None:
    """Round-6 finding 4: `finished_episodes` accumulates, the event must not.

    Live `v6g01` had two genuine cancellations and **nine**
    `ix_hold_forced_close` + `ix_close_no_message` pairs, because every later
    turn re-read the finished episode. Anything counting cancellations off the
    event stream over-counted 4.5 to 1.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_cap_responses())
        loop, _ = loop_factory(provider, "task_write", "task_send")

        turns = await _run_to_the_cap(loop)

        fired = [
            (index, event["name"])
            for index, turn in enumerate(turns)
            for event in turn.metadata["tool_events"]
            if event.get("name") in {"ix_hold_forced_close", "ix_close_no_message"}
        ]
        assert fired == [
            (HOLD_ROUNDS_HARD_CAP - 1, "ix_hold_forced_close"),
            (HOLD_ROUNDS_HARD_CAP - 1, "ix_close_no_message"),
        ]

    run(scenario())


def test_a_cancelled_item_is_on_the_record_and_in_the_reminder_afterwards(
    loop_factory,
) -> None:
    """Owner decision 3(b)/(c): the turns *after* the cancellation.

    The turn that lists a cancelled item as done withholds nothing, so no
    structural detector can reach it — `v6g01` t35's "Confirmation email sent to
    Dr. Okafor" fired nothing at all. Two answers, neither of them NLP: the PA
    is reminded on every later turn that the item was never executed, and
    `cancelled_items` is on every later turn's blob so the judge can read the
    prose against it.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_cap_responses())
        loop, _ = loop_factory(provider, "task_write", "task_send")

        turns = await _run_to_the_cap(loop)
        after = turns[HOLD_ROUNDS_HARD_CAP]

        assert after.metadata["ix_render"]["cancelled_items"] == [{
            "attribute": "topic_management", "id": "b2", "kind": "batch",
            "intent": _EMAIL_LABEL,
        }]
        # Nothing is held any more, and the episode is gone …
        assert after.metadata["ix_render"]["episodes"] == {}
        assert after.metadata["ix_render"]["held"]["calls"] == []
        # … but the PA is still told, in its own internal reminder.
        reminders = [
            message["content"] for message in provider.requests[-1]
            if message.get("_mempa_internal")
        ]
        assert any(
            f"b2 ({_EMAIL_LABEL}): cancelled, never executed" in text
            for text in reminders
        ), reminders
        # The reminder is internal: it reaches neither the user nor history.
        assert "cancelled, never executed" not in after.content

    run(scenario())


# ---------------------------------------------------------------------------
# §3.7 (part 4) (R3) · the cancel line names the label, never the held question
#
# A `Batch` is cancelled by its `label`, which is already a safe short string.
# A `Question` was cancelled by its `text` — the whole withheld question — so
# reaching the round cap delivered on the public face the one thing four rounds
# of holding had kept off it, and then repeated it in every later reminder.
# ---------------------------------------------------------------------------

_HELD_Q_TEXT = "Which room did you want, the HELD_SECRET_Q2 one?"
_HELD_Q_LABEL = "room"


def _elicitation(*questions: dict[str, str]) -> Any:
    return ix(
        "IX_render_information_elicitation",
        elicitation_id="el1", questions=list(questions), assumptions=[],
    )


def _question_cap_responses(*, about: str) -> list[LLMResponse]:
    """Four elicitation turns that keep `q2` held until the round cap cancels it.

    `Iterative` releases the first question and holds the rest, and the payload
    is resent each round exactly as the model would resend it while the ask is
    still open. No `closing_message` is ever written, so the cancellation lands
    on the renderer's template line.
    """
    def payload() -> Any:
        return _elicitation(
            {"id": "q1", "text": "Which date works?", "about": "date"},
            {"id": "q2", "text": _HELD_Q_TEXT, "about": about},
        )

    # `q2`'s section is the wording the hold withholds every round, and it is
    # what `_next_hold` files as the held `Question.text`. The payload field
    # carries the same string so neither carrier can leak unnoticed.
    prose = f"[[q:q1]] Which date works?\n\n[[q:q2]] {_HELD_Q_TEXT}"
    responses = [
        LLMResponse(content=None, tool_calls=[
            selection("information_elicitation", "Iterative"), payload(),
        ]),
        LLMResponse(content=prose),
    ]
    for _ in range(2, HOLD_ROUNDS_HARD_CAP + 1):
        responses.append(LLMResponse(content=None, tool_calls=[
            payload(),
            ix("IX_continue_information_elicitation", decisions=[
                {"id": "q2", "decision": "keep", "user_evidence": ""},
            ]),
        ]))
        responses.append(LLMResponse(content=prose))
    # One turn past the cancellation, so its follow-up reminder is observable.
    responses.append(LLMResponse(content=None, tool_calls=[_elicitation()]))
    responses.append(LLMResponse(content="Noted."))
    return responses


async def _run_elicitation_to_the_cap(loop) -> list[Any]:
    return [
        await run_turn(loop, f"Turn {index}.", active_ix=["information_elicitation"])
        for index in range(HOLD_ROUNDS_HARD_CAP + 1)
    ]


def test_a_cancelled_question_keeps_its_held_text_off_every_public_face(
    loop_factory,
) -> None:
    """R3 + owner decision 2026-09-06: the cap says nothing on the public face.

    R3 kept the withheld wording out of the close line; the close line itself is
    now gone, so neither the question, nor its label, nor its id is delivered.
    The reminder to the PA still names both, on the internal plane.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_question_cap_responses(about=_HELD_Q_LABEL))
        loop, adapter = loop_factory(provider)

        turns = await _run_elicitation_to_the_cap(loop)
        cancelling = turns[HOLD_ROUNDS_HARD_CAP - 1]

        for absent in ("I have not done", "on hold", _HELD_Q_LABEL, "q2"):
            assert absent not in cancelling.content, absent
        assert cancelling.metadata["ix_render"]["close_line_source"] == "none"

        session = loop.sessions.get(SESSION_KEY)
        reminders = [
            str(message.get("content") or "")
            for message in provider.requests[-1]
            if message.get("_mempa_internal") is True
        ]
        assert any(
            f"q2 ({_HELD_Q_LABEL}): cancelled, never executed" in text
            for text in reminders
        ), reminders

        faces = {
            "public turn": cancelling.content,
            "runtime_history public assistant": _blob([
                message["content"] for message in session.messages
                if message.get("role") == "assistant" and not message.get("tool_calls")
            ]),
            "memory_projection": _blob(build_turn_pool(session.messages).messages),
            "memory adapter events": _blob(
                [event.messages for event in adapter.events]
            ),
            "next-turn reminder": _blob(reminders),
        }
        for face, text in faces.items():
            assert _HELD_Q_TEXT not in text, face
            assert "HELD_SECRET" not in text, face

    run(scenario())


def test_a_cancelled_question_without_a_label_drops_the_subject(
    loop_factory,
) -> None:
    """R3, the no-label case: after the 2026-09-06 decision there is no line at all.

    The unlabelled variant existed because the id is no subject for a user.
    Nothing is composed now, so the reply is the PA's prose and the cancellation
    is read off `cancelled_items`.
    """
    async def scenario() -> None:
        provider = ScriptProvider(_question_cap_responses(about=""))
        loop, _ = loop_factory(provider)

        turns = await _run_elicitation_to_the_cap(loop)
        cancelling = turns[HOLD_ROUNDS_HARD_CAP - 1]

        for absent in (
            "did not get to", "on hold", "I have not done",
            _HELD_Q_TEXT, "HELD_SECRET", "q2",
        ):
            assert absent not in cancelling.content, absent
        blob = cancelling.metadata["ix_render"]
        assert blob["close_line_source"] == "none"
        assert [item["id"] for item in blob["cancelled_items"]] == ["q2"]

    run(scenario())


# ---------------------------------------------------------------------------
# §3.7 (part 3) · `consent[]` survives the repair that follows the authorisation
# ---------------------------------------------------------------------------

def test_a_gate_repair_keeps_the_consent_row_of_a_call_that_already_ran(
    loop_factory,
) -> None:
    """Round-6 finding 5: two authorisations, one `consent[]` row.

    `v6g02` t20 released `item_2`, ran `appointment_create`, then opened a second
    gate for `item_3` whose ask was missing. The §3.3 (part 2)·C repair calls
    `clear_holds()`, which wiped `consent[]` wholesale — and the retry has no
    reason to re-issue a call that already ran, so the row was gone for good. A
    judge counting authorisations off `consent[]` under-counted two to one. The
    dedupe key was never the problem; the row for an executed call is a fact of
    the turn and follows `calls`, not `held`.
    """
    async def scenario() -> None:
        action_types = {"task_send": "external_action", "task_write": "internal_write"}
        provider = ScriptProvider([
            LLMResponse(content="Drafting it.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                ),
                task_call("send_1", "task_send", "i1", ask="Shall I mail Dale?"),
            ]),
            # The consent turn: `i1` is released and runs, and a second gate
            # opens on `i2` — with no ask, so the repair fires after the send.
            LLMResponse(content="On it.", tool_calls=[
                plan_call(
                    {"id": "a1", "bound_item_ids": ["i1"]},
                    {"id": "a2", "bound_item_ids": ["i2"]},
                ),
                ix("IX_continue_autonomy_level", decisions=[
                    {"id": "i1", "decision": "release", "user_evidence": "yes, send it"},
                ]),
                task_call("send_2", "task_send", "i1", ask="Shall I mail Dale?"),
                task_call("write_2", "task_write", "i2"),
            ]),
            LLMResponse(content=None, tool_calls=[
                task_call("write_3", "task_write", "i2", ask="Shall I log it too?"),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send", "task_write")

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"], action_types=action_types,
        )
        second = await run_turn(
            loop, "Yes, send it.", active_ix=["autonomy_level"],
            action_types=action_types,
            consent=[{"interaction_item_id": "i1", "decision": "allow"}],
        )

        # The repair really did run, and the second ask really was delivered.
        assert any(
            event.get("name") == "ix_gate_no_message"
            for event in second.metadata["tool_events"]
        )
        assert "Shall I log it too?" in second.content
        assert executed_tools(second) == ["task_send"]
        # …and the authorisation that bought the send is still on the record.
        assert [
            (row["attribute"], row["item_id"], row["sideband_decision"], row["released"])
            for row in second.metadata["ix_render"]["consent"]
        ] == [("autonomy_level", "i1", "allow", True)]

    run(scenario())


# ---------------------------------------------------------------------------
# §4.3 · the reminder an episode holding no calls used to lose
# ---------------------------------------------------------------------------

def _held_episode(attribute: str, **held: Any) -> IXHold:
    return IXHold(open_episodes={
        attribute: HoldEpisode(episode_id="e1", attribute=attribute, **held),
    })


def test_a_question_only_episode_is_reminded_by_id_and_marker_not_by_its_text() -> None:
    """Owner decision 2026-09-05: no held *call* is not the same as nothing held.

    `Iterative` can withhold a question on a turn that makes no tool call, so
    `calls_held` is empty while `IX_continue_information_elicitation` is still
    required next turn (`episode_has_held_items`). The old
    `not episode.calls_held` skip left that turn with no reminder at all.
    """
    prompt = hold_prompt(_held_episode(
        "information_elicitation",
        questions_held=[
            Question(id="q2", text="Which room do you want?", about="room"),
            Question(id="q3", text="How many nights is it?", about="nights"),
        ],
    ))

    assert "q2 (room)" in prompt
    assert "q3 (nights)" in prompt
    assert "IX_continue_information_elicitation" in prompt
    # The markers are named as what divides the reply — not as something owed.
    assert "[[q:id]]" in prompt
    assert "released whole" in prompt
    # The held wording stays in `audit_trace`. The PA asks again in its own
    # words, so the cached sentence is never handed back to be replayed.
    assert "Which room do you want?" not in prompt
    assert "How many nights is it?" not in prompt


def test_a_batch_only_episode_is_reminded_by_id_and_label() -> None:
    """The topic_management half of the same decision, down to its own marker."""
    prompt = hold_prompt(_held_episode(
        "topic_management",
        batches_held=[Batch(batch_id="b2", label="the invoice", body="Held prose.")],
    ))

    assert "b2 (the invoice)" in prompt
    assert "IX_continue_topic_management" in prompt
    assert "[[batch:id]]" in prompt
    assert "Held prose." not in prompt


def test_a_call_holding_episode_keeps_the_reminder_it_already_had() -> None:
    """The wording an episode with held calls gets: no marker line, one refusal.

    acc_009 t=12/t=14 added the second sentence. A held call produces no tool
    result by design, so before it the PA was told only that it "still holds"
    something — and it showed the user a draft that was never created and then
    announced a send that never happened.
    """
    prompt = hold_prompt(_held_episode(
        "autonomy_level",
        calls_held=[{"interaction_item_id": "i1", "tool": "task_send"}],
    ))

    assert prompt == (
        "[interaction wrapper] You still hold: i1 (task_send). Call "
        "IX_continue_autonomy_level and give every id a release / keep / skip. "
        "When you release an outward-facing item, fill user_evidence with the "
        "user's own words from their latest message (audit field). Nothing ran "
        "for task_send; those calls were stopped before they executed, changed "
        "nothing, and must not be reported to the user as done or sent."
    )
    # A question / batch hold holds no call, so it does not get the sentence.
    assert "Nothing ran for" not in hold_prompt(_held_episode(
        "information_elicitation",
        questions_held=[Question(id="q2", text="Which room?", about="room")],
    ))


def test_nothing_open_and_nothing_held_produce_no_reminder() -> None:
    assert hold_prompt(IXHold()) == ""
    # A closed episode is still one the PA must not be nudged about, however
    # much it was holding when it closed.
    assert hold_prompt(_held_episode(
        "information_elicitation",
        closed=True,
        questions_held=[Question(id="q2", text="Which room?", about="room")],
    )) == ""


def test_the_turn_after_a_question_only_hold_really_receives_the_reminder(
    loop_factory,
) -> None:
    """End to end: the loop hands the whole hold to `hold_prompt` unfiltered, so
    the fix has to show up in the next turn's request as well as in the string."""
    def _questions(*questions: dict[str, str]) -> Any:
        return ix(
            "IX_render_information_elicitation",
            elicitation_id="el1", questions=list(questions), assumptions=[],
        )

    async def scenario() -> None:
        provider = ScriptProvider([
            # An asking turn with no task call at all: `q2` is held, and the
            # episode that holds it has `calls_held == []`.
            LLMResponse(content=None, tool_calls=[
                selection("information_elicitation", "Iterative"),
                _questions(
                    {"id": "q1", "text": "Which date?", "about": "date"},
                    {"id": "q2", "text": "Which room?", "about": "room"},
                ),
            ]),
            LLMResponse(
                content="[[q:q1]] Which date works?\n\n[[q:q2]] Which room do you want?",
            ),
            LLMResponse(content=None, tool_calls=[
                _questions(),
                ix("IX_continue_information_elicitation", decisions=[
                    {"id": "q2", "decision": "skip", "user_evidence": ""},
                ]),
            ]),
            LLMResponse(content="Noted."),
        ])
        loop, _ = loop_factory(provider, "task_book")

        first = await run_turn(
            loop, "Book me a room.", active_ix=["information_elicitation"],
        )
        episode = loop._ix_hold_cache[
            (SESSION_KEY, SCRIPT_SESSION)
        ].open_episodes["information_elicitation"]
        assert [question.id for question in episode.questions_held] == ["q2"]
        assert episode.calls_held == []
        assert [
            question["id"] for question in first.metadata["ix_render"]["held"]["questions"]
        ] == ["q2"]

        start = len(provider.requests)
        await run_turn(loop, "The 14th.", active_ix=["information_elicitation"])
        reminders = [
            str(message.get("content") or "")
            for message in provider.requests[start]
            if message.get("_mempa_internal") is True
            and "You still hold" in str(message.get("content") or "")
        ]
        assert len(reminders) == 1
        assert "You still hold: q2 (room)" in reminders[0]
        assert "[[q:id]]" in reminders[0]
        # The model's own earlier payload still carries the cached wording; the
        # wrapper's reminder is what must not hand it back.
        assert "Which room?" not in reminders[0]

    run(scenario())
