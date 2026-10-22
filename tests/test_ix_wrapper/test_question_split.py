"""Playability audit §4 — the undivided `[[q:…]]` block, and the nudge that splits it.

`information_elicitation` releases whole marker blocks, not questions:
`render.py::_release_count` returns 1 under `Iterative` and `total` under
`Structured`, counting `[[q:…]]` sections. Two missing fields written into
**one** block therefore render identically under both settings — the actor sees
the same reply either way and the attribute under test leaves no trace in the
transcript.

`topic_management` already repairs its version of this (`IX_BATCH_SPLIT_REPAIR`,
`ix_batch_undivided`). This is the elicitation half, built as a nudge rather
than a contract: the wrapper cannot write the second question for a model that
refuses to split, so a second undivided reply is delivered as it stands and the
failure stays visible instead of silent.
"""

from __future__ import annotations

from typing import Any

from pa.agent.runner import (
    IX_PER_TURN_FORCED_PREFIX,
    IX_PER_TURN_REMINDER_PREFIX,
    IX_QUESTION_MARKER_REPAIR,
    IX_QUESTION_SPLIT_REPAIR,
)
from pa.providers.base import LLMResponse

from .conftest import ScriptProvider, ix, run, run_turn, selection

#: One block, two questions: the shape the audit named.
UNDIVIDED = "[[q:q1]] Which airport? Which seat?"
#: The same two questions, one per block: what the nudge asks for.
DIVIDED = "[[q:q1]] Which airport?\n\n[[q:q2]] Aisle or window?"


def _event_names(outbound: Any) -> list[str]:
    return [
        str(event.get("name") or "")
        for event in outbound.metadata.get("tool_events") or []
        if event.get("ix_event")
    ]


def _nudges(request: list[dict[str, Any]]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in request
        if str(message.get("content") or "") == IX_QUESTION_SPLIT_REPAIR
    ]


def _payload(*ids: str) -> Any:
    return ix(
        "IX_render_information_elicitation",
        elicitation_id="el1",
        questions=[
            {"id": qid, "text": f"Question {qid}?", "about": qid} for qid in ids
        ],
        assumptions=[],
    )


def _opening(setting: str, *ids: str) -> LLMResponse:
    return LLMResponse(content=None, tool_calls=[
        selection("information_elicitation", setting), _payload(*ids),
    ])


def test_an_undivided_question_block_is_nudged_once_and_the_retry_is_delivered(
    loop_factory,
) -> None:
    """(a) One nudge, one event, and the model keeps the last word.

    The retry here still writes both questions into one block. That is
    deliberate: the nudge is a nudge, not a contract — the wrapper has no way to
    invent the second block, so whatever comes back is delivered untouched and
    the undivided reply is scored as the PA failure it is.
    """
    retry = "[[q:q1]] Which departure airport? Aisle or window?"

    async def scenario() -> None:
        provider = ScriptProvider([
            _opening("Iterative", "q1", "q2"),
            LLMResponse(content=UNDIVIDED),
            LLMResponse(content=retry),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Book me a flight.", active_ix=["information_elicitation"],
        )

        assert _event_names(outbound).count("ix_questions_undivided") == 1
        event = next(
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_questions_undivided"
        )
        assert event["attribute"] == "information_elicitation"
        assert event["block_id"] == "q1"
        assert event["question_count"] == 2

        # Nudged once, and the block it replaced never reaches the user.
        assert len(provider.requests) == 3
        assert _nudges(provider.requests[-1]) == [IX_QUESTION_SPLIT_REPAIR]
        assert "Which departure airport?" in outbound.content
        assert "Aisle or window?" in outbound.content
        assert "Which airport?" not in outbound.content

    run(scenario())


def test_the_split_retry_releases_one_block_and_holds_the_other(loop_factory) -> None:
    """(b) The intended repair: two blocks, and `Iterative` renders as written."""
    async def scenario() -> None:
        provider = ScriptProvider([
            _opening("Iterative", "q1", "q2"),
            LLMResponse(content=UNDIVIDED),
            LLMResponse(content=DIVIDED),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Book me a flight.", active_ix=["information_elicitation"],
        )

        # The first block goes out, the second waits its turn.
        assert "Which airport?" in outbound.content
        assert "Aisle or window?" not in outbound.content
        assert [
            question["id"]
            for question in outbound.metadata["ix_render"]["held"]["questions"]
        ] == ["q2"]
        # The split answered the nudge, so it is not asked again.
        assert _event_names(outbound).count("ix_questions_undivided") == 1
        assert len(provider.requests) == 3

    run(scenario())


def test_no_nudge_under_structured(loop_factory) -> None:
    """(c) `Structured` is allowed to batch: one block, both questions, no nudge."""
    async def scenario() -> None:
        provider = ScriptProvider([
            _opening("Structured", "q1"),
            LLMResponse(content=UNDIVIDED),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Book me a flight.", active_ix=["information_elicitation"],
        )

        assert "ix_questions_undivided" not in _event_names(outbound)
        assert len(provider.requests) == 2
        assert _nudges(provider.requests[-1]) == []
        assert "Which airport?" in outbound.content
        assert "Which seat?" in outbound.content

    run(scenario())


def test_no_nudge_when_each_block_asks_one_question(loop_factory) -> None:
    """(d) Two blocks, one question each: exactly what the setting asks for."""
    async def scenario() -> None:
        provider = ScriptProvider([
            _opening("Iterative", "q1", "q2"),
            LLMResponse(content=DIVIDED),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Book me a flight.", active_ix=["information_elicitation"],
        )

        assert "ix_questions_undivided" not in _event_names(outbound)
        assert len(provider.requests) == 2
        assert _nudges(provider.requests[-1]) == []
        assert "Which airport?" in outbound.content
        assert "Aisle or window?" not in outbound.content

    run(scenario())


def test_the_nudge_never_pins_a_tool_or_spends_the_payload_ladder(
    loop_factory,
) -> None:
    """(e) Its own budget: no `tool_choice`, and no step off the §3.0 ladder."""
    async def scenario() -> None:
        provider = ScriptProvider([
            _opening("Iterative", "q1", "q2"),
            LLMResponse(content=UNDIVIDED),
            LLMResponse(content=DIVIDED),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Book me a flight.", active_ix=["information_elicitation"],
        )

        assert "ix_questions_undivided" in _event_names(outbound)
        assert provider.tool_choices == [None, None, None]
        # The render-payload ladder never moved: no miss recorded, no pin spent,
        # and the only internal message the nudge added is its own.
        for ladder_event in ("ix_render_payload_missing", "ix_render_payload_forced"):
            assert ladder_event not in _event_names(outbound)
        internal = [
            str(message.get("content") or "")
            for message in provider.requests[-1]
            if message.get("_mempa_internal") is True
        ]
        assert [
            message for message in internal
            if message.startswith((IX_PER_TURN_REMINDER_PREFIX, IX_PER_TURN_FORCED_PREFIX))
        ] == []
        assert IX_QUESTION_SPLIT_REPAIR in internal

    run(scenario())


# ---------------------------------------------------------------------------
# acc_009 · the hole beside the split nudge — a payload nobody marked
# ---------------------------------------------------------------------------

#: The framing acc_009 t=8 delivered: a reference to two questions, and neither
#: question anywhere in the reply.
UNMARKED = (
    "Let me know your answers to those two points, and I'll put together a "
    "concrete proposal from there."
)


def _marker_nudges(request: list[dict[str, Any]]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in request
        if str(message.get("content") or "") == IX_QUESTION_MARKER_REPAIR
    ]


def test_a_payload_marked_nowhere_is_nudged_once_and_the_retry_delivers(
    loop_factory,
) -> None:
    """acc_009 t=8: both questions in the payload, no `[[q:…]]` in the prose.

    `_select_sections` selects question sections out of the prose, so with
    nothing marked it released nothing, held nothing and traced nothing, and the
    user read a reference to "those two points" with the two points missing.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _opening("Structured", "q1", "q2"),
            LLMResponse(content=UNMARKED),
            LLMResponse(content=DIVIDED),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Propose a time.", active_ix=["information_elicitation"],
        )

        assert _marker_nudges(provider.requests[-1]) != []
        # The nudge took, so nothing was dropped and nothing is recorded.
        assert "ix_questions_unmarked" not in _event_names(outbound)
        assert "Which airport?" in outbound.content
        assert "Aisle or window?" in outbound.content

    run(scenario())


def test_an_ignored_marker_nudge_is_put_once_more(loop_factory) -> None:
    """acc_009 t=8 and t=10 — one nudge was ignored on both turns.

    Each turn burned its single nudge and shipped the framing without the
    questions; the questions arrived at t=13, two turns late. The second ask is
    still the wrapper asking, not writing.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _opening("Structured", "q1", "q2"),
            LLMResponse(content=UNMARKED),
            LLMResponse(content=UNMARKED),
            LLMResponse(content=DIVIDED),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Propose a time.", active_ix=["information_elicitation"],
        )

        assert len(provider.requests) == 4
        assert len(_marker_nudges(provider.requests[-1])) == 2
        # The second ask took, so nothing was dropped and nothing is recorded.
        assert "ix_questions_unmarked" not in _event_names(outbound)
        assert "Which airport?" in outbound.content
        assert "Aisle or window?" in outbound.content

    run(scenario())


def test_questions_dropped_for_want_of_a_marker_are_always_recorded(
    loop_factory,
) -> None:
    """The nudge is a nudge; the drop is recorded whether or not it takes.

    A third unmarked reply is delivered as it stands — the wrapper will not
    write the PA's questions into its prose — so the audit is what says the ask
    never happened.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _opening("Structured", "q1", "q2"),
            LLMResponse(content=UNMARKED),
            LLMResponse(content=UNMARKED),
            LLMResponse(content=UNMARKED),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Propose a time.", active_ix=["information_elicitation"],
        )

        assert _marker_nudges(provider.requests[-1]) != []
        # Nudged twice at most, however many unmarked replies come back.
        assert len(provider.requests) == 4
        event = next(
            item for item in outbound.metadata["tool_events"]
            if item.get("name") == "ix_questions_unmarked"
        )
        assert event["attribute"] == "information_elicitation"
        assert event["question_count"] == 2
        # …and the trace says it too, where `_select_sections` used to say nothing.
        assert any(
            "no [[q:...]] marker in the prose" in step
            for step in outbound.metadata["ix_render"]["trace"]
        )

    run(scenario())


def test_a_marked_reply_records_no_drop(loop_factory) -> None:
    """The mirror: markers present, nothing dropped, no nudge, no event."""
    async def scenario() -> None:
        provider = ScriptProvider([
            _opening("Structured", "q1", "q2"),
            LLMResponse(content=DIVIDED),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Propose a time.", active_ix=["information_elicitation"],
        )

        assert "ix_questions_unmarked" not in _event_names(outbound)
        assert _marker_nudges(provider.requests[-1]) == []
        assert len(provider.requests) == 2

    run(scenario())
