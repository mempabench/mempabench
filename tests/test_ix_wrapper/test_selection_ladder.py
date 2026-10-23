"""A turn that owes a selection *and* a render payload at the same time.

`information_elicitation` is the shape live `pre_run_test_9` (acc_009 t1) died
on: natural selection, so the turn owes `IX_information_elicitation`
(`required == "always"`) while `IX_render_information_elicitation` is owed on
every turn the attribute sits in `session_active_ix`. Two carriers, and
`tool_choice` pins exactly one function per request.

Before the fix the conditional-payload branch voided *every* response that
still owed a carrier, including the one that answered a pin: the delivered call
was never appended to `messages`, never executed and never banked in
`ix_completed`, so each pin was spent on a call that was thrown away and the
turn ended on `ix_contract_failure` even for a fully compliant model. These
cases pin the ladder's four shapes and the budget that still has to end a turn
the model never answers.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from pa.agent.ix.render import IX_EMPTY_REPLY_LINE
from pa.agent.loop import AgentProviderError
from pa.providers.base import LLMResponse, ToolCallRequest

from .conftest import (
    SESSION_KEY,
    ScriptProvider,
    ix,
    run,
    run_turn,
    selection,
    task_call,
)

ATTRIBUTE = "information_elicitation"
SELECTION_TOOL = "IX_information_elicitation"
RENDER_TOOL = "IX_render_information_elicitation"


def _selection_call() -> ToolCallRequest:
    return selection(ATTRIBUTE, "Structured")


def _render_call() -> ToolCallRequest:
    """A valid payload that opens no episode: `Infer`-shaped, no questions."""
    return ix(
        RENDER_TOOL,
        elicitation_id="el1",
        questions=[],
        assumptions=[{"about": "duration", "assumed": "about an hour"}],
    )


class PinnedScriptProvider(ScriptProvider):
    """A scripted provider that answers a `tool_choice` pin like a real one.

    `tool_choice={"type": "function", ...}` is the provider's own guarantee that
    exactly the named function comes back, so a pinned request never reaches the
    script: the pinned call is synthesised here. The scripted response is still
    consumed, so the script stays aligned with the request count.
    """

    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        pinned: dict[str, Callable[[], ToolCallRequest]],
    ) -> None:
        super().__init__(responses)
        self._pinned = dict(pinned)

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
            self._pinned[tool_choice["function"]["name"]](),
        ])


def _pins(provider: ScriptProvider) -> list[str]:
    """The tools `tool_choice` pinned, request by request."""
    return [
        choice["function"]["name"]
        for choice in provider.tool_choices
        if isinstance(choice, dict)
    ]


def _event_names(outbound: Any) -> list[str]:
    return [
        str(event.get("name") or "")
        for event in outbound.metadata["tool_events"] or []
    ]


def _forced_tools(outbound: Any) -> list[str]:
    return [
        str(event.get("tool") or "")
        for event in outbound.metadata["tool_events"] or []
        if event.get("name") == "ix_render_payload_forced"
    ]


def _completed_ix(outbound: Any) -> set[str]:
    """The IX carriers that really ran and parsed — what `ix_completed` banks."""
    return {
        str(event.get("name") or "")
        for event in outbound.metadata["tool_events"] or []
        if event.get("ix")
    }


def test_a_selection_and_its_payload_in_separate_responses_both_land(
    loop_factory,
) -> None:
    """(a) The model splits the two carriers over two responses, unprompted.

    Neither response satisfies the turn on its own, and before the fix the first
    was voided for owing the payload and the second for owing the selection —
    two pins spent on calls that were discarded, then `ix_contract_failure`.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[_selection_call()]),
            LLMResponse(content=None, tool_calls=[_render_call()]),
            LLMResponse(content="About an hour, I would say."),
            # Slack the fixed runner never reaches: it keeps both carriers, so
            # three requests finish the turn. Without the fix the ladder spends
            # two more requests on pins before failing, and these keep that
            # failure the contract error rather than an exhausted script.
            LLMResponse(content="About an hour, I would say."),
            LLMResponse(content="About an hour, I would say."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
        )

        assert "About an hour, I would say." in outbound.content
        assert _completed_ix(outbound) == {SELECTION_TOOL, RENDER_TOOL}
        # The turn cost no more than the four requests the whole ladder allows.
        assert len(provider.tool_choices) <= 4
        # Every forced event names a tool that was really pinned, and vice
        # versa: a pin the model answered is progress, never a recorded miss.
        assert _forced_tools(outbound) == _pins(provider)
        assert "ix_render_payload_missing" not in _event_names(outbound)
        assert outbound.metadata["ix_render"]["stop_reason"] == "completed"

    run(scenario())


def test_a_payload_first_response_then_the_selection_on_the_pin_renders(
    loop_factory,
) -> None:
    """(b) The payload arrives first, so the pin has to buy the selection.

    A response owing an `always` selection never reaches tool execution — the
    tool-order gate sends it back — so it banks nothing and the nudge/pin budget
    still has to carry the turn. The pinned selection then runs, and the payload
    the model resends after it completes the contract.
    """
    async def scenario() -> None:
        provider = PinnedScriptProvider(
            [
                LLMResponse(content=None, tool_calls=[_render_call()]),
                LLMResponse(content=None, tool_calls=[_render_call()]),
                # Request three is the pinned one; its scripted stand-in is
                # consumed but replaced by the selection call.
                LLMResponse(content="ignored"),
                LLMResponse(content=None, tool_calls=[_render_call()]),
                LLMResponse(content="About an hour, I would say."),
            ],
            pinned={
                SELECTION_TOOL: _selection_call,
                RENDER_TOOL: _render_call,
            },
        )
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
        )

        assert "About an hour, I would say." in outbound.content
        assert _completed_ix(outbound) == {SELECTION_TOOL, RENDER_TOOL}
        assert _pins(provider) == [SELECTION_TOOL]
        assert _forced_tools(outbound) == [SELECTION_TOOL]

    run(scenario())


def test_both_carriers_in_one_response_still_cost_two_requests(
    loop_factory,
) -> None:
    """(c) The compliant shape is untouched: no nudge, no pin, two requests."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                _selection_call(), _render_call(),
            ]),
            LLMResponse(content="About an hour, I would say."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
        )

        assert "About an hour, I would say." in outbound.content
        assert provider.tool_choices == [None, None]
        assert _completed_ix(outbound) == {SELECTION_TOOL, RENDER_TOOL}

    run(scenario())


def test_a_model_that_delivers_neither_carrier_still_fails_on_third_response(
    loop_factory,
) -> None:
    """(d) One unresolved wrapper operation gets three responses in total.

    The turn owing two carriers gets its initial response, one reminder and one
    pinned repair. It dies on that third invalid response, before another pin
    can multiply the budget across selection and render ladders.
    """
    async def scenario() -> None:
        provider = ScriptProvider([LLMResponse(content="About an hour.")] * 4)
        loop, adapter = loop_factory(provider)
        before_messages = list(loop.sessions.get(SESSION_KEY).messages)

        with pytest.raises(AgentProviderError, match="IX render contract"):
            await run_turn(
                loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            )

        assert loop.sessions.get(SESSION_KEY).messages == before_messages
        assert adapter.events == []
        assert _pins(provider) == [SELECTION_TOOL]
        assert len(provider.tool_choices) == 3

    run(scenario())


def test_a_legal_mixed_selection_is_banked_before_the_render_is_repaired(
    loop_factory,
) -> None:
    """A valid selection is progress even when its sibling task is withheld."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="About an hour."),
            LLMResponse(content=None, tool_calls=[
                _selection_call(),
                task_call("read_1", "task_read", "i1", value="not-run"),
            ]),
            LLMResponse(content=None, tool_calls=[_render_call()]),
            LLMResponse(content="About an hour, I would say."),
        ])
        loop, adapter = loop_factory(provider, "task_read")

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            action_types={"task_read": "read"},
        )

        assert len(provider.tool_choices) == 4
        assert provider.responses == []
        assert adapter.events
        assert _completed_ix(outbound) == {SELECTION_TOOL, RENDER_TOOL}
        events = outbound.metadata["tool_events"] or []
        assert not [event for event in events if event.get("name") == "task_read"]
        assert [
            event for event in events
            if event.get("name") == "ix_task_calls_deferred_for_carrier"
        ] == [{
            "ix_event": True,
            "name": "ix_task_calls_deferred_for_carrier",
            "calls": [{"id": "read_1", "name": "task_read"}],
            "owed": [RENDER_TOOL],
        }]
        assistant_calls = [
            call["function"]["name"]
            for message in provider.requests[-1]
            if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        ]
        tool_results = [
            message.get("name")
            for message in provider.requests[-1]
            if message.get("role") == "tool"
        ]
        assert "task_read" not in assistant_calls
        assert "task_read" not in tool_results
        assert assistant_calls.count(SELECTION_TOOL) == 1
        assert tool_results.count(SELECTION_TOOL) == 1

    run(scenario())


def test_a_response_carrying_task_calls_banks_ix_but_defers_the_work(
    loop_factory,
) -> None:
    """Each IX carrier lands; ordinary work waits for the complete contract."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                _selection_call(),
                task_call("read_1", "task_read", "i1", value="voided"),
            ]),
            LLMResponse(content=None, tool_calls=[
                _selection_call(), _render_call(),
                task_call("read_2", "task_read", "i1", value="resent"),
            ]),
            LLMResponse(content="About an hour, I would say."),
        ])
        loop, _ = loop_factory(provider, "task_read")

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            action_types={"task_read": "read"},
        )

        # The first response's task was deferred. The second response completed
        # the carrier set, so its resent task ran once.
        assert [
            event["args"]["value"]
            for event in outbound.metadata["tool_events"] or []
            if event.get("name") == "task_read"
        ] == ["resent"]
        assert _completed_ix(outbound) == {SELECTION_TOOL, RENDER_TOOL}

    run(scenario())



def test_repeated_valid_selections_do_not_reset_the_missing_render_budget(
    loop_factory,
) -> None:
    """Only the first selection is new progress; duplicates cannot loop forever."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                selection(ATTRIBUTE, setting),
                task_call(f"read_{index}", "task_read", f"i{index}", value="not-run"),
            ])
            for index, setting in enumerate(
                ("Infer", "Structured", "Iterative", "Infer"), 1
            )
        ])
        loop, adapter = loop_factory(provider, "task_read")

        with pytest.raises(AgentProviderError, match="IX render contract") as caught:
            await run_turn(
                loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
                action_types={"task_read": "read"},
            )

        assert len(provider.tool_choices) == 4
        assert adapter.events == []
        assert caught.value.ix_contract_exhaustion == {
            "attempts": 3,
            "contract": "render",
            "operation": "required_render_carrier",
        }
        events = caught.value.failed_turn_events["tool_events"]
        assert [
            event["setting"] for event in events
            if event.get("name") == SELECTION_TOOL
        ] == ["Infer"]
        assert not [event for event in events if event.get("name") == "task_read"]

    run(scenario())

def test_three_invalid_mixed_selections_still_exhaust_and_never_run_tasks(
    loop_factory,
) -> None:
    """Mixed batching cannot turn a rejected IX payload into progress."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                ix(SELECTION_TOOL, call_id=f"bad_{index}", setting="not-a-setting"),
                task_call(f"read_{index}", "task_read", f"i{index}", value="not-run"),
            ])
            for index in range(1, 4)
        ])
        loop, adapter = loop_factory(provider, "task_read")

        with pytest.raises(AgentProviderError, match="IX selection contract") as caught:
            await run_turn(
                loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
                action_types={"task_read": "read"},
            )

        assert len(provider.requests) == 3
        assert adapter.events == []
        assert caught.value.ix_contract_exhaustion == {
            "attempts": 3,
            "contract": "selection_args",
            "operation": "ix_tool_arguments",
        }
        events = caught.value.failed_turn_events["tool_events"]
        assert not [event for event in events if event.get("name") == "task_read"]

    run(scenario())


def test_same_selection_tool_in_three_independent_operations_has_no_shared_budget(
    loop_factory,
) -> None:
    """The repair budget is local to a runner operation, not a tool name."""
    async def scenario() -> None:
        for index in range(3):
            provider = ScriptProvider([
                LLMResponse(content=None, tool_calls=[_selection_call()]),
                LLMResponse(content=None, tool_calls=[_render_call()]),
                LLMResponse(content=f"done-{index}"),
            ])
            loop, _ = loop_factory(provider)
            outbound = await run_turn(
                loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            )
            assert outbound.content == f"done-{index}"
            assert _completed_ix(outbound) == {SELECTION_TOOL, RENDER_TOOL}
            assert len(provider.requests) == 3

    run(scenario())


def test_a_delivered_pinned_call_is_executed_exactly_once(loop_factory) -> None:
    """(e) The kept response is appended once, so its call runs once.

    Falling through to normal execution rather than `continue` must not leave a
    second copy of the assistant message or of its tool result in `messages` —
    a duplicate `IX_information_elicitation` result would read to the model as
    two selections for one attribute.
    """
    async def scenario() -> None:
        provider = PinnedScriptProvider(
            [
                LLMResponse(content=None, tool_calls=[_render_call()]),
                LLMResponse(content=None, tool_calls=[_render_call()]),
                LLMResponse(content="ignored"),
                LLMResponse(content=None, tool_calls=[_render_call()]),
                LLMResponse(content="About an hour, I would say."),
            ],
            pinned={
                SELECTION_TOOL: _selection_call,
                RENDER_TOOL: _render_call,
            },
        )
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
        )

        final_request = provider.requests[-1]
        assert [
            message for message in final_request
            if message.get("role") == "tool"
            and message.get("name") == SELECTION_TOOL
        ] != []
        assert len([
            message for message in final_request
            if message.get("role") == "tool"
            and message.get("name") == SELECTION_TOOL
        ]) == 1
        assert len([
            message for message in final_request
            if message.get("role") == "assistant"
            and any(
                call["function"]["name"] == SELECTION_TOOL
                for call in message.get("tool_calls") or []
            )
        ]) == 1
        assert _event_names(outbound).count(SELECTION_TOOL) == 1

    run(scenario())


# --- the pin follows what is still owed *after* the void ---------------------


def test_a_mixed_selection_banks_and_the_pin_buys_only_the_render(
    loop_factory,
) -> None:
    """A valid selection survives the mixed reply; its task waits for render."""
    async def scenario() -> None:
        provider = PinnedScriptProvider(
            [
                # The nudge is spent on a response that owes both carriers.
                LLMResponse(content=None, tool_calls=[
                    task_call("read_1", "task_read", "i1", value="nudged"),
                ]),
                # The selection arrives with work that cannot run yet. The
                # selection banks; the work is withheld.
                LLMResponse(content=None, tool_calls=[
                    _selection_call(),
                    task_call("read_2", "task_read", "i1", value="voided"),
                ]),
                # Request three is the pinned one; its stand-in is consumed.
                LLMResponse(content="ignored"),
                LLMResponse(content=None, tool_calls=[
                    _render_call(),
                    task_call("read_3", "task_read", "i1", value="ran"),
                ]),
                LLMResponse(content="About an hour, I would say."),
            ],
            pinned={
                SELECTION_TOOL: _selection_call,
                RENDER_TOOL: _render_call,
            },
        )
        loop, _ = loop_factory(provider, "task_read")

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            action_types={"task_read": "read"},
        )

        assert "About an hour, I would say." in outbound.content
        assert _pins(provider) == [RENDER_TOOL]
        assert _forced_tools(outbound) == [RENDER_TOOL]
        assert _completed_ix(outbound) == {SELECTION_TOOL, RENDER_TOOL}
        assert "ix_render_payload_missing" not in _event_names(outbound)
        # Neither task call ran: the first lacked all carriers and the second
        # was withheld; the pin itself is IX-only.
        assert [
            event["args"]["value"]
            for event in outbound.metadata["tool_events"] or []
            if event.get("name") == "task_read"
        ] == []

    run(scenario())


# --- the pinned re-emit must not empty the selection it re-sends -------------


def _bare_selection(setting: str = "Structured") -> ToolCallRequest:
    """The pinned re-emit as it comes back live: the setting and nothing else."""
    return ix(SELECTION_TOOL, setting=setting)


def _selection_events(outbound: Any) -> list[dict[str, Any]]:
    return [
        event for event in outbound.metadata["tool_events"] or []
        if event.get("name") == SELECTION_TOOL
    ]


def _mixed_then_render_pinned(pinned: Callable[[], ToolCallRequest]) -> ScriptProvider:
    """Evidence lands with the selection; only the later render is pinned."""
    return PinnedScriptProvider(
        [
            LLMResponse(content=None, tool_calls=[
                task_call("read_1", "task_read", "i1", value="nudged"),
            ]),
            # The model's own selection, reasoned: `selection()` fills evidence
            # and application. Its sibling task is withheld until render lands.
            LLMResponse(content=None, tool_calls=[
                _selection_call(),
                task_call("read_2", "task_read", "i1", value="voided"),
            ]),
            LLMResponse(content="ignored"),
            LLMResponse(content=None, tool_calls=[
                pinned(), _render_call(),
                task_call("read_3", "task_read", "i1", value="ran"),
            ]),
            LLMResponse(content="About an hour, I would say."),
        ],
        pinned={SELECTION_TOOL: pinned, RENDER_TOOL: _render_call},
    )


def test_the_mixed_selection_keeps_its_evidence_without_a_re_emit(
    loop_factory,
) -> None:
    """The model's reasoned selection banks once; the pin asks only for render."""
    async def scenario() -> None:
        provider = _mixed_then_render_pinned(_bare_selection)
        loop, _ = loop_factory(provider, "task_read")

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            action_types={"task_read": "read"},
        )

        assert _pins(provider) == [RENDER_TOOL]
        events = _selection_events(outbound)
        assert [event["setting"] for event in events] == ["Structured"]
        assert events[0]["evidence"] == "the user said so last time"
        assert events[0]["application"] == "apply it to this turn"

    run(scenario())


def test_a_later_duplicate_selection_cannot_replace_the_banked_setting(
    loop_factory,
) -> None:
    """Once Structured banks, a later duplicate cannot relabel its evidence."""
    async def scenario() -> None:
        provider = _mixed_then_render_pinned(lambda: _bare_selection("Iterative"))
        loop, _ = loop_factory(provider, "task_read")

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            action_types={"task_read": "read"},
        )

        events = _selection_events(outbound)
        assert [event["setting"] for event in events] == ["Structured"]
        assert events[0]["evidence"] == "the user said so last time"
        assert events[0]["application"] == "apply it to this turn"

    run(scenario())


# --- the provider ran out of room, and nobody decided anything ---------------


def _truncated() -> LLMResponse:
    """The live acc_026 shape: the whole budget spent, nothing sent.

    `finish_reason == "length"`, `completion_tokens == 4096`, no content and no
    tool calls -- deepseek-v4-flash reasoning to the cap with the answer still
    unwritten.
    """
    return LLMResponse(
        content="",
        finish_reason="length",
        usage={"prompt_tokens": 9000, "completion_tokens": 4096},
    )


def test_two_truncated_replies_are_answered_and_cost_the_ladder_nothing(
    loop_factory,
) -> None:
    """(f) A cut-off reply is the provider running out of room, not a miss.

    Live acc_026: two of these were read as two ladder misses, so the
    reminder and the pin were spent before the model had said anything at all,
    and the turn died on `ix_render_payload_missing` with the render payload
    never asked for. Each is answered once instead, and the budget is still
    whole when the carriers do arrive.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _truncated(),
            _truncated(),
            LLMResponse(content=None, tool_calls=[
                _selection_call(), _render_call(),
            ]),
            LLMResponse(content="About an hour, I would say."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
        )

        assert "About an hour, I would say." in outbound.content
        assert _completed_ix(outbound) == {SELECTION_TOOL, RENDER_TOOL}
        assert _event_names(outbound).count("provider_truncated") == 2
        # Nothing on the ladder moved: no miss recorded, nothing pinned.
        assert "ix_render_payload_forced" not in _event_names(outbound)
        assert "ix_render_payload_missing" not in _event_names(outbound)
        assert provider.tool_choices == [None, None, None, None]
        assert outbound.metadata["ix_render"]["stop_reason"] == "completed"

    run(scenario())


def test_a_third_empty_truncation_fails_as_provider_not_ix(
    loop_factory,
) -> None:
    """(g) Two retries, then a provider failure without buying an IX miss."""
    async def scenario() -> None:
        provider = ScriptProvider(
            [_truncated()] * 3 + [LLMResponse(content="About an hour.")] * 3
        )
        loop, adapter = loop_factory(provider)
        before_messages = list(loop.sessions.get(SESSION_KEY).messages)

        with pytest.raises(AgentProviderError, match="provider_truncation_exhausted"):
            await run_turn(
                loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            )

        assert loop.sessions.get(SESSION_KEY).messages == before_messages
        assert adapter.events == []
        assert _pins(provider) == []
        assert len(provider.tool_choices) == 3

    run(scenario())


# --- and then the fragment it left behind shipped as the turn ----------------


#: acc_006 t=8, verbatim: narration written between two document
#: edits. It is not an answer to anything, and the second sentence has no end.
INTERSTITIAL = "Let me remove the remaining fragments. I'll work from the bottom up"

#: acc_006 t=4, verbatim: the same shape with no sentence end at all, so the
#: truncation rule keeps nothing of it.
INTERSTITIAL_UNPUNCTUATED = (
    "Let me start by looking at the research folder to find the citation notes"
)


def _carrier_response(content: str) -> LLMResponse:
    """Both carriers plus prose, i.e. narration in the same message as calls."""
    return LLMResponse(
        content=content, tool_calls=[_selection_call(), _render_call()],
    )


def test_a_truncated_turn_does_not_ship_its_unfinished_interstitial(
    loop_factory,
) -> None:
    """(i) The turn was cut off, so its dangling last line is not the reply.

    Live acc_006: three truncations closed t=8, the renderer's
    `_final_segment` reached back past them to the narration the model had
    written between its `document_update` calls, and the user's whole PA turn
    was `Let me remove the remaining fragments. I'll work from the bottom up`.
    What survives here is the sentence that was finished; the tail after it is
    dropped and counted. The fixture now ends with nonempty truncated prose;
    three empty truncations instead take the independent provider failure path.
    """
    async def scenario() -> None:
        provider = ScriptProvider(
            [_carrier_response(INTERSTITIAL)] + [_truncated()] * 2
            + [LLMResponse(content=INTERSTITIAL, finish_reason="length",
                           usage={"completion_tokens": 4096})]
        )
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Merge the duplicate entries.", active_ix=[ATTRIBUTE],
        )

        assert "Let me remove the remaining fragments." in outbound.content
        assert "I'll work from the bottom up" not in outbound.content
        events = outbound.metadata["tool_events"]
        truncated = [
            event for event in events if event.get("name") == "turn_truncated"
        ]
        assert len(truncated) == 1
        assert truncated[0]["completion_tokens"] == 4096
        assert truncated[0]["dropped_chars"] == len(" I'll work from the bottom up")
        assert _event_names(outbound).count("provider_truncated") == 2

    run(scenario())


def test_the_same_reply_is_untouched_when_the_provider_did_not_truncate(
    loop_factory,
) -> None:
    """(j) The rule reads `finish_reason`, not the prose.

    An unpunctuated last line is ordinary: plenty of replies end without a full
    stop and the model meant them that way. Only a response the provider ended
    at the token limit loses its tail, so the identical text delivered by a
    whole response reaches the user whole and records nothing.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _carrier_response(""), LLMResponse(content=INTERSTITIAL),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Merge the duplicate entries.", active_ix=[ATTRIBUTE],
        )

        assert outbound.content == INTERSTITIAL
        assert "turn_truncated" not in _event_names(outbound)

    run(scenario())


def test_a_truncated_turn_with_no_finished_sentence_takes_the_empty_path(
    loop_factory,
) -> None:
    """(k) Nothing complete left means the turn has said nothing.

    acc_006 t=4's fragment has no sentence end anywhere, so the rule keeps none
    of it and hands the turn to the empty-reply rule below -- which spends its
    one request asking the PA for the sentence the turn owes, and delivers the
    template line when that comes back empty too. No new path, and the fragment
    reaches the user in neither.

    The final truncated response is nonempty, so this still exercises the
    existing partial-prose rule rather than empty-provider exhaustion.
    """
    async def scenario() -> None:
        provider = ScriptProvider(
            [_carrier_response(INTERSTITIAL_UNPUNCTUATED)]
            + [_truncated()] * 2
            + [LLMResponse(content=INTERSTITIAL_UNPUNCTUATED, finish_reason="length",
                           usage={"completion_tokens": 4096})]
            + [LLMResponse(content="")]
        )
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "Merge the duplicate entries.", active_ix=[ATTRIBUTE],
        )

        assert outbound.content == IX_EMPTY_REPLY_LINE
        assert INTERSTITIAL_UNPUNCTUATED not in outbound.content
        names = _event_names(outbound)
        assert names.count("turn_truncated") == 1
        assert "ix_empty_reply" in names

    run(scenario())


# --- the provider answered with an error, and the wrapper read it as a turn --


#: The live acc_003 t=12 body, verbatim: the upstream error and its
#: request id, handed back as the response's `content`.
PROVIDER_ERROR_BODY = (
    "The request failed because it is missing `messages.content` parameter. "
    "Request id: 021788775363717beabf2da32de8288dd26a2be1ddd6126859bde"
)


def _provider_error() -> LLMResponse:
    return LLMResponse(content=PROVIDER_ERROR_BODY, finish_reason="error")


def test_a_provider_error_is_retried_once_and_the_turn_still_lands(
    loop_factory,
) -> None:
    """(h) An error response is not a reply, so it costs one more request.

    Live acc_003 t=12: the user had just authorised two document
    merges, the provider answered with its error body, and the wrapper released
    that string as the whole PA turn with zero tool calls. The same messages go
    back once instead, and the turn the model then writes is the one delivered.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _provider_error(),
            LLMResponse(content=None, tool_calls=[
                _selection_call(), _render_call(),
            ]),
            LLMResponse(content="About an hour, I would say."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
        )

        assert "About an hour, I would say." in outbound.content
        assert PROVIDER_ERROR_BODY not in (outbound.content or "")
        assert _completed_ix(outbound) == {SELECTION_TOOL, RENDER_TOOL}
        assert _event_names(outbound).count("provider_error_retried") == 1
        # The retry is not a ladder miss: nothing reminded, nothing pinned.
        assert "ix_render_payload_forced" not in _event_names(outbound)
        assert provider.tool_choices == [None, None, None]

    run(scenario())


def test_a_second_provider_error_ends_the_turn_and_delivers_no_text(
    loop_factory,
) -> None:
    """(i) One retry, then the existing error path -- never the error body.

    The retry is a safety net, not a loop: a second error ends the turn on
    `stop_reason == "error"`, which the loop raises on, so nothing of the
    gateway's text reaches the user, the session or the memory adapter.
    """
    async def scenario() -> None:
        provider = ScriptProvider([_provider_error(), _provider_error()])
        loop, adapter = loop_factory(provider)
        before_messages = list(loop.sessions.get(SESSION_KEY).messages)

        with pytest.raises(AgentProviderError, match="failed to produce"):
            await run_turn(
                loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            )

        assert loop.sessions.get(SESSION_KEY).messages == before_messages
        assert adapter.events == []
        assert len(provider.tool_choices) == 2

    run(scenario())


def test_a_reply_with_no_text_leaves_no_contentless_assistant_message(
    loop_factory,
) -> None:
    """(j) The root cause of acc_003 t=12: what made the request invalid.

    A reply with no text and no tool calls used to be written onto the history
    as `{"role": "assistant", "content": None}`, and the closing request the
    empty-reply rule sends straight afterwards was rejected for exactly that --
    "missing `messages.content` parameter". The empty string goes on instead,
    which `_sanitize_empty_content` already knows how to send.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                _selection_call(), _render_call(),
            ]),
            LLMResponse(content=None),
            LLMResponse(content="About an hour, I would say."),
        ])
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
        )

        assert "About an hour, I would say." in outbound.content
        assert len(provider.requests) == 3
        assert not [
            message for message in provider.requests[-1]
            if message.get("content") is None
        ]

    run(scenario())


def _pin_messages(provider: ScriptProvider) -> list[str]:
    """The wrapper's pin instructions, in the order they were sent.

    The pin rides as an internal user message appended just before the pinned
    request, so it is the last message of the request that carries the
    `tool_choice`.
    """
    return [
        str(request[-1].get("content") or "")
        for request, choice in zip(provider.requests, provider.tool_choices)
        if isinstance(choice, dict)
    ]


def test_the_one_available_pin_asks_selection_for_evidence_and_application(
    loop_factory,
) -> None:
    """A pinned selection came back as the bare `setting`.

    `tool_choice` is answered with the schema's `required` list and nothing
    else, and `setting` is the only required field on a selection tool. Every
    live session whose selection was pinned (`acc_002`, `007`, `009`,
    `019`, `026`, `035`) banked `evidence: ""`, and so did all three of an
    earlier run's; no pinned selection in either run carried one. The pin now
    names the two fields the pinned call would otherwise drop. The render
    payload's pin is untouched: it has no such fields.
    """
    async def scenario() -> None:
        provider = ScriptProvider([LLMResponse(content="About an hour.")] * 4)
        loop, _ = loop_factory(provider)

        with pytest.raises(AgentProviderError, match="IX render contract"):
            await run_turn(
                loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
            )

        assert _pins(provider) == [SELECTION_TOOL]
        [selection_pin] = _pin_messages(provider)

        assert SELECTION_TOOL in selection_pin
        assert "`application`" in selection_pin
        assert "`evidence`" in selection_pin
        # The standing rule survives the ask: an empty `evidence` is still the
        # right answer with nothing to cite, and is never to be invented.
        assert "never invent it" in selection_pin

    run(scenario())


def test_a_forced_event_says_how_many_calls_preceded_the_pin(
    loop_factory,
) -> None:
    """The pin's position in `tool_events` is not its position in the turn.

    `collector.events` are appended after every executed call, so a pin that
    fired before the model had made a single call still sits at the tail of the
    list. Three reads took that for chronology and reported the
    wrapper pinning a carrier the turn already had (`acc_007` t2 pinned
    `IX_proactive_outreach` and the record shows the selection first;
    `acc_026` t4 the same). `call_index` says it plainly.
    """
    async def scenario() -> None:
        provider = PinnedScriptProvider(
            [
                # Two text-only replies: nothing is called, so the pin goes out
                # with no call recorded behind it.
                LLMResponse(content="About an hour."),
                LLMResponse(content="About an hour."),
                LLMResponse(content="ignored"),
                LLMResponse(content=None, tool_calls=[_render_call()]),
                LLMResponse(content="About an hour, I would say."),
            ],
            pinned={
                SELECTION_TOOL: _selection_call,
                RENDER_TOOL: _render_call,
            },
        )
        loop, _ = loop_factory(provider)

        outbound = await run_turn(
            loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE],
        )

        forced = [
            event for event in outbound.metadata["tool_events"] or []
            if event.get("name") == "ix_render_payload_forced"
        ]
        assert [event["tool"] for event in forced] == [SELECTION_TOOL]
        assert forced[0]["call_index"] == 0
        # ...while the event itself sits behind the call it bought.
        names = _event_names(outbound)
        assert names.index(SELECTION_TOOL) < names.index(
            "ix_render_payload_forced"
        )

    run(scenario())


def test_provider_error_retry_keeps_the_pinned_tool_choice(loop_factory) -> None:
    """A pinned request answered with a provider error is retried with the same
    `tool_choice`, never unpinned."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[_render_call()]),
            LLMResponse(content=None, tool_calls=[_render_call()]),
            LLMResponse(content="gateway error", finish_reason="error"),
            LLMResponse(content=None, tool_calls=[_selection_call()]),
            LLMResponse(content=None, tool_calls=[_render_call()]),
            LLMResponse(content="About an hour, I would say."),
            LLMResponse(content="About an hour, I would say."),
        ])
        loop, _ = loop_factory(provider)
        await run_turn(loop, "How long is the Reyes visit?", active_ix=[ATTRIBUTE])
        choices = provider.tool_choices
        assert isinstance(choices[2], dict) and choices[2]["function"]["name"] == SELECTION_TOOL, choices
        assert choices[3] == choices[2], choices

    run(scenario())
