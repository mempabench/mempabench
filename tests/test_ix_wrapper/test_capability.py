"""T21 — the two-stage capability boundary, end to end and fail-closed.

Plan §7.2 P7, §3.5. `tests/test_pa/test_runner_ix.py` owns the runner-level
pair (missing, invalid -- each fails on the third response charged to the
shared IX repair budget). What is here is the whole path: a task
tool returning the `capability_limit` code, the resolve tool appearing and
being forced on the next iteration, the two settings parting company, and what
the step is left holding when the contract is not met.
"""

from __future__ import annotations

import pytest

from harness.ix_seed import seed_ix_selection
from pa.agent.ix.render_types import STOP_REASON_IX_CONTRACT_FAILURE
from pa.agent.loop import AgentProviderError
from pa.providers.base import LLMResponse

from .conftest import (
    SCRIPT_SESSION,
    SESSION_KEY,
    CapabilityLimitTool,
    ScriptProvider,
    executed_tools,
    ix,
    plan_call,
    run,
    run_turn,
    selection,
    task_call,
)

HANDOFF = "That paper is not in the retrievable catalogue. Shall I write to Imran?"


def _resolve(**overrides) -> object:
    arguments = {
        "boundary_id": "bd1",
        "blocked_goal": "get the full text",
        "boundary_kind": "missing_data",
        "recovery_options": [{"id": "r1", "intent": "email the author"}],
        "handoff_message": HANDOFF,
    }
    arguments.update(overrides)
    return ix("IX_capability_resolve", **arguments)


def _loop_with_blocked_read(loop_factory, provider: ScriptProvider):
    loop, adapter = loop_factory(provider)
    loop.tools.register(CapabilityLimitTool("task_paper_read"))
    return loop, adapter


def _first_turn(setting: str) -> LLMResponse:
    return LLMResponse(content="Fetching it.", tool_calls=[
        selection("capability_boundary", setting),
        task_call("read_1", "task_paper_read", "i1"),
    ])


# ---------------------------------------------------------------------------
# the happy path of the two-stage protocol
# ---------------------------------------------------------------------------

def test_a_capability_limit_forces_the_resolve_tool_and_ends_with_the_handoff(
    loop_factory,
) -> None:
    """§3.5 (parts 1/4): declare the boundary against the real result, then hand off."""
    async def scenario() -> None:
        provider = ScriptProvider([
            _first_turn("Find and hand off"),
            LLMResponse(content="Declaring the boundary.", tool_calls=[_resolve()]),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_paper_read": "read"},
        )

        # The tool ran and reported the stable code; that is not an error turn.
        assert executed_tools(outbound) == ["task_paper_read"]
        limit_events = [
            event for event in outbound.metadata["tool_events"]
            if event.get("status") == "capability_limit"
        ]
        assert [event["name"] for event in limit_events] == ["task_paper_read"]
        # The resolve tool only exists once the limit has been seen …
        assert "IX_capability_resolve" in provider.offered(-1)
        assert "IX_capability_resolve" not in provider.offered(0)
        # … and `Find and hand off` closes the turn on the PA's own line.
        assert outbound.content.rstrip().endswith(HANDOFF)
        gate = outbound.metadata["ix_render"]["gate"]
        assert gate["attribute"] == "capability_boundary"
        assert gate["message_source"] == "handoff_message"
        # The read itself ran, so nothing is a held *call* -- and owner ruling
        # 2026-09-18: the delivered handoff *is* the hand-back, so the boundary
        # is not left on the books either.
        assert outbound.metadata["ix_render"]["held"]["calls"] == []
        assert outbound.metadata["ix_render"]["episodes"] == {}
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def test_suggest_alternatives_declares_the_boundary_without_handing_off(
    loop_factory,
) -> None:
    """§3.5 (part 3): the other setting keeps going, so nothing is held and no gate opens."""
    async def scenario() -> None:
        provider = ScriptProvider([
            _first_turn("Suggest alternatives"),
            LLMResponse(content="Declaring the boundary.", tool_calls=[_resolve()]),
            LLMResponse(content="The full text is out of reach; the preprint covers it."),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_paper_read": "read"},
        )

        assert outbound.content == (
            "The full text is out of reach; the preprint covers it."
        )
        assert HANDOFF not in outbound.content
        assert outbound.metadata["ix_render"]["gate"] is None
        assert outbound.metadata["ix_render"]["held"]["calls"] == []
        assert outbound.metadata["ix_render"]["episodes"] == {}

    run(scenario())


# ---------------------------------------------------------------------------
# owner, 2026-09-06 · the countable difference between the two settings
# ---------------------------------------------------------------------------

def test_find_and_hand_off_lets_a_read_through_after_the_limit(loop_factory) -> None:
    """Owner ruling 2026-09-17: a read after the limit runs; the turn still hands off.

    `Find and hand off` allows the diagnostic lookups the hand-back needs, so
    the gate no longer holds every call after `capability_limit` — only the
    non-read classes (next test). Whether the read was diagnosis or a recovery
    attempt is the judge's semantic call; the wrapper records it and moves on.
    The turn still ends on the PA's `handoff_message`, verbatim.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _first_turn("Find and hand off"),
            LLMResponse(content="Declaring, and checking the preprint index.", tool_calls=[
                _resolve(),
                task_call("look_1", "task_echo", "i2"),
            ]),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_paper_read": "read", "task_echo": "read"},
        )

        assert executed_tools(outbound) == ["task_paper_read", "task_echo"]
        assert loop.tools.get("task_echo").calls == [{"value": "i2"}]
        assert outbound.content.rstrip().endswith(HANDOFF)
        assert outbound.metadata["ix_render"]["held"]["calls"] == []
        assert outbound.metadata["ix_render"]["gate"]["reason"] == "handoff"

    run(scenario())


@pytest.mark.parametrize("action_type", ["draft", "internal_write", "external_action"])
def test_find_and_hand_off_holds_every_non_read_call_after_the_limit(
    loop_factory, action_type: str,
) -> None:
    """The three world-changing classes stay held: no execution, no tool-log row.

    The handoff gate is decided after `_execute_tools` returns, so a response
    that answers the forced `IX_capability_resolve` *and* attaches a workaround
    call used to get the workaround executed and then ship the handoff line —
    the user is told nothing can be done while something was in fact tried.
    A held call never touches the world, so the tool never sees it and it is
    absent from the executed events the tool log is built from.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _first_turn("Find and hand off"),
            LLMResponse(content="Declaring, and trying the preprint.", tool_calls=[
                _resolve(),
                task_call("work_1", "task_echo", "i2"),
            ]),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_paper_read": "read", "task_echo": action_type},
        )

        assert executed_tools(outbound) == ["task_paper_read"]
        assert loop.tools.get("task_echo").calls == []
        assert outbound.content.rstrip().endswith(HANDOFF)
        held = outbound.metadata["ix_render"]["held"]["calls"]
        assert [(row["tool"], row["status"]) for row in held] == [
            ("task_echo", "held_for_handoff")
        ]

    run(scenario())


def test_suggest_alternatives_may_call_a_tool_after_the_limit(loop_factory) -> None:
    """At least one call after the limit — the same response is the fastest one."""
    async def scenario() -> None:
        provider = ScriptProvider([
            _first_turn("Suggest alternatives"),
            LLMResponse(content="Declaring, and trying the preprint.", tool_calls=[
                _resolve(),
                task_call("work_1", "task_echo", "i2"),
            ]),
            LLMResponse(content="The full text is out of reach; the preprint covers it."),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_paper_read": "read", "task_echo": "read"},
        )

        assert executed_tools(outbound) == ["task_paper_read", "task_echo"]
        assert outbound.metadata["ix_render"]["gate"] is None
        assert outbound.metadata["ix_render"]["held"]["calls"] == []

    run(scenario())


# ---------------------------------------------------------------------------
# T21 (a) / (b) / (c) · the contract failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "second,reason",
    [
        # (a) the resolve tool is simply never called
        (LLMResponse(content="Sorry, I could not get it."), "missing"),
        # (b) it is called with a `boundary_kind` outside the enum
        (
            LLMResponse(content="Declaring.", tool_calls=[
                _resolve(boundary_kind="not_a_real_kind"),
            ]),
            "invalid_payload",
        ),
    ],
)
def test_two_misses_fail_the_step_and_leave_nothing_behind(
    loop_factory, second: LLMResponse, reason: str
) -> None:
    """T21 (a)(b)(c): no scorable answer, nothing persisted, a clean retry.

    The turn raises instead of returning, so `_save_turn` and the memory write
    never run. What survives the failure is cleared at the step boundary, which
    is what makes the retry a second attempt rather than a resumed one.
    """
    async def scenario() -> None:
        provider = ScriptProvider([_first_turn("Find and hand off"), second, second, second])
        loop, adapter = _loop_with_blocked_read(loop_factory, provider)
        before_messages = list(loop.sessions.get(SESSION_KEY).messages)

        with pytest.raises(AgentProviderError, match="IX capability contract"):
            await run_turn(
                loop, "Get me the full text.", active_ix=["capability_boundary"],
                action_types={"task_paper_read": "read"},
            )

        # (c) nothing reached the record: no answer, no memory, no session turn.
        assert loop.sessions.get(SESSION_KEY).messages == before_messages
        assert adapter.events == []
        # And the step boundary hands back a clean slate for the retry.
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []
        key = (SESSION_KEY, SCRIPT_SESSION)
        assert key not in loop._ix_hold_cache
        assert key not in loop._ix_selection_cache

    run(scenario())


def test_the_stop_reason_of_a_capability_contract_failure_is_its_own(
    loop_factory,
) -> None:
    """§3.5 (part 6): this failure is not `error` and not `max_iterations`.

    A step that failed the contract has to be distinguishable from a provider
    error, because only one of the two is worth retrying unchanged.
    """
    async def scenario() -> None:
        miss = LLMResponse(content="Sorry, I could not get it.")
        provider = ScriptProvider([_first_turn("Find and hand off"), miss, miss, miss])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        seen: list[str] = []
        original = loop._render_turn

        def spy(result, **kwargs):
            seen.append(result.stop_reason)
            return original(result, **kwargs)

        loop._render_turn = spy  # type: ignore[method-assign]
        with pytest.raises(AgentProviderError):
            await run_turn(
                loop, "Get me the full text.", active_ix=["capability_boundary"],
                action_types={"task_paper_read": "read"},
            )

        assert seen == [STOP_REASON_IX_CONTRACT_FAILURE]

    run(scenario())


# ---------------------------------------------------------------------------
# round-3 finding 1 · the two scopes of the two-stage protocol
# ---------------------------------------------------------------------------

def test_a_settled_setting_governs_a_beat_that_does_not_list_the_attribute(
    loop_factory,
) -> None:
    """Round-3 finding 1: stage 2 is session-scoped, so stage 3 must be too.

    The runner turns `capability_limit_seen` on from the tool result alone, not
    from the beat's `session_active_ix` — a forced seed or an earlier beat's
    choice both settle the setting elsewhere. So a `capability_limit` on a beat
    that does not list `capability_boundary` still declares the boundary. The
    setting reaches stage 3 through the turn directive, which was built from
    the beat-filtered selections only: live, `v3g07` and `v3g08` both resolved
    the boundary on beat 1 and then rendered byte-identically, `gate: null` and
    zero holds, because `capability_boundary_policy` was empty there.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Fetching it.", tool_calls=[
                task_call("read_1", "task_paper_read", "i1"),
            ]),
            LLMResponse(content="Declaring the boundary.", tool_calls=[_resolve()]),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)
        seed_ix_selection(
            loop,
            session_key=SESSION_KEY,
            script_session_id=SCRIPT_SESSION,
            attribute="capability_boundary",
            setting="Find and hand off",
        )

        # The beat lists nothing: this is `v3g07`'s `b1`, two beats before
        # `acc_026` activates the attribute.
        outbound = await run_turn(
            loop, "Say more about the unreferenced coupling.",
            active_ix=[],
            action_types={"task_paper_read": "read"},
        )

        # Stage 2 ran, as it already did: the resolve tool appeared and was called.
        assert "IX_capability_resolve" in provider.offered(-1)
        # Stage 3 now runs on the same beat.
        gate = outbound.metadata["ix_render"]["gate"]
        assert gate is not None
        assert gate["attribute"] == "capability_boundary"
        assert gate["message_source"] == "handoff_message"
        assert outbound.content.rstrip().endswith(HANDOFF)
        # Owner ruling 2026-09-18: delivered, so nothing is left open.
        assert outbound.metadata["ix_render"]["episodes"] == {}

    run(scenario())


def test_suggest_alternatives_stays_open_on_the_same_off_beat(loop_factory) -> None:
    """The pair `v3g07` / `v3g08` could not tell apart: the other setting has
    to keep going on exactly the beat where `Find and hand off` gates."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Fetching it.", tool_calls=[
                task_call("read_1", "task_paper_read", "i1"),
            ]),
            LLMResponse(content="Declaring the boundary.", tool_calls=[_resolve()]),
            LLMResponse(content="Out of reach; the preprint covers it."),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)
        seed_ix_selection(
            loop,
            session_key=SESSION_KEY,
            script_session_id=SCRIPT_SESSION,
            attribute="capability_boundary",
            setting="Suggest alternatives",
        )

        outbound = await run_turn(
            loop, "Say more about the unreferenced coupling.",
            active_ix=[],
            action_types={"task_paper_read": "read"},
        )

        assert outbound.content == "Out of reach; the preprint covers it."
        assert HANDOFF not in outbound.content
        assert outbound.metadata["ix_render"]["gate"] is None
        assert outbound.metadata["ix_render"]["episodes"] == {}

    run(scenario())


# ---------------------------------------------------------------------------
# pre-run-test audit §2 (acc_026) · the trigger activates the attribute
# ---------------------------------------------------------------------------

def _limit_read() -> LLMResponse:
    return LLMResponse(content="Fetching it.", tool_calls=[
        task_call("read_1", "task_paper_read"),
    ])


def _boundary_selection() -> object:
    return selection("capability_boundary", "Suggest alternatives")


def test_mixed_boundary_selection_is_progress_but_work_waits_for_resolve(
    loop_factory,
) -> None:
    """A legal boundary choice survives; only later missing resolves spend budget."""
    async def scenario() -> None:
        provider = ScriptProvider([
            _limit_read(),
            *[
                LLMResponse(content=None, tool_calls=[
                    _boundary_selection(),
                    task_call(f"work_{index}", "task_echo", f"i{index}"),
                ])
                for index in range(1, 5)
            ],
        ])
        loop, adapter = _loop_with_blocked_read(loop_factory, provider)

        with pytest.raises(AgentProviderError, match="capability contract") as caught:
            await run_turn(
                loop, "Get me the full text.", active_ix=[],
                action_types={"task_paper_read": "read", "task_echo": "read"},
            )

        assert len(provider.requests) == 5
        assert loop.tools.get("task_echo").calls == []
        assert adapter.events == []
        assert caught.value.ix_contract_exhaustion == {
            "attempts": 3,
            "contract": "capability",
            "operation": "capability_resolve",
        }
        events = caught.value.failed_turn_events["tool_events"]
        assert [
            event.get("name") for event in events if event.get("ix")
        ] == ["IX_capability_boundary"]

    run(scenario())


def test_a_capability_limit_activates_the_attribute_on_a_session_that_never_lists_it(
    loop_factory,
) -> None:
    """Audit §2 (acc_026): `capability_boundary` follows the trigger, not the beat.

    Live, the `capability_limit` fired on `b1` while the beat carrying the
    attribute in `active_skills` was `b3`. The PA substituted two other papers
    with no setting chosen, and by `b3` there was nothing left to decide. So the
    first limit in the session puts `IX_capability_boundary` on that same turn --
    required, once -- and leaves the attribute in the session's active set.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            # turn 1 -- an ordinary beat, no limit anywhere
            LLMResponse(content=None, tool_calls=[selection("verbosity", "Terse")]),
            LLMResponse(content="Here it is."),
            # turn 2 -- the limit lands mid-turn
            _limit_read(),
            # ... the declaration alone is not enough: the setting is owed too
            LLMResponse(content="Declaring.", tool_calls=[_resolve()]),
            LLMResponse(content="Choosing and declaring.", tool_calls=[
                _boundary_selection(), _resolve(),
            ]),
            LLMResponse(content="The full text is out of reach; the preprint covers it."),
            # turn 3 -- the attribute is settled, so nothing is asked again
            LLMResponse(content="Here is the preprint summary."),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        await run_turn(loop, "How is the review going?", active_ix=["verbosity"])
        assert "IX_capability_boundary" not in provider.offered(0)
        assert loop.triggered_ix_skills(SESSION_KEY, SCRIPT_SESSION) == []

        second = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["verbosity"],
            action_types={"task_paper_read": "read"},
        )

        # The beat offered no selection tool at all; the trigger put one on the
        # very next request of the same turn, next to the resolve tool.
        assert "IX_capability_boundary" not in provider.offered(2)
        assert "IX_capability_boundary" in provider.offered(3)
        assert "IX_capability_resolve" in provider.offered(3)
        # Required, not merely offered: the resolve-only response is voided and
        # the selection is asked for by name before the turn may go on.
        assert any(
            "IX_capability_boundary" in str(message.get("content") or "")
            for message in provider.requests[4]
        )
        assert second.content == (
            "The full text is out of reach; the preprint covers it."
        )
        assert loop.triggered_ix_skills(SESSION_KEY, SCRIPT_SESSION) == [
            "capability_boundary"
        ]

        await run_turn(loop, "And the preprint?", active_ix=["verbosity"])

        # Withdrawn like any other settled selection, but still governing.
        assert "IX_capability_boundary" not in provider.offered(6)
        assert loop.triggered_ix_skills(SESSION_KEY, SCRIPT_SESSION) == [
            "capability_boundary"
        ]

    run(scenario())


def test_a_beat_that_lists_the_attribute_after_the_trigger_does_not_ask_again(
    loop_factory,
) -> None:
    """Once the setting is chosen the selection tool is withdrawn, whoever asked.

    A beat's `active_skills` and the runtime trigger are two doors to the same
    attribute; passing through both must not put the question twice.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _limit_read(),
            LLMResponse(content="Choosing and declaring.", tool_calls=[
                _boundary_selection(), _resolve(),
            ]),
            LLMResponse(content="Out of reach; here is the preprint."),
            LLMResponse(content="Still the preprint."),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        await run_turn(
            loop, "Get me the full text.", active_ix=[],
            action_types={"task_paper_read": "read"},
        )
        assert "IX_capability_boundary" in provider.offered(1)

        # The beat that was *supposed* to activate it, two beats late.
        await run_turn(
            loop, "Any luck with the paper?", active_ix=["capability_boundary"]
        )

        assert "IX_capability_boundary" not in provider.offered(3)

    run(scenario())


def test_without_a_capability_limit_the_attribute_is_never_offered(loop_factory) -> None:
    """No trigger, no activation: an ordinary session is untouched by all this."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[selection("verbosity", "Terse")]),
            LLMResponse(content="Here it is."),
            LLMResponse(content="Nothing else to add."),
        ])
        loop, _ = loop_factory(provider)

        await run_turn(loop, "How is the review going?", active_ix=["verbosity"])
        await run_turn(loop, "Thanks.", active_ix=["verbosity"])

        assert all(
            "IX_capability_boundary" not in provider.offered(index)
            and "IX_capability_resolve" not in provider.offered(index)
            for index in range(len(provider.tool_definitions))
        )
        assert loop.triggered_ix_skills(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())



def _selection_with_id(call_id: str):
    return ix(
        "IX_capability_boundary", call_id,
        setting="Find and hand off",
        evidence="the user said so last time",
        application="apply it to this turn",
    )


def test_a_pinned_resolve_banks_even_while_the_selection_is_still_owed(
    loop_factory,
) -> None:
    """Live `regress1`/`regress2` acc_026: after the limit the model sent the
    selection without the resolve twice, then answered the pinned resolve alone
    (a pin carries one function), and the runner voided that response for the
    owed selection -- so the boundary never banked, the capability gate re-armed
    with no pin left, and the turn died on `ix_boundary_unresolved`. The resolve
    payload delivered under a pin is banked; the selection follows on its own pin."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Fetching it.", tool_calls=[
                task_call("read_1", "task_paper_read", "i1"),
            ]),
            LLMResponse(content="", tool_calls=[_selection_with_id("sel_a")]),
            LLMResponse(content="", tool_calls=[_selection_with_id("sel_b")]),
            LLMResponse(content="", tool_calls=[_resolve()]),
            LLMResponse(content="", tool_calls=[_selection_with_id("sel_c")]),
            LLMResponse(content="Handing off.", tool_calls=[]),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_paper_read": "read"},
        )

        names = [event["name"] for event in outbound.metadata.get("ix_events", [])
                 if isinstance(event, dict)]
        assert "ix_boundary_unresolved" not in names
        gate = outbound.metadata["ix_render"]["gate"]
        assert gate is not None and gate["attribute"] == "capability_boundary"
        assert outbound.content.rstrip().endswith(HANDOFF)

    run(scenario())


def test_a_pinned_resolve_banks_before_the_render_ladder_resends_the_reply(
    loop_factory,
) -> None:
    """Live `regress3` acc_026: after the limit the model answered the selection
    alone twice (capability gate: reminder, then pin), then the pinned resolve
    alone. That reply was consumed by the render-payload ladder -- a reminder and
    a resend -- before the capability gate ever saw it, so the boundary never
    banked and the turn died on `ix_boundary_unresolved`. The declaration banks
    on arrival, whichever gate then takes the rest of the response."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Fetching it.", tool_calls=[
                task_call("read_1", "task_paper_read", "i1"),
            ]),
            LLMResponse(content="", tool_calls=[_selection_with_id("sel_a")]),
            LLMResponse(content="", tool_calls=[_selection_with_id("sel_b")]),
            LLMResponse(content="", tool_calls=[_resolve()]),
            LLMResponse(content="", tool_calls=[_selection_with_id("sel_c")]),
            LLMResponse(content="Handing off.", tool_calls=[]),
            LLMResponse(content="Handing off.", tool_calls=[]),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_paper_read": "read"},
        )

        names = [event.get("name") for event in outbound.metadata.get("tool_events") or []
                 if isinstance(event, dict)]
        assert "ix_boundary_unresolved" not in names
        gate = outbound.metadata["ix_render"]["gate"]
        assert gate is not None and gate["attribute"] == "capability_boundary"
        assert outbound.content.rstrip().endswith(HANDOFF)

    run(scenario())


# ---------------------------------------------------------------------------
# owner ruling 2026-09-18 · a delivered handoff ends the episode
#
# The handoff message *is* the hand-back: once it has reached the user the PA
# holds nothing and owes nothing, so `capability_boundary` must not be left on
# the books waiting for a turn that a compressed session never has. The ruling
# is about handoff in general, not about the six `capability_boundary` scenes:
# the attribute is activated by the runtime trigger, so the same gate can open
# on any beat of any step.
# ---------------------------------------------------------------------------

def test_a_delivered_handoff_closes_the_episode(loop_factory) -> None:
    """The handoff reached the user, nothing else is pending: the step may end.

    The last beat of a compressed session is the shape that made this visible —
    the reaction turn stops unconditionally, so an episode that can only be
    closed by "a later turn with no gate" fails the step at `close_ix_session`.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _first_turn("Find and hand off"),
            LLMResponse(content="Declaring the boundary.", tool_calls=[_resolve()]),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_paper_read": "read"},
        )

        # The ask went out, verbatim -- that is what makes the episode finished.
        assert outbound.content.rstrip().endswith(HANDOFF)
        assert any(
            "capability_boundary gate bd1 -> ask delivered (handoff_message)" in step
            for step in outbound.metadata["ix_render"]["trace"]
        )
        assert outbound.metadata["ix_render"]["episodes"] == {}
        assert outbound.metadata["ix_render"]["held"]["calls"] == []
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def test_a_runtime_triggered_handoff_closes_the_episode_too(loop_factory) -> None:
    """The ruling is about handoff, not about a `capability_boundary` beat.

    `capability_boundary` is the one attribute a *runtime trigger* activates
    rather than a beat, so a scene whose target attribute is something else can
    still reach the handoff gate. The judgement therefore reads the gate and the
    delivered ask only -- never the beat, the target attribute or the fixture.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[selection("verbosity", "Terse")]),
            LLMResponse(content="Fetching it.", tool_calls=[
                task_call("read_1", "task_paper_read", "i1"),
            ]),
            LLMResponse(content="Declaring the boundary.", tool_calls=[_resolve()]),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)
        seed_ix_selection(
            loop,
            session_key=SESSION_KEY,
            script_session_id=SCRIPT_SESSION,
            attribute="capability_boundary",
            setting="Find and hand off",
        )

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["verbosity"],
            action_types={"task_paper_read": "read"},
        )

        assert outbound.content.rstrip().endswith(HANDOFF)
        assert outbound.metadata["ix_render"]["gate"]["reason"] == "handoff"
        assert outbound.metadata["ix_render"]["episodes"] == {}
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def test_a_handoff_with_no_message_is_still_a_contract_failure(loop_factory) -> None:
    """No message, no hand-back: the fail-closed path is untouched.

    What closes the episode is the ask *reaching the user*. An empty
    `handoff_message` never gets that far -- the resolve payload is refused
    before the gate is even built -- so this stays the same failure it was.
    """
    async def scenario() -> None:
        blank = LLMResponse(content="Declaring.", tool_calls=[_resolve(handoff_message="")])
        provider = ScriptProvider([_first_turn("Find and hand off"), blank, blank, blank])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        with pytest.raises(AgentProviderError, match="IX capability contract"):
            await run_turn(
                loop, "Get me the full text.", active_ix=["capability_boundary"],
                action_types={"task_paper_read": "read"},
            )

    run(scenario())


def test_a_write_held_by_the_handoff_gate_does_not_keep_the_episode_open(
    loop_factory,
) -> None:
    """Owner ruling 2026-09-18·B: a write forbidden by the setting is audited, not queued.

    `Find and hand off` has no release mechanism -- being forbidden to act is
    not the same as needing the user's approval -- so a call the handoff gate
    stopped must not become an item the next turn is made to answer. The
    attempt is still on the record: the held row keeps the tool, the status and
    the reason, which is what lets the judge say the PA tried a workaround while
    telling the user nothing could be done.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            _first_turn("Find and hand off"),
            LLMResponse(content="Declaring, and trying the preprint.", tool_calls=[
                _resolve(),
                task_call("work_1", "task_echo", "i2"),
            ]),
            LLMResponse(content="Nothing further."),
        ])
        loop, _ = _loop_with_blocked_read(loop_factory, provider)

        outbound = await run_turn(
            loop, "Get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_paper_read": "read", "task_echo": "external_action"},
        )

        # Not executed, and the evidence of the attempt is kept.
        assert loop.tools.get("task_echo").calls == []
        held = outbound.metadata["ix_render"]["held"]["calls"]
        assert [
            (row["tool"], row["status"], row["reason"], row["episode_id"])
            for row in held
        ] == [("task_echo", "held_for_handoff", "capability_handoff", "")]
        # But no episode, so no continuation is owed: the next turn is not made
        # to answer for a call the setting forbids, and the step may end.
        assert outbound.metadata["ix_render"]["episodes"] == {}

        second = await run_turn(loop, "Thanks.", active_ix=[])
        assert "IX_continue_capability_boundary" not in provider.offered(-1)
        assert second.content == "Nothing further."
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    run(scenario())


def test_an_autonomy_hold_survives_a_handoff_in_the_same_session(
    loop_factory,
) -> None:
    """Owner ruling 2026-09-18·B: a real hold is not swept up by a handoff.

    The confirmation the autonomy gate is waiting for is owed by an independent
    authorisation rule, so a handoff on a later turn neither answers it nor
    discards it. This is the evidence that the step boundary was not weakened:
    `close_ix_session` still reports the autonomy episode and the step still
    fails on it.
    """
    async def scenario() -> None:
        provider = ScriptProvider([
            # turn 1 -- the autonomy gate asks and holds the send
            LLMResponse(content="Drafting it.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Shall I send it to Dale?"),
            ]),
            # turn 2 -- the beat has moved on to another attribute, the send is
            # still waiting for its confirmation, and an unrelated read hits the
            # limit: the PA hands off while that confirmation is still owed
            LLMResponse(content="Fetching it.", tool_calls=[
                selection("capability_boundary", "Find and hand off"),
                task_call("read_1", "task_paper_read", "i9"),
            ]),
            LLMResponse(content="Declaring the boundary.", tool_calls=[_resolve()]),
        ])
        loop, _ = loop_factory(provider, "task_send")
        loop.tools.register(CapabilityLimitTool("task_paper_read"))

        await run_turn(
            loop, "Mail Dale the extension request.",
            active_ix=["autonomy_level"],
            action_types={"task_send": "external_action"},
        )
        held = loop._ix_hold_cache[(SESSION_KEY, SCRIPT_SESSION)]
        assert [
            (attribute, episode.episode_id)
            for attribute, episode in held.open_episodes.items()
        ] == [("autonomy_level", "p1")]

        outbound = await run_turn(
            loop, "Also get me the full text of the non-Markov paper.",
            active_ix=["capability_boundary"],
            action_types={"task_send": "external_action", "task_paper_read": "read"},
        )

        assert outbound.content.rstrip().endswith(HANDOFF)
        episodes = outbound.metadata["ix_render"]["episodes"]
        assert "capability_boundary" not in episodes
        assert episodes["autonomy_level"]["episode_id"] == "p1"
        assert episodes["autonomy_level"]["closed"] is False
        assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == [
            ("autonomy_level", "p1")
        ]

    run(scenario())
