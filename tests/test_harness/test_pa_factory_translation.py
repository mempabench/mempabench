"""The harness translates the PA's IX face into keys the simulator can read.

`simulator/` never imports `pa/` and never branches on `ix_render`, so the two
things it does need from a rendered turn — the proactive send/suppress decision
and the held ids the consent sideband refers to — are restated by `RecordingPA`,
the object the simulator holds as `pa_loop`.

    PYTHONPATH=. python -m pytest tests/test_harness/test_pa_factory_translation.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from harness.pa_factory import (
    RecordingPA,
    build_pa_from_provider_profile,
    has_pending_interaction_from_ix_render,
    held_items_from_ix_render,
    outreach_keys_from_ix_render,
    reply_facts_from_ix_render,
)
from harness.run_plan import DEFAULT_PA_MAX_TOKENS
from pa.agent.ix.render_types import IX_CONTRACT_VERSION, IX_RENDER_SCHEMA_VERSION
from pa.agent.loop import AgentLoop
from pa.agent.tools.base import Tool
from pa.bus.queue import MessageBus
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest


@dataclass
class _Outbound:
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class _StubPA:
    def __init__(self, outbound: _Outbound) -> None:
        self.outbound = outbound

    async def process_direct_message(self, **_kwargs: Any) -> _Outbound:
        return self.outbound

    async def process_environment_event(self, **_kwargs: Any) -> _Outbound:
        return self.outbound


def _ix_render(**overrides: Any) -> dict:
    blob = {
        "schema_version": IX_RENDER_SCHEMA_VERSION,
        "ix_contract_version": IX_CONTRACT_VERSION,
        "episodes": {},
        "released": {"batches": [], "questions": [], "plan_executed": []},
        "held": {
            "batches": [],
            "questions": [],
            "plan_remaining": [],
            "calls": [],
            "cancelled": [],
            "outreach_message": "",
        },
        "gate": None,
        "consent": [],
        "notices": [],
        "completion": {"released": True, "text": ""},
        "reasoning": {
            "shown": False,
            "source": "user_facing_rationale",
            "provider_reasoning_present": False,
        },
        "certainties": [],
        "proactive_decision": None,
        "trace": [],
    }
    blob.update(overrides)
    return blob


# ---------------------------------------------------------------------------
# (1) proactive decision -> outreach_decision / outreach_delivered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, delivered",
    [("send", True), ("suppress", False)],
)
def test_proactive_decision_becomes_the_two_keys_the_simulator_reads(
    action: str, delivered: bool
) -> None:
    keys = outreach_keys_from_ix_render(
        _ix_render(
            proactive_decision={
                "action": action,
                "setting": "High",
                "message": "Your flight moved to 07:40.",
            }
        )
    )

    assert keys == {"outreach_decision": action, "outreach_delivered": delivered}


def test_the_translation_carries_no_setting_and_no_suppressed_message() -> None:
    blob = _ix_render(
        proactive_decision={
            "action": "suppress",
            "setting": "Low",
            "message": "SUPPRESSED-MARKER your flight moved.",
        },
        held={
            "batches": [],
            "questions": [],
            "plan_remaining": [],
            "calls": [],
            "cancelled": [],
            "outreach_message": "SUPPRESSED-MARKER your flight moved.",
        },
    )

    keys = outreach_keys_from_ix_render(blob)

    assert "SUPPRESSED-MARKER" not in str(keys)
    assert "Low" not in str(keys.values())


def test_an_ordinary_turn_gets_no_outreach_keys() -> None:
    assert outreach_keys_from_ix_render(_ix_render()) == {}


@pytest.mark.asyncio
async def test_environment_event_result_carries_the_translated_decision() -> None:
    outbound = _Outbound(
        content="Your flight moved to 07:40.",
        metadata={
            "ix_render": _ix_render(
                proactive_decision={
                    "action": "send",
                    "setting": "High",
                    "message": "Your flight moved to 07:40.",
                }
            )
        },
    )
    pa = RecordingPA(_StubPA(outbound))

    result = await pa.process_environment_event(detail="flight change")

    # This is exactly the shape `simulator/loop.py` asserts on.
    assert result.metadata["outreach_decision"] == "send"
    assert result.metadata["outreach_delivered"] is True
    assert pa.received[0]["source"] == "environment"


# ---------------------------------------------------------------------------
# (2) held ids -> held_items for the consent sideband
# ---------------------------------------------------------------------------


def _open(attribute: str = "autonomy_level", episode_id: str = "p1") -> dict:
    return {attribute: {"episode_id": episode_id, "rounds_open": 1, "closed": False}}


def test_only_the_ids_the_ask_named_are_offered_for_consent() -> None:
    """§3.7 part 3: the criterion is "shown to the user", not "still held".

    Live, an ask that named one cancellation came back with an `allow` for six
    ids. `ix_render.held` still lists every held call for the judge; the
    sideband is offered only what the ask put to the user.
    """
    blob = _ix_render(
        episodes=_open(),
        held={
            "batches": [],
            "questions": [],
            "plan_remaining": [
                {"id": "a3", "intent": "email the team", "bound_item_ids": ["i3"]},
                {"id": "a4", "intent": "search the calendar", "bound_item_ids": ["i4"]},
            ],
            "calls": [
                {
                    "interaction_item_id": "i3",
                    "plan_item_id": "a3",
                    "episode_id": "p1",
                    "tool": "mcp_state_email_message_send",
                    "reason": "external_action_floor",
                    "status": "held_for_confirmation",
                    "attribute": "autonomy_level",
                    "asked": True,
                },
                {
                    "interaction_item_id": "i4",
                    "plan_item_id": "a4",
                    "episode_id": "p1",
                    "tool": "mcp_state_calendar_event_search",
                    "reason": "after_open_gate",
                    "status": "held_for_confirmation",
                    "attribute": "autonomy_level",
                    "asked": False,
                },
            ],
            "cancelled": [],
            "outreach_message": "",
        },
        gate={
            "attribute": "autonomy_level",
            "item_id": "i3",
            "reason": "external_action_floor",
            "action_type": "external_action",
            "message_source": "_ix_confirmation_message",
        },
    )

    assert held_items_from_ix_render(blob) == [
        {"interaction_item_id": "i3", "intent": "email the team"}
    ]


def test_held_items_carry_the_id_and_intent_and_nothing_else() -> None:
    blob = _ix_render(
        episodes=_open(),
        held={
            "batches": [],
            "questions": [],
            "plan_remaining": [
                {"id": "a3", "intent": "email the team", "bound_item_ids": ["i3"]}
            ],
            "calls": [
                {
                    "interaction_item_id": "i3",
                    "plan_item_id": "a3",
                    "episode_id": "p1",
                    "tool": "mcp_state_email_message_send",
                    "reason": "external_action_floor",
                    "status": "held_for_confirmation",
                    "attribute": "autonomy_level",
                }
            ],
            "cancelled": [],
            "outreach_message": "",
        },
    )

    items = held_items_from_ix_render(blob)

    assert items == [{"interaction_item_id": "i3", "intent": "email the team"}]
    text = str(items)
    for leak in ("autonomy_level", "external_action_floor", "held_for_confirmation"):
        assert leak not in text


def test_a_bound_plan_item_without_a_held_call_still_gets_an_entry() -> None:
    blob = _ix_render(
        episodes=_open(),
        held={
            "batches": [],
            "questions": [],
            "plan_remaining": [
                {"id": "a4", "intent": "book the hotel", "bound_item_ids": ["i4"]}
            ],
            "calls": [],
            "cancelled": [],
            "outreach_message": "",
        },
    )

    assert held_items_from_ix_render(blob) == [
        {"interaction_item_id": "i4", "intent": "book the hotel"}
    ]


def test_a_plan_item_the_wrapper_cannot_address_is_not_offered() -> None:
    """The sideband keys on `interaction_item_id`; no id, no verdict."""
    blob = _ix_render(
        episodes=_open(),
        held={
            "batches": [],
            "questions": [],
            "plan_remaining": [{"id": "a5", "intent": "unbound step"}],
            "calls": [],
            "cancelled": [],
            "outreach_message": "",
        },
    )

    assert held_items_from_ix_render(blob) == []


def test_a_held_call_no_open_episode_owns_is_not_offered_for_consent() -> None:
    """A verdict on an id the wrapper is not holding is consent to nothing.

    A fail-closed binding batch and a spent release are both recorded in
    `held.calls` for the judge, and neither belongs to an episode. Offering
    them would leave a stale `allow` sitting ready for whatever reuses the id.
    """
    call = {
        "interaction_item_id": "i3",
        "plan_item_id": "",
        "episode_id": "",
        "tool": "mcp_state_email_message_send",
        "reason": "binding_invalid",
        "status": "held_for_confirmation",
        "attribute": "autonomy_level",
    }
    held = {
        "batches": [],
        "questions": [],
        "plan_remaining": [],
        "calls": [call],
        "cancelled": [],
        "outreach_message": "",
    }

    # No episode at all, and an episode that does not own this call.
    assert held_items_from_ix_render(_ix_render(held=held)) == []
    assert held_items_from_ix_render(_ix_render(episodes=_open(), held=held)) == []
    # The same row, once an open episode owns it.
    assert held_items_from_ix_render(
        _ix_render(episodes=_open(), held={**held, "calls": [{**call, "episode_id": "p1"}]})
    ) == [{"interaction_item_id": "i3", "intent": ""}]


@pytest.mark.asyncio
async def test_direct_message_result_carries_held_items(tmp_path) -> None:
    outbound = _Outbound(
        content="Shall I send it?",
        metadata={
            "ix_render": _ix_render(
                episodes=_open(),
                held={
                    "batches": [],
                    "questions": [],
                    "plan_remaining": [
                        {"id": "a3", "intent": "email the team", "bound_item_ids": ["i3"]}
                    ],
                    "calls": [
                        {
                            "interaction_item_id": "i3",
                            "plan_item_id": "a3",
                            "episode_id": "p1",
                            "tool": "mcp_state_email_message_send",
                            "reason": "external_action_floor",
                            "status": "held_for_confirmation",
                        }
                    ],
                    "cancelled": [],
                    "outreach_message": "",
                },
            )
        },
    )
    pa = RecordingPA(_StubPA(outbound))

    result = await pa.process_direct_message(user_message="mail the team")

    assert result.metadata["held_items"] == [
        {"interaction_item_id": "i3", "intent": "email the team"}
    ]


# ---------------------------------------------------------------------------
# (3) has_pending_interaction -> the simulator's pacing flag
# ---------------------------------------------------------------------------


def _question_hold(question_id: str = "q2") -> dict:
    """An Iterative hold: q2 kept back, no call and no plan item behind it."""
    return {
        "batches": [],
        "questions": [{"id": question_id, "text": "HELD_SECRET_Q2", "about": "budget"}],
        "plan_remaining": [],
        "calls": [],
        "cancelled": [],
        "outreach_message": "",
    }


def test_a_question_only_hold_is_pending_but_offers_nothing_to_consent_to() -> None:
    """The R7 case: `held_items` is empty, yet the PA is mid-question.

    Advancing on the empty list walks out of the ask, and on the last beat ends
    the session with the episode open.
    """
    blob = _ix_render(
        episodes=_open("information_elicitation", "e1"), held=_question_hold()
    )

    assert held_items_from_ix_render(blob) == []
    assert has_pending_interaction_from_ix_render(blob) is True


def test_the_pending_flag_says_only_that_something_waits() -> None:
    blob = _ix_render(
        episodes=_open("topic_management", "b1"),
        held={
            "batches": [{"batch_id": "b2", "label": "the rest", "body": "HELD_BODY"}],
            "questions": [],
            "plan_remaining": [],
            "calls": [],
            "cancelled": [],
            "outreach_message": "",
        },
    )

    flag = has_pending_interaction_from_ix_render(blob)

    # A bool carries no body, no id and no setting name by construction.
    assert flag is True and isinstance(flag, bool)


def test_a_closed_or_absent_episode_is_not_pending() -> None:
    assert has_pending_interaction_from_ix_render(_ix_render()) is False
    closed = {"information_elicitation": {
        "episode_id": "e1", "rounds_open": 2, "closed": True,
    }}
    assert has_pending_interaction_from_ix_render(
        _ix_render(episodes=closed, held=_question_hold())
    ) is False


@pytest.mark.asyncio
async def test_direct_message_result_carries_the_pending_flag() -> None:
    outbound = _Outbound(
        content="What is the budget?",
        metadata={
            "ix_render": _ix_render(
                episodes=_open("information_elicitation", "e1"),
                held=_question_hold(),
            )
        },
    )
    pa = RecordingPA(_StubPA(outbound))

    result = await pa.process_direct_message(user_message="book something")

    assert result.metadata["has_pending_interaction"] is True
    # The consent list stays exactly what it was: nothing to answer here.
    assert "held_items" not in result.metadata
    assert "HELD_SECRET_Q2" not in str(result.metadata["has_pending_interaction"])


@pytest.mark.asyncio
async def test_a_turn_without_ix_render_gains_nothing_but_the_trigger_list() -> None:
    """`triggered_skills` is pacing state, not a projection of `ix_render`.

    It says which attributes a runtime trigger has activated in this session, so
    it is written on every PA turn — a turn that rendered nothing still has to
    say "nothing has fired". Everything else stays untouched.
    """
    outbound = _Outbound(content="hello", metadata={"tool_events": []})
    pa = RecordingPA(_StubPA(outbound))

    result = await pa.process_direct_message(user_message="hi")

    assert result.metadata == {"tool_events": [], "triggered_skills": []}


# ---------------------------------------------------------------------------
# The run id must not reach anything the model reads
# ---------------------------------------------------------------------------

RUN_ID = "ixval_proactive_low_002"
SESSION_KEY = f"{RUN_ID}_main"


class _CapturingProvider(LLMProvider):
    """Records every message list handed to the provider."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.calls: list[list[dict[str, Any]]] = []
        self.responses = list(responses)

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return self.responses.pop(0)


def _prompt_text(provider: _CapturingProvider) -> str:
    return json.dumps(provider.calls, ensure_ascii=False, default=str)


def _loop(tmp_path: Path, provider: _CapturingProvider) -> AgentLoop:
    loop = AgentLoop(
        bus=MessageBus(),
        pa_llm_provider=provider,
        workspace=tmp_path,
        pa_model="test-model",
    )
    loop.sessions.create(SESSION_KEY)
    return loop


@pytest.mark.asyncio
async def test_the_environment_event_prompt_carries_no_run_id(tmp_path: Path) -> None:
    """The chat id the PA sees is opaque, so it cannot name the condition.

    `pa/agent/context.py` renders `Channel:` / `Chat ID:` into the runtime
    context block, and the caller used to pass the run-derived session key. Two
    live proactive runs then justified their own answer with "the channel name
    ixval_proactive_low_002_main indicates this turn evaluates the Low setting".
    """
    provider = _CapturingProvider([
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(
                id="ix_proactive",
                name="IX_proactive_outreach",
                arguments={
                    "setting": "Low",
                    "evidence": "Retrieved preference evidence.",
                    "application": "suppress this contact.",
                    "outreach_action": "suppress",
                    "outreach_message": "The water shutoff is tomorrow morning.",
                },
            )
        ]),
        LLMResponse(content="Internal completion text."),
    ])
    pa = RecordingPA(_loop(tmp_path, provider))

    await pa.process_environment_event(
        trigger_id="water_shutoff",
        source_type="external_event",
        prior_thread="Building management notices.",
        detail="A temporary water shutoff is scheduled.",
        user_impact="User A may want to plan around it.",
        session_key=SESSION_KEY,
        # What the simulator passes; the harness seam overrides both.
        channel="benchmark",
        chat_id=SESSION_KEY,
        metadata={"run_id": RUN_ID, "persona_id": "user_a"},
    )

    prompt = _prompt_text(provider)
    assert provider.calls
    assert RUN_ID not in prompt
    assert "Chat ID: main" in prompt
    assert "Channel: benchmark" in prompt


@pytest.mark.asyncio
async def test_the_direct_message_prompt_carries_no_run_id(tmp_path: Path) -> None:
    provider = _CapturingProvider([LLMResponse(content="Noted.")])
    pa = RecordingPA(_loop(tmp_path, provider))

    await pa.process_direct_message(
        user_message="Where do we stand?",
        session_key=SESSION_KEY,
        metadata={"run_id": RUN_ID, "persona_id": "user_a"},
    )

    prompt = _prompt_text(provider)
    assert provider.calls
    assert RUN_ID not in prompt
    assert "Chat ID: main" in prompt


# ---------------------------------------------------------------------------
# `triggered_skills` — an attribute a runtime trigger activated (audit §2)
# ---------------------------------------------------------------------------

SCRIPT_SESSION = "acc_026"


class _BlockedReadTool(Tool):
    """A read whose record the fixture marks as unobtainable (§3.5 part 3)."""

    @property
    def name(self) -> str:
        return "task_paper_read"

    @property
    def description(self) -> str:
        return "Read a paper's full text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"paper_id": {"type": "string"}}}

    async def execute(self, **_kwargs: Any) -> str:
        return json.dumps({
            "ok": False,
            "error_code": "capability_limit",
            "boundary_kind": "missing_data",
        })


def _ix_call(name: str, **arguments: Any) -> ToolCallRequest:
    return ToolCallRequest(id=f"{name}_1", name=name, arguments=arguments)


async def _beat(pa: RecordingPA, message: str) -> Any:
    """One benchmark turn on a beat whose script names no IX attribute."""
    return await pa.process_direct_message(
        user_message=message,
        session_key=SESSION_KEY,
        metadata={
            "run_id": RUN_ID,
            "persona_id": "user_a",
            "script_session_id": SCRIPT_SESSION,
            "active_ix": [],
            "ix_context": {
                "session_active_ix": [],
                "tool_action_types": {"task_paper_read": "read"},
            },
        },
    )


@pytest.mark.asyncio
async def test_triggered_skills_is_written_every_turn_and_sticks_from_the_trigger(
    tmp_path: Path,
) -> None:
    """Audit §2 (acc_026): the capability limit fires on a beat that lists nothing.

    The simulator cannot read `ix_render`, so this is how it learns that
    `capability_boundary` is live on a beat whose script never named it. Written
    on every PA turn — `[]` before anything fires — and sticky afterwards.
    """
    provider = _CapturingProvider([
        LLMResponse(content="Noted."),
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(
                id="read_1", name="task_paper_read", arguments={"paper_id": "p1"},
            ),
        ]),
        LLMResponse(content="Choosing and declaring.", tool_calls=[
            _ix_call(
                "IX_capability_boundary",
                setting="Suggest alternatives",
                evidence="she asked for options last time",
                application="offer the preprint instead",
            ),
            _ix_call(
                "IX_capability_resolve",
                boundary_id="bd1",
                blocked_goal="get the full text",
                boundary_kind="missing_data",
                recovery_options=[{"id": "r1", "intent": "email the author"}],
                handoff_message="Shall I write to the author?",
            ),
        ]),
        LLMResponse(content="The full text is out of reach; the preprint covers it."),
        LLMResponse(content="Anything else?"),
    ])
    loop = _loop(tmp_path, provider)
    loop.tools.register(_BlockedReadTool())
    pa = RecordingPA(loop)

    before = await _beat(pa, "How is the review going?")
    assert before.metadata["triggered_skills"] == []

    triggered = await _beat(pa, "Get me the full text of that paper.")
    assert triggered.metadata["triggered_skills"] == ["capability_boundary"]

    after = await _beat(pa, "And the preprint?")
    assert after.metadata["triggered_skills"] == ["capability_boundary"]

    # The step boundary clears it with the rest of the session's IX state.
    pa.close_ix_session(SESSION_KEY, SCRIPT_SESSION)
    assert loop.triggered_ix_skills(SESSION_KEY, SCRIPT_SESSION) == []


# ---------------------------------------------------------------------------
# The frozen plan's PA completion-token limit reaches the request
# ---------------------------------------------------------------------------


class _MaxTokensProvider(LLMProvider):
    """Records the completion-token limit each request carried."""

    def __init__(self) -> None:
        super().__init__()
        self.max_tokens: list[int] = []

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.max_tokens.append(max_tokens)
        return LLMResponse(content="Noted.")


class _StubProviderProfile:
    """Stands in for `AssistantLLMProviderProfile` without a policy catalog."""

    provider_name = "stub"

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.roles: list[str] = []

    def create_provider(self, *, model: str, role: str, **_kwargs: Any) -> LLMProvider:
        self.roles.append(role)
        return self.provider


@pytest.mark.asyncio
@pytest.mark.parametrize("pa_max_tokens", [DEFAULT_PA_MAX_TOKENS, 16_384])
async def test_the_plans_pa_max_tokens_reaches_the_pa_request(
    tmp_path: Path, pa_max_tokens: int
) -> None:
    """regress16 acc_006 was cut off six times at the provider dataclass's 4096.

    Nothing on the PA path ever set a limit: `AgentRunSpec.max_tokens` stays
    `None`, so the request inherits the provider's `GenerationSettings`. That is
    the one place the frozen plan's value has to land, and this drives a real
    turn to prove it arrives in the request the provider receives.
    """
    assert DEFAULT_PA_MAX_TOKENS == 8192

    provider = _MaxTokensProvider()
    profile = _StubProviderProfile(provider)
    loop = build_pa_from_provider_profile(
        provider_profile=profile,
        workspace=tmp_path,
        pa_model="test-model",
        pa_max_tokens=pa_max_tokens,
    )
    loop.sessions.create(SESSION_KEY)

    await _beat(RecordingPA(loop), "How is the review going?")

    assert profile.roles == ["pa"]
    assert provider.max_tokens == [pa_max_tokens]


# ---------------------------------------------------------------------------
# (5) the delivered reply -> reply_facts for the actor's private record
# ---------------------------------------------------------------------------


def test_reply_facts_count_released_notices_completion_and_brackets() -> None:
    """Owner ruling 2026-09-17: the actor is told how many progress lines and
    whether a closing line were delivered, and whether bracketed confidence
    tags appeared — counts and flags off the delivered reply, nothing else."""
    blob = _ix_render(
        notices=[
            {"order": 0, "tool": "mcp_state_document_read", "text": "Checking the folder."},
            {"order": 1, "tool": "mcp_state_document_update", "text": "Updating the resume."},
        ],
        completion={"released": True, "text": "The resume is updated."},
        certainties=[
            {"ref": "c1", "level": "high", "rendered": " (confidence: high)"},
            {"ref": "c2", "level": "low", "rendered": ""},
        ],
    )

    assert reply_facts_from_ix_render(blob) == {
        "notices_released": 2,
        "completion_released": True,
        "confidence_brackets": True,
    }
    assert reply_facts_from_ix_render(_ix_render()) == {
        "notices_released": 0,
        "completion_released": True,
        "confidence_brackets": False,
    }


def test_reply_facts_carry_nothing_held_no_gate_and_no_setting() -> None:
    blob = _ix_render(
        episodes=_open("topic_management", "b7"),
        held={
            "batches": [{"batch_id": "SECRET_BATCH", "text": "SECRET_BATCH_BODY"}],
            "questions": [{"text": "SECRET_QUESTION"}],
            "plan_remaining": [{"id": "a9", "intent": "SECRET_INTENT"}],
            "calls": [],
            "cancelled": [],
            "outreach_message": "",
        },
        gate={
            "attribute": "autonomy_level",
            "item_id": "SECRET_ITEM",
            "reason": "external_action_floor",
            "action_type": "external_action",
            "message_source": "_ix_confirmation_message",
        },
        completion={"released": False, "text": "SECRET_CLOSING", "skipped": "gate"},
        certainties=[{"ref": "c1", "level": "low", "rendered": "", "claim": "SECRET_CLAIM"}],
    )

    keys = reply_facts_from_ix_render(blob)

    assert set(keys) == {"notices_released", "completion_released", "confidence_brackets"}
    assert "SECRET" not in str(keys)
    assert keys["completion_released"] is False
    assert keys["confidence_brackets"] is False


@pytest.mark.asyncio
async def test_direct_message_result_carries_reply_facts_next_to_held_items() -> None:
    outbound = _Outbound(
        content="Updated the resume.",
        metadata={
            "ix_render": _ix_render(
                notices=[{"order": 0, "tool": "mcp_state_document_update", "text": "Updating."}],
                completion={"released": True, "text": "Done."},
            )
        },
    )
    pa = RecordingPA(_StubPA(outbound))

    result = await pa.process_direct_message(user_message="update the resume")

    assert result.metadata["reply_facts"] == {
        "notices_released": 1,
        "completion_released": True,
        "confidence_brackets": False,
    }


@pytest.mark.asyncio
async def test_a_turn_without_a_render_writes_no_reply_facts() -> None:
    pa = RecordingPA(_StubPA(_Outbound(content="hi", metadata={})))
    result = await pa.process_direct_message(user_message="hi")
    assert "reply_facts" not in result.metadata
