"""`--force-ix`: forcing one IX setting for the PA in a validation run.

Some validation cases need settings user_a's matrix never produces (`Bookend`,
`Reactive`). They assumed the simulator's preference overlay could make them,
but that overlay only rewrites the *user* matrix — nothing of it reaches the PA.
`harness/ix_seed.py` seeds the PA's own per-session selection cache instead, and
these tests pin the three things that makes true: the seeded setting really
renders, the attribute's selection tool is no longer offered, and the seed dies
at the step boundary like any other selection.
"""

from __future__ import annotations

import json

import pytest

from harness.ix_seed import seed_ix_selection
from pa.agent.ix.render import settings_from_selections
from pa.providers.base import LLMResponse
from scripts.run_orchestrator import _parse_force_ix

from .conftest import (
    SCRIPT_SESSION,
    SESSION_KEY,
    ScriptProvider,
    ix,
    run,
    run_turn,
    task_call,
)


def _seed(loop, attribute: str, setting: str):
    return seed_ix_selection(
        loop,
        session_key=SESSION_KEY,
        script_session_id=SCRIPT_SESSION,
        attribute=attribute,
        setting=setting,
    )


# ---------------------------------------------------------------------------
# (a) the seeded setting renders, and its selection tool is not offered
# ---------------------------------------------------------------------------

def test_seeded_setting_renders_and_its_selection_tool_is_not_offered(
    loop_factory,
) -> None:
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content=None, tool_calls=[
                ix("IX_render_process_visibility",
                   completion_message="Monday morning is in the calendar."),
                task_call("c1", "task_search", "", notice="Let me look for a free slot."),
                task_call("c2", "task_create", "", notice="Putting the meeting in now."),
            ]),
            LLMResponse(content="Monday 10:00 is booked."),
        ])
        loop, _ = loop_factory(provider, "task_search", "task_create")
        _seed(loop, "process_visibility", "Full narration")

        outbound = await run_turn(
            loop, "Book Monday morning.", active_ix=["process_visibility"],
            action_types={"task_search": "read", "task_create": "internal_write"},
        )

        # `Full narration` released both notices and the completion line: the
        # seed reached the renderer, no selection tool call anywhere in sight.
        assert "Let me look for a free slot." in outbound.content
        assert "Putting the meeting in now." in outbound.content
        assert "Monday morning is in the calendar." in outbound.content
        # The PA was never asked to choose: the setting is already made.
        assert "IX_process_visibility" not in provider.offered(0)
        # The per-turn payload tool still is offered — it carries this turn's
        # notices and completion line, whoever chose the setting.
        assert "IX_render_process_visibility" in provider.offered(0)
        assert settings_from_selections(
            loop._ix_selection_cache[(SESSION_KEY, SCRIPT_SESSION)].values()
        ) == {"process_visibility": "Full narration"}

    run(scenario())


# ---------------------------------------------------------------------------
# (a2) the seed survives the environment-event turn
# ---------------------------------------------------------------------------
# The proactive round inherits no settings,
# so `IX_proactive_outreach` is still offered -- it has to be, it is the only
# carrier of the candidate contact -- and the model's own answer used to
# overwrite the seed in `_ix_selection_cache` before the renderer read it. The
# seed is `forced` now, so the renderer applies its override instead.


async def _proactive(loop):
    return await loop.process_environment_event(
        trigger_id="pickup_deadline",
        source_type="deadline",
        prior_thread="The comic order is ready for pickup.",
        detail="The store closes in two hours.",
        user_impact="User A may want time to collect the order today.",
        session_key=SESSION_KEY,
        metadata={
            "run_id": "run_001",
            "persona_id": "user_a",
            "script_session_id": SCRIPT_SESSION,
        },
    )


def _proactive_provider(*, setting: str, action: str, message: str) -> ScriptProvider:
    return ScriptProvider([
        LLMResponse(content=None, tool_calls=[
            ix("IX_proactive_outreach",
               setting=setting,
               evidence="nothing on file",
               application=f"{action} this contact",
               outreach_action=action,
               outreach_message=message),
        ]),
        LLMResponse(content="Internal completion text."),
    ])


def test_seeded_low_suppresses_the_contact_the_pa_wanted_to_send(loop_factory) -> None:
    async def scenario() -> None:
        message = "The comic store closes at 6 PM today."
        # The PA reads the event as worth sending; the seeded Low overrules it.
        provider = _proactive_provider(setting="High", action="send", message=message)
        loop, adapter = loop_factory(provider)
        _seed(loop, "proactive_outreach", "Low")

        outbound = await _proactive(loop)

        assert outbound.content == ""
        assert outbound.metadata["outreach_decision"] == "suppress"
        assert outbound.metadata["outreach_delivered"] is False
        ix_render = outbound.metadata["ix_render"]
        assert ix_render["proactive_decision"]["setting"] == "Low"
        assert ix_render["held"]["outreach_message"] == message
        # Nothing of the suppressed *contact* reaches history or memory; the
        # decision to stay silent does, because that is what is scored.
        assert loop.sessions.get(SESSION_KEY).messages == []
        assert len(adapter.events) == 1
        assert [item.kind for item in adapter.events[0].items][-1] == "no_reply"
        assert message not in json.dumps(adapter.events[0].messages, default=str)
        # The tool stays on the turn: it is what wrote the candidate message.
        assert "IX_proactive_outreach" in provider.offered(0)
        assert loop._ix_selection_cache[(SESSION_KEY, SCRIPT_SESSION)][
            "proactive_outreach"
        ].setting == "Low"

    run(scenario())


def test_seeded_high_delivers_the_contact_the_pa_wanted_to_suppress(loop_factory) -> None:
    async def scenario() -> None:
        message = "The comic store closes at 6 PM today."
        provider = _proactive_provider(setting="Low", action="suppress", message=message)
        loop, adapter = loop_factory(provider)
        _seed(loop, "proactive_outreach", "High")

        outbound = await _proactive(loop)

        assert outbound.content == message
        assert outbound.metadata["outreach_decision"] == "send"
        assert outbound.metadata["outreach_delivered"] is True
        ix_render = outbound.metadata["ix_render"]
        assert ix_render["proactive_decision"]["setting"] == "High"
        assert ix_render["held"]["outreach_message"] == ""
        assert loop.sessions.get(SESSION_KEY).messages[-1]["content"] == message
        assert adapter.events

    run(scenario())


# ---------------------------------------------------------------------------
# (b) the seed is session-scoped like any other selection
# ---------------------------------------------------------------------------

def test_close_ix_session_drops_the_seed(loop_factory) -> None:
    loop, _ = loop_factory(ScriptProvider([]))
    _seed(loop, "autonomy_level", "Reactive")
    assert (SESSION_KEY, SCRIPT_SESSION) in loop._ix_selection_cache

    assert loop.close_ix_session(SESSION_KEY, SCRIPT_SESSION) == []

    assert (SESSION_KEY, SCRIPT_SESSION) not in loop._ix_selection_cache


# ---------------------------------------------------------------------------
# (c) a typo is a crash, never a silently unforced run
# ---------------------------------------------------------------------------

def test_unknown_attribute_raises(loop_factory) -> None:
    loop, _ = loop_factory(ScriptProvider([]))
    with pytest.raises(KeyError):
        _seed(loop, "process_visibilty", "Full narration")
    assert loop._ix_selection_cache == {}


def test_unknown_setting_raises(loop_factory) -> None:
    loop, _ = loop_factory(ScriptProvider([]))
    with pytest.raises(ValueError, match="unknown setting for process_visibility"):
        _seed(loop, "process_visibility", "Full Narration")
    assert loop._ix_selection_cache == {}


# ---------------------------------------------------------------------------
# (d) the CLI form
# ---------------------------------------------------------------------------

def test_parse_force_ix_keeps_the_space_in_a_two_word_setting() -> None:
    assert _parse_force_ix(["acc_003:process_visibility=Full narration"]) == {
        "acc_003": {"process_visibility": "Full narration"},
    }


def test_parse_force_ix_takes_two_entries() -> None:
    assert _parse_force_ix([
        "acc_003:process_visibility=Bookend",
        "acc_040:autonomy_level=Reactive",
    ]) == {
        "acc_003": {"process_visibility": "Bookend"},
        "acc_040": {"autonomy_level": "Reactive"},
    }


def test_parse_force_ix_defaults_to_nothing_forced() -> None:
    assert _parse_force_ix(None) == {}
    assert _parse_force_ix([]) == {}


@pytest.mark.parametrize(
    "raw",
    ["acc_003", "acc_003:process_visibility", "process_visibility=Bookend",
     ":process_visibility=Bookend", "acc_003:=Bookend", "acc_003:autonomy_level="],
)
def test_parse_force_ix_rejects_a_malformed_value(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid --force-ix value"):
        _parse_force_ix([raw])
