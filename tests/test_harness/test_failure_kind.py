"""The typed failure vocabulary: one word per way a step can die."""

from __future__ import annotations

import pytest

from harness.failure_kind import (
    FAILURE_KIND_ACTIONS,
    FAILURE_KINDS,
    JUDGE_VOID_KINDS,
    PROBE_RETRY_KINDS,
    classify_failure,
)
from harness.forced_event_executor import ForcedEventExecutionError
from harness.memory_io import MemoryIOAuditError
from harness.pa_factory import OpenHoldEpisodeError
from harness.run_plan import RunPlanLoadError
from pa.agent.ix.render_types import IXContractVersionError
from pa.agent.loop import AgentLoop, AgentProviderError
from pa.agent.runner import AgentRunResult
from simulator.judge.client import JudgeUnavailable
from simulator.output import (
    SESSION_INVALID_RULES,
    SessionInvalidError,
    SimulatorOutputContractError,
)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (OpenHoldEpisodeError("acc_002", [("autonomy_level", "p1")]), "ix_open_hold"),
        (IXContractVersionError("ix_render written under 1.0.0"), "ix_contract"),
        (
            AgentProviderError(
                "PA turn failed the IX render contract: a required "
                "IX_render_<attribute> payload was missing twice, so this "
                "turn had no per-turn render data"
            ),
            "ix_runtime",
        ),
        (AgentProviderError("PA provider failed to produce a response"), "provider"),
        (
            SessionInvalidError(
                "beat_ask_unfulfilled", beat="b3_probe", attempts=3
            ),
            "session_invalid",
        ),
        (
            JudgeUnavailable(
                "ollama is not reachable at http://127.0.0.1:11434: "
                "[Errno 61] Connection refused"
            ),
            "judge_unavailable",
        ),
        # The three judge failures the loop voids the session on. They stop the
        # round like any other session-level failure, but they keep their own
        # word, because "look at the simulator" is the wrong first move.
        (
            SessionInvalidError("judge_timeout", beat="b3_probe", attempts=1),
            "judge_timeout",
        ),
        (
            SessionInvalidError("judge_unparseable", beat="b3_probe", attempts=2),
            "judge_unparseable",
        ),
        (
            SessionInvalidError(
                "judge_evidence_invalid", beat="b3_probe", attempts=2
            ),
            "judge_evidence_invalid",
        ),
        # A spent question budget is the run's own setting, not a judge that
        # failed, so it stays an ordinary invalid session.
        (
            SessionInvalidError(
                "judge_budget_exhausted", beat="b3_probe", attempts=1
            ),
            "session_invalid",
        ),
        (
            SimulatorOutputContractError(["simulator_output_format_invalid"]),
            "output_contract",
        ),
        (MemoryIOAuditError("audit sink refused the write"), "memory_backend"),
        # A forced event that could not be replayed is not one of the five
        # named kinds, and a run that hits one still needs a human.
        (ForcedEventExecutionError("evolv_03 could not be replayed"), "other"),
        (ValueError("something nobody has classified"), "other"),
    ],
)
def test_one_exception_per_kind(error: BaseException, expected: str) -> None:
    assert classify_failure(error) == expected


def test_a_dead_judge_is_not_a_session_to_rerun() -> None:
    """The shape `SimulatorLoop.run_session` raises when the judge dies.

    It writes the session's `session_invalid` record and then re-raises the
    `JudgeUnavailable` from inside that `except`, so the exception carries a
    `SessionInvalidError` as its `__context__`. Classified off that context the
    ledger would say "discard the session and rerun it", which is the one thing
    not to do: rerunning it without Ollama back does the same thing again.
    """
    dead = JudgeUnavailable("ollama is not reachable at http://127.0.0.1:11434")
    try:
        try:
            raise SessionInvalidError(
                "judge_unavailable", beat="b3_probe", attempts=1
            ) from dead
        except SessionInvalidError as invalid:
            raise invalid.__cause__ from None
    except JudgeUnavailable as raised:
        assert classify_failure(raised) == "judge_unavailable"
        assert "cannot run" in FAILURE_KIND_ACTIONS["judge_unavailable"]


SESSION_LEVEL_KINDS = (
    "session_invalid",
    "output_contract",
    "judge_timeout",
    "judge_unparseable",
    "judge_evidence_invalid",
    "judge_unavailable",
)


@pytest.mark.parametrize("kind", SESSION_LEVEL_KINDS)
def test_every_session_level_action_says_the_round_stopped_and_resumes(
    kind: str,
) -> None:
    """A regression round is an accumulation run, so no session is ever skipped
    past. Whatever killed it, the action a reader
    gets is the same two facts: the round ended here, and the same run id picks
    it up here.
    """
    action = FAILURE_KIND_ACTIONS[kind]
    assert "the round stopped here" in action
    assert "resumes at this session" in action


def test_every_invalidating_rule_has_a_kind() -> None:
    """The rules the simulator can void a session on all reach the vocabulary:
    the three judge ones by name, everything else as `session_invalid`."""
    for rule in SESSION_INVALID_RULES:
        kind = classify_failure(
            SessionInvalidError(rule, beat="b3_probe", attempts=1)
        )
        assert kind in SESSION_LEVEL_KINDS
        assert kind in FAILURE_KINDS


def test_a_wrapped_failure_is_classified_by_what_it_wraps() -> None:
    """The class that matters is often one link down, not the one raised."""
    cause = OpenHoldEpisodeError("acc_002", [("autonomy_level", "p1")])
    try:
        try:
            raise cause
        except OpenHoldEpisodeError as inner:
            raise RuntimeError("step acc_002 did not finish") from inner
    except RuntimeError as wrapper:
        assert classify_failure(wrapper) == "ix_open_hold"

    # `__context__` alone (a bare re-raise, no `from`) is enough as well.
    try:
        try:
            raise AgentProviderError("openrouter timed out")
        except AgentProviderError:
            raise RuntimeError("step acc_002 did not finish")
    except RuntimeError as wrapper:
        assert classify_failure(wrapper) == "provider"


def test_only_an_ix_contract_message_makes_a_provider_error_ix_runtime() -> None:
    """One class, two unrelated failures, two different things to do.

    `pa/agent/loop.py::AgentLoop._run_agent_loop` raises `AgentProviderError`
    when OpenRouter fails: the run is stale and the last checkpoint is where it
    comes back from. `AgentLoop._process_message` raises the same class when
    the PA broke an IX contract mid-turn -- nothing is stale, the PA simply
    did not honour the wrapper, and a rerun from a checkpoint addresses none
    of it. Matching is on the prefix, so the detail half stays free to change.
    """
    assert classify_failure(
        AgentProviderError(
            "PA turn failed the IX render contract: a required "
            "IX_render_<attribute> payload was missing twice, so this turn "
            "had no per-turn render data"
        )
    ) == "ix_runtime"
    assert classify_failure(
        AgentProviderError(
            "PA turn failed the IX capability contract: IX_capability_resolve "
            "was missing or invalid twice after a tool reported a capability "
            "limit"
        )
    ) == "ix_runtime"
    # Everything else under the class is still a provider failure.
    assert classify_failure(AgentProviderError("openrouter 502")) == "provider"


def test_a_proactive_event_left_undecided_is_ix_runtime() -> None:
    """acc_007: the PA answered a proactive environment event with
    three tool-calling turns and never made the IX `proactive_outreach`
    send/suppress decision. The PA failed an IX contract, as it does when it
    leaves out a render payload, so the run resumes and a probe is rerun."""
    with pytest.raises(RuntimeError, match="without a valid send/suppress decision") as raised:
        AgentLoop._proactive_decision(AgentRunResult(final_content=None, messages=[]))
    assert classify_failure(raised.value) == "ix_runtime"


def test_a_plan_load_failure_is_other() -> None:
    """A run plan no longer carries an IX contract check, so an unreadable or
    invalid plan is never a contract failure."""
    assert classify_failure(
        RunPlanLoadError(
            "Invalid run-plan YAML: path=run_plan.yaml "
            "error=ScannerError: mapping values are not allowed here"
        )
    ) == "other"


def test_the_vocabulary_stays_closed() -> None:
    """Nothing may leak a ninth word into the ledger or into Admin's display."""
    assert set(FAILURE_KIND_ACTIONS) == set(FAILURE_KINDS)
    for error in (
        OpenHoldEpisodeError("acc_002", []),
        IXContractVersionError("boom"),
        AgentProviderError("boom"),
        AgentProviderError("PA turn failed the IX render contract: boom"),
        SessionInvalidError("beat_ask_unfulfilled", beat="b1", attempts=3),
        JudgeUnavailable("boom"),
        MemoryIOAuditError("boom"),
        RunPlanLoadError("boom"),
        KeyboardInterrupt(),
        ValueError("boom"),
    ):
        assert classify_failure(error) in FAILURE_KINDS


def test_probe_retry_kinds_hold_the_owners_2026_09_19_line():
    """Which failures buy a probe another attempt.

    The cap is three and `ix_runtime` is in: the PA breaking an IX contract
    mid-turn is something another attempt can pass.
    `ix_contract` stays out — `FAILURE_KIND_ACTIONS` says "archive; do not
    retry under the current code", and a version or checkpoint mismatch fails
    identically every time. `ix_open_hold` stays out too: a step that ended
    with a hold episode still open is a material question, not a flaky turn.

    `other` is retried too, three attempts before it is terminal.
    """
    assert "ix_runtime" in PROBE_RETRY_KINDS
    assert "other" in PROBE_RETRY_KINDS
    assert JUDGE_VOID_KINDS <= PROBE_RETRY_KINDS
    assert PROBE_RETRY_KINDS == JUDGE_VOID_KINDS | {"ix_runtime", "other"}
    for never_retried in ("ix_contract", "ix_open_hold", "judge_unavailable"):
        assert never_retried not in PROBE_RETRY_KINDS
    assert PROBE_RETRY_KINDS <= set(FAILURE_KINDS)


def test_an_ix_runtime_failure_is_classified_into_the_retried_set():
    """The exception three live holes carried lands in the set."""
    kind = classify_failure(
        AgentProviderError(
            "PA turn failed the IX render contract: a required "
            "IX_render_<attribute> payload was missing twice, so this turn "
            "had no per-turn render data"
        )
    )
    assert kind == "ix_runtime"
    assert kind in PROBE_RETRY_KINDS
