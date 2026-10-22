"""One typed vocabulary for why a run step died.

The ledger has always kept the exception's ``str()`` and nothing else, so a run
that died because the PA ended a step still holding a confirmation and a run
that died because OpenRouter timed out are the same shape to Admin
(``non_resumable(acc_002)`` for both) while their handling has nothing in
common. This module is the single owner of the mapping from an exception to
one of the words in :data:`FAILURE_KINDS`, and of the one line each implies.

No exception class is imported here. The classes live in ``harness``,
``pa.agent``, ``memory.backends`` and ``simulator``; importing them would cycle
back into ``harness`` (``pa_factory`` and ``checkpoint`` both import from it)
and would drag every optional memory-backend dependency into a module whose
only job is to name a failure. Matching is by class name over the raised
class's MRO instead, which also means a backend that renames its module keeps
working and two identically named classes (``MemoryConsolidationError`` exists
in both ``pa.agent.memory`` and ``simulator.memory.store``) need one entry.
"""

from __future__ import annotations


#: The closed vocabulary. `classify_failure` never returns anything else.
FAILURE_KINDS: tuple[str, ...] = (
    "ix_open_hold",
    "ix_contract",
    "ix_runtime",
    "session_invalid",
    "output_contract",
    "judge_timeout",
    "judge_unparseable",
    "judge_evidence_invalid",
    "judge_unavailable",
    "provider",
    "memory_backend",
    "other",
)

#: What to do with a run that died this way, one line per kind.
FAILURE_KIND_ACTIONS: dict[str, str] = {
    "ix_open_hold": "human look: the step ended with a hold episode still open",
    "ix_contract": "archive; do not retry under the current code",
    "ix_runtime": (
        "a probe is rerun for it (`PROBE_RETRY_KINDS`, three attempts); a "
        "human look once it has failed all three, or on any other step kind: "
        "the PA broke the IX render or capability contract mid-turn"
    ),
    # The six session-level kinds below all stopped the round where they stood
    # (owner ruling 2026-09-09). A regression round is an ACCUMULATION run —
    # each session's memory is the next one's input — so a session that
    # measured nothing cannot be skipped past: a hole breaks the chain. The
    # round therefore ends at the failed session, and rerunning the SAME run id
    # (without `--overwrite`) rolls back to the checkpoint taken before it,
    # runs that session again, and continues.
    "session_invalid": (
        "the round stopped here: the simulator failed as an instrument, so "
        "this session measured nothing — rerun the same run id and it resumes "
        "at this session"
    ),
    "output_contract": (
        "the round stopped here: the simulator's output never satisfied the "
        "three-block contract — rerun the same run id and it resumes at this "
        "session"
    ),
    "judge_timeout": (
        "the round stopped here: the local judge ran out of clock on a call — "
        "rerun the same run id and it resumes at this session"
    ),
    "judge_unparseable": (
        "the round stopped here: the local judge's answer did not parse — "
        "rerun the same run id and it resumes at this session"
    ),
    "judge_evidence_invalid": (
        "the round stopped here: the local judge could not quote what it read "
        "— rerun the same run id and it resumes at this session"
    ),
    "judge_unavailable": (
        "the round stopped here: the local judge model cannot run — bring the "
        "judge endpoint back up on the pinned model, then rerun the same run id "
        "and it resumes at this session"
    ),
    "provider": (
        "restore from the last checkpoint; a provider failure usually passes "
        "on the rerun"
    ),
    "memory_backend": "restore from the last checkpoint",
    "other": "human look",
}

#: The kinds whose action above is to rerun the same run id (restoring from the
#: last checkpoint is what a rerun does): `mempabench run` exits 75 on them and
#: `scripts/run_with_resume.sh` reruns the command. `ix_runtime` is in because
#: one turn breaking an IX contract usually passes on a rerun. Excluded:
#: `ix_open_hold` -- "human look": the step ended with a hold episode still open;
#: `ix_contract` -- "archive; do not retry under the current code";
#: `other` -- "human look".
RESUMABLE_FAILURE_KINDS: frozenset[str] = frozenset(
    {
        "ix_runtime",
        "session_invalid",
        "output_contract",
        "judge_timeout",
        "judge_unparseable",
        "judge_evidence_invalid",
        "judge_unavailable",
        "provider",
        "memory_backend",
    }
)

#: The attribute the orchestrator sets, where it marks a step failed, to that
#: step's id on the exception it then raises; `mempabench run` prints it.
FAILED_STEP_ID_ATTR = "failed_step_id"

_KIND_BY_EXCEPTION_NAME: dict[str, str] = {
    # Contract 7: the step ended with the PA waiting on a confirmation nobody
    # will give (`harness/pa_factory.py:448`).
    "OpenHoldEpisodeError": "ix_open_hold",
    # Contract 12: an `ix_render` record written under another wrapper major
    # or contract version (`pa/agent/ix/render_types.py`); nothing to retry.
    "IXContractVersionError": "ix_contract",
    # Every OpenRouter transport, HTTP and timeout failure reaches a step as
    # this one class: both `OpenAICompatProvider`s turn any exception from the
    # client into `LLMResponse(finish_reason="error")`
    # (`OpenAICompatProvider._handle_error` in `pa/providers/` and in
    # `simulator/providers/openai_compat_provider.py`), and the agent loop
    # turns that into `AgentProviderError`
    # (`pa/agent/loop.py::AgentLoop._run_agent_loop`). No `openai`
    # or `httpx` exception escapes the provider, so none is listed here. The
    # same class also carries the IX contract failures, which are not
    # provider failures at all; `_direct_kind` splits them off by message.
    "AgentProviderError": "provider",
    # The PA answered a proactive environment event without the IX
    # `proactive_outreach` send/suppress decision
    # (`pa/agent/loop.py::AgentLoop._proactive_decision`): the PA failed an IX
    # contract, as with a missing render payload below.
    "ProactiveDecisionMissingError": "ix_runtime",
    # Audit item 6 (owner ruling 2026-09-08): the simulator failed as an
    # instrument — a beat's scripted ask never reached the PA, or a three-block
    # output never parsed — so the session measured nothing and the step is
    # invalid rather than merely dead (`simulator/output.py`). The distinction
    # from `ix_*` matters: the PA's own contract failures stay
    # continue-and-record, because the PA is the subject.
    "SessionInvalidError": "session_invalid",
    # The three-block output contract, exhausted on one candidate turn and not
    # turned into a `SessionInvalidError` above it. Session-level all the same:
    # the round stops here and resumes here.
    "SimulatorOutputContractError": "output_contract",
    # The local judge stopped answering mid-round: the endpoint went away, or
    # the pinned model was unloaded, after the run's preflight had passed. Its own
    # kind and not `session_invalid`, because the two want opposite things done
    # — one session is discarded and rerun, a dead judge is not rerun at all
    # until the model is back. `simulator.loop.SimulatorLoop.run_session` writes
    # the session's `session_invalid` record and then re-raises THIS class for
    # exactly that reason, so the class the ledger sees is this one.
    "JudgeUnavailable": "judge_unavailable",
    # An external memory service failed. The run's own state is intact, so the
    # last cut point is where it comes back from.
    "Mem0PersistenceError": "memory_backend",
    "Mem0ResponseError": "memory_backend",
    "MemOSClientError": "memory_backend",
    "MemOSResponseError": "memory_backend",
    "MemOSTransportError": "memory_backend",
    "AdapterRetrievalError": "memory_backend",
    "AdapterWriteCommitError": "memory_backend",
    "MemoryConsolidationError": "memory_backend",
    "MemoryConsolidationContractError": "memory_backend",
    "MemoryIOAuditError": "memory_backend",
}

#: `AgentProviderError` is raised for two unrelated things. The provider one is
#: raised in `pa/agent/loop.py::AgentLoop._run_agent_loop`. These are raised in
#: `AgentLoop._process_message`, where the
#: PA broke the wrapper's own contract mid-turn: nothing is stale, so restoring
#: from a checkpoint addresses nothing, and the failed turn wants a reader.
#: The raise site builds each message as the literal "PA turn failed the IX "
#: plus the matching entry of `pa/agent/loop.py::_IX_CONTRACT_FAILURE_DETAILS`,
#: which is the contract word ("render contract: ") followed by its
#: `IX_*_CONTRACT_FAILURE_DETAIL` constant.
#: Importing them anyway is not an option -- `pa.agent.loop` reaches `harness`
#: through `pa_factory`, which is the cycle this module's docstring exists to
#: avoid -- so the prefixes are restated, matched as a prefix so the detail half
#: stays free to change.
#: All five, because a prefix missing here does not fall to `other` -- it keeps
#: the class's own kind, `provider`, whose action is "restore from the last
#: checkpoint; a provider failure usually passes on the rerun". That is the
#: opposite of what a PA contract failure needs, and the notice contract has
#: been reaching runs that way since it was added. The loop names one kind per
#: fatal rule (`pa/agent/runner.py`'s `IX_CONTRACT_*`), so this list is
#: complete when it has one entry per contract word those kinds map to in
#: `_IX_CONTRACT_FAILURE_DETAILS` (`selection_args` and `render_args` reuse
#: `selection` and `render`).
IX_RENDER_CONTRACT_FAILURE_PREFIX = "PA turn failed the IX render contract"
IX_CAPABILITY_CONTRACT_FAILURE_PREFIX = (
    "PA turn failed the IX capability contract"
)
IX_NOTICE_CONTRACT_FAILURE_PREFIX = "PA turn failed the IX notice contract"
IX_SELECTION_CONTRACT_FAILURE_PREFIX = (
    "PA turn failed the IX selection contract"
)
IX_GATE_CONTRACT_FAILURE_PREFIX = "PA turn failed the IX gate contract"
_IX_RUNTIME_PREFIXES = (
    IX_RENDER_CONTRACT_FAILURE_PREFIX,
    IX_CAPABILITY_CONTRACT_FAILURE_PREFIX,
    IX_NOTICE_CONTRACT_FAILURE_PREFIX,
    IX_SELECTION_CONTRACT_FAILURE_PREFIX,
    IX_GATE_CONTRACT_FAILURE_PREFIX,
)


#: The `SessionInvalidError.rule` values that name a judge failure rather than
#: a simulator one. `judge_budget_exhausted` is deliberately absent: a spent
#: budget is the run's own setting, not a judge that failed, so it reads as an
#: ordinary invalid session.
_JUDGE_RULE_KINDS: dict[str, str] = {
    "judge_timeout": "judge_timeout",
    "judge_unparseable": "judge_unparseable",
    "judge_evidence_invalid": "judge_evidence_invalid",
}

#: The kinds that say the local judge voided the session it was watching. Read
#: by the probe batch (owner decision D5: a voided probe is rerun once, at
#: once), which needs the set rather than one kind, and derived from the map
#: above so a judge rule added there is a void without a second edit here.
#: `judge_unavailable` is deliberately not one of them: the judge did not void
#: a session, it is gone, and D1 makes it a batch stopper.
JUDGE_VOID_KINDS: frozenset[str] = frozenset(_JUDGE_RULE_KINDS.values())

#: The kinds a probe is rerun for. Owner 2026-09-19, superseding the D5 half of
#: "twice = ask": a probe that died because the PA broke an IX contract
#: mid-turn gets the same second (and now third) chance a judge void gets, so
#: the PA is given three attempts before the cell is written off as a hole.
#:
#: `ix_runtime` is in; `ix_contract` deliberately is not. The first is the PA
#: failing the contract on this turn, which another attempt can pass. The
#: second is a version or checkpoint mismatch — `FAILURE_KIND_ACTIONS` says
#: "archive; do not retry under the current code" — and rerunning it would
#: fail identically. `ix_open_hold` is also out: the step ended with a hold
#: episode still open, which is a material question, not a flaky turn.
#:
#: Owner 2026-09-23 (HY repeat_02/fin_23, 「再试两次吧 都到三次这样 other 其他的
#: 也是」): `other` joins the retried kinds, so an unclassified failure also gets
#: three attempts and is terminal only when all three are spent. This revises
#: the 2026-09-19 line above for `other` only; the exclusions stand.
PROBE_RETRY_KINDS: frozenset[str] = JUDGE_VOID_KINDS | {"ix_runtime", "other"}


def classify_failure(exc: BaseException) -> str:
    """One of :data:`FAILURE_KINDS` for the exception that ended a step."""
    kind = _direct_kind(exc)
    if kind is not None:
        return kind
    # A generic wrapper (a bare `RuntimeError`, a re-raise around the real
    # failure) carries the class that matters one link down. One level is
    # enough to reach it and shallow enough not to blame an unrelated
    # exception that merely happened to be in flight.
    for nested in (exc.__cause__, exc.__context__):
        if nested is None:
            continue
        kind = _direct_kind(nested)
        if kind is not None:
            return kind
    return "other"


def _direct_kind(exc: BaseException) -> str | None:
    """The kind this exception's own class, or a base of it, names."""
    for cls in type(exc).__mro__:
        kind = _KIND_BY_EXCEPTION_NAME.get(cls.__name__)
        if kind is None:
            continue
        if cls.__name__ == "AgentProviderError" and str(exc).startswith(
            _IX_RUNTIME_PREFIXES
        ):
            return "ix_runtime"
        if cls.__name__ == "SessionInvalidError":
            # Every one of `SESSION_INVALID_RULES` stops the round the same
            # way, but three of them want a different first move from the
            # reader than "look at the simulator": the local judge failed, and
            # the rule the error names is already the judge's own event name.
            return _JUDGE_RULE_KINDS.get(getattr(exc, "rule", ""), kind)
        return kind
    return None
