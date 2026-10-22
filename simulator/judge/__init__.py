"""Local semantic judge for the simulator's message-quality rules.

One small local model answers the checks that need the meaning of a line read,
pointing at a numbered sentence of the text for every verdict that claims
something is in it. Mechanical checks — did a tool run, is a quote in the
transcript, did the dictated key terms appear, how many contractions are there
— stay in code and never reach the judge.

Phase 2 wires it in. `JudgeSession` is the object a simulator session holds:
it carries the mode (`MEMPA_JUDGE`), the per-generation budget, the door that
turns a judge that cannot be read into `SessionInvalidError`, and the release
door for the one failure that is the judge's own — a reference that will not
resolve twice, which costs its group's verdicts and nothing more.
"""

from simulator.judge.session import (
    JUDGE_MODES,
    JudgeSession,
    current_mode,
    preflight,
)
from simulator.judge.guards import (
    JudgeBudgetExhausted,
    JudgeEvidenceInvalid,
    JudgeFailure,
    JudgeTimeout,
    JudgeUnparseable,
    TurnBudget,
    Verdict,
    ask,
    parse_verdicts,
    strip_fences,
)

__all__ = [
    "JUDGE_MODES",
    "JudgeBudgetExhausted",
    "JudgeEvidenceInvalid",
    "JudgeFailure",
    "JudgeSession",
    "JudgeTimeout",
    "JudgeUnparseable",
    "TurnBudget",
    "Verdict",
    "ask",
    "current_mode",
    "parse_verdicts",
    "preflight",
    "strip_fences",
]
