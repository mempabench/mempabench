"""Parser for simulator's per-turn structured output.

Decoupled module — knows nothing about prompt templates, LLM providers, or schemas.
Pure string-in / dataclass-out.

Under the "Factual Reporter / dual-assessment architecture" (2026-08-31 schema):
  <message>            — PA-visible dialog (always present)
  <factual_check>      — eval-only, conditional (empty when no PA fact errors)
  <turn_assessment>    — eval-only, every turn — next_beat + goal_already_met
  <consent_decisions>  — eval-only, conditional (2026-09-05):
                         the PA-blind consent sideband. One entry per held item
                         the PA's last message asked this user to approve.

Access boundary (PA blindness for the latter three) is enforced in harness/,
not here. This module just parses; downstream decides who sees what.

The per-turn <emotion_event> block was removed 2026-08-31 — the session-end
self-report is the only emotion/experience source. Earlier (2026-05-01) the
per-turn eval fields preference_met / attitude_delta / trust_delta moved to
session end for the same reason. See MemPABench_Evaluation_Metrics.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import yaml

_MESSAGE_RE = re.compile(r"<message>\s*(.*?)\s*</message>", re.DOTALL)
_FACTUAL_CHECK_RE = re.compile(r"<factual_check>\s*(.*?)\s*</factual_check>", re.DOTALL)
_TURN_ASSESSMENT_RE = re.compile(
    r"<turn_assessment>\s*(.*?)\s*</turn_assessment>", re.DOTALL
)
_CONSENT_DECISIONS_RE = re.compile(
    r"<consent_decisions>\s*(.*?)\s*</consent_decisions>", re.DOTALL
)
# Legacy leak guard: <emotion_event> is no longer part of the output schema,
# but a model may still emit the old tag. The regex only exists so the
# missing-<message> fallback can strip it before PA visibility.
_EMOTION_EVENT_RE = re.compile(
    r"<emotion_event>\s*(.*?)\s*</emotion_event>", re.DOTALL
)

# next_beat: "stay", "advance", or "branch:<id>"
_NEXT_BEAT_SIMPLE = frozenset({"stay", "advance"})
_NEXT_BEAT_BRANCH_RE = re.compile(r"^branch:[A-Za-z0-9_\-]+$")
_KNOWN_TAGS = {
    "message",
    "messag",
    "factual_check",
    "turn_assessment",
    "consent_decisions",
    "emotion_event",  # legacy tag; normalized so the leak-guard strip matches
}

# The three values the simulator may write. "missing" is a wrapper-side audit
# state ("this held id never appeared in the sideband"), never something the
# simulated user can claim, so it is rejected on the input path.
_CONSENT_DECISION_VALUES = frozenset({"allow", "deny", "unclear"})


def _normalize_model_artifacts(raw: str) -> str:
    """Normalize BPE-looking artifacts seen from some model outputs."""
    text = raw.replace("Ġ", " ").replace("Ċ", "\n")

    def normalize_tag(match: re.Match[str]) -> str:
        closing = "/" if match.group(1) else ""
        name = re.sub(r"\s+", "", match.group(2).lower())
        if name == "messag":
            name = "message"
        if name in _KNOWN_TAGS:
            return f"<{closing}{name}>"
        return match.group(0)

    text = re.sub(r"<\s*(/)?\s*([^<>]+?)\s*>", normalize_tag, text)
    text = re.sub(r"\bnext\s*_\s*be\s*at\b", "next_beat", text)
    text = re.sub(r"\bn\s*/\s*a\b", "n/a", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text


@dataclass
class FactualViolation:
    """One PA factual error caught by simulator (vs session task_facts GT)."""
    fact: str
    expected: Any = None
    pa_said: Any = None


@dataclass
class TurnAssessment:
    """Simulator's per-turn beat-routing decision (PA-blind, eval-only).

    Schema: `next_beat` plus `goal_already_met`, the actor's judgement on the
    first turn of a beat that the PA has already accomplished this beat's goal.
    Older preference_met / attitude_delta / trust_delta were removed when
    per-turn eval migrated to session-end self-report.
    """
    next_beat: str = "stay"  # "stay" | "advance" | "branch:<id>"
    goal_already_met: bool = False


@dataclass
class ConsentDecision:
    """One entry of the PA-blind consent sideband.

    The simulated user knows whether this turn agreed or refused; the wrapper
    reads that as gate state, never as text for the model. `decision` is one of
    allow / deny / unclear — `missing` is a wrapper-side audit value and is
    never produced here.
    """
    interaction_item_id: str
    decision: str


@dataclass
class SimulatorOutput:
    """Parsed per-turn output. `message` is PA-visible; the rest are eval-only."""
    message: str
    factual_check: list[FactualViolation] = field(default_factory=list)
    turn_assessment: TurnAssessment = field(default_factory=TurnAssessment)
    consent_decisions: list[ConsentDecision] = field(default_factory=list)
    raw_response: str = ""
    parse_warnings: list[str] = field(default_factory=list)
    rejected_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Non-blocking findings on the message that was PLAYED, as a list of
    # finding names — the same shape a rejected attempt carries under its own
    # `diagnostics` key. A rejected attempt records these on its way out; an
    # accepted one had nowhere to put them, so a diagnostic on the turn the PA
    # actually answered left no trace at all. Never a retry trigger: the agent
    # loop decides that, and this field only carries the verdict out to the
    # session loop.
    diagnostics: list[str] = field(default_factory=list)
    # What a diagnostic fired on, keyed by its name — the part of the verdict
    # a bare name loses. `probe_fact_missing` is the case it was added for:
    # regress21 `acc_027` recorded it twice with no key, and both reads had to
    # re-derive the keys against the draft by hand.
    diagnostics_detail: dict[str, list[str]] = field(default_factory=dict)


class SimulatorOutputContractError(RuntimeError):
    """Raised after simulator output exhausts contract correction attempts."""

    def __init__(
        self,
        findings: list[str],
        *,
        rejected_attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        self.findings = list(findings)
        self.rejected_attempts = list(rejected_attempts or [])
        last = self.rejected_attempts[-1] if self.rejected_attempts else {}
        detail = ""
        if last:
            raw = str(last.get("raw_response") or "")[:600]
            detail = (
                f" | parse_warnings={last.get('parse_warnings')!r}"
                f" | last_raw_head={raw!r}"
            )
        super().__init__(
            "INVALID_SIMULATOR_OUTPUT: " + ", ".join(self.findings) + detail
        )


#: The simulator-side rules that end a session as invalid (owner ruling
#: 2026-09-08). The first two were play-anyway hatches:
#: the beat whose scripted ask never reached the PA, and the three-block output
#: that never parsed and was played on a defaulted `next_beat`. The third is the
#: same failure one step further on: the dictation the session gate is keyed on,
#: delivered without the information the store checks for, so the PA is handed a
#: sentence the world will refuse whatever it does with it. The simulator is the
#: instrument, so any of them means the session measured nothing.
SESSION_INVALID_RULES: tuple[str, ...] = (
    "beat_ask_unfulfilled",
    "simulator_output_format_invalid",
    "dictation_missing_key_terms",
    # The four judge failures. A judge that
    # cannot be parsed, cannot quote what it read, ran out of clock or ran out
    # of budget has not answered, and a session recorded on a missing answer is
    # a session measured against nothing. No fallback, no pass-through.
    "judge_unparseable",
    "judge_evidence_invalid",
    "judge_timeout",
    "judge_budget_exhausted",
    # The judge endpoint gone after the preflight passed. Void like the four
    # above, but the
    # loop re-raises the underlying `JudgeUnavailable` so the ROUND stops too:
    # a dead judge is not one bad session, it is every session after it.
    "judge_unavailable",
)


class SessionInvalidError(RuntimeError):
    """The simulator failed as an instrument; this session measured nothing.

    Distinct from :class:`SimulatorOutputContractError`, which the backend
    raises about one candidate turn: by the time this is raised the retry
    budget is spent, nothing is delivered to the PA, and the session ends on
    `session_invalid` rather than `session_end`. The PA's own contract failures
    are the other way round — the PA is the subject, so those stay
    continue-and-record (ruling 4).
    """

    def __init__(self, rule: str, *, beat: str, attempts: int) -> None:
        self.rule = rule
        self.beat = beat
        self.attempts = attempts
        super().__init__(
            f"SESSION_INVALID: rule={rule} beat={beat} attempts={attempts}"
        )


def _parse_factual_check(inner: str) -> tuple[list[FactualViolation], list[str]]:
    """Parse the inner content of <factual_check> as YAML list of dicts."""
    inner = inner.strip()
    if not inner:
        return [], []
    if inner.lower() in {"ok", "n/a", "na"}:
        return [], []
    try:
        data = yaml.safe_load(inner)
    except yaml.YAMLError as e:
        return [], [f"factual_check YAML parse error: {e}"]
    if data is None:
        return [], []
    if not isinstance(data, list):
        return [], [f"factual_check expected list, got {type(data).__name__}"]

    violations: list[FactualViolation] = []
    warnings: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            warnings.append(f"factual_check[{i}] not a dict, skipping")
            continue
        fact = item.get("fact")
        if not fact:
            warnings.append(f"factual_check[{i}] missing 'fact' field, skipping")
            continue
        violations.append(
            FactualViolation(
                fact=str(fact),
                expected=item.get("expected"),
                pa_said=item.get("pa_said"),
            )
        )
    return violations, warnings


def _validate_next_beat(value: str) -> tuple[str, list[str]]:
    """Validate next_beat. Returns (normalized_value, warnings)."""
    v = value.strip().lower()
    if v in _NEXT_BEAT_SIMPLE:
        return v, []
    if _NEXT_BEAT_BRANCH_RE.match(v):
        return v, []
    return "stay", [
        f"next_beat={value!r} not in {{stay, advance, branch:<id>}}; using 'stay'"
    ]


def _parse_turn_assessment(inner: str) -> tuple[TurnAssessment, list[str]]:
    """Parse <turn_assessment> as YAML mapping: `next_beat`, `goal_already_met`.

    Unknown keys are ignored for tooling but are not part of the canonical schema.
    """
    inner = inner.strip()
    if not inner:
        return TurnAssessment(), []
    try:
        data = yaml.safe_load(inner)
    except yaml.YAMLError as e:
        return TurnAssessment(), [f"turn_assessment YAML parse error: {e}"]
    if not isinstance(data, dict):
        return (
            TurnAssessment(),
            [f"turn_assessment expected dict, got {type(data).__name__}"],
        )

    raw_next = data.get("next_beat", "stay")
    next_beat, warnings = _validate_next_beat(str(raw_next))
    raw_met = data.get("goal_already_met", False)
    goal_already_met = (
        raw_met
        if isinstance(raw_met, bool)
        else str(raw_met).strip().lower() in {"true", "yes"}
    )
    return (
        TurnAssessment(next_beat=next_beat, goal_already_met=goal_already_met),
        warnings,
    )


def _parse_consent_decisions(inner: str) -> tuple[list[ConsentDecision], list[str]]:
    """Parse <consent_decisions> as a YAML list of {interaction_item_id, decision}.

    An empty block means "no held item was put to me this turn" and is the
    normal case. Malformed entries are dropped with a warning: a decision the
    wrapper cannot read is treated as absent, and absence is fail-closed there.
    """
    inner = inner.strip()
    if not inner:
        return [], []
    if inner.lower() in {"none", "n/a", "na"}:
        return [], []
    try:
        data = yaml.safe_load(inner)
    except yaml.YAMLError as e:
        return [], [f"consent_decisions YAML parse error: {e}"]
    if data is None:
        return [], []
    if not isinstance(data, list):
        return [], [f"consent_decisions expected list, got {type(data).__name__}"]

    decisions: list[ConsentDecision] = []
    warnings: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            warnings.append(f"consent_decisions[{i}] not a dict, skipping")
            continue
        item_id = item.get("interaction_item_id")
        decision = str(item.get("decision") or "").strip().lower()
        if not item_id:
            warnings.append(
                f"consent_decisions[{i}] missing 'interaction_item_id', skipping"
            )
            continue
        if decision not in _CONSENT_DECISION_VALUES:
            warnings.append(
                f"consent_decisions[{i}] decision={item.get('decision')!r} not in "
                f"{sorted(_CONSENT_DECISION_VALUES)}, skipping"
            )
            continue
        decisions.append(
            ConsentDecision(interaction_item_id=str(item_id), decision=decision)
        )
    return decisions, warnings


def filter_consent_decisions(
    decisions: list[ConsentDecision],
    held_item_ids: Iterable[str],
) -> tuple[list[ConsentDecision], list[str]]:
    """Keep only the entries for ids the harness said were actually held.

    The held ids reach the simulated user as stage direction, so anything else
    in the block is invented — a model-authored id can never become consent for
    a real action. With nothing held, every entry is dropped: no ask, no
    consent. Dropping is warned about, not raised on; the wrapper reads a
    dropped id as absent, and absence is fail-closed there.
    """
    known = set(held_item_ids)
    kept: list[ConsentDecision] = []
    warnings: list[str] = []
    for decision in decisions:
        if decision.interaction_item_id not in known:
            warnings.append(
                "consent_decisions dropped unknown "
                f"interaction_item_id={decision.interaction_item_id!r} "
                f"(held ids: {sorted(known)})"
            )
            continue
        kept.append(decision)
    return kept, warnings


def parse_simulator_output(raw: str) -> SimulatorOutput:
    """Parse LLM raw response into the four structured blocks.

    Robust to missing/malformed blocks — populates `parse_warnings` rather than raising.
    """
    parse_raw = _normalize_model_artifacts(raw)
    warnings: list[str] = []
    if parse_raw != raw:
        warnings.append("normalized model tokenization artifacts before parsing")

    msg_match = _MESSAGE_RE.search(parse_raw)
    if msg_match:
        message = msg_match.group(1).strip()
    else:
        # LLM forgot the <message>...</message> wrapper — but we still must
        # NOT leak the eval-only blocks to PA. Strip those out of the raw
        # text first, then take what's left as the message. The regexes for
        # the eval blocks still work (they don't depend on <message>).
        candidate = _FACTUAL_CHECK_RE.sub("", parse_raw)
        candidate = _TURN_ASSESSMENT_RE.sub("", candidate)
        candidate = _CONSENT_DECISIONS_RE.sub("", candidate)
        candidate = _EMOTION_EVENT_RE.sub("", candidate)
        message = candidate.strip()
        warnings.append(
            "no <message> block found; stripped eval-only blocks and used remainder "
            "(access-boundary safety fallback)"
        )

    factual_check: list[FactualViolation] = []
    fc_match = _FACTUAL_CHECK_RE.search(parse_raw)
    if fc_match:
        factual_check, fc_warnings = _parse_factual_check(fc_match.group(1))
        warnings.extend(fc_warnings)

    turn_assessment = TurnAssessment()
    ta_match = _TURN_ASSESSMENT_RE.search(parse_raw)
    if ta_match:
        turn_assessment, ta_warnings = _parse_turn_assessment(ta_match.group(1))
        warnings.extend(ta_warnings)
    else:
        warnings.append("no <turn_assessment> block found; using defaults")

    # Conditional block, like <factual_check>: absent means "nothing was put to
    # me for approval this turn", which is the normal case and not a warning.
    consent_decisions: list[ConsentDecision] = []
    cd_match = _CONSENT_DECISIONS_RE.search(parse_raw)
    if cd_match:
        consent_decisions, cd_warnings = _parse_consent_decisions(cd_match.group(1))
        warnings.extend(cd_warnings)

    return SimulatorOutput(
        message=message,
        factual_check=factual_check,
        turn_assessment=turn_assessment,
        consent_decisions=consent_decisions,
        raw_response=raw,
        parse_warnings=warnings,
    )
