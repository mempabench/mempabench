"""Session-end self-report: one call, answered by the simulator itself.

HCI framing (2026-08-31): the questionnaire is filled in by the participant.
The simulator's own dialogue model — with full character card and life
context — writes the reflection, the closing emotion, and the five
self-report answers in a single response. There is no external calibrator
and no persona-specific anchor file; every persona answers the same
questionnaire, and individual color comes from the answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from simulator.prompting import render_life_context
from simulator.providers.base import LLMResponse
from simulator.schemas import SessionScript


_REFLECTION_RE = re.compile(
    r"<session_end_reflection>\s*(.*?)\s*</session_end_reflection>", re.DOTALL
)
_FINAL_EMOTION_RE = re.compile(
    r"<final_emotion>\s*(.*?)\s*</final_emotion>", re.DOTALL
)
_SELF_REPORT_RE = re.compile(r"<self_report>\s*(.*?)\s*</self_report>", re.DOTALL)
_EVIDENCE_RE = re.compile(
    r"<reflection_evidence>\s*(.*?)\s*</reflection_evidence>", re.DOTALL
)

#: Completion budget for the one session-end call. The provider default
#: (4096) was hit on 7 of 13 regress21 sessions: google/gemini-3.8-flash
#: spends its thinking tokens inside `max_tokens`, so the reflection came
#: back cut mid-block (completion_tokens=4092) and cost a second call.
#: 8192 was then hit in turn on 6 of ~300 user_b sessions
#: (completion_tokens=8188, finish_reason=length): the reasoning tokens are
#: charged inside the budget, and the cut landed on the last block, so
#: `<reflection_evidence>` came back truncated.
SELF_REPORT_MAX_TOKENS = 16384
_CURLY_QUOTES = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"'}
)


@dataclass
class EmotionEvent:
    """Closing emotional note from the session-end self-report (PA-blind)."""

    trigger: str = ""
    reaction: str = ""


class SelfReportAnswers(BaseModel):
    """The five self-report fields: three 1-5 integers, one enum, one text."""

    model_config = ConfigDict(extra="forbid")

    felt_understood_in_this_context: int = Field(ge=1, le=5)
    shift_perception: Literal[
        "pa_finally_got_it",
        "pa_still_does_not_get_it",
        "pa_got_it_last_session",
        "no_shift_here",
    ]
    felt_effort: int = Field(ge=1, le=5)
    noticed_contradictions: str
    overall_experience: int = Field(ge=1, le=5)

    @model_validator(mode="after")
    def _validate_text(self) -> "SelfReportAnswers":
        self.noticed_contradictions = self.noticed_contradictions.strip()
        if not self.noticed_contradictions:
            raise ValueError("noticed_contradictions must not be empty")
        return self


@dataclass
class SessionEndReflectionResult:
    reflection: str
    final_emotion: EmotionEvent
    self_report: SelfReportAnswers
    rejected_attempts: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


class SelfReportContractError(RuntimeError):
    def __init__(
        self,
        findings: list[str],
        *,
        rejected_attempts: list[dict[str, Any]],
    ) -> None:
        self.findings = list(findings)
        self.rejected_attempts = list(rejected_attempts)
        super().__init__("INVALID_SELF_REPORT: " + ", ".join(findings))


def _parse_final_emotion(raw: str) -> tuple[EmotionEvent | None, list[str]]:
    match = _FINAL_EMOTION_RE.search(raw)
    if not match:
        return None, ["final_emotion_missing"]
    findings: list[str] = []
    try:
        data = yaml.safe_load(match.group(1).strip())
    except yaml.YAMLError:
        return None, ["final_emotion_malformed"]
    if not isinstance(data, dict):
        return None, ["final_emotion_malformed"]
    trigger = str(data.get("trigger") or "").strip()
    reaction = str(data.get("reaction") or "").strip()
    if not trigger:
        findings.append("final_emotion_trigger_blank")
    if not reaction:
        findings.append("final_emotion_reaction_blank")
    if findings:
        return None, findings
    return EmotionEvent(trigger=trigger, reaction=reaction), []


def _parse_self_report(raw: str) -> tuple[SelfReportAnswers | None, list[str]]:
    match = _SELF_REPORT_RE.search(raw)
    if not match:
        return None, ["self_report_block_missing"]
    try:
        data = yaml.safe_load(match.group(1).strip())
        if not isinstance(data, dict):
            return None, ["self_report_not_mapping"]
        if isinstance(data.get("noticed_contradictions"), bool):
            data["noticed_contradictions"] = (
                "yes" if data["noticed_contradictions"] else "no"
            )
        return SelfReportAnswers.model_validate(data), []
    except (yaml.YAMLError, ValueError):
        return None, ["self_report_schema_invalid"]


def _parse_reflection_output(
    raw: str,
) -> tuple[str, EmotionEvent, SelfReportAnswers | None, list[str]]:
    findings: list[str] = []
    reflection_match = _REFLECTION_RE.search(raw)
    reflection = reflection_match.group(1).strip() if reflection_match else ""
    if not reflection:
        findings.append("session_end_reflection_missing")

    emotion, emotion_findings = _parse_final_emotion(raw)
    findings.extend(emotion_findings)

    answers, answer_findings = _parse_self_report(raw)
    findings.extend(answer_findings)

    return reflection, emotion or EmotionEvent(), answers, findings


def _norm(text: Any) -> str:
    """Case, whitespace, curly quotes and terminal punctuation folded, so a
    quote can be located. A sentence copied with the period it ended on has to
    sit inside a claim that was quoted without one, and the other way round.
    """
    folded = re.sub(r"\s+", " ", str(text or "").translate(_CURLY_QUOTES)).lower()
    return re.sub(r"[\s.,;:!?]+$", "", folded.strip())


def _rendered_turns(history: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    """(index, speaker, content) per turn, content untouched.

    In `history` the simulated person speaks as "assistant" and the PA as
    "user"; the turn index is the position in `history`.
    """
    return [
        (
            index,
            "me" if turn.get("role") == "assistant" else "assistant",
            str(turn.get("content") or ""),
        )
        for index, turn in enumerate(history)
    ]


def _render_transcript(history: list[dict[str, Any]]) -> str:
    """Plain indexed blocks. YAML would quote a scalar holding a colon and
    double its apostrophes, so a verbatim copy of what the model read would
    not be found in the transcript; here the two are the same text.
    """
    return "\n\n".join(
        f"turn {index} — {speaker}:\n{content}"
        for index, speaker, content in _rendered_turns(history)
    )


def _transcript_t(history: list[dict[str, Any]], turn: int | None) -> int | None:
    """The transcript `t` of the history entry `turn` names.

    `turn` indexes `history`, which is what the rendered transcript numbers and
    what every check here resolves against. A reader of `transcript.jsonl`
    numbers the same message by the event `t` the session loop stamped on it,
    and the two do not line up: the regress21 reads of `acc_002`, `acc_004`,
    `acc_009`, `acc_011`, `acc_019` and `acc_023` all took one for the other.
    The row carries both. `None` when no index was given, when it names no
    entry, or when the entry carries no `t` (a history built by hand).
    """
    if turn is None or turn < 0 or turn >= len(history):
        return None
    value = history[turn].get("t")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _four_words(text: str) -> bool:
    """The floor for anything that has to carry content of its own."""
    return len(_norm(text).split()) >= 4


def _sentences(text: str) -> list[str]:
    """Split on terminal punctuation, but never inside a double-quoted span.

    A reflection sentence that quotes the PA carries the PA's own punctuation,
    and a full stop inside the quotation does not end the claim. Live regress21
    `acc_019` attempt 2 was rejected `reflection_evidence_uncovered` on exactly
    that: the claim read `the assistant acknowledged that the draft now stood
    "without hedging. If you want" further adjustments.`, the splitter cut it at
    `hedging.`, and the (complete, correct) evidence row no longer sat inside
    the truncated claim. Quote state is tracked over straight and curly double
    quotes; single quotes are left alone, since an apostrophe is not a
    delimiter. An unbalanced quote runs to the end of the reflection, which
    merges claims rather than splitting one — the safer direction here.
    """
    text = text.strip()
    parts: list[str] = []
    start = 0
    index = 0
    in_quote = False
    while index < len(text):
        char = text[index]
        if char == '"':
            in_quote = not in_quote
        elif char == "“":
            in_quote = True
        elif char == "”":
            in_quote = False
        elif char in ".!?" and not in_quote:
            end = index + 1
            after = end
            while after < len(text) and text[after].isspace():
                after += 1
            if after > end:
                parts.append(text[start:end])
                start = index = after
                continue
        index += 1
    parts.append(text[start:])
    # Fragments under four words are abbreviation splits ("Dr.") or say
    # nothing an evidence entry could sit inside.
    return [part.strip() for part in parts if _four_words(part)]


def _match_entry(
    entries: list[dict[str, Any]], claim: str, start: int
) -> tuple[dict[str, Any] | None, int]:
    """First entry at or after `start` whose sentence sits inside `claim`.

    Only that direction: a claim inside a stub sentence proves nothing, so one
    junk entry cannot cover the whole reflection. Entries are read in order and
    each is consumed once, so evidence written out of order leaves a claim
    uncovered.
    """
    target = _norm(claim)
    for index in range(start, len(entries)):
        sentence = _norm(entries[index]["sentence"])
        if _four_words(sentence) and sentence in target:
            return entries[index], index + 1
    return None, start


def _check_reflection_evidence(
    entries: list[dict[str, Any]],
    *,
    reflection: str,
    emotion: EmotionEvent,
    answers: SelfReportAnswers,
    history: list[dict[str, Any]],
) -> list[str]:
    """Existence checks only: a quote is either in the transcript or it is not."""
    findings: list[str] = []

    def add(code: str) -> None:
        if code not in findings:
            findings.append(code)

    # Quotes are checked against the same rendered text the model read.
    rendered = _rendered_turns(history)
    user_turns = [_norm(content) for _, speaker, content in rendered if speaker == "me"]
    pa_turns = {
        index: _norm(content)
        for index, speaker, content in rendered
        if speaker == "assistant"
    }

    claims = [("sentence", sentence) for sentence in _sentences(reflection)]
    trigger = emotion.trigger.strip()
    if trigger:
        claims.append(("trigger", trigger))
    contradictions = answers.noticed_contradictions.strip()
    # A bare `yes` answers the question without describing anything to quote.
    if contradictions.lower() not in ("", "no", "none") and _four_words(contradictions):
        claims.append(("contradiction", contradictions))

    cursor = 0
    for kind, claim in claims:
        entry, cursor = _match_entry(entries, claim, cursor)
        if entry is None:
            add("reflection_evidence_uncovered")
            continue
        if kind == "trigger" and not entry["pa_quote"]:
            add("reflection_trigger_without_quote")

    for entry in entries:
        user_quote = _norm(entry["user_quote"])
        if user_quote:
            if not _four_words(user_quote):
                add("reflection_quote_too_short")
            elif not any(user_quote in turn for turn in user_turns):
                add("reflection_quote_not_in_transcript")
        pa_quote = _norm(entry["pa_quote"])
        turn = entry["turn"]
        if pa_quote:
            if not _four_words(pa_quote):
                add("reflection_quote_too_short")
            elif not any(pa_quote in text for text in pa_turns.values()):
                add("reflection_quote_not_in_transcript")
            elif turn is not None and pa_quote not in pa_turns.get(turn, ""):
                # `turn` names the assistant turn the quote was copied from.
                add("reflection_quote_turn_mismatch")
        elif turn is not None:
            add("reflection_quote_turn_mismatch")

    return findings


def _parse_reflection_evidence(
    raw: str,
    *,
    reflection: str,
    emotion: EmotionEvent,
    answers: SelfReportAnswers,
    history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    match = _EVIDENCE_RE.search(raw)
    if not match:
        return [], ["reflection_evidence_missing"]
    try:
        data = yaml.safe_load(match.group(1).strip())
    except yaml.YAMLError:
        data = None
    entries: list[dict[str, Any]] = []
    parse_findings: list[str] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        turn = item.get("turn")
        empty = turn is None or (isinstance(turn, str) and not turn.strip())
        index = isinstance(turn, int) and not isinstance(turn, bool) and turn >= 0
        if not empty and not index:
            # `turn: true` used to fold into None and read as "no turn given".
            if "reflection_turn_not_an_index" not in parse_findings:
                parse_findings.append("reflection_turn_not_an_index")
        turn = turn if index else None
        entries.append(
            {
                "sentence": str(item.get("sentence") or "").strip(),
                "user_quote": str(item.get("user_quote") or "").strip(),
                "pa_quote": str(item.get("pa_quote") or "").strip(),
                "turn": turn,
                "t": _transcript_t(history, turn),
            }
        )
    if not entries:
        # A block that cannot be read carries no evidence at all.
        return [], ["reflection_evidence_missing"]
    return entries, parse_findings + _check_reflection_evidence(
        entries,
        reflection=reflection,
        emotion=emotion,
        answers=answers,
        history=history,
    )


class SessionEndSelfReporter:
    """One session-end call: reflection + closing emotion + five answers."""

    def __init__(
        self,
        provider: Any,
        *,
        profile_context: str,
        model: str,
        output_contract_retries: int = 2,
    ) -> None:
        if output_contract_retries < 0:
            raise ValueError("output_contract_retries must be non-negative")
        self.provider = provider
        self.profile_context = profile_context
        self.model = model
        self.output_contract_retries = output_contract_retries

    async def generate(
        self,
        *,
        session: SessionScript,
        history: list[dict[str, Any]],
    ) -> SessionEndReflectionResult:
        template = (
            Path(__file__).parent / "prompts" / "session_end_reflection.md"
        ).read_text(encoding="utf-8")
        prompt = (
            template.replace("<<CHARACTER_CARD>>", self.profile_context)
            .replace("<<LIFE_CONTEXT>>", render_life_context(session.life_context))
            # Indexed, and with the chat roles named, so an evidence entry can
            # point at a turn and at who spoke it.
            .replace("<<SESSION_TRANSCRIPT>>", _render_transcript(history))
        )
        messages = [{"role": "system", "content": prompt}]
        rejected: list[dict[str, Any]] = []
        for attempt in range(1, self.output_contract_retries + 2):
            response = await self._call(messages)
            raw = response.content
            reflection, emotion, answers, findings = _parse_reflection_output(raw)
            evidence: list[dict[str, Any]] = []
            evidence_findings: list[str] = []
            if not findings and answers is not None:
                evidence, evidence_findings = _parse_reflection_evidence(
                    raw,
                    reflection=reflection,
                    emotion=emotion,
                    answers=answers,
                    history=history,
                )
                # Evidence findings are recorded, never raised: the owner ruled
                # out a hard checker on what the reflection says. Record-only
                # since the owner ruling of 2026-09-14 ("只记"): no re-ask on an
                # evidence finding either; restore the condition
                # `if not evidence_findings or attempt > self.output_contract_retries:`
                # to bring the re-ask back.
                if True:
                    # Accepted, so still no retry and no void. But a
                    # `finish_reason=length` says the completion budget cut
                    # the reply (6 user_b session-end calls came back at
                    # completion_tokens=8188 with `<reflection_evidence>`
                    # truncated), and this path returns before the
                    # `_attempt_record` below, so without a row of its own the
                    # cut is invisible in
                    # `simulator_validation_attempts.jsonl`.
                    if response.finish_reason == "length":
                        rejected.append(
                            {
                                **self._attempt_record(
                                    attempt=attempt,
                                    findings=evidence_findings,
                                    session=session,
                                    response=response,
                                ),
                                "accepted_record_only": True,
                            }
                        )
                    return SessionEndReflectionResult(
                        reflection=reflection,
                        final_emotion=emotion,
                        self_report=answers,
                        rejected_attempts=rejected,
                        evidence=evidence,
                        findings=evidence_findings,
                    )
            findings = findings + evidence_findings
            rejected.append(
                self._attempt_record(
                    attempt=attempt,
                    findings=findings,
                    session=session,
                    response=response,
                )
            )
            if attempt > self.output_contract_retries:
                raise SelfReportContractError(findings, rejected_attempts=rejected)
            messages = self._correction_messages(messages, raw, findings)
        raise AssertionError("unreachable self-report retry state")

    async def _call(self, messages: list[dict[str, str]]) -> LLMResponse:
        import asyncio

        last_finish_reason = None
        for attempt in range(5):
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=[],
                model=self.model,
                max_tokens=SELF_REPORT_MAX_TOKENS,
            )
            if response.content:
                return response
            # Some models intermittently return an empty final message; retry.
            last_finish_reason = response.finish_reason
            await asyncio.sleep(3 * (attempt + 1))
        raise RuntimeError(
            "Self-report model returned no final content: "
            f"finish_reason={last_finish_reason!r}"
        )

    def _attempt_record(
        self,
        *,
        attempt: int,
        findings: list[str],
        session: SessionScript,
        response: LLMResponse,
    ) -> dict[str, Any]:
        # `finish_reason` and `usage` on the row so a cut response reads as a
        # token-budget hit on the record, not as a schema miss by the model.
        return {
            "phase": "self_report",
            "attempt": attempt,
            "error_codes": findings,
            "model": self.model,
            "session_id": session.session_id,
            "beat": "session_end",
            "raw_response": response.content,
            "finish_reason": response.finish_reason,
            "usage": dict(response.usage or {}),
        }

    @staticmethod
    def _correction_messages(
        messages: list[dict[str, str]], raw: str, findings: list[str]
    ) -> list[dict[str, str]]:
        return [
            *messages,
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "The output violated the private schema: "
                    f"{', '.join(findings)}. Regenerate the same result in the "
                    "required blocks without discussing this retry."
                ),
            },
        ]
