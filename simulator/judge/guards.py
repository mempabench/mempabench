"""The mechanical guards around an untrusted judge.

Four failure events, four exceptions, each carrying its event dict. This module
decides nothing about what happens next: the ladder (Phase 2) maps every one of
them onto `SessionInvalidError`. Here they are just raised.

Nothing in this file asks the model anything semantic. It strips fences, checks
the schema, checks that the evidence points at a numbered sentence of the block
the question pointed at, counts retries, and counts the per-generation budget.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

#: The verdict values of the judge reply schema, for a rule that returns one
#: verdict of its own.
VERDICT_VALUES = ("obeyed", "violated", "not_applicable")

#: The ask gate is a branch of its own: the turn either puts something to the
#: assistant or it does not.
ASK_GATE_VALUES = ("ask", "not_applicable")

#: Sub-questions — the ones whose answers a compose function reads — answer in
#: this pair and nothing else (amendment of 2026-09-09). `obeyed` / `violated`
#: carry a meaning the model acts on: asked whether a turn claims the assistant
#: never asked, it quoted the denial, wrote "the turn claims that the assistant
#: failed to ask for details" in its reason, and answered `obeyed`. The
#: experiment's raw true/false had nothing to read into, and this restores it.
YES_NO_VALUES = ("yes", "no")

#: `YES_NO_VALUES` plus the answer for a sub-question a previous sub-question
#: has already emptied of content (audit finding I, 2026-09-12). The frame tells
#: the judge to answer each question on its own, so a question that only makes
#: sense after another said `yes` needs a way to say "there was nothing to judge"
#: instead of being told to copy the other one's `no` — a `no` that would then
#: carry two meanings. A group may mix this with `YES_NO_VALUES`: the schema
#: union is a superset and `parse_verdicts` still holds each question to its own
#: set.
YES_NO_NA_VALUES = ("yes", "no", "not_applicable")

#: The side of each vocabulary that asserts a thing is PRESENT in the text. A
#: question's default `quotes_on` is whichever of these its answers contain, and
#: `Verdict.is_yes` reads the same side as the boolean's true.
PRESENCE_ANSWERS = ("yes", "violated", "ask")

#: `evidence` is a REFERENCE, not a copy: `<block>#<n>` names one of the
#: numbered sentences the block was printed with, and several are written
#: `candidate_turn#2,#3` or `candidate_turn#2, candidate_turn#3`. It is required
#: only on the answers its question declares in `quotes_on` — the answers that
#: point at words which are in the text. On every other answer nothing is
#: referenced and nothing is checked: told to quote on an answer that asserts an
#: ABSENCE — a cue that was not said, a clause that was not broken — the model
#: writes its reasoning into `evidence` instead (owner ruling 2026-09-09,
#: "没违反不引用").
#:
#: Copying was the old form and it is gone (owner ruling 2026-09-09). In the
#: live batch of that day five of seven session stops were a correct verdict
#: thrown away on the copy: the model quoted the question's own text, shortened
#: a long clause with an ellipsis, paraphrased, or copied a 460-character turn
#: and dropped one word of it. A number cannot be mistyped that way, and the
#: resolution below puts the sentence text back on the verdict, so nothing
#: downstream has to learn what a reference is.

#: One call gets at most three requests: the first plus the bounded re-asks.
#: Evidence gets exactly one retry inside that same cap.
MAX_REQUESTS_PER_CALL = 3
MAX_EVIDENCE_RETRIES = 1

#: What the one evidence re-ask carries (owner ruling 2026-09-09).
#: The re-ask used to be the identical request sent again, and in the live batch
#: of that day every one of the three evidence failures came back with the same
#: wrong answer. So it names the question, shows the reference the judge gave
#: back to it, says what is wrong with that reference, and names the blocks the
#: question may point at. The blocks themselves are not repeated: they are
#: already in the prompt this is appended to.
SCHEMA_FEEDBACK = (
    "\n\nYour previous answer was rejected: {defect}. Every verdict needs a "
    "`question_id`, a `verdict` from that question's list, and one sentence in "
    "`reason` — including an obeyed/no verdict whose `evidence` is empty. "
    "Answer every question again, in the same format as before."
)

#: Decoding is deterministic (temperature 0, prefix cache), so a byte-identical
#: re-ask returns the byte-identical defective reply: regress23m/acc_087 spent
#: three calls on the same blank `reason` before the session died. The schema
#: cannot forbid whitespace (xgrammar rejects `pattern`), so the retry changes
#: the prompt instead.
EVIDENCE_FEEDBACK = (
    "\n\nYour previous answer was rejected. For question {question_id} you gave "
    'the evidence "{evidence}", and {reason}. That question may point at: '
    "{block_names}. Answer every question again, in the same format as before. "
    "Where an answer needs evidence, write a reference and nothing else: the "
    "block name, then `#`, then the number printed in front of the sentence — "
    "for example `{block_name}#2`, and `{block_name}#1,#2` for two of them. Do "
    "not copy any words into `evidence`."
)

#: A reply cut off at the token cap is re-asked ONCE, at twice the width, and
#: never a third time: re-asking the identical request cannot succeed, and the
#: smoke run of 2026-09-09 spent three 30-second calls proving it before
#: voiding the session. 4096 is the ceiling on that one widening.
MAX_NUM_PREDICT = 4096

#: Question groups (calls) per candidate generation, not requests: the parse and
#: evidence retries above are bounded separately.
#: Worst case measured on the Phase 3 shapes is 13 (a beat that is both a
#: read-back first turn and a compressed reaction beat, after the two
#: diagnostics became live questions), plus one. A placeholder until regress22
#: gives the real distribution.
DEFAULT_TURN_BUDGET = 14

_WHITESPACE_RE = re.compile(r"\s+")
_CURLY_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*\n?|\n?\s*```\s*$")


# --------------------------------------------------------------------- events


class JudgeFailure(RuntimeError):
    """Base for the four events that void a session.

    Every subclass carries the event dict the record wants under `.event`. The
    module never falls back to a regex and never silently accepts a verdict.
    """

    event: dict[str, Any]


class JudgeUnparseable(JudgeFailure):
    """The requests ran out and the reply never satisfied the schema.

    Two ways to get here, and the read has to be able to tell them apart:
    three replies that were malformed, or two that were cut off at the token
    cap. The cut carries `truncated` and both `completion_tokens`, so the
    record names the cause instead of showing well-formed JSON that stops
    mid-value.
    """

    def __init__(
        self,
        *,
        question_id: str,
        attempts: int,
        raw_response: str,
        truncated: bool = False,
        completion_tokens: Sequence[int | None] = (),
    ) -> None:
        self.question_id = question_id
        self.attempts = attempts
        self.raw_response = raw_response
        self.truncated = truncated
        self.completion_tokens = list(completion_tokens)
        self.event = {
            "event": "judge_unparseable",
            "question_id": question_id,
            "attempts": attempts,
            "raw_response": raw_response,
        }
        if truncated:
            self.event["truncated"] = True
            self.event["completion_tokens"] = list(completion_tokens)
        super().__init__(
            f"judge_unparseable: question_id={question_id} attempts={attempts} "
            f"truncated={truncated} completion_tokens={list(completion_tokens)} "
            f"raw_response={raw_response[:400]!r}"
        )


class JudgeEvidenceInvalid(JudgeFailure):
    """The evidence reference does not name a sentence of the block it may read.

    The event carries what the read needs to tell the causes apart — the model
    wrote prose instead of a reference, named a block the question does not
    point at, or numbered a sentence that is not there. `block_id` names the
    block the question pointed at (`candidate_turn`, `beat_avoid`, several
    joined by `+`), `block_text` is that block in the numbered form the judge
    was given, and `raw_response` is the judge's last reply. Neither text is
    truncated: the candidate turn the judge read is dropped with the attempt,
    so this event is the only copy.

    Since the owner's ruling of 2026-09-09 this no longer voids the session:
    `JudgeSession.ask` records it and releases the turn (no verdict, so no
    finding from that group). It is still raised here — the guards decide
    nothing about what happens next.
    """

    def __init__(
        self,
        *,
        question_id: str,
        evidence: str,
        block_id: str = "",
        block_text: str = "",
        raw_response: str = "",
    ) -> None:
        self.question_id = question_id
        self.evidence = evidence
        self.block_id = block_id
        self.block_text = block_text
        self.raw_response = raw_response
        self.event = {
            "event": "judge_evidence_invalid",
            "question_id": question_id,
            "evidence": evidence,
            "block_id": block_id,
            "block_text": block_text,
            "raw_response": raw_response,
        }
        super().__init__(
            f"judge_evidence_invalid: question_id={question_id} "
            f"evidence={evidence[:200]!r} block_id={block_id}"
        )


class JudgeTimeout(JudgeFailure):
    """One call passed its wall-clock deadline."""

    def __init__(
        self,
        *,
        question_ids: Sequence[str],
        seconds: float,
        partial_response: str = "",
        partial_tokens: int = 0,
    ) -> None:
        self.question_ids = list(question_ids)
        self.seconds = seconds
        self.partial_response = partial_response
        self.partial_tokens = partial_tokens
        self.event = {
            "event": "judge_timeout",
            "question_ids": list(question_ids),
            "seconds": seconds,
            # Untruncated: a runaway generation is only readable whole.
            "partial_response": partial_response,
            "partial_tokens": partial_tokens,
        }
        super().__init__(
            f"judge_timeout: question_ids={list(question_ids)} seconds={seconds:.1f}"
        )


class JudgeBudgetExhausted(JudgeFailure):
    """This candidate generation asked for more question groups than it may."""

    def __init__(self, *, turn_uid: str, used: int) -> None:
        self.turn_uid = turn_uid
        self.used = used
        self.event = {
            "event": "judge_budget_exhausted",
            "turn_uid": turn_uid,
            "used": used,
        }
        super().__init__(f"judge_budget_exhausted: turn_uid={turn_uid} used={used}")


class JudgeSchemaError(ValueError):
    """Internal: one reply failed the schema. Retried, then re-raised as
    `JudgeUnparseable`."""


class JudgeEvidenceError(ValueError):
    """Internal: one verdict's reference does not resolve. Retried once, then
    re-raised as `JudgeEvidenceInvalid`. `reason` is the half-sentence the
    re-ask shows the judge."""

    def __init__(self, question_id: str, evidence: str, reason: str = "") -> None:
        self.question_id = question_id
        self.evidence = evidence
        self.reason = reason
        super().__init__(f"{question_id}: {evidence[:120]!r} — {reason}")


# ---------------------------------------------------------------- normalising


def strip_fences(raw: str) -> str:
    """The JSON object out of whatever the model wrapped it in.

    Three shapes, all measured on this model: a ```json fence around the whole
    reply, prose ahead of the first `{`, and prose after the last `}`.
    """
    text = (raw or "").strip()
    while True:
        stripped = _FENCE_RE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def normalise(text: str) -> str:
    """The evidence comparison form: whitespace collapsed, case folded, curly
    quotes straightened, accented letters folded to their base letter (NFKD)."""
    decomposed = unicodedata.normalize("NFKD", (text or "").translate(_CURLY_QUOTES))
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WHITESPACE_RE.sub(" ", folded).strip().casefold()


# ------------------------------------------------------- numbering, references

#: A sentence ends at a terminator followed by whitespace — the same cut
#: `general.split_clauses` makes on an `avoid` clause. Lines are cut first, so a
#: markdown bullet or a table row is a piece of its own whatever it ends with.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: One reference: `candidate_turn#2`, or a bare `#3` or `3` continuing the block
#: named before it in the same string.
_REF_RE = re.compile(r"^(?:(?P<block>[^#\s,]*)#)?(?P<index>\d+)$")


def number_sentences(text: str) -> list[str]:
    """The numbered pieces of one block, each verbatim.

    The one segmentation in the module: `numbered_block` prints these and
    `check_evidence` resolves against them, so the number the judge reads and
    the number the guard counts cannot drift apart.
    """
    pieces: list[str] = []
    for line in (text or "").splitlines():
        for part in _SENTENCE_SPLIT_RE.split(line):
            if part.strip():
                pieces.append(part.strip())
    return pieces


def numbered_block(name: str, text: str) -> str:
    """One block as the judge is given it: `<name>`, its numbered sentences,
    `</name>`."""
    body = "\n".join(
        f"[{index}] {sentence}"
        for index, sentence in enumerate(number_sentences(text), start=1)
    )
    return f"<{name}>\n{body}\n</{name}>"


def parse_refs(evidence: str) -> list[tuple[str, int]]:
    """`"candidate_turn#2, #3"` -> `[("candidate_turn", 2), ("candidate_turn", 3)]`.

    `"candidate_turn#2,3"` is that same list: a piece that is a bare number
    continues the block named before it, so one pointer may list several
    sentence numbers of its block. About 1% of this model's answers write that
    shape (owner ruling, 2026-09-16); every number in it still has to resolve.

    Raises `ValueError` with the half-sentence the re-ask shows the judge.
    """
    refs: list[tuple[str, int]] = []
    block = ""
    for piece in (evidence or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        match = _REF_RE.match(piece)
        if not match:
            raise ValueError(
                f'"{piece}" is not a reference of the form <block>#<number>'
            )
        block = match.group("block") or block
        if not block:
            raise ValueError(f'"{piece}" names no block')
        refs.append((block, int(match.group("index"))))
    if not refs:
        raise ValueError("it is empty and that answer needs a reference")
    return refs


# ------------------------------------------------------------------- verdicts


@dataclass(frozen=True)
class Verdict:
    """One answer: which question, what it says, the words it says it with.

    The model writes a reference into `evidence`; `check_evidence` resolves it
    and puts the sentence text back in that field, keeping the reference in
    `evidence_ref`. Everything downstream — the composes, the events, the
    replay predictions, the `judge_call` line — therefore still reads text.
    """

    question_id: str
    verdict: str
    evidence: str
    reason: str = ""
    evidence_ref: str = ""

    @property
    def is_yes(self) -> bool:
        """The question's answer read as a boolean.

        A sub-question answers `yes`/`no`. A whole-rule verdict answers
        `violated` when the thing the rule names happened, and the ask gate
        answers `ask`; those are the yes side too.
        """
        return self.verdict in PRESENCE_ANSWERS


def parse_verdicts(
    raw: str,
    *,
    allowed: Mapping[str, Sequence[str]],
) -> list[Verdict]:
    """Parse and schema-check one reply. Raises `JudgeSchemaError` on any fault.

    `allowed` maps each question's id to the values that question may answer in.
    Its keys are the id set, which must match exactly — too many, too few, or a
    wrong one is a schema failure — and its values are checked per
    question, so a sub-question answering `obeyed` is as much a schema failure
    as one answering nonsense.
    """
    question_ids = list(allowed)
    text = strip_fences(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeSchemaError(f"reply is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeSchemaError("reply is not a JSON object")
    items = payload.get("verdicts")
    if not isinstance(items, list):
        raise JudgeSchemaError("reply has no `verdicts` array")

    verdicts: list[Verdict] = []
    for item in items:
        if not isinstance(item, dict):
            raise JudgeSchemaError("a verdict entry is not an object")
        question_id = item.get("question_id")
        verdict = item.get("verdict")
        evidence = item.get("evidence", "")
        reason = item.get("reason", "")
        if not isinstance(question_id, str) or not isinstance(verdict, str):
            raise JudgeSchemaError("a verdict entry has a non-string id or verdict")
        if not isinstance(evidence, str) or not isinstance(reason, str):
            raise JudgeSchemaError(f"{question_id}: evidence and reason must be strings")
        # A blank `reason` is a dropped answer, not a considered one: ~8 live
        # answers came back with both fields empty and no replay could tell
        # the two apart (regress23 read, finding 2). The schema's `minLength`
        # stops "" at the sampler; this stops " " and "\n", which it does not,
        # and `withheld_fact_before_ask` needs a first word to read.
        if not reason.strip():
            raise JudgeSchemaError(f"{question_id}: reason is empty")
        values = allowed.get(question_id)
        if values is None:
            raise JudgeSchemaError(f"{question_id}: not a question that was asked")
        if verdict not in values:
            raise JudgeSchemaError(
                f"{question_id}: verdict {verdict!r} is not one of {list(values)}"
            )
        verdicts.append(Verdict(question_id, verdict, evidence, reason))

    got = [verdict.question_id for verdict in verdicts]
    if sorted(got) != sorted(question_ids) or len(got) != len(set(got)):
        raise JudgeSchemaError(
            f"question ids do not match: asked {list(question_ids)}, got {got}"
        )
    return verdicts


def check_evidence(
    verdicts: Sequence[Verdict],
    blocks_by_name: Mapping[str, str],
    quotes_on: Mapping[str, Sequence[str]],
    question_blocks: Mapping[str, Sequence[str]],
) -> list[Verdict]:
    """Resolve every quoting answer's reference, or raise `JudgeEvidenceError`.

    `blocks_by_name` maps block name -> its text, the same blocks the prompt
    printed. `question_blocks` maps question_id -> the block names that
    question may point at: the candidate turn by default, the named context
    block for a two-step q2, and both of a multi-block question's names.
    `quotes_on` maps question_id -> the answers that question
    requires a reference on; an answer outside that set asserts something is
    NOT in the text, so there is nothing to point at and its evidence is
    neither required nor read.

    The returned verdicts are the ones to carry on with: a resolved answer's
    `evidence` is the sentence text its reference names, and `evidence_ref`
    keeps the reference.
    """
    resolved: list[Verdict] = []
    for verdict in verdicts:
        if verdict.verdict not in quotes_on.get(verdict.question_id, ()):
            resolved.append(verdict)
            continue
        names = tuple(question_blocks.get(verdict.question_id, ()))
        try:
            refs = parse_refs(verdict.evidence)
        except ValueError as exc:
            raise JudgeEvidenceError(
                verdict.question_id, verdict.evidence, str(exc)
            ) from exc
        sentences: list[str] = []
        for name, index in refs:
            if name not in names:
                raise JudgeEvidenceError(
                    verdict.question_id,
                    verdict.evidence,
                    f"{name} is not a block this question may point at",
                )
            numbered = number_sentences(blocks_by_name.get(name, ""))
            if not 1 <= index <= len(numbered):
                raise JudgeEvidenceError(
                    verdict.question_id,
                    verdict.evidence,
                    f"{name} has no sentence [{index}]",
                )
            sentences.append(numbered[index - 1])
        resolved.append(
            replace(
                verdict,
                evidence=" ".join(sentences),
                evidence_ref=verdict.evidence,
            )
        )
    return resolved


# --------------------------------------------------------------------- budget


class TurnBudget:
    """Question groups spent on one candidate generation."""

    def __init__(self, turn_uid: str, limit: int | None = None) -> None:
        self.turn_uid = turn_uid
        self.limit = limit if limit is not None else _env_int(
            "MEMPA_JUDGE_TURN_BUDGET", DEFAULT_TURN_BUDGET
        )
        self.used = 0

    def spend(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise JudgeBudgetExhausted(turn_uid=self.turn_uid, used=self.used)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


# ----------------------------------------------------------------- one call


def ask(
    client,
    *,
    system: str,
    prompt: str,
    allowed: Mapping[str, Sequence[str]],
    blocks_by_name: Mapping[str, str],
    quotes_on: Mapping[str, Sequence[str]],
    question_blocks: Mapping[str, Sequence[str]],
    num_predict_kind: str = "yes_no",
    budget: TurnBudget | None = None,
) -> list[Verdict]:
    """One question group, guarded.

    Parse failure re-asks the same payload up to `MAX_REQUESTS_PER_CALL` times
    then raises `JudgeUnparseable`; a reference that does not resolve re-asks
    once then raises `JudgeEvidenceInvalid`. A timeout comes straight out of
    the client. Neither retry counts against the budget, which counts calls.

    The evidence re-ask is not the identical request: the prompt is the original
    plus `EVIDENCE_FEEDBACK`, which names the question, the reference the judge
    gave, what is wrong with it, and the blocks that question may point at.
    Everything else about the request is unchanged, and it is still the same
    guarded call.

    A reply cut off at the token cap is the one failure the identical re-ask
    cannot fix, so it is not one: the request is re-sent once at twice the
    width, and a second cut raises `JudgeUnparseable` there and then, carrying
    `truncated` and both `completion_tokens` so the read sees why.
    """
    if not allowed:
        # No questions, so no question group and no call. Most beats owe no
        # required facts at all: 413 of the 546 corpus records build an empty
        # `probe_fact_missing` group, and a call for one would ask Ollama to
        # sample against an empty enum.
        return []

    if budget is not None:
        budget.spend()

    question_ids = list(allowed)
    last_raw = ""
    last_schema_error: JudgeSchemaError | None = None
    evidence_retries = 0
    cut_tokens: list[int | None] = []
    widened: int | None = None
    sent_prompt = prompt
    for _ in range(MAX_REQUESTS_PER_CALL):
        # The width override is sent only on the widened re-ask, so a client
        # that never returns a cut reply never sees the argument.
        override = {"num_predict": widened} if widened is not None else {}
        call = client.generate(
            system=system,
            prompt=sent_prompt,
            allowed=allowed,
            num_predict_kind=num_predict_kind,
            **override,
        )
        last_raw = call.raw
        if getattr(call, "truncated", False):
            cut_tokens.append(call.eval_tokens)
            if len(cut_tokens) == 1:
                widened = min(call.num_predict * 2, MAX_NUM_PREDICT)
                continue
            raise JudgeUnparseable(
                question_id=question_ids[0] if question_ids else "",
                attempts=len(cut_tokens),
                raw_response=last_raw,
                truncated=True,
                completion_tokens=cut_tokens,
            )
        try:
            verdicts = parse_verdicts(call.raw, allowed=allowed)
        except JudgeSchemaError as exc:
            last_schema_error = exc
            sent_prompt = prompt + SCHEMA_FEEDBACK.format(defect=exc)
            continue
        try:
            return check_evidence(
                verdicts, blocks_by_name, quotes_on, question_blocks
            )
        except JudgeEvidenceError as exc:
            names = tuple(question_blocks.get(exc.question_id, ())) or (
                "candidate_turn",
            )
            if evidence_retries < MAX_EVIDENCE_RETRIES:
                evidence_retries += 1
                sent_prompt = prompt + EVIDENCE_FEEDBACK.format(
                    question_id=exc.question_id,
                    evidence=exc.evidence,
                    reason=exc.reason,
                    block_names=", ".join(names),
                    block_name=names[0],
                )
                continue
            raise JudgeEvidenceInvalid(
                question_id=exc.question_id,
                evidence=exc.evidence,
                block_id="+".join(names),
                block_text="\n\n".join(
                    numbered_block(name, blocks_by_name.get(name, ""))
                    for name in names
                ),
                raw_response=last_raw,
            ) from exc

    raise JudgeUnparseable(
        question_id=question_ids[0] if question_ids else "",
        attempts=MAX_REQUESTS_PER_CALL,
        raw_response=last_raw,
    ) from last_schema_error
