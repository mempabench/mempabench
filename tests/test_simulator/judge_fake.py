"""One scripted judge for every test that needs the ladder's judge answered.

Not a fixture and not a mock framework: the judge is a socket, and this is the
thing that stands in for it. It answers from a table, and a question with no
entry answers the last value of its own vocabulary (`no`, `not_applicable`),
which needs no evidence — so a test writes down only the answers it is about.

Two conveniences, both off by default, for the tests that are about something
else entirely and only have to get past the ask gate:

* `no_ask` — the candidate turns the gate answers `not_applicable` on. Every
  other turn reads as an ask, pointed at the candidate itself. This is how a
  test says "this turn asks nothing" without writing a per-attempt queue.
* the same switch forgives a scripted quote that is not in the prompt it was
  asked about: a scripted verdict is about one candidate turn, and the retry
  that removed the offending words is a different one.

A test still writes the WORDS it wants pointed at. Since the owner's ruling of
2026-09-09 the judge answers with a reference (`candidate_turn#2`) into blocks
printed with their sentences numbered, so `_reference` looks the scripted words
up in the prompt and answers with the number in front of them — which is what
the model does. Words that are nowhere in the prompt go out unchanged, so a
test can still script the answer the guard has to refuse.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from simulator.judge.client import JudgeCall

_CANDIDATE_RE = re.compile(r"<candidate_turn>\n(.*?)\n</candidate_turn>", re.S)
_BLOCK_RE = re.compile(r"<([A-Za-z0-9_:+.-]+)>\n(.*?)\n</\1>", re.S)
_NUMBERED_RE = re.compile(r"^\[(\d+)\] (.*)$")
_REF_RE = re.compile(r"^[^#\s,]+#\d+$")


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _unnumber(body: str) -> str:
    return _normalise(
        " ".join(_NUMBERED_RE.sub(r"\2", line) for line in body.splitlines())
    )


def _may_point_at(prompt: str, question_id: str) -> list[str]:
    """The block names the rendered question says its evidence may point at."""
    start = prompt.find(f"[{question_id}]\n")
    if start == -1:
        return []
    end = prompt.find("\n\n[", start)
    line = prompt[start : end if end != -1 else len(prompt)]
    _, _, tail = line.partition("numbered sentence of ")
    return re.findall(r"<([A-Za-z0-9_:+.-]+)>", tail.split("\n")[0])


def _reference(prompt: str, evidence: str, names: list[str]) -> str:
    """The reference that points at `evidence` where the prompt printed it.

    Only the blocks the question may point at are searched, the way the model
    reads the question: the same sentence often stands in two blocks, and a
    reference into the wrong one is refused.
    """
    if not evidence or _REF_RE.match(evidence):
        return evidence
    printed = dict(_BLOCK_RE.findall(prompt))
    needle = _normalise(evidence)
    for name in names or list(printed):
        for line in printed.get(name, "").splitlines():
            found = _NUMBERED_RE.match(line)
            if found and needle in _normalise(found.group(2)):
                return f"{name}#{found.group(1)}"
    return evidence


class ScriptedJudgeClient:
    """A judge that answers from a table, and never opens a socket."""

    model = "scripted-judge"
    base_url = "http://10.0.0.2:30001/v1"

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        *,
        raw: list[str] | None = None,
        raises: Exception | None = None,
        no_ask: Iterable[str] | None = None,
    ) -> None:
        self.answers = dict(answers or {})
        self.raw = list(raw or [])
        self.raises = raises
        #: When this is set the gate answers itself, `not_applicable` on these
        #: turns and `ask` on every other one.
        self.no_ask = {_normalise(text) for text in (no_ask or ())}
        self.reads_the_gate_itself = no_ask is not None
        self.calls: list[list[str]] = []
        self.prompts: list[str] = []

    def _candidate(self, prompt: str) -> str:
        found = _CANDIDATE_RE.search(prompt)
        return _unnumber(found.group(1)) if found else ""

    def _answer(self, question_id: str, values, prompt: str) -> tuple[str, str]:
        entry = self.answers.get(question_id)
        if isinstance(entry, list):
            entry = entry.pop(0) if entry else None
        if entry is None:
            if self.reads_the_gate_itself and question_id == "carries_an_ask":
                if self._candidate(prompt) in self.no_ask:
                    return "not_applicable", ""
                return "ask", "candidate_turn#1"
            return values[-1], ""
        verdict, evidence = entry
        reference = _reference(prompt, evidence, _may_point_at(prompt, question_id))
        if self.reads_the_gate_itself and evidence and not _REF_RE.match(reference):
            return values[-1], ""
        return verdict, reference

    def generate(self, *, system, prompt, allowed, num_predict_kind="yes_no"):
        if self.raises is not None:
            raise self.raises
        self.calls.append(list(allowed))
        self.prompts.append(prompt)
        if self.raw:
            body = self.raw.pop(0)
        else:
            verdicts = []
            for question_id, values in allowed.items():
                # One lookup per question: a queued answer is popped once.
                verdict, evidence = self._answer(question_id, values, prompt)
                verdicts.append(
                    {
                        "question_id": question_id,
                        "verdict": verdict,
                        "evidence": evidence,
                        "reason": "scripted",
                    }
                )
            body = json.dumps({"verdicts": verdicts})
        return JudgeCall(raw=body, seconds=0.01, prompt_tokens=1, eval_tokens=1)
