"""Pure-function rendering of `reasoning_visibility` and `uncertainty_expression`.

Rules:

- `reasoning_visibility`  the render source is **only** the `user_facing_rationale` the PA wrote
  itself in `IX_render_reasoning_visibility`; the provider-side
  `reasoning_content` / `reasoning` / `reasoning_details` / `<think>` is never
  rendered, it goes to `AuditTrace.provider_reasoning` alone. That field is
  **required under both** `Show` and `Hide` -- whether it is required must not
  leak the setting; whether it is released is what the setting decides.
- `uncertainty_expression`  `Express`: append `(confidence: low)` at every
  certainty; `Hide`: append nothing, just drop the
  markers. The **primary carrier for level and position is the `<certainties>`
  block at the end of the reply** (carrier F): tool calls are produced
  before the prose, so the model cannot quote words it has not written yet; the
  trailing block is it **copying the prose it just finished**, which is where
  the ordering problem gets solved. With no trailing block it falls back to the
  `IX_render_uncertainty_expression` tool arguments, and `[[c:*]]` markers are
  still honoured.
- the unset fallback (no selection this session for the attribute)  no
  setting-specific transform (no parenthesis appended, `user_facing_rationale`
  not released), but `[[c:*]]` markers are **still deterministically stripped**
  (contract 6).
- the JSON shapes of `ix_render.reasoning` (`ReasoningJSON`) and
  `ix_render.certainties` (`CertaintyJSON`).

This module is pure: no IO, no LLM calls, it never reads provider reasoning.
`render.py` calls the entry points here and folds the results into
`RenderedTurn` / `AuditTrace`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from pa.agent.ix.render_types import (
    CERTAINTY_MARKER_RE,
    Certainty,
    CertaintyJSON,
    ReasoningJSON,
    ReasoningVisibilitySetting,
    RenderStep,
    UncertaintyExpressionSetting,
)

#: After this turn `metadata["ix_render"].reasoning.source`
#: has only this one value.
RATIONALE_SOURCE = "user_facing_rationale"

#: The parenthetical wording. Every benchmark persona speaks English (user_a),
#: and a Chinese parenthetical would reach live English prose, so what ships is
#: `(confidence: low)`, with `high` / `medium` in the same shape.
CERTAINTY_LABELS: dict[str, str] = {"high": "high", "medium": "medium", "low": "low"}

#: The levels each setting really appends a parenthesis for.
#: A setting outside this table — the unset fallback included — appends
#: none.
_ANNOTATED_LEVELS: dict[str, frozenset[str]] = {
    "Express": frozenset(CERTAINTY_LABELS),
    "Hide": frozenset(),
}


def certainty_annotation(level: str) -> str:
    """The parenthesis for `level`; "" when `level` is not in `CERTAINTY_LABELS`."""
    label = CERTAINTY_LABELS.get(str(level))
    return f"(confidence: {label})" if label else ""


def normalize_rationale(raw: object) -> str:
    """The PA's `user_facing_rationale` normalised to a string; not really filled -> "".

    Type and whitespace checks only, no semantic judgement -- "is this passage
    evidence and a causal chain, or is it inner monologue" is constrained on the
    model by the setting descriptions in `specs.py`, not decided by the renderer.
    """
    return raw.strip() if isinstance(raw, str) else ""


#: acc_011 t=27 — the `user_facing_rationale` that opened the delivered reply
#: with *"The user approved the prioritization logic and asked me to save the
#: document."* and then addressed that same user in the second person two lines
#: later. Under `Show` the field is printed **ahead of** the answer, so this is
#: the first thing they read. The parameter contract's own *Bad, a work log*
#: example is almost verbatim this, and `specs.py` says it too; the model
#: ignored both, and nothing counted it because the call was well-formed.
#:
#: Deliberately narrow — the opening only. "Is this passage reasons or a recap"
#: is not a judgement the renderer makes; what this catches is the one
#: mechanical tell, a rationale that starts by talking *about* the reader.
_WORK_LOG_OPENING_RE = re.compile(
    r"\A[\s*_>#-]*the\s+(?:user|client)\b",
    re.IGNORECASE,
)


def reads_as_a_work_log(rationale: object) -> bool:
    """Does this rationale open by talking about the user in the third person?"""
    return bool(_WORK_LOG_OPENING_RE.match(normalize_rationale(rationale)))


@dataclass(frozen=True)
class RationaleRender:
    """This turn's render result for `reasoning_visibility`."""

    prefix: str = ""
    """Under `Show` with a non-empty field = the passage rendered **before** the
    answer; "" in every other case. The renderer **never writes for the PA**: a
    missing field leaves "" here, no substitute text is synthesised."""
    shown: bool = False
    reasoning: ReasoningJSON = field(
        default_factory=lambda: ReasoningJSON(
            shown=False, source=RATIONALE_SOURCE, provider_reasoning_present=False
        )
    )
    """`ix_render.reasoning`."""
    trace: list[RenderStep] = field(default_factory=list)
    """Goes to `audit_trace` only, never to the text plane."""


def render_rationale(
    setting: ReasoningVisibilitySetting | str,
    rationale: object,
    *,
    provider_reasoning_present: bool = False,
) -> RationaleRender:
    """`Show` releases `user_facing_rationale`; `Hide` / unset keep it in `audit_trace`.

    `provider_reasoning_present` is only the single boolean "was any captured" --
    the **content** of the provider reasoning never passes through here, and it
    certainly never lands in `prefix`.
    """
    text = normalize_rationale(rationale)
    shown = setting == "Show" and bool(text)

    if shown:
        step = f"reasoning_visibility=Show -> rationale released ({len(text)} chars)"
    elif setting == "Show":
        step = "reasoning_visibility=Show -> user_facing_rationale empty, nothing released"
    elif setting == "Hide":
        step = "reasoning_visibility=Hide -> rationale held (audit only)"
    else:
        # No selection for this attribute, or one outside the domain: the unset
        # fallback releases no internal field. It goes to the trace, never
        # silently.
        step = f"reasoning_visibility={setting or 'unset'} -> unset fallback, nothing released"

    return RationaleRender(
        prefix=text if shown else "",
        shown=shown,
        reasoning=ReasoningJSON(
            shown=shown,
            source=RATIONALE_SOURCE,
            provider_reasoning_present=bool(provider_reasoning_present),
        ),
        trace=[step],
    )


@dataclass(frozen=True)
class CertaintyRender:
    """This turn's render result for `uncertainty_expression`."""

    text: str = ""
    """The prose: every `[[c:*]]` marker is stripped and, per setting, replaced
    in place by the parenthesis ("" when this setting appends nothing)."""
    rows: list[CertaintyJSON] = field(default_factory=list)
    """`ix_render.certainties`, one row per `certainties` entry the PA
    submitted; `rendered` is the parenthesis **actually** appended, "" for none."""
    trace: list[RenderStep] = field(default_factory=list)


def _flatten(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text, plus the index table "flat char i came from original position n"."""
    chars: list[str] = []
    index: list[int] = []
    previous_space = False
    for position, char in enumerate(text):
        if char.isspace():
            if chars and not previous_space:
                chars.append(" ")
                index.append(position)
            previous_space = True
            continue
        chars.append(char)
        index.append(position)
        previous_space = False
    return "".join(chars), index


#: The markdown emphasis marks the model brings along when it copies its own
#: prose -- stripped from both sides before comparing.
_EMPHASIS_CHARS = "*_`~"

#: A clause quoted as a sentence: the model adds punctuation at the end, while
#: in the prose it is usually followed by ` and …`. In a live run 2 of 9 claims
#: differed by exactly that one character.
_EDGE_PUNCTUATION = " \t.。,，、;；:：!！?？'\"“”‘’()（）[]【】《》<>-—…"

#: Tokenisation for the fallback pass: a run of ASCII alphanumerics is one
#: token, each non-ASCII (CJK) character is one token.
_TOKEN_RE = re.compile(r"[0-9a-z]+|[^\x00-\x7f]")

#: A claim's tokens must be hit in order at at least this rate to count as
#: "this is that passage" (0.8).
_TOKEN_OVERLAP_THRESHOLD = 0.8


def _normalized(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed + emphasis-stripped + lowercased text, plus the index table back into the original.

    The three go together because they have to be done **on both sides at
    once**: the `**` in the needle and the `**` in the prose are not necessarily
    in the same place, and a verbatim comparison then misses the whole claim
    over a single asterisk.
    """
    chars: list[str] = []
    index: list[int] = []
    previous_space = False
    for position, char in enumerate(text):
        if char in _EMPHASIS_CHARS:
            continue
        if char.isspace():
            if chars and not previous_space:
                chars.append(" ")
                index.append(position)
            previous_space = True
            continue
        lowered = char.lower()
        # A few characters change length when lowercased (İ -> i̇), which would
        # wreck the index table, so keep the original character here.
        chars.append(lowered if len(lowered) == 1 else char)
        index.append(position)
        previous_space = False
    return "".join(chars), index


def _tokens(text: str) -> list[tuple[str, int]]:
    """`(token, index just past the token)`, an exclusive right end in the **normalised** string."""
    return [(match.group(0), match.end()) for match in _TOKEN_RE.finditer(text)]


def _token_overlap_end(
    haystack: Sequence[tuple[str, int]], needle: Sequence[str]
) -> int | None:
    """Right end of the earliest window covering >= the threshold of needle tokens in order (normalised index).

    The model occasionally rewrites a word or two of the sentence it promised
    before putting it in the prose. A verbatim comparison is helpless against
    that, while "this passage is that claim" is in fact perfectly clear, so the
    last resort is token coverage. The window is bounded, so words scattered
    across the whole reply cannot be assembled into a hit.
    """
    if not needle or not haystack:
        return None
    required = _TOKEN_OVERLAP_THRESHOLD * len(needle)
    span = 2 * len(needle) + 4
    # acc_035 — the skip window, proportional to the claim instead of fixed at
    # two. The declared claim was the prose sentence with `on quiz 1 question 2`
    # inserted into it: 21 of 26 tokens present, 81% and over the threshold, but
    # a five-token run broke the in-order walk and the pass returned `None`. The
    # first `low` claim in four rounds went unbracketed, on the cell that exists
    # to produce exactly that bracket.
    #
    # The complement of the threshold, not a second knob: 20% of the needle is
    # already the most that may go missing in total, so a run longer than that
    # can never reach `required` anyway and a wider window would be dead range.
    # The floor of two keeps the old behaviour for short claims.
    skip = max(2, math.ceil((1 - _TOKEN_OVERLAP_THRESHOLD) * len(needle)))
    for start in range(len(haystack)):
        matched = 0
        last: int | None = None
        at = 0
        for position in range(start, min(len(haystack), start + span)):
            if at >= len(needle):
                break
            token, end = haystack[position]
            ahead = [
                offset
                for offset in range(at, min(at + skip + 1, len(needle)))
                if needle[offset] == token
            ]
            if not ahead:
                continue
            matched += 1
            last = end
            at = ahead[0] + 1
        if last is not None and matched >= required:
            return last
    return None


def _after_emphasis(text: str, end: int) -> int:
    """Push the insertion point past a trailing `**` / `` ` ``, so the parenthesis does not land inside the bold."""
    while end < len(text) and text[end] in _EMPHASIS_CHARS:
        end += 1
    return end


#: Where a confidence parenthesis may land: the end of a clause or a sentence.
_CLAUSE_BOUNDARY = ".;,:。；，："


def _clause_boundary(text: str, end: int) -> int | None:
    """The first clause or sentence boundary at or after `end`; `None` when the clause runs on.

    `acc_035` t=3: the overlap window stopped on `rather`, because the
    claim's tail (`than as a phase`) fell outside the bounded window, and the
    parenthesis went in as `rather(confidence: high) than` — inside a word pair,
    in the middle of a clause.

    The overlap pass answers *which passage the claim is*, not where that
    passage ends, so its index is a starting point and not an insertion point:
    the marker is pushed to the end of the clause it landed in. A line break
    ends a clause too — in a markdown list the claim's line is the clause, and
    scanning on would carry the marker into the next bullet — so it is a
    boundary, and the insertion goes before it rather than after.

    When nothing but whitespace follows, the claim ended the reply and that end
    is the boundary. When prose runs on with no boundary at all, there is no
    place to put the marker and the caller falls back to the not-found path.
    """
    for position in range(end, len(text)):
        char = text[position]
        if char in _CLAUSE_BOUNDARY:
            return position + 1
        if char == "\n":
            return position
    if text[end:].strip(_EMPHASIS_CHARS + " \t"):
        return None
    return len(text)


@dataclass(frozen=True)
class ClaimMatch:
    """Where `claim` landed in the prose, and which pass found it."""

    end: int
    """Index just past the claim's last character in the original string -- the parenthesis goes here."""
    strategy: str
    """`exact` / `normalized` / `token_overlap`, goes to `trace` only."""


def claim_end(text: str, claim: str) -> ClaimMatch | None:
    """The index just past the first occurrence of `claim` in `text`; `None` when not found.

    Three passes, each wider than the last; the name of the first one that hits
    goes to `trace`:

    1. `exact` -- verbatim after whitespace collapsing (the sentence the model
       wrote and the prose often differ in line breaks / indentation);
    2. `normalized` -- both sides also lose markdown emphasis and case, and the
       needle loses sentence-final punctuation at either end: in live runs the
       commonest reason a "copied word for word" claim missed was the model
       adding a full stop in the tool call while the prose sentence is followed
       by ` and …`;
    3. `token_overlap` -- >= 80% of the tokens hit in order counts as found (the
       rewritten kind). This pass says which passage the claim is, not where it
       ends, so its index is snapped forward to the end of the clause it landed
       in (`_clause_boundary`); a match with no clause end ahead of it is not a
       match, because the only alternative is a marker inside a word pair.
    """
    needle, _ = _flatten(claim)
    needle = needle.strip()
    if not needle:
        return None

    flat, flat_index = _flatten(text)
    at = flat.find(needle)
    if at >= 0:
        return ClaimMatch(_after_emphasis(text, flat_index[at + len(needle) - 1] + 1), "exact")

    norm_text, norm_index = _normalized(text)
    norm_needle, _ = _normalized(claim)
    norm_needle = norm_needle.strip(_EDGE_PUNCTUATION)
    if not norm_needle:
        return None
    at = norm_text.find(norm_needle)
    if at >= 0:
        end = norm_index[at + len(norm_needle) - 1] + 1
        return ClaimMatch(_after_emphasis(text, end), "normalized")

    overlap = _token_overlap_end(_tokens(norm_text), [token for token, _ in _tokens(norm_needle)])
    if overlap is not None:
        boundary = _clause_boundary(
            text, _after_emphasis(text, norm_index[overlap - 1] + 1)
        )
        if boundary is not None:
            return ClaimMatch(_after_emphasis(text, boundary), "token_overlap")
    return None


# ---------------------------------------------------------------------------
# carrier F · the `<certainties>` block at the end of the reply
# ---------------------------------------------------------------------------

#: The trailing block's boundary. Case-insensitive; the block is stripped along
#: with the whitespace around it.
CERTAINTY_BLOCK_RE = re.compile(
    r"[ \t]*<certainties>(?P<body>.*?)</certainties>[ \t]*",
    re.DOTALL | re.IGNORECASE,
)

#: One row inside the block: `- ref: c1 | level: low | claim: …`. `claim` comes
#: last, so a `|` inside it does no harm. The leading `-` / `*` / `•` is optional.
_BLOCK_ROW_RE = re.compile(
    r"^\s*[-*•]?\s*ref\s*:\s*(?P<ref>[^|]+?)\s*\|"
    r"\s*level\s*:\s*(?P<level>[^|]+?)\s*\|"
    r"\s*claim\s*:\s*(?P<claim>.+?)\s*$",
    re.IGNORECASE,
)

#: The instruction shown to the model. **Setting-blind**: all three settings read
#: the same passage, and writing the block or not leaks no setting. The tool
#: descriptions (`tools.py`) and the per-turn restatement (`collector.py`) read
#: this same one.
CERTAINTY_BLOCK_INSTRUCTION = (
    "End every reply to the user with a confidence block, after all of your prose, "
    "on its own lines:\n"
    "<certainties>\n"
    "- ref: c1 | level: low | claim: <a span copied out of the reply above>\n"
    "- ref: c2 | level: high | claim: <...>\n"
    "</certainties>\n"
    "One line per claim in that reply whose confidence matters — a date, an "
    "availability, a recommendation, a lookup you could not fully verify. `level` is "
    "high, medium or low, and `claim` must be copied word for word out of the prose "
    "you just wrote: plain running prose, not a heading, not a table cell, not a "
    "bullet you reformatted, because the claim is what locates the confidence in your "
    "text. Only sentences of this reply: a claim from an earlier reply of yours is not "
    "in this text and locates nothing. A reply where no claim's confidence matters ends with "
    "`<certainties></certainties>`. The block is wrapper bookkeeping, identical under "
    "every setting: it is removed before the user sees the reply, so never mention it, "
    "and never hedge in the prose instead."
)


@dataclass(frozen=True)
class CertaintyBlock:
    """What the block at the end of the reply parsed into."""

    found: bool = False
    """Whether the block is there. Only when it is **not** do we fall back to the
    tool arguments (they are the fallback, not the primary channel)."""
    rows: list[Certainty] = field(default_factory=list)
    text: str = ""
    """The prose with the block stripped out."""
    trace: list[RenderStep] = field(default_factory=list)


def has_certainty_block(text: str) -> bool:
    """Whether this turn's reply has a trailing block -- the required predicate reads it."""
    return bool(text) and CERTAINTY_BLOCK_RE.search(text) is not None


def strip_certainty_block(text: str) -> str:
    """Strip every `<certainties>` block; return the text unchanged when there is none.

    The contract: **under every setting** the block is barred from
    `public_turn` / `runtime_history` / memory / judge pack, and stays only in
    `audit_trace`'s `segments` and the provider trace.
    """
    if not text:
        return text or ""
    if CERTAINTY_BLOCK_RE.search(text) is None:
        return text
    stripped = CERTAINTY_BLOCK_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def certainty_block_text(text: str) -> str:
    """The trailing `<certainties>` block verbatim, tags included, or "".

    `raw_blocks`: the block is a carrier the renderer strips out of every
    delivered surface, so without this the one thing that says which claims the
    model committed to would exist nowhere on the audit plane either -- under
    `Hide` nobody could tell a located-and-withheld claim from one never found.
    The last block wins, the same rule `parse_certainty_block` reads by.
    """
    matches = list(CERTAINTY_BLOCK_RE.finditer(text or ""))
    return matches[-1].group(0).strip() if matches else ""


def parse_certainty_block(text: str) -> CertaintyBlock:
    """Read the `<certainties>` block at the end of the reply. With several, the **last** wins.

    A line that will not parse only goes to the trace, it does not raise: the
    block is handwritten by the model, and one botched line should not void the
    whole turn -- the backstop for shape validation is still "no block and no
    tool arguments -> one reminder -> contract failure".
    """
    matches = list(CERTAINTY_BLOCK_RE.finditer(text or ""))
    if not matches:
        return CertaintyBlock(text=text or "")

    rows: list[Certainty] = []
    trace: list[RenderStep] = []
    seen: set[str] = set()
    for line in matches[-1].group("body").splitlines():
        if not line.strip():
            continue
        parsed = _BLOCK_ROW_RE.match(line)
        if parsed is None:
            trace.append(f"certainty block: line not parsed {line.strip()[:60]!r}")
            continue
        ref = parsed.group("ref").strip()
        if not ref or ref in seen:
            trace.append(f"certainty block: duplicate or empty ref {ref!r} ignored")
            continue
        seen.add(ref)
        rows.append(Certainty(
            ref=ref,
            level=parsed.group("level").strip().lower(),
            claim=parsed.group("claim").strip(),
        ))
    trace.insert(0, f"certainty block: {len(rows)} row(s) read from the reply")
    return CertaintyBlock(
        found=True, rows=rows, text=strip_certainty_block(text), trace=trace
    )


def render_certainties(
    setting: UncertaintyExpressionSetting | str,
    text: str,
    certainties: Sequence[Certainty] = (),
) -> CertaintyRender:
    """Append the parenthesis per setting; all four branches must strip `[[c:*]]` markers and the `<certainties>` block clean.

    Two production channels, markers first and claims second: a `[[c:ref]]`
    marker present in the prose is replaced where it stands; a ref with no
    marker is located in the prose by its `claim`, and the parenthesis goes in
    after that passage. **The claim is the primary channel** -- tool calls are
    produced before the prose, so the model cannot place a marker in text it has
    not written yet (two live runs had a constant marker hit rate of 0). A claim
    that is not found inserts nothing and only writes a trace line: the payload
    was given, so it is not a payload miss. A claim that is found but not
    appended under this setting writes a line too -- otherwise `Hide`'s "no
    parenthesis" is unfalsifiable: the renderer finding every claim and
    withholding per setting looks exactly like it finding none.

    The appended parenthesis is **not part of the prose** -- the model does not
    hedge in prose, the parenthesis is generated here.
    """
    levels = _ANNOTATED_LEVELS.get(str(setting))
    trace: list[RenderStep] = []
    if levels is None:
        # No selection for this attribute, or one outside the domain: the unset
        # fallback appends not one parenthesis, and the markers are stripped all
        # the same.
        levels = frozenset()
        trace.append(f"uncertainty_expression={setting or 'unset'} -> unset fallback")

    # The trailing block is a **carrier**, not prose: under every setting it is
    # stripped before rendering, and no claim may be located inside it.
    if has_certainty_block(text):
        text = strip_certainty_block(text)
        trace.append(
            f"uncertainty_expression={setting or 'unset'} -> "
            "trailing block stripped from the reply"
        )

    by_ref: dict[str, str] = {}
    claims: dict[str, str] = {}
    for item in certainties:
        if item.ref in by_ref:
            trace.append(f"uncertainty_expression -> duplicate ref {item.ref} ignored")
            continue
        by_ref[item.ref] = str(item.level)
        claims[item.ref] = str(item.claim or "")

    annotated: set[str] = set()
    unknown: list[str] = []
    seen: set[str] = set()
    marker_count = 0
    annotated_count = 0

    def _replace(match) -> str:
        nonlocal marker_count, annotated_count
        marker_count += 1
        ref = match.group(1)
        seen.add(ref)
        level = by_ref.get(ref)
        if level is None:
            unknown.append(ref)
            return ""
        if level not in levels:
            return ""
        annotated.add(ref)
        annotated_count += 1
        return certainty_annotation(level)

    rendered_text = CERTAINTY_MARKER_RE.sub(_replace, text)

    # The claim channel: only refs with no marker in the prose, since a marker
    # has already placed it and it must not be inserted a second time.
    claim_refs = [ref for ref in by_ref if claims[ref].strip() and ref not in seen]
    claim_trace: list[RenderStep] = []
    inserts: list[tuple[int, str]] = []
    strategy_by_ref: dict[str, str] = {}
    for ref in claim_refs:
        match = claim_end(rendered_text, claims[ref])
        strategy_by_ref[ref] = match.strategy if match is not None else "not_found"
        if match is None:
            claim_trace.append(f"certainty ref {ref}: claim not found")
            continue
        annotation = certainty_annotation(by_ref[ref]) if by_ref[ref] in levels else ""
        if not annotation:
            claim_trace.append(
                f"certainty ref {ref}: located ({match.strategy}), "
                f"withheld ({setting or 'unset'})"
            )
            continue
        claim_trace.append(f"certainty ref {ref}: located ({match.strategy}), annotated")
        inserts.append((match.end, annotation))
        annotated.add(ref)
    # Insert back to front, so an earlier insertion cannot shift an index that
    # was already computed for a later one.
    for position, annotation in sorted(inserts, key=lambda item: item[0], reverse=True):
        rendered_text = rendered_text[:position] + annotation + rendered_text[position:]

    # `rendered` alone cannot distinguish "found it and this setting withholds
    # the parenthesis" from "never found it" -- under `Hide` both are the empty
    # string. `claim` and `strategy` put that on the audit plane.
    rows = [
        CertaintyJSON(
            ref=item.ref,
            level=str(item.level),
            rendered=(
                certainty_annotation(by_ref[item.ref]) if item.ref in annotated else ""
            ),
            **(
                {"claim": str(item.claim)} if str(item.claim or "").strip() else {}
            ),
            **(
                {"strategy": strategy_by_ref[item.ref]}
                if item.ref in strategy_by_ref else {}
            ),
        )
        for item in certainties
    ]

    trace.append(
        f"uncertainty_expression={setting or 'unset'} -> "
        f"annotated {annotated_count}/{marker_count} markers"
    )
    if claim_refs:
        trace.append(
            f"uncertainty_expression={setting or 'unset'} -> "
            f"annotated {len(inserts)}/{len(claim_refs)} claims"
        )
    trace.extend(claim_trace)
    for ref in unknown:
        trace.append(f"uncertainty_expression -> unknown marker ref {ref} stripped")
    for ref in by_ref:
        if ref not in seen and not claims[ref].strip():
            trace.append(f"uncertainty_expression -> ref {ref} has no [[c:…]] marker in the text")

    return CertaintyRender(text=rendered_text, rows=rows, trace=trace)
