"""The IX renderer: a pure function from one PA turn to the one message the user sees.

No IO, no LLM call, no wall clock: every per-attribute rule is an `if` in `render()`.

What it does, in the order the parts reach the user:

    [process notices] [reasoning rationale] [prose] [gate ask | completion] [closing]

and what it never does: rewrite the model's wording, invent a sentence the model
did not write (a missing ask or closing line is a recorded PA failure, not
something the renderer papers over), or let an internal marker or a held body
reach the user.

Three inputs, one output: `render(IXRenderInput, settings) -> RenderedTurn`.
`settings` maps attribute -> chosen setting for this session; an attribute that
is absent or empty takes the **unset fallback** — no setting-specific transform,
markers still stripped, internal payload fields never released, no gate opened.
The fallback is reachable only for attributes this session never exposed: an
exposed attribute's `IX_<attribute>` tool is `required == "always"`, so a turn
that never calls it ends in `ix_contract_failure` rather than being rendered.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pa.agent.ix.rationale import render_certainties, render_rationale
from pa.agent.ix.render_types import (
    BATCH_MARKER_RE,
    CERTAINTY_MARKER_RE,
    HELD_STATUS_BY_ATTRIBUTE,
    HOLD_ROUNDS_HARD_CAP,
    IX_TOOL_NAME_PREFIX,
    PER_CALL_INJECTED_FIELDS,
    QUESTION_MARKER_RE,
    UNRECOGNISED_MARKER_RE,
    Batch,
    HeldItem,
    HeldNarration,
    HoldEpisode,
    HoldStatus,
    IXHold,
    IXRenderInput,
    PlanItem,
    ProactiveDecision,
    Question,
    RenderedTurn,
    RenderStep,
    ToolCallRecord,
    TurnPayloads,
)

# --- markers ---------------------------------------------------------------

#: A `[[batch:…]]` or `[[q:…]]` marker opens a section that owns the prose after
#: it until the next marker. Prose before the first marker carries no id and is
#: delivered every turn (§8 item 2, owner option A).
_SECTION_MARKER_RE = re.compile(
    f"(?P<batch>{BATCH_MARKER_RE.pattern})|(?P<question>{QUESTION_MARKER_RE.pattern})"
)

#: §3.7 (part 4), owner decision 2026-09-06 (§10 item d) · the forced close writes
#: **no** user-visible text. The renderer used to compose one sentence here when
#: the round cap cancelled an item and the PA wrote no `closing_message`, and the
#: line did more harm than the silence it covered: live `v11s02` t11 appended
#: "I have not done meeting_duration; it is on hold." underneath an answer that
#: had just been given, contradicting it and putting a snake_case intent id on
#: the public face. The force-close is now an audit fact only
#: (`ix_hold_forced_close` / `ix_close_no_message` / `cancelled_items` /
#: `close_line_source: none`); the PA's own `closing_message` still delivers.

#: §3.3 (part 2)·A · the second safety line, on the other fail-closed exit. A binding
#: refusal takes every task call in the turn and opens no gate, so the user is
#: owed a sentence saying the steps did not run — live `v7r06` t18 dropped three
#: reads in silence. Same justification as the cancelled line: the alternative
#: is not a worse sentence, it is a user who is told nothing at all.
#:
#: Round 9 item 2: **appended**, never substituted. It used to replace the turn
#: only when the message was otherwise empty, which read as a safe default and
#: was not one — live `v8s02` t3 declared four questions under `Structured`,
#: bound its task calls to the question ids, and the refusal shipped this line
#: *instead of* the four questions the setting exists to deliver. The refusal
#: and the questions are two different facts and the user gets both.
_BINDING_CLOSE_LINE = "I could not carry out those steps this turn; I will retry."

#: Round 9 item 1 · the universal empty-reply line (owner decision: an empty
#: reply is a miss, whatever emptied it). The runner has already spent one extra
#: provider request asking for a real sentence by the time this is reached; see
#: `IX_EMPTY_REPLY_PROMPT` there. Three distinct routes to an empty turn have
#: now been found in four rounds (a `binding_invalid` fail-closed exit, a hold
#: whose closing request came back silent, and a provider completion with no
#: text and no tool calls at all), so the fallback is written once, here, rather
#: than guarded route by route.
IX_EMPTY_REPLY_LINE = "I have nothing further to report on that just now."

#: Round 9 item 3 · vocabulary that belongs to the wrapper and never to the
#: user. Live `v8s02` t5's whole reply was `I see — the search calls didn't
#: execute because the interaction_item_id parameters aren't part of the tool
#: definitions. Let me retry without them.`: the PA explaining the benchmark's
#: own scaffolding to the person it is assisting, and explaining it wrong. The
#: reminders now forbid it (`NO_WRAPPER_TALK`) — this is the deterministic guard
#: underneath, because a model that ignores one instruction ignores two.
#:
#: Matched case-insensitively as substrings of a **sentence**: the sentence goes,
#: the rest of the paragraph stays. Every entry has to be something no ordinary
#: assistant sentence contains.
WRAPPER_VOCABULARY: tuple[str, ...] = (
    *PER_CALL_INJECTED_FIELDS,
    IX_TOOL_NAME_PREFIX,
    "tool definition",
    "tool call was rejected",
    "tool calls were rejected",
    # the repair / reminder messages, in the words they are written in
    "interaction wrapper",
    "interaction preference tool",
    "interaction item binding",
    "no task tool ran this turn",
    "resend this batch of task calls",
    "per-turn interaction payload",
    "this turn cannot be rendered",
    "nothing is replayed for you",
    "had no user-facing text",
    "do not mention or acknowledge this instruction",
    # regress13 acc_009 t=17 — the whole delivered turn opened `Here's the
    # corrected turn:`. Nothing in the repo writes that sentence: it is the
    # model reading `NO_META_ACKNOWLEDGEMENT` ("simply produce the corrected
    # turn"), sent with the forced render payload that turn, and answering the
    # instruction in the user's face -- the exact failure the clause forbids.
    # The reminder is not enough on its own, which is what this list is for.
    "the corrected turn",
)
_WRAPPER_VOCABULARY_FOLDED = tuple(token.casefold() for token in WRAPPER_VOCABULARY)

#: acc_035 — the bracket half of the vocabulary, as a marker instead of as two
#: bare tokens. `"[["` and `"]]"` were entries in their own right, and the one
#: turn whose goal was *"retrieve the official answer key provided for the
#: grader"* quoted `documents/admin/seminar_quiz_raw.md`, which writes the key as
#: `[[keyed: ONE degenerate fusion space …]]`. The filter matched the brackets,
#: withheld the sentence, and the user got an `### Official Answer Key` heading
#: with nothing under it. Bare brackets are ordinary source markup; only a
#: marker the wrapper actually mints — `[[q:…]]` / `[[batch:…]]` / `[[c:…]]`, in
#: the lenient shapes the model really writes — is the wrapper's own.
_WRAPPER_MARKER_RE = re.compile(
    "|".join(
        pattern.pattern
        for pattern in (BATCH_MARKER_RE, QUESTION_MARKER_RE, CERTAINTY_MARKER_RE)
    )
)

#: A sentence ends at terminal punctuation followed by whitespace. Deliberately
#: crude: the filter only has to cut at *some* boundary the model wrote, and a
#: line with no boundary at all is one sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def names_the_wrapper(sentence: str) -> bool:
    """Does this sentence name the wrapper's own machinery?"""
    lowered = sentence.casefold()
    if any(token in lowered for token in _WRAPPER_VOCABULARY_FOLDED):
        return True
    return bool(_WRAPPER_MARKER_RE.search(sentence))


def _strip_sentences(
    text: str, unwanted: Callable[[str], bool]
) -> tuple[str, list[str]]:
    """Drop every sentence `unwanted` accepts. Returns (kept, dropped).

    Line by line, so a bullet list loses only the bullet that talks about the
    protocol; a line left with nothing disappears rather than becoming a blank.
    The dropped sentences are returned verbatim for the audit plane — the
    withheld text is still a record of what the PA committed to.
    """
    if not text:
        return text, []
    kept_lines: list[str] = []
    dropped: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            kept_lines.append(line)
            continue
        sentences = _SENTENCE_SPLIT_RE.split(line)
        kept = [part for part in sentences if not unwanted(part)]
        dropped.extend(part.strip() for part in sentences if unwanted(part))
        if len(kept) == len(sentences):
            kept_lines.append(line)
        elif kept:
            kept_lines.append(" ".join(part.strip() for part in kept))
    if not dropped:
        return text, []
    return _BLANK_RUN_RE.sub("\n\n", "\n".join(kept_lines)).strip(), dropped


def strip_wrapper_vocabulary(text: str) -> tuple[str, list[str]]:
    """Drop every sentence that names the wrapper. Returns (kept, dropped)."""
    return _strip_sentences(text, names_the_wrapper)


#: acc_039 — the PA claiming the very action the gate is stopping. Two of the
#: three gate turns opened in the present progressive (*"Adding the checkup
#: appointment …"*, *"Adjusting the calendar block end time …"*) over calls that
#: were held, and the third put *"I'll proceed with booking now"* four lines
#: above its own *"Shall I go ahead?"*. Nothing executed on any of them. A closed
#: list, in the shapes the model actually wrote them.
_CLAIM_OF_ACTION_CUES: tuple[str, ...] = (
    "i'll proceed",
    "i will proceed",
    "i'm proceeding",
    "proceeding with",
    "i'll go ahead",
    "i will go ahead",
    "going ahead with",
    "i've booked",
    "i have booked",
    "i've created",
    "i have created",
    "i've added",
    "i have added",
    "i've updated",
    "i have updated",
    "i've scheduled",
    "i have scheduled",
    "i've sent",
    "i have sent",
    "has been booked",
    "has been created",
    "has been added",
    "has been updated",
    "has been scheduled",
    "has been sent",
)
#: The other live shape: a sentence that *opens* on the held act, in the present
#: progressive. Anchored, because the claim is the sentence's subject and not a
#: word inside it.
_CLAIM_OF_ACTION_OPENING_RE = re.compile(
    r"\A[\s*_>#-]*"
    r"(?:adding|adjusting|booking|creating|updating|scheduling|sending|filing)\b",
    re.IGNORECASE,
)
#: A modal turns the same opening into a proposal — *"Adding a second slot would
#: help"* is the sentence a gate turn is *for*, and it must survive.
_HYPOTHETICAL_RE = re.compile(r"\b(?:would|could|might|should|if)\b", re.IGNORECASE)


def claims_the_held_action(sentence: str) -> bool:
    """Does this sentence assert the action the gate is still asking about?"""
    lowered = sentence.casefold()
    if any(cue in lowered for cue in _CLAIM_OF_ACTION_CUES):
        return True
    return bool(_CLAIM_OF_ACTION_OPENING_RE.match(sentence)) and not _HYPOTHETICAL_RE.search(
        sentence
    )


def strip_claim_of_action(text: str) -> tuple[str, list[str]]:
    """Drop every sentence claiming the held action. Returns (kept, dropped)."""
    return _strip_sentences(text, claims_the_held_action)


#: The provider's own tool-call syntax, arriving as prose. acc_001 t32/t34: the
#: model typed `<｜DSML｜tool_calls>` / `<｜DSML｜invoke name="…">` blocks into its
#: message instead of calling the tool, and 7056 characters of unparsed
#: `document_update` parameters went to the user. t34's block was a write that
#: never ran and that t38 then reported as done, so this is not only noise --
#: the markup is the PA's evidence for a claim nothing backs. The full-width
#: `｜` (U+FF5C) is what the model actually emits; the ASCII pipe is matched too.
_DSML_OPEN = r"<[|｜]\s*DSML\s*[|｜]>?"
#: A closed block is cut whole, so prose after it survives. An unclosed one
#: takes the rest of the text: t34's was truncated mid-token, and what follows
#: an opening tag is more markup, never a sentence.
_DSML_BLOCK_RE = re.compile(
    rf"{_DSML_OPEN}\s*tool_calls>.*?</[|｜]\s*DSML\s*[|｜]>?\s*tool_calls>"
    rf"|{_DSML_OPEN}.*",
    re.DOTALL,
)


def strip_dsml_markup(text: str) -> tuple[str, list[str]]:
    """Take the model's raw tool-call syntax out of the prose. Returns (kept, dropped).

    Same contract as `strip_wrapper_vocabulary`: the dropped blocks come back
    verbatim for the audit plane, because an unparsed call is a record of what
    the PA meant to do and did not.
    """
    if not text or "DSML" not in text:
        return text, []
    dropped: list[str] = []

    def _cut(match: re.Match[str]) -> str:
        dropped.append(match.group(0))
        return ""

    kept = _DSML_BLOCK_RE.sub(_cut, text)
    if not dropped:
        return text, []
    return _BLANK_RUN_RE.sub("\n\n", kept).strip(), dropped


def reply_is_only_markup(text: str) -> bool:
    """Was the whole reply an unparsed tool call? (acc_001 t32)

    A turn with nothing but markup in it has said nothing to the user, and the
    stripper is about to leave it empty -- so it goes back for a real reply on
    the unfinished-turn path rather than being delivered as a completed turn.
    """
    _kept, dropped = strip_dsml_markup(text or "")
    return bool(dropped) and not clean_delivered_text(text or "").strip()


def _spaced(pattern: re.Pattern[str]) -> re.Pattern[str]:
    """The marker plus the blanks around it, so removing it leaves clean prose."""
    return re.compile(rf"[ \t]*{pattern.pattern}[ \t]*")


#: An emphasis pair the marker's removal emptied. The model writes
#: `**[[q1]]** What is …`; the section split takes the marker *and* everything
#: after it, so the opening `**` stays behind on the previous part and the
#: closing `**` opens the next one. Live (`v3g05` t3) the released ask arrived as
#: three stray `**` lines around one question. The pair only counts as emptied
#: when the gap it now spans is nothing at all (`****`) or crosses a line break
#: — that is the section-join signature, and it leaves `**bold** **also bold**`
#: alone.
_EMPTY_EMPHASIS_RE = re.compile(r"[ \t]*(\*\*|__)((?:[ \t]*\n\s*)?)\1[ \t]*")
#: A line left holding one emphasis delimiter and nothing else. At most two
#: characters, so a `***` / `___` horizontal rule is not a match.
_ORPHAN_EMPHASIS_LINE_RE = re.compile(r"^[ \t]*[*_]{1,2}[ \t]*$", re.MULTILINE)
#: A marker that owned its own line leaves the line blank behind it.
_BLANK_RUN_RE = re.compile(r"\n{3,}")
#: A line that is nothing but emphasis / rule characters. `***` and `___` are
#: horizontal rules, not broken emphasis, so the balancing pass leaves them be.
_RULE_LINE_RE = re.compile(r"^[ \t]*[*_-]{3,}[ \t]*$")


def _balance_emphasis_line(line: str) -> str:
    """Drop the delimiter whose partner the marker split took away.

    The two passes above remove the *opening* half — it is left stranded on the
    previous section, usually as a line of its own — and that is exactly what
    orphans the closing half, which is not on a line of its own and so survives
    both of them. Live (`v7r07` t3, `Structured`) the four released questions
    each shipped as `Who is Prof. Reyes?** Could you share …`: one `**` per
    line, four in the turn, so a whole-text parity check sees an even number and
    finds nothing. Parity is per line, which is also where markdown emphasis
    lives.
    """
    if _RULE_LINE_RE.match(line):
        return line
    for delimiter in ("**", "__"):
        if line.count(delimiter) % 2:
            cut = line.rfind(delimiter)
            line = line[:cut] + line[cut + len(delimiter):]
    return line


def _drop_orphan_emphasis(text: str) -> str:
    """Close the markdown a stripped marker left hanging open.

    Order matters: collapse the emptied pairs first, because the half that
    still carries the question's text is not an emphasis-only line and would
    survive the line pass with a dangling `**` in front of it. The balancing
    pass runs last, on what the first two left behind.
    """
    text = _EMPTY_EMPHASIS_RE.sub(lambda match: match.group(2), text)
    text = _ORPHAN_EMPHASIS_LINE_RE.sub("", text)
    text = "\n".join(_balance_emphasis_line(line) for line in text.split("\n"))
    return _BLANK_RUN_RE.sub("\n\n", text)


def clean_delivered_text(text: str) -> str:
    """The marker/emphasis cleanup every delivered surface owes (contract 6).

    Public because the renderer is not the only path a message can leave by:
    when `render()` produced no message the loop ships `final_content`, which
    never passed through `_strip_markers`. One function, so the two paths cannot
    disagree about what a user is allowed to see.

    Round 9 item 3 folds the wrapper-vocabulary filter in here for the same
    reason: the fallback path is exactly the one a meta-acknowledgement reply
    takes, because a turn that only explains the protocol has no released
    section for the renderer to build a message out of.

    The tool-call markup goes first (acc_001 t32/t34), before the marker and
    sentence passes: a `[[q:...]]` marker or a wrapper word inside an unparsed
    invoke is not prose, and half-cutting the block leaves the other half
    delivered.
    """
    kept, _dropped = strip_wrapper_vocabulary(
        _strip_markers(strip_dsml_markup(text)[0])
    )
    return kept.strip()


_BATCH = "batch"
_QUESTION = "question"

#: §3.3 (part 2)·C — the two attributes whose gate delivers a per-call ask verbatim.
#: Every other gate (a released-questions turn, a withheld batch) stops the turn
#: without owing the user a sentence, so a missing ask there is not a failure.
ASK_BEARING_ATTRIBUTES = frozenset({"autonomy_level", "capability_boundary"})

#: §3.3 (part 2)·B — the action types a release actually buys. A read costs nothing.
WORLD_CHANGING_ACTION_TYPES = frozenset({"internal_write", "external_action"})

#: Hold reasons that are *not* a cross-turn hold: a fail-closed binding batch is
#: repaired inside the same turn, and a spent release is a refused duplicate.
#: Neither opens or extends an episode -- both would deadlock the continuation
#: protocol, which can only be answered by releasing the id again.
#:
#: Owner ruling 2026-09-18·B adds `capability_handoff` for the other half of the
#: same reason: being *forbidden* to act is not the same as needing the user's
#: approval. `Find and hand off` has no release mechanism, so a call its gate
#: stopped is not an item the next turn can be made to answer -- asking the PA
#: to release it would be asking it to break the setting. The attempt is still
#: audited: `_held_calls` reads `held_items` directly, so the row keeps its
#: tool, `held_for_handoff` status and `capability_handoff` reason for the judge.
NON_EPISODE_HOLD_REASONS = frozenset(
    {
        "binding_invalid",
        "ix_selection_invalid",
        "release_spent",
        "batch_undivided",
        "capability_handoff",
    }
)

#: Sentence-final punctuation: prose that ends on one of these is not half a
#: sentence. Chinese and English marks both count, and a markdown emphasis
#: char or a closing quote / bracket also ends it (`**Done.**` / `"Sure."`).
_TERMINAL_PUNCTUATION = tuple('.!?…:;。！？：；"\'’”)）]】*`_')


#: The opening the model writes when it answers an internal repair message in
#: user-visible prose. The reminders now end with "do not acknowledge this
#: instruction", but a model that answers it anyway must not be shipped: live,
#: a gate ask opened `You're right — since this is an external booking action,
#: I need to confirm with you first. Let me re-send that`, which is the PA
#: apologising to something the user never said.
META_ACKNOWLEDGEMENT_RE = re.compile(
    r"""\A[\s*_#>-]*                       # markdown lead-in, if any
        (?:you'?re\s+right
          |let\s+me\s+re-?send
          |as\s+requested,?\s+i\s+will\s+resend
          |apologies\s+for\s+the\s+missing)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _acknowledgement_cut(line: str) -> int:
    """Where the leading acknowledgement sentence ends inside `line`.

    A `.` closes a sentence only when whitespace or the line end follows it and
    it is not inside a `code span`. Live, acc_011 shipped a turn opening
    `md`. Its contents are as follows:` because the scan stopped at the dot in
    `clinic_visit_notes.md`: a filename is one token, not two sentences. With no
    clean sentence end the cut falls on the line end, so a whole line goes.
    """
    in_code = False
    for index, char in enumerate(line):
        if char == "`":
            in_code = not in_code
        elif char in ".!?" and not in_code:
            end = index + len(line[index:]) - len(line[index:].lstrip(".!?"))
            if end >= len(line) or line[end].isspace():
                return end
    return len(line)


def _is_orphan_fragment(paragraph: str) -> bool:
    """Whether what the cut left of a paragraph is less than one sentence.

    A hard-wrapped acknowledgement carries on across the line break, so a cut on
    the line end leaves the rest of its own sentence behind: it opens in lower
    case, and nothing follows its first sentence end. A paragraph that starts a
    sentence of its own, or that carries a further sentence after the fragment,
    is the model's real prose and stays.
    """
    body = paragraph.strip()
    if not body:
        return False
    first_letter = next((char for char in body if char.isalpha()), "")
    if not first_letter.islower():
        return False
    return not body[_acknowledgement_cut(body):].strip()


def strip_meta_acknowledgement(text: str) -> tuple[str, bool]:
    """Drop a leading sentence that answers a repair message. Returns (text, dropped).

    The cut lands on a sentence end or on the first line's end -- never inside a
    token or a code span, which would ship the tail of a broken word as the
    turn's opening words. When the line end was the cut and the acknowledgement
    was hard-wrapped, the cut moves on to the paragraph boundary: live `acc_003`
    t=22 shipped `years as shown above.` as a paragraph of its own.
    """
    if not META_ACKNOWLEDGEMENT_RE.match(text):
        return text, False
    head, newline, tail = text.partition("\n")
    remainder = head[_acknowledgement_cut(head):].lstrip()
    stripped = f"{remainder}{newline}{tail}" if remainder else tail.lstrip()
    paragraph, separator, rest = stripped.partition("\n\n")
    if _is_orphan_fragment(paragraph):
        stripped = rest.lstrip() if separator else ""
    return stripped, True


@dataclass(frozen=True)
class _Section:
    """One stretch of the model's prose: unmarked framing, a batch, or a question."""

    kind: str  # "" | _BATCH | _QUESTION
    id: str
    text: str


# ---------------------------------------------------------------------------
# public helpers (P2 builds the §4.4 blob with these, so the rules live once)
# ---------------------------------------------------------------------------


def settings_from_selections(selections: Iterable) -> dict[str, str]:
    """`{attribute: setting}` from IX selections, latest selection per attribute wins."""
    settings: dict[str, str] = {}
    for selection in selections:
        attribute = getattr(selection, "attribute", "")
        if attribute:
            settings[attribute] = getattr(selection, "setting", "") or ""
    return settings


def released_item_ids(payloads: TurnPayloads, setting: str) -> set[str] | None:
    """Which ``interaction_item_id`` a ``topic_management`` turn may execute.

    ``None`` means "no batch restriction" (Organize, unset, or no batches).
    ``One-at-a-time`` releases only the first batch, so only calls bound to it
    run; the rest are held before execution, not after (§3.4 (part 3)).

    It lives here rather than next to the gate because the renderer needs the
    same answer to decide which held calls belong to the topic episode, and two
    copies of this rule would be two chances to disagree.
    """
    if not payloads.batches:
        return None
    if (setting or "").casefold() != "one-at-a-time":
        return None
    return set(payloads.batches[0].call_binding)


def released_ids(
    render_input: IXRenderInput, settings: Mapping[str, str] | None = None
) -> dict[str, list[str]]:
    """§4.4 ``released`` — what this turn actually let out, by id.

    Pure, and deliberately re-derived from the same inputs `render` used rather
    than threaded out of it: `RenderedTurn` has no slot for these audit rows.

    A batch counts as released when its prose section went out **or** when a
    task call bound to it ran — the model may bind calls to a batch without
    writing the `[[batch:…]]` marker, and the audit record has to say that the
    work went out either way. `plan_executed` is every plan item, from this
    turn's payload or from the open episode, that had a bound call execute.
    """
    if render_input.proactive_decision is not None:
        return {"batches": [], "questions": [], "plan_executed": []}
    chosen = {attr: (value or "").strip() for attr, value in (settings or {}).items()}
    sections = _split_sections(_final_segment(delivered_segments(
        render_input.segments, render_input.held_narration,
    )))
    released, _held_batches, _held_questions = _select_sections(sections, chosen, [])

    executed = {
        call.interaction_item_id
        for call in render_input.calls
        if call.executed and call.interaction_item_id
    }
    batch_ids: list[str] = []
    for batch in render_input.payloads.batches:
        released_here = (_BATCH, batch.batch_id) in released or bool(
            executed & set(batch.call_binding)
        )
        if batch.batch_id and released_here and batch.batch_id not in batch_ids:
            batch_ids.append(batch.batch_id)
    for kind, section_id in released:
        if kind == _BATCH and section_id and section_id not in batch_ids:
            batch_ids.append(section_id)

    question_ids = [
        question.id
        for question in render_input.payloads.questions
        if (_QUESTION, question.id) in released
    ]
    question_ids += sorted(
        section_id
        for kind, section_id in released
        if kind == _QUESTION and section_id and section_id not in question_ids
    )

    plan_items = [
        *render_input.payloads.plan,
        *(
            item
            for episode in render_input.hold.open_episodes.values()
            for item in episode.plan_remaining
        ),
    ]
    plan_executed: list[str] = []
    for item in plan_items:
        if item.id and item.id not in plan_executed and executed & set(item.bound_item_ids):
            plan_executed.append(item.id)

    return {
        "batches": batch_ids,
        "questions": question_ids,
        "plan_executed": plan_executed,
    }


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def render(
    render_input: IXRenderInput,
    settings: Mapping[str, str] | None = None,
    *,
    held_items: Sequence[HeldItem] = (),
) -> RenderedTurn:
    """Render one PA turn. Pure: same inputs, same `RenderedTurn`, no side effects.

    `held_items` are the gate's own hold records for this turn (`attribute`,
    `reason`, plan and batch ids). They are what tells the write-back which
    episode a held call belongs to: a call withheld because its batch was not
    released belongs to `topic_management`, not to whichever attribute happened
    to own the gate — and on a batch hold no gate is opened at all (§3.4 (part 3)).
    Callers that have no such records get the older gate-owned reading.
    """
    chosen = {attr: (value or "").strip() for attr, value in (settings or {}).items()}
    trace: list[RenderStep] = []

    if render_input.proactive_decision is not None:
        return _render_proactive(render_input, chosen, trace)

    payloads = render_input.payloads
    body, meta_dropped = strip_meta_acknowledgement(_final_segment(delivered_segments(
        render_input.segments, render_input.held_narration, trace
    )))
    if meta_dropped:
        trace.append("meta acknowledgement dropped")
    sections = _split_sections(body)

    released, held_batches, held_questions = _select_sections(sections, chosen, trace)
    # acc_009: the drop is silent because `_select_sections` returns before its
    # own trace line. Say it here, so the reviewer reading the page can see two
    # questions in the payload and a reply without them and know why.
    dropped_questions = unmarked_questions(render_input)
    if dropped_questions:
        trace.append(
            f"information_elicitation -> {dropped_questions} payload question(s) "
            "dropped, no [[q:...]] marker in the prose"
        )
    claims_not_found: list[str] = []
    leadin: list[str] = []
    prose = _prose(
        sections, released, chosen, payloads, trace, claims_not_found, leadin,
    )
    # A gate — or a fail-closed hold — ends the turn mid-sentence. The ask that
    # follows is the message; the half sentence in front of it is not.
    if prose and (
        render_input.gate is not None
        or any(not call.executed for call in render_input.calls)
    ):
        prose = _drop_trailing_fragment(prose, trace)
    # acc_039 — the turn ends on an ask, so nothing in front of it may say the
    # thing already happened or is happening. Withheld, not rewritten: what the
    # user is owed on a gate turn is the `_ix_confirmation_message`, which the PA
    # wrote, and the wrapper writes none of it.
    if prose and render_input.gate is not None:
        prose, action_claims = strip_claim_of_action(prose)
        if action_claims:
            trace.append(
                f"gate open -> claim of action withheld "
                f"({len(action_claims)} sentence(s))"
            )
    else:
        action_claims = []

    parts: list[str] = []
    parts.extend(_notices(render_input.calls, chosen.get("process_visibility", ""), trace))
    # §3.2 — P3's pure branch: `Show` releases the PA's own rationale, everything
    # else keeps it on the audit side. Provider reasoning never reaches it.
    rationale = render_rationale(
        chosen.get("reasoning_visibility", ""),
        payloads.user_facing_rationale,
        provider_reasoning_present=bool(render_input.provider_reasoning),
    )
    trace.extend(rationale.trace)
    if rationale.prefix:
        parts.append(rationale.prefix)
    if prose:
        parts.append(prose)

    gate_ask = _gate_ask(render_input, payloads, trace)
    completion_delivered = False
    completion_skipped = ""
    completion_line = ""
    if gate_ask is not None:
        parts.append(gate_ask)
        # §4.4: the ask is the end of the turn, so the closing line the process
        # setting would have released has nowhere to go. Recorded, because the
        # audit plane must say what reached the user — not what the setting
        # would have allowed.
        completion_skipped = "gate"
        trace.append("gate -> completion line skipped")
    else:
        completion = _completion(
            chosen.get("process_visibility", ""), payloads, render_input.calls, trace,
        )
        # §3.1 process row: the completion line is a field, but the model often
        # writes the same sentence as its own closing prose *and* puts it in
        # `completion_message`. Appending it then ships it twice — live `v7r01`
        # t13 and t22 under `Full narration`. Released either way; the field is
        # what the audit plane records, and the user reads it once.
        if completion and _already_ends_with(parts, completion):
            trace.append("process_visibility -> completion already in prose, not repeated")
            completion_delivered = True
            completion_line = completion
            completion = ""
        if completion:
            parts.append(completion)
            completion_delivered = True
            completion_line = completion

    asked_ids = _asked_item_ids(render_input, gate_ask, trace)
    hold_next, cancelled_now, folded_now = _next_hold(
        render_input, chosen, released, held_batches, held_questions, held_items,
        asked_ids, gate_ask, trace,
    )
    close_line_source = ""
    if cancelled_now:
        closing = payloads.closing_message.strip()
        if closing:
            parts.append(closing)
            close_line_source = "payload"
            trace.append("hold -> forced close, closing_message delivered")
        else:
            # §3.7 (part 4), owner decision 2026-09-06: the renderer writes nothing.
            # `ix_close_no_message` is still the recorded PA failure and
            # `cancelled_items` still carries the dropped ids to the judge on
            # every later beat -- what goes away is the sentence, which the user
            # read as the PA contradicting itself in the PA's own voice.
            close_line_source = "none"
            trace.append(
                "hold -> forced close, closing_message missing "
                "(ix_close_no_message), no close line delivered"
            )

    message = "\n\n".join(part for part in parts if part.strip())
    # Round 9 item 3 — the last thing every delivered surface passes through.
    # After the sections are chosen, so a sentence inside a released question is
    # judged on its own and the rest of that question survives.
    # acc_001 t32/t34 — before the sentence filter, for the reason
    # `clean_delivered_text` takes it first: an unparsed call is not prose and
    # must not be judged a sentence at a time.
    message, dsml_blocks = strip_dsml_markup(message)
    message, wrapper_talk = strip_wrapper_vocabulary(message)
    withheld_narration = [
        *(
            HeldNarration(item_id="claim_of_action", text=sentence)
            for sentence in action_claims
        ),
        *(
            HeldNarration(item_id="dsml_markup", text=block)
            for block in dsml_blocks
        ),
        *(
            HeldNarration(item_id="wrapper_vocabulary", text=sentence)
            for sentence in wrapper_talk
        ),
    ]
    if dsml_blocks:
        trace.append(f"tool-call markup stripped ({len(dsml_blocks)} block(s))")
    if wrapper_talk:
        trace.append(f"wrapper vocabulary withheld ({len(wrapper_talk)} sentence(s))")
    # The completion flag was set while the parts were being assembled, before
    # the filter above could take the line back out again. `completion.released`
    # has to say what the reply carried, so it is re-read off the finished
    # message: a closing line that named the wrapper never reached the user.
    if completion_delivered and not _still_present(message, completion_line):
        completion_delivered = False
        completion_skipped = "stripped"
        trace.append("completion line stripped by the wrapper filter -> not delivered")
    # §3.3 (part 2)·A: the binding refusal took every task call and opened no gate, so
    # the user is owed the sentence saying so. Appended, not substituted (round 9
    # item 2): on a `questions_gate` turn the released questions are the reply
    # and this line is a postscript to them.
    if render_input.binding_refused:
        if _prose_admits_held_step(message):
            trace.append(
                "binding_invalid -> prose already names the failed step, "
                "template close line suppressed"
            )
        else:
            message = (
                f"{message.rstrip()}\n\n{_BINDING_CLOSE_LINE}"
                if message.strip() else _BINDING_CLOSE_LINE
            )
            close_line_source = "template"
            trace.append("binding_invalid -> template close line appended")
    # Round 9 item 1: the runner has already asked once for a real sentence and
    # got nothing back. One rule, one line, whatever emptied the turn.
    if not message.strip() and render_input.empty_reply:
        message = IX_EMPTY_REPLY_LINE
        close_line_source = "template"
        trace.append("empty turn after retry -> template close line")

    return RenderedTurn(
        message=message,
        withheld_narration=withheld_narration,
        held=_held_calls(
            render_input, held_batches, held_questions, hold_next, asked_ids,
            held_items,
        ),
        hold_next=hold_next,
        proactive=None,
        executed_call_ids=[call.tool_call_id for call in render_input.calls if call.executed],
        held_call_ids=[call.tool_call_id for call in render_input.calls if not call.executed],
        trace=trace,
        cancelled_now=cancelled_now,
        release_folded=folded_now,
        close_line_source=close_line_source,
        completion_delivered=completion_delivered,
        completion_skipped=completion_skipped,
        claims_not_found=claims_not_found,
        leadin_dropped=leadin[0] if leadin else "",
        prose_held=bool(held_batches or held_questions),
    )


# --- proactive --------------------------------------------------------------


def _render_proactive(
    render_input: IXRenderInput,
    settings: Mapping[str, str],
    trace: list[RenderStep],
) -> RenderedTurn:
    """§3.1 proactive row, contract 11 — the decision the rest of the turn is built on.

    `High` delivers, `Low` suppresses, and the unset fallback passes the model's
    own action through. The candidate message always exists; on suppress it goes
    to the audit side only and `public_turn` is the empty string.
    """
    decision = render_input.proactive_decision
    assert decision is not None
    setting = (settings.get("proactive_outreach") or decision.setting or "").strip()
    action = decision.action
    if setting == "High":
        action = "send"
    elif setting == "Low":
        action = "suppress"
    if action != decision.action:
        trace.append(f"proactive_outreach={setting or 'unset'} -> {decision.action} overridden")
    trace.append(f"proactive_outreach={setting or 'unset'} -> {action}")
    return RenderedTurn(
        message=decision.message if action == "send" else "",
        held=[],
        hold_next=render_input.hold,
        proactive=ProactiveDecision(
            action=action, setting=setting or decision.setting, message=decision.message
        ),
        executed_call_ids=[call.tool_call_id for call in render_input.calls if call.executed],
        held_call_ids=[call.tool_call_id for call in render_input.calls if not call.executed],
        trace=trace,
    )


# --- prose ------------------------------------------------------------------


def _final_segment(segments: Sequence[str]) -> str:
    """The answer prose is the last iteration's text; earlier iterations are internal."""
    for segment in reversed(list(segments)):
        if segment and segment.strip():
            return segment
    return ""


def _marked_only(text: str) -> str:
    """Everything from the first `[[batch:…]]` / `[[q:…]]` marker on, or nothing."""
    match = _SECTION_MARKER_RE.search(text)
    return text[match.start():] if match is not None else ""


def delivered_segments(
    segments: Sequence[str],
    held_narration: Sequence[HeldNarration] = (),
    trace: list[RenderStep] | None = None,
) -> list[str]:
    """`segments` with every held-narration segment cut back to its markers.

    A segment written in the same assistant message as a call the gate stopped
    narrates work that did not happen (§3.3 (part 2)·D addendum): live, `📧 Email
    confirmation sent to Dr. Okafor` alongside a held send. That free prose is
    audit-only. What survives the cut is what the renderer selects by marker
    anyway — the questions and the released batches — because those are the ask,
    not the report; and a segment left with nothing falls through to the last
    iteration that still has prose, whose calls did run.
    """
    withheld = {
        narration.text for narration in held_narration
        if narration.text and narration.text.strip()
    }
    if not withheld:
        return list(segments)
    if trace is not None:
        for narration in held_narration:
            if narration.text and narration.text.strip():
                trace.append(f"held narration withheld ({narration.item_id})")
    return [
        _marked_only(segment) if segment in withheld else segment
        for segment in segments
    ]


def _split_sections(body: str) -> list[_Section]:
    """Cut the prose at every `[[batch:…]]` / `[[q:…]]` marker.

    A marker owns everything after it until the next marker. Text before the
    first marker is unmarked framing prose and is released every turn.
    """
    sections: list[_Section] = []
    position = 0

    def _close(chunk: str) -> None:
        """The prose since the last marker belongs to that marker's section."""
        if sections and sections[-1].kind:
            sections[-1] = replace(sections[-1], text=chunk)
        elif chunk.strip():
            sections.append(_Section("", "", chunk))

    for match in _SECTION_MARKER_RE.finditer(body):
        _close(body[position : match.start()])
        kind = _BATCH if match.group("batch") else _QUESTION
        pattern = BATCH_MARKER_RE if kind == _BATCH else QUESTION_MARKER_RE
        marker = pattern.search(match.group(0))
        assert marker is not None
        sections.append(_Section(kind, marker.group(1), ""))
        position = match.end()
    _close(body[position:])
    if not sections:
        sections.append(_Section("", "", body))
    return sections


def unmarked_questions(render_input: IXRenderInput) -> int:
    """Payload questions this turn declared and marked nowhere in its prose.

    Live `acc_009` t=8: the PA selected `Structured`, wrote both questions into
    the `IX_render_information_elicitation` payload, and wrote no `[[q:…]]`
    marker at all. `_select_sections` selects question sections out of the
    *prose*, so with nothing marked it takes the `if not marked: continue`
    branch — before its own trace line. Nothing was released, nothing was held,
    no trace line, no event, and the user read a reference to "those two points"
    with the two points missing. Everything the rest of that session did wrong
    descends from it.

    The payload is never the source of the released text, so this is a count of
    what was lost, not something the renderer can recover on its own.
    """
    if not any(question.id for question in render_input.payloads.questions):
        return 0
    body = _final_segment(
        delivered_segments(render_input.segments, render_input.held_narration)
    )
    if QUESTION_MARKER_RE.search(body):
        return 0
    return sum(1 for question in render_input.payloads.questions if question.id)


def _select_sections(
    sections: Sequence[_Section],
    settings: Mapping[str, str],
    trace: list[RenderStep],
) -> tuple[set[tuple[str, str]], list[_Section], list[_Section]]:
    """Which marked sections go out this turn (§3.4). Returns (released, held batches, held questions)."""
    released: set[tuple[str, str]] = set()
    held_batches: list[_Section] = []
    held_questions: list[_Section] = []

    for kind, attribute, held in (
        (_BATCH, "topic_management", held_batches),
        (_QUESTION, "information_elicitation", held_questions),
    ):
        marked = [section for section in sections if section.kind == kind]
        if not marked:
            continue
        setting = settings.get(attribute, "")
        keep = _release_count(attribute, setting, len(marked))
        for index, section in enumerate(marked):
            if index < keep:
                released.add((kind, section.id))
            else:
                held.append(section)
        label = setting or "unset"
        trace.append(f"{attribute}={label} -> release {keep}/{len(marked)}")
    return released, held_batches, held_questions


def _release_count(attribute: str, setting: str, total: int) -> int:
    """How many marked sections of this attribute the setting releases this turn."""
    if not setting:
        # Unset fallback: nothing is withheld, no gate is opened.
        return total
    if attribute == "topic_management":
        return 1 if setting == "One-at-a-time" else total
    if setting == "Infer":
        return 0
    if setting == "Iterative":
        return 1
    return total


#: regress13 acc_009 — the second shape of a lead-in, for the sentence that
#: names its object without a colon. Deliberately the one form that session
#: produced ("I need two details from you"), not a general theory of lead-ins:
#: the framing sentence is the PA's own prose and a wider rule would start
#: eating turns whose framing says something on its own.
_LEADIN_OBJECT_RE = re.compile(
    r"\b(?:details?|questions?)\s+from\s+you\b", re.IGNORECASE
)
#: Where the last sentence of a paragraph starts. Same crudeness as
#: `_SENTENCE_SPLIT_RE` and for the same reason: the cut only has to land on a
#: boundary the model itself wrote.
_LAST_SENTENCE_RE = re.compile(r"(?<=[.!?。！？…])\s+|\n+")
#: A colon inside a bold run still points at the block that has gone:
#: `**I need two details from you:**` is the same sentence.
_LEADIN_TRAILING_CHARS = "*_`~ \t"
#: The list bullet an excised section left behind, at the very end of the text.
#: `_split_sections` cuts at the marker and the model writes the marker *inside*
#: its list item, so the item's own `- ` stays on whatever came before it. A
#: bullet with nothing after it introduces nothing, whichever section it ended
#: up on. Anchored at the end, so the bullets of the items still being delivered
#: are untouched; `---` is a horizontal rule and not a match.
_ORPHAN_LIST_MARKER_RE = re.compile(r"\n[ \t]*(?:[-*+•]|\d+[.)])[ \t]*\Z")


def _drop_orphan_list_marker(text: str) -> str:
    """`text` without the trailing list bullet a held section left behind."""
    body = text.rstrip()
    match = _ORPHAN_LIST_MARKER_RE.search(body)
    return body[: match.start()].rstrip() if match else body


def _drop_leadin(text: str, trace: list[RenderStep]) -> tuple[str, str]:
    """Cut a lead-in sentence off framing prose whose object has just been held.

    regress13 acc_009: under `Infer` the release is 0/N, so the questions come
    out and the sentence that introduced them ships alone. Three turns of that
    session reached the user as *"Happy to hear the draft structure is
    acceptable! To finalize it, I need two details from you:"* and nothing else
    — a colon with nothing after it, twice, and a third turn whose "here's what
    I've incorporated" pointed at an incorporation the user could not see. The
    PA had written a real question; the wrapper's own excision is what made the
    turn unreadable, so the wrapper is what has to clean up after it.

    Two shapes only, and only ever the **last** sentence of the framing: a
    sentence that ends on a colon (the block it was introducing is gone, so the
    colon points at nothing), and `_LEADIN_OBJECT_RE`'s object-naming form.
    Everything before it is prose that stands by itself and stays. Returns
    (kept, dropped).

    regress15 acc_009 t7: the excised item's own bullet stayed on the framing,
    so the "last sentence" this reads was the bare `-` and neither shape
    matched. The turn shipped *"Before I write the draft, I just need a few
    clarifications from you:"* and a lone `-`. The bullet comes off first, so
    the sentence the rule is written about is the one it sees.
    """
    body = _drop_orphan_list_marker(text)
    if body != text.rstrip():
        trace.append("held block -> orphan list marker dropped")
    if not body:
        return body, ""
    breaks = list(_LAST_SENTENCE_RE.finditer(body))
    start = breaks[-1].end() if breaks else 0
    sentence = body[start:].strip()
    if not sentence:
        return body, ""
    if not (
        sentence.rstrip(_LEADIN_TRAILING_CHARS).endswith(":")
        or _LEADIN_OBJECT_RE.search(sentence)
    ):
        return body, ""
    trace.append("held block -> lead-in sentence dropped")
    return body[:start].rstrip(), sentence


def _prose(
    sections: Sequence[_Section],
    released: set[tuple[str, str]],
    settings: Mapping[str, str],
    payloads: TurnPayloads,
    trace: list[RenderStep],
    not_found: list[str],
    leadin: list[str],
) -> str:
    """Assemble the delivered prose: framing plus released sections, markers resolved.

    `not_found` is filled the way `trace` is: the refs whose claim the locator
    could not place, so the loop can record them (acc_035). `leadin` is filled
    the same way with the framing sentence a held block left dangling (acc_009).
    """
    # `_split_sections` gives the unmarked framing one section, always the
    # first, so "the sentence immediately before the first excised block" is
    # only ever the tail of `sections[0]` — and only when the very next section
    # is the one that went. Under `Iterative` the first question is released and
    # the lead-in still has something to introduce, which is why this reads the
    # section that follows the framing rather than "any block was held".
    # Questions only: `_release_count` never returns 0 for `topic_management`,
    # so a batch immediately after the framing is released by construction and
    # a batch branch here would be a rule with no turn to apply it to.
    if (
        len(sections) > 1
        and sections[0].kind == ""
        and sections[1].kind == _QUESTION
        and (sections[1].kind, sections[1].id) not in released
    ):
        kept, dropped = _drop_leadin(sections[0].text, trace)
        if dropped:
            leadin.append(dropped)
        # The bullet comes off whether or not a lead-in sentence went with it.
        if kept != sections[0].text:
            sections = [replace(sections[0], text=kept), *sections[1:]]
    parts = [
        section.text.strip()
        for section in sections
        if section.kind == "" or (section.kind, section.id) in released
    ]
    parts = [part for part in parts if part]
    # §3.1 uncertainty row — P3's pure branch turns each released `[[c:…]]` into a
    # parenthetical or into nothing; the hedging is the renderer's, never the prose's.
    certainties = render_certainties(
        settings.get("uncertainty_expression", ""),
        "\n\n".join(parts),
        payloads.certainties,
    )
    trace.extend(certainties.trace)
    not_found.extend(
        str(row["ref"]) for row in certainties.rows
        if row.get("strategy") == "not_found"
    )
    prose = _strip_markers(certainties.text, trace).strip()
    # The last delivered section carries the bullet of the section behind it, so
    # `Iterative` ends on an orphan the same way `Infer` does — one released
    # question and a `-` pointing at the two that were held. Only the very end
    # of the finished prose: the bullets between delivered items still have
    # their items.
    trimmed = _drop_orphan_list_marker(prose)
    if trimmed != prose:
        trace.append("held block -> orphan list marker dropped")
    return trimmed


def _substitute(pattern: re.Pattern[str], text: str, replace_id: Callable[[str], str]) -> str:
    """Replace `pattern` (and the blanks hugging it) with whatever `replace_id` returns."""

    def _one(match: re.Match[str]) -> str:
        inner = pattern.search(match.group(0))
        assert inner is not None
        rendered = replace_id(inner.group(1))
        whole = match.group(0)
        if rendered:
            lead = whole[: whole.index("[[")]
            trail = whole[whole.index("]]") + 2 :]
            return f"{lead}{rendered}{trail}"
        # Removing a marker between two words must not glue them together.
        return " " if whole.startswith((" ", "\t")) and whole.endswith((" ", "\t")) else ""

    return _spaced(pattern).sub(_one, text)


def _strip_markers(text: str, trace: list[RenderStep] | None = None) -> str:
    """All three marker kinds come out, whatever the setting, unset included.

    The certainty pass has normally consumed `[[c:…]]` already; it stays in this
    list so no marker can survive a branch that skipped that pass.

    Anything still wearing double brackets afterwards is a marker shape nobody
    recognises. It carries no id the wrapper can use, and it must not reach the
    user either, so it comes out too and says so in the trace.
    """
    for pattern in (BATCH_MARKER_RE, QUESTION_MARKER_RE, CERTAINTY_MARKER_RE):
        text = _substitute(pattern, text, lambda _ref: "")
    if trace is not None:
        for leftover in UNRECOGNISED_MARKER_RE.finditer(text):
            trace.append(f"unrecognised marker stripped: {leftover.group(0)}")
    text = _substitute(UNRECOGNISED_MARKER_RE, text, lambda _ref: "")
    return _drop_orphan_emphasis(text)


def _drop_trailing_fragment(text: str, trace: list[RenderStep]) -> str:
    """Cut a half-written sentence off the end of a turn that stopped early.

    A gate (or a fail-closed hold) ends the turn where the model was still
    typing, so the prose it had produced up to that point ends mid-sentence:
    live, a whole `public_turn` read `Appointment confirmed! Now let me add it
    to your`. The completed sentences before it are true and stay; only the
    dangling tail goes, and only when the last line has a sentence boundary to
    cut back to — a trailing line that is one unpunctuated sentence is the
    model's own phrasing, not a truncation, and is left alone.
    """
    body = text.rstrip()
    if not body or body.endswith(_TERMINAL_PUNCTUATION):
        return text
    head, _, last_line = body.rpartition("\n")
    cut = max(last_line.rfind(mark) for mark in ".!?。！？…")
    if cut < 0:
        return text
    kept = f"{head}\n{last_line[: cut + 1]}" if head else last_line[: cut + 1]
    trace.append("trailing fragment dropped")
    return kept.rstrip()


def split_truncated_tail(text: str) -> tuple[str, str]:
    """Split prose the provider cut off into the finished part and the tail.

    `_drop_trailing_fragment` is the gate's version of this and is deliberately
    forgiving: an unpunctuated last line is left alone, because the model may
    simply have written it that way. A completion the provider ended with
    `finish_reason == "length"` gets no such benefit of the doubt — it stopped
    mid-token, so whatever follows the last sentence end was never finished.
    A text with no sentence end anywhere is all tail, and the caller then has
    nothing to deliver (regress16 acc_006 t=4: `Let me start by looking at the
    research folder to find the citation notes` shipped as the whole PA turn).
    """
    body = text.rstrip()
    if not body or body.endswith(_TERMINAL_PUNCTUATION):
        return text, ""
    cut = max(body.rfind(mark) for mark in ".!?。！？…")
    if cut < 0:
        return "", text
    return body[: cut + 1], body[cut + 1:]


# --- per-attribute pieces ---------------------------------------------------


def _notices(
    calls: Sequence[ToolCallRecord], setting: str, trace: list[RenderStep]
) -> list[str]:
    """§3.1 process row — Silent none, Bookend the first, Full narration all of them."""
    available = [call.notice.strip() for call in calls if call.executed and call.notice.strip()]
    if setting == "Bookend":
        released = available[:1]
    elif setting == "Full narration":
        released = list(available)
    else:
        released = []
    trace.append(
        f"process_visibility={setting or 'unset'} -> notices {len(released)}/{len(available)}"
    )
    return released


def _normalized_line(text: str) -> str:
    """One sentence, comparable across the two places the model typed it.

    Whitespace collapses because the field and the prose are the same sentence
    typed twice and differ only in wrapping. Backticks go for the same reason:
    live `acc_003` t=13 wrote the citation key bare in its closing prose and in
    a code span in `completion_message`, and the exact-string comparison let the
    sentence ship twice.
    """
    return " ".join(text.replace("`", "").split())


def _already_ends_with(parts: Sequence[str], line: str) -> bool:
    """Whether the prose assembled so far already closes with `line`."""
    tail = _normalized_line("\n\n".join(parts))
    return bool(tail) and tail.endswith(_normalized_line(line))


def prose_within_completion(prose: str, completion: str) -> bool:
    """Is the whole reply just its own closing line? (acc_003 t=20)

    Asked for the exact updated metadata fields, the PA replied `Summary of the
    two updated entries is complete.` — 47 characters, no summary, no tool call
    — and the wrapper released the completion on top of it. A closing line
    reports work; it is not the work, and a reply that is nothing but the
    closing line has not answered anything.

    Normalised the same way the de-duplication is: the model types the same
    sentence into the field and into its prose with different wrapping and code
    spans, and the two have to compare equal.
    """
    return bool(prose.strip()) and _normalized_line(prose) in _normalized_line(completion)


def _still_present(message: str, line: str) -> bool:
    """Whether `line` survived into the finished message, normalised.

    Normalised for the same reason `_already_ends_with` is: the line may have
    been delivered as the prose tail, where the model's own wrapping and code
    spans are not the field's. A line the wrapper filter cut in half does not
    count as present — what was withheld is exactly the sentence the flag was
    claiming.
    """
    return _normalized_line(line) in _normalized_line(message)


#: The PA's own ways of saying what `_BINDING_CLOSE_LINE` says. Every entry is
#: about a step of *this* turn not having been carried out — "could not find" and
#: friends are deliberately absent, because a failed lookup the PA reports is not
#: the refusal the close line covers.
_INCOMPLETE_STEP_CUES: tuple[str, ...] = (
    "didn't complete",
    "did not complete",
    "wasn't completed",
    "was not completed",
    "didn't finish",
    "did not finish",
    "didn't go through",
    "did not go through",
    "couldn't carry out",
    "could not carry out",
    "wasn't able to",
    "was not able to",
    "still outstanding",
    "i'll retry",
    "i will retry",
)


def _prose_admits_held_step(message: str) -> bool:
    """Whether the delivered prose already tells the user a step did not run.

    Live `acc_039` t=22 closed with *"One thing that didn't complete: updating
    your `clinic_visit_notes.md`… I'll take care of that at the start of the next
    turn."* and then the wrapper appended *"I could not carry out those steps
    this turn; I will retry."* underneath it — the same admission twice, in
    adjacent sentences. The close line exists so the user is never left with
    silence about a refused batch (live `v7r06` t18 dropped three reads without a
    word); where the PA has already said it, the line adds nothing and costs the
    reply its composure.

    Matched on the normalised message for the same reason the completion line is:
    the sentence the model wrote wraps and code-spans differently from any
    sentence the wrapper holds.
    """
    folded = _normalized_line(message).casefold()
    return any(cue in folded for cue in _INCOMPLETE_STEP_CUES)


def _completion(
    setting: str,
    payloads: TurnPayloads,
    calls: Sequence[ToolCallRecord],
    trace: list[RenderStep],
) -> str:
    """§3.1 process row — the closing line is a field, never the last paragraph.

    Released only on a turn that executed a task call. The closing line reports
    work the turn did; a turn with no task call has none to report, and under
    Bookend and Full narration the line then carries no signal at all (both
    settings would show the same sentence). Live `regress5 acc_003` shipped it
    on three text-only turns, each time as a near-duplicate of the PA's own
    closing prose, and once — `regress3 acc_003` t=20 — as the wrapper's
    endorsement of a completion the PA had not done. Owner decision 2026-09-07:
    no task call, no closing line.
    """
    text = payloads.completion_message.strip()
    if setting not in ("Bookend", "Full narration") or not text:
        trace.append(f"process_visibility={setting or 'unset'} -> completion held")
        return ""
    if not any(
        call.executed and not call.tool_name.startswith(IX_TOOL_NAME_PREFIX)
        for call in calls
    ):
        trace.append(
            f"process_visibility={setting} -> completion held (no task call this turn)"
        )
        return ""
    trace.append(f"process_visibility={setting} -> completion released")
    return text


def _gate_ask(
    render_input: IXRenderInput, payloads: TurnPayloads, trace: list[RenderStep]
) -> str | None:
    """§3.3 (part 2)·C/D and §3.5 — deliver the PA's own ask, verbatim, or nothing.

    Returns `None` when no gate fired, `""` when one fired without the message the
    PA was supposed to write (recorded, never invented).
    """
    gate = render_input.gate
    if gate is None:
        return None
    if gate.attribute not in ASK_BEARING_ATTRIBUTES:
        # A released-questions turn already carries its own message: the
        # questions. Demanding an ask here would report every such turn as a
        # PA failure and would print a second, unrelated sentence.
        trace.append(f"{gate.attribute} gate {gate.item_id} -> no ask required")
        return ""
    if gate.attribute == "capability_boundary":
        boundary = payloads.boundary
        ask = boundary.handoff_message.strip() if boundary is not None else ""
        source = "handoff_message"
    else:
        ask = triggering_ask(
            render_input.calls, render_input.payloads.plan, gate.item_id
        )
        source = gate.message_source
    if not ask:
        trace.append(f"{gate.attribute} gate {gate.item_id} -> no ask written (ix_gate_no_message)")
        return ""
    trace.append(f"{gate.attribute} gate {gate.item_id} -> ask delivered ({source})")
    return ask


#: What makes a `_ix_confirmation_message` an *ask*: it puts the decision to the
#: user, either as a question or as an explicit request for their word. Live
#: regress19 `acc_039` t41, the floor held the booking and the PA's ask was
#: "I'll book the November 10, 08:30 appointment with Dr. Patel …" — the whole
#: delivered turn announced an act that was in fact blocked, while the trace
#: read "ask delivered". Non-emptiness was the only test, so a declarative
#: sentence passed it. The alternatives below stay deliberately broad: a gate
#: that finds no ask twice fails the turn closed, so the cost of refusing a real
#: ask is higher than the cost of accepting a marginal one, and what has to be
#: caught is the announcement, which invites nothing.
ASK_SURFACE_RE = re.compile(
    r"\?"
    r"|\b(?:"
    r"let me know|tell me|say the word|give (?:me )?the go[- ]ahead"
    r"|(?:please |just |simply )?confirm\b"
    r"|if you(?:'d| would) (?:like|prefer|rather)|if you want me to"
    r"|shall i|should i|do you want|would you like|want me to|ok(?:ay)? to"
    r")",
    re.IGNORECASE,
)


def asks_the_user(text: str) -> bool:
    """§3.3 (part 2)·C — does this text ask, or does it only announce?"""
    return bool(ASK_SURFACE_RE.search(text))


def triggering_ask(
    calls: Sequence[ToolCallRecord],
    plan: Sequence[PlanItem],
    item_id: str,
) -> str:
    """The gate's ask: the first held call for `item_id` that actually wrote one.

    Held calls with no `_ix_confirmation_message` — or with one that asks the
    user nothing (`ASK_SURFACE_RE`) — are skipped rather than answered with
    `""`. Live `v9s03` t24, one item held two calls — a `binding_invalid`
    `document_read` with no ask ahead of the `document_update` that carried the
    real one — and taking the first id match reported the ask as missing while
    the runner, which required a non-empty one, concluded it existed. Neither
    the repair nor the ask fired, and every held call shipped `asked: false`, so
    the user's next `allow` would have authorised nothing. Both sides read this
    one function now, so they cannot disagree.
    """
    held = [
        call for call in calls
        if not call.executed and asks_the_user(call.ask_text.strip())
    ]
    for call in held:
        if call.interaction_item_id and call.interaction_item_id == item_id:
            return call.ask_text.strip()
    for plan_item in plan:
        if plan_item.id == item_id:
            for call in held:
                if call.interaction_item_id in plan_item.bound_item_ids:
                    return call.ask_text.strip()
    return held[0].ask_text.strip() if held else ""


# --- held content and the next hold ----------------------------------------


def _held_status(
    render_input: IXRenderInput, held_batches: Sequence[_Section], held_questions: Sequence[_Section]
) -> HoldStatus:
    if render_input.gate is not None:
        status = HELD_STATUS_BY_ATTRIBUTE.get(render_input.gate.attribute)
        if status is not None:
            return status
    if held_questions:
        return "held_for_questions"
    if held_batches:
        return "held_for_batch"
    return "held_for_confirmation"


def _asked_item_ids(
    render_input: IXRenderInput, gate_ask: str | None, trace: list[RenderStep]
) -> set[str]:
    """The held ids the ask this turn actually put to the user (§3.7 (part 3)).

    §3.7 (part 3)'s criterion is "every held id shown to the user", and what is shown
    is one ask: the gate's own item, plus any other held id the PA named in the
    same `_ix_confirmation_message`. Live, an ask that mentioned only a
    cancellation came back with an `allow` for six ids, five of them searches
    the user was never asked about. Ids outside this set stay held until they
    get an ask of their own.
    """
    gate = render_input.gate
    if gate is None or not gate_ask:
        return set()
    asked = {gate.item_id} if gate.item_id else set()
    asked |= {
        call.interaction_item_id
        for call in render_input.calls
        if not call.executed
        and call.interaction_item_id
        and call.interaction_item_id in gate_ask
    }
    if asked:
        trace.append(f"ask named {len(asked)} held id(s): {', '.join(sorted(asked))}")
    return asked


def _hold_record(
    unmatched: list[HeldItem], call: ToolCallRecord
) -> HeldItem | None:
    """The gate's own record for this call, consumed so two calls never share one.

    The collector appends one record per held call in call order, so matching
    on `(interaction_item_id, tool)` and popping the first hit keeps the two
    lists aligned even when one message holds the same tool twice.
    """
    for index, record in enumerate(unmatched):
        if (
            record.interaction_item_id == call.interaction_item_id
            and record.tool == call.tool_name
        ):
            return unmatched.pop(index)
    return None


def _held_calls(
    render_input: IXRenderInput,
    held_batches: Sequence[_Section],
    held_questions: Sequence[_Section],
    hold_next: IXHold,
    asked_ids: set[str],
    held_items: Sequence[HeldItem] = (),
) -> list[HeldItem]:
    """§3.3 (part 2)·D — one record per call the gate stopped before it ran.

    `episode_id` is stamped from the write-back, so a row says whether the call
    is really waiting on an **open** episode or was refused for the turn only
    (a fail-closed binding batch, a spent release). The harness reads exactly
    that field to decide which ids the consent sideband may answer.

    `status`, `attribute` and `reason` are copied from the gate's own record for
    that call. Derived from the gate instead, a batch or question hold — which
    opens no gate — wrote `attribute: ""`, `reason: ""` and, whenever the model
    marked no prose, `status: held_for_confirmation`: live (`v5g01`) the audit
    said an autonomy gate held three calls on turns where `gate` was `null`, and
    nothing downstream could tell which turns took the `ix_hold` exit. The
    gate-derived values remain the fallback for a caller with no records (a
    direct `render()` in a unit test).
    """
    status = _held_status(render_input, held_batches, held_questions)
    attribute = render_input.gate.attribute if render_input.gate is not None else ""
    reason = render_input.gate.reason if render_input.gate is not None else ""
    unmatched = list(held_items)
    episode_by_item = {
        str(call.get("interaction_item_id") or ""): episode.episode_id
        for episode in hold_next.open_episodes.values()
        for call in episode.calls_held
        if call.get("interaction_item_id")
    }
    plan_by_item = {
        item_id: plan_item.id
        for plan_item in render_input.payloads.plan
        for item_id in plan_item.bound_item_ids
    }
    batch_by_item = {
        item_id: batch.batch_id
        for batch in render_input.payloads.batches
        for item_id in batch.call_binding
    }
    rows: list[HeldItem] = []
    for call in render_input.calls:
        if call.executed:
            continue
        record = _hold_record(unmatched, call)
        rows.append(HeldItem(
            status=record.status if record is not None else status,
            interaction_item_id=call.interaction_item_id,
            attribute=record.attribute if record is not None else attribute,
            tool=call.tool_name,
            plan_item_id=plan_by_item.get(call.interaction_item_id, ""),
            episode_id=episode_by_item.get(call.interaction_item_id, ""),
            batch_id=batch_by_item.get(call.interaction_item_id, ""),
            reason=record.reason if record is not None else reason,
            repaired=record.repaired if record is not None else False,
            repaired_by=record.repaired_by if record is not None else "",
            asked=call.interaction_item_id in asked_ids,
        ))
    return rows


def _calls_held_by_attribute(
    render_input: IXRenderInput,
    held_items: Sequence[HeldItem],
    gate_attribute: str,
    asked_ids: set[str],
) -> dict[str, list[dict]]:
    """Which attribute's episode owns each held call (§4.1 `calls_held`).

    With the gate's own records this is the attribute the gate wrote on the
    hold; a batch hold therefore lands on `topic_management` even though it
    opens no gate, which is what makes it survive the turn. Without them (a
    direct `render()` call in a unit test) every held call falls to the gate.
    """
    if not held_items:
        return {
            gate_attribute: [
                {
                    "interaction_item_id": call.interaction_item_id,
                    "tool": call.tool_name,
                    "plan_item_id": "",
                    "batch_id": "",
                    "reason": "",
                    "asked": call.interaction_item_id in asked_ids,
                }
                for call in render_input.calls
                if not call.executed
            ]
        } if gate_attribute else {}

    by_attribute: dict[str, list[dict]] = {}
    for item in held_items:
        if not item.attribute or item.reason in NON_EPISODE_HOLD_REASONS:
            continue
        by_attribute.setdefault(item.attribute, []).append({
            "interaction_item_id": item.interaction_item_id,
            "tool": item.tool,
            "plan_item_id": item.plan_item_id,
            "batch_id": item.batch_id,
            "reason": item.reason,
            # §3.7 (part 3): consent next turn is scoped to the ids this turn's ask
            # actually named, so the episode has to carry that fact forward.
            "asked": item.asked or item.interaction_item_id in asked_ids,
        })
    return by_attribute


def _pending(
    carried: Sequence[Any],
    held_now: Sequence[Any],
    settled: set[str],
    item_id: Callable[[Any], str],
) -> list[Any]:
    """The episode's still-pending items: what it carried, plus this turn's holds.

    `held_here` used to read only the holds produced *on this turn*, so the
    normal answers to a continuation closed the episode and lost every id still
    waiting: `keep` re-holds nothing, and a turn that answers in prose alone
    re-sends no call and re-marks no section. With several ids held, settling
    one silently dropped the rest -- live (`v11s02` acc_009) `q2` and `q3` were
    held on t3 under `Iterative` and were simply gone on t8, episode closed,
    `cancelled` empty, no close line. What settles an id is that its call ran or
    its section went out, or that the PA reported `skip`; `keep` and no decision
    at all leave it pending, and the episode closes only when nothing is left
    (§3.7 (part 3), §4.1–4.2).

    Questions are the one exception, and it is the caller that passes it in
    `settled`: a bare `release` settles a carried question, because §4.3 tells
    the PA that "a reply carrying none of them is released whole" -- the
    question was folded into this reply's prose. Batches and calls keep the
    delivery rule.
    """
    fresh = {item_id(item) for item in held_now}
    return [
        *(item for item in carried if item_id(item) not in settled | fresh),
        *held_now,
    ]


def _next_hold(
    render_input: IXRenderInput,
    settings: Mapping[str, str],
    released: set[tuple[str, str]],
    held_batches: Sequence[_Section],
    held_questions: Sequence[_Section],
    held_items: Sequence[HeldItem],
    asked_ids: set[str],
    gate_ask: str | None,
    trace: list[RenderStep],
) -> tuple[IXHold, list[HoldEpisode], list[dict]]:
    """Write back one episode per hold-bearing attribute (§4.1, §3.7 (part 4)).

    The second return value is the episodes cancelled **on this turn**, not
    every cancelled episode of the session: `finished_episodes` accumulates, and
    reading it per turn made one cancellation re-fire `ix_hold_forced_close` on
    every later turn (live `v6g01`: two cancellations, nine events).

    The third is the carried questions released by decision without their marker
    (§4.3, "folded into the reply"), for `ix_release_folded`.
    """
    payloads = render_input.payloads
    open_episodes = dict(render_input.hold.open_episodes)
    finished = list(render_input.hold.finished_episodes)
    cancelled_now: list[HoldEpisode] = []
    folded_now: list[dict] = []
    # acc_023 — a fully released batch set closes no episode and leaves nothing
    # behind, so the session's only memory of what a `batch_id` meant is this.
    # `setdefault`: the first binding is the one a redeclaration is checked
    # against, and a rejected turn must not overwrite it.
    #
    # Keyed by the set it was declared under, not by the id alone. Round 4 keyed
    # it by the id and fail-closed acc_023 twice for doing exactly what the
    # `batch_set_id` description asks: t18 opened `report_2026_02` and re-used
    # `b1`. A batch id is a handle inside its set, and `b1` in a new set is a new
    # handle; what must not happen is the same set re-pointing `b1` somewhere
    # else, which is the t18/t20 defect and is still caught.
    batch_bindings = dict(render_input.hold.batch_bindings)
    for declared_batch in payloads.batches:
        if declared_batch.batch_id:
            batch_bindings.setdefault(
                (payloads.batch_set_id, declared_batch.batch_id),
                tuple(declared_batch.call_binding),
            )

    plan_remaining = list(render_input.gate_remaining) if render_input.gate is not None else []
    boundary = payloads.boundary if render_input.gate is not None else None
    # Owner ruling 2026-09-18: the handoff message *is* the hand-back. Once it
    # has reached the user the PA holds nothing and owes nothing, so the
    # boundary is a record of a resolved limit -- not an item waiting on a
    # confirmation. Left on the books it could only be closed by "a later turn
    # with no gate", which a compressed session's reaction turn never has, and
    # the step then failed at `close_ix_session` with nothing actually pending.
    #
    # The judgement is about handoff in general, never about a
    # `capability_boundary` scene: `capability_boundary` is the one attribute a
    # *runtime trigger* activates rather than a beat, so this gate can open on
    # any beat of any step. Accordingly it reads the gate's own three fields
    # (written by `_capability_handoff_gate`, not by the model) and whether the
    # ask went out -- never the beat, the target attribute or the fixture.
    #
    # `gate_ask` is passed in rather than recomputed: `triggering_ask` was once
    # judged on both sides and the two answers drifted, so the renderer and the
    # runner read one value (see `triggering_ask`).
    if (
        render_input.gate is not None
        and render_input.gate.attribute == "capability_boundary"
        and render_input.gate.reason == "handoff"
        and render_input.gate.message_source == "handoff_message"
        and gate_ask
    ):
        boundary = None
        trace.append("capability_boundary handoff delivered -> nothing held")
    gate_attribute = render_input.gate.attribute if render_input.gate is not None else ""
    calls_by_attribute = _calls_held_by_attribute(
        render_input, held_items, gate_attribute, asked_ids
    )
    # §3.3 (part 2)·A / §3.7: one release buys one world-changing call. The ids that
    # spent theirs ride on the episode so the next turn still knows.
    # Owner ruling 2026-09-08: it is the *world change* that spends the release,
    # so a call the tool refused spends nothing. Live acc_039 t31 had
    # `appointment_create` come back an error and the id was booked as spent
    # anyway, so the next turn's re-issue needed a second authorisation for a
    # booking that had never happened. Within one response the duplicate is
    # still refused (`release_spent`): the gate runs before execution and cannot
    # know the outcome yet.
    executed_now = {
        call.interaction_item_id
        for call in render_input.calls
        if call.executed
        and call.status == "ok"
        and call.interaction_item_id
        and call.action_type in WORLD_CHANGING_ACTION_TYPES
    }
    # Anything that ran this turn settles its held id, world-changing or not.
    ran_now = {
        call.interaction_item_id
        for call in render_input.calls
        if call.executed and call.interaction_item_id
    }
    skipped_now = {
        decision.id
        for decision in payloads.release_decisions
        if decision.decision == "skip"
    }
    # §4.3, owner decision 2026-09-06: for **questions only**, a `release` with
    # no `[[q:…]]` marker means the PA folded that question into this reply's
    # prose -- that is exactly what the hold reminder tells it ("a reply
    # carrying none of them is released whole"). So a bare `release` settles the
    # question and no carry-over is left. Batches and calls are unchanged: they
    # settle on delivery / a bound call running / `skip` / the round cap.
    released_by_decision = {
        decision.id
        for decision in payloads.release_decisions
        if decision.decision == "release"
    }
    # A section settles the same way a call does: by going out. `release` is the
    # PA's intent (§3.7 (part 3)) -- what removes the id from the episode is the
    # renderer actually delivering that section this turn.
    released_now = {
        kind: {section_id for section_kind, section_id in released if section_kind == kind}
        for kind in (_BATCH, _QUESTION)
    }

    for attribute, episode_id, batches, questions, plan, boundary_state in (
        (
            "topic_management",
            # acc_023 b_review_proposals declared `bs_002` with `item_count: 0`
            # and `batches: []` — a no-op that would still have renamed an open
            # episode to a batch set holding nothing. A render that declares no
            # batch declares no batch set either.
            payloads.batch_set_id if payloads.batches else "",
            [
                Batch(
                    batch_id=section.id,
                    label=_batch_label(payloads, section.id),
                    body=section.text,
                    call_binding=_batch_binding(payloads, section.id),
                )
                for section in held_batches
            ],
            [],
            [],
            None,
        ),
        (
            "information_elicitation",
            payloads.elicitation_id,
            [],
            [
                Question(id=section.id, text=section.text, about=_question_about(payloads, section.id))
                for section in held_questions
            ],
            [],
            None,
        ),
        ("autonomy_level", payloads.plan_id, [], [], plan_remaining, None),
        ("capability_boundary", payloads.boundary_id, [], [], [], boundary),
    ):
        existing = open_episodes.get(attribute)
        carried = existing if existing is not None else HoldEpisode("", attribute)
        calls_held = _pending(
            carried.calls_held,
            calls_by_attribute.get(attribute, []),
            ran_now | skipped_now,
            lambda call: str(call.get("interaction_item_id") or ""),
        )
        batches = _pending(
            carried.batches_held,
            batches,
            # §4.4 `released`: a batch went out when its section did **or** when
            # a call bound to it ran -- the model may release the work without
            # ever writing the `[[batch:…]]` marker again.
            released_now[_BATCH] | skipped_now | {
                batch.batch_id for batch in carried.batches_held
                if ran_now & set(batch.call_binding)
            },
            lambda batch: batch.batch_id,
        )
        folded_ids = [
            question.id
            for question in carried.questions_held
            if question.id in released_by_decision
            and question.id not in released_now[_QUESTION]
            and question.id not in {held_again.id for held_again in questions}
        ]
        if folded_ids:
            folded_now.append({
                "attribute": attribute,
                # The folded questions were carried, so the episode they belong
                # to is the open one -- not this turn's fresh payload id.
                "episode_id": carried.episode_id or episode_id,
                "question_ids": folded_ids,
            })
            trace.append(
                f"{attribute} questions folded into the reply (no marker): "
                + ", ".join(folded_ids)
            )
        questions = _pending(
            carried.questions_held,
            questions,
            released_now[_QUESTION] | skipped_now | released_by_decision,
            lambda question: question.id,
        )
        held_here = bool(batches or questions or plan or boundary_state or calls_held)
        if not held_here:
            if existing is not None and not existing.closed:
                open_episodes.pop(attribute)
                finished.append(replace(existing, closed=True))
                trace.append(f"{attribute} episode {existing.episode_id} -> closed")
            continue

        rounds_open = (existing.rounds_open if existing is not None else 0) + 1
        episode = HoldEpisode(
            episode_id=episode_id or (existing.episode_id if existing is not None else ""),
            attribute=attribute,
            opened_at_turn=existing.opened_at_turn if existing is not None else render_input.turn_index,
            rounds_open=rounds_open,
            closed=False,
            last_gate=render_input.gate if attribute == gate_attribute else (
                existing.last_gate if existing is not None else None
            ),
            plan_remaining=list(plan),
            batches_held=list(batches),
            questions_held=list(questions),
            boundary=boundary_state,
            calls_held=list(calls_held),
            executed_item_ids=sorted(
                set(existing.executed_item_ids if existing is not None else [])
                | executed_now
            ),
        )
        if rounds_open >= HOLD_ROUNDS_HARD_CAP:
            # §3.7 (part 4): hitting the cap cancels, it never releases by default.
            open_episodes.pop(attribute, None)
            cancelled = replace(
                episode,
                closed=True,
                plan_remaining=[],
                batches_held=[],
                questions_held=[],
                calls_held=[],
                cancelled=_cancelled_entries(
                    attribute, batches, questions, plan, calls_held
                ),
            )
            cancelled_now.append(cancelled)
            finished.append(cancelled)
            trace.append(
                f"{attribute} episode {episode.episode_id} -> cancelled at round {rounds_open}"
            )
            continue
        open_episodes[attribute] = episode
        trace.append(
            f"{attribute} episode {episode.episode_id} -> held, round {rounds_open}"
        )

    return (
        IXHold(
            open_episodes=open_episodes,
            finished_episodes=finished,
            batch_bindings=batch_bindings,
            # Owner ruling 2026-09-08: an unspent release outlives the episode
            # it was granted in -- acc_039's re-mint closed the plan and opened
            # a new one, and the authorisation has to be there when it does.
            # The collector computes it (the sideband is its side of the wall);
            # here it is only carried across.
            act_releases=dict(render_input.act_releases),
        ),
        cancelled_now,
        folded_now,
    )


def _cancelled_entries(
    attribute: str,
    batches: Sequence[Batch],
    questions: Sequence[Question],
    plan: Sequence[PlanItem],
    calls: Sequence[dict] = (),
) -> list[dict]:
    """The cancelled rows, each carrying the item's own `intent` text.

    `intent` is what the close line and the next turns' reminder say out loud,
    so it is written here rather than re-derived later: a batch's is its label,
    a question's is its `about`, a plan item's is its intent.

    A question's `about` and **not** its `text`: `text` is the wording the
    wrapper withheld, and cancelling at the round cap used to copy the whole of
    it into the public close line and into every later turn's reminder. The
    short label is what a user needs to hear; `Question.text` stays audit-only.

    regress15 acc_039: a held **call** was none of these three, so the episode
    holding `apt_book_001` and nothing else was force-closed reporting
    `cancelled: 0` and the call was dropped without a trace. A cancelled hold
    now says what it dropped whatever kind of item it was holding; the call's
    intent is its plan item's, which is the sentence the PA wrote about it.
    """
    intents = {item.id: item.intent for item in plan}
    entries = [
        {"attribute": attribute, "id": batch.batch_id, "kind": "batch",
         "intent": batch.label}
        for batch in batches
    ]
    entries += [
        {"attribute": attribute, "id": question.id, "kind": "question",
         "intent": question.about}
        for question in questions
    ]
    entries += [
        {"attribute": attribute, "id": item.id, "kind": "plan_item",
         "intent": item.intent}
        for item in plan
    ]
    # A call bound to a batch or a plan item is already on the list under that
    # item's own row -- `b2` and the `c2` it binds are one cancellation, not two.
    # What is added here is the call nothing else accounts for.
    covered = {
        bound
        for batch in batches for bound in batch.call_binding
    } | {
        bound
        for item in plan for bound in item.bound_item_ids
    }
    entries += [
        {"attribute": attribute,
         "id": str(call.get("interaction_item_id") or ""),
         "kind": "call",
         "intent": intents.get(str(call.get("plan_item_id") or ""), "")}
        for call in calls
        if str(call.get("interaction_item_id") or "") not in covered
    ]
    return entries


def _batch_label(payloads: TurnPayloads, batch_id: str) -> str:
    for batch in payloads.batches:
        if batch.batch_id == batch_id:
            return batch.label
    return ""


def _batch_binding(payloads: TurnPayloads, batch_id: str) -> list[str]:
    for batch in payloads.batches:
        if batch.batch_id == batch_id:
            return list(batch.call_binding)
    return []


def _question_about(payloads: TurnPayloads, question_id: str) -> str:
    for question in payloads.questions:
        if question.id == question_id:
            return question.about
    return ""
