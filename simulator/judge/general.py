"""The general form: judge a beat's own direction text, one clause at a time.

This form guards nothing but what the beat itself wrote down — its
`user_instruction`, its `avoid`, and (on the cue beat's opening turn) its cue.
The cross-session defects stay per-rule in the catalogue, because they were each
added after a regression showed the simulator making a mistake nobody could have
foreseen while writing the beat, and the beat's `avoid` has no such grain.

Three things the measurement forced:

- `avoid` is split into one question per sentence, each traceable on its own. A
  whole `avoid` in one question loses which half broke.
- an instruction covers a beat, and a beat runs over several turns; it is judged
  only on the turns it binds. Eight of the 28 false alarms were the judge
  reading a beat-wide instruction as owed by every turn of the beat.
- must-ask and beat position are printed as facts. Left to work it out, the
  judge invented a must-ask constraint on 40 of 58 calls and failed every one.

The form is asked over several calls rather than one. A ten-question general
call was failing on the length of its quotes and its JSON, not on time, and one
bad quote among ten answers voids and re-asks the whole group; `avoid` is
therefore chunked at `GENERAL_AVOID_CHUNK`, so a re-ask costs one chunk. The
clause index stays global across the chunks, so a finding still quotes the
author's own words back.
"""

from __future__ import annotations

import re

from simulator.judge.catalogue import (
    RULE_PROMPTS,
    Question,
    QuestionGroup,
    TurnContext,
)
from simulator.judge.guards import VERDICT_VALUES

#: Clause boundaries: a sentence terminator only. A semicolon continues the
#: sentence, so its second half is not a prohibition that stands on its own. The
#: clause keeps its own text verbatim, terminator included, so a violation can be
#: quoted back to the author in the words the author wrote.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?])\s*")

#: `avoid` clauses per call. Measured on 主-解读's step-0 gate_cue replay
#: 2026-09-09 (deadline 60, load average 11-14): one question 7.6 s mean, five
#: 26.0 s, ten 32.1 s, and 13 of ~50 rows failed — on quote and JSON length,
#: not on time, with one bad quote voiding and re-asking all ten answers. At
#: most four per call is about 15 s, and a bad quote re-asks only its chunk;
#: one question per call would be ~76 s against 32 and would eat the turn
#: budget.
GENERAL_AVOID_CHUNK = 4

_AVOID_LEAD = (
    "The line must not do what this clause describes. The clause is the "
    "author's summary, not words to look for or quote. "
    # The ban read as a requirement: the judge answered `violated` because the
    # line did NOT do the banned thing (9 of 24 avoid FPs, memos r2 live round,
    # 2026-09-15 — e.g. acc_031 `beat_4_probe_order_and_settle`, reason "does
    # not authorize … which is exactly what the avoid clause describes").
    "The clause describes what the line must not do: a line that refrains "
    "from it, or does the opposite, has obeyed it, and obeying the clause is "
    "never a violation. "
    # Commented out, owner ruling 2026-09-14: a scope-note clause is a draft defect.
    # "Some clauses only say "
    # "what this turn is about, judges, or evaluates: a clause like that forbids "
    # "nothing, so answer obeyed with empty evidence. "
    "<beat_goal> names the "
    "thing the line is there to evaluate, and saying how that thing landed — "
    "its reasons and its wording included — is obeyed. "
    # Commented out, owner ruling 2026-09-14: read as a standalone ban (32/33 FP).
    # "Every other act this "
    # "clause bans is still violated even where the goal touches it: telling "
    # "the assistant to proceed or what to do next, prescribing length or "
    # "format, opening new work, stating a standing rule. Answer violated for "
    # "one of those:"
    # The bridge the commented-out list's colon used to provide.
    "The clause:"
)
#: The opening of the generator's fixed scope-sentence template for a
#: measurement reaction beat. It mirrors `_REACTION_SCOPE_SENTENCE` in
#: `scripts/validate_normal_event.py`: a literal the generator writes, matched
#: as a literal prefix, never a wording variant.
_REACTION_SCOPE_SENTENCE = "This turn judges only"
_INSTRUCTION_LEAD = (
    "The line must do what this direction describes. The direction is the "
    "author's summary of the moment, not words to look for or quote:"
)


def split_clauses(text: str) -> list[str]:
    """One clause per sentence, verbatim."""
    return [part.strip() for part in _CLAUSE_SPLIT_RE.split(text or "") if part.strip()]


def build_general_groups(ctx: TurnContext) -> list[QuestionGroup]:
    """This beat's own direction, as one call for its head and one per chunk of
    `avoid` clauses.

    `must_ask` and `beat_position` go in as blocks, never as questions:
    `carries_an_ask` keeps its own per-rule call,
    and the judge is told where the turn sits rather than asked.

    The head call carries the beat's `user_instruction` on the turn that binds
    it and, on the cue beat's opening turn, the cue; it is omitted when the turn
    owes neither. The `avoid` clauses follow in chunks of at most
    `GENERAL_AVOID_CHUNK`, each chunk carrying only the clauses it asks about
    and keeping the clause's global index in both its question id and its
    `avoid::<i>` block.
    """
    base = dict(ctx.base_blocks())
    base["beat_position"] = ctx.beat_position or ctx.beat_id
    base["turn_owes_an_ask"] = "yes" if ctx.must_ask else "no"
    base["instruction_binds_this_turn"] = (
        "yes" if ctx.instruction_binds_this_turn else "no"
    )

    groups: list[QuestionGroup] = []

    head_blocks = dict(base)
    head: list[Question] = []
    if ctx.beat_user_instruction.strip() and ctx.instruction_binds_this_turn:
        head_blocks["beat_user_instruction"] = ctx.beat_user_instruction
        head.append(
            # `obeyed` quotes: the turn did the thing, so the words are in it.
            # `violated` is the absence — the instruction was not followed and
            # there is nothing in the turn to point at (owner ruling
            # 2026-09-09, "没违反不引用").
            Question(
                "user_instruction",
                f"{_INSTRUCTION_LEAD} {ctx.beat_user_instruction}",
                "candidate_turn",
                VERDICT_VALUES,
                ("obeyed",),
            )
        )
    if ctx.cue_owed and ctx.cue.strip():
        head_blocks["cue"] = ctx.cue
        head.append(
            # Same shape as `user_instruction`: `obeyed` points at the words
            # that carry the cue's meaning; `violated` says the meaning is
            # nowhere in the turn, which is nothing to quote.
            Question(
                "gate_cue_missing",
                RULE_PROMPTS["gate_cue_missing"].text("gate_cue_missing", cue=ctx.cue),
                "candidate_turn",
                VERDICT_VALUES,
                ("obeyed",),
            )
        )
    if head:
        groups.append(_group(tuple(head), head_blocks))

    # The reaction beat's scope sentence is authoring metadata: it tells the
    # validator which beat measures the attribute's surface
    # (`_REACTION_SCOPE_SENTENCE`, scripts/validate_normal_event.py). It is not
    # a ban, but the judge read it as one and answered `violated` on the very
    # behaviour the beat measures (20 of 28 residual avoid FPs, 2026-09-14 live
    # round). It is dropped here; the kept clauses hold their global index so
    # diagnostics and replays stay comparable.
    kept = [
        (index, clause)
        for index, clause in enumerate(split_clauses(ctx.beat_avoid), start=1)
        if not clause.startswith(_REACTION_SCOPE_SENTENCE)
    ]
    for offset in range(0, len(kept), GENERAL_AVOID_CHUNK):
        chunk_blocks = dict(base)
        # What the beat is FOR, once per chunk. Without it the judge fired
        # `violated` on the very behaviour the beat exists to measure —
        # reacting to confidence tagging read as commenting on derivations,
        # saying how an unprompted message landed read as judging tone (owner
        # labels, 2026-09-12). At runtime `beat_intent` is the beat's
        # `user_instruction` (schemas.py:832,850; agent/loop.py:370-373); the
        # second name is only for contexts built without an intent.
        goal = (ctx.beat_intent or ctx.beat_user_instruction).strip()
        if goal:
            chunk_blocks["beat_goal"] = goal
        questions: list[Question] = []
        for index, clause in kept[offset : offset + GENERAL_AVOID_CHUNK]:
            chunk_blocks[f"avoid::{index}"] = clause
            questions.append(
                # An avoid clause is a ban, so `violated` is the presence side
                # and the default holds: quote the banned thing the line did.
                Question(
                    f"avoid_{index}",
                    f"{_AVOID_LEAD} {clause}",
                    "candidate_turn",
                    VERDICT_VALUES,
                )
            )
        groups.append(_group(tuple(questions), chunk_blocks))

    return groups


def _group(questions: tuple[Question, ...], blocks: dict[str, str]) -> QuestionGroup:
    return QuestionGroup(
        rule="general",
        shape="general",
        questions=questions,
        blocks=blocks,
        num_predict_kind="general",
    )
