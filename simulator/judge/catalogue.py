"""The rule catalogue: which questions a rule asks, and how its answers compose.

The question TEXT is not here. It lives in `prompts/rules/<rule>.md`, which is
the measured text from the experiment wherever a measurement exists — the
prompt files are the rule, and contract §9's English column is a gloss of them.
That is the only sanctioned patch channel, so it is a file and not a string.

What is here is the shape: which questions a rule asks, which context blocks
they get, which block each question's evidence is checked against, and the
compose function that turns the answers into the rule's boolean. The last
boolean is always computed here, never by the model — that single change took
`reaction_authorises_pa_action` from 1 miss / 1 false alarm to 0 / 0.

Two vocabularies. A question whose answer feeds a compose function answers
`yes` / `no` and nothing else; only a rule that returns a verdict of its own
uses `obeyed` / `violated` / `not_applicable`, and the ask gate uses
`ask` / `not_applicable`. The split is not cosmetic: asked in the
`obeyed`/`violated` vocabulary whether a turn claims the assistant never asked,
the model quoted the denial, wrote "the turn claims that the assistant failed
to ask for details", and answered `obeyed`.

Two things are never asked of the model and always given to it: arming (does
this turn owe an ask, is the gate open, which facts are on-request) and beat
position. In the general form the judge invented a must-ask constraint on 40 of
58 calls when it was left to work it out.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from simulator.judge.guards import (
    ASK_GATE_VALUES,
    PRESENCE_ANSWERS,
    VERDICT_VALUES,
    YES_NO_NA_VALUES,
    YES_NO_VALUES,
    Verdict,
    normalise,
    numbered_block,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"
RULES_DIR = PROMPTS_DIR / "rules"

#: A carve-out is at most this many clauses. Five is the draft validator's
#: error at authoring time, never a judge event at run time (§9).
MAX_EXCEPTION_CLAUSES = 4

_EXCEPTION_RE = re.compile(
    r"\b(?:beyond|except(?:\s+for)?|other than|apart from|aside from|save for|unless)\b",
    re.IGNORECASE,
)
_CLAUSE_END_RE = re.compile(r"[.;]|(?<=\w),(?=\s)")
_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)

#: The effects whose call changes state, read off the same catalog file —
#: every write, in-sandbox (`internal_write`: documents, contacts, calendar,
#: notes) as well as out (`external_write`, `external_action`). Read-only
#: effects (`read`, `verify`) stay out: looking is not an act.
#: Owner ruling of 2026-09-12: WHICH acts the assistant carried out is
#: mechanical, so code collects them and hands the list to the model; what a
#: line claims stays the model's. The mechanical half of that brief was
#: written against `harness/gate.py:_catalog_external_tools`, whose set is the
#: external two, because a session releases at most one external tool — but
#: the question is about ANY act, and the boolean it replaced (`pa_wrote`,
#: a suffix test) counted in-sandbox writes too. Live user_b `acc_001` on the
#: mem0 line (2026-09-13) is the cost of the narrower set: the PA really ran
#: `document_update`, the list read "none yet", and four rewrites of the
#: user's thanks were all judged fictions. Not imported from `harness`:
#: `simulator/` does not read that package, the way `simulator.loop` and
#: `simulator.message_quality` keep their own copies of the coarser suffix
#: test rather than import each other's.
_STATE_CHANGING_EFFECTS = frozenset(
    {"external_write", "external_action", "internal_write"}
)

#: The state server's MCP prefix on a catalog name: the catalog declares
#: `email_message_send` and the PA calls `mcp_state_email_message_send`.
_STATE_TOOL_PREFIX = "mcp_state_"

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data/state_tools/catalog.yaml"

#: What the evidence block says when the assistant has carried out nothing.
NO_EXECUTED_ACTS = "none yet"

#: A delivered proactive outreach as one line of the same list. It is the one
#: act with no catalog entry: the send happens outside the tool record
#: (`SimulatorLoop._run_proactive_trigger`), so no name carries it and the
#: label is written here rather than read off the catalog. Live regress24m
#: `acc_029` is why it is on the list at all — the PA really did contact the
#: user, and the actor's true "having it sent was unwelcome" was judged a
#: fiction without it.
OUTREACH_ACT = "proactive_outreach — send the user an unprompted message"


@lru_cache(maxsize=1)
def executed_act_labels() -> dict[str, str]:
    """Catalog name -> one line saying what that tool does, for every write.

    The label is the first sentence of the catalog's own description, so there
    is no second table of prose to drift from the tools it names. A tool that
    only reads is not in here at all, so a read can never reach the list.
    """
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    labels = {}
    for name, spec in (catalog.get("tools") or {}).items():
        if not isinstance(spec, dict):
            continue
        if not _STATE_CHANGING_EFFECTS & set(spec.get("allowed_effects") or ()):
            continue
        description = str(spec.get("description") or "").strip()
        first = re.split(r"(?<=[.!?])\s+", description)[0].rstrip(".")
        labels[str(name)] = first[:1].lower() + first[1:]
    return labels


def executed_act_label(name: str) -> str | None:
    """`<bare name> — <catalog line>` for one executed call, or `None` for a read.

    Strips the state server's MCP prefix and looks the bare name up in
    `executed_act_labels()`. Shared by the judge's `executed_acts` and the
    actor's own record (`simulator.reply_facts`), so the two never describe
    the same write in two vocabularies.
    """
    bare = str(name or "")
    if bare.startswith(_STATE_TOOL_PREFIX):
        bare = bare[len(_STATE_TOOL_PREFIX) :]
    label = executed_act_labels().get(bare)
    return f"{bare} — {label}" if label else None


# ------------------------------------------------------------- prompt files


@dataclass(frozen=True)
class RulePrompt:
    """One `prompts/rules/<rule>.md`: its provenance, preamble and questions."""

    rule: str
    status: str
    source: str
    preamble: str
    sections: Mapping[str, str]

    def text(self, section: str, **substitutions: str) -> str:
        try:
            body = self.sections[section]
        except KeyError:  # pragma: no cover - a malformed prompt file
            raise KeyError(
                f"{self.rule}.md has no '## {section}' section; it has "
                f"{sorted(self.sections)}"
            ) from None
        for name, value in substitutions.items():
            body = body.replace("{{" + name + "}}", value)
        return body


def load_rule_prompt(rule: str) -> RulePrompt:
    path = RULES_DIR / f"{rule}.md"
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: must open with a '---' front-matter block")
    front, _, body = text[3:].partition("\n---\n")
    meta = yaml.safe_load(front) or {}
    if meta.get("rule") != rule:
        raise ValueError(f"{path}: front matter names rule {meta.get('rule')!r}")
    if meta.get("status") not in ("measured", "adapted", "unmeasured"):
        raise ValueError(f"{path}: status must be measured / adapted / unmeasured")

    sections: dict[str, str] = {}
    marks = list(_SECTION_RE.finditer(body))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(body)
        sections[mark.group(1).strip()] = body[mark.end() : end].strip()
    if "preamble" not in sections:
        raise ValueError(f"{path}: needs a '## preamble' section")
    preamble = sections.pop("preamble")
    if not sections:
        raise ValueError(f"{path}: needs at least one question section")
    return RulePrompt(rule, meta["status"], str(meta.get("source", "")), preamble, sections)


RULE_PROMPTS: dict[str, RulePrompt] = {
    path.stem: load_rule_prompt(path.stem) for path in sorted(RULES_DIR.glob("*.md"))
}


# ----------------------------------------------------------------- the input


@dataclass
class TurnContext:
    """Everything the harness hands the judge about one candidate turn.

    Verbatim text only: no persona, no outline, no setting name, no attribute
    name, no world state (§2). The fields below the two verbatim blocks are the
    context the catalogue lets specific questions draw on, plus the
    code-computed facts that the judge is told rather than asked.
    """

    turn_uid: str = ""
    pa_previous_turn: str = ""
    candidate_turn: str = ""

    # context blocks the catalogue draws on
    pa_turns: tuple[str, ...] = ()
    tool_log: tuple[str, ...] = ()
    previous_user_turns: tuple[str, ...] = ()
    contractions: tuple[str, ...] = ()
    fact_values: Mapping[str, str] = field(default_factory=dict)
    beat_intent: str = ""
    beat_avoid: str = ""
    beat_id: str = ""
    beat_user_instruction: str = ""
    gate_dictated_text: str = ""
    gate_key_terms: tuple[str, ...] = ()
    gate_reveal_beat: str = ""
    cue: str = ""
    #: The attribute this session measures. The one place the judge is told an
    #: attribute name (§2 otherwise forbids it): `reaction_beat_off_attribute`
    #: asks whether the turn sends the speaker at some OTHER attribute's
    #: surface, and that question cannot be put without naming this one.
    target_attribute: str = ""

    # arming and code-side answers: given, never asked
    on_request_fact_keys: frozenset[str] = frozenset()
    pa_asked: bool = False
    missing_key_terms: tuple[str, ...] = ()
    must_ask: bool = False
    beat_position: str = ""
    instruction_binds_this_turn: bool = True
    cue_owed: bool = False
    # which arm of `withheld_fact_before_ask` armed the turn: the arming site
    # sets it and `compose_withheld_fact_before_ask` reads it.
    pre_reveal: bool = False
    #: The names of the tools the assistant actually ran this session, in call
    #: order and already minus the calls that errored — the input to
    #: `executed_acts`, never a block the judge reads as it stands.
    executed_tools: tuple[str, ...] = ()
    #: Whether a triggered proactive outreach was DELIVERED this session. The
    #: send happens outside the tool record (`SimulatorLoop._run_proactive_
    #: trigger`), so no name ever carries it; it is a flag of its own rather
    #: than a synthetic tool name, because `executed_tools` also feeds the retry
    #: note's clause and the `<tool_log>` window.
    outreach_sent: bool = False

    @property
    def executed_acts(self) -> tuple[str, ...]:
        """The acts this session has really carried out, one line each.

        The owner's ruling of 2026-09-12 replaced the session boolean
        `pa_wrote`: one flag meant any real write switched
        `reaction_asserts_unexecuted_pa_action` off for every later line, so an
        actor could fabricate "you already sent it" after the PA had merely
        placed an order, and the transcript carried the fiction into PA memory.
        The judge is handed the list instead and asked whether the line claims
        something off it.

        An act is any call that changed state (`_STATE_CHANGING_EFFECTS`: the
        catalog's writes, in-sandbox as well as external), plus a delivered
        outreach. The in-sandbox writes are on it because the question is
        about any act the line credits the assistant with, and a document the
        PA really edited is one: live user_b `acc_001` (2026-09-13) thanked it
        for a `document_update` it had just run and the turn was voided as a
        fiction. A read is never an act. The list is as long as the session's
        writes; a tool called twice is still one act, because the list says
        what was done, not how many calls it took.
        """
        acts: list[str] = []
        for name in self.executed_tools:
            line = executed_act_label(name)
            if line and line not in acts:
                acts.append(line)
        if self.outreach_sent:
            acts.append(OUTREACH_ACT)
        return tuple(acts)

    def base_blocks(self) -> dict[str, str]:
        return {
            "pa_previous_turn": self.pa_previous_turn,
            "candidate_turn": self.candidate_turn,
        }


# -------------------------------------------------------------- the question


@dataclass(frozen=True)
class Question:
    """One question, where its evidence comes from, and what it may answer.

    `evidence_block` names one block, or several when the question invites a
    quote from more than one of them: `probe_fact_missing` says in as many
    words that a fact already delivered on an earlier turn counts, so a correct
    answer may quote an earlier turn, and checking it against the candidate
    turn alone would fail every one of those.
    """

    question_id: str
    text: str
    evidence_block: str | tuple[str, ...]
    answers: tuple[str, ...] = YES_NO_VALUES
    quotes_on: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.quotes_on is None:
            object.__setattr__(
                self,
                "quotes_on",
                tuple(a for a in self.answers if a in PRESENCE_ANSWERS),
            )
        unknown = set(self.quotes_on) - set(self.answers)
        if unknown:
            raise ValueError(
                f"{self.question_id}: quotes_on {sorted(unknown)} are not "
                f"answers this question allows ({list(self.answers)})"
            )

    @property
    def evidence_block_names(self) -> tuple[str, ...]:
        if isinstance(self.evidence_block, str):
            return (self.evidence_block,)
        return tuple(self.evidence_block)


@dataclass(frozen=True)
class QuestionGroup:
    """One call: a rule's whole set of steps (§2, "一次调用一个问题组")."""

    rule: str
    shape: str
    questions: tuple[Question, ...]
    blocks: Mapping[str, str]
    preamble: str = ""
    num_predict_kind: str = "yes_no"

    @property
    def question_ids(self) -> list[str]:
        return [question.question_id for question in self.questions]

    @property
    def is_empty(self) -> bool:
        """A group with nothing to ask. No call is made for one (§2: a call is
        a question group, and there is no group here) — an empty `format` enum
        is not a request Ollama can honour anyway."""
        return not self.questions

    def allowed(self) -> dict[str, tuple[str, ...]]:
        return {question.question_id: question.answers for question in self.questions}

    def quotes_on(self) -> dict[str, tuple[str, ...]]:
        """question_id -> the answers that must be backed by a verbatim quote."""
        return {
            question.question_id: tuple(question.quotes_on or ())
            for question in self.questions
        }

    def question_blocks(self) -> dict[str, tuple[str, ...]]:
        """question_id -> the names of the blocks its evidence may reference.

        The blocks themselves are `self.blocks`, printed with their sentences
        numbered; a reference names one of these and a number in it.
        """
        return {
            question.question_id: question.evidence_block_names
            for question in self.questions
        }


@dataclass(frozen=True)
class RuleSpec:
    """A catalogue row: how to build the call, and how to read the answers."""

    rule: str
    build: Callable[[TurnContext], QuestionGroup]
    compose: Callable[[dict[str, Verdict], TurnContext], Any]


# ---------------------------------------------------------------- the builds


def _joined(lines: Sequence[str]) -> str:
    return "\n\n".join(line for line in lines if str(line).strip())


#: Longest one argument value may be in a `<tool_log>` line. The judge has to
#: see WHICH record a call touched, never the whole anchor or the whole
#: replacement a `document_update` carries.
TOOL_LOG_ARG_CHARS = 80


def _clip(text: str) -> str:
    return text if len(text) <= TOOL_LOG_ARG_CHARS else text[:TOOL_LOG_ARG_CHARS] + "…"


def format_tool_log_line(
    name: Any, args: Any = None, status: Any = None
) -> str:
    """One `<tool_log>` line: what ran, what it ran on, how it came back.

    Names alone are not enough. Live regress22 `acc_006` stopped four times on
    `reaction_asserts_unexecuted_pa_action` after seven `mcp_state_document_update`
    calls that all came back ok: handed seven bare names, the judge could not
    see WHAT had been updated and answered "the tool log shows no such action"
    every time. Live and replay both build their lines here, so the corpus
    measures the block the harness actually sends.
    """
    parts = [str(name or "")]
    if isinstance(args, Mapping):
        if args:
            parts.append(
                json.dumps(
                    {
                        str(key): _clip(
                            value
                            if isinstance(value, str)
                            else json.dumps(value, ensure_ascii=False, default=str)
                        )
                        for key, value in args.items()
                    },
                    ensure_ascii=False,
                )
            )
    elif args:
        parts.append(_clip(json.dumps(args, ensure_ascii=False, default=str)))
    if status:
        parts.append(f"status={status}")
    return " ".join(parts)


def tool_log_lines(events: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """The `<tool_log>` lines for one PA turn's tool events.

    Same window as `SimulatorLoop._pa_executed_tools`, minus the IX calls: the
    wrapper's control records (`ix_event`) are bookkeeping rather than a call
    the PA made, an errored call changed nothing, and an IX selection is not
    work on the world, which is all this block is asked about.
    """
    return tuple(
        format_tool_log_line(
            event.get("name"), event.get("args"), event.get("status")
        )
        for event in events
        if not event.get("ix")
        and not event.get("ix_event")
        and not event.get("error")
    )


def exception_clauses(beat_avoid: str) -> list[str]:
    """The carve-outs the beat's `avoid` hangs off its bans.

    Each becomes its own `qc<i>` question, because the model reads a carve-out
    and then fails to apply it when it has to compute the boolean itself.
    """
    clauses: list[str] = []
    for match in _EXCEPTION_RE.finditer(beat_avoid or ""):
        tail = beat_avoid[match.start() :]
        stop = _CLAUSE_END_RE.search(tail)
        clause = tail[: stop.start() if stop else len(tail)].strip()
        if clause:
            clauses.append(clause)
    if len(clauses) > MAX_EXCEPTION_CLAUSES:
        raise ValueError(
            f"{len(clauses)} exception clauses in `avoid`, at most "
            f"{MAX_EXCEPTION_CLAUSES} are allowed; this is a draft-validator "
            f"error at authoring time, not a run-time judge event"
        )
    return clauses


def _group(
    rule: str,
    ctx: TurnContext,
    questions: Sequence[Question],
    *,
    shape: str = "per_rule",
    extra_blocks: Mapping[str, str] | None = None,
    num_predict_kind: str = "yes_no",
) -> QuestionGroup:
    blocks = dict(ctx.base_blocks())
    blocks.update(extra_blocks or {})
    return QuestionGroup(
        rule=rule,
        shape=shape,
        questions=tuple(questions),
        blocks=blocks,
        preamble=RULE_PROMPTS[rule].preamble,
        num_predict_kind=num_predict_kind,
    )


def build_carries_an_ask(ctx: TurnContext) -> QuestionGroup:
    prompt = RULE_PROMPTS["carries_an_ask"]
    return _group(
        "carries_an_ask",
        ctx,
        [
            Question(
                "carries_an_ask",
                prompt.text("carries_an_ask"),
                "candidate_turn",
                ASK_GATE_VALUES,
            )
        ],
    )


def build_asserts_unexecuted(ctx: TurnContext) -> QuestionGroup:
    """One question: does the turn claim an act that is not on the record.

    q2 asked the model to find the claimed change in `<tool_log>`, and the 12B
    judge answered it by hunting for a tool named after the user's phrasing
    (live regress22 `acc_006`, twice). Owner ruling B of 2026-09-10 deleted it
    and `<tool_log>` with it, and made "did anything run" a code-side boolean;
    the owner's ruling of 2026-09-12 replaced that boolean with this list.
    `<executed_acts>` is the difference: not the raw log the judge could not
    read, but the acts themselves, named and glossed, which is a list a line
    can be checked against. `<assistant_turns_so_far>` stays — the question
    turns on what the assistant said it WOULD do.
    """
    rule = "reaction_asserts_unexecuted_pa_action"
    prompt = RULE_PROMPTS[rule]
    return _group(
        rule,
        ctx,
        [Question(f"{rule}__q1", prompt.text(f"{rule}__q1"), "candidate_turn")],
        extra_blocks={
            "assistant_turns_so_far": _joined(ctx.pa_turns),
            # Never empty: a block with no text is dropped from the prompt
            # (`render_blocks`), and the question names this one.
            "executed_acts": "\n".join(ctx.executed_acts) or NO_EXECUTED_ACTS,
        },
    )


def build_misstates_contractions(ctx: TurnContext) -> QuestionGroup:
    rule = "reaction_misstates_contractions"
    prompt = RULE_PROMPTS[rule]
    return _group(
        rule,
        ctx,
        [
            Question(f"{rule}__q1", prompt.text(f"{rule}__q1"), "candidate_turn"),
            # q2 has a third answer for "q1 said there is no report", so the
            # frame's "answer each question on its own" holds and a wrong `no`
            # no longer has to be copied down (audit finding I, 2026-09-12).
            Question(
                f"{rule}__q2",
                prompt.text(f"{rule}__q2"),
                "candidate_turn",
                YES_NO_NA_VALUES,
            ),
        ],
        shape="two_step",
        extra_blocks={
            "contraction_list": (
                ", ".join(ctx.contractions)
                if ctx.contractions
                else "(the reply contains no contractions)"
            )
        },
        num_predict_kind="wide",
    )


def build_probe_fact_missing(ctx: TurnContext) -> QuestionGroup:
    """One question per fact, all in one call.

    One fact at a time so the evidence always points at words that are really in
    this turn: a question asking the model to quote what is *not* there cannot
    pass the §4(2) check by construction.
    """
    prompt = RULE_PROMPTS["probe_fact_missing"]
    questions = [
        Question(
            f"probe_fact_missing:{key}",
            prompt.text("probe_fact_missing:<fact_key>", fact=str(value)),
            ("candidate_turn", "earlier_user_turns"),
        )
        for key, value in ctx.fact_values.items()
    ]
    return _group(
        "probe_fact_missing",
        ctx,
        questions,
        extra_blocks={
            "beat_intent": ctx.beat_intent,
            "earlier_user_turns": _joined(ctx.previous_user_turns),
        },
        num_predict_kind="general",
    )


def build_authorises_pa_action(ctx: TurnContext) -> QuestionGroup:
    """The experiment's five questions, plus one per carve-out cut from `avoid`.

    Contract §9 names only q1 and the carve-outs. The other three — does the
    turn release the assistant, is the release the speaker's own next step, is
    it held or conditional — are the measured form, and the boolean that scored
    0 misses / 0 false alarms is over all five.
    """
    rule = "reaction_authorises_pa_action"
    prompt = RULE_PROMPTS[rule]
    # `assistant_turns_so_far` is not here: no sub-question reads it, and the
    # assistant's own go-aheads inside it pull q2 onto the wrong speaker
    # (audit finding G, 2026-09-12).
    blocks = {"beat_avoid": ctx.beat_avoid}
    questions = [Question(f"{rule}__q1", prompt.text(f"{rule}__q1"), "beat_avoid")]
    for index, clause in enumerate(exception_clauses(ctx.beat_avoid), start=1):
        blocks[f"exception::{index}"] = clause
        questions.append(
            Question(
                f"{rule}__qc{index}",
                prompt.text(f"{rule}__qc<i>", exception=clause),
                "beat_avoid",
            )
        )
    for suffix in ("q2", "q2_first_person", "q2_conditional"):
        questions.append(
            Question(
                f"{rule}__{suffix}",
                prompt.text(f"{rule}__{suffix}"),
                "candidate_turn",
            )
        )
    return _group(
        rule, ctx, questions, extra_blocks=blocks, num_predict_kind="general"
    )


def _single_verdict(rule: str, ctx: TurnContext, **kwargs) -> QuestionGroup:
    prompt = RULE_PROMPTS[rule]
    substitutions = kwargs.pop("substitutions", {})
    return _group(
        rule,
        ctx,
        [
            Question(
                rule,
                prompt.text(rule, **substitutions),
                kwargs.pop("evidence_block", "candidate_turn"),
                VERDICT_VALUES,
                kwargs.pop("quotes_on", None),
            )
        ],
        **kwargs,
    )


def build_states_future_policy(ctx: TurnContext) -> QuestionGroup:
    return _single_verdict("reaction_states_future_policy", ctx)


def build_withheld_fact_before_ask(ctx: TurnContext) -> QuestionGroup:
    """The handing-over question and the two exemptions.

    Delivery — are the key terms all there — is `missing_key_terms()` and stays
    in code: both errors of the two-step experiment were on that count, and none
    on reading an exemption. `q_handed` is the question of the owner's ruling of
    2026-09-09: it took over `pre_reveal_substitute_for_gated_fact` (did the turn
    supply wording of its own for a thing not revealed yet) and nothing else —
    ruling A of the same day keeps both exemptions as questions of their own, so
    `q_handed` asks only whether the turn fills the slot. It works the slot noun
    out first the way `q_names` does, and answers on whether the words it quotes
    could be used AS the thing: replayed without that, the deleted rule's text
    scored P .10 over 150 rows, calling deferrals, instructions and bare
    mentions of the topic wording of the turn's own.

    The three questions read three different blocks (owner ruling, 2026-09-09).
    `q_handed` quotes the handed or invented wording out of the candidate turn.
    `q_wording` asks for the wording *of this sentence*, which only the turn the
    user is replying to can have asked for. `q_names` asks whether the assistant
    ever asked what the thing is, and an assistant that asked two turns ago and
    got an answer to something else has still asked: the user handing the
    sentence over after that is not withholding it. So it reads the whole
    session's assistant turns, and its evidence is a verbatim span of any one of
    them.

    That is the on-request arm. The pre-reveal arm asks `q_handed` alone, under
    `## preamble_pre_reveal`: ruling A leaves the exemptions unread there, so
    asking them buys nothing, and the two blocks only they read go with them.
    """
    rule = "withheld_fact_before_ask"
    prompt = RULE_PROMPTS[rule]
    handed = Question(
        f"{rule}__q_handed",
        prompt.text(f"{rule}__q_handed"),
        "candidate_turn",
    )
    gate_blocks = {
        "gate_dictated_text": ctx.gate_dictated_text,
        "gate_key_terms": ", ".join(ctx.gate_key_terms),
        "gate_reveal_beat": ctx.gate_reveal_beat,
        "beat_id": ctx.beat_id,
        "key_terms_delivered": (
            "no, missing: " + ", ".join(ctx.missing_key_terms)
            if ctx.missing_key_terms
            else "yes"
        ),
    }
    if ctx.pre_reveal:
        # `compose_withheld_fact_before_ask` reads neither exemption on this arm
        # (owner ruling A, 2026-09-09), so neither is asked, and the two blocks
        # only they read — `pa_previous_turn`, `assistant_turns_so_far` — are not
        # printed to a call that has no question for them (audit finding H,
        # restored by the owner 2026-09-12).
        return QuestionGroup(
            rule=rule,
            shape="two_step",
            questions=(handed,),
            blocks={"candidate_turn": ctx.candidate_turn, **gate_blocks},
            preamble=prompt.text("preamble_pre_reveal"),
            num_predict_kind="wide",
        )
    return _group(
        rule,
        ctx,
        [
            handed,
            Question(
                f"{rule}__q_wording",
                prompt.text(f"{rule}__q_wording"),
                "pa_previous_turn",
            ),
            Question(
                f"{rule}__q_names",
                prompt.text(f"{rule}__q_names"),
                "assistant_turns_so_far",
            ),
        ],
        shape="two_step",
        extra_blocks={"assistant_turns_so_far": _joined(ctx.pa_turns), **gate_blocks},
        num_predict_kind="wide",
    )


def build_reaction_beat_off_attribute(ctx: TurnContext) -> QuestionGroup:
    """Owner ruling C-3 (2026-09-11): the frozen word table's finding, asked.

    `_REACTION_SURFACE_TERMS` in `scripts/validate_normal_event.py` takes no
    new rows; whether a shared word opens another attribute's surface is a
    reading, so it is asked here instead of matched there.
    """
    return _single_verdict(
        "reaction_beat_off_attribute",
        ctx,
        extra_blocks={"target_attribute": ctx.target_attribute},
    )


def build_reaction_names_setting(ctx: TurnContext) -> QuestionGroup:
    """The normal line's read of the forced line's `forced_setting_name_leak`."""
    return _single_verdict("reaction_names_setting", ctx)


# --------------------------------------------------------------- the composes


def _yes(verdicts: Mapping[str, Verdict], question_id: str) -> bool:
    verdict = verdicts.get(question_id)
    return bool(verdict and verdict.is_yes)


def compose_carries_an_ask(verdicts, ctx) -> bool:
    """True when the turn put something to the assistant."""
    return _yes(verdicts, "carries_an_ask")


def compose_asserts_unexecuted(verdicts, ctx) -> bool:
    """q1 alone: the turn claims an act `<executed_acts>` does not carry.

    There is no second half any more. Owner ruling B of 2026-09-10 made it the
    session boolean `pa_wrote` and the ruling of 2026-09-12 took that out: one
    flag cleared every claim on a turn as soon as anything at all had run, so
    the particular act a sentence names went unchecked. Code still owns the
    record — it builds the list — and the model is asked the one question a list
    cannot answer for itself.
    """
    return _yes(verdicts, "reaction_asserts_unexecuted_pa_action__q1")


def compose_misstates_contractions(verdicts, ctx) -> bool:
    """q1 ∧ ¬q2, and q2 is read only when q1 said there is a report.

    q2 answers `not_applicable` when q1 is no, so nothing here has to tell the
    two answers apart: the `and` never reaches q2 on that branch (audit finding
    I, 2026-09-12).
    """
    if not _yes(verdicts, "reaction_misstates_contractions__q1"):
        return False
    return not _yes(verdicts, "reaction_misstates_contractions__q2")


def compose_probe_fact_missing(verdicts, ctx) -> list[str]:
    """The fact keys this turn owes and has not said.

    An on-request fact is owed only once the assistant has asked for something;
    until then the beat's own `avoid` forbids saying it. That flag is the
    harness's (`_on_request_fact_keys` plus the previous PA turn), never the
    judge's.
    """
    missing = []
    for key in ctx.fact_values:
        if _yes(verdicts, f"probe_fact_missing:{key}"):
            continue
        owed = key.split("[", 1)[0] not in ctx.on_request_fact_keys or ctx.pa_asked
        if owed:
            missing.append(key)
    return missing


def compose_authorises_pa_action(verdicts, ctx) -> bool:
    """The experiment's boolean: the ban holds, uncarved, and a release lands.

    q1 ∧ ¬(qc1 ∨ … ∨ qcn) ∧ q2 ∧ ¬first_person ∧ ¬conditional. The model
    answers; every AND and NOT is here.
    """
    rule = "reaction_authorises_pa_action"
    if not _yes(verdicts, f"{rule}__q1"):
        return False
    if any(_yes(verdicts, qid) for qid in verdicts if "__qc" in qid):
        return False
    if not _yes(verdicts, f"{rule}__q2"):
        return False
    if _yes(verdicts, f"{rule}__q2_first_person"):
        return False
    return not _yes(verdicts, f"{rule}__q2_conditional")


def compose_states_future_policy(verdicts, ctx) -> bool:
    return _yes(verdicts, "reaction_states_future_policy")


def named_noun(reason: str) -> str:
    """The thing the judge says the assistant asked about.

    The prompt tells it to open `reason` with that name, then a colon, then one
    sentence, because the §3 schema has no field of its own for it. The first
    word run is what is read, so `noun:`, `noun —` and `noun.` all resolve;
    the separator is the prompt's business, not this function's. A `reason`
    with no word in it yields "", which fails the rule closed — the exemption
    below needs a noun that the quoted question actually contains.
    """
    words = re.findall(r"[\w-]+", reason or "")
    return words[0].casefold() if words else ""


def compose_withheld_fact_before_ask(verdicts, ctx) -> bool:
    """Pre-reveal: q_handed alone. On request: (delivered OR q_handed) AND NOT
    q_wording AND NOT asked_by_name.

    Before the reveal beat the exemptions do not apply (owner ruling A,
    2026-09-09): the turn must not hand a stand-in for the gated sentence over
    that early even when the assistant asked for it. Past the reveal beat the
    sentence is the actor's to give once asked, and the two exemptions are what
    excuse the handing over.

    Delivery is code's — `missing_key_terms()` — and `q_handed` is the judge's
    half of the same fact, the one that catches a turn handing over wording of
    its own before the reveal beat, where the code test has no real sentence to
    match against. Either one is a handing over; the two exemptions are what
    excuse it, and both keep their own question (owner ruling A, 2026-09-09).
    Folded into `q_handed` the wording exemption could not reach a turn the code
    had already called delivered.

    The judge saying "it asked" is not enough on its own — it will call any
    question about the task a question about the thing. It has to have named the
    thing and that name has to be in the question it quoted, which is what
    `message_quality._pa_question_names()` tests.
    """
    rule = "withheld_fact_before_ask"
    if ctx.pre_reveal:
        return _yes(verdicts, f"{rule}__q_handed")
    delivered = not ctx.missing_key_terms
    if not (delivered or _yes(verdicts, f"{rule}__q_handed")):
        return False
    if _yes(verdicts, f"{rule}__q_wording"):
        return False
    names = verdicts.get(f"{rule}__q_names")
    noun = named_noun(names.reason if names else "")
    asked_by_name = (
        names is not None
        and names.is_yes
        and bool(noun)
        and noun in normalise(names.evidence)
    )
    return not asked_by_name


def compose_reaction_beat_off_attribute(verdicts, ctx) -> bool:
    return _yes(verdicts, "reaction_beat_off_attribute")


def compose_reaction_names_setting(verdicts, ctx) -> bool:
    return _yes(verdicts, "reaction_names_setting")


# ------------------------------------------------------------- the catalogue


CATALOGUE: tuple[RuleSpec, ...] = (
    RuleSpec("carries_an_ask", build_carries_an_ask, compose_carries_an_ask),
    RuleSpec(
        "reaction_asserts_unexecuted_pa_action",
        build_asserts_unexecuted,
        compose_asserts_unexecuted,
    ),
    RuleSpec(
        "reaction_misstates_contractions",
        build_misstates_contractions,
        compose_misstates_contractions,
    ),
    RuleSpec("probe_fact_missing", build_probe_fact_missing, compose_probe_fact_missing),
    RuleSpec(
        "reaction_authorises_pa_action",
        build_authorises_pa_action,
        compose_authorises_pa_action,
    ),
    # Commented out, owner ruling 2026-09-13 ("先comment"): the rule is not
    # asked this round; gold, prompt and builder stay.
    # RuleSpec(
    #     "reaction_states_future_policy",
    #     build_states_future_policy,
    #     compose_states_future_policy,
    # ),
    RuleSpec(
        "withheld_fact_before_ask",
        build_withheld_fact_before_ask,
        compose_withheld_fact_before_ask,
    ),
    RuleSpec(
        "reaction_beat_off_attribute",
        build_reaction_beat_off_attribute,
        compose_reaction_beat_off_attribute,
    ),
    RuleSpec(
        "reaction_names_setting",
        build_reaction_names_setting,
        compose_reaction_names_setting,
    ),
)

BY_RULE: dict[str, RuleSpec] = {spec.rule: spec for spec in CATALOGUE}


# ------------------------------------------------------------------ rendering


def load_prompt(shape: str) -> tuple[str, str]:
    """(system, body) for one shape. The prompt files are the patch channel, so
    the text lives there rather than in a Python string."""
    text = (PROMPTS_DIR / f"{shape}.md").read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{shape}.md: must open with a '---' front-matter block")
    front, _, body = text[3:].partition("\n---\n")
    meta = yaml.safe_load(front) or {}
    system = str(meta.get("system") or "").strip()
    if not system:
        raise ValueError(f"{shape}.md: front matter must declare a `system: |` block")
    return system, body.lstrip("\n")


def render_questions(group: QuestionGroup) -> str:
    lines = []
    for question in group.questions:
        answers = " / ".join(f'"{value}"' for value in question.answers)
        blocks = " or ".join(f"<{name}>" for name in question.evidence_block_names)
        quoting = " or ".join(f'"{value}"' for value in (question.quotes_on or ()))
        # The grammar of a reference is in the system prompt, once. Per
        # question only the two facts that vary: which answers need one and
        # which block it may point at (audit 2026-09-12, "多不多").
        lines.append(
            f"[{question.question_id}]\n{question.text}\n"
            f"  answer with one of {answers}\n"
            + (
                f"  evidence: on {quoting} → {blocks}#N; otherwise empty"
                if quoting
                else "  evidence: always empty"
            )
        )
    return "\n\n".join(lines)


def render_blocks(group: QuestionGroup) -> str:
    """Every block, its sentences numbered — the numbers `evidence` points at.

    The one segmentation (`guards.number_sentences`) prints these and resolves
    the references, so what the judge counts and what the guard counts cannot
    drift apart.
    """
    return "\n\n".join(
        numbered_block(name, value)
        for name, value in group.blocks.items()
        if str(value).strip()
    )


def render(group: QuestionGroup) -> tuple[str, str]:
    """(system, prompt) for one question group."""
    system, body = load_prompt(group.shape)
    prompt = (
        body.replace("{{context}}", render_blocks(group))
        .replace("{{preamble}}", group.preamble)
        .replace("{{questions}}", render_questions(group))
    )
    return system, prompt


# --------------------------------------------------------------------- probe


_PROBE_CONTEXT = TurnContext(
    turn_uid="probe",
    pa_previous_turn=(
        "I've opened the working bibliography. What would you like me to change?"
    ),
    candidate_turn=(
        "Update the year field in the Sarpong entry to 2021, and leave the "
        "annotation as it stands."
    ),
)
PROBE_SYSTEM, PROBE_PROMPT = render(build_carries_an_ask(_PROBE_CONTEXT))
