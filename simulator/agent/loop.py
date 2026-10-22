"""Simulator agent loop built on the simulator's own AgentRunner.

Every judge call this module makes that reaches the endpoint goes out through
`asyncio.to_thread` (`begin_generation` and `last_evidence` read local state and
stay where they are). The judge client is synchronous `urllib` under a
process-wide `threading.Lock`
(`simulator/judge/client.py`), so on the event loop one call left every other
coroutine's in-flight HTTP unread for its whole length — a 60 s deadline floor,
doubled by one timeout failover, against a 600 s PA request budget. In a thread
the lock still serialises the calls, which is what one local judge box wants;
what it stops doing is freezing the loop. It matters from the moment more than
one session runs at a time.
"""

from __future__ import annotations

import re

import asyncio
from collections.abc import Iterable
from typing import Any

from simulator.agent.context import SimulatorContextBuilder
from simulator.agent.runner import AgentRunner, AgentRunSpec
from simulator.judge.catalogue import TurnContext
from simulator.judge.general import build_general_groups
from simulator.message_quality import (
    _asks_by_marker,
    _CONTRACTION_RE,
    beat_carries_authorisation,
    delivers_dictated_text,
    dictation_missing_key_terms_finding,
    missing_key_terms,
    missing_required_facts,
    quoted_span_not_in_pa_turn_finding,
    reaction_authorisation_clause,
    sentences_carrying,
    simulator_message_quality_findings,
    simulator_style_findings,
    # Non-blocking, owner ruling 2026-09-13: judged and recorded for
    # observation this round, never a finding — the only caller was the retry
    # note commented out below.
    # unexecuted_pa_action_clause,
    value_is_never_spoken_verbatim,
)
from simulator.output import (
    SimulatorOutput,
    SimulatorOutputContractError,
    parse_simulator_output,
)
from simulator.profiles import SimulatorProfile
from simulator.prompting import (
    beats_before,
    reception_carry_attributes,
)
from simulator.reply_facts import ReplyFacts
from simulator.schemas import OPENING_BEATS, Beat, PreferenceMatrix, SessionScript


#: The two findings the general form produces. `user_instruction` and
#: `avoid` are what the draft's own author wrote down as this beat's "must" and
#: "must not", so a violation is hard (owner ruling 2026-09-09: a rule that is
#: only recorded is not a rule).
BEAT_INSTRUCTION_NOT_FOLLOWED = "beat_instruction_not_followed"
BEAT_AVOID_VIOLATED = "beat_avoid_violated"

#: Hard findings that are retried but played once the budget is spent, with the
#: rule carried on the turn's diagnostics: each of them degrades the turn
#: without handing the PA a sentence the world will refuse, so the session
#: still measures something. The exception is
#: `reaction_asserts_unexecuted_pa_action` on the compressed reaction beat,
#: where that one exchange IS the measurement and the finding blocks.
#: `reaction_authorises_pa_action` is not on this list either: it inherits the
#: blocking consequence of the `authorised_before_its_beat` regex it replaced
#: (owner ruling 2026-09-09, "不改后果") — a go-ahead the actor keeps repeating
#: past the retry budget hands the PA a release the session never meant to
#: give, so the step dies instead of playing it.
PLAYED_AFTER_BUDGET: frozenset[str] = frozenset(
    {
        "gate_cue_missing",
        "withheld_fact_before_ask",
        "reaction_asserts_unexecuted_pa_action",
        "reaction_misstates_contractions",
        # A turn that argued with its own direction is degraded, but it hands
        # the assistant nothing the world will refuse, so the session still
        # measures something.
        BEAT_INSTRUCTION_NOT_FOLLOWED,
        BEAT_AVOID_VIOLATED,
        # Hard from Phase 3 (裁定 vi): it was a diagnostic while a pattern
        # decided it, and the judge decides it now.
        "reaction_states_future_policy",
    }
)

# Owner ruling 2026-09-14 ("只记"): every judgement about the turn is recorded,
# none triggers a rewrite or a void this round. Remove a name to make it
# blocking again.
RECORD_ONLY_FINDINGS: frozenset[str] = frozenset(
    {
        # The general form: the beat's own instruction, its `avoid`
        # clauses, and the cue.
        BEAT_INSTRUCTION_NOT_FOLLOWED,
        BEAT_AVOID_VIOLATED,
        "gate_cue_missing",
        # The judged per-rule checks on the reception and reaction beats.
        "reaction_asserts_unexecuted_pa_action",
        "reaction_authorises_pa_action",
        "reaction_misstates_contractions",
        # Not asked this round (the rule is commented out at its call site);
        # named here so the set answers the question either way.
        "reaction_states_future_policy",
        # The judged content checks: what the turn had to say, and what it had
        # to keep back.
        "probe_fact_missing",
        "withheld_fact_before_ask",
        # dictation_missing_key_terms is blocking again (owner ruling
        # 2026-09-15): a dropped term is the one judgement the run cannot
        # absorb, because the PA is then holding a sentence the store will
        # refuse and the session measures nothing either way. It retries with
        # the note naming the missing words and voids past the budget.
        # Mechanical judgements about what the turn said, from
        # `simulator_message_quality_findings`.
        "names_a_file_you_do_not_know",
        "identifier_volunteered_unprompted",
        # Every style finding, from `simulator_style_findings`.
        "grades_pa_before_acting",
        "reviewer_voice_in_message",
        # sentence_over_35_words is blocking again (owner ruling 2026-09-14: 19/19 correct live).
    }
)


def record_only(
    findings: list[str], output: SimulatorOutput, detail: dict[str, str] | None = None
) -> list[str]:
    """Move every `RECORD_ONLY_FINDINGS` name onto the turn's diagnostics.

    The one place a record-only finding stops being a finding. Returns what is
    left of `findings` — what still costs a retry — and puts each name it took
    on `output.diagnostics`, with whatever the finding already carries on
    `output.diagnostics_detail`, the shape `probe_fact_missing` uses. A finding
    that carries no detail of its own is recorded by name alone rather than
    given an invented one.
    """
    kept: list[str] = []
    for finding in findings:
        if finding not in RECORD_ONLY_FINDINGS:
            kept.append(finding)
            continue
        if finding not in output.diagnostics:
            output.diagnostics.append(finding)
        carried = (detail or {}).get(finding)
        if carried and finding not in output.diagnostics_detail:
            output.diagnostics_detail[finding] = [carried]
    return kept



def _fact_strings(value: Any) -> list[str]:
    """Every string value inside task_facts, including nested dicts and lists."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _fact_strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _fact_strings(item)]
    return []


def _instruction_names_fact(instruction: str, key: str) -> bool:
    """True when a beat instruction names a `task_facts` key.

    Both forms the drafts use: the interpolation `$task_facts.<key>` and the
    bare key inside the prose ("says the student's name is student_name").
    """
    return bool(
        re.search(rf"\$task_facts\.{re.escape(key)}(?![\w])", instruction)
        or re.search(rf"(?<![\w.]){re.escape(key)}(?![\w])", instruction)
    )


#: The draft saying a fact waits for the assistant to ask for it.
_ON_REQUEST_RE = re.compile(
    r"\b(?:only )?(?:when|once|if|until|after|unless) (?:the )?(?:PA|assistant|it) asks\b"
    r"|\bunprompted\b|\b(?:when|if|until|unless) asked\b"
    r"|\bhands? (?:any of )?(?:them|it|these|those|the rest) over only\b"
    r"|\bon request\b",
    re.I,
)


def _reveal_at_fact_keys(session: SessionScript, beat: Beat) -> list[str]:
    """`task_facts` keys this beat holds back until the PA asks for them.

    The beat's own `reveal_at` keys, less the ones it also `uses` or names in
    its instruction — those two are the beat's content, not a fact it is
    sitting on. Live `acc_009` beat_1 keeps all three of its `reveal_at` facts
    "and hands any of them over only when the PA asks for it", with an `avoid`
    against volunteering them; `probe_fact_missing` reads this so it does not
    demand them from a turn that is correctly withholding.
    """
    # On a triggered session `reveal_at` on the played opening is a hiding
    # device, not a promise: it keeps the notice facts off the send branch,
    # and the suppress opening only "may mention the notice in passing" (live
    # verify007__suppress t6 said the shutoff and nothing else, and the
    # diagnostic fired on the nine facts it was never asked to say). Only
    # `uses` and the instruction's own names count there.
    if session.proactive_trigger is not None and beat.beat in OPENING_BEATS:
        return []
    return [
        key
        for key, beat_id in (session.reveal_at or {}).items()
        if beat_id == beat.beat
        and key in session.task_facts
        and key not in beat.uses
        and not _instruction_names_fact(beat.actor_instruction or "", key)
    ]


def _on_request_fact_keys(session: SessionScript, beat: Beat) -> list[str]:
    """The reveal_at keys this beat holds back until the assistant asks.

    Only a beat whose own prose says so holds a fact on request ("hands any of
    them over only when the PA asks for it", "unprompted"). A reveal beat that
    simply delivers the fact — live `acc_023` pose_both_topics, "giving the
    dictated text from task_facts in his own words" — owes it from its first
    turn; regress19 read that beat as on-request and `withheld_fact_before_ask`
    stripped the dictation out of the probe turn.
    """
    if not _ON_REQUEST_RE.search(f"{beat.actor_instruction or ''} {beat.avoid or ''}"):
        return []
    return _reveal_at_fact_keys(session, beat)


def _required_fact_values(session: SessionScript, beat: Beat) -> dict[str, str]:
    """`task_facts` values this beat has to actually say, keyed by fact key.

    Three sources, all the draft's own statement of what the beat is for: the
    keys whose `reveal_at` is this beat — the beat where the user first has
    them, the mirror of `gated_fact_keys` — the keys the beat's `uses` lists,
    and the keys the beat's own instruction names. `uses` is the only link for
    a fact the instruction refers to without naming (live `acc_002` beat 1
    says "the walking limit from task_facts", never `max_walking_minutes`).
    A value nobody says as written is dropped: a fixture
    path, because requiring it would be a rule against the path-fact rule
    (`acc_011` `prep_list_path`), and a machine literal such as `2026-11-10`,
    which a person says as "November 10". A non-string fact contributes each of
    its strings under an indexed key.
    """
    # The `reveal_at` group, less the keys the other two groups pick up again
    # below — same set either way (see `_reveal_at_fact_keys`, which owns the
    # triggered-opening carve-out both derivations need).
    keys = _reveal_at_fact_keys(session, beat)
    keys += [
        key
        for key in beat.uses
        if key not in keys and key in session.task_facts
    ]
    keys += [
        key
        for key in session.task_facts
        if key not in keys
        and _instruction_names_fact(beat.actor_instruction or "", key)
    ]
    required: dict[str, str] = {}
    for key in keys:
        values = [
            text
            for text in _fact_strings(session.task_facts.get(key))
            if text.strip() and not value_is_never_spoken_verbatim(text)
        ]
        if len(values) == 1:
            required[key] = values[0]
        else:
            required.update(
                {f"{key}[{index}]": text for index, text in enumerate(values)}
            )
    return required


def _judge_probe_fact_missing(
    judge: Any,
    output: SimulatorOutput,
    *,
    session: SessionScript,
    beat: Beat,
    required_values: dict[str, str],
    on_request_keys: Iterable[str],
    user_history: list[str],
    last_pa_message: str,
) -> list[str]:
    """The judge's own list of the facts this turn owes and has not said.

    One question per fact, all in one call, and every flag the answer composes
    against is the harness's: which keys are held on request
    (`_on_request_fact_keys`) and whether the assistant has asked for anything
    (`_latest_user_message`). The judge is asked only whether the words are in
    the turn.

    Hard since the owner's ruling of 2026-09-13, where it was a diagnostic
    before: the keys go back to the caller, which turns them into a finding
    like any other judge finding — the turn is said again with the missing
    values named in the note, and a fact still missing when the retries are
    gone voids the session. The judge is the only decider: with the judge off
    the rule is not checked.

    Returns the fact keys this turn still owes; the keys also stay on
    `diagnostics_detail`, so a read of a rejected attempt has them.
    """
    ctx = TurnContext(
        turn_uid=f"{session.session_id}:{beat.beat}",
        pa_previous_turn=last_pa_message,
        candidate_turn=output.message,
        previous_user_turns=tuple(user_history),
        fact_values=dict(required_values),
        beat_intent=beat.actor_instruction or "",
        on_request_fact_keys=frozenset(on_request_keys),
        pa_asked=_asks_by_marker(last_pa_message),
    )
    decided = judge.judge("probe_fact_missing", ctx)
    if decided:
        if "probe_fact_missing" not in output.diagnostics:
            output.diagnostics.append("probe_fact_missing")
        output.diagnostics_detail["probe_fact_missing"] = decided
    else:
        if "probe_fact_missing" in output.diagnostics:
            output.diagnostics.remove("probe_fact_missing")
        output.diagnostics_detail.pop("probe_fact_missing", None)
    return list(decided or [])


def _judge_withheld_fact_before_ask(
    judge: Any,
    output: SimulatorOutput,
    *,
    session: SessionScript,
    beat: Beat,
    pa_history: list[str],
    last_pa_message: str,
    pre_reveal: bool,
) -> str | None:
    """Hard: the turn handed the gated sentence over before it was asked for.

    One rule for both ends of the gated beat since the owner's ruling of
    2026-09-09, so it is called on the UNION of the two windows the two rules
    used to have and on nothing wider: an armed script, and either a beat
    earlier than the gate's reveal beat whose turn does not deliver the real
    dictated text (where wording is necessarily invented), or a beat whose
    `reveal_at` prose holds the fact on request, with gate key terms and a
    previous PA turn to read (where the sentence is the actor's to give, but
    only when asked).

    Delivery stays code — `missing_key_terms()` is the store's own test, and
    its conclusion goes in as context (`compose_withheld_fact_before_ask` reads
    it off `ctx.missing_key_terms`). The judge answers the two questions: did
    the turn hand the thing over unasked, and did the assistant ever ask what
    the thing is.
    """
    ctx = TurnContext(
        turn_uid=f"{session.session_id}:{beat.beat}",
        pa_previous_turn=last_pa_message,
        candidate_turn=output.message,
        pa_turns=tuple(pa_history),
        gate_dictated_text=session.gate_dictated_text or "",
        gate_key_terms=tuple(session.gate_key_terms or ()),
        gate_reveal_beat=session.gate_reveal_beat or "",
        beat_id=beat.beat,
        missing_key_terms=tuple(
            missing_key_terms(output.message, session.gate_key_terms)
        ),
        pre_reveal=pre_reveal,
    )
    if judge.judge("withheld_fact_before_ask", ctx):
        return "withheld_fact_before_ask"
    return None


def _judge_general(
    judge: Any,
    findings: list[str],
    *,
    session: SessionScript,
    beat: Beat,
    output: SimulatorOutput,
    last_pa_message: str,
    user_history: list[str],
    cue_owed: bool,
    first_turn_in_beat: bool,
    must_ask: bool,
) -> dict[str, str]:
    """The beat's own direction text, judged over one call per chunk.

    Three things and no more: the beat's `user_instruction` on the turn that
    binds it, each `avoid` clause on its own, and — on the cue beat's opening
    turn — the cue. They go out as the head call and one call per
    `GENERAL_AVOID_CHUNK` clauses; each spends one group against the turn
    budget, which is what the budget counts. Everything cross-session stays per-rule in the
    catalogue: those rules were each added after a regression, and the beat's
    own `avoid` has no grain for them.

    `gate_cue_missing` is asked here rather than per-rule, and the regex that
    used to answer it is gone: the judge is its sole decider. Its consequence
    is unchanged — still hard, still `PLAYED_AFTER_BUDGET`.

    An instruction covers a beat and a beat runs over several turns, so it is
    judged only on the turn it binds: `first_turn_in_beat`, which is also the
    only turn of a one-turn beat. Eight of the 28 false alarms a
    measurement counted were the judge reading a beat-wide instruction as owed
    by every turn of the beat.

    Returns the clause behind each finding, so the retry note can quote the
    author's own words back instead of naming a rule.
    """
    ctx = TurnContext(
        turn_uid=f"{session.session_id}:{beat.beat}",
        pa_previous_turn=last_pa_message,
        candidate_turn=output.message,
        previous_user_turns=tuple(user_history),
        beat_intent=beat.actor_instruction or "",
        beat_avoid=beat.avoid or "",
        beat_id=beat.beat,
        beat_user_instruction=beat.actor_instruction or "",
        cue=session.gate_cue or "",
        cue_owed=cue_owed,
        must_ask=must_ask,
        instruction_binds_this_turn=first_turn_in_beat,
    )
    verdicts: dict[str, Any] = {}
    blocks: dict[str, str] = {}
    for group in build_general_groups(ctx):
        verdicts.update(judge.ask(group))
        blocks.update(group.blocks)
    clauses: dict[str, str] = {}
    for question_id, verdict in verdicts.items():
        if verdict.verdict != "violated":
            continue
        if question_id == "gate_cue_missing":
            rule, clause = "gate_cue_missing", ctx.cue
        elif question_id == "user_instruction":
            rule = BEAT_INSTRUCTION_NOT_FOLLOWED
            clause = ctx.beat_user_instruction
        else:
            rule = BEAT_AVOID_VIOLATED
            clause = blocks.get(
                f"avoid::{question_id.removeprefix('avoid_')}", ""
            )
        clauses.setdefault(rule, clause)
        if rule not in findings:
            findings.append(rule)
    return clauses


class _NoTools:
    """Empty tool registry. The simulator is a user actor and calls no tools."""

    def get_definitions(self) -> list[dict[str, Any]]:
        return []

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        raise RuntimeError(f"simulator has no tools: name={name!r}")


def _latest_user_message(history: list[dict] | None) -> str:
    for message in reversed(history or []):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _declaring_beat_index(session: SessionScript) -> int | None:
    """Index of the beat that declares the session's measured attribute."""
    return next(
        (i for i, item in enumerate(session.beats) if item.active_skills), None
    )


def _authorising_beat_index(session: SessionScript) -> int | None:
    """Index of the beat whose own instruction hands the gated work over.

    That is the beat the go-ahead belongs to — `acc_039` `beat_5`, which
    "gives the go-ahead for the Dr. Patel booking at the clinic" — and it is
    read off the beats with the same predicate that exempts the beat itself,
    so the arming and the exemption cannot disagree.

    The LAST match, not the first: a reaction beat that branches on whether
    the assistant asked for a go-ahead reads like an authorising beat from a
    distance, and taking the first match let it close the window early. Live
    regress14c `acc_039`: beat_4's branch wording disarmed the check from
    beat_4 on, one beat before beat_5, the beat that actually gives it.
    """
    return next(
        (
            i
            for i, item in reversed(list(enumerate(session.beats)))
            if beat_carries_authorisation(f"{item.actor_instruction} {item.goal}")
        ),
        None,
    )


def authorising_beat_instruction(session: SessionScript) -> str:
    """Instruction and goal of the beat that carries the go-ahead, or "".

    What the gate is FOR, so a caller can tell a hand-over of the gated act
    from a hand-over of some other act an earlier beat owns.
    """
    index = _authorising_beat_index(session)
    if index is None:
        return ""
    item = session.beats[index]
    return f"{item.actor_instruction} {item.goal}"


def before_authorising_beat(session: SessionScript, beat: Beat) -> bool:
    """True when this beat comes before the beat that carries the go-ahead.

    When no beat carries one, the boundary sits one past the declaring beat,
    so the probe beat itself stays armed (owner ruling 2026-09-08): a session
    with nothing to authorise has no beat where a go-ahead belongs, and the
    probe is exactly where one was slipping through — `acc_011`
    `b5_probe_prep_list` lost its arming when the authorising-beat predicate
    stopped matching b6, and regress9 `acc_001` t22 gave the go-ahead on the
    declaring beat.
    """
    order = {item.beat: i for i, item in enumerate(session.beats)}
    current = order.get(beat.beat)
    gate = _authorising_beat_index(session)
    if gate is None:
        declaring = _declaring_beat_index(session)
        gate = None if declaring is None else declaring + 1
    return current is not None and gate is not None and current < gate


def _delivers_gated_dictation(
    session: SessionScript, beat: Beat, message: str
) -> bool:
    """True when this turn delivers the sentence the session gate is keyed on.

    False on an unarmed script — a session with no gate, or a caller that never
    set the gate fields — and false before the reveal beat, where the fact is
    withheld from the prompt entirely (`simulator.prompting.gated_fact_keys`)
    and a turn that carried it would be an invention, not a
    dictation. From the reveal beat onwards every delivery counts, on whatever
    beat it lands: live regress15 `acc_039` was asked for the reason again on
    `beat_6`, five beats past the reveal, and restated it there.
    """
    if not session.gate_key_terms or not session.gate_dictated_text:
        return False
    order = {item.beat: index for index, item in enumerate(session.beats)}
    reveal = order.get(session.gate_reveal_beat or "")
    here = order.get(beat.beat)
    if reveal is None or here is None or here < reveal:
        return False
    return delivers_dictated_text(message, session.gate_dictated_text)


def _gated_fact_key(session: SessionScript) -> str:
    """The `task_facts` key the session gate is keyed on, or "".

    `SessionScript` carries the gate as its three enforcement fields, not as the
    key it came from, so the key is read back off the value — the harness sets
    `gate_dictated_text` from `task_facts[gate.task_fact]`, so the two are the
    same string. What the key is wanted for is its noun: `checkup_reason_the_
    user_dictates` is a reason, and a pre-reveal turn that says what the reason
    is has handed a substitute over.
    """
    if not session.gate_dictated_text:
        return ""
    return next(
        (
            key
            for key, value in session.task_facts.items()
            if isinstance(value, str) and value == session.gate_dictated_text
        ),
        "",
    )


def _before_gate_reveal(session: SessionScript, beat: Beat) -> bool:
    """True on a beat EARLIER than the gate's reveal beat, on an armed script.

    The complement of the window `_delivers_gated_dictation` works in, and
    unarmed on the same terms: a session with no gate, or a caller that never
    set the gate fields, has no pre-reveal window at all.
    """
    if not session.gate_key_terms or not session.gate_dictated_text:
        return False
    order = {item.beat: index for index, item in enumerate(session.beats)}
    reveal = order.get(session.gate_reveal_beat or "")
    here = order.get(beat.beat)
    if reveal is None or here is None:
        return False
    return here < reveal


def _from_gate_reveal(session: SessionScript, beat: Beat) -> bool:
    """True on the gate's reveal beat or any beat after it, on an armed script.

    Where the terms are already the actor's to protect: it has said the
    sentence, or is about to, and from here on the store refuses any write on
    the gate tool that drops one of them.
    """
    if not session.gate_key_terms or not session.gate_dictated_text:
        return False
    order = {item.beat: index for index, item in enumerate(session.beats)}
    reveal = order.get(session.gate_reveal_beat or "")
    here = order.get(beat.beat)
    if reveal is None or here is None:
        return False
    return here >= reveal


def _script_dictates(
    session: SessionScript, beat: Beat, user_history: Iterable[str] = ()
) -> bool:
    """True when the beat's own script still owes the sentence the gate keys on.

    The other way a turn owes the key terms. `_delivers_gated_dictation` reads
    the message, so a turn that never says the sentence is never armed — and a
    beat whose `user_instruction` dictates the title is exactly the beat where
    not saying it is the failure: live `fin_05` was scripted to ask for a hold
    "titled Headshot pickup run" and asked for one titled "Headshot print
    counter pickup", which no reading of the message is a delivery of, so
    nothing fired and the PA had no wording the store would take.

    Same window and same arming conditions as the delivery arm — an unarmed
    script checks nothing, and before the reveal beat the fact is out of the
    prompt entirely — so `_from_gate_reveal` carries both. What is read is the
    beat's own prose rather than the whole draft: a beat whose `avoid` forbids
    volunteering the sentence does not dictate it, and stays unarmed until it
    is delivered (live regress15 `acc_009` t2), as does a reaction beat, whose
    script reacts and dictates nothing.

    It disarms on the first turn of the session that carries every key term
    (owner, 2026-09-15). A dictating beat is otherwise armed on all of its
    turns, and the dictation only has to land once: replayed over the four
    finished user_b runs that cost 10 false positives on 6 probes, every one a
    follow-up or a closing after the title had already been said correctly —
    "That works, thanks." failed for not repeating "Chair run 5D". Once the
    session has said it, the beat has done what it was scripted to do.
    """
    if not _from_gate_reveal(session, beat) or not delivers_dictated_text(
        beat.actor_instruction, session.gate_dictated_text
    ):
        return False
    return all(
        missing_key_terms(turn, session.gate_key_terms) for turn in user_history
    )


def _is_reaction_beat(session: SessionScript, beat: Beat) -> bool:
    """True for the beat right after a beat that declares skills.

    Same shape the loop's collapse guard protects: it declares nothing itself,
    carries the reception rules of the beat that did the asking, and is where
    the surface is read.

    On a triggered compressed session that shape does not hold and this cannot
    be derived from `session.beats` at all: the probe was the PA's own trigger
    turn, the played opening IS the reaction, it is the first beat entered, and
    it declares the target attribute by contract. Both halves of the derivation
    below reject it — `if not current` excludes index 0, `not beat.active_skills`
    excludes a declaring beat — so the reception relaxations were dead on both
    openings (live regress14c `acc_007`). The loop stamps the answer on the
    script as it enters the beat; take it from there.
    """
    if session.compressed_reaction_beat is not None:
        return session.compressed_reaction_beat == beat.beat
    order = {item.beat: i for i, item in enumerate(session.beats)}
    current = order.get(beat.beat)
    if not current:  # None, or the opening beat, which follows nothing
        return False
    return bool(session.beats[current - 1].active_skills) and not beat.active_skills


class SimulatorAgentLoop:
    """One simulated-user agent turn with simulator-specific output contract."""

    name = "nanobot"

    def __init__(
        self,
        provider: Any,
        profile: SimulatorProfile,
        matrix: PreferenceMatrix,
        *,
        actor_skill_specs_root=None,
        max_iterations: int = 1,
        record_memory: bool = True,
        output_contract_retries: int = 3,
    ) -> None:
        if output_contract_retries < 0:
            raise ValueError("output_contract_retries must be non-negative")
        self.provider = provider
        self.profile = profile
        self.matrix = matrix
        self.max_iterations = max_iterations
        self.record_memory = record_memory
        self.output_contract_retries = output_contract_retries
        self.pending_memory_exchanges: list[dict] = []
        self.memory_model: str | None = None
        self.memory_timestamp: str | None = None
        self.runner = AgentRunner(provider)
        self.context = SimulatorContextBuilder(
            profile,
            matrix,
            actor_skill_specs_root=actor_skill_specs_root,
        )

    def get_default_model(self) -> str:
        return self.provider.get_default_model()

    async def generate_turn(
        self,
        session: SessionScript,
        beat: Beat,
        *,
        history: list[dict] | None = None,
        model: str | None = None,
        first_turn_in_beat: bool = True,
        # The harness already decided whether this turn owes an ask
        # (`SimulatorLoop.require_ask`); the judge is told, never asked.
        must_ask: bool = False,
        benchmark_timestamp: str | None = None,
        held_items: Iterable[dict[str, str]] = (),
        pa_executed_tools: Iterable[str] = (),
        pa_tool_log: Iterable[str] = (),
        # A delivered proactive outreach is a PA action the tool record cannot
        # show: the send happens outside it. It reaches `TurnContext` as its
        # own flag, and joins the list `ctx.executed_acts` hands the judge.
        pa_outreach_sent: bool = False,
        judge: Any | None = None,
        # The actor's private record of what the latest PA reply did (owner
        # 2026-09-17); rendered into the trigger, never into the judge's input.
        reply_facts: ReplyFacts | None = None,
    ) -> SimulatorOutput:
        if beat.line:
            raise SimulatorOutputContractError(["fixed_beat_line_bypasses_contract"])

        # `first_turn_in_beat` is decided by the session loop, which owns the
        # per-beat turn counter; this object is rebuilt per turn and cannot
        # remember it. It stays fixed across every retry attempt below.
        request_messages = self.context.build_messages(
            session,
            beat,
            history,
            first_turn_in_beat=first_turn_in_beat,
            held_items=held_items,
            reply_facts=reply_facts,
        )
        rejected_attempts: list[dict[str, Any]] = []
        reaction_beat = _is_reaction_beat(session, beat)
        # Which tools the PA has actually run this session. Only the session
        # loop sees them, so they arrive as an argument; an empty tuple (the
        # prompt-only path, and every caller that does not pass them) reads as
        # "nothing was executed", so the check below is armed on the beat that
        # takes the reaction and nowhere else.
        executed_tools = list(pa_executed_tools)
        # The same calls as lines the judge can read — name, arguments, status.
        # The names above stay the mechanical clause's input; this is the
        # `<tool_log>` block, and a name with no arguments cannot say which
        # record a write touched (live regress22 `acc_006`).
        tool_log = tuple(pa_tool_log)
        # The window `simulator.loop` already computed the diagnostic over: a
        # beat carrying reception rules that declares none of its own, OR the
        # compressed reaction beat itself.
        #
        # The second disjunct has to stand on its own, and a conjunction here
        # is a no-op. On a triggered compressed session the played opening is
        # the first beat entered, so `reception_carry_attributes` returns `[]`
        # (nothing walked before it) and the attribute it does carry is the one
        # it declares itself, which the carry subtracts. `bool(...)` is
        # therefore false on exactly the beat that IS the reaction, and the
        # reception checks below were dead on every triggered shape —
        # regress21 `acc_007` among them. `_is_reaction_beat` (445-467) already
        # reads the stamp this way; this line was the one that had not caught
        # up.
        reception_attributes = reception_carry_attributes(session, beat)
        reception_beat = (
            bool(reception_attributes) and not beat.active_skills
        ) or session.compressed_reaction_beat == beat.beat
        required_values = _required_fact_values(session, beat)
        on_request_fact_keys = _on_request_fact_keys(session, beat)
        # The cue is owed by the beat's OPENING turn: the drafts ask for it in
        # its own sentence there, and demanding it again on the beat's later
        # turns would only have the actor repeat the same phrase. Unarmed when
        # the harness set no cue beat — live `acc_023` states its cue in the
        # scenario alone, on no beat of its own.
        cue_owed = bool(
            session.gate_cue
            and session.gate_cue_beat == beat.beat
            and first_turn_in_beat
        )
        # Commented out with the judge half of withheld_fact_before_ask (owner
        # ruling 2026-09-14, see the call site below): `on_request_arm` was its
        # only reader. Uncomment it together with that block.
        # gated_fact_key = _gated_fact_key(session)
        selected_model = model or self.get_default_model()
        for attempt in range(1, self.output_contract_retries + 2):
            if judge is not None and judge.enabled:
                # One candidate generation, one question-group budget.
                judge.begin_generation(
                    f"{session.session_id}:{beat.beat}:a{attempt}", beat=beat.beat
                )
            result = None
            for empty_retry in range(5):
                result = await self.runner.run(
                    AgentRunSpec(
                        initial_messages=request_messages,
                        tools=_NoTools(),
                        model=selected_model,
                        max_iterations=self.max_iterations,
                    )
                )
                if result.final_content is not None:
                    break
                # The model intermittently returns an empty final message;
                # wait and ask again before giving up on the whole run.
                await asyncio.sleep(3 * (empty_retry + 1))
            if result.final_content is None:
                raise RuntimeError(
                    f"Simulator agent returned no final content after retries "
                    f"(stop_reason={result.stop_reason!r})"
                )
            output = parse_simulator_output(result.final_content)
            # The actor answered with a tool call although the simulator
            # registers none (`runner.py`, which reports it instead of raising).
            # The names go on the attempt's own record, and the finding below
            # is added whatever the text parsed to: a tool call is unusable
            # even when three well-formed blocks came with it.
            tool_call_error = (
                result.error or "" if result.stop_reason == "tool_call" else ""
            )
            if tool_call_error:
                output.parse_warnings.append(tool_call_error)
            dictated_texts = [
                value
                for value in session.task_facts.values()
                if isinstance(value, str)
            ]
            # In simulator history the PA speaks in the "user" role.
            pa_history = [
                str(message.get("content") or "")
                for message in (history or [])
                if message.get("role") == "user"
            ]
            user_history = [
                str(message.get("content") or "")
                for message in (history or [])
                if message.get("role") == "assistant"
            ]
            findings = simulator_message_quality_findings(
                output.message,
                output.parse_warnings,
                beat_names=[b.beat for b in session.beats],
                known_texts=[
                    *_fact_strings(session.task_facts),
                    *(
                        str(message.get("content") or "")
                        for message in (history or [])
                        if message.get("role") in {"assistant", "user"}
                    ),
                ],
                fact_identifiers=_fact_strings(session.task_facts),
                last_pa_message=_latest_user_message(history),
                dictated_texts=dictated_texts,
                pa_messages=pa_history,
            )
            if (
                tool_call_error
                and "simulator_output_format_invalid" not in findings
            ):
                findings.append("simulator_output_format_invalid")
            # Hard, and armed on the turns that actually deliver the dictation:
            # any turn from the reveal beat onwards whose message says the gated
            # sentence, verbatim or in the user's own words. Only when the
            # harness armed the script with the gate — an unarmed script checks
            # nothing.
            #
            # Not "the first turn of the reveal beat", which is what the
            # `require_ask` shape would suggest, because `reveal_at` marks where
            # a fact is UNLOCKED, not where it is said. Replayed over the 1156
            # simulator turns of the five gated sessions, the first-turn rule
            # aborts regress15 `acc_009` at t2 — a turn whose own draft `avoid`
            # forbids volunteering the sentence, which the user then dictates
            # correctly at t8 — and misses the case this check exists for,
            # `acc_039` t55 on `beat_6`, five beats past the reveal. Arming on
            # delivery catches t55 and nothing else in regress15.
            #
            # The second arm is the beat whose OWN script dictates the sentence
            # (owner, 2026-09-15). Reading the message alone leaves the one case
            # the gate cannot survive unchecked: a turn that paraphrases the
            # title away is not a delivery, so nothing fires, and the PA is left
            # with no wording to write — live `fin_05`. It is spent once the
            # session says the sentence, so the beat's later turns are free.
            dictation_armed = _delivers_gated_dictation(
                session, beat, output.message
            ) or _script_dictates(session, beat, user_history)
            if dictation_armed:
                dictation_finding = dictation_missing_key_terms_finding(
                    output.message, key_terms=session.gate_key_terms
                )
                if dictation_finding:
                    findings.append(dictation_finding)
            # The other half of the same window: the gated sentence is the
            # assistant's to ask for, and a turn that hands it over anyway has
            # answered the elicitation probe on the assistant's behalf. One rule
            # for both ends of the beat since the owner's ruling of 2026-09-09,
            # so it is armed on the UNION of the two windows it was split
            # across, and no wider. Before the reveal the gated fact is not in
            # the prompt, so wording handed over there is invented — and the
            # closed gate is what makes the PA ask for it — which is the first
            # arm, on a turn that does not deliver the real dictated text. The
            # second is the reveal beat whose own prose holds the fact on
            # request, where there is a real sentence to withhold and a PA turn
            # to have withheld it from. Not an abort rule: past the budget the
            # turn is played.
            pre_reveal_arm = _before_gate_reveal(
                session, beat
            ) and not delivers_dictated_text(
                output.message, session.gate_dictated_text
            )
            # Commented out, owner ruling 2026-09-14: the judge half of
            # withheld_fact_before_ask is off — 29/29 live verdicts false, 0
            # real leaks, ~13 s per pre-reveal turn. The key-term code check
            # stays. Uncomment to restore.
            # on_request_arm = (
            #     gated_fact_key in on_request_fact_keys
            #     and bool(session.gate_key_terms)
            #     and bool(_latest_user_message(history).strip())
            # )
            # if (
            #     judge is not None
            #     and judge.enabled
            #     and (pre_reveal_arm or on_request_arm)
            # ):
            #     withheld_finding = _judge_withheld_fact_before_ask(
            #         judge,
            #         output,
            #         session=session,
            #         beat=beat,
            #         pa_history=pa_history,
            #         last_pa_message=_latest_user_message(history),
            #         pre_reveal=pre_reveal_arm,
            #     )
            #     if withheld_finding:
            #         findings.append(withheld_finding)
            #
            # The code half stays and is recorded below, past the point where
            # `output.diagnostics` is rebuilt.
            pre_reveal_key_terms_delivered = bool(
                pre_reveal_arm
                and session.gate_key_terms
                and not missing_key_terms(output.message, session.gate_key_terms)
            )
            # Hard wherever the reaction is taken (owner ruling 2026-09-08: an
            # invented PA action is an instrument error, and an after-the-fact
            # claim the record can settle is hard). The reaction beat is the
            # measurement, so a reaction to work the assistant never did scores
            # the attribute against a fiction — live regress16 `acc_006` t7
            # credited the PA with reformatting the bibliography into APA style
            # after a turn that had only read two documents, and the turn was
            # delivered because the check only ran in the session loop, after
            # the retry budget was gone. The window was the compressed beat
            # alone for one round; over regress12-19 the three uncompressed hits
            # are all true (regress14c `acc_039` t17, regress18 `acc_009` t13,
            # regress18 `acc_039` t23), so it is armed on all of them.
            # The judge is the only decider on both: the semantic
            # patterns they used to run on are gone, so with `MEMPA_JUDGE=off`
            # the reception beat is not judged at all. The second of them is
            # still narrowed by the beat's own `avoid` — where the draft says
            # this turn hands nothing over, a go-ahead spoken here releases
            # work no beat authorises (live regress20 `acc_002` t6 "so I will
            # proceed with that" after the PA closed "Want me to book it?",
            # and it booked) — but the ban and its carve-outs are read out of
            # `avoid` by the judge's own questions. It BLOCKS: it took over the
            # deleted `authorised_before_its_beat` regex and its consequence
            # with it, so it is not in `PLAYED_AFTER_BUDGET`.
            asserts_unexecuted: list[str] = []
            if reception_beat and judge is not None and judge.enabled:
                reception_ctx = TurnContext(
                    turn_uid=f"{session.session_id}:{beat.beat}",
                    pa_previous_turn=_latest_user_message(history),
                    candidate_turn=output.message,
                    pa_turns=tuple(pa_history),
                    tool_log=tool_log,
                    executed_tools=tuple(executed_tools),
                    outreach_sent=pa_outreach_sent,
                    previous_user_turns=tuple(user_history),
                    beat_avoid=beat.avoid,
                    beat_id=beat.beat,
                )
                for rule in (
                    "reaction_asserts_unexecuted_pa_action",
                    "reaction_authorises_pa_action",
                ):
                    if await asyncio.to_thread(judge.judge, rule, reception_ctx):
                        # Non-blocking, owner ruling 2026-09-13: judged and
                        # recorded for observation this round, never a finding.
                        # The call above still writes its `judge_call` row with
                        # the verdict, the evidence and the reason; the verdict
                        # itself goes on the turn's diagnostics below, once the
                        # list is rebuilt. To make it hard again, drop this
                        # branch and let the rule fall through to `findings`.
                        if rule == "reaction_asserts_unexecuted_pa_action":
                            asserts_unexecuted = [
                                judge.last_evidence(f"{rule}__q1") or output.message
                            ]
                            continue
                        findings.append(rule)
            # Armed wider than the check above, because these need no tool
            # record — only the assistant's own turns — so every reception
            # beat gets them. The judge is the only decider on both,
            # so with `MEMPA_JUDGE=off` the reaction beat is
            # not judged. One of them changed standing with the wiring:
            # `reaction_states_future_policy` is hard (裁定 vi) and leaves the
            # diagnostics list below.
            #
            # The contraction rule's arming stays mechanical: only
            # `tone_formality` reads the register off the reply, and the
            # contractions the reply actually carries go in as context, never
            # the attribute name.
            if reaction_beat and judge is not None and judge.enabled:
                last_pa_message = _latest_user_message(history)
                reaction_ctx = TurnContext(
                    turn_uid=f"{session.session_id}:{beat.beat}",
                    pa_previous_turn=last_pa_message,
                    candidate_turn=output.message,
                    pa_turns=tuple(pa_history),
                    contractions=tuple(_CONTRACTION_RE.findall(last_pa_message)),
                    beat_id=beat.beat,
                )
                rules = [
                    # Commented out, owner ruling 2026-09-13 ("先comment"): the
                    # rule is not asked this round; gold, prompt and builder
                    # stay.
                    # "reaction_states_future_policy",
                ]
                if (
                    "tone_formality" in reception_attributes
                    and last_pa_message.strip()
                ):
                    rules.insert(0, "reaction_misstates_contractions")
                for rule in rules:
                    if await asyncio.to_thread(judge.judge, rule, reaction_ctx):
                        findings.append(rule)
            # Set on `output` rather than only on the rejected-attempt record
            # below, which is built after the clean-pass return: whichever of
            # the three accept paths plays this candidate carries its own
            # diagnostics out with it.
            output.diagnostics = [
                finding
                for finding in (
                    # The reception specs ask the actor to quote the offending
                    # words back; regress14b `acc_004` t5 quoted "wouldn't"
                    # against a reply that does not contain it. Diagnostic on
                    # its first round: one hit over regress12-14b is too thin
                    # a base to spend a retry on.
                    quoted_span_not_in_pa_turn_finding(
                        output.message,
                        last_pa_message=_latest_user_message(history),
                    ),
                )
                if finding
            ]
            # Non-blocking, owner ruling 2026-09-13: judged and recorded for
            # observation this round, never a finding. It is appended after the
            # list above is rebuilt, the shape `probe_fact_missing` uses — the
            # rule on `diagnostics`, the judge's own evidence on
            # `diagnostics_detail` — so the played turn and every rejected
            # attempt carry it. To make it hard again, move it back to
            # `findings` at the reception call site.
            if asserts_unexecuted:
                if "reaction_asserts_unexecuted_pa_action" not in output.diagnostics:
                    output.diagnostics.append("reaction_asserts_unexecuted_pa_action")
                output.diagnostics_detail[
                    "reaction_asserts_unexecuted_pa_action"
                ] = asserts_unexecuted
            # The code half of `withheld_fact_before_ask`, all that is left of
            # the rule since the owner's ruling of 2026-09-14 (see the arming
            # site above): before the reveal beat the gated sentence is not in
            # the prompt, so a turn carrying every key term of it has handed
            # the thing over. `missing_key_terms()` is the store's own test and
            # costs nothing. Recorded here, after the rebuild, for the same
            # reason the two rules around it are — the shape
            # `probe_fact_missing` uses. Record-only, like the name itself
            # (`RECORD_ONLY_FINDINGS`): it never reaches `findings`, so it
            # costs no retry and voids nothing.
            if pre_reveal_key_terms_delivered:
                if "withheld_fact_before_ask" not in output.diagnostics:
                    output.diagnostics.append("withheld_fact_before_ask")
                output.diagnostics_detail.setdefault(
                    "withheld_fact_before_ask", ["key_terms_delivered"]
                )
            # `probe_fact_missing` is the judge's alone: with the
            # judge off the rule is simply not checked. The detail list is the
            # keys it named, so a reader gets them without the draft.
            #
            # Hard since the owner's ruling of 2026-09-13: a fact the judge
            # does not read in the turn is a finding like any other, so the
            # turn is said again with the missing values in its note. It is out
            # of `PLAYED_AFTER_BUDGET`, so a fact still missing when the
            # retries are gone raises instead — a probe that never states what
            # it is about measures nothing, so the session is not played.
            judged_missing_facts: list[str] = []
            if judge is not None and judge.enabled and required_values:
                judged_missing_facts = await asyncio.to_thread(
                    _judge_probe_fact_missing,
                    judge,
                    output,
                    session=session,
                    beat=beat,
                    required_values=required_values,
                    on_request_keys=on_request_fact_keys,
                    user_history=user_history,
                    last_pa_message=_latest_user_message(history),
                )
                if judged_missing_facts:
                    findings.append("probe_fact_missing")
            # One call per candidate generation, whatever the beat: the draft's
            # own instruction and `avoid` bind every turn of every session, so
            # this form has no arming of its own.
            general_clauses: dict[str, str] = {}
            if judge is not None and judge.enabled:
                general_clauses = await asyncio.to_thread(
                    _judge_general,
                    judge,
                    findings,
                    session=session,
                    beat=beat,
                    output=output,
                    last_pa_message=_latest_user_message(history),
                    user_history=user_history,
                    cue_owed=cue_owed,
                    first_turn_in_beat=first_turn_in_beat,
                    must_ask=must_ask,
                )
            # The sentences the turn has to keep, as one list (owner,
            # 2026-09-16): the strings the style retry note below quotes back
            # word for word are exactly the strings that buy a sentence out of
            # the 35-word cap. Three sources — the required fact values no
            # earlier turn has delivered, the session's gate cue whether this
            # beat owes it or a later beat restates it, and the gate's key
            # terms on a turn the dictation rule is armed on.
            #
            # Each is a sentence a shortening retry otherwise pays the cut
            # with. The facts: live regress19 `acc_027` b1 mandates three of
            # them in one sentence, so the cap alone is unsatisfiable — four
            # retries each sheared something else out and attempt 4 played at
            # 43 words anyway. The cue: regress18 deleted it on `acc_009` t2,
            # `acc_039` t3 and `acc_011` t7, and regress20 `acc_003` t10
            # dropped a later beat's restatement of it, which is the same
            # sentence the gate depends on, whoever owes it. The key terms: a
            # 35-word rewrite of a mixed message drops them and the blocking
            # dictation rule then fires a second retry (memos r2 `acc_059`).
            #
            # The exemption is per sentence, so a turn that keeps one sentence
            # and runs on in another is still trimmed.
            owed_facts = missing_required_facts(
                "",
                required_values=required_values,
                earlier_user_messages=user_history,
            )
            kept_strings = [required_values[key] for key in owed_facts]
            if session.gate_cue:
                kept_strings.append(session.gate_cue)
            if dictation_armed:
                kept_strings.extend(str(term) for term in session.gate_key_terms or ())
            style_findings = simulator_style_findings(
                output.message,
                dictated_texts=dictated_texts,
                on_reaction_beat=reaction_beat,
                owed_strings=kept_strings,
            )
            # Record-only, owner ruling 2026-09-14 (live FP watch: 51/71
            # rewrite findings were false positives): the one place a
            # record-only finding stops being a finding. It runs before
            # anything acts on the two lists, so the accept below, the
            # rejection record, the `blocking` split, the retry note and the
            # contract raise all see only what is left. `general_clauses` is
            # the detail the general form's findings already carry;
            # `probe_fact_missing` wrote its own keys earlier and keeps them.
            findings = record_only(findings, output, general_clauses)
            style_findings = record_only(style_findings, output)
            if not findings and not style_findings:
                output.rejected_attempts = rejected_attempts
                self._schedule_memory_record(
                    output, session, beat, history, model, benchmark_timestamp
                )
                return output

            record: dict[str, Any] = {
                "phase": "simulator_turn",
                "attempt": attempt,
                "error_codes": [*findings, *style_findings],
                "parse_warnings": list(output.parse_warnings),
                "model": selected_model,
                "session_id": session.session_id,
                "beat": beat.beat,
                "raw_response": result.final_content,
            }
            if output.diagnostics:
                record["diagnostics"] = list(output.diagnostics)
                if output.diagnostics_detail:
                    record["diagnostics_detail"] = dict(output.diagnostics_detail)
            if style_findings:
                record["style_findings"] = style_findings
                if not findings:
                    record["severity"] = "style"
            rejected_attempts.append(record)
            style_budget = self.output_contract_retries
            # On the compressed reaction beat an invented PA action is an
            # instrument error (owner ruling 2026-09-08): that one exchange IS
            # the measurement, so it is not played past the budget there.
            compressed_reaction = (
                session.compressed_reaction_beat is not None
                and session.compressed_reaction_beat == beat.beat
            )
            blocking = [
                rule
                for rule in findings
                if rule not in PLAYED_AFTER_BUDGET
                or (
                    compressed_reaction
                    and rule == "reaction_asserts_unexecuted_pa_action"
                )
            ]
            if not blocking and attempt > style_budget:
                # Style findings alone never kill a turn: on the last allowed
                # attempt the message is accepted, with the findings on record.
                # A `PLAYED_AFTER_BUDGET` rule leaves by the same door and rides
                # out on the turn's diagnostics, so the session loop writes it
                # to the transcript the way it writes any other diagnostic.
                record["accepted_after_retries"] = True
                for rule in findings:
                    if rule not in output.diagnostics:
                        output.diagnostics.append(rule)
                output.rejected_attempts = rejected_attempts
                self._schedule_memory_record(
                    output, session, beat, history, model, benchmark_timestamp
                )
                return output
            if blocking and attempt > self.output_contract_retries:
                # A `simulator_output_format_invalid` that survives the budget
                # used to be played on its parse fallback (recovered message,
                # default `next_beat: stay`), which routed the beat off a value
                # the actor never produced. The simulator is the instrument, so
                # the session loop turns this raise into an invalid session
                # rather than a degraded turn (owner ruling 2026-09-08).
                raise SimulatorOutputContractError(
                    findings,
                    rejected_attempts=rejected_attempts,
                )
            # Prepended to the format hint below, so a hallucinated tool call
            # is told what it called and that it has nothing to call, in the
            # same note that asks for the three blocks. Empty on every other
            # attempt, which leaves that hint as it was.
            tool_call_hint = (
                "you answered with a tool call ("
                + tool_call_error.partition(": ")[2]
                + ") and you have no tools — never call one, answer with the "
                "three text blocks only. "
                if tool_call_error
                else ""
            )
            # Once, for the two places that need it: the dictation hint below
            # lists them, and the keep clause subtracts them.
            dictation_missing = missing_key_terms(
                output.message, session.gate_key_terms
            )
            finding_hints = {
                "message_blank": "the <message> block was empty",
                "authoring_metadata_in_message": (
                    "the <message> leaked authoring or benchmark vocabulary — "
                    "speak only as the character in the scene"
                ),
                "reviewer_voice_in_message": (
                    "the <message> sounded like a reviewer grading the "
                    "assistant — react as the character, not as an evaluator"
                ),
                "consent_block_in_message": (
                    "the <message> carried the <consent_decisions> block. That "
                    "block is a private note the assistant never sees: say yes "
                    "or no to it in your own words inside <message>, and put "
                    "the structured entries in their own block."
                ),
                "simulator_output_format_invalid": tool_call_hint + (
                    "the three-block output format was malformed — emit exactly "
                    "the three required blocks, with the spoken dialog wrapped "
                    "in <message>...</message> (never as bare prose outside "
                    "the tags); next_beat accepts ONLY stay, advance, or "
                    "branch:<id> (never close/end/done)"
                ),
                "grades_pa_before_acting": (
                    "Your first sentence graded the assistant. Put that "
                    "judgement in <turn_assessment>; the message starts with "
                    "what you want done or the one thing you object to."
                ),
                "sentence_over_35_words": (
                    "One sentence is over 35 words. Break it up or drop the "
                    "clauses you do not need right now."
                ),
                "names_a_file_you_do_not_know": (
                    "You named a file, path, DOI, or id you do not actually "
                    "have. You have none of those beyond the session facts. A "
                    "real person says so ('I don't have the DOI on me') and "
                    "gives what they do have: the title, the authors, what it "
                    "is ('my citation notes', 'the draft'). Never make one up; "
                    "if the assistant cannot find it, let that be the outcome."
                ),
                "identifier_volunteered_unprompted": (
                    "You handed over an exact identifier the assistant had not "
                    "asked for. Refer to it by what it is and let the assistant "
                    "ask if it needs the exact value."
                ),
                "dictation_missing_key_terms": (
                    "Your last attempt dictated the text without these exact "
                    "words: "
                    + ", ".join(dictation_missing)
                    + ". This step gives the assistant the wording it must "
                    "write; say the note in your own voice but keep these "
                    "exact phrases in it."
                ),
                # Fixed wording, and the facts by VALUE rather than by key:
                # the key alone left the actor without the subject on
                # regress17 `acc_002`, and the whole point of the retry is
                # that the words the beat exists to deliver end up in the turn.
                "probe_fact_missing": (
                    "This turn still has to state these facts, in your own "
                    "words: "
                    + "; ".join(
                        required_values[key] for key in judged_missing_facts
                    )
                    + "."
                ),
                "gate_cue_missing": (
                    'This step has to say "'
                    + str(session.gate_cue or "")
                    + '". That is what gives the assistant a reason to ask you '
                    "for the rest; say it in your own voice with those words "
                    "in it, and make room by cutting something else."
                ),
                BEAT_INSTRUCTION_NOT_FOLLOWED: (
                    "Your last attempt did not do what this step asks of you ("
                    + general_clauses.get(BEAT_INSTRUCTION_NOT_FOLLOWED, "")
                    + "). Say the turn again and do it."
                ),
                BEAT_AVOID_VIOLATED: (
                    "Your last attempt did the one thing this step rules out ("
                    + general_clauses.get(BEAT_AVOID_VIOLATED, "")
                    + "). Say the turn again without it."
                ),
                "withheld_fact_before_ask": (
                    "You handed over the wording for this step before the "
                    "assistant asked you for it — either the wording itself, "
                    "or something of your own standing in for a fact you do "
                    "not have yet. You keep that back until it asks. Answer "
                    "what it actually put to you and say nothing about what "
                    "the work should say, and do not offer a substitute or a "
                    "placeholder."
                ),
                # Non-blocking, owner ruling 2026-09-13: judged and recorded for
                # observation this round, never a finding. The rule never
                # reaches `findings`, so this note is never asked for; it is
                # commented out so the clause is not computed either. Uncomment
                # it together with the reception call site above.
                # "reaction_asserts_unexecuted_pa_action": (
                #     "Your last attempt credited the assistant with an action "
                #     "it did not take ("
                #     + (
                #         unexecuted_pa_action_clause(
                #             output.message,
                #             on_reaction_beat=True,
                #             pa_executed_tools=executed_tools,
                #         )
                #         or "an action it never ran"
                #     )
                #     + "). React only to what the assistant actually did and "
                #     "showed; do not invent work."
                # ),
                "reaction_misstates_contractions": (
                    "Your last attempt described the assistant's contractions "
                    "backwards — you said its reply had none where it had "
                    "them, or the other way round. Read the reply again and "
                    "say how its register actually struck you."
                ),
                "reaction_authorises_pa_action": (
                    'Your last attempt handed the assistant a go-ahead ("'
                    + (
                        reaction_authorisation_clause(
                            output.message, on_reaction_beat=True
                        )
                        or "a release it can act on"
                    )
                    + '"). This step judges the reply you were given and hands '
                    "over nothing: do not accept, endorse or answer whatever "
                    "the assistant offered to do next, and do not tell it to "
                    "go ahead."
                ),
                "reaction_states_future_policy": (
                    "Your last attempt laid down a rule for the sessions to "
                    "come instead of reacting to the reply in front of you. "
                    "Say what is wrong with THIS reply, in your own terms, and "
                    "leave what should always happen out of it."
                ),
            }
            detail = "; ".join(
                finding_hints.get(finding, finding)
                for finding in [*findings, *style_findings]
            )
            # A style retry asks for a shorter turn and nothing else, and on
            # regress15 `acc_002` attempt 3 paid for the third sentence with
            # the two facts the beat existed to deliver. Name them in the note
            # so the shortening happens somewhere else.
            keep_facts = (
                " This step must still say "
                + ", ".join(f"{key}: {required_values[key]}" for key in owed_facts)
                + "."
                if style_findings and owed_facts
                else ""
            )
            # Naming the strings is not enough: regress18 `acc_002` t4 had the
            # note with the value spelled out and still rewrote "Marcus is free
            # from 18:00" into "you are available from 18:00". So the note also
            # quotes back the sentences of the REJECTED attempt that already
            # carry a kept string — they stay as they are and the cut comes out
            # of the rest. The list is `kept_strings` above, the one the 35-word
            # cap exempts: the note and the cap have to read the same list, or
            # the note protects a sentence the cap is still counting — live
            # memos r2 `acc_108` was a note holding the 37-word sentence that
            # was the whole message, so every attempt was asked for a shorter
            # turn and for that sentence whole.
            #
            # Minus the key terms this attempt's dictation finding reports
            # missing (owner, 2026-09-16): a term the dictation rule asks the
            # actor to put back is never protected word for word. The two
            # matchers read the same sentence differently — `sentences_carrying`
            # keeps a sentence that covers 70% of a string's tokens, while
            # `missing_key_terms` wants the term's words in order — so a rewrite
            # that scatters a phrase leaves the finding calling the term missing
            # and the note quoting the scattered sentence as one to keep, which
            # no retry can do both of.
            kept_sentences = (
                sentences_carrying(
                    output.message,
                    [
                        string
                        for string in kept_strings
                        if string not in dictation_missing
                    ]
                    if "dictation_missing_key_terms" in findings
                    else kept_strings,
                )
                if style_findings
                else []
            )
            keep_sentences = ""
            if kept_sentences:
                keep_sentences = (
                    " Keep these sentences word for word: "
                    + " ".join(f'"{sentence}"' for sentence in kept_sentences)
                    + " Cut elsewhere."
                )
            # On the attempt's own record, so a read can see what the actor was
            # told rather than infer it from the rewrite. `record` is the dict
            # already in `rejected_attempts`, and the harness writes every key
            # of it to `simulator_validation_attempts.jsonl`. The whole note,
            # `detail` first: writing only the keep clauses left every row whose
            # finding was not a style one with an empty note — regress20
            # `acc_023` had three such rows, including the t15
            # `dictation_missing_key_terms` one whose note names the terms the
            # dictation dropped.
            record["retry_note"] = detail + keep_facts + keep_sentences
            request_messages = [
                *request_messages,
                {"role": "assistant", "content": result.final_content},
                {
                    "role": "user",
                    "content": (
                        "Your previous output violated the private output contract: "
                        f"{detail}. Regenerate the same user turn. "
                        "Preserve the current beat's facts, intent, and routing. Output "
                        "all three blocks with a natural in-character <message>. Do not "
                        "mention this retry in <message>."
                        + keep_facts
                        + keep_sentences
                    ),
                },
            ]

        raise AssertionError("unreachable simulator output retry state")

    def take_pending_memory(self) -> list[dict]:
        """Hand off this turn's buffered exchange for the session-close write."""
        exchanges = self.pending_memory_exchanges
        self.pending_memory_exchanges = []
        return exchanges

    def _schedule_memory_record(
        self,
        output: SimulatorOutput,
        session: SessionScript,
        beat: Beat,
        history: list[dict] | None,
        model: str | None,
        benchmark_timestamp: str | None = None,
    ) -> None:
        """Buffer the exchange; memory is written once, at session close."""
        if not self.record_memory:
            return
        self.memory_model = model or self.get_default_model()
        # In-world time of this turn; the session-close write stamps HISTORY
        # with it, never with wall-clock time.
        if benchmark_timestamp:
            self.memory_timestamp = benchmark_timestamp
        pa_text = _latest_user_message(history).strip()
        if pa_text:
            self.pending_memory_exchanges.append({"role": "pa", "content": pa_text})
        self.pending_memory_exchanges.append(
            {"role": "simulated_user", "content": output.message.strip()}
        )
