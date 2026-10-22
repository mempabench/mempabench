"""Simulator prompt rendering — extracted from simulator/loop.py (M1).

Section renderers + template assembly for simulator LLM calls.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from simulator.reply_facts import ReplyFacts, render_reply_facts_block
from simulator.schemas import (
    Beat,
    BeatBranch,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)
from simulator.skills import reception_only, resolve_active_skills


PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "generate_message.md"
#: The probe session's sibling template: same blocks, but the
#: user corrects content errors only and never the assistant's manner.
PROBE_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "generate_message_probe.md"
#: `SessionScript.artifact_kind` value that selects the probe template.
PROBE_ARTIFACT_KIND = "probe_session"
DIALOGUE_NATURALNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/scripts/prompts/user_dialogue_naturalness.md"
)
_PATH_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
    r"|(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+\.(?:csv|docx|json|jsonl|md|pdf|txt|yaml|yml)\b",
    re.IGNORECASE,
)
_ALREADY_SAID = "(already said)"
_NOT_SAID_YET = "(known to you; not said yet)"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

PERSONALITY_MODEL_PRIOR = (
    "Compressed prior. Where it conflicts with a specific trait, reaction or "
    "speech item above, the specific item wins."
)

# Beat-block line for a beat past the one that reacted to the assistant's
# manner. The GLOBAL RULES carry the same rule for the whole session; this
# puts it on the step itself, where the visible conversation is loudest. Soft
# on purpose: the residue of the reaction is in character and stays, and what
# the step owes is its own content, not silence about the manner.
POST_REACTION_NOTE = (
    "You have already said your piece on the way the assistant answers. It can "
    "still colour how you sound here. The main content of this step is the "
    "task: an answer, a go-ahead, or the next thing you want done."
)

# Beat-block line for the played opening of a triggered session. It stands
# where the ask line stands on every other declaring beat: this beat declares
# the attribute by contract, and the thing it declares is the assistant's own
# decision to get in touch or stay quiet, which has already happened. There is
# nothing left to ask for.
TRIGGERED_OPENING_NOTE = (
    "This is the one turn where you react to whether the assistant got in "
    "touch; ask it nothing and do not discuss what the message said."
)


def _dash_join(*parts: str | None) -> str:
    return " — ".join(p for p in parts if p)


def render_character_card(identity: PersonaIdentity) -> str:
    lines: list[str] = []
    lines.append(f"Name: {identity.name}" + (f" (from {identity.source})" if identity.source else ""))
    if identity.name.startswith("User ") and not identity.source:
        lines.append(
            "Anonymization rule: refer to yourself exactly by this user label; "
            "do not invent or disclose a real personal name."
        )
    if identity.background:
        lines.append(f"Background: {identity.background}")
    life = identity.life
    occupation, living = life.occupation, life.living
    occupation_line = _dash_join(
        occupation.role, occupation.field, occupation.workplace, occupation.workplace_type
    )
    living_line = _dash_join(living.place, living.arrangement)
    if any([occupation_line, living_line, life.relationships, life.routines, life.hobbies]):
        lines.append("")
        lines.append("Life:")
        if occupation_line:
            lines.append(f"- Occupation: {occupation_line}")
        if living_line:
            lines.append(f"- Living: {living_line}")
        if life.relationships:
            lines.append("- People in your life:")
            lines.extend(
                f"  - {_dash_join(r.name, r.role, r.closeness)}" for r in life.relationships
            )
        if life.routines:
            lines.append("- Routines:")
            lines.extend(f"  - {r}" for r in life.routines)
        if life.hobbies:
            lines.append("- Hobbies:")
            lines.extend(f"  - {h}" for h in life.hobbies)
    if identity.personality_traits:
        lines.append("")
        lines.append("Personality traits:")
        lines.extend(f"- {t}" for t in identity.personality_traits)
    triggers = identity.triggers
    if triggers.irritated_by or triggers.pleased_by:
        lines.append("")
        lines.append("Triggers:")
        if triggers.irritated_by:
            lines.append("- Irritated by:")
            lines.extend(f"  - {t}" for t in triggers.irritated_by)
        if triggers.pleased_by:
            lines.append("- Pleased by:")
            lines.extend(f"  - {t}" for t in triggers.pleased_by)
    model = identity.personality_model
    rated_dims = [(k, v.level) for k, v in model.big_five.items() if v.level]
    if rated_dims or model.jungian_stack:
        lines.append("")
        lines.append(PERSONALITY_MODEL_PRIOR)
        lines.append("")
        lines.append("Personality model:")
        if rated_dims:
            lines.append("- Big Five:")
            lines.extend(f"  - {name}: {level}" for name, level in rated_dims)
        if model.jungian_stack:
            lines.append("- Function stack:")
            lines.extend(
                f"  - {_dash_join(fn.fn, fn.level, fn.shows_as)}" for fn in model.jungian_stack
            )
    style = identity.speech_style
    if any([style.tone, style.vocabulary_level, style.sentence_patterns, style.catchphrases, style.quirks]):
        lines.append("")
        lines.append("Speech style:")
        if style.tone:
            lines.append(f"- Tone: {style.tone}")
        if style.vocabulary_level:
            lines.append(f"- Vocabulary level: {style.vocabulary_level}")
        if style.sentence_patterns:
            lines.append("- Sentence patterns:")
            lines.extend(f"  - {p}" for p in style.sentence_patterns)
        if style.catchphrases:
            lines.append("- Catchphrases:")
            lines.extend(f"  - {c}" for c in style.catchphrases)
        if style.quirks:
            lines.append("- Quirks:")
            lines.extend(f"  - {q}" for q in style.quirks)
    register = identity.register_by_audience
    register_fields = [
        ("Close people", register.close),
        ("Colleagues", register.colleague),
        ("Institutions", register.institution),
        ("Strangers and service staff", register.stranger_or_service),
    ]
    if any(value for _, value in register_fields):
        lines.append("")
        lines.append("Register by audience:")
        lines.extend(f"- {label}: {value}" for label, value in register_fields if value)
    reaction = identity.reaction
    reaction_fields = [
        ("Someone gets it wrong", reaction.to_others.gets_it_wrong),
        ("Someone gets it right", reaction.to_others.gets_it_right),
        ("Someone questions you", reaction.to_others.questions_them),
        ("Someone acts without asking", reaction.to_others.acts_without_asking),
        ("A plan is disrupted", reaction.to_events.plan_disrupted),
        ("An outcome is bad", reaction.to_events.bad_outcome),
        ("Others make a demand", reaction.to_events.demand_from_others),
        ("A third party is in conflict with you", reaction.to_events.conflict_with_third_party),
        ("Your own error is exposed", reaction.to_events.own_error_exposed),
    ]
    if any(value for _, value in reaction_fields):
        lines.append("")
        lines.append("How you react:")
        lines.extend(f"- {label}: {value}" for label, value in reaction_fields if value)
    if identity.act_repertoire:
        lines.append("")
        lines.append("Acts you habitually use:")
        lines.extend(f"- {a}" for a in identity.act_repertoire)
    bounds = identity.knowledge_bounds
    if bounds.expert_in or bounds.weak_in:
        lines.append("")
        lines.append("What you know:")
        if bounds.expert_in:
            lines.append("- Expert in:")
            lines.extend(f"  - {k}" for k in bounds.expert_in)
        if bounds.weak_in:
            lines.append("- Weak in:")
            lines.extend(f"  - {k}" for k in bounds.weak_in)
    if identity.sim_rules:
        lines.append("")
        lines.append("Style rules (active for this character):")
        lines.extend(f"- {r}" for r in identity.sim_rules)
    return "\n".join(lines)


def render_active_skills(
    skills: list[tuple[str, str, str]],
    reception_skills: list[tuple[str, str, str]] | None = None,
) -> str:
    """skills = list of (attribute, setting, spec_body).

    `reception_skills` are the same triples carried over from the earlier beats
    of this session, already cut down to their reception sections.

    The order to take a position leads, so it binds on whichever reception
    rules are in the prompt: a beat that declares the attribute itself renders
    the reception section of its own spec and carries nothing, because
    `reception_carry_attributes` subtracts its own skills — and the surface can
    already be there on that beat (live `acc_007`: the PA's unprompted ping is
    the `proactive_outreach` surface, and it lands on the opening beat).

    The order is conditional on the surface actually being in the assistant's
    last reply. On a declaring beat's first turn there is usually no such reply
    yet, and asserting one as a fact pushed the actor to take a position on
    nothing (live `acc_011`).
    """
    reception_skills = reception_skills or []
    if not skills and not reception_skills:
        return "(no active skills declared for this beat — fall back to identity)"
    blocks: list[str] = [
        "If the assistant's last reply carries one of the surfaces these rules "
        "are about, this message takes a position on that surface, in "
        "your own voice, in the same turn as whatever else the beat has you "
        "do. If the reply did what the rules below describe, say what you "
        "are taking about the surface itself; a general \"fine\" or "
        "\"acceptable\" is not it. If the reply went against them, make the "
        "visible move the reception item names: push back, correct it, ask "
        "again pointedly, or go short. Answering the task alone, or letting "
        "the surface pass unmentioned, is not available on the first turn "
        "after it appears. Where no reply of the assistant's has shown one of "
        "these surfaces yet, there is nothing to take a position on: play the "
        "beat as it is written and leave the rules below for the turn the "
        "surface arrives on. One or two clauses on top of what the beat asks, "
        "never a lecture, and never the name of the preference or the "
        "attribute."
    ]
    for attr, setting, body in skills:
        blocks.append(f"## Skill: {attr} = {setting}\n\n{body}")
    if reception_skills:
        carried = "\n\n".join(
            f"### Skill: {attr} = {setting} (reception only)\n\n{body}"
            for attr, setting, body in reception_skills
        )
        blocks.append(
            "## Reacting to what you asked for earlier in this conversation\n\n"
            "The rules below govern only how you take what the assistant gives "
            "you on the things you asked for earlier in this conversation, not "
            "anything you ask for or produce yourself."
            f"\n\n{carried}"
        )
    return "\n\n---\n\n".join(blocks)


def beats_before(session: SessionScript, beat: Beat) -> list[Beat]:
    """The outline beats preceding `beat`, in order."""
    index = next(
        (i for i, item in enumerate(session.beats) if item.beat == beat.beat),
        None,
    )
    return session.beats[:index] if index is not None else []


def reception_carry_attributes(session: SessionScript, beat: Beat) -> list[str]:
    """Attributes whose reception rules are loaded for this beat.

    A beat that asks for something is answered in a LATER beat, whose own
    `active_skills` is usually empty; without this the actor reacts to the
    answer with no spec at all (live run `acc_002`, beats `b4`/`b5`). That one
    beat — the first beat after the one that asked — is where the reaction is
    taken, and it is the only beat that carries the rules. Minus the
    attributes this beat declares itself, which render in full.

    The carry used to run to the end of the session, on the reasoning that the
    assistant keeps behaving that way long after the asking beat. It does, and
    so did the complaint: live regress11 `acc_035` objected to the confidence
    tags on its reaction beat (`t=13`, correct) and then again at `t=16`, 19,
    21, 24 and 29, six retries and a session that is all complaint. The IX
    selection is made once, at the declaring beat, and the assistant cannot
    re-select; a user who keeps grading a manner that cannot change is not
    measuring anything. So the actor reacts once and then proceeds with the
    task under the surface it got.

    `session.session_reception_carry` is what the loop actually loaded on this
    beat; the outline is the fallback for the prompt-only path, which has no
    runtime history and reads the beat right before this one.
    """
    walked = session.session_reception_carry
    if walked is None:
        before = beats_before(session, beat)
        walked = list(before[-1].active_skills or []) if before else []
    own = set(beat.active_skills or [])
    carried: list[str] = []
    for attr in walked:
        if attr not in own and attr not in carried:
            carried.append(attr)
    return carried


def reception_already_taken(session: SessionScript, beat: Beat) -> bool:
    """True on a beat past the one that reacted to the surface.

    The reaction beat has been played, its rules are no longer loaded, and
    this beat's job is the task. The beat block says so, so the instruction is
    in front of the actor on the very turn it would otherwise re-open the
    complaint from the visible conversation alone.
    """
    if beat.active_skills or reception_carry_attributes(session, beat):
        return False
    if session.session_reception_carry is not None:
        return bool(session.session_reception_skills)
    return any(item.active_skills for item in beats_before(session, beat))


def render_active_skills_block(
    matrix: PreferenceMatrix,
    session: SessionScript,
    beat: Beat,
    root: Path | str | None = None,
) -> str:
    """The `<<ACTIVE_SKILLS>>` block: this beat's skills in full, then the
    session's earlier skills in reception-only form.

    A carried setting resolves the way the beat that declared it resolved it,
    that beat's own `active_skill_settings` override included.
    """
    skills_kwargs = {"root": root} if root else {}
    skills = resolve_active_skills(
        matrix,
        session.context,
        beat.active_skills,
        setting_overrides=beat.active_skill_settings,
        **skills_kwargs,
    )
    carried_overrides: dict[str, str] = {}
    for item in beats_before(session, beat):
        carried_overrides.update(item.active_skill_settings or {})
    reception = [
        (attr, setting, reception_only(body))
        for attr, setting, body in resolve_active_skills(
            matrix,
            session.context,
            reception_carry_attributes(session, beat),
            setting_overrides=carried_overrides,
            **skills_kwargs,
        )
    ]
    return render_active_skills(skills, reception)


def render_session_clock(session: SessionScript) -> str:
    """One line telling the simulated user what day and time it is in-world."""
    clock = session.session_clock
    if clock is None:
        return ""
    return f"- Now: {clock.strftime('%A, %Y-%m-%d %H:%M')}"


def render_life_context_block(session: SessionScript) -> str:
    """Life context preceded by the in-world clock when the harness set one."""
    clock_line = render_session_clock(session)
    body = render_life_context(session.life_context)
    return f"{clock_line}\n{body}" if clock_line else body


def render_life_context(lc: LifeContext) -> str:
    lines: list[str] = []
    if lc.what_happened:
        lines.append(f"- What happened: {lc.what_happened}")
    if lc.mood:
        lines.append(f"- Mood: {lc.mood}")
    if lc.time_pressure:
        lines.append(f"- Time pressure: {lc.time_pressure}")
    if lc.energy:
        lines.append(f"- Energy: {lc.energy}")
    if lc.continuity_note:
        lines.append(f"- Continuity note: {lc.continuity_note}")
    return "\n".join(lines) if lines else "(no life context for this session)"


def _append_labeled_line(lines: list[str], label: str, value: str) -> None:
    if value:
        lines.append(f"{label}: {value}")


def _render_branch_block(branch_id: str, branch: BeatBranch) -> list[str]:
    lines = [f"- {branch_id}:"]
    fields = [
        ("Condition", branch.condition),
        ("Goal", branch.goal),
        ("Constraint", branch.constraint),
        ("Actor instruction", branch.actor_instruction),
        ("Language style", branch.language_style),
        ("Avoid", branch.avoid),
    ]
    rendered = [f"  - {label}: {value}" for label, value in fields if value]
    if rendered:
        lines.extend(rendered)
    else:
        lines.append("  - (no branch-specific instruction)")
    return lines


def render_session_outline(session: SessionScript, beat: Beat) -> str:
    """What is settled and where you are now: the task, then the beats so far.

    Plain numbers only — beat ids never reach the actor prompt. Earlier beats
    are marked `(done)` or `(skipped)`, the current one `(now)`. Later beats are
    NOT listed, so the actor cannot run ahead; only their count is given.
    Numbering stays absolute: the current beat keeps its real number.
    """
    lines: list[str] = []
    if session.task:
        lines.append(session.task)
    skipped = set(session.skipped_beats)
    current_index = next(
        (i for i, item in enumerate(session.beats) if item.beat == beat.beat),
        None,
    )
    remaining = 0
    for index, item in enumerate(session.beats):
        goal = item.goal or item.actor_instruction
        if not goal:
            continue
        if current_index is not None and index > current_index:
            remaining += 1
            continue
        if current_index is not None and index == current_index:
            mark = " (now)"
        elif item.beat in skipped:
            mark = " (skipped)"
        elif current_index is not None and index < current_index:
            mark = " (done)"
        else:
            mark = ""
        lines.append(f"{index + 1}. {goal}{mark}")
    if remaining:
        lines.append(
            f"Then {remaining} more step{'s' if remaining != 1 else ''} after this."
        )
    if not lines:
        return "(no outline for this session)"
    return "\n".join(lines)


def render_beat_block(
    beat: Beat,
    *,
    first_turn_in_beat: bool = True,
    reception_already_taken: bool = False,
    on_compressed_reaction_beat: bool = False,
) -> str:
    lines = [f"Beat id: {beat.beat}"]
    _append_labeled_line(lines, "Dramatic function", beat.dramatic_function)
    _append_labeled_line(lines, "Goal", beat.goal)
    # Only the first turn of the beat is the one that poses the request; on
    # later turns this line would push the actor to re-ask what it already
    # asked instead of reacting to the answer.
    if beat.active_skills and first_turn_in_beat:
        # A declaring beat that is also the reaction beat is the played opening
        # of a triggered session: it declares the attribute by contract and
        # asks for nothing. Live regress14c `acc_007` got the ask line a few
        # lines above its own "Avoid: do not ask what the message said", and
        # the prompt contradicting itself is half of why the turn asked.
        lines.append(
            TRIGGERED_OPENING_NOTE
            if on_compressed_reaction_beat
            else "This is the step where you ask for what this conversation is about."
        )
    if reception_already_taken:
        lines.append(POST_REACTION_NOTE)
    _append_labeled_line(lines, "Constraint", beat.constraint)
    _append_labeled_line(lines, "Actor instruction", beat.actor_instruction)
    _append_labeled_line(lines, "Language style", beat.language_style)
    _append_labeled_line(lines, "How life context colors this beat", beat.life_context_hint)
    _append_labeled_line(lines, "Avoid", beat.avoid)
    _append_labeled_line(
        lines, "What you were waiting for from the PA", beat.entry_condition
    )
    _append_labeled_line(lines, "If that has not happened", beat.if_not_met)
    if beat.reactions:
        lines.append("")
        lines.append("Conditional branch instructions for this reaction:")
        for branch_id, branch in beat.reactions.items():
            lines.extend(_render_branch_block(branch_id, branch))
    return "\n".join(lines)


def chat_pairs(history: list[dict] | None) -> list[dict]:
    """The chat keys of each history entry, and nothing else.

    The session loop also stamps the transcript `t` on every entry so a record
    indexing `history` can name the turn a reader sees in `transcript.jsonl`.
    That key is ours; a provider payload takes `role` and `content` alone.
    """
    return [
        {"role": entry["role"], "content": entry["content"]}
        for entry in history or []
    ]


def _actor_visible_context(history: list[dict] | None) -> str:
    # Both sides of the visible conversation can say a fact out loud: the
    # simulator's own past lines (role assistant) and the PA's replies (role
    # user). The beat block is private stage direction, not something spoken,
    # so it does not count as "already said".
    return "\n".join(
        str(message.get("content") or "")
        for message in history or []
        if message.get("role") in {"assistant", "user"}
    ).casefold()


def _fact_status(value: Any, actor_visible_context: str) -> str:
    """Tag one fact value as already spoken or still unspoken.

    Values carrying path literals are judged on those literals (a path is the
    part that gets quoted verbatim); everything else on the whole value.
    """
    text = str(value)
    paths = _PATH_LITERAL_RE.findall(text)
    if paths:
        said = all(path.casefold() in actor_visible_context for path in paths)
    else:
        said = bool(text.strip()) and text.casefold() in actor_visible_context
    return _ALREADY_SAID if said else _NOT_SAID_YET


def gated_fact_keys(session: SessionScript, beat: Beat) -> set[str]:
    """`task_facts` keys this session gates to a beat LATER than `beat`.

    A gated fact is the one the preference probe depends on: it does not exist
    for the simulated user until its own beat is played, so it cannot be said
    early and cannot let the PA do the probe's work ahead of time.
    """
    order = {item.beat: index for index, item in enumerate(session.beats)}
    current = order.get(beat.beat)
    if current is None:
        return set()
    return {
        key
        for key, beat_id in (session.reveal_at or {}).items()
        if beat_id in order and order[beat_id] > current
    }


# Rendered only on a beat that actually has facts withheld from it. A hidden
# key leaves the actor holding a hole rather than a fact, and a PA blocked by
# the closed gate asks for exactly that hole: live `check039` `acc_039` t12,
# asked for the visit reason two beats before the reveal, answered "The reason
# for the visit is a routine preventive checkup. Use that exact phrasing for
# the record." Nothing in the prompt told it that the honest move there is to
# defer, so it filled the hole with a plausible substitute and the PA then hit
# the closed gate thirteen times with it.
WITHHELD_FACTS_LINE = (
    "**Some facts belong to later steps and are withheld from you for now.** If "
    "the assistant asks you for one of them — wording it must use for a record, "
    "a note, an email, a reason — say you will give it when you get to that "
    "step; do not make one up and do not offer a placeholder."
)


def render_task_facts(
    facts: dict[str, Any],
    *,
    history: list[dict] | None = None,
    hidden_keys: Iterable[str] = (),
) -> str:
    """Render session.task_facts as a readable YAML-ish list for the system prompt.

    Every value is rendered in full: the simulated user is a real person and
    knows all of them. Each line carries a status tag saying whether it has
    already been said in the visible conversation.

    `hidden_keys` are gated facts (see `gated_fact_keys`): they belong to a
    later beat and are omitted entirely, tag and all, so the actor cannot
    reveal them before the beat that introduces them. When there are any,
    `WITHHELD_FACTS_LINE` follows the list and says what to do when the
    assistant asks for one; a beat with nothing hidden never sees it.

    Empty facts -> an explicit "no facts" message rather than an empty section.
    """
    hidden = set(hidden_keys)
    facts = {key: value for key, value in facts.items() if key not in hidden}
    context = _actor_visible_context(history)
    lines: list[str] = []
    for key, value in facts.items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            lines.extend(f"  - {item} {_fact_status(item, context)}" for item in value)
        elif isinstance(value, dict):
            lines.append(f"- {key}:")
            for k, v in value.items():
                lines.append(f"  - {k}: {v} {_fact_status(v, context)}")
        else:
            lines.append(f"- {key}: {value} {_fact_status(value, context)}")
    block = (
        "\n".join(lines)
        if lines
        else "(no specific ground-truth facts declared for this session)"
    )
    return f"{block}\n\n{WITHHELD_FACTS_LINE}" if hidden else block


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def prompt_template_path(session: SessionScript) -> Path:
    """The system-prompt template for this session's kind.

    `probe_session` drafts get `PROBE_PROMPT_TEMPLATE_PATH`; every other
    artifact kind keeps `PROMPT_TEMPLATE_PATH`.
    """
    if session.artifact_kind == PROBE_ARTIFACT_KIND:
        return PROBE_PROMPT_TEMPLATE_PATH
    return PROMPT_TEMPLATE_PATH


def _load_template(session: SessionScript) -> str:
    return prompt_template_path(session).read_text(encoding="utf-8")


def load_dialogue_naturalness() -> str:
    return DIALOGUE_NATURALNESS_PATH.read_text(encoding="utf-8").strip()


def _render_system_message(
    identity: PersonaIdentity,
    matrix: PreferenceMatrix,
    session: SessionScript,
    beat: Beat,
    actor_skill_specs_root: Path | str | None = None,
    history: list[dict] | None = None,
) -> str:
    template = _load_template(session)
    return (
        template
        .replace("<<CHARACTER_CARD>>", render_character_card(identity))
        .replace(
            "<<ACTIVE_SKILLS>>",
            render_active_skills_block(matrix, session, beat, actor_skill_specs_root),
        )
        .replace("<<LIFE_CONTEXT>>", render_life_context_block(session))
        .replace("<<SESSION_OUTLINE>>", render_session_outline(session, beat))
        .replace(
            "<<BEAT_BLOCK>>",
            render_beat_block(
                beat,
                reception_already_taken=reception_already_taken(session, beat),
                on_compressed_reaction_beat=(
                    session.compressed_reaction_beat == beat.beat
                ),
            ),
        )
        .replace(
            "<<TASK_FACTS>>",
            render_task_facts(
                session.task_facts,
                history=history,
                hidden_keys=gated_fact_keys(session, beat),
            ),
        )
        .replace("<<DIALOGUE_NATURALNESS>>", load_dialogue_naturalness())
    )


def render_held_items_block(held_items: Iterable[dict[str, str]]) -> str:
    """Private per-turn instruction naming the items the assistant is holding.

    The ids come from the harness, not from the assistant's visible text, so
    the simulated user is told them here instead of having to spot them in the
    reply. This block is stage direction: it tells the actor what is waiting on
    an answer and asks for one structured entry per id. It is never spoken.
    """
    items = list(held_items)
    if not items:
        return ""
    lines = [
        "The assistant is holding these items until you say what to do with them:"
    ]
    for item in items:
        intent = str(item.get("intent") or "").strip()
        item_id = item["interaction_item_id"]
        lines.append(f"- {item_id}" + (f" ({intent})" if intent else ""))
    ids = ", ".join(item["interaction_item_id"] for item in items)
    lines.append(
        "Answer them in <message> the way this person would, in your own words. "
        "Never read these ids out loud; they are not something the user would say. "
        "Then add a <consent_decisions> block with one entry for each of these "
        f"ids and no others ({ids}): decision `allow` if your message gives the "
        "go-ahead for that item, `deny` if it refuses or puts it off, `unclear` "
        "if your message leaves it open. The block must agree with what you "
        "actually said."
    )
    return "\n".join(lines)


def _render_user_trigger(
    session: SessionScript,
    beat: Beat,
    history: list[dict] | None,
    held_items: Iterable[dict[str, str]] = (),
    *,
    reply_facts: ReplyFacts | None = None,
) -> str:
    """Trigger message that prompts the LLM to produce the next line.

    The current beat is the only actor-facing action boundary. `reply_facts`
    is the private record of what the latest PA reply did (owner 2026-09-17);
    it is rendered only once there is a reply to react to.
    """
    suffix = (
        "\n\nOutput exactly three XML-like blocks, plus <consent_decisions> only "
        "if held ids are listed above. Your "
        "response must start with <message> and the PA-visible dialogue must be "
        "inside <message>...</message>."
        "\n\n<message>\n...\n</message>\n\n<factual_check>\n</factual_check>\n\n"
        "<turn_assessment>\nnext_beat: stay | advance | branch:<branch_id>\n</turn_assessment>"
    )
    # Per-turn stage direction: what the assistant is waiting on an answer for.
    held_block = render_held_items_block(held_items)
    if held_block:
        suffix = f"\n\n{held_block}{suffix}"
    # Per-turn stage direction: what the assistant's latest reply actually
    # did. Only after a reply exists — the opening turn and the suppress
    # opening of a triggered session have nothing to record.
    facts_block = render_reply_facts_block(reply_facts) if history else ""
    if facts_block:
        suffix = f"\n\n{facts_block}{suffix}"
    # One-shot note set by SimulatorLoop when it jumped over stalled beats to
    # reach this one; it is consumed by the loop after this first trigger.
    if session.pending_jump_note:
        suffix = f"\n\n{session.pending_jump_note}{suffix}"
    # The suppress branch of a triggered session: the assistant stayed silent,
    # so nothing was delivered and `_history` is empty. Without this the turn
    # falls to the opening-beat trigger below, which tells the actor to open
    # with the ask itself — the opposite of what the beat block just said, and
    # the same contradiction the beat's ask line used to carry.
    if (
        session.proactive_trigger is not None
        and session.compressed_reaction_beat == beat.beat
        and not history
    ):
        return (
            f"The assistant stayed silent: nothing reached you, and there is no "
            f"conversation yet. Produce only the current `{beat.beat}` opening "
            "beat. Follow its Actor instruction and Avoid field. This is your "
            "one turn and it is a reaction to not having been contacted about "
            "it — say how that sits with you and stop there. Ask the assistant "
            "for nothing. Send the first message in character."
        ) + suffix
    if history:
        return (
            f"Produce only the current `{beat.beat}` beat. Follow its Actor instruction, "
            "select exactly one applicable conditional branch from the latest PA reply "
            "and the private record of what it did when branches are present, and obey "
            "its Avoid field. Do not continue the "
            "conversation freely or introduce a new task. Produce your next message in "
            "character."
        ) + suffix
    return (
        f"The conversation has not started yet. Produce only the current "
        f"`{beat.beat}` opening beat. Follow its Actor instruction and Avoid field. "
        "Exact values wait until the assistant asks; open with the ask itself. "
        "Send the first message "
        "in character."
    ) + suffix


def assemble_prompt(
    identity: PersonaIdentity,
    matrix: PreferenceMatrix,
    session: SessionScript,
    beat: Beat,
    history: list[dict] | None = None,
    actor_skill_specs_root: Path | str | None = None,
) -> list[dict]:
    """Assemble OpenAI-style chat messages for the simulator LLM call.

    Returns: [{role: system, content: ...}, *history, {role: user, content: trigger}].
    """
    system = _render_system_message(
        identity,
        matrix,
        session,
        beat,
        actor_skill_specs_root,
        history,
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(chat_pairs(history))
    messages.append({"role": "user", "content": _render_user_trigger(session, beat, history)})
    return messages
