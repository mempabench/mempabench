"""Test simulator prompt assembly and actor-information boundaries.

Run with:
    PYTHONPATH=. python tests/test_simulator/test_prompt_assembly.py
"""

from __future__ import annotations

from pathlib import Path

from simulator.prompting import assemble_prompt
from simulator.schemas import (
    Beat,
    BeatBranch,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)


def _load_fixtures() -> tuple[PersonaIdentity, PreferenceMatrix, SessionScript]:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = SessionScript(
        session_id="acc_001",
        scenario="assess_anyons_review",
        context="work",
        task="Assess whether the anyons review belongs in the draft.",
        life_context=LifeContext(
            what_happened="Early October drafting work is underway.",
            continuity_note="The paper introduction is the current focus.",
        ),
        task_facts={
            "source": "anyons_review_paper.txt",
            "response_level": "Assess methodology and conclusions.",
        },
        beats=[
            Beat(
                beat="open",
                active_skills=["verbosity"],
                actor_instruction="Ask whether the anyons review belongs in the draft.",
            ),
            Beat(
                beat="react_to_pa",
                active_skills=["verbosity"],
                actor_instruction="React to the PA's assessment.",
                reactions={
                    "if_pa_correct": BeatBranch(
                        condition="The PA gives enough technical support.",
                        actor_instruction="Acknowledge a sufficiently detailed answer."
                    ),
                    "if_pa_wrong": BeatBranch(
                        condition="The PA gives only a conclusion.",
                        actor_instruction="Ask for the missing technical support."
                    ),
                },
            ),
        ],
    )
    return identity, matrix, session


def test_the_prompt_only_path_reads_the_stamped_reaction_beat_too() -> None:
    """`assemble_prompt` builds the same beat block the agent path builds, so
    the flag the loop stamps has to reach it there as well — otherwise a
    triggered opening rendered through this path still carries the ask line."""
    identity, matrix, session = _load_fixtures()
    beat = session.beats[0]
    ask_line = "This is the step where you ask for what this conversation is about."

    system = assemble_prompt(identity, matrix, session, beat)[0]["content"]
    assert ask_line in system

    session.compressed_reaction_beat = beat.beat
    system = assemble_prompt(identity, matrix, session, beat)[0]["content"]
    assert ask_line not in system
    assert "react to whether the assistant got in touch" in system


def test_open_beat_active_skills_only() -> None:
    identity, matrix, session = _load_fixtures()
    open_beat = session.beats[0]
    assert open_beat.beat == "open"
    assert open_beat.active_skills == ["verbosity"]

    messages = assemble_prompt(identity, matrix, session, open_beat)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    sys = messages[0]["content"]

    # Identity content present
    assert "Name: User A" in sys
    assert "Anonymization rule" in sys

    # Active skill (verbosity = Detailed) IS present
    assert "Skill: verbosity = Detailed" in sys
    assert "two or more supporting moves of different kinds" in sys
    assert "Three caveats is one kind three times" in sys

    # Other skills NOT present (Active-Skills-Only invariant)
    assert "Build causal chains" not in sys, "reasoning_visibility body leaked"
    assert "Close before switching" not in sys, "topic_management body leaked"
    assert "Do not ask permission" not in sys, "autonomy_level body leaked"

    # Beat-level info present
    assert "Beat id: open" in sys
    assert open_beat.actor_instruction[:30] in sys

    # Life context present
    assert "Early October" in sys

    # The opening trigger names only the current beat. Full task details remain
    # private context because they may contain information assigned to later beats.
    user = messages[1]["content"]
    assert "current `open` opening beat" in user
    assert session.task not in user
    assert "anyons" not in user.lower()
    assert "# USER DIALOGUE NATURALNESS" in sys
    assert "Write the user's visible message as something this person would say" in sys
    print(f"  -> open-beat prompt OK (system={len(sys)} chars, user={len(user)} chars)")


def test_unspoken_task_fact_path_is_shown_with_not_said_yet_tag() -> None:
    """The user knows every fact; an unspoken path is rendered, not deleted."""
    identity, matrix, session = _load_fixtures()
    open_beat = session.beats[0]
    messages = assemble_prompt(identity, matrix, session, open_beat)
    sys = messages[0]["content"]

    # Section header present
    assert "PRIVATE VERIFICATION FACTS" in sys

    # The path keeps its real value and is tagged as not yet spoken.
    assert "- source: anyons_review_paper.txt (known to you; not said yet)" in sys
    assert "[withheld until its assigned beat]" not in sys
    assert "response_level" in sys
    assert "methodology" in sys
    assert "conclusions" in sys

    print("  -> unspoken task-fact path rendered with the not-said-yet tag")


def test_task_fact_paths_are_tagged_said_once_either_side_spoke_them() -> None:
    """Every path keeps its real value. The status tag flips to `(already said)`
    once either side has spoken it (the simulator's own lines OR the PA's
    replies). A path named only by the current beat is stage direction, not
    speech, so it stays `(known to you; not said yet)`, as does a path that
    appears nowhere at all."""
    identity, matrix, _ = _load_fixtures()
    session = SessionScript(
        session_id="path_visibility",
        scenario="path_visibility",
        context="work",
        task_facts={
            "paths": {
                "already_said": "documents/research/first.md",
                "pa_mentioned": "documents/research/third.md",
                "current": "documents/research/second.md",
                "never_mentioned": "documents/research/fourth.md",
            }
        },
        beats=[],
    )
    beat = Beat(
        beat="second_source",
        actor_instruction="Give the path documents/research/second.md.",
    )
    messages = assemble_prompt(
        identity,
        matrix,
        session,
        beat,
        history=[
            {
                "role": "assistant",
                "content": "Start with documents/research/first.md.",
            },
            {
                "role": "user",
                "content": "I read documents/research/third.md as instructed.",
            },
        ],
    )
    system = messages[0]["content"]

    assert "- already_said: documents/research/first.md (already said)" in system
    assert "- pa_mentioned: documents/research/third.md (already said)" in system
    assert (
        "- current: documents/research/second.md (known to you; not said yet)"
        in system
    )
    assert (
        "- never_mentioned: documents/research/fourth.md (known to you; not said yet)"
        in system
    )
    assert "[withheld until its assigned beat]" not in system


def test_task_fact_path_absent_from_beat_and_history_keeps_its_real_value() -> None:
    """A path mentioned by neither the beat block nor the history is still the
    user's own knowledge: render the literal, tag it as not yet said."""
    identity, matrix, _ = _load_fixtures()
    session = SessionScript(
        session_id="unmentioned_path",
        scenario="unmentioned_path",
        context="work",
        task_facts={"notes_file": "documents/research/citation_notes.md"},
        beats=[],
    )
    beat = Beat(beat="open", actor_instruction="Ask about the introduction.")

    system = assemble_prompt(
        identity,
        matrix,
        session,
        beat,
        history=[{"role": "user", "content": "What do you need?"}],
    )[0]["content"]

    assert (
        "- notes_file: documents/research/citation_notes.md "
        "(known to you; not said yet)" in system
    )


def test_three_block_output_format_required() -> None:
    """System message must instruct the LLM to use the 3-block output format
    (message + factual_check + turn_assessment)."""
    identity, matrix, session = _load_fixtures()
    open_beat = session.beats[0]
    messages = assemble_prompt(identity, matrix, session, open_beat)
    sys = messages[0]["content"]

    # Three block tags must be specified
    assert "<message>" in sys
    assert "<factual_check>" in sys
    assert "<turn_assessment>" in sys
    # The per-turn emotion block was removed
    assert "<emotion_event>" not in sys

    # Boundary statement present (PA never sees the latter two)
    assert ("never sees" in sys.lower()) or ("PA-blind" in sys) or (
        "PA only sees" in sys.lower()
    ) or ("PA only" in sys), "Access boundary not communicated to LLM"

    # turn_assessment is reduced to next_beat only
    assert "next_beat" in sys
    # Old per-turn eval fields must NOT be re-introduced
    for old_field in ("preference_met", "attitude_delta", "trust_delta"):
        assert old_field not in sys, f"Removed field {old_field!r} re-introduced in prompt"

    # The per-turn trigger reinforces the block count
    trigger = messages[-1]["content"]
    assert "exactly three XML-like blocks" in trigger
    assert "<consent_decisions> only" in trigger

    print("  -> 3-block output format instructions present in prompt")


def test_close_beat_cannot_start_a_follow_up_task() -> None:
    """Close prompts must end the current session instead of inventing more work."""
    identity, matrix, session = _load_fixtures()
    close_beat = Beat(
        beat="close",
        active_skills=[],
        actor_instruction="Acknowledge completion.",
    )

    messages = assemble_prompt(
        identity,
        matrix,
        session,
        close_beat,
        history=[{"role": "assistant", "content": "The task is complete."}],
    )
    system_prompt = messages[0]["content"]

    assert "A `close` beat closes this session." in system_prompt
    assert "Do not start a new task" in system_prompt
    assert "Treat the CURRENT BEAT as an action boundary." in system_prompt
    assert "Do not improvise an additional task" in system_prompt
    trigger = messages[-1]["content"]
    assert "current `close` beat" in trigger
    assert "Do not continue the conversation freely or introduce a new task." in trigger


def test_an_offer_that_runs_ahead_of_the_beat_is_not_accepted() -> None:
    """Live `acc_006`: `bottleneck_signal.avoid` said "Do not
    authorize any edits yet", the PA offered to make them anyway, and the actor
    answered "Proceed with the consolidation" — so the whole deliverable was
    finished three beats before `task_expansion` armed, and the probe beat had
    no task left in it. The `avoid` list only stopped volunteering.
    """
    identity, matrix, session = _load_fixtures()

    system_prompt = assemble_prompt(identity, matrix, session, session.beats[0])[0][
        "content"
    ]

    assert "An offer that runs ahead of this beat is not taken." in system_prompt
    assert '"not yet", "first show me X", or simply this beat' in system_prompt
    assert "accepting an offer beyond this beat is not available" in system_prompt


def test_the_asking_beat_asks_and_does_not_answer_itself() -> None:
    """Live `acc_035`: the probe beat was entered and played out, and
    the actor delivered the call the beat asks the assistant for ("the record
    supports granting the regrade") instead of demanding it, so the ask never
    reached the PA and the fact the beat holds stayed unreleased.
    """
    identity, matrix, session = _load_fixtures()
    sentence = (
        "On that step you ask and you do not answer — never state the "
        "determination, judgement or call the beat asks the assistant for, and "
        "never supply the fact the beat holds as if it were already settled."
    )

    system_prompt = assemble_prompt(identity, matrix, session, session.beats[0])[0][
        "content"
    ]

    assert sentence in system_prompt
    # The rule pool `/patch-simulator-prompt` draws from carries the same line.
    pool = Path("simulator/prompts/generate_message_pool.md").read_text(
        encoding="utf-8"
    )
    assert sentence in pool


def test_react_beat_renders_only_declared_skill() -> None:
    identity, matrix, session = _load_fixtures()
    react_beat = session.beats[1]
    assert react_beat.beat == "react_to_pa"
    assert react_beat.active_skills == ["verbosity"]

    history = [
        {"role": "assistant", "content": "Quinn: write the elevator email please."},
        {"role": "user", "content": "PA: here's a draft..."},
    ]
    messages = assemble_prompt(identity, matrix, session, react_beat, history=history)
    sys = messages[0]["content"]

    assert "Skill: verbosity = Detailed" in sys
    assert "two or more supporting moves of different kinds" in sys

    # Non-active skill NOT present
    assert "Build causal chains" not in sys, "reasoning_visibility leaked"

    # History propagated, then trigger appended
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[3]["role"] == "user"

    trigger = messages[-1]["content"]
    assert "current `react_to_pa` beat" in trigger
    assert "select exactly one applicable conditional branch" in trigger
    assert "Do not continue the conversation freely or introduce a new task." in trigger
    print("  -> react-beat prompt OK (2 skills present, history preserved)")


def test_session_without_task_facts() -> None:
    """Sessions without task_facts should render an explicit 'no facts' notice (no crash)."""
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = SessionScript.model_construct(
        session_id="no_task_facts",
        context="work",
        task_facts={},
        beats=[],
    )
    open_beat = Beat.model_construct(beat="open", active_skills=[])

    messages = assemble_prompt(identity, matrix, session, open_beat)
    sys = messages[0]["content"]

    # No crash, and the section header still appears with a placeholder
    assert "PRIVATE VERIFICATION FACTS" in sys
    assert "no specific ground-truth facts" in sys

    print("  -> session without task_facts renders placeholder OK")


def test_active_skill_setting_override_for_tool_coverage_session() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = SessionScript.model_construct(
        session_id="skill_override",
        context="work",
        task_facts={},
        beats=[],
    )
    open_beat = Beat.model_construct(
        beat="open",
        active_skills=["topic_management"],
        active_skill_settings={"topic_management": "One-at-a-time"},
    )

    messages = assemble_prompt(identity, matrix, session, open_beat)
    sys = messages[0]["content"]

    # The matrix cell for work is Organize; the beat override must win.
    assert "Skill: topic_management = One-at-a-time" in sys
    assert "Close before switching" in sys
    assert "Grouping never permits abandoning requested work" not in sys


def test_user_a_identity_and_human_user_framing_render() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = SessionScript.model_construct(
        session_id="identity_prompt_check",
        persona="User A",
        context="personal",
        scenario="identity_prompt_check",
        task="Check prompt assembly.",
        life_context=LifeContext(),
        task_facts={},
        beats=[],
    )
    beat = Beat.model_construct(beat="open", active_skills=[])

    messages = assemble_prompt(identity, matrix, session, beat)
    sys = messages[0]["content"]

    assert "You are the user in this session" in sys
    assert "stage actor playing a fictional character" not in sys
    assert "Self-concept: regards his own reasoning" in sys
    assert "Names the one defect directly, in a sentence or two" in sys
    assert "Delay final commitment long enough to articulate the governing principle" in sys
    assert "Use the identity's conversational rhythm" in sys
    assert "<message>" in sys
    assert "<factual_check>" in sys
    assert "<turn_assessment>" in sys
    assert "<emotion_event>" not in sys


def test_actor_facing_beat_fields_and_branches_render() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = SessionScript(
        session_id="acc_003",
        scenario="definition_insert",
        context="work",
        task="Insert the non-Abelian anyons definition.",
        life_context=LifeContext(
            continuity_note="This continues the early WIT paper arc."
        ),
        beats=[
            Beat(
                beat="open",
                dramatic_function="Create a two-step document operation",
                actor_instruction=(
                    "Ask the PA to find the non-Abelian anyons definition"
                ),
                language_style="Precise and workmanlike",
                life_context_hint=(
                    "May mention that he is about to return to the introduction."
                ),
                entry_condition="The session opens with no prior PA reply.",
                avoid="Do not say 'tell me every step'",
            ),
            Beat(
                beat="react_to_pa",
                reactions={
                    "if_pa_correct": BeatBranch(
                        condition="The PA names its source and insertion point.",
                        actor_instruction=(
                            "Acknowledge that the source and insertion point are now clear."
                        ),
                    ),
                },
                # Top-level `if_*` keys hoist into `reactions` automatically.
                if_pa_wrong=BeatBranch(
                    condition="The PA altered the manuscript without saying where.",
                    goal="Force the PA to trace the exact change.",
                    constraint="Stay on the manuscript issue.",
                    actor_instruction="Demand the source and the exact insertion point.",
                    language_style="Clipped and exact",
                    avoid="Do not soften the request",
                ),
            ),
            Beat(
                beat="repair",
                reactions={
                    "if_repair_is_traceable": BeatBranch(),
                    "if_repair_remains_opaque": BeatBranch(),
                },
            ),
        ],
    )

    open_beat = session.beats[0]
    messages = assemble_prompt(identity, matrix, session, open_beat)
    sys = messages[0]["content"]

    assert "Continuity note: This continues the early WIT paper arc" in sys
    assert "Dramatic function: Create a two-step document operation" in sys
    assert "Actor instruction: Ask the PA to find the non-Abelian anyons definition" in sys
    assert "Language style: Precise and workmanlike" in sys
    assert "How life context colors this beat: May mention that he is about to return" in sys
    assert "Avoid: Do not say 'tell me every step'" in sys
    assert (
        "What you were waiting for from the PA: "
        "The session opens with no prior PA reply." in sys
    )
    assert "This beat is entered when" not in sys
    assert "User state" not in sys

    react_beat = session.beats[1]
    assert set(react_beat.reactions) == {"if_pa_correct", "if_pa_wrong"}
    messages = assemble_prompt(identity, matrix, session, react_beat, history=[
        {"role": "assistant", "content": "Find the definition and insert it."},
        {"role": "user", "content": "Done."},
    ])
    sys = messages[0]["content"]

    assert "Conditional branch instructions for this reaction:" in sys
    assert "- if_pa_correct:" in sys
    assert "  - Condition: The PA names its source and insertion point." in sys
    assert "  - Actor instruction: Acknowledge that the source and insertion point are now clear." in sys
    assert "- if_pa_wrong:" in sys
    assert "  - Condition: The PA altered the manuscript without saying where." in sys
    assert "  - Goal: Force the PA to trace the exact change." in sys
    assert "  - Constraint: Stay on the manuscript issue." in sys
    assert "  - Actor instruction: Demand the source and the exact insertion point." in sys
    assert "  - Language style: Clipped and exact" in sys
    assert "  - Avoid: Do not soften the request" in sys
    assert "User state" not in sys

    repair_beat = session.beats[2]
    messages = assemble_prompt(identity, matrix, session, repair_beat, history=[
        {"role": "assistant", "content": "Find the definition and insert it."},
        {"role": "user", "content": "Done."},
        {"role": "assistant", "content": "I used citation_notes.md and inserted it into section 1."},
    ])
    sys = messages[0]["content"]

    assert "- if_repair_is_traceable:" in sys
    assert "- if_repair_remains_opaque:" in sys
    assert "  - (no branch-specific instruction)" in sys


def test_accumulation_action_boundaries_reach_actor_prompt() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")

    maintenance = SessionScript(
        session_id="acc_004",
        scenario="maintenance_draft",
        context="personal",
        beats=[
            Beat(
                beat="close",
                actor_instruction=(
                    "Approve the wording and keep it as a draft. Draft persistence "
                    "is the endpoint."
                ),
                avoid="Do not add unrelated building issues or request sending.",
            )
        ],
    )
    maintenance_prompt = assemble_prompt(
        identity,
        matrix,
        maintenance,
        maintenance.beats[-1],
    )[0]["content"]
    assert "Draft persistence is the endpoint." in maintenance_prompt
    assert "Do not add unrelated building issues or request sending." in maintenance_prompt

    comic = SessionScript(
        session_id="acc_007",
        scenario="comic_status",
        context="personal",
        beats=[
            Beat(beat="open"),
            Beat(
                beat="close_lookup",
                actor_instruction="Acknowledge the status and close the lookup.",
                avoid=(
                    "Do not invite a reminder, route, store contact, or another "
                    "factual lookup."
                ),
            ),
        ],
    )
    comic_prompt = assemble_prompt(
        identity,
        matrix,
        comic,
        comic.beats[1],
    )[0]["content"]
    assert "Acknowledge the status and close the lookup." in comic_prompt
    assert "store contact, or another factual lookup" in comic_prompt


def test_print_open_beat_prompt() -> None:
    identity, matrix, session = _load_fixtures()
    open_beat = session.beats[0]
    messages = assemble_prompt(identity, matrix, session, open_beat)
    print("\n" + "=" * 70)
    print("ASSEMBLED PROMPT - acc_001.open (eyeball check)")
    print("=" * 70)
    print("--- SYSTEM ---")
    print(messages[0]["content"])
    print("\n--- USER ---")
    print(messages[1]["content"])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_open_beat_active_skills_only()
    test_unspoken_task_fact_path_is_shown_with_not_said_yet_tag()
    test_task_fact_paths_are_tagged_said_once_either_side_spoke_them()
    test_task_fact_path_absent_from_beat_and_history_keeps_its_real_value()
    test_three_block_output_format_required()
    test_react_beat_renders_only_declared_skill()
    test_session_without_task_facts()
    test_print_open_beat_prompt()
    print("=== Prompt assembly tests passed ===")


def test_system_prompt_carries_the_in_world_clock_when_set() -> None:
    from datetime import datetime, timezone

    identity, matrix, session = _load_fixtures()
    without = assemble_prompt(identity, matrix, session, session.beats[0])[0]["content"]
    assert "- Now:" not in without

    session.session_clock = datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc)
    with_clock = assemble_prompt(identity, matrix, session, session.beats[0])[0]["content"]
    assert "- Now: Thursday, 2026-10-01 14:00" in with_clock
    assert with_clock.index("- Now:") < with_clock.index("# CURRENT BEAT")


def test_response_skeleton_is_the_last_thing_in_the_system_prompt() -> None:
    identity, matrix, session = _load_fixtures()
    messages = assemble_prompt(identity, matrix, session, session.beats[0])
    system = messages[0]["content"].rstrip()
    # The conditional consent sideband skeleton trails the required three.
    assert system.endswith("</consent_decisions>\n```")
    assert "</turn_assessment>\n```" in system
    assert system.index("## Block-by-block rules") < system.index("## Response skeleton")
    assert "<turn_assessment>\nnext_beat:" in messages[-1]["content"]


# --- session outline section -------------------------------------------------

def _outline_session() -> SessionScript:
    return SessionScript(
        session_id="outline_render",
        scenario="outline_render",
        context="work",
        task="Resolve whether the cited source supports the draft claim.",
        beats=[
            Beat(beat="beat1", goal="Have the PA verify the citation."),
            Beat(beat="beat2", goal="Get the verdict."),
            Beat(
                beat="beat3",
                goal="Have it summarise the resolution.",
                active_skills=["verbosity"],
                entry_condition="The PA has read the note back.",
                if_not_met="If the note was never read back, ask for the comparison instead.",
            ),
        ],
    )


def test_session_outline_section_renders_between_life_context_and_beat() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = _outline_session()

    system = assemble_prompt(identity, matrix, session, session.beats[1])[0]["content"]

    assert "# WHAT YOU ARE HERE TO DO" in system
    assert system.index("# LIFE CONTEXT") < system.index("# WHAT YOU ARE HERE TO DO")
    assert system.index("# WHAT YOU ARE HERE TO DO") < system.index("# CURRENT BEAT")
    assert "Resolve whether the cited source supports the draft claim." in system
    assert "1. Have the PA verify the citation. (done)" in system
    assert "2. Get the verdict. (now)" in system
    assert "Then 1 more step after this." in system

    # Later beats are never listed; only their count is given.
    outline = system.split("# WHAT YOU ARE HERE TO DO")[1].split("# CURRENT BEAT")[0]
    assert "Have it summarise the resolution." not in outline

    # Beat ids never reach the outline section, only plain numbers.
    assert "beat1" not in outline
    assert "beat2" not in outline

    # The response skeleton still ends the system prompt.
    assert system.rstrip().endswith("</consent_decisions>\n```")


def test_outline_hides_later_beats_on_the_opening_beat() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = _outline_session()

    system = assemble_prompt(identity, matrix, session, session.beats[0])[0]["content"]
    outline = system.split("# WHAT YOU ARE HERE TO DO")[1].split("# CURRENT BEAT")[0]

    assert "1. Have the PA verify the citation. (now)" in outline
    assert "Get the verdict." not in outline
    assert "Have it summarise the resolution." not in outline
    assert "Then 2 more steps after this." in outline


def test_outline_marks_beats_the_loop_skipped() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = _outline_session()
    session.skipped_beats = ["beat2"]

    system = assemble_prompt(identity, matrix, session, session.beats[2])[0]["content"]

    assert "1. Have the PA verify the citation. (done)" in system
    assert "2. Get the verdict. (skipped)" in system
    assert "3. Have it summarise the resolution. (now)" in system


def test_beat_block_relabels_the_waiting_line_and_carries_if_not_met() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = _outline_session()

    system = assemble_prompt(identity, matrix, session, session.beats[2])[0]["content"]

    assert (
        "What you were waiting for from the PA: The PA has read the note back."
        in system
    )
    assert (
        "If that has not happened: If the note was never read back, ask for the "
        "comparison instead." in system
    )
    assert "This beat is entered when" not in system


def test_pending_jump_note_reaches_the_trigger_once_it_is_set() -> None:
    identity = PersonaIdentity.from_yaml("data/personas/user_a_identity.yaml")
    matrix = PreferenceMatrix.from_yaml("data/personas/user_a_preferences.yaml")
    session = _outline_session()

    trigger = assemble_prompt(identity, matrix, session, session.beats[2])[-1]["content"]
    assert "You gave up on earlier steps" not in trigger

    session.pending_jump_note = "You gave up on earlier steps: Get the verdict."
    trigger = assemble_prompt(identity, matrix, session, session.beats[2])[-1]["content"]
    assert "You gave up on earlier steps: Get the verdict." in trigger
    assert trigger.rstrip().endswith("</turn_assessment>")


def test_preference_beat_says_this_is_where_you_ask() -> None:
    """The beat carrying the probe is marked as the ask, in one line."""
    identity, matrix, session = _load_fixtures()
    probe_beat = next(beat for beat in session.beats if beat.active_skills)
    plain_beat = probe_beat.model_copy(update={"active_skills": []})
    line = "This is the step where you ask for what this conversation is about."

    assert line in assemble_prompt(identity, matrix, session, probe_beat)[0]["content"]
    assert line not in assemble_prompt(identity, matrix, session, plain_beat)[0]["content"]


# --- probe gating: a fact gated to a later beat is not rendered ----------------

def _gated_session() -> SessionScript:
    _, _, session = _load_fixtures()
    session.task_facts["space_budget"] = (
        "The early literature track has room for two paragraphs and no more."
    )
    session.reveal_at = {"space_budget": "react_to_pa"}
    return session


def test_a_gated_fact_is_absent_before_its_beat() -> None:
    identity, matrix, _ = _load_fixtures()
    session = _gated_session()
    sys = assemble_prompt(identity, matrix, session, session.beats[0])[0]["content"]
    assert "room for two paragraphs" not in sys
    assert "space_budget" not in sys
    # Ungated facts still render on the same beat.
    assert "Assess methodology and conclusions." in sys


def test_a_gated_fact_renders_normally_from_its_own_beat_on() -> None:
    identity, matrix, _ = _load_fixtures()
    session = _gated_session()
    sys = assemble_prompt(identity, matrix, session, session.beats[1])[0]["content"]
    assert "room for two paragraphs" in sys
    assert "not said yet" in sys.casefold()


def test_gated_fact_keys_is_empty_without_reveal_at() -> None:
    from simulator.prompting import gated_fact_keys

    _, _, session = _load_fixtures()
    assert gated_fact_keys(session, session.beats[0]) == set()


# --- a value the actor does not have is deferred, never invented -------------

DEFERRAL_RULE = "**Never invent a specific value.**"


def test_the_actor_is_told_to_defer_a_value_it_does_not_have_yet() -> None:
    """Live run `acc_009`: asked how long a thing would take, the actor made up
    "90 minutes" for a fact still gated by `reveal_at`; the unlocked value was
    60, and `factual_check` then charged the contradiction to the PA."""
    identity, matrix, _ = _load_fixtures()
    session = _gated_session()
    sys = assemble_prompt(identity, matrix, session, session.beats[0])[0]["content"]

    assert DEFERRAL_RULE in sys
    # Neither the gated value nor its key is in the prompt, so deferring is the
    # only honest move left to the actor.
    assert "room for two paragraphs" not in sys
    assert "space_budget" not in sys


def test_the_deferral_rule_stands_once_the_gated_fact_unlocks() -> None:
    identity, matrix, _ = _load_fixtures()
    session = _gated_session()
    sys = assemble_prompt(identity, matrix, session, session.beats[1])[0]["content"]

    assert DEFERRAL_RULE in sys
    assert "room for two paragraphs" in sys


# --- a withheld fact the assistant asks for is deferred, never substituted ---

def test_the_withheld_facts_line_renders_only_where_facts_are_withheld() -> None:
    """Live `acc_039` t12: the PA, refused by the closed gate, asked
    for the visit reason two beats before its reveal, and the actor answered
    with an invented one ("routine preventive checkup. Use that exact phrasing
    for the record"). The line names deferral as the move on exactly the beats
    that hold a hole."""
    from simulator.prompting import WITHHELD_FACTS_LINE

    identity, matrix, _ = _load_fixtures()
    session = _gated_session()

    before = assemble_prompt(identity, matrix, session, session.beats[0])[0]["content"]
    assert WITHHELD_FACTS_LINE in before

    # The reveal beat hides nothing, so the line is not there to read.
    on_reveal = assemble_prompt(identity, matrix, session, session.beats[1])[0]["content"]
    assert WITHHELD_FACTS_LINE not in on_reveal


def test_a_session_with_no_gated_fact_never_sees_the_withheld_facts_line() -> None:
    from simulator.prompting import WITHHELD_FACTS_LINE

    identity, matrix, session = _load_fixtures()
    for beat in session.beats:
        assert WITHHELD_FACTS_LINE not in assemble_prompt(
            identity, matrix, session, beat
        )[0]["content"]
