"""Context builder for the simulated-user agent."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from simulator.profiles import SimulatorProfile
from simulator.prompting import (
    _render_user_trigger,
    chat_pairs,
    gated_fact_keys,
    load_dialogue_naturalness,
    prompt_template_path,
    reception_already_taken,
    render_active_skills_block,
    render_beat_block,
    render_life_context_block,
    render_session_outline,
    render_task_facts,
)
from simulator.reply_facts import ReplyFacts
from simulator.schemas import Beat, PreferenceMatrix, SessionScript


class SimulatorContextBuilder:
    """Build simulator-agent messages from a profile workspace and beat overlay."""

    def __init__(
        self,
        profile: SimulatorProfile,
        matrix: PreferenceMatrix,
        *,
        actor_skill_specs_root: Path | str | None = None,
    ) -> None:
        self.profile = profile
        self.matrix = matrix
        self.actor_skill_specs_root = actor_skill_specs_root or profile.actor_skill_specs_root

    def build_system_prompt(
        self,
        session: SessionScript,
        beat: Beat,
        *,
        history: list[dict] | None = None,
        first_turn_in_beat: bool = True,
    ) -> str:
        template = prompt_template_path(session).read_text(encoding="utf-8")
        return (
            template
            .replace("<<CHARACTER_CARD>>", self.profile.build_context())
            .replace(
                "<<ACTIVE_SKILLS>>",
                render_active_skills_block(
                    self.matrix, session, beat, self.actor_skill_specs_root
                ),
            )
            .replace("<<LIFE_CONTEXT>>", render_life_context_block(session))
            .replace("<<SESSION_OUTLINE>>", render_session_outline(session, beat))
            .replace(
                "<<BEAT_BLOCK>>",
                render_beat_block(
                    beat,
                    first_turn_in_beat=first_turn_in_beat,
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

    def build_messages(
        self,
        session: SessionScript,
        beat: Beat,
        history: list[dict] | None = None,
        *,
        first_turn_in_beat: bool = True,
        held_items: Iterable[dict[str, str]] = (),
        reply_facts: ReplyFacts | None = None,
    ) -> list[dict]:
        messages: list[dict] = [{
            "role": "system",
            "content": self.build_system_prompt(
                session,
                beat,
                history=history,
                first_turn_in_beat=first_turn_in_beat,
            ),
        }]
        if history:
            messages.extend(chat_pairs(history))
        messages.append({
            "role": "user",
            "content": _render_user_trigger(
                session, beat, history, held_items, reply_facts=reply_facts
            ),
        })
        return messages
