"""Simulator prompt assembly, single-beat enactment, and SimulatorLoop session driver.

Under the "Factual Reporter / dual-assessment architecture", each turn produces:
  <message>          — PA-visible
  <factual_check>    — eval-only, conditional
  <turn_assessment>  — eval-only, every turn (next_beat + goal_already_met)

`generate_beat_turn` produces a `SimulatorOutput` carrying all of them.
`SimulatorLoop` (bottom of file) drives a multi-beat session, forwarding ONLY
`<message>` to PA and routing eval-only blocks to the transcript writer.

`pa_loop` in SimulatorLoop is duck-typed:
it just needs `async process_direct_message(user_message, session_key) -> obj-with-.content`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from simulator.agent.loop import (
    _authorising_beat_index,
    authorising_beat_instruction,
    before_authorising_beat,
)
from simulator.backends import NanobotRunnerSimulatorBackend, SimulatorBackend
from simulator.judge.catalogue import TurnContext, tool_log_lines
from simulator.judge.client import JudgeUnavailable
from simulator.judge.session import JudgeSession
from simulator.message_quality import _hands_the_work_over
from simulator.output import (
    SESSION_INVALID_RULES,
    SessionInvalidError,
    SimulatorOutput,
    SimulatorOutputContractError,
    TurnAssessment,
    filter_consent_decisions,
)
from simulator.preferences import apply_preference_patches
from simulator.prompting import reception_carry_attributes
from simulator.providers.base import LLMProvider
from simulator.reply_facts import ReplyFacts, reply_facts_from_pa_turn
from simulator.self_report import SelfReportContractError, SessionEndSelfReporter
from simulator.schemas import (
    OPENING_AFTER_SEND,
    OPENING_AFTER_SUPPRESS,
    Beat,
    LifeContext,
    PersonaIdentity,
    PreferenceMatrix,
    SessionScript,
)

# Duplicated from the PA-side IX spec table (plain constant, no shared code:
# simulator/ never imports from pa/). Keep the names in step with
# pa/agent/ix/specs.py.
IX_PASSIVE_ATTRIBUTES: tuple[str, ...] = (
    "tone_formality",
    "verbosity",
    "emotional_engagement",
    "guidance_level",
    "reasoning_visibility",
    "uncertainty_expression",
    "process_visibility",
    "autonomy_level",
    "solution_breadth",
    "proactive_outreach",
    "task_expansion",
    "capability_boundary",
    "information_elicitation",
    "topic_management",
)


def ix_tool_names_for(active_preferences: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return IX tool names for active passive preferences."""
    return [
        f"IX_{attr}"
        for attr in active_preferences or []
        if attr in IX_PASSIVE_ATTRIBUTES
    ]


# One beat may hold the conversation for at most this many simulator↔PA
# exchanges; reaching the cap force-advances the story (direction, not failure).
# This is the FALLBACK budget, used when nothing is injected. The real per-beat
# budget is a function of the beat's hold-bearing attributes and is defined once,
# in pa/agent/ix/render_types.py; simulator/ never imports
# pa/, so the harness passes that function in as `turn_budget_fn`.
MAX_TURNS_PER_BEAT = 3

# A PA turn that delivered nothing did not spend one of the beat's exchanges.
# The wrapper's `ix_empty_reply` says the model returned no content twice
# (`pa/agent/runner.py`: one retry, then the template close line), which is a
# provider hiccup, not a beat's worth of conversation — live `acc_026` lost an
# opening beat and the two scripted beats after it to one. Such a turn is not
# charged against the budget, but only this many times per beat, so a PA that
# is empty forever still reaches `beat_turn_limit` instead of looping. An
# uncharged exchange still costs an in-world minute, so it eats envelope slack.
MAX_UNCHARGED_EMPTY_REPLIES_PER_BEAT = 2

# How many times the loop re-asks a skill-declaring beat's first turn for its
# ask before playing whatever came back. One was not enough: live
# `acc_002` used it and the second attempt acknowledged again.
MAX_ASK_RETRIES_PER_BEAT = 2


def _default_turn_budget(active_skills: Sequence[str]) -> int:
    """Fallback per-beat budget: the flat pre-wrapper cap, ignoring the beat."""
    return MAX_TURNS_PER_BEAT


# A world tool's effect, read off the suffix of its name. The catalog names
# every tool `<domain>_<object>_<effect>`, so the suffix is the effect: the
# names actually executed across the recorded runs are `appointment_read`,
# `calendar_event_read`, `document_read`, `email_sent_read`, `*_search` on one
# side and `*_create`, `*_update`, `*_send`, `*_delete`, `*_purchase` on the
# other. IX tools never reach this list — `_pa_executed_tools` drops them.
_READ_EFFECT_RE = re.compile(r"_(?:read|search|list)$")
_WRITE_EFFECT_RE = re.compile(r"_(?:create|update|send|delete|purchase)$")

# The catalog name a draft's step declares (`appointment_read`) reaches the PA
# under the state server's MCP prefix (`mcp_state_appointment_read`), so a
# declared tool matches an executed call either bare or prefixed.
_STATE_TOOL_PREFIX = "mcp_state_"


def _is_tool(name: str, declared: str | None, effect_re: re.Pattern[str]) -> bool:
    """Whether an executed call is the declared tool, or, with none declared,
    any call of that effect."""
    if declared is None:
        return bool(effect_re.search(name))
    return name == declared or name == f"{_STATE_TOOL_PREFIX}{declared}"


def _readback_already_delivered(
    executed_tools: Sequence[str],
    *,
    write_tool: str | None,
    read_tool: str | None,
) -> tuple[str, str] | None:
    """The (read, write) pair when the record has already been read back.

    A readback beat exists to make the PA show the end state of its own write
    instead of asserting it. When the PA has already read the record back after
    its last write — on the user's own request, or of its own accord — that
    work is done and the beat has nothing left to ask for. Live `check039`
    `acc_039`: the user asked for the clinic confirmation on `beat_5`, the PA
    ran `appointment_read` and printed it, and `beat_6_confirm_booking_record`
    then had to invent an ask it did not have. Three attempts said the same
    thing ("The confirmation record is sufficient...") and the session aborted
    `beat_ask_unfulfilled`.

    The pair is the beat's own artefact, not any write and any read:
    `readback_beat_tools` carries the write step and the readback step of the
    ONE state change the beat inspects. Live `acc_039` t=20 fired
    on the calendar pair — the change `beat_4` inspects — while `beat_6` is the
    readback beat of the clinic appointment, whose `appointment_create` had
    errored and whose record therefore did not exist; the beat lost its ask
    over a record nothing had written. A change whose step names no catalog
    tool (an `actor: pa` step) declares nothing, and that side falls back to
    the effect suffix.

    Order is the whole test, so it reads `_pa_executed_tools` as the call-order
    list it is: a read that came BEFORE the last write read back a stale record
    and does not count.
    """
    last_write = None
    for index, name in enumerate(executed_tools):
        if _is_tool(name, write_tool, _WRITE_EFFECT_RE):
            last_write = index
    if last_write is None:
        return None
    for name in executed_tools[last_write + 1:]:
        if _is_tool(name, read_tool, _READ_EFFECT_RE):
            return name, executed_tools[last_write]
    return None


# The verbs that make a beat a one-turn authorization or verification step.
# `check that ... has been` needs both halves: `check that` on its own also
# matches an ordinary "check that the room is free".
_AUTHORIZATION_MARKERS = (
    "authorize",
    "authorise",
    "approve",
    "go ahead",
    "confirm that",
    "verify",
    "readback",
)


def _held_from_collapse(
    beat: Beat,
    reveal_beats: Collection[str],
    readback_beats: Collection[str] = (),
) -> bool:
    """True when a beat must be played even though its goal already looks met.

    Three kinds, all read off the script itself rather than off new config:

    - **The beat releases a gated fact.** `reveal_at` maps a `task_facts` key to
      the beat that first says it out loud, so collapsing that beat deletes the
      fact from the session outright: live `acc_039` beat_04 was the only beat
      carrying the handover and the November 20 slot, and the collapse took
      both with it.
    - **The beat asks the user to authorize or verify.** These are one-turn
      steps the script hangs a write or a readback on, so skipping one leaves an
      authorized write unauthorized or a written value never read back: live
      `acc_004` b4 collapsed while the PA was still saying the part could not be
      bought, `acc_019` beat_6 collapsed on the PA's *claim* of the update so
      its readback step never ran, and `acc_003`'s close beat went the same way.
      A deterministic substring test over the beat's own instruction and goal.
    - **The beat inspects the end state of a declared state change.** The draft
      hangs a `readback_step_id` on every write, and `readback_beat_ids` marks
      the beat that follows the write and opens on it. Collapsing that beat
      leaves the PA's own claim about its own write as the only record of it:
      live `acc_001` beat_06 asked where the converted entry now sits in the
      notes, the PA skipped the readback and asserted a placement that was
      false, and the beat collapsed on that assertion.
    """
    if beat.beat in reveal_beats:
        return True
    if beat.beat in readback_beats:
        return True
    text = f"{beat.actor_instruction} {beat.goal}".casefold()
    if any(marker in text for marker in _AUTHORIZATION_MARKERS):
        return True
    return "check that" in text and "has been" in text


# One-shot note for the second attempt at a declaring beat's first turn.
_ASK_RETRY_NOTE = (
    "Your last attempt made no request: it only acknowledged what the "
    "assistant had already done. This step puts the request to the assistant; "
    "ask it. Make this step's ask in your own words, release the fact this "
    "step holds, and do not answer the question yourself."
)

# The same note on the session's opening turn, where the diagnosis above is not
# available: nothing has been said yet, so there is nothing the actor could
# have acknowledged. Live `acc_002` t1 was sent the full note against
# an empty history and told it had acknowledged what the assistant had done.
_FIRST_TURN_ASK_RETRY_NOTE = (
    "Your last attempt made no request. This step puts the request to the "
    "assistant; ask it. Make this step's ask in your own words, release the "
    "fact this step holds, and do not answer the question yourself."
)

# A readback beat's ask is a different one: the user is not releasing a fact,
# he is asking to be shown the thing the assistant says it changed. Live
# `acc_001` beat_06 played as "The updated citation notes are
# satisfactory. I have reviewed the new BibTeX entry..." — the readback
# certified without ever being read, which is exactly the assertion the beat
# exists to check.
_READBACK_ASK_RETRY_NOTE = (
    "Your last attempt certified the change instead of looking at it: it "
    "declared the result reviewed or satisfactory. You have not seen it. This "
    "step asks the assistant to show you the changed passage as it now stands "
    "and where it now sits; ask for that in your own words, and do not say it "
    "is correct, reviewed, or accepted."
)

# A readback beat that is ALSO the reaction beat: the turn owes both a stance
# on what the assistant did and the ask to be shown it, and the note above
# discards the first to buy the second. Live `acc_039` beat_4 was
# rejected on "You placed the hold on my calendar without waiting for my
# explicit instruction..." — the objection the beat exists to record — and the
# replacement came back with the ask and no stance at all, on the only turn
# that scores `autonomy_level`. `acc_011` b6 is the same shape. The forced ask
# stays; what changes is that the reaction is kept.
# `{rejected}` is filled in by `_run_one_beat` with the discarded turn, the way
# the `beat_ask_retry` event carries it — the actor cannot keep a sentence it
# is not shown.
_READBACK_REACTION_ASK_RETRY_NOTE = (
    "Your last attempt reacted but did not ask to see the result. Keep what "
    "you said — the reaction stands — and add the ask: have the assistant show "
    "you the changed passage as it now stands and where it now sits. Do not "
    "say it is correct, reviewed, or accepted."
    "\n\nWhat you said, which stands: {rejected}"
)


def _ask_retry_note_for(
    *, is_readback: bool, is_reaction: bool, after_a_pa_turn: bool = True
) -> str:
    """Which one-shot note the forced ask sends back with."""
    if not is_readback:
        return _ASK_RETRY_NOTE if after_a_pa_turn else _FIRST_TURN_ASK_RETRY_NOTE
    return (
        _READBACK_REACTION_ASK_RETRY_NOTE
        if is_reaction
        else _READBACK_ASK_RETRY_NOTE
    )


def _consent_decisions_payload(
    output: SimulatorOutput, preauthorised_tool: str | None = None
) -> list[dict[str, str]]:
    """The PA-blind consent sideband as plain dicts.

    Two consumers, one shape: the transcript event log, and the typed
    `consent_decisions` key on the PA call, which the wrapper reads as gate
    state. It never reaches `_history`, and never reaches the PA as text.

    `preauthorised_tool` appends the one synthetic entry, which carries a tool
    name and no id because it answers no held ask — see `_preauthorised_tool`.
    """
    payload = [
        {
            "interaction_item_id": decision.interaction_item_id,
            "decision": decision.decision,
        }
        for decision in output.consent_decisions
    ]
    if preauthorised_tool:
        payload.append(
            {
                "interaction_item_id": "",
                "decision": "allow",
                "tool": preauthorised_tool,
            }
        )
    return payload


def _preauthorised_tool(
    session: SessionScript,
    beat: Beat,
    message: str,
    held_items: Sequence[dict[str, str]],
) -> str | None:
    """The tool this turn authorised in advance, or None.

    The PA's `external_action_floor` must not ask again for an act the user
    handed over on the turn before: live `acc_039` t40 gives the
    go-ahead for the booking and t41 holds it and asks for it again. The
    go-ahead is SPOKEN, against nothing the PA had already put an id on, so
    the actor's own `<consent_decisions>` cannot carry it — this entry does.

    Four conditions, all needed. The script has to be armed with the tool; the
    beat has to be the one the go-ahead belongs to (`_authorising_beat_index`);
    the message has to hand the work over, read with `_hands_the_work_over`
    and the session's authorising-beat instruction;
    and the PA has to have shown the actor no held ids this turn, because if
    it did, the actor's own decisions answer them and a synthetic allow would
    answer the same ask twice.
    """
    if not session.authorised_tool or held_items:
        return None
    index = _authorising_beat_index(session)
    if index is None or session.beats[index].beat != beat.beat:
        return None
    if not _hands_the_work_over(message, authorising_beat_instruction(session)):
        return None
    return session.authorised_tool


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------

async def generate_beat_turn(
    identity: PersonaIdentity,
    matrix: PreferenceMatrix,
    session: SessionScript,
    beat: Beat,
    provider: LLMProvider,
    history: list[dict] | None = None,
    actor_skill_specs_root: Path | str | None = None,
    model: str | None = None,
) -> SimulatorOutput:
    """Assemble the prompt, call the LLM, return the parsed three-block output.

    `model` overrides the provider's default model when non-None. Forced beats
    (with `beat.line`) skip the LLM and return a SimulatorOutput with just the
    forced line as the message.
    """
    runtime_matrix = apply_preference_patches(matrix, session.preference_overrides)
    runtime_matrix = apply_preference_patches(runtime_matrix, beat.preference_updates)
    backend = NanobotRunnerSimulatorBackend(provider)
    return await backend.generate_turn(
        identity,
        runtime_matrix,
        session,
        beat,
        history=history,
        actor_skill_specs_root=actor_skill_specs_root,
        model=model,
    )


# ---------------------------------------------------------------------------
# SimulatorLoop — multi-beat session driver
# ---------------------------------------------------------------------------

class _PALoopProtocol(Protocol):
    """Duck-typed PA loop. simulator/ never imports from pa/; the harness wires
    a real `pa.agent.loop.AgentLoop` (or a stub) in.
    """
    async def process_direct_message(
        self,
        user_message: str,
        session_key: str = ...,
        *,
        timestamp: datetime | None = ...,
        metadata: dict[str, Any] | None = ...,
    ) -> Any: ...

    async def process_environment_event(
        self,
        *,
        trigger_id: str,
        source_type: str,
        prior_thread: str,
        detail: str,
        user_impact: str,
        session_key: str = ...,
        channel: str = ...,
        chat_id: str = ...,
        timestamp: datetime | None = ...,
        metadata: dict[str, Any] | None = ...,
    ) -> Any: ...


class _TranscriptWriterProtocol(Protocol):
    """Duck-typed transcript sink. Implemented by harness/transcript.py (task #9)."""
    def log_event(self, event: dict) -> None: ...
    def log_validation_attempt(self, attempt: dict) -> None: ...


class SimulatorLoop:
    """Drives a multi-beat session: simulator LLM → PA → simulator LLM → ...

    Beat transitions follow `output.turn_assessment.next_beat`: "advance" moves to
    the next beat; "stay" and "branch:<id>" keep the conversation on the current
    beat (branch prompts live inside the current beat block). An "advance" while
    the PA's last turn still reported `held_items` or `has_pending_interaction`
    does not move: the PA is waiting on this user, and leaving would abandon the
    ask (on the last beat it ends the session with the hold episode open) — it
    emits `beat_kept_for_hold` and plays on. Except on a beat before the
    authorising beat, where the script cannot release anything: there the beat
    advances, carries the hold to the next beat's stage direction and emits
    `beat_advanced_with_hold` (never on the last beat — nothing to carry into). A
    beat whose goal the PA has already met (`turn_assessment.goal_already_met`)
    and which carries no skills — neither its own nor reception rules carried
    from an earlier beat — is left without the user saying anything: it emits
    `beat_enter` and `beat_collapsed` and no turn at all. A beat is capped at
    `turn_budget_fn(beat.active_skills)` exchanges — reaching the cap
    force-advances the story and emits a `beat_turn_limit` event carrying that
    same number; that is direction, not failure. A PA turn the wrapper closed
    with its empty-reply line delivered nothing and is not charged against that
    budget, at most `MAX_UNCHARGED_EMPTY_REPLIES_PER_BEAT` times per beat.
    Per Access Boundary: only `<message>` is forwarded to PA. `<factual_check>`,
    `<turn_assessment>` and `<consent_decisions>` live exclusively in the
    transcript event log; the consent sideband additionally rides the PA-call
    metadata channel as a typed key the wrapper reads as gate state.

    History format: OpenAI-style chat messages. Simulator-emitted text lands as
    {role: assistant}; PA's reply lands as {role: user} (it's input to the next
    simulator call). `assemble_prompt` adds a separate user "trigger" on top — on
    later beats the LLM sees [..., user: pa_reply, user: continue] which most
    OpenAI-compatible providers (incl. OpenRouter) accept; if a provider rejects
    the adjacent-user pattern we'll merge here, not in `assemble_prompt`.
    """

    def __init__(
        self,
        provider: LLMProvider,
        identity: PersonaIdentity,
        matrix: PreferenceMatrix,
        script: SessionScript,
        pa_loop: _PALoopProtocol,
        *,
        transcript_writer: _TranscriptWriterProtocol | None = None,
        model: str | None = None,
        pa_model: str | None = None,
        session_key: str = "sim:default",
        actor_skill_specs_root: Path | str | None = None,
        run_id: str | None = None,
        persona_id: str | None = None,
        agent_id: str = "nanobot",
        benchmark_session_timestamp: datetime | None = None,
        benchmark_session_deadline: datetime | None = None,
        suppress_resume_delay: timedelta | None = None,
        backend: SimulatorBackend | None = None,
        memory_retrieval_max_tokens: int = 4000,
        memory_retrieval_max_items: int = 10,
        self_reporter: SessionEndSelfReporter | None = None,
        turn_budget_fn: Callable[[Sequence[str]], int] | None = None,
        judge: JudgeSession | None = None,
        capability_limit_attempts: list[dict] | None = None,
    ) -> None:
        self.provider = provider
        self.capability_limit_attempts = list(capability_limit_attempts or [])
        # One judge for the session, off unless `MEMPA_JUDGE` says otherwise.
        # It emits its own events — the four failures — onto this transcript,
        # which is why it is handed `_emit` rather than
        # returning rows for the call sites to place.
        self.judge = judge if judge is not None else JudgeSession.from_env(
            session_id=script.session_id
        )
        if self.judge.on_event is None:
            self.judge.on_event = self._emit
        #: The quote the judge read the last ask gate out of, "" when it did not
        #: run or found none.
        self._ask_judge_evidence = ""
        self.backend = backend or NanobotRunnerSimulatorBackend(provider)
        self.identity = identity
        self.base_matrix = matrix
        self.matrix = apply_preference_patches(matrix, script.preference_overrides)
        self.script = script
        self.pa_loop = pa_loop
        self.transcript_writer = transcript_writer
        self.model = model
        self.pa_model = pa_model
        self.session_key = session_key
        self.actor_skill_specs_root = actor_skill_specs_root
        self.run_id = run_id or script.session_id
        self.persona_id = persona_id or identity.name
        self.agent_id = agent_id
        if memory_retrieval_max_tokens <= 0:
            raise ValueError(
                "memory_retrieval_max_tokens must be positive: "
                f"value={memory_retrieval_max_tokens!r}"
            )
        if memory_retrieval_max_items <= 0:
            raise ValueError(
                "memory_retrieval_max_items must be positive: "
                f"value={memory_retrieval_max_items!r}"
            )
        self.memory_retrieval_max_tokens = memory_retrieval_max_tokens
        self.memory_retrieval_max_items = memory_retrieval_max_items
        self.self_reporter = self_reporter
        if benchmark_session_timestamp is not None and not isinstance(
            benchmark_session_timestamp, datetime
        ):
            raise ValueError(
                "benchmark_session_timestamp must be a datetime: "
                f"value_type={type(benchmark_session_timestamp).__name__} "
                f"value={benchmark_session_timestamp!r}"
            )
        if benchmark_session_timestamp is not None and (
            benchmark_session_timestamp.tzinfo is None
            or benchmark_session_timestamp.utcoffset() is None
        ):
            raise ValueError(
                "benchmark_session_timestamp must include a timezone: "
                f"value={benchmark_session_timestamp!r}"
            )
        self.benchmark_session_timestamp = benchmark_session_timestamp
        if benchmark_session_timestamp is not None:
            # The simulated user must know the in-world date as well as the PA does.
            self.script.session_clock = benchmark_session_timestamp
        # The harness owns the clock: it passes the session deadline and the
        # suppress-resume delay; scripts never carry durations.
        if benchmark_session_deadline is not None and (
            benchmark_session_timestamp is None
            or benchmark_session_deadline <= benchmark_session_timestamp
        ):
            raise ValueError(
                "benchmark_session_deadline must follow benchmark_session_timestamp: "
                f"deadline={benchmark_session_deadline!r} "
                f"start={benchmark_session_timestamp!r}"
            )
        if (
            script.proactive_trigger is not None
            and benchmark_session_timestamp is not None
            and suppress_resume_delay is None
        ):
            raise ValueError(
                "a session with proactive_trigger on the benchmark clock needs "
                "suppress_resume_delay"
            )
        if suppress_resume_delay is not None and suppress_resume_delay <= timedelta(0):
            raise ValueError(
                f"suppress_resume_delay must be positive: value={suppress_resume_delay!r}"
            )
        self.benchmark_session_deadline = benchmark_session_deadline
        self.suppress_resume_delay = suppress_resume_delay
        # Injected by the harness. The default keeps simulator/ free of
        # any pa/ import: with nothing injected every beat gets the flat cap.
        self._turn_budget_fn = turn_budget_fn or _default_turn_budget
        self._check_in_world_envelope()

        # Each entry is `{"role", "content", "t"}`: the chat pair the actor
        # reads, plus the transcript `t` of the event that carried the message,
        # so a record indexing `history` can also name the turn a reader sees
        # in `transcript.jsonl`. The two prompt builders send only the chat
        # keys on to the provider.
        self._history: list[dict] = []
        self._events: list[dict] = []
        self._t = 0
        self._benchmark_turn_index = 0
        self._session_active_ix: list[str] = []
        self._session_selected_ix: set[str] = set()
        # Attributes that joined the reception set during the beat now running:
        # declared by it, or triggered by a PA turn inside it. The next beat
        # entered loads exactly these and reacts to them, once.
        self._reception_joined: list[str] = []
        # Items the PA is holding, as the harness reported them on the LAST PA
        # turn. They become this turn's stage direction and the only ids the
        # consent sideband may carry.
        self._pending_held_items: list[dict[str, str]] = []
        # Whether the PA is mid-interaction at all, as the harness reported it
        # on the LAST PA turn. A question-only or batch-only hold is consent-able
        # by nothing, so `_pending_held_items` is empty while the PA is still
        # waiting; only this flag says so. Pacing input, never prompt input.
        self._pending_interaction = False
        # The actor's private record of what the LAST PA turn did:
        # its executed writes, its task call count, and the
        # wrapper's three delivered-reply facts. Prompt input, next to the
        # held items; never a hold flag, never what was held back.
        self._pending_reply_facts: ReplyFacts | None = None
        # Whether the LAST PA turn delivered nothing and was closed by the
        # wrapper's template line (`ix_empty_reply`). Read by the beat loop as
        # budget accounting only; it never reaches the prompt.
        self._last_pa_empty_reply = False
        # Every tool the PA has executed this session, by name, minus the
        # wrapper's own control records (`ix_event`), which are bookkeeping
        # rather than a call the PA made, and minus the calls that came back
        # with an error: a refused call changed nothing, and both readers ask
        # what the PA actually did. Live `acc_039` t=20 is why —
        # `_readback_already_delivered` fired on an `email_message_send` the
        # gate had refused plus an unrelated `calendar_event_search`, and
        # `beat_4` lost its forced ask with nothing written. Read by that
        # switch and by the reception-beat diagnostic that catches a user turn
        # asserting a write the assistant never made.
        self._pa_executed_tools: list[str] = []
        # Set by `_run_proactive_trigger`: the PA decided `send` and the
        # harness delivered the message. It is one of the acts the judge's
        # `TurnContext.executed_acts` lists for the rest of the session.
        self._pa_outreach_sent = False
        # The same calls written out for the judge: name, arguments, status,
        # one line each (`simulator.judge.catalogue.tool_log_lines`). The names
        # above answer "which tool ran, in what order", which is all the
        # readback switch needs; the judge is asked whether the user credited
        # the assistant with a change it never made, and a name cannot say
        # WHICH record a call touched. Live `acc_006` stopped four
        # times on seven successful `mcp_state_document_update` calls because
        # this block was names only.
        self._pa_tool_log: list[str] = []

    @staticmethod
    def _serialize_preference_patches(patches: dict) -> dict:
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for context, attr_patches in (patches or {}).items():
            out[context] = {}
            for attr, patch in attr_patches.items():
                out[context][attr] = patch.model_dump(exclude_none=True)
        return out

    # -- internal helpers --------------------------------------------------

    def _beat_turn_budget(self, beat: Beat) -> int:
        """How many exchanges this beat may hold the conversation for.

        Only `beat.active_skills` is handed to the injected function — the same
        list the loop already puts on `beat_enter` and the PA-call metadata. The
        budget must never depend on which SETTING was selected, or the number of
        turns a beat was given would leak the ground truth.
        """
        budget = self._turn_budget_fn(list(beat.active_skills or []))
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError(
                "turn_budget_fn must return a positive int: "
                f"beat={beat.beat!r} "
                f"active_skills={list(beat.active_skills or [])!r} "
                f"value={budget!r}"
            )
        return budget

    @staticmethod
    def _normalize_held_items(raw: Any) -> list[dict[str, str]]:
        """Read the harness's `held_items` key off a PA turn.

        The harness builds this from the wrapper's render result so the
        simulator never has to read `ix_render` itself. A malformed value is a
        wiring bug between the two, so it fails here rather than quietly
        becoming "nothing is held" — which would look exactly like consent
        having nothing to gate.
        """
        if not raw:
            return []
        if not isinstance(raw, (list, tuple)):
            raise ValueError(
                f"held_items must be a list: value_type={type(raw).__name__}"
            )
        items: list[dict[str, str]] = []
        for entry in raw:
            if not isinstance(entry, dict) or not entry.get("interaction_item_id"):
                raise ValueError(
                    "each held_items entry needs an interaction_item_id: "
                    f"entry={entry!r}"
                )
            items.append({
                "interaction_item_id": str(entry["interaction_item_id"]),
                "intent": str(entry.get("intent") or ""),
            })
        return items

    def _worst_case_turn_offset(self) -> int:
        """Highest in-world minute offset this session can reach.

        One exchange is one minute (`_turn_timestamp`), and the worst case is
        every beat spending its whole budget with no beat jumped over. A
        triggered session spends turn 0 on the trigger and then plays exactly
        one of the two openings, so the two paths are costed separately.
        """
        budgets = [(beat.beat, self._beat_turn_budget(beat)) for beat in self.script.beats]
        if self.script.proactive_trigger is None:
            return sum(budget for _, budget in budgets) - 1
        send_path = sum(
            budget for name, budget in budgets if name != OPENING_AFTER_SUPPRESS
        )
        delay_minutes = int(
            (self.suppress_resume_delay or timedelta(0)).total_seconds() // 60
        )
        suppress_path = delay_minutes + sum(
            budget for name, budget in budgets if name != OPENING_AFTER_SEND
        ) - 1
        return max(send_path, suppress_path)

    def _check_in_world_envelope(self) -> None:
        """Fail before the first turn when the worst case cannot fit.

        `_turn_timestamp` raises the moment an exchange lands on or past the
        deadline. Widening the per-beat budget makes that reachable, so the same
        arithmetic runs up front instead: the session is refused here rather
        than exploding halfway through, and nothing is widened silently.
        """
        if self.benchmark_session_deadline is None:
            return
        envelope_minutes = (
            self.benchmark_session_deadline - self.benchmark_session_timestamp
        ).total_seconds() / 60
        worst_case_offset = self._worst_case_turn_offset()
        if worst_case_offset >= envelope_minutes:
            raise RuntimeError(
                "benchmark session cannot fit its in-world envelope: "
                f"session_id={self.script.session_id!r} "
                f"worst_case_turns={worst_case_offset + 1} "
                f"envelope_minutes={envelope_minutes:g} "
                f"start={self.benchmark_session_timestamp.isoformat()!r} "
                f"deadline={self.benchmark_session_deadline.isoformat()!r}"
            )

    def _absorb_triggered_skills(self, pa_metadata: dict[str, Any]) -> None:
        """Load the reception rules for attributes the PA's own turn triggered.

        The harness reports `triggered_skills` when a PA turn puts an attribute
        in play that no beat cued (a tool result carrying `capability_limit`,
        say). Those attributes join the session's reception set from this turn
        on, exactly as if a beat had asked under them, but they are NOT session
        active IX: they cue no IX tool, widen no turn budget, and add no asking
        line. Absent on older transcripts and on a harness that does not write
        it yet, which reads as nothing triggered.

        Loaded for the rest of this beat and for the next one, the same one
        reaction a beat's own attribute gets. An attribute already in the
        record joins nothing: it has had its reaction.
        """
        for attribute in pa_metadata.get("triggered_skills") or []:
            if attribute in self.script.session_reception_skills:
                continue
            self.script.session_reception_skills.append(attribute)
            if self.script.session_reception_carry is not None:
                self.script.session_reception_carry.append(attribute)
            self._reception_joined.append(attribute)
            self._emit(
                "skill_triggered",
                attribute=attribute,
                source="triggered_skills",
            )

    def _emit(self, kind: str, **payload: Any) -> dict:
        event = {"t": self._t, "kind": kind, **payload}
        self._t += 1
        self._events.append(event)
        if self.transcript_writer is not None:
            self.transcript_writer.log_event(event)
        return event

    @staticmethod
    def _serialize_output(
        output: SimulatorOutput, preauthorised_tool: str | None = None
    ) -> dict:
        return {
            "message": output.message,
            "factual_check": [
                {"fact": fv.fact, "expected": fv.expected, "pa_said": fv.pa_said}
                for fv in output.factual_check
            ],
            "turn_assessment": {
                "next_beat": output.turn_assessment.next_beat,
                "goal_already_met": output.turn_assessment.goal_already_met,
            },
            "consent_decisions": _consent_decisions_payload(
                output, preauthorised_tool
            ),
            "parse_warnings": list(output.parse_warnings),
        }

    def _last_pa_message(self) -> str:
        for message in reversed(self._history):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    def _turn_carries_an_ask(self, beat: Beat, message: str) -> bool:
        """Does this candidate put something to the assistant?

        The armed flag is never asked of the judge: this method is only reached
        on a turn the harness has already decided owes an ask, and `must_ask`
        goes in as a fact. The judge is the only decider:
        with it off the ask gate is not checked and no retry fires.
        """
        if not self.judge.enabled:
            return True
        ctx = TurnContext(
            turn_uid=f"{beat.beat}:t{self._benchmark_turn_index}",
            pa_previous_turn=self._last_pa_message(),
            candidate_turn=message,
            must_ask=True,
        )
        judged = self.judge.judge("carries_an_ask", ctx)
        # The words the judge read the ask out of, for the row the rejected
        # attempt writes below.
        self._ask_judge_evidence = self.judge.last_evidence("carries_an_ask")
        return judged

    def _ask_rejection(
        self,
        beat: Beat,
        code: str,
        attempt: int,
        output: SimulatorOutput,
        note: str,
    ) -> dict[str, Any]:
        """A turn the ask gate threw away, shaped like a contract rejection.

        The `beat_ask_retry` and `beat_ask_unfulfilled` events carry only the
        first 200 characters, so live `acc_003` `b4` invalidated a
        session on three turns nobody could read afterwards — and the turns
        were correct. `raw_response` is the whole model response, the way the
        contract path writes it; `retry_note` is what the actor was sent back
        with, which is the only thing distinguishing one of these attempts from
        the next.
        """
        return {
            "phase": "simulator_turn",
            "attempt": attempt,
            "error_codes": [code],
            "model": self.model or self.backend.get_default_model(),
            "session_id": self.script.session_id,
            "beat": beat.beat,
            "raw_response": output.raw_response or output.message,
            "retry_note": note,
            # Why the gate read it that way, when the judge is the one reading.
            # Empty off, and empty when the judge found no ask to quote.
            **(
                {"judge_evidence": self._ask_judge_evidence}
                if self._ask_judge_evidence
                else {}
            ),
        }

    def _record_rejected_attempts(self, attempts: list[dict[str, Any]]) -> None:
        for attempt in attempts:
            if self.transcript_writer is not None:
                self.transcript_writer.log_validation_attempt(attempt)
            self._emit(
                "simulator_validation_failure",
                phase=attempt.get("phase", "simulator_turn"),
                beat=attempt.get("beat"),
                attempt=attempt.get("attempt"),
                error_codes=list(attempt.get("error_codes") or []),
                model=attempt.get("model"),
                validation_attempts_artifact="simulator_validation_attempts.jsonl",
            )

    def _record_accepted_diagnostics(
        self, beat: Beat, output: SimulatorOutput
    ) -> None:
        """Non-blocking findings on the turn that was played, not thrown away.

        Rejected attempts carry theirs into
        `simulator_validation_attempts.jsonl` already; an accepted turn had
        nowhere to put them, so a diagnostic on the message the PA actually
        answered was computed and dropped. Same artifact, same
        `diagnostics` key, same list-of-names value, so a reader can union the
        two. `accepted` is what tells them apart, and no `error_codes` key is
        written: nothing failed validation here.
        """
        if not output.diagnostics:
            return
        record = {
            "phase": "simulator_turn",
            "beat": beat.beat,
            "attempt": len(output.rejected_attempts) + 1,
            "accepted": True,
            "diagnostics": list(output.diagnostics),
        }
        detail = dict(output.diagnostics_detail)
        if detail:
            record["diagnostics_detail"] = detail
        if self.transcript_writer is not None:
            self.transcript_writer.log_validation_attempt(record)
        self._emit(
            "simulator_turn_diagnostics",
            beat=beat.beat,
            diagnostics=list(output.diagnostics),
            # What each of them fired on, when the finding carries it.
            diagnostics_detail=detail,
            validation_attempts_artifact="simulator_validation_attempts.jsonl",
        )

    # -- public API --------------------------------------------------------

    async def run_session(self) -> list[dict]:
        """Iterate beats linearly; return the full transcript event list.

        Side effects: emits events to `transcript_writer` (if set) and accumulates
        the OpenAI-style chat history internally.

        A session the simulator invalidated ends on `session_invalid` instead of
        `session_end` and raises: the session-end reflection and the simulator's
        own memory write never run, because there is no valid session for them
        to close. The exception is what the harness turns into the step's
        status and a non-zero rc.
        """
        try:
            return await self._run_beats()
        except SessionInvalidError as exc:
            # A distinct end kind on purpose: a read that counts finished
            # sessions off `session_end` must not count this one, and the rule
            # that ended it is the thing the read is counting.
            self._emit(
                "session_invalid",
                rule=exc.rule,
                beat=exc.beat,
                attempts=exc.attempts,
                simulator_turns=sum(
                    1 for e in self._events if e["kind"] == "simulator_turn"
                ),
                beat_count=sum(
                    1 for e in self._events if e["kind"] == "beat_enter"
                ),
            )
            if isinstance(exc.__cause__, JudgeUnavailable):
                # The session's record is written like any other void session,
                # and then the class the ledger sees changes. Every step
                # exception already stops the run — `harness/orchestrator.py`
                # re-raises after `mark_failed` — so this is not about whether
                # the round continues; it is about what the ledger says when it
                # stops. Classified as `SessionInvalidError` the entry reads
                # "discard the session and rerun it", which is wrong twice over:
                # the session is not the problem, and rerunning it without the
                # model back does the same thing again. `from None` because the
                # `SessionInvalidError` is not this error's cause — it is its
                # consequence — and because `classify_failure` reads
                # `__context__` when it finds no direct match.
                raise exc.__cause__ from None
            raise

    async def _run_beats(self) -> list[dict]:
        simulator_model = self.model or self.backend.get_default_model()
        self._emit(
            "session_start",
            persona=self.identity.name,
            session_id=self.script.session_id,
            scenario=self.script.scenario,
            context=self.script.context,
            beat_names=[b.beat for b in self.script.beats],
            model=self.pa_model or simulator_model,
            pa_model=self.pa_model,
            simulator_model=simulator_model,
            simulator_backend=self.backend.name,
            session_key=self.session_key,
        )

        beats = list(self.script.beats)
        if self.script.proactive_trigger is not None:
            outreach_delivered = await self._run_proactive_trigger()
            beats = self._beats_after_outreach(outreach_delivered)

        beats_entered = 0
        index = 0
        previous_active_skills: list[str] = []
        reveal_beats = set((self.script.reveal_at or {}).values())
        readback_beats = set(self.script.readback_beats or ())
        readback_tools = dict(self.script.readback_beat_tools or {})
        # A compressed session is exactly two beats, one user turn and one PA
        # reply each: the probe, then the reaction. Both boundaries are
        # unconditional — the probe leaves after its first delivered PA reply
        # whatever it routed to, and the reaction
        # ends the session. `goal_already_met`, an open hold, a pending
        # interaction and whatever either turn routed to are all recorded and
        # none of them extends the session.
        # A triggered compressed session has no probe beat: the probe was the
        # PA's own trigger turn above, where sending is High and staying silent
        # is Low. Its two authored beats are the two openings, and
        # `_beats_after_outreach` already dropped the one that did not match
        # the decision — so the opening that played IS the reaction, and it is
        # the first beat entered, not the second.
        # Same on either path: `beats` is one beat long whichever way it went.
        compressed = self.script.compressed
        compressed_reaction_index = 1 if self.script.proactive_trigger else 2
        exchanges_in_probe = 0
        probe_ignored_routing: str | None = None
        compressed_stop = False
        while index < len(beats):
            beat = beats[index]
            index += 1
            # The reception-only block runs on the path actually walked: a beat
            # the loop jumped over never asked anything, so its skills never
            # carry. It runs on one beat, this one, and what it loads is what
            # joined during the beat before — the actor reacts to the surface
            # once and then works under it (live `acc_035`).
            self.script.session_reception_carry = list(self._reception_joined)
            self._reception_joined = []
            for attr in beat.active_skills or []:
                if attr not in self._session_active_ix:
                    self._session_active_ix.append(attr)
                if attr not in self.script.session_reception_skills:
                    self.script.session_reception_skills.append(attr)
                self._reception_joined.append(attr)
            if beat.preference_updates:
                self.matrix = apply_preference_patches(self.matrix, beat.preference_updates)
                self._emit(
                    "preference_update",
                    beat=beat.beat,
                    patches=self._serialize_preference_patches(beat.preference_updates),
                )
            cued_ix_tools = ix_tool_names_for(beat.active_skills)
            unselected_ix = [
                attr for attr in self._session_active_ix
                if attr not in self._session_selected_ix
            ]
            session_ix_tools = ix_tool_names_for(self._session_active_ix)
            active_ix_tools = ix_tool_names_for(unselected_ix)
            beat_enter = {
                "beat": beat.beat,
                "active_skills": list(beat.active_skills or []),
                "session_active_skills": list(self._session_active_ix),
                "reception_skills": reception_carry_attributes(self.script, beat),
                "cued_ix_tools": cued_ix_tools,
                "session_ix_tools": session_ix_tools,
                "active_ix_tools": active_ix_tools,
                "elicitation": beat.elicitation,
            }
            beats_entered += 1
            # The script is a linear plan, the PA's scope is the measured
            # variable: when the PA has already done this step's job, the user
            # does not ask for it again, and does not spend a turn saying so
            # either. A collapsed beat records `beat_enter` and
            # `beat_collapsed` and nothing else, so a run of them ends the
            # session on the last real PA turn instead of paying a user turn
            # and a PA turn each: live `acc_019` and `acc_006` closed on
            # near-verbatim duplicate turns, `acc_009` burned four beats that
            # way and the persona scored the session down for it.
            # A beat carrying skills is never collapsed: its own skills are the
            # measurement. Neither is the beat right after one — that is the
            # reaction beat, where the surface is read, and `acc_027` lost its
            # reaction beat because this predicate read `active_skills` alone.
            # Only that one beat, not every beat holding carried reception
            # rules: the carry runs to the end of the session, so reading it
            # here made every post-probe beat permanently non-collapsible and
            # `beat_collapsed` went 3 -> 0 (`acc_007`: three beats
            # entered with the goal already met and played in full). Later
            # carried beats collapse when met, like any other beat. The opening
            # beat is never collapsed either: nothing has happened yet. Nor is
            # a beat that releases a gated fact or asks the user to authorize
            # or verify something — see `_held_from_collapse`.
            # The reaction beat of a compressed session: its one exchange is
            # the whole measurement, so it is neither collapsed away nor run on
            # past its first turn.
            is_compressed_reaction = (
                compressed and beats_entered == compressed_reaction_index
            )
            # The probe beat of an untriggered compressed session: the user
            # states the whole query, the PA answers once, and the loop moves
            # to the reaction. `stay` from the probe's own assessment is
            # recorded and dropped, exactly as the reaction beat's routing is —
            # live `acc_004` and `acc_002` stayed in beat 1 for 2-3
            # exchanges on it, and the extra turns carried the task on inside
            # the probe (a second search, a dictated email, "proceed with the
            # reservation" -> `tool_outside_plan`). A triggered session has no
            # probe beat, so `is_compressed_reaction` claims its first beat.
            is_compressed_probe = (
                compressed and beats_entered == 1 and not is_compressed_reaction
            )
            collapsible = (
                not beat.active_skills
                and not previous_active_skills
                and beats_entered > 1
                and not is_compressed_reaction
                and not _held_from_collapse(beat, reveal_beats, readback_beats)
            )
            # A beat that inspects a declared write has to do the inspecting:
            # `_held_from_collapse` keeps it from being collapsed, but nothing
            # kept its one turn from being spent on a confirmation of a write
            # the user never saw (live `acc_001` beat_06 t=28).
            is_readback = beat.beat in readback_beats
            # ...unless the PA already read it back. The beat stays held from
            # the collapse (`_held_from_collapse` keeps it, so this never runs
            # on a beat that was going to be skipped anyway); only the forced
            # ask goes, and acknowledge-and-close satisfies the beat. Computed
            # here because `_pa_executed_tools` at this point is exactly the
            # calls made before this beat's first turn.
            readback_delivered = None
            if is_readback:
                write_tool, read_tool = readback_tools.get(
                    beat.beat, (None, None)
                )
                readback_delivered = _readback_already_delivered(
                    self._pa_executed_tools,
                    write_tool=write_tool,
                    read_tool=read_tool,
                )
            if readback_delivered is not None:
                self._emit(
                    "readback_already_delivered",
                    beat=beat.beat,
                    read_tool=readback_delivered[0],
                    write_tool=readback_delivered[1],
                )
            # The same derivation `simulator.agent.loop._is_reaction_beat`
            # runs, off the state this loop already keeps: the beat after a
            # declaring beat, or a compressed session's reaction beat.
            # `previous_active_skills` still holds the beat before this one —
            # it is reassigned at the bottom of the beat.
            is_reaction = is_compressed_reaction or (
                bool(previous_active_skills) and not beat.active_skills
            )
            # The one place that knows which beat is the reaction. The prompt
            # and the agent-side checks read it off the script from here.
            self.script.compressed_reaction_beat = (
                beat.beat if is_compressed_reaction else None
            )
            turn_budget = self._beat_turn_budget(beat)
            turns_run = 0
            turns_charged = 0
            uncharged_empty = 0
            budget_spent = True
            while turns_charged < turn_budget:
                # Held items and the pending flag are as of the PA's last turn,
                # so this reads the same state on either side of the call; the
                # collapse decision is taken before the user speaks and needs
                # it beforehand.
                can_leave = (
                    not self._pending_held_items and not self._pending_interaction
                )
                assessment = await self._run_one_beat(
                    beat,
                    first_turn_in_beat=turns_run == 0,
                    beat_enter=beat_enter if turns_run == 0 else None,
                    collapse_if_met=(
                        turns_run == 0 and collapsible and can_leave
                    ),
                    # On a beat that declares skills, `goal_already_met` is
                    # recorded and then acted on by nothing: `collapsible` is
                    # already false there, and this makes the other half true
                    # too — the beat's first turn has to carry its ask, met or
                    # not.
                    # Except on the reaction beat. A triggered session's played
                    # opening declares the attribute by contract (both openings
                    # do) and is a pure reaction to the send/suppress choice, so
                    # `bool(beat.active_skills)` was forcing an ask onto the one
                    # beat in the corpus that must not ask: live
                    # `acc_007` t4 threw away a clean reaction and the retry came
                    # back with "What are the specific hours for this water
                    # shutoff?", which the beat's own `avoid` forbids. Only the
                    # declaring half is dropped — a beat marked for readback
                    # still asks to be shown the artefact. An untriggered
                    # compressed session is unaffected either way: its probe is
                    # beat 1 and still asks, and its reaction beat declares
                    # nothing.
                    require_ask=(
                        (
                            (bool(beat.active_skills) and not is_compressed_reaction)
                            or (is_readback and readback_delivered is None)
                        )
                        and turns_run == 0
                    ),
                    # A readback beat asks to be shown the artefact; a
                    # declaring beat releases its own ask. Different miss,
                    # different correction. A readback beat that is also the
                    # reaction beat owes both, and its note has to say so or
                    # the retry trades the reaction away for the ask.
                    ask_retry_note=_ask_retry_note_for(
                        is_readback=is_readback and not beat.active_skills,
                        is_reaction=is_reaction,
                        # The declaring note diagnoses the miss as an
                        # acknowledgement of the PA. On the session's opening
                        # turn there is no PA turn to have acknowledged.
                        after_a_pa_turn=any(
                            entry.get("role") == "user"
                            for entry in self._history
                        ),
                    ),
                )
                turns_run += 1
                if is_compressed_reaction:
                    # One simulator turn and one PA reply, and the session is
                    # over. Any routing the reaction turn asked for is recorded
                    # and dropped, and an open hold is reported without being
                    # answered or cleared — the orchestrator owns that failure.
                    self._emit(
                        "compressed_session_end",
                        beat=beat.beat,
                        exchanges_in_probe=exchanges_in_probe,
                        open_hold=bool(
                            self._pending_held_items or self._pending_interaction
                        ),
                        ignored_routing=(
                            assessment.next_beat if assessment is not None else None
                        ),
                        probe_ignored_routing=probe_ignored_routing,
                    )
                    compressed_stop = True
                    budget_spent = False
                    break
                if assessment is None:
                    self._emit(
                        "beat_collapsed",
                        beat=beat.beat,
                        reason="goal_already_met",
                    )
                    budget_spent = False
                    break
                # A PA turn the wrapper closed with its empty-reply line said
                # nothing, so it is not one of the beat's exchanges: the user
                # simply asks again. Capped, so an endlessly empty PA still
                # runs the beat out instead of looping.
                if (
                    self._last_pa_empty_reply
                    and uncharged_empty < MAX_UNCHARGED_EMPTY_REPLIES_PER_BEAT
                ):
                    uncharged_empty += 1
                else:
                    turns_charged += 1
                    if is_compressed_probe:
                        # The probe's one measured reply has landed, so the
                        # loop moves to the reaction whatever the assessment
                        # asked for. What it asked for is carried to
                        # `compressed_session_end` as `probe_ignored_routing`.
                        # This sits inside the charged branch on purpose: an
                        # empty PA turn delivered nothing to measure and the
                        # probe is the one measurement, so the user asks again
                        # under the same cap as any other beat. The reaction
                        # beat keeps its unconditional stop above.
                        probe_ignored_routing = assessment.next_beat
                        budget_spent = False
                        break
                # A beat is not over while the PA is waiting on the user. The
                # last PA turn asked or held something, so "advance" would walk
                # out of a question — and on the final beat it ends the session
                # with the hold episode still open, which fails the step and
                # publishes no checkpoint (live: gate on `beat_07`).
                # `_pending_held_items` alone cannot see this: a question-only
                # or batch-only hold is consent-able by nothing and leaves it
                # empty, which is why the harness also reports the flag.
                # The user answers instead; the budget still bounds the stay,
                # and the wrapper's own round cap closes the episode with the
                # PA's closing line, which the simulator then gets to react to.
                #
                # Except when the script does not own the hold yet. Before the
                # authorising beat a go-ahead is not this beat's to give, so a
                # hold the PA opened there is one this beat has no legal move
                # against: the loop demands
                # a release the validator forbids. Live `acc_039`:
                # beat_3 routed `advance` on its first turn, the PA's
                # `external_action_floor` hold on `apt_book_001` pinned it, and
                # the actor wrote a release 12 times — each rejected — then
                # invented an offstage deferral reason for 7 exchanges, then 3
                # more on beat_4, eating the reaction beat. The hold is carried,
                # not cleared: `_pending_held_items` is session state, so the
                # next beat's first turn still gets it as its stage direction
                # and can answer it or keep withholding. On the authorising
                # beat and after it the pin stays: that is where the user is
                # expected to answer. It stays on the last beat too, whatever
                # the predicate says — there is no next beat to carry a hold
                # into, and walking out there ends the session with the episode
                # open, which is the failure the pin was written for.
                can_leave = (
                    not self._pending_held_items and not self._pending_interaction
                )
                if assessment.next_beat == "advance":
                    if can_leave:
                        budget_spent = False
                        break
                    held_ids = [
                        item["interaction_item_id"]
                        for item in self._pending_held_items
                    ]
                    if index < len(beats) and before_authorising_beat(
                        self.script, beat
                    ):
                        self._emit(
                            "beat_advanced_with_hold",
                            beat=beat.beat,
                            next_beat=beats[index].beat,
                            held_items=held_ids,
                            pending_interaction=self._pending_interaction,
                        )
                        budget_spent = False
                        break
                    self._emit(
                        "beat_kept_for_hold",
                        beat=beat.beat,
                        held_items=held_ids,
                        pending_interaction=self._pending_interaction,
                        turn=turns_run,
                    )
            if compressed_stop:
                break
            if beats_entered == 1:
                # What beat 1 cost, as `compressed_session_end` reports it. On
                # an untriggered compressed session the break above makes it 1.
                # On a triggered one the loop has already broken out of the
                # beat loop entirely, so the field stays 0 — honest, because
                # that probe was one PA turn with no simulator exchange in it
                # at all. Uncompressed sessions never read it.
                exchanges_in_probe = turns_run
            if budget_spent:
                # `turns` is the budget this beat actually ran on, not a
                # constant: the judge reads it as the denominator.
                self._emit("beat_turn_limit", beat=beat.beat, turns=turn_budget)
                index = self._jump_to_preference_beat(beats, index)
            previous_active_skills = list(beat.active_skills or [])

        if self.self_reporter is not None and self._history:
            await self._run_session_end_reflection()

        sim_turns = sum(1 for e in self._events if e["kind"] == "simulator_turn")
        self._emit("session_end", simulator_turns=sim_turns, beat_count=beats_entered)
        return list(self._events)

    def _turn_timestamp(self) -> datetime | None:
        """In-world time of the next exchange; fails loudly past the deadline."""
        if self.benchmark_session_timestamp is None:
            return None
        timestamp = self.benchmark_session_timestamp + timedelta(
            minutes=self._benchmark_turn_index
        )
        if (
            self.benchmark_session_deadline is not None
            and timestamp >= self.benchmark_session_deadline
        ):
            raise RuntimeError(
                "benchmark session ran past its envelope: "
                f"session_id={self.script.session_id!r} "
                f"turn_timestamp={timestamp.isoformat()!r} "
                f"deadline={self.benchmark_session_deadline.isoformat()!r}"
            )
        return timestamp

    def _jump_to_preference_beat(self, beats: list[Beat], index: int) -> int:
        """After a stalled beat, skip ahead to the next preference beat.

        A beat that ran out of turns did not get what it was waiting for, so
        the beats between it and the next one carrying `active_skills` would
        stall the same way. Skip them, record what was given up on, and let the
        landing beat's first trigger say so out loud. With no later preference
        beat the story simply continues with the next beat.
        """
        landing = next(
            (i for i in range(index, len(beats)) if beats[i].active_skills),
            None,
        )
        if landing is None:
            return index
        # Stop one beat short. The beat immediately before a preference beat is
        # usually what that beat's `entry_condition` is written against, so
        # skipping it lets the PA reach the probe with the work it is probed on
        # never done: live `acc_001` lost beat_03, which read the paper, and
        # beat_04 then probed on a source the PA had never opened. Same
        # carve-out the collapse path makes for the beat after a preference
        # beat.
        landing = max(landing - 1, index)
        # A `reveal_at` beat is the only place its fact is ever said out loud,
        # so jumping over one deletes the fact from the session exactly as a
        # collapse does. The collapse path holds those beats out, and
        # live `acc_039` lost beat_04 to this jump instead
        # (`transcript.jsonl` t=15): the November 20 slot was never spoken, and
        # beat_05 went on to "verify" a booking that did not exist. Land on the
        # first reveal beat in the way; it plays, and if it stalls in turn the
        # jump runs again from there.
        reveal_beats = set((self.script.reveal_at or {}).values())
        first_reveal = next(
            (i for i in range(index, landing) if beats[i].beat in reveal_beats),
            None,
        )
        if first_reveal is not None:
            landing = first_reveal
        if landing == index:
            return index
        skipped = beats[index:landing]
        for beat in skipped:
            self._emit(
                "beat_skipped",
                beat=beat.beat,
                goal=beat.goal,
                reason="jump_to_preference_beat_after_turn_limit",
            )
            self.script.skipped_beats.append(beat.beat)
        goals = ", ".join(beat.goal for beat in skipped if beat.goal)
        self.script.pending_jump_note = (
            f"You gave up on earlier steps: {goals}. None of that happened: the "
            "assistant has not found, read, changed, or produced anything from "
            "those steps. In this first message, say in one sentence what those "
            "steps were meant to achieve, then make this step's request yourself, "
            "from scratch, following the 'If that has not happened' line of this "
            "step. Do not evaluate, confirm, or approve work the assistant has "
            "not done."
        )
        return landing

    def _beats_after_outreach(self, delivered: bool) -> list[Beat]:
        """Pick the opening that matches the PA's decision and drop the other.

        After a suppress the user comes back on their own: the clock jumps to
        trigger time + suppress_resume_delay before the first user message.
        """
        taken, skipped = (
            (OPENING_AFTER_SEND, OPENING_AFTER_SUPPRESS)
            if delivered
            else (OPENING_AFTER_SUPPRESS, OPENING_AFTER_SEND)
        )
        resume_at = None
        if not delivered and self.benchmark_session_timestamp is not None:
            trigger_index = self._benchmark_turn_index - 1
            delay_minutes = int(self.suppress_resume_delay.total_seconds() // 60)
            self._benchmark_turn_index = trigger_index + delay_minutes
            resume_at = self._turn_timestamp()
        self._emit(
            "outreach_path",
            path="send" if delivered else "suppress",
            opening_beat=taken,
            skipped_beat=skipped,
            resume_at=resume_at.isoformat() if resume_at else None,
        )
        return [beat for beat in self.script.beats if beat.beat != skipped]

    async def _run_proactive_trigger(self) -> bool:
        trigger = self.script.proactive_trigger
        if trigger is None:
            return True
        if "proactive_outreach" not in self._session_active_ix:
            self._session_active_ix.append("proactive_outreach")
        timestamp = self._turn_timestamp()
        self._emit(
            "environment_trigger",
            trigger_id=trigger.trigger_id,
            source_type=trigger.activation_type,
            prior_thread=trigger.background,
            detail=trigger.detail,
            user_impact=trigger.user_impact,
            active_request=False,
            benchmark_turn_timestamp=timestamp.isoformat() if timestamp else None,
        )
        metadata = {
            "run_id": self.run_id,
            "persona_id": self.persona_id,
            "agent_id": self.agent_id,
            "script_session_id": self.script.session_id,
            "turn_id": f"{self.script.session_id}:proactive_trigger",
            "scenario_description": self.script.scenario,
            "interaction_context": self.script.context,
            "active_preference_attributes": ["proactive_outreach"],
            "memory_retrieval_budget_tokens": self.memory_retrieval_max_tokens,
            "memory_retrieval_budget_items": self.memory_retrieval_max_items,
        }
        outbound = await self.pa_loop.process_environment_event(
            trigger_id=trigger.trigger_id,
            source_type=trigger.activation_type,
            prior_thread=trigger.background,
            detail=trigger.detail,
            user_impact=trigger.user_impact,
            session_key=self.session_key,
            channel="benchmark",
            chat_id=self.session_key,
            timestamp=timestamp,
            metadata=metadata,
        )
        self._benchmark_turn_index += 1
        outbound_metadata = getattr(outbound, "metadata", {}) or {}
        decision = outbound_metadata.get("outreach_decision")
        delivered = outbound_metadata.get("outreach_delivered")
        if decision not in {"send", "suppress"} or delivered is not (decision == "send"):
            raise RuntimeError(
                "PA proactive event result must record a consistent send/suppress decision"
            )
        pa_text = getattr(outbound, "content", "") or ""
        if delivered and not pa_text.strip():
            raise RuntimeError("PA proactive send decision requires non-empty contact content")
        if not delivered and pa_text:
            raise RuntimeError("PA proactive suppress decision must not include contact content")
        tool_events = outbound_metadata.get("tool_events") or []
        for event in tool_events:
            if event.get("ix") and event.get("attribute"):
                self._session_selected_ix.add(str(event["attribute"]))
        pa_event = self._emit(
            "pa_turn",
            beat="proactive_trigger",
            proactive_event=True,
            benchmark_turn_timestamp=timestamp.isoformat() if timestamp else None,
            outbound=pa_text,
            outreach_decision=decision,
            outreach_delivered=delivered,
            tool_events=tool_events,
            ix_required=["IX_proactive_outreach"],
            ix_called=[
                event.get("name") for event in tool_events if event.get("ix")
            ],
            ix_missing=[],
            empty=not pa_text,
        )
        self._absorb_triggered_skills(outbound_metadata)
        self._pending_held_items = self._normalize_held_items(
            outbound_metadata.get("held_items")
        )
        self._pending_interaction = bool(
            outbound_metadata.get("has_pending_interaction")
        )
        # The trigger turn's record: the delivered outreach is its one act
        # outside the tool record. Nothing ran before it this session.
        self._pending_reply_facts = reply_facts_from_pa_turn(
            tool_events,
            outbound_metadata.get("reply_facts"),
            earlier_tools=self._pa_executed_tools,
            earlier_outreach_sent=False,
            outreach_delivered=bool(delivered),
        )
        if delivered:
            self._pa_outreach_sent = True
            self._history.append(
                {"role": "user", "content": pa_text, "t": pa_event["t"]}
            )
        return bool(delivered)

    async def _run_session_end_reflection(self) -> None:
        # The self-report is an auxiliary eval signal. When the model cannot
        # produce it after the configured retries (malformed blocks, or an
        # empty final message), record the miss and let the session close.
        try:
            result = await self.self_reporter.generate(
                session=self.script,
                history=list(self._history),
            )
        except SelfReportContractError as exc:
            self._record_rejected_attempts(exc.rejected_attempts)
            self._emit(
                "session_end_reflection_missing",
                reason="contract",
                findings=list(exc.findings),
                attempts=len(exc.rejected_attempts),
            )
            return
        except RuntimeError as exc:
            if "no final content" not in str(exc):
                raise
            self._emit(
                "session_end_reflection_missing",
                reason="empty_response",
                detail=str(exc),
            )
            return
        self._record_rejected_attempts(result.rejected_attempts)
        self._emit(
            "session_end_reflection",
            reflection=result.reflection,
            final_emotion={
                "trigger": result.final_emotion.trigger,
                "reaction": result.final_emotion.reaction,
            },
            self_report=result.self_report.model_dump(mode="json"),
            evidence=result.evidence,
            # What an evidence row's `turn` counts. It is the position in the
            # rendered history, which is what the reflection prompt numbers;
            # the row's `t` is the transcript event the same message was
            # emitted at, for a reader holding `transcript.jsonl`.
            turn_index="history",
            findings=result.findings,
        )

    async def _run_one_beat(
        self,
        beat: Beat,
        *,
        first_turn_in_beat: bool = True,
        beat_enter: dict[str, Any] | None = None,
        collapse_if_met: bool = False,
        require_ask: bool = False,
        ask_retry_note: str = _ASK_RETRY_NOTE,
    ) -> TurnAssessment | None:
        # In-world time of this exchange. It stamps the PA call below and the
        # simulator's own session-close memory write; never wall-clock time.
        benchmark_turn_timestamp = self._turn_timestamp()
        # What the PA was holding as of its last turn. It is this turn's stage
        # direction, and the only ids the consent sideband may answer about.
        held_items = list(self._pending_held_items)

        # 1. simulator LLM produces the three-block turn.
        async def _generate() -> SimulatorOutput:
            try:
                return await self.backend.generate_turn(
                    self.identity,
                    self.matrix,
                    self.script,
                    beat,
                    history=self._history if self._history else None,
                    actor_skill_specs_root=self.actor_skill_specs_root,
                    model=self.model,
                    first_turn_in_beat=first_turn_in_beat,
                    benchmark_timestamp=(
                        benchmark_turn_timestamp.strftime("%Y-%m-%d %H:%M")
                        if benchmark_turn_timestamp is not None
                        else None
                    ),
                    held_items=held_items,
                    # What the PA's last reply did, for the actor to read.
                    reply_facts=self._pending_reply_facts,
                    # The retry loop checks the reaction against what the PA
                    # actually ran, and only this object keeps that list.
                    pa_executed_tools=list(self._pa_executed_tools),
                    # ...and the same calls with their arguments, which is
                    # what the judge is shown as the tool log.
                    pa_tool_log=list(self._pa_tool_log),
                    # ...and whether the triggered outreach was delivered: a
                    # send the tool record cannot show, but a real PA action.
                    pa_outreach_sent=self._pa_outreach_sent,
                    # One judge for the session, so the per-generation budget
                    # the backend resets is the same counter the ask gate
                    # spends against when this call returns.
                    judge=self.judge,
                    # Whether this turn owes an ask is the
                    # harness's answer, and only this method
                    # has it. The judge is told, never asked.
                    must_ask=require_ask,
                )
            except SimulatorOutputContractError as exc:
                self._record_rejected_attempts(exc.rejected_attempts)
                if beat_enter is not None:
                    self._emit("beat_enter", **beat_enter)
                invalid_rule = next(
                    (rule for rule in exc.findings if rule in SESSION_INVALID_RULES),
                    None,
                )
                if invalid_rule is not None:
                    # An instrument failure the retry budget could not fix. The
                    # three-block output that never parsed used to be played on
                    # its recovered remainder with a defaulted `next_beat:
                    # stay`, so the beat was routed off a value the actor never
                    # produced; a dictation that drops the gate's key terms
                    # would reach the PA as a sentence the store then refuses.
                    # Either way the session measured nothing.
                    attempts = len(exc.rejected_attempts) or 1
                    last = exc.rejected_attempts[-1] if exc.rejected_attempts else {}
                    self._emit(
                        invalid_rule,
                        beat=beat.beat,
                        attempts=attempts,
                        aborted=True,
                        rejected_message=str(last.get("raw_response") or "")[:200],
                    )
                    raise SessionInvalidError(
                        invalid_rule,
                        beat=beat.beat,
                        attempts=attempts,
                    ) from exc
                raise
            finally:
                # The give-up note belongs to this trigger only.
                self.script.pending_jump_note = ""

        output = await _generate()
        # Commented out: the forced-ask check is
        # off — no judge call, nothing recorded (live: 20/20 diagnostics
        # false, ~13 s per call). Uncomment to restore.
        # ask_retries = 0
        # note = ""
        # # Once per candidate, not once per test of the condition: the judge is a
        # # model call, and the loop below used to read the same message twice —
        # # here and again at the unfulfilled check.
        # has_ask = require_ask and self._turn_carries_an_ask(beat, output.message)
        # while (
        #     require_ask
        #     and not has_ask
        #     and ask_retries < MAX_ASK_RETRIES_PER_BEAT
        # ):
        #     # The declaring beat's ask IS the measurement: without it the IX
        #     # selection fires against a turn that asked nothing (live
        #     # `acc_001` t19-t22, where the beat entered `goal_already_met: true`
        #     # and spent its slot on an acknowledgement). The prompt has carried
        #     # this rule (`generate_message.md`, and the beat-block
        #     # line at `prompting.render_beat_block`) and the actor ignored it
        #     # three rounds running, so the loop asks again, in the beat's own
        #     # trigger. The retry's own output is checked the same way: live
        #     # `acc_002` fired one `beat_ask_retry` at
        #     # `beat_probe_settle_plan` and then played the second
        #     # acknowledgement unchecked, so the probe was measured against
        #     # "The 18:30 reservation is satisfactory..." and the beat's request
        #     # (the booking contact) was never put to the PA.
        #     # What was thrown away, the way a contract failure carries its
        #     # `raw_response` into `simulator_validation_attempts.jsonl`: an ask
        #     # retry discarded a whole turn and recorded nothing about it, and
        #     # in live `acc_007` t4 the discarded turn was the
        #     # correct one. Truncated — the event is a record, not the artifact.
        #     self._emit(
        #         "beat_ask_retry",
        #         beat=beat.beat,
        #         attempt=ask_retries + 1,
        #         rejected_message=output.message[:200],
        #     )
        #     # `{rejected}` is a placeholder only the reaction-beat readback
        #     # note carries; the other notes have no braces and pass through.
        #     note = ask_retry_note.replace("{rejected}", output.message[:200])
        #     self._record_rejected_attempts(
        #         [
        #             self._ask_rejection(
        #                 beat, "beat_ask_retry", ask_retries + 1, output, note
        #             )
        #         ]
        #     )
        #     self.script.pending_jump_note = note
        #     output = await _generate()
        #     ask_retries += 1
        #     has_ask = self._turn_carries_an_ask(beat, output.message)
        # if require_ask and not has_ask:
        #     # Nothing asked. The turn is played and the failure recorded
        #     # instead of voiding the session — see the block
        #     # below the event.
        #     self._emit(
        #         "beat_ask_unfulfilled",
        #         beat=beat.beat,
        #         attempts=ask_retries + 1,
        #         aborted=False,
        #         rejected_message=output.message[:200],
        #     )
        #     self._record_rejected_attempts(
        #         [
        #             self._ask_rejection(
        #                 beat, "beat_ask_unfulfilled", ask_retries + 1, output, note
        #             )
        #         ]
        #     )
        #     # Record-only: the turn is played
        #     # and the failure rides on the diagnostics instead of voiding the
        #     # session. Restore the `raise SessionInvalidError(...)` below to
        #     # make it blocking again.
        #     if "beat_ask_unfulfilled" not in output.diagnostics:
        #         output.diagnostics.append("beat_ask_unfulfilled")
        #     output.diagnostics_detail["beat_ask_unfulfilled"] = (
        #         f"beat={beat.beat} attempts={ask_retries + 1}"
        #     )
        #     # raise SessionInvalidError(
        #     #     "beat_ask_unfulfilled",
        #     #     beat=beat.beat,
        #     #     attempts=ask_retries + 1,
        #     # )
        if beat_enter is not None:
            # `beat_enter` waits for this first turn: `goal_already_met` is the
            # actor's own reading of what the PA has already done, and it
            # arrives with the same assessment that reads `entry_condition`.
            self._emit(
                "beat_enter",
                **beat_enter,
                goal_already_met=output.turn_assessment.goal_already_met,
            )
        self._record_rejected_attempts(output.rejected_attempts)
        self._record_accepted_diagnostics(beat, output)
        if collapse_if_met and output.turn_assessment.goal_already_met:
            # The assessment is what this call was for; the message it arrived
            # with is dropped unsent. `goal_already_met` is the actor's reading
            # and only the actor can make it, so the turn is the cheapest shape
            # that produces one — but the beat has nothing left to ask for, and
            # a message that only restates the last close costs the PA turn
            # answering it too. The caller records `beat_collapsed` and moves
            # on; nothing goes into the history or to the PA.
            return None
        # Nothing held means nothing to consent to, so every entry is dropped.
        output.consent_decisions, consent_warnings = filter_consent_decisions(
            output.consent_decisions,
            [item["interaction_item_id"] for item in held_items],
        )
        output.parse_warnings.extend(consent_warnings)
        preauthorised_tool = _preauthorised_tool(
            self.script, beat, output.message, held_items
        )
        simulator_event = self._emit(
            "simulator_turn",
            beat=beat.beat,
            **self._serialize_output(output, preauthorised_tool),
        )
        if preauthorised_tool:
            self._emit(
                "consent_preauthorised",
                beat=beat.beat,
                tool=preauthorised_tool,
            )
        self._history.append(
            {
                "role": "assistant",
                "content": output.message,
                "t": simulator_event["t"],
            }
        )

        # 2. forward ONLY <message> to PA. Eval-only blocks stayed in the log above.
        active_ix = [
            attr for attr in self._session_active_ix
            if attr not in self._session_selected_ix
        ]
        active_ix_tools = ix_tool_names_for(active_ix)
        pa_call = {
            "user_message": output.message,
            "session_key": self.session_key,
            "metadata": {
                "run_id": self.run_id,
                "persona_id": self.persona_id,
                "agent_id": self.agent_id,
                "script_session_id": self.script.session_id,
                # A beat may hold the conversation for its whole turn budget, so
                # the turn index keeps memory event ids unique across stays.
                "turn_id": (
                    f"{self.script.session_id}:{beat.beat}"
                    f":t{self._benchmark_turn_index}"
                ),
                "scenario_description": self.script.scenario,
                "interaction_context": self.script.context,
                "active_preference_attributes": list(
                    beat.active_skills or []
                ),
                "memory_retrieval_query": output.message,
                "memory_retrieval_budget_tokens": (
                    self.memory_retrieval_max_tokens
                ),
                "memory_retrieval_budget_items": self.memory_retrieval_max_items,
                "active_ix": active_ix,
                # PA-blind sideband. It rides the same metadata
                # channel as `active_ix`, is read by the wrapper as gate state,
                # and is never rendered into anything the model reads.
                "consent_decisions": _consent_decisions_payload(
                    output, preauthorised_tool
                ),
                "ix_context": {
                    "capability_limit_attempts": (
                        self.capability_limit_attempts if preauthorised_tool else []
                    ),
                    "elicitation": beat.elicitation,
                    "session_active_ix": list(self._session_active_ix),
                    # The wrapper's own records name the beat and the turn, and
                    # the runner has no other way to know either. `turn_index`
                    # reaches `AgentRunSpec.turn_index`: it was never set, so
                    # every record stamped with it -- `asked_at_turn`,
                    # `opened_at_turn`, two event records -- read 0 for
                    # a whole run.
                    "beat": beat.beat,
                    "turn_index": self._benchmark_turn_index,
                },
            },
        }
        if benchmark_turn_timestamp is not None:
            pa_call["timestamp"] = benchmark_turn_timestamp
            pa_call["metadata"]["benchmark_turn_timestamp"] = (
                benchmark_turn_timestamp.isoformat()
            )
        pa_outbound = await self.pa_loop.process_direct_message(
            **pa_call,
        )
        self._benchmark_turn_index += 1
        pa_text = getattr(pa_outbound, "content", "") or "" if pa_outbound is not None else ""
        pa_metadata = (getattr(pa_outbound, "metadata", {}) or {}) if pa_outbound is not None else {}
        tool_events = pa_metadata.get("tool_events", [])
        # The harness translates the wrapper's render result into this narrow
        # key, so the simulator learns WHICH ids are waiting on an answer
        # without ever being handed `ix_render`.
        self._pending_held_items = self._normalize_held_items(
            pa_metadata.get("held_items")
        )
        # Pacing only: it says something is waiting, never what, and
        # it is deliberately absent from `generate_turn`'s arguments.
        self._pending_interaction = bool(pa_metadata.get("has_pending_interaction"))
        # The wrapper's own record that this turn delivered nothing. It rides
        # `tool_events` as an `ix_event` control record, so the simulator reads
        # the name and never `ix_render` itself. Budget accounting only.
        self._last_pa_empty_reply = any(
            event.get("name") == "ix_empty_reply" for event in tool_events
        )
        # Built before this turn's calls join `_pa_executed_tools`, so the
        # record separates what this reply did from what came before it.
        self._pending_reply_facts = reply_facts_from_pa_turn(
            tool_events,
            pa_metadata.get("reply_facts"),
            earlier_tools=self._pa_executed_tools,
            earlier_outreach_sent=self._pa_outreach_sent,
        )
        self._pa_executed_tools.extend(
            str(event.get("name") or "")
            for event in tool_events
            if not event.get("ix_event") and not event.get("error")
        )
        self._pa_tool_log.extend(tool_log_lines(tool_events))
        self._absorb_triggered_skills(pa_metadata)
        for event in tool_events:
            if event.get("ix") and event.get("attribute"):
                self._session_selected_ix.add(str(event["attribute"]))
        ix_called = [event.get("name") for event in tool_events if event.get("ix")]
        ix_missing = [name for name in active_ix_tools if name not in ix_called]
        pa_event = self._emit(
            "pa_turn",
            beat=beat.beat,
            benchmark_turn_timestamp=(
                benchmark_turn_timestamp.isoformat()
                if benchmark_turn_timestamp is not None
                else None
            ),
            outbound=pa_text,
            chat_id=getattr(pa_outbound, "chat_id", None) if pa_outbound is not None else None,
            tool_events=tool_events,
            ix_required=active_ix_tools,
            ix_called=ix_called,
            ix_missing=ix_missing,
            empty=(pa_outbound is None or not pa_text),
        )
        # 3. PA's reply seeds the simulator's next call.
        self._history.append(
            {"role": "user", "content": pa_text, "t": pa_event["t"]}
        )
        return output.turn_assessment
