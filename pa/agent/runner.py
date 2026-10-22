"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from harness.capability_attempt import (
    CONTEXT_KEY, current as attempt_current, matches as attempt_matches,
)
from pa.agent.hook import AgentHook, AgentHookContext
from pa.agent.ix.action_types import tool_action_type as _ix_tool_action_type
from pa.agent.ix.collector import (
    ASK_BEARING_ATTRIBUTES,
    IX_BATCH_REBIND_REPAIR,
    IX_BATCH_SPLIT_REPAIR,
    NO_META_ACKNOWLEDGEMENT,
    NO_WRAPPER_TALK,
    WORLD_CHANGING_ACTION_TYPES,
    RecurrenceRepair,
    TurnCollector,
    batch_decomposition_undivided,
    binding_attribute,
    boundary_payload_invalid,
    catalog_tool_name,
    collect_payloads,
    declared_item_ids,
    gate_opens,
    is_capability_limit,
    live_act_releases,
    plan_item_for,
    preauthorised_acts,
    rebound_batch_ids,
    released_item_ids,
    render_turn,
    sideband_decision,
    turn_reminder_message,
    validate_bindings,
)
from pa.agent.ix.events import IXDirective, IXSelection, extract_ix_payload
from pa.agent.ix.policy import directive_for, selections_from_tool_events
from pa.agent.ix.rationale import (
    reads_as_a_work_log,
    render_certainties,
    strip_certainty_block,
)
from pa.agent.ix.render import (
    IX_EMPTY_REPLY_LINE,
    NON_EPISODE_HOLD_REASONS,
    clean_delivered_text,
    prose_within_completion,
    reply_is_only_markup,
    settings_from_selections,
    split_truncated_tail,
    strip_dsml_markup,
    triggering_ask,
)
from pa.agent.ix.render_types import (
    BATCH_MARKER_RE,
    CAPABILITY_BOUNDARY_ATTRIBUTE,
    CERTAINTY_MARKER_RE,
    HOLD_BEARING_ATTRIBUTES,
    ID_DECLARING_ATTRIBUTES,
    INTERACTION_ITEM_ID_FIELD,
    IX_CAPABILITY_RESOLVE_TOOL,
    IX_CONFIRMATION_MESSAGE_FIELD,
    IX_CONTINUE_TOOL_NAMES,
    IX_RENDER_TOOL_NAMES,
    IX_TOOL_NAME_PREFIX,
    PER_CALL_INJECTED_FIELDS,
    QUESTION_MARKER_RE,
    STOP_REASON_IX_CONTRACT_FAILURE,
    STOP_REASON_IX_GATE,
    STOP_REASON_IX_HOLD,
    UNKNOWN_TOOL_ACTION_TYPE,
    held_status,
    ConsentDecision,
    ConsentRecord,
    GateRecord,
    HeldItem,
    IXHold,
    IXRenderInput,
    RequiredContext,
)
from pa.agent.ix.specs import IX_PASSIVE_SPECS
from pa.agent.tools.registry import ToolRegistry
from pa.providers.base import LLMProvider, LLMTaskIdentity, ToolCallRequest
from pa.utils.helpers import build_assistant_message

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
IX_TOOL_ORDER_REMINDER_PREFIX = (
    "Before calling task tools for this turn, call the required "
    "interaction preference tool(s): "
)
IX_PER_TURN_REMINDER_PREFIX = (
    "This turn cannot be rendered without the per-turn interaction payload. "
    "Call: "
)
IX_UNAVAILABLE_TOOL_REPAIR = (
    "The following interaction tool(s) are not available on this turn: {tools}. "
    "Do not call them again. Use only interaction tools provided in the current "
    "tool list. If no interaction tool is applicable or required, continue "
    "handling the user's request without one. A tool appearing in history "
    "does not mean it is available now. "
)
#: Sent with the forced-tool retry (§3.0): the model is about to see the tool
#: pinned by `tool_choice`, so the message says what to fill in, not "call it".
IX_PER_TURN_FORCED_PREFIX = (
    "The interaction payload for this turn is still missing and is now "
    "required. Fill in and send: "
)
#: hy4 `acc_031` (table row 5) — the demand nothing ever answered. The reminder
#: above and the pin below stay in `messages` for the rest of the turn, and the
#: receipt a landed carrier gets is its own payload echoed back, so a model that
#: has just complied still reads two live instructions to call the same tool.
#: hy4's fourth response called `IX_render_process_visibility` again, this time
#: with the selection tool's fields. Said once per carrier per turn, and only
#: for a carrier this turn actually asked for. It forbids nothing: a later
#: payload for the same attribute still overrides the banked one
#: (`collector.collect_payloads`), which is how an `autonomy_level` plan is
#: marked `plan_done` after the work runs.
IX_CARRIER_RECORDED_NOTE = (
    "[interaction wrapper] {tool} is recorded for this turn, so the request "
    "for it is answered. Continue your reply. " + NO_WRAPPER_TALK
)
#: A mixed response may make valid IX protocol progress before the ordinary
#: work in that response is allowed to run. Keep the protocol call, withhold
#: the work, and say exactly what the next response has to do. The withheld
#: calls are deliberately absent from the assistant/tool protocol below: no
#: task result is invented for work that did not run.
IX_TASK_CALLS_DEFERRED_FOR_CARRIER = (
    "[interaction wrapper] The interaction preference call in your previous "
    "response was recorded, but its ordinary task call(s) were not executed "
    "because this turn still owes interaction payload: {tools}. Send the "
    "remaining interaction payload first, then reissue any still-needed task "
    "calls. " + NO_META_ACKNOWLEDGEMENT
)
#: Appended to the pin when the tool being pinned is a *selection* tool. A
#: `tool_choice` pin gets answered with the schema's `required` list and
#: nothing else -- for a selection tool that is `setting` alone -- so the turn
#: banks a setting with no reasoning behind it. Live: every regress21 session
#: whose selection was pinned (`acc_002`, `007`, `009`, `019`, `026`, `035`)
#: recorded `evidence: ""`, and so did all three of regress20's (`002`, `026`,
#: `035`); no pinned selection in either round carried one. The other half of
#: the same wound already has an answer -- `_restore_voided_selection_fields`
#: refills the blanks from a copy the *same turn* wrote and §3.0 then voided --
#: but on the ordinary ladder path there is no earlier copy to refill from,
#: because the model never sent the call at all. So the pin asks for the
#: fields by name. `evidence` keeps its own rule from
#: `tools.py`/`SETTING_COMMITMENT_NOTE`: empty is the right answer with nothing
#: to cite, and it is never to be invented to fill the field.
IX_FORCED_SELECTION_FIELDS = (
    "Fill `application` with how this setting shapes the reply you are about "
    "to write, and `evidence` with what you are going on -- leave `evidence` "
    "empty if you have nothing to cite, and never invent it. "
)
#: regress12 acc_026 — the model spent the whole `max_tokens` budget on
#: reasoning and returned nothing: `finish_reason == "length"`, no tool calls,
#: no content. The ladder read each of those as a miss, so two truncations
#: bought a reminder and a pin and the turn died with the render payload never
#: asked for. A cut-off reply is the provider running out of room, not the model
#: refusing, so it is answered with this and costs the ladder nothing.
PROVIDER_TRUNCATED_RETRY = (
    "[interaction wrapper] Your previous reply was cut off at the token limit "
    "before anything came through -- no text and no tool calls. Answer again, "
    "briefly, and make the required tool calls first. "
    + NO_META_ACKNOWLEDGEMENT
)
#: Empty truncations may retry twice; exhaustion is a provider failure and
#: cannot spend any IX repair budget.
PROVIDER_TRUNCATION_RETRIES = 2
#: regress12 acc_003 t=12 — the provider answered the closing request with the
#: upstream error body ("The request failed because it is missing
#: `messages.content` parameter" plus a request id) and the runner released that
#: text as a finished PA turn: zero tool calls, and the two document merges the
#: user had just authorised were swallowed. An error response is not a reply, so
#: it buys one more request with the same messages before the turn ends on the
#: error path -- and it is never what the user reads.
PROVIDER_ERROR_RETRIES = 1
#: regress12 acc_004 — `shopping_order_create` was called 46 times with guessed
#: `delivery_address_id` values (addr_home, addr_001, addr_4b, ...) and answered
#: "unknown delivery address" every time: 274 tool calls, 1531 s, and an invented
#: order id at the end. The gate held mechanically and the PA never asked. After
#: this many identical failures of one tool the runner stops running it for the
#: rest of the turn.
TOOL_FAILURE_STRIKE_LIMIT = 3
#: The tool result a struck call gets instead of another attempt. Model-facing:
#: it is a tool result, so it never reaches the user.
TOOL_FAILURE_STRIKE_RESULT = (
    "Error: not run. {tool} has already failed {limit} times on this turn with "
    "the same error, so it will not be called again on this turn. The value it "
    "is missing is not in the world and cannot be guessed."
)
#: The internal message that goes with it, once per struck tool per turn.
TOOL_FAILURE_STRIKE_PROMPT = (
    "[interaction wrapper] {tools} failed {limit} times on this turn with the "
    "same error and will not run again on this turn. Stop trying other values "
    "for it: ask the user for the one it needs. " + NO_META_ACKNOWLEDGEMENT
)
#: gpt `acc_082`, 2026-09-18 — the strike's one exception. Six
#: `calendar_event_create` calls came in with `recurrence: '>TBD'` on an evening
#: the user had authorised in as many words ("Nothing is on the calendar yet, so
#: put it on with the title ..."); the store refused every one of them before
#: writing anything, and on the third the prompt above sent the PA to ask the
#: user for a value the user had never been asked about and did not have. The
#: fault is a parameter the schema rejects, not a missing fact, so this one says
#: which field and which way out -- and names the authorised item, because the
#: turn's other blockage was the model minting a fresh id for the corrected call
#: and walking back into the confirmation gate.
TOOL_FAILURE_STRIKE_RECURRENCE_PROMPT = (
    "[interaction wrapper] {tool} failed {limit} times on this turn, every time "
    "because `recurrence` was rejected before anything was written: the entry "
    "does not exist yet and item {item_id} is still authorised. This is a "
    "parameter the tool refuses, not a fact the user has to supply, so do not "
    "ask them for it. If the arrangement happens once, send the call again with "
    "`recurrence` omitted or null, the same interaction_item_id ({item_id}), "
    "the same render binding, and every other argument unchanged -- that is "
    "your one attempt. If you do not know whether it repeats, ask the user "
    "about the repeat and nothing else. " + NO_META_ACKNOWLEDGEMENT
)
#: The same correction arriving under a new id while one item is still waiting
#: for it. A tool result, so it never reaches the user. The binding is the
#: model's to repair: rewriting the id here would move a user's authorisation
#: onto a call they never saw.
RECURRENCE_REBIND_PREFIX = "Error: not run. This is the correction "
RECURRENCE_REBIND_RESULT = (
    RECURRENCE_REBIND_PREFIX + "{item_id} is waiting for, but it came under a "
    "different interaction_item_id, so it reads as a new act needing its own "
    "authorisation. Re-send it with interaction_item_id {item_id} and the same "
    "render binding; the one attempt is still unspent."
)
#: Owner ruling 2026-09-08 — the strike's other half. The strike stops one tool
#: inside one response; this stops the *turn* when the same tool has come back an
#: error on this many separate iterations, which is the shape the runner spins
#: on: error result, re-prompt, same call, error result. Counted per iteration,
#: not per call, because a burst of identical failures in one response is what
#: the strike already answers (regress12 acc_004's 46 guessed address ids).
#:
#: One *above* the strike limit, and derived rather than written out (review item
#: 4, 2026-09-09): at the same value the strike's "ask the user for the one it
#: needs" prompt was appended on the very iteration the cap ended, so it was
#: written into the history and no request ever carried it. The extra iteration
#: is what delivers it.
TOOL_RETRY_CAP = TOOL_FAILURE_STRIKE_LIMIT + 1
#: regress20 acc_027 — a per-session selection tool is offered on the beat that
#: activates its attribute and withdrawn once the session has the setting. The
#: PA called `IX_emotional_engagement` on the following beat, the base registry
#: answered "Tool ... not found", the PA read that as a broken call and re-sent
#: it, and the third refusal ended the session on the selection contract. The
#: setting is already recorded, so there is nothing to refuse and nothing to
#: redo: the call is answered like any other tool call and dropped.
IX_RESELECTION_IGNORED_RESULT = (
    "Selection for {attribute} is already recorded for this session; this call "
    "is ignored. Continue your reply."
)
#: The per-turn payload carriers. A missing one is a contract failure; a missing
#: `IX_continue_<attribute>` is not (§3.0 required table vs §3.7).
_RENDER_TOOL_NAMES = frozenset(IX_RENDER_TOOL_NAMES.values())
#: The per-session selection carriers, the `required == "always"` tier. Every
#: exposed attribute must be given a named setting and has no unset surface, so
#: a selection the model never makes leaves the turn as unrenderable as a
#: missing payload does — same ladder, same fatal end.
_SELECTION_TOOL_NAMES = frozenset(spec.name for spec in IX_PASSIVE_SPECS.values())


def _fatal_missing(missing: Sequence[str]) -> list[str]:
    """The owed IX tools whose absence ends the turn, in pin order.

    Selections come first: `tool_choice` pins one tool per request, and a render
    payload is only meaningful once the setting it renders under has been
    chosen, so a turn owing both spends its first pin on the selection.
    """
    return [
        *(name for name in missing if name in _SELECTION_TOOL_NAMES),
        *(name for name in missing if name in _RENDER_TOOL_NAMES),
    ]


def _ix_refusal_stage(names: Sequence[str]) -> str:
    """Which tier of the IX contract the refused calls belong to (hy4 acc_031).

    The two tiers are refused by the same registry check and used to be filed
    under the same cause, so the record could not say whether the model failed
    to *choose* a setting or failed to *render* under one it had already chosen.
    Read off the tool names, which is the one fact that separates them.
    """
    kinds = {
        (
            "selection" if name in _SELECTION_TOOL_NAMES
            else "render" if name in _RENDER_TOOL_NAMES
            else "other"
        )
        for name in names
    }
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _missing_required_fields(
    definitions: Sequence[dict[str, Any]], name: str, arguments: Any
) -> list[str]:
    """The tool's own required parameters this call left out, in schema order.

    Taken from the schema the turn actually offered rather than from the
    refusal text, so the record names fields and not a parsed sentence.
    """
    sent = arguments if isinstance(arguments, dict) else {}
    for definition in definitions:
        function = definition.get("function") or {}
        if str(function.get("name") or "") != name:
            continue
        required = (function.get("parameters") or {}).get("required") or []
        return [field for field in required if field not in sent]
    return []
#: Holds that open no gate *here*: a withheld batch is the setting working as
#: written, and a spent release is a refused duplicate -- neither owes the user
#: an ask. A `capability_handoff` hold does owe one, but the ask is the
#: boundary's `handoff_message`, so its gate is opened by
#: `_capability_handoff_gate` with the right `message_source` rather than by the
#: generic path below.
_NO_GATE_HOLD_REASONS = frozenset({
    "batch_not_released", "release_spent", "capability_handoff",
})
#: §3.4 (part 3) — the two hold statuses that end the turn on their own. A batch or a
#: question hold opens no gate, so before this the loop went back to the
#: provider and the provider's *next* message narrated the held call as done:
#: live, `v3g03` withheld the send and the following text-only message read
#: `📧 Email confirmation sent to Dr. Okafor`. The same-message narration rule
#: cannot reach that shape — deepseek writes the call with empty content and
#: the prose one message later — so the turn stops here, exactly as a gate does.
_TURN_ENDING_HOLD_STATUSES = frozenset({"held_for_batch", "held_for_questions"})
#: §3.4 (part 5) — the only tools the closing iteration is shown. Every task tool is
#: gone, because nothing more may run this turn; neither of these is required
#: there, so the closing message owes no payload.
_CLOSING_TOOL_NAMES = frozenset({
    *IX_RENDER_TOOL_NAMES.values(),
    *IX_CONTINUE_TOOL_NAMES.values(),
})
#: §3.4 (part 5) — the result the closing iteration sees for a call the hold stopped.
#: Executed calls keep their real results; this one says in as many words that
#: its call did not run, so the closing text has nothing to report it as.
IX_HELD_TOOL_RESULT = json.dumps(
    {
        "status": "held",
        "detail": (
            "not executed this turn — waiting for the user; "
            "do not describe it as done"
        ),
    },
    ensure_ascii=False,
)
#: §3.3 (part 2)·A — the same result for the other fail-closed exit. A binding refusal
#: is not "waiting for the user": nothing was put to anybody, the ids were the
#: problem, and the turn's whole batch is gone. Live `v7r06` t18 held all six
#: calls this way and shipped the loop placeholder, because the closing
#: iteration was only ever run on the batch / question path.
IX_BINDING_HELD_TOOL_RESULT = json.dumps(
    {
        "status": "held",
        "detail": (
            "not executed: missing/duplicate interaction_item_id — "
            "re-issue with ids next turn"
        ),
    },
    ensure_ascii=False,
)
#: regress23m `acc_001` / `acc_004`, 2026-09-12 — the tool protocol's last
#: resort. A call the wrapper withheld gets no result at the point it is
#: withheld (§2.4: a held call never touches the world), and the paths that
#: re-request inside the same turn -- the `ix_selection_invalid` retry, the
#: batch/binding repairs, the deduped duplicate selection -- left its
#: `tool_call_id` unanswered in `messages`. deepseek/gemini/qwen tolerate that;
#: OpenAI does not, and `openai/gpt-5.6-sol` returned
#: `400 No tool output found for function call <id>` on the next request.
#: Emitted only immediately before a request, and only for ids nothing else has
#: answered, so every path that does write its own result keeps it.
WITHHELD_TOOL_RESULT = json.dumps(
    {
        "status": "withheld",
        "detail": (
            "not executed this turn — the call was withheld before it ran; "
            "do not describe it as done"
        ),
    },
    ensure_ascii=False,
)
#: Owner decision 2026-09-07 (regress5 acc_003): a turn that selected Bookend or
#: Full narration and executed task calls with no `_ix_notice` on any of them
#: reaches the user as if the setting were Silent -- the spec says so in as many
#: words, and the model ignored it on the one turn the session exists to test.
#: The wrapper asks for the opening line, up to twice; a turn still without one
#: fails the IX contract like a missing render payload, so no accumulation
#: session persists a narrating turn that did not narrate. Task tools are
#: withdrawn here as in every other closing request: this can only produce the
#: sentence, never start work.
IX_MISSING_NOTICE_PROMPT = (
    "[interaction wrapper] None of the task calls you made this turn carried "
    "`_ix_notice`, and your process_visibility selection shows the user an "
    "opening line before the result. Write that one line now: one sentence, "
    "in the user's terms, saying what you set out to do in this turn end to "
    "end -- not what you found. Text only; do not call tools. "
    + NO_META_ACKNOWLEDGEMENT
)
#: How many times the opening line is asked for before the turn fails.
IX_MISSING_NOTICE_ATTEMPTS = 2
#: §3.4 (part 5) — what the closing iteration is asked for. Ending at the hold left
#: the turn with nothing to say whenever the model narrates only *after* its
#: calls (deepseek writes the tool-call message with empty content and the prose
#: one message later), and the loop shipped its placeholder. One more iteration
#: asks for that message under conditions where it can do no harm: no task tool
#: is offered, and the held results say what actually happened.
IX_HOLD_CLOSING_PROMPT = (
    "[interaction wrapper] This turn ends here; nothing further will run in it. "
    "The results above mark which of your calls were held — those did not "
    "happen. Write the user-facing message for this turn now: what was actually "
    "done, and what you are putting to the user. Do not describe a held item as "
    "done, sent, booked or filed. Keep each `[[batch:...]]` / `[[q:...]]` marker "
    "in front of the text it owns. Text only; do not call tools. "
    + NO_META_ACKNOWLEDGEMENT
)
#: Round 9 item 1 — the one extra request an otherwise-silent turn buys. Not a
#: closing request: there may be no hold at all. `v8s01` t26 answered a
#: substantive instruction (book a room, move an event) with an empty completion
#: and no tool calls, and what the user read was the loop's placeholder. Task
#: tools are gone here, exactly as in the closing request, so this cannot start
#: work — it can only produce the sentence the turn owes.
IX_EMPTY_REPLY_PROMPT = (
    "[interaction wrapper] Your previous message had no user-facing text. "
    "Write your reply to the user now; do not call tools. "
    + NO_META_ACKNOWLEDGEMENT
)
#: acc_003 t=20 — the other way a turn can arrive with nothing in it. Asked for
#: the exact updated metadata fields, the PA sent `Summary of the two updated
#: entries is complete.` and no tool call, and the wrapper released the
#: completion over it; the user accepted the empty claim and the session closed.
#: Not a template: the wrapper says what is missing and the PA writes the turn,
#: exactly once. Task tools are withdrawn here as in the other closing request,
#: so this cannot start work it did not do.
IX_UNFINISHED_TURN_PROMPT = (
    "[interaction wrapper] Your closing line says the work is done, and your "
    "reply contains nothing else: no answer, no summary, and no tool call this "
    "turn. A closing line reports the work; it is not the work. Write the reply "
    "the user asked for now -- the content itself, in your own words -- and "
    "leave the closing line to the end of it. "
    + NO_META_ACKNOWLEDGEMENT
)
#: acc_003 t=17 — the second shape of the same defect. The prose there is not
#: the closing line repeated but *different* text making the same claim ("That
#: covers all five remaining entries", over a completion asserting a list that
#: is nowhere in the reply), so the containment test passes it. What the two
#: shapes share is a closing line that asserts delivery, and these are the verbs
#: that assert it. A closed list on purpose: anything wider starts catching
#: turns that merely mention the word.
WORK_ASSERTED_RE = re.compile(
    r"\b(?:summari[sz]ed|updated|wrote|written|saved|sent|completed|done)\b",
    re.IGNORECASE,
)
#: The prose a closing line can stand over. Past it the reply has a body of its
#: own and the closing line is doing its actual job; acc_003 t=17's entire
#: delivered turn was 130 characters.
UNFINISHED_TURN_PROSE_LIMIT = 300
IX_CAPABILITY_RESOLVE_REMINDER = (
    "A tool reported a capability limit. Call "
    f"{IX_CAPABILITY_RESOLVE_TOOL} for this turn and declare blocked_goal, "
    "boundary_kind, recovery_options and handoff_message before doing anything "
    "else. " + NO_WRAPPER_TALK
)
#: §3.3 (part 2)·C — the renderer never writes the ask for the PA, so a gated call
#: that carried no `_ix_confirmation_message` leaves the turn with nothing to
#: say. One repair, then the same contract failure the render payloads get.
IX_GATE_MESSAGE_REPAIR = (
    "[interaction wrapper] A call in that response has to be put to the user "
    "before it can run, and it carried no {field}, so there is nothing to ask "
    "with. Resend that call with {field} filled in: the sentence you would say "
    "to the user, with the real recipient, draft or amount you now have. "
    "Nothing is written for you if you leave it out. "
    + NO_META_ACKNOWLEDGEMENT
)
#: §3.7 (part 3) — a `release` is a promise to act in this same reply, and releases are
#: never auto-replayed (§3.3 (part 3)). Live, `v3g03` released the held email, issued
#: no call, closed the episode, and told the user it had gone out. Asked once; a
#: second response without the call keeps the item held (`ix_release_missing`).
IX_RELEASE_REPAIR = (
    "[interaction wrapper] You reported `release` for {ids}, but this response "
    "carries no task call for {ids}. Nothing is replayed for you: a release "
    "means you perform the action in this same reply. Issue the call now with "
    f"the same {INTERACTION_ITEM_ID_FIELD}, or report `keep` / `skip` instead. "
    + NO_META_ACKNOWLEDGEMENT
)
#: acc_039 — the double ask. Under a setting whose floor stops external actions,
#: the PA wrote `Shall I go ahead and book …?` as plain prose with no call and no
#: plan item; the user authorised it; the next turn's booking call hit the floor
#: and the wrapper asked the same question again, so the user authorised it
#: twice. The gate stays fail-closed (it is the only thing that makes consent
#: real), so the ask is what moves: it has to go out through the id path. One
#: nudge per turn, and whatever comes back is delivered -- a nudge, not a
#: contract, because a prose question is sometimes exactly right (a missing
#: fact, a material ambiguity) and the wrapper cannot tell which this is.
IX_PROSE_ASK_REPAIR = (
    "[interaction wrapper] That reply ends by asking the user's permission, but "
    "the turn issued no call the wrapper could stop and bound no plan item, so "
    f"the ask carries no {INTERACTION_ITEM_ID_FIELD}. Issue the call now with its "
    f"`{IX_CONFIRMATION_MESSAGE_FIELD}` — the wrapper stops the turn and puts that "
    "sentence to the user for you — or declare the proposal as a plan item with a "
    f"bound {INTERACTION_ITEM_ID_FIELD}. If the question is a missing fact rather "
    "than permission to act, ask it again as it stands. "
    + NO_META_ACKNOWLEDGEMENT
)
#: Playability audit §4 — `information_elicitation` splits per `[[q:…]]` **block**,
#: not per question (`render.py::_release_count`). Two missing fields written
#: into one block are released together under `Iterative` exactly as they are
#: under `Structured`, so both settings produce the identical reply and the
#: attribute under test disappears from the transcript. `topic_management` has
#: the same wound and repairs it with `IX_BATCH_SPLIT_REPAIR`; this one is a
#: nudge rather than a contract, the way the prose ask is -- the wrapper cannot
#: write the second question for a model that refuses to split, and release
#: semantics are untouched either way.
IX_QUESTION_SPLIT_REPAIR = (
    "[interaction wrapper] One `[[q:...]]` block in that reply asks more than one "
    "question. The wrapper releases whole blocks, so both questions would go out "
    "together and the asking would not be one at a time. Resend the reply with one "
    "question per block -- one block per missing field -- and the wrapper releases "
    "them a block at a time. "
    + NO_META_ACKNOWLEDGEMENT
)
#: The mirror of `IX_QUESTION_SPLIT_REPAIR` (live `acc_009` t=8). The split
#: repair catches two questions inside one block; this one catches the hole
#: beside it — a payload full of questions and a reply that marks none, which
#: `render.py::_select_sections` drops without releasing, holding or tracing
#: anything. The user read "let me know your answers to those two points" with
#: the two points nowhere in the reply, and the wrapper said nothing to either
#: side. A nudge rather than a contract, for the same reason the split is: the
#: wrapper will not write the PA's questions into its prose.
IX_QUESTION_MARKER_REPAIR = (
    "[interaction wrapper] Your IX_render_information_elicitation payload "
    "declares questions and your reply marks none of them. The wrapper releases "
    "the prose behind a `[[q:<id>]]` marker, so an unmarked question is not "
    "asked at all -- the user reads the framing and never sees the question. "
    "Resend the reply with each declared question written out in your prose "
    "behind its own `[[q:<id>]]` marker, using the ids the payload declares. "
    + NO_META_ACKNOWLEDGEMENT
)
#: acc_011 t=27 — the rationale that opened the reply talking about the reader.
#: Under `Show` the field prints ahead of the answer, so *"The user approved the
#: prioritization logic and asked me to save the document"* was the first thing
#: they read, two lines above "Done! …has been saved to your desktop". Not one
#: coherent assistant. Asked, never rewritten: the renderer will not compose the
#: reasons, and a second work log ships as it stands with the event as the
#: record. Round 2 fixed this by wording, round 3 held, round 4 lost it.
IX_RATIONALE_WORK_LOG_REPAIR = (
    "[interaction wrapper] Your `user_facing_rationale` starts by describing "
    "the user in the third person. That field is printed to them, ahead of your "
    "answer, so it has to be written to them in the second person and it has to "
    "give the reasons behind the answer -- not a recap of what they asked for or "
    "of what you did. Resend the payload with the reasons, or with the field "
    "empty if this turn decided nothing. " + NO_META_ACKNOWLEDGEMENT
)
#: regress13 acc_035 t=7 — three declared certainties, `Express`, and the
#: locator placed none of them, so the one attribute under test had no surface
#: at all on the very turn the reception beat reads and the actor approved a
#: reply with no hedges in it. The read says why: three of the four drops that
#: session were claims copied out of an *earlier* turn's reply, and the fourth
#: ("I will be available during office hours on Thursday, October 15, at 15:00")
#: was a paraphrase of a line that never said it. So this is not locator
#: tolerance -- the locator already token-matches at 80% -- it is the model
#: quoting the wrong text, and the only thing that fixes it is asking again for
#: a fragment of *this* reply. The wrapper never appends the claim itself: an
#: unlocated certainty written into the message would be the wrapper inventing
#: PA prose, which is the one thing §3.1 forbids it.
IX_CERTAINTY_CLAIM_REPAIR = (
    "[interaction wrapper] Every claim in your `<certainties>` block failed to "
    "match a sentence of the reply you just wrote, so the confidence you "
    "declared could not be placed anywhere and none of it reached the user. A "
    "claim from an earlier reply of yours, or a paraphrase of what you meant, "
    "locates nothing. Resend the reply with the block rewritten: each `claim` "
    "copied word for word out of the running prose of *this* reply, as a "
    "sentence fragment you can point at. If no claim in this reply has a "
    "confidence worth stating, send `<certainties></certainties>`. "
    + NO_META_ACKNOWLEDGEMENT
)
#: acc_009 t=8 *and* t=10 — one nudge was not enough. Both turns burned it, both
#: shipped anyway, and the user read "those two points" twice with the two points
#: missing; the questions arrived at t=13, two turns late. Two, not more: the
#: nudge is the wrapper asking, never writing, and a model that has refused the
#: same marker twice is a PA failure the audit should record rather than a
#: request worth paying for a third time.
IX_QUESTION_MARKER_NUDGE_LIMIT = 2
#: Either section marker: a `[[q:…]]` block owns the prose after it until the
#: next marker of *either* kind, the way `render.py::_split_sections` cuts it.
_SECTION_MARKER_RE = re.compile(
    f"(?P<batch>{BATCH_MARKER_RE.pattern})|(?P<question>{QUESTION_MARKER_RE.pattern})"
)
#: A sentence ends at `.`, `!` or `?` followed by whitespace or the end of the
#: text. Deliberately crude, the same reading `render.py` uses for its own
#: sentence filter: the count only has to tell one question from two.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: Which IX contract a `STOP_REASON_IX_CONTRACT_FAILURE` turn broke. Five
#: unrelated rules fail closed on the one stop reason, so the stop reason alone
#: cannot say what the turn owed; each fatal site names its own kind here and
#: the loop maps the kind to the sentence a reader gets. Live regress20
#: `acc_027`: the reaction beat offers no IX tools, the PA re-sent
#: `IX_emotional_engagement` three times, and the turn was reported as
#: "IX_capability_resolve was missing or invalid twice after a tool reported a
#: capability limit" -- no tool on that turn reported anything of the sort. The
#: loop's old mapping read event names and fell through to the capability text
#: by default, which is why a path with no capability in it wore that label.
IX_CONTRACT_RENDER = "render"
IX_CONTRACT_CAPABILITY = "capability"
IX_CONTRACT_SELECTION = "selection"
#: regress23m `acc_001`: the same break, the other cause. `IX_verbosity` *was*
#: offered (`active_ix_tools: ["IX_verbosity"]`); the PA called it with `{}` and
#: the registry refused it for "missing required setting" three times. The turn
#: still wore the not-offered sentence, so the record pointed the reader at the
#: beat's tool list, which was fine, instead of at the arguments, which were not.
IX_CONTRACT_SELECTION_ARGS = "selection_args"
#: hy4 `acc_031`, 2026-09-18: the same break, one tier further down. The turn's
#: setting was already chosen and the *render* payload was refused on its own
#: arguments -- `IX_render_process_visibility` sent the selection tool's
#: `setting`/`evidence`/`application` and left out its own required
#: `completion_message`. The old classification asked only "was the tool
#: offered", which both refusals answer the same way, so the record read
#: `selection_args` and pointed the reader at a choice of setting the model had
#: made correctly one response earlier.
IX_CONTRACT_RENDER_ARGS = "render_args"
IX_CONTRACT_GATE_MESSAGE = "gate"
IX_CONTRACT_NOTICE = "notice"


def internal_user_message(content: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": content,
        "_mempa_internal": True,
    }


def _tool_result_error(result: Any) -> Any | None:
    payload = result
    if isinstance(result, str):
        stripped = result.strip()
        if stripped.startswith((
            "Error",
            "(MCP tool call timed out",
            "(MCP tool call was cancelled",
            "(MCP tool call failed",
        )):
            return stripped
        try:
            payload = json.loads(stripped)
        except (TypeError, json.JSONDecodeError):
            return None

    if isinstance(payload, dict) and payload.get("ok") is False:
        return payload.get("error") or payload
    return None


def _tool_error_class(tool_call: ToolCallRequest, error: Any) -> str:
    """One failure's shape, with the values this call guessed taken out.

    regress12 acc_004: 46 `shopping_order_create` calls, 46 different
    `delivery_address_id`s, one error -- "unknown delivery address: <the value>".
    The id is what varies and the sentence is what repeats, so every string
    argument the call sent is removed before the texts are compared. Short
    values are left in place: a `quantity` of "1" appears in unrelated prose.
    """
    text = str(error)
    arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
    for value in arguments.values():
        if isinstance(value, str) and len(value) >= 3:
            text = text.replace(value, "")
    return " ".join(text.split())


#: The store's code for a `recurrence` the calendar refused before writing
#: anything (gpt `acc_082`). `pa/` imports nothing from `state_server/` and
#: `state_server/` imports nothing from `pa/`, so this is the second copy of one
#: literal, the way `CAPABILITY_LIMIT_ERROR_CODE` already is
#: (`state_server/store.py` says so at its own copy); the source of truth is
#: `state_server.store.CALENDAR_RECURRENCE_INVALID_ERROR_CODE` and
#: `tests/test_pa/test_runner_ix.py` asserts the two stay equal.
CALENDAR_RECURRENCE_INVALID_ERROR_CODE = "calendar_recurrence_invalid"


def _is_recurrence_invalid(error: Any) -> bool:
    """gpt acc_082 — the store's own code for "recurrence wrong, nothing written".

    Read off `error_code`, never off the sentence: `_tool_error_class` strips
    the call's string arguments out of the text, so the live failure signature
    was `recurrence='' is not …` and could not be told apart from a call that
    really sent an empty value.
    """
    return (
        isinstance(error, dict)
        and error.get("error_code") == CALENDAR_RECURRENCE_INVALID_ERROR_CODE
    )


def _business_args_signature(arguments: Any) -> str:
    """One call's arguments without `recurrence` or the wrapper's own fields.

    Two calls share a signature when they are the same act: a changed date,
    title, location or participant list is a different arrangement and must go
    back through the gate even under the same item id (§4).
    """
    args = arguments if isinstance(arguments, dict) else {}
    return json.dumps(
        {
            key: value for key, value in args.items()
            if key not in PER_CALL_INJECTED_FIELDS and key != "recurrence"
        },
        sort_keys=True, default=str,
    )


def _recurrence_repair(
    collector: TurnCollector, tool_call: ToolCallRequest, item_id: str
) -> tuple[str, str, RecurrenceRepair] | None:
    """Does this struck call spend the turn's one recurrence exception (§3.3)?

    `("run", item_id, repair)` for a correction on the same authorised item;
    `("rebind", original_id, repair)` for the same correction under a fresh id
    while exactly one item is waiting for it; `None` for everything else -- a
    changed arrangement, a recurrence value that is still there, a second call
    once the one attempt is spent, or an ambiguous choice of two waiting items.
    """
    arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
    # Omitted or null, and nothing else: a corrected *legal* recurrence is a
    # different arrangement from the one-off the user authorised, and the
    # wrapper does not decide that on their behalf.
    if arguments.get("recurrence") is not None:
        return None
    signature = _business_args_signature(arguments)
    pending = {
        candidate: repair
        for candidate, repair in collector.recurrence_repairs.items()
        if not repair.spent
        and repair.tool == tool_call.name
        and repair.business_args == signature
    }
    if (repair := pending.get(item_id)) is not None:
        return ("run", item_id, repair)
    if len(pending) == 1:
        original, repair = next(iter(pending.items()))
        return ("rebind", original, repair)
    return None


def _struck_tools(collector: TurnCollector) -> set[str]:
    """The tools that have spent their strikes on one error this turn."""
    return {
        tool for (tool, _error_class), failures
        in collector.tool_failure_strikes.items()
        if failures >= TOOL_FAILURE_STRIKE_LIMIT
    }


def _no_ask_hold_signatures(
    collector: TurnCollector, start: int
) -> set[tuple[str, str]]:
    """`(tool, arguments)` of the holds recorded from `start` on that ask nothing.

    Only `NON_EPISODE_HOLD_REASONS` -- a consent ask opens an episode and is
    answered by the user, so re-issuing the call after it is the protocol, not
    a loop. The arguments come off the call record the hold names, so the same
    booking held twice reads as one signature and a second, different act on
    the tool reads as another.
    """
    args_by_id = {
        call.interaction_item_id: call.args
        for call in collector.calls
        if call.interaction_item_id
    }
    return {
        (
            item.tool,
            json.dumps(
                args_by_id.get(item.interaction_item_id, {}),
                sort_keys=True, default=str,
            ),
        )
        for item in collector.held[start:]
        if item.reason in NON_EPISODE_HOLD_REASONS
    }


def _answer_withheld_tool_calls(messages: list[dict[str, Any]]) -> list[str]:
    """Give every unanswered `tool_call_id` in `messages` a result, in place.

    The OpenAI-side contract, not an IX rule: a request whose history carries an
    assistant `tool_calls` entry with no matching `role: "tool"` message is
    rejected outright (`400 No tool output found for function call <id>`).
    Called immediately before each provider request, so a path that writes its
    own result for a withheld call -- `_closing_iteration`'s
    `IX_HELD_TOOL_RESULT` / `IX_BINDING_HELD_TOOL_RESULT` -- has already written
    it and is left alone. Returns the ids it had to answer, for the audit line.
    """
    answered = {
        str(message.get("tool_call_id"))
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    repaired: list[str] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(tool_calls, list):
            index += 1
            continue
        # The synthetic results go at the end of this message's own run of tool
        # replies: the provider reads them positionally, so one appended at the
        # tail of the whole history would answer the wrong assistant message.
        insert_at = index + 1
        while (
            insert_at < len(messages)
            and messages[insert_at].get("role") == "tool"
        ):
            insert_at += 1
        missing = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            call_id = str(tool_call.get("id") or "")
            if not call_id or call_id in answered:
                continue
            answered.add(call_id)
            repaired.append(call_id)
            missing.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": str(
                    (tool_call.get("function") or {}).get("name")
                    or tool_call.get("name")
                    or ""
                ),
                "content": WITHHELD_TOOL_RESULT,
            })
        messages[insert_at:insert_at] = missing
        index = insert_at + len(missing)
    return repaired


def _with_capability_boundary(ix_context: dict[str, Any]) -> dict[str, Any]:
    """This turn's IX context with `capability_boundary` in the active set.

    The trigger, not the beat, activates the attribute (§3.5; audit §2,
    acc_026). A copy, not an in-place edit: the caller's dict belongs to the
    turn, and the session-scoped half of the stickiness is the loop's
    (`pa/agent/loop.py`, `_remember_ix_triggers`).
    """
    active = list(ix_context.get("session_active_ix") or ())
    if CAPABILITY_BOUNDARY_ATTRIBUTE in active:
        return ix_context
    return {
        **ix_context,
        "session_active_ix": [*active, CAPABILITY_BOUNDARY_ATTRIBUTE],
    }


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    required_tool_names: list[str] = field(default_factory=list)
    ix_context: dict[str, Any] = field(default_factory=dict)
    inherited_ix_selections: list[IXSelection] = field(default_factory=list)
    # §3.6: restated once before the first provider request of every turn, so a
    # pass-through setting does not lapse on turns without tool calls.
    turn_reminder: str = ""
    # §4.3: the controlled hold reminder. Hold ids and intents only -- it never
    # copies the user's own words, and never the sideband verdict.
    hold_prompt: str = ""
    hold: IXHold = field(default_factory=IXHold)
    # §3.7 (part 3): PA-blind consent sideband. Gate state, never prompt text.
    consent_decisions: list[ConsentDecision] = field(default_factory=list)
    turn_index: int = 0
    # §3.5 (part 4): rebuilds the per-turn tool view once a tool has reported a
    # capability limit, so `IX_capability_resolve` can be offered and forced on
    # the *next* provider iteration rather than declared up front.
    tools_factory: Callable[[dict[str, Any]], Any] | None = None
    llm_task_identity: LLMTaskIdentity | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    #: One audit line per provider iteration (tool names, content length,
    #: finish reason, pinned tool): diagnostics only, outside the IX event
    #: stream, so a turn that ends in a contract failure still shows what the
    #: model answered with.
    iteration_log: list[dict[str, Any]] = field(default_factory=list)
    #: The turn's selection calls `_dedupe_ix_selections` dropped before
    #: execution, as `{id, name, args, outcome}`. Memory-pool input only: they
    #: ran nothing, so they emit no tool event and reach the collector's call
    #: records through no other path (§3.3 `outcome=ignored_duplicate`).
    ignored_ix_selections: list[dict[str, Any]] = field(default_factory=list)
    # §2.2: the collector's product, consumed by the renderer in the loop.
    render_input: IXRenderInput | None = None
    collector: Any = None
    #: One of the `IX_CONTRACT_*` kinds while `stop_reason` is
    #: `ix_contract_failure`, `None` otherwise. Set at the fatal site itself,
    #: because the event stream cannot say which rule ended the turn: the same
    #: events are emitted by the non-fatal repairs that precede it.
    ix_contract_failure_kind: str | None = None
    #: Present only when an existing in-turn IX repair ladder consumed exactly
    #: three responses for the same unresolved operation.  The harness may use
    #: this typed proof to keep the live partial state and skip an ordinary ACC.
    ix_contract_exhaustion: dict[str, Any] | None = None


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        #: §3.0 / §7.1 — *attempted* is `tools_used`; this is *completed*. An IX
        #: call the registry refused (live: `IX_autonomy_level(setting="Suggest")`,
        #: a setting the three-way enum no longer has) returns an error and no
        #: payload, so it contributes nothing to the directive. Counting it as
        #: satisfying the requirement let the turn finish on an empty policy,
        #: which gates nothing at all.
        ix_completed: set[str] = set()
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        #: Named at every site that sets `STOP_REASON_IX_CONTRACT_FAILURE`, so
        #: the loop never has to guess which contract this turn broke.
        contract_failure_kind: str | None = None
        contract_exhaustion: dict[str, Any] | None = None
        tool_events: list[dict[str, Any]] = []
        iteration_log: list[dict[str, Any]] = []
        #: Selection calls `_dedupe_ix_selections` dropped before execution.
        #: They are nowhere else: no tool event is emitted for a call that
        #: never ran, and the collector records only calls that reached it.
        ignored_ix_selections: list[dict[str, Any]] = []
        ix_directive = directive_for(spec.inherited_ix_selections)
        collector = TurnCollector(turn_index=spec.turn_index)

        # §3.6 / §4.3: both are internal messages -- `_save_turn` drops them, so
        # they never reach runtime_history, memory or the user.
        for prompt in (spec.turn_reminder, spec.hold_prompt):
            if prompt:
                messages.append(internal_user_message(prompt))

        binding_retries = 0
        #: A fail-closed refusal is waiting to see whether its retry runs. Set
        #: when the repair goes out, cleared when the resent batch executes and
        #: the refused rows are stamped repaired.
        binding_refusal_pending = False
        conditional_retries = 0
        #: hy4 `acc_031` — the argument refusals' own budget. It used to share
        #: `conditional_retries` with the missing-carrier ladder, so a turn that
        #: had already been reminded and pinned arrived at its *first* invalid
        #: argument with the budget at 2 and ended there: the model sent a
        #: correct `IX_render_process_visibility` payload, then re-sent the tool
        #: with the selection tool's fields, and that one mistake was fatal with
        #: no repair asked for. Two different faults with two different repairs,
        #: so two counters -- and neither is ever reset on a success, because a
        #: turn alternating a good call and a bad one would then retry forever.
        #: `spec.max_iterations` is still the ceiling over both.
        ix_argument_retries = 0
        #: The payload carriers this turn has asked for, by reminder or by pin.
        #: A landed carrier is answered once and only if it was demanded
        #: (`IX_CARRIER_RECORDED_NOTE`).
        carriers_demanded: set[str] = set()
        carriers_recorded_note_sent: set[str] = set()
        #: `IX_<attribute>` arguments this turn sent in a response that was
        #: voided, by tool name. A voided response banks nothing, but the model
        #: reasoned once and is not asked to reason again, so a later selection
        #: for the same setting reads its blanks back from here (§3.0).
        voided_selections: dict[str, dict[str, Any]] = {}
        #: The tools already recorded as missing this turn, so the second miss is
        #: recorded once rather than once per remaining iteration (§3.0 / T16).
        payload_missing_reported: set[str] = set()
        #: Owner ruling 2026-09-08 — how many *iterations* of this turn each tool
        #: has come back an error on. One per iteration however many of its calls
        #: failed in it: the burst inside a single response is the strike's
        #: business, and this one is about the re-prompt loop.
        retry_failures: dict[str, int] = {}
        #: The same backstop for the calls the *wrapper* stops. A held call gets
        #: no error result, so `failing_now` never sees it: live regress21r
        #: acc_039 t19 re-issued one booking 25 times, the wrapper held every
        #: one of them `release_spent`, and 25 of the turn's 40 iterations went
        #: on a loop nothing could break. Keyed on (tool, arguments) so an
        #: identical call held again counts and a different act does not, and
        #: only for the reasons that ask the user nothing
        #: (`NON_EPISODE_HOLD_REASONS`) -- the first hold of a consent ask is
        #: the protocol working and is never counted. `held_cursor` is the
        #: number of hold rows already counted; `clear_holds()` shortens the
        #: list, so it is clamped before use.
        held_repeats: dict[tuple[str, str], int] = {}
        held_cursor = 0
        gate_message_retries = 0
        release_repairs = 0
        #: acc_039 — the unbound prose ask is nudged once per turn and never
        #: pinned, so it has its own counter: spending `conditional_retries` here
        #: would take a pin off the render-payload ladder for a repair whose
        #: whole point is that ignoring it is allowed.
        prose_ask_nudges = 0
        #: Playability audit §4 — the undivided `[[q:…]]` block, nudged once per
        #: turn. Its own counter for the same reason as `prose_ask_nudges`: it
        #: pins nothing and ending the turn on it is not an option, so it must
        #: not spend the render-payload ladder's budget.
        question_split_nudges = 0
        #: acc_009 — the payload's questions marked nowhere, nudged once per turn.
        #: Its own counter for the same reason as the two above.
        question_marker_nudges = 0
        #: acc_011 — the `user_facing_rationale` written about the reader instead
        #: of to them, nudged once per turn. Same reason again.
        rationale_work_log_nudges = 0
        #: regress13 acc_035 — the whole `<certainties>` block missing the prose,
        #: re-asked once per turn. Its own counter like the three above; owner
        #: ruling 4 (2026-09-08) made the second all-drop ship untagged with the
        #: failure recorded, rather than ending the step.
        certainty_claim_retries = 0
        #: §3.7 (part 3) — set for the one response that answers the release repair,
        #: so the payload reminder cannot discard the call the repair asked for.
        release_repair_sent = False
        capability_required = False
        capability_attempts = 0
        #: One unresolved IX operation may change shape while it is repaired:
        #: missing selection -> selected but missing render payload is still the
        #: same operation.  Count provider responses across those local ladders
        #: so their individual budgets cannot multiply into 3x3 requests.
        ix_repair_attempts = 0
        # A carrier repair can complete in the same response that opens a new
        # confirmation gate. Track its concrete debt and fresh successful events:
        # a tool name merely received (or completed before this debt) is not proof.
        ix_carrier_debt: set[str] = set()
        ix_carrier_repaired: set[str] = set()

        def _reset_ix_repair_attempts() -> None:
            nonlocal ix_repair_attempts, ix_argument_retries, conditional_retries
            ix_repair_attempts = 0
            ix_argument_retries = 0
            conditional_retries = 0
            ix_carrier_debt.clear()
            ix_carrier_repaired.clear()

        def _spend_ix_repair_attempt(
            contract: str, operation: str
        ) -> dict[str, Any] | None:
            nonlocal ix_repair_attempts
            ix_repair_attempts += 1
            if contract in (IX_CONTRACT_RENDER, IX_CONTRACT_RENDER_ARGS):
                # This iteration's response broke the render contract: the
                # upstream that served it is charged if the response is empty
                # (provider avoidance, `harness/provider_avoidance.py`).
                self.provider.record_contract_failure(
                    provider_response, f"pa_ix_{contract}: {operation}"
                )
            if ix_repair_attempts < 3:
                return None
            return {
                "attempts": ix_repair_attempts,
                "contract": contract,
                "operation": operation,
            }
        #: regress12 acc_026 — replies the provider cut off at the token limit,
        #: counted apart from every ladder budget: nothing was decided in them,
        #: so nothing may be spent on them.
        provider_truncations = 0
        #: regress12 acc_003 — the same, for a response the provider marked an
        #: error: no reply came back, so no budget may be charged for one.
        provider_errors = 0
        #: regress12 acc_004 — tools already struck off for this turn *and* named
        #: to the model. The strike itself is on the collector; this is only so
        #: the instruction is written once however many iterations follow.
        strike_prompted: set[str] = set()
        #: Attributes whose §3.6 restatement has already been said on this turn,
        #: so a selection made mid-turn is restated once rather than once per
        #: remaining iteration.
        restated_in_turn: set[str] = set()
        #: The tool `tool_choice` pins on the next request, for one request only
        #: (§3.0, third attempt). `None` leaves the provider on "auto".
        forced_tool: str | None = None
        pinned_tool: str | None = None
        forced_tools_used: set[str] = set()
        #: Bound here as well as per iteration, so the single exit below can hand
        #: it to the empty-reply rule even when the loop body never ran.
        hook_context = AgentHookContext(iteration=0, messages=messages)

        for iteration in range(spec.max_iterations):
            hook_context = AgentHookContext(iteration=iteration, messages=messages)
            await hook.before_iteration(hook_context)
            # Last thing before the request: whatever this turn withheld, the
            # history that goes out has a result for every call it carries.
            if (withheld := _answer_withheld_tool_calls(messages)):
                collector.emit(
                    "tool_call_withheld_answered",
                    count=len(withheld),
                    turn_index=spec.turn_index,
                    iteration=iteration,
                )
            provider_request: dict[str, Any] = {
                "messages": messages,
                "tools": spec.tools.get_definitions(),
                "model": spec.model,
            }
            if forced_tool is not None:
                provider_request["tool_choice"] = {
                    "type": "function",
                    "function": {"name": forced_tool},
                }
                pinned_tool = forced_tool
                forced_tool = None
            if spec.temperature is not None:
                provider_request["temperature"] = spec.temperature
            if spec.max_tokens is not None:
                provider_request["max_tokens"] = spec.max_tokens
            if spec.reasoning_effort is not None:
                provider_request["reasoning_effort"] = spec.reasoning_effort

            if hook.wants_streaming():
                async def _stream(delta: str) -> None:
                    await hook.on_stream(hook_context, delta)

                streaming_request = {
                    **provider_request,
                    "on_content_delta": _stream,
                }
                if spec.llm_task_identity is not None:
                    streaming_request["task_identity"] = spec.llm_task_identity
                provider_response = await self.provider.chat_stream_with_retry(
                    **streaming_request
                )
            else:
                if spec.llm_task_identity is not None:
                    provider_request["task_identity"] = spec.llm_task_identity
                provider_response = await self.provider.chat_with_retry(
                    **provider_request
                )

            raw_usage = provider_response.usage or {}
            usage = {
                "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
            }
            # Audit line per iteration: what the model answered with, so a turn
            # that ends in a contract failure still shows whether a pinned tool
            # was called, something else was called, or only prose came back.
            iteration_entry: dict[str, Any] = {
                "tool_names": [call.name for call in (provider_response.tool_calls or [])],
                "ix_arguments": {
                    call.name: {
                        key: (value if isinstance(value, (int, float, bool)) else str(value)[:160])
                        for key, value in (call.arguments or {}).items()
                    }
                    for call in (provider_response.tool_calls or [])
                    if call.name.startswith(IX_TOOL_NAME_PREFIX)
                },
                "content_chars": len(provider_response.content or ""),
                "finish_reason": str(getattr(provider_response, "finish_reason", "") or ""),
                "forced_tool": pinned_tool,
                "completion_tokens": usage["completion_tokens"],
                "capability_required": capability_required,
                "outcome": "",
            }
            iteration_log.append(iteration_entry)
            # regress16 acc_006 — carried to the single exit, and reset on every
            # response that came back whole, so what it says at the exit is
            # "the last thing the provider sent was cut off at the token limit".
            collector.truncated_completion_tokens = (
                usage["completion_tokens"]
                if iteration_entry["finish_reason"] == "length"
                else None
            )
            request_pinned_tool = pinned_tool
            pinned_tool = None
            hook_context.response = provider_response
            hook_context.usage = usage
            hook_context.tool_calls = list(provider_response.tool_calls)
            # regress12 acc_026 — a reply the provider cut off at `max_tokens`
            # with nothing in it: no text, no calls, `finish_reason == "length"`.
            # The model decided nothing there, so it is answered before any
            # nudge, ladder or gate can read it as a miss and spend a budget on
            # it. Two retries keep the existing request budget; a third empty
            # truncation ends as a provider failure, never an IX ladder miss.
            if (
                iteration_entry["finish_reason"] == "length"
                and not provider_response.has_tool_calls
                and not (provider_response.content or "").strip()
            ):
                provider_truncations += 1
                iteration_entry["outcome"] = "provider_truncated"
                collector.emit(
                    "provider_truncated",
                    attempt=provider_truncations,
                    **(
                        {"completion_tokens": usage["completion_tokens"]}
                        if raw_usage.get("completion_tokens") else {}
                    ),
                )
                if provider_truncations <= PROVIDER_TRUNCATION_RETRIES:
                    messages.append(
                        internal_user_message(PROVIDER_TRUNCATED_RETRY)
                    )
                    await hook.after_iteration(hook_context)
                    continue
                collector.emit("provider_truncation_exhausted", attempts=provider_truncations)
                if hook.wants_streaming():
                    await hook.on_stream_end(hook_context, resuming=False)
                stop_reason = "error"
                error = f"provider_truncation_exhausted: empty length responses={provider_truncations}"
                final_content = spec.error_message or _DEFAULT_ERROR_MESSAGE
                hook_context.final_content = final_content
                hook_context.error = error
                hook_context.stop_reason = stop_reason
                await hook.after_iteration(hook_context)
                break
            # regress12 acc_003 t=12 — a response the provider marked an error.
            # Its `content` is the gateway's error body, not the model's words,
            # so nothing below may read it: it must not become a segment, an
            # assistant message, a ladder miss or the turn's text. One more
            # request with the same messages, and then the error exit -- the
            # same one that used to sit past the ladder, moved up here so a
            # response nobody wrote cannot spend a reminder or a pin on its way
            # to it.
            if provider_response.finish_reason == "error":
                provider_errors += 1
                if provider_errors <= PROVIDER_ERROR_RETRIES:
                    iteration_entry["outcome"] = "provider_error_retried"
                    collector.emit("provider_error_retried", attempt=provider_errors)
                    # C/USER_FORCED_TOOL_RETRY_FIX_20260924.json: the retry re-sends the same
                    # request, so a pinned tool_choice must be pinned again.
                    if request_pinned_tool is not None:
                        forced_tool = request_pinned_tool
                    await hook.after_iteration(hook_context)
                    continue
                iteration_entry["outcome"] = "provider_error"
                if hook.wants_streaming():
                    await hook.on_stream_end(hook_context, resuming=False)
                final_content = (
                    hook.finalize_content(hook_context, provider_response.content)
                    or spec.error_message
                    or _DEFAULT_ERROR_MESSAGE
                )
                stop_reason = "error"
                error = final_content
                hook_context.final_content = final_content
                hook_context.error = error
                hook_context.stop_reason = stop_reason
                await hook.after_iteration(hook_context)
                break
            collector.add_reasoning(provider_response.reasoning_content)
            # The provider seam lifted tool-call syntax the model typed into its
            # message back into real calls (regress5 acc_006). Recorded per
            # iteration, because a run whose calls all arrive this way is a
            # model that is not really speaking the tool protocol and the batch
            # review has to be able to count it.
            if provider_response.dsml_converted:
                collector.emit(
                    "ix_dsml_calls_recovered",
                    calls=provider_response.dsml_converted,
                    tools=[
                        tool_call.name for tool_call in provider_response.tool_calls
                        if tool_call.id.startswith("dsml_")
                    ],
                )
            # Converted or still unparsed, the turn is now marked for the exit:
            # a model that typed its calls out had work to report, so the
            # empty-reply template must not be what the user reads.
            if provider_response.dsml_converted or strip_dsml_markup(
                provider_response.content or ""
            )[1]:
                collector.dsml_markup_seen = True
            # The boundary declaration banks the moment it arrives, before any
            # gate can consume the response: live regress3 acc_026 the pinned
            # `IX_capability_resolve` reply was taken by the render-payload
            # ladder (a reminder and a resend), never reached the capability
            # gate, and the turn died on `ix_boundary_unresolved` although the
            # payload was correct. A pin carries one function, so the rest of
            # the debt is still reminded or pinned on the next iteration.
            if capability_required and provider_response.has_tool_calls:
                resolve_now = [
                    tool_call for tool_call in provider_response.tool_calls
                    if tool_call.name == IX_CAPABILITY_RESOLVE_TOOL
                ]
                if resolve_now:
                    collector.add_payloads(resolve_now)
                    if not boundary_payload_invalid(collector.payloads):
                        capability_required = False
                        iteration_entry["outcome"] = "resolve_banked"
                        # The carriers still owed (selection, render payload)
                        # were unanswerable while the boundary was open -- the
                        # render tool is only offered under a chosen setting --
                        # yet every iteration spent on the boundary counted as
                        # one of their misses. The ladder budget restarts here
                        # (regress4 acc_026: fatal at the third "miss" with the
                        # render tool never pinned).
                        conditional_retries = 0

            response_tool_names = [
                tool_call.name for tool_call in provider_response.tool_calls
            ]
            # §3.7 (part 3): a response that carries the call a release already
            # promised is executed before the payload reminder can throw it
            # away. The repair is asked once, so discarding that response is
            # how the action the user authorised gets lost -- the second miss
            # only holds the id. The render payload is still owed; it is
            # reminded on the next iteration as usual, with the forced-tool
            # ladder untouched.
            answers_release = self._answers_release_repair(
                collector, provider_response, repair_sent=release_repair_sent,
            )
            release_repair_sent = False
            conditional_missing = self._conditional_required_missing(
                spec, collector, ix_completed, response_tool_names, provider_response,
            )
            #: The `required == "always"` selections this response neither
            #: carries nor has already banked. Hoisted out of the tool-order
            #: gate below, which is the only consumer: a response owing one of
            #: these never reaches tool execution, so it delivers nothing
            #: whatever else it carries.
            selections_owed_first = [
                name for name in spec.required_tool_names
                if name not in ix_completed and name not in response_tool_names
            ]
            # A response can contain one valid protocol step and ordinary work
            # that is not allowed to run until a later carrier lands. Before
            # this split, the ordinary call made `ladder_delivered` false and
            # the whole response was charged and discarded. Execute only the IX
            # calls through the existing registry validator; the task calls are
            # omitted from the assistant/tool protocol and must be reissued
            # after the remaining carrier. An invalid IX call is not progress:
            # the existing rejected-argument path below charges it normally.
            deferred_task_calls: list[ToolCallRequest] = []
            deferred_owed = list(dict.fromkeys([
                *selections_owed_first,
                *conditional_missing,
            ]))
            if capability_required and IX_CAPABILITY_RESOLVE_TOOL not in deferred_owed:
                deferred_owed.append(IX_CAPABILITY_RESOLVE_TOOL)
            progress_names = {
                name for name in response_tool_names
                if name not in ix_completed
                and (
                    name in _SELECTION_TOOL_NAMES
                    or name in _RENDER_TOOL_NAMES
                    or name == IX_CAPABILITY_RESOLVE_TOOL
                )
            }
            if deferred_owed and progress_names:
                deferred_task_calls = [
                    call for call in provider_response.tool_calls
                    if not call.name.startswith(IX_TOOL_NAME_PREFIX)
                ]
                if deferred_task_calls:
                    provider_response = replace(
                        provider_response,
                        tool_calls=[
                            call for call in provider_response.tool_calls
                            if call.name.startswith(IX_TOOL_NAME_PREFIX)
                        ],
                    )
                    response_tool_names = [
                        call.name for call in provider_response.tool_calls
                    ]
                    collector.emit(
                        "ix_task_calls_deferred_for_carrier",
                        calls=[
                            {"id": call.id, "name": call.name}
                            for call in deferred_task_calls
                        ],
                        owed=deferred_owed,
                    )
            #: §3.0 — the ladder carriers this response *did* send, has not
            #: banked yet, and will actually get to run. Same wound as
            #: `answers_release`, one level up: live `pre_run_test_9` (acc_009
            #: t1) died here. A natural-selection turn owes two carriers at once
            #: -- `IX_<attribute>` (`required == "always"`) and
            #: `IX_render_<attribute>` (owed on every turn the attribute is in
            #: `session_active_ix`) -- while `tool_choice` pins exactly one tool
            #: per request. Voiding the response that answers a pin spends the
            #: pin on a call that is never executed and never banked in
            #: `ix_completed`, so once both pins are spent `_forcible_tool`
            #: returns None and the turn ends on `ix_contract_failure` although
            #: the model sent everything it was asked for. So a response
            #: delivering a still-owed carrier falls through to normal execution
            #: instead: the call runs and is banked, and the rest of the debt is
            #: reminded or pinned on the next iteration with the nudge/pin
            #: budget (`conditional_retries`, `forced_tools_used`) untouched. A
            #: delivered pin is progress, not a miss, so no
            #: `ix_render_payload_missing` is recorded against it.
            #:
            #: A mixed response reaches here only after the split above removed
            #: ordinary calls whose ordering still depends on another carrier.
            #: Calls left in a response with a newly delivered carrier are safe
            #: to execute because no later IX carrier is owed. A pinned response
            #: is IX-only by construction: `tool_choice` names one function and
            #: the provider answers with that one call.
            ladder_delivered = [] if (
                selections_owed_first
                or any(
                    not name.startswith(IX_TOOL_NAME_PREFIX)
                    for name in response_tool_names
                )
            ) else [
                name for name in response_tool_names
                if name not in ix_completed
                and (
                    name in _SELECTION_TOOL_NAMES
                    or name in _RENDER_TOOL_NAMES
                    # The boundary declaration answers its own pin (regress4
                    # acc_026: it was counted as the render payload's third
                    # miss and the turn went fatal without the render ever
                    # being pinned).
                    or name == IX_CAPABILITY_RESOLVE_TOOL
                )
            ]
            if conditional_missing and not answers_release and not ladder_delivered:
                ix_carrier_debt.update(
                    name for name in _fatal_missing([
                        *conditional_missing, *selections_owed_first,
                    ])
                )
                exhausted = _spend_ix_repair_attempt(
                    IX_CONTRACT_RENDER, "required_render_carrier"
                )
                if exhausted is not None:
                    stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                    contract_failure_kind = IX_CONTRACT_RENDER
                    contract_exhaustion = exhausted
                    final_content = None
                    await hook.after_iteration(hook_context)
                    break
                # regress19 acc_009 t8 — the reasoning this response wrote is
                # kept even though the response is not. What the pin re-buys is
                # the *selection*, and it comes back as the bare `setting` the
                # pin named: nothing asks the model to derive its evidence a
                # second time, and live it does not, so the turn banked a
                # setting chosen for no stated reason while the model's own
                # `evidence`/`application` went into the bin with the response
                # that carried them. Filed under the setting they justify, and
                # read back below only for that setting.
                for tool_call in provider_response.tool_calls:
                    arguments = (
                        tool_call.arguments
                        if isinstance(tool_call.arguments, dict) else {}
                    )
                    if (
                        tool_call.name in _SELECTION_TOOL_NAMES
                        and str(arguments.get("setting") or "")
                    ):
                        voided_selections[tool_call.name] = dict(arguments)
                conditional_retries += 1
                if conditional_retries <= 1:
                    carriers_demanded.update(conditional_missing)
                    messages.append(internal_user_message(
                        IX_PER_TURN_REMINDER_PREFIX + ", ".join(conditional_missing)
                        + ". " + NO_META_ACKNOWLEDGEMENT
                    ))
                    await hook.after_iteration(hook_context)
                    continue
                # Third attempt before failing: pin the tool with `tool_choice`
                # instead of asking again. A reminder is a suggestion the model
                # can ignore twice -- live, three of nine runs died here -- while
                # a forced choice is the provider's own guarantee. One forced
                # call per tool, so an ignored force still ends the turn. Only
                # the carriers whose absence is fatal are worth a third call: a
                # missing `IX_continue_<attribute>` keeps its record-and-deliver
                # rule (§3.7, T16) and must not cost the turn an extra request.
                # The pin has to name a carrier that is still owed *after* this
                # response is voided, and a voided response banks nothing: the
                # carriers it did send are owed too. `conditional_missing`
                # reads those off `response_tool_names` and calls them
                # satisfied, which only holds for a response that runs. Live
                # `regress19r` acc_009 t1: the model answered the nudge with
                # the selection alongside a task call, so §3.0 voided the
                # response whole, but the ladder had already counted that
                # selection as made and pinned the render instead. The
                # selection went with the discarded response, the render came
                # back under its pin and was voided in turn for owing it, the
                # second pin went on re-buying the selection, and the turn died
                # with the render never re-pinned -- although the model had
                # sent every carrier it was asked for. Selection still goes
                # first (`_fatal_missing`): a payload means nothing under no
                # setting.
                pin_order = _fatal_missing(conditional_missing)
                if pin_order:
                    pin_order = _fatal_missing([
                        *conditional_missing,
                        *dict.fromkeys(
                            name for name in response_tool_names
                            if name not in ix_completed
                        ),
                    ])
                forced = self._forcible_tool(spec, pin_order, forced_tools_used)
                if forced is not None:
                    forced_tool = forced
                    forced_tools_used.add(forced)
                    carriers_demanded.add(forced)
                    collector.emit(
                        "ix_render_payload_forced",
                        tool=forced,
                        tools=pin_order,
                        call_index=len(collector.calls),
                    )
                    messages.append(internal_user_message(
                        IX_PER_TURN_FORCED_PREFIX + forced + ". "
                        + (
                            IX_FORCED_SELECTION_FIELDS
                            if forced in _SELECTION_TOOL_NAMES else ""
                        )
                        + NO_META_ACKNOWLEDGEMENT
                    ))
                    await hook.after_iteration(hook_context)
                    continue
                # §3.0 / §3.7 (T16): "remind once, record **one**
                # `ix_render_payload_missing` on the second miss, then deliver".
                # `IX_continue_<attribute>` is not forcible and not fatal, so the
                # check re-fires on every later iteration of the same turn: live
                # (`v11s02` acc_009 t8) one owed continuation was recorded eight
                # times. The budget is already spent -- nothing is re-asked, so
                # the repeats are audit noise, not a repair loop.
                if not set(conditional_missing) <= payload_missing_reported:
                    payload_missing_reported.update(conditional_missing)
                    collector.emit(
                        "ix_render_payload_missing", tools=conditional_missing
                    )
                # A spent pin is only a spent provider hint, not proof that
                # this unresolved operation exhausted its three attempts. Keep
                # the response void and ask for the still-owed carrier without
                # re-pinning. The counter above alone publishes typed exhaustion
                # on the third actual miss. Nonfatal continuations retain their
                # existing record-and-deliver path.
                if _fatal_missing(conditional_missing):
                    messages.append(internal_user_message(
                        IX_PER_TURN_REMINDER_PREFIX + ", ".join(conditional_missing)
                        + ". " + NO_META_ACKNOWLEDGEMENT
                    ))
                    await hook.after_iteration(hook_context)
                    continue
            # A response answering the selection's pin is executed even while
            # the resolve is still owed: voiding it here wasted the selection's
            # only pin on a call that never banked, and the ladder then went
            # fatal on that carrier once the boundary was declared (regress4
            # acc_026). The resolve is reminded or pinned on the next iteration.
            if (
                capability_required
                and IX_CAPABILITY_RESOLVE_TOOL not in response_tool_names
                and not ladder_delivered
            ):
                iteration_entry["outcome"] = "capability_gate_missing"
                capability_attempts += 1
                exhausted = _spend_ix_repair_attempt(
                    IX_CONTRACT_CAPABILITY, "capability_resolve"
                )
                if exhausted is not None:
                    stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                    contract_failure_kind = IX_CONTRACT_CAPABILITY
                    contract_exhaustion = exhausted
                    collector.emit(
                        "ix_boundary_unresolved",
                        attempts=ix_repair_attempts,
                        reason="missing",
                    )
                    final_content = None
                    await hook.after_iteration(hook_context)
                    break
                if capability_attempts >= 2:
                    forced = self._forcible_tool(
                        spec, [IX_CAPABILITY_RESOLVE_TOOL], forced_tools_used
                    )
                    if forced is None and capability_attempts >= 3:
                        stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                        contract_failure_kind = IX_CONTRACT_CAPABILITY
                        contract_exhaustion = {
                            "attempts": capability_attempts,
                            "contract": IX_CONTRACT_CAPABILITY,
                            "operation": "capability_resolve",
                        }
                        collector.emit(
                            "ix_boundary_unresolved", attempts=2, reason="missing",
                        )
                        final_content = None
                        await hook.after_iteration(hook_context)
                        break
                    if forced is not None:
                        forced_tool = forced
                        forced_tools_used.add(forced)
                        collector.emit(
                            "ix_render_payload_forced", tool=forced, tools=[forced],
                            call_index=len(collector.calls),
                        )
                # A trigger-activated boundary owes two carriers on the same
                # iteration -- the setting and the declaration -- and this gate
                # runs before the selection gate below, so asking for the
                # resolve alone would spend an iteration per carrier and count
                # each one as a capability miss. Both are named at once.
                messages.append(internal_user_message(
                    IX_CAPABILITY_RESOLVE_REMINDER
                    + (
                        " Call " + ", ".join(selections_owed_first)
                        + " in the same response."
                        if selections_owed_first else ""
                    )
                ))
                await hook.after_iteration(hook_context)
                continue

            # Playability audit §4: under `Iterative` the renderer releases the
            # first `[[q:…]]` block, so two questions inside one block go out
            # together and the reply is indistinguishable from `Structured`.
            # Nudged once per turn towards one question per block, before the
            # response is used for anything -- so nothing is executed, held or
            # collected against a decomposition that is about to be resent.
            # Whatever comes back is delivered as it stands: the wrapper cannot
            # split a question the model refuses to write twice, and a second
            # undivided reply is a visible PA failure, not a wrapper failure.
            # A response still owing a selection is voided below anyway, and
            # spending the one nudge on it would lose the split repair.
            undivided = (
                self._undivided_question_block(provider_response.content)
                if question_split_nudges == 0
                and not selections_owed_first
                and ix_directive.information_elicitation == "iterative"
                else None
            )
            if undivided is not None:
                question_split_nudges = 1
                collector.emit(
                    "ix_questions_undivided",
                    attribute="information_elicitation",
                    block_id=undivided[0],
                    question_count=undivided[1],
                )
                messages.append(internal_user_message(IX_QUESTION_SPLIT_REPAIR))
                await hook.after_iteration(hook_context)
                continue

            # acc_009 t=8 — the hole beside the undivided nudge: a payload
            # declaring questions and a reply that marks none of them. The
            # renderer drops them in silence, so the ask never happens and
            # nothing on either side says so. Asked once, before the response is
            # used for anything, for the same reason the split repair is.
            unmarked = (
                self._unmarked_payload_questions(collector, provider_response)
                if question_marker_nudges < IX_QUESTION_MARKER_NUDGE_LIMIT
                and not selections_owed_first
                else 0
            )
            if unmarked:
                question_marker_nudges += 1
                iteration_entry["outcome"] = "question_marker_nudge"
                messages.append(internal_user_message(IX_QUESTION_MARKER_REPAIR))
                await hook.after_iteration(hook_context)
                continue

            # acc_011 t=27 — a rationale written about the reader rather than to
            # them. `Show` prints the field ahead of the answer, so the reply
            # opened "The user approved …" two lines above "Done! … saved to
            # your desktop". Asked once, in the same place and for the same
            # reason as the two nudges above; the renderer composes no reasons,
            # so a second work log ships and the event is the record.
            work_log = (
                rationale_work_log_nudges == 0
                and not selections_owed_first
                and reads_as_a_work_log(
                    collect_payloads(
                        provider_response.tool_calls, collector.payloads,
                    ).user_facing_rationale
                )
            )
            if work_log:
                rationale_work_log_nudges = 1
                iteration_entry["outcome"] = "rationale_work_log_nudge"
                collector.emit(
                    "ix_rationale_work_log", attribute="reasoning_visibility",
                )
                messages.append(
                    internal_user_message(IX_RATIONALE_WORK_LOG_REPAIR)
                )
                await hook.after_iteration(hook_context)
                continue

            if provider_response.has_tool_calls:
                if selections_owed_first:
                    iteration_entry["outcome"] = "void_owed_selection"
                    iteration_entry["selections_owed"] = list(selections_owed_first)
                    exhausted = _spend_ix_repair_attempt(
                        IX_CONTRACT_SELECTION, "required_selection_carrier"
                    )
                    if exhausted is not None:
                        stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                        contract_failure_kind = IX_CONTRACT_SELECTION
                        contract_exhaustion = exhausted
                        final_content = None
                        hook_context.stop_reason = stop_reason
                        await hook.after_iteration(hook_context)
                        break
                    # A pinned `tool_choice` carries one function, so a response
                    # answering a forced `IX_capability_resolve` can never also
                    # carry the still-owed `IX_<attribute>` selection. Discarding
                    # it whole re-arms the capability gate on the next iteration
                    # with no pin left (`_forcible_tool` returns None) and the
                    # turn ends on `ix_boundary_unresolved` although the model
                    # sent the payload it was asked for -- live `regress1` /
                    # `regress2` acc_026, three of four runs. The resolve payload
                    # is banked here; the rest of the response is still resent.
                    resolve_calls = [
                        tool_call for tool_call in provider_response.tool_calls
                        if tool_call.name == IX_CAPABILITY_RESOLVE_TOOL
                    ]
                    if resolve_calls and capability_required:
                        collector.add_payloads(resolve_calls)
                        if not boundary_payload_invalid(collector.payloads):
                            capability_required = False
                            iteration_entry["outcome"] = "void_owed_selection_resolve_banked"
                        else:
                            iteration_entry["outcome"] = "void_owed_selection_resolve_invalid"
                            # Same budget as the main gate: an illegal payload is
                            # one attempt, and the reminder names what is owed.
                            capability_attempts += 1
                            exhausted = _spend_ix_repair_attempt(
                                IX_CONTRACT_CAPABILITY, "capability_resolve"
                            )
                            if exhausted is not None:
                                stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                                contract_failure_kind = IX_CONTRACT_CAPABILITY
                                contract_exhaustion = exhausted
                                collector.emit(
                                    "ix_boundary_unresolved",
                                    attempts=ix_repair_attempts,
                                    reason="invalid_payload",
                                )
                                final_content = None
                                await hook.after_iteration(hook_context)
                                break
                            if capability_attempts >= 3:
                                stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                                contract_failure_kind = IX_CONTRACT_CAPABILITY
                                contract_exhaustion = {
                                    "attempts": capability_attempts,
                                    "contract": IX_CONTRACT_CAPABILITY,
                                    "operation": "capability_resolve",
                                }
                                collector.emit(
                                    "ix_boundary_unresolved",
                                    attempts=capability_attempts,
                                    reason="invalid_payload",
                                )
                                final_content = None
                                await hook.after_iteration(hook_context)
                                break
                            messages.append(
                                internal_user_message(IX_CAPABILITY_RESOLVE_REMINDER)
                            )
                    messages.append(
                        internal_user_message(
                            IX_TOOL_ORDER_REMINDER_PREFIX
                            + ", ".join(selections_owed_first)
                            + ". Choose the setting, give brief evidence, and state how "
                            "you will apply it. " + NO_WRAPPER_TALK
                        )
                    )
                    await hook.after_iteration(hook_context)
                    continue

                if hook.wants_streaming():
                    await hook.on_stream_end(hook_context, resuming=True)

                # One decision, one selection event. Filter duplicate selection
                # declarations before writing the assistant protocol message as
                # well as before execution; otherwise a banked selection that is
                # re-sent beside a later carrier has a declaration with no tool
                # result and makes runtime_history malformed.
                executed_calls, ignored_selections = self._dedupe_ix_selections(
                    provider_response.tool_calls, ix_completed
                )
                self._restore_voided_selection_fields(
                    executed_calls, voided_selections,
                )
                for dropped in ignored_selections:
                    arguments = (
                        dropped.arguments
                        if isinstance(dropped.arguments, dict) else {}
                    )
                    collector.emit(
                        "ix_selection_duplicate",
                        tool=dropped.name,
                        ignored_setting=str(arguments.get("setting") or ""),
                    )
                    ignored_ix_selections.append({
                        "id": dropped.id,
                        "name": dropped.name,
                        "args": {
                            key: value for key, value in arguments.items()
                            if key not in PER_CALL_INJECTED_FIELDS
                        },
                        "outcome": "ignored_duplicate",
                    })
                collector.add_segment(provider_response.content)
                messages.append(build_assistant_message(
                    provider_response.content or "",
                    tool_calls=[
                        tool_call.to_openai_tool_call()
                        for tool_call in executed_calls
                    ],
                    reasoning_content=provider_response.reasoning_content,
                    thinking_blocks=provider_response.thinking_blocks,
                ))
                tools_used.extend(
                    tool_call.name for tool_call in executed_calls
                )
                collector.add_payloads(executed_calls)

                # The IX-only carrier allowed through the earlier ladder gate
                # must reach execution and be banked. A selection-only response
                # does not submit an invalid resolve: that separate declaration
                # is still owed on the next request. Discarding the selection
                # here charged a miss for answering its pin, then charged another
                # when the valid resolve arrived without the discarded selection.
                # A submitted-but-invalid resolve, or any business-tool mixture,
                # still follows the existing bounded failure path.
                if capability_required and not (
                    ladder_delivered
                    and IX_CAPABILITY_RESOLVE_TOOL not in response_tool_names
                ):
                    capability_required = False
                    if boundary_payload_invalid(collector.payloads):
                        capability_attempts += 1
                        exhausted = _spend_ix_repair_attempt(
                            IX_CONTRACT_CAPABILITY, "capability_resolve"
                        )
                        if exhausted is not None:
                            stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                            contract_failure_kind = IX_CONTRACT_CAPABILITY
                            contract_exhaustion = exhausted
                            collector.emit(
                                "ix_boundary_unresolved",
                                attempts=ix_repair_attempts,
                                reason="invalid_payload",
                            )
                            final_content = None
                            await hook.after_iteration(hook_context)
                            break
                        if capability_attempts >= 3:
                            stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                            contract_failure_kind = IX_CONTRACT_CAPABILITY
                            contract_exhaustion = {
                                "attempts": capability_attempts,
                                "contract": IX_CONTRACT_CAPABILITY,
                                "operation": "capability_resolve",
                            }
                            collector.emit(
                                "ix_boundary_unresolved",
                                attempts=capability_attempts,
                                reason="invalid_payload",
                            )
                            final_content = None
                            await hook.after_iteration(hook_context)
                            break
                        capability_required = True
                        messages.append(
                            internal_user_message(IX_CAPABILITY_RESOLVE_REMINDER)
                        )
                        await hook.after_iteration(hook_context)
                        continue

                iteration_entry["outcome"] = iteration_entry["outcome"] or "executed"
                await hook.before_execute_tools(hook_context)

                collector.repair = ""
                collector.batch_repair = ""
                results, new_events = await self._execute_tools(
                    spec,
                    executed_calls,
                    ix_directive,
                    collector,
                    tool_events,
                )
                tool_events.extend(new_events)
                ix_completed.update(
                    str(event.get("name") or "")
                    for event in new_events
                    if event.get("ix")
                )
                ix_carrier_repaired.update(
                    str(event.get("name") or "")
                    for event in new_events
                    if event.get("ix") and event.get("name") in ix_carrier_debt
                )
                ix_directive = self._directive_for_events(spec, tool_events)
                for tool_call, result in zip(executed_calls, results):
                    if tool_call.id in collector.skipped_call_ids:
                        continue
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    })
                if deferred_task_calls:
                    messages.append(internal_user_message(
                        IX_TASK_CALLS_DEFERRED_FOR_CARRIER.format(
                            tools=", ".join(deferred_owed),
                        )
                    ))
                # regress12 acc_004 — after the strike limit the tool is out for
                # the turn, and the model has to be told what to do instead:
                # ask, rather than guess again. Once per struck tool per turn,
                # and after the tool results so the protocol stays well formed.
                newly_struck = sorted(_struck_tools(collector) - strike_prompted)
                if newly_struck:
                    strike_prompted.update(newly_struck)
                    # gpt acc_082 — "ask the user for the one it needs" is the
                    # wrong instruction for a parameter the schema refuses: the
                    # user had already said to put the evening on the calendar
                    # and was never going to be asked about a recurrence rule.
                    # A tool with a correction owed on it gets that correction
                    # named instead; every other struck tool keeps the ask.
                    repairable = {
                        item_id: repair
                        for item_id, repair in collector.recurrence_repairs.items()
                        if not repair.spent and repair.tool in set(newly_struck)
                    }
                    generic = [
                        name for name in newly_struck
                        if name not in {
                            repair.tool for repair in repairable.values()
                        }
                    ]
                    if generic:
                        messages.append(internal_user_message(
                            TOOL_FAILURE_STRIKE_PROMPT.format(
                                tools=", ".join(generic),
                                limit=TOOL_FAILURE_STRIKE_LIMIT,
                            )
                        ))
                    for item_id, repair in sorted(repairable.items()):
                        messages.append(internal_user_message(
                            TOOL_FAILURE_STRIKE_RECURRENCE_PROMPT.format(
                                tool=repair.tool,
                                limit=TOOL_FAILURE_STRIKE_LIMIT,
                                item_id=item_id,
                            )
                        ))
                # hy4 table row 5 — a carrier this turn asked for has landed, so
                # the reminder and the pin still sitting in `messages` are
                # answered. Said here, after the tool results, for the same
                # reason the strike prompt is.
                for name in sorted(
                    (carriers_demanded & ix_completed)
                    - carriers_recorded_note_sent
                ):
                    carriers_recorded_note_sent.add(name)
                    messages.append(internal_user_message(
                        IX_CARRIER_RECORDED_NOTE.format(tool=name)
                    ))
                # Owner ruling 2026-09-09 — the setting the model just chose
                # shapes *this* turn's reply. regress21 `acc_002` t5 selected
                # `solution_breadth: Low` at call 0 and still delivered a
                # five-row table; the Low surface arrived on t8, because §3.6's
                # restatement is built from `inherited_ix_selections`, which the
                # loop writes only after the run. The message slot is the same
                # one the next turn uses (`spec.turn_reminder`) and the text is
                # the same builder's output, so nothing new is said to the
                # model — it is said one turn earlier. Empty, and therefore
                # absent, for every attribute §3.6 does not restate: those are
                # the ones the wrapper enforces at render or gate time, and
                # both already read this turn's events.
                fresh = [
                    selection
                    for selection in selections_from_tool_events(new_events)
                    if selection.attribute not in restated_in_turn
                ]
                if (restatement := turn_reminder_message(fresh)):
                    restated_in_turn.update(
                        selection.attribute for selection in fresh
                    )
                    messages.append(internal_user_message(restatement))
                hook_context.tool_results = list(results)
                hook_context.tool_events = list(new_events)

                # §3.0 / §7.1: an IX call the registry refused is owed again,
                # exactly like a missing payload -- so a model that keeps
                # re-sending the same invalid argument ends the turn instead of
                # spending every iteration on it. The refused result is already
                # in `messages` above, so the model sees what was wrong with it.
                # The response's task calls were withheld in
                # `_execute_tools_ix_first`; they are reissued with the
                # corrected selection or not at all.
                #
                # hy4 `acc_031`: the budget is this repair's own, not the
                # missing-carrier ladder's. Sharing it meant a reminded and
                # pinned turn had nothing left for the first argument mistake.
                rejected = self._rejected_ix_tools(new_events)
                if rejected:
                    ix_argument_retries += 1
                    # Which of the two refusals this was. The beat's own
                    # `session_active_ix` cannot answer it: regress20 `acc_027`
                    # carried `emotional_engagement` there while the reaction
                    # beat offered no IX tool at all. This turn's tool view is
                    # what the registry dispatched against, so a rejected name
                    # inside it was offered and refused on its arguments
                    # (regress23m `acc_001`, `IX_verbosity` called with `{}`),
                    # and a name outside it was never offered.
                    offered_now = set(spec.tools.tool_names)
                    all_offered = all(name in offered_now for name in rejected)
                    # hy4 `acc_031`: "was it offered" alone cannot tell the two
                    # tiers apart, so an offered *render* tool refused on its
                    # own arguments wore the selection cause. The tool names
                    # say which tier it was; an unoffered name keeps the
                    # not-offered cause whichever tier it belongs to, because
                    # there the refusal is about the beat's tool list and not
                    # about the arguments at all.
                    stage = _ix_refusal_stage(rejected)
                    cause = (
                        IX_CONTRACT_RENDER_ARGS
                        if all_offered and stage == "render"
                        else IX_CONTRACT_SELECTION_ARGS if all_offered
                        else IX_CONTRACT_SELECTION
                    )
                    exhausted = _spend_ix_repair_attempt(
                        cause, "ix_tool_arguments"
                    )
                    # regress20 acc_027 recorded the refused tool and nothing
                    # else, so the record could not say *why* the registry
                    # refused it. The beat's own attribute list is the answer
                    # in the common case -- there the reaction beat offered
                    # none at all and the PA kept asking for
                    # `IX_emotional_engagement` -- and the attempt number says
                    # which of the three refusals ended the turn. hy4 adds the
                    # tier, the fields the call left out and the budget the
                    # attempt was charged to: under `render` the reader needs to
                    # see `completion_message`, not a setting.
                    definitions = spec.tools.get_definitions()
                    missing_fields = list(dict.fromkeys(
                        field
                        for event in new_events
                        if str(event.get("name") or "") in set(rejected)
                        for field in _missing_required_fields(
                            definitions,
                            str(event.get("name") or ""),
                            event.get("args"),
                        )
                    ))
                    collector.emit(
                        "ix_selection_invalid",
                        tools=rejected,
                        attempt=ix_argument_retries,
                        cause=cause,
                        stage=stage,
                        missing_fields=missing_fields,
                        budget="ix_argument_retries",
                        turn_index=spec.turn_index,
                        session_active_ix=list(
                            spec.ix_context.get("session_active_ix") or ()
                        ),
                    )
                    if exhausted is not None:
                        stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                        contract_failure_kind = cause
                        contract_exhaustion = exhausted
                        final_content = None
                        hook_context.stop_reason = stop_reason
                        await hook.after_iteration(hook_context)
                        break
                    # Same trade as the batch repair below, for the same reason:
                    # nothing ran, so the refused attempt's hold records go with
                    # it and the retry starts clean. Live regress18 `acc_001` t6
                    # kept them: the reissued `document_update`s executed --
                    # `state_diff.jsonl` and `canonical_stage` carry both -- while
                    # `held.calls` still reported them `held_for_binding` and
                    # `ix_claims_unexecuted_action` fired on the very turn that
                    # carried them out. The refusal itself stays on the record as
                    # the `ix_selection_invalid` event above.
                    collector.clear_holds()
                    # An unavailable tool cannot satisfy a payload demand.
                    # Keep argument repairs only for tools still offered here.
                    offered_rejected = [name for name in rejected if name in offered_now]
                    unavailable = [name for name in rejected if name not in offered_now]
                    carriers_demanded.update(offered_rejected)
                    repair = (
                        IX_UNAVAILABLE_TOOL_REPAIR.format(tools=", ".join(unavailable))
                        if unavailable else ""
                    )
                    if offered_rejected:
                        repair += IX_PER_TURN_REMINDER_PREFIX + ", ".join(offered_rejected) + ". "
                    messages.append(internal_user_message(repair + NO_META_ACKNOWLEDGEMENT))
                    await hook.after_iteration(hook_context)
                    continue

                # Owner ruling 2026-09-08 — the max-steps backstop. Counted here,
                # where the results are in hand, so an iteration that ends on any
                # of the repairs below still counts; *acted on* at the very
                # bottom of this branch, so every rule of the ladder -- the
                # boundary declaration, the gate, the hold -- decides the turn
                # first and the cap only replaces the final `continue` (review
                # items 1 and 3, 2026-09-09).
                #
                # A `capability_limit` is excluded from the count for the same
                # reason the strike excludes it: §3.5 owns the blocked-capability
                # case, and its two-stage repair is exactly the re-prompting this
                # would cut short. The recurrence rebind instruction (gpt
                # acc_082 §4) is excluded for exactly that reason too: it is the
                # wrapper telling the model which id to use, not a tool coming
                # back an error, and counting it spent the one iteration the
                # re-bound call needed. The still-illegal retry is *not*
                # excluded -- that one really is another parameter failure, and
                # the cap is what stops it going round again.
                failing_now = {
                    tool_call.name
                    for tool_call, result in zip(executed_calls, results)
                    if tool_call.id not in collector.skipped_call_ids
                    and _tool_result_error(result) is not None
                    and not is_capability_limit(result)
                    and not (
                        isinstance(result, str)
                        and result.startswith(RECURRENCE_REBIND_PREFIX)
                    )
                }
                # The wrapper's own half of the same loop: a call it held again,
                # unchanged, for a reason that asks the user nothing. Counted
                # per iteration like the failures, and from the *second* hold of
                # a signature, so the first refusal is free.
                held_cursor = min(held_cursor, len(collector.held))
                new_holds = _no_ask_hold_signatures(collector, held_cursor)
                held_cursor = len(collector.held)
                held_again = set()
                for signature in sorted(new_holds):
                    held_repeats[signature] = held_repeats.get(signature, 0) + 1
                    if held_repeats[signature] > 1:
                        held_again.add(signature[0])
                failing_now |= held_again
                for name in failing_now:
                    retry_failures[name] = retry_failures.get(name, 0) + 1
                capped = next(
                    (
                        name for name in sorted(failing_now)
                        if retry_failures[name] >= TOOL_RETRY_CAP
                    ),
                    None,
                )

                if collector.batch_repair:
                    # §3.4 (part 3): asked once, and only once. Nothing ran, so the
                    # refused attempt's hold records go with it -- the retry
                    # resends the same calls under a new decomposition.
                    repair = collector.batch_repair
                    collector.batch_repair = ""
                    collector.clear_holds()
                    messages.append(internal_user_message(repair))
                    await hook.after_iteration(hook_context)
                    continue

                if collector.repair:
                    binding_retries += 1
                    if binding_retries > 1:
                        collector.emit("ix_binding_invalid", attempts=2)
                        # Round 9 item 2: only *this* exit owes the user the
                        # close line. A first refusal is repaired inside the
                        # turn, and its hold rows stay on the audit plane
                        # afterwards -- scanning them would tell the user the
                        # steps failed on a turn where the retry ran them.
                        collector.binding_refused = True
                        # §3.4 (part 5) applies here too. A binding refusal takes the
                        # whole turn's batch, so `last_segment` is whatever the
                        # refused response happened to write -- live `v7r06` t18
                        # that was nothing at all, and the loop's old placeholder
                        # sentence was the entire reply. The closing iteration
                        # was only ever run on the batch / question path; the
                        # fail-closed path needs it more, because there is no
                        # ask and no released section to carry the turn.
                        await self._closing_iteration(
                            spec, hook, hook_context, messages,
                            provider_response, collector,
                            held_result=IX_BINDING_HELD_TOOL_RESULT,
                        )
                        final_content = collector.last_segment
                        await hook.after_iteration(hook_context)
                        break
                    binding_refusal_pending = True
                    messages.append(internal_user_message(collector.repair))
                    await hook.after_iteration(hook_context)
                    continue

                # Past the refusal check with the resent batch executed: the
                # refusal is history, not an outstanding hold. Live `acc_009`
                # scanned those rows and told the user the steps could not be
                # carried out on the turn whose retry carried them out.
                if binding_refusal_pending:
                    repaired_by = next(
                        (
                            call.tool_call_id
                            for call in reversed(collector.calls)
                            if call.executed
                            and not call.tool_name.startswith(IX_TOOL_NAME_PREFIX)
                        ),
                        "",
                    )
                    if repaired_by:
                        collector.mark_binding_repaired(repaired_by)
                        binding_refusal_pending = False

                # Reaching here proves schema and binding repairs accepted this
                # response. Resolve only the recorded carrier debt before a new
                # gate consumes its first attempt. The debt is then cleared, so
                # repeated/unrelated IX calls cannot reset an unresolved gate.
                if ix_carrier_debt and ix_carrier_debt <= ix_carrier_repaired:
                    _reset_ix_repair_attempts()

                if collector.capability_limit_seen and not collector.boundary_declared:
                    # Declare the boundary against the real result, not ahead of
                    # it: the tool view is rebuilt so the resolve tool exists on
                    # the next iteration and is forced there.
                    capability_required = True
                    # The same rebuild carries the activation (audit §2,
                    # acc_026). `session_active_ix` gains `capability_boundary`
                    # so the rest of the turn injects, gates and renders under
                    # it, and the factory puts `IX_capability_boundary` on the
                    # turn when the session has not chosen a setting yet --
                    # `required_tool_names` is re-read off the new view rather
                    # than second-guessing whether the selection is still owed.
                    spec.ix_context = _with_capability_boundary(spec.ix_context)
                    if spec.tools_factory is not None:
                        spec.tools = spec.tools_factory({
                            **spec.ix_context,
                            "capability_limit_seen": True,
                            "hold": spec.hold,
                        })
                        spec.required_tool_names = list(
                            spec.tools.required_ix_tool_names
                        )

                gate = self._capability_handoff_gate(collector, ix_directive)
                if gate is not None:
                    collector.gate = gate
                if collector.gate is not None:
                    ask = self._gate_message(collector)
                    if not ask and collector.gate.attribute in ASK_BEARING_ATTRIBUTES:
                        # §3.3 (part 2)·C: the renderer never invents the ask, so a
                        # gate without one asks the user nothing and ships the
                        # PA's own prose instead -- observed live twice, with
                        # the whole turn reading `Let me set up the plan`. One
                        # repair, then fail closed like the render payloads.
                        collector.emit(
                            "ix_gate_no_message",
                            attribute=collector.gate.attribute,
                            item_id=collector.gate.item_id,
                        )
                        gate_message_retries += 1
                        exhausted = _spend_ix_repair_attempt(
                            IX_CONTRACT_GATE_MESSAGE,
                            "gate_confirmation_message",
                        )
                        if exhausted is None:
                            collector.clear_holds()
                            messages.append(internal_user_message(
                                IX_GATE_MESSAGE_REPAIR.format(
                                    field=IX_CONFIRMATION_MESSAGE_FIELD
                                )
                            ))
                            await hook.after_iteration(hook_context)
                            continue
                        stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                        contract_failure_kind = IX_CONTRACT_GATE_MESSAGE
                        contract_exhaustion = exhausted
                        final_content = None
                        hook_context.stop_reason = stop_reason
                        await hook.after_iteration(hook_context)
                        break
                    stop_reason = STOP_REASON_IX_GATE
                    final_content = ask
                    collector.emit(
                        "beat_gate_turn",
                        attribute=collector.gate.attribute,
                        item_id=collector.gate.item_id,
                        reason=collector.gate.reason,
                    )
                    hook_context.final_content = final_content
                    hook_context.stop_reason = stop_reason
                    await hook.after_iteration(hook_context)
                    break

                # §3.4 (part 3): a batch or question hold ends the turn's *work* the
                # way a gate does -- no task tool runs after this point. What
                # ships is the released batches or questions and whatever frame
                # the payloads already in hand let the renderer build. Those
                # payloads are guaranteed present: the response that produced
                # the hold has already passed the conditional required check
                # above, and a batch hold cannot exist without the batch payload
                # it was decided against. The episode stays open, and
                # `IX_continue_<attribute>` resumes it next turn.
                if any(
                    item.status in _TURN_ENDING_HOLD_STATUSES
                    for item in collector.held
                ):
                    stop_reason = STOP_REASON_IX_HOLD
                    # One closing iteration first (§3.4 (part 5)): the prose may still
                    # be owed. It adds a segment or it does not, and either way
                    # `last_segment` is the same fallback as before.
                    await self._closing_iteration(
                        spec, hook, hook_context, messages,
                        provider_response, collector,
                    )
                    final_content = collector.last_segment
                    hook_context.final_content = final_content
                    hook_context.stop_reason = stop_reason
                    await hook.after_iteration(hook_context)
                    break

                # §3.7 (part 3): a released id whose call never arrived. Ask once for
                # the call the release promised -- nothing is replayed for the
                # model, so an unanswered release is simply an action the user
                # authorised and never got.
                missing_releases = (
                    self._releases_without_calls(spec, collector)
                    if release_repairs == 0 else []
                )
                if missing_releases:
                    release_repairs = 1
                    release_repair_sent = True
                    messages.append(internal_user_message(
                        IX_RELEASE_REPAIR.format(
                            ids=", ".join(item_id for item_id, _a, _t in missing_releases)
                        )
                    ))
                    await hook.after_iteration(hook_context)
                    continue

                # The max-steps backstop, in place of the final `continue`: every
                # rule above has declined to end the turn, so the only thing left
                # is another request, and the tool it would carry has said
                # everything it is going to say. One exception (review item 1): a
                # `capability_limit` seen this turn owes §3.5's declaration, and
                # that is exactly the re-prompting the cap would cut short --
                # the turn would ship `completed` with the block unresolved and
                # no `ix_boundary_unresolved` on the record.
                if capped is not None and not (
                    collector.capability_limit_seen and not collector.boundary_declared
                ):
                    collector.emit(
                        "tool_retry_cap_reached",
                        tool=capped,
                        count=retry_failures[capped],
                        turn_index=spec.turn_index,
                        beat=str(spec.ix_context.get("beat") or ""),
                    )
                    iteration_entry["outcome"] = "tool_retry_cap_reached"
                    # Review item 2: what ships here is a user-facing message, so
                    # the turn owes its render payloads exactly as a text reply
                    # does. The response that reached the cap carried tool calls,
                    # so the predicate's own "visible message" test says no --
                    # it is overridden, because this exit is the delivery.
                    capped_missing = self._conditional_required_missing(
                        spec, collector, ix_completed,
                        response_tool_names, provider_response,
                        produces_user_visible_message=True,
                    )
                    if not set(capped_missing) <= payload_missing_reported:
                        payload_missing_reported.update(capped_missing)
                        collector.emit(
                            "ix_render_payload_missing", tools=capped_missing
                        )
                    if _fatal_missing(capped_missing):
                        # §3.0's own consequence: unrenderable is unrenderable,
                        # however the turn arrived at its last message.
                        stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
                        contract_failure_kind = IX_CONTRACT_RENDER
                        final_content = None
                    else:
                        final_content = collector.last_segment or None
                    hook_context.final_content = final_content
                    hook_context.stop_reason = stop_reason
                    await hook.after_iteration(hook_context)
                    break

                # This response survived every repair and gate check. A later
                # request therefore starts a new IX operation and must not
                # inherit misses from the operation just resolved. In
                # particular, do not clear this before `ix_gate_no_message`
                # above: the reissued held call is still the same unresolved
                # confirmation operation.
                # A successful business call or deduplicated selection does
                # not repair an outstanding carrier. Only the fresh accepted
                # events checked above can close that concrete debt.
                if ix_repair_attempts and not ix_carrier_debt:
                    _reset_ix_repair_attempts()
                await hook.after_iteration(hook_context)
                continue

            if hook.wants_streaming():
                await hook.on_stream_end(hook_context, resuming=False)

            clean = hook.finalize_content(hook_context, provider_response.content)

            missing_required = [
                name for name in spec.required_tool_names if name not in ix_completed
            ]
            if missing_required:
                messages.append(
                    internal_user_message(
                        "Before producing the final answer for this turn, call the required "
                        "interaction preference tool(s): "
                        + ", ".join(missing_required)
                        + ". Choose the setting, give brief evidence, and state how "
                        "you will apply it. " + NO_WRAPPER_TALK
                    )
                )
                await hook.after_iteration(hook_context)
                continue

            # acc_039: the ask the wrapper would have delivered, written as prose
            # instead. Nudged once, then whatever comes back is delivered.
            if prose_ask_nudges == 0 and self._prose_ask_unbound(
                spec, collector, ix_directive, clean,
            ):
                prose_ask_nudges = 1
                collector.emit(
                    "ix_prose_ask_unbound",
                    attribute="autonomy_level",
                    # `collector.settings` is filled by the loop after the run
                    # returns, so the label is read off the selections here.
                    setting=settings_from_selections([
                        *spec.inherited_ix_selections,
                        *selections_from_tool_events(tool_events),
                    ]).get("autonomy_level", ""),
                )
                messages.append(internal_user_message(IX_PROSE_ASK_REPAIR))
                await hook.after_iteration(hook_context)
                continue

            collector.add_segment(clean)
            # `clean or ""`, never `None`: a reply with no text at all puts an
            # assistant message with a null `content` and no `tool_calls` on the
            # history, and the next request built from it is rejected outright
            # -- regress12 acc_003 t=12, "missing `messages.content` parameter",
            # raised by the closing request the empty-reply rule sends straight
            # after this. An empty string is what `_sanitize_empty_content`
            # already knows how to send.
            messages.append(build_assistant_message(
                clean or "",
                reasoning_content=provider_response.reasoning_content,
                thinking_blocks=provider_response.thinking_blocks,
            ))
            # regress13 acc_035 — after `add_segment`, because §5.3 carrier F
            # says the trailing block replaces the tool payload and this is the
            # call that reads it; and after the assistant message, so the re-ask
            # goes out with the reply it is about on the history. `last_segment`
            # is the same delivery surface the renderer locates in, minus the
            # released-section selection -- which only ever *shrinks* the text,
            # so a claim that misses here misses there too and this cannot
            # re-ask a turn the renderer would have rendered.
            all_dropped = self._certainties_all_dropped(
                collector,
                settings_from_selections([
                    *spec.inherited_ix_selections,
                    *selections_from_tool_events(tool_events),
                ]),
            )
            if all_dropped:
                certainty_claim_retries += 1
                if certainty_claim_retries <= 1:
                    collector.emit(
                        "ix_certainties_all_dropped",
                        attribute="uncertainty_expression",
                        declared=all_dropped,
                        attempt=certainty_claim_retries,
                    )
                    iteration_entry["outcome"] = "certainty_claim_repair"
                    messages.append(
                        internal_user_message(IX_CERTAINTY_CLAIM_REPAIR)
                    )
                    await hook.after_iteration(hook_context)
                    continue
                # Owner ruling 4 (2026-09-08): the retry is spent and the turn
                # ships anyway, untagged. Every claim missed the prose, so the
                # renderer locates none of them and appends no parenthesis --
                # the reply reaches the user with no confidence on it at all,
                # which is the fact the judge needs and this event carries.
                # Losing the whole step over it costs a run to buy nothing:
                # nothing is appended to the prose either way, because a
                # certainty the model did not write into its reply is not the
                # wrapper's sentence to add.
                collector.emit(
                    "ix_certainties_all_dropped",
                    attribute="uncertainty_expression",
                    declared=all_dropped,
                    attempt=certainty_claim_retries,
                    detail="certainties contract",
                    continued=True,
                )
            # §5.3 carrier F: the renderer strips the trailing `<certainties>`
            # block out of the prose it delivers, but `final_content` is the
            # fallback the loop ships when the renderer produced no message at
            # all -- so the block has to come off here too, or the one path
            # that bypasses the renderer is the one that leaks it.
            final_content = strip_certainty_block(clean) if clean else clean
            hook_context.final_content = final_content
            hook_context.stop_reason = stop_reason
            await hook.after_iteration(hook_context)
            break
        else:
            stop_reason = "max_iterations"
            template = spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE
            final_content = template.format(max_iterations=spec.max_iterations)

        # The single exit, so the give-up runs once however the turn ended --
        # at the gate, at the final text, or out of iterations. The truncation
        # rule goes first because it decides what text there *is*: the three
        # below read the turn's prose, and on a cut-off turn some of that prose
        # was never finished. It spends no request of its own, so putting it
        # ahead of them costs nothing and keeps them off a stale fragment.
        final_content = self._resolve_truncated_turn(final_content, collector, stop_reason)
        # Then the missing notice: it is the one rule here that can fail the
        # turn, and the two below skip a failed one. Then the unfinished turn:
        # it has text, so the empty-reply rule would pass it straight through,
        # and both spend the same one extra request.
        stop_reason_before_notice = stop_reason
        final_content, stop_reason = await self._resolve_missing_notice(
            spec, hook, hook_context, messages, collector, tool_events,
            final_content, stop_reason,
        )
        if stop_reason != stop_reason_before_notice:
            # The notice rule is the only one after the loop that can fail a
            # turn, and it returns the stop reason it was given in every other
            # case -- so a change here is that failure and nothing else.
            contract_failure_kind = IX_CONTRACT_NOTICE
            contract_exhaustion = {
                "attempts": 1 + IX_MISSING_NOTICE_ATTEMPTS,
                "contract": IX_CONTRACT_NOTICE,
                "operation": "process_visibility_notice",
            }
        final_content = await self._resolve_unfinished_turn(
            spec, hook, hook_context, messages, collector, final_content, stop_reason,
        )
        final_content = await self._resolve_empty_reply(
            spec, hook, hook_context, messages, collector, tool_events,
            final_content, stop_reason,
        )
        self._keep_unexecuted_releases_held(spec, collector)
        self._expire_preauthorisations(spec, collector)
        tool_events.extend(collector.events)
        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            iteration_log=iteration_log,
            ignored_ix_selections=ignored_ix_selections,
            render_input=collector.to_render_input(
                hold=spec.hold,
                consent_decisions=spec.consent_decisions,
            ),
            collector=collector,
            ix_contract_failure_kind=contract_failure_kind,
            ix_contract_exhaustion=contract_exhaustion,
        )

    @staticmethod
    def _conditional_required_missing(
        spec: AgentRunSpec,
        collector: TurnCollector,
        ix_completed: set[str],
        response_tool_names: list[str],
        response: Any,
        produces_user_visible_message: bool | None = None,
    ) -> list[str]:
        """Which conditionally-required IX tools this response still owes.

        The predicate is evaluated *after* the response, because that is the
        only point where "would this turn be renderable without it" can be
        answered. `IX_capability_resolve` is excluded: §3.5 (part 6) gives it its own
        two-attempt contract-failure rule rather than the generic one.

        "Produces a user-visible message" normally means "this response called
        no tool", which is the same thing on every path that reaches the user
        through the model's next reply. The retry cap is the one exit where it
        is not: it ends the turn on a response that *did* call tools and ships
        that response's prose, so it passes the answer in rather than letting it
        be derived (review item 2, 2026-09-09).
        """
        evaluate = getattr(spec.tools, "required_ix_tool_names_for", None)
        if evaluate is None:
            return []
        text = " ".join([*collector.segments, response.content or ""]).strip()
        markers = (
            *(f"batch:{item}" for item in BATCH_MARKER_RE.findall(text)),
            *(f"q:{item}" for item in QUESTION_MARKER_RE.findall(text)),
            *(f"c:{item}" for item in CERTAINTY_MARKER_RE.findall(text)),
        )
        item_ids = tuple(
            str(tool_call.arguments.get(INTERACTION_ITEM_ID_FIELD))
            for tool_call in response.tool_calls
            if isinstance(tool_call.arguments, dict)
            and tool_call.arguments.get(INTERACTION_ITEM_ID_FIELD)
        )
        required = evaluate(RequiredContext(
            session_active_ix=tuple(spec.ix_context.get("session_active_ix") or ()),
            hold=spec.hold,
            response_tool_names=tuple(response_tool_names),
            response_interaction_item_ids=item_ids,
            produces_user_visible_message=(
                (not response.has_tool_calls)
                if produces_user_visible_message is None
                else produces_user_visible_message
            ),
            capability_limit_seen=collector.capability_limit_seen,
            response_text=text,
            response_markers=markers,
        ))
        return [
            name for name in required
            if name != IX_CAPABILITY_RESOLVE_TOOL
            and name not in response_tool_names
            and name not in ix_completed
        ]

    @staticmethod
    def _rejected_ix_tools(events: Sequence[dict[str, Any]]) -> list[str]:
        """The IX tools this response called and the registry refused (§3.0).

        An IX result carries `ix` only once it parsed into a payload, so a
        setting the enum does not allow lands here rather than in the directive
        -- which would otherwise be rebuilt empty and gate nothing.
        """
        return [
            str(event.get("name") or "")
            for event in events
            if str(event.get("name") or "").startswith(IX_TOOL_NAME_PREFIX)
            and not event.get("ix")
        ]

    @staticmethod
    def _answers_release_repair(
        collector: TurnCollector, response: Any, *, repair_sent: bool
    ) -> bool:
        """§3.7 (part 3) — does this response carry a task call a release promised?

        True for a call bound to an id already released this turn, and for any
        task call in the response answering the one release repair: the model
        may re-bind the call there, and that response is still the last chance
        the released id gets.
        """
        task_calls = [
            tool_call for tool_call in response.tool_calls
            if not tool_call.name.startswith(IX_TOOL_NAME_PREFIX)
        ]
        if not task_calls:
            return False
        if repair_sent:
            return True
        released = {
            decision.id
            for decision in collector.payloads.release_decisions
            if decision.decision == "release"
        }
        return any(
            isinstance(tool_call.arguments, dict)
            and str(tool_call.arguments.get(INTERACTION_ITEM_ID_FIELD) or "") in released
            for tool_call in task_calls
        )

    def _forcible_tool(
        self,
        spec: AgentRunSpec,
        missing: Sequence[str],
        already_forced: set[str],
    ) -> str | None:
        """The tool a third attempt may pin with `tool_choice`, or None.

        None means no forcing hint is available: the model policy does not
        support a forced choice (`configs/model_request_policies.yaml`), the
        tool is not in this turn's tool view, or it was already forced once.
        This is not evidence that the operation's repair budget is exhausted.
        """
        supports = getattr(self.provider, "supports_forced_tool_choice", None)
        if supports is None or not supports(spec.model):
            return None
        offered = {
            definition.get("function", {}).get("name")
            for definition in spec.tools.get_definitions()
        }
        return next(
            (
                name for name in missing
                if name not in already_forced and name in offered
            ),
            None,
        )

    @staticmethod
    def _prose_ask_unbound(
        spec: AgentRunSpec,
        collector: TurnCollector,
        directive: IXDirective,
        text: str | None,
    ) -> bool:
        """acc_039 — is this reply a permission ask the id path never saw?

        Five conditions, all read off state the turn already has:

        1. `autonomy_level` is armed, and armed at a setting whose floor stops
           external actions (`Reactive`, `Self-directed`). Under `Autonomous`
           nothing stops, so a prose ask cannot be duplicated by a gate.
        2. The delivered prose's last non-empty line ends with `?`.
        3. No gate is open, and
        4. nothing is held. Together these two *are* "the turn attempted no call
           the floor would have stopped": under a gating policy such a call
           either opens the gate (3) or is held (4), and both exits leave the
           loop long before this point. Requiring literally zero tool calls
           would miss the audited shape, where the PA read the clinic's free
           slots and only then asked about the booking in prose.
        5. No plan item is left unexecuted, so there is no bound id the next
           turn's call could inherit the ask from.
        """
        if not AgentRunner._attribute_engaged(spec, "autonomy_level"):
            return False
        if not gate_opens(directive.autonomy_level_policy, "external_action", False):
            return False
        if collector.gate is not None or collector.held:
            return False
        # The certainty block is never delivered, so it must not hide the `?`
        # that is (§5.3 carrier F).
        prose = strip_certainty_block(text) if text else ""
        lines = [line for line in prose.splitlines() if line.strip()]
        if not lines or not lines[-1].rstrip().endswith("?"):
            return False
        executed = {
            call.interaction_item_id for call in collector.calls if call.executed
        }
        return not any(
            not (set(item.bound_item_ids) & executed)
            for item in collector.payloads.plan
        )

    @staticmethod
    def _undivided_question_block(text: str | None) -> tuple[str, int] | None:
        """The first `[[q:…]]` block in `text` that asks two questions or more.

        Returns `(block_id, question_count)`, or `None` when every block asks
        at most one thing. A block owns the prose from its marker to the next
        marker of either kind, the way `render.py::_split_sections` cuts it, and
        a question is a sentence ending in `?` under the crude split the
        renderer uses elsewhere -- one `?` is one question, and the reading only
        has to tell one from two.
        """
        if not text:
            return None
        markers = list(_SECTION_MARKER_RE.finditer(text))
        for index, marker in enumerate(markers):
            if not marker.group("question"):
                continue
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            questions = sum(
                1
                for sentence in _SENTENCE_SPLIT_RE.split(text[marker.end():end])
                if sentence.strip().endswith("?")
            )
            if questions >= 2:
                block = QUESTION_MARKER_RE.search(marker.group(0))
                assert block is not None
                return block.group(1), questions
        return None

    @staticmethod
    def _unmarked_payload_questions(
        collector: TurnCollector, response: Any,
    ) -> int:
        """How many declared questions this turn has written no `[[q:…]]` for.

        Read off the payloads *plus* this response's own tool calls, because the
        nudge runs before `add_payloads`: the payload and the prose usually
        arrive together (live `acc_009` t=8), but a payload sent on an earlier
        iteration counts just the same. Zero while the response wrote no prose —
        there is nothing to mark yet, and a nudge there would be premature.
        """
        if not (response.content or "").strip():
            return 0
        payloads = collect_payloads(response.tool_calls, collector.payloads)
        declared = sum(1 for question in payloads.questions if question.id)
        if not declared:
            return 0
        text = " ".join([*collector.segments, response.content or ""])
        return 0 if QUESTION_MARKER_RE.search(text) else declared

    @staticmethod
    def _certainties_all_dropped(
        collector: TurnCollector, settings: Mapping[str, str],
    ) -> int:
        """How many certainties this turn declared when the locator placed **none** of them.

        Zero unless every one of them dropped: a turn that hedged one claim out
        of three still put the `Express` surface in front of the user, and the
        per-claim `ix_certainty_claim_not_found` rows are the record for that.
        Only the all-or-nothing turn is worth a request, because that is the one
        where the attribute under test rendered nothing at all (regress13
        acc_035 t=7: three declared, `annotated 0/3 claims`, a plain reply, and
        the reception beat scored it as if the PA had chosen `Hide`).

        `Express` only. Under `Hide` the renderer appends nothing by design, so
        an unplaced claim changes not one byte of the surface and re-asking
        would spend a request on a difference the user cannot see -- and would
        leak the setting into the request pattern besides.

        A row carries `strategy` only when it went down the claim channel, so a
        ref placed by a `[[c:…]]` marker, or one with an empty claim, is not a
        locator drop and holds the turn back from this exit.
        """
        if settings.get("uncertainty_expression", "") != "Express":
            return 0
        declared = list(collector.payloads.certainties)
        if not declared:
            return 0
        rows = render_certainties(
            "Express", collector.last_segment, declared,
        ).rows
        if any(row.get("strategy") != "not_found" for row in rows):
            return 0
        return len(rows)

    @staticmethod
    def _gate_message(collector: TurnCollector) -> str:
        """The PA's own `_ix_confirmation_message` for the call that gated.

        Delivered verbatim: the wrapper never writes the ask on the PA's behalf.
        """
        if collector.gate is None:
            return ""
        if collector.gate.attribute == "capability_boundary":
            boundary = collector.payloads.boundary
            return boundary.handoff_message if boundary else ""
        return triggering_ask(
            collector.calls, collector.payloads.plan, collector.gate.item_id
        )

    @staticmethod
    def _capability_handoff_gate(
        collector: TurnCollector, directive: IXDirective
    ) -> GateRecord | None:
        """`Find and hand off` gates once the boundary has been declared (§3.5)."""
        if directive.capability_boundary_policy != "find_handoff_path":
            return None
        if not collector.boundary_declared or collector.gate is not None:
            return None
        return GateRecord(
            attribute="capability_boundary",
            item_id=collector.payloads.boundary_id,
            reason="handoff",
            asked_at_turn=collector.turn_index,
            action_type="read",
            message_source="handoff_message",
        )

    def _delivers_something(
        self,
        spec: AgentRunSpec,
        collector: TurnCollector,
        settings: dict[str, str],
        final_content: str | None,
    ) -> bool:
        """Would this turn hand the user any text at all?

        The renderer answers, not a second copy of its rules: what the user gets
        is `render()`'s message, or -- when that is empty -- the loop's cleaned
        `final_content` fallback. `render()` is pure, so asking it here costs
        nothing and cannot disagree with the answer the loop gets later.
        `collector.empty_reply` is still False at this point, so the template
        line is not yet in play and the question stays answerable.
        """
        rendered = render_turn(
            collector.to_render_input(
                hold=spec.hold, consent_decisions=spec.consent_decisions,
            ),
            settings,
            held=collector.held,
        )
        if rendered.message.strip():
            return True
        # regress13 acc_009 — the fallback reads the model's raw text, which on
        # a hold turn still carries every excised `[[q:…]]` section: under
        # `Infer` (release 0/N) a reply with no framing at all rendered to
        # nothing here, answered "yes, it delivers", and `loop.py`'s matching
        # fallback then handed the user both withheld questions verbatim. When
        # the renderer held prose, an empty message is its decision and the raw
        # text is not a substitute for it -- the turn goes to the one re-ask
        # below and asks the PA what it can say without the questions.
        if rendered.prose_held:
            return False
        return bool(clean_delivered_text(final_content or "").strip())

    @staticmethod
    def _completion_stands_in_for_the_turn(
        collector: TurnCollector,
        final_content: str | None,
        *,
        claim_only: bool = False,
    ) -> bool:
        """acc_003 t=20 — a closing line delivered in place of the answer.

        Three conditions, all read off the turn the wrapper already has: the
        render call carried a `completion_message`, no task tool ran, and the
        prose is nothing the completion does not already say. Any one of them
        alone is ordinary — a short exchange releases a completion over real
        prose, and a turn with no task call may still answer in full — but
        together they are a turn that reported work it never did.

        `calls` records the IX carriers alongside the task calls, and those are
        never the work: a turn is bound to have called `IX_render_*` to have a
        completion at all.

        acc_003 t=17 is the same turn one notch wider: the prose is *different*
        text making the completion's claim rather than repeating it, so
        containment does not see it. The second trigger reads the claim itself —
        a closing line whose verb asserts delivery, over prose too short to be
        the delivery — and lands on the same one re-prompt.

        `claim_only` drops that second trigger, and the recheck after the
        re-prompt uses it. The wider trigger is a suspicion, cheap to act on
        because acting costs one request and the reply ships either way; it is
        not evidence, and a short answer that arrives on the retry would be
        recorded as a defect it is not. Containment is the one shape the wrapper
        can be certain did not answer, so that is what the audit is allowed to
        record.

        An empty reply is a different rule with its own request (round 9 item 1),
        so it is left alone here.
        """
        completion = collector.payloads.completion_message.strip()
        if not completion:
            return False
        if any(
            call.executed and not call.tool_name.startswith(IX_TOOL_NAME_PREFIX)
            for call in collector.calls
        ):
            return False
        prose = clean_delivered_text(final_content or "").strip()
        if prose_within_completion(prose, completion):
            return True
        return (
            not claim_only
            and bool(prose)
            and len(prose) < UNFINISHED_TURN_PROSE_LIMIT
            and bool(WORK_ASSERTED_RE.search(completion))
        )

    async def _resolve_missing_notice(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        hook_context: AgentHookContext,
        messages: list[dict[str, Any]],
        collector: TurnCollector,
        tool_events: list[dict[str, Any]],
        final_content: str | None,
        stop_reason: str,
    ) -> tuple[str | None, str]:
        """Owner decision 2026-09-07 — a narrating turn must carry its notice.

        regress5 acc_003 t=3: the PA selected Bookend, made three catalog
        lookups, wrote `_ix_notice` on none of them, and the user read a reply
        with no opening line -- the surface of Silent under a setting that is
        not Silent, on the one turn the session probes. The renderer counted
        `notices 0/0` and shipped it.

        Up to `IX_MISSING_NOTICE_ATTEMPTS` requests ask for the opening line;
        the first non-empty reply becomes the notice of the first executed task
        call, which is the one Bookend releases and the first Full narration
        releases. A turn still without one ends on `ix_contract_failure`, as a
        missing render payload does: no scorable reply, nothing persisted, the
        step retried from a clean state. `ix_notice_repaired` records the
        repaired case with its attempt count; `ix_notice_missing` rides on the
        failure for the harness to write beside the step.

        Zero-call turns are not this rule's business (the closing line is held
        there anyway), and the empty-reply / unfinished-turn rules keep their
        own one request: `collector.closing_iteration` is left alone.
        """
        if stop_reason in (STOP_REASON_IX_CONTRACT_FAILURE, "error"):
            return final_content, stop_reason
        if spec.ix_context.get("proactive_event"):
            return final_content, stop_reason
        setting = settings_from_selections([
            *spec.inherited_ix_selections,
            *selections_from_tool_events(tool_events),
        ]).get("process_visibility", "")
        if setting not in ("Bookend", "Full narration"):
            return final_content, stop_reason
        task_indexes = [
            index for index, call in enumerate(collector.calls)
            if call.executed and not call.tool_name.startswith(IX_TOOL_NAME_PREFIX)
        ]
        if not task_indexes or any(
            collector.calls[index].notice.strip() for index in task_indexes
        ):
            return final_content, stop_reason
        for attempt in range(1, IX_MISSING_NOTICE_ATTEMPTS + 1):
            reply = await self._text_only_request(
                spec, hook, hook_context, messages, collector,
                IX_MISSING_NOTICE_PROMPT,
            )
            line = clean_delivered_text(reply or "").strip()
            if not line:
                continue
            first = task_indexes[0]
            collector.calls[first] = replace(collector.calls[first], notice=line)
            messages.append(build_assistant_message(line))
            collector.emit("ix_notice_repaired", attempts=attempt, setting=setting)
            return final_content, stop_reason
        collector.emit(
            "ix_notice_missing", attempts=IX_MISSING_NOTICE_ATTEMPTS, setting=setting,
        )
        hook_context.stop_reason = STOP_REASON_IX_CONTRACT_FAILURE
        return None, STOP_REASON_IX_CONTRACT_FAILURE

    @staticmethod
    def _resolve_truncated_turn(
        final_content: str | None,
        collector: TurnCollector,
        stop_reason: str,
    ) -> str | None:
        """regress16 acc_006 — a turn the provider cut off does not ship a fragment.

        Both PA turns of that session reached the user as a mid-tool-sequence
        aside: `Let me start by looking at the research folder to find the
        citation notes`, and then `Let me remove the remaining fragments. I'll
        work from the bottom up`. Six `provider_truncated` events, three per
        turn, `completion_tokens == 4096` each — the model spent its whole
        budget reasoning and the closing prose was never written. The renderer's
        `_final_segment` then reached back past the empty responses to the last
        iteration that still had words in it, which was narration written
        *between* tool calls, and `_delivers_something` saw a non-empty message
        and let the turn through with the empty-reply rule never firing.

        So the tail goes: whatever follows the last sentence end was not
        finished, and a text with no sentence end at all leaves nothing. What
        remains, if anything, is the turn; what does not is the empty-reply
        rule's, one rule further down — it spends one request asking the PA for
        the sentence the turn owes, and delivers `IX_EMPTY_REPLY_LINE` when that
        comes back empty too. The token limit itself is not touched here.
        """
        if collector.truncated_completion_tokens is None:
            return final_content
        if stop_reason in (STOP_REASON_IX_CONTRACT_FAILURE, "error"):
            # Both are raised by the loop; the turn is not going to be delivered
            # and its text is never read.
            return final_content
        delivered = collector.delivered_segment or (final_content or "")
        # The trailing `<certainties>` block comes after the prose, so it is
        # either whole -- and then the prose in front of it is whole too and
        # nothing is dropped below -- or it was cut off with everything else.
        # Either way the rows are already banked in `payloads` by `add_segment`.
        kept, dropped = split_truncated_tail(strip_certainty_block(delivered))
        if not dropped:
            return final_content
        collector.emit(
            "turn_truncated",
            completion_tokens=collector.truncated_completion_tokens,
            dropped_chars=len(delivered) - len(kept),
        )
        collector.replace_delivered_segment(kept)
        # Re-read rather than return `kept`: this is the loop's unfiltered
        # `public_turn` fallback, and `last_segment` is the one form of the text
        # that has been through every delivery strip (regress13 acc_009).
        return collector.last_segment or None

    async def _resolve_unfinished_turn(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        hook_context: AgentHookContext,
        messages: list[dict[str, Any]],
        collector: TurnCollector,
        final_content: str | None,
        stop_reason: str,
    ) -> str | None:
        """One re-prompt for a turn whose whole reply is its own closing line.

        The wrapper writes none of the answer: it says what is missing and the
        PA writes the turn. A second reply that is still nothing but the closing
        line is delivered as it stands, with `ix_completion_without_content` as
        the record — the same shape as the other nudges, because the wrapper
        cannot produce the summary the PA claimed to have written.

        acc_001 t32 comes in the same door: a reply that was nothing but an
        unparsed `<｜DSML｜…>` tool call has said nothing to the user either, and
        the stripper is about to leave it empty. It buys the one request here
        and, if the second reply is empty too, the empty-reply rule below owns
        the record — `ix_completion_without_content` would be the wrong name for
        a turn whose closing line claimed nothing.
        """
        if stop_reason in (STOP_REASON_IX_CONTRACT_FAILURE, "error"):
            return final_content
        if spec.ix_context.get("proactive_event"):
            return final_content
        if not (
            reply_is_only_markup(final_content or "")
            or self._completion_stands_in_for_the_turn(collector, final_content)
        ):
            return final_content
        if collector.closing_iteration is None:
            await self._closing_iteration(
                spec, hook, hook_context, messages, None, collector,
                prompt=IX_UNFINISHED_TURN_PROMPT,
            )
            retried = collector.last_segment or final_content
            if not self._completion_stands_in_for_the_turn(
                collector, retried, claim_only=True,
            ):
                return retried
            final_content = retried
        collector.emit(
            "ix_completion_without_content",
            chars=len(clean_delivered_text(final_content or "").strip()),
        )
        return final_content

    async def _resolve_empty_reply(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        hook_context: AgentHookContext,
        messages: list[dict[str, Any]],
        collector: TurnCollector,
        tool_events: list[dict[str, Any]],
        final_content: str | None,
        stop_reason: str,
    ) -> str | None:
        """Round 9 item 1 — an empty reply is a miss, whatever emptied it.

        One rule at the single exit, replacing the reason-specific guards: three
        distinct routes to a turn with no user-facing text were found in four
        rounds (a `binding_invalid` fail-closed exit, a hold whose closing
        request came back silent, and -- `v8s01` t26 -- a provider completion
        with no text and no tool calls at all, answering a two-part instruction
        the user then heard nothing about). Guarding them one at a time was not
        converging, so the emptiness itself is what is checked.

        The turn buys exactly one extra request: a hold that already spent its
        closing request goes straight to the template rather than asking twice.
        """
        if stop_reason in (STOP_REASON_IX_CONTRACT_FAILURE, "error"):
            # The loop raises on both; a provider request here would be spent on
            # a turn that is already not going to be delivered.
            return final_content
        if spec.ix_context.get("proactive_event"):
            # A suppressed reminder delivers nothing *by design* (contract 11).
            return final_content
        settings = settings_from_selections([
            *spec.inherited_ix_selections,
            *selections_from_tool_events(tool_events),
        ])
        if self._delivers_something(spec, collector, settings, final_content):
            return final_content
        if collector.closing_iteration is None:
            await self._closing_iteration(
                spec, hook, hook_context, messages, None, collector,
                prompt=IX_EMPTY_REPLY_PROMPT,
            )
            if self._delivers_something(
                spec, collector, settings, collector.last_segment or final_content
            ):
                return collector.last_segment or final_content
        if collector.dsml_markup_seen:
            # regress5 acc_006: five turns whose reply was tool-call syntax
            # shipped `I have nothing further to report on that just now.` The
            # PA had four document edits in hand and typed every one of them --
            # the line is false, and it is false in the direction that hides the
            # defect, because a turn that reads like a polite no-op is scored as
            # a turn and not as the four missing writes it really was. The
            # request above was still spent asking for the reply; if that came
            # back empty too the turn ships empty, and the audit plane
            # (`ix_empty_reply` beside `ix_dsml_calls_recovered` /
            # `ix_markup_stripped`) is what says why.
            collector.emit(
                "ix_empty_reply", stop_reason=stop_reason, dsml_markup=True,
            )
            # `None`, never `final_content`: reaching here means every delivery
            # surface already cleaned that text to nothing, and `loop.py`'s
            # `public_turn` fallback reads `final_content` unfiltered -- handing
            # the markup back would undo 03ee0f7 and put it in front of the user.
            return None
        collector.empty_reply = True
        collector.emit("ix_empty_reply", stop_reason=stop_reason)
        return IX_EMPTY_REPLY_LINE

    async def _closing_iteration(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        hook_context: AgentHookContext,
        messages: list[dict[str, Any]],
        provider_response: Any,
        collector: TurnCollector,
        held_result: str = IX_HELD_TOOL_RESULT,
        prompt: str = IX_HOLD_CLOSING_PROMPT,
    ) -> None:
        """§3.4 (part 5) — the one closing request after a batch / question hold.

        The hold is already recorded and no task tool is offered any more, so
        this cannot start work; it exists because the model that writes its
        prose in the message *after* its calls has not written the turn's
        message yet. A response that carries tool calls anyway is used for its
        text alone, and a response with no text leaves the turn exactly where
        the hold left it.

        Round 9 item 1 reuses it for the empty-reply rule with a different
        `prompt` and no `provider_response`: the conditions it needs — task
        tools withdrawn, one request, text taken or not — are the same, and the
        turn is allowed exactly one of these however it emptied.
        """
        for tool_call in getattr(provider_response, "tool_calls", ()) or ():
            if tool_call.id in collector.skipped_call_ids:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": held_result,
                })
        clean = await self._text_only_request(
            spec, hook, hook_context, messages, collector, prompt,
        )
        collector.closing_iteration = bool(clean and clean.strip())
        if not collector.closing_iteration:
            return
        # Added after the holds were recorded, so it shares no assistant message
        # with a held call: the §3.3 (part 3)·a narration cut does not reach it, and
        # the renderer selects its `[[batch:…]]` / `[[q:…]]` sections as usual.
        collector.add_segment(clean)
        messages.append(build_assistant_message(clean))

    async def _text_only_request(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        hook_context: AgentHookContext,
        messages: list[dict[str, Any]],
        collector: TurnCollector,
        prompt: str,
    ) -> str | None:
        """One provider request for text alone: task tools withdrawn.

        The closing iteration and the missing-notice request share it. The
        prompt goes on the history; what the caller does with the reply -- a
        prose segment, a notice field -- is the caller's, so nothing is
        recorded here beyond the reasoning and the ignored tool calls.
        """
        # The closing / notice request is a provider request like any other, and
        # it runs on the same history: it needs the same guarantee.
        _answer_withheld_tool_calls(messages)
        messages.append(internal_user_message(prompt))

        request: dict[str, Any] = {
            "messages": messages,
            "tools": [
                definition for definition in spec.tools.get_definitions()
                if definition.get("function", {}).get("name") in _CLOSING_TOOL_NAMES
            ],
            "model": spec.model,
        }
        if spec.temperature is not None:
            request["temperature"] = spec.temperature
        if spec.max_tokens is not None:
            request["max_tokens"] = spec.max_tokens
        if spec.reasoning_effort is not None:
            request["reasoning_effort"] = spec.reasoning_effort
        if spec.llm_task_identity is not None:
            request["task_identity"] = spec.llm_task_identity
        response = await self.provider.chat_with_retry(**request)
        # regress12 acc_003 t=12 — this is the request that failed, and its
        # error body left here as prose: the closing iteration took it for the
        # turn's text and the wrapper released it, request id and all. The main
        # loop's error branch never sees this request, so the guard is repeated
        # rather than shared: one more try with the same messages, then no text.
        if response.finish_reason == "error":
            collector.emit("provider_error_retried", attempt=1)
            response = await self.provider.chat_with_retry(**request)
            if response.finish_reason == "error":
                return None

        if response.tool_calls:
            collector.emit(
                "ix_hold_closing_tool_call_ignored",
                tools=[tool_call.name for tool_call in response.tool_calls],
            )
        collector.add_reasoning(response.reasoning_content)
        return hook.finalize_content(hook_context, response.content)

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        ix_directive: IXDirective,
        collector: TurnCollector,
        prior_events: Sequence[dict[str, Any]] = (),
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        has_ix_call = any(
            tool_call.name.startswith(IX_TOOL_NAME_PREFIX) for tool_call in tool_calls
        )
        # The wrapper path is not conditional on this particular response
        # looking interesting. `has_control_directive` is False whenever no
        # selection carries a directive -- an unselected attribute, or
        # `information_elicitation = Infer` -- so a
        # response that happened to carry no IX call used to fall through to
        # the plain path -- which never validates bindings and never gates.
        # Live: an `external_action` ran with `interaction_item_id = None` while
        # autonomy had no directive. While the IX layer is engaged at all, every
        # response goes through it. (A `capability_limit` is noticed on both
        # paths: it is what activates the attribute in the first place.)
        if (
            self._ix_layer_engaged(spec)
            or spec.ix_context.get(CONTEXT_KEY)
            or spec.required_tool_names
            or ix_directive.has_control_directive
            or has_ix_call
        ):
            return await self._execute_tools_ix_first(
                spec, tool_calls, ix_directive, collector, prior_events
            )

        if spec.concurrent_tools:
            tool_results = await asyncio.gather(*(
                self._run_tool(spec, tool_call)
                for tool_call in tool_calls
            ))
        else:
            tool_results = [
                await self._run_tool(spec, tool_call)
                for tool_call in tool_calls
            ]

        results: list[Any] = []
        events: list[dict[str, Any]] = []
        for tool_call, (result, event) in zip(tool_calls, tool_results):
            results.append(result)
            events.append(event)
            collector.record_call(
                tool_call,
                action_type=self._tool_action_type(tool_call.name, spec.ix_context),
                executed=True,
                status=str(event.get("status") or "ok"),
            )
            # The one signal that reaches into the IX layer from outside it. A
            # capability limit activates `capability_boundary` even on a turn no
            # attribute was governing: live `acc_026` hit the limit on `b1`,
            # while the first beat listing any skill was `b3`, so gating the
            # detection on an already-engaged layer is what let the PA route
            # around the block unobserved. The main loop rebuilds the tool view
            # from here, and every later iteration of the turn takes the IX
            # path because the selection is now owed.
            if is_capability_limit(result):
                collector.capability_limit_seen = True
        return results, events

    async def _execute_tools_ix_first(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        ix_directive: IXDirective,
        collector: TurnCollector,
        prior_events: Sequence[dict[str, Any]] = (),
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        ix_calls = [
            tool_call for tool_call in tool_calls
            if tool_call.name.startswith(IX_TOOL_NAME_PREFIX)
        ]
        task_calls = [
            tool_call for tool_call in tool_calls
            if not tool_call.name.startswith(IX_TOOL_NAME_PREFIX)
        ]

        results_by_id: dict[str, Any] = {}
        events: list[dict[str, Any]] = []

        for tool_call in ix_calls:
            reselected = self._reselected_attribute(spec, tool_call)
            if reselected is not None:
                # No event: the refusal ladder reads `events`, and a call that
                # asks for something already recorded is neither a refusal nor
                # a repair. Nothing is banked either -- the session's selection
                # is read off the tool events this call does not join.
                results_by_id[tool_call.id] = IX_RESELECTION_IGNORED_RESULT.format(
                    attribute=reselected
                )
                record = collector.record_call(
                    tool_call, action_type="read", executed=False,
                    status="ignored",
                )
                collector.emit(
                    "ix_reselection_ignored",
                    attribute=reselected,
                    setting=str(record.args.get("setting") or ""),
                    turn_index=spec.turn_index,
                    beat=str(spec.ix_context.get("beat") or ""),
                    call_index=record.order,
                )
                continue
            result, event = await self._run_tool(spec, tool_call)
            results_by_id[tool_call.id] = result
            events.append(event)
            # IX tools are declarations, never world actions: they always run,
            # and they stay in the tool protocol so it remains well formed.
            collector.record_call(
                tool_call, action_type="read", executed=True,
                status=str(event.get("status") or "ok"),
            )

        # §3.0 / §7.1: the selection has to be *valid* before this turn's task
        # calls are filtered against it. A refused one contributes no payload,
        # so the directive below is rebuilt empty -- and an empty policy gates
        # nothing: live, `IX_autonomy_level(setting="Suggest")` errored and the
        # same response's `email_message_send` went out unasked. The task calls
        # are withheld here, before the filter. Nothing is refused for good: the
        # loop asks for the selection again and the model reissues them.
        if self._rejected_ix_tools(events):
            for tool_call in task_calls:
                record = collector.record_call(
                    tool_call,
                    action_type=self._tool_action_type(tool_call.name, spec.ix_context),
                    executed=False,
                    status="held",
                )
                collector.skipped_call_ids.add(tool_call.id)
                collector.hold_call(
                    record,
                    attribute=self._hold_attribute(
                        set(spec.ix_context.get("session_active_ix") or [])
                    ),
                    reason="ix_selection_invalid",
                )
            return [
                results_by_id.get(tool_call.id, "") for tool_call in tool_calls
            ], events

        # Every selection the turn has made, not only this response's: a
        # response that resends the render payload after a repair carries no
        # selection tool, and rebuilding the directive from it alone dropped the
        # setting mid-turn -- `One-at-a-time` stopped holding on the very retry
        # that was asked for.
        directive = (
            self._directive_for_events(spec, [*prior_events, *events])
            if ix_calls
            else ix_directive
        )
        # qwen `acc_082`, 2026-09-18 — name resolution, then authorisation, then
        # execution. Qwen called `mcp_state_calendar_list`, which does not exist
        # (the beat offered `..._event_search` and `..._event_read`), and the
        # invented name went straight into the business-call list: §3.8's
        # fail-closed floor read it as an `external_action`, Self-directed
        # withheld it for a confirmation, and the registry's "Tool ... not
        # found" -- which was sitting right there -- never reached the model.
        # The turn ended reported as carrying no ask. Deciding whether an
        # operation is authorised before knowing the operation exists is the
        # wrong order. The name is checked against this turn's own registry
        # view, never probed by running anything, and §3.8's classification
        # defaults are untouched: a tool that does not exist simply does not
        # reach them.
        offered = set(spec.tools.tool_names)
        unknown_calls = [
            call for call in task_calls if call.name not in offered
        ]
        task_calls = [call for call in task_calls if call.name in offered]
        for tool_call in unknown_calls:
            # The registry's own not-found path, so the message, the event and
            # the call id's tool result are the ones every other tool error
            # gets. `executed=False` is load-bearing and not cosmetic:
            # `render.released_ids` derives `plan_executed` from the bound
            # calls' `executed` flags, so a record marked only `status: error`
            # would close out the plan step the call was bound to.
            result, event = await self._run_tool(spec, tool_call)
            results_by_id[tool_call.id] = result
            events.append(event)
            collector.record_call(
                tool_call,
                action_type=UNKNOWN_TOOL_ACTION_TYPE,
                executed=False,
                status=str(event.get("status") or "error"),
                error=str(event.get("error") or ""),
            )

        if task_calls:
            events.extend(self._process_events(directive, "start", task_calls))
            self._gate_task_calls(spec, task_calls, directive, collector)
            struck = _struck_tools(collector)
            limit_attempt_changed = False
            for tool_call in task_calls:
                record = collector.record_for(tool_call.id)
                if record is None or not record.executed:
                    continue
                action_type = self._tool_action_type(tool_call.name, spec.ix_context)
                # Earlier siblings run sequentially and may have changed the
                # world since the batch gate. Recheck immediately before MCP.
                args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
                bound_attempts = [a for a in spec.ix_context.get(CONTEXT_KEY, [])
                                  if attempt_matches(a, catalog_tool_name(tool_call.name), {
                                      k: v for k, v in args.items()
                                      if k not in PER_CALL_INJECTED_FIELDS
                                  })]
                if ((limit_attempt_changed and action_type != "read") or
                        (bound_attempts and not any(attempt_current(a) for a in bound_attempts))):
                    limit_attempt_changed = True
                    collector.update_call(tool_call.id, status="held", executed=False)
                    collector.skipped_call_ids.add(tool_call.id)
                    collector.executed_item_ids.discard(record.interaction_item_id)
                    collector.consent = [
                        replace(row, released=False)
                        if row.item_id in {record.interaction_item_id, catalog_tool_name(tool_call.name)}
                        else row for row in collector.consent
                    ]
                    collector.hold_call(collector.record_for(tool_call.id), attribute="autonomy_level",
                                        reason="external_action_floor")
                    held_ids_now = {item.interaction_item_id for item in collector.held}
                    collector.gate_remaining = [item for item in collector.payloads.plan
                                                if held_ids_now & set(item.bound_item_ids)]
                    if collector.gate is None:
                        collector.gate = GateRecord(
                            attribute="autonomy_level", item_id=record.interaction_item_id or tool_call.name,
                            reason="external_action_floor", asked_at_turn=collector.turn_index,
                            action_type=action_type,
                        )
                    continue
                # X-3, owner 2026-09-19: task calls in one provider response run
                # in order.  Once one of them returns a capability limit, the
                # already-selected Find-and-hand-off policy applies to the
                # remaining siblings immediately.  Earlier calls have already
                # happened and reads may still collect diagnosis; later writes
                # and outward actions must wait for the handoff instead of
                # slipping through before the next provider iteration.
                if (
                    directive.capability_boundary_policy == "find_handoff_path"
                    and collector.capability_limit_seen
                    and action_type != "read"
                ):
                    collector.update_call(
                        tool_call.id,
                        status="held",
                        executed=False,
                    )
                    held = collector.record_for(tool_call.id)
                    if held is not None:
                        collector.skipped_call_ids.add(tool_call.id)
                        collector.hold_call(
                            held,
                            attribute="capability_boundary",
                            reason="capability_handoff",
                        )
                    continue
                # regress12 acc_004 — the same tool, the same error, over and
                # over on one guessed argument. The call still gets a result, so
                # the tool protocol stays well formed, but the tool does not run:
                # the value is not in the world and the next guess will not find
                # it either. `run()` tells the model to ask the user instead.
                if tool_call.name in struck:
                    # gpt acc_082 — the one exception. A calendar entry the user
                    # authorised, refused three times on `recurrence` alone and
                    # never written, is a parameter the model can fix; blocking
                    # the fixed call was the first of that turn's two dead ends.
                    # The exception is spent here, before the call runs, so a
                    # second one in the same response cannot also have it, and
                    # the strike count is left alone: this is an extra budget,
                    # not a reset. Authorisation is not bypassed -- the gate
                    # above has already decided this call may run.
                    repair = _recurrence_repair(
                        collector, tool_call, record.interaction_item_id
                    )
                    if repair is None:
                        strike_result = TOOL_FAILURE_STRIKE_RESULT.format(
                            tool=tool_call.name, limit=TOOL_FAILURE_STRIKE_LIMIT,
                        )
                        results_by_id[tool_call.id] = strike_result
                        collector.update_call(
                            tool_call.id, status="error", error=strike_result,
                        )
                        continue
                    kind, item_id, pending = repair
                    if kind == "rebind":
                        rebind_result = RECURRENCE_REBIND_RESULT.format(
                            item_id=item_id
                        )
                        results_by_id[tool_call.id] = rebind_result
                        collector.update_call(
                            tool_call.id, status="error", error=rebind_result,
                        )
                        collector.emit(
                            "tool_recurrence_repair_rebind",
                            tool=tool_call.name,
                            item_id=item_id,
                        )
                        continue
                    pending.spent = True
                    collector.emit(
                        "tool_recurrence_repair_allowed",
                        tool=tool_call.name,
                        item_id=item_id,
                    )
                events.extend(self._process_events(directive, "progress", [tool_call]))
                result, event = await self._run_tool(spec, tool_call)
                results_by_id[tool_call.id] = result
                events.append(event)
                collector.update_call(
                    tool_call.id,
                    status=str(event.get("status") or "ok"),
                    error=str(event.get("error") or ""),
                )
                # A `capability_limit` has its own gate below and is never
                # counted here; two different tools never share a strike.
                if event.get("status") == "error":
                    strike = (
                        tool_call.name,
                        _tool_error_class(tool_call, event.get("error") or ""),
                    )
                    failures = collector.tool_failure_strikes.get(strike, 0) + 1
                    collector.tool_failure_strikes[strike] = failures
                    if failures == TOOL_FAILURE_STRIKE_LIMIT:
                        struck.add(tool_call.name)
                        collector.emit(
                            "tool_failure_strike",
                            tool=strike[0],
                            error_code=strike[1][:120],
                            count=failures,
                        )
                    # gpt acc_082 — a `recurrence` the calendar refused before
                    # writing anything. The item stays authorised (no executed,
                    # `status == "ok"` call, so no release is consumed), so the
                    # one correction is owed to *this* item: recorded under its
                    # id, with the arrangement's own arguments, so a later call
                    # that changes the evening is not mistaken for the fix.
                    if (
                        _is_recurrence_invalid(event.get("error"))
                        and record.interaction_item_id
                    ):
                        collector.recurrence_repairs.setdefault(
                            record.interaction_item_id,
                            RecurrenceRepair(
                                tool=tool_call.name,
                                business_args=_business_args_signature(
                                    tool_call.arguments
                                ),
                            ),
                        )
                # Every tool result, on every turn -- and whether or not the
                # attribute was already engaged. The pre-run-test audit (§2,
                # acc_026) found the limit firing two beats before the beat that
                # listed `capability_boundary`, so gating detection on the beat
                # let the PA route around the block with no setting chosen. The
                # trigger activates the attribute; `_with_capability_boundary`
                # and the tool-view rebuild put the selection tool on the turn.
                if is_capability_limit(result):
                    collector.capability_limit_seen = True
            events.extend(self._process_events(directive, "completion", task_calls))

        return [
            results_by_id.get(tool_call.id, "") for tool_call in tool_calls
        ], events

    def _gate_task_calls(
        self,
        spec: AgentRunSpec,
        task_calls: list[ToolCallRequest],
        directive: IXDirective,
        collector: TurnCollector,
    ) -> None:
        """Decide, before anything runs, which task calls this turn may execute.

        The gate is a pre-execution filter, not an after-the-fact apology: a
        held call never touches the world, so no synthetic tool result is
        produced for it and it never enters `runtime_history` (§2.4, §3.3 (part 2)·D).
        """
        session_active = set(spec.ix_context.get("session_active_ix") or [])
        gate_bearing = bool(session_active & HOLD_BEARING_ATTRIBUTES)

        if gate_bearing and self._binding_contract_active(spec):
            # A read needs no binding: it opens no gate, joins no batch release
            # and changes nothing, so an id on it authorises nothing either.
            # Live `acc_039` P10 spent a whole turn on this — four unbound
            # calls, all of them lookups, refused as a batch, and the user was
            # told the steps could not be carried out. Only the calls a binding
            # governs — draft, internal write, external action — are checked.
            binding_calls = [
                tool_call for tool_call in task_calls
                if self._tool_action_type(tool_call.name, spec.ix_context) != "read"
            ]
            declared = declared_item_ids(collector.payloads, spec.hold)
            # acc_023 t20: `b1` came back bound to a different call than the one
            # it was released for at t18. Same fail-closed exit as any other bad
            # binding — the id no longer names the calls it was authorised for.
            rebound = rebound_batch_ids(collector.payloads, spec.hold)
            if rebound:
                collector.emit("ix_batch_rebound", batch_ids=rebound)
                repair: str | None = IX_BATCH_REBIND_REPAIR.format(
                    batch_ids=", ".join(rebound),
                )
            else:
                repair = validate_bindings(
                    binding_calls,
                    declared,
                    {
                        question.id
                        for question in collector.payloads.questions
                        if question.id
                    },
                    ids_declarable=bool(session_active & ID_DECLARING_ATTRIBUTES),
                )
            if repair is not None:
                collector.repair = repair
                for tool_call in task_calls:
                    record = collector.record_call(
                        tool_call,
                        action_type=self._tool_action_type(tool_call.name, spec.ix_context),
                        executed=False,
                        status="error",
                    )
                    collector.skipped_call_ids.add(tool_call.id)
                    collector.hold_call(
                        record,
                        attribute=binding_attribute(
                            collector.payloads,
                            spec.hold,
                            self._hold_attribute(session_active),
                        ),
                        reason="binding_invalid",
                    )
                return

        # §3.4 (part 3): `One-at-a-time` releases the first batch, so a turn that
        # declared one batch for several topics releases everything and holds
        # nothing. Ask once for a per-topic split; a second undivided response
        # is accepted and recorded, because the wrapper cannot name the topics
        # for the model.
        if batch_decomposition_undivided(
            collector.payloads, directive.topic_management
        ):
            if collector.batch_repair_attempts == 0:
                collector.batch_repair_attempts = 1
                collector.batch_repair = IX_BATCH_SPLIT_REPAIR.format(
                    item_count=collector.payloads.item_count,
                )
                for tool_call in task_calls:
                    record = collector.record_call(
                        tool_call,
                        action_type=self._tool_action_type(tool_call.name, spec.ix_context),
                        executed=False,
                        status="held",
                    )
                    collector.skipped_call_ids.add(tool_call.id)
                    collector.hold_call(
                        record, attribute="topic_management", reason="batch_undivided",
                    )
                return
            collector.emit(
                "ix_batch_undivided",
                item_count=collector.payloads.item_count,
                batches=len(collector.payloads.batches),
            )

        questions_gate = (
            "information_elicitation" in session_active
            and directive.information_elicitation in {"structured", "iterative"}
            and bool(collector.payloads.questions)
        )
        # Owner, 2026-09-06: the countable difference between the two
        # `capability_boundary` settings is "at least one tool call after the
        # limit this turn" versus "zero". `_capability_handoff_gate` only runs
        # once `_execute_tools` has returned, so the response that answers the
        # forced `IX_capability_resolve` could attach a workaround call, execute
        # it, and *then* ship the handoff line -- the user is handed the problem
        # back while the PA quietly tried something, and the two settings become
        # indistinguishable in the transcript. The limit is only seen once the
        # limiting call has already run, so the response that hit it keeps its
        # own siblings. Owner, 2026-09-17: `Find and hand off` allows the
        # diagnostic lookups the hand-back needs, so afterwards only read calls
        # still run and every draft / internal write / external action is held
        # (`action_types.py`); whether a read was diagnosis or a recovery
        # attempt is the judge's semantic call, not the wrapper's.
        handoff_stops_the_turn = (
            directive.capability_boundary_policy == "find_handoff_path"
            and collector.capability_limit_seen
        )
        batch_allowed = released_item_ids(collector.payloads, directive.topic_management)
        held_ids: dict[str, str] = {
            str(call.get(INTERACTION_ITEM_ID_FIELD) or ""): attribute
            for attribute, episode in spec.hold.open_episodes.items()
            for call in episode.calls_held
        }
        # §3.3 (part 3) / §3.7 (part 3) (owner, round 10): the user authorised *an action*,
        # not an id. Releasing the same id for a different tool is a different
        # action -- live shape: `i1` held as `email_message_send`, allowed, then
        # reissued as `bill_payment_create` and executed with no second ask. It
        # inherits nothing and goes back through the gate as a fresh ask, which
        # is not an error: the PA may legitimately have changed how it does the
        # step. Only the tool name is compared; the arguments are the PA's to
        # revise, and the ask itself is what the user answers.
        held_tools: dict[str, str] = {
            str(call.get(INTERACTION_ITEM_ID_FIELD) or ""): str(call.get("tool") or "")
            for episode in spec.hold.open_episodes.values()
            for call in episode.calls_held
        }
        # §3.7 (part 3): consent is scoped to what the ask actually put to the user.
        # An `allow` for a held id that ask never named authorises nothing —
        # live, an ask about one cancellation came back with an `allow` for six
        # ids. Only episodes that did ask something are held to this: a batch or
        # question hold opens no gate and writes no ask, and its continuation
        # protocol is the whole release mechanism (§3.4 (part 3)).
        asked_ids = {
            str(call.get(INTERACTION_ITEM_ID_FIELD) or "")
            for episode in spec.hold.open_episodes.values()
            for call in episode.calls_held
            if call.get("asked")
        }
        asking_attributes = {
            attribute
            for attribute, episode in spec.hold.open_episodes.items()
            if any(call.get("asked") for call in episode.calls_held)
        }
        released = {
            decision.id
            for decision in collector.payloads.release_decisions
            if decision.decision == "release"
        }
        # Owner ruling 2026-09-08: which tools still carry an unspent release,
        # so a re-minted item does not re-ask for an act the user has already
        # authorised. Computed here as well as in the write-back because the
        # re-mint can happen in the very response that releases the old id
        # (acc_039 t28) -- the carried map alone would be one turn late.
        act_releases = live_act_releases(
            spec.hold, collector.payloads, spec.consent_decisions, collector.calls,
        )
        # Owner ruling 2026-09-08 ("这个肯定要改"): the go-ahead the user spoke in
        # the turn just gone, for an act nothing was holding yet. acc_039 t40
        # authorised the booking, t41 held it and asked again. Recomputed per
        # response like `act_releases`, and never written back to the hold: the
        # authorisation covers this turn only.
        preauthorised = preauthorised_acts(spec.consent_decisions, collector.calls)
        # §3.4 (part 3) (owner, round 6): the release *is* the release. An id the topic
        # episode is holding, released this turn through
        # `IX_continue_topic_management`, is past the batch restriction even when
        # the same response re-declares it inside an unreleased batch. Without
        # this the two protocols livelock: `v6g01` released `email1` on eight
        # turns, issued `email_message_send` on all eight, and the re-declared
        # `b2` re-held it every time -- the action was authorised eight times and
        # never happened. It only lifts `batch_not_released`; the id still goes
        # through the held-item path below, so a `deny` from the consent sideband
        # still keeps it held.
        batch_released = {
            item_id for item_id in released
            if held_ids.get(item_id) == "topic_management"
        }
        # §3.3 (part 2)·A — one release, one execution. `validate_bindings` resets its
        # duplicate check per provider response, and a turn is many responses,
        # so without this an id that was released once carried an unlimited
        # number of world-changing calls across the turn's iterations.
        collector.executed_item_ids |= {
            str(item_id)
            for episode in spec.hold.open_episodes.values()
            for item_id in episode.executed_item_ids
        }
        # Owner ruling 2026-09-08, the half this path was missing: it is the
        # *world change* that spends the release, so a call the tool refused
        # spends nothing. `live_act_releases` and `render.executed_now` both
        # test `status == "ok"`; this set was filled at gate time, before the
        # outcome was known, and never revised. Live regress21r acc_039 t19 --
        # the released `appointment_create` came back `unknown provider`, the
        # id was booked as spent anyway, and the 25 retries of a booking that
        # had never happened were all refused `release_spent`. An id whose
        # world-changing calls this turn have all errored gives its release
        # back here, so the retry runs under the same authorisation. A
        # duplicate inside the *same* response is unaffected: its record still
        # reads `ok`, because the gate runs before execution.
        errored_ids: set[str] = set()
        ok_ids: set[str] = set()
        for call in collector.calls:
            if not (
                call.executed
                and call.interaction_item_id
                and call.action_type in WORLD_CHANGING_ACTION_TYPES
            ):
                continue
            target = ok_ids if call.status == "ok" else errored_ids
            target.add(call.interaction_item_id)
        collector.executed_item_ids -= errored_ids - ok_ids
        stopped = False

        for tool_call in task_calls:
            args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            item_id = str(args.get(INTERACTION_ITEM_ID_FIELD) or "")
            action_type = self._tool_action_type(tool_call.name, spec.ix_context)
            plan_item = plan_item_for(collector.payloads, item_id)
            is_key = bool(plan_item.is_key) if plan_item is not None else False
            tool_changed = bool(
                held_tools.get(item_id)
                and held_tools[item_id] != tool_call.name
            )
            if tool_changed:
                collector.emit(
                    "ix_consent_tool_changed",
                    item_id=item_id,
                    held_tool=held_tools[item_id],
                    tool=tool_call.name,
                )

            attribute, reason = "", ""
            if stopped:
                attribute, reason = collector.gate.attribute, "after_open_gate"
            elif handoff_stops_the_turn and action_type != "read":
                attribute, reason = "capability_boundary", "capability_handoff"
            elif questions_gate:
                attribute, reason = "information_elicitation", "questions_gate"
            elif (
                item_id
                and item_id in collector.executed_item_ids
                and action_type in WORLD_CHANGING_ACTION_TYPES
            ):
                # The id already spent its release on a world-changing call.
                # Refused for this turn, and deliberately *not* recorded as a
                # hold on the episode: the only way to answer a hold is to
                # release the id again, which is exactly what is being refused.
                attribute = held_ids.get(item_id) or self._hold_attribute(session_active)
                reason = "release_spent"
                collector.emit(
                    "ix_release_spent", item_id=item_id, tool=tool_call.name,
                    action_type=action_type,
                )
            elif (
                batch_allowed is not None
                and item_id not in batch_allowed
                and item_id not in batch_released
            ):
                attribute, reason = "topic_management", "batch_not_released"
            elif item_id in held_ids and not tool_changed:
                # An item the user has already been asked about is governed by
                # the continuation protocol alone. Re-running the autonomy gate
                # on it would make consent unreachable: the same ask would fire
                # again on the turn the user answered it. A call that changed
                # tools is a different action, so it falls past this to the gate
                # below and is asked afresh.
                if item_id not in released:
                    attribute, reason = held_ids[item_id], "release_missing"
                    collector.emit(
                        "ix_release_missing", item_id=item_id, tool=tool_call.name,
                    )
                elif (
                    action_type == "external_action"
                    and held_ids[item_id] in asking_attributes
                    and item_id not in asked_ids
                ):
                    # The user was never asked about this one: it keeps waiting
                    # for an ask of its own, whatever the sideband said.
                    attribute, reason = held_ids[item_id], "consent_unasked"
                    collector.emit(
                        "ix_consent_unasked", item_id=item_id, tool=tool_call.name,
                    )
                elif action_type == "external_action":
                    verdict = sideband_decision(spec.consent_decisions, item_id)
                    if verdict != "allow":
                        attribute, reason = held_ids[item_id], f"consent_{verdict}"
                        collector.emit(
                            "ix_consent_denied" if verdict == "deny"
                            else "ix_consent_unresolved",
                            item_id=item_id,
                            sideband_decision=verdict,
                        )
            elif gate_opens(directive.autonomy_level_policy, action_type, is_key):
                inherited = (
                    act_releases.get(tool_call.name, "")
                    if action_type == "external_action" else ""
                )
                if inherited:
                    # Owner ruling 2026-09-08: the same act under a new id. The
                    # user authorised this tool and the act has not gone through
                    # yet, so the floor does not ask a second time; the grant is
                    # consumed by the first `ok` call on the tool, which is why
                    # a real second act still gets its own ask.
                    #
                    # Taken out of the map here as well, so one authorisation
                    # still buys one call (§3.3 (part 2)·A): a response issuing
                    # two re-minted items on the same tool gets the second one
                    # asked about. Only this turn's copy is touched — an errored
                    # call leaves the persisted grant standing.
                    del act_releases[tool_call.name]
                    collector.emit(
                        "ix_release_inherited",
                        item_id=item_id or tool_call.name,
                        tool=tool_call.name,
                        release_inherited_from=inherited,
                    )
                elif (
                    action_type == "external_action"
                    and (catalog_tool_name(tool_call.name) in preauthorised or any(
                        attempt_matches(a, catalog_tool_name(tool_call.name), {
                            k: v for k, v in args.items() if k not in PER_CALL_INJECTED_FIELDS
                        }) and attempt_current(a)
                        for a in spec.ix_context.get(CONTEXT_KEY, [])
                    ))
                ):
                    # Owner ruling 2026-09-08: the user authorised this act in
                    # the turn immediately before, on a beat where nothing was
                    # held — so there is no id the sideband could have answered
                    # and no ask this act is still waiting for. Asking again
                    # takes the same permission twice (spec anti-behaviour 7).
                    #
                    # Discarded here for the same reason `act_releases` is: one
                    # entry buys one call, and a second act on the tool in this
                    # response goes back through the floor. The turn-level copy
                    # is rebuilt per response, so a call the world refuses
                    # leaves the authorisation standing for the retry.
                    #
                    # The grant is named in catalog space throughout -- the
                    # sideband's spelling, and the one
                    # `ix_preauthorisation_expired` reports -- so the release
                    # and the lapse of the same grant read as the same tool.
                    act = catalog_tool_name(tool_call.name)
                    preauthorised.discard(act)
                    collector.emit(
                        "ix_release_preauthorised",
                        item_id=item_id or act,
                        tool=act,
                    )
                    # §3.7 (part 3): the judge counts authorisations off
                    # `consent[]`, and this one never passed through a hold, so
                    # the loop below (keyed on `held_ids`) would not record it.
                    # The tool name stands in for the item id, as it already
                    # does in `GateRecord` and `ix_release_inherited`.
                    if not any(
                        row.attribute == "autonomy_level"
                        and row.item_id == act
                        for row in collector.consent
                    ):
                        collector.consent.append(ConsentRecord(
                            attribute="autonomy_level",
                            item_id=act,
                            # The PA decided nothing: it never saw the sideband
                            # and was never holding the act.
                            pa_decision="keep",
                            sideband_decision="allow",
                            released=True,
                        ))
                else:
                    attribute = "autonomy_level"
                    reason = (
                        "external_action_floor"
                        if action_type == "external_action"
                        else f"{action_type}_gate"
                    )

            # §3.7 (part 3): `consent[]` is the record of one authorisation, so it
            # gets one row per (attribute, item) — not one per call carrying
            # the id. Live, `v3g02` t24 wrote four rows for a single `allow`
            # (one read, one write, two duplicates the spent-release check
            # refused), and a judge counting authorisations off `consent[]`
            # over-counted by four to one. What each individual call did is
            # already in `held.calls` and in the `ix_release_spent` notices.
            if item_id in held_ids and not any(
                row.attribute == held_ids[item_id] and row.item_id == item_id
                for row in collector.consent
            ):
                verdict = sideband_decision(spec.consent_decisions, item_id)
                pa_decision = next(
                    (
                        decision.decision
                        for decision in collector.payloads.release_decisions
                        if decision.id == item_id
                    ),
                    "keep",
                )
                evidence = next(
                    (
                        decision.user_evidence
                        for decision in collector.payloads.release_decisions
                        if decision.id == item_id
                    ),
                    "",
                )
                collector.consent.append(ConsentRecord(
                    attribute=held_ids[item_id],
                    item_id=item_id,
                    pa_decision=pa_decision,
                    sideband_decision=verdict,
                    released=not reason,
                    user_evidence=evidence,
                    evidence_verified=self._evidence_verified(
                        evidence, spec.ix_context.get("user_message", "")
                    ),
                    tool_changed=tool_changed,
                ))

            record = collector.record_call(
                tool_call,
                action_type=action_type,
                executed=not reason,
                status="ok" if not reason else "held",
            )
            if not reason:
                if item_id and action_type in WORLD_CHANGING_ACTION_TYPES:
                    collector.executed_item_ids.add(item_id)
                continue
            collector.skipped_call_ids.add(tool_call.id)
            collector.hold_call(
                record,
                attribute=attribute,
                reason=reason,
                plan_item_id=plan_item.id if plan_item is not None else "",
            )
            if collector.gate is None and reason not in _NO_GATE_HOLD_REASONS:
                collector.gate = GateRecord(
                    attribute=attribute,
                    item_id=item_id or record.tool_name,
                    reason=reason,
                    asked_at_turn=collector.turn_index,
                    action_type=action_type,
                )
                stopped = reason != "questions_gate"

        # The plan steps behind the held calls stay claimed by the open episode.
        # Without this the next turn's continuation looks like a brand new
        # binding, and the model is asked to re-declare a plan it already gave.
        held_ids_now = {item.interaction_item_id for item in collector.held}
        collector.gate_remaining = [
            item for item in collector.payloads.plan
            if held_ids_now & set(item.bound_item_ids)
        ]

    @staticmethod
    def _releases_without_calls(
        spec: AgentRunSpec, collector: TurnCollector
    ) -> list[tuple[str, str, str]]:
        """`(item_id, attribute, tool)` for every held id released without a call.

        Read off the open episodes rather than off `decisions`, so an id the PA
        invented is not mistaken for a hold, and so the attribute and tool the
        record needs come from the episode that is actually waiting.

        **Only an executed call satisfies a release.** `collector.calls` holds
        the withheld ones too, so counting them made the release/call diff empty
        whenever the wrapper itself re-held the call -- `v6g01` released `email1`
        eight times, issued it eight times, and got zero `ix_release_missing`
        while the email never went out. A call the wrapper refused this turn is
        not the missing call the repair is for: the refusal is already on the
        record as a held row, and asking the model to re-issue it would only get
        it held again -- so a held id is excluded here rather than reported.
        """
        released = {
            decision.id
            for decision in collector.payloads.release_decisions
            if decision.decision == "release"
        }
        if not released:
            return []
        executed = {
            call.interaction_item_id
            for call in collector.calls
            if call.interaction_item_id and call.executed
        }
        held_now = {
            item.interaction_item_id
            for item in collector.held
            if item.interaction_item_id
        }
        missing: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for attribute, episode in spec.hold.open_episodes.items():
            for call in episode.calls_held:
                item_id = str(call.get(INTERACTION_ITEM_ID_FIELD) or "")
                if (
                    item_id in released
                    and item_id not in executed
                    and item_id not in held_now
                    and item_id not in seen
                ):
                    seen.add(item_id)
                    missing.append((item_id, attribute, str(call.get("tool") or "")))
        return missing

    @staticmethod
    def _keep_unexecuted_releases_held(
        spec: AgentRunSpec, collector: TurnCollector
    ) -> None:
        """§3.7 (part 3): a release that never issued its call leaves the item held.

        Without this the write-back sees nothing held for the attribute and
        closes the episode, so the id the user authorised is gone -- live,
        `v3g03` lost `email_01` exactly this way after the repair would have had
        its second chance.
        """
        for item_id, attribute, tool in AgentRunner._releases_without_calls(
            spec, collector
        ):
            collector.emit("ix_release_missing", item_id=item_id, tool=tool)
            collector.held.append(HeldItem(
                status=held_status(attribute, "release_missing"),
                interaction_item_id=item_id,
                attribute=attribute,
                tool=tool,
                episode_id=spec.hold.open_episodes[attribute].episode_id,
                reason="release_missing",
            ))

    @staticmethod
    def _expire_preauthorisations(
        spec: AgentRunSpec, collector: TurnCollector
    ) -> None:
        """Owner ruling 2026-09-08: a standing authorisation lasts one PA turn.

        The user said "you may proceed" about the act it named, on the beat they
        said it. Nothing carries it forward: it is read from
        `consent_decisions`, which arrives with the user message, and it is
        never written into `IXHold`. So all that is left here is the notice --
        a tool the go-ahead covered whose act never went through this turn, so
        the next one is asked for in the ordinary way.
        """
        for tool in sorted(preauthorised_acts(
            spec.consent_decisions, collector.calls
        )):
            collector.emit("ix_preauthorisation_expired", tool=tool)

    @staticmethod
    def _hold_attribute(session_active: set[str]) -> str:
        """Which hold-bearing attribute owns a refusal the gate did not raise."""
        for attribute in (
            "autonomy_level",
            "topic_management",
            "information_elicitation",
            "capability_boundary",
        ):
            if attribute in session_active:
                return attribute
        return "autonomy_level"

    @staticmethod
    def _evidence_verified(evidence: str, user_message: Any) -> bool:
        """Whitespace-normalised substring check. Audit only -- never a gate."""
        if not evidence or not isinstance(user_message, str):
            return False
        normalize = lambda text: " ".join(text.split())  # noqa: E731
        return normalize(evidence) in normalize(user_message)

    @staticmethod
    def _attribute_engaged(spec: AgentRunSpec, attribute: str) -> bool:
        """True while one attribute governs this session at all.

        Three sources, because an attribute can be live before it is in the
        beat's `session_active_ix`: a forced seed (`--force-ix`) or a natural
        choice made on an earlier beat both sit in the session's selection
        cache, which is where `session_ix_selected` comes from. Used by the
        unbound-prose-ask nudge; capability detection no longer needs it.
        """
        context = spec.ix_context
        return (
            attribute in (context.get("session_active_ix") or ())
            or attribute in (context.get("session_ix_selected") or ())
            or any(
                selection.attribute == attribute
                for selection in spec.inherited_ix_selections
            )
        )

    @staticmethod
    def _reselected_attribute(
        spec: AgentRunSpec, tool_call: ToolCallRequest
    ) -> str | None:
        """The attribute of a selection call this turn no longer offers, or None.

        Three conditions, all needed (regress20 acc_027). It has to be a
        per-session selection tool -- an `IX_render_*` payload or
        `IX_capability_resolve` the turn does not offer is a different mistake
        and keeps its refusal. It has to be absent from *this* turn's tool view,
        or it is an ordinary call. And the session has to have the setting
        already, because that is what the answer says: with no selection
        recorded there is nothing to point the model at, so a name the wrapper
        never offered stays a refusal.
        """
        name = tool_call.name
        if name not in _SELECTION_TOOL_NAMES or name in set(spec.tools.tool_names):
            return None
        attribute = name[len(IX_TOOL_NAME_PREFIX):]
        recorded = {
            *(str(attr) for attr in (spec.ix_context.get("session_ix_selected") or ())),
            *(selection.attribute for selection in spec.inherited_ix_selections),
        }
        return attribute if attribute in recorded else None

    @staticmethod
    def _ix_layer_engaged(spec: AgentRunSpec) -> bool:
        """True while any attribute is active or already chosen this session."""
        context = spec.ix_context
        return bool(
            context.get("session_active_ix")
            or context.get("session_ix_selected")
            or spec.inherited_ix_selections
        )

    @staticmethod
    def _restore_voided_selection_fields(
        tool_calls: list[ToolCallRequest],
        voided_selections: dict[str, dict[str, Any]],
    ) -> None:
        """Refill a selection's blank reasoning from this turn's voided copy.

        Live regress19 acc_009 t8: the response carrying
        `IX_information_elicitation{setting: "Structured", evidence: "The user
        explicitly asked to be asked for the remaining parameters before
        drafting…", application: …}` also carried a task call, so §3.0 voided
        it whole; the pin then re-bought the selection and the model answered it
        with the bare `setting` the pin named. That empty payload is the one
        that banks, so the run recorded a setting chosen for no stated reason
        and `evidence: ""` came back on the watch list.

        The pinned selection still decides: its `setting` is untouched, and a
        setting it *changed* keeps its own blanks, because evidence justifies
        one setting and says nothing about another (`regress19r` t1 re-emitted
        the same selection as `Iterative`). Only blanks are filled, and only
        from what this same turn already wrote for that same setting.
        """
        for tool_call in tool_calls:
            voided = voided_selections.get(tool_call.name)
            if voided is None or not isinstance(tool_call.arguments, dict):
                continue
            if str(tool_call.arguments.get("setting") or "") != str(
                voided.get("setting") or ""
            ):
                continue
            for field in ("evidence", "application"):
                if not str(tool_call.arguments.get(field) or "").strip():
                    tool_call.arguments[field] = voided.get(field, "")

    @staticmethod
    def _dedupe_ix_selections(
        tool_calls: list[ToolCallRequest],
        ix_completed: set[str],
    ) -> tuple[list[ToolCallRequest], list[ToolCallRequest]]:
        """Keep the first `IX_<attribute>` selection of the turn, drop the rest.

        A selection is one decision, so a second copy — in the same response or
        in a later one, once the turn has banked the attribute in
        `ix_completed` — decides nothing and only doubles the record. Live
        `acc_026` t14 executed `IX_capability_boundary` twice off one pinned
        response (`ix_render_payload_forced`), and a repeat naming a *different*
        setting used to win silently, because `directive_for` reads the latest
        selection. First wins; the dropped **calls** are returned so the caller
        can record them — the memory pool's `[ix_selection
        outcome=ignored_duplicate]` block needs the call's own id and its full
        arguments (§3.3), which a `(name, setting)` pair cannot supply. Payload
        and continue tools are untouched: those legitimately repeat within a
        turn.
        """
        kept: list[ToolCallRequest] = []
        ignored: list[ToolCallRequest] = []
        seen = set(ix_completed)
        for tool_call in tool_calls:
            if tool_call.name in _SELECTION_TOOL_NAMES:
                if tool_call.name in seen:
                    ignored.append(tool_call)
                    continue
                seen.add(tool_call.name)
            kept.append(tool_call)
        return kept, ignored

    @staticmethod
    def _binding_contract_active(spec: AgentRunSpec) -> bool:
        """True once the registry injects `interaction_item_id` on task tools.

        The fail-closed binding check only applies where the contract is
        actually in force; the field injection itself belongs to the tool
        registry (plan §5.3 carrier C).
        """
        injected = getattr(spec.tools, "injected_per_call_fields", ())
        return INTERACTION_ITEM_ID_FIELD in injected

    @staticmethod
    def _directive_for_events(spec: AgentRunSpec, events: list[dict[str, Any]]) -> IXDirective:
        return directive_for([
            *spec.inherited_ix_selections,
            *selections_from_tool_events(events),
        ])

    @staticmethod
    def _tool_action_type(tool_name: str, ix_context: dict[str, Any]) -> str:
        return _ix_tool_action_type(tool_name, ix_context.get("tool_action_types"))

    @staticmethod
    def _process_events(
        directive: IXDirective,
        stage: str,
        tool_calls: list[ToolCallRequest],
    ) -> list[dict[str, Any]]:
        if stage == "start" and not directive.emit_start:
            return []
        if stage == "progress" and not directive.emit_progress:
            return []
        if stage == "completion" and not directive.emit_completion:
            return []

        names = [tool_call.name for tool_call in tool_calls]
        if stage == "progress":
            return [
                {
                    "process": True,
                    "visibility": directive.process_visibility,
                    "stage": "progress",
                    "tool_name": name,
                    "message": f"Executing {name}.",
                }
                for name in names
            ]
        return [{
            "process": True,
            "visibility": directive.process_visibility,
            "stage": stage,
            "tool_names": names,
            "message": (
                f"Starting task tool sequence: {', '.join(names)}."
                if stage == "start"
                else f"Completed task tool sequence: {', '.join(names)}."
            ),
        }]

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
    ) -> tuple[Any, dict[str, Any]]:
        # Carrier C fields are wrapper state, never tool arguments. The registry
        # already strips them off the call it forwards, but the event below is
        # what the transcript keeps, and it carried them verbatim: live
        # `acc_039` P12 shows `calendar_event_create` logged with an empty
        # `_ix_confirmation_message` among its real arguments. The wrapper reads
        # the fields off `tool_call.arguments` elsewhere, so they are dropped
        # here rather than on the call itself.
        arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        arguments = {
            key: value for key, value in arguments.items()
            if key not in PER_CALL_INJECTED_FIELDS
        }
        result = await spec.tools.execute(tool_call.name, arguments)
        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        error = _tool_result_error(result)
        event = {
            "name": tool_call.name,
            "status": (
                "capability_limit" if is_capability_limit(result)
                else "error" if error is not None
                else "ok"
            ),
            "args": arguments,
            "detail": detail,
        }
        if error is not None:
            event["error"] = error
        ix_payload = extract_ix_payload(tool_call.name, result)
        if ix_payload is not None:
            event["ix"] = True
            event.update(ix_payload)
        return result, event
