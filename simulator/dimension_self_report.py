"""Dimension self-report: the same actor rates, after a finished test probe,
the one attribute the probe tests — a 1-5 score and a first-person reason.
`scripts/self_report_completed_run.py` runs it over a run's final probes; the
mean is the run's PRS (Perceived Realization Score).

The prompt text is `prompts/dimension_self_report/`. This module rebuilds the
blocks the actor is given from a finished probe's recorded data with the live
section renderers — never with the whole-prompt assemblers, which also pull in
material the actor is not given here (the message-writing template, profile
memory, the naturalness guide, the per-turn instructions) — fills them in,
calls the simulator model the way the session-end self-report does, and
parses the one block it asks for.

Where the live actor's input had small slips, the rebuild corrects them; the
conversation itself is never changed:

- the unplayed opening of a triggered (proactive) probe is marked `(skipped)`
  in the outline when it stands before the played one, and is left out of the
  "Then N more step(s)" count when it stands after it (live showed `(done)` and
  one more step to come);
- the private notes' tool-call count leaves out the wrapper's progress-notice
  records (`"process": true`, no tool name), which live counted;
- every delivered PA reply gets its private notes, the last one included (live
  never showed that one: the conversation ended there); an undelivered
  proactive turn gets none, which would reveal the IX choice;
- an item held for approval shows only its description, never the wrapper's
  internal id, and an item without one shows `HELD_ITEM_WITHOUT_DESCRIPTION`.

The harness-side per-turn facts (`reply_facts`, `held_items`) are derived from
`ix_render` by the caller, exactly as `harness/pa_factory.py` derived them
live; this package never reads `ix_render` (`simulator/` does not import
`pa/` or `harness/`).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from simulator.preferences import apply_preference_patches
from simulator.profiles import (
    BOOTSTRAP_FILES,
    GROUND_RULES_PATH,
    ROLE_FRAME_PATH,
    SimulatorProfile,
)
from simulator.prompting import (
    gated_fact_keys,
    reception_already_taken,
    reception_carry_attributes,
    render_active_skills_block,
    render_beat_block,
    render_character_card,
    render_held_items_block,
    render_life_context_block,
    render_session_outline,
    render_task_facts,
)
from simulator.providers.base import LLMResponse
from simulator.reply_facts import reply_facts_from_pa_turn, render_reply_facts_block
from simulator.schemas import PreferenceMatrix, SessionScript
from simulator.self_report import (
    SELF_REPORT_MAX_TOKENS,
    SessionEndSelfReporter,
    _render_transcript,
)

PROMPT_DIR = Path(__file__).parent / "prompts" / "dimension_self_report"
PLACEHOLDERS = (
    "CHARACTER_CARD", "ACTING_SHEET", "LIFE_CONTEXT", "SESSION_SETUP",
    "CONVERSATION", "PRIVATE_NOTES", "EXAMPLE", "QUESTION", "ANCHORS",
)
NOTES_HEADER = "private notes after turn {index}:"
HELD_ITEM_WITHOUT_DESCRIPTION = "- an item (no description given)"
#: Wall-clock ceiling on one provider call; keep-alive bytes can keep resetting
#: the HTTP read timeout, so this is the one deadline that always fires.
CALL_DEADLINE_S = 600
#: Consecutive empty completions tolerated per attempt, as in `self_report._call`.
EMPTY_RETRIES = 5

#: The only recorded fields the rebuild reads. Everything else in a transcript
#: — IX calls and choices, tool arguments and results, judge calls, the actor's
#: private blocks and its earlier self-report, the environment trigger — is
#: dropped before the replay sees the event.
_READ_FIELDS = {
    "pa_turn": ("t", "outbound", "proactive_event", "outreach_delivered"),
    "simulator_turn": ("t", "beat", "message"),
    "beat_enter": ("beat",),
    "outreach_path": ("skipped_beat",),
}
#: `error` is read as a flag only: its text (the tool's own failure detail) is
#: never rendered, so the whitelist does not carry it.
_READ_TOOL_EVENT_FIELDS = ("name", "ix", "ix_event", "error", "process")
#: Events the live loop reacts to in ways this rebuild does not reproduce.
_UNSUPPORTED_EVENTS = ("beat_skipped", "skill_triggered")

_REPLY_FACTS_FIRST_LINE = "Private record of what the assistant's latest reply actually did"
_REPLY_FACTS_LAST_LINE = "A change the reply describes but this record does not list"
_BLOCK_RE = re.compile(r"<dimension_self_report>(.*?)</dimension_self_report>", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"<<([A-Z_]+)>>")


class RebuildError(ValueError):
    """The recorded data does not rebuild into what the actor was given."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RebuildError(message)


# ------------------------------------------------------------- prompt files


@lru_cache(maxsize=1)
def load_prompt_files() -> tuple[str, dict[str, Any]]:
    """The frame and the per-attribute question data, read once."""
    frame = (PROMPT_DIR / "frame.md").read_text(encoding="utf-8")
    attributes = yaml.safe_load((PROMPT_DIR / "attributes.yaml").read_text(encoding="utf-8"))
    return frame, attributes


def prompt_files_sha256() -> str:
    """One hash over the prompt directory: sha256 of `name\\0bytes\\0` for
    `attributes.yaml`, then `frame.md`."""
    digest = hashlib.sha256()
    for name in ("attributes.yaml", "frame.md"):
        digest.update(name.encode() + b"\0" + (PROMPT_DIR / name).read_bytes() + b"\0")
    return digest.hexdigest()


def build_prompt(blocks: Mapping[str, str], attribute: str, setting: str) -> str:
    """Fill the frame in one pass, so text inside a block is never re-read as
    a placeholder. `setting` is the probe's true setting; the acting sheet must
    have been rendered for the same one."""
    frame, attributes = load_prompt_files()
    entry = attributes[attribute]
    _require(
        f"## Skill: {attribute} = {setting}\n" in blocks["ACTING_SHEET"],
        f"acting sheet is not {attribute} = {setting}",
    )
    fills = {
        **blocks,
        "EXAMPLE": entry["example"],
        "QUESTION": entry["question"].replace("{who}", entry["who"][setting]),
        "ANCHORS": "\n".join(f"{score}: {text}" for score, text in entry["anchors"].items()),
    }
    _require(
        set(_PLACEHOLDER_RE.findall(frame)) == set(PLACEHOLDERS) == set(fills),
        "placeholder set",
    )
    return _PLACEHOLDER_RE.sub(lambda match: fills[match.group(1)], frame)


# ------------------------------------------------------------------ rebuild


def character_card(profile: SimulatorProfile) -> str:
    """`SimulatorProfile.build_context()` without the profile memory, which
    the actor is not given here. Same parts, same joins."""
    _require(not profile._read_profile_skills(), "profile skills present")
    role_frame = ROLE_FRAME_PATH.read_text(encoding="utf-8").strip().replace(
        "<<DISPLAY_NAME>>", profile.display_name
    )
    parts = [
        f"## Identity\n\n{role_frame}\n\n{render_character_card(profile.load_identity())}",
        "## Ground rules\n\n" + GROUND_RULES_PATH.read_text(encoding="utf-8").strip(),
    ]
    for filename in BOOTSTRAP_FILES:
        path = profile.root / filename
        if path.exists():
            parts.append(f"## {filename}\n\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n---\n\n".join(parts)


def _read(event: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = event.get("kind")
    if kind in _UNSUPPORTED_EVENTS:
        raise RebuildError(f"{kind} in transcript")
    if kind not in _READ_FIELDS:
        return None
    kept = {"kind": kind, **{key: event[key] for key in _READ_FIELDS[kind] if key in event}}
    if kind == "pa_turn":
        kept["tool_events"] = [
            {
                key: bool(item[key]) if key == "error" else item[key]
                for key in _READ_TOOL_EVENT_FIELDS
                if key in item
            }
            for item in event.get("tool_events") or []
        ]
    return kept


def _outline(script: SessionScript, beat, skipped_opening: str | None) -> str:
    """The session outline as the actor last saw it, with a triggered probe's
    unplayed opening marked skipped (before the played one) or left out
    (after it)."""
    if skipped_opening:
        order = [item.beat for item in script.beats]
        if order.index(skipped_opening) < order.index(beat.beat):
            script = script.model_copy(
                update={"skipped_beats": [*script.skipped_beats, skipped_opening]}
            )
        else:
            script = script.model_copy(
                update={"beats": [b for b in script.beats if b.beat != skipped_opening]}
            )
    return render_session_outline(script, beat)


def _held_lines(held_items: list[dict[str, str]]) -> list[str]:
    """The held-items block without its last line, which tells the actor how
    to answer; each item by its description only."""
    block = render_held_items_block(held_items)
    if not block:
        return []
    return [block.split("\n")[0]] + [
        f"- {intent}" if intent else HELD_ITEM_WITHOUT_DESCRIPTION
        for intent in (str(item.get("intent") or "").strip() for item in held_items)
    ]


def _notes(index: int, facts, held_items: list[dict[str, str]]) -> str:
    """The reply-facts lines and the held items under a fixed header. The
    reply-facts block's first and last lines are turn instructions to the
    actor, not facts, and are dropped."""
    facts_lines = render_reply_facts_block(facts).split("\n")
    _require(
        facts_lines[0].startswith(_REPLY_FACTS_FIRST_LINE)
        and facts_lines[-1].startswith(_REPLY_FACTS_LAST_LINE),
        "reply-facts layout",
    )
    return "\n".join([NOTES_HEADER.format(index=index), *facts_lines[1:-1], *_held_lines(held_items)])


@dataclass(frozen=True)
class Rebuilt:
    blocks: dict[str, str]
    #: The visible conversation, as the live `_history` held it (with `t`).
    history: list[dict[str, Any]]
    setting: str


def rebuild_blocks(
    events: Iterable[Mapping[str, Any]],
    *,
    script: SessionScript,
    session_clock: datetime,
    profile: SimulatorProfile,
    preferences: PreferenceMatrix,
    recorded_overlay: Mapping[str, Any] | None,
    derived: Mapping[int, Mapping[str, Any]],
    attribute: str,
    specs_root: Path,
) -> Rebuilt:
    """Replay one finished probe's transcript the way `SimulatorLoop` walked
    it and return the six blocks the actor is given.

    `script` is the probe draft as live loaded it; `preferences` the run-level
    matrix and `recorded_overlay` the step's recorded `preference_overlay`
    (never recomputed); `derived` maps each `pa_turn` `t` to the harness keys
    `reply_facts` and `held_items` live derived from its `ix_render`.
    """
    script = script.model_copy(deep=True)
    script.session_clock = session_clock
    # A skill the PA's turn triggers loads its rules from the actor's next turn
    # on (`SimulatorLoop._absorb_triggered_skills`); after the actor's last
    # turn it changed nothing the actor was given. Earlier, it is refused.
    events = list(events)
    last_turn = max((i for i, e in enumerate(events) if e.get("kind") == "simulator_turn"), default=-1)
    events = [e for i, e in enumerate(events) if not (e.get("kind") == "skill_triggered" and i > last_turn)]
    matrix = apply_preference_patches(preferences, recorded_overlay or None)
    beats = {beat.beat: beat for beat in script.beats}
    compressed_reaction_index = 1 if script.proactive_trigger else 2
    history: list[dict[str, Any]] = []
    notes: list[str] = []
    beat_blocks: list[str] = []
    executed: list[str] = []
    outreach_sent = False
    skipped_opening: str | None = None
    joined: list[str] = []
    beat = None
    beats_entered = 0
    first_turn = False
    sheet = outline = facts_block = None
    for event in filter(None, map(_read, events)):
        kind = event["kind"]
        if kind == "pa_turn":
            proactive = bool(event.get("proactive_event"))
            delivered = bool(event.get("outreach_delivered")) if proactive else True
            _require(event["t"] in derived, f"no harness keys for pa_turn t={event['t']}")
            harness = derived[event["t"]]
            tool_events = event["tool_events"]
            facts = reply_facts_from_pa_turn(
                [e for e in tool_events if not e.get("process")],
                harness.get("reply_facts"),
                earlier_tools=executed,
                earlier_outreach_sent=False if proactive else outreach_sent,
                outreach_delivered=proactive and delivered,
            )
            held = list(harness.get("held_items") or [])
            if proactive:
                _require(delivered or not held, "suppressed outreach holds items")
                outreach_sent = outreach_sent or delivered
            else:
                executed.extend(
                    str(e.get("name") or "")
                    for e in tool_events
                    if not e.get("ix_event") and not e.get("error")
                )
            if delivered:
                history.append(
                    {"role": "user", "content": event.get("outbound") or "", "t": event["t"]}
                )
                notes.append(_notes(len(history) - 1, facts, held))
        elif kind == "outreach_path":
            skipped_opening = event["skipped_beat"]
        elif kind == "beat_enter":
            beat = beats[event["beat"]]
            beats_entered += 1
            first_turn = True
            script.session_reception_carry = list(joined)
            joined = []
            for attr in beat.active_skills or []:
                if attr not in script.session_reception_skills:
                    script.session_reception_skills.append(attr)
                joined.append(attr)
            script.compressed_reaction_beat = (
                beat.beat
                if script.compressed and beats_entered == compressed_reaction_index
                else None
            )
        elif kind == "simulator_turn":
            _require(beat is not None and event.get("beat") == beat.beat, "turn outside its beat")
            if first_turn:
                beat_blocks.append(
                    render_beat_block(
                        beat,
                        first_turn_in_beat=True,
                        reception_already_taken=reception_already_taken(script, beat),
                        on_compressed_reaction_beat=script.compressed_reaction_beat == beat.beat,
                    )
                )
                if attribute in (beat.active_skills or []):
                    _require(beat.active_skills == [attribute], "declaring beat has other skills")
                    _require(
                        not reception_carry_attributes(script, beat),
                        "reception carry on the sheet",
                    )
                    sheet = render_active_skills_block(matrix, script, beat, specs_root)
            outline = _outline(script, beat, skipped_opening)
            facts_block = render_task_facts(
                script.task_facts,
                history=history or None,
                hidden_keys=gated_fact_keys(script, beat),
            )
            history.append(
                {"role": "assistant", "content": event.get("message") or "", "t": event["t"]}
            )
            first_turn = False
    _require(sheet is not None, f"no played beat declares {attribute}")
    _require(bool(notes), "no delivered PA reply")
    return Rebuilt(
        blocks={
            "CHARACTER_CARD": character_card(profile),
            "ACTING_SHEET": sheet,
            "LIFE_CONTEXT": render_life_context_block(script),
            "SESSION_SETUP": "\n\n".join([outline, *beat_blocks, facts_block]),
            "CONVERSATION": _render_transcript(history),
            "PRIVATE_NOTES": "\n\n".join(notes),
        },
        history=history,
        setting=matrix.matrix[script.context][attribute].setting,
    )


# ------------------------------------------------------------------- parser


@dataclass(frozen=True)
class ParsedReport:
    score: int | None
    reason: str | None
    findings: tuple[str, ...]
    #: Text outside the block: recorded, not re-asked.
    outside_text: bool


def parse_report(raw: str) -> ParsedReport:
    """Exactly one block holding a `score:` line (a digit 1-5) and then a
    `reason:` line. Counted with findall, so a copied example is caught."""
    found = _BLOCK_RE.findall(raw)
    outside = bool(_BLOCK_RE.sub("", raw).strip())
    if len(found) != 1:
        code = "dimension_self_report_missing" if not found else "dimension_self_report_repeated"
        return ParsedReport(None, None, (code,), outside)
    lines = [line.strip() for line in found[0].splitlines() if line.strip()]
    if len(lines) != 2 or not lines[0].startswith("score:") or not lines[1].startswith("reason:"):
        return ParsedReport(None, None, ("dimension_self_report_malformed",), outside)
    score = lines[0][len("score:"):].strip()
    reason = lines[1][len("reason:"):].strip()
    if len(reason) >= 2 and reason[0] == reason[-1] == '"':
        reason = reason[1:-1].strip()
    findings = tuple(
        code
        for code, bad in (
            ("score_not_1_to_5", score not in {"1", "2", "3", "4", "5"}),
            ("reason_empty", not reason),
        )
        if bad
    )
    if findings:
        return ParsedReport(None, None, findings, outside)
    return ParsedReport(int(score), reason, (), outside)


# ------------------------------------------------------------------- caller


class _Stop(Exception):
    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


class DimensionSelfReporter:
    """One rating: the session-end self-report's provider, model and params
    (`tools=[]`, `SELF_REPORT_MAX_TOKENS`, the provider's own temperature),
    its correction message and its empty-reply retry.

    Give each probe its own provider instance: the per-call cost is read off
    the provider's `_extract_cost`, which is wrapped on the instance.
    """

    def __init__(self, provider: Any, *, model: str, output_contract_retries: int) -> None:
        if output_contract_retries < 0:
            raise ValueError("output_contract_retries must be non-negative")
        if "_dsr_costs" in vars(provider):
            raise ValueError("provider is already metered; use one provider per probe")
        extract = provider._extract_cost
        costs: list[Any] = []

        def metered(provider_response: Any) -> Any:
            cost = extract(provider_response)
            costs.append(cost)
            return cost

        provider._extract_cost = metered
        provider._dsr_costs = costs
        self.provider = provider
        self.model = model
        self.output_contract_retries = output_contract_retries
        self._costs = costs

    async def generate(self, prompt: str) -> dict[str, Any]:
        """Status is `ok`, `contract_exhausted` (every attempt unparseable),
        `empty_output` (only empty completions) or `provider_error` (an error
        response or the wall-clock deadline)."""
        messages = [{"role": "system", "content": prompt}]
        calls: list[dict[str, Any]] = []
        parsed: ParsedReport | None = None
        status = "contract_exhausted"
        for attempt in range(1, self.output_contract_retries + 2):
            try:
                response = await self._call(messages, calls, attempt)
            except _Stop as stop:
                status = stop.status
                break
            parsed = parse_report(response.content)
            calls[-1].update(findings=list(parsed.findings), outside_text=parsed.outside_text)
            if not parsed.findings:
                status = "ok"
                break
            messages = SessionEndSelfReporter._correction_messages(
                messages, response.content, list(parsed.findings)
            )
        known = [call["cost_usd"] for call in calls if call["cost_usd"] is not None]
        return {
            "status": status,
            "score": parsed.score if status == "ok" else None,
            "reason": parsed.reason if status == "ok" else None,
            "calls": calls,
            "usage_total": {
                key: sum(int(call["usage"].get(key) or 0) for call in calls)
                for key in ("prompt_tokens", "completion_tokens")
            },
            "cost_usd_total": sum(known),
            "cost_missing_calls": len(calls) - len(known),
        }

    async def _call(
        self,
        messages: list[dict[str, str]],
        calls: list[dict[str, Any]],
        attempt: int,
    ) -> LLMResponse:
        for empty in range(EMPTY_RETRIES):
            before = len(self._costs)
            started = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    self.provider.chat_with_retry(
                        messages=messages,
                        tools=[],
                        model=self.model,
                        max_tokens=SELF_REPORT_MAX_TOKENS,
                    ),
                    CALL_DEADLINE_S,
                )
            except asyncio.TimeoutError:
                response = LLMResponse(content=None, finish_reason="timeout")
            new = self._costs[before:]
            calls.append({
                "attempt": attempt,
                "raw": response.content,
                "finish_reason": response.finish_reason,
                "usage": dict(response.usage or {}),
                "cost_usd": None if not new or None in new else float(sum(new)),
                "seconds": round(time.monotonic() - started, 3),
                "findings": [],
                "outside_text": None,
            })
            if response.finish_reason in {"error", "timeout"}:
                raise _Stop("provider_error")
            if response.content:
                return response
            await asyncio.sleep(3 * (empty + 1))
        raise _Stop("empty_output")
