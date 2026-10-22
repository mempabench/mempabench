"""File-backed run-local task state for the MemPABench MCP server.

Every tool handler is a deterministic function of (sandbox state, arguments).
Writes persist to real store files; readbacks read the written state. There is
no scripted-case layer and no call-count logic.

Date visibility: when MEMPA_SESSION_DATE is set, read tools hide records whose
arrival/creation date is after the session date. MEMPA_SESSION_TIMESTAMP adds
the session's hour, so a stamp written `YYYY-MM-DDTHH:MM` is hidden from an
earlier session of the same day (`stamp_visible`); a date-only stamp keeps its
day-granular meaning. Visibility keys on when a record came into existence,
never on the date it is about.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple


_ACTOR_HIDDEN_ROOTS = {"authoring", "scripted"}
# `gate_open` and `gate_satisfied` join the two audit files: they are harness
# and store bookkeeping sitting in the state root (see `GATE_OPEN_MARKER`,
# `GATE_SATISFIED_MARKER`), and a PA that could read them would be reading
# whether its own refusal is about to lift, or has already lifted.
_ACTOR_HIDDEN_FILES = {
    "state_diff.jsonl",
    "tool_log.jsonl",
    "gate_open",
    "gate_satisfied",
    "session_closing",
}

# Session gate (2026-09-08 owner ruling). A gated session's write payload is
# the user's own words, said only at the probe beat. Until then the store
# refuses the call, and the refusal is the message the PA reads: it has to say
# "ask the user", because a PA told only that the value is wrong invents a
# better-looking one (live acc_039 wrote its own `reason` three rounds running).
# `MEMPA_GATE` carries {tool, param, task_fact, dictated_text, key_terms,
# reveal_beat, pre_reveal_write_tools} per step.
GATE_REFUSAL = (
    "{param} for {tool} must be the user's own words, and the user has not "
    "given them in this conversation. Do not compose it yourself: ask the user "
    "for it and stop."
)
#: Refusal once the gate is open. By then the user *has* spoken, so the closed
#: gate's "has not given them" is a falsehood the PA has to reconcile against
#: the transcript: live acc_039 t52 refused a paraphrase of a reason the user
#: had dictated at t15, and the PA went back asking for the words "as stated in
#: this conversation" -- correct behaviour reached through a false premise.
#: What failed there is the match, not the giving, so this says so and still
#: ends at the user: the wording is theirs to give, not the PA's to improve.
#: Like `GATE_REFUSAL` it never quotes the dictated text or the key terms -- a
#: refusal that showed the answer would be the gate handing over what it guards.
GATE_MISMATCH_REFUSAL = (
    "{param} for {tool} must carry what the user told you for it, in the words "
    "the user used in this conversation -- this value does not. Ask the user to "
    "give or confirm it, and stop."
)
#: Refusal for a write on the gate tool AFTER the gate is satisfied that would
#: leave the artifact without the user's information (owner ruling A,
#: 2026-09-08). The parameter used to be fully open from the first matching
#: write, and regress16/17/18 acc_009 spent that opening: two free revisions of
#: the same draft dropped "line-by-line", so the saved artifact no longer held
#: the focus the user had dictated. The refusal is worded at the artifact, not
#: at the value, because a write that changes something else and keeps the words
#: is legal -- and it leaks neither the sentence nor the terms, like the other
#: two.
GATE_DROPS_TERMS_REFUSAL = (
    "{tool} refused: {param} must keep what the user told you for it, in the "
    "words the user used in this conversation, and this write would drop them. "
    "Keep the user's words as given, or ask the user."
)
#: The gate is keyed on INFORMATION, not on words (owner, 2026-09-08). The
#: outline authors `gate_key_terms` off the dictated sentence, and a value
#: passes when it carries every one of them; a PA that reworded the sentence
#: around the terms has still asked, and one that dropped a term has not. Two is
#: the floor for hand-picked terms because a single one of those is a word, not
#: the information -- live acc_009 turned on three filler tokens under the old
#: 0.7 token ratio, which this replaces.
#:
#: Exactly one single term escapes the floor: the whole dictated text itself
#: (owner, 2026-09-14). The probe line authors no terms -- `harness/gate.py:
#: _probe_gate_wording` has only the sentence the user says -- so it names that
#: sentence entire as the one term. That is the opposite of a word standing in
#: for the information: the payload has to carry all of it, which is stricter
#: than any pair an author could pick out of the same sentence, never looser.
GATE_MIN_KEY_TERMS = 2

#: Number words the key-term normalisation reads as their digits, so "three
#: weeks" and "3 weeks" are the same information (owner ruling 6, 2026-09-08).
#: regress16 acc_039 refused "began approximately 3 weeks ago" against a term of
#: "three weeks": the PA had asked, said what the user said, and lost the gate to
#: a spelling. Only standalone words are mapped -- "twenty-one" is two words
#: after punctuation becomes a break, and normalises to "20 1", which is what a
#: payload writing it the same way normalises to as well.
NUMBER_WORDS: dict[str, str] = {
    word: str(value)
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
        "twenty".split()
    )
}
NUMBER_WORDS.update(
    {
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
        "hundred": "100",
    }
)

#: A comma with a digit on either side groups the digits of one number. It is
#: dropped before the fold, not turned into a space, so "1,200" and "1200" are
#: the same word on both sides of the comparison.
DIGIT_GROUP_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

#: Endings a key term's word may wear in the payload, and the payload's word in
#: the term, without being a different word (owner, 2026-09-08). regress19
#: acc_009 refused a body that said everything the user had dictated because the
#: PA wrote "hardest" against a term of "hard"; a round earlier "I confirm
#: receipt" lost to "Confirming receipt". The gate is keyed on the information,
#: and an ordinary inflection carries the same information.
#:
#: Inflections only, and no "-ly": that suffix is the one whose acceptance would
#: make "hardly" carry "hard", which is a different word. Nothing here is a
#: stemmer -- a word that merely BEGINS with the term (hardware, harden) never
#: matches it.
GATE_INFLECTION_SUFFIXES = ("s", "es", "ed", "d", "ing", "er", "est")

#: NOT the gate match rule any more -- `scripts/validate_normal_event.py` is the
#: one remaining importer, where it is the authoring-time guessability floor
#: (how much of the dictated text the world already hands over). It goes when
#: that check moves to the key terms.
GATE_TOKEN_RATIO = 0.7

#: How much of a dictated payload `tool_log.jsonl` keeps. The row used to record
#: every `appointment_create` argument except `reason` -- the one argument the
#: gate is about -- so a finished run could not say whether the PA had asked the
#: user or written the words itself. A length is not evidence; the words are.
GATE_LOG_VALUE_LIMIT = 200

#: The same rule one level out: how many matched record ids a search row names
#: in `result_summary.ids` before it says `truncated: true` instead. A count on
#: its own cannot be attributed to anything -- live acc_096's row was
#: `contact_search {"query": "Sarah Bennett"} -> {"matches": 1}`, so the
#: finished run never said the record was `ct_0012`, and nothing could tell
#: that fin_28's probe object had already been handed to the timeline PA.
SEARCH_LOG_ID_LIMIT = 20

#: Every field of `MEMPA_GATE` that has to carry text. A blank one is not a
#: weaker gate, it is a gate that silently enforces nothing -- an empty `tool`
#: would guard no tool, an empty `reveal_beat` would never shut -- so each is
#: checked at init rather than discovered in a live round. `dictated_text` is
#: no longer compared against the payload (the key terms are), but it stays
#: required: it is what the terms were authored off, the dry-resolve check in
#: `scripts/validate_normal_event.py` holds the draft to it, and a finished
#: run's `meta.yaml` gate block is unreadable without it.
GATE_REQUIRED_TEXT_KEYS = ("tool", "param", "task_fact", "dictated_text", "reveal_beat")

#: `tool_log.jsonl` row written once per step when the server receives a gate.
#: It is the only proof a finished run offers that the gate reached the
#: subprocess at all; without it, a step whose gate never arrived is
#: indistinguishable from a step the PA passed honestly.
GATE_LOADED_TOOL = "gate_loaded"

#: The harness touches this file in the state root when the session enters the
#: beat at which the user says the dictated text (`harness/gate.py`). Until it
#: exists the gate is *closed*, and closed is the default: the text check alone
#: left two ways through. A value the user leaked before the probe beat, or one
#: the PA happened to compose close enough, opened the gate early; and a sibling
#: write tool reached the same outcome without touching the gate at all
#: (acc_039: `calendar_event_create` a checkup and call the task done).
GATE_OPEN_MARKER = "gate_open"

#: The store writes this next to `gate_open` when a gated write first carries
#: every key term. From then on the PA may write that parameter in its own
#: words -- it has asked, and making it re-quote the user on every later write
#: is what ate acc_009's reaction beat -- but the ARTIFACT has to keep carrying
#: the information (`_enforce_gate_keeps_terms`, owner ruling A). It is a file
#: rather than an attribute because the MCP server builds a fresh store per tool
#: call, and `harness.state_runtime` clears it with the other markers on step
#: entry and exit so it cannot unlock a later session. The same name is the
#: `tool_log.jsonl` row that records the moment, the way `gate_loaded` records
#: the gate arriving.
GATE_SATISFIED_MARKER = "gate_satisfied"

#: Gate tools that write INTO an existing record, so the artifact after the
#: write is not the value the call passed: `document_update` substitutes one
#: span of a file, and `calendar_event_update` / `contact_update` leave every
#: field they were not given. For these `_enforce_gate` cannot judge ruling A
#: from the payload, so their handlers call `_enforce_gate_keeps_terms`
#: themselves, at the point the merged record exists and before it is
#: persisted. Every other gate tool creates a record whose gated field is
#: exactly what the call passed, and is checked at the seam.
GATE_MERGING_TOOLS = frozenset(
    {"document_update", "calendar_event_update", "contact_update"}
)

#: Refusal for the second lock -- a write tool that is not the gate tool and
#: that this draft does not call before the reveal. It says NOTHING about the
#: gate (owner ruling B, 2026-09-08): it used to name the gate tool and param,
#: and the call it refuses is by definition one the PA improvised, so that text
#: announced a step the session had not reached. regress18 acc_011 took three
#: of these inside one beat-1 turn and asked the user to dictate the file four
#: beats early. `GATE_REFUSAL` -- lock 1, on the gate tool itself -- still names
#: what it needs, because there the ask is the only way forward.
GATE_CLOSED_WRITE_REFUSAL = (
    "{tool} is not open yet on this task. Ask the user what they want done "
    "before writing anything, and stop."
)

#: The harness touches this file in the state root once a COMPRESSED session's
#: reaction beat has delivered its user turn to the PA (`harness/gate.py`), so
#: it is down for exactly one PA turn: the last one of the session. From then on
#: every write tool is refused. Rule E says the reaction beat never asks for a
#: product, so no beat of such a session can want a write there -- and nothing
#: scores that turn either, because the session is over the moment it ends. Live
#: regress19 `acc_035` t8 spent it executing an offer the user had not answered:
#: two roster writes that invented an 8 -> 9 change and corrupted a note.
#: A file rather than an attribute for the same reason as `gate_open`: the MCP
#: server builds a fresh store per tool call, and `harness.state_runtime` clears
#: it with the gate markers on step entry and exit.
SESSION_CLOSING_MARKER = "session_closing"

#: Refusal for a write on that final turn. Neutral by construction: it names no
#: attribute and no setting, because the turn it refuses is the one the session
#: measures, and a refusal that said "you are being watched on proactivity"
#: would be the harness coaching the answer. It says the only true thing there
#: is to say -- the conversation is ending and nothing was asked for -- and ends
#: at the user, like the gate refusals.
SESSION_CLOSING_WRITE_REFUSAL = (
    "{tool} cannot run: this conversation is at its close and nothing the user "
    "asked for remains to be written. Reply to the user, and stop."
)

#: Every write tool in `data/state_tools/catalog.yaml` -- `allowed_effects`
#: carrying `external_write` or `internal_write`, the same rule
#: `scripts/validate_normal_event.py:_tool_effect` applies -- and the store
#: method each one runs. Six tools are named differently on the store than in
#: the catalog, which is why this is a map and not a set.
#:
#: The catalog is repeated here rather than read, because the store runs as an
#: MCP subprocess whose working directory is not the project root -- the same
#: reason the capability-limit literals above are repeated instead of imported.
#: `tests/test_state_server_gate.py` asserts these keys equal the catalog's
#: write tools, so a new write tool cannot quietly escape the closed gate.
WRITE_TOOL_METHODS: dict[str, str] = {
    "appointment_create": "appointment_create",
    "appointment_update": "appointment_update",
    "bill_payment_create": "bill_payment_create",
    "calendar_event_create": "calendar_event_create",
    "calendar_event_update": "calendar_event_update",
    "cinema_ticket_purchase": "buy_cinema_tickets",
    "conference_registration_create": "conference_registration_create",
    "contact_create": "contact_create",
    "contact_update": "contact_update",
    "corporate_reporting_submit": "corporate_reporting_submit",
    "document_create": "create_document",
    "document_delete": "delete_document",
    "document_update": "update_document",
    "email_draft_create": "save_email_draft",
    "email_message_send": "send_email",
    "exhibition_ticket_purchase": "exhibition_ticket_purchase",
    "flight_book": "flight_book",
    "grade_submission_create": "grade_submission_create",
    "hotel_book": "hotel_book",
    "hotel_booking_update": "hotel_booking_update",
    "message_send": "message_send",
    "media_transcode_job_create": "media_transcode_job_create",
    "phone_call_dial": "phone_call_dial",
    "reimbursement_submission_create": "reimbursement_submission_create",
    "restaurant_reservation_create": "restaurant_reservation_create",
    "restaurant_reservation_update": "restaurant_reservation_update",
    "room_booking_create": "room_booking_create",
    "room_booking_update": "room_booking_update",
    "shopping_order_create": "shopping_order_create",
    "shopping_order_update": "shopping_order_update",
    "shopping_return_case_update": "shopping_return_case_update",
    "utility_account_holder_update": "utility_account_holder_update",
}


# Capability-limit contract.
# The frozen source of these four values is pa/agent/ix/render_types.py:
# CAPABILITY_LIMIT_ERROR_CODE, CAPABILITY_LIMIT_TOOL_STATUS,
# CAPABILITY_LIMIT_RESULT_KEYS and the BoundaryKind Literal (whose members are
# CapabilityLimitResult's allowed boundary_kind values). state_server/ imports
# nothing from pa/, so the literals are repeated here and
# tests/test_state_server_capability_limit.py asserts the two copies stay equal.
CAPABILITY_LIMIT_ERROR_CODE = "capability_limit"
CAPABILITY_LIMIT_TOOL_STATUS = "capability_limit"
CAPABILITY_LIMIT_RESULT_KEYS = ("ok", "error_code", "boundary_kind")
BOUNDARY_KINDS = (
    "missing_data",
    "missing_capability",
    "action_not_permitted",
    "external_dependency",
)

# Fixture-only control keys on a resource record. A record carrying
# `unavailable: true` is one the fixture deliberately withholds: searches still
# list it, but reading/retrieving that resolved id returns the capability-limit
# result instead of the record. The keys never reach the model.
_CAPABILITY_MARKER_KEYS = ("unavailable", "boundary_kind", "unavailable_reason")

# Run bookkeeping stamped onto every record this store creates. It belongs on
# disk and in `tool_log.jsonl` -- that is how a record is attributed to a run --
# but it must never travel back to the model in a tool result. Live, the PA
# created an appointment and read it back, and the reply carried
# `"run_id": "v2g03"` into the provider payload: a descriptive run id would have
# named the setting under test. `strip_run_identity` is applied at the one seam
# every model-facing result crosses (`state_server/server.py`), so the harness's
# own reads keep the full record.
_RUN_IDENTITY_KEYS = ("run_id", "user_id", "session_id")

# World-visibility bookkeeping, the same kind of leak one level down. A fixture
# record carries `visible_from` (see `_date_visible`) so the store can hide it
# before its story date; the model has no use for it and it says out loud that
# the world is staged -- live, `restaurant_reservation_read` handed back
# `reservation.visible_from` and `comic_store_inventory_read` put
# `matches[].visible_from` on every row. Stripped at the same seam, so the
# filter still reads the full record off disk.
#
# `visibility` is deliberately NOT in this tuple. It is not a marker the store
# ever writes or reads -- `_date_visible` filters on `visible_from` alone -- and
# it is a real weather field: user_d's forecasts carry
# `forecast.visibility: "poor before 08:00 on the Route 41 spur"`, which the
# whole point of the tool is to hand to the PA.
_WORLD_VISIBILITY_KEYS = ("visible_from", "visible_until")

_STRIPPED_RESULT_KEYS = (*_RUN_IDENTITY_KEYS, *_WORLD_VISIBILITY_KEYS)

# The positions those keys occupy, and the only positions they are removed
# from. Both kinds are stamped on the top level of a *record*, and a record
# reaches a model-facing result in exactly three shapes: it is the result
# itself (`run_id`), it is one field of the result (`reservation.run_id`,
# `booking.visible_from`), or it is a row of one of the result's lists
# (`matches[].visible_from`). Anything deeper is a record's own sub-object --
# `forecast.visibility` is weather data, not bookkeeping -- so the walk stops
# looking there instead of deleting every matching name at every depth.
#
# In path terms (`*` = any key, `[]` = any list): the allowed positions are
# `<key>`, `*.<key>` and `*[].<key>`. `_RECORD_DEPTH` is that rule expressed as
# the number of dict hops from the result root, since list hops do not name a
# level.
_RECORD_DEPTH = 1


def strip_run_identity(value: Any, _depth: int = 0) -> Any:
    """The same value without run identity or world-visibility bookkeeping.

    Only at the documented record positions (`_RECORD_DEPTH`): a key of the
    result, of a record the result holds, or of a row in one of its lists. A
    same-named field deeper inside a record is data and survives.
    """
    if isinstance(value, dict):
        strippable = _depth <= _RECORD_DEPTH
        return {
            key: strip_run_identity(item, _depth + 1)
            for key, item in value.items()
            if not (strippable and key in _STRIPPED_RESULT_KEYS)
        }
    if isinstance(value, list):
        return [strip_run_identity(item, _depth) for item in value]
    return value

# Top-level folders document_read exposes for an empty/'.' path. Keep in sync
# with the document_read backing_stores in data/state_tools/catalog.yaml.
_DOCUMENT_TOP_LEVEL_DIRS = ("documents", "my_desktop")

# Section-level date scoping inside a markdown document (owner, 2026-09-12).
#
# `<!-- visible_from: YYYY-MM-DD -->` on a line of its own opens a scoped block.
# The block runs to the next `visible_from` marker or to `<!-- visible_from: end
# -->`, whichever comes first, and to end of file when neither appears. Content
# before the first marker is always visible, and a block is visible when its
# stamp is `<=` the session clock -- the same `<=` a record's `visible_from`
# gets in `_date_visible`. Without a session clock nothing is dropped.
#
# One file can then serve readers months apart: a working document that records
# what its author knew in October and what they added in December is one byte
# stream, and a session set in October never reads the December lines. The
# markers themselves are stripped from what the reader gets.
_VISIBLE_FROM_RE = re.compile(
    r"^[ \t]*<!--[ \t]*visible_from:[ \t]*(?P<value>\S+)[ \t]*-->[ \t]*$"
)
_VISIBLE_FROM_END = "end"
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Hour-granular form of the same stamp (owner, 2026-09-14). A pre-event probe
# sits 60 minutes before its forced event, on a day that already carries earlier
# accumulation sessions, so a day-granular stamp cannot hide a probe-only object
# from that morning's sessions. A stamp that carries a time is compared as a
# timestamp; a date-only stamp keeps its day-granular meaning exactly.
_ISO_TIMED_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _as_datetime(value: str) -> datetime | None:
    """`value` as a naive local datetime, or None when it carries no time.

    Any timezone offset is dropped rather than converted: both sides of every
    comparison come from the same authored wall clock (a fixture stamp and the
    benchmark session clock), so an offset on one side only would otherwise
    shift a stamp by hours against a stamp that has none.
    """
    if not _ISO_TIMED_RE.match(value):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace(" ", "T", 1))
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


def stamp_visible(
    stamp: str, session_date: str | None, session_timestamp: str | None = None
) -> bool:
    """True when a `visible_from`-style stamp has already come round.

    `stamp` is either `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM` (seconds and an offset
    are accepted and ignored). The session clock is the same two shapes:
    `session_date` is the day the session runs, `session_timestamp` its full
    start time when the caller knows it.

    Four combinations, and only one of them is new:

    * date-only stamp, any session -> `stamp <= session_date`, as before.
    * timed stamp, timed session -> the two timestamps are compared, so an
      object stamped 17:00 is hidden from that morning's 10:00 session.
    * timed stamp, date-only session -> **compared by date part**, i.e. exactly
      today's behaviour. A caller that knows only the day cannot place the
      stamp against the session's hour, and the safe reading is the one that
      changes nothing: the authoring-time validators (which know only a day)
      keep their current verdicts, and `validate_probe_outline`'s A25 is what
      keeps a same-day timed stamp inside its probe's window.
    * no session clock at all -> everything is visible, as before.
    """
    if not session_date and not session_timestamp:
        return True
    stamp_at = _as_datetime(stamp)
    session_at = _as_datetime(session_timestamp or "")
    if stamp_at is not None and session_at is not None:
        return stamp_at <= session_at
    day = session_date or (session_timestamp or "")[:10]
    return stamp[:10] <= day


def filter_visible_blocks(
    content: str, session_date: str | None, session_timestamp: str | None = None
) -> str:
    """`content` with the blocks scoped after the session clock dropped.

    Markers are removed either way, so a document with no marker comes back
    unchanged. A marker value is either `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM` and
    is judged by `stamp_visible`. A malformed marker is not this function's
    problem -- the fixture validator rejects one before a run ever starts, and
    here an unparsable date keeps its block visible rather than silently hiding
    it.
    """
    if "visible_from:" not in content:
        return content
    lines = content.splitlines(keepends=True)
    kept: list[str] = []
    visible = True
    for line in lines:
        match = _VISIBLE_FROM_RE.match(line.rstrip("\r\n"))
        if match is None:
            if visible:
                kept.append(line)
            continue
        value = match.group("value")
        if value == _VISIBLE_FROM_END or not (session_date or session_timestamp):
            visible = True
        elif _ISO_DATE_RE.match(value) or _ISO_TIMED_RE.match(value):
            visible = stamp_visible(value, session_date, session_timestamp)
        else:
            visible = True
    return "".join(kept)

_STORE_DIRS = (
    "email",
    "message",
    "documents",
    "calendar",
    "maps",
    "places",
    "weather",
    "shopping",
    "shipping",
    "finance",
    "hotel",
    "appointment",
    "restaurant",
    "room",
    "flight",
    "scholarly",
    "recipe",
    "reimbursement",
    "conference_registration",
    "grade_submission",
    "comic_store",
    "cinema",
    "household",
    "exhibition",
    "media_transcode",
)

# Calendar recurrence. A recurring fixture record stores one anchor `date` plus
# a `recurrence` phrase; date-filtered lookups expand the phrase so the record
# is present on every date the rule generates from the anchor onward. The
# vocabulary is closed: anything else raises when the calendar file is read.
_WEEKDAY_INDEXES = {
    "mondays": 0,
    "tuesdays": 1,
    "wednesdays": 2,
    "thursdays": 3,
    "fridays": 4,
    "saturdays": 5,
    "sundays": 6,
}
_RECURRENCE_THROUGH_RE = re.compile(
    r"^(?P<rule>.+?)\s+through\s+(?P<until>\d{4}-\d{2}-\d{2})$"
)
_RECURRENCE_WEEKLY_RE = re.compile(r"^weekly on (?P<days>.+)$")
_RECURRENCE_DAY_SPLIT_RE = re.compile(r",|\band\b")
RECURRENCE_FORMAT = (
    "'daily', 'yearly', or 'weekly on <day names>' "
    "(e.g. 'weekly on Mondays and Wednesdays'), "
    "each optionally suffixed ' through YYYY-MM-DD'"
)
# gpt acc_082 (2026-09-18) — the one calendar failure that is a parameter fault
# on an otherwise complete call, and the one where nothing has been written when
# it fires. Six `calendar_event_create` calls came in with `recurrence: '>TBD'`
# and were refused before the append; the wrapper had only the refusal sentence
# to go on, and `_tool_error_class` strips string arguments out of it, so the
# failure signature read `recurrence='' is not …`. The code is what the wrapper
# classifies on -- same shape as `ANCHOR_IS_HEADING_ERROR_CODE` below. Reading
# the calendar file still *raises*: an unparseable phrase already on disk is a
# fixture fault, not a PA one.
CALENDAR_RECURRENCE_INVALID_ERROR_CODE = "calendar_recurrence_invalid"


def _parse_recurrence(
    recurrence: str, event_id: str
) -> tuple[str, frozenset[int], str | None]:
    """Parse one recurrence phrase into (kind, weekdays, until).

    Supported: "daily", "yearly", "weekly on <day names>", each optionally
    suffixed " through YYYY-MM-DD". Free text raises with the record id rather
    than silently matching nothing.
    """
    text = " ".join(str(recurrence).split()).casefold()
    until = None
    through = _RECURRENCE_THROUGH_RE.match(text)
    if through:
        text = through.group("rule")
        until = through.group("until")
    if text in ("daily", "yearly"):
        return text, frozenset(), until
    weekly = _RECURRENCE_WEEKLY_RE.match(text)
    if weekly:
        days = set()
        for token in _RECURRENCE_DAY_SPLIT_RE.split(weekly.group("days")):
            name = token.strip()
            if not name:
                continue
            if name not in _WEEKDAY_INDEXES:
                raise ValueError(
                    f"{event_id}: unsupported calendar recurrence {recurrence!r} "
                    f"(unknown day {name!r})"
                )
            days.add(_WEEKDAY_INDEXES[name])
        if days:
            return "weekly", frozenset(days), until
    raise ValueError(
        f"{event_id}: recurrence={recurrence!r} is not "
        f"{RECURRENCE_FORMAT}"
    )


# Argument formats. Every date/time the PA sends is later compared as a string
# (`slot["date"] == date`), parsed (`strptime`), or both, so a PA that wrote
# "Sep 12" or "3pm" used to get a silent mismatch -- an empty search, or a stored
# record no later read could find. These validators turn that into a tool error
# naming the parameter, the value, and the expected format, and the same format
# strings appear in the tool parameter descriptions in `state_server/server.py`.
DATE_FORMAT = "YYYY-MM-DD"
TIME_FORMAT = "HH:MM"
PERIOD_FORMAT = "YYYY-MM"

_DATE_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_PERIOD_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# Statuses the store's own logic recognises on an update. The guards that block
# a second change read `record["status"] == "cancelled"`, so a PA writing
# "canceled" or "Cancelled" would leave a booking the store still treats as live.
CALENDAR_STATUSES = ("confirmed", "cancelled")
BOOKING_STATUSES = ("confirmed", "cancelled")
HOTEL_STATUSES = ("booked", "cancelled")
ORDER_STATUSES = ("placed", "cancelled")


def _slot_max_party(slot: dict[str, Any] | None, restaurant_id: str) -> int:
    """A restaurant slot's seating cap. A slot without one is a broken fixture,
    never an uncapped table."""
    value = (slot or {}).get("max_party")
    if value is None:
        raise ValueError(
            f"slot {(slot or {}).get('date')} {(slot or {}).get('time')} at "
            f"{restaurant_id} has no max_party; the fixture must state the cap"
        )
    return int(value)


def _require_date(param: str, value: str | None) -> None:
    """Reject anything but a real, zero-padded YYYY-MM-DD calendar date.

    The pattern runs before `strptime` because `strptime` is lenient about
    padding: it reads "2026-11-6" as 6 Nov 2026. Everything downstream compares
    dates as strings against the world's own zero-padded values, so an unpadded
    date was accepted and then matched nothing -- a weather lookup came back
    empty, and a calendar event created on "2026-11-6" was invisible to a
    search for "2026-11-06". Only the canonical form is a date.
    """
    if value is None:
        return
    text = str(value)
    if not _DATE_RE.match(text):
        raise ValueError(f"{param}={value!r} is not {DATE_FORMAT}")
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{param}={value!r} is not {DATE_FORMAT}") from None


def _require_time(param: str, value: str | None) -> None:
    """Reject anything but a 24-hour HH:MM clock time."""
    if value is None:
        return
    if not _TIME_RE.match(str(value)):
        raise ValueError(f"{param}={value!r} is not 24-hour {TIME_FORMAT}")


def _require_period(param: str, value: str | None) -> None:
    """Reject anything but a YYYY-MM billing period."""
    if value is None:
        return
    if not _PERIOD_RE.match(str(value)):
        raise ValueError(f"{param}={value!r} is not {PERIOD_FORMAT}")


def _require_choice(param: str, value: str | None, allowed: tuple[str, ...]) -> None:
    """Reject a value outside the closed vocabulary the store matches on."""
    if value is None:
        return
    if value not in allowed:
        raise ValueError(
            f"{param}={value!r} is not one of: {', '.join(allowed)}"
        )


# Kinds of named record a PA reaches for when a parameter wants a location, and
# the tool that actually takes that kind. Only ever used to explain a rejection.
LOCATION_KIND_TOOLS: dict[str, str] = {
    "restaurant": "restaurant_reservation_search or place_search(category='restaurant')",
    "hotel": "hotel_search",
    "product": "shopping_product_search",
    "contact": "contact_search",
    "place": "place_search with that listing's own category",
}


def _names_a_record(value: str, names: list[str]) -> bool:
    """True when `value` is one of `names`, or carries one of them whole.

    Equality, or a multi-word name sitting inside the value ("Casa Marisol,
    Alder St"). Never the other direction -- "Glenmont" sits inside half the
    contact names of the first world without being any of them, and a hint
    drawn that way would misname what the PA typed.
    """
    needle = value.casefold().strip()
    if not needle:
        return False
    for name in names:
        candidate = str(name).casefold().strip()
        if not candidate:
            continue
        if needle == candidate:
            return True
        if len(candidate.split()) >= 2 and candidate in needle:
            return True
    return False


def _require_known_location(
    param: str,
    value: str,
    known: list[str],
    takes: str,
    other_kinds: dict[str, list[str]],
) -> str:
    """Validate a location-kind value and return the form the query must use.

    Returns the value stripped of surrounding whitespace. Validation used to
    `.strip()` the value while the handler filtered on the raw argument, so
    `hotel_search(city=" Chicago ")` passed the check and then matched nothing
    -- the silent-empty answer this helper exists to abolish, reintroduced by
    the helper itself. Every call site uses this return value for both the
    filter and the echoed result, so there is one normalised value.

    `known` is what the handler matches this parameter against -- the field
    values of the records visible at the session date -- so a value outside it
    can only ever match nothing. The store used to answer those calls with an
    empty result, and a PA that had passed a restaurant name where a street was
    wanted read the emptiness as "nothing nearby" and moved on: regress16
    acc_027 called place_search with location='Casa Marisol', then 'Golden Peak
    Szechuan', then 'downtown', and got 0 matches three times.

    A world with no such records has no vocabulary to check against and keeps
    the empty answer it gave before; that is the absence of a subsystem, not a
    soft path around the rule.

    The message names the parameter, the value, what the parameter takes and
    the known values; when the value is one of the world's other named records
    it also says which kind that is and which tool takes it. Nothing else of
    that record -- no address, id or phone -- reaches the text.
    """
    normalised = str(value).strip()
    if not known:
        return normalised
    needle = normalised.casefold()
    if needle and any(needle in str(candidate).casefold() for candidate in known):
        return normalised
    listed = sorted(
        {str(candidate).strip() for candidate in known if str(candidate).strip()}
    )
    shown = ", ".join(listed[:12]) + (", …" if len(listed) > 12 else "")
    hint = ""
    for kind, names in other_kinds.items():
        if _names_a_record(value, names):
            hint = f" {value!r} is a {kind} name; use {LOCATION_KIND_TOOLS[kind]}."
            break
    raise ValueError(
        f"{param}={value!r} is not a location this world knows. "
        f"{param} takes {takes}.{hint} "
        f"Known {param} values: {shown}"
    )


# Order ids are the merchant's, not the store's. In the first world every seeded
# order in `shopping/order_store.json` is `GM-<6 digits>` -- GM-118427,
# GM-121058, GM-124309 -- and every record that quotes one names GlenMart: the
# confirmation emails (`Order GM-118427 confirmed`, from `GlenMart Orders
# <orders@glenmart.example>`), `shipping/tracking.json` ("GlenMart order
# GM-118427") and the household inventory notes. So GM is GlenMart's code, and
# the store numbers new orders in that scheme instead of one of its own.
#
# These two are the fallback for a fixture that names no merchant of its own;
# `order_merchant` below is where a persona's merchant actually comes from.
ORDER_MERCHANT_CODES: dict[str, str] = {"GlenMart": "GM"}
ORDER_MERCHANT = "GlenMart"


def order_merchant_code(merchant: str) -> str:
    """Order-id code for a merchant.

    The table first; otherwise the merchant's first two uppercase letters (which
    is also how GlenMart earns GM), or its first two characters upper-cased when
    the name has fewer than two capitals.
    """
    if merchant in ORDER_MERCHANT_CODES:
        return ORDER_MERCHANT_CODES[merchant]
    capitals = [char for char in merchant if char.isupper()]
    if len(capitals) >= 2:
        return "".join(capitals[:2])
    return merchant[:2].upper()


def order_merchant(order_store: dict[str, Any]) -> tuple[str, str]:
    """`(name, id code)` of the shop this order store mints under.

    The merchant is the persona's, not the module's, so it is declared in the
    persona's own fixture -- at the top of `shopping/order_store.json`, beside
    the record list:

        {"merchant": {"name": "Harbor Market", "code": "HM"}, "orders": [...]}

    A dict rather than a list on purpose: the loaders that take "the single list
    value" of a store document (`scripts/world_context.py::_load_store`,
    `list_record_key` below) still find `orders`, and
    `scripts/validate_l0_world.py` sees a record with no id, date or status to
    check. `code` is optional -- without it the name's own two capitals stand --
    and so is the whole key, which leaves the GlenMart defaults above.
    """
    declared = order_store.get("merchant") or {}
    name = str(declared.get("name") or ORDER_MERCHANT)
    code = str(declared.get("code") or order_merchant_code(name))
    return name, code


class IdScheme(NamedTuple):
    """Shape of one store's record ids: ``<code><sep><number>``, the number
    zero-padded to `digits`.

    `code` is what the store mints with. `code_pattern` is what counts as a valid
    code where the store is not the one choosing it -- an order carries its
    merchant's code (`ORDER_MERCHANT_CODES`), so any two-letter code is valid
    there; everywhere else the code is the literal one.
    """

    code: str
    sep: str = "_"
    digits: int = 4
    code_pattern: str = ""

    def mint(self, number: int, code: str | None = None) -> str:
        return f"{code or self.code}{self.sep}{number:0{self.digits}d}"

    def valid_re(self) -> re.Pattern[str]:
        """A well-formed id of this shape: the rule the minter satisfies and
        `scripts/validate_l0_world.py` enforces on seeded records."""
        return re.compile(
            rf"^{self.code_pattern or re.escape(self.code)}"
            rf"{re.escape(self.sep)}\d{{{self.digits}}}$"
        )

    def number_re(self, code: str | None = None) -> re.Pattern[str]:
        """One code's ids, capturing the number, so the next id is one past the
        highest that code has already used. Width is open here so a store that
        outgrows its padding still numbers upward."""
        return re.compile(
            rf"^{re.escape(code or self.code)}{re.escape(self.sep)}(\d+)$"
        )

    def label(self) -> str:
        """The shape as a human reads it in an error message: ``ev_NNNN``."""
        return f"{self.code_pattern or self.code}{self.sep}{'N' * self.digits}"


# Id schemes, one row per record type the store mints ids for. The handler that
# appends to a store and the fixture that seeds it have to agree: a PA that reads
# a seeded `res_0001` and then books a table should not get an id from a second
# family. (`restaurant_reservation_create` used to mint `rsv_`, and
# `shopping_order_create` used to mint `order_0001` beside seeded `GM-118427` --
# the same split twice.) Minting sites take the scheme from here and
# `scripts/validate_l0_world.py` checks seeded records against the same table, so
# neither side can drift. Value is (id field, scheme).
def _phone_digits(number: Any) -> str:
    """A phone number reduced to its digits, for comparing two spellings."""
    return re.sub(r"\D", "", str(number or ""))


ID_PREFIXES: dict[str, tuple[str, IdScheme]] = {
    "calendar/events.json": ("event_id", IdScheme("ev")),
    "contacts.json": ("contact_id", IdScheme("ct")),
    "email/inbox.jsonl": ("id", IdScheme("em")),
    "email/drafts.jsonl": ("draft_id", IdScheme("draft")),
    "email/sent.jsonl": ("sent_id", IdScheme("sent")),
    "message/threads.jsonl": ("message_id", IdScheme("msg")),
    "phone/calls.jsonl": ("call_id", IdScheme("call")),
    "corporate/reporting_submissions.jsonl": ("submission_id", IdScheme("corp")),
    "appointment/appointments.jsonl": ("appointment_id", IdScheme("apt")),
    "restaurant/reservations.jsonl": ("reservation_id", IdScheme("res")),
    "room/bookings.jsonl": ("booking_id", IdScheme("rb")),
    "hotel/bookings.jsonl": ("booking_id", IdScheme("hb")),
    "flight/bookings.jsonl": ("booking_id", IdScheme("fb")),
    "reimbursement/submissions.jsonl": ("submission_id", IdScheme("rmb")),
    "conference_registration/registrations.jsonl": ("registration_id", IdScheme("reg")),
    "grade_submission/submissions.jsonl": ("submission_id", IdScheme("grd")),
    "finance/payments.jsonl": ("payment_id", IdScheme("pay")),
    "cinema/tickets.jsonl": ("ticket_id", IdScheme("ticket")),
    "exhibition/tickets.jsonl": ("ticket_id", IdScheme("exh")),
    "shopping/order_store.json": (
        "order_id",
        IdScheme(
            ORDER_MERCHANT_CODES[ORDER_MERCHANT], "-", 6, code_pattern="[A-Z]{2}"
        ),
    ),
    "media_transcode/jobs.jsonl": ("job_id", IdScheme("mtj")),
}

# `ID_PREFIXES` lists exactly the transactional stores -- the ones the user and the
# PA add records to. The resource catalogs (rooms, flights, hotels, restaurant and
# provider slots, products, bills, tracking, places, showtimes and the rest) are the
# world's fixed inventory: the store only reads them back by exact match, and nothing
# is ever minted into them.


def id_scheme(store_path: str) -> IdScheme:
    """Id shape for the record type this store file holds."""
    return ID_PREFIXES[store_path][1]


def list_record_key(document: dict[str, Any], id_field: str) -> str:
    """The key of the record list inside a JSON-document store."""
    lists = {
        key: value for key, value in document.items() if isinstance(value, list)
    }
    for key, value in lists.items():
        if any(isinstance(item, dict) and id_field in item for item in value):
            return key
    if len(lists) == 1:
        return next(iter(lists))
    raise ValueError(
        f"cannot tell which key holds the records (lists: {sorted(lists)})"
    )


#: A probe's record-level overlay, beside the whole files of its `fixture/`
#: directory. One JSON object per line, in the shape
#: `scripts/apply_fixture_additions.py` writes into a run's
#: `authoring/fixture_additions.jsonl`: `store`, `id_field`, `record_id`,
#: `record`.
FIXTURE_RECORDS_NAME = "records.jsonl"


def fixture_record_rows(overlay: Path) -> list[dict[str, Any]]:
    """The rows of an overlay's record file; `[]` when it carries none."""
    path = overlay / FIXTURE_RECORDS_NAME
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_already_there(
    existing: dict[str, Any], record: dict[str, Any], store: str, record_id: str
) -> None:
    """The id is taken. Either by this same record -- then the overlay is simply
    idempotent -- or by another one, and the overlay must not be applied.

    The second case is real: jsonl ids are minted by counting rows, so a run's
    own writes take the next ids while the fixture is later hand-given records
    under the same ones. Overwriting or silently keeping either record makes the
    probe read something nobody wrote for it.
    """
    if strip_run_identity(existing) == strip_run_identity(record):
        return
    raise ValueError(
        "fixture record overlay collides with a different record already in the "
        f"world: store={store!r} record_id={record_id!r} "
        f"in_world={json.dumps(existing, ensure_ascii=False, default=str)} "
        f"overlay={json.dumps(record, ensure_ascii=False, default=str)}"
    )


def run_wrote(record: dict[str, Any]) -> bool:
    """True when a run created this record rather than the fixture.

    The store stamps `run_id` / `user_id` / `session_id` onto every record it
    writes (`_RUN_IDENTITY_KEYS`), and strips them again at the model-facing
    seam; a fixture record carries none of them. That is the whole test for
    "the PA wrote this", and it is what an `op: replace` row is held to.
    """
    return any(key in record for key in _RUN_IDENTITY_KEYS)


def _must_replace(
    existing: dict[str, Any], record: dict[str, Any], store: str, record_id: str
) -> bool:
    """Whether an `op: replace` row still has work to do on the world's copy.

    False when the world already holds this record -- the overlay stays
    idempotent. Raises when the run wrote the copy: `replace` exists for a
    record S0 grew *inside* after the run was frozen (three February 2027 slots
    added to `prov_larkspur_nails`, every other field identical), which an
    append by id cannot fix, and it must never become a way to drop a write the
    PA made under that id.
    """
    if run_wrote(existing):
        raise ValueError(
            "fixture record overlay would replace a record this run wrote, "
            f"which is never allowed: store={store!r} record_id={record_id!r} "
            f"in_world={json.dumps(existing, ensure_ascii=False, default=str)} "
            f"overlay={json.dumps(record, ensure_ascii=False, default=str)}"
        )
    return strip_run_identity(existing) != strip_run_identity(record)


def _row_op(row: dict[str, Any], store: str, record_id: str) -> str:
    op = str(row.get("op") or "append")
    if op not in {"append", "replace"}:
        raise ValueError(
            "fixture record overlay op must be 'append' or 'replace': "
            f"store={store!r} record_id={record_id!r} op={op!r}"
        )
    return op


def apply_fixture_records(world: Path, overlay: Path) -> list[str]:
    """Append the overlay's records to `world`; return the ids actually added.

    Record-level, not file-level: copying a whole jsonl store over the world
    erases every write the timeline accumulated into it, so a scene that needs one
    record adds that record and leaves the rest of the store alone. A record
    the world already holds unchanged is skipped, which makes the overlay
    idempotent; one held under the same id in a different shape is refused
    (`_record_already_there`); a store file the world does not have at all is an
    error, because the overlay describes a world it was written against.

    A row may ask for `op: replace` instead of the default `append`. That is
    for the one shape an append cannot reach: S0 grew a record *inside* after
    the run was frozen, so the id is there, every other field matches, and the
    checkpoint's copy is simply older. It overwrites that copy and nothing
    else -- never a record the run itself wrote (`_must_replace` raises).
    """
    applied: list[str] = []
    for row in fixture_record_rows(overlay):
        store, id_field = str(row["store"]), str(row["id_field"])
        record_id, record = str(row["record_id"]), dict(row["record"])
        op = _row_op(row, store, record_id)
        path = world / store
        if not path.is_file():
            raise FileNotFoundError(
                f"fixture record overlay names a store the world does not hold: "
                f"store={store!r} record_id={record_id!r} path={str(path)!r}"
            )
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            held = [item for item in rows if item.get(id_field) == record_id]
            if held and op == "append":
                _record_already_there(held[0], record, store, record_id)
                continue
            if held:
                if not _must_replace(held[0], record, store, record_id):
                    continue
                rows = [
                    record if item.get(id_field) == record_id else item
                    for item in rows
                ]
                path.write_text(
                    "".join(
                        json.dumps(item, ensure_ascii=False) + "\n"
                        for item in rows
                    ),
                    encoding="utf-8",
                )
            else:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
            key = list_record_key(document, id_field)
            held = [
                item for item in document[key] if item.get(id_field) == record_id
            ]
            if held and op == "append":
                _record_already_there(held[0], record, store, record_id)
                continue
            if held:
                if not _must_replace(held[0], record, store, record_id):
                    continue
                document[key] = [
                    record if item.get(id_field) == record_id else item
                    for item in document[key]
                ]
            else:
                document[key].append(record)
            path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        applied.append(f"{store}:{record_id}")
    return applied


def _recurrence_covers(record: dict[str, Any], date: str) -> bool:
    """True when the record's recurrence rule places it on `date`."""
    anchor = str(record.get("date") or "")
    event_id = str(record.get("event_id") or "?")
    kind, weekdays, until = _parse_recurrence(record["recurrence"], event_id)
    if not anchor or date < anchor:
        return False
    if until and date > until:
        return False
    try:
        target = datetime.strptime(date, "%Y-%m-%d")
        start = datetime.strptime(anchor, "%Y-%m-%d")
    except ValueError:
        return False
    if kind == "daily":
        return True
    if kind == "weekly":
        return target.weekday() in weekdays
    return (target.month, target.day) == (start.month, start.day)


# Anchor matching for `document_update`. The PA reads a document through an MCP
# result that is JSON-encoded, so what it copies back is the escaped rendering,
# not the file's text: JSON escapes copied literally, a run of newlines
# miscounted, a hard-wrapped line collapsed to one space. The match ladder below
# recovers those transcriptions. A fallback tier counts only when it lands on
# exactly one span, and the replacement always splices the DOCUMENT's span.
MATCH_TIERS = ("exact", "unescaped", "whitespace")

# A fallback tier rescued the call, so the PA has to be told which span it
# actually edited: live (regress13 acc_003) two `unescaped` rescues went back as
# a bare `match_tier` field the PA never acted on, and it kept sending escapes.
_MATCH_TIER_NOTES = {
    "unescaped": (
        "anchor matched after undoing the backslash escapes it carried; the "
        "document's own span was replaced -- read the file back before the next edit"
    ),
    "whitespace": (
        "anchor matched only after loosening its whitespace; the document's own "
        "span was replaced -- read the file back before the next edit"
    ),
}

# A bare markdown heading line is never a safe anchor: it matches, so the
# replacement consumes the heading and whatever the replacement opens with
# becomes the heading for entries written under the old one. Live (regress13
# acc_001 t=18) `## Raw -- needs converting to a proper entry` was replaced by a
# new heading plus a new entry, and the raw note that heading introduced was
# left under it.
ANCHOR_IS_HEADING_ERROR_CODE = "anchor_is_heading"
_HEADING_LINE = re.compile(r"#{1,6} .*")

_ANCHOR_UNESCAPES = {'"': '"', "\\": "\\", "`": "`", "n": "\n"}


def _is_bare_heading_anchor(anchor: str) -> bool:
    """True when the anchor's whole content is one markdown heading line."""
    stripped = anchor.strip()
    return "\n" not in stripped and _HEADING_LINE.fullmatch(stripped) is not None


# A span that ends at the end of a physical line inside a wrapped list item is
# never a safe place to write: the replacement lands between the item's first
# line and its continuation lines, and whatever the replacement ends with becomes
# the head of the item's tail. Live (regress19 acc_035 t=8) the PA anchored on
# the first line of a two-line "Notes to self" bullet and inserted a new bullet;
# the old bullet's clause was cut in half around it.
ANCHOR_ENDS_MID_ITEM_ERROR_CODE = "anchor_ends_mid_item"
_LIST_ITEM_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")


def _continuation_after_span(document: str, end: int) -> str | None:
    """The wrapped list-item line the span at [.., end) stops short of, or None.

    Only a span that ends on a line boundary can stop short of a wrapped line;
    a span that ends mid-line replaces part of one line and leaves the rest of
    it in place. An indented line counts as a continuation only when the
    lines above it, back to the first unindented one, are a list item — a
    BibTeX field or an indented code line under a plain line is not.
    """
    if end <= 0 or end > len(document):
        return None
    if document[end - 1] == "\n":
        boundary = end - 1
    elif end < len(document) and document[end] == "\n":
        boundary = end
    else:
        return None
    next_end = document.find("\n", boundary + 1)
    next_line = document[boundary + 1 : next_end if next_end >= 0 else len(document)]
    if not next_line.strip() or next_line[0] not in " \t":
        return None
    lines_above = document[:boundary].splitlines()
    while lines_above and lines_above[-1][:1] in (" ", "\t"):
        lines_above.pop()
    if not lines_above or not _LIST_ITEM_LINE.match(lines_above[-1]):
        return None
    return next_line.strip()


def _unescape_anchor(anchor: str) -> str:
    """Undo the JSON escaping the PA sees in `document_read` output."""
    out: list[str] = []
    index = 0
    while index < len(anchor):
        char = anchor[index]
        replacement = _ANCHOR_UNESCAPES.get(anchor[index + 1 : index + 2])
        if char == "\\" and replacement is not None:
            out.append(replacement)
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _literal_spans(document: str, needle: str) -> list[tuple[int, int]]:
    """Every non-overlapping span of `needle`, matching `str.count` semantics."""
    if not needle:
        return []
    spans: list[tuple[int, int]] = []
    start = document.find(needle)
    while start != -1:
        spans.append((start, start + len(needle)))
        start = document.find(needle, start + len(needle))
    return spans


def _whitespace_spans(document: str, anchor: str) -> list[tuple[int, int]]:
    """Spans where every run of whitespace in `anchor` matches any run in the
    document. The document itself is never normalised, so the span returned is a
    real slice of the file."""
    tokens = anchor.split()
    if not tokens:
        return []
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    if anchor[:1].isspace():
        pattern = r"\s+" + pattern
    if anchor[-1:].isspace():
        pattern = pattern + r"\s+"
    return [match.span() for match in re.finditer(pattern, document)]


def _anchor_tiers(document: str, anchor: str):
    """The match tiers in order; the first tier with any hit decides."""
    yield "exact", _literal_spans(document, anchor)
    unescaped = _unescape_anchor(anchor)
    if unescaped != anchor:
        yield "unescaped", _literal_spans(document, unescaped)
    yield "whitespace", _whitespace_spans(document, anchor)


class StateStore:
    """Read/write task state under one run namespace.

    The store intentionally contains task artifacts only. It must not store
    derived interaction preferences or memory summaries.
    """

    def __init__(
        self,
        state_root: Path | str,
        *,
        run_id: str,
        user_id: str,
        session_id: str,
        session_date: str | None = None,
        session_timestamp: str | None = None,
        gate: dict[str, Any] | None = None,
    ) -> None:
        self.state_root = Path(state_root).expanduser().resolve()
        self.run_id = run_id
        self.user_id = user_id
        self.session_id = session_id
        # The day the session runs, and its full start time when the caller
        # knows it. A full timestamp is the store's write clock as well as its
        # visibility clock; date-only authoring callers keep the deterministic
        # noon fallback in `_now()`.
        self.session_date = session_date or (
            str(session_timestamp)[:10] if session_timestamp else None
        )
        self.session_timestamp = session_timestamp
        self.gate = gate
        self.state_root.mkdir(parents=True, exist_ok=True)
        for name in _STORE_DIRS:
            (self.state_root / name).mkdir(parents=True, exist_ok=True)
        if gate is not None:
            self._check_write_gate(gate)
            self._log_gate_loaded(gate)
        self._install_write_guards()

    @classmethod
    def from_env(cls) -> StateStore:
        state_root = os.environ.get("MEMPA_STATE_ROOT")
        if not state_root:
            raise RuntimeError("MEMPA_STATE_ROOT is required for the state server")
        return cls(
            state_root,
            run_id=os.environ.get("MEMPA_RUN_ID", "run"),
            user_id=os.environ.get("MEMPA_USER_ID", "user"),
            session_id=os.environ.get("MEMPA_SESSION_ID", "session"),
            session_date=os.environ.get("MEMPA_SESSION_DATE") or None,
            session_timestamp=os.environ.get("MEMPA_SESSION_TIMESTAMP") or None,
            gate=json.loads(os.environ["MEMPA_GATE"])
            if os.environ.get("MEMPA_GATE")
            else None,
        )

    # Session gate

    def _check_write_gate(self, gate: dict[str, Any]) -> None:
        """Refuse a gate the store could not enforce.

        A gate that does not say which beat opens it is an old freeze, not a
        gate the store can enforce, so it is an error here rather than a silent
        fallback to the text check alone.
        """
        missing = [
            key
            for key in GATE_REQUIRED_TEXT_KEYS
            if not str(gate.get(key) or "").strip()
        ]
        if "pre_reveal_write_tools" not in gate:
            missing.append("pre_reveal_write_tools")
        if missing:
            raise ValueError(
                "MEMPA_GATE is missing or blank at " + ", ".join(sorted(missing))
                + ": a gate the store cannot enforce is worse than no gate, so "
                "there is no fallback to the text check alone"
            )
        if not self._usable_key_terms(
            gate.get("key_terms"), gate.get("dictated_text")
        ):
            raise ValueError(
                f"MEMPA_GATE needs at least {GATE_MIN_KEY_TERMS} non-blank "
                "key_terms: the gate is keyed on the information the user gave, "
                "and one term is a word rather than the information"
            )

    def _install_write_guards(self) -> None:
        """Put the two write-refusing checks in front of every write tool.

        They are bound onto the instance instead of written into each of the 27
        write handlers: one seam, and `WRITE_TOOL_METHODS` is then the whole
        list of what is locked -- which is what the catalog test can hold to
        account. The six payload handlers keep their own `_enforce_gate` call;
        by the time their body runs the gate is open, so that call is left with
        the text check it always did.

        Installed on every store, not only a gated one. The session-closing
        refusal is about a COMPRESSED session, and a compressed session has no
        gate at all -- so binding this to `gate is not None` would leave the one
        shape it exists for unguarded. Both checks read a marker file and return
        at once when it is absent, so an ungated, unclosed session behaves
        exactly as before.
        """
        for tool_name, method_name in WRITE_TOOL_METHODS.items():
            method = getattr(self, method_name)
            setattr(self, method_name, self._write_guarded(tool_name, method))

    def _log_gate_loaded(self, gate: dict[str, Any]) -> None:
        """Record, once per step, that the server actually received the gate.

        Nothing else in a finished run proves it. A step whose gate never
        reached the subprocess -- an env var that did not survive, a resolve
        that quietly returned None -- ran ungated and looked exactly like a step
        the PA passed honestly. This row is the evidence, and `log_tool` writes
        it to the same `tool_log.jsonl` the rest of the audit lives in.

        The MCP server builds a fresh store per tool call, so the row is written
        only when the log does not already carry one.
        """
        rows = self._read_jsonl(self.state_root / "tool_log.jsonl")
        if any(row.get("tool") == GATE_LOADED_TOOL for row in rows):
            return
        self.log_tool(
            GATE_LOADED_TOOL,
            {
                "gate_tool": gate.get("tool"),
                "gate_param": gate.get("param"),
                "task_fact": gate.get("task_fact"),
                "reveal_beat": gate.get("reveal_beat"),
                "pre_reveal_write_tools": list(
                    gate.get("pre_reveal_write_tools") or ()
                ),
            },
            {"loaded": True, "open": self._gate_is_open()},
        )

    def _write_guarded(self, tool_name: str, method: Any) -> Any:
        def guarded(*args: Any, **kwargs: Any) -> Any:
            self._enforce_session_closing(tool_name)
            self._enforce_write_gate(tool_name)
            return method(*args, **kwargs)

        return guarded

    def _gate_is_open(self) -> bool:
        """Whether the session has reached the beat where the user dictates."""
        return (self.state_root / GATE_OPEN_MARKER).exists()

    def _gate_is_satisfied(self) -> bool:
        """Whether a gated write has already carried the user's information."""
        return (self.state_root / GATE_SATISFIED_MARKER).exists()

    def _mark_gate_satisfied(self) -> None:
        """Record the first gated write that carried every key term.

        The marker is what makes the gate a one-time hold: after this the PA may
        write the same parameter in its own words, because it has asked. The row
        is the audit half -- a finished run has to be able to say which call
        satisfied the gate, not merely that some call did.
        """
        marker = self.state_root / GATE_SATISFIED_MARKER
        if marker.exists():
            return
        marker.touch()
        gate = self.gate or {}
        self.log_tool(
            GATE_SATISFIED_MARKER,
            {
                "gate_tool": gate.get("tool"),
                "gate_param": gate.get("param"),
                "task_fact": gate.get("task_fact"),
            },
            {"satisfied": True},
        )

    def _session_is_closing(self) -> bool:
        """Whether the PA's turn now running is a compressed session's last."""
        return (self.state_root / SESSION_CLOSING_MARKER).exists()

    def _enforce_session_closing(self, tool_name: str) -> None:
        """Refuse every write on a compressed session's final PA turn.

        Owner ruling 2026-09-09. A compressed session is exactly two beats --
        the probe, then the reaction -- and the session ends with the PA's reply
        to the reaction. Rule E says that reaction beat never asks for a
        product, so there is no beat of such a session whose work a write on
        that turn could be finishing; and nothing scores what the PA does there,
        because the transcript stops. Live regress19 `acc_035` t8 executed an
        offer the user never answered, on exactly that turn: two roster writes
        carrying an invented 8 -> 9 and a corrupted note, into the committed
        stage, unseen and unscored.

        The refusal names nothing about the measurement (like lock 2, and for
        the same reason). Reads are untouched -- the PA may still look anything
        up to answer -- and an uncompressed session never has the marker, so it
        never reaches this.
        """
        if not self._session_is_closing():
            return
        self.log_tool(
            tool_name,
            {},
            {
                "error": "gate_refusal",
                "gate_refusal": True,
                "session_closing": True,
            },
            status="error",
        )
        raise ValueError(SESSION_CLOSING_WRITE_REFUSAL.format(tool=tool_name))

    def _enforce_write_gate(self, tool_name: str) -> None:
        """Refuse a write the session's gate has not opened yet.

        Two refusals, one condition. The gate tool itself is refused whatever
        the PA passes -- the words cannot be the user's, because the user has
        not said them -- and so is every other write tool except the ones the
        frozen draft actually calls before the reveal. Reads and IX tools are
        never touched: the PA is free to look the world up, it just cannot act
        on it before it has asked.

        The two texts differ in what they may say. Lock 1 names the parameter it
        wants, because asking for it is the only way past. Lock 2 names nothing
        of the gate (ruling B): the call it refuses is one the PA improvised, so
        anything it says about the gate announces a step the session has not
        reached.
        """
        gate = self.gate or {}
        if not gate or self._gate_is_open():
            return
        param = str(gate.get("param") or "")
        gate_tool = str(gate.get("tool") or "")
        if tool_name != gate_tool and tool_name in set(
            gate.get("pre_reveal_write_tools") or ()
        ):
            return
        self.log_tool(
            tool_name,
            {"gate_tool": gate_tool, "gate_param": param},
            {"error": "gate_refusal", "gate_refusal": True, "closed": True},
            status="error",
        )
        if tool_name == gate_tool:
            raise ValueError(GATE_REFUSAL.format(param=param, tool=tool_name))
        raise ValueError(GATE_CLOSED_WRITE_REFUSAL.format(tool=tool_name))

    def _payload_log_args(
        self, tool_name: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """The payload half of a gate-capable write's `tool_log.jsonl` row.

        `payload` is the handler's WHOLE gate payload -- the same dict it hands
        `_enforce_gate`, every free-text parameter an outline may gate on --
        because the gated one is the one `matched` is about and the handler does
        not know which it is. Live fin_08 gated on `subject` and this was handed
        `{"body": body}`, so the match was computed against a parameter the gate
        was not on and the row read `matched: false` over a stored draft whose
        subject was the dictated sentence exactly.

        The dictated value itself, capped at `GATE_LOG_VALUE_LIMIT`, plus what
        the gate made of this call. `matched` is about this call only: it is
        false for a write to any tool the gate is not on, because such a call
        does not satisfy the gate however it is worded. `satisfied` is about the
        session: true from the call that first carried every key term onwards.
        Since ruling A, a row on the gate tool reading `matched: false,
        satisfied: true` is a refusal -- the write that would have emptied the
        artifact -- and the successful later writes read `matched: true` on the
        PA's own wording. `document_update` is the exception worth knowing: its
        `matched` is about the replacement span, so a legitimate edit elsewhere
        in the gated file lands with `matched: false`.
        """
        args: dict[str, Any] = {
            key: value[:GATE_LOG_VALUE_LIMIT] if isinstance(value, str) else value
            for key, value in payload.items()
        }
        gate = self.gate or {}
        if gate:
            param = str(gate.get("param") or "")
            args["gate"] = {
                "param": param,
                "matched": gate.get("tool") == tool_name
                and self._carries_key_terms(
                    gate.get("key_terms"), payload.get(param)
                ),
                "satisfied": self._gate_is_satisfied(),
                "open": self._gate_is_open(),
            }
        return args

    def _enforce_gate(self, tool_name: str, payload: dict[str, Any]) -> None:
        """Refuse a gated write that does not carry the user's information.

        Called by every write handler that persists a free-text parameter, with
        all of that handler's such parameters -- `calendar_event_update` passes
        `title` and `notes` together. Only the gated one is compared; the rest
        are there so the pair the outline names is always present to check.

        The gate holds the FIRST write here (owner, 2026-09-08). Once a call has
        carried every key term the marker goes down and this stops demanding the
        words of the payload: the PA has asked, and acc_009's session was eaten
        by a PA re-quoting the user turn after turn. What later writes still owe
        is the ARTIFACT, which is `_enforce_gate_keeps_terms`.

        It is a no-op unless this session has a gate on this tool. When the gate
        is set the parameter is required even where the signature makes it
        optional -- an absent `reason` is the PA skipping the gate, not
        declining an option.

        A gate on this tool naming a parameter the handler does not carry is a
        misconfiguration, not a gate to skip. A gate param is one of
        `scripts/outline_schema.py:GATE_PAYLOAD_PARAMS`: every persisted
        free-text value a user could dictate, and nothing the world can hand
        over -- a value the world holds is a value the PA can guess, and
        acc_039's old `appointment_create.time` gate was defeated three live
        rounds running by one undated search. So the only way to reach this is a
        broken freeze, and returning quietly would leave the session ungated.
        """
        gate = self.gate or {}
        if gate.get("tool") != tool_name:
            return
        param = gate.get("param")
        if param not in payload:
            raise ValueError(
                f"gate misconfiguration: the session gates {param!r} on "
                f"{tool_name}, which carries no such payload parameter"
            )
        if self._gate_is_satisfied():
            if tool_name not in GATE_MERGING_TOOLS:
                self._enforce_gate_keeps_terms(tool_name, payload, payload[param])
            return
        if self._carries_key_terms(gate.get("key_terms"), payload[param]):
            self._mark_gate_satisfied()
            return
        self.log_tool(
            tool_name,
            {"gate_param": param, **self._payload_log_args(tool_name, payload)},
            {"error": "gate_refusal", "gate_refusal": True},
            status="error",
        )
        raise ValueError(GATE_MISMATCH_REFUSAL.format(param=param, tool=tool_name))

    def _enforce_gate_keeps_terms(
        self, tool_name: str, payload: dict[str, Any], artifact: Any
    ) -> None:
        """Refuse a post-`gate_satisfied` write that empties the artifact.

        `artifact` is what the gated parameter's record says once THIS write has
        landed -- the whole file for `document_update`, the merged field for the
        other two merging tools, and the payload value everywhere else, since a
        create tool's record is what the call passed. Every key term has to
        still be in there (owner ruling A, 2026-09-08): acc_009's PA asked, got
        the words, wrote them, and then revised them away twice, and the run's
        artifact no longer held the focus the user dictated.

        This is deliberately not a check on the value the PA passed. acc_003's
        year fix elsewhere in the gated file carries no term at all and is a
        legitimate write; what it must not do is remove the note.

        The refusal changes no gate state -- the marker stays down -- and the
        merging tools call this before persisting, so a refused write leaves the
        record exactly as it was.
        """
        gate = self.gate or {}
        if gate.get("tool") != tool_name or not self._gate_is_satisfied():
            return
        param = str(gate.get("param") or "")
        if self._carries_key_terms(gate.get("key_terms"), artifact):
            return
        self.log_tool(
            tool_name,
            {"gate_param": param, **self._payload_log_args(tool_name, payload)},
            {"error": "gate_refusal", "gate_refusal": True, "drops_terms": True},
            status="error",
        )
        raise ValueError(GATE_DROPS_TERMS_REFUSAL.format(tool=tool_name, param=param))

    def _gate_param(self) -> str:
        """The gated parameter's name, for a handler reading it off its record."""
        return str((self.gate or {}).get("param") or "")

    @classmethod
    def _carries_key_terms(cls, key_terms: Any, value: Any) -> bool:
        """Whether `value` carries every piece of information the user gave.

        The gate is keyed on information, not on words: each key term the
        outline authored off the dictated sentence has to appear in the value,
        and how the PA arranges them around those terms is its own business. A
        multi-word term is a phrase -- its words in that order -- so "blood
        panel" is not satisfied by a value that says "panel" somewhere and
        "blood" somewhere else. Matching is case-insensitive, whitespace is
        normalised, and punctuation is a word break on both sides, so
        "chest-tightness" carries "chest tightness" and a term at the end of a
        sentence is not lost to its full stop. Word endings are not information
        either, so a word matches an ordinary inflection of itself
        (`_is_inflection`) -- "hardest" carries "hard". Verbatim needs no path of
        its own: the dictated sentence carries all of its own terms.

        Terms that survive none of that refuse rather than accept -- an empty
        term would match everything -- as does a missing or blank value.
        `_install_write_gate` already rejects such a gate at init; this is the
        second half of the same rule.
        """
        if not isinstance(value, str) or not value.strip():
            return False
        if not isinstance(key_terms, list) or not key_terms:
            return False
        terms = [cls._term_words(str(term)).split() for term in key_terms]
        if not all(terms):
            return False
        words = cls._term_words(value).split()
        return all(cls._carries_term(term, words) for term in terms)

    @classmethod
    def _carries_term(cls, term: list[str], words: list[str]) -> bool:
        """Whether `words` says `term`'s words, in that order and side by side.

        The phrase rule, now that a word is compared rather than a string: a
        window of the value the length of the term, every position matching.
        """
        span = len(term)
        return any(
            all(
                cls._same_word(term_word, value_word)
                for term_word, value_word in zip(term, words[start : start + span])
            )
            for start in range(len(words) - span + 1)
        )

    @classmethod
    def _same_word(cls, term_word: str, value_word: str) -> bool:
        """The same word, whichever of the two is wearing the ending.

        Both directions, because the outline authors the term off the dictated
        sentence and the user says it either way round: "printed" against a PA
        that wrote "print" is the same information as the reverse.
        """
        return (
            term_word == value_word
            or cls._is_inflection(term_word, value_word)
            or cls._is_inflection(value_word, term_word)
        )

    @staticmethod
    def _is_inflection(stem: str, form: str) -> bool:
        """Whether `form` is `stem` plus one `GATE_INFLECTION_SUFFIXES` ending.

        With the two spelling changes those endings bring: the silent e drops
        before one (settle -> settling) and a final consonant doubles (plan ->
        planned). Nothing else. This is deliberately not a stemmer: the ending
        has to be one of the listed few, so "hardware" and "harden" are not
        "hard", and "-ly" is not among them, so neither is "hardly".
        """
        for suffix in GATE_INFLECTION_SUFFIXES:
            if len(form) <= len(stem) or not form.endswith(suffix):
                continue
            root = form[: -len(suffix)]
            if root == stem:  # confirm -> confirming
                return True
            if stem.endswith("e") and root == stem[:-1]:  # settle -> settling
                return True
            if (  # plan -> planned
                stem[-1:].isalpha()
                and stem[-1] not in "aeiou"
                and root == stem + stem[-1]
            ):
                return True
        return False

    @classmethod
    def _usable_key_terms(cls, key_terms: Any, dictated_text: Any = None) -> bool:
        """Whether `MEMPA_GATE.key_terms` is a gate the store can enforce.

        `GATE_MIN_KEY_TERMS` terms, or one term that is the whole dictated text
        -- the probe line's gate, which names the user's sentence entire because
        it has no authored terms to name (owner, 2026-09-14). "The whole
        dictated text" is decided by `_term_words`, the same normalisation the
        matcher compares in, so the term that passes here is exactly the one a
        payload has to carry whole to satisfy `_carries_key_terms`. Any other
        single term is still a word standing in for the information, and refuses.
        """
        if not isinstance(key_terms, list) or not key_terms:
            return False
        if not all(
            isinstance(term, str) and cls._term_words(term) for term in key_terms
        ):
            return False
        if len(key_terms) >= GATE_MIN_KEY_TERMS:
            return True
        return cls._term_words(key_terms[0]) == cls._term_words(
            dictated_text if isinstance(dictated_text, str) else ""
        )

    @staticmethod
    def _term_words(text: str) -> str:
        """`text` case-folded, punctuation to spaces, one space between words.

        A standalone number word becomes its digits (`NUMBER_WORDS`), on both
        sides of the comparison, so "three weeks" and "3 weeks" are the same
        term: the gate is keyed on the information, and how the PA spells a
        number is not information the user gave it. A digit-grouping comma is
        dropped rather than folded to a space, on both sides too, so the term
        "1,200 dollars" is carried by a payload that writes either 1,200 or
        1200.
        """
        text = DIGIT_GROUP_COMMA_RE.sub("", text)
        words = "".join(ch.casefold() if ch.isalnum() else " " for ch in text).split()
        return " ".join(NUMBER_WORDS.get(word, word) for word in words)

    # Date visibility

    def _date_visible(self, record: dict[str, Any], *fields: str) -> bool:
        """True when the record already exists at the session clock.

        Checks visible_from first, then the given per-store arrival fields.
        Records without any stamp are always visible. Without a session clock
        everything is visible. A stamp of `YYYY-MM-DDTHH:MM` is compared
        against `session_timestamp` when the runtime handed one down, and by
        date part otherwise -- see `stamp_visible`.
        """
        if not self.session_date and not self.session_timestamp:
            return True
        for key in ("visible_from", *fields):
            stamp = record.get(key)
            if stamp:
                return stamp_visible(
                    str(stamp), self.session_date, self.session_timestamp
                )
        return True

    def _location_kind_names(self) -> dict[str, list[str]]:
        """The world's other named records, by kind, for a rejection message.

        Only the kinds a location-kind parameter gets confused with: the shops
        and venues of `places/nearby.json` (split by whether the listing calls
        itself a restaurant), the bookable restaurants, hotels, products and
        contacts. Names only -- `_require_known_location` puts nothing else of
        a record into the error. Read per call, like every other store read,
        and used only when that call is already being rejected.
        """

        def named(relative: str, key: str) -> list[str]:
            path = self.state_root / relative
            if not path.exists():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
            records = (data.get(key) or []) if isinstance(data, dict) else data
            return [
                str(record.get("name"))
                for record in records
                if isinstance(record, dict) and record.get("name")
            ]

        def eats(record: dict[str, Any]) -> bool:
            labels = [str(record.get("category") or "")]
            labels += [str(tag) for tag in record.get("tags") or []]
            return "restaurant" in " ".join(labels).casefold()

        listings: list[dict[str, Any]] = []
        path = self.state_root / "places" / "nearby.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            records = (data.get("places") or []) if isinstance(data, dict) else data
            listings = [
                record
                for record in records
                if isinstance(record, dict) and record.get("name")
            ]
        return {
            "restaurant": [
                *(str(record["name"]) for record in listings if eats(record)),
                *named("restaurant/slots.json", "restaurants"),
            ],
            "hotel": [
                str(record.get("name"))
                for record in self._hotels()
                if isinstance(record, dict) and record.get("name")
            ],
            "product": named("shopping/products.json", "products"),
            "contact": [
                str(record.get("name"))
                for record in self._contacts()[1]
                if isinstance(record, dict) and record.get("name")
            ],
            "place": [
                str(record["name"]) for record in listings if not eats(record)
            ],
        }

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        """Public facts, excluding controls and the answer to a future call."""
        return {
            k: v for k, v in record.items()
            if k not in _CAPABILITY_MARKER_KEYS and k != "phone_answer"
        }

    def _capability_limit(
        self, tool: str, args: dict[str, Any], record: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The capability-limit result for a resolved resource the fixture marks
        unavailable, or None when the record carries no marker.

        Only a read/retrieve of an already-resolved id may produce it: a search
        with no hits stays an ordinary empty result, and an id that matches no
        record stays the existing not_found error.
        """
        if record.get("unavailable") is not True:
            return None
        boundary_kind = record.get("boundary_kind")
        if boundary_kind not in BOUNDARY_KINDS:
            raise ValueError(
                "A fixture record marked unavailable needs a boundary_kind from "
                f"{BOUNDARY_KINDS!r}: tool={tool!r} got={boundary_kind!r}"
            )
        result: dict[str, Any] = {
            "ok": False,
            "error_code": CAPABILITY_LIMIT_ERROR_CODE,
            "boundary_kind": boundary_kind,
        }
        reason = record.get("unavailable_reason")
        if reason:
            result["detail"] = str(reason)
        self.log_tool(
            tool,
            args,
            {
                "error_code": CAPABILITY_LIMIT_ERROR_CODE,
                "boundary_kind": boundary_kind,
            },
            status=CAPABILITY_LIMIT_TOOL_STATUS,
        )
        return result

    # Documents

    def copier_entitlement_read(
        self, contract_id: str, component: str
    ) -> dict[str, Any]:
        data = json.loads(
            (self.state_root / "copier" / "entitlements.json").read_text(
                encoding="utf-8"
            )
        )
        wanted_component = component.casefold()
        record = next(
            (
                item
                for item in data.get("entitlements", [])
                if item.get("contract_id") == contract_id
                and str(item.get("component") or "").casefold() == wanted_component
                and self._date_visible(item)
            ),
            None,
        )
        args = {"contract_id": contract_id, "component": component}
        if record is None:
            self.log_tool(
                "copier_entitlement_read", args, {"error": "not_found"}, status="error"
            )
            raise FileNotFoundError(f"{contract_id}: {component}")
        limited = self._capability_limit("copier_entitlement_read", args, record)
        if limited is not None:
            return limited
        self.log_tool("copier_entitlement_read", args, {"found": True})
        return {"entitlement": self._public_record(record)}

    def corporate_reporting_submit(self, file_id: str) -> dict[str, Any]:
        self._enforce_gate("corporate_reporting_submit", {})
        report_path = self._resolve_inside(file_id)
        if report_path.is_dir():
            raise IsADirectoryError(self._relative(report_path))
        data = json.loads(
            (
                self.state_root / "corporate" / "reporting_accounts.json"
            ).read_text(encoding="utf-8")
        )
        account = next(
            (
                item
                for item in data.get("accounts", [])
                if file_id in (item.get("accepted_file_ids") or [])
                and self._date_visible(item)
            ),
            None,
        )
        args = {"file_id": file_id}
        if account is None:
            self.log_tool(
                "corporate_reporting_submit", args, {"error": "not_found"}, status="error"
            )
            raise FileNotFoundError(f"no reporting account accepts {file_id}")
        limited = self._capability_limit("corporate_reporting_submit", args, account)
        if limited is not None:
            return limited
        submissions = self.state_root / "corporate" / "reporting_submissions.jsonl"
        submission_id = self._mint_id(submissions)
        record = {
            "submission_id": submission_id,
            "account_id": account.get("account_id"),
            "file_id": self._relative(report_path),
            "status": "submitted",
            "submitted_at": self._now(),
            "run_id": self.run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(submissions, record)
        result = {"submission_id": submission_id, "status": "submitted"}
        self.log_tool("corporate_reporting_submit", args, result)
        self.log_state_diff(
            "corporate.reporting_submissions",
            "append",
            submission_id,
            f"Submitted corporate report {file_id}",
        )
        return result

    def read_document(self, path: str) -> dict[str, Any]:
        if path.strip().rstrip("/") in ("", "."):
            folders = [
                f"{name}/"
                for name in _DOCUMENT_TOP_LEVEL_DIRS
                if (self.state_root / name).is_dir()
            ]
            result = {"path": ".", "type": "directory", "files": folders}
            self.log_tool(
                "document_read",
                {"path": path},
                {"path": ".", "files": len(folders), "type": "directory"},
            )
            return result
        doc_path = self._resolve_inside(path)
        if doc_path.is_dir():
            files = [
                self._relative(p)
                for p in sorted(doc_path.rglob("*"))
                if p.is_file() and self._is_actor_visible(p)
            ]
            result = {
                "path": self._relative(doc_path),
                "type": "directory",
                "files": files,
            }
            self.log_tool(
                "document_read",
                {"path": path},
                {"path": result["path"], "files": len(files), "type": "directory"},
            )
            return result
        content = filter_visible_blocks(
            doc_path.read_text(encoding="utf-8"),
            self.session_date,
            self.session_timestamp,
        )
        result = {
            "path": self._relative(doc_path),
            "type": "file",
            "content": content,
            "bytes": len(content.encode("utf-8")),
        }
        self.log_tool("document_read", {"path": path}, {"path": result["path"], "bytes": result["bytes"]})
        return result

    def create_document(self, path: str, content: str) -> dict[str, Any]:
        """Create a new document. An existing path is left untouched."""
        self._enforce_gate("document_create", {"content": content})
        doc_path = self._resolve_writable_inside(path)
        if doc_path.exists() and doc_path.is_dir():
            raise IsADirectoryError(self._relative(doc_path))
        if doc_path.exists():
            result = {
                "path": self._relative(doc_path),
                "status": "exists",
                "bytes": doc_path.stat().st_size,
            }
            self.log_tool(
                "document_create",
                {
                    "path": path,
                    **self._payload_log_args("document_create", {"content": content}),
                },
                result,
            )
            return result
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(content, encoding="utf-8")
        result = {
            "path": self._relative(doc_path),
            "status": "created",
            "bytes": len(content.encode("utf-8")),
        }
        self.log_tool(
            "document_create",
            {
                "path": path,
                "content_chars": len(content),
                **self._payload_log_args("document_create", {"content": content}),
            },
            result,
        )
        self.log_state_diff(
            "documents",
            "write",
            result["path"],
            f"Created document: {result['path']}",
        )
        return result

    def update_document(self, path: str, anchor: str, replacement: str) -> dict[str, Any]:
        """Replace one unique `anchor` span inside an existing document.

        The anchor is matched by the `_anchor_tiers` ladder, so a transcription
        artifact of the JSON-escaped `document_read` output still lands on the
        span the PA meant. Ambiguity at any tier is still an error, a bare
        heading line is refused before any tier runs, and a span rescued by a
        fallback tier says so in the result.
        """
        self._enforce_gate("document_update", {"replacement": replacement})
        doc_path = self._resolve_writable_inside(path)
        if not doc_path.is_file():
            raise FileNotFoundError(self._relative(doc_path))
        logged_args = {
            "path": path,
            "anchor_chars": len(anchor),
            "replacement_chars": len(replacement),
            **self._payload_log_args("document_update", {"replacement": replacement}),
        }
        if _is_bare_heading_anchor(anchor):
            self.log_tool(
                "document_update",
                logged_args,
                {"error_code": ANCHOR_IS_HEADING_ERROR_CODE},
                status="error",
            )
            return {
                "ok": False,
                "error_code": ANCHOR_IS_HEADING_ERROR_CODE,
                "detail": (
                    "the anchor is a bare markdown heading line, so this edit would "
                    "replace the heading itself and leave everything under it beneath "
                    "the replacement's own first line; anchor on a content line under "
                    "the heading, or on the heading plus its first content line"
                ),
            }
        current = doc_path.read_text(encoding="utf-8")
        spans: list[tuple[int, int]] = []
        match_tier = MATCH_TIERS[0]
        for match_tier, spans in _anchor_tiers(current, anchor):
            if spans:
                break
        if not spans:
            first_line = anchor.strip().splitlines()[0][:60] if anchor.strip() else ""
            shown = f", starting: {first_line}" if first_line else ""
            raise ValueError(
                f"anchor not found in {self._relative(doc_path)} "
                f"({len(anchor)} chars{shown}); copy it from document_read as "
                "plain text — real newlines, plain quotes, no backslash escapes"
            )
        if len(spans) > 1:
            raise ValueError(
                f"anchor occurs {len(spans)} times in {self._relative(doc_path)}; "
                "extend it until it matches exactly one span"
            )
        start, end = spans[0]
        continuation = _continuation_after_span(current, end)
        if continuation is not None:
            self.log_tool(
                "document_update",
                logged_args,
                {"error_code": ANCHOR_ENDS_MID_ITEM_ERROR_CODE},
                status="error",
            )
            return {
                "ok": False,
                "error_code": ANCHOR_ENDS_MID_ITEM_ERROR_CODE,
                "detail": (
                    "the anchor ends inside a wrapped list item: the next line "
                    f"({continuation[:60]!r}) continues the same item, so the "
                    "replacement would be wedged between the item's first line "
                    "and its continuation; extend the anchor through the item's "
                    "last wrapped line, or anchor on that last line instead"
                ),
            }
        updated = current[:start] + replacement + current[end:]
        # Ruling A on the merging path: the file as it stands after the write,
        # judged on a copy so a refusal leaves the document untouched.
        self._enforce_gate_keeps_terms(
            "document_update", {"replacement": replacement}, updated
        )
        doc_path.write_text(updated, encoding="utf-8")
        result = {
            "path": self._relative(doc_path),
            "status": "updated",
            "bytes": len(updated.encode("utf-8")),
            "match_tier": match_tier,
        }
        note = _MATCH_TIER_NOTES.get(match_tier)
        if note:
            result["note"] = note
        self.log_tool("document_update", logged_args, result)
        self.log_state_diff(
            "documents",
            "write",
            result["path"],
            f"Updated document: {result['path']}",
        )
        return result

    def delete_document(self, path: str) -> dict[str, Any]:
        """Delete an existing document."""
        doc_path = self._resolve_writable_inside(path)
        if not doc_path.is_file():
            raise FileNotFoundError(self._relative(doc_path))
        relative = self._relative(doc_path)
        doc_path.unlink()
        result = {"path": relative, "status": "deleted"}
        self.log_tool("document_delete", {"path": path}, result)
        self.log_state_diff(
            "documents",
            "write",
            relative,
            f"Deleted document: {relative}",
        )
        return result

    # Email

    def save_email_draft(
        self,
        to: str,
        subject: str,
        body: str,
        attachments: list[str] | None = None,
        draft_id: str | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate("email_draft_create", {"subject": subject, "body": body})
        drafts_path = self.state_root / "email" / "drafts.jsonl"
        attachment_paths = self._resolve_attachments(attachments)
        if draft_id is not None:
            records = self._read_jsonl(drafts_path)
            target = next(
                (r for r in records if r.get("draft_id") == draft_id), None
            )
            if target is None:
                raise FileNotFoundError(draft_id)
            target.update({
                "to": to,
                "subject": subject,
                "body": body,
                "attachments": attachment_paths,
                "updated_at": self._now(),
            })
            self._write_jsonl(drafts_path, records)
            result = {"draft_id": draft_id, "status": "updated"}
            self.log_tool(
                "email_draft_create",
                {
                    "to": to,
                    "subject": subject,
                    "body_chars": len(body),
                    "draft_id": draft_id,
                    **self._payload_log_args(
                        "email_draft_create", {"subject": subject, "body": body}
                    ),
                },
                result,
            )
            self.log_state_diff("email.drafts", "write", draft_id, f"Revised draft: {subject}")
            return result
        new_id = self._mint_id(drafts_path)
        record = {
            "draft_id": new_id,
            "to": to,
            "subject": subject,
            "body": body,
            "attachments": attachment_paths,
            "created_at": self._now(),
            "run_id": self.run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(drafts_path, record)
        result = {"draft_id": new_id, "status": "saved"}
        self.log_tool(
            "email_draft_create",
            {
                "to": to,
                "subject": subject,
                "body_chars": len(body),
                "attachments": attachment_paths,
                **self._payload_log_args(
                    "email_draft_create", {"subject": subject, "body": body}
                ),
            },
            result,
        )
        self.log_state_diff("email.drafts", "append", new_id, f"Saved draft: {subject}")
        return result

    def read_email_draft(self, draft_id: str) -> dict[str, Any]:
        drafts_path = self.state_root / "email" / "drafts.jsonl"
        for record in self._read_jsonl(drafts_path):
            if record.get("draft_id") == draft_id:
                result = {
                    key: record.get(key)
                    for key in ("draft_id", "to", "subject", "body", "attachments")
                }
                self.log_tool("email_draft_read", {"draft_id": draft_id}, {"found": True})
                return result
        self.log_tool("email_draft_read", {"draft_id": draft_id}, {"error": "not_found"}, status="error")
        raise FileNotFoundError(draft_id)

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        attachments: list[str] | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate("email_message_send", {"subject": subject, "body": body})
        sent_path = self.state_root / "email" / "sent.jsonl"
        sent_id = self._mint_id(sent_path)
        attachment_paths = self._resolve_attachments(attachments)
        record = {
            "sent_id": sent_id,
            "to": to,
            "subject": subject,
            "body": body,
            "attachments": attachment_paths,
            "sent_at": self._now(),
            "run_id": self.run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(sent_path, record)
        result = {"sent_id": sent_id, "status": "sent"}
        self.log_tool(
            "email_message_send",
            {
                "to": to,
                "subject": subject,
                "body_chars": len(body),
                "attachments": attachment_paths,
                **self._payload_log_args(
                    "email_message_send", {"subject": subject, "body": body}
                ),
            },
            result,
        )
        self.log_state_diff("email.sent", "append", sent_id, f"Sent email: {subject}")
        return result

    def read_sent_email(self, sent_id: str) -> dict[str, Any]:
        sent_path = self.state_root / "email" / "sent.jsonl"
        for record in self._read_jsonl(sent_path):
            if record.get("sent_id") != sent_id:
                continue
            if not self._date_visible(record, "date", "sent_at"):
                continue
            result = {
                key: record.get(key)
                for key in ("sent_id", "to", "subject", "body", "attachments", "sent_at")
            }
            self.log_tool("email_sent_read", {"sent_id": sent_id}, {"found": True})
            return result
        self.log_tool("email_sent_read", {"sent_id": sent_id}, {"error": "not_found"}, status="error")
        raise FileNotFoundError(sent_id)

    def search_email(self, query: str) -> dict[str, Any]:
        inbox_path = self.state_root / "email" / "inbox.jsonl"
        matches = []
        for record in self._read_jsonl(inbox_path):
            if not self._date_visible(record, "date"):
                continue
            haystack = " ".join(
                str(record.get(key) or "")
                for key in ("from", "subject", "body", "date")
            )
            if self._query_matches(query, haystack):
                body = str(record.get("body") or "")
                matches.append({
                    "id": record.get("id"),
                    "from": record.get("from"),
                    "date": record.get("date"),
                    "subject": record.get("subject"),
                    "snippet": body[:220],
                })
        result = {"query": query, "matches": matches}
        self.log_tool(
            "email_message_search",
            {"query": query},
            self._search_log_result(matches, "id"),
        )
        return result

    def read_email(self, email_id: str) -> dict[str, Any]:
        for record in self._read_jsonl(self.state_root / "email" / "inbox.jsonl"):
            if record.get("id") != email_id:
                continue
            if not self._date_visible(record, "date"):
                continue
            result = {
                "id": record.get("id"),
                "from": record.get("from"),
                "to": record.get("to"),
                "date": record.get("date"),
                "subject": record.get("subject"),
                "body": record.get("body"),
            }
            self.log_tool("email_message_read", {"email_id": email_id}, {"id": email_id})
            return result
        self.log_tool("email_message_read", {"email_id": email_id}, {"error": "not_found"}, status="error")
        raise FileNotFoundError(email_id)

    # Messages (informal channel)

    def message_send(self, to: str, body: str) -> dict[str, Any]:
        self._enforce_gate("message_send", {"body": body})
        threads_path = self.state_root / "message" / "threads.jsonl"
        message_id = self._mint_id(threads_path)
        record = {
            "message_id": message_id,
            "with": to,
            "direction": "outgoing",
            "body": body,
            "date": self.session_date or self._now(),
            "run_id": self.run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(threads_path, record)
        result = {"message_id": message_id, "status": "sent"}
        self.log_tool(
            "message_send",
            {
                "to": to,
                "body_chars": len(body),
                **self._payload_log_args("message_send", {"body": body}),
            },
            result,
        )
        self.log_state_diff("message.threads", "append", message_id, f"Sent message to {to}")
        return result

    def message_search(self, query: str) -> dict[str, Any]:
        threads_path = self.state_root / "message" / "threads.jsonl"
        matches = []
        for record in self._read_jsonl(threads_path):
            if not self._date_visible(record, "date"):
                continue
            haystack = " ".join(
                str(record.get(key) or "")
                for key in ("with", "body", "date")
            )
            if self._query_matches(query, haystack):
                body = str(record.get("body") or "")
                matches.append({
                    "message_id": record.get("message_id"),
                    "with": record.get("with"),
                    "direction": record.get("direction"),
                    "date": record.get("date"),
                    "snippet": body[:160],
                })
        result = {"query": query, "matches": matches}
        self.log_tool(
            "message_search",
            {"query": query},
            self._search_log_result(matches, "message_id"),
        )
        return result

    # Environment delivery (forced-event third-party messages land in the world)

    def inbox_deliver(self, *, sender: str, subject: str, body: str) -> dict[str, Any]:
        inbox_path = self.state_root / "email" / "inbox.jsonl"
        recipients = [
            json.loads(line).get("to")
            for line in inbox_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if inbox_path.exists() else []
        record = {
            "id": self._mint_id(inbox_path),
            "from": sender,
            "to": next((r for r in recipients if r), "user@example"),
            "date": self.session_date or self._now()[:10],
            "subject": subject,
            "body": body,
        }
        self._append_jsonl(inbox_path, record)
        self.log_state_diff("email.inbox", "append", record["id"], f"Delivered email from {sender}")
        return record

    def message_deliver(self, *, sender: str, body: str) -> dict[str, Any]:
        threads_path = self.state_root / "message" / "threads.jsonl"
        record = {
            "message_id": self._mint_id(threads_path),
            "with": sender,
            "direction": "incoming",
            "body": body,
            "date": self.session_date or self._now()[:10],
        }
        self._append_jsonl(threads_path, record)
        self.log_state_diff("message.threads", "append", record["message_id"], f"Delivered message from {sender}")
        return record

    def message_read(self, message_id: str) -> dict[str, Any]:
        threads_path = self.state_root / "message" / "threads.jsonl"
        for record in self._read_jsonl(threads_path):
            if record.get("message_id") != message_id:
                continue
            if not self._date_visible(record, "date"):
                continue
            result = {
                key: record.get(key)
                for key in ("message_id", "with", "direction", "body", "date")
            }
            self.log_tool("message_read", {"message_id": message_id}, {"found": True})
            return result
        self.log_tool("message_read", {"message_id": message_id}, {"error": "not_found"}, status="error")
        raise FileNotFoundError(message_id)

    # Calendar

    def _calendar_events(self) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
        path = self.state_root / "calendar" / "events.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"events": []}
        events = data.setdefault("events", [])
        for record in events:
            if record.get("recurrence"):
                _parse_recurrence(
                    record["recurrence"], str(record.get("event_id") or "?")
                )
        return path, data, events

    @staticmethod
    def _calendar_date_matches(record: dict[str, Any], date: str) -> bool:
        """True when the record occupies `date`.

        A record without a recurrence matches its stored date exactly, or any
        day of the span `date`..`end_date` when it carries an `end_date`; a
        recurring one also matches every later date its rule generates.
        """
        if str(record.get("date") or "") == date:
            return True
        if not record.get("recurrence"):
            end_date = str(record.get("end_date") or "")
            start = str(record.get("date") or "")
            return bool(end_date and start and start <= date <= end_date)
        return _recurrence_covers(record, date)

    def calendar_event_search(
        self,
        query: str,
        date: str | None = None,
    ) -> dict[str, Any]:
        _require_date("date", date)
        _, _, events = self._calendar_events()
        query_tokens = self._search_tokens(query)
        matches = []
        for record in events:
            if not self._date_visible(record, "created_date"):
                continue
            haystack = json.dumps(record, ensure_ascii=False)
            if query_tokens and not (query_tokens & self._search_tokens(haystack)):
                continue
            if date:
                if not self._calendar_date_matches(record, date):
                    continue
                if record.get("recurrence") and str(record.get("date") or "") != date:
                    # Matched through the recurrence rule: report the occurrence
                    # the caller asked about, not the series anchor. Same id.
                    record = {**record, "date": date}
            matches.append(record)
        result = {"query": query, "date": date, "matches": matches}
        self.log_tool(
            "calendar_event_search",
            {"query": query, "date": date},
            self._search_log_result(matches, "event_id"),
        )
        return result

    def calendar_event_read(self, event_id: str) -> dict[str, Any]:
        _, _, events = self._calendar_events()
        for record in events:
            if record.get("event_id") != event_id:
                continue
            if not self._date_visible(record, "created_date"):
                continue
            self.log_tool("calendar_event_read", {"event_id": event_id}, {"found": True})
            return {"event": record}
        self.log_tool("calendar_event_read", {"event_id": event_id}, {"error": "not_found"}, status="error")
        raise FileNotFoundError(event_id)

    def calendar_event_create(
        self,
        title: str,
        date: str,
        start_time: str | None = None,
        end_time: str | None = None,
        location: str | None = None,
        participants: list[str] | None = None,
        recurrence: str | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate("calendar_event_create", {"title": title})
        _require_date("date", date)
        _require_time("start_time", start_time)
        _require_time("end_time", end_time)
        path, data, events = self._calendar_events()
        existing_ids = {str(record.get("event_id")) for record in events}
        scheme = id_scheme("calendar/events.json")
        index = len(events) + 1
        while scheme.mint(index) in existing_ids:
            index += 1
        event_id = scheme.mint(index)
        record: dict[str, Any] = {
            "event_id": event_id,
            "title": title,
            "date": date,
            "scope": "personal",
            "status": "confirmed",
            "created_date": (self.session_date or self._now()[:10]),
        }
        if start_time:
            record["start_time"] = start_time
        if end_time:
            record["end_time"] = end_time
        if location:
            record["location"] = location
        if participants:
            record["participants"] = participants
        if recurrence:
            # Reject free text here too: an unparseable phrase written into the
            # file would make every later calendar read raise. Returned with a
            # stable `error_code` rather than raised (gpt acc_082): nothing has
            # been appended at this point, and the wrapper has to be able to
            # tell "this one parameter is wrong and the entry does not exist
            # yet" from every other calendar failure without reading the
            # sentence. The sentence itself is unchanged.
            try:
                _parse_recurrence(recurrence, event_id)
            except ValueError as exc:
                self.log_tool(
                    "calendar_event_create",
                    {
                        "date": date,
                        "recurrence": recurrence,
                        **self._payload_log_args(
                            "calendar_event_create", {"title": title}
                        ),
                    },
                    {"error_code": CALENDAR_RECURRENCE_INVALID_ERROR_CODE},
                    status="error",
                )
                return {
                    "ok": False,
                    "error_code": CALENDAR_RECURRENCE_INVALID_ERROR_CODE,
                    "detail": str(exc),
                }
            record["recurrence"] = recurrence
        events.append(record)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        result = {"event_id": event_id, "status": "created"}
        self.log_tool(
            "calendar_event_create",
            {
                "date": date,
                "recurrence": recurrence,
                **self._payload_log_args("calendar_event_create", {"title": title}),
            },
            result,
        )
        self.log_state_diff("calendar.events", "append", event_id, f"Created event: {title}")
        return result

    def calendar_event_update(
        self,
        event_id: str,
        title: str | None = None,
        date: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        location: str | None = None,
        participants: list[str] | None = None,
        status: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate("calendar_event_update", {"title": title, "notes": notes})
        _require_date("date", date)
        _require_time("start_time", start_time)
        _require_time("end_time", end_time)
        _require_choice("status", status, CALENDAR_STATUSES)
        path, data, events = self._calendar_events()
        record = next((r for r in events if r.get("event_id") == event_id), None)
        if record is None or not self._date_visible(record, "created_date"):
            self.log_tool("calendar_event_update", {"event_id": event_id}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(event_id)
        if record.get("scope") == "world":
            raise ValueError(f"world events cannot be changed: {event_id}")
        changes = {
            key: value
            for key, value in {
                "title": title,
                "date": date,
                "start_time": start_time,
                "end_time": end_time,
                "location": location,
                "participants": participants,
                "status": status,
                "notes": notes,
            }.items()
            if value is not None
        }
        if not changes:
            raise ValueError("calendar_event_update requires at least one field to change")
        record.update(changes)
        # Ruling A on the merging path: a call that never names the gated field
        # leaves the user's words where they were, so the record is what counts.
        self._enforce_gate_keeps_terms(
            "calendar_event_update",
            {"title": title, "notes": notes},
            record.get(self._gate_param()),
        )
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        result = {"event_id": event_id, "status": record.get("status", "confirmed")}
        self.log_tool(
            "calendar_event_update",
            {
                "event_id": event_id,
                **changes,
                **self._payload_log_args(
                    "calendar_event_update", {"title": title, "notes": notes}
                ),
            },
            result,
        )
        self.log_state_diff(
            "calendar.events",
            "write",
            event_id,
            f"Updated event: {record.get('title')}",
        )
        return result

    # Contacts

    def _contacts(self) -> tuple[Path, list[dict[str, Any]]]:
        path = self.state_root / "contacts.json"
        if not path.exists():
            return path, []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("contacts"), list):
            return path, raw["contacts"]
        if isinstance(raw, dict):
            return path, list(raw.values())
        if isinstance(raw, list):
            return path, raw
        return path, []

    def _write_contacts(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.write_text(
            json.dumps({"contacts": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def lookup_contacts(self, query: str) -> dict[str, Any]:
        _, records = self._contacts()
        query_tokens = self._search_tokens(query)
        scored_matches = []
        for record in records:
            if not self._date_visible(record):
                continue
            public = self._public_record(record)
            haystack = json.dumps(public, ensure_ascii=False)
            haystack_tokens = self._search_tokens(haystack)
            score = len(query_tokens & haystack_tokens)
            if self._query_matches(query, haystack):
                score += 10
            if score:
                scored_matches.append((score, public))
        scored_matches.sort(key=lambda item: item[0], reverse=True)
        matches = [record for _, record in scored_matches]
        result = {"query": query, "matches": matches}
        self.log_tool(
            "contact_search",
            {"query": query},
            self._search_log_result(matches, "contact_id"),
        )
        return result

    def contact_read(self, contact_id: str) -> dict[str, Any]:
        _, records = self._contacts()
        for record in records:
            if record.get("contact_id") != contact_id:
                continue
            if not self._date_visible(record):
                continue
            limited = self._capability_limit(
                "contact_read", {"contact_id": contact_id}, record
            )
            if limited is not None:
                return limited
            self.log_tool("contact_read", {"contact_id": contact_id}, {"found": True})
            return {"contact": self._public_record(record)}
        self.log_tool("contact_read", {"contact_id": contact_id}, {"error": "not_found"}, status="error")
        raise FileNotFoundError(contact_id)

    def contact_create(
        self,
        name: str,
        email: str | None = None,
        phone: str | None = None,
        organization: str | None = None,
        relation: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate("contact_create", {"relation": relation, "notes": notes})
        path, records = self._contacts()
        existing_ids = {str(record.get("contact_id")) for record in records}
        scheme = id_scheme("contacts.json")
        index = len(records) + 1
        while scheme.mint(index) in existing_ids:
            index += 1
        contact_id = scheme.mint(index)
        record: dict[str, Any] = {"contact_id": contact_id, "name": name}
        for key, value in {
            "email": email,
            "phone": phone,
            "organization": organization,
            "relation": relation,
            "notes": notes,
        }.items():
            if value is not None:
                record[key] = value
        records.append(record)
        self._write_contacts(path, records)
        result = {"contact_id": contact_id, "status": "created"}
        self.log_tool(
            "contact_create",
            {
                "name": name,
                **self._payload_log_args(
                    "contact_create", {"relation": relation, "notes": notes}
                ),
            },
            result,
        )
        self.log_state_diff("contacts", "append", contact_id, f"Created contact: {name}")
        return result

    def contact_update(
        self,
        contact_id: str,
        name: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        organization: str | None = None,
        relation: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate("contact_update", {"relation": relation, "notes": notes})
        path, records = self._contacts()
        record = next((r for r in records if r.get("contact_id") == contact_id), None)
        if record is None:
            self.log_tool("contact_update", {"contact_id": contact_id}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(contact_id)
        changes = {
            key: value
            for key, value in {
                "name": name,
                "email": email,
                "phone": phone,
                "organization": organization,
                "relation": relation,
                "notes": notes,
            }.items()
            if value is not None
        }
        if not changes:
            raise ValueError("contact_update requires at least one field to change")
        record.update(changes)
        # Ruling A on the merging path, as on `calendar_event_update`.
        self._enforce_gate_keeps_terms(
            "contact_update",
            {"relation": relation, "notes": notes},
            record.get(self._gate_param()),
        )
        self._write_contacts(path, records)
        result = {"contact_id": contact_id, "status": "updated"}
        self.log_tool(
            "contact_update",
            {
                "contact_id": contact_id,
                **changes,
                **self._payload_log_args(
                    "contact_update", {"relation": relation, "notes": notes}
                ),
            },
            result,
        )
        self.log_state_diff("contacts", "write", contact_id, f"Updated contact: {record.get('name')}")
        return result

    # Phone

    def _dialled_record(self, number: str) -> dict[str, Any] | None:
        """The world record whose `phone` is this number, or None.

        Contacts first, then places: a number can belong to either, and both
        stores carry it under the same field. Compared on digits alone, because
        the world writes a number the way the listing prints it -- "(555)
        026-2200" on the contact, "555 741 0188" on the place -- and the PA
        passes back whatever it read.
        """
        wanted = _phone_digits(number)
        if not wanted:
            return None
        _, contacts = self._contacts()
        for record in contacts:
            if self._date_visible(record) and _phone_digits(record.get("phone")) == wanted:
                return record
        places_path = self.state_root / "places" / "nearby.json"
        if not places_path.exists():
            return None
        places = json.loads(places_path.read_text(encoding="utf-8"))
        if isinstance(places, dict):
            places = places.get("places") or []
        for record in places:
            if self._date_visible(record, "published_date") and _phone_digits(
                record.get("phone")
            ) == wanted:
                return record
        return None

    def phone_call_dial(self, number: str, purpose: str) -> dict[str, Any]:
        """Place a call to a number the world lists, and record what was said.

        Whether the line answers is the callee's own record talking: a record
        the fixture marks `unavailable` gives the same capability-limit result
        `contact_read` gives, and a record that registers a `phone_answer` --
        what that line says when it picks up -- hands it back. A record that
        registers none still answers; the world simply has not authored what
        was said, and this returns facts rather than inventing them.
        """
        self._enforce_gate("phone_call_dial", {"purpose": purpose})
        record = self._dialled_record(number)
        if record is None:
            self.log_tool(
                "phone_call_dial", {"number": number}, {"error": "not_found"}, status="error"
            )
            raise FileNotFoundError(number)
        limited = self._capability_limit(
            "phone_call_dial", {"number": number}, record
        )
        if limited is not None:
            return limited
        path = self.state_root / "phone" / "calls.jsonl"
        call_id = self._mint_id(path)
        call: dict[str, Any] = {
            "call_id": call_id,
            "number": number,
            "purpose": purpose,
            "answered_by": record.get("name"),
            "status": "answered",
            "date": self.session_date or self._now(),
            "run_id": self.run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        answer = record.get("phone_answer")
        if answer:
            call["answer"] = str(answer)
        self._append_jsonl(path, call)
        result = {
            "call_id": call_id,
            "status": "answered",
            "number": number,
            "answered_by": record.get("name"),
        }
        if answer:
            result["answer"] = str(answer)
        self.log_tool(
            "phone_call_dial",
            {
                "number": number,
                "answered_by": record.get("name"),
                **self._payload_log_args("phone_call_dial", {"purpose": purpose}),
            },
            {"call_id": call_id, "status": "answered"},
        )
        self.log_state_diff(
            "phone.calls", "append", call_id, f"Called {record.get('name')}"
        )
        return result

    def phone_call_read(self, call_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="phone_call_read",
            path=self.state_root / "phone" / "calls.jsonl",
            id_field="call_id",
            record_id=call_id,
            wrap="call",
        )

    # Maps and places

    # Words that, on their own, name the place the user lives rather than a place
    # on the map. "the"/"my"/"our" are dropped before the check so "the apartment"
    # reads the same as "apartment".
    _HOME_WORDS = frozenset({"home", "apartment", "apt", "flat", "house", "residence"})
    _HOME_FILLER = frozenset({"the", "our", "his", "her", "their", "own", "user"})

    def _home_address_parts(self) -> list[str]:
        """Casefolded comma-separated pieces of the user's own street address."""
        path = self.state_root / "shipping" / "addresses.json"
        if not path.exists():
            return []
        try:
            records = json.loads(path.read_text(encoding="utf-8")).get("addresses") or []
        except (json.JSONDecodeError, OSError):
            return []
        records = [record for record in records if isinstance(record, dict)]
        record = next((r for r in records if r.get("default")), None)
        if record is None and records:
            record = records[0]
        if record is None:
            return []
        parts = [part.strip().casefold() for part in str(record.get("street") or "").split(",")]
        return [part for part in parts if len(part) >= 6]

    def _names_home(self, endpoint: str) -> bool:
        """True when this endpoint text names where the user lives."""
        text = endpoint.casefold().strip()
        if not text:
            return False
        if any(part in text for part in self._home_address_parts()):
            return True
        tokens = self._search_tokens(text) - self._HOME_FILLER
        return bool(tokens) and tokens <= self._HOME_WORDS

    def _home_route_endpoint(self, endpoints: list[str]) -> str:
        """The stored route endpoint that stands for the user's home, if any."""
        return next((endpoint for endpoint in endpoints if self._names_home(endpoint)), "")

    @classmethod
    def _endpoint_matches(cls, query: str, stored: str) -> bool:
        query_text = query.casefold().strip()
        stored_text = stored.casefold().strip()
        if not query_text or not stored_text:
            return query_text == stored_text
        if query_text in stored_text or stored_text in query_text:
            return True
        query_tokens = cls._search_tokens(query_text)
        stored_tokens = cls._search_tokens(stored_text)
        overlap = query_tokens & stored_tokens
        # One side's words must all appear on the other. Accepting half of the
        # shorter side let a place name win on a single shared town name: the
        # home address "12 Corvid Ave, Glenmont" shared only "glenmont" with
        # "Glenmont Heights apartment" (below half of three) but that one token
        # was half of two-token "Glenmont Cinema", so walking times came back
        # from the cinema. regress14b acc_002.
        return bool(overlap) and overlap in (query_tokens, stored_tokens)

    def route_options(self, origin: str, destination: str, stop: str = "") -> dict[str, Any]:
        route_path = self.state_root / "maps" / "route_options.json"
        data = json.loads(route_path.read_text(encoding="utf-8"))
        route: dict[str, Any] = {}
        if isinstance(data, dict) and isinstance(data.get("routes"), list):
            routes = [candidate for candidate in data["routes"] if isinstance(candidate, dict)]
            # An endpoint that names the user's home resolves to the home record
            # first; no place name gets to compete for it on a shared token.
            home = self._home_route_endpoint(
                [str(candidate.get(field) or "") for candidate in routes for field in ("origin", "destination")]
            )
            origin_query = home if home and self._names_home(origin) else origin
            destination_query = home if home and self._names_home(destination) else destination
            candidates = [
                candidate
                for candidate in routes
                if self._endpoint_matches(origin_query, str(candidate.get("origin") or ""))
                and self._endpoint_matches(destination_query, str(candidate.get("destination") or ""))
            ]
            if stop.strip():
                matching_stop = [
                    candidate
                    for candidate in candidates
                    if self._endpoint_matches(stop, str(candidate.get("stop") or ""))
                ]
                route = matching_stop[0] if matching_stop else {}
            else:
                direct = [
                    candidate
                    for candidate in candidates
                    if not str(candidate.get("stop") or "").strip()
                ]
                route = (direct or candidates or [{}])[0]
        elif isinstance(data, dict):
            route = data
        result = {
            "origin": origin,
            "stop": stop,
            "destination": destination,
            "options": route.get("options") or [],
        }
        self.log_tool(
            "map_route_search",
            {"origin": origin, "stop": stop, "destination": destination},
            {"options": len(result["options"])},
        )
        return result

    def place_search(
        self,
        location: str | None = None,
        category: str | None = None,
        query: str | None = None,
        limit: int = 3,
    ) -> dict[str, Any]:
        """Nearby places by street, kind of place, or the listing's own name.

        `places/nearby.json` records carry no id, so before `query` the only way
        to reach one the user named was to guess its street and its category:
        every outline ingredient over this store cites its record by name, and
        the PA had no parameter to pass that name to. The three selectors are
        independent and combinable; at least one has to be given.

        `category` is checked the same way `location` is. It used to be the one
        selector on this tool where a wrong value returned an empty list instead
        of an error -- exactly the silent-empty path `_require_known_location`
        was written to close on `location`.
        """
        if location is None and category is None and query is None:
            raise ValueError(
                "place_search needs at least one of location, category or query."
            )
        places_path = self.state_root / "places" / "nearby.json"
        places = json.loads(places_path.read_text(encoding="utf-8"))
        if isinstance(places, dict):
            places = places.get("places") or []
        visible = [
            record for record in places
            if self._date_visible(record, "published_date")
        ]
        if location is not None:
            location = _require_known_location(
                "location",
                location,
                [str(record.get("near") or "") for record in visible],
                "a street or landmark near the user as the world names it, never a "
                "business or person name",
                self._location_kind_names(),
            )
        if category is not None:
            category = _require_known_location(
                "category",
                category,
                [
                    label
                    for record in visible
                    for label in [str(record.get("category") or "")]
                    + [str(tag) for tag in record.get("tags") or []]
                ],
                "a kind of place as this world's listings label it, never a "
                "business or person name",
                self._location_kind_names(),
            )
        needle_location = (location or "").casefold()
        needle_category = (category or "").casefold()
        matches = [
            self._public_record(record) for record in visible
            if (
                location is None
                or needle_location in str(record.get("near") or "").casefold()
            )
            and (
                category is None
                or needle_category in str(record.get("category") or "").casefold()
                or any(
                    needle_category in str(tag).casefold()
                    for tag in record.get("tags") or []
                )
            )
            and (
                query is None
                or self._query_matches(query, str(record.get("name") or ""))
            )
        ][:max(limit, 0)]
        result = {
            "location": location,
            "category": category,
            "query": query,
            "limit": limit,
            "matches": matches,
        }
        self.log_tool(
            "place_search",
            {
                "location": location,
                "category": category,
                "query": query,
                "limit": limit,
            },
            {"matches": len(matches)},
        )
        return result

    def club_closure_calendar_read(
        self, place_id: str, date_range: str
    ) -> dict[str, Any]:
        data = json.loads(
            (self.state_root / "places" / "closure_calendars.json").read_text(
                encoding="utf-8"
            )
        )
        record = next(
            (
                item
                for item in data.get("calendars", [])
                if item.get("place_id") == place_id
                and item.get("date_range") == date_range
                and self._date_visible(item)
            ),
            None,
        )
        args = {"place_id": place_id, "date_range": date_range}
        if record is None:
            self.log_tool(
                "club_closure_calendar_read", args, {"error": "not_found"}, status="error"
            )
            raise FileNotFoundError(f"{place_id}: {date_range}")
        limited = self._capability_limit("club_closure_calendar_read", args, record)
        if limited is not None:
            return limited
        self.log_tool("club_closure_calendar_read", args, {"found": True})
        return {"calendar": self._public_record(record)}

    # Weather

    _TIME_WINDOW_LABELS = {
        "morning": (5, 12),
        "midday": (11, 14),
        "afternoon": (12, 17),
        "evening": (17, 23),
        "tonight": (17, 23),
        "night": (20, 24),
    }

    @classmethod
    def _time_window_matches(cls, requested: str, stored: str) -> bool:
        requested = requested.strip().casefold()
        stored = stored.strip().casefold()
        if requested == stored:
            return True
        label_range = cls._TIME_WINDOW_LABELS.get(requested)
        match = re.match(r"^(\d{1,2}):\d{2}-(\d{1,2}):\d{2}$", stored)
        if label_range is None or match is None:
            return False
        start, end = int(match.group(1)), int(match.group(2))
        return start < label_range[1] and end > label_range[0]

    def weather_forecast_read(
        self,
        location: str,
        date: str,
        time_window: str,
    ) -> dict[str, Any]:
        _require_date("date", date)
        forecast_path = self.state_root / "weather" / "forecasts.json"
        if not forecast_path.exists():
            legacy = self.state_root / "weather" / "glenmont_evening_forecast.json"
            if legacy.exists():
                forecast_path = legacy
        data = json.loads(forecast_path.read_text(encoding="utf-8"))
        visible = [
            candidate
            for candidate in data.get("forecasts", [])
            if isinstance(candidate, dict)
            and self._date_visible(candidate, "published_date")
        ]
        location = _require_known_location(
            "location",
            location,
            [str(candidate.get("location") or "") for candidate in visible],
            "a city or neighbourhood the forecast records name",
            self._location_kind_names(),
        )
        requested_location = location.casefold()
        forecast = next(
            (
                candidate
                for candidate in visible
                if requested_location
                in str(candidate.get("location") or "").casefold()
                and str(candidate.get("date") or "") == date
                and self._time_window_matches(
                    time_window, str(candidate.get("time_window") or "")
                )
            ),
            {},
        )
        result = {
            "location": location,
            "date": date,
            "time_window": time_window,
            "forecast": forecast.get("forecast") or {},
        }
        self.log_tool(
            "weather_forecast_read",
            {"location": location, "date": date, "time_window": time_window},
            {"matched": bool(result["forecast"])},
        )
        return result

    # Shopping

    def shopping_product_search(
        self,
        query: str,
        max_price: float | None = None,
        delivery_method: str | None = None,
    ) -> dict[str, Any]:
        data = json.loads(
            (self.state_root / "shopping" / "products.json").read_text(
                encoding="utf-8"
            )
        )
        matches = []
        for record in data.get("products", []):
            if not self._date_visible(record):
                continue
            haystack = " ".join(
                str(record.get(key) or "")
                for key in ("item_type", "name", "description")
            )
            if not self._query_matches(query, haystack):
                continue
            if max_price is not None and float(record.get("price") or 0) > float(max_price):
                continue
            # Out-of-stock products stay in the results, carrying
            # `in_stock: false`. Dropping them made the store lie about its
            # catalog -- the PA could not tell "we do not sell that" from "it
            # is sold out" -- and left every withheld product unreachable, so
            # no order could ever reach the capability limit on it.
            if delivery_method is not None and not any(
                delivery_method.casefold() == str(method).casefold()
                for method in record.get("delivery_methods") or []
            ):
                continue
            matches.append(self._public_record(record))
        result = {
            "query": query,
            "max_price": max_price,
            "delivery_method": delivery_method,
            "currency": data.get("currency"),
            "matches": matches,
        }
        self.log_tool(
            "shopping_product_search",
            {"query": query, "max_price": max_price, "delivery_method": delivery_method},
            self._search_log_result(matches, "product_id"),
        )
        return result

    def shopping_order_create(
        self,
        product_id: str,
        quantity: int,
        delivery_address_id: str,
        payment_method_id: str,
    ) -> dict[str, Any]:
        if quantity < 1:
            raise ValueError("quantity must be positive")
        products = json.loads(
            (self.state_root / "shopping" / "products.json").read_text(
                encoding="utf-8"
            )
        )
        product = next(
            (
                item
                for item in products.get("products", [])
                if item.get("product_id") == product_id
                and self._date_visible(item)
            ),
            None,
        )
        if product is None:
            raise ValueError(f"unknown product: {product_id}")
        # Two different "no" answers. A product the fixture marks `unavailable`
        # is a world boundary the PA has to work around (user_d acc_066: the
        # promised letterhead is back-ordered at the mill with no restock
        # date), so it reports the formal capability limit. A product that is
        # merely out of stock is a wait, and stays an ordinary error.
        limited = self._capability_limit(
            "shopping_order_create",
            {
                "product_id": product_id,
                "quantity": quantity,
                "delivery_address_id": delivery_address_id,
                "payment_method_id": payment_method_id,
            },
            product,
        )
        if limited is not None:
            return limited
        if not product.get("in_stock"):
            raise ValueError(f"product is out of stock: {product_id}")
        addresses = json.loads(
            (self.state_root / "shipping" / "addresses.json").read_text(
                encoding="utf-8"
            )
        )
        address = next(
            (
                item
                for item in addresses.get("addresses", [])
                if item.get("address_id") == delivery_address_id
            ),
            None,
        )
        if address is None:
            raise ValueError(
                f"unknown delivery address: {delivery_address_id}. "
                "delivery_address_read lists every saved address id"
            )
        self._require_payment_method(payment_method_id)
        path = self.state_root / "shopping" / "order_store.json"
        store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        # The world's own shop, declared at the top of this document
        # (`order_merchant`); GlenMart when the fixture names none.
        # Only the code is used: putting the name on the record would change
        # what `shopping_order_read` returns, and the merchant is already
        # readable from the document itself.
        _, merchant_code = order_merchant(store)
        order_id = self._mint_order_id(store.get("orders", []), merchant_code)
        record = {
            "order_id": order_id,
            "product_id": product_id,
            "item_type": product.get("item_type"),
            "product_name": product.get("name"),
            "unit_price": product.get("price"),
            "currency": products.get("currency"),
            "quantity": quantity,
            "delivery_method": "home_delivery",
            "delivery_address_id": delivery_address_id,
            "payment_method_id": payment_method_id,
            "status": "placed",
            "placed_at": self._now(),
        }
        store.setdefault("orders", []).append(record)
        path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {"order_id": order_id, "status": "placed"}
        self.log_tool(
            "shopping_order_create",
            {
                "product_id": product_id,
                "quantity": quantity,
                "delivery_address_id": delivery_address_id,
                "payment_method_id": payment_method_id,
            },
            result,
        )
        self.log_state_diff(
            "shopping.orders",
            "append",
            order_id,
            f"Placed order for {product.get('name')}",
        )
        return result

    def shopping_order_read(self, order_id: str) -> dict[str, Any]:
        store = json.loads(
            (self.state_root / "shopping" / "order_store.json").read_text(
                encoding="utf-8"
            )
        )
        record = next(
            (
                item
                for item in store.get("orders", [])
                if item.get("order_id") == order_id
                and self._date_visible(item, "placed_at")
            ),
            None,
        )
        if record is None:
            raise ValueError(f"unknown order: {order_id}")
        # `placed_at` is wall-clock bookkeeping the PA must not read back. The
        # order's `delivery_address_id` stays out too: `delivery_address_read`
        # is where a saved address is looked up, so an order read has no reason
        # to be a second, rawer source for the same id.
        stable_record = {
            key: value
            for key, value in record.items()
            if key not in ("placed_at", "delivery_address_id")
        }
        result = {"order": stable_record}
        self.log_tool(
            "shopping_order_read",
            {"order_id": order_id},
            {"found": True},
        )
        return result

    def shopping_order_update(
        self,
        order_id: str,
        status: str | None = None,
        quantity: int | None = None,
        delivery_address_id: str | None = None,
    ) -> dict[str, Any]:
        _require_choice("status", status, ORDER_STATUSES)
        path = self.state_root / "shopping" / "order_store.json"
        store = json.loads(path.read_text(encoding="utf-8"))
        record = next(
            (
                item
                for item in store.get("orders", [])
                if item.get("order_id") == order_id
                and self._date_visible(item, "placed_at")
            ),
            None,
        )
        if record is None:
            raise ValueError(f"unknown order: {order_id}")
        if record.get("status") in ("shipped", "delivered", "cancelled"):
            raise ValueError(
                f"order can no longer be changed: {order_id} is {record.get('status')}"
            )
        changes: dict[str, Any] = {}
        if status is not None:
            changes["status"] = status
        if quantity is not None:
            if quantity < 1:
                raise ValueError("quantity must be positive")
            changes["quantity"] = quantity
        if delivery_address_id is not None:
            addresses = json.loads(
                (self.state_root / "shipping" / "addresses.json").read_text(
                    encoding="utf-8"
                )
            )
            if not any(
                item.get("address_id") == delivery_address_id
                for item in addresses.get("addresses", [])
            ):
                raise ValueError(
                f"unknown delivery address: {delivery_address_id}. "
                "delivery_address_read lists every saved address id"
            )
            changes["delivery_address_id"] = delivery_address_id
        if not changes:
            raise ValueError("shopping_order_update requires at least one field to change")
        record.update(changes)
        record["updated_at"] = self._now()
        path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {"order_id": order_id, "status": record.get("status")}
        self.log_tool("shopping_order_update", {"order_id": order_id, **changes}, result)
        self.log_state_diff(
            "shopping.orders",
            "write",
            order_id,
            f"Updated order {order_id}: {record.get('status')}",
        )
        return result

    def shopping_return_case_read(self, case_id: str) -> dict[str, Any]:
        path = self.state_root / "shopping" / "return_cases.jsonl"
        record = next(
            (row for row in self._read_jsonl(path) if row.get("case_id") == case_id),
            None,
        )
        if record is None:
            raise ValueError(f"unknown return case: {case_id}")
        result = {"case": dict(record)}
        self.log_tool("shopping_return_case_read", {"case_id": case_id}, {"found": True})
        return result

    def shopping_return_case_update(
        self,
        case_id: str,
        route: str,
        filing: str,
        note_draft_id: str,
    ) -> dict[str, Any]:
        self._enforce_gate("shopping_return_case_update", {})
        values = {"route": route, "filing": filing, "note_draft_id": note_draft_id}
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        path = self.state_root / "shopping" / "return_cases.jsonl"
        records = self._read_jsonl(path)
        index = next(
            (i for i, row in enumerate(records) if row.get("case_id") == case_id),
            None,
        )
        if index is None:
            raise ValueError(f"unknown return case: {case_id}")
        record = records[index]
        if record.get("status") != "open_on_hold":
            raise ValueError(
                f"return case cannot be changed: {case_id} is {record.get('status')}"
            )
        drafts = self._read_jsonl(self.state_root / "email" / "drafts.jsonl")
        if not any(row.get("draft_id") == note_draft_id for row in drafts):
            raise ValueError(f"unknown email draft: {note_draft_id}")

        replacement = {
            **record,
            "route": route.strip(),
            "filing": filing.strip(),
            "note_draft_id": note_draft_id,
            "updated_at": self._now(),
        }
        records[index] = replacement
        self._write_return_cases_atomically(path, records)
        result = {"case": dict(replacement)}
        self.log_tool(
            "shopping_return_case_update",
            {"case_id": case_id, **values},
            {"case_id": case_id, "status": replacement["status"]},
        )
        self.log_state_diff(
            "shopping.return_cases",
            "write",
            case_id,
            f"Updated return case {case_id}; status remains {replacement['status']}",
        )
        return result

    @staticmethod
    def _write_return_cases_atomically(
        path: Path, records: list[dict[str, Any]]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                for record in records:
                    temporary.write(json.dumps(record, ensure_ascii=False) + "\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    # Shipping

    def _delivery_addresses(self) -> list[dict[str, Any]]:
        path = self.state_root / "shipping" / "addresses.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("addresses", [])

    def delivery_address_read(self, address_id: str | None = None) -> dict[str, Any]:
        addresses = [
            record for record in self._delivery_addresses() if self._date_visible(record)
        ]
        if address_id is not None:
            addresses = [
                record for record in addresses if record.get("address_id") == address_id
            ]
            if not addresses:
                self.log_tool("delivery_address_read", {"address_id": address_id}, {"error": "not_found"}, status="error")
                raise FileNotFoundError(address_id)
        result = {"addresses": addresses}
        self.log_tool(
            "delivery_address_read",
            {"address_id": address_id},
            {"addresses": len(addresses)},
        )
        return result

    def package_tracking_read(self, tracking_id: str) -> dict[str, Any]:
        data = json.loads(
            (self.state_root / "shipping" / "tracking.json").read_text(
                encoding="utf-8"
            )
        )
        record = next(
            (
                item
                for item in data.get("packages", [])
                if item.get("tracking_id") == tracking_id
                and self._date_visible(item)
            ),
            None,
        )
        if record is None:
            self.log_tool("package_tracking_read", {"tracking_id": tracking_id}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(tracking_id)
        limited = self._capability_limit(
            "package_tracking_read", {"tracking_id": tracking_id}, record
        )
        if limited is not None:
            return limited
        events = [
            event
            for event in record.get("events", [])
            if self._date_visible(event, "date", "timestamp")
        ]
        status = events[-1].get("status") if events else record.get("status")
        result = {
            "tracking_id": tracking_id,
            "carrier": record.get("carrier"),
            "status": status,
            "events": events,
        }
        self.log_tool(
            "package_tracking_read",
            {"tracking_id": tracking_id},
            {"events": len(events), "status": status},
        )
        return result

    def package_delivery_proof_read(self, tracking_id: str) -> dict[str, Any]:
        data = json.loads(
            (self.state_root / "shipping" / "delivery_proofs.json").read_text(
                encoding="utf-8"
            )
        )
        record = next(
            (
                item
                for item in data.get("proofs", [])
                if item.get("tracking_id") == tracking_id
                and self._date_visible(item)
            ),
            None,
        )
        args = {"tracking_id": tracking_id}
        if record is None:
            self.log_tool(
                "package_delivery_proof_read", args, {"error": "not_found"}, status="error"
            )
            raise FileNotFoundError(tracking_id)
        limited = self._capability_limit("package_delivery_proof_read", args, record)
        if limited is not None:
            return limited
        self.log_tool("package_delivery_proof_read", args, {"found": True})
        return {"proof": self._public_record(record)}

    # Finance

    def utility_bill_read(self, account: str, period: str | None = None) -> dict[str, Any]:
        _require_period("period", period)
        data = json.loads(
            (self.state_root / "finance" / "bills.json").read_text(encoding="utf-8")
        )
        # The description has always told the PA to pass "the provider name as
        # the bill gives it", but the handler only ever matched `account`. For
        # most bills the provider is a substring of the account string, so that
        # held by accident; the four where it is not -- Fairhaven Renters
        # Insurance, Corvid Court Homeowners Association, Corvid Mutual,
        # Glenmont Credit Union -- raised not_found for the name printed on the
        # bill. Either field matching is enough.
        needle = account.casefold()
        bills = [
            record
            for record in data.get("bills", [])
            if (
                needle in str(record.get("account") or "").casefold()
                or needle in str(record.get("provider") or "").casefold()
            )
            and self._date_visible(record, "issued_date")
            and (period is None or str(record.get("period") or "") == period)
        ]
        if not bills:
            self.log_tool("utility_bill_read", {"account": account, "period": period}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(f"no bill for account: {account}")
        # Several bills can match one account and period -- a card statement and
        # the minimum due on it, or three subscriptions with one provider. The
        # handler used to sort by period and hand back `bills[0]`, so the other
        # matches were unreachable and the result did not say they existed.
        bills.sort(key=lambda record: str(record.get("period") or ""), reverse=True)
        result = {"account": account, "period": period, "bills": bills}
        self.log_tool(
            "utility_bill_read",
            {"account": account, "period": period},
            {"bills": len(bills)},
        )
        return result

    def utility_account_holder_update(
        self, account_id: str, remove_holder: str
    ) -> dict[str, Any]:
        self._enforce_gate("utility_account_holder_update", {})
        path = self.state_root / "finance" / "utility_accounts.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        record = next(
            (
                item
                for item in data.get("accounts", [])
                if item.get("account_id") == account_id
                and self._date_visible(item)
            ),
            None,
        )
        args = {"account_id": account_id, "remove_holder": remove_holder}
        if record is None:
            self.log_tool(
                "utility_account_holder_update", args, {"error": "not_found"}, status="error"
            )
            raise FileNotFoundError(account_id)
        limited = self._capability_limit("utility_account_holder_update", args, record)
        if limited is not None:
            return limited
        holders = list(record.get("holders") or [])
        exact = next(
            (
                holder
                for holder in holders
                if str(holder).casefold() == remove_holder.casefold()
            ),
            None,
        )
        if exact is None:
            raise ValueError(f"holder not on account: {remove_holder}")
        if len(holders) <= 1:
            raise ValueError("cannot remove the account's only holder")
        record["holders"] = [holder for holder in holders if holder != exact]
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        result = {
            "account_id": account_id,
            "status": "updated",
            "holders": record["holders"],
        }
        self.log_tool("utility_account_holder_update", args, result)
        self.log_state_diff(
            "finance.utility_accounts",
            "write",
            account_id,
            f"Removed an account holder from {account_id}",
        )
        return result

    def _payment_methods(self) -> list[dict[str, Any]]:
        path = self.state_root / "finance" / "payment_methods.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("payment_methods", [])

    def _require_payment_method(self, method_id: str) -> dict[str, Any]:
        method = next(
            (
                record
                for record in self._payment_methods()
                if record.get("method_id") == method_id
            ),
            None,
        )
        if method is None:
            raise ValueError(f"unknown payment method: {method_id}")
        return method

    def payment_method_read(self, method_id: str | None = None) -> dict[str, Any]:
        methods = [
            record for record in self._payment_methods() if self._date_visible(record)
        ]
        if method_id is not None:
            methods = [
                record for record in methods if record.get("method_id") == method_id
            ]
            if not methods:
                self.log_tool("payment_method_read", {"method_id": method_id}, {"error": "not_found"}, status="error")
                raise FileNotFoundError(method_id)
        result = {"payment_methods": methods}
        self.log_tool(
            "payment_method_read",
            {"method_id": method_id},
            {"methods": len(methods)},
        )
        return result

    def bill_payment_create(
        self,
        bill_id: str,
        payment_method_id: str,
        amount: float | None = None,
    ) -> dict[str, Any]:
        self._require_payment_method(payment_method_id)
        bills_path = self.state_root / "finance" / "bills.json"
        bills_data = json.loads(bills_path.read_text(encoding="utf-8"))
        bill = next(
            (
                record
                for record in bills_data.get("bills", [])
                if record.get("bill_id") == bill_id
                and self._date_visible(record, "issued_date")
            ),
            None,
        )
        if bill is None:
            raise ValueError(f"unknown bill: {bill_id}")
        if bill.get("status") == "paid":
            raise ValueError(f"bill is already paid: {bill_id}")
        bill_total = float(bill.get("total") or 0)
        if amount is not None and float(amount) != bill_total:
            raise ValueError(
                f"amount {amount} does not match the bill total {bill_total}"
            )
        payments_path = self.state_root / "finance" / "payments.jsonl"
        payment_id = self._mint_id(payments_path)
        record = {
            "payment_id": payment_id,
            "bill_id": bill_id,
            "account": bill.get("account"),
            "amount": bill_total,
            "payment_method_id": payment_method_id,
            "status": "completed",
            "paid_at": self._now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(payments_path, record)
        bill["status"] = "paid"
        bill["paid_by"] = payment_id
        bills_path.write_text(
            json.dumps(bills_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {"payment_id": payment_id, "status": "completed"}
        self.log_tool(
            "bill_payment_create",
            {"bill_id": bill_id, "payment_method_id": payment_method_id, "amount": amount},
            result,
        )
        self.log_state_diff(
            "finance.payments",
            "append",
            payment_id,
            f"Paid bill {bill_id}: {bill_total}",
        )
        return result

    def bill_payment_read(self, payment_id: str) -> dict[str, Any]:
        payments_path = self.state_root / "finance" / "payments.jsonl"
        for record in self._read_jsonl(payments_path):
            if record.get("payment_id") != payment_id:
                continue
            if not self._date_visible(record):
                continue
            self.log_tool("bill_payment_read", {"payment_id": payment_id}, {"found": True})
            return {"payment": record}
        self.log_tool("bill_payment_read", {"payment_id": payment_id}, {"error": "not_found"}, status="error")
        raise FileNotFoundError(payment_id)

    # Slotted reservation contexts (appointment / restaurant / room) share one
    # mechanical core; each context keeps its own store, ids, and tool names.

    def _slotted_resources(self, domain: str, filename: str, key: str) -> list[dict[str, Any]]:
        path = self.state_root / domain / filename
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get(key, [])

    def _slotted_search(
        self,
        *,
        tool: str,
        domain: str,
        filename: str,
        key: str,
        id_field: str,
        query: str,
        date: str | None,
    ) -> dict[str, Any]:
        _require_date("date", date)
        query_tokens = self._search_tokens(query)
        matches = []
        for record in self._slotted_resources(domain, filename, key):
            if not self._date_visible(record, "published_date"):
                continue
            haystack = json.dumps(record, ensure_ascii=False)
            if query_tokens and not (query_tokens & self._search_tokens(haystack)):
                continue
            slots = record.get("slots") or []
            if date is not None:
                slots = [slot for slot in slots if str(slot.get("date") or "") == date]
            shown = {k: v for k, v in record.items() if k != "slots"}
            shown["slots"] = slots
            matches.append(shown)
        result = {"query": query, "date": date, "matches": matches}
        self.log_tool(
            tool,
            {"query": query, "date": date},
            self._search_log_result(matches, id_field),
        )
        return result

    def _require_slot(
        self,
        *,
        domain: str,
        filename: str,
        key: str,
        id_field: str,
        resource_id: str,
        date: str,
        time: str,
    ) -> dict[str, Any]:
        # Every slotted create/update funnels through here, so one pair of
        # checks covers appointment, restaurant and room in both directions.
        _require_date("date", date)
        _require_time("time", time)
        # `_slotted_search` drops a resource the world has not published yet;
        # this path did not, so a booking could be made against a resource the
        # session could not have seen. Same check, same answer as an id that
        # matches nothing.
        resource = next(
            (
                record
                for record in self._slotted_resources(domain, filename, key)
                if record.get(id_field) == resource_id
                and self._date_visible(record, "published_date")
            ),
            None,
        )
        if resource is None:
            raise ValueError(f"unknown {id_field.removesuffix('_id')}: {resource_id}")
        slots = resource.get("slots")
        if not isinstance(slots, list):
            # A resource with no slots list is a broken fixture, not an open
            # calendar; the validator refuses it too.
            raise ValueError(f"{resource_id} has no slots list; nothing can be booked there")
        if not any(
            str(slot.get("date") or "") == date and str(slot.get("time") or "") == time
            for slot in slots
        ):
            raise ValueError(f"no open slot at {resource_id} on {date} {time}")
        return resource

    def _record_by_id(
        self, *, tool: str, path: Path, id_field: str, record_id: str, wrap: str
    ) -> dict[str, Any]:
        for record in self._read_jsonl(path):
            if record.get(id_field) != record_id:
                continue
            if not self._date_visible(record):
                continue
            self.log_tool(tool, {id_field: record_id}, {"found": True})
            return {wrap: record}
        self.log_tool(tool, {id_field: record_id}, {"error": "not_found"}, status="error")
        raise FileNotFoundError(record_id)

    # Appointment (clinics and services)

    def appointment_search(self, service: str, date: str | None = None) -> dict[str, Any]:
        return self._slotted_search(
            tool="appointment_search",
            domain="appointment",
            filename="availability.json",
            key="providers",
            id_field="provider_id",
            query=service,
            date=date,
        )

    def appointment_create(
        self,
        provider_id: str,
        date: str,
        time: str,
        patient: str,
        reason: str | None = None,
        booking_contact_id: str | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate("appointment_create", {"reason": reason})
        provider = self._require_slot(
            domain="appointment",
            filename="availability.json",
            key="providers",
            id_field="provider_id",
            resource_id=provider_id,
            date=date,
            time=time,
        )
        booking_route = None
        if routes := provider.get("booking_routes"):
            if booking_contact_id is None:
                raise ValueError(
                    f"provider {provider_id} requires a booking_contact_id from its booking_routes"
                )
            booking_route = next(
                (
                    route
                    for route in routes
                    if route.get("contact_id") == booking_contact_id
                ),
                None,
            )
            if booking_route is None:
                raise ValueError(
                    f"booking contact {booking_contact_id} is not a route for provider {provider_id}"
                )
        elif booking_contact_id is not None:
            raise ValueError(f"provider {provider_id} does not publish booking routes")
        booking_fields = (
            {
                "booking_contact_id": booking_contact_id,
                "booking_role": booking_route.get("role"),
            }
            if booking_route
            else {}
        )
        path = self.state_root / "appointment" / "appointments.jsonl"
        appointment_id = self._mint_id(path)
        record = {
            "appointment_id": appointment_id,
            "provider_id": provider_id,
            "provider_name": provider.get("name"),
            "location": provider.get("location"),
            "date": date,
            "time": time,
            "patient": patient,
            "reason": reason,
            **booking_fields,
            "status": "confirmed",
            "created_at": self._now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(path, record)
        result = {
            "appointment_id": appointment_id,
            "status": "confirmed",
            **booking_fields,
        }
        self.log_tool(
            "appointment_create",
            {
                "provider_id": provider_id,
                "date": date,
                "time": time,
                "patient": patient,
                "booking_contact_id": booking_contact_id,
                **self._payload_log_args("appointment_create", {"reason": reason}),
            },
            result,
        )
        self.log_state_diff(
            "appointment.appointments",
            "append",
            appointment_id,
            f"Booked {provider.get('name')} on {date} {time}",
        )
        return result

    def appointment_read(self, appointment_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="appointment_read",
            path=self.state_root / "appointment" / "appointments.jsonl",
            id_field="appointment_id",
            record_id=appointment_id,
            wrap="appointment",
        )

    def appointment_update(
        self,
        appointment_id: str,
        status: str | None = None,
        date: str | None = None,
        time: str | None = None,
    ) -> dict[str, Any]:
        _require_choice("status", status, BOOKING_STATUSES)
        path = self.state_root / "appointment" / "appointments.jsonl"
        records = self._read_jsonl(path)
        record = next(
            (r for r in records if r.get("appointment_id") == appointment_id), None
        )
        if record is None:
            self.log_tool("appointment_update", {"appointment_id": appointment_id}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(appointment_id)
        if record.get("status") == "cancelled":
            raise ValueError(f"appointment is already cancelled: {appointment_id}")
        changes: dict[str, Any] = {}
        if status is not None:
            changes["status"] = status
        if date is not None or time is not None:
            new_date = date or str(record.get("date"))
            new_time = time or str(record.get("time"))
            self._require_slot(
                domain="appointment",
                filename="availability.json",
                key="providers",
                id_field="provider_id",
                resource_id=str(record.get("provider_id")),
                date=new_date,
                time=new_time,
            )
            changes["date"] = new_date
            changes["time"] = new_time
        if not changes:
            raise ValueError("appointment_update requires at least one field to change")
        record.update(changes)
        record["updated_at"] = self._now()
        self._write_jsonl(path, records)
        result = {"appointment_id": appointment_id, "status": record.get("status")}
        self.log_tool("appointment_update", {"appointment_id": appointment_id, **changes}, result)
        self.log_state_diff(
            "appointment.appointments",
            "write",
            appointment_id,
            f"Updated appointment {appointment_id}: {record.get('status')}",
        )
        return result

    # Restaurant reservations

    def restaurant_reservation_search(
        self,
        query: str,
        date: str | None = None,
        party_size: int | None = None,
    ) -> dict[str, Any]:
        result = self._slotted_search(
            tool="restaurant_reservation_search",
            domain="restaurant",
            filename="slots.json",
            key="restaurants",
            id_field="restaurant_id",
            query=query,
            date=date,
        )
        if party_size is not None:
            for match in result["matches"]:
                match["slots"] = [
                    slot
                    for slot in match["slots"]
                    if _slot_max_party(slot, match["restaurant_id"]) >= party_size
                ]
        result["party_size"] = party_size
        return result

    def restaurant_reservation_create(
        self,
        restaurant_id: str,
        date: str,
        time: str,
        party_size: int,
        contact_phone: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate("restaurant_reservation_create", {"note": note})
        # The persona's own number is on no fixture, so only the user can supply
        # it; a blank or truncated value means the PA guessed instead of asking.
        if sum(char.isdigit() for char in str(contact_phone)) < 7:
            raise ValueError(
                f"contact_phone={contact_phone!r} is not a phone number "
                "(at least 7 digits, such as (555) 014-3388)"
            )
        restaurant = self._require_slot(
            domain="restaurant",
            filename="slots.json",
            key="restaurants",
            id_field="restaurant_id",
            resource_id=restaurant_id,
            date=date,
            time=time,
        )
        # `restaurant_reservation_search` already drops slots that cannot seat
        # the party, so create is the only path that could book past the cap.
        slot = next(
            (
                candidate
                for candidate in (restaurant.get("slots") or [])
                if str(candidate.get("date") or "") == date
                and str(candidate.get("time") or "") == time
            ),
            None,
        )
        max_party = _slot_max_party(slot, restaurant_id)
        if party_size > max_party:
            raise ValueError(
                f"party_size={party_size} exceeds max_party={max_party} for the "
                f"{date} {time} slot at {restaurant_id}"
            )
        path = self.state_root / "restaurant" / "reservations.jsonl"
        reservation_id = self._mint_id(path)
        record = {
            "reservation_id": reservation_id,
            "restaurant_id": restaurant_id,
            "restaurant_name": restaurant.get("name"),
            "date": date,
            "time": time,
            "party_size": party_size,
            "contact_phone": contact_phone,
            "note": note,
            "status": "confirmed",
            "created_at": self._now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(path, record)
        result = {"reservation_id": reservation_id, "status": "confirmed"}
        self.log_tool(
            "restaurant_reservation_create",
            {
                "restaurant_id": restaurant_id,
                "date": date,
                "time": time,
                "party_size": party_size,
                "contact_phone": contact_phone,
                **self._payload_log_args(
                    "restaurant_reservation_create", {"note": note}
                ),
            },
            result,
        )
        self.log_state_diff(
            "restaurant.reservations",
            "append",
            reservation_id,
            f"Reserved {restaurant.get('name')} on {date} {time} (contact {contact_phone})",
        )
        return result

    def restaurant_reservation_read(self, reservation_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="restaurant_reservation_read",
            path=self.state_root / "restaurant" / "reservations.jsonl",
            id_field="reservation_id",
            record_id=reservation_id,
            wrap="reservation",
        )

    def restaurant_reservation_update(
        self,
        reservation_id: str,
        status: str | None = None,
        date: str | None = None,
        time: str | None = None,
        party_size: int | None = None,
    ) -> dict[str, Any]:
        _require_choice("status", status, BOOKING_STATUSES)
        path = self.state_root / "restaurant" / "reservations.jsonl"
        records = self._read_jsonl(path)
        record = next(
            (r for r in records if r.get("reservation_id") == reservation_id), None
        )
        if record is None:
            self.log_tool("restaurant_reservation_update", {"reservation_id": reservation_id}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(reservation_id)
        if record.get("status") == "cancelled":
            raise ValueError(f"reservation is already cancelled: {reservation_id}")
        changes: dict[str, Any] = {}
        if status is not None:
            changes["status"] = status
        if date is not None or time is not None:
            new_date = date or str(record.get("date"))
            new_time = time or str(record.get("time"))
            self._require_slot(
                domain="restaurant",
                filename="slots.json",
                key="restaurants",
                id_field="restaurant_id",
                resource_id=str(record.get("restaurant_id")),
                date=new_date,
                time=new_time,
            )
            changes["date"] = new_date
            changes["time"] = new_time
        if party_size is not None:
            changes["party_size"] = party_size
        if not changes:
            raise ValueError("restaurant_reservation_update requires at least one field to change")
        record.update(changes)
        record["updated_at"] = self._now()
        self._write_jsonl(path, records)
        result = {"reservation_id": reservation_id, "status": record.get("status")}
        self.log_tool("restaurant_reservation_update", {"reservation_id": reservation_id, **changes}, result)
        self.log_state_diff(
            "restaurant.reservations",
            "write",
            reservation_id,
            f"Updated reservation {reservation_id}: {record.get('status')}",
        )
        return result

    # Workplace room bookings

    def room_booking_search(self, query: str, date: str | None = None) -> dict[str, Any]:
        return self._slotted_search(
            tool="room_booking_search",
            domain="room",
            filename="rooms.json",
            key="rooms",
            id_field="room_id",
            query=query,
            date=date,
        )

    def room_booking_create(
        self,
        room_id: str,
        date: str,
        time: str,
        end_time: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate("room_booking_create", {"purpose": purpose})
        _require_time("end_time", end_time)
        room = self._require_slot(
            domain="room",
            filename="rooms.json",
            key="rooms",
            id_field="room_id",
            resource_id=room_id,
            date=date,
            time=time,
        )
        path = self.state_root / "room" / "bookings.jsonl"
        booking_id = self._mint_id(path)
        record = {
            "booking_id": booking_id,
            "room_id": room_id,
            "room_name": room.get("name"),
            "date": date,
            "time": time,
            "end_time": end_time,
            "purpose": purpose,
            "status": "confirmed",
            "created_at": self._now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(path, record)
        result = {"booking_id": booking_id, "status": "confirmed"}
        self.log_tool(
            "room_booking_create",
            {
                "room_id": room_id,
                "date": date,
                "time": time,
                **self._payload_log_args("room_booking_create", {"purpose": purpose}),
            },
            result,
        )
        self.log_state_diff(
            "room.bookings",
            "append",
            booking_id,
            f"Booked {room.get('name')} on {date} {time}",
        )
        return result

    def room_booking_read(self, booking_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="room_booking_read",
            path=self.state_root / "room" / "bookings.jsonl",
            id_field="booking_id",
            record_id=booking_id,
            wrap="booking",
        )

    def room_booking_update(
        self,
        booking_id: str,
        status: str | None = None,
        date: str | None = None,
        time: str | None = None,
    ) -> dict[str, Any]:
        _require_choice("status", status, BOOKING_STATUSES)
        path = self.state_root / "room" / "bookings.jsonl"
        records = self._read_jsonl(path)
        record = next(
            (r for r in records if r.get("booking_id") == booking_id), None
        )
        if record is None:
            self.log_tool("room_booking_update", {"booking_id": booking_id}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(booking_id)
        if record.get("status") == "cancelled":
            raise ValueError(f"room booking is already cancelled: {booking_id}")
        changes: dict[str, Any] = {}
        if status is not None:
            changes["status"] = status
        if date is not None or time is not None:
            new_date = date or str(record.get("date"))
            new_time = time or str(record.get("time"))
            self._require_slot(
                domain="room",
                filename="rooms.json",
                key="rooms",
                id_field="room_id",
                resource_id=str(record.get("room_id")),
                date=new_date,
                time=new_time,
            )
            changes["date"] = new_date
            changes["time"] = new_time
        if not changes:
            raise ValueError("room_booking_update requires at least one field to change")
        record.update(changes)
        record["updated_at"] = self._now()
        self._write_jsonl(path, records)
        result = {"booking_id": booking_id, "status": record.get("status")}
        self.log_tool("room_booking_update", {"booking_id": booking_id, **changes}, result)
        self.log_state_diff(
            "room.bookings",
            "write",
            booking_id,
            f"Updated room booking {booking_id}: {record.get('status')}",
        )
        return result

    # Hotel stays

    def _hotels(self) -> list[dict[str, Any]]:
        path = self.state_root / "hotel" / "hotels.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("hotels", [])

    def hotel_search(self, city: str, check_in: str, check_out: str) -> dict[str, Any]:
        self._require_stay_dates(check_in, check_out)
        visible = [
            record
            for record in self._hotels()
            if self._date_visible(record, "published_date")
        ]
        city = _require_known_location(
            "city",
            city,
            [str(record.get("city") or "") for record in visible],
            "a city name as the hotel records give it",
            self._location_kind_names(),
        )
        needle = city.casefold()
        matches = [
            record
            for record in visible
            if needle in str(record.get("city") or "").casefold()
        ]
        result = {
            "city": city,
            "check_in": check_in,
            "check_out": check_out,
            "matches": matches,
        }
        self.log_tool(
            "hotel_search",
            {"city": city, "check_in": check_in, "check_out": check_out},
            self._search_log_result(matches, "hotel_id"),
        )
        return result

    @staticmethod
    def _require_stay_dates(check_in: str, check_out: str) -> int:
        from datetime import date as _date

        _require_date("check_in", check_in)
        _require_date("check_out", check_out)
        nights = (_date.fromisoformat(check_out) - _date.fromisoformat(check_in)).days
        if nights < 1:
            raise ValueError(
                f"check_out={check_out!r} must be after check_in={check_in!r}"
            )
        return nights

    def hotel_book(
        self,
        hotel_id: str,
        check_in: str,
        check_out: str,
        room_type: str,
        guest: str,
        payment_method_id: str,
    ) -> dict[str, Any]:
        nights = self._require_stay_dates(check_in, check_out)
        hotel = next(
            (
                record
                for record in self._hotels()
                if record.get("hotel_id") == hotel_id
                and self._date_visible(record, "published_date")
            ),
            None,
        )
        if hotel is None:
            raise ValueError(f"unknown hotel: {hotel_id}")
        rates = hotel.get("nightly_rates") or {}
        if rates and room_type not in rates:
            raise ValueError(f"room type {room_type} is not offered at {hotel_id}")
        self._require_payment_method(payment_method_id)
        nightly = float(rates.get(room_type) or 0)
        path = self.state_root / "hotel" / "bookings.jsonl"
        booking_id = self._mint_id(path)
        record = {
            "booking_id": booking_id,
            "hotel_id": hotel_id,
            "hotel_name": hotel.get("name"),
            "city": hotel.get("city"),
            "check_in": check_in,
            "check_out": check_out,
            "nights": nights,
            "room_type": room_type,
            "nightly_rate": nightly,
            "total": round(nightly * nights, 2),
            "guest": guest,
            "payment_method_id": payment_method_id,
            "status": "booked",
            "booked_at": self._now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(path, record)
        result = {"booking_id": booking_id, "status": "booked"}
        self.log_tool(
            "hotel_book",
            {
                "hotel_id": hotel_id,
                "check_in": check_in,
                "check_out": check_out,
                "room_type": room_type,
                "guest": guest,
                "payment_method_id": payment_method_id,
            },
            result,
        )
        self.log_state_diff(
            "hotel.bookings",
            "append",
            booking_id,
            f"Booked {hotel.get('name')} {check_in}..{check_out}",
        )
        return result

    def hotel_booking_read(self, booking_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="hotel_booking_read",
            path=self.state_root / "hotel" / "bookings.jsonl",
            id_field="booking_id",
            record_id=booking_id,
            wrap="booking",
        )

    def hotel_booking_update(
        self,
        booking_id: str,
        status: str | None = None,
        check_in: str | None = None,
        check_out: str | None = None,
        room_type: str | None = None,
    ) -> dict[str, Any]:
        _require_choice("status", status, HOTEL_STATUSES)
        path = self.state_root / "hotel" / "bookings.jsonl"
        records = self._read_jsonl(path)
        record = next(
            (r for r in records if r.get("booking_id") == booking_id), None
        )
        if record is None:
            self.log_tool("hotel_booking_update", {"booking_id": booking_id}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(booking_id)
        if record.get("status") == "cancelled":
            raise ValueError(f"hotel booking is already cancelled: {booking_id}")
        changes: dict[str, Any] = {}
        if status is not None:
            changes["status"] = status
        if check_in is not None or check_out is not None or room_type is not None:
            new_in = check_in or str(record.get("check_in"))
            new_out = check_out or str(record.get("check_out"))
            nights = self._require_stay_dates(new_in, new_out)
            new_type = room_type or str(record.get("room_type"))
            hotel = next(
                (
                    h
                    for h in self._hotels()
                    if h.get("hotel_id") == record.get("hotel_id")
                ),
                None,
            )
            rates = (hotel or {}).get("nightly_rates") or {}
            if rates and new_type not in rates:
                raise ValueError(
                    f"room type {new_type} is not offered at {record.get('hotel_id')}"
                )
            nightly = float(rates.get(new_type) or record.get("nightly_rate") or 0)
            changes.update({
                "check_in": new_in,
                "check_out": new_out,
                "nights": nights,
                "room_type": new_type,
                "nightly_rate": nightly,
                "total": round(nightly * nights, 2),
            })
        if not changes:
            raise ValueError("hotel_booking_update requires at least one field to change")
        record.update(changes)
        record["updated_at"] = self._now()
        self._write_jsonl(path, records)
        result = {"booking_id": booking_id, "status": record.get("status")}
        self.log_tool("hotel_booking_update", {"booking_id": booking_id, **changes}, result)
        self.log_state_diff(
            "hotel.bookings",
            "write",
            booking_id,
            f"Updated hotel booking {booking_id}: {record.get('status')}",
        )
        return result

    # Reimbursement (department finance portal)

    def reimbursement_submission_create(
        self,
        purpose: str,
        amount: float,
        attachments: list[str] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        self._enforce_gate(
            "reimbursement_submission_create", {"purpose": purpose, "notes": notes}
        )
        if amount <= 0:
            raise ValueError("amount must be positive")
        path = self.state_root / "reimbursement" / "submissions.jsonl"
        submission_id = self._mint_id(path)
        record = {
            "submission_id": submission_id,
            "purpose": purpose,
            "amount": amount,
            "attachments": self._resolve_attachments(attachments),
            "notes": notes,
            "status": "submitted",
            "submitted_at": self._now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(path, record)
        result = {"submission_id": submission_id, "status": "submitted"}
        self.log_tool(
            "reimbursement_submission_create",
            {
                "amount": amount,
                **self._payload_log_args(
                    "reimbursement_submission_create",
                    {"purpose": purpose, "notes": notes},
                ),
            },
            result,
        )
        self.log_state_diff(
            "reimbursement.submissions",
            "append",
            submission_id,
            f"Filed reimbursement: {purpose}",
        )
        return result

    def reimbursement_submission_read(self, submission_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="reimbursement_submission_read",
            path=self.state_root / "reimbursement" / "submissions.jsonl",
            id_field="submission_id",
            record_id=submission_id,
            wrap="submission",
        )

    # Conference registration (external event portal)

    def conference_registration_create(
        self,
        conference: str,
        fee: float,
        payment_method_id: str,
        attendee: str | None = None,
    ) -> dict[str, Any]:
        if fee < 0:
            raise ValueError("fee must not be negative")
        self._require_payment_method(payment_method_id)
        path = self.state_root / "conference_registration" / "registrations.jsonl"
        registration_id = self._mint_id(path)
        record = {
            "registration_id": registration_id,
            "conference": conference,
            "fee": fee,
            "payment_method_id": payment_method_id,
            "attendee": attendee or self.user_id,
            "status": "registered",
            "registered_at": self._now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(path, record)
        result = {"registration_id": registration_id, "status": "registered"}
        self.log_tool(
            "conference_registration_create",
            {"conference": conference, "fee": fee, "payment_method_id": payment_method_id},
            result,
        )
        self.log_state_diff(
            "conference_registration.registrations",
            "append",
            registration_id,
            f"Registered for {conference}",
        )
        return result

    def conference_registration_read(self, registration_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="conference_registration_read",
            path=self.state_root / "conference_registration" / "registrations.jsonl",
            id_field="registration_id",
            record_id=registration_id,
            wrap="registration",
        )

    # Grade submission (university roster portal)

    def grade_submission_create(
        self,
        course: str,
        roster: dict[str, Any],
        term: str | None = None,
    ) -> dict[str, Any]:
        if not roster:
            raise ValueError("roster must not be empty")
        path = self.state_root / "grade_submission" / "submissions.jsonl"
        submission_id = self._mint_id(path)
        record = {
            "submission_id": submission_id,
            "course": course,
            "term": term,
            "roster": roster,
            "status": "submitted",
            "submitted_at": self._now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(path, record)
        result = {"submission_id": submission_id, "status": "submitted"}
        self.log_tool(
            "grade_submission_create",
            {"course": course, "term": term, "students": len(roster)},
            result,
        )
        self.log_state_diff(
            "grade_submission.submissions",
            "append",
            submission_id,
            f"Submitted grades: {course}",
        )
        return result

    def grade_submission_read(self, submission_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="grade_submission_read",
            path=self.state_root / "grade_submission" / "submissions.jsonl",
            id_field="submission_id",
            record_id=submission_id,
            wrap="submission",
        )

    # Flights

    def _flights(self) -> list[dict[str, Any]]:
        path = self.state_root / "flight" / "flights.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("flights", [])

    def flight_search(self, origin: str, destination: str, date: str) -> dict[str, Any]:
        _require_date("date", date)
        visible = [
            record
            for record in self._flights()
            if self._date_visible(record, "published_date")
        ]
        kinds = self._location_kind_names()
        takes = "an airport code or city as the flight records name it"
        origin = _require_known_location(
            "origin",
            origin,
            [
                " ".join(
                    str(record.get(key) or "") for key in ("origin", "origin_city")
                ).strip()
                for record in visible
            ],
            takes,
            kinds,
        )
        destination = _require_known_location(
            "destination",
            destination,
            [
                " ".join(
                    str(record.get(key) or "")
                    for key in ("destination", "destination_city")
                ).strip()
                for record in visible
            ],
            takes,
            kinds,
        )
        needle_origin = origin.casefold()
        needle_destination = destination.casefold()
        matches = [
            record
            for record in visible
            if needle_origin in " ".join(
                str(record.get(key) or "") for key in ("origin", "origin_city")
            ).casefold()
            and needle_destination in " ".join(
                str(record.get(key) or "")
                for key in ("destination", "destination_city")
            ).casefold()
            and str(record.get("date") or "") == date
        ]
        result = {
            "origin": origin,
            "destination": destination,
            "date": date,
            "matches": matches,
        }
        self.log_tool(
            "flight_search",
            {"origin": origin, "destination": destination, "date": date},
            self._search_log_result(matches, "flight_id"),
        )
        return result

    def flight_read(self, flight_id: str) -> dict[str, Any]:
        record = next(
            (
                item
                for item in self._flights()
                if item.get("flight_id") == flight_id
                and self._date_visible(item, "published_date")
            ),
            None,
        )
        if record is None:
            self.log_tool("flight_read", {"flight_id": flight_id}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(flight_id)
        self.log_tool("flight_read", {"flight_id": flight_id}, {"found": True})
        return {"flight": record}

    def flight_book(
        self,
        flight_id: str,
        passenger: str,
        payment_method_id: str,
        fare_class: str | None = None,
    ) -> dict[str, Any]:
        flight = next(
            (
                item
                for item in self._flights()
                if item.get("flight_id") == flight_id
                and self._date_visible(item, "published_date")
            ),
            None,
        )
        if flight is None:
            raise ValueError(f"unknown flight: {flight_id}")
        if flight.get("seats_available") == 0:
            raise ValueError(f"flight is sold out: {flight_id}")
        self._require_payment_method(payment_method_id)
        fares = flight.get("fares") or {}
        if fare_class is not None:
            if fares and fare_class not in fares:
                raise ValueError(
                    f"fare class {fare_class} is not offered on {flight_id}"
                )
            selected_class = fare_class
        else:
            selected_class = next(iter(fares), None)
        bookings_path = self.state_root / "flight" / "bookings.jsonl"
        booking_id = self._mint_id(bookings_path)
        record = {
            "booking_id": booking_id,
            "flight_id": flight_id,
            "carrier": flight.get("carrier"),
            "origin": flight.get("origin"),
            "destination": flight.get("destination"),
            "date": flight.get("date"),
            "depart_time": flight.get("depart_time"),
            "passenger": passenger,
            "fare_class": selected_class,
            "fare": fares.get(selected_class) if selected_class else None,
            "payment_method_id": payment_method_id,
            "status": "booked",
            "booked_at": self._now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(bookings_path, record)
        result = {"booking_id": booking_id, "status": "booked"}
        self.log_tool(
            "flight_book",
            {
                "flight_id": flight_id,
                "passenger": passenger,
                "payment_method_id": payment_method_id,
                "fare_class": fare_class,
            },
            result,
        )
        self.log_state_diff(
            "flight.bookings",
            "append",
            booking_id,
            f"Booked flight {flight_id} for {passenger}",
        )
        return result

    def flight_booking_read(self, booking_id: str) -> dict[str, Any]:
        bookings_path = self.state_root / "flight" / "bookings.jsonl"
        for record in self._read_jsonl(bookings_path):
            if record.get("booking_id") != booking_id:
                continue
            if not self._date_visible(record):
                continue
            self.log_tool("flight_booking_read", {"booking_id": booking_id}, {"found": True})
            return {"booking": record}
        self.log_tool("flight_booking_read", {"booking_id": booking_id}, {"error": "not_found"}, status="error")
        raise FileNotFoundError(booking_id)

    # Scholarly literature

    def _papers(self) -> list[dict[str, Any]]:
        path = self.state_root / "scholarly" / "papers.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8")).get("papers", [])

    def scholarly_paper_search(self, query: str) -> dict[str, Any]:
        query_tokens = self._search_tokens(query)
        scored = []
        for record in self._papers():
            if not self._date_visible(record, "published_date"):
                continue
            public = self._public_record(record)
            haystack = json.dumps(public, ensure_ascii=False)
            score = len(query_tokens & self._search_tokens(haystack))
            if self._query_matches(query, haystack):
                score += 10
            if score:
                summary = {k: v for k, v in public.items() if k != "fulltext"}
                scored.append((score, summary))
        scored.sort(key=lambda item: item[0], reverse=True)
        matches = [record for _, record in scored]
        result = {"query": query, "matches": matches}
        self.log_tool(
            "scholarly_paper_search",
            {"query": query},
            self._search_log_result(matches, "paper_id"),
        )
        return result

    def scholarly_paper_read(self, paper_id: str) -> dict[str, Any]:
        record = next(
            (
                item
                for item in self._papers()
                if item.get("paper_id") == paper_id
                and self._date_visible(item, "published_date")
            ),
            None,
        )
        if record is None:
            self.log_tool("scholarly_paper_read", {"paper_id": paper_id}, {"error": "not_found"}, status="error")
            raise FileNotFoundError(paper_id)
        limited = self._capability_limit(
            "scholarly_paper_read", {"paper_id": paper_id}, record
        )
        if limited is not None:
            return limited
        self.log_tool("scholarly_paper_read", {"paper_id": paper_id}, {"found": True})
        return {"paper": record}

    # Recipes

    def recipe_search(
        self,
        query: str,
        max_minutes: int | None = None,
        dietary: str | None = None,
    ) -> dict[str, Any]:
        path = self.state_root / "recipe" / "recipes.json"
        recipes = []
        if path.exists():
            recipes = json.loads(path.read_text(encoding="utf-8")).get("recipes", [])
        matches = []
        for record in recipes:
            if not self._date_visible(record):
                continue
            haystack = " ".join(
                [str(record.get("name") or "")]
                + [str(item) for item in record.get("ingredients") or []]
                + [str(tag) for tag in record.get("tags") or []]
            )
            if not self._query_matches(query, haystack):
                continue
            if max_minutes is not None and int(record.get("total_minutes") or 0) > max_minutes:
                continue
            if dietary is not None and not any(
                dietary.casefold() in str(tag).casefold()
                for tag in record.get("tags") or []
            ):
                continue
            matches.append(record)
        result = {
            "query": query,
            "max_minutes": max_minutes,
            "dietary": dietary,
            "matches": matches,
        }
        self.log_tool(
            "recipe_search",
            {"query": query, "max_minutes": max_minutes, "dietary": dietary},
            {"matches": len(matches)},
        )
        return result

    # Persona layer: comic_store / cinema / exhibition / household

    def comic_store_inventory_read(self, item: str) -> dict[str, Any]:
        inventory_path = self.state_root / "comic_store" / "items.json"
        # A persona layer, so a world without a comic shop has no store file.
        # Same answer the cinema gives then (`_showtimes_on`): the read still
        # returns its own shape and logs, with nothing matched.
        if not inventory_path.exists():
            items = []
        else:
            items = json.loads(inventory_path.read_text(encoding="utf-8"))
            if isinstance(items, dict):
                items = items.get("items") or []
        needle_item = item.casefold()
        matches = [
            record for record in items
            if self._date_visible(record)
            and needle_item in str(record.get("item") or "").casefold()
        ]
        result = {"item": item, "matches": matches}
        self.log_tool(
            "comic_store_inventory_read",
            {"item": item},
            {"matches": len(matches)},
        )
        return result

    def _showtimes_on(self, cinema: str, movie: str, date: str) -> list[dict[str, Any]]:
        showtimes_path = self.state_root / "cinema" / "showtimes.json"
        if not showtimes_path.exists():
            return []
        showtimes = json.loads(showtimes_path.read_text(encoding="utf-8"))
        if isinstance(showtimes, dict):
            showtimes = showtimes.get("showtimes") or []
        needle_cinema = cinema.casefold()
        needle_movie = movie.casefold()
        return [
            record for record in showtimes
            if self._date_visible(record, "published_date")
            and needle_cinema in str(record.get("cinema") or "").casefold()
            and needle_movie in str(record.get("movie") or "").casefold()
            and str(record.get("date") or "") == date
        ]

    def cinema_showtime_search(self, cinema: str, movie: str, date: str) -> dict[str, Any]:
        _require_date("date", date)
        matches = self._showtimes_on(cinema, movie, date)
        result = {"cinema": cinema, "movie": movie, "date": date, "matches": matches}
        self.log_tool(
            "cinema_showtime_search",
            {"cinema": cinema, "movie": movie, "date": date},
            {"matches": len(matches)},
        )
        return result

    def buy_cinema_tickets(
        self,
        cinema: str,
        movie: str,
        date: str,
        time: str,
        quantity: int,
        payment_method_id: str,
    ) -> dict[str, Any]:
        if quantity < 1:
            raise ValueError("quantity must be positive")
        _require_date("date", date)
        _require_time("time", time)
        # Same rule as every other slotted purchase: the slot has to exist.
        # Matching is the one `cinema_showtime_search` uses, so the strings the
        # PA got back from a search buy the screening it saw.
        if not any(
            str(record.get("time") or "") == time
            for record in self._showtimes_on(cinema, movie, date)
        ):
            raise ValueError(f"no screening of {movie} at {cinema} on {date} {time}")
        self._require_payment_method(payment_method_id)
        tickets_path = self.state_root / "cinema" / "tickets.jsonl"
        ticket_id = self._mint_id(tickets_path)
        record = {
            "ticket_id": ticket_id,
            "cinema": cinema,
            "movie": movie,
            "date": date,
            "time": time,
            "quantity": quantity,
            "payment_method_id": payment_method_id,
            "purchased_at": self._now(),
            "run_id": self.run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(tickets_path, record)
        result = {"ticket_id": ticket_id, "status": "purchased"}
        self.log_tool(
            "cinema_ticket_purchase",
            {
                "cinema": cinema,
                "movie": movie,
                "date": date,
                "time": time,
                "quantity": quantity,
                "payment_method_id": payment_method_id,
            },
            result,
        )
        self.log_state_diff("cinema.tickets", "append", ticket_id, f"Bought tickets: {movie}")
        return result

    def cinema_ticket_read(self, ticket_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="cinema_ticket_read",
            path=self.state_root / "cinema" / "tickets.jsonl",
            id_field="ticket_id",
            record_id=ticket_id,
            wrap="ticket",
        )

    def exhibition_search(
        self,
        query: str | None = None,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Exhibitions and their open days, by name or by a day they are open.

        `exhibition_ticket_purchase` validates `venue`, `exhibit` and `date`
        against this store, and nothing read it: the PA was pointed at the
        `place_search` listing instead, which carries the hours but not the
        per-day slots the purchase actually checks. The cinema parallel --
        `cinema_showtime_search` before `cinema_ticket_purchase` -- is what this
        restores. Returns the fields the purchase validates on, so a search
        result can be passed straight through.
        """
        if date is not None:
            _require_date("date", date)
        path = self.state_root / "exhibition" / "exhibitions.json"
        exhibitions: list[dict[str, Any]] = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            exhibitions = (
                data.get("exhibitions") or [] if isinstance(data, dict) else data
            )
        matches = []
        for record in exhibitions:
            if not self._date_visible(record, "published_date"):
                continue
            if query is not None and not self._query_matches(
                query, f"{record.get('name') or ''} {record.get('venue') or ''}"
            ):
                continue
            slots = [
                str(slot.get("date") or "") for slot in record.get("slots") or []
            ]
            if date is not None and date not in slots:
                continue
            matches.append({
                "name": record.get("name"),
                "venue": record.get("venue"),
                "note": record.get("note"),
                "published_date": record.get("published_date"),
                "open_dates": slots,
            })
        result = {"query": query, "date": date, "matches": matches}
        self.log_tool(
            "exhibition_search",
            {"query": query, "date": date},
            {"matches": len(matches)},
        )
        return result

    def _require_open_exhibition_day(self, venue: str, exhibit: str, date: str) -> None:
        """Reject admission for a day the exhibition is not open.

        The exhibition parallel to `cinema/showtimes.json`: `exhibition_search`
        reads the open days out of this file and the write path checks the day
        against the same records, the way every other purchase checks its slot.
        Optional: a persona whose fixture has no `exhibitions.json` keeps the
        old unchecked behaviour rather than failing every purchase.
        """
        path = self.state_root / "exhibition" / "exhibitions.json"
        if not path.exists():
            return
        exhibitions = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(exhibitions, dict):
            exhibitions = exhibitions.get("exhibitions") or []
        visible = [
            record
            for record in exhibitions
            if self._date_visible(record, "published_date")
        ]
        venue = _require_known_location(
            "venue",
            venue,
            [str(record.get("venue") or "") for record in visible],
            "a venue name as exhibition_search gives it",
            self._location_kind_names(),
        )
        needle_venue = venue.casefold()
        needle_exhibit = exhibit.casefold()
        listed = [
            record
            for record in visible
            if needle_exhibit in str(record.get("name") or "").casefold()
            and needle_venue in str(record.get("venue") or "").casefold()
        ]
        if not listed:
            raise ValueError(f"unknown exhibition: {exhibit} at {venue}")
        if not any(
            str(slot.get("date") or "") == date
            for record in listed
            for slot in record.get("slots") or []
        ):
            raise ValueError(f"no open day for {exhibit} at {venue} on {date}")

    def exhibition_ticket_purchase(
        self,
        venue: str,
        exhibit: str,
        date: str,
        quantity: int,
        payment_method_id: str,
    ) -> dict[str, Any]:
        if quantity < 1:
            raise ValueError("quantity must be positive")
        _require_date("date", date)
        self._require_open_exhibition_day(venue, exhibit, date)
        self._require_payment_method(payment_method_id)
        tickets_path = self.state_root / "exhibition" / "tickets.jsonl"
        ticket_id = self._mint_id(tickets_path)
        record = {
            "ticket_id": ticket_id,
            "venue": venue,
            "exhibit": exhibit,
            "date": date,
            "quantity": quantity,
            "payment_method_id": payment_method_id,
            "purchased_at": self._now(),
            "run_id": self.run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        self._append_jsonl(tickets_path, record)
        result = {"ticket_id": ticket_id, "status": "purchased"}
        self.log_tool(
            "exhibition_ticket_purchase",
            {
                "venue": venue,
                "exhibit": exhibit,
                "date": date,
                "quantity": quantity,
                "payment_method_id": payment_method_id,
            },
            result,
        )
        self.log_state_diff("exhibition.tickets", "append", ticket_id, f"Bought admission: {exhibit}")
        return result

    def exhibition_ticket_read(self, ticket_id: str) -> dict[str, Any]:
        return self._record_by_id(
            tool="exhibition_ticket_read",
            path=self.state_root / "exhibition" / "tickets.jsonl",
            id_field="ticket_id",
            record_id=ticket_id,
            wrap="ticket",
        )

    def household_inventory_search(
        self,
        location: str,
        item: str,
    ) -> dict[str, Any]:
        inventory_path = self.state_root / "household" / "inventory.json"
        # As for the comic shop above: a world without a household inventory
        # answers with an empty match list rather than raising.
        data = (
            json.loads(inventory_path.read_text(encoding="utf-8"))
            if inventory_path.exists()
            else {}
        )
        visible = [
            record for record in data.get("items", []) if self._date_visible(record)
        ]
        location = _require_known_location(
            "location",
            location,
            [str(record.get("location") or "") for record in visible],
            "a place in the home as the inventory names it",
            self._location_kind_names(),
        )
        location_text = location.casefold()
        item_text = item.casefold()
        matches = [
            record
            for record in visible
            if location_text in str(record.get("location") or "").casefold()
            and item_text in str(record.get("item") or "").casefold()
        ]
        result = {
            "location": location,
            "item": item,
            "usable_item_available": any(
                bool(record.get("usable")) and int(record.get("quantity") or 0) > 0
                for record in matches
            ),
            "matches": matches,
        }
        self.log_tool(
            "household_inventory_search",
            {"location": location, "item": item},
            {
                **self._search_log_result(matches, "id"),
                "usable": result["usable_item_available"],
            },
        )
        return result

    # Professional media transcode

    def _media_transcode_services(self) -> list[dict[str, Any]]:
        path = self.state_root / "media_transcode" / "services.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8")).get("services", [])

    def media_transcode_service_search(
        self, source_format: str, target_format: str, deliver_by: str
    ) -> dict[str, Any]:
        _require_date("deliver_by", deliver_by)
        matches = []
        resolved_target = target_format
        for service in self._media_transcode_services():
            if not self._date_visible(service):
                continue
            if source_format.casefold() not in {
                str(value).casefold()
                for value in service.get("accepted_source_formats", [])
            }:
                continue
            offered_targets = [str(value) for value in service.get("target_formats", [])]
            canonical_target = next(
                (
                    value
                    for value in offered_targets
                    if value.casefold() == target_format.casefold()
                    or value.casefold().startswith(target_format.casefold() + "/")
                ),
                None,
            )
            if canonical_target is None:
                continue
            resolved_target = canonical_target
            promised = str(service.get("promised_delivery_at") or "")
            if not promised or promised[:10] > deliver_by:
                continue
            matches.append(
                {
                    "service_id": service["service_id"],
                    "provider": service["provider"],
                    "fee": service["fee"],
                    "currency": service["currency"],
                    "promised_delivery_at": promised,
                    "source_manifest_path": service["source_manifest_path"],
                    "required_source_locator": service["required_source_locator"],
                    "delivery_terms": service["delivery_terms"],
                }
            )
        result = {
            "source_format": source_format,
            "target_format": resolved_target,
            "deliver_by": deliver_by,
            "matches": matches,
        }
        self.log_tool(
            "media_transcode_service_search",
            {
                "source_format": source_format,
                "target_format": target_format,
                "deliver_by": deliver_by,
            },
            self._search_log_result(matches, "service_id"),
        )
        return result

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _bmff_boxes(
        data: bytes, start: int, end: int
    ) -> list[tuple[bytes, int, int]]:
        boxes = []
        cursor = start
        while cursor < end:
            if end - cursor < 8:
                raise ValueError("truncated MP4 box header")
            size = int.from_bytes(data[cursor : cursor + 4], "big")
            kind = data[cursor + 4 : cursor + 8]
            header = 8
            if size == 1:
                if end - cursor < 16:
                    raise ValueError("truncated extended MP4 box header")
                size = int.from_bytes(data[cursor + 8 : cursor + 16], "big")
                header = 16
            elif size == 0:
                size = end - cursor
            if size < header or cursor + size > end:
                raise ValueError("invalid MP4 box size")
            boxes.append((kind, cursor + header, cursor + size))
            cursor += size
        return boxes

    @classmethod
    def _validate_mp4_h264(cls, path: Path) -> None:
        data = path.read_bytes()
        top = cls._bmff_boxes(data, 0, len(data))
        if not top or top[0][0] != b"ftyp":
            raise ValueError("transcode artifact is not an MP4 container")

        def children(kind: bytes, start: int, end: int) -> bool:
            if kind == b"stsd":
                if end - start < 8:
                    raise ValueError("truncated MP4 sample description")
                entries = cls._bmff_boxes(data, start + 8, end)
                return any(entry_kind in {b"avc1", b"avc3"} for entry_kind, _, _ in entries)
            if kind not in {b"moov", b"trak", b"mdia", b"minf", b"stbl"}:
                return False
            return any(
                children(child_kind, child_start, child_end)
                for child_kind, child_start, child_end in cls._bmff_boxes(data, start, end)
            )

        if not any(children(kind, start, end) for kind, start, end in top):
            raise ValueError("transcode artifact has no H.264 video sample entry")

    def media_transcode_job_create(
        self,
        service_id: str,
        source_path: str,
        target_format: str,
        deliver_by: str,
        payment_method_id: str,
    ) -> dict[str, Any]:
        _require_date("deliver_by", deliver_by)
        service = next(
            (
                item
                for item in self._media_transcode_services()
                if item.get("service_id") == service_id and self._date_visible(item)
            ),
            None,
        )
        if service is None:
            raise ValueError(f"unknown transcode service: {service_id}")
        if target_format.casefold() not in {
            str(value).casefold() for value in service.get("target_formats", [])
        }:
            raise ValueError(
                f"service {service_id} does not deliver target format {target_format}"
            )
        promised = str(service.get("promised_delivery_at") or "")
        if not promised or promised[:10] > deliver_by:
            raise ValueError(f"service {service_id} cannot deliver by {deliver_by}")
        payment = self._require_payment_method(payment_method_id)

        source = self._resolve_inside(source_path)
        if source.is_dir():
            raise IsADirectoryError(source_path)
        source_sha256 = self._sha256(source)
        artifacts_path = self.state_root / "media_transcode" / "artifacts.json"
        artifacts = (
            json.loads(artifacts_path.read_text(encoding="utf-8")).get("artifacts", [])
            if artifacts_path.exists()
            else []
        )
        artifact = next(
            (
                item
                for item in artifacts
                if item.get("service_id") == service_id
                and item.get("source_path") == source_path
                and item.get("source_sha256") == source_sha256
                and str(item.get("target_format") or "").casefold()
                == target_format.casefold()
            ),
            None,
        )
        if artifact is None:
            raise ValueError(
                "no verified transcode artifact for this service, source, and target"
            )
        output_relative = str(artifact.get("path") or "")
        if not output_relative.startswith("media_transcode/deliveries/"):
            raise ValueError("transcode artifact path is outside the delivery store")
        output = self._resolve_inside(output_relative)
        if output.is_dir():
            raise IsADirectoryError(output_relative)
        output_sha256 = self._sha256(output)
        output_size = output.stat().st_size
        if output_sha256 != artifact.get("sha256"):
            raise ValueError("transcode artifact SHA-256 does not match its manifest")
        if output_size != artifact.get("size_bytes"):
            raise ValueError("transcode artifact size does not match its manifest")
        self._validate_mp4_h264(output)
        if artifact.get("container") != "mp4" or artifact.get("video_codec") != "h264":
            raise ValueError("transcode artifact manifest is not MP4/H.264")

        jobs_path = self.state_root / "media_transcode" / "jobs.jsonl"
        job_id = self._mint_id(jobs_path)
        accepted_at = self._now()
        charge_id = "mtc_" + job_id.removeprefix("mtj_")
        manifest = {
            "artifact_id": artifact["artifact_id"],
            "path": output_relative,
            "container": "mp4",
            "video_codec": "h264",
            "source_sha256": source_sha256,
            "job_id": job_id,
            "sha256": output_sha256,
            "size_bytes": output_size,
        }
        record = {
            "job_id": job_id,
            "service_id": service_id,
            "provider": service["provider"],
            "source_path": source_path,
            "source_format": artifact["source_format"],
            "target_format": target_format,
            "deliver_by": deliver_by,
            "accepted_at": accepted_at,
            "delivered_at": promised,
            "status": "delivered",
            "charge": {
                "charge_id": charge_id,
                "status": "captured",
                "amount": service["fee"],
                "currency": service["currency"],
                "payment_method_id": payment["method_id"],
                "charged_at": accepted_at,
            },
            "artifact": manifest,
            "run_id": self.run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        records = self._read_jsonl(jobs_path)
        records.append(record)
        temporary = jobs_path.with_suffix(".jsonl.tmp")
        self._write_jsonl(temporary, records)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(jobs_path)

        result = {
            "job_id": job_id,
            "status": "delivered",
            "charge": record["charge"],
            "artifact": manifest,
        }
        self.log_tool(
            "media_transcode_job_create",
            {
                "service_id": service_id,
                "source_path": source_path,
                "target_format": target_format,
                "deliver_by": deliver_by,
                "payment_method_id": payment_method_id,
            },
            {
                "job_id": job_id,
                "status": "delivered",
                "charge_id": charge_id,
                "artifact_id": artifact["artifact_id"],
            },
        )
        self.log_state_diff(
            "media_transcode.jobs",
            "append",
            job_id,
            f"Accepted, charged, and delivered transcode job {job_id}",
        )
        return result

    def media_transcode_job_read(self, job_id: str) -> dict[str, Any]:
        path = self.state_root / "media_transcode" / "jobs.jsonl"
        record = next(
            (
                item
                for item in self._read_jsonl(path)
                if item.get("job_id") == job_id
                and self._date_visible(item, "accepted_at")
            ),
            None,
        )
        if record is None:
            raise ValueError(f"unknown transcode job: {job_id}")
        result = {"job": self._public_record(record)}
        self.log_tool(
            "media_transcode_job_read", {"job_id": job_id}, {"job_id": job_id}
        )
        return result

    # Logging

    def _search_log_result(
        self, matches: list[dict[str, Any]], id_field: str
    ) -> dict[str, Any]:
        """The `result_summary` of a search row: how many matched, and which.

        `ids` is the matched records' own identifiers, in the order the tool
        returned them, capped at `SEARCH_LOG_ID_LIMIT` with `truncated: true`
        past it. Without them the row could not be attributed at all: the
        exposure scan reads these ids to say which record a search handed the
        PA, and a count says only that it handed one over.

        The key is absent when no matched record carries an id -- an empty
        result, or a store whose records have none (`places/nearby.json`,
        `recipe/recipes.json`, the persona layers) -- because there is then
        nothing to name.
        """
        ids = [
            str(record[id_field])
            for record in matches
            if isinstance(record, dict) and record.get(id_field) is not None
        ]
        summary: dict[str, Any] = {"matches": len(matches)}
        if ids:
            summary["ids"] = ids[:SEARCH_LOG_ID_LIMIT]
            if len(ids) > SEARCH_LOG_ID_LIMIT:
                summary["truncated"] = True
        return summary

    def log_tool(self, tool: str, args: dict[str, Any], result_summary: dict[str, Any], status: str = "ok") -> None:
        self._append_jsonl(
            self.state_root / "tool_log.jsonl",
            {
                "t": self._next_jsonl_index(self.state_root / "tool_log.jsonl"),
                "timestamp": self._now(),
                "run_id": self.run_id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "tool": tool,
                "args": args,
                "result_summary": result_summary,
                "status": status,
            },
        )

    def log_state_diff(self, namespace: str, op: str, item_id: str, summary: str) -> None:
        self._append_jsonl(
            self.state_root / "state_diff.jsonl",
            {
                "t": self._next_jsonl_index(self.state_root / "state_diff.jsonl"),
                "timestamp": self._now(),
                "run_id": self.run_id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "namespace": namespace,
                "op": op,
                "id": item_id,
                "summary": summary,
            },
        )

    # Path helpers

    def _resolve_inside(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            raise ValueError("absolute paths are not allowed")
        resolved = (self.state_root / candidate).resolve()
        if self.state_root not in resolved.parents and resolved != self.state_root:
            raise ValueError("path escapes state root")
        if not resolved.exists():
            raise FileNotFoundError(self._relative(resolved))
        self._require_actor_visible(resolved)
        return resolved

    def _resolve_writable_inside(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            raise ValueError("absolute paths are not allowed")
        resolved = (self.state_root / candidate).resolve()
        if self.state_root not in resolved.parents and resolved != self.state_root:
            raise ValueError("path escapes state root")
        self._require_actor_visible(resolved)
        return resolved

    def _is_actor_visible(self, path: Path) -> bool:
        relative = path.resolve().relative_to(self.state_root)
        if not relative.parts:
            return False
        return (
            relative.parts[0] not in _ACTOR_HIDDEN_ROOTS
            and relative.as_posix() not in _ACTOR_HIDDEN_FILES
        )

    def _require_actor_visible(self, path: Path) -> None:
        if not self._is_actor_visible(path):
            raise PermissionError("path is not user-visible task state")

    def _resolve_attachments(self, attachments: list[str] | None) -> list[str]:
        resolved_paths: list[str] = []
        for raw_path in attachments or []:
            attachment = self._resolve_inside(raw_path)
            if attachment.is_dir():
                raise IsADirectoryError(self._relative(attachment))
            resolved_paths.append(self._relative(attachment))
        return resolved_paths

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.state_root))

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _mint_id(self, path: Path) -> str:
        """Next id for a record store, shape taken from `ID_PREFIXES`.

        Keyed by the store's path relative to the state root, so the id a
        handler writes and the id the fixture validator expects come out of one
        table.
        """
        scheme = id_scheme(path.relative_to(self.state_root).as_posix())
        return scheme.mint(self._next_jsonl_index(path, scheme.number_re()))

    @staticmethod
    def _mint_order_id(orders: list[dict[str, Any]], code: str) -> str:
        """Next order number in the merchant's own scheme, `code` from
        `order_merchant`.

        The order store is one JSON document rather than a jsonl file, and the
        number continues that merchant's own run (GlenMart's seeded orders end at
        GM-124309, so the next is GM-124310), leaving another merchant's numbers
        untouched.
        """
        scheme = id_scheme("shopping/order_store.json")
        number_pattern = scheme.number_re(code)
        highest = max(
            (
                int(match.group(1))
                for order in orders
                if (match := number_pattern.match(str(order.get("order_id", ""))))
            ),
            default=0,
        )
        return scheme.mint(highest + 1, code)

    @staticmethod
    def _next_jsonl_index(path: Path, id_pattern: re.Pattern[str] | None = None) -> int:
        """Next index for a jsonl store.

        With an id pattern, returns the highest number it captures + 1 so that
        deleted rows never make a new record reuse a live id. Without one
        (tool log, state diff) it is simply row count + 1.
        """
        if not path.exists():
            return 1
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if id_pattern is None:
            return len(rows) + 1
        highest = 0
        for line in rows:
            for value in json.loads(line).values():
                if isinstance(value, str) and (match := id_pattern.match(value)):
                    highest = max(highest, int(match.group(1)))
        return highest + 1

    @staticmethod
    def _search_tokens(text: str) -> set[str]:
        normalized = "".join(ch.casefold() if ch.isalnum() else " " for ch in text)
        return {token for token in normalized.split() if len(token) >= 3}

    @staticmethod
    def _query_matches(query: str, haystack: str) -> bool:
        # Every whitespace-separated word of the query has to appear somewhere in
        # the record, in any order. Matching the whole query as one substring made
        # a multi-word phrase miss records holding all of its words in a different
        # run ("TA-15 aerator" against "TA-15 tap aerator"). Splitting on
        # whitespace only, so a part number such as "ta-15" or "m22×1" stays one
        # token. An empty query still matches everything.
        text = haystack.casefold()
        return all(token in text for token in query.casefold().split())

    def _now(self) -> str:
        # A real run supplies its frozen full clock. Preserve that wall-clock
        # string and offset exactly: fixture visibility compares authored wall
        # clocks without converting offsets, and persisted writes must use the
        # same convention. Date-only authoring callers retain the historical
        # deterministic noon fallback byte-for-byte.
        if self.session_timestamp is not None:
            try:
                parsed = _as_datetime(self.session_timestamp)
            except TypeError as exc:
                raise ValueError("session_timestamp must be a timed ISO string") from exc
            if parsed is None:
                raise ValueError("session_timestamp must be a timed ISO string")
            if self.session_date != parsed.date().isoformat():
                raise ValueError(
                    "session_timestamp date must match session_date: "
                    f"timestamp={self.session_timestamp!r} date={self.session_date!r}"
                )
            return self.session_timestamp
        if self.session_date:
            return f"{self.session_date}T12:00:00+00:00"
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
