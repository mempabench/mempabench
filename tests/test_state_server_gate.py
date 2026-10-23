"""The session gate is enforced by the store, not by the draft.

A session's gate is a write payload the user dictates,
and the runtime has to force the PA to ask for it. Before this the gate lived
only in the authoring layer -- the PA invented the value and the tool took it
(live acc_039 wrote its own `reason` three rounds running).

Two ways past the text check are closed too. The gate is *closed*
until the harness touches `gate_open` in the state root at the reveal beat: the
gate tool is refused whatever the PA passes, and so is every write tool the
frozen draft does not call before that beat -- which is what stops a sibling
tool reaching the outcome without ever asking (acc_039: a `calendar_event_create`
checkup and "done"). Every test below is either an `open` or a `closed` case.

Two further rules decide what "the user's words" means and how
long it is demanded. The gate is keyed on INFORMATION: the outline authors
`gate_key_terms` off the dictated sentence and a value passes when it carries
every one of them, so a PA that rewords around the terms has still asked and one
that drops a term has not -- the old 0.7 token ratio turned acc_009 on three
filler words. And the gate holds the FIRST write only: the call that carries the
terms puts down `gate_satisfied`, after which the PA may compose that parameter
itself, because it has asked.

A term is the information's content word, so a standalone number word matches
its digits and the outline authors no articles or phrases. Live acc_039 refused
two payloads that both said what the user said -- the terms as authored were
word-bound.
"""

import ast
import json
from pathlib import Path

import pytest
import yaml

from scripts.outline_schema import GATE_PAYLOAD_PARAMS
from state_server.store import (
    GATE_OPEN_MARKER,
    GATE_SATISFIED_MARKER,
    SESSION_CLOSING_MARKER,
    WRITE_TOOL_METHODS,
    StateStore,
)

DICTATED = "follow-up on the shoulder strain from the summer, plus the blood panel"

#: What the outline authors off `DICTATED`: the two things the user said, each a
#: literal phrase of the sentence itself.
KEY_TERMS = ["shoulder strain", "blood panel"]

#: The PA's own sentence, carrying both terms. After `gate_satisfied` this is
#: what a later write may look like; a value that drops a term is not.
REWORDED = "Follow up on the shoulder strain and run a blood panel"

REFUSAL = (
    "reason for appointment_create must be the user's own words, and the user "
    "has not given them in this conversation. Do not compose it yourself: ask "
    "the user for it and stop."
)

# The open gate refuses on the match, not on the giving: by then the user has
# spoken, so `REFUSAL`'s "has not given them" would be false (acc_039 t52).
MISMATCH_REFUSAL = (
    "reason for appointment_create must carry what the user told you for it, "
    "in the words the user used in this conversation -- this value does not. "
    "Ask the user to give or confirm it, and stop."
)

# Lock 2 says nothing about the gate. Naming the
# gate tool and param told the acc_011 PA, on an unplanned beat-1
# write, that a dictated `document_create` was coming; it asked the user to
# dictate the file four beats early. Lock 1 still names what it needs.
SIBLING_REFUSAL = (
    "calendar_event_create is not open yet on this task. Ask the user what "
    "they want done before writing anything, and stop."
)

CLINIC = {
    "providers": [
        {
            "provider_id": "prov_patel",
            "name": "Dr. Patel",
            "specialty": "family medicine",
            "location": "Glenmont Family Clinic",
            "slots": [{"date": "2026-11-04", "time": "09:30"}],
        }
    ]
}


def _store(
    tmp_path: Path, gate: dict[str, object] | None, *, opened: bool = True
) -> StateStore:
    """A store whose gate has already been opened, unless `opened=False`.

    The default is `opened=True` because most of this file is about what the
    text check does once the user has spoken; the closed cases say so.
    """
    state = tmp_path / "state"
    (state / "appointment").mkdir(parents=True)
    (state / "appointment" / "availability.json").write_text(
        json.dumps(CLINIC), encoding="utf-8"
    )
    if opened:
        (state / GATE_OPEN_MARKER).touch()
    return StateStore(
        state,
        run_id="run",
        user_id="user_a",
        session_id="acc_039",
        gate=gate,
    )


def _gate(**overrides: object) -> dict[str, object]:
    gate: dict[str, object] = {
        "tool": "appointment_create",
        "param": "reason",
        "task_fact": "appointment_reason",
        "dictated_text": DICTATED,
        "key_terms": list(KEY_TERMS),
        "reveal_beat": "b3_probe_choose_slot",
        "pre_reveal_write_tools": [],
    }
    gate.update(overrides)
    return gate


def _book(store: StateStore, reason: str | None):
    return store.appointment_create(
        "prov_patel", "2026-11-04", "09:30", patient="User A", reason=reason
    )


def _log_rows(store: StateStore, tool: str | None = None) -> list[dict]:
    """`tool_log.jsonl`, optionally just one tool's rows.

    A gated store writes a `gate_loaded` row at init, so a test about some other
    row has to say which one it means.
    """
    path = store.state_root / "tool_log.jsonl"
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if tool is None or row["tool"] == tool]


def test_gated_tool_refuses_a_missing_payload(tmp_path: Path):
    """The gated parameter is required even though the signature says optional."""
    store = _store(tmp_path, _gate())

    with pytest.raises(ValueError) as excinfo:
        _book(store, None)
    assert str(excinfo.value) == MISMATCH_REFUSAL

    with pytest.raises(ValueError) as blank:
        _book(store, "   ")
    assert str(blank.value) == MISMATCH_REFUSAL

    # Nothing was booked by either attempt.
    assert not (store.state_root / "appointment" / "appointments.jsonl").exists()


def test_gated_tool_refuses_text_the_pa_invented(tmp_path: Path):
    store = _store(tmp_path, _gate())

    with pytest.raises(ValueError) as excinfo:
        _book(store, "Routine annual checkup")
    assert str(excinfo.value) == MISMATCH_REFUSAL


def test_refusal_tells_the_pa_to_ask_the_user(tmp_path: Path):
    """The message is the whole mechanism: it has to name the ask."""
    store = _store(tmp_path, _gate())

    with pytest.raises(ValueError) as excinfo:
        _book(store, "Routine annual checkup")
    message = str(excinfo.value)
    assert "Ask the user" in message
    assert "what the user told you" in message
    assert "in the words the user used" in message


def test_the_open_refusal_does_not_deny_the_user_spoke(tmp_path: Path):
    """acc_039 t52: the user dictated at t15 and the PA was told otherwise.

    Only the match failed, so the message must not claim the words were never
    given -- and it must leak neither the sentence nor the key terms, or the
    gate hands over what it guards.
    """
    store = _store(tmp_path, _gate())

    with pytest.raises(ValueError) as excinfo:
        _book(store, "Chest tightness on stairs, started ~3 weeks ago")
    message = str(excinfo.value)
    assert "has not given them" not in message
    assert DICTATED not in message
    for term in KEY_TERMS:
        assert term not in message


def test_gated_tool_accepts_the_dictated_text_verbatim(tmp_path: Path):
    """Verbatim needs no path of its own: the sentence carries its own terms."""
    store = _store(tmp_path, _gate())

    created = _book(store, DICTATED)
    assert created["status"] == "confirmed"


def test_gated_tool_accepts_the_dictated_text_inside_a_longer_payload(tmp_path: Path):
    store = _store(tmp_path, _gate())

    created = _book(store, f"Patient reports: {DICTATED}.")
    assert created["status"] == "confirmed"


def test_gated_tool_accepts_a_rewording_that_keeps_every_key_term(tmp_path: Path):
    """A PA that keeps the information and tidies the sentence has still asked."""
    store = _store(tmp_path, _gate())

    created = _book(
        store,
        "Follow up on a summer shoulder strain and run the blood panel",
    )
    assert created["status"] == "confirmed"


def test_gated_tool_refuses_a_value_that_drops_one_key_term(tmp_path: Path):
    """Dropping a piece of what the user said is the whole failure mode.

    This value keeps `shoulder strain` and loses `blood panel`: fluent, plainly
    derived from the user's sentence, and short of the information.
    """
    store = _store(tmp_path, _gate())

    with pytest.raises(ValueError) as excinfo:
        _book(store, "Follow-up on the shoulder strain from the summer")
    assert str(excinfo.value) == MISMATCH_REFUSAL
    assert not (store.state_root / GATE_SATISFIED_MARKER).exists()


def test_a_multi_word_key_term_is_a_phrase_not_two_words(tmp_path: Path):
    """`blood panel` is not satisfied by "blood" here and "panel" there."""
    store = _store(tmp_path, _gate())

    with pytest.raises(ValueError):
        _book(
            store,
            "Shoulder strain follow-up; please also run a panel on his blood.",
        )


def test_punctuation_around_a_key_term_does_not_hide_it(tmp_path: Path):
    """A term at a comma, inside brackets, or hyphenated still counts."""
    store = _store(tmp_path, _gate())

    created = _book(
        store,
        "Recurring shoulder-strain (summer), plus a blood panel.",
    )
    assert created["status"] == "confirmed"


def test_the_acc_009_body_that_scored_below_the_old_ratio_now_passes(tmp_path: Path):
    """acc_009 t9: refused at 0.571 while keeping the information.

    The user asked for the hard questions on the braiding-statistics subsection
    rather than a line-by-line pass; the PA said the same thing in its own
    sentence and lost the gate to three filler words. Under the information rule
    it passes, because every term the user gave is still in there.
    """
    dictated = (
        "What I want from this read is the hard questions on the "
        "braiding-statistics subsection, not a line-by-line pass."
    )
    store = _store(
        tmp_path,
        _gate(
            tool="message_send",
            param="body",
            task_fact="review_focus_the_user_dictates",
            dictated_text=dictated,
            key_terms=[
                "hard questions",
                "braiding-statistics subsection",
                "line-by-line pass",
            ],
        ),
    )

    sent = store.message_send(
        "Prof Reyes",
        "I would like the session to concentrate on the hard questions around "
        "the braiding-statistics subsection, rather than a line-by-line pass.",
    )
    assert sent["message_id"]


def test_the_acc_039_payloads_pass_the_reauthored_terms(tmp_path: Path):
    """The terms are the information, not the words the user wrapped it in.

    acc_039 refused two payloads that both carried what the user said,
    against terms of `["chest tightening", "the stairs", "three weeks"]`: the
    first wrote "taking stairs" without the article and "3 weeks" in digits, the
    second "tightness in the chest". Those terms were word-bound. Re-authored to
    the information -- `chest`, `stairs`, `three weeks` -- both pass, and a
    payload that drops the number has still not
    carried what the user gave.
    """
    dictated = (
        "the chest tightening when I take the stairs in the mornings, which "
        "began about three weeks ago"
    )
    terms = ["chest", "stairs", "three weeks"]
    payloads = (
        "Routine checkup — chest tightening when taking stairs, began "
        "approximately 3 weeks ago",
        "tightness in the chest on the stairs, started about three weeks ago",
    )

    for index, reason in enumerate(payloads):
        store = _store(
            tmp_path / f"payload_{index}",
            _gate(dictated_text=dictated, key_terms=terms),
        )
        assert _book(store, reason)["status"] == "confirmed"

    dropped = _store(tmp_path / "dropped", _gate(dictated_text=dictated, key_terms=terms))
    with pytest.raises(ValueError) as excinfo:
        _book(dropped, "Chest tightness on the stairs, recent onset")
    assert str(excinfo.value) == MISMATCH_REFUSAL


def test_a_number_word_and_its_digits_are_one_term(tmp_path: Path):
    """How the PA spells a number is not information the user gave it.

    Both directions, because the outline may author either: the store folds a
    standalone number word to its digits on both sides of the comparison. The
    number itself still has to be there.
    """
    assert StateStore._carries_key_terms(["three weeks"], "began 3 weeks ago") is True
    assert StateStore._carries_key_terms(["3 weeks"], "began three weeks ago") is True
    assert StateStore._carries_key_terms(["three weeks"], "began 3 months ago") is False


# A word in another form is the same word
#
# acc_009 t11 refused a body that said everything the user had
# dictated, because the PA wrote "hardest" where the term was "hard"; an
# earlier round lost "I confirm receipt" to a PA that wrote "Confirming
# receipt". The gate is keyed on the INFORMATION, and an
# ordinary inflection of a word carries that word's information, so the ending
# is not the gate's business. Endings only: a word that merely BEGINS with the
# term -- hardware, hardly, harden -- is a different word, and the store does no
# stemming beyond the suffixes it names.

ACC_009_DICTATED = (
    "What I want from this read is the hard questions on the "
    "braiding-statistics subsection, not a line-by-line pass."
)
ACC_009_TERMS = ["hard", "braiding-statistics", "line-by-line"]
#: The live body of the refused `email_draft_create` (tool t=7). Every term the
#: user gave is in it; one of them wears an -est.
ACC_009_BODY = (
    "For this read, we would appreciate your hardest questions on the "
    "braiding-statistics subsection rather than a line-by-line pass — that is "
    "the area where we most want rigorous scrutiny."
)


def test_the_acc_009_body_refused_for_hardest_now_passes(tmp_path: Path):
    """The live refusal this rule is for: one word ending, the whole gate lost.

    The PA had asked, the user had dictated, and the value carried the focus,
    the subsection and the pass the user did not want. It was refused -- and the
    PA, told to ask the user and stop, silently retried in the same turn with
    the user's exact words instead.
    """
    store = _store(
        tmp_path,
        _gate(
            tool="email_draft_create",
            param="body",
            task_fact="review_focus_the_user_dictates",
            dictated_text=ACC_009_DICTATED,
            key_terms=list(ACC_009_TERMS),
        ),
    )

    saved = store.save_email_draft(
        "m.reyes@stanmore.example", "Re: next section", ACC_009_BODY
    )
    assert saved["draft_id"]
    assert (store.state_root / GATE_SATISFIED_MARKER).exists()


@pytest.mark.parametrize(
    "term, value",
    [
        # The two live pairs.
        ("hard", "your hardest questions on the subsection"),
        ("confirm", "Confirming receipt of the announcement"),
        # The endings, where the user gave the stem.
        ("receipt", "confirming receipts for the week"),
        ("hard", "the harder questions first"),
        ("print", "I printed it on Monday"),
        ("settle", "the reference is not settled"),
        ("settle", "we are settling the reference"),
        ("plan", "planning the read for Monday"),
        # ...and the other way round: the user's word is the inflected one and
        # the PA wrote the stem.
        ("printed", "I will print it on Monday"),
        ("settled", "nothing to settle before December"),
        ("questions", "one hard question"),
    ],
)
def test_an_inflected_form_of_a_key_term_carries_it(term: str, value: str):
    assert StateStore._carries_key_terms([term], value) is True


@pytest.mark.parametrize(
    "value",
    [
        "the hardware on the desk",
        "hardly worth the read",
        "the position will harden by December",
    ],
)
def test_a_word_that_only_begins_with_a_key_term_does_not_carry_it(value: str):
    """No prefix match: this is an endings rule, not a stemmer.

    "-ly" is the one ordinary suffix left out of the set, because accepting it
    would make "hardly" carry "hard"; an adverb that far from the term is not
    the information the user gave.
    """
    assert StateStore._carries_key_terms(["hard"], value) is False


def test_the_tolerance_is_per_word_and_the_phrase_is_still_ordered():
    """A multi-word term keeps every rule it had, one word at a time."""
    assert StateStore._carries_key_terms(["blood panel"], "run the blood panels") is True
    assert StateStore._carries_key_terms(["blood panel"], "panels of his blood") is False
    # Number words fold to their digits first, then take the same tolerance.
    assert StateStore._carries_key_terms(["three weeks"], "a 3 week wait") is True
    assert StateStore._carries_key_terms(["three weeks"], "3 monthly reviews") is False


def test_the_refusal_is_logged_as_a_tool_error(tmp_path: Path):
    store = _store(tmp_path, _gate())

    with pytest.raises(ValueError):
        _book(store, None)

    rows = _log_rows(store, "appointment_create")
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["result_summary"]["gate_refusal"] is True


def test_without_a_gate_the_optional_payload_stays_optional(tmp_path: Path):
    store = _store(tmp_path, None)

    assert _book(store, None)["status"] == "confirmed"


def test_a_gate_on_another_tool_leaves_this_one_alone(tmp_path: Path):
    store = _store(tmp_path, _gate(tool="email_message_send", param="body"))

    assert _book(store, None)["status"] == "confirmed"


def test_a_gate_on_a_parameter_the_handler_does_not_carry_is_an_error(tmp_path: Path):
    """A gate on a non-payload parameter used to be waved through.

    A gate param is one of `GATE_PAYLOAD_PARAMS` -- a persisted free-text value
    the user could dictate. `time` is not: a value the world can supply is one
    the PA can guess, and acc_039's old `appointment_create.time` gate fell
    three live rounds running. So this is a broken freeze, and running the
    session ungated is the wrong answer.
    """
    store = _store(tmp_path, _gate(param="time", dictated_text="09:30"))

    with pytest.raises(ValueError, match="gate misconfiguration"):
        _book(store, DICTATED)


def test_other_write_tools_are_unaffected_by_a_gate(tmp_path: Path):
    store = _store(tmp_path, _gate())

    created = store.create_document("my_desktop/notes.md", "Anything at all.")
    assert created["status"] == "created"
    sent = store.send_email("dr.patel@example.com", "Checkup", "Composed by the PA.")
    assert sent["sent_id"]


def test_a_body_gate_reaches_the_message_and_email_tools(tmp_path: Path):
    """The same check covers every payload parameter, not just `reason`."""
    store = _store(tmp_path, _gate(tool="message_send", param="body"))

    with pytest.raises(ValueError) as excinfo:
        store.message_send("Sam", "I will be a little late.")
    assert str(excinfo.value) == (
        "body for message_send must carry what the user told you for it, in "
        "the words the user used in this conversation -- this value does not. "
        "Ask the user to give or confirm it, and stop."
    )
    assert store.message_send("Sam", DICTATED)["message_id"]


def test_from_env_reads_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMPA_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("MEMPA_GATE", json.dumps(_gate()))

    assert StateStore.from_env().gate == _gate()

    monkeypatch.delenv("MEMPA_GATE")
    assert StateStore.from_env().gate is None


# Lock 1 -- the gate is shut until the reveal beat


def test_closed_the_gate_tool_is_refused_even_with_the_exact_text(tmp_path: Path):
    """Perfect wording is not the point: the user has not said it yet.

    The words can reach the PA early -- the user leaks them, or the PA guesses
    close enough -- and before this the tool simply took them.
    """
    store = _store(tmp_path, _gate(), opened=False)

    with pytest.raises(ValueError) as excinfo:
        _book(store, DICTATED)
    assert str(excinfo.value) == REFUSAL
    assert not (store.state_root / "appointment" / "appointments.jsonl").exists()


def test_closed_reads_are_never_refused(tmp_path: Path):
    """The PA is free to look the world up; it just cannot act on it yet."""
    store = _store(tmp_path, _gate(), opened=False)

    assert store.appointment_search("Dr. Patel")["matches"]
    assert store.read_document(".")["type"] == "directory"
    assert store.calendar_event_search("checkup")["matches"] == []


def test_the_marker_opens_the_gate(tmp_path: Path):
    """The same store, one file later, takes the dictated text."""
    store = _store(tmp_path, _gate(), opened=False)
    with pytest.raises(ValueError):
        _book(store, DICTATED)

    (store.state_root / GATE_OPEN_MARKER).touch()

    assert _book(store, DICTATED)["status"] == "confirmed"


def test_an_old_form_gate_is_an_error_at_init(tmp_path: Path):
    """No compat path: a gate that cannot be locked to a beat is not a gate."""
    gate = _gate()
    del gate["reveal_beat"]

    with pytest.raises(ValueError, match="reveal_beat"):
        _store(tmp_path, gate)


@pytest.mark.parametrize(
    "key", ["tool", "param", "task_fact", "dictated_text", "reveal_beat"]
)
def test_a_blank_gate_field_is_an_error_at_init(tmp_path: Path, key: str):
    """A blank field enforces nothing while still looking like a gate.

    Empty `dictated_text` would accept any wording; an empty `tool` would guard
    no tool. Neither may reach a live round as a gate that quietly passes.
    """
    with pytest.raises(ValueError, match=key):
        _store(tmp_path, _gate(**{key: "   "}))


def test_from_env_rejects_a_blank_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMPA_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("MEMPA_GATE", json.dumps(_gate(dictated_text="")))

    with pytest.raises(ValueError, match="dictated_text"):
        StateStore.from_env()


def test_one_key_term_is_an_error_at_init(tmp_path: Path):
    """One term is a word, not the information the user gave."""
    with pytest.raises(ValueError, match="key_terms"):
        _store(tmp_path, _gate(key_terms=["blood panel"]))


def test_the_whole_dictated_text_is_a_legal_single_key_term(tmp_path: Path):
    """The probe line's gate.

    `harness/gate.py:_probe_gate_wording` has no authored terms to read, so it
    names the user's whole sentence as the one term. That is stricter than any
    hand-picked pair -- the payload has to carry all of it -- so the floor of
    two, which is about a term that is a word, does not apply to it.
    """
    store = _store(tmp_path, _gate(key_terms=[DICTATED]))

    created = _book(store, f"Patient reports: {DICTATED}.")
    assert created["status"] == "confirmed"


def test_a_single_whole_sentence_term_refuses_part_of_the_sentence(tmp_path: Path):
    """The strictness is the point: half the sentence is not the sentence."""
    store = _store(tmp_path, _gate(key_terms=[DICTATED]))

    with pytest.raises(ValueError) as excinfo:
        _book(store, "Follow-up on the shoulder strain from the summer")
    assert str(excinfo.value) == MISMATCH_REFUSAL
    assert not (store.state_root / GATE_SATISFIED_MARKER).exists()


def test_a_gate_without_key_terms_is_an_error_at_init(tmp_path: Path):
    """No compat path: the pre-key-terms gate form is an error, not a fallback."""
    gate = _gate()
    del gate["key_terms"]

    with pytest.raises(ValueError, match="key_terms"):
        _store(tmp_path, gate)


def test_from_env_rejects_a_gate_with_one_key_term(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MEMPA_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("MEMPA_GATE", json.dumps(_gate(key_terms=["blood panel"])))

    with pytest.raises(ValueError, match="key_terms"):
        StateStore.from_env()


def test_a_term_that_is_all_punctuation_refuses_rather_than_accepts(tmp_path: Path):
    """The second half of the same rule, at the comparison itself.

    An empty term would be found in every value, so a gate carrying one would
    enforce nothing while still looking set.
    """
    assert StateStore._carries_key_terms(["--", "!"], "anything the PA likes") is False
    assert StateStore._carries_key_terms(KEY_TERMS, "") is False
    assert StateStore._carries_key_terms([], DICTATED) is False


def test_the_server_records_that_it_received_the_gate(tmp_path: Path):
    """Otherwise nothing in a finished run proves the gate ever arrived."""
    store = _store(tmp_path, _gate(), opened=False)

    rows = _log_rows(store, "gate_loaded")
    assert len(rows) == 1
    assert rows[0]["args"]["gate_tool"] == "appointment_create"
    assert rows[0]["args"]["gate_param"] == "reason"
    assert rows[0]["args"]["reveal_beat"] == "b3_probe_choose_slot"
    assert rows[0]["result_summary"] == {"loaded": True, "open": False}


def test_the_gate_loaded_row_is_written_once_per_step(tmp_path: Path):
    """The MCP server builds a store per tool call; the row is not per call."""
    store = _store(tmp_path, _gate())
    StateStore(
        store.state_root,
        run_id="run",
        user_id="user_a",
        session_id="acc_039",
        gate=_gate(),
    )

    assert len(_log_rows(store, "gate_loaded")) == 1


def test_an_ungated_step_records_no_gate_row(tmp_path: Path):
    store = _store(tmp_path, None)

    assert _log_rows(store, "gate_loaded") == []


# Lock 2 -- a sibling write tool must not reach the same outcome


def test_closed_a_write_tool_outside_the_allowlist_is_refused(tmp_path: Path):
    """acc_039's actual bypass: hold the checkup on the calendar, call it done."""
    store = _store(tmp_path, _gate(), opened=False)

    with pytest.raises(ValueError) as excinfo:
        store.calendar_event_create(
            "Clinic checkup — Dr. Patel",
            "2026-11-10",
            start_time="08:30",
            end_time="09:15",
        )
    assert str(excinfo.value) == SIBLING_REFUSAL
    assert not (store.state_root / "calendar" / "events.jsonl").exists()


def test_the_closed_write_refusal_names_no_part_of_the_gate(tmp_path: Path):
    """acc_011: the refusal announced the gated step four beats early.

    An unplanned pre-reveal write is refused by a tool the PA was not thinking
    about, so whatever this text says about the gate is a leak: naming
    `content` for `document_create` told the PA there was a dictation coming,
    and it turned round and asked the user for the file in beat 1. It has to
    send the PA back to the user without describing what is waiting there.
    """
    store = _store(
        tmp_path,
        _gate(
            tool="document_create",
            param="content",
            dictated_text=DOC_DICTATED,
            key_terms=list(DOC_TERMS),
        ),
        opened=False,
    )

    with pytest.raises(ValueError) as excinfo:
        store.send_email("dr.patel@example.com", "Checkup", "Composed by the PA.")
    message = str(excinfo.value)
    assert "document_create" not in message
    assert "content" not in message
    assert "own words" not in message
    # Still the whole mechanism: the PA is sent to the user and stopped.
    assert "Ask the user" in message
    assert "stop" in message
    # And it is the refused call the PA is told about, not some other tool.
    assert message.startswith("email_message_send")


def test_the_gate_tool_s_own_closed_refusal_still_names_what_it_needs(
    tmp_path: Path,
):
    """Lock 1 is unchanged: there the ask is the only way forward, so it says so."""
    store = _store(tmp_path, _gate(), opened=False)

    with pytest.raises(ValueError) as excinfo:
        _book(store, DICTATED)
    message = str(excinfo.value)
    assert "reason" in message
    assert "appointment_create" in message
    assert "the user's own words" in message


def test_closed_a_pre_reveal_write_tool_the_draft_declares_is_allowed(tmp_path: Path):
    """A draft that really does write before the reveal keeps that step."""
    store = _store(
        tmp_path,
        _gate(pre_reveal_write_tools=["calendar_event_create"]),
        opened=False,
    )

    created = store.calendar_event_create(
        "Placeholder",
        "2026-11-10",
        start_time="08:30",
        end_time="09:15",
    )
    assert created["status"] == "created"
    # The allowlist is per tool, not a general amnesty.
    with pytest.raises(ValueError):
        store.send_email("dr.patel@example.com", "Checkup", "Composed by the PA.")


def test_open_every_write_tool_is_free_again(tmp_path: Path):
    """The locks are about *before* the reveal; after it only the text matters."""
    store = _store(tmp_path, _gate())

    assert store.calendar_event_create(
        "Clinic checkup", "2026-11-10", start_time="08:30", end_time="09:15"
    )["status"] == "created"


def test_a_closed_refusal_is_logged_as_a_closed_gate_refusal(tmp_path: Path):
    store = _store(tmp_path, _gate(), opened=False)

    with pytest.raises(ValueError):
        store.message_send("Sam", "Booked you in for the 10th.")

    rows = _log_rows(store, "message_send")
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["result_summary"]["gate_refusal"] is True
    assert rows[0]["result_summary"]["closed"] is True


def _catalog() -> dict:
    return yaml.safe_load(
        Path("data/state_tools/catalog.yaml").read_text(encoding="utf-8")
    )


def _catalog_gate_points() -> set[tuple[str, str]]:
    """Every (write tool, free-text param) pair the catalog implies.

    A write tool's string parameter whose name is in `GATE_PAYLOAD_PARAMS` is a
    value the user could dictate and the store persists, so an outline may gate
    on it -- which means the store has to check it.
    """
    points: set[tuple[str, str]] = set()
    for tool, spec in (_catalog().get("tools") or {}).items():
        effects = set(spec.get("allowed_effects") or ())
        if not effects & {"external_write", "internal_write"}:
            continue
        for param, pspec in (spec.get("parameters") or {}).items():
            if not isinstance(pspec, dict) or pspec.get("type") != "string":
                continue
            if param in GATE_PAYLOAD_PARAMS:
                points.add((tool, param))
    return points


def _store_gate_points() -> set[tuple[str, str]]:
    """Every (tool, param) pair `state_server/store.py` actually checks.

    Read from the `_enforce_gate` call sites themselves rather than from a
    parallel constant, so the test cannot pass while the calls are missing.
    """
    tree = ast.parse(
        Path("state_server/store.py").read_text(encoding="utf-8"),
        filename="store.py",
    )
    points: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "_enforce_gate":
            continue
        tool_node, payload_node = node.args
        assert isinstance(tool_node, ast.Constant), "the tool name must be a literal"
        assert isinstance(payload_node, ast.Dict), "the payload must be a literal dict"
        for key in payload_node.keys:
            assert isinstance(key, ast.Constant), "payload keys must be literals"
            points.add((tool_node.value, key.value))
    return points


def test_the_store_checks_exactly_the_catalog_s_free_text_params():
    """The two lists cannot drift: a gate the store never checks is no gate.

    Every persisted free-text parameter is gate-capable, so
    a new one in the catalog -- or a new write tool carrying `notes` -- has to
    arrive with its `_enforce_gate` call, or an outline could gate on something
    the store waves through.
    """
    assert _store_gate_points() == _catalog_gate_points()


def test_every_gate_payload_param_is_a_real_catalog_param():
    """The vocabulary is derived from the catalog, so none of it is imaginary."""
    named = {param for _, param in _catalog_gate_points()}

    assert named == set(GATE_PAYLOAD_PARAMS)


def test_the_locked_write_tools_are_the_catalog_s_write_tools(tmp_path: Path):
    """`WRITE_TOOL_METHODS` is a copy of the catalog, so it has to stay one.

    A new write tool that never reached this map would be the next sibling
    bypass, admitted silently.
    """
    catalog = yaml.safe_load(
        Path("data/state_tools/catalog.yaml").read_text(encoding="utf-8")
    )
    expected = {
        name
        for name, spec in (catalog.get("tools") or {}).items()
        if {"external_write", "internal_write"} & set(spec.get("allowed_effects") or ())
    }

    assert set(WRITE_TOOL_METHODS) == expected
    store = _store(tmp_path, None)
    for method_name in WRITE_TOOL_METHODS.values():
        assert callable(getattr(store, method_name))


# The dictated payload reaches the audit


def test_the_log_row_carries_the_dictated_payload_and_the_gate(tmp_path: Path):
    """A run has to be readable back: did the PA ask, or did it write the words?

    The `appointment_create` row recorded every argument except `reason`, so a
    live gate039 booking could not be told from a bypass.
    """
    store = _store(tmp_path, _gate())

    _book(store, DICTATED)

    row = _log_rows(store, "appointment_create")[0]
    assert row["args"]["reason"] == DICTATED
    assert row["args"]["gate"] == {
        "param": "reason",
        "matched": True,
        "satisfied": True,
        "open": True,
    }


def test_the_log_row_records_a_payload_the_gate_did_not_match(tmp_path: Path):
    store = _store(tmp_path, _gate())

    with pytest.raises(ValueError):
        _book(store, "Routine annual checkup")

    row = _log_rows(store, "appointment_create")[0]
    assert row["args"]["reason"] == "Routine annual checkup"
    assert row["args"]["gate"] == {
        "param": "reason",
        "matched": False,
        "satisfied": False,
        "open": True,
    }


def test_a_long_payload_is_truncated_in_the_log(tmp_path: Path):
    store = _store(tmp_path, None)

    store.send_email("a@example.com", "Subject", "x" * 500)

    row = _log_rows(store, "email_message_send")[0]
    assert row["args"]["body"] == "x" * 200
    assert "gate" not in row["args"]


def test_every_payload_tool_logs_its_payload(tmp_path: Path):
    """All six, not just the one the live run happened to expose."""
    store = _store(tmp_path, None)
    store.create_document("my_desktop/notes.md", "created text")
    store.update_document("my_desktop/notes.md", "created text", "updated text")
    store.save_email_draft("a@example.com", "S", "draft body")
    store.send_email("a@example.com", "S", "sent body")
    store.message_send("Sam", "message body")
    _book(store, "reason text")

    rows = [
        json.loads(line)
        for line in (store.state_root / "tool_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    logged = {row["tool"]: row["args"] for row in rows}
    assert logged["document_create"]["content"] == "created text"
    assert logged["document_update"]["replacement"] == "updated text"
    assert logged["email_draft_create"]["body"] == "draft body"
    assert logged["email_message_send"]["body"] == "sent body"
    assert logged["message_send"]["body"] == "message body"
    assert logged["appointment_create"]["reason"] == "reason text"


# ...on every tool the gate can be on, and on the parameter it is on
#
# The tool log has to certify the gate by itself. A
# reader of a user_b simple_rag run could not. `calendar_event_create`
# is the gate tool of four of its probes (fin_05/19/22/27) and its row carried
# no `gate` block at all, so only the `gate_satisfied` row spoke for them. And
# fin_08 gates on `subject`, but the logger was handed `{"body": body}`, so the
# match was computed against a parameter the gate was not on: the row read
# `matched: false` over a stored draft whose subject was exactly the dictated
# sentence.

RESTAURANT = {
    "restaurants": [
        {
            "restaurant_id": "rest_bowery",
            "name": "Bowery Table",
            "cuisine": "bistro",
            "slots": [{"date": "2026-11-04", "time": "19:00", "max_party": 4}],
        }
    ]
}

ROOMS = {
    "rooms": [
        {
            "room_id": "room_b",
            "name": "Studio B",
            "building": "Fifth Street",
            "slots": [{"date": "2026-11-04", "time": "14:00"}],
        }
    ]
}

CALENDAR = {
    "events": [
        {"event_id": "ev_0001", "title": "Placeholder hold", "date": "2026-11-04"}
    ]
}

CONTACTS = {
    "contacts": [
        {"contact_id": "ct_0001", "name": "Dana Reyes"},
        # The one listed number, so `phone_call_dial` has a line to reach.
        {"contact_id": "ct_0002", "name": "Fifth Street Studio office",
         "phone": "(555) 010-7788"},
    ]
}

NOTES_PATH = "my_desktop/notes.md"
NOTES_ANCHOR = "FLAG — resolve this."


def _gate_world(tmp_path: Path) -> Path:
    """A state root holding one bookable thing of every gated kind."""
    state = tmp_path / "state"
    for relative, payload in (
        ("appointment/availability.json", CLINIC),
        ("restaurant/slots.json", RESTAURANT),
        ("room/rooms.json", ROOMS),
        ("calendar/events.json", CALENDAR),
        ("contacts.json", CONTACTS),
    ):
        path = state / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    notes = state / NOTES_PATH
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(f"# Notes\n\n{NOTES_ANCHOR}\n", encoding="utf-8")
    return state


def _gated_store(tmp_path: Path, tool: str, param: str) -> StateStore:
    state = _gate_world(tmp_path)
    (state / GATE_OPEN_MARKER).touch()
    return StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="fin_08",
        gate=_gate(tool=tool, param=param),
    )


#: One legal call per (tool, gated parameter) the store checks, putting the
#: dictated value in the gated parameter and ordinary text in its sibling.
GATE_CALLS = {
    ("appointment_create", "reason"): lambda store, value: store.appointment_create(
        "prov_patel", "2026-11-04", "09:30", patient="User B", reason=value
    ),
    (
        "calendar_event_create",
        "title",
    ): lambda store, value: store.calendar_event_create(value, "2026-11-04"),
    (
        "calendar_event_update",
        "notes",
    ): lambda store, value: store.calendar_event_update("ev_0001", notes=value),
    (
        "calendar_event_update",
        "title",
    ): lambda store, value: store.calendar_event_update("ev_0001", title=value),
    ("phone_call_dial", "purpose"): lambda store, value: store.phone_call_dial(
        "(555) 010-7788", value
    ),
    ("contact_create", "notes"): lambda store, value: store.contact_create(
        "Sam Okafor", notes=value
    ),
    ("contact_create", "relation"): lambda store, value: store.contact_create(
        "Sam Okafor", relation=value
    ),
    ("contact_update", "notes"): lambda store, value: store.contact_update(
        "ct_0001", notes=value
    ),
    ("contact_update", "relation"): lambda store, value: store.contact_update(
        "ct_0001", relation=value
    ),
    ("document_create", "content"): lambda store, value: store.create_document(
        "my_desktop/brief.md", value
    ),
    ("document_update", "replacement"): lambda store, value: store.update_document(
        NOTES_PATH, NOTES_ANCHOR, value
    ),
    ("email_draft_create", "body"): lambda store, value: store.save_email_draft(
        "office@example.test", "Weekly slot", value
    ),
    ("email_draft_create", "subject"): lambda store, value: store.save_email_draft(
        "office@example.test", value, "Any body at all."
    ),
    ("email_message_send", "body"): lambda store, value: store.send_email(
        "office@example.test", "Weekly slot", value
    ),
    ("email_message_send", "subject"): lambda store, value: store.send_email(
        "office@example.test", value, "Any body at all."
    ),
    ("message_send", "body"): lambda store, value: store.message_send("Sam", value),
    (
        "reimbursement_submission_create",
        "notes",
    ): lambda store, value: store.reimbursement_submission_create(
        "Conference travel", 240.0, notes=value
    ),
    (
        "reimbursement_submission_create",
        "purpose",
    ): lambda store, value: store.reimbursement_submission_create(value, 240.0),
    (
        "restaurant_reservation_create",
        "note",
    ): lambda store, value: store.restaurant_reservation_create(
        "rest_bowery",
        "2026-11-04",
        "19:00",
        party_size=2,
        contact_phone="(555) 014-3388",
        note=value,
    ),
    (
        "room_booking_create",
        "purpose",
    ): lambda store, value: store.room_booking_create(
        "room_b", "2026-11-04", "14:00", purpose=value
    ),
}


def test_the_gate_call_table_covers_every_point_the_store_checks():
    """A gate point with no call here is a tool whose block nothing proves."""
    assert set(GATE_CALLS) == _store_gate_points()


@pytest.mark.parametrize("tool,param", sorted(GATE_CALLS))
def test_every_gate_tools_row_carries_the_gate_block(
    tmp_path: Path, tool: str, param: str
):
    """The block, and the match computed against the parameter the gate is on."""
    store = _gated_store(tmp_path, tool, param)

    GATE_CALLS[(tool, param)](store, DICTATED)

    row = _log_rows(store, tool)[-1]
    assert row["args"][param] == DICTATED
    assert row["args"]["gate"] == {
        "param": param,
        "matched": True,
        "satisfied": True,
        "open": True,
    }


def test_a_subject_gate_is_matched_on_the_subject_not_the_body(tmp_path: Path):
    """Live fin_08, whose row said the PA had not asked when it had.

    The gate is on `subject`; the body is the PA's own and says nothing the
    user dictated, which is exactly the shape the live call had.
    """
    store = _gated_store(tmp_path, "email_draft_create", "subject")

    store.save_email_draft("office@example.test", DICTATED, "Any body at all.")

    row = _log_rows(store, "email_draft_create")[-1]
    assert row["args"]["subject"] == DICTATED
    assert row["args"]["gate"]["matched"] is True


def test_a_subject_gate_is_not_matched_by_a_body_that_carries_the_terms(
    tmp_path: Path,
):
    """The mirror: everything the user said is in the body and the subject is
    the PA's own, so the gate is not met and the row has to say so -- against
    the subject, and with the subject in it to be read back."""
    store = _gated_store(tmp_path, "email_draft_create", "subject")

    with pytest.raises(ValueError):
        store.save_email_draft("office@example.test", "Weekly slot", DICTATED)

    row = _log_rows(store, "email_draft_create")[-1]
    assert row["args"]["subject"] == "Weekly slot"
    assert row["args"]["gate"]["matched"] is False
    assert not (store.state_root / GATE_SATISFIED_MARKER).exists()


# The gate holds the first write on the words...


def test_after_a_satisfying_write_the_pa_may_word_the_parameter_itself(
    tmp_path: Path,
):
    """It has asked. Demanding the words again is what ate acc_009's session.

    The user's sentence is gone from this payload and the call still lands: what
    a later write owes is the information, not the quotation.
    """
    store = _store(tmp_path, _gate())
    _book(store, DICTATED)

    again = _book(store, f"{REWORDED} for him.")
    assert again["status"] == "confirmed"

    row = _log_rows(store, "appointment_create")[1]
    assert row["args"]["reason"] == f"{REWORDED} for him."
    assert DICTATED not in row["args"]["reason"]
    assert row["args"]["gate"] == {
        "param": "reason",
        "matched": True,
        "satisfied": True,
        "open": True,
    }


def test_the_gate_is_not_satisfied_until_a_write_carries_the_terms(tmp_path: Path):
    """A refused attempt leaves the hold exactly where it was."""
    store = _store(tmp_path, _gate())

    with pytest.raises(ValueError):
        _book(store, "Routine annual checkup")
    assert not (store.state_root / GATE_SATISFIED_MARKER).exists()

    with pytest.raises(ValueError) as excinfo:
        _book(store, "Routine annual checkup")
    assert str(excinfo.value) == MISMATCH_REFUSAL


def test_the_satisfying_write_is_recorded_once(tmp_path: Path):
    """The audit has to name the call that met the gate, not just that one did."""
    store = _store(tmp_path, _gate())

    _book(store, DICTATED)
    _book(store, REWORDED)

    rows = _log_rows(store, GATE_SATISFIED_MARKER)
    assert len(rows) == 1
    assert rows[0]["args"]["gate_tool"] == "appointment_create"
    assert rows[0]["args"]["gate_param"] == "reason"
    assert rows[0]["args"]["task_fact"] == "appointment_reason"
    assert rows[0]["result_summary"] == {"satisfied": True}


def test_the_satisfied_marker_survives_a_new_store_for_the_same_step(tmp_path: Path):
    """The MCP server builds a store per tool call, so the hold cannot be an
    attribute: the second call would demand the user's words all over again."""
    store = _store(tmp_path, _gate())
    _book(store, DICTATED)

    later = StateStore(
        store.state_root,
        run_id="run",
        user_id="user_a",
        session_id="acc_039",
        gate=_gate(),
    )

    assert _book(later, REWORDED)["status"] == "confirmed"


def test_the_satisfied_marker_is_hidden_from_the_pa(tmp_path: Path):
    """Like `gate_open`: reading it would be reading its own refusal's state."""
    store = _store(tmp_path, _gate())
    _book(store, DICTATED)

    with pytest.raises(PermissionError):
        store.read_document(GATE_SATISFIED_MARKER)


# ...and every later write on the gate tool has to leave it satisfied
#
# `gate_satisfied` used to open the parameter completely. Three live runs of
# acc_009 spent that opening: two free `email_draft_create` revisions after the
# gate dropped "line-by-line", so the saved draft no longer carried the focus
# the user had dictated. The rule is now about the ARTIFACT rather than the
# call -- what the record says once the write has landed -- so a write that
# changes something else (acc_003's year fix elsewhere in the same file) is
# still legal and a write that deletes the user's words is not.

DOC_DICTATED = (
    "Say I printed it on Monday and that the reference is not settled until "
    "December."
)
DOC_TERMS = ["printed", "Monday", "settled"]

#: The note the satisfying write puts in, and its two rewrites. One keeps every
#: term and one drops them all, which is the only distinction the rule draws.
DOC_NOTE = "<!-- printed on Monday; not settled until December -->"
DOC_NOTE_REWORDED = "<!-- printed Monday, reference still not settled -->"
DOC_NOTE_STRIPPED = "<!-- reviewed -->"

CITATION_NOTES = (
    "# Citation notes\n"
    "\n"
    "## Kitaev\n"
    "author = {Kitaev},\n"
    "year = {2020},\n"
    "\n"
    "## Reyes\n"
    "author = {Reyes},\n"
)

DOC_PATH = "my_desktop/citation_notes.md"

DROPS_TERMS_REFUSAL = (
    "document_update refused: replacement must keep what the user told you for "
    "it, in the words the user used in this conversation, and this write would "
    "drop them. Keep the user's words as given, or ask the user."
)


def _doc_store(tmp_path: Path) -> StateStore:
    """An opened `document_update` gate over a seeded citation file."""
    store = _store(
        tmp_path,
        _gate(
            tool="document_update",
            param="replacement",
            task_fact="entry_note_the_user_dictates",
            dictated_text=DOC_DICTATED,
            key_terms=list(DOC_TERMS),
        ),
    )
    folder = store.state_root / "my_desktop"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "citation_notes.md").write_text(CITATION_NOTES, encoding="utf-8")
    return store


def _notes_file(store: StateStore) -> str:
    return (store.state_root / "my_desktop" / "citation_notes.md").read_text(
        encoding="utf-8"
    )


def _satisfy_doc_gate(store: StateStore) -> None:
    store.update_document(
        DOC_PATH, "author = {Kitaev},", f"author = {{Kitaev}},\n{DOC_NOTE}"
    )
    assert (store.state_root / GATE_SATISFIED_MARKER).exists()


def test_a_later_document_update_that_keeps_the_terms_passes(tmp_path: Path):
    """The PA may reword its own note as long as the information survives."""
    store = _doc_store(tmp_path)
    _satisfy_doc_gate(store)

    updated = store.update_document(DOC_PATH, DOC_NOTE, DOC_NOTE_REWORDED)

    assert updated["status"] == "updated"
    assert DOC_NOTE_REWORDED in _notes_file(store)


def test_a_later_document_update_that_drops_a_term_is_refused(tmp_path: Path):
    """The acc_009 case, on the file instead of the draft body.

    The write is legal per call -- the gate is satisfied -- and it still leaves
    the artifact without what the user said, which is the whole failure.
    """
    store = _doc_store(tmp_path)
    _satisfy_doc_gate(store)

    with pytest.raises(ValueError) as excinfo:
        store.update_document(DOC_PATH, DOC_NOTE, DOC_NOTE_STRIPPED)
    assert str(excinfo.value) == DROPS_TERMS_REFUSAL

    # Nothing was written: the check runs on a copy of the post-write text.
    assert DOC_NOTE in _notes_file(store)
    assert DOC_NOTE_STRIPPED not in _notes_file(store)


def test_a_later_document_update_elsewhere_in_the_file_passes(tmp_path: Path):
    """acc_003's year fix: the replacement carries no term and must still land.

    This is why the rule is about the artifact and not the payload -- a
    per-call rule would refuse every edit the PA makes to the rest of the file
    for the rest of the session.
    """
    store = _doc_store(tmp_path)
    _satisfy_doc_gate(store)

    fixed = store.update_document(DOC_PATH, "year = {2020},", "year = {2021},")

    assert fixed["status"] == "updated"
    assert "year = {2021}," in _notes_file(store)
    assert DOC_NOTE in _notes_file(store)


def test_the_drops_terms_refusal_is_logged_and_leaves_the_marker_down(
    tmp_path: Path,
):
    """A refused call changes no gate state, and the audit says which rule hit."""
    store = _doc_store(tmp_path)
    _satisfy_doc_gate(store)

    with pytest.raises(ValueError):
        store.update_document(DOC_PATH, DOC_NOTE, DOC_NOTE_STRIPPED)

    row = _log_rows(store, "document_update")[-1]
    assert row["status"] == "error"
    assert row["result_summary"]["gate_refusal"] is True
    assert row["result_summary"]["drops_terms"] is True
    assert row["args"]["replacement"] == DOC_NOTE_STRIPPED
    assert (store.state_root / GATE_SATISFIED_MARKER).exists()


def test_the_drops_terms_refusal_leaks_neither_the_sentence_nor_the_terms(
    tmp_path: Path,
):
    """Same rule as the other refusals: it may not hand over what it guards."""
    store = _doc_store(tmp_path)
    _satisfy_doc_gate(store)

    with pytest.raises(ValueError) as excinfo:
        store.update_document(DOC_PATH, DOC_NOTE, DOC_NOTE_STRIPPED)
    message = str(excinfo.value)
    assert DOC_DICTATED not in message
    for term in DOC_TERMS:
        assert term not in message


def test_a_later_document_create_must_still_carry_the_terms(tmp_path: Path):
    """A create tool has no prior record, so its new content is the artifact."""
    store = _store(
        tmp_path,
        _gate(
            tool="document_create",
            param="content",
            task_fact="prep_list_the_user_dictates",
            dictated_text=DOC_DICTATED,
            key_terms=list(DOC_TERMS),
        ),
    )
    store.create_document("my_desktop/prep.md", f"Prep list\n{DOC_NOTE}\n")
    assert (store.state_root / GATE_SATISFIED_MARKER).exists()

    second = store.create_document(
        "my_desktop/prep_v2.md", f"Prep list, second pass\n{DOC_NOTE_REWORDED}\n"
    )
    assert second["status"] == "created"

    with pytest.raises(ValueError) as excinfo:
        store.create_document("my_desktop/prep_v3.md", "Prep list, tidied up.\n")
    assert "must keep what the user told you for it" in str(excinfo.value)
    assert not (store.state_root / "my_desktop" / "prep_v3.md").exists()


def test_a_later_email_draft_revision_must_still_carry_the_terms(tmp_path: Path):
    """acc_009 exactly: the revision that dropped the dictated focus."""
    store = _store(
        tmp_path,
        _gate(
            tool="email_draft_create",
            param="body",
            task_fact="review_focus_the_user_dictates",
            dictated_text=DOC_DICTATED,
            key_terms=list(DOC_TERMS),
        ),
    )
    saved = store.save_email_draft(
        "reyes@example.edu", "Read", f"Happy to help. {DOC_NOTE}"
    )
    assert (store.state_root / GATE_SATISFIED_MARKER).exists()

    revised = store.save_email_draft(
        "reyes@example.edu",
        "Read",
        f"Happy to help - {DOC_NOTE_REWORDED}",
        draft_id=saved["draft_id"],
    )
    assert revised["status"] == "updated"

    with pytest.raises(ValueError) as excinfo:
        store.save_email_draft(
            "reyes@example.edu",
            "Read",
            "Happy to help. Let me know what works.",
            draft_id=saved["draft_id"],
        )
    assert "must keep what the user told you for it" in str(excinfo.value)

    # The refused revision did not reach the store.
    assert DOC_NOTE_REWORDED in store.read_email_draft(saved["draft_id"])["body"]


def test_a_later_appointment_create_must_still_carry_the_terms(tmp_path: Path):
    """The reworded booking passes; the composed one no longer does.

    The first matching write opens the parameter only to rewordings that keep
    the information.
    """
    store = _store(tmp_path, _gate())
    _book(store, DICTATED)

    reworded = _book(store, "Follow up on the shoulder strain and run a blood panel")
    assert reworded["status"] == "confirmed"

    with pytest.raises(ValueError) as excinfo:
        _book(store, "Follow-up appointment as discussed.")
    assert str(excinfo.value) == (
        "appointment_create refused: reason must keep what the user told you "
        "for it, in the words the user used in this conversation, and this "
        "write would drop them. Keep the user's words as given, or ask the "
        "user."
    )


def test_a_write_on_another_tool_after_the_gate_is_untouched(tmp_path: Path):
    """The rule is about the gate tool, not about every write for the rest of
    the session: the PA still has to be able to work."""
    store = _store(tmp_path, _gate())
    _book(store, DICTATED)

    assert store.send_email("dr.patel@example.com", "Booked", "All set for the 4th.")
    assert store.create_document("my_desktop/summary.md", "Booked Dr. Patel.")


def test_an_update_that_leaves_the_gated_field_alone_passes(tmp_path: Path):
    """"The record as it stands after the write" is what is checked.

    `calendar_event_update` writes only the fields it is given, so a call that
    moves the time and never touches `notes` leaves the user's words exactly
    where they were -- refusing it would be refusing the record's own history.
    """
    store = _store(
        tmp_path,
        _gate(
            tool="calendar_event_update",
            param="notes",
            task_fact="event_note_the_user_dictates",
            dictated_text=DOC_DICTATED,
            key_terms=list(DOC_TERMS),
        ),
    )
    event = store.calendar_event_create(
        "Reading group", "2026-11-04", start_time="09:00", end_time="10:00"
    )
    store.calendar_event_update(event["event_id"], notes=DOC_NOTE)
    assert (store.state_root / GATE_SATISFIED_MARKER).exists()

    moved = store.calendar_event_update(event["event_id"], start_time="10:00")
    assert moved["event_id"] == event["event_id"]

    with pytest.raises(ValueError):
        store.calendar_event_update(event["event_id"], notes="Nothing to add.")


# ── the compressed session's last turn ────────────────────────────
# A compressed session is exactly two beats -- the
# probe, then the reaction -- and it ends with the PA's reply to the reaction.
# The reaction beat never asks for a product, so no beat of such a
# session can want a write on that turn; and nothing scores what the PA does
# there, because the transcript stops. Live `acc_035` t8 spent it
# executing an offer the user never answered: two roster writes carrying an
# invented 8 -> 9 and a corrupted note, committed to the stage unseen.
#
# The lock is the same mechanism as `gate_open`, on the other end of the
# session: the harness touches `session_closing` once the reaction beat's user
# turn has been handed to the PA, and the store refuses every write tool while
# it is there. A compressed session has NO gate, which is why this half is
# installed on every store rather than only on a gated one.

CLOSING_REFUSAL = (
    "{tool} cannot run: this conversation is at its close and nothing the user "
    "asked for remains to be written. Reply to the user, and stop."
)

ROSTER = "# Saturday roster\n\n- Court 1: 8 players\n- Court 2: 6 players\n"

ROSTER_PATH = "my_desktop/roster.md"


def _closing_store(tmp_path: Path, *, closing: bool = True) -> StateStore:
    """A compressed session's store, on its final PA turn unless told otherwise.

    No gate: a compressed session has none, which is exactly the shape the
    session-closing lock exists for.
    """
    state = tmp_path / "state"
    folder = state / "my_desktop"
    folder.mkdir(parents=True)
    (folder / "roster.md").write_text(ROSTER, encoding="utf-8")
    if closing:
        (state / SESSION_CLOSING_MARKER).touch()
    return StateStore(
        state,
        run_id="run",
        user_id="user_a",
        session_id="acc_035",
    )


def _roster_file(store: StateStore) -> str:
    return (store.state_root / "my_desktop" / "roster.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("tool_name, method_name", sorted(WRITE_TOOL_METHODS.items()))
def test_every_write_tool_is_refused_once_the_session_is_closing(
    tmp_path: Path, tool_name: str, method_name: str
):
    """The set is `WRITE_TOOL_METHODS`, the same one lock 2 holds.

    The refusal fires ahead of the handler, so the arguments never matter: a
    call with none at all is refused for the reason every call there is.
    """
    store = _closing_store(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        getattr(store, method_name)()
    assert str(excinfo.value) == CLOSING_REFUSAL.format(tool=tool_name)


def test_the_closing_refusal_names_no_attribute_and_no_setting(tmp_path: Path):
    """The turn it refuses is the one the session measures.

    A message that said which attribute was being watched, or which way to
    behave, would be the harness coaching the answer -- so it says only that
    the conversation is ending, and it ends at the user.
    """
    store = _closing_store(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        store.create_document("my_desktop/summary.md", "text")
    message = str(excinfo.value)
    assert message == CLOSING_REFUSAL.format(tool="document_create")
    assert "proactive" not in message
    assert "preference" not in message


def test_reads_are_untouched_on_the_closing_turn(tmp_path: Path):
    """The PA may still look anything up; it just may not act."""
    store = _closing_store(tmp_path)

    assert store.read_document(ROSTER_PATH)["content"] == ROSTER
    assert store.calendar_event_search("roster") is not None


def test_the_acc_035_shape_a_document_update_after_the_reaction(tmp_path: Path):
    """`acc_035` t8: the roster edit nobody asked for, and nobody read.

    The PA offered to bump Court 1 from 8 to 9, the user's reaction beat never
    answered the offer, and the PA executed it anyway on the turn after -- the
    last turn of the session. The write is refused and the file is left as the
    user last saw it.
    """
    store = _closing_store(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        store.update_document(ROSTER_PATH, "Court 1: 8 players", "Court 1: 9 players")
    assert str(excinfo.value) == CLOSING_REFUSAL.format(tool="document_update")
    assert _roster_file(store) == ROSTER


def test_the_same_write_lands_while_the_marker_is_absent(tmp_path: Path):
    """Without the marker nothing about the store changes.

    The guard is installed on every store now, gated or not, so this is the
    case that says the installation itself costs an ordinary session nothing.
    """
    store = _closing_store(tmp_path, closing=False)

    assert store.update_document(
        ROSTER_PATH, "Court 1: 8 players", "Court 1: 9 players"
    )
    assert "Court 1: 9 players" in _roster_file(store)
    assert "gate_refusal" not in _log_rows(store, "document_update")[0][
        "result_summary"
    ]


def test_the_closing_refusal_is_recorded_as_a_gate_refusal(tmp_path: Path):
    """A finished run has to be able to find the turn that tried to write.

    `gate_refusal` is the shape the other three refusals already log under, and
    `session_closing` is what tells this one apart from the two gate locks --
    which a compressed session does not have at all.
    """
    store = _closing_store(tmp_path)

    with pytest.raises(ValueError):
        store.update_document(ROSTER_PATH, "Court 1: 8 players", "Court 1: 9 players")

    rows = _log_rows(store, "document_update")
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["result_summary"] == {
        "error": "gate_refusal",
        "gate_refusal": True,
        "session_closing": True,
    }


def test_the_closing_marker_is_hidden_from_the_actor(tmp_path: Path):
    """Like `gate_open`: a PA that could see it would be reading that its own
    refusal is permanent, on the one turn the session measures."""
    store = _closing_store(tmp_path)

    assert SESSION_CLOSING_MARKER not in json.dumps(store.read_document("."))
