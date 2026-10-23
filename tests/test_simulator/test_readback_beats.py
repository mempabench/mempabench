"""Which beat inspects the end state of a declared state change.

`state_changes` carries a `readback_step_id` — the read that proves the write
landed — but nothing in the schema links a step to a beat, so
`readback_beat_ids` finds the beat from the beats: the first beat after the
write whose `entry_condition` reports that change's target as done. The loop
holds those beats back from the `goal_already_met` collapse, because otherwise
the PA's own claim about its own write is what deletes the check on it (live
`acc_001` beat_06).

A COMPRESSED session has no readback beat at all — the write and its readback
both happen inside beat 1 — so the marking is skipped
there; see `test_compressed_stop.py` for what that spares the actor.

The compressed side is read from `tests/fixtures/authoring/`, not from the
live drafts: every live draft of a compressed attribute is still authored in
the old seven-beat shape and no longer validates. The uncompressed side stays
on the live drafts that do (`acc_003`, `acc_010`, `acc_011`); `acc_007` is the
draft that declares no state change at all.

Beat ids move whenever a draft is regenerated, so the expectations name the
held beats by what their `entry_condition` says, not by id.

Run with:
    PYTHONPATH=. python -m pytest tests/test_simulator/test_readback_beats.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from simulator.schemas import SessionScript, readback_beat_ids

DRAFTS = Path("Authoring/sessions/normal/drafts/user_a")
COMPRESSED_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/authoring/normal_session_compressed.yaml"
)

# The sentence the compressed generator prompt now tells drafts NOT to write on
# beat 2 — it carries a done-verb and the write target's own word, so it is
# exactly what the scan would otherwise catch.
SHOWN_AND_UPDATED = "the PA has updated the note and shown what it now says"


def _readback_beats(session_id: str) -> list[str]:
    script = SessionScript.from_authored_session_yaml(DRAFTS / f"{session_id}.yaml")
    return list(script.readback_beats)


def _draft(session_id: str) -> dict:
    return yaml.safe_load(
        (DRAFTS / f"{session_id}.yaml").read_text(encoding="utf-8")
    )


def _marked(data: dict) -> list[str]:
    from scripts.session_authoring_schema import NormalSessionDraft

    return readback_beat_ids(NormalSessionDraft.model_validate(data))


def _beat_saying(data: dict, phrase: str) -> str:
    """The id of the one beat whose `entry_condition` carries `phrase`."""
    ids = [
        beat["beat"]
        for beat in data["beats"]
        if phrase in beat["entry_condition"].casefold()
    ]
    assert len(ids) == 1, (phrase, ids)
    return ids[0]


def test_a_compressed_session_holds_no_beat_however_its_beats_read() -> None:
    """The two-beat `verbosity` fixture, given the worst beat 2 the scan could
    be handed: a declared write of the citation notes, and an
    `entry_condition` that says the PA updated the note and showed it. The
    beat is the scan's first candidate and matches on both counts, and it is
    still not marked, because the write and its readback both
    belong to beat 1.
    """
    data = yaml.safe_load(COMPRESSED_FIXTURE.read_text(encoding="utf-8"))
    assert _marked(data) == []

    data["task_execution"]["state_changes"] = [
        {
            "change_id": "record_the_placement",
            "atomic_step_id": "report_detailed_assessment",
            "target": "documents/research/citation_notes.md",
            "operation": "append the placement decision",
            "readback_step_id": "read_citation_notes",
        }
    ]
    data["beats"][1]["entry_condition"] = SHOWN_AND_UPDATED
    assert _marked(data) == []


def test_only_the_target_attribute_decides_it_on_otherwise_identical_beats() -> None:
    """Same draft, same beats, same `entry_condition`s, same `state_changes`:
    the only difference is whether the target attribute is compressed.

    The pair cannot be two files any more — a compressed draft is exactly two
    beats — so the target is swapped on the loaded model. That is the whole
    input `readback_beat_ids` reads to decide.
    """
    from scripts.session_authoring_schema import NormalSessionDraft

    data = _draft("acc_003")
    held = _beat_saying(
        data, "the correction note has been saved to the citation notes"
    )

    draft = NormalSessionDraft.model_validate(data)
    assert draft.meta.target_attribute == "process_visibility"
    assert readback_beat_ids(draft) == [held]

    compressed = draft.model_copy(
        update={"meta": draft.meta.model_copy(update={"target_cell": "work_verbosity"})}
    )
    assert readback_beat_ids(compressed) == []


def test_the_beat_that_says_the_write_was_shown_is_marked_only_uncompressed() -> None:
    """The wording banned on a compressed beat 2 is still a readback on
    a beat of an uncompressed session — the guard is on the attribute, not on
    the sentence. `acc_003` writes the same citation notes; put the sentence on
    an earlier beat and its one write holds that beat instead.
    """
    data = _draft("acc_003")
    assert _marked(data) == [
        _beat_saying(data, "the correction note has been saved to the citation notes")
    ]

    data["beats"][1]["entry_condition"] = SHOWN_AND_UPDATED
    assert _marked(data) == [data["beats"][1]["beat"]]


def test_a_session_that_writes_nothing_holds_no_beat() -> None:
    assert _readback_beats("acc_007") == []


def test_acc_011_holds_the_beat_that_opens_on_its_write() -> None:
    """`acc_011` writes the prep list to the desktop, and its one verify beat
    opens on that file as created, so the write holds it."""
    data = _draft("acc_011")
    assert _marked(data) == [
        _beat_saying(data, "prep list document has been created")
    ]


def test_each_state_change_holds_one_beat_and_they_do_not_share() -> None:
    """Two writes in `acc_010` (`autonomy_level`, not compressed) — the
    calendar hold and then the room booking — reported done by two separate
    beats: a later change may not reuse an earlier change's beat, so the
    marking walks forward and each write holds its own beat."""
    data = _draft("acc_010")
    hold_beat = _beat_saying(data, "the session has been entered on the calendar")
    booking_beat = _beat_saying(data, "the room booking has been placed")
    assert booking_beat != hold_beat
    assert _readback_beats("acc_010") == [hold_beat, booking_beat]


def test_two_changes_landing_in_one_write_beat_hold_only_that_beat() -> None:
    """`acc_003` applies its citation fix before one readback beat, `b6`, the
    beat that opens on the entry as written. Declare a second fix to the same
    file — both land before that same beat — and nothing further is held: the
    closing beat asks only whether the rest of the annotation survived, so it
    reports no citation-notes write, and the second change may not reuse the
    first change's beat."""
    data = _draft("acc_003")
    held = _beat_saying(
        data, "the correction note has been saved to the citation notes"
    )
    assert _marked(data) == [held]

    second_fix = copy.deepcopy(data["task_execution"]["state_changes"][0])
    second_fix["change_id"] = "change_sarpong_page_range"
    data["task_execution"]["state_changes"].append(second_fix)
    assert _marked(data) == [held]


def test_acc_039_carries_the_write_and_read_tool_of_each_change() -> None:
    """Each held beat also carries the pair of catalog tools its own state
    change names — `atomic_step_id` writes, `readback_step_id` reads back.

    Live `acc_039` t=20 is what this is for: the loop's "already
    read back" switch keyed on any write followed by any read, took the
    calendar hold `beat_4` inspects as proof for `beat_6`, and dropped the
    appointment beat's ask while the clinic booking had errored out.
    """
    script = SessionScript.from_authored_session_yaml(DRAFTS / "acc_039.yaml")
    hold_beat, booking_beat = script.readback_beats
    assert script.readback_beat_tools == {
        hold_beat: ("calendar_event_create", "calendar_event_read"),
        booking_beat: ("appointment_create", "appointment_read"),
    }


def test_a_session_that_writes_nothing_carries_no_pair() -> None:
    script = SessionScript.from_authored_session_yaml(DRAFTS / "acc_007.yaml")
    assert script.readback_beat_tools == {}
