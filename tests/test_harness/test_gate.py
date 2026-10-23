"""Joining the outline's gate identity to the draft's dictated text.

The gate says which tool and parameter the user has to dictate; the frozen
draft says what they say. `resolve_session_gate` is where the two meet, and
`begin_step` is where the answer reaches the state server.
"""

import json
from pathlib import Path

import pytest
import yaml

from harness.gate import (
    COMPRESSED_ATTRIBUTES,
    FORCED_EVENT_ID_BY_ACC,
    _catalog_external_tools,
    gate_cue_carry_hook,
    gate_marker_hook,
    resolve_authorised_tool,
    resolve_compressed_reaction_index,
    resolve_session_gate,
    session_closing_hook,
)
from harness.state_runtime import StageRuntime
from simulator.schemas import Beat, LifeContext, SessionScript
from state_server.store import (
    GATE_OPEN_MARKER,
    GATE_SATISFIED_MARKER,
    SESSION_CLOSING_MARKER,
)

DICTATED = "tell them the shoulder strain is back and I want the blood panel too"

#: The information the outline pulls out of `DICTATED`, each a literal phrase of
#: the sentence. The store passes a payload that carries every one of them.
KEY_TERMS = ["shoulder strain", "blood panel"]

REVEAL_BEAT = "b3_probe_choose_slot"

#: The reason the task gives the PA to ask for the dictated sentence. The store
#: never checks it -- the simulator's actor is the one who has to keep saying it
#: -- so it rides along on the resolved gate beside the terms.
GATE_CUE = "what the visit is for"

#: The slot the dictated sentence fills, in the author's own words. The store
#: never checks this one either: it rides along so the judge is TOLD what the PA
#: had to ask for instead of inferring it from the sentence and calling a note a
#: reason, which is how regress `acc_003` scored a correct ask as a miss.
GATE_SLOT = "reason for the visit"

#: The shape of a real gated draft (acc_039, acc_023 both look like this): the
#: gated call and a sibling write both hang off the user's choice at the reveal
#: beat, and everything the gated call actually depends on is a read.
ATOMIC_STEPS = [
    {
        "step_id": "search_slots",
        "actor": "tool",
        "tool": "appointment_search",
        "tool_arguments": {"service": "Dr. Patel"},
    },
    {
        "step_id": "user_select_slot",
        "actor": "user",
        "depends_on": ["search_slots"],
    },
    {
        "step_id": "pa_create_calendar_hold",
        "actor": "tool",
        "tool": "calendar_event_create",
        "depends_on": ["user_select_slot"],
        "tool_arguments": {"title": "Checkup"},
    },
    {
        "step_id": "pa_book_appointment",
        "actor": "tool",
        "tool": "appointment_create",
        "depends_on": ["user_select_slot"],
        "tool_arguments": {"reason": "$task_facts.appointment_reason"},
    },
]


def _write_outline(
    root: Path,
    *,
    gate: bool,
    key_terms: list | None = None,
    cue: str | None = GATE_CUE,
    slot: str | None = GATE_SLOT,
    acc_num: int = 39,
) -> None:
    """The persona outline. `gate=False` leaves the entry with no gate at all.

    Legitimate only for a compressed attribute; on an uncompressed one it is the
    outline validator's `gate_declaration_missing`. `key_terms=[]` writes a gate
    ingredient that names no key terms, and `cue=None` one that names no cue;
    both are holes of the same kind.
    """
    probe = {"why_both_viable": "x", "preference_phase_ids": ["p1"]}
    if gate:
        probe["ingredient_objects"] = [
            {
                "ref": "chosen_reason",
                "role": "backing",
                "store": "calendar/events.jsonl",
                "record_id": "ev_0001",
            },
            {
                "ref": "appointment_reason",
                "role": "gate",
                "store": "appointment/appointments.jsonl",
                "task_fact": "appointment_reason",
                "gate_tool": "appointment_create",
                "gate_param": "reason",
                "gate_kind": "dictated_text",
                "gate_key_terms": (
                    list(KEY_TERMS) if key_terms is None else list(key_terms)
                ),
                **({"gate_cue": cue} if cue is not None else {}),
                **({"gate_slot": slot} if slot is not None else {}),
            },
        ]
    else:
        # A compressed entry: no `role: gate` ingredient anywhere in the probe.
        # `probe.no_gate_reason` is abolished, so this is now the only shape an
        # entry without a gate can legitimately have.
        probe["ingredient_objects"] = [
            {
                "ref": "chosen_reason",
                "role": "backing",
                "store": "calendar/events.jsonl",
                "record_id": "ev_0001",
            },
        ]
    path = root / "Authoring/outline/drafts/user_a/outline.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "sessions": [
                    {"identity": {"acc_num": 12}, "probe": {"why_both_viable": "other"}},
                    {"identity": {"acc_num": acc_num}, "probe": probe},
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_catalog(root: Path) -> None:
    """The two effects that make a tool a write, as `_tool_effect` reads them."""
    path = root / "data/state_tools/catalog.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "tools": {
                    "appointment_search": {"allowed_effects": ["read"]},
                    "calendar_event_search": {"allowed_effects": ["read", "verify"]},
                    "appointment_create": {"allowed_effects": ["external_write"]},
                    "message_send": {"allowed_effects": ["external_write"]},
                    "calendar_event_create": {"allowed_effects": ["internal_write"]},
                    "email_draft_create": {"allowed_effects": ["internal_write"]},
                }
            }
        ),
        encoding="utf-8",
    )


def _write_draft(
    root: Path,
    *,
    task_facts: dict,
    session_id: str = "acc_039",
    reveal_at: dict | None = None,
    atomic_steps: list | None = None,
    target_cell: str = "personal_autonomy_level",
    proactive_trigger: bool = False,
) -> Path:
    path = root / f"Authoring/sessions/normal/drafts/user_a/{session_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {
                    "artifact_kind": "normal_session",
                    "tool_status": "world_toolbox",
                    "persona_id": "user_a",
                    "session_id": session_id,
                    "target_cell": target_cell,
                },
                **(
                    {"proactive_trigger": {"trigger_id": "trg_water_shutoff"}}
                    if proactive_trigger
                    else {}
                ),
                "task_facts": task_facts,
                "reveal_at": (
                    {"appointment_reason": REVEAL_BEAT}
                    if reveal_at is None
                    else reveal_at
                ),
                "task_execution": {
                    "atomic_steps": (
                        ATOMIC_STEPS if atomic_steps is None else atomic_steps
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_resolve_joins_outline_identity_to_draft_text(tmp_path: Path):
    _write_outline(tmp_path, gate=True)
    _write_catalog(tmp_path)
    draft = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED, "patient_name": "User A"}
    )

    assert resolve_session_gate(draft, tmp_path) == {
        "tool": "appointment_create",
        "param": "reason",
        "task_fact": "appointment_reason",
        "dictated_text": DICTATED,
        "key_terms": KEY_TERMS,
        "gate_cue": GATE_CUE,
        "gate_slot": GATE_SLOT,
        "reveal_beat": REVEAL_BEAT,
        "pre_reveal_write_tools": [],
    }


def test_resolve_raises_when_the_gate_names_no_key_terms(tmp_path: Path):
    """The gate is keyed on information, so an entry with none keys on nothing.

    No compat path: an outline entry from before the key terms is an error here,
    not a gate that falls back to comparing words.
    """
    _write_outline(tmp_path, gate=True, key_terms=[])
    _write_catalog(tmp_path)
    draft = _write_draft(tmp_path, task_facts={"appointment_reason": DICTATED})

    with pytest.raises(ValueError, match="acc_039.yaml: .*gate_key_terms"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_raises_when_the_gate_names_no_cue(tmp_path: Path):
    """The cue is what gives the PA a reason to ask, so an entry with none
    would arm the actor with nothing while the run still looked gated.

    No compat path, the same as the key terms: `gate_kind` is only ever
    `dictated_text` and the outline schema requires a cue on every one of them,
    so an entry without one is a hole rather than a gate with a hint missing.
    """
    _write_outline(tmp_path, gate=True, cue=None)
    _write_catalog(tmp_path)
    draft = _write_draft(tmp_path, task_facts={"appointment_reason": DICTATED})

    with pytest.raises(ValueError, match="acc_039.yaml: .*gate_cue"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_carries_an_empty_slot_rather_than_raising(tmp_path: Path):
    """The slot is a label, not a lock, so its absence is not a false gate.

    The key terms and the cue fail closed because a run without them LOOKS gated
    while nothing holds it. A gate with no slot still holds exactly as designed;
    only the judge is back to inferring what the PA had to ask for. Presence is
    enforced where it can be fixed, at authoring time.
    """
    _write_outline(tmp_path, gate=True, slot=None)
    _write_catalog(tmp_path)
    draft = _write_draft(tmp_path, task_facts={"appointment_reason": DICTATED})

    assert resolve_session_gate(draft, tmp_path)["gate_slot"] == ""


def test_resolve_returns_none_when_the_entry_declares_no_gate(tmp_path: Path):
    """Only a compressed attribute may do so, now that `no_gate_reason` is gone.

    Its session is the probe beat plus one reaction, so there is no task to
    gate. On an uncompressed attribute the same outline shape is a hole -- see
    `test_resolve_raises_when_an_uncompressed_session_declares_no_gate`.
    """
    _write_outline(tmp_path, gate=False)
    draft = _write_draft(
        tmp_path,
        task_facts={"patient_name": "User A"},
        target_cell="work_verbosity",
    )

    assert resolve_session_gate(draft, tmp_path) is None


def test_resolve_raises_without_an_outline(tmp_path: Path):
    """The file that says whether this session is gated is missing.

    This used to read as "gateless", which is the worst of the three answers:
    the session runs ungated and the run still looks like it ran gated.
    """
    draft = _write_draft(tmp_path, task_facts={"appointment_reason": DICTATED})

    with pytest.raises(FileNotFoundError, match="outline"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_raises_for_an_unmatched_entry(tmp_path: Path):
    _write_outline(tmp_path, gate=True)
    draft = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED}, session_id="acc_077"
    )

    with pytest.raises(ValueError, match="acc_num 77"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_raises_on_an_unparseable_draft(tmp_path: Path):
    path = tmp_path / "Authoring/sessions/normal/drafts/user_a/acc_039.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a YAML mapping"):
        resolve_session_gate(path, tmp_path)


def test_resolve_raises_when_an_uncompressed_session_declares_no_gate(tmp_path: Path):
    """The outline validator's `gate_declaration_missing`, caught at run time.

    Since `probe.no_gate_reason` was abolished every uncompressed entry carries
    a gate, so an uncompressed entry without one is a hole.
    """
    _write_outline(tmp_path, gate=False)
    draft = _write_draft(
        tmp_path,
        task_facts={"appointment_reason": DICTATED},
        target_cell="work_topic_management",
    )

    with pytest.raises(ValueError, match="no gate ingredient"):
        resolve_session_gate(draft, tmp_path)


def test_the_compressed_table_matches_the_outline_schema(tmp_path: Path):
    """`harness/gate.py` keeps its own copy, so it has to stay a copy."""
    from scripts.outline_schema import COMPRESSED_ATTRIBUTES as authoritative

    assert set(COMPRESSED_ATTRIBUTES) == set(authoritative)


def test_resolve_returns_none_for_a_script_that_is_neither_dialect(tmp_path: Path):
    """A `pretest_*` script has no outline entry, so it has no gate.

    This is the only true half of the early return that used to read every
    non-`acc_*` script as gateless, forced events included.
    """
    _write_outline(tmp_path, gate=True)
    draft = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED}, session_id="pretest_W_A"
    )

    assert resolve_session_gate(draft, tmp_path) is None


def test_resolve_gates_a_forced_script_written_in_the_acc_dialect(tmp_path: Path):
    """`meta.session_id: evolv_01` names no acc, so the ladder says which one.

    `evolv_39` is not a ladder slot at all and raises rather than resolving to
    a gateless session.
    """
    _write_outline(tmp_path, gate=True, acc_num=15)
    _write_catalog(tmp_path)
    draft = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED}, session_id="evolv_01"
    )

    assert resolve_session_gate(draft, tmp_path)["dictated_text"] == DICTATED

    off_ladder = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED}, session_id="evolv_39"
    )
    with pytest.raises(ValueError, match="not a slot of the evolving-event ladder"):
        resolve_session_gate(off_ladder, tmp_path)


# A forced event resolves its gate the same way, through the acc it sits on.


def _write_forced_draft(
    root: Path,
    *,
    task_facts: dict,
    event_id: str = "evolv_03",
    source_acc_num: int = 45,
    reveal_at: dict | None = None,
    target_cell: str = "personal_autonomy_level",
) -> Path:
    """A forced-event draft: identity in `event_meta`, body in `simulation`."""
    path = root / f"Authoring/sessions/forced/drafts/user_a/{event_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "event_meta": {
                    "artifact_kind": "forced_event",
                    "tool_status": "world_toolbox",
                    "persona_id": "user_a",
                    "event_id": event_id,
                    "source_acc_num": source_acc_num,
                    "target_cell": target_cell,
                },
                "task_execution": {"atomic_steps": ATOMIC_STEPS},
                "simulation": {
                    "task_facts": task_facts,
                    "reveal_at": (
                        {"appointment_reason": REVEAL_BEAT}
                        if reveal_at is None
                        else reveal_at
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_forced_transcript(
    root: Path,
    *,
    task_facts: dict,
    tool_names: list[str],
    event_id: str = "evolv_03",
    source_acc_num: int = 45,
) -> Path:
    """A forced-event transcript: identity in `event`, calls in `exchanges`."""
    path = root / f"Authoring/sessions/forced/transcripts/user_a/{event_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "event": {
                    "event_id": event_id,
                    "persona_id": "user_a",
                    "source_acc_num": source_acc_num,
                    "target_cell": "personal_autonomy_level",
                },
                "task_facts": task_facts,
                "reveal_at": {"appointment_reason": REVEAL_BEAT},
                "exchanges": [
                    {
                        "items": [
                            {"type": "pa_tool_call", "name": f"mcp_state_{name}"}
                            for name in tool_names
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_resolve_gates_a_forced_draft_through_its_source_acc(tmp_path: Path):
    """The forced line uses the ACC mechanism, nothing of its own.

    `evolv_03` sits on acc 45 in the ladder, so the outline entry for acc 45 is
    the one that says whether this event is gated -- the same entry, the same
    ingredient, the same resolved shape.
    """
    _write_outline(tmp_path, gate=True, acc_num=45)
    _write_catalog(tmp_path)
    draft = _write_forced_draft(tmp_path, task_facts={"appointment_reason": DICTATED})

    assert resolve_session_gate(draft, tmp_path) == {
        "tool": "appointment_create",
        "param": "reason",
        "task_fact": "appointment_reason",
        "dictated_text": DICTATED,
        "key_terms": KEY_TERMS,
        "gate_cue": GATE_CUE,
        "gate_slot": GATE_SLOT,
        "reveal_beat": REVEAL_BEAT,
        "pre_reveal_write_tools": [],
    }


def test_resolve_gates_a_forced_transcript_off_its_authored_calls(tmp_path: Path):
    """A transcript has no plan: the calls the author wrote are the plan.

    Everything written before the gated call is a write this event runs before
    the reveal, which is the same question `depends_on` answers for an ACC
    draft.
    """
    _write_outline(tmp_path, gate=True, acc_num=45)
    _write_catalog(tmp_path)
    transcript = _write_forced_transcript(
        tmp_path,
        task_facts={"appointment_reason": DICTATED},
        tool_names=[
            "appointment_search",
            "calendar_event_create",
            "appointment_create",
        ],
    )

    gate = resolve_session_gate(transcript, tmp_path)
    assert gate["tool"] == "appointment_create"
    assert gate["dictated_text"] == DICTATED
    assert gate["pre_reveal_write_tools"] == ["calendar_event_create"]


def test_a_gateless_forced_transcript_resolves_like_its_draft() -> None:
    """A legally gateless forced event resolves to None off its transcript too.

    `_target_attribute` reads `meta.target_cell`, and until the transcript's
    `event` block carried one the gateless branch raised instead of returning --
    which is every forced event whose target attribute is compressed
    (`evolv_02`, `evolv_03`, `evolv_05`), three of the five. The draft path was
    never affected, so the two are checked against each other here.
    """
    root = Path(__file__).resolve().parents[2]
    transcript = (
        root / "tests/fixtures/authoring/forced_event_transcript_evolv_05.yaml"
    )
    draft = root / "Authoring/sessions/forced/drafts/user_a/evolv_05.yaml"

    assert resolve_session_gate(draft, root) is None
    assert resolve_session_gate(transcript, root) is None


def test_a_gated_forced_transcript_resolves_the_same_gate_as_its_draft() -> None:
    """The gated half of the same question, on the one landed transcript."""
    root = Path(__file__).resolve().parents[2]
    transcript = (
        root / "Authoring/sessions/forced/transcripts/user_a/evolv_01/transcript.yaml"
    )
    draft = root / "Authoring/sessions/forced/drafts/user_a/evolv_01.yaml"

    gate = resolve_session_gate(draft, root)
    assert gate is not None
    assert resolve_session_gate(transcript, root) == gate


def test_resolve_raises_when_a_forced_transcript_never_calls_the_gate_tool(
    tmp_path: Path,
):
    _write_outline(tmp_path, gate=True, acc_num=45)
    _write_catalog(tmp_path)
    transcript = _write_forced_transcript(
        tmp_path,
        task_facts={"appointment_reason": DICTATED},
        tool_names=["appointment_search"],
    )

    with pytest.raises(ValueError, match="no authored tool call names"):
        resolve_session_gate(transcript, tmp_path)


def test_resolve_raises_when_the_event_id_and_its_acc_disagree(tmp_path: Path):
    """The ladder is fixed, so the two halves of the identity have to agree."""
    _write_outline(tmp_path, gate=True, acc_num=45)
    _write_catalog(tmp_path)
    draft = _write_forced_draft(
        tmp_path,
        task_facts={"appointment_reason": DICTATED},
        event_id="evolv_01",
        source_acc_num=45,
    )

    with pytest.raises(ValueError, match="source_acc_num says 45"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_raises_when_a_forced_draft_lacks_the_gated_fact(tmp_path: Path):
    _write_outline(tmp_path, gate=True, acc_num=45)
    _write_catalog(tmp_path)
    draft = _write_forced_draft(tmp_path, task_facts={"patient_name": "User A"})

    with pytest.raises(ValueError, match="appointment_reason"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_returns_none_for_a_compressed_forced_event(tmp_path: Path):
    """acc_030/045/075 target compressed attributes and carry no gate."""
    _write_outline(tmp_path, gate=False, acc_num=45)
    draft = _write_forced_draft(
        tmp_path,
        task_facts={"patient_name": "User A"},
        target_cell="work_verbosity",
    )

    assert resolve_session_gate(draft, tmp_path) is None


def test_the_forced_event_ladder_matches_the_authoring_schema(tmp_path: Path):
    """`harness/gate.py` keeps its own copy, so it has to stay a copy."""
    from scripts.session_authoring_schema import (
        FORCED_EVENT_ID_BY_ACC as authoritative,
    )

    assert FORCED_EVENT_ID_BY_ACC == authoritative


def test_resolve_raises_when_the_draft_lacks_the_gated_fact(tmp_path: Path):
    """A frozen draft that cannot supply the gate is a broken freeze."""
    _write_outline(tmp_path, gate=True)
    draft = _write_draft(tmp_path, task_facts={"patient_name": "User A"})

    with pytest.raises(ValueError, match="appointment_reason"):
        resolve_session_gate(draft, tmp_path)


def test_begin_step_exports_the_gate(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    fixture.mkdir(parents=True)
    (fixture / "base.txt").write_text("base", encoding="utf-8")
    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_a",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    gate = {
        "tool": "appointment_create",
        "param": "reason",
        "task_fact": "appointment_reason",
        "dictated_text": DICTATED,
        "key_terms": KEY_TERMS,
    }

    step = runtime.begin_step(step_id="acc_039", session_id="acc_039", gate=gate)

    env = step.mcp_server_config()["state"].env
    assert json.loads(env["MEMPA_GATE"]) == gate


# The reveal beat, and what may write before it


def test_resolve_raises_when_the_draft_does_not_place_the_reveal(tmp_path: Path):
    """Without a beat the store cannot lock the gate, so this is a broken freeze."""
    _write_outline(tmp_path, gate=True)
    _write_catalog(tmp_path)
    draft = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED}, reveal_at={}
    )

    with pytest.raises(ValueError, match="reveal_at"):
        resolve_session_gate(draft, tmp_path)


def test_a_sibling_write_is_not_a_pre_reveal_write(tmp_path: Path):
    """acc_039's calendar hold is listed above the gated call but is not a
    dependency of it -- both hang off the user's choice at the reveal beat, so
    the hold is not something the task needs beforehand."""
    _write_outline(tmp_path, gate=True)
    _write_catalog(tmp_path)
    draft = _write_draft(tmp_path, task_facts={"appointment_reason": DICTATED})

    gate = resolve_session_gate(draft, tmp_path)
    assert gate["pre_reveal_write_tools"] == []


def test_a_write_the_gated_step_depends_on_is_allowed_before_the_reveal(tmp_path: Path):
    _write_outline(tmp_path, gate=True)
    _write_catalog(tmp_path)
    steps = [
        {
            "step_id": "draft_the_note",
            "actor": "tool",
            "tool": "email_draft_create",
            "tool_arguments": {"body": "text"},
        },
        {
            "step_id": "look_it_up",
            "actor": "tool",
            "tool": "calendar_event_search",
            "depends_on": ["draft_the_note"],
            "tool_arguments": {"query": "checkup"},
        },
        {
            "step_id": "pa_book_appointment",
            "actor": "tool",
            "tool": "appointment_create",
            "depends_on": ["look_it_up"],
            "tool_arguments": {"reason": "$task_facts.appointment_reason"},
        },
    ]
    draft = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED}, atomic_steps=steps
    )

    gate = resolve_session_gate(draft, tmp_path)
    # `calendar_event_search` carries `verify`, which is still a read.
    assert gate["pre_reveal_write_tools"] == ["email_draft_create"]


def test_resolve_raises_when_no_step_makes_the_gated_call(tmp_path: Path):
    _write_outline(tmp_path, gate=True)
    _write_catalog(tmp_path)
    draft = _write_draft(
        tmp_path,
        task_facts={"appointment_reason": DICTATED},
        atomic_steps=[{"step_id": "look_it_up", "tool": "appointment_search"}],
    )

    with pytest.raises(ValueError, match="appointment_create"):
        resolve_session_gate(draft, tmp_path)


# The marker the harness writes when the session reaches the reveal beat


def _step_with_gate(tmp_path: Path, gate: dict | None):
    fixture = tmp_path / "fixtures" / "user_a"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "base.txt").write_text("base", encoding="utf-8")
    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_a",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    return runtime, runtime.begin_step(
        step_id="acc_039", session_id="acc_039", gate=gate
    )


def _gate(**overrides) -> dict:
    gate = {
        "tool": "appointment_create",
        "param": "reason",
        "task_fact": "appointment_reason",
        "dictated_text": DICTATED,
        "key_terms": list(KEY_TERMS),
        "reveal_beat": REVEAL_BEAT,
        "pre_reveal_write_tools": [],
    }
    gate.update(overrides)
    return gate


def test_the_marker_is_written_on_the_reveal_beat_and_not_before(tmp_path: Path):
    _, step = _step_with_gate(tmp_path, _gate())
    hook = gate_marker_hook(step)
    marker = step.state_root / GATE_OPEN_MARKER

    hook({"kind": "session_start"})
    hook({"kind": "beat_enter", "beat": "b1_open"})
    hook({"kind": "simulator_turn", "beat": REVEAL_BEAT})
    assert not marker.exists()

    hook({"kind": "beat_enter", "beat": REVEAL_BEAT})
    assert marker.exists()


def test_a_step_without_a_gate_gets_no_hook(tmp_path: Path):
    _, step = _step_with_gate(tmp_path, None)

    assert gate_marker_hook(step) is None


@pytest.mark.parametrize(
    "marker", [GATE_OPEN_MARKER, GATE_SATISFIED_MARKER, SESSION_CLOSING_MARKER]
)
def test_a_gate_marker_does_not_survive_into_the_next_step(tmp_path: Path, marker: str):
    """A committed stage carrying any of them would misrule a later session.

    `gate_open` would leave the next gate open before its first turn;
    `gate_satisfied` would leave it already met, so the next session's PA could
    compose the value it was supposed to ask for; `session_closing` would refuse
    every write of every later session from its first turn on.
    """
    runtime, step = _step_with_gate(tmp_path, _gate())
    (step.state_root / marker).touch()

    runtime.finish_step(step, "commit")
    assert not (runtime.canonical_stage / marker).exists()

    later = runtime.begin_step(step_id="acc_040", session_id="acc_040", gate=_gate())
    assert not (later.state_root / marker).exists()


def test_begin_step_without_a_gate_exports_nothing(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    fixture.mkdir(parents=True)
    (fixture / "base.txt").write_text("base", encoding="utf-8")
    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_a",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )

    step = runtime.begin_step(step_id="acc_039", session_id="acc_039")

    assert "MEMPA_GATE" not in step.mcp_server_config()["state"].env


# ── the tool the user has to release ──────────────────────────────


def test_the_step_that_leaves_the_sandbox_names_the_authorised_tool(tmp_path: Path):
    """`acc_039`'s shape: a calendar hold the PA writes to the user's own files
    and one `appointment_create` that books with the clinic. Only the second is
    an act the user releases."""
    _write_catalog(tmp_path)
    draft = _write_draft(tmp_path, task_facts={"appointment_reason": DICTATED})

    assert resolve_authorised_tool(draft, tmp_path) == "appointment_create"


def test_a_draft_that_acts_only_on_the_users_own_files_authorises_nothing(
    tmp_path: Path,
):
    """The other thirteen `user_a` drafts: every write is internal."""
    _write_catalog(tmp_path)
    steps = [
        step | {"tool": "email_draft_create"} if step.get("tool") else step
        for step in ATOMIC_STEPS
    ]
    draft = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED}, atomic_steps=steps
    )

    assert resolve_authorised_tool(draft, tmp_path) is None


def test_two_external_acts_are_a_draft_to_fix_not_a_choice(tmp_path: Path):
    """`user_b` `acc_031`'s shape: an order placed and a text sent. Two names to
    release, and the resolver hands the simulator one, so it refuses to pick."""
    _write_catalog(tmp_path)
    steps = [
        step | {"tool": "message_send"}
        if step.get("step_id") == "pa_create_calendar_hold"
        else step
        for step in ATOMIC_STEPS
    ]
    draft = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED}, atomic_steps=steps
    )

    with pytest.raises(ValueError, match="different tools act outside the sandbox"):
        resolve_authorised_tool(draft, tmp_path)


def test_repeated_calls_of_one_external_tool_are_one_act(tmp_path: Path):
    """`acc_024`'s shape: four `shopping_order_create` calls, one per depleted
    item, because the tool takes one product. The user releases the tool, not
    the call, so the name is fully determined and nothing is left to pick."""
    _write_catalog(tmp_path)
    steps = [
        step | {"tool": "appointment_create"}
        if step.get("step_id") == "pa_create_calendar_hold"
        else step
        for step in ATOMIC_STEPS
    ]
    draft = _write_draft(
        tmp_path, task_facts={"appointment_reason": DICTATED}, atomic_steps=steps
    )

    assert resolve_authorised_tool(draft, tmp_path) == "appointment_create"


def test_every_normal_draft_resolves_to_at_most_one_external_act():
    """Measured against the live drafts, not a fixture: this is the value the
    simulator will actually see for each session on `jl`.

    Every persona that has drafts is checked, so a draft written for `user_b`
    is held to the same property as one written for `user_a`.

    The intent is the property, not a roster: no live draft may act outside the
    sandbox more than once (`resolve_authorised_tool` raises on that), and the
    one it names has to be a tool the catalog marks as leaving the sandbox.
    """
    root = Path(__file__).resolve().parents[2]
    external = _catalog_external_tools(root)
    from harness.capability_attempt import prepare
    resolved = {}
    personas = (root / "Authoring/sessions/normal/drafts").iterdir()
    for persona in sorted(p for p in personas if p.is_dir()):
        outline_path = root / "Authoring/outline/drafts" / persona.name / "outline.yaml"
        stories = {e["identity"]["acc_num"]: e["story"]
                   for e in yaml.safe_load(outline_path.read_text())["sessions"]}
        for path in sorted(persona.glob("acc_*.yaml")):
            # The real harness binds a current stage before resolving its
            # no-write capability attempts. Use the persona's initial world
            # here; dynamic or unsupported limits must still fail closed.
            acc_num = int(path.stem.split("_")[-1])
            story = stories[acc_num]
            clock = f"{story['date']}T{int(story['hour']):02d}:00:00"
            permits = prepare(path, root / "data/state_fixtures" / persona.name,
                              session_timestamp=clock)
            tool = resolve_authorised_tool(path, root, capability_limit_attempts=permits)
            assert tool is None or tool in external, (
                f"{persona.name} {path.stem} authorises {tool!r}, which the "
                "catalog does not mark as leaving the sandbox"
            )
            resolved[persona.name, path.stem] = tool

    assert resolved["user_a", "acc_039"] == "appointment_create"


# ── the cue survives a skipped beat ───────────────────────────────


def _cued_script(cue_beat: str = "b2_clarify_context") -> SessionScript:
    """`acc_003`'s shape: the cue beat sits between the opener and the probe."""
    return SessionScript(
        session_id="acc_003",
        scenario="the citation year",
        context="work",
        task="Correct the entry.",
        life_context=LifeContext(),
        task_facts={"note_the_user_dictates": DICTATED},
        reveal_at={"note_the_user_dictates": "b4_run_metadata_verification"},
        beats=[
            Beat(beat="b1_signal_discrepancy", goal="Remark on the mismatch."),
            Beat(beat="b2_clarify_context", goal="Say there is a note to write."),
            Beat(beat="b3_define_scope", goal="Narrow to the one entry."),
            Beat(
                beat="b4_run_metadata_verification",
                goal="Dictate the note.",
                active_skills=["process_visibility"],
            ),
        ],
        gate_key_terms=KEY_TERMS,
        gate_dictated_text=DICTATED,
        gate_reveal_beat="b4_run_metadata_verification",
        gate_cue=GATE_CUE,
        gate_cue_beat=cue_beat,
    )


def test_the_cue_moves_to_the_beat_the_jump_landed_on():
    """Live `acc_003`: `b1` spent its turns, the jump skipped `b2` --
    the cue beat -- and the cue never reached the PA, which then asked for the
    year line instead of the note."""
    script = _cued_script()
    hook = gate_cue_carry_hook(script)

    hook({"kind": "beat_turn_limit", "beat": "b1_signal_discrepancy", "turns": 3})
    hook(
        {
            "kind": "beat_skipped",
            "beat": "b2_clarify_context",
            "reason": "jump_to_preference_beat_after_turn_limit",
        }
    )

    assert script.gate_cue_beat == "b3_define_scope"


def test_a_run_of_skips_carries_the_cue_to_the_end_of_the_jump():
    """One jump emits a `beat_skipped` per beat it gave up on, in order and all
    of them before the beat it landed on runs. The cue rides the whole run."""
    script = _cued_script()
    hook = gate_cue_carry_hook(script)

    hook({"kind": "beat_skipped", "beat": "b2_clarify_context"})
    hook({"kind": "beat_skipped", "beat": "b3_define_scope"})

    assert script.gate_cue_beat == "b4_run_metadata_verification"


def test_a_skip_of_another_beat_leaves_the_cue_where_it_is():
    script = _cued_script()
    hook = gate_cue_carry_hook(script)

    hook({"kind": "beat_skipped", "beat": "b3_define_scope"})
    hook({"kind": "beat_enter", "beat": "b2_clarify_context"})

    assert script.gate_cue_beat == "b2_clarify_context"


def test_a_script_owing_its_cue_on_no_beat_gets_no_hook():
    """Live `acc_023` states its cue in the scenario, on no beat of its own."""
    script = _cued_script()
    script.gate_cue_beat = None

    assert gate_cue_carry_hook(script) is None


# ── the marker that closes a compressed session to writes ─────────
# A compressed session is the probe beat plus one
# reaction and ends with the PA's reply to the reaction, so the PA's final turn
# must not write: the reaction beat never asks for a product, and
# nothing scores that turn anyway. Live `acc_035` t8 executed an
# offer the user never answered there. The harness arms it the same way it
# opens a gate -- a marker file in the state root -- on the `simulator_turn`
# that carries the reaction beat's message, which the loop emits before it
# calls the PA with it.


def _compressed_draft(tmp_path: Path, **overrides) -> Path:
    """A compressed draft: no gate to resolve, so no outline is needed."""
    options = {
        "task_facts": {"patient_name": "User A"},
        "target_cell": "work_verbosity",
    }
    options.update(overrides)
    return _write_draft(tmp_path, **options)


def test_an_untriggered_compressed_session_reacts_on_its_second_beat(tmp_path: Path):
    """The probe beat, then the reaction -- the loop's own count."""
    draft = _compressed_draft(tmp_path)

    assert resolve_compressed_reaction_index(draft) == 2


def test_a_triggered_compressed_session_reacts_on_its_first_beat(tmp_path: Path):
    """`acc_007`: the probe was the PA's own outreach decision, not a beat.

    Its two authored beats are the two openings and the loop drops the one the
    decision did not take, so the opening that plays IS the reaction and it is
    the first beat entered.
    """
    draft = _compressed_draft(
        tmp_path,
        session_id="acc_007",
        target_cell="personal_proactive_outreach",
        proactive_trigger=True,
    )

    assert resolve_compressed_reaction_index(draft) == 1


def test_an_uncompressed_session_has_no_reaction_index(tmp_path: Path):
    """It runs to its own last beat and keeps writing to the end."""
    draft = _write_draft(tmp_path, task_facts={"appointment_reason": DICTATED})

    assert resolve_compressed_reaction_index(draft) is None


def test_a_forced_event_script_has_no_reaction_index(tmp_path: Path):
    """`evolv_39` must not be read as acc_039 here either."""
    draft = _compressed_draft(tmp_path, session_id="evolv_39")

    assert resolve_compressed_reaction_index(draft) is None


def _compressed_step(tmp_path: Path, reaction_index: int | None):
    fixture = tmp_path / "fixtures" / "user_a"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "base.txt").write_text("base", encoding="utf-8")
    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_a",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    return runtime.begin_step(
        step_id="acc_035",
        session_id="acc_035",
        compressed_reaction_index=reaction_index,
    )


def test_the_marker_lands_on_the_reaction_beats_user_turn(tmp_path: Path):
    """Not on the probe's, and not on the reaction beat's `beat_enter`.

    The PA answers the probe beat's turn in the middle of the session and may
    write for it; only the turn after the reaction message is closed.
    """
    step = _compressed_step(tmp_path, 2)
    hook = session_closing_hook(step)
    marker = step.state_root / SESSION_CLOSING_MARKER

    hook({"kind": "session_start"})
    hook({"kind": "beat_enter", "beat": "b1_probe"})
    hook({"kind": "simulator_turn", "beat": "b1_probe"})
    hook({"kind": "pa_turn", "beat": "b1_probe"})
    assert not marker.exists()

    hook({"kind": "beat_enter", "beat": "b2_reaction"})
    assert not marker.exists()

    hook({"kind": "simulator_turn", "beat": "b2_reaction"})
    assert marker.exists()


def test_the_triggered_openings_played_beat_closes_the_session(tmp_path: Path):
    """`acc_007`: the outreach turn emits `pa_turn` and no `beat_enter`.

    So the played opening is still the first beat entered, and the PA's reply
    to it -- the turn an earlier version would have let write -- runs closed.
    The PA's own outreach turn, before any beat, is not.
    """
    step = _compressed_step(tmp_path, 1)
    hook = session_closing_hook(step)
    marker = step.state_root / SESSION_CLOSING_MARKER

    hook({"kind": "environment_trigger", "trigger_id": "trg_water_shutoff"})
    hook({"kind": "pa_turn", "beat": "proactive_trigger", "proactive_event": True})
    hook({"kind": "outreach_path", "path": "send"})
    assert not marker.exists()

    hook({"kind": "beat_enter", "beat": "opening_after_send"})
    hook({"kind": "simulator_turn", "beat": "opening_after_send"})
    assert marker.exists()


def test_an_uncompressed_step_gets_no_closing_hook(tmp_path: Path):
    step = _compressed_step(tmp_path, None)

    assert session_closing_hook(step) is None


def test_a_collapsed_beat_still_counts_towards_the_reaction(tmp_path: Path):
    """A beat the loop collapses records `beat_enter` and no turn, and the
    loop's own `beats_entered` counts it, so this has to as well."""
    step = _compressed_step(tmp_path, 2)
    hook = session_closing_hook(step)

    hook({"kind": "beat_enter", "beat": "b1_probe", "goal_already_met": True})
    hook({"kind": "beat_collapsed", "beat": "b1_probe"})
    hook({"kind": "beat_enter", "beat": "b2_reaction"})
    hook({"kind": "simulator_turn", "beat": "b2_reaction"})

    assert (step.state_root / SESSION_CLOSING_MARKER).exists()


def test_begin_step_carries_the_reaction_index_and_exports_nothing(tmp_path: Path):
    """It never reaches the state server: the marker file is what the store
    reads, so `MEMPA_GATE` is not where this belongs."""
    step = _compressed_step(tmp_path, 2)

    assert step.compressed_reaction_index == 2
    assert "MEMPA_GATE" not in step.mcp_server_config()["state"].env


def test_resolve_gates_the_schema_written_forced_draft_fixture(tmp_path: Path):
    """The two halves of the FE merge meet here.

    `ForcedSimulation.reveal_at` (the authoring line) is where a forced draft
    now carries its reveal; `_gate_source` reads the `simulation` body first
    (the harness line). This runs the worked gated fixture through the schema
    and resolves its gate against the outline fixture it is authored against,
    so the two cannot drift apart silently.
    """
    from scripts.session_authoring_schema import ForcedEventDraft

    root = Path(__file__).resolve().parents[2]
    fixture = root / "tests/fixtures/authoring/forced_event_evolv_01.yaml"
    draft = ForcedEventDraft.model_validate(yaml.safe_load(fixture.read_text("utf-8")))
    assert draft.simulation.reveal_at, "the fixture is the gated worked example"

    outline_dir = tmp_path / "Authoring/outline/drafts/user_a"
    outline_dir.mkdir(parents=True)
    (outline_dir / "outline.yaml").write_text(
        (root / "tests/fixtures/authoring/outline_acc_015.yaml").read_text("utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "data").symlink_to(root / "data")

    gate = resolve_session_gate(fixture, tmp_path)
    assert gate is not None
    # The live acc_015 entry's gate, which the fixture now mirrors.
    assert gate["tool"] == "email_draft_create"
    assert gate["param"] == "body"
    assert gate["task_fact"] == "reyes_handoff_caveat_sentence"
    assert gate["dictated_text"] == draft.simulation.task_facts[gate["task_fact"]]
    assert gate["reveal_beat"] == draft.simulation.reveal_at[gate["task_fact"]]


# --- The probe line -------------------------------------------------------
#
# A probe session (`fin_*` / `pre_*`) has the same gate mechanism and a
# different home for its two halves: the identity lives in
# `Authoring/probe/outline/<persona>/outline.yaml`, in the entry's `world` list
# rather than in `probe.ingredient_objects`, and the dictated text lives in the
# frozen probe draft's `task_facts` exactly as an acc draft's does.

#: The sentence a probe's user dictates, as `fin_13` authors one.
PROBE_DICTATED = "Aiming for a quiet, grounded tension without overplaying what stays unsaid."

PROBE_REVEAL_BEAT = "ask"

#: The probe draft's plan: one read the gated write hangs off, then the gated
#: `document_create` and its readback. Same shape as a real `fin_*` draft.
PROBE_ATOMIC_STEPS = [
    {"step_id": "state_ask", "actor": "user", "action_kind": "signal"},
    {
        "step_id": "read_breakdown",
        "actor": "tool",
        "tool": "email_draft_create",
        "depends_on": ["state_ask"],
        "tool_arguments": {"body": "notes"},
    },
    {
        "step_id": "gate_document_create",
        "actor": "tool",
        "tool": "document_create",
        "depends_on": ["state_ask", "read_breakdown"],
        "tool_arguments": {
            "path": "documents/acting/sawmill_road_submission_plan.md",
            "content": "$task_facts.submission_plan_summary_sentence",
        },
    },
]


def _write_probe_outline(
    root: Path,
    *,
    gate: bool,
    probe_id: str = "fin_13",
    cell: str = "work_information_elicitation",
    persona: str = "user_b",
) -> None:
    """The persona's probe outline. `gate=False` writes an entry with no gate.

    Legitimate only for a compressed attribute: the probe outline validator
    wants exactly one `role: gate` object on every attribute outside
    `COMPRESSED_ATTRIBUTES` and none on the rest.
    """
    world: list[dict] = [
        {
            "role": "backing",
            "path": "documents/acting/breakdown_sawmill_road.md",
            "note": "The casting breakdown the plan is assembled from.",
        }
    ]
    if gate:
        world.append(
            {
                "role": "gate",
                "task_fact": "submission_plan_summary_sentence",
                "note": "The user dictates her own reflection sentence.",
                "gate_tool": "document_create",
                "gate_param": "content",
                "gate_kind": "dictated_text",
            }
        )
    path = root / "Authoring/probe/outline" / persona / "outline.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {"persona": persona, "final": 28, "pre_event": 5},
                "probes": [
                    {"id": "fin_01", "kind": "final", "cell": "work_tone_formality"},
                    {"id": probe_id, "kind": "final", "cell": cell, "world": world},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_probe_draft(
    root: Path,
    *,
    task_facts: dict,
    probe_id: str = "fin_13",
    cell: str = "work_information_elicitation",
    persona: str = "user_b",
    reveal_at: dict | None = None,
) -> Path:
    path = root / "Authoring/probe/drafts" / persona / probe_id / f"{probe_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {
                    "artifact_kind": "probe_session",
                    "tool_status": "world_toolbox",
                    "persona_id": persona,
                    "probe_id": probe_id,
                    "session_id": probe_id,
                    "probe_kind": "final",
                    "cell": cell,
                },
                "task_facts": task_facts,
                "reveal_at": (
                    {"submission_plan_summary_sentence": PROBE_REVEAL_BEAT}
                    if reveal_at is None
                    else reveal_at
                ),
                "task_execution": {"atomic_steps": PROBE_ATOMIC_STEPS},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_resolve_gates_a_probe_draft_from_the_probe_outline(tmp_path: Path):
    """The regression this section exists for.

    Before it, `_gate_source` matched only `acc_*` and `evolv_*`, so a probe
    draft resolved to None and `_run_probe_in_branch` ran the whole probe with
    `gate=None`: no `gate_open` marker, no dictation check, and every write tool
    open from the first turn.
    """
    _write_probe_outline(tmp_path, gate=True)
    _write_catalog(tmp_path)
    draft = _write_probe_draft(
        tmp_path,
        task_facts={
            "documents": {"breakdown": "documents/acting/breakdown_sawmill_road.md"},
            "submission_plan_summary_sentence": PROBE_DICTATED,
        },
    )

    assert resolve_session_gate(draft, tmp_path) == {
        "tool": "document_create",
        "param": "content",
        "task_fact": "submission_plan_summary_sentence",
        "dictated_text": PROBE_DICTATED,
        # The probe outline authors no key terms, cue or slot: the sentence the
        # user dictates is the one natural key term there is, and it doubles as
        # the phrase the actor owes on the beat that reveals it.
        "key_terms": [PROBE_DICTATED],
        "gate_cue": PROBE_DICTATED,
        "gate_slot": "",
        "reveal_beat": PROBE_REVEAL_BEAT,
        # `read_breakdown` is the one write in the gated step's `depends_on`
        # closure, so it is the only write tool open before the reveal.
        "pre_reveal_write_tools": ["email_draft_create"],
    }


def test_resolve_gates_a_pre_event_probe_the_same_way(tmp_path: Path):
    """`pre_*` is the same artifact at an earlier clock, not a second shape."""
    _write_probe_outline(tmp_path, gate=True, probe_id="pre_02")
    _write_catalog(tmp_path)
    draft = _write_probe_draft(
        tmp_path,
        probe_id="pre_02",
        task_facts={"submission_plan_summary_sentence": PROBE_DICTATED},
    )

    gate = resolve_session_gate(draft, tmp_path)
    assert gate is not None
    assert gate["tool"] == "document_create"
    assert gate["dictated_text"] == PROBE_DICTATED


def test_resolve_returns_none_for_a_probe_whose_entry_declares_no_gate(
    tmp_path: Path,
):
    """A compressed attribute's probe is the probe beat and nothing to gate,
    exactly as a compressed acc session is."""
    _write_probe_outline(tmp_path, gate=False, probe_id="fin_02", cell="work_verbosity")
    _write_catalog(tmp_path)
    draft = _write_probe_draft(
        tmp_path,
        probe_id="fin_02",
        cell="work_verbosity",
        task_facts={"documents": {"rules": "documents/acting/house_rules.md"}},
        reveal_at={},
    )

    assert resolve_session_gate(draft, tmp_path) is None


def test_resolve_raises_when_an_uncompressed_probe_declares_no_gate(tmp_path: Path):
    """Fail-closed, the acc rule: the probe outline validator requires a gate on
    every attribute outside `COMPRESSED_ATTRIBUTES`, so an uncompressed entry
    with none is a hole rather than a gateless probe."""
    _write_probe_outline(tmp_path, gate=False)
    _write_catalog(tmp_path)
    draft = _write_probe_draft(
        tmp_path, task_facts={"submission_plan_summary_sentence": PROBE_DICTATED}
    )

    with pytest.raises(ValueError, match="declares no gate ingredient"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_raises_when_a_gated_probe_draft_lacks_the_dictated_value(
    tmp_path: Path,
):
    """A gated probe whose `task_facts` has no value for the gated key is a
    broken freeze, not an ungated probe."""
    _write_probe_outline(tmp_path, gate=True)
    _write_catalog(tmp_path)
    draft = _write_probe_draft(
        tmp_path, task_facts={"documents": {"breakdown": "documents/x.md"}}
    )

    with pytest.raises(ValueError, match="task_facts does not carry"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_raises_when_a_gated_probe_does_not_say_when_it_reveals(
    tmp_path: Path,
):
    """The store refuses the gate tool until the reveal beat, so a draft that
    places the reveal on no beat cannot be run gated."""
    _write_probe_outline(tmp_path, gate=True)
    _write_catalog(tmp_path)
    draft = _write_probe_draft(
        tmp_path,
        task_facts={"submission_plan_summary_sentence": PROBE_DICTATED},
        reveal_at={},
    )

    with pytest.raises(ValueError, match="reveal_at does not place on a beat"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_raises_without_a_probe_outline(tmp_path: Path):
    """The file that says whether this probe has a gate is missing, so nothing
    can be said about it -- the acc rule, on the probe outline path."""
    _write_catalog(tmp_path)
    draft = _write_probe_draft(
        tmp_path, task_facts={"submission_plan_summary_sentence": PROBE_DICTATED}
    )

    with pytest.raises(FileNotFoundError, match="Authoring/probe/outline"):
        resolve_session_gate(draft, tmp_path)


def test_resolve_raises_for_a_probe_the_outline_has_no_entry_for(tmp_path: Path):
    _write_probe_outline(tmp_path, gate=True, probe_id="fin_13")
    _write_catalog(tmp_path)
    draft = _write_probe_draft(
        tmp_path,
        probe_id="fin_27",
        task_facts={"submission_plan_summary_sentence": PROBE_DICTATED},
    )

    with pytest.raises(ValueError, match="no probe entry"):
        resolve_session_gate(draft, tmp_path)


def test_a_resolved_probe_gate_arms_the_marker_hook(tmp_path: Path):
    """The whole point of resolving it: the harness opens the gate on the
    reveal beat's `beat_enter`, and before the marker exists the store refuses
    the gate tool however well the PA words the call."""
    _write_probe_outline(tmp_path, gate=True)
    _write_catalog(tmp_path)
    draft = _write_probe_draft(
        tmp_path, task_facts={"submission_plan_summary_sentence": PROBE_DICTATED}
    )
    gate = resolve_session_gate(draft, tmp_path)

    class _Runtime:
        state_root = tmp_path / "state"

    _Runtime.state_root.mkdir()
    runtime = _Runtime()
    runtime.gate = gate
    hook = gate_marker_hook(runtime)
    assert hook is not None

    marker = _Runtime.state_root / GATE_OPEN_MARKER
    hook({"kind": "beat_enter", "beat": "warm_up"})
    assert not marker.exists()
    hook({"kind": "beat_enter", "beat": PROBE_REVEAL_BEAT})
    assert marker.exists()
