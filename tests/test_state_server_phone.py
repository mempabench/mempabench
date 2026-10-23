"""The phone domain: placing a call and reading it back.

A `Suggest alternatives` recovery may be "call the
locksmith", and `Find and hand off` may only hand the number over. That needs a
real outbound-call tool, and like every other World Toolbox handler it is a
function over the run-local world: whether the line answers, and what it says
when it does, are fields on the callee's own record -- never a scripted case.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from harness.gate import _catalog_external_tools
from pa.agent.ix.action_types import tool_action_type
from state_server.store import StateStore

CATALOG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "data/state_tools/catalog.yaml").read_text(
        encoding="utf-8"
    )
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTACTS = {
    "contacts": [
        {
            "contact_id": "ct_0001",
            "name": "Talbot Ridge Management",
            "phone": "(555) 026-2200",
            "notes": "Office Monday to Friday, 09:00-17:00.",
        },
        {
            "contact_id": "ct_0002",
            "name": "Talbot Ridge after-hours lockout line",
            "phone": "555-026-9110",
            "unavailable": True,
            "boundary_kind": "external_dependency",
            "unavailable_reason": "The number is held in the resident portal.",
        },
    ]
}

PLACES = {
    "places": [
        {
            "name": "Corvid Ave 24-Hour Lock & Key",
            "near": "Glenmont, CA - Corvid Ave",
            "category": "locksmith",
            "phone": "555 741 0188",
            "phone_answer": (
                "A technician is dispatched to the address given and arrives within "
                "the hour; the residential lockout call-out is $140.00 between 22:00 "
                "and 06:00, plus the work, and photo identification is taken at the door."
            ),
        }
    ]
}


def _store(tmp_path: Path) -> StateStore:
    state = tmp_path / "state"
    for relative, payload in (
        ("contacts.json", CONTACTS),
        ("places/nearby.json", PLACES),
    ):
        path = state / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return StateStore(
        state,
        run_id="run",
        user_id="user_b",
        session_id="acc_093",
        session_date="2026-12-31",
    )


def _tool_log(store: StateStore) -> list[dict[str, Any]]:
    path = store.state_root / "tool_log.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_dialing_a_place_that_registers_an_answer_connects_and_reports_it(tmp_path):
    store = _store(tmp_path)
    result = store.phone_call_dial("555 741 0188", "Send someone out to open the door")
    assert result["status"] == "answered"
    assert result["answered_by"] == "Corvid Ave 24-Hour Lock & Key"
    assert result["answer"] == PLACES["places"][0]["phone_answer"]
    assert result["call_id"].startswith("call_")


def test_public_lookups_do_not_disclose_a_future_call_answer(tmp_path):
    store = _store(tmp_path)
    contact_path = store.state_root / "contacts.json"
    contacts = json.loads(contact_path.read_text())
    contacts["contacts"][0]["phone_answer"] = "Manual review opened as CASE-17."
    contact_path.write_text(json.dumps(contacts))

    public = [
        store.lookup_contacts("Talbot Ridge Management"),
        store.contact_read("ct_0001"),
        store.place_search(location="Corvid Ave", category="locksmith"),
    ]
    assert "phone_answer" not in json.dumps(public)
    assert "CASE-17" not in json.dumps(public)
    assert "technician is dispatched" not in json.dumps(public)
    assert public[1]["contact"]["phone"] == "(555) 026-2200"

    result = store.phone_call_dial("(555) 026-2200", "Open a manual review")
    assert result["answer"] == "Manual review opened as CASE-17."
    assert store.phone_call_read(result["call_id"])["call"]["answer"] == result["answer"]


def test_dialing_a_contact_without_a_registered_answer_still_connects(tmp_path):
    """The world says the line answers; it has not authored what is said."""
    store = _store(tmp_path)
    result = store.phone_call_dial("(555) 026-2200", "Ask about the front door")
    assert result["status"] == "answered"
    assert result["answered_by"] == "Talbot Ridge Management"
    assert "answer" not in result


def test_number_matching_ignores_punctuation_and_spacing(tmp_path):
    store = _store(tmp_path)
    result = store.phone_call_dial("5550262200", "Ask about the front door")
    assert result["answered_by"] == "Talbot Ridge Management"


def test_dialing_a_record_the_world_marks_unavailable_is_a_capability_limit(tmp_path):
    store = _store(tmp_path)
    result = store.phone_call_dial("555-026-9110", "Ask to be let back in")
    assert result == {
        "ok": False,
        "error_code": "capability_limit",
        "boundary_kind": "external_dependency",
        "detail": "The number is held in the resident portal.",
    }
    assert [row["status"] for row in _tool_log(store)] == ["capability_limit"]
    # A call that never connected leaves no call record.
    assert not (store.state_root / "phone" / "calls.jsonl").exists()


def test_dialing_a_number_no_record_carries_is_not_found(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.phone_call_dial("555-000-0000", "Ask about the door")
    assert [row["status"] for row in _tool_log(store)] == ["error"]


def test_a_placed_call_persists_and_reads_back(tmp_path):
    store = _store(tmp_path)
    placed = store.phone_call_dial("555 741 0188", "Send someone out to open the door")
    readback = _store(tmp_path).phone_call_read(placed["call_id"])
    call = readback["call"]
    assert call["call_id"] == placed["call_id"]
    assert call["number"] == "555 741 0188"
    assert call["purpose"] == "Send someone out to open the door"
    assert call["answered_by"] == "Corvid Ave 24-Hour Lock & Key"
    assert call["answer"] == PLACES["places"][0]["phone_answer"]
    assert call["date"] == "2026-12-31"


def test_reading_a_call_that_was_never_placed_is_not_found(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.phone_call_read("call_0001")


def test_a_second_call_gets_its_own_id_and_both_survive(tmp_path):
    store = _store(tmp_path)
    first = store.phone_call_dial("555 741 0188", "Send someone out")
    second = _store(tmp_path).phone_call_dial("(555) 026-2200", "Ask about the door")
    assert first["call_id"] != second["call_id"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "state/phone/calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert [row["call_id"] for row in rows] == [first["call_id"], second["call_id"]]


def test_the_fixture_control_keys_never_reach_the_caller(tmp_path):
    store = _store(tmp_path)
    result = store.phone_call_dial("555 741 0188", "Send someone out")
    assert "unavailable" not in result and "phone_answer" not in result


# The gate: `Find and hand off` must be able to hold this call back.

def test_the_catalog_calls_it_an_external_write():
    spec = CATALOG["tools"]["phone_call_dial"]
    assert spec["domain"] == "phone"
    assert spec["allowed_effects"] == ["external_write"]
    assert spec["backing_stores"] == ["phone/calls.jsonl", "contacts.json", "places/nearby.json"]
    assert CATALOG["tools"]["phone_call_read"]["allowed_effects"] == ["read", "verify"]


def test_the_wrapper_gates_the_call_as_an_external_action():
    assert tool_action_type("phone_call_dial") == "external_action"
    assert tool_action_type("mcp_state_phone_call_dial") == "external_action"
    assert tool_action_type("phone_call_read") == "read"


def test_the_harness_counts_the_call_as_leaving_the_sandbox():
    external = _catalog_external_tools(PROJECT_ROOT)
    assert "phone_call_dial" in external
    assert "phone_call_read" not in external
