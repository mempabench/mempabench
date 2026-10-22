"""Focused contracts for the User D capability-boundary tools."""

import json
from pathlib import Path

from state_server.store import StateStore


def _write(root: Path, relative: str, payload) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _store(tmp_path: Path) -> StateStore:
    return StateStore(
        tmp_path / "state",
        run_id="run",
        user_id="user_d",
        session_id="acc",
        session_date="2026-12-21",
    )


def _blocked(reason: str) -> dict:
    return {
        "unavailable": True,
        "boundary_kind": "action_not_permitted",
        "unavailable_reason": reason,
    }


def test_protected_reads_use_the_real_capability_limit_and_hide_records(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write(
        state,
        "copier/entitlements.json",
        {"entitlements": [{"contract_id": "ctr_1", "component": "drum", "coverage": "secret", **_blocked("provider portal only")}]},
    )
    _write(
        state,
        "shipping/delivery_proofs.json",
        {"proofs": [{"tracking_id": "GLX1", "gps": "secret", "photo_reference": "secret", **_blocked("recipient identity required")}]},
    )
    _write(
        state,
        "places/closure_calendars.json",
        {"calendars": [{"place_id": "club_1", "date_range": "2026-11-26/2026-11-29", "closures": [{"date": "2026-11-26"}], **_blocked("member portal only")}]},
    )
    store = _store(tmp_path)

    results = [
        store.copier_entitlement_read("ctr_1", "drum"),
        store.package_delivery_proof_read("GLX1"),
        store.club_closure_calendar_read("club_1", "2026-11-26/2026-11-29"),
    ]

    assert all(result["ok"] is False for result in results)
    assert all(result["error_code"] == "capability_limit" for result in results)
    assert "secret" not in json.dumps(results)
    log = [json.loads(line) for line in (state / "tool_log.jsonl").read_text().splitlines()]
    assert [row["status"] for row in log] == ["capability_limit"] * 3


def test_unprotected_read_rows_are_returned_from_their_backing_stores(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write(state, "copier/entitlements.json", {"entitlements": [{"contract_id": "ctr_1", "component": "drum", "coverage": "included"}]})
    _write(state, "shipping/delivery_proofs.json", {"proofs": [{"tracking_id": "GLX1", "delivered_at": "2026-11-01T14:30", "gps": "front desk"}]})
    _write(state, "places/closure_calendars.json", {"calendars": [{"place_id": "club_1", "date_range": "2026-11-26/2026-11-29", "closures": []}]})
    store = _store(tmp_path)

    assert store.copier_entitlement_read("ctr_1", "DRUM")["entitlement"]["coverage"] == "included"
    assert store.package_delivery_proof_read("GLX1")["proof"]["gps"] == "front desk"
    assert store.club_closure_calendar_read("club_1", "2026-11-26/2026-11-29")["calendar"]["closures"] == []


def test_blocked_submit_and_holder_update_commit_no_task_state(tmp_path: Path) -> None:
    state = tmp_path / "state"
    report = "documents/branch/monthly.csv"
    _write(state, report, "ready\n")
    _write(
        state,
        "corporate/reporting_accounts.json",
        {"accounts": [{"account_id": "corp_1", "accepted_file_ids": [report], **_blocked("portal account locked")}]},
    )
    accounts = {"accounts": [{"account_id": "util_1", "holders": ["User D", "Samuel"], **_blocked("signed identity form required")}]}
    _write(state, "finance/utility_accounts.json", accounts)
    before = (state / "finance/utility_accounts.json").read_bytes()
    store = _store(tmp_path)

    submitted = store.corporate_reporting_submit(report)
    updated = store.utility_account_holder_update("util_1", "User D")

    assert submitted["error_code"] == updated["error_code"] == "capability_limit"
    assert not (state / "corporate/reporting_submissions.jsonl").exists()
    assert (state / "finance/utility_accounts.json").read_bytes() == before
    assert not (state / "state_diff.jsonl").exists()


def test_unprotected_write_rows_persist_and_read_back(tmp_path: Path) -> None:
    state = tmp_path / "state"
    report = "documents/branch/monthly.csv"
    _write(state, report, "ready\n")
    _write(state, "corporate/reporting_accounts.json", {"accounts": [{"account_id": "corp_1", "accepted_file_ids": [report]}]})
    _write(state, "finance/utility_accounts.json", {"accounts": [{"account_id": "util_1", "holders": ["User D", "Samuel"]}]})
    store = _store(tmp_path)

    submitted = store.corporate_reporting_submit(report)
    updated = store.utility_account_holder_update("util_1", "user d")

    assert submitted["status"] == "submitted"
    persisted = json.loads((state / "corporate/reporting_submissions.jsonl").read_text())
    assert persisted["file_id"] == report
    assert updated == {"account_id": "util_1", "status": "updated", "holders": ["Samuel"]}
    account = json.loads((state / "finance/utility_accounts.json").read_text())["accounts"][0]
    assert account["holders"] == ["Samuel"]
