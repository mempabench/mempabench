import hashlib
import json
import shutil
from pathlib import Path

import pytest

from pa.agent.ix.action_types import tool_action_type
from harness.gate import resolve_authorised_tool
from harness.state_runtime import StageRuntime
from state_server.store import StateStore


ROOT = Path(__file__).resolve().parents[1]


def _store(tmp_path: Path) -> StateStore:
    state = tmp_path / "state"
    shutil.copytree(ROOT / "data/state_fixtures/user_c", state)
    return StateStore(
        state,
        run_id="test",
        user_id="user_c",
        session_id="acc_102",
        session_date="2027-01-21",
    )


def _create(store: StateStore, **overrides: str) -> dict:
    arguments = {
        "service_id": "mts_rapidframe_realtime",
        "source_path": "media_transcode/source_media/branch_video_hvx2_proxy.mxf",
        "target_format": "mp4/H.264",
        "deliver_by": "2027-01-22",
        "payment_method_id": "pm_visa_5501",
    }
    arguments.update(overrides)
    return store.media_transcode_job_create(**arguments)


def test_search_is_read_only_and_returns_the_published_fee(tmp_path: Path):
    store = _store(tmp_path)
    jobs = store.state_root / "media_transcode/jobs.jsonl"
    before = jobs.read_bytes()

    result = store.media_transcode_service_search(
        "HVX-2 long-GOP 10-bit 4:2:2", "mp4/H.264", "2027-01-22"
    )

    assert result["matches"] == [
        {
            "service_id": "mts_rapidframe_realtime",
            "provider": "RapidFrame Professional Media API",
            "fee": 89.0,
            "currency": "USD",
            "promised_delivery_at": "2027-01-21T09:15:00",
            "source_manifest_path": "documents/zz_service_manifests/branch_video_source.json",
            "required_source_locator": (
                "media_transcode/source_media/branch_video_hvx2_proxy.mxf"
            ),
            "delivery_terms": (
                "Real-time API delivery after the commission is accepted and the "
                "quoted fee is captured."
            ),
        }
    ]
    assert jobs.read_bytes() == before


def test_search_resolves_mp4_to_the_service_canonical_h264_target(tmp_path: Path):
    result = _store(tmp_path).media_transcode_service_search(
        "HVX-2 long-GOP 10-bit 4:2:2", "mp4", "2027-01-22"
    )

    assert result["target_format"] == "mp4/H.264"
    assert result["matches"][0]["service_id"] == "mts_rapidframe_realtime"


def test_create_atomically_persists_charge_and_verified_artifact(tmp_path: Path):
    store = _store(tmp_path)
    result = _create(store)

    assert result["job_id"] == "mtj_0001"
    assert result["status"] == "delivered"
    assert result["charge"] == {
        "charge_id": "mtc_0001",
        "status": "captured",
        "amount": 89.0,
        "currency": "USD",
        "payment_method_id": "pm_visa_5501",
        "charged_at": "2027-01-21T12:00:00+00:00",
    }
    output = store.state_root / result["artifact"]["path"]
    assert result["artifact"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["artifact"]["size_bytes"] == output.stat().st_size
    assert result["artifact"]["container"] == "mp4"
    assert result["artifact"]["video_codec"] == "h264"
    persisted = json.loads(
        (store.state_root / "media_transcode/jobs.jsonl").read_text().strip()
    )
    assert persisted["charge"] == result["charge"]
    assert persisted["artifact"] == result["artifact"]


def test_job_read_is_idempotent_and_does_not_change_job_or_artifact(tmp_path: Path):
    store = _store(tmp_path)
    created = _create(store)
    jobs = store.state_root / "media_transcode/jobs.jsonl"
    output = store.state_root / created["artifact"]["path"]
    jobs_before = jobs.read_bytes()
    output_before = output.read_bytes()

    first = store.media_transcode_job_read(created["job_id"])
    second = store.media_transcode_job_read(created["job_id"])

    assert first == second
    assert first["job"]["charge"] == created["charge"]
    assert first["job"]["artifact"] == created["artifact"]
    assert jobs.read_bytes() == jobs_before
    assert output.read_bytes() == output_before


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"service_id": "mts_missing"}, "unknown transcode service"),
        ({"source_path": "media_transcode/source_media/missing.mxf"}, "missing.mxf"),
        ({"payment_method_id": "pm_missing"}, "unknown payment method"),
        ({"deliver_by": "2027-01-20"}, "cannot deliver"),
        ({"target_format": "mov/ProRes"}, "does not deliver"),
    ],
)
def test_invalid_create_leaves_no_job_or_charge(
    tmp_path: Path, overrides: dict[str, str], error: str
):
    store = _store(tmp_path)
    jobs = store.state_root / "media_transcode/jobs.jsonl"
    before = jobs.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError), match=error):
        _create(store, **overrides)
    assert jobs.read_bytes() == before


def test_mismatched_artifact_hash_leaves_no_job_or_charge(tmp_path: Path):
    store = _store(tmp_path)
    output = store.state_root / "media_transcode/deliveries/branch_video_awards.mp4"
    output.write_bytes(output.read_bytes() + b"tampered")
    jobs = store.state_root / "media_transcode/jobs.jsonl"
    with pytest.raises(ValueError, match="SHA-256"):
        _create(store)
    assert jobs.read_bytes() == b""


def test_container_check_parses_boxes_instead_of_searching_for_strings(tmp_path: Path):
    store = _store(tmp_path)
    output = store.state_root / "media_transcode/deliveries/branch_video_awards.mp4"
    fake = b"not-a-box-ftyp-and-avc1"
    output.write_bytes(fake)
    artifacts_path = store.state_root / "media_transcode/artifacts.json"
    artifacts = json.loads(artifacts_path.read_text())
    artifacts["artifacts"][0]["sha256"] = hashlib.sha256(fake).hexdigest()
    artifacts["artifacts"][0]["size_bytes"] = len(fake)
    artifacts_path.write_text(json.dumps(artifacts))

    with pytest.raises(ValueError, match="MP4"):
        _create(store)
    assert (store.state_root / "media_transcode/jobs.jsonl").read_bytes() == b""


def test_unknown_job_is_rejected_and_create_uses_authorization_floor(tmp_path: Path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="unknown transcode job"):
        store.media_transcode_job_read("mtj_9999")
    assert tool_action_type("media_transcode_job_create") == "external_action"
    assert tool_action_type("mcp_state_media_transcode_job_create") == "external_action"
    assert resolve_authorised_tool(
        ROOT / "Authoring/sessions/normal/drafts/user_c/acc_102.yaml", ROOT
    ) == "media_transcode_job_create"


def test_stage_runtime_clones_media_and_commits_job_for_next_step(tmp_path: Path):
    runtime = StageRuntime(
        run_id="transcode-ready",
        user_id="user_c",
        fixture_root=ROOT / "data/state_fixtures",
        runs_root=tmp_path / "runs",
        project_root=ROOT,
    )
    create_step = runtime.begin_step(
        step_id="acc_102",
        session_id="acc_102",
        runtime_artifact_path=ROOT / "Authoring/sessions/normal/drafts/user_c/acc_102.yaml",
        session_date="2027-01-21",
    )
    assert {
        "media_transcode_service_search",
        "media_transcode_job_create",
        "media_transcode_job_read",
    } <= set(create_step.enabled_tools)
    store = StateStore(
        create_step.state_root,
        run_id="transcode-ready",
        user_id="user_c",
        session_id="acc_102",
        session_date="2027-01-21",
    )
    created = _create(store)
    delivered = create_step.state_root / created["artifact"]["path"]
    assert delivered.is_file()
    assert hashlib.sha256(delivered.read_bytes()).hexdigest() == created["artifact"]["sha256"]
    runtime.finish_step(create_step, "commit")

    read_step = runtime.begin_step(
        step_id="acc_103",
        session_id="acc_103",
        session_date="2027-01-26",
    )
    read_store = StateStore(
        read_step.state_root,
        run_id="transcode-ready",
        user_id="user_c",
        session_id="acc_103",
        session_date="2027-01-26",
    )
    readback = read_store.media_transcode_job_read(created["job_id"])
    assert readback["job"]["charge"]["payment_method_id"] == "pm_visa_5501"
    assert (read_step.state_root / readback["job"]["artifact"]["path"]).is_file()
