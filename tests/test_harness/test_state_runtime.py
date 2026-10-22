import inspect
import hashlib
import json
import sys
from pathlib import Path

import pytest

from harness.state_runtime import (
    StageRuntime,
    StageStepRuntime,
    StateRuntime,
    initialize_state_runtime,
    preserve_step_state_artifacts,
)
from state_server.store import StateStore
from state_server.server import _PublicStore


from state_server.store import StateStore
from state_server.server import _PublicStore


def _step_runtime(tmp_path: Path, **clock) -> StageStepRuntime:
    return StageStepRuntime(
        run_id="run",
        user_id="user_b",
        session_id="acc_014",
        step_id="acc_014",
        state_root=tmp_path,
        project_root=tmp_path,
        enabled_tools=["document_read"],
        **clock,
    )


def test_the_session_timestamp_reaches_the_state_server(tmp_path: Path) -> None:
    """Hour-granular `visible_from` needs the session's hour, and the day stays
    its own variable: `MEMPA_SESSION_DATE` is the store's clock (`_now`, every
    authored `date` default) and widening it would change every stamp a run
    writes."""
    env = _step_runtime(
        tmp_path,
        session_date="2026-10-05",
        session_timestamp="2026-10-05T14:00:00+00:00",
    ).mcp_server_config()["state"].env

    assert env["MEMPA_SESSION_DATE"] == "2026-10-05"
    assert env["MEMPA_SESSION_TIMESTAMP"] == "2026-10-05T14:00:00+00:00"


def test_a_step_with_no_timestamp_sets_no_timestamp_variable(tmp_path: Path) -> None:
    env = _step_runtime(tmp_path, session_date="2026-10-05").mcp_server_config()[
        "state"
    ].env

    assert env["MEMPA_SESSION_DATE"] == "2026-10-05"
    assert "MEMPA_SESSION_TIMESTAMP" not in env


def test_the_static_state_runtime_carries_the_timestamp_too(tmp_path: Path) -> None:
    env = StateRuntime(
        run_id="run",
        user_id="user_b",
        session_id="acc_014",
        state_root=tmp_path,
        project_root=tmp_path,
        enabled_tools=["document_read"],
        session_date="2026-10-05",
        session_timestamp="2026-10-05T14:00:00+00:00",
    ).mcp_server_config()["state"].env

    assert env["MEMPA_SESSION_TIMESTAMP"] == "2026-10-05T14:00:00+00:00"


@pytest.mark.parametrize(
    ("persona", "session_id", "day", "timestamp", "to", "subject", "body"),
    [
        (
            "user_a",
            "fin_12",
            "2027-01-31",
            "2027-01-31T10:00:00+00:00",
            "alerts-admin@scholarnet.example",
            "Change to existing ScholarNet literature alert",
            "Please change my existing weekly ScholarNet literature alert so that it "
            "returns published articles only on decoherence and non-Abelian anyons. "
            "Please remove preprints from its scope.",
        ),
        (
            "user_c",
            "fin_26",
            "2027-02-01",
            "2027-02-01T11:00:00+00:00",
            "care@northgateretail.example",
            "Address correction for order NR-58120",
            "My order NR-58120, the kitchen tap aerator set, is held at the Glenmont "
            "depot pending an address correction that the carrier will only take from "
            "the sender. Could you submit the correction so the consignment can be released?",
        ),
    ],
)
def test_probe_stage_public_store_reads_back_pre_noon_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persona: str,
    session_id: str,
    day: str,
    timestamp: str,
    to: str,
    subject: str,
    body: str,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    stage = StageRuntime(
        run_id="clock_test",
        user_id=persona,
        fixture_root=project_root / "data/state_fixtures",
        project_root=project_root,
        run_dir=tmp_path / persona,
    )
    step = stage.begin_step(
        step_id=session_id,
        session_id=session_id,
        enabled_tools=["email_message_send", "email_sent_read"],
        session_date=day,
        session_timestamp=timestamp,
    )
    env = step.mcp_server_config()["state"].env
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    store = _PublicStore(StateStore.from_env())

    sent = store.send_email(to, subject, body, [])
    assert sent == {"sent_id": "sent_0001", "status": "sent"}
    assert store.read_sent_email(sent["sent_id"]) == {
        "sent_id": "sent_0001",
        "to": to,
        "subject": subject,
        "body": body,
        "attachments": [],
        "sent_at": timestamp,
    }


def test_state_runtime_freezes_no_root_in_its_signature() -> None:
    """Neither signature may carry a root of its own.

    The default used to be the relative `Path("workspace/runs")`, which made a
    run's state land under whatever directory the process was standing in; a
    resolved constant would be no better, because it would bind the configured
    root at import time, before anything can set it."""
    for owner in (initialize_state_runtime, StageRuntime):
        assert inspect.signature(owner).parameters["runs_root"].default is None


def test_state_runtime_defaults_stay_under_the_configured_runs_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    monkeypatch.setenv("MEMPA_RUNS_ROOT", str(tmp_path / "moved"))

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="acc_001",
        run_id="run_x",
        fixture_root=fixture.parent,
        project_root=tmp_path,
    )
    stage = StageRuntime(
        run_id="run_x", user_id="user_a", fixture_root=fixture.parent
    )

    assert runtime.state_root == tmp_path / "moved" / "run_x" / "state"
    assert stage.run_dir == tmp_path / "moved" / "run_x"


def test_an_explicit_runs_root_is_the_run_s_parent_and_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`runs_root` here is the run's *direct parent*, not the global root: a
    formal launch passes `run_dir.parent`, which is four levels below the root
    (`scripts/run_orchestrator.py`). Read as a global root instead, every run
    of every model and persona would be flattened into one directory."""
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    monkeypatch.setenv("MEMPA_RUNS_ROOT", str(tmp_path / "moved"))
    parent = tmp_path / "moved" / "model" / "cond" / "user_a"

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="acc_001",
        run_id="run_x",
        fixture_root=fixture.parent,
        runs_root=parent,
        project_root=tmp_path,
    )
    stage = StageRuntime(
        run_id="run_x",
        user_id="user_a",
        fixture_root=fixture.parent,
        runs_root=parent,
    )
    branch = StageRuntime(
        run_id="run_x",
        user_id="user_a",
        fixture_root=fixture.parent,
        runs_root=parent,
        run_dir=tmp_path / "branches" / "01",
    )

    assert runtime.state_root == parent / "run_x" / "state"
    assert stage.run_dir == parent / "run_x"
    # `run_dir` still overrides both: a probe branch is a run directory of its
    # own and is never derived from a root plus a run id.
    assert branch.run_dir == tmp_path / "branches" / "01"


def test_initialize_state_runtime_copies_fixture(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "documents" / "doc.md").write_text("hello", encoding="utf-8")
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_01",
        run_id="run_001",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )

    assert runtime.state_root == (tmp_path / "runs" / "run_001" / "state").resolve()
    assert (runtime.state_root / "documents" / "doc.md").read_text(encoding="utf-8") == "hello"

    cfg = runtime.mcp_server_config()["state"]
    assert cfg.command == sys.executable
    assert cfg.args == ["-m", "state_server.server"]
    assert cfg.env["MEMPA_STATE_ROOT"] == str(runtime.state_root)
    assert cfg.env["PYTHONPATH"] == str(tmp_path.resolve())
    assert cfg.enabled_tools == ["document_read", "email_draft_create"]


def test_session_04_exposes_route_planning_tools(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_04",
        run_id="run_004",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "email_message_search", "email_message_read", "map_route_search"],
    )

    assert runtime.enabled_tools == [
        "document_read",
        "email_draft_create",
        "email_message_search",
        "email_message_read",
        "map_route_search",
    ]



TOOLS_RESOLVED_DRAFT_YAML = """\
meta:
  artifact_kind: normal_session
  tool_status: world_toolbox
  persona_id: user_a
  session_id: acc_001
  session_type: stable
  target_cell: work_verbosity
  target_setting: Detailed
  contract_class: output_realization
assistance_emergence:
  pattern: explicit_request
  rationale: The user opens with a direct request for a source-grounded summary.
  evidence_beat_ids:
  - ex_open_request
task_execution:
  atomic_steps:
  - step_id: open_request
    source_outline_phase_id: identify_source
    purpose: User asks for a usable summary of the anyons review paper.
    actor: user
    action_kind: signal
    required_observation: The user names the paper and the draft it supports.
  - step_id: read_review_paper
    source_outline_phase_id: extract_content
    purpose: Read the candidate review paper before summarizing it.
    actor: tool
    action_kind: read
    depends_on:
    - open_request
    tool: document_read
    tool_arguments:
      path: documents/research/anyons_review_paper.txt
    required_observation: The review paper content is available for summarization.
  - step_id: deliver_summary
    source_outline_phase_id: deliver_summary
    purpose: Report the summary grounded in the paper content.
    actor: pa
    action_kind: report
    depends_on:
    - read_review_paper
    required_observation: The user receives a summary tied to the paper content.
scenario: User A is drafting his manuscript and wants a summary of a candidate
  review paper on anyons before deciding how to use it.
task: Summarize the available anyons review paper for the manuscript draft.
life_context:
  what_happened: The drafting session reached the literature framing decision.
  mood: Focused and exacting.
  time_pressure: He wants the summary within the current work block.
  energy: Mentally engaged.
  continuity_note: Early in the research-writing arc.
task_facts:
  user_knows:
  - The review paper is available to the PA.
beats:
- beat: ex_open_request
  goal: Get a summary of the review paper.
  user_instruction: Ask directly for a summary of the anyons review paper for
    the current draft.
  entry_condition: The literature decision needs the summary.
  avoid: Do not mention preference labels or evaluation.
- beat: ex_confirm_scope
  goal: Make sure the summary serves the draft decision.
  user_instruction: Confirm the summary should serve the draft decision.
  entry_condition: The PA has restated the request.
  avoid: Do not restate the whole task.
- beat: ex_wait_for_read
  goal: Let the paper actually get read.
  user_instruction: Respond briefly while the PA checks the paper.
  entry_condition: The PA is reading the paper.
  avoid: Do not invent paper facts.
- beat: ex_scope_reaction
  goal: React to what the paper covers.
  user_instruction: React briefly to the reported scope of the paper.
  entry_condition: The PA has described the paper scope.
  avoid: Do not drift into unrelated topics.
- beat: ex_claims_reaction
  goal: React to what the paper claims.
  user_instruction: React briefly to the reported claims of the paper.
  entry_condition: The PA has described the paper claims.
  avoid: Do not request unrelated work.
- beat: ex_receive_summary
  goal: Take the summary and see where it leaves the draft.
  active_skills:
  - verbosity
  if_not_met: If no summary has arrived, ask straight out what the paper covers and whether it is worth using.
  user_instruction: Acknowledge the delivered summary.
  entry_condition: The PA has delivered the summary.
  avoid: Do not use meta-language about testing.
- beat: ex_close_session
  goal: Close out with a next step.
  user_instruction: Close the session with a short next-step remark.
  entry_condition: The summary discussion is complete.
  avoid: Do not open a new task.
"""


def test_authored_runtime_uses_resolved_tools_without_text_inference(
    tmp_path: Path,
):
    project_root = tmp_path
    fixture_doc = (
        project_root
        / "data/state_fixtures/user_a/documents/research/anyons_review_paper.txt"
    )
    fixture_doc.parent.mkdir(parents=True)
    fixture_doc.write_text("Anyon review fixture", encoding="utf-8")
    catalog = project_root / "data/state_tools/catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        "tools:\n"
        "  document_read:\n"
        "    status: implemented\n"
        "    description: Read one document from the state root.\n"
        "    parameters:\n"
        "      path:\n"
        "        type: string\n"
        "        required: true\n"
        "    returns:\n"
        "      content: string\n",
        encoding="utf-8",
    )
    bundle = project_root / "data/state_tools/builds/test_build.yaml"
    bundle.parent.mkdir(parents=True)
    bundle.write_text(
        "build_id: test_build\nsupporting_fixtures: []\n",
        encoding="utf-8",
    )
    draft_path = project_root / "Authoring/sessions/normal/drafts/user_a/acc_001.yaml"
    draft_path.parent.mkdir(parents=True)
    draft_path.write_text(TOOLS_RESOLVED_DRAFT_YAML, encoding="utf-8")

    stage = StageRuntime(
        run_id="run_authored_acc_001",
        user_id="user_a",
        fixture_root=project_root / "data/state_fixtures",
        runs_root=tmp_path / "runs",
        project_root=project_root,
    )

    runtime = stage.begin_step(
        step_id="acc_001",
        session_id="acc_001",
        runtime_artifact_path=draft_path,
        session_date="2026-10-02",
        session_timestamp="2026-10-02T10:00:00+00:00",
    )

    # World-toolbox artifact: no manifest, the full implemented catalog is
    # exposed, and the complete world was staged.
    assert runtime.enabled_tools == ["document_read"]
    assert runtime.session_date == "2026-10-02"
    assert runtime.mcp_server_config()["state"].env["MEMPA_SESSION_DATE"] == "2026-10-02"
    assert (
        runtime.mcp_server_config()["state"].env["MEMPA_SESSION_TIMESTAMP"]
        == "2026-10-02T10:00:00+00:00"
    )
    assert (
        runtime.state_root / "documents/research/anyons_review_paper.txt"
    ).read_text(encoding="utf-8") == "Anyon review fixture"


def test_session_06_exposes_one_at_a_time_tools(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_06",
        run_id="run_006",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "comic_store_inventory_read", "cinema_showtime_search", "place_search"],
    )

    assert runtime.enabled_tools == [
        "document_read",
        "email_draft_create",
        "comic_store_inventory_read",
        "cinema_showtime_search",
        "place_search",
    ]


def test_session_08_exposes_follow_flow_tools(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_08",
        run_id="run_007",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "comic_store_inventory_read", "cinema_showtime_search", "place_search"],
    )

    assert runtime.enabled_tools == [
        "document_read",
        "email_draft_create",
        "comic_store_inventory_read",
        "cinema_showtime_search",
        "place_search",
    ]


def test_session_09_exposes_autonomy_boundary_tools(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "email" / "sent.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_09",
        run_id="run_009",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "email_message_send"],
    )

    assert runtime.enabled_tools == [
        "document_read",
        "email_draft_create",
        "email_message_send",
    ]


def test_session_12_exposes_autonomous_external_action_tools(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "email" / "sent.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_12",
        run_id="run_012",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "email_message_send"],
    )

    assert runtime.enabled_tools == [
        "document_read",
        "email_draft_create",
        "email_message_send",
    ]


def test_session_14_exposes_self_directed_boundary_tools(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "planning").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "email" / "sent.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_14",
        run_id="run_014",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "planning_note_create", "email_message_send"],
    )

    assert runtime.enabled_tools == [
        "document_read",
        "email_draft_create",
        "planning_note_create",
        "email_message_send",
    ]


def test_session_15_exposes_route_option_tool_for_solution_breadth(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "maps").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_15",
        run_id="run_015",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "map_route_search"],
    )

    assert runtime.enabled_tools == [
        "document_read",
        "email_draft_create",
        "map_route_search",
    ]


def test_session_16_exposes_route_option_tool_for_solution_breadth(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "maps").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_16",
        run_id="run_016",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "map_route_search"],
    )

    assert runtime.enabled_tools == [
        "document_read",
        "email_draft_create",
        "map_route_search",
    ]


def test_stage_runtime_commits_working_stage(tmp_path: Path):
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

    step = runtime.begin_step(step_id="acc_001", session_id="session_01")
    (step.state_root / "new.txt").write_text("committed", encoding="utf-8")
    runtime.finish_step(step, "commit")

    assert (tmp_path / "runs" / "run_001" / "canonical_stage" / "new.txt").read_text(
        encoding="utf-8"
    ) == "committed"


def test_stage_runtime_preserves_only_current_step_audit(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    fixture.mkdir(parents=True)
    (fixture / "tool_log.jsonl").write_text('{"tool":"old"}\n', encoding="utf-8")
    (fixture / "state_diff.jsonl").write_text('{"id":"old"}\n', encoding="utf-8")
    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_a",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )

    step = runtime.begin_step(step_id="acc_001", session_id="session_01")
    assert not (step.state_root / "tool_log.jsonl").exists()
    assert not (step.state_root / "state_diff.jsonl").exists()
    (step.state_root / "tool_log.jsonl").write_text(
        '{"tool":"current"}\n', encoding="utf-8"
    )

    artifacts = preserve_step_state_artifacts(step, tmp_path / "steps" / "acc_001")
    runtime.finish_step(step, "commit")

    assert Path(artifacts["tool_log"]).read_text(encoding="utf-8") == (
        '{"tool":"current"}\n'
    )
    assert Path(artifacts["state_diff"]).read_text(encoding="utf-8") == ""
    assert not (runtime.canonical_stage / "tool_log.jsonl").exists()
    assert not (runtime.canonical_stage / "state_diff.jsonl").exists()


def test_stage_runtime_discards_probe_changes(tmp_path: Path):
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

    step = runtime.begin_step(step_id="pretest", session_id="session_01")
    (step.state_root / "new.txt").write_text("discarded", encoding="utf-8")
    cfg = step.mcp_server_config()["state"]
    runtime.finish_step(step, "discard")

    assert not (tmp_path / "runs" / "run_001" / "canonical_stage" / "new.txt").exists()
    assert cfg.env["MEMPA_STATE_ROOT"].endswith("stage_work/pretest")


def test_session_17_exposes_route_option_tool_for_solution_breadth(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "maps").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_17",
        run_id="run_017",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "map_route_search"],
    )

    assert runtime.enabled_tools == [
        "document_read",
        "email_draft_create",
        "map_route_search",
    ]


def test_sessions_18_to_20_expose_archive_tools_for_proactive_outreach(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "planning").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    for session_id in ("session_18", "session_19", "session_20"):
        runtime = initialize_state_runtime(
            user_id="user_a",
            session_id=session_id,
            run_id=f"run_{session_id[-2:]}",
            fixture_root=tmp_path / "fixtures",
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
            enabled_tools=["document_read", "email_draft_create", "planning_note_create"],
        )

        assert runtime.enabled_tools == [
            "document_read",
            "email_draft_create",
            "planning_note_create",
        ]


def test_sessions_21_to_22_expose_contacts_for_capability_boundary(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "contacts.json").write_text("{}", encoding="utf-8")

    for session_id in ("session_21", "session_22"):
        runtime = initialize_state_runtime(
            user_id="user_a",
            session_id=session_id,
            run_id=f"run_{session_id[-2:]}",
            fixture_root=tmp_path / "fixtures",
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
            enabled_tools=["document_read", "email_draft_create", "contact_search"],
        )

        assert runtime.enabled_tools == [
            "document_read",
            "email_draft_create",
            "contact_search",
        ]


def test_sessions_23_to_25_expose_planning_notes_for_task_expansion(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "planning").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    for session_id in ("session_23", "session_24", "session_25"):
        runtime = initialize_state_runtime(
            user_id="user_a",
            session_id=session_id,
            run_id=f"run_{session_id[-2:]}",
            fixture_root=tmp_path / "fixtures",
            runs_root=tmp_path / "runs",
            project_root=tmp_path,
            enabled_tools=["document_read", "email_draft_create", "planning_note_create"],
        )

        assert runtime.enabled_tools == [
            "document_read",
            "email_draft_create",
            "planning_note_create",
        ]



# Static runtime regression tests

def test_static_fixture_mode_does_not_infer_email_message_send_from_session_id(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_09",
        run_id="run_s7_009",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    assert "email_message_send" not in runtime.enabled_tools
    assert runtime.enabled_tools == ["document_read", "email_draft_create"]


def test_static_fixture_mode_does_not_infer_store_tools_from_session_id(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_06",
        run_id="run_s7_006",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    assert "store_inventory_read" not in runtime.enabled_tools
    assert runtime.enabled_tools == ["document_read", "email_draft_create"]


def test_explicit_enabled_tools_still_work(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_09",
        run_id="run_s7_explicit",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        enabled_tools=["document_read", "email_draft_create", "email_message_send"],
    )
    assert runtime.enabled_tools == ["document_read", "email_draft_create", "email_message_send"]


def test_user_c_forced_claim_packet_survives_runtime_send_and_readback(tmp_path: Path):
    runtime = initialize_state_runtime(
        user_id="user_c",
        session_id="evolv_02",
        run_id="run_c_claim",
        fixture_root=Path("data/state_fixtures"),
        runs_root=tmp_path / "runs",
        project_root=Path.cwd(),
        enabled_tools=["document_read", "email_message_send", "email_sent_read"],
    )
    store = StateStore(
        runtime.state_root,
        run_id=runtime.run_id,
        user_id=runtime.user_id,
        session_id=runtime.session_id,
        session_date="2026-10-10",
    )
    attachments = [
        "claim_packets/coastline_claim/claim_body.md",
        "claim_packets/coastline_claim/pallet_damage_photo.jpg",
        "claim_packets/coastline_claim/signed_delivery_note.md",
    ]
    fixture = Path("data/state_fixtures/user_c")
    before = {p: hashlib.sha256((fixture / p).read_bytes()).hexdigest() for p in attachments}

    sent_id = store.send_email(
        "Coastline Freight",
        "Damage claim filing - Pallet delivery verification",
        "Delivery time verified against dock weighbridge slip stamp showing 14:15 arrival.",
        attachments,
    )["sent_id"]
    readback = store.read_sent_email(sent_id)
    after = {
        p: hashlib.sha256((runtime.state_root / p).read_bytes()).hexdigest()
        for p in readback["attachments"]
    }

    assert readback["attachments"] == attachments
    assert before == after
    claim_body = store.read_document(attachments[0])["content"]
    assert "lower-left carton edge is visibly crushed" in claim_body
    assert "both black securing straps are broken" in claim_body
    assert "forklift" not in claim_body
    assert "SIGNED — Curtis Robinson" in store.read_document(attachments[2])["content"]
    assert (runtime.state_root / attachments[1]).read_bytes().startswith(b"\xff\xd8")


def test_user_c_forced_case_and_group_repairs_are_visible_in_runtime(tmp_path: Path):
    runtime = initialize_state_runtime(
        user_id="user_c",
        session_id="evolv_04_05",
        run_id="run_c_case_group",
        fixture_root=Path("data/state_fixtures"),
        runs_root=tmp_path / "runs",
        project_root=Path.cwd(),
    )
    store = StateStore(
        runtime.state_root,
        run_id=runtime.run_id,
        user_id=runtime.user_id,
        session_id=runtime.session_id,
        session_date="2026-12-11",
    )
    draft_id = store.save_email_draft(
        "GlenMart returns", "Return Case Details - Order GM-126104", "Prepared note"
    )["draft_id"]
    store.shopping_return_case_update(
        "rcase_0001",
        "the returns desk, not the shop floor on the plaza",
        "the case already open on the order, not a new one raised from scratch",
        draft_id,
    )
    case = store.shopping_return_case_read("rcase_0001")["case"]
    contact = store.contact_read("ct_0059")["contact"]
    message_id = store.message_send(contact["name"], "Venue update for all eight.")["message_id"]
    message = store.message_read(message_id)

    assert case["status"] == "open_on_hold"
    assert case["note_draft_id"] == draft_id
    assert "eight recipients total" in contact["notes"]
    assert message["with"] == "Saturday run — game-night eight"


def _world_toolbox_artifact(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "meta:\n  tool_status: world_toolbox\n",
        encoding="utf-8",
    )
    return path


def test_stage_runtime_accepts_explicit_run_dir(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    fixture.mkdir(parents=True)
    (fixture / "base.txt").write_text("fixture", encoding="utf-8")
    branch = tmp_path / "branches" / "fin_03" / "01"
    (branch / "canonical_stage").mkdir(parents=True)
    (branch / "canonical_stage" / "base.txt").write_text(
        "from checkpoint", encoding="utf-8"
    )

    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_a",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
        run_dir=branch,
    )

    assert runtime.run_dir == branch
    assert runtime.canonical_stage == branch / "canonical_stage"
    assert runtime.working_root == branch / "stage_work"
    # The branch already carries a world, so the fixture is not copied over it.
    assert (runtime.canonical_stage / "base.txt").read_text(
        encoding="utf-8"
    ) == "from checkpoint"
    assert not (tmp_path / "runs").exists()


def test_probe_fixture_overlays_without_resetting_world(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "documents" / "notes.md").write_text("cut point", encoding="utf-8")
    (fixture / "documents" / "kept.md").write_text("inherited", encoding="utf-8")
    catalog = tmp_path / "data/state_tools/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "tools:\n  document_read:\n    status: implemented\n", encoding="utf-8"
    )
    probe = _world_toolbox_artifact(tmp_path / "probe" / "fin_03.yaml")
    overlay = tmp_path / "probe" / "fixture" / "documents"
    overlay.mkdir(parents=True)
    (overlay / "notes.md").write_text("probe scene", encoding="utf-8")

    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_a",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    step = runtime.begin_step(
        step_id="fin_03",
        session_id="fin_03",
        runtime_artifact_path=probe,
    )

    assert (step.state_root / "documents" / "notes.md").read_text(
        encoding="utf-8"
    ) == "probe scene"
    assert (step.state_root / "documents" / "kept.md").read_text(
        encoding="utf-8"
    ) == "inherited"

    # A paired final probe is handed its partner's overlay explicitly.
    paired = runtime.begin_step(
        step_id="pre_01",
        session_id="pre_01",
        runtime_artifact_path=_world_toolbox_artifact(
            tmp_path / "other" / "pre_01.yaml"
        ),
        fixture_overlay=tmp_path / "probe" / "fixture",
    )
    assert (paired.state_root / "documents" / "notes.md").read_text(
        encoding="utf-8"
    ) == "probe scene"

    # No overlay directory: unchanged behaviour.
    plain = runtime.begin_step(
        step_id="acc_001",
        session_id="acc_001",
        runtime_artifact_path=_world_toolbox_artifact(
            tmp_path / "normal" / "acc_001.yaml"
        ),
    )
    assert (plain.state_root / "documents" / "notes.md").read_text(
        encoding="utf-8"
    ) == "cut point"


def _records_overlay_world(tmp_path: Path) -> Path:
    """A fixture whose two stores already hold one record each."""
    fixture = tmp_path / "fixtures" / "user_b"
    (fixture / "email").mkdir(parents=True)
    (fixture / "email" / "inbox.jsonl").write_text(
        json.dumps({"id": "em_0001", "subject": "accumulated"}) + "\n",
        encoding="utf-8",
    )
    (fixture / "contacts.json").write_text(
        json.dumps({"contacts": [{"contact_id": "ct_0001", "name": "Alex"}]}),
        encoding="utf-8",
    )
    catalog = tmp_path / "data/state_tools/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "tools:\n  document_read:\n    status: implemented\n", encoding="utf-8"
    )
    return fixture


def _write_records_overlay(overlay: Path, rows: list[dict]) -> Path:
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return overlay


_NEW_EMAIL = {
    "store": "email/inbox.jsonl",
    "id_field": "id",
    "record_id": "em_0051",
    "record": {"id": "em_0051", "subject": "the probe's scene"},
}
_NEW_CONTACT = {
    "store": "contacts.json",
    "id_field": "contact_id",
    "record_id": "ct_0036",
    "record": {"contact_id": "ct_0036", "name": "Bridgeline Payroll"},
}
#: The row the world already holds, verbatim: the overlay is idempotent.
_HELD_EMAIL = {
    "store": "email/inbox.jsonl",
    "id_field": "id",
    "record_id": "em_0001",
    "record": {"id": "em_0001", "subject": "accumulated"},
}


def test_a_probe_records_overlay_appends_without_erasing_the_store(tmp_path: Path):
    """A record the world gained after the run was frozen is appended for the
    probe's branch alone; a whole-file overlay would drop every accumulated
    write in the same store."""
    fixture = _records_overlay_world(tmp_path)
    probe = _world_toolbox_artifact(tmp_path / "probe" / "fin_09.yaml")
    _write_records_overlay(
        tmp_path / "probe" / "fixture",
        [_NEW_EMAIL, _NEW_CONTACT, _HELD_EMAIL],
    )

    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_b",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    step = runtime.begin_step(
        step_id="fin_09",
        session_id="fin_09",
        runtime_artifact_path=probe,
        merge_records=True,
    )

    inbox = [
        json.loads(line)
        for line in (step.state_root / "email" / "inbox.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["id"] for row in inbox] == ["em_0001", "em_0051"]
    contacts = json.loads(
        (step.state_root / "contacts.json").read_text(encoding="utf-8")
    )["contacts"]
    assert [row["contact_id"] for row in contacts] == ["ct_0001", "ct_0036"]
    # The rows are records to append, never a file of the world.
    assert not (step.state_root / "records.jsonl").exists()


def test_a_records_overlay_on_a_non_probe_step_is_refused(tmp_path: Path):
    fixture = _records_overlay_world(tmp_path)
    assert fixture.is_dir()
    session = _world_toolbox_artifact(tmp_path / "normal" / "acc_001.yaml")
    _write_records_overlay(tmp_path / "normal" / "fixture", [_NEW_EMAIL])

    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_b",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    with pytest.raises(ValueError, match="probe-only overlay"):
        runtime.begin_step(
            step_id="acc_001",
            session_id="acc_001",
            runtime_artifact_path=session,
        )


def test_a_records_overlay_colliding_with_a_run_written_record_is_refused(
    tmp_path: Path,
):
    """The store mints jsonl ids by counting rows, so the run's own writes take
    the ids a later fixture record also claims. Keeping either one silently
    hands the probe a record nobody wrote for it."""
    fixture = _records_overlay_world(tmp_path)
    (fixture / "email" / "inbox.jsonl").write_text(
        json.dumps({"id": "em_0051", "subject": "the PA wrote this"}) + "\n",
        encoding="utf-8",
    )
    probe = _world_toolbox_artifact(tmp_path / "probe" / "fin_09.yaml")
    _write_records_overlay(tmp_path / "probe" / "fixture", [_NEW_EMAIL])

    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_b",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    with pytest.raises(ValueError, match="collides with a different record"):
        runtime.begin_step(
            step_id="fin_09",
            session_id="fin_09",
            runtime_artifact_path=probe,
            merge_records=True,
        )


def test_a_records_overlay_naming_an_absent_store_is_refused(tmp_path: Path):
    _records_overlay_world(tmp_path)
    probe = _world_toolbox_artifact(tmp_path / "probe" / "fin_09.yaml")
    _write_records_overlay(
        tmp_path / "probe" / "fixture",
        [{**_NEW_EMAIL, "store": "message/threads.jsonl"}],
    )

    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_b",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="message/threads.jsonl"):
        runtime.begin_step(
            step_id="fin_09",
            session_id="fin_09",
            runtime_artifact_path=probe,
            merge_records=True,
        )


def test_a_failed_step_leaves_the_canonical_stage_untouched(tmp_path: Path):
    """The "no state change" guarantee's actual mechanism.

    A turn that later fails the IX contract has already run its tools -- the
    raise happens after the agent loop returns -- so the world write exists.
    What makes it harmless is where it lives: `begin_step` copies the canonical
    stage into `stage_work/<step_id>`, every tool writes there, and only
    `finish_step(..., "commit")` copies it back. A failed step never reaches
    that call (the orchestrator raises first), so the write dies with the
    working stage and the canonical world is byte-for-byte what it was.
    """
    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "appointment").mkdir(parents=True)
    (fixture / "appointment" / "appointments.jsonl").write_text("", encoding="utf-8")
    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_a",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    canonical = tmp_path / "runs" / "run_001" / "canonical_stage"
    before = sorted(
        (path.relative_to(canonical), path.read_bytes())
        for path in canonical.rglob("*") if path.is_file()
    )

    step = runtime.begin_step(step_id="acc_039", session_id="acc_039")
    booked = step.state_root / "appointment" / "appointments.jsonl"
    booked.write_text('{"appointment_id":"apt_0003"}\n', encoding="utf-8")

    # The step fails: `finish_step` is never reached (harness/orchestrator.py
    # calls it after `executor.execute`, which raised).
    assert booked.read_text(encoding="utf-8") == '{"appointment_id":"apt_0003"}\n'
    assert sorted(
        (path.relative_to(canonical), path.read_bytes())
        for path in canonical.rglob("*") if path.is_file()
    ) == before
    assert "apt_0003" not in (
        canonical / "appointment" / "appointments.jsonl"
    ).read_text(encoding="utf-8")

    # And a step that is finished with `discard` drops the working stage too.
    runtime.finish_step(step, "discard")
    assert not (tmp_path / "runs" / "run_001" / "stage_work" / "acc_039").exists()
    assert sorted(
        (path.relative_to(canonical), path.read_bytes())
        for path in canonical.rglob("*") if path.is_file()
    ) == before


#: The fourth collision class: S0 grew this record *inside* after the run was
#: frozen (three February 2027 slots on an availability record that is
#: otherwise identical), so appending by id cannot reach it.
_GROWN_AVAILABILITY = {
    "store": "appointment/availability.json",
    "id_field": "provider_id",
    "record_id": "prov_larkspur",
    "op": "replace",
    "record": {
        "provider_id": "prov_larkspur",
        "name": "Larkspur Nails",
        "slots": ["2027-01-04T10:00", "2027-02-01T10:00"],
    },
}


def _availability_world(tmp_path: Path, record: dict) -> Path:
    fixture = _records_overlay_world(tmp_path)
    (fixture / "appointment").mkdir(parents=True, exist_ok=True)
    (fixture / "appointment" / "availability.json").write_text(
        json.dumps({"providers": [record]}), encoding="utf-8"
    )
    return fixture


def _availability_step(tmp_path: Path, rows: list[dict]):
    probe = _world_toolbox_artifact(tmp_path / "probe" / "fin_16.yaml")
    _write_records_overlay(tmp_path / "probe" / "fixture", rows)
    runtime = StageRuntime(
        run_id="run_001",
        user_id="user_b",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    return runtime.begin_step(
        step_id="fin_16",
        session_id="fin_16",
        runtime_artifact_path=probe,
        merge_records=True,
    )


def test_a_replace_row_overwrites_a_record_the_run_never_wrote(tmp_path: Path):
    _availability_world(
        tmp_path,
        {
            "provider_id": "prov_larkspur",
            "name": "Larkspur Nails",
            "slots": ["2027-01-04T10:00"],
        },
    )

    step = _availability_step(tmp_path, [_GROWN_AVAILABILITY])

    providers = json.loads(
        (step.state_root / "appointment" / "availability.json").read_text(
            encoding="utf-8"
        )
    )["providers"]
    assert providers == [_GROWN_AVAILABILITY["record"]]


def test_a_replace_row_refuses_a_record_the_run_wrote(tmp_path: Path):
    """`replace` is for a copy the fixture grew past, never a way to drop a
    write the PA made under the same id: the store stamps its own records with
    the run identity, and that is the whole test."""
    _availability_world(
        tmp_path,
        {
            "provider_id": "prov_larkspur",
            "name": "Larkspur Nails",
            "slots": ["2027-01-04T10:00"],
            "run_id": "run_001",
            "user_id": "user_b",
            "session_id": "acc_042",
        },
    )

    with pytest.raises(ValueError, match="would replace a record this run wrote"):
        _availability_step(tmp_path, [_GROWN_AVAILABILITY])


def test_a_replace_row_that_changes_nothing_is_idempotent(tmp_path: Path):
    _availability_world(tmp_path, dict(_GROWN_AVAILABILITY["record"]))

    step = _availability_step(tmp_path, [_GROWN_AVAILABILITY])

    providers = json.loads(
        (step.state_root / "appointment" / "availability.json").read_text(
            encoding="utf-8"
        )
    )["providers"]
    assert providers == [_GROWN_AVAILABILITY["record"]]


def test_an_unknown_overlay_op_is_refused(tmp_path: Path):
    _availability_world(tmp_path, dict(_GROWN_AVAILABILITY["record"]))

    with pytest.raises(ValueError, match="op must be 'append' or 'replace'"):
        _availability_step(
            tmp_path, [{**_GROWN_AVAILABILITY, "op": "merge_slots"}]
        )
